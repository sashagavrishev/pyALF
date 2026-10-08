"""Bin counts read from ALF's ``data.h5``."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from typing import Any

from .simulation import Simulation

logger = logging.getLogger(__name__)


_bin_cache: dict[Any, int] = {}
# Keys read once while their job was already in a terminal state. Such a count
# can never change again, so it is served from _bin_cache without touching the
# filesystem. Callers monitoring many finished simulations would otherwise
# re-open every data.h5 on every refresh.
_bin_final: set[Any] = set()

# (st_mtime_ns, st_size) of the data.h5 each cached count was read from, used to
# skip the h5py open when the file has not changed since.
_bin_stat: dict[Any, tuple[int, int]] = {}

# HDF5 open failures that mean "ALF is part-way through writing this file", not
# "this file is broken".  ALF rewrites data.h5 in place, so a reader can catch it
# after the superblock records a new end-of-file but before the data is flushed --
# hence "truncated file: eof = ... < stored_eof = ...".  The condition clears
# itself as soon as the write completes.
_MIDWRITE_MARKERS = (
    "truncated file",
    "file signature not found",
    "unable to read superblock",
    "bad object header version number",
    "unable to lock file",
    "resource temporarily unavailable",
)

# Backoff between re-open attempts.  ALF's write window is short, so one or two
# retries usually turn a mid-write into a good read rather than a stale row.
_H5_RETRY_DELAYS = (0.05, 0.15)

# Consecutive failed reads of one file before a mid-write stops being treated as
# transient and gets reported.  At a 30s refresh this is minutes of failure, by
# which point the file is genuinely damaged rather than being written.
_MIDWRITE_LOG_AFTER = 5

_bin_read_failures: dict[Any, int] = {}


def _is_midwrite_error(exc: BaseException | str) -> bool:
    """True if *exc* looks like a read that raced ALF's writer.

    Accepts a string as well as an exception: a worker process's mid-write
    verdict has to cross back to the caller through
    :func:`_read_bin_count`'s return value, since the exception itself is not
    always picklable and does not need to survive the trip -- only whether it
    matched one of these markers does.
    """
    text = (exc if isinstance(exc, str) else str(exc)).lower()
    return any(marker in text for marker in _MIDWRITE_MARKERS)


# Filesystems record mtimes coarsely -- NFS commonly only to the second -- so a
# change landing in the same tick as a reading is invisible to it.  Caching such
# a reading would pin a stale answer until something else moved the mtime past
# the tick, so a reading is only trusted once its mtime has settled.
_MTIME_SETTLE_NS = 2_000_000_000


def _mtime_settled(mtime_ns: int) -> bool:
    """True if *mtime_ns* is far enough in the past to be a safe cache key."""
    return time.time_ns() - mtime_ns > _MTIME_SETTLE_NS


# h5py wraps every call into the HDF5 C library in a single process-wide lock
# (h5py._objects.phil) -- confirmed empirically: N threads each holding it for
# 50ms behave identically to N sequential 50ms calls, regardless of how many
# threads there are. So fanning bin-count reads out across _get_io_pool's
# threads (as Campaign.status() does) never actually overlaps the blocking
# h5py.File() open itself, only the Python-level dispatch around it. Only a
# separate OS process gets its own independent HDF5 library instance -- and
# hence its own phil -- so this pool is what actually lets two data.h5 opens
# progress at once. Deliberately smaller than _MAX_IO_WORKERS: each worker
# imports h5py/numpy into its own address space, which costs real memory an
# idle thread would not, and (unlike the thread pool) is opt-in per call
# rather than the default -- see _bin_count's ``use_process_pool``.
_process_pool: ProcessPoolExecutor | None = None
_process_pool_lock = threading.Lock()


def _process_pool_size() -> int:
    """Worker count for the h5py read pool, capped to any SLURM allocation.

    A bare process count defaults to the whole node's cores, not the cgroup a SLURM
    task was actually granted, and a status check run as its own job (e.g.
    ``reconcile`` on a timer) must not oversubscribe that allocation the way
    a handful of extra processes importing h5py/numpy each could.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    budget = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)
    return min(8, max(2, budget))


def _get_process_pool() -> ProcessPoolExecutor:
    """The shared process pool for h5py reads, created on first use."""
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            _process_pool = ProcessPoolExecutor(max_workers=_process_pool_size())
        return _process_pool


def _stat_sig(filename: str) -> tuple[int, int] | None:
    """``(st_mtime_ns, st_size)`` of *filename*, or None if it cannot be stat'd."""
    try:
        st = os.stat(filename)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_bin_count(
    filename: str, counting_obs: str
) -> tuple[int, bool, str | None, tuple[int, int] | None]:
    """Open *filename* and read its bin count, retrying through a mid-write race.

    Pure function of its arguments -- no cache, no module state read or
    written -- so it is safe to run inside a worker process via
    :func:`_get_process_pool`. The caller owns everything stateful: caching,
    the stale-value fallback, and the failure-count bookkeeping, none of which
    a worker process could share back with the caller's copy of those module
    dicts anyway.

    The file's (mtime, size) is taken here and handed back rather than stat'd by
    the caller, so that on a batch a grid's worth of stats runs across the pool
    instead of serially in one process -- on a networked filesystem a stat is
    milliseconds, which at campaign scale is minutes. Taken *before* the read
    for the reason :func:`_bin_count` documents: a write landing in between then
    leaves a stale signature that forces a re-read next time, which is the safe
    direction to be wrong in.
    """
    import h5py

    sig = _stat_sig(filename)
    N_bins = 0
    read_ok = False
    last_exc: BaseException | None = None
    for attempt in range(len(_H5_RETRY_DELAYS) + 1):
        try:
            # POSIX file locking stalls (or errors) on networked filesystems, and
            # a read-only probe gets nothing from it.
            with h5py.File(filename, "r", locking=False) as f:
                if counting_obs in f:
                    N_bins = f[counting_obs + "/obser"].shape[0]
            read_ok = True
            break
        except FileNotFoundError:
            break
        except (OSError, KeyError) as e:
            last_exc = e
            # Only a mid-write is worth retrying, and only while attempts remain.
            if not _is_midwrite_error(e) or attempt == len(_H5_RETRY_DELAYS):
                break
            time.sleep(_H5_RETRY_DELAYS[attempt])

    return N_bins, read_ok, repr(last_exc) if last_exc is not None else None, sig


def _bin_count(
    sim: Simulation,
    counting_obs: str = "Ener_scal",
    refresh: bool = False,
    data_dir: str | None = None,
    final: bool = False,
    force: bool = False,
    use_process_pool: bool = False,
) -> int:
    """
    Counts bins for a given observable in simulation data, with caching.
    Args:
        sim: Simulation instance.
        counting_obs: Observable name.
        refresh: Whether to refresh cache.
        data_dir: Directory holding ``data.h5`` to read instead of
            ``sim.sim_dir`` — used to count bins in a single ``Temp_i/``
            realisation of a PARALLEL_PARAMS job.
        final: Whether the job has reached a terminal state. The file is still
            read once (the last bins may have landed since the previous
            refresh), but the result is then frozen and served from cache.
        force: Skip the (mtime, size) short-circuit and re-read the file even if
            it looks unchanged.
        use_process_pool: Run the actual read in :func:`_get_process_pool`
            instead of this process. Worth it only when many chains are being
            probed at once (h5py serializes every call in-process regardless
            of thread count -- see that pool's docstring), so this defaults
            off: a lone call, like a worker job checking its own bin count,
            would pay a process pool's startup cost for nothing.
    Returns:
        Number of bins.
    """
    filename = os.path.join(
        data_dir if data_dir is not None else sim.sim_dir, "data.h5"
    )
    key = (filename, counting_obs)

    if key in _bin_final:
        return _bin_cache.get(key, 0)

    if (key in _bin_cache) and (not refresh):
        return _bin_cache[key]

    # ALF rewrites data.h5 as a whole, so an unchanged (mtime, size) means
    # unchanged content: a stat is far cheaper than letting h5py parse the file
    # structure, and a running job appends a bin far less often than the monitor
    # polls.  Stat *before* reading — a write landing between the two then leaves
    # a stale signature that forces a re-read next time, whereas stat-after would
    # pair the new signature with the old count and never re-read.
    stat_sig: tuple[int, int] | None = None
    try:
        st = os.stat(filename)
        stat_sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    if (
        not force
        and stat_sig is not None
        and key in _bin_cache
        and _bin_stat.get(key) == stat_sig
    ):
        return _bin_cache[key]

    if use_process_pool:
        read = (
            _get_process_pool().submit(_read_bin_count, filename, counting_obs).result()
        )
    else:
        read = _read_bin_count(filename, counting_obs)

    return _absorb_bin_read(key, filename, read, final)


def _absorb_bin_read(
    key: tuple[str, str],
    filename: str,
    read: tuple[int, bool, str | None, tuple[int, int] | None],
    final: bool,
) -> int:
    """Fold one :func:`_read_bin_count` result into the module caches.

    Split out of :func:`_bin_count` so the batched path
    (:func:`read_bin_counts`) inherits the same caching, stale-value fallback and
    failure bookkeeping instead of reimplementing them. The signature travels
    in ``read`` because the reader takes it (see :func:`_read_bin_count`).
    """
    N_bins, read_ok, error_text, stat_sig = read

    if error_text is not None and not read_ok:
        fails = _bin_read_failures.get(key, 0) + 1
        _bin_read_failures[key] = fails
        # Racing ALF's writer is expected and self-correcting, so stay quiet
        # about it; a file that keeps failing is a real problem worth surfacing.
        if _is_midwrite_error(error_text) and fails < _MIDWRITE_LOG_AFTER:
            logger.debug(
                "%s is mid-write (attempt %d), keeping the cached bin count: %s",
                filename,
                fails,
                error_text,
            )
        else:
            logger.error(f"Error reading {filename}: {error_text}")
        # Keep the last known good value rather than caching a spurious 0 that
        # would be shown briefly on the next refresh.
        return _bin_cache.get(key, 0)

    _bin_read_failures.pop(key, None)

    # Don't let a transient 0 overwrite a previously-seen non-zero count —
    # ALF truncates and rewrites data.h5 between bins, so a 0 mid-write is
    # not meaningful and would cause the progress bar to flicker.  The stat
    # signature is deliberately not recorded here, so the next refresh re-reads
    # rather than caching the mid-write state.
    if N_bins == 0 and _bin_cache.get(key, 0) > 0:
        return _bin_cache[key]

    _bin_cache[key] = N_bins
    if read_ok and stat_sig is not None and _mtime_settled(stat_sig[0]):
        _bin_stat[key] = stat_sig
    # Only freeze a count that came from an actual read: a terminal job whose
    # data.h5 is missing or unreadable may still appear once the filesystem
    # catches up, or once a truncated file is repaired.
    if final and read_ok:
        _bin_final.add(key)
    return N_bins


# Files per task handed to the process pool by read_bin_counts. One file per task
# makes every read cost a full pickle/IPC round trip, which on a campaign-sized
# batch dominates the h5py open it was meant to overlap; a chunk amortises that
# over many reads while staying small enough that the workers finish together.
_BIN_BATCH_CHUNK = 64


def read_bin_counts(
    filenames: list[str],
    counting_obs: str = "Ener_scal",
    final: bool = False,
    force: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> list[int]:
    """Bin counts for many ``data.h5`` paths at once, in caller order.

    The batched counterpart of :func:`_bin_count`, for a caller holding
    thousands of paths (:meth:`py_alf.campaign.Campaign.status`). It differs
    only in *dispatch*: the cheap (mtime, size) short-circuit runs here, and
    whatever survives it goes to :func:`_get_process_pool` as one chunked
    ``map`` rather than one blocking ``submit``/``result`` per file. Reading
    one file at a time through the pool costs an IPC round trip per read and
    serialises on the executor's single work queue -- measured on a
    24k-chain campaign, that fan-out achieved no concurrency at all.

    ``on_progress(n)`` is called with how many paths a step settled, so a caller
    can drive a bar over a batch big enough to be worth watching.
    """
    keys = [(name, counting_obs) for name in filenames]
    counts: list[int | None] = [None] * len(filenames)
    pending: list[int] = []

    for i, key in enumerate(keys):
        if key in _bin_final:
            counts[i] = _bin_cache.get(key, 0)
            continue
        # Stat only when there is a previous reading for the stat to validate.
        # _bin_stat lives in this process, so a freshly started CLI holds none
        # and every such stat is a guaranteed miss -- on a networked filesystem
        # a stat is milliseconds, and a grid's worth of them cost more than they
        # could ever save. The chains that do have one (a second pass within a
        # run, e.g. reconcile's launch after its status) still short-circuit,
        # and the reads themselves take their own signature inside the pool.
        if not force and key in _bin_cache:
            # Both sides return None when absent, so an unrecorded signature and
            # a vanished file would otherwise compare equal and serve the cache
            # without ever looking at the disk.
            recorded = _bin_stat.get(key)
            if recorded is not None and recorded == _stat_sig(filenames[i]):
                counts[i] = _bin_cache[key]
                continue
        pending.append(i)

    # Everything settled by the caches above cost nothing, so report it in one
    # step: a bar that inched through them would misrepresent where the time is.
    if on_progress is not None and len(pending) < len(filenames):
        on_progress(len(filenames) - len(pending))

    if pending:
        reads = _get_process_pool().map(
            _read_bin_count,
            [filenames[i] for i in pending],
            repeat(counting_obs, len(pending)),
            chunksize=max(
                1, min(_BIN_BATCH_CHUNK, len(pending) // _process_pool_size())
            ),
        )
        for i, read in zip(pending, reads):
            counts[i] = _absorb_bin_read(keys[i], filenames[i], read, final)
            if on_progress is not None:
                on_progress(1)

    return [0 if c is None else c for c in counts]


Simulation.bin_count = _bin_count
