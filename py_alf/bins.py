"""Bin counts read from ALF's ``data.h5``."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

logger = logging.getLogger(__name__)


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
# retries usually turn a mid-write into a good read.
_H5_RETRY_DELAYS = (0.05, 0.15)


def _is_midwrite_error(text: str) -> bool:
    """True if an HDF5 error message looks like a read that raced ALF's writer."""
    text = text.lower()
    return any(marker in text for marker in _MIDWRITE_MARKERS)


# h5py serialises every call into the HDF5 library behind one process-wide lock,
# so threads cannot overlap data.h5 opens; only separate processes can. Smaller
# than the I/O thread pool because each worker imports h5py and numpy.
_process_pool: ProcessPoolExecutor | None = None
_process_pool_lock = threading.Lock()


def _process_pool_size() -> int:
    """Worker count for the h5py read pool, capped to any SLURM allocation."""
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


def _read_bin_count(filename: str, counting_obs: str) -> tuple[int, str | None]:
    """``(bins, error)`` for one file, retrying through a mid-write race.

    A top-level function with no module state, so the process pool can run it.
    A missing file is 0 bins and no error: the chain has not started.
    """
    import h5py

    error = None
    for attempt in range(len(_H5_RETRY_DELAYS) + 1):
        try:
            # POSIX file locking stalls (or errors) on networked filesystems, and
            # a read-only probe gets nothing from it.
            with h5py.File(filename, "r", locking=False) as f:
                if counting_obs not in f:
                    return 0, None
                return int(f[counting_obs + "/obser"].shape[0]), None
        except FileNotFoundError:
            return 0, None
        except (OSError, KeyError) as e:
            error = repr(e)
            if not _is_midwrite_error(error) or attempt == len(_H5_RETRY_DELAYS):
                break
            time.sleep(_H5_RETRY_DELAYS[attempt])
    return 0, error


def _report(filename: str, error: str) -> None:
    if _is_midwrite_error(error):
        logger.warning(
            "%s is still being written, counted as 0 bins: %s", filename, error
        )
    else:
        logger.error("Error reading %s: %s", filename, error)


def read_bin_count(filename: str, counting_obs: str = "Ener_scal") -> int:
    """Bins of *counting_obs* in one ``data.h5``; 0 if it is missing or unreadable."""
    bins, error = _read_bin_count(filename, counting_obs)
    if error is not None:
        _report(filename, error)
    return bins


# Files per task handed to the process pool: one file per task would make every
# read a full pickle round trip, outweighing the open it was meant to overlap.
_BIN_BATCH_CHUNK = 64


def read_bin_counts(
    filenames: list[str],
    counting_obs: str = "Ener_scal",
    on_progress: Callable[[int], None] | None = None,
) -> list[int]:
    """:func:`read_bin_count` for many files at once, in caller order.

    Reads run in the process pool as one chunked ``map``. ``on_progress(n)`` is
    called as each file is settled.
    """
    if not filenames:
        return []
    reads = _get_process_pool().map(
        _read_bin_count,
        filenames,
        repeat(counting_obs, len(filenames)),
        chunksize=max(1, min(_BIN_BATCH_CHUNK, len(filenames) // _process_pool_size())),
    )
    counts = []
    for filename, (bins, error) in zip(filenames, reads, strict=True):
        if error is not None:
            _report(filename, error)
        counts.append(bins)
        if on_progress is not None:
            on_progress(1)
    return counts
