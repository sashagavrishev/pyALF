"""Shared thread pool for filesystem probes over many simulation directories."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

# Upper bound on concurrent filesystem probes (bin counts, submitit log reads,
# ...).  These threads spend most of their time blocked on a networked
# filesystem, but h5py's own parsing of each data.h5 is real CPU work, and a
# login node is shared with everyone else logged into it -- so on a machine
# with room to spare the width tracks its core count rather than growing
# purely with however much latency there is to hide.  Clamped at both ends:
# a floor of 8 keeps real overlap available on a constrained sandbox or CI
# container (whose core count reflects nothing about a login node and can be
# too low to run the fan-out concurrently at all), and a ceiling of 32 avoids
# oversubscribing a very large machine for what is still I/O-bound work.
# Below _MIN_FANOUT items the pool costs more to start than the I/O it would
# overlap. Shared by any caller that probes many sim directories at once,
# such as :class:`py_alf.campaign.Campaign`.
_MAX_IO_WORKERS = min(32, max(8, os.cpu_count() or 16))
_MIN_FANOUT = 3


_io_pool: ThreadPoolExecutor | None = None
_io_pool_lock = threading.Lock()


def _get_io_pool() -> ThreadPoolExecutor:
    """The shared probe pool, created on first use.

    Reused across refreshes rather than rebuilt each time: spawning the
    workers costs more than the probes themselves once the filesystem is
    fast. The threads are joined by concurrent.futures' own atexit hook, so
    there is no lifecycle to manage here.
    """
    global _io_pool
    with _io_pool_lock:
        if _io_pool is None:
            _io_pool = ThreadPoolExecutor(
                max_workers=_MAX_IO_WORKERS, thread_name_prefix="alf-io"
            )
        return _io_pool


def map_io(fn, items: list) -> list:
    """Apply *fn* to *items*, concurrently when there is enough work to justify it.

    The probes are independent and block on filesystem latency, so on a
    cluster filesystem the fan-out dominates: at ~5 ms per operation this
    turns a 32-row refresh from ~200 ms into ~15 ms. On a local disk the pool
    is pure overhead, but well under a millisecond either way.

    *fn* must not itself call map_io: the pool is shared and finite, so a
    nested call could wait on a worker that never frees.
    """
    if len(items) < _MIN_FANOUT:
        return [fn(item) for item in items]
    return list(_get_io_pool().map(fn, items))
