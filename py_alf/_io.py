"""Shared thread pool for filesystem probes over many simulation directories."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor

# The probes mostly wait on a networked filesystem, but a login node is shared,
# so the pool follows the core count, clamped to [8, 32]. Below _MIN_FANOUT
# items starting the pool costs more than the overlap saves.
_MAX_IO_WORKERS = min(32, max(8, os.cpu_count() or 16))
_MIN_FANOUT = 3


_io_pool: ThreadPoolExecutor | None = None
_io_pool_lock = threading.Lock()


def _get_io_pool() -> ThreadPoolExecutor:
    """The shared probe pool, created on first use and joined at exit."""
    global _io_pool
    with _io_pool_lock:
        if _io_pool is None:
            _io_pool = ThreadPoolExecutor(
                max_workers=_MAX_IO_WORKERS, thread_name_prefix="alf-io"
            )
        return _io_pool


def map_io(fn, items: list) -> list:
    """Apply *fn* to *items*, through the shared pool when there are enough.

    *fn* must not itself call map_io: the pool is finite, so a nested call
    could wait on a worker that never frees.
    """
    if len(items) < _MIN_FANOUT:
        return [fn(item) for item in items]
    return list(_get_io_pool().map(fn, items))
