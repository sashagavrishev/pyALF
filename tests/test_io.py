"""Tests for the shared filesystem-probe pool in py_alf._io."""

import threading

from py_alf._io import _MIN_FANOUT, map_io


def test_map_io_runs_small_inputs_without_a_pool():
    """Below the fan-out threshold the work runs inline — no thread spawn."""

    caller = threading.current_thread()
    threads = map_io(lambda i: threading.current_thread(), list(range(_MIN_FANOUT - 1)))
    assert all(t is caller for t in threads)


def test_map_io_fans_out_large_inputs_and_preserves_order():
    """Above the threshold work is spread across threads but stays ordered."""

    n = max(_MIN_FANOUT, 8)
    barrier = threading.Barrier(n, timeout=5)

    def work(i):
        # Deadlocks unless the items really run concurrently.
        barrier.wait()
        return i * 2

    assert map_io(work, list(range(n))) == [i * 2 for i in range(n)]
