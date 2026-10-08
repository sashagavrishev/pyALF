"""Tests for the data.h5 bin readers in py_alf.bins."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import pytest

from py_alf.simulation import Simulation


@pytest.fixture(autouse=True)
def _clear_bin_caches():
    """Keep the module-level bin caches from leaking between tests."""
    from py_alf import bins

    for cache in (bins._bin_cache, bins._bin_stat, bins._bin_read_failures):
        cache.clear()
    bins._bin_final.clear()
    yield


# --- _bin_count stat gate ---


def _write_bins(path: Path, n_bins: int) -> None:
    """Write a data.h5 holding *n_bins* bins of the counting observable."""
    import h5py
    import numpy as np

    with h5py.File(path, "w") as f:
        f.create_dataset("Ener_scal/obser", data=np.zeros((n_bins, 1)))


def _settle(path: Path) -> None:
    """Age a file's mtime past the settle window so its reading may be cached.

    A just-written file is deliberately not cached (a coarse-mtime filesystem
    could record a further change in the same tick), so tests that exercise the
    cache have to represent a file whose writer has moved on.
    """
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 5_000_000_000))


def _bin_count_sim(sim_dir: Path):
    sim = MagicMock()
    sim.__class__ = Simulation
    sim.sim_dir = str(sim_dir)
    return sim


def test_bin_count_skips_reopen_when_file_unchanged(tmp_path):
    """An unchanged data.h5 is served from cache without an h5py open."""
    from py_alf.bins import _bin_count

    _write_bins(tmp_path / "data.h5", 5)
    _settle(tmp_path / "data.h5")
    sim = _bin_count_sim(tmp_path)

    assert _bin_count(sim, refresh=True) == 5

    # A second refresh must not reach h5py: the file has not moved.
    with patch("h5py.File", side_effect=AssertionError("data.h5 re-opened")):
        assert _bin_count(sim, refresh=True) == 5


def test_bin_count_rereads_when_file_changes(tmp_path):
    """A new bin landing changes (mtime, size), so the count is re-read."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 5)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 5

    _write_bins(h5, 9)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
    assert _bin_count(sim, refresh=True) == 9


def test_bin_count_force_bypasses_stat_gate(tmp_path):
    """force=True re-reads even when the stat signature is unchanged."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 5)
    _settle(h5)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 5

    with patch("h5py.File", side_effect=AssertionError("should not be re-opened")):
        assert _bin_count(sim, refresh=True) == 5

    # Same signature, but the user asked for a real read.
    reads: list[int] = []
    real_file = h5py.File

    def _counting_open(*args, **kwargs):
        reads.append(1)
        return real_file(*args, **kwargs)

    with patch("h5py.File", side_effect=_counting_open):
        assert _bin_count(sim, refresh=True, force=True) == 5
    assert reads == [1]


def test_bin_count_does_not_cache_signature_of_midwrite_zero(tmp_path):
    """A mid-write 0 keeps the old count and is not frozen by the stat gate."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 7)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 7

    # ALF truncates and rewrites data.h5 between bins; catch it holding 0 bins.
    _write_bins(h5, 0)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
    assert _bin_count(sim, refresh=True) == 7, "transient 0 must not overwrite"

    # The signature of that mid-write state must not have been recorded, or the
    # real count would never be picked up again.
    _write_bins(h5, 8)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 20))
    assert _bin_count(sim, refresh=True) == 8


# --- _bin_count mid-write handling ---


def _truncate(path: Path, fraction: float = 0.9) -> None:
    """Shorten data.h5 so its superblock disagrees with its size, as a
    reader racing ALF's writer would observe."""
    size = path.stat().st_size
    with open(path, "r+b") as fh:
        fh.truncate(int(size * fraction))


def test_midwrite_read_keeps_cached_count_and_stays_quiet(tmp_path, caplog):
    """Racing ALF's writer is expected: keep the last count, log nothing loud."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 40

    _truncate(h5)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
    with caplog.at_level(logging.ERROR, logger="py_alf.bins"):
        assert _bin_count(sim, refresh=True) == 40, "must fall back to cached count"
    assert caplog.records == [], "a mid-write must not be logged as an error"


def test_midwrite_error_is_reported_once_it_stops_being_transient(tmp_path, caplog):
    """A file that keeps failing is real damage and must surface."""
    from py_alf.bins import _MIDWRITE_LOG_AFTER, _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 40
    _truncate(h5)

    with caplog.at_level(logging.ERROR, logger="py_alf.bins"):
        for _ in range(_MIDWRITE_LOG_AFTER - 1):
            os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
            _bin_count(sim, refresh=True)
        assert caplog.records == [], "still within the transient window"

        os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
        _bin_count(sim, refresh=True)
    assert len(caplog.records) == 1
    assert "truncated file" in caplog.records[0].getMessage()


def test_midwrite_failure_streak_resets_after_a_good_read(tmp_path, caplog):
    """A recovered file must not carry its old failure count toward the alarm."""
    from py_alf.bins import _MIDWRITE_LOG_AFTER, _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    _bin_count(sim, refresh=True)

    for _ in range(_MIDWRITE_LOG_AFTER - 1):
        _truncate(h5)
        os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
        _bin_count(sim, refresh=True)

    # The writer finishes; the next read succeeds and clears the streak.
    _write_bins(h5, 41)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 20))
    assert _bin_count(sim, refresh=True) == 41

    _truncate(h5)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 30))
    with caplog.at_level(logging.ERROR, logger="py_alf.bins"):
        _bin_count(sim, refresh=True)
    assert caplog.records == [], "streak should have reset after the good read"


def test_midwrite_read_is_retried(tmp_path):
    """A file that settles between attempts is read rather than reported stale."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    _bin_count(sim, refresh=True)

    good = h5.read_bytes()
    _truncate(h5)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))

    real_open = h5py.File
    calls: list[int] = []

    def _settling_open(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # the writer completes before the second attempt
            h5.write_bytes(good)
        return real_open(*args, **kwargs)

    with patch("h5py.File", side_effect=_settling_open):
        assert _bin_count(sim, refresh=True, force=True) == 40
    assert len(calls) >= 2, "a mid-write must be retried, not given up on"


def test_missing_file_is_not_counted_as_a_read_failure(tmp_path, caplog):
    """A sim that has not started yet is not an error."""
    from py_alf.bins import _bin_count

    sim = _bin_count_sim(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="py_alf.bins"):
        assert _bin_count(sim, refresh=True) == 0
    assert caplog.records == []


# --- _bin_count process pool (h5py's phil serializes reads within a process) ---


def test_bin_count_use_process_pool_reads_correctly(tmp_path):
    """use_process_pool=True still reads the right count, via a worker process."""
    from py_alf.bins import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 12)
    sim = _bin_count_sim(tmp_path)

    assert _bin_count(sim, refresh=True, force=True, use_process_pool=True) == 12


def test_bin_count_use_process_pool_missing_file_returns_zero(tmp_path):
    """A chain with no data.h5 yet reads as 0 bins through the process pool too."""
    from py_alf.bins import _bin_count

    sim = _bin_count_sim(tmp_path)  # no data.h5 written

    assert _bin_count(sim, refresh=True, force=True, use_process_pool=True) == 0


def test_bin_count_use_process_pool_many_files_no_cross_contamination(tmp_path):
    """Each dispatch must come back matched to its own file, not another's."""
    from py_alf.bins import _bin_count

    sims = []
    for i in range(6):
        d = tmp_path / f"chain_{i}"
        d.mkdir()
        _write_bins(d / "data.h5", i + 1)
        sims.append(_bin_count_sim(d))

    counts = [
        _bin_count(sim, refresh=True, force=True, use_process_pool=True) for sim in sims
    ]
    assert counts == [1, 2, 3, 4, 5, 6]


def test_read_bin_count_is_a_pure_module_level_function():
    """_read_bin_count must stay a plain, picklable top-level function.

    ProcessPoolExecutor sends the callable to worker processes by reference
    (pickling its qualified name), so turning this into a closure, a bound
    method, or a lambda would break silently the next time it is actually
    dispatched to a worker rather than called in-process by a test.
    """
    from py_alf.bins import _read_bin_count

    assert _read_bin_count.__module__ == "py_alf.bins"
    assert _read_bin_count.__qualname__ == "_read_bin_count"


# --- read_bin_counts: the batched read behind Campaign.status ---


def test_bin_counts_keeps_results_in_caller_order(tmp_path):
    """Chunked pool.map returns results by task; they must map back to their file."""
    from py_alf.bins import read_bin_counts

    paths = []
    for i in range(12):
        d = tmp_path / f"chain_{i}"
        d.mkdir()
        _write_bins(d / "data.h5", i + 1)
        paths.append(str(d / "data.h5"))

    assert read_bin_counts(paths, force=True) == list(range(1, 13))


def test_bin_counts_reports_zero_for_a_missing_file(tmp_path):
    """An unstarted chain has no data.h5; that is 0 bins, not a failure."""
    from py_alf.bins import read_bin_counts

    (tmp_path / "there").mkdir()
    _write_bins(tmp_path / "there" / "data.h5", 7)
    counts = read_bin_counts(
        [
            str(tmp_path / "there" / "data.h5"),
            str(tmp_path / "gone" / "data.h5"),
            str(tmp_path / "there" / "data.h5"),
        ],
        force=True,
    )
    assert counts == [7, 0, 7]


def test_bin_counts_skips_the_read_for_an_unchanged_file(tmp_path):
    """The (mtime, size) short-circuit is what keeps a repeat check cheap."""
    from py_alf.bins import read_bin_counts

    d = tmp_path / "chain"
    d.mkdir()
    _write_bins(d / "data.h5", 9)
    _settle(d / "data.h5")
    path = str(d / "data.h5")

    assert read_bin_counts([path]) == [9]
    with patch(
        "py_alf.bins._read_bin_count",
        side_effect=AssertionError("re-read an unchanged file"),
    ):
        assert read_bin_counts([path]) == [9]


def test_bin_counts_sees_a_file_that_grew(tmp_path):
    """A chain that ran more bins must not be masked by the previous reading."""
    from py_alf.bins import read_bin_counts

    d = tmp_path / "grower"
    d.mkdir()
    _write_bins(d / "data.h5", 5)
    _settle(d / "data.h5")
    path = str(d / "data.h5")
    assert read_bin_counts([path]) == [5]

    _write_bins(d / "data.h5", 25)
    _settle(d / "data.h5")
    assert read_bin_counts([path]) == [25]
