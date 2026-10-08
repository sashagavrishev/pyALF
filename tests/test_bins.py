"""Tests for the data.h5 bin readers in py_alf.bins."""

import logging
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from py_alf.bins import _read_bin_count, read_bin_count, read_bin_counts
from py_alf.simulation import Simulation


def _write_bins(path: Path, n_bins: int) -> None:
    """Write a data.h5 holding *n_bins* bins of the counting observable."""
    with h5py.File(path, "w") as f:
        f.create_dataset("Ener_scal/obser", data=np.zeros((n_bins, 1)))


def _truncate(path: Path, fraction: float = 0.9) -> None:
    """Shorten data.h5 so its superblock disagrees with its size, as a
    reader racing ALF's writer would observe."""
    size = path.stat().st_size
    with open(path, "r+b") as fh:
        fh.truncate(int(size * fraction))


# --- read_bin_count ---


def test_read_bin_count_reads_the_counting_observable(tmp_path):
    _write_bins(tmp_path / "data.h5", 12)
    assert read_bin_count(str(tmp_path / "data.h5")) == 12


def test_read_bin_count_is_zero_for_an_absent_observable(tmp_path):
    _write_bins(tmp_path / "data.h5", 12)
    assert read_bin_count(str(tmp_path / "data.h5"), "Kin_scal") == 0


def test_missing_file_is_zero_and_not_a_failure(tmp_path, caplog):
    """A chain that has not started yet is not an error."""
    with caplog.at_level(logging.DEBUG, logger="py_alf.bins"):
        assert read_bin_count(str(tmp_path / "data.h5")) == 0
    assert caplog.records == []


def test_midwrite_read_is_retried(tmp_path):
    """A file that settles between attempts is read rather than reported."""
    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    good = h5.read_bytes()
    _truncate(h5)

    real_open = h5py.File
    calls: list[int] = []

    def _settling_open(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:  # the writer completes before the second attempt
            h5.write_bytes(good)
        return real_open(*args, **kwargs)

    with patch("h5py.File", side_effect=_settling_open):
        assert read_bin_count(str(h5)) == 40
    assert len(calls) == 2


def test_a_persistent_midwrite_is_a_warning_not_an_error(tmp_path, caplog):
    """Racing ALF's writer is expected; the count is 0 for this read only."""
    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    _truncate(h5)

    with caplog.at_level(logging.WARNING, logger="py_alf.bins"):
        assert read_bin_count(str(h5)) == 0
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "truncated file" in caplog.records[0].getMessage()


def test_a_damaged_file_is_reported_as_an_error(tmp_path, caplog):
    h5 = tmp_path / "data.h5"
    with h5py.File(h5, "w") as f:
        f.create_group("Ener_scal")  # no obser dataset

    with caplog.at_level(logging.WARNING, logger="py_alf.bins"):
        assert read_bin_count(str(h5)) == 0
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def test_simulation_bin_count_reads_its_data_file(tmp_path):
    _write_bins(tmp_path / "data.h5", 7)
    sim = object.__new__(Simulation)
    sim.sim_dir = str(tmp_path)
    assert sim.bin_count() == 7


# --- read_bin_counts: the batched read behind Campaign.status ---


def test_read_bin_count_is_a_pure_module_level_function():
    """The process pool sends the reader to workers by qualified name."""
    assert _read_bin_count.__module__ == "py_alf.bins"
    assert _read_bin_count.__qualname__ == "_read_bin_count"


def test_bin_counts_keeps_results_in_caller_order(tmp_path):
    """Chunked pool.map returns results by task; they must map back to their file."""
    paths = []
    for i in range(12):
        d = tmp_path / f"chain_{i}"
        d.mkdir()
        _write_bins(d / "data.h5", i + 1)
        paths.append(str(d / "data.h5"))

    assert read_bin_counts(paths) == list(range(1, 13))


def test_bin_counts_reports_zero_for_a_missing_file(tmp_path):
    """An unstarted chain has no data.h5; that is 0 bins, not a failure."""
    (tmp_path / "there").mkdir()
    _write_bins(tmp_path / "there" / "data.h5", 7)
    there = str(tmp_path / "there" / "data.h5")
    gone = str(tmp_path / "gone" / "data.h5")
    assert read_bin_counts([there, gone, there]) == [7, 0, 7]


def test_bin_counts_reports_each_file_to_the_progress_hook(tmp_path):
    _write_bins(tmp_path / "data.h5", 3)
    steps: list[int] = []
    read_bin_counts([str(tmp_path / "data.h5")] * 5, on_progress=steps.append)
    assert steps == [1] * 5


def test_bin_counts_of_nothing_starts_no_pool():
    with patch("py_alf.bins._get_process_pool", side_effect=AssertionError):
        assert read_bin_counts([]) == []
