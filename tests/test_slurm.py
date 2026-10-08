"""Tests for SLURM job state, timeout detection and logs in py_alf.slurm."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from py_alf.slurm import _sanitise_nodelist, job_log


@pytest.fixture(autouse=True)
def _clear_status_caches():
    """Keep the module-level status caches from leaking between tests."""
    from py_alf import slurm

    slurm._terminal_status_cache.clear()
    slurm._submitit_timeout_cache.clear()
    yield


# --- log helpers ---


def test_find_job_log_submitit(tmp_path):
    """job_log returns the submitit-named log when submit_dir is provided."""
    submit_dir = tmp_path / "logs"
    submit_dir.mkdir()
    log_file = submit_dir / "42_0_0_log.out"
    log_file.write_text("output")

    result = job_log("42_0", submit_dir=submit_dir)
    assert result == log_file


def test_find_job_log_falls_back_to_legacy(tmp_path):
    """job_log falls back to the job-*.log glob when no submitit log exists."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    legacy_log = sim_dir / "job-42.log"
    legacy_log.write_text("output")

    result = job_log("42", root_dir=[str(sim_dir)])
    assert result == legacy_log


# --- job_states parent-ID queries ---


def _mock_subprocess(stdout: str):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = 0
    return patch("py_alf.slurm.subprocess.run", return_value=result)


def test_get_slurm_status_bulk_queries_parent_id_for_array_tasks():
    """squeue is called with the array parent ID, not individual task IDs."""
    from py_alf.slurm import job_states

    squeue_output = (
        "99000 99000_0 COMPLETED 01:00:00 node01\n"
        "99000 99000_1 RUNNING   00:30:00 node02\n"
    )
    with _mock_subprocess(squeue_output) as mock_run:
        result = job_states(["99000_0", "99000_1"])

    first_cmd = mock_run.call_args_list[0][0][0]
    j_arg = first_cmd[first_cmd.index("-j") + 1]
    queried = j_arg.split(",")
    assert "99000" in queried
    assert "99000_0" not in queried
    assert "99000_1" not in queried

    assert result["99000_0"]["status"] == "COMPLETED"
    assert result["99000_1"]["status"] == "RUNNING"


def test_get_slurm_status_bulk_non_array_job_passed_through():
    """Non-array job IDs are forwarded to squeue unchanged."""
    from py_alf.slurm import job_states

    squeue_output = "77777 77777 PENDING 0:00 (Priority)\n"
    with _mock_subprocess(squeue_output) as mock_run:
        job_states(["77777"])

    first_cmd = mock_run.call_args_list[0][0][0]
    j_arg = first_cmd[first_cmd.index("-j") + 1]
    assert "77777" in j_arg.split(",")


def test_sacct_fallback_filters_by_job_id():
    """The sacct fallback asks for specific jobs, never the whole day's history."""
    from py_alf.slurm import _job_states_sacct

    sacct_output = "88000_0|COMPLETED|01:00:00|node01\n"
    with _mock_subprocess(sacct_output) as mock_run:
        _job_states_sacct(["88000_0"])

    cmd = mock_run.call_args_list[0][0][0]
    assert cmd[0] == "sacct"
    assert "-j" in cmd, "sacct must be filtered by job id"
    assert "88000" in cmd[cmd.index("-j") + 1].split(",")


def test_sacct_fallback_ignores_substep_rows():
    """sacct's .batch/.extern sub-steps must not leak into the status map."""
    from py_alf.slurm import _job_states_sacct

    sacct_output = (
        "88001 COMPLETED 01:00:00 node01\n"
        "88001.batch FAILED 01:00:00 node01\n"
        "88001.extern COMPLETED 01:00:00 node01\n"
    )
    with _mock_subprocess(sacct_output):
        result = _job_states_sacct(["88001"])

    assert set(result) == {"88001"}
    assert result["88001"]["status"] == "COMPLETED"


def test_terminal_status_is_cached_and_not_requeried():
    """A finished job is served from cache, sparing SLURM a query per refresh."""
    from py_alf.slurm import job_states

    squeue_output = "99100 99100 RUNNING 00:30:00 node07\n"
    with _mock_subprocess(squeue_output):
        first = job_states(["99100"])
    assert first["99100"]["status"] == "RUNNING"

    # Still RUNNING → not cached, so the next refresh must query again.
    with _mock_subprocess(squeue_output) as mock_run:
        job_states(["99100"])
    assert mock_run.call_count > 0

    # Now it completes; squeue no longer lists it and sacct reports the state.
    with _mock_subprocess("99100 COMPLETED 01:00:00 node07\n"):
        done = job_states(["99100"])
    assert done["99100"]["status"] == "COMPLETED"

    # Terminal states are immutable — no further subprocess calls.
    with _mock_subprocess("") as mock_run:
        cached = job_states(["99100"])
    assert mock_run.call_count == 0
    assert cached["99100"]["status"] == "COMPLETED"


def test_terminal_cache_still_queries_unfinished_jobs():
    """A mixed session queries only the jobs that can still change."""
    from py_alf.slurm import job_states

    with _mock_subprocess("99200 COMPLETED 01:00:00 node01\n"):
        job_states(["99200"])

    squeue_output = "99201 99201 RUNNING 00:10:00 node02\n"
    with _mock_subprocess(squeue_output) as mock_run:
        result = job_states(["99200", "99201"])

    queried = mock_run.call_args_list[0][0][0]
    j_arg = queried[queried.index("-j") + 1].split(",")
    assert j_arg == ["99201"], "cached terminal job must be excluded from the query"
    assert result["99200"]["status"] == "COMPLETED"
    assert result["99201"]["status"] == "RUNNING"


# --- is_timeout ---


def test_submitit_timeout_detected_from_log(tmp_path):
    from py_alf.slurm import is_timeout

    (tmp_path / "77_0_log.out").write_text("... this job is timed-out ...")
    assert is_timeout("77", tmp_path) is True


def test_submitit_missing_log_is_not_cached(tmp_path):
    """A log that lags the job's state change must not pin "not timed out"."""
    from py_alf.slurm import is_timeout

    assert is_timeout("78", tmp_path) is False

    # The log lands on a later refresh; the timeout must now be reported.
    (tmp_path / "78_0_log.out").write_text("... this job is timed-out ...")
    assert is_timeout("78", tmp_path) is True


def test_submitit_present_log_is_cached(tmp_path):
    """Once read, the log is not read again -- its verdict cannot change."""
    from py_alf.slurm import is_timeout

    log = tmp_path / "79_0_log.out"
    log.write_text("nothing interesting")
    assert is_timeout("79", tmp_path) is False

    with patch.object(Path, "read_text", side_effect=AssertionError("re-read")):
        assert is_timeout("79", tmp_path) is False


# --- _sanitise_nodelist ---


def test_sanitise_nodelist_returns_real_node():
    assert _sanitise_nodelist("compute01") == "compute01"
    assert _sanitise_nodelist("node[001-004]") == "node[001-004]"


def test_sanitise_nodelist_rejects_pending_reason():
    assert _sanitise_nodelist("(Priority)") is None
    assert _sanitise_nodelist("(Resources)") is None
    assert _sanitise_nodelist("(None)") is None


def test_sanitise_nodelist_rejects_sacct_none_literal():
    assert _sanitise_nodelist("None") is None
    assert _sanitise_nodelist("N/A") is None
    assert _sanitise_nodelist("none") is None


def test_sanitise_nodelist_rejects_empty_and_none():
    assert _sanitise_nodelist("") is None
    assert _sanitise_nodelist(None) is None
