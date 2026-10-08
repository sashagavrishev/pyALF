"""Tests for ClusterSubmitter in py_alf.cluster_submission."""

import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import pytest

from py_alf.cluster_submission import (
    _MIN_FANOUT,
    ClusterSubmitter,
    _exec_alf_binary,
    _find_job_log,
    _map_io,
    _normalise_partition_spec,
    _parse_mem_gb,
    _parse_slurm_time_hours,
    _run_alf,
    _sanitise_nodelist,
    detect_partition_rules,
)
from py_alf.simulation import Simulation

_RULES = {"short": 8, "long": 168}


@pytest.fixture(autouse=True)
def _clear_module_caches():
    """Keep the module-level status/bin caches from leaking between tests."""
    from py_alf import cluster_submission as _cs

    for cache in (
        _cs._terminal_status_cache,
        _cs._bin_cache,
        _cs._bin_stat,
        _cs._bin_read_failures,
        _cs._submitit_timeout_cache,
    ):
        cache.clear()
    _cs._bin_final.clear()
    yield


# --- __init__ validation ---


def test_init_defaults():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules=_RULES)
    assert cs.executor == "slurm"
    assert cs.submit_dir.name == ".pyalf"
    assert cs.submit_dir.is_absolute()
    assert cs.slurm_mem == "2G"
    # partition_rules is normalised to PartitionSpec dicts at construction time
    assert cs.partition_rules == {
        "short": {"max_hours": 8.0},
        "long": {"max_hours": 168.0},
    }
    assert cs.job_name is None
    assert cs.mail_type is None
    assert cs.wckey is None
    assert cs.stderr_to_stdout is False
    assert cs.slurm_kwargs == {}


def test_init_custom():
    cs = ClusterSubmitter(
        "slurm",
        submit_dir="/tmp/logs",
        slurm_mem="8G",
        partition_rules={"gpu": 24},
        slurm_extra="foo",
    )
    assert cs.submit_dir == Path("/tmp/logs").resolve()
    assert cs.executor == "slurm"
    assert cs.slurm_mem == "8G"
    assert cs.partition_rules == {"gpu": {"max_hours": 24.0}}
    assert cs.slurm_kwargs == {"slurm_extra": "foo"}


def test_init_requires_slurm_mem():
    with pytest.raises(ValueError, match="slurm_mem"):
        ClusterSubmitter(partition_rules=_RULES)


def test_init_requires_partition_rules():
    with pytest.raises(ValueError, match="partition_rules"):
        ClusterSubmitter(slurm_mem="2G")


def test_init_rejects_invalid_executor():
    with pytest.raises(ValueError, match="executor"):
        ClusterSubmitter("chronos", slurm_mem="2G", partition_rules=_RULES)


def test_init_local_executor_no_slurm_params():
    cs = ClusterSubmitter("local", submit_dir="/tmp/logs")
    assert cs.executor == "local"
    assert cs.slurm_mem is None
    assert cs.partition_rules is None


def test_init_debug_executor_no_slurm_params():
    cs = ClusterSubmitter("debug")
    assert cs.executor == "debug"


def test_init_local_rejects_slurm_mem():
    with pytest.raises(ValueError, match="slurm_mem"):
        ClusterSubmitter("local", slurm_mem="4G")


def test_init_local_rejects_partition_rules():
    with pytest.raises(ValueError, match="partition_rules"):
        ClusterSubmitter("local", partition_rules=_RULES)


def test_init_local_rejects_slurm_prefixed_kwargs():
    with pytest.raises(ValueError, match="slurm_constraint"):
        ClusterSubmitter("local", slurm_constraint="gpu")


def test_init_local_rejects_mail_type():
    with pytest.raises(ValueError, match="mail_type"):
        ClusterSubmitter("local", mail_type="END")


def test_init_local_rejects_wckey():
    with pytest.raises(ValueError, match="wckey"):
        ClusterSubmitter("local", wckey="my-project")


def test_init_slurm_extra_fields():
    cs = ClusterSubmitter(
        slurm_mem="4G",
        partition_rules=_RULES,
        job_name="my-job",
        mail_type="END",
        wckey="proj-key",
        stderr_to_stdout=True,
    )
    assert cs.job_name == "my-job"
    assert cs.mail_type == "END"
    assert cs.wckey == "proj-key"
    assert cs.stderr_to_stdout is True


def test_init_job_name_and_stderr_accepted_for_local():
    cs = ClusterSubmitter("local", job_name="my-job", stderr_to_stdout=True)
    assert cs.job_name == "my-job"
    assert cs.stderr_to_stdout is True


# --- _select_partition ---


def test_select_partition_picks_tightest_fit():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": 8, "long": 168})
    assert cs._select_partition(1) == "short"
    assert cs._select_partition(8) == "short"
    assert cs._select_partition(9) == "long"
    assert cs._select_partition(168) == "long"


def test_select_partition_raises_when_no_fit():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": 8})
    with pytest.raises(ValueError, match="9h"):
        cs._select_partition(9)


# --- submit ---


def test_submit_type_error():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules=_RULES)
    with pytest.raises(TypeError, match="Expected Simulation"):
        cs.submit("not_a_simulation")


def test_submit_empty_list():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules=_RULES)
    result = cs.submit([])
    assert result == []


def test_submit_single_sim(tmp_path):
    """Single simulation is submitted as a one-element job list."""
    sim = _make_mock_sim(tmp_path / "sim0")

    mock_job = MagicMock()
    mock_job.job_id = "42"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        jobs = cs.submit(sim)

    assert jobs == [mock_job]
    assert (tmp_path / "sim0" / "jobid.txt").read_text() == "42"
    mock_executor.return_value.submit.assert_called_once_with(_run_alf, sim)


def test_submit_multiple_sims_uses_map_array(tmp_path):
    """Multiple simulations are submitted via map_array."""
    sims = [_make_mock_sim(tmp_path / f"sim{i}") for i in range(3)]

    mock_jobs = [MagicMock(job_id=f"99_{i}") for i in range(3)]

    with _patch_submitit(mock_jobs, multi=True) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        jobs = cs.submit(sims)

    assert jobs == mock_jobs
    for i, _sim in enumerate(sims):
        assert (tmp_path / f"sim{i}" / "jobid.txt").read_text() == f"99_{i}"
    mock_executor.return_value.map_array.assert_called_once_with(_run_alf, sims)


def test_submit_heterogeneous_resources_raises(tmp_path):
    """Array submission raises ValueError when sims have different resource shapes."""
    sim_a = _make_mock_sim(tmp_path / "sim0")
    sim_b = _make_mock_sim(tmp_path / "sim1")
    sim_b.n_omp = 8  # differs from sim_a's n_omp=4

    with pytest.raises(ValueError, match="n_omp"):
        ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        ).submit([sim_a, sim_b])


def test_submit_skips_running_job(tmp_path):
    """A simulation whose jobid.txt reports RUNNING is skipped."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    with patch(
        "py_alf.cluster_submission._get_slurm_status_sacct",
        return_value={"status": "RUNNING"},
    ):
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        jobs = cs.submit(sim)

    assert jobs == []


def test_submit_skips_pending_job(tmp_path):
    """A simulation whose jobid.txt reports PENDING is skipped."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    with patch(
        "py_alf.cluster_submission._get_slurm_status_sacct",
        return_value={"status": "PENDING"},
    ):
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        jobs = cs.submit(sim)

    assert jobs == []


def test_submit_skips_leftover_running_file_by_default(tmp_path):
    """A stale RUNNING file keeps the sim out, without prompting."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "RUNNING").write_text("")

    with _patch_submitit(MagicMock()) as mock_executor:
        cs = ClusterSubmitter("local", submit_dir=tmp_path / "logs")
        assert cs.submit(sim) == []

    mock_executor.return_value.submit.assert_not_called()
    assert (tmp_path / "sim0" / "RUNNING").exists()


def test_submit_removes_leftover_running_file_when_asked(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "RUNNING").write_text("")
    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job):
        cs = ClusterSubmitter("local", submit_dir=tmp_path / "logs")
        assert cs.submit(sim, stale_running="remove") == [mock_job]

    assert not (tmp_path / "sim0" / "RUNNING").exists()


def test_submit_local_does_not_check_slurm_status(tmp_path):
    """Local executor never calls sacct even if a jobid.txt exists."""
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    mock_job = MagicMock()
    mock_job.job_id = "local_0"

    with (
        _patch_submitit(mock_job) as _mock_executor,
        patch("py_alf.cluster_submission._get_slurm_status_sacct") as mock_sacct,
    ):
        cs = ClusterSubmitter("local", submit_dir=tmp_path / "logs")
        jobs = cs.submit(sim)

    mock_sacct.assert_not_called()
    assert jobs == [mock_job]


def test_submit_executor_parameters_auto_selects_partition(tmp_path):
    """Executor is configured with correct SLURM parameters, partition auto-selected."""
    sim = _make_mock_sim(tmp_path / "sim0")
    # CPU_MAX=2 → timeout_hours=2 → fits "short" (≤8h)
    sim.sim_dict = {"CPU_MAX": 2}

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="4G",
            partition_rules={"short": 8, "long": 168},
        )
        cs.submit(sim, job_properties={"timeout_min": 120})

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_partition"] == "short"
    assert call_kwargs["slurm_mem"] == "4G"
    assert call_kwargs["timeout_min"] == 120
    assert call_kwargs["cpus_per_task"] == sim.n_omp
    assert call_kwargs["tasks_per_node"] == 1  # non-MPI sim


def test_submit_auto_selects_long_partition(tmp_path):
    """Jobs with CPU_MAX exceeding 'short' limit are assigned to 'long'."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 24}  # 24h → exceeds short (8h), fits long (168h)

    mock_job = MagicMock()
    mock_job.job_id = "2"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="4G",
            partition_rules={"short": 8, "long": 168},
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_partition"] == "long"


def test_submit_job_properties_slurm_time_selects_partition(tmp_path):
    """A per-call slurm_time beats the instance one for partition and --time alike."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 2}

    mock_job = MagicMock()
    mock_job.job_id = "3"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="4G",
            partition_rules={"short": 8, "long": 168},
            slurm_time=60,
        )
        cs.submit(sim, job_properties={"slurm_time": "24:00:00"})

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_partition"] == "long"
    assert call_kwargs["slurm_time"] == 24 * 60


def test_submit_job_name_overrides_ham_name(tmp_path):
    """Explicit job_name takes precedence over sim.ham_name."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock(job_id="1")

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
            job_name="custom-name",
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["name"] == "custom-name"


def test_submit_default_name_is_ham_name(tmp_path):
    """When job_name is None the hamiltonian name is used."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock(job_id="1")

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["name"] == sim.ham_name


def test_submit_mail_type_and_wckey_in_slurm_params(tmp_path):
    """mail_type and wckey appear in SLURM update_parameters call."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock(job_id="1")

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
            mail_type="END",
            wckey="proj-key",
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_mail_type"] == "END"
    assert call_kwargs["slurm_wckey"] == "proj-key"


def test_submit_stderr_to_stdout_forwarded(tmp_path):
    """stderr_to_stdout is forwarded for both slurm and local executors."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock(job_id="1")

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
            stderr_to_stdout=True,
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["stderr_to_stdout"] is True


def test_submit_stderr_to_stdout_false_not_forwarded(tmp_path):
    """stderr_to_stdout=False (default) is not included in params."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock(job_id="1")

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert "stderr_to_stdout" not in call_kwargs


def test_submit_local_executor_omits_slurm_params(tmp_path):
    """Local executor parameters do not include slurm_mem or slurm_partition."""
    sim = _make_mock_sim(tmp_path / "sim0")

    mock_job = MagicMock()
    mock_job.job_id = "local_0"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter("local", submit_dir=tmp_path / "logs")
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert "slurm_mem" not in call_kwargs
    assert "slurm_partition" not in call_kwargs


def test_submit_uses_correct_executor_cluster_arg(tmp_path):
    """AutoExecutor is called with the cluster matching the executor setting."""
    sim = _make_mock_sim(tmp_path / "sim0")
    mock_job = MagicMock()
    mock_job.job_id = "d0"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter("debug", submit_dir=tmp_path / "logs")
        cs.submit(sim)

    mock_executor.assert_called_once()
    _, ctor_kwargs = mock_executor.call_args
    assert ctor_kwargs.get("cluster") == "debug"


# --- log helpers ---


def test_find_job_log_submitit(tmp_path):
    """_find_job_log returns the submitit-named log when submit_dir is provided."""
    submit_dir = tmp_path / "logs"
    submit_dir.mkdir()
    log_file = submit_dir / "42_0_0_log.out"
    log_file.write_text("output")

    result = _find_job_log("42_0", submit_dir=submit_dir)
    assert result == log_file


def test_find_job_log_falls_back_to_legacy(tmp_path):
    """_find_job_log falls back to the job-*.log glob when no submitit log exists."""
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    legacy_log = sim_dir / "job-42.log"
    legacy_log.write_text("output")

    result = _find_job_log("42", root_dir=[str(sim_dir)])
    assert result == legacy_log


# --- helpers ---


def _make_mock_sim(sim_dir: Path):
    sim_dir.mkdir(parents=True, exist_ok=True)
    sim = MagicMock()
    sim.__class__ = Simulation  # makes isinstance(sim, Simulation) return True
    sim.sim_dir = str(sim_dir)
    sim.ham_name = "Hubbard"
    sim.n_omp = 4
    sim.n_mpi = 1
    sim.mpi = False
    sim.sim_dict = {"CPU_MAX": 2}
    return sim


def _patch_submitit(job_or_jobs, multi=False):
    mock_executor = MagicMock()
    if multi:
        mock_executor.return_value.map_array.return_value = job_or_jobs
    else:
        mock_executor.return_value.submit.return_value = job_or_jobs
    return patch("py_alf.cluster_submission.submitit.AutoExecutor", mock_executor)


# --- _parse_mem_gb ---


def test_parse_mem_gb_gigabytes():
    assert _parse_mem_gb("8G") == pytest.approx(8.0)


def test_parse_mem_gb_megabytes():
    assert _parse_mem_gb("512M") == pytest.approx(0.5)


def test_parse_mem_gb_terabytes():
    assert _parse_mem_gb("1T") == pytest.approx(1024.0)


def test_parse_mem_gb_kilobytes():
    # 1024 KB = 1 MB = 1/1024 GB
    assert _parse_mem_gb("1024K") == pytest.approx(1.0 / 1024)


def test_parse_mem_gb_no_suffix_is_megabytes():
    # SLURM default: no suffix means MB
    assert _parse_mem_gb("256") == pytest.approx(256 / 1024)


def test_parse_mem_gb_empty_raises():
    with pytest.raises(ValueError):
        _parse_mem_gb("")


def test_parse_mem_gb_bad_value_raises():
    with pytest.raises(ValueError):
        _parse_mem_gb("badval")


# --- PartitionSpec normalisation ---


def test_normalise_plain_float_returns_max_hours_dict():
    spec = _normalise_partition_spec("short", 2.0)
    assert spec == {"max_hours": 2.0}


def test_normalise_plain_int_returns_float_max_hours():
    spec = _normalise_partition_spec("short", 8)
    assert spec == {"max_hours": 8.0}
    assert isinstance(spec["max_hours"], float)


def test_normalise_full_dict_passes_through():
    spec = _normalise_partition_spec(
        "short", {"max_hours": 2, "max_cpus": 64, "max_mem_gb": 128}
    )
    assert spec["max_hours"] == 2
    assert spec["max_cpus"] == 64
    assert spec["max_mem_gb"] == 128


def test_normalise_partial_dict_max_hours_only():
    spec = _normalise_partition_spec("short", {"max_hours": 8})
    assert spec == {"max_hours": 8}


def test_normalise_missing_max_hours_raises():
    with pytest.raises(ValueError, match="max_hours"):
        _normalise_partition_spec("short", {"max_cpus": 64})


def test_normalise_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown"):
        _normalise_partition_spec("short", {"max_hours": 8, "bogus_key": 1})


def test_init_accepts_plain_float_spec():
    """Plain float is backward-compatible; ClusterSubmitter must accept it."""
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": 2.0})
    assert "short" in cs.partition_rules


def test_init_accepts_plain_int_spec():
    """Plain int is backward-compatible; ClusterSubmitter must accept it."""
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": 8})
    assert "short" in cs.partition_rules


def test_init_accepts_full_dict_spec():
    cs = ClusterSubmitter(
        slurm_mem="2G",
        partition_rules={"short": {"max_hours": 2, "max_cpus": 64, "max_mem_gb": 128}},
    )
    assert "short" in cs.partition_rules


def test_init_accepts_partial_dict_spec():
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": {"max_hours": 8}})
    assert "short" in cs.partition_rules


def test_init_rejects_dict_missing_max_hours():
    with pytest.raises(ValueError, match="max_hours"):
        ClusterSubmitter(slurm_mem="2G", partition_rules={"short": {"max_cpus": 64}})


def test_init_rejects_dict_with_unknown_key():
    with pytest.raises(ValueError):
        ClusterSubmitter(
            slurm_mem="2G",
            partition_rules={"short": {"max_hours": 8, "bogus_key": 1}},
        )


# --- _select_partition still works after normalisation ---


def test_select_partition_with_plain_float_rules():
    """Plain float specs are still usable for partition selection."""
    cs = ClusterSubmitter(slurm_mem="2G", partition_rules={"short": 2, "long": 48})
    assert cs._select_partition(1) == "short"
    assert cs._select_partition(2) == "short"
    assert cs._select_partition(3) == "long"
    assert cs._select_partition(48) == "long"


def test_select_partition_with_dict_rules():
    """Dict specs with max_hours are usable for partition selection."""
    cs = ClusterSubmitter(
        slurm_mem="2G",
        partition_rules={
            "short": {"max_hours": 2, "max_cpus": 64},
            "long": {"max_hours": 48, "max_cpus": 128},
        },
    )
    assert cs._select_partition(1) == "short"
    assert cs._select_partition(2) == "short"
    assert cs._select_partition(3) == "long"
    assert cs._select_partition(48) == "long"


# --- _check_node_fit ---


def test_check_node_fit_cpu_within_limit(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.n_omp = 4
    sim.mpi = False
    cs = ClusterSubmitter(
        slurm_mem="8G",
        partition_rules={"short": {"max_hours": 8, "max_cpus": 8, "max_mem_gb": 16}},
    )
    # n_omp=4 <= max_cpus=8 => no exception
    cs._check_node_fit(sim, "short")


def test_check_node_fit_cpu_exceeds_raises(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.n_omp = 16
    sim.mpi = False
    cs = ClusterSubmitter(
        slurm_mem="8G",
        partition_rules={"short": {"max_hours": 8, "max_cpus": 8, "max_mem_gb": 16}},
    )
    with pytest.raises(ValueError, match="(?i)cpu"):
        cs._check_node_fit(sim, "short")


def test_check_node_fit_mem_within_limit(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = ClusterSubmitter(
        slurm_mem="8G",
        partition_rules={"short": {"max_hours": 8, "max_mem_gb": 16}},
    )
    # 8 GB <= 16 GB => no exception
    cs._check_node_fit(sim, "short")


def test_check_node_fit_mem_exceeds_raises(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = ClusterSubmitter(
        slurm_mem="32G",
        partition_rules={"short": {"max_hours": 8, "max_mem_gb": 16}},
    )
    with pytest.raises(ValueError, match="(?i)memory"):
        cs._check_node_fit(sim, "short")


def test_check_node_fit_no_limits_no_exception(tmp_path):
    """Plain-float spec has no max_cpus or max_mem_gb, so no exception is raised."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.n_omp = 128
    cs = ClusterSubmitter(
        slurm_mem="512G",
        partition_rules={"short": 8},  # plain float spec => no CPU/mem limits
    )
    cs._check_node_fit(sim, "short")


def test_check_node_fit_slurm_mem_override_exceeds(tmp_path):
    """The slurm_mem keyword overrides the instance default when checking memory."""
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = ClusterSubmitter(
        slurm_mem="4G",  # instance default is within limit
        partition_rules={"short": {"max_hours": 8, "max_mem_gb": 16}},
    )
    # Override with a value that exceeds the limit
    with pytest.raises(ValueError, match="(?i)memory"):
        cs._check_node_fit(sim, "short", slurm_mem="32G")


def test_check_node_fit_slurm_mem_override_within_limit(tmp_path):
    """slurm_mem override within limit should not raise even if instance default is too large."""
    sim = _make_mock_sim(tmp_path / "sim0")
    cs = ClusterSubmitter(
        slurm_mem="32G",  # instance default exceeds limit
        partition_rules={"short": {"max_hours": 8, "max_mem_gb": 16}},
    )
    # Override with a value that is within the limit => no exception
    cs._check_node_fit(sim, "short", slurm_mem="4G")


# --- submit resource validation ---


def test_submit_exceeds_max_cpus_raises_before_executor(tmp_path):
    """Exceeding max_cpus raises ValueError before submitit is ever invoked."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.n_omp = 16  # exceeds max_cpus=8
    sim.mpi = False

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="8G",
            partition_rules={"short": {"max_hours": 8, "max_cpus": 8}},
        )
        with pytest.raises(ValueError, match="(?i)cpu"):
            cs.submit(sim)

    mock_executor.return_value.submit.assert_not_called()
    mock_executor.return_value.map_array.assert_not_called()


def test_submit_exceeds_max_mem_raises_before_executor(tmp_path):
    """Exceeding max_mem_gb raises ValueError before submitit is ever invoked."""
    sim = _make_mock_sim(tmp_path / "sim0")

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="32G",  # exceeds max_mem_gb=16
            partition_rules={"short": {"max_hours": 8, "max_mem_gb": 16}},
        )
        with pytest.raises(ValueError, match="(?i)memory"):
            cs.submit(sim)

    mock_executor.return_value.submit.assert_not_called()
    mock_executor.return_value.map_array.assert_not_called()


# --- MPI use_srun=False ---


def test_submit_mpi_sim_sets_use_srun_false(tmp_path):
    """MPI simulation gets use_srun=False to prevent nested srun conflict."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.mpi = True
    sim.n_mpi = 2
    sim.n_omp = 4

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_use_srun"] is False
    assert "use_srun" not in call_kwargs  # legacy (deprecated) key not passed


def test_submit_non_mpi_sim_has_no_use_srun(tmp_path):
    """Non-MPI simulation must NOT have use_srun in its executor parameters."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.mpi = False

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert "use_srun" not in call_kwargs
    assert "slurm_use_srun" not in call_kwargs


def test_submit_uses_prefixed_additional_parameters(tmp_path):
    """The auto wall-time goes via slurm_additional_parameters, not the
    deprecated unprefixed additional_parameters key."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 1}  # CPU_MAX mode → submit derives a wall time

    mock_job = MagicMock()
    mock_job.job_id = "1"

    with _patch_submitit(mock_job) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert "additional_parameters" not in call_kwargs  # legacy key not passed
    assert "time" in call_kwargs["slurm_additional_parameters"]


# --- _parse_slurm_time_hours ---


@pytest.mark.parametrize(
    "s,expected",
    [
        ("2:00:00", 2.0),
        ("00:30:00", 0.5),
        ("1-00:00:00", 24.0),
        ("14-00:00:00", 336.0),
        ("2-12:00:00", 60.0),
        ("00:10:00", 1 / 6),
        ("30:00", 0.5),  # MM:SS form
        ("120", 2.0 / 60),  # seconds-only form
        ("UNLIMITED", None),
        ("unlimited", None),
        ("INFINITE", None),
        ("NOT_SET", None),
        ("", None),
    ],
)
def test_parse_slurm_time_hours(s, expected):
    result = _parse_slurm_time_hours(s)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, rel=1e-6)


# --- detect_partition_rules ---


def _mock_sinfo(stdout: str):
    """Return a context manager that patches subprocess.run for sinfo calls."""
    result = MagicMock()
    result.stdout = stdout
    result.returncode = 0
    return patch(
        "py_alf.cluster_submission.subprocess.run",
        return_value=result,
    )


_SINFO_TYPICAL = (
    "short|2:00:00|64|257000\n"
    "medium|2-00:00:00|128|515000\n"
    "long|14-00:00:00|128|515000\n"
)


def test_detect_basic():
    """Happy-path: three partitions are detected with correct specs."""
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules()
    assert set(rules) == {"short", "medium", "long"}
    assert rules["short"]["max_hours"] == pytest.approx(2.0)
    assert rules["medium"]["max_hours"] == pytest.approx(48.0)
    assert rules["long"]["max_hours"] == pytest.approx(336.0)
    assert rules["short"]["max_cpus"] == 64
    assert rules["medium"]["max_cpus"] == 128
    # Memory: 257000 MB / 1024 − 2 GB headroom
    assert rules["short"]["max_mem_gb"] == pytest.approx(257000 / 1024 - 2, rel=1e-4)


def test_detect_strips_default_partition_star():
    """The '*' suffix on the default partition name is stripped."""
    sinfo_out = "short*|2:00:00|64|257000\n"
    with _mock_sinfo(sinfo_out):
        rules = detect_partition_rules()
    assert "short" in rules
    assert "short*" not in rules


def test_detect_skips_unlimited():
    """Partitions with UNLIMITED time are excluded."""
    sinfo_out = "short|2:00:00|64|257000\ninfinite|UNLIMITED|128|515000\n"
    with _mock_sinfo(sinfo_out):
        rules = detect_partition_rules()
    assert "short" in rules
    assert "infinite" not in rules


def test_detect_exclude():
    """Named partitions are dropped when listed in exclude."""
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules(exclude=["long", "medium"])
    assert set(rules) == {"short"}


def test_detect_exclude_case_insensitive():
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules(exclude=["LONG", "MEDIUM"])
    assert "long" not in rules
    assert "medium" not in rules


def test_detect_include():
    """Only listed partitions are returned when include is set."""
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules(include=["short", "medium"])
    assert set(rules) == {"short", "medium"}
    assert "long" not in rules


def test_detect_include_case_insensitive():
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules(include=["SHORT"])
    assert "short" in rules


def test_detect_heterogeneous_nodes_takes_minimum():
    """Multiple sinfo lines for the same partition → conservative min values."""
    sinfo_out = (
        "short|2:00:00|128|515000\n"  # larger node group
        "short|2:00:00|64|257000\n"  # smaller node group
    )
    with _mock_sinfo(sinfo_out):
        rules = detect_partition_rules()
    assert rules["short"]["max_cpus"] == 64  # minimum
    assert rules["short"]["max_mem_gb"] == pytest.approx(257000 / 1024 - 2, rel=1e-4)


def test_detect_mem_headroom_applied():
    """mem_headroom_gb is subtracted from the raw memory figure."""
    sinfo_out = "short|2:00:00|64|10240\n"  # exactly 10 GB
    with _mock_sinfo(sinfo_out):
        rules = detect_partition_rules(mem_headroom_gb=1.0)
    assert rules["short"]["max_mem_gb"] == pytest.approx(9.0)


def test_detect_custom_mem_headroom():
    sinfo_out = "short|2:00:00|64|10240\n"
    with _mock_sinfo(sinfo_out):
        rules_2 = detect_partition_rules(mem_headroom_gb=2.0)
        rules_4 = detect_partition_rules(mem_headroom_gb=4.0)
    assert rules_2["short"]["max_mem_gb"] == pytest.approx(8.0)
    assert rules_4["short"]["max_mem_gb"] == pytest.approx(6.0)


def test_detect_sinfo_not_found_raises():
    with (
        patch(
            "py_alf.cluster_submission.subprocess.run",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(RuntimeError, match="sinfo"),
    ):
        detect_partition_rules()


def test_detect_sinfo_timeout_raises():
    import subprocess as _sp

    with (
        patch(
            "py_alf.cluster_submission.subprocess.run",
            side_effect=_sp.TimeoutExpired("sinfo", 10),
        ),
        pytest.raises(RuntimeError, match="timed out"),
    ):
        detect_partition_rules()


def test_detect_no_usable_partitions_raises():
    """All UNLIMITED → RuntimeError."""
    sinfo_out = "bigmem|UNLIMITED|256|1024000\n"
    with (
        _mock_sinfo(sinfo_out),
        pytest.raises(RuntimeError, match="no usable partitions"),
    ):
        detect_partition_rules()


def test_detect_all_excluded_raises():
    with _mock_sinfo(_SINFO_TYPICAL), pytest.raises(RuntimeError):
        detect_partition_rules(exclude=["short", "medium", "long"])


def test_detect_result_is_valid_for_cluster_submitter():
    """The returned dict passes directly into ClusterSubmitter without errors."""
    with _mock_sinfo(_SINFO_TYPICAL):
        rules = detect_partition_rules()
    cs = ClusterSubmitter(slurm_mem="8G", partition_rules=rules)
    assert cs._select_partition(1) == "short"
    assert cs._select_partition(24) == "medium"
    assert cs._select_partition(200) == "long"


def test_detect_cpu_count_with_plus_suffix():
    """sinfo rows like '72+' are parsed as 72, not skipped with a warning."""
    sinfo_out = "compute|8:00:00|72+|257000\n"
    with _mock_sinfo(sinfo_out):
        rules = detect_partition_rules()
    assert "compute" in rules
    assert rules["compute"]["max_cpus"] == 72


# --- submit_dir resolution ---


def test_init_submit_dir_relative_is_resolved_to_absolute():
    cs = ClusterSubmitter(
        slurm_mem="2G", partition_rules=_RULES, submit_dir="relative/path"
    )
    assert cs.submit_dir.is_absolute()
    assert cs.submit_dir.name == "path"


def test_init_submit_dir_absolute_stays_unchanged():
    cs = ClusterSubmitter(
        slurm_mem="2G", partition_rules=_RULES, submit_dir="/abs/path"
    )
    assert cs.submit_dir == Path("/abs/path").resolve()
    assert cs.submit_dir.is_absolute()


# --- _get_slurm_status_bulk parent-ID queries ---


def _mock_subprocess(stdout: str):
    result = MagicMock()
    result.stdout = stdout
    result.returncode = 0
    return patch("py_alf.cluster_submission.subprocess.run", return_value=result)


def test_get_slurm_status_bulk_queries_parent_id_for_array_tasks():
    """squeue is called with the array parent ID, not individual task IDs."""
    from py_alf.cluster_submission import _get_slurm_status_bulk

    squeue_output = (
        "99000 99000_0 COMPLETED 01:00:00 node01\n"
        "99000 99000_1 RUNNING   00:30:00 node02\n"
    )
    with _mock_subprocess(squeue_output) as mock_run:
        result = _get_slurm_status_bulk(["99000_0", "99000_1"])

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
    from py_alf.cluster_submission import _get_slurm_status_bulk

    squeue_output = "77777 77777 PENDING 0:00 (Priority)\n"
    with _mock_subprocess(squeue_output) as mock_run:
        _get_slurm_status_bulk(["77777"])

    first_cmd = mock_run.call_args_list[0][0][0]
    j_arg = first_cmd[first_cmd.index("-j") + 1]
    assert "77777" in j_arg.split(",")


def test_sacct_fallback_filters_by_job_id():
    """The sacct fallback asks for specific jobs, never the whole day's history."""
    from py_alf.cluster_submission import _get_slurm_status_bulk_sacct

    sacct_output = "88000_0|COMPLETED|01:00:00|node01\n"
    with _mock_subprocess(sacct_output) as mock_run:
        _get_slurm_status_bulk_sacct(["88000_0"])

    cmd = mock_run.call_args_list[0][0][0]
    assert cmd[0] == "sacct"
    assert "-j" in cmd, "sacct must be filtered by job id"
    assert "88000" in cmd[cmd.index("-j") + 1].split(",")


def test_sacct_fallback_ignores_substep_rows():
    """sacct's .batch/.extern sub-steps must not leak into the status map."""
    from py_alf.cluster_submission import _get_slurm_status_bulk_sacct

    sacct_output = (
        "88001 COMPLETED 01:00:00 node01\n"
        "88001.batch FAILED 01:00:00 node01\n"
        "88001.extern COMPLETED 01:00:00 node01\n"
    )
    with _mock_subprocess(sacct_output):
        result = _get_slurm_status_bulk_sacct(["88001"])

    assert set(result) == {"88001"}
    assert result["88001"]["status"] == "COMPLETED"


def test_terminal_status_is_cached_and_not_requeried():
    """A finished job is served from cache, sparing SLURM a query per refresh."""
    from py_alf.cluster_submission import _get_slurm_status_bulk

    squeue_output = "99100 99100 RUNNING 00:30:00 node07\n"
    with _mock_subprocess(squeue_output):
        first = _get_slurm_status_bulk(["99100"])
    assert first["99100"]["status"] == "RUNNING"

    # Still RUNNING → not cached, so the next refresh must query again.
    with _mock_subprocess(squeue_output) as mock_run:
        _get_slurm_status_bulk(["99100"])
    assert mock_run.call_count > 0

    # Now it completes; squeue no longer lists it and sacct reports the state.
    with _mock_subprocess("99100 COMPLETED 01:00:00 node07\n"):
        done = _get_slurm_status_bulk(["99100"])
    assert done["99100"]["status"] == "COMPLETED"

    # Terminal states are immutable — no further subprocess calls.
    with _mock_subprocess("") as mock_run:
        cached = _get_slurm_status_bulk(["99100"])
    assert mock_run.call_count == 0
    assert cached["99100"]["status"] == "COMPLETED"


def test_terminal_cache_still_queries_unfinished_jobs():
    """A mixed session queries only the jobs that can still change."""
    from py_alf.cluster_submission import _get_slurm_status_bulk

    with _mock_subprocess("99200 COMPLETED 01:00:00 node01\n"):
        _get_slurm_status_bulk(["99200"])

    squeue_output = "99201 99201 RUNNING 00:10:00 node02\n"
    with _mock_subprocess(squeue_output) as mock_run:
        result = _get_slurm_status_bulk(["99200", "99201"])

    queried = mock_run.call_args_list[0][0][0]
    j_arg = queried[queried.index("-j") + 1].split(",")
    assert j_arg == ["99201"], "cached terminal job must be excluded from the query"
    assert result["99200"]["status"] == "COMPLETED"
    assert result["99201"]["status"] == "RUNNING"


# --- _is_submitit_timeout ---


def test_submitit_timeout_detected_from_log(tmp_path):
    from py_alf.cluster_submission import _is_submitit_timeout

    (tmp_path / "77_0_log.out").write_text("... this job is timed-out ...")
    assert _is_submitit_timeout("77", tmp_path) is True


def test_submitit_missing_log_is_not_cached(tmp_path):
    """A log that lags the job's state change must not pin "not timed out"."""
    from py_alf.cluster_submission import _is_submitit_timeout

    assert _is_submitit_timeout("78", tmp_path) is False

    # The log lands on a later refresh; the timeout must now be reported.
    (tmp_path / "78_0_log.out").write_text("... this job is timed-out ...")
    assert _is_submitit_timeout("78", tmp_path) is True


def test_submitit_present_log_is_cached(tmp_path):
    """Once read, the log is not read again -- its verdict cannot change."""
    from py_alf.cluster_submission import _is_submitit_timeout

    log = tmp_path / "79_0_log.out"
    log.write_text("nothing interesting")
    assert _is_submitit_timeout("79", tmp_path) is False

    with patch.object(Path, "read_text", side_effect=AssertionError("re-read")):
        assert _is_submitit_timeout("79", tmp_path) is False


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
    from py_alf.cluster_submission import _bin_count

    _write_bins(tmp_path / "data.h5", 5)
    _settle(tmp_path / "data.h5")
    sim = _bin_count_sim(tmp_path)

    assert _bin_count(sim, refresh=True) == 5

    # A second refresh must not reach h5py: the file has not moved.
    with patch("h5py.File", side_effect=AssertionError("data.h5 re-opened")):
        assert _bin_count(sim, refresh=True) == 5


def test_bin_count_rereads_when_file_changes(tmp_path):
    """A new bin landing changes (mtime, size), so the count is re-read."""
    from py_alf.cluster_submission import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 5)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 5

    _write_bins(h5, 9)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
    assert _bin_count(sim, refresh=True) == 9


def test_bin_count_force_bypasses_stat_gate(tmp_path):
    """force=True re-reads even when the stat signature is unchanged."""
    from py_alf.cluster_submission import _bin_count

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
    from py_alf.cluster_submission import _bin_count

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
    from py_alf.cluster_submission import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 40

    _truncate(h5)
    os.utime(h5, (h5.stat().st_atime, h5.stat().st_mtime + 10))
    with caplog.at_level(logging.ERROR, logger="py_alf.cluster_submission"):
        assert _bin_count(sim, refresh=True) == 40, "must fall back to cached count"
    assert caplog.records == [], "a mid-write must not be logged as an error"


def test_midwrite_error_is_reported_once_it_stops_being_transient(tmp_path, caplog):
    """A file that keeps failing is real damage and must surface."""
    from py_alf.cluster_submission import _MIDWRITE_LOG_AFTER, _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 40)
    sim = _bin_count_sim(tmp_path)
    assert _bin_count(sim, refresh=True) == 40
    _truncate(h5)

    with caplog.at_level(logging.ERROR, logger="py_alf.cluster_submission"):
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
    from py_alf.cluster_submission import _MIDWRITE_LOG_AFTER, _bin_count

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
    with caplog.at_level(logging.ERROR, logger="py_alf.cluster_submission"):
        _bin_count(sim, refresh=True)
    assert caplog.records == [], "streak should have reset after the good read"


def test_midwrite_read_is_retried(tmp_path):
    """A file that settles between attempts is read rather than reported stale."""
    from py_alf.cluster_submission import _bin_count

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
    from py_alf.cluster_submission import _bin_count

    sim = _bin_count_sim(tmp_path)
    with caplog.at_level(logging.DEBUG, logger="py_alf.cluster_submission"):
        assert _bin_count(sim, refresh=True) == 0
    assert caplog.records == []


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


# --- _exec_alf_binary data.h5 backup ---


def test_exec_alf_binary_backs_up_data_on_fresh_run(tmp_path):
    """data.h5 is renamed before a fresh run (no confin_* present)."""
    data = tmp_path / "data.h5"
    data.write_bytes(b"old")
    binary = tmp_path / "ALF.out"
    binary.touch()

    with (
        patch("subprocess.run"),
        patch.dict("os.environ", {"SLURM_JOB_ID": "99999"}, clear=False),
    ):
        _exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

    assert not data.exists(), "data.h5 should have been renamed"
    assert (tmp_path / "data_99999.h5").exists(), "backup file should exist"


def test_exec_alf_binary_preserves_data_on_checkpoint_restart(tmp_path):
    """data.h5 is left untouched when confin_* files are present."""
    data = tmp_path / "data.h5"
    data.write_bytes(b"accumulated")
    (tmp_path / "confin_0").touch()
    binary = tmp_path / "ALF.out"
    binary.touch()

    with patch("subprocess.run"):
        _exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

    assert data.exists(), "data.h5 must not be touched during checkpoint restart"
    assert data.read_bytes() == b"accumulated"


def test_exec_alf_binary_passes_extra_env(tmp_path):
    """A Simulation's env reaches the ALF process."""
    (tmp_path / "ALF.out").touch()

    with patch("subprocess.run") as run:
        _exec_alf_binary(
            tmp_path, n_omp=1, n_mpi=1, mpi=False, extra_env={"ALF_DELAY_K": "32"}
        )

    assert run.call_args.kwargs["env"]["ALF_DELAY_K"] == "32"


def test_exec_alf_binary_no_backup_when_no_data(tmp_path):
    """No error and no backup file when data.h5 does not exist."""
    binary = tmp_path / "ALF.out"
    binary.touch()

    with patch("subprocess.run"):
        _exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

    backups = list(tmp_path.glob("data_*.h5"))
    assert backups == []


def test_map_io_runs_small_inputs_without_a_pool():
    """Below the fan-out threshold the work runs inline — no thread spawn."""
    import threading

    caller = threading.current_thread()
    threads = _map_io(
        lambda i: threading.current_thread(), list(range(_MIN_FANOUT - 1))
    )
    assert all(t is caller for t in threads)


def test_map_io_fans_out_large_inputs_and_preserves_order():
    """Above the threshold work is spread across threads but stays ordered."""
    import threading

    n = max(_MIN_FANOUT, 8)
    barrier = threading.Barrier(n, timeout=5)

    def work(i):
        # Deadlocks unless the items really run concurrently.
        barrier.wait()
        return i * 2

    assert _map_io(work, list(range(n))) == [i * 2 for i in range(n)]


# --- _bin_count process pool (h5py's phil serializes reads within a process) ---


def test_bin_count_use_process_pool_reads_correctly(tmp_path):
    """use_process_pool=True still reads the right count, via a worker process."""
    from py_alf.cluster_submission import _bin_count

    h5 = tmp_path / "data.h5"
    _write_bins(h5, 12)
    sim = _bin_count_sim(tmp_path)

    assert _bin_count(sim, refresh=True, force=True, use_process_pool=True) == 12


def test_bin_count_use_process_pool_missing_file_returns_zero(tmp_path):
    """A chain with no data.h5 yet reads as 0 bins through the process pool too."""
    from py_alf.cluster_submission import _bin_count

    sim = _bin_count_sim(tmp_path)  # no data.h5 written

    assert _bin_count(sim, refresh=True, force=True, use_process_pool=True) == 0


def test_bin_count_use_process_pool_many_files_no_cross_contamination(tmp_path):
    """Each dispatch must come back matched to its own file, not another's."""
    from py_alf.cluster_submission import _bin_count

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
    from py_alf.cluster_submission import _read_bin_count

    assert _read_bin_count.__module__ == "py_alf.cluster_submission"
    assert _read_bin_count.__qualname__ == "_read_bin_count"


# --- _bin_counts: the batched read behind Campaign.status ---


def test_bin_counts_keeps_results_in_caller_order(tmp_path):
    """Chunked pool.map returns results by task; they must map back to their file."""
    from py_alf.cluster_submission import _bin_counts

    paths = []
    for i in range(12):
        d = tmp_path / f"chain_{i}"
        d.mkdir()
        _write_bins(d / "data.h5", i + 1)
        paths.append(str(d / "data.h5"))

    assert _bin_counts(paths, force=True) == list(range(1, 13))


def test_bin_counts_reports_zero_for_a_missing_file(tmp_path):
    """An unstarted chain has no data.h5; that is 0 bins, not a failure."""
    from py_alf.cluster_submission import _bin_counts

    (tmp_path / "there").mkdir()
    _write_bins(tmp_path / "there" / "data.h5", 7)
    counts = _bin_counts(
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
    from py_alf.cluster_submission import _bin_counts

    d = tmp_path / "chain"
    d.mkdir()
    _write_bins(d / "data.h5", 9)
    _settle(d / "data.h5")
    path = str(d / "data.h5")

    assert _bin_counts([path]) == [9]
    with patch(
        "py_alf.cluster_submission._read_bin_count",
        side_effect=AssertionError("re-read an unchanged file"),
    ):
        assert _bin_counts([path]) == [9]


def test_bin_counts_sees_a_file_that_grew(tmp_path):
    """A chain that ran more bins must not be masked by the previous reading."""
    from py_alf.cluster_submission import _bin_counts

    d = tmp_path / "grower"
    d.mkdir()
    _write_bins(d / "data.h5", 5)
    _settle(d / "data.h5")
    path = str(d / "data.h5")
    assert _bin_counts([path]) == [5]

    _write_bins(d / "data.h5", 25)
    _settle(d / "data.h5")
    assert _bin_counts([path]) == [25]
