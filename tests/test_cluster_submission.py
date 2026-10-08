"""Tests for ClusterSubmitter in py_alf.cluster_submission."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from py_alf.cluster_submission import (
    ClusterSubmitter,
    _normalise_partition_spec,
    _parse_mem_gb,
    _parse_slurm_time_hours,
)
from py_alf.execute import exec_alf_binary
from py_alf.simulation import Simulation

_RULES = {"short": 8, "long": 168}


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
    mock_executor.return_value.submit.assert_called_once_with(exec_alf_binary, sim)


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
    mock_executor.return_value.map_array.assert_called_once_with(exec_alf_binary, sims)


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
        "py_alf.cluster_submission.job_states",
        return_value={"7": {"status": "RUNNING"}},
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
        "py_alf.cluster_submission.job_states",
        return_value={"7": {"status": "PENDING"}},
    ):
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        jobs = cs.submit(sim)

    assert jobs == []


def test_submit_queries_slurm_once_for_many_sims(tmp_path):
    """The active check is one bulk query, not one sacct call per sim."""
    sims = []
    for i in range(3):
        sim = _make_mock_sim(tmp_path / f"sim{i}")
        (tmp_path / f"sim{i}" / "jobid.txt").write_text(f"7_{i}")
        sims.append(sim)
    states = {"7_0": {"status": "RUNNING"}, "7_1": {"status": "COMPLETED"}}

    with (
        patch("py_alf.cluster_submission.job_states", return_value=states) as query,
        _patch_submitit(
            [MagicMock(job_id="8_0"), MagicMock(job_id="8_1")], multi=True
        ) as mock_executor,
    ):
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sims)

    query.assert_called_once()
    submitted = mock_executor.return_value.map_array.call_args.args[1]
    assert [s.sim_dir for s in submitted] == [sims[1].sim_dir, sims[2].sim_dir]


def test_submit_skip_active_false_trusts_the_caller(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    (tmp_path / "sim0" / "jobid.txt").write_text("7")

    with (
        patch("py_alf.cluster_submission.job_states") as query,
        _patch_submitit(MagicMock(job_id="8")),
    ):
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        assert len(cs.submit(sim, skip_active=False)) == 1

    query.assert_not_called()


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
        patch("py_alf.cluster_submission.job_states") as mock_sacct,
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
