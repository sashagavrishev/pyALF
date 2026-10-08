"""Tests for ClusterSubmitter in py_alf.cluster_submission."""

import operator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from py_alf.cluster_submission import (
    ClusterSubmitter,
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
    # partition_rules is normalised to float hours at construction time
    assert cs.partition_rules == {"short": 8.0, "long": 168.0}
    assert cs.job_name is None
    assert cs.executor_params == {}


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
    assert cs.partition_rules == {"gpu": 24.0}
    assert cs.executor_params == {"slurm_extra": "foo"}


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


def test_init_local_rejects_slurm_options_passed_through():
    with pytest.raises(ValueError, match="slurm_wckey"):
        ClusterSubmitter("local", slurm_wckey="my-project")


def test_init_job_name_and_stderr_accepted_for_local():
    cs = ClusterSubmitter("local", job_name="my-job", stderr_to_stdout=True)
    assert cs.job_name == "my-job"
    assert cs.executor_params == {"stderr_to_stdout": True}


def test_init_rejects_a_partition_rule_that_is_not_hours():
    with pytest.raises(ValueError, match="hours"):
        ClusterSubmitter(slurm_mem="2G", partition_rules={"short": {"max_hours": 2}})


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


def test_submit_without_cpu_max_needs_one_on_slurm(tmp_path):
    """With no CPU_MAX there is no wall time to derive."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 0}
    cs = ClusterSubmitter(
        submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
    )
    with (
        _patch_submitit(MagicMock(job_id="1")),
        pytest.raises(ValueError, match="set CPU_MAX"),
    ):
        cs.submit(sim)


def test_submit_wall_time_is_cpu_max_plus_ten_percent(tmp_path):
    """ALF gets room to finish its last bin, within the partition's limit."""
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 2}

    with _patch_submitit(MagicMock(job_id="1")) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["timeout_min"] == 132
    assert "slurm_additional_parameters" not in call_kwargs


def test_submit_wall_time_is_capped_at_the_partition_limit(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    sim.sim_dict = {"CPU_MAX": 7.5}  # 7.5 h * 1.1 exceeds "short"'s 8 h

    with _patch_submitit(MagicMock(job_id="1")) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sim)

    call_kwargs = mock_executor.return_value.update_parameters.call_args.kwargs
    assert call_kwargs["slurm_partition"] == "short"
    assert call_kwargs["timeout_min"] == 8 * 60


def test_submit_max_requeues_reaches_the_executor(tmp_path):
    sim = _make_mock_sim(tmp_path / "sim0")
    with _patch_submitit(MagicMock(job_id="1")) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs", slurm_mem="2G", partition_rules=_RULES
        )
        cs.submit(sim, max_requeues=5)

    assert mock_executor.call_args.kwargs["slurm_max_num_timeout"] == 5
    params = mock_executor.return_value.update_parameters.call_args.kwargs
    assert "slurm_max_num_timeout" not in params


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


def test_submit_passes_slurm_options_through(tmp_path):
    """slurm_* options given to the constructor reach update_parameters."""
    sim = _make_mock_sim(tmp_path / "sim0")

    with _patch_submitit(MagicMock(job_id="1")) as mock_executor:
        cs = ClusterSubmitter(
            submit_dir=tmp_path / "logs",
            slurm_mem="2G",
            partition_rules=_RULES,
            slurm_mail_type="END",
            slurm_wckey="proj-key",
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


# --- submit_dir templates, end to end on the local executor ---


def test_submit_into_an_array_folder_template_runs_locally(tmp_path):
    """A %A template becomes one real folder per job; no literal %A is made by us."""
    alf_dir = tmp_path / "ALF"
    (alf_dir / "Prog").mkdir(parents=True)
    (alf_dir / "Prog" / "ALF.out").touch()
    sim = SimpleNamespace(
        sim_dir=str(tmp_path / "sim0"),
        sim_dict={"CPU_MAX": 0.1},
        ham_name="Hubbard",
        n_omp=1,
        n_mpi=1,
        mpi=False,
        run=None,
        alf_src=SimpleNamespace(alf_dir=str(alf_dir)),
    )
    template = tmp_path / "jobs" / "L8" / "%A"

    cs = ClusterSubmitter("local", submit_dir=tmp_path / "default")
    # The worker process must be able to import the runner, so no test-local one.
    (job,) = cs.submit(
        sim, submit_dir=template, runner=operator.attrgetter("sim_dir"), prep=False
    )

    assert job.result() == sim.sim_dir
    folder = tmp_path / "jobs" / "L8" / job.job_id
    assert (folder / f"{job.job_id}_0_log.out").exists()
    assert not (tmp_path / "default").exists()
