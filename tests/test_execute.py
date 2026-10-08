"""Tests for launching the ALF binary in py_alf.execute."""

from unittest.mock import patch

from py_alf.execute import exec_alf_binary

# --- exec_alf_binary data.h5 backup ---


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
        exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

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
        exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

    assert data.exists(), "data.h5 must not be touched during checkpoint restart"
    assert data.read_bytes() == b"accumulated"


def test_exec_alf_binary_passes_extra_env(tmp_path):
    """A Simulation's env reaches the ALF process."""
    (tmp_path / "ALF.out").touch()

    with patch("subprocess.run") as run:
        exec_alf_binary(
            tmp_path, n_omp=1, n_mpi=1, mpi=False, extra_env={"ALF_DELAY_K": "32"}
        )

    assert run.call_args.kwargs["env"]["ALF_DELAY_K"] == "32"


def test_exec_alf_binary_no_backup_when_no_data(tmp_path):
    """No error and no backup file when data.h5 does not exist."""
    binary = tmp_path / "ALF.out"
    binary.touch()

    with patch("subprocess.run"):
        exec_alf_binary(tmp_path, n_omp=1, n_mpi=1, mpi=False)

    backups = list(tmp_path.glob("data_*.h5"))
    assert backups == []
