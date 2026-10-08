"""Tests for launching the ALF binary in py_alf.execute."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from py_alf.execute import exec_alf_binary
from py_alf.simulation import Simulation

# --- exec_alf_binary: fresh start versus resume ---


def _data_h5(sim_dir, n_bins):
    """A data.h5 shaped like ALF's: parameters plus one observable's bins."""
    with h5py.File(sim_dir / "data.h5", "w") as f:
        f.create_group("parameters")
        f.create_dataset("Ener_scal/obser", data=np.zeros((n_bins, 1, 2)))


def _sim(sim_dir, **overrides):
    """The attributes exec_alf_binary reads from a Simulation."""
    attrs = {
        "sim_dir": str(sim_dir),
        "alf_src": SimpleNamespace(alf_dir="."),
        "config": "GNU NOMPI",
        "n_omp": 1,
        "n_mpi": 1,
        "mpi": False,
        "mpiexec": "mpiexec",
        "mpiexec_args": [],
        "env": {},
    }
    return SimpleNamespace(**{**attrs, **overrides})


@contextlib.contextmanager
def _no_shell():
    """Stub out configure.sh and the ALF process; yield the process mock."""
    with (
        patch("py_alf.execute.getenv", return_value={}),
        patch("py_alf.execute.subprocess.run") as run,
    ):
        yield run


def test_exec_alf_binary_refuses_to_start_over_existing_bins(tmp_path):
    """ALF would append a second, independent chain to these bins."""
    _data_h5(tmp_path, 3)
    (tmp_path / "ALF.out").touch()

    with _no_shell() as run, pytest.raises(RuntimeError):
        exec_alf_binary(_sim(tmp_path))

    run.assert_not_called()
    assert (tmp_path / "data.h5").exists()


def test_exec_alf_binary_clears_a_data_file_with_no_bins(tmp_path):
    """A run that died before its first bin left only parameters behind."""
    _data_h5(tmp_path, 0)
    (tmp_path / "ALF.out").touch()

    with _no_shell() as run:
        exec_alf_binary(_sim(tmp_path))

    run.assert_called_once()
    assert not (tmp_path / "data.h5").exists()


def test_exec_alf_binary_resumes_onto_existing_bins(tmp_path):
    """With a checkpoint, ALF appends to data.h5, which must be left alone."""
    _data_h5(tmp_path, 3)
    (tmp_path / "confin_0.h5").touch()
    (tmp_path / "ALF.out").touch()

    with _no_shell() as run:
        exec_alf_binary(_sim(tmp_path))

    run.assert_called_once()
    with h5py.File(tmp_path / "data.h5", "r") as f:
        assert f["Ener_scal/obser"].shape[0] == 3


def test_exec_alf_binary_passes_the_sim_env(tmp_path):
    """A Simulation's env reaches the ALF process."""
    (tmp_path / "ALF.out").touch()

    with _no_shell() as run:
        exec_alf_binary(_sim(tmp_path, env={"ALF_DELAY_K": "32"}))

    assert run.call_args.kwargs["env"]["ALF_DELAY_K"] == "32"


def test_exec_alf_binary_wraps_mpi_runs_in_mpiexec(tmp_path):
    (tmp_path / "ALF.out").touch()

    with _no_shell() as run:
        exec_alf_binary(
            _sim(tmp_path, mpi=True, n_mpi=4, mpiexec_args=["--oversubscribe"])
        )

    assert run.call_args.args[0] == [
        "mpiexec",
        "-n",
        "4",
        "--oversubscribe",
        str(tmp_path / "ALF.out"),
    ]


def test_simulation_run_launches_through_exec_alf_binary_with_env(tmp_path):
    """Simulation.run used to build its own command and drop sim.env."""
    sim = object.__new__(Simulation)
    vars(sim).update(vars(_sim(tmp_path, env={"ALF_DELAY_K": "32"})))
    sim.tempering = False
    sim.sim_dict = {}
    sim.mc_seed = None
    sim.ham_name = "Hubbard"

    with patch("py_alf.simulation._prep_sim_dir"), _no_shell() as run:
        sim.run(bin_in_sim_dir=True)

    assert run.call_args.args[0] == [str(tmp_path / "ALF.out")]
    assert run.call_args.kwargs["env"]["ALF_DELAY_K"] == "32"
