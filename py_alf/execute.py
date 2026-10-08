"""How an ALF binary is launched inside a simulation directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import h5py

from .simulation import Simulation, cd, getenv


def _check_fresh_start(sim_dir: Path) -> None:
    """Refuse to start a second chain on top of an earlier chain's bins.

    ALF appends to an existing ``data.h5`` whether or not it resumes from a
    checkpoint, so a fresh start there would mix two independent chains. A file
    with no bins holds only the parameters of a run that died before its first
    bin, and is removed.
    """
    data_file = sim_dir / "data.h5"
    if not data_file.exists() or any(sim_dir.glob("confin_*")):
        return
    with h5py.File(data_file, "r", locking=False) as f:
        has_bins = any(
            isinstance(obs, h5py.Group) and "obser" in obs and obs["obser"].shape[0]
            for obs in f.values()
        )
    if has_bins:
        raise RuntimeError(
            f"{data_file} holds bins but {sim_dir} has no confin_* to resume from; "
            "archive or remove it before starting a new chain there."
        )
    data_file.unlink()


def exec_alf_binary(sim: Simulation, executable: str | Path | None = None) -> None:
    """Run ALF for *sim* inside its ``sim_dir``, with ``sim.env`` applied.

    The one place an ALF process is started, locally or on a node.
    *executable* defaults to the copy of ``ALF.out`` in ``sim_dir``.
    """
    sim_dir = Path(sim.sim_dir)
    _check_fresh_start(sim_dir)
    env = getenv(sim.config, sim.alf_src.alf_dir)
    # Under SLURM, match the threads to the cores actually allocated.
    env["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", str(sim.n_omp))
    env.update(sim.env)

    binary = str(executable) if executable is not None else str(sim_dir / "ALF.out")
    if sim.mpi:
        cmd = [sim.mpiexec, "-n", str(sim.n_mpi), *sim.mpiexec_args, binary]
    else:
        cmd = [binary]
    with cd(str(sim_dir)):
        subprocess.run(cmd, check=True, env=env)
