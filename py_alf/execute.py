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


def exec_alf_binary(
    sim_dir: str | Path,
    n_omp: int,
    n_mpi: int,
    mpi: bool,
    mpiexec: str = "mpiexec",
    mpiexec_args: list[str] | None = None,
    config: str = "",
    alf_dir: str = ".",
    extra_env: dict[str, str] | None = None,
) -> None:
    """Execute the ALF binary already present in *sim_dir*.

    Single source of truth for how an ALF job is launched on a worker node.
    """
    sim_dir_path = Path(sim_dir)
    _check_fresh_start(sim_dir_path)
    executable = os.path.join(str(sim_dir), "ALF.out")
    env = getenv(config, alf_dir)
    # Prefer SLURM_CPUS_PER_TASK so OMP_NUM_THREADS exactly matches the
    # allocated CPU slots, which is best practice for hybrid MPI+OpenMP jobs.
    env["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", str(n_omp))
    env.update(extra_env or {})

    if mpi:
        cmd: list[str] = [mpiexec, "-n", str(n_mpi), *(mpiexec_args or []), executable]
    else:
        cmd = [executable]
    with cd(str(sim_dir)):
        subprocess.run(cmd, check=True, env=env)


def run_alf(sim: Simulation) -> None:
    """
    Execute an ALF simulation on a cluster node.

    Called by submitit on the remote worker. Assumes the binary has already been copied
    into sim.sim_dir by the pre-submission preparation step.
    """
    exec_alf_binary(
        sim.sim_dir,
        sim.n_omp,
        sim.n_mpi,
        getattr(sim, "mpi", False),
        mpiexec=getattr(sim, "mpiexec", "mpiexec"),
        mpiexec_args=getattr(sim, "mpiexec_args", []),
        config=getattr(sim, "config", ""),
        alf_dir=getattr(sim.alf_src, "alf_dir", "."),
        extra_env=getattr(sim, "env", None),
    )
