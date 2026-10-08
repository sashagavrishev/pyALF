"""How an ALF binary is launched inside a simulation directory."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .simulation import Simulation, cd, getenv


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
    executable = os.path.join(str(sim_dir), "ALF.out")
    env = getenv(config, alf_dir)
    # Prefer SLURM_CPUS_PER_TASK so OMP_NUM_THREADS exactly matches the
    # allocated CPU slots, which is best practice for hybrid MPI+OpenMP jobs.
    env["OMP_NUM_THREADS"] = os.environ.get("SLURM_CPUS_PER_TASK", str(n_omp))
    env.update(extra_env or {})

    # Guard against overwriting data from a previous independent run.
    # When confin_* files are present ALF will checkpoint-restart and
    # accumulate bins into the existing data.h5 — the intended behaviour.
    # When no confin_* exist ALF starts fresh and would overwrite data.h5.
    # In that case we archive the old file under the current SLURM job ID
    # so it is not lost.
    has_checkpoint = any(
        name.startswith("confin_") for name in os.listdir(sim_dir_path)
    )
    if not has_checkpoint:
        data_file = sim_dir_path / "data.h5"
        if data_file.exists():
            import time as _time

            job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get(
                "SLURM_JOB_ID"
            )
            suffix = job_id if job_id else str(int(_time.time()))
            data_file.rename(sim_dir_path / f"data_{suffix}.h5")

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
