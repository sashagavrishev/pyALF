"""Submitting ALF simulations through submitit, to SLURM or locally."""

from __future__ import annotations

__author__ = "Johannes Hofmann"
__copyright__ = "Copyright 2020-2025, The ALF Project"
__license__ = "GPL"

import hashlib
import logging
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import submitit
from submitit.core.utils import JobPaths

from .execute import exec_alf_binary
from .simulation import Simulation
from .slurm import ACTIVE_STATES, job_states

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    """Nearest ancestor of the CWD holding ``.git`` or a project file, else the CWD.

    Anchors the default ``.pyalf`` at the repository root, wherever a script runs.
    """
    markers = {".git", "pyproject.toml", "setup.py", "setup.cfg"}
    current = Path.cwd()
    while True:
        if any((current / m).exists() for m in markers):
            return current
        parent = current.parent
        if parent == current:
            return Path.cwd()
        current = parent


def _format_hours(h: float) -> str:
    """Return a human-readable string for a duration expressed in hours."""
    if h < 1:
        return f"{round(h * 60)}min"
    if h < 24:
        return f"{h:g}h"
    days = h / 24
    return f"{int(days)}d" if days == int(days) else f"{days:.1f}d"


class ClusterSubmitter:
    """Submits simulations as submitit jobs, one SLURM array per call.

    Parameters
    ----------
    executor : {'slurm', 'local', 'debug'}
        ``'local'`` runs jobs in local processes, ``'debug'`` inline.
    submit_dir : str or Path, optional
        Where submitit writes job files; defaults to ``.pyalf`` at the project root.
    slurm_mem : str
        Memory per node, e.g. ``'8G'``. Required for SLURM.
    partition_rules : dict[str, float]
        Partition name to its wall-time limit in hours. Required for SLURM.
        Each job goes to the partition with the smallest limit that fits it.
    job_name : str, optional
        Job name. Defaults to the Hamiltonian name.
    **executor_params
        Forwarded to submitit's ``update_parameters()``, e.g.
        ``slurm_mail_type='FAIL'``, ``slurm_wckey=...``, ``slurm_setup=[...]``
        or ``stderr_to_stdout=True``. ``slurm_*`` keys need the SLURM executor.

    Raises
    ------
    ValueError
        If *executor* is not one of the accepted values; if SLURM-specific
        parameters are supplied for a non-SLURM executor; or if required
        SLURM parameters are missing for a SLURM executor.
    """

    _VALID_EXECUTORS = ("slurm", "local", "debug")

    def __init__(
        self,
        executor: Literal["slurm", "local", "debug"] = "slurm",
        *,
        submit_dir: str | Path | None = None,
        slurm_mem: str | None = None,
        partition_rules: dict[str, Any] | None = None,
        job_name: str | None = None,
        **executor_params,
    ):
        if executor not in self._VALID_EXECUTORS:
            raise ValueError(
                f"executor must be one of {self._VALID_EXECUTORS!r}, got {executor!r}"
            )

        if executor != "slurm":
            slurm_specific = [k for k in executor_params if k.startswith("slurm_")]
            problems = (
                (["slurm_mem"] if slurm_mem is not None else [])
                + (["partition_rules"] if partition_rules is not None else [])
                + slurm_specific
            )
            if problems:
                raise ValueError(
                    f"Parameters {problems!r} are only valid for executor='slurm'"
                )
        else:
            if slurm_mem is None:
                raise ValueError("slurm_mem is required for executor='slurm'")
            if partition_rules is None:
                raise ValueError(
                    "partition_rules is required for executor='slurm'. "
                    "Example: partition_rules={'short': 2, 'medium': 48, 'long': 336}"
                )

        if partition_rules is not None:
            try:
                partition_rules = {
                    name: float(hours) for name, hours in partition_rules.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"partition_rules maps each partition to its hours: {exc}"
                ) from exc

        self.executor = executor
        self.submit_dir = (
            Path(submit_dir).resolve()
            if submit_dir is not None
            else (_project_root() / ".pyalf")
        )
        self.slurm_mem = slurm_mem
        self.partition_rules: dict[str, float] | None = partition_rules
        self.job_name = job_name
        self.executor_params = executor_params

    def _select_partition(self, timeout_hours: float) -> str:
        """Select the smallest-limit partition that accommodates *timeout_hours*."""
        for name, limit in sorted(self.partition_rules.items(), key=lambda kv: kv[1]):
            if timeout_hours <= limit:
                return name
        configured = ", ".join(
            f"{n}: {_format_hours(h)}" for n, h in self.partition_rules.items()
        )
        raise ValueError(
            f"No configured partition fits a {_format_hours(timeout_hours)} timeout. "
            f"Extend partition_rules or reduce CPU_MAX. "
            f"Configured: {{{configured}}}"
        )

    def _wall_time(self, sim: Simulation) -> tuple[int, str | None]:
        """``(minutes, partition)`` for *sim*; the partition is None off SLURM.

        The job asks for ``CPU_MAX`` plus 10%, so ALF can finish its last bin
        and write its output after stopping, capped at the limit of the
        partition ``CPU_MAX`` fits.
        """
        sim_dict = sim.sim_dict[0] if isinstance(sim.sim_dict, list) else sim.sim_dict
        return self._wall_time_for(float(sim_dict.get("CPU_MAX", 0)))

    def _wall_time_for(self, cpu_max: float) -> tuple[int, str | None]:
        """``(minutes, partition)`` for a run budget of *cpu_max* hours."""
        if self.executor != "slurm":
            return int(max(cpu_max, 0.0) * 60), None
        if cpu_max <= 0:
            raise ValueError(
                "CPU_MAX=0 gives ALF no time limit, so no SLURM wall time can be "
                "derived; set CPU_MAX."
            )
        partition = self._select_partition(cpu_max)
        hours = min(cpu_max * 1.1, self.partition_rules[partition])
        return int(hours * 60), partition

    @staticmethod
    def _active_jobs(sims: list[Simulation]) -> dict[str, str]:
        """``sim_dir -> job id`` for each sim whose recorded job is still active."""
        jobids = {}
        for s in sims:
            jobid_file = Path(s.sim_dir) / "jobid.txt"
            if jobid_file.exists():
                jobids[s.sim_dir] = jobid_file.read_text().strip()
        if not jobids:
            return {}
        states = job_states(list(jobids.values()))
        return {
            sim_dir: jid
            for sim_dir, jid in jobids.items()
            if (states.get(jid) or {}).get("status") in ACTIVE_STATES
        }

    def submit(
        self,
        sims: Simulation | Iterable[Simulation],
        job_properties: dict[str, Any] | None = None,
        submit_dir: str | Path | None = None,
        runner: Callable[[Simulation], None] | None = None,
        prep: bool = True,
        stale_running: Literal["remove", "skip"] = "skip",
        skip_active: bool = True,
        max_requeues: int | None = None,
    ) -> list[submitit.Job]:
        """Submit simulations, as one array when there are several.

        Simulations with an active job or a leftover ``RUNNING`` are left out.
        Each submitted job's id is written to ``jobid.txt`` in its ``sim_dir``.
        The wall time is ``CPU_MAX`` plus 10%, capped at the partition limit.

        Parameters
        ----------
        sims : Simulation or iterable of Simulation
            Simulation(s) to submit.
        job_properties : dict, optional
            Per-call overrides for submitit's ``update_parameters()``.
        submit_dir : str or Path, optional
            Directory for submitit logs and state for this submission, which
            may contain submitit's ``%A``/``%j`` placeholders. Overrides the
            instance-level ``submit_dir`` set at construction.
        runner : callable, optional
            Called with the simulation on the worker; defaults to
            :func:`~py_alf.execute.exec_alf_binary`. Pass one that decides on
            the node how to run, usually with ``prep=False``.
        prep : bool, default=True
            Prepare each ``sim_dir`` now. ``False`` leaves it to *runner*, which a
            requeued job needs: the ``confout -> confin`` rename has to happen at
            each start. The ALF binary is copied in either way.
        stale_running : {'remove', 'skip'}, default='skip'
            What to do with a ``RUNNING`` file left behind by a previous run
            whose job is no longer active: ``'skip'`` leaves the simulation out,
            ``'remove'`` deletes the file and submits anyway.
        skip_active : bool, default=True
            Leave out simulations whose ``jobid.txt`` names a job SLURM still
            holds. A caller that has already established this passes ``False``.
        max_requeues : int, optional
            How many times submitit may requeue a task that hit its wall time
            (SLURM only; submitit's default is 3).

        Returns
        -------
        list of submitit.Job
            One Job object per submitted simulation.
        """

        if (
            isinstance(sims, Iterable)
            and not isinstance(sims, (str, bytes))
            and not all(hasattr(sims, a) for a in _SIM_ATTRS)
        ):
            sim_list = list(sims)
        else:
            sim_list = [sims]
        _check_sims(sim_list)

        active = (
            self._active_jobs(sim_list)
            if skip_active and self.executor == "slurm"
            else {}
        )
        filtered_sims = []

        for s in sim_list:
            if s.sim_dir in active:
                logger.info(f"Skipping {s.sim_dir}: job {active[s.sim_dir]} is active")
                continue
            if _clear_stale_running(s, stale_running):
                filtered_sims.append(s)

        if not filtered_sims:
            logger.info("No inactive simulations to submit.")
            return []

        sim = filtered_sims[0]
        _check_uniform(filtered_sims)
        timeout_min, partition = self._wall_time(sim)
        executor = self._executor(
            sim, timeout_min, partition, job_properties, submit_dir, max_requeues
        )

        for s in filtered_sims:
            _stage(s, prep)

        run_fn = runner if runner is not None else exec_alf_binary
        if len(filtered_sims) == 1:
            jobs = [executor.submit(run_fn, filtered_sims[0])]
        else:
            jobs = executor.map_array(run_fn, filtered_sims)

        # jobid.txt is how a later submit recognises a chain that is still active.
        for s, job in zip(filtered_sims, jobs):
            Path(s.sim_dir, "jobid.txt").write_text(job.job_id)

        logger.info(f"Submitted {len(jobs)} job(s): {[j.job_id for j in jobs]}")
        return jobs

    def submit_packs(
        self,
        packs: list[list[Simulation]],
        hours: float,
        runner: Callable[[list[Simulation], float], None],
        job_properties: dict[str, Any] | None = None,
        submit_dir: str | Path | None = None,
        max_requeues: int | None = None,
    ) -> list[submitit.Job]:
        """Submit one array whose task ``i`` calls ``runner(packs[i], hours)``.

        Each task runs its pack's simulations in turn on one slot, within
        ``hours`` plus the 10% :meth:`submit` also allows. Directories are
        staged as ``submit(prep=False)`` does, a leftover ``RUNNING`` is
        removed, and every simulation's ``jobid.txt`` names its pack's job. The
        caller must already have left out simulations with an active job.
        """
        packs = [list(p) for p in packs if p]
        if not packs:
            return []
        sims = [s for p in packs for s in p]
        _check_sims(sims)
        _check_uniform(sims)
        for s in sims:
            _clear_stale_running(s, "remove")
            _stage(s, prep=False)

        timeout_min, partition = self._wall_time_for(hours)
        executor = self._executor(
            sims[0], timeout_min, partition, job_properties, submit_dir, max_requeues
        )
        jobs = executor.map_array(runner, packs, [hours] * len(packs))

        for pack, job in zip(packs, jobs, strict=True):
            for s in pack:
                Path(s.sim_dir, "jobid.txt").write_text(job.job_id)

        logger.info(f"Submitted {len(jobs)} pack(s) of {len(sims)} simulations.")
        return jobs

    def _executor(
        self,
        sim: Simulation,
        timeout_min: int,
        partition: str | None,
        job_properties: dict[str, Any] | None,
        submit_dir: str | Path | None,
        max_requeues: int | None,
    ) -> submitit.AutoExecutor:
        """A submitit executor configured for an array shaped like *sim*."""
        # Defaults, then the instance's options, then this call's. Each MPI rank
        # is a task slot of n_omp cores, matching mpiexec -n n_mpi with
        # OMP_NUM_THREADS=n_omp.
        params: dict[str, Any] = {
            "name": self.job_name if self.job_name is not None else sim.ham_name,
            "timeout_min": timeout_min,
            "nodes": 1,
            "cpus_per_task": sim.n_omp,
            "tasks_per_node": sim.n_mpi if sim.mpi else 1,
        }
        if self.executor == "slurm":
            params["slurm_mem"] = self.slurm_mem
            params["slurm_partition"] = partition
            if sim.mpi:
                # submitit's srun would start one launcher per task slot, each
                # running its own mpiexec (submitit#1757); one launcher calls
                # mpiexec once instead.
                params["slurm_use_srun"] = False
        params.update(self.executor_params)
        params.update(job_properties or {})

        effective_submit_dir = (
            Path(submit_dir) if submit_dir is not None else self.submit_dir
        )
        # A %A/%j template is filled in per job; create only the part before it.
        JobPaths.get_first_id_independent_folder(effective_submit_dir).mkdir(
            parents=True, exist_ok=True
        )

        # submitit takes the requeue budget when the executor is constructed.
        executor_kwargs: dict[str, Any] = {}
        if max_requeues is not None and self.executor == "slurm":
            executor_kwargs["slurm_max_num_timeout"] = int(max_requeues)

        executor = submitit.AutoExecutor(
            folder=str(effective_submit_dir),
            cluster=self.executor,
            **executor_kwargs,
        )
        executor.update_parameters(**params)
        return executor


_SIM_ATTRS = ("sim_dir", "sim_dict", "ham_name", "n_omp", "n_mpi", "mpi", "run")


def _check_sims(sims: list) -> None:
    for s in sims:
        missing = [a for a in _SIM_ATTRS if not hasattr(s, a)]
        if missing:
            raise TypeError(
                f"Expected Simulation-like object (missing {missing!r}), got {type(s)}"
            )


def _check_uniform(sims: list[Simulation]) -> None:
    """One array shares one set of SLURM parameters."""
    sim = sims[0]
    for s in sims[1:]:
        if s.n_omp != sim.n_omp or s.n_mpi != sim.n_mpi or s.mpi != sim.mpi:
            raise ValueError(
                "All simulations in an array job must have the same n_omp, n_mpi, "
                f"and mpi settings (derived from the first: n_omp={sim.n_omp}, "
                f"n_mpi={sim.n_mpi}, mpi={sim.mpi}), but {s.sim_dir} has "
                f"n_omp={s.n_omp}, n_mpi={s.n_mpi}, mpi={s.mpi}."
            )


def _clear_stale_running(sim: Simulation, stale_running: str) -> bool:
    """Handle a leftover ``RUNNING``; False if *sim* must be left out."""
    running_file = Path(sim.sim_dir) / "RUNNING"
    if not running_file.exists():
        return True
    if stale_running == "remove":
        running_file.unlink()
        logger.warning(f"Removed a leftover RUNNING file in {sim.sim_dir}.")
        return True
    logger.warning(
        f"Skipping {sim.sim_dir}: leftover RUNNING file from a "
        "previous run (pass stale_running='remove' to clear it)."
    )
    return False


def _stage(sim: Simulation, prep: bool) -> None:
    """Prepare *sim*'s directory now, or only place the binary for a node-side prep."""
    if prep:
        sim.run(only_prep=True, copy_bin=True)
    else:
        Path(sim.sim_dir).mkdir(parents=True, exist_ok=True)
        _place_binary(Path(sim.alf_src.alf_dir, "Prog", "ALF.out"), Path(sim.sim_dir))


BINARY_DIR = ".alf_bin"


def _frozen_binary(src: Path, root: Path) -> Path:
    """One read-only copy of *src* under *root*, named by its content.

    A rebuild gets a new name, so a queued chain keeps the binary it was staged
    with, as a copy per directory guaranteed.
    """
    stat = src.stat()
    key = (str(src), stat.st_mtime_ns, stat.st_size, str(root))
    if key not in _FROZEN:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        frozen = root / BINARY_DIR / f"ALF-{digest}.out"
        if not frozen.exists():
            frozen.parent.mkdir(parents=True, exist_ok=True)
            tmp = frozen.with_suffix(f".tmp{os.getpid()}")
            shutil.copy2(src, tmp)
            tmp.chmod(0o555)
            os.replace(tmp, frozen)
        _FROZEN[key] = frozen
    return _FROZEN[key]


_FROZEN: dict[tuple, Path] = {}


def _place_binary(src: Path, sim_dir: Path) -> None:
    """Hard-link the frozen binary as ``sim_dir/ALF.out``; copy across filesystems.

    Copying a multi-megabyte binary into every directory dominated submitting
    a grid of hundreds of thousands of chains; a link costs one metadata write.
    """
    dest = sim_dir / "ALF.out"
    frozen = _frozen_binary(src, sim_dir.parent)
    if dest.exists() and dest.samefile(frozen):
        return
    dest.unlink(missing_ok=True)
    try:
        os.link(frozen, dest)
    except OSError:
        shutil.copy(src, dest)
