"""

cluster_submission
==================

Provides interfaces for running ALF simulations on a cluster.

"""

from __future__ import annotations

__author__ = "Johannes Hofmann"
__copyright__ = "Copyright 2020-2025, The ALF Project"
__license__ = "GPL"

import logging
import os
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal, TypedDict

import submitit
from submitit.core.utils import JobPaths

from .execute import exec_alf_binary
from .simulation import Simulation
from .slurm import ACTIVE_STATES, job_states

logger = logging.getLogger(__name__)


class PartitionSpec(TypedDict, total=False):
    """Per-partition SLURM node resource limits.

    Used as values in the *partition_rules* mapping passed to
    :class:`ClusterSubmitter`.  A bare ``float`` is also accepted and is
    interpreted as *max_hours* only.

    max_hours : float
        Wall-time limit in hours (required).
    max_cpus : int, optional
        Maximum CPUs available per node.  When supplied the submitter
        raises :class:`ValueError` if the job's ``n_mpi * n_omp`` (or
        just ``n_omp`` for non-MPI runs) would exceed this.
    max_mem_gb : float, optional
        Maximum node memory in GB.  When supplied the submitter raises
        :class:`ValueError` if ``slurm_mem`` would exceed this.
    """

    max_hours: float
    max_cpus: int
    max_mem_gb: float


def _parse_mem_gb(mem_str: str) -> float:
    """Parse a SLURM-style memory string into GB.

    Accepted suffixes (case-insensitive): K, M, G, T.
    No suffix → megabytes (SLURM's default unit for ``--mem``).

    Examples
    --------
    >>> _parse_mem_gb("8G")
    8.0
    >>> _parse_mem_gb("512M")
    0.5
    >>> _parse_mem_gb("1T")
    1024.0
    """
    s = mem_str.strip()
    if not s:
        raise ValueError("Empty memory string")
    suffix = s[-1].upper() if s[-1].isalpha() else ""
    try:
        num = float(s[:-1]) if suffix else float(s)
    except ValueError as err:
        raise ValueError(f"Cannot parse memory string: {mem_str!r}") from err
    factors: dict[str, float] = {
        "K": 1 / 1024**2,  # KB → GB
        "M": 1 / 1024,  # MB → GB
        "G": 1.0,
        "T": 1024.0,  # TB → GB
        "": 1 / 1024,  # no suffix = MB (SLURM default)
    }
    if suffix not in factors:
        raise ValueError(f"Unknown memory suffix {suffix!r} in {mem_str!r}")
    return num * factors[suffix]


def _normalise_partition_spec(
    name: str, value: float | int | PartitionSpec | dict
) -> PartitionSpec:
    """Coerce a *partition_rules* value to a :class:`PartitionSpec` dict."""
    if isinstance(value, (int, float)):
        return PartitionSpec(max_hours=float(value))
    d = dict(value)
    if "max_hours" not in d:
        raise ValueError(
            f"partition_rules[{name!r}]: dict entries must contain 'max_hours'; "
            f"got keys {sorted(d)!r}"
        )
    unknown = set(d) - {"max_hours", "max_cpus", "max_mem_gb"}
    if unknown:
        raise ValueError(
            f"partition_rules[{name!r}]: unknown keys {sorted(unknown)!r}; "
            f"valid keys are 'max_hours', 'max_cpus', 'max_mem_gb'"
        )
    return PartitionSpec(**d)


def _parse_slurm_time_hours(time_str: str) -> float | None:
    """Parse a SLURM time-limit string into fractional hours.

    Accepted formats (case-insensitive):

    * ``UNLIMITED`` / ``INFINITE`` / ``NOT_SET`` → *None*
    * ``MM``
    * ``MM:SS``
    * ``HH:MM:SS``
    * ``D-HH:MM:SS``

    Returns *None* for unlimited or unparseable values.
    """
    s = time_str.strip()
    if not s or s.upper() in ("UNLIMITED", "INFINITE", "NOT_SET"):
        return None
    try:
        days = 0
        if "-" in s:
            day_part, s = s.split("-", 1)
            days = int(day_part)
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        elif len(parts) == 2:
            h, m, sec = 0, int(parts[0]), int(parts[1])
        elif len(parts) == 1:
            h, m, sec = 0, 0, int(parts[0])
        else:
            return None
        return days * 24 + h + m / 60 + sec / 3600
    except (ValueError, IndexError):
        return None


def _project_root() -> Path:
    """Walk up from CWD to find the project root, identified by .git or common markers.

    Falls back to CWD when no root is found (e.g. outside any repository).
    This anchors .pyalf like .git — always at the repo root, never scattered
    across sub-directories depending on where the script was launched from.
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


def _slurm_time_to_minutes(value: int | str) -> int:
    """Normalise a slurm_time value to integer minutes.

    Accepts an integer (already in minutes) or an HH:MM:SS / D-HH:MM:SS string.
    Raises ValueError for unrecognised strings.
    """
    if isinstance(value, int):
        return value
    hours = _parse_slurm_time_hours(value)
    if hours is None:
        raise ValueError(
            f"Cannot parse slurm_time {value!r} — expected an integer (minutes) "
            "or a string in HH:MM:SS / D-HH:MM:SS format."
        )
    return int(hours * 60)


class ClusterSubmitter:
    """
    Handles job submission using submitit.

    Parameters
    ----------
    executor : {'slurm', 'local', 'debug'}
        Backend to use. ``'slurm'`` submits to a SLURM cluster; ``'local'``
        runs jobs in local processes (useful for testing without SLURM);
        ``'debug'`` runs jobs inline and synchronously.
    submit_dir : str or Path
        Directory where submitit writes job logs and state.
    slurm_mem : str
        Memory request per node (e.g. ``'2G'``, ``'8G'``). Required when
        *executor* is ``'slurm'``.
    partition_rules : dict[str, float | PartitionSpec]
        Mapping of SLURM partition name → resource limits.  Each value is
        either a plain ``float`` (wall-time limit in hours, backward
        compatible) or a :class:`PartitionSpec` dict with keys:

        * ``max_hours`` (**required**) – wall-time limit in hours
          (fractions allowed, e.g. ``10/60`` for 10 minutes).
        * ``max_cpus`` (*optional*) – maximum CPUs per node; submission
          fails if ``n_mpi × n_omp`` would exceed this.
        * ``max_mem_gb`` (*optional*) – maximum node memory in GB;
          submission fails if ``slurm_mem`` would exceed this.

        At submission time the partition with the smallest *max_hours*
        that is still ≥ the job's ``CPU_MAX`` is selected automatically.
        Required when *executor* is ``'slurm'``.  Exclude GPU-only
        partitions from CPU workloads.

        Example (typical HPC cluster, minimal)::

            partition_rules={
                "short":      2,      # 2 h
                "medium":     48,     # 2 days
                "long":       336,    # 14 days
                "extra_long": 672,    # 28 days
            }

        Example with per-node resource limits::

            partition_rules={
                "short":  {"max_hours": 2,   "max_cpus": 64,  "max_mem_gb": 256},
                "medium": {"max_hours": 48,  "max_cpus": 128, "max_mem_gb": 512},
                "long":   {"max_hours": 336, "max_cpus": 128, "max_mem_gb": 512},
            }

        The ``debug`` partition (10-minute wall time) is intentionally
        omitted here because ``CPU_MAX`` is always at least 1 hour; submit
        debug-partition jobs explicitly via ``job_properties``.

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
                    name: _normalise_partition_spec(name, spec)
                    for name, spec in partition_rules.items()
                }
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid partition_rules: {exc}") from exc

        self.executor = executor
        self.submit_dir = (
            Path(submit_dir).resolve()
            if submit_dir is not None
            else (_project_root() / ".pyalf")
        )
        self.slurm_mem = slurm_mem
        self.partition_rules: dict[str, PartitionSpec] | None = partition_rules
        self.job_name = job_name
        self.executor_params = executor_params

    def _select_partition(self, timeout_hours: float) -> str:
        """Select the smallest-limit partition that accommodates *timeout_hours*."""
        for name, spec in sorted(
            self.partition_rules.items(), key=lambda kv: kv[1]["max_hours"]
        ):
            if timeout_hours <= spec["max_hours"]:
                return name
        configured = ", ".join(
            f"{n}: {_format_hours(s['max_hours'])}"
            for n, s in self.partition_rules.items()
        )
        raise ValueError(
            f"No configured partition fits a {_format_hours(timeout_hours)} timeout. "
            f"Extend partition_rules or reduce CPU_MAX. "
            f"Configured: {{{configured}}}"
        )

    def _wall_time(
        self, sim: Simulation, slurm_time: int | str | None
    ) -> tuple[int, str | None]:
        """``(minutes, partition)`` for *sim*; the partition is None off SLURM.

        An explicit *slurm_time* is used as given. Otherwise the job asks for
        ``CPU_MAX`` plus 10%, so ALF can finish its last bin and write its output
        after stopping, capped at the limit of the partition ``CPU_MAX`` fits.
        """
        slurm = self.executor == "slurm"
        if slurm_time is not None:
            minutes = _slurm_time_to_minutes(slurm_time)
            return minutes, self._select_partition(minutes / 60) if slurm else None

        sim_dict = sim.sim_dict[0] if isinstance(sim.sim_dict, list) else sim.sim_dict
        cpu_max = float(sim_dict.get("CPU_MAX", 0))
        if not slurm:
            return int(max(cpu_max, 0.0) * 60), None
        if cpu_max <= 0:
            raise ValueError(
                "CPU_MAX=0 means ALF stops after Nbin bins with no internal "
                "time limit, so a SLURM wall time cannot be derived automatically. "
                "Pass slurm_time (int minutes or HH:MM:SS) to ClusterSubmitter "
                "or job_properties."
            )
        partition = self._select_partition(cpu_max)
        hours = min(cpu_max * 1.1, float(self.partition_rules[partition]["max_hours"]))
        return int(hours * 60), partition

    def _check_node_fit(
        self,
        sim: Simulation,
        partition: str,
        slurm_mem: str | None = None,
    ) -> None:
        """Raise :class:`ValueError` if resources exceed the partition's per-node limits.

        Parameters
        ----------
        sim:
            The simulation whose ``n_mpi``, ``n_omp``, and ``mpi`` attributes
            define the CPU footprint.
        partition:
            Name of the SLURM partition that has been (or will be) selected.
        slurm_mem:
            Memory string to check (e.g. ``'8G'``).  Defaults to
            ``self.slurm_mem`` when *None*.

        Raises
        ------
        ValueError
            If ``n_mpi × n_omp`` exceeds ``max_cpus``, or if the requested
            memory exceeds ``max_mem_gb`` for the given *partition*.
        """
        spec = self.partition_rules[partition]

        # ── CPU check ──────────────────────────────────────────────────────────
        total_cpus = (sim.n_mpi if sim.mpi else 1) * sim.n_omp
        max_cpus = spec.get("max_cpus")
        if max_cpus is not None and total_cpus > max_cpus:
            detail = (
                f"n_mpi={sim.n_mpi} × n_omp={sim.n_omp}"
                if sim.mpi
                else f"n_omp={sim.n_omp}"
            )
            raise ValueError(
                f"Requested {total_cpus} CPU(s) ({detail}) exceeds "
                f"partition '{partition}' per-node CPU limit of {max_cpus}."
            )

        # ── Memory check ───────────────────────────────────────────────────────
        effective_mem = slurm_mem if slurm_mem is not None else self.slurm_mem
        max_mem_gb = spec.get("max_mem_gb")
        if max_mem_gb is not None and effective_mem:
            try:
                req_gb = _parse_mem_gb(effective_mem)
            except ValueError:
                return  # unparseable → skip
            if req_gb > max_mem_gb:
                raise ValueError(
                    f"Requested memory {effective_mem} ({req_gb:.3g} GB) exceeds "
                    f"partition '{partition}' per-node memory limit of {max_mem_gb} GB."
                )

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
        """
        Submit one or more Simulation instances to the SLURM cluster.

        Prepares simulation directories, filters out already-running or broken
        jobs, then submits via submitit. Job IDs are written to ``jobid.txt``
        inside each simulation directory so that the status-checking helpers
        in this module continue to work.

        Parameters
        ----------
        sims : Simulation or iterable of Simulation
            Simulation(s) to submit.
        job_properties : dict, optional
            Override default SLURM parameters. Keys must match
            ``executor.update_parameters()`` keyword arguments.
        submit_dir : str or Path, optional
            Directory for submitit logs and state for this submission, which
            may contain submitit's ``%A``/``%j`` placeholders. Overrides the
            instance-level ``submit_dir`` set at construction.
        runner : callable, optional
            Function submitit executes on the worker, called with one
            ``Simulation``. Defaults to :func:`~py_alf.execute.exec_alf_binary`, which execs the binary
            directly. A caller that must decide *on the node* how to run (e.g.
            sizing ``CPU_MAX`` from the bins already on disk) passes its own,
            usually together with ``prep=False``.
        prep : bool, default=True
            Run ``sim.run(only_prep=True)`` for each simulation at submission
            time, writing ``parameters``/``seeds`` and renaming
            ``confout_* -> confin_*``. Set ``False`` when *runner* preps the
            directory itself: for a job that may be requeued the rename must
            happen each time it starts, not once when it was queued. The ALF
            binary is copied into the simulation directory either way, so the
            worker can rely on it being there.
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

        _SIM_ATTRS = ("sim_dir", "sim_dict", "ham_name", "n_omp", "n_mpi", "mpi", "run")
        if (
            isinstance(sims, Iterable)
            and not isinstance(sims, (str, bytes))
            and not all(hasattr(sims, a) for a in _SIM_ATTRS)
        ):
            sim_list = list(sims)
        else:
            sim_list = [sims]

        for s in sim_list:
            missing = [a for a in _SIM_ATTRS if not hasattr(s, a)]
            if missing:
                raise TypeError(
                    f"Expected Simulation-like object (missing {missing!r}), got {type(s)}"
                )

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

            running_file = Path(s.sim_dir) / "RUNNING"
            if running_file.exists():
                if stale_running == "remove":
                    running_file.unlink()
                    logger.warning(f"Removed a leftover RUNNING file in {s.sim_dir}.")
                else:
                    logger.warning(
                        f"Skipping {s.sim_dir}: leftover RUNNING file from a "
                        "previous run (pass stale_running='remove' to clear it)."
                    )
                    continue

            filtered_sims.append(s)

        if not filtered_sims:
            logger.info("No inactive simulations to submit.")
            return []

        sim = filtered_sims[0]

        # Guard: all sims in an array job must share the same resource shape,
        # since submitit applies one set of SLURM parameters to every task.
        if len(filtered_sims) > 1:
            for s in filtered_sims[1:]:
                if s.n_omp != sim.n_omp or s.n_mpi != sim.n_mpi or s.mpi != sim.mpi:
                    raise ValueError(
                        "All simulations in an array job must have the same n_omp, n_mpi, "
                        f"and mpi settings (derived from filtered_sims[0]: n_omp={sim.n_omp}, "
                        f"n_mpi={sim.n_mpi}, mpi={sim.mpi}), but {s.sim_dir} has "
                        f"n_omp={s.n_omp}, n_mpi={s.n_mpi}, mpi={s.mpi}."
                    )

        # A per-call slurm_time wins over the instance one; without either the
        # wall time follows CPU_MAX.
        slurm_time = (job_properties or {}).get(
            "slurm_time", self.executor_params.get("slurm_time")
        )
        timeout_min, partition = self._wall_time(sim, slurm_time)

        # Build executor parameters from defaults, instance-level kwargs,
        # then per-call overrides.
        #
        # Resource layout for a hybrid MPI + OpenMP job
        # -----------------------------------------------
        # tasks_per_node = n_mpi  →  SLURM allocates n_mpi task slots per node,
        #                            each with cpus_per_task = n_omp CPU cores.
        # Total cores on the node  = n_mpi × n_omp, matching exactly what
        # `mpiexec -n n_mpi ./ALF.out` with OMP_NUM_THREADS=n_omp will consume.
        #
        # For a pure-OpenMP (no MPI) job tasks_per_node is 1, so a single task
        # slot owns all n_omp cores and OMP_NUM_THREADS=n_omp fills them.
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
            self._check_node_fit(sim, partition)
            if sim.mpi:
                # submitit's default batch script wraps the Python launcher in
                # `srun` *without* an explicit -n flag.  With
                # #SBATCH --ntasks-per-node=n_mpi that outer srun therefore
                # spawns n_mpi copies of the Python process.  Each copy then
                # independently calls `mpiexec -n n_mpi ./ALF.out`, producing
                # n_mpi² ALF processes and triggering a nested srun / mpiexec
                # PMI conflict (see facebookincubator/submitit#1757).
                #
                # use_srun=False makes the batch script call Python directly
                # (exactly one process).  That single process then invokes
                # `mpiexec -n n_mpi`, which sees the n_mpi SLURM task slots
                # and distributes processes correctly across them.
                #
                # OMP_NUM_THREADS is set to sim.n_omp inside sim.run() before
                # mpiexec is called, consistent with cpus_per_task=n_omp so
                # each MPI rank fills exactly its allocated cores with threads.
                params["slurm_use_srun"] = False
        params.update(self.executor_params)
        params.update(job_properties or {})
        # Consumed into timeout_min above.
        params.pop("slurm_time", None)

        # Prepare simulation directories and copy binary. With prep=False the
        # runner preps on the node, so only the binary is staged here.
        for s in filtered_sims:
            if prep:
                s.run(only_prep=True, copy_bin=True)
            else:
                Path(s.sim_dir).mkdir(parents=True, exist_ok=True)
                shutil.copy(
                    os.path.join(s.alf_src.alf_dir, "Prog", "ALF.out"), s.sim_dir
                )

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
