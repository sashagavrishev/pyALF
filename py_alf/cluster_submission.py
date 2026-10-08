"""

cluster_submission
==================

Provides interfaces for running ALF simulations on a cluster.

"""

from __future__ import annotations

__author__ = "Johannes Hofmann"
__copyright__ = "Copyright 2020-2025, The ALF Project"
__license__ = "GPL"

import contextlib
import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Literal, TypedDict

import submitit

from .simulation import Simulation, getenv

logger = logging.getLogger(__name__)


# Upper bound on concurrent filesystem probes (bin counts, submitit log reads,
# ...).  These threads spend most of their time blocked on a networked
# filesystem, but h5py's own parsing of each data.h5 is real CPU work, and a
# login node is shared with everyone else logged into it -- so on a machine
# with room to spare the width tracks its core count rather than growing
# purely with however much latency there is to hide.  Clamped at both ends:
# a floor of 8 keeps real overlap available on a constrained sandbox or CI
# container (whose core count reflects nothing about a login node and can be
# too low to run the fan-out concurrently at all), and a ceiling of 32 avoids
# oversubscribing a very large machine for what is still I/O-bound work.
# Below _MIN_FANOUT items the pool costs more to start than the I/O it would
# overlap. Shared by any caller that probes many sim directories at once,
# such as :class:`py_alf.campaign.Campaign`.
_MAX_IO_WORKERS = min(32, max(8, os.cpu_count() or 16))
_MIN_FANOUT = 3


_io_pool: ThreadPoolExecutor | None = None
_io_pool_lock = threading.Lock()


def _get_io_pool() -> ThreadPoolExecutor:
    """The shared probe pool, created on first use.

    Reused across refreshes rather than rebuilt each time: spawning the
    workers costs more than the probes themselves once the filesystem is
    fast. The threads are joined by concurrent.futures' own atexit hook, so
    there is no lifecycle to manage here.
    """
    global _io_pool
    with _io_pool_lock:
        if _io_pool is None:
            _io_pool = ThreadPoolExecutor(
                max_workers=_MAX_IO_WORKERS, thread_name_prefix="alf-io"
            )
        return _io_pool


def _map_io(fn, items: list) -> list:
    """Apply *fn* to *items*, concurrently when there is enough work to justify it.

    The probes are independent and block on filesystem latency, so on a
    cluster filesystem the fan-out dominates: at ~5 ms per operation this
    turns a 32-row refresh from ~200 ms into ~15 ms. On a local disk the pool
    is pure overhead, but well under a millisecond either way.

    *fn* must not itself call _map_io: the pool is shared and finite, so a
    nested call could wait on a worker that never frees.
    """
    if len(items) < _MIN_FANOUT:
        return [fn(item) for item in items]
    return list(_get_io_pool().map(fn, items))


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


def _sanitise_nodelist(raw: str | None) -> str | None:
    """Return the raw SLURM nodelist string, or *None* when there is no real node.

    ``squeue``'s ``%N`` field returns a parenthesised reason string such as
    ``(Priority)`` for pending or blocked jobs, and the actual node list
    (e.g. ``compute01`` or ``node[1-4]``) for running jobs.
    ``sacct``'s ``NodeList`` column returns the literal string ``"None"`` when
    no allocation has been made yet.
    """
    if not raw or raw in ("None", "N/A", "none"):
        return None
    if raw.startswith("("):  # pending-reason e.g. "(Priority)", "(Resources)"
        return None
    return raw


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


def detect_partition_rules(
    exclude: list[str] | None = None,
    include: list[str] | None = None,
    mem_headroom_gb: float = 2.0,
    timeout: float = 10.0,
) -> dict[str, PartitionSpec]:
    """Query the local SLURM installation and build a *partition_rules* dict.

    Runs ``sinfo -o "%P|%l|%c|%m" --noheader`` and converts the output into
    a mapping of partition name → :class:`PartitionSpec`.  When a partition
    has multiple node groups (multiple ``sinfo`` lines), the *minimum* CPU
    count and *minimum* memory are used — the conservative choice that
    guarantees the limits hold for every node in the partition.

    Partitions with an ``UNLIMITED`` time limit are excluded because
    :class:`ClusterSubmitter` requires a finite ``max_hours`` to select
    a partition automatically.

    Parameters
    ----------
    exclude : list of str, optional
        Partition names to ignore, e.g. GPU-only or interactive partitions.
        Matching is case-insensitive.
    include : list of str, optional
        If given, *only* these partition names are returned; all others are
        dropped.  Matching is case-insensitive.
    mem_headroom_gb : float
        Gigabytes subtracted from the raw per-node memory reported by
        ``sinfo`` to leave headroom for OS and system daemons.
        Default is ``2.0``.
    timeout : float
        Seconds to wait for the ``sinfo`` subprocess before raising.
        Default is ``10.0``.

    Returns
    -------
    dict[str, PartitionSpec]
        Ready to pass directly to :class:`ClusterSubmitter` as
        *partition_rules*.

    Raises
    ------
    RuntimeError
        If ``sinfo`` is not found on PATH, times out, or returns no
        partitions that survive the filters and have finite time limits.

    Examples
    --------
    Detect all finite-time-limit partitions, excluding the GPU queue::

        rules = detect_partition_rules(exclude=["gpu"])
        cs = ClusterSubmitter("slurm", slurm_mem="8G", partition_rules=rules)

    Detect only specific partitions::

        rules = detect_partition_rules(include=["short", "medium", "long"])
    """
    exclude_set = {p.lower() for p in (exclude or [])}
    include_set = {p.lower() for p in include} if include else None

    try:
        result = subprocess.run(
            ["sinfo", "-o", "%P|%l|%c|%m", "--noheader"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "detect_partition_rules: 'sinfo' not found — "
            "is SLURM installed and on PATH?"
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"detect_partition_rules: 'sinfo' timed out after {timeout} s"
        ) from None

    # Accumulate (max_hours, cpus, mem_mb) tuples per partition name.
    # Multiple tuples arise when a partition spans several node groups.
    raw: dict[str, list[tuple[float, int, int]]] = {}

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 4:
            logger.debug("detect_partition_rules: skipping unrecognised line %r", line)
            continue
        name_raw, time_raw, cpus_raw, mem_raw = parts[:4]

        # SLURM marks the default partition with a trailing '*'
        name = name_raw.strip().rstrip("*").strip()
        if not name:
            continue

        if name.lower() in exclude_set:
            continue
        if include_set is not None and name.lower() not in include_set:
            continue

        hours = _parse_slurm_time_hours(time_raw)
        if hours is None:
            logger.debug(
                "detect_partition_rules: skipping partition %r (UNLIMITED time)", name
            )
            continue

        try:
            # sinfo may report "72+" meaning ≥72 CPUs; strip any trailing non-digit chars
            cpus = int(cpus_raw.strip().rstrip("+"))
        except ValueError:
            logger.warning(
                "detect_partition_rules: cannot parse CPU count %r for %r — skipping row",
                cpus_raw,
                name,
            )
            continue

        try:
            mem_mb = int(mem_raw.strip().rstrip("+"))
        except ValueError:
            logger.warning(
                "detect_partition_rules: cannot parse memory %r for %r — skipping row",
                mem_raw,
                name,
            )
            continue

        raw.setdefault(name, []).append((hours, cpus, mem_mb))

    if not raw:
        raise RuntimeError(
            "detect_partition_rules: no usable partitions found after filtering. "
            "Verify that 'sinfo' is working and adjust the exclude/include lists."
        )

    rules: dict[str, PartitionSpec] = {}
    for name, entries in raw.items():
        max_hours = min(h for h, _, _ in entries)
        max_cpus = min(c for _, c, _ in entries)
        raw_mem_gb = min(m for _, _, m in entries) / 1024 - mem_headroom_gb
        if raw_mem_gb <= 0:
            logger.warning(
                "detect_partition_rules: partition %r has %.1f GB after headroom "
                "deduction — skipping.",
                name,
                raw_mem_gb + mem_headroom_gb,
            )
            continue
        rules[name] = PartitionSpec(
            max_hours=max_hours,
            max_cpus=max_cpus,
            max_mem_gb=round(raw_mem_gb, 3),
        )

    if not rules:
        raise RuntimeError(
            "detect_partition_rules: all detected partitions were excluded or had "
            "unusable specs (e.g. memory too small after headroom deduction)."
        )

    return rules


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


def _unique_slurm_job_name(base_name: str) -> str:
    """Return a SLURM job name that is not currently held by any queued job.

    Queries ``squeue`` for all active jobs whose name starts with *base_name*
    and appends a numeric suffix (``_2``, ``_3``, …) until an unused name is
    found.  If ``squeue`` is unavailable the original name is returned unchanged
    so that submission is never blocked.
    """
    try:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%j"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        active_names: set[str] = {
            line.strip() for line in result.stdout.splitlines() if line.strip()
        }
    except Exception:
        return base_name

    if base_name not in active_names:
        return base_name

    for suffix in range(2, 10000):
        candidate = f"{base_name}_{suffix}"
        if candidate not in active_names:
            return candidate

    return base_name


def _exec_alf_binary(
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
    from .simulation import cd

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


def _run_alf(sim: Simulation) -> None:
    """
    Execute an ALF simulation on a cluster node.

    Called by submitit on the remote worker. Assumes the binary has already been copied
    into sim.sim_dir by the pre-submission preparation step.
    """
    _exec_alf_binary(
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
        SLURM job name (``--job-name``). Defaults to the hamiltonian name.
        Accepted for all executors.
    mail_type : str, optional
        SLURM mail event type, e.g. ``'END'``, ``'FAIL'``, ``'ALL'``.
        Only valid when *executor* is ``'slurm'``.
    wckey : str, optional
        SLURM workload-characterisation key (``--wckey``).
        Only valid when *executor* is ``'slurm'``.
    stderr_to_stdout : bool
        Redirect stderr to the stdout log file. Accepted for all executors.
    **slurm_kwargs
        Additional keyword arguments forwarded to
        ``executor.update_parameters()``. Keys prefixed with ``slurm_``
        are sent as raw ``#SBATCH`` directives. Only valid when *executor*
        is ``'slurm'``. ``slurm_max_num_timeout`` is the exception: it is
        a submitit executor setting (how many times a timed-out task may be
        requeued, default 3) and is forwarded to the executor's constructor
        instead.

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
        mail_type: str | None = None,
        wckey: str | None = None,
        stderr_to_stdout: bool = False,
        **slurm_kwargs,
    ):
        if executor not in self._VALID_EXECUTORS:
            raise ValueError(
                f"executor must be one of {self._VALID_EXECUTORS!r}, got {executor!r}"
            )

        if executor != "slurm":
            slurm_specific = [k for k in slurm_kwargs if k.startswith("slurm_")]
            problems = (
                (["slurm_mem"] if slurm_mem is not None else [])
                + (["partition_rules"] if partition_rules is not None else [])
                + (["mail_type"] if mail_type is not None else [])
                + (["wckey"] if wckey is not None else [])
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
        self.mail_type = mail_type
        self.wckey = wckey
        self.stderr_to_stdout = stderr_to_stdout
        self.slurm_kwargs = slurm_kwargs

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

    def submit(
        self,
        sims: Simulation | Iterable[Simulation],
        job_properties: dict[str, Any] | None = None,
        submit_dir: str | Path | None = None,
        runner: Callable[[Simulation], None] | None = None,
        prep: bool = True,
        stale_running: Literal["remove", "skip"] = "skip",
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
            Directory for submitit logs and state for this submission.
            Overrides the instance-level ``submit_dir`` set at construction.
        runner : callable, optional
            Function submitit executes on the worker, called with one
            ``Simulation``. Defaults to :func:`_run_alf`, which execs the binary
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

        filtered_sims = []

        for s in sim_list:
            jobid_file = Path(s.sim_dir) / "jobid.txt"
            running_file = Path(s.sim_dir) / "RUNNING"

            jobid: str | None = (
                jobid_file.read_text().strip() if jobid_file.exists() else None
            )

            if jobid is not None and self.executor == "slurm":
                status_entry = _get_slurm_status_sacct(jobid)
                if status_entry.get("status") in ("PENDING", "RUNNING"):
                    logger.info(
                        f"Skipping {s.sim_dir}: job {jobid} is \
                           {status_entry.get('status')}"
                    )
                    continue

            if running_file.exists():
                if jobid is not None and self.executor == "slurm":
                    status_entry = _get_slurm_status_sacct(jobid)
                    if status_entry.get("status") == "RUNNING":
                        logger.info(f"Skipping {s.sim_dir}: job {jobid} is RUNNING")
                        continue
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

        # Same precedence as the params merge below, so the partition is chosen
        # for the wall time actually requested.
        _raw_slurm_time = (job_properties or {}).get("slurm_time")
        if _raw_slurm_time is None:
            _raw_slurm_time = (self.slurm_kwargs or {}).get("slurm_time")
        if _raw_slurm_time is not None:
            timeout_hours = _slurm_time_to_minutes(_raw_slurm_time) / 60
        else:
            _sim_dict0 = (
                sim.sim_dict[0] if isinstance(sim.sim_dict, list) else sim.sim_dict
            )
            cpu_max = float(_sim_dict0.get("CPU_MAX", 0))
            if cpu_max <= 0 and self.executor == "slurm":
                raise ValueError(
                    "CPU_MAX=0 means ALF stops after Nbin bins with no internal "
                    "time limit, so a SLURM wall time cannot be derived automatically. "
                    "Pass slurm_time (int minutes or HH:MM:SS) to ClusterSubmitter "
                    "or job_properties."
                )
            timeout_hours = cpu_max if cpu_max > 0 else 0.0

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
        base_name = self.job_name if self.job_name is not None else sim.ham_name
        if self.executor == "slurm" and self.job_name is None:
            base_name = _unique_slurm_job_name(base_name)
        params: dict[str, Any] = {
            "name": base_name,
            "timeout_min": int(timeout_hours * 60),
            "nodes": 1,
            "cpus_per_task": sim.n_omp,
            "tasks_per_node": sim.n_mpi if sim.mpi else 1,
        }
        if self.executor == "slurm":
            params["slurm_mem"] = self.slurm_mem
            params["slurm_partition"] = self._select_partition(timeout_hours)
            self._check_node_fit(sim, params["slurm_partition"])
            if self.mail_type is not None:
                params["slurm_mail_type"] = self.mail_type
            if self.wckey is not None:
                params["slurm_wckey"] = self.wckey
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
        if self.stderr_to_stdout:
            params["stderr_to_stdout"] = True
        params.update(self.slurm_kwargs)
        if job_properties:
            params.update(job_properties)
        if "slurm_time" in params:
            params["slurm_time"] = _slurm_time_to_minutes(params["slurm_time"])

        if self.executor == "slurm":
            extra = dict(params.get("slurm_additional_parameters") or {})
            # Add 10% buffer so ALF can finish writing output after CPU_MAX;
            # cap at the selected partition's wall-time limit.
            # Skip auto-computation when the caller already supplied slurm_time
            # (a submitit-style kwarg) or an explicit "time" in
            # slurm_additional_parameters, so user-set wall times are never
            # silently overwritten.
            if "slurm_time" not in params and "time" not in extra:
                slurm_time_h = timeout_hours * 1.1
                selected = params.get("slurm_partition")
                if (
                    selected
                    and self.partition_rules
                    and selected in self.partition_rules
                ):
                    max_h = float(
                        self.partition_rules[selected].get("max_hours", slurm_time_h)
                    )
                    slurm_time_h = min(slurm_time_h, max_h)
                extra["time"] = int(slurm_time_h * 60)
            params["slurm_additional_parameters"] = extra

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
        effective_submit_dir.mkdir(parents=True, exist_ok=True)

        # submitit takes the requeue budget when the executor is *constructed*,
        # not through update_parameters(), so it is pulled back out of params.
        max_num_timeout = params.pop("slurm_max_num_timeout", None)
        executor_kwargs: dict[str, Any] = {}
        if max_num_timeout is not None and self.executor == "slurm":
            executor_kwargs["slurm_max_num_timeout"] = int(max_num_timeout)

        executor = submitit.AutoExecutor(
            folder=str(effective_submit_dir),
            cluster=self.executor,
            **executor_kwargs,
        )
        executor.update_parameters(**params)

        run_fn = runner if runner is not None else _run_alf
        if len(filtered_sims) == 1:
            jobs = [executor.submit(run_fn, filtered_sims[0])]
        else:
            jobs = executor.map_array(run_fn, filtered_sims)

        # jobid.txt is how a later submit recognises a chain that is still active.
        for s, job in zip(filtered_sims, jobs):
            Path(s.sim_dir, "jobid.txt").write_text(job.job_id)

        logger.info(f"Submitted {len(jobs)} job(s): {[j.job_id for j in jobs]}")
        return jobs


# --- Status functions ---
def _normalize_slurm_state(raw: str) -> str:
    """Return a canonical SLURM state from a raw sacct or squeue value.

    sacct truncates state strings to its column width and appends ``+``, e.g.
    ``CANCELLED+`` (cancelled by uid) or ``OUT_OF_ME+`` (out of memory).
    This strips the truncation marker and restores full names.
    """
    state = raw.split()[0].rstrip("+")
    if state == "OUT_OF_ME":
        return "OUT_OF_MEMORY"
    return state


def _get_slurm_status_sacct(jobid: str) -> dict[str, str | None]:
    """
    Query SLURM sacct for job status, elapsed time, and allocated node.
    Returns dict: {'status': ..., 'runtime': ..., 'nodelist': ...}
    """
    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                jobid,
                "--format=State,Elapsed,NodeList",
                "--noheader",
                "--array",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        lines = result.stdout.strip().splitlines()
        logger.debug(lines)
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                state = _normalize_slurm_state(parts[0])
                runtime = parts[1] if len(parts) > 1 else None
                raw_node = parts[2] if len(parts) > 2 else None
                return {
                    "status": state,
                    "runtime": runtime,
                    "nodelist": _sanitise_nodelist(raw_node),
                }
        return {"status": "UNKNOWN", "runtime": None, "nodelist": None}
    except Exception as e:
        logger.error(f"sacct error for job {jobid}: {e}")
        return {"status": "ERROR", "runtime": None, "nodelist": None}


def _parent_ids(jobids: list[str]) -> list[str]:
    """Return the distinct array-parent IDs behind *jobids*, preserving order.

    Querying individual task IDs (e.g. "12345_1") is unreliable on some SLURM
    versions; the parent ID "12345" always returns every task row.
    """
    seen: dict[str, None] = {}
    for jid in jobids:
        parts = jid.rsplit("_", 1)
        seen[parts[0] if len(parts) == 2 and parts[1].isdigit() else jid] = None
    return list(seen)


def _get_slurm_status_bulk_sacct(
    jobids: list[str],
) -> dict[str, dict[str, str | None]]:
    """
    Query SLURM sacct for multiple job IDs (including array tasks) in one call.
    Returns dict: jobid[_index] -> {'status': ..., 'runtime': ..., 'nodelist': ...}
    """
    status_map: dict[str, dict[str, str | None]] = {
        jid: {"status": "UNKNOWN", "runtime": None, "nodelist": None} for jid in jobids
    }
    if not jobids:
        return status_map

    try:
        result = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(_parent_ids(jobids)),
                "--format=JobID,State,Elapsed,NodeList",
                "--noheader",
                "--array",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                jobid = parts[0]
                # sacct also returns ".batch"/".extern" sub-steps; keep only the
                # job rows that were actually asked for.
                if jobid not in status_map:
                    continue
                state = _normalize_slurm_state(parts[1])
                runtime = parts[2] if len(parts) > 2 else None
                raw_node = parts[3] if len(parts) > 3 else None
                status_map[jobid] = {
                    "status": state,
                    "runtime": runtime,
                    "nodelist": _sanitise_nodelist(raw_node),
                }
    except Exception as e:
        logger.error(f"sacct bulk error: {e}")
        for jid in jobids:
            status_map[jid] = {"status": "ERROR", "runtime": None, "nodelist": None}
    return status_map


def _get_slurm_status_bulk(jobids: list[str]) -> dict[str, dict[str, str | None]]:
    """
    Query SLURM for multiple job IDs (including array tasks) in one call.

    Uses ``squeue`` first (fast, live data); falls back to ``sacct`` for jobs
    that are no longer in the scheduler queue (completed, failed, etc.).

    Returns
    -------
    dict
        Mapping ``jobid[_task]`` → ``{'status': str, 'runtime': str|None,
        'nodelist': str|None}``.  ``nodelist`` is the allocated compute node
        for running jobs, or *None* for pending / finished / inactive jobs.
    """
    if not jobids:
        return {}

    # A job in a terminal state can never change again, so it is served from
    # cache without touching SLURM.  This matters most for the sacct fallback
    # below: terminal jobs have left the queue, so leaving them in the query set
    # would make every refresh of a partly-finished session pay for an sacct
    # call that can only return what is already known.
    cached = {
        jid: _terminal_status_cache[jid]
        for jid in jobids
        if jid in _terminal_status_cache
    }
    jobids = [jid for jid in jobids if jid not in _terminal_status_cache]
    if not jobids:
        return cached

    status_map: dict[str, dict[str, str | None]] = {
        jid: {"status": "FINISHED_OR_NOT_FOUND", "runtime": None, "nodelist": None}
        for jid in jobids
    }
    found_in_squeue = set()

    try:
        result = subprocess.run(
            [
                "squeue",
                "-h",
                "-o",
                "%A %i %T %M %N",
                "--array",
                "-j",
                ",".join(_parent_ids(jobids)),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        squeue_failed = False
    except subprocess.TimeoutExpired:
        logger.warning(
            "squeue command timed out. Falling back to sacct for job status."
        )
        squeue_failed = True
    except Exception as e:
        logger.error(f"Error running squeue: {e}")
        squeue_failed = True

    if not squeue_failed:
        for line in result.stdout.strip().splitlines():
            try:
                # maxsplit=4: the 5th token is the nodelist (may be absent for
                # pending jobs that squeue shows with an empty nodelist field)
                parts = line.split(maxsplit=4)
                if len(parts) < 4:
                    logger.warning(f"Unexpected squeue output line: '{line}'")
                    continue
                jid, idx, raw_state, runtime = parts[:4]
                raw_node = parts[4] if len(parts) > 4 else None
                full_id = jid if idx == "N/A" else idx
                status_map[full_id] = {
                    "status": _normalize_slurm_state(raw_state),
                    "runtime": runtime,
                    "nodelist": _sanitise_nodelist(raw_node),
                }
                found_in_squeue.add(full_id)
            except Exception as e:
                logger.error(f"Error parsing squeue output line '{line}': {e}")
                continue
        # For jobs not found in squeue, fall back to sacct.
        # Also include COMPLETING jobs: that state means the job has finished
        # executing and SLURM is cleaning up — sacct may already have the true
        # terminal state (TIMEOUT, COMPLETED, FAILED, …).
        missing_jobids = [jid for jid in jobids if jid not in found_in_squeue]
        completing_ids = [
            jid
            for jid in found_in_squeue
            if status_map[jid].get("status") == "COMPLETING"
        ]
        need_sacct = missing_jobids + completing_ids
        if need_sacct:
            sacct_statuses = _get_slurm_status_bulk_sacct(need_sacct)
            for jid in missing_jobids:
                status_map[jid] = sacct_statuses.get(
                    jid, {"status": "UNKNOWN", "runtime": None, "nodelist": None}
                )
            for jid in completing_ids:
                entry = sacct_statuses.get(jid, {})
                if entry.get("status") not in (None, "UNKNOWN"):
                    status_map[jid] = entry
                # else: sacct hasn't caught up yet — keep COMPLETING
    else:
        # squeue failed, use sacct bulk for all jobids
        sacct_statuses = _get_slurm_status_bulk_sacct(jobids)
        for jid in jobids:
            status_map[jid] = sacct_statuses.get(
                jid, {"status": "UNKNOWN", "runtime": None, "nodelist": None}
            )

    for jid, entry in status_map.items():
        if entry.get("status") in _TERMINAL_STATES:
            _terminal_status_cache[jid] = dict(entry)

    status_map.update(cached)
    return status_map


_submitit_timeout_cache: dict[str, tuple[bool, bool]] = {}

# Terminal SLURM states are immutable, so they are cached per job ID and never
# re-queried.  Keyed by the full task ID ("12345" or "12345_7").
_terminal_status_cache: dict[str, dict[str, str | None]] = {}


def _is_submitit_timeout(
    jobid: str, submit_dir: str | Path, status: str = "FAILED"
) -> bool:
    """Return True if a FAILED or COMPLETED job was actually a wall-time timeout.

    submitit can cause SLURM to misreport the terminal state in two ways:

    - FAILED: submitit's SIGUSR1 handler fires before SLURM's kill, decides the
      job timed out, and exits non-zero.  Indicator: "this job is timed-out".
    - COMPLETED: SLURM sends SIGTERM at the wall time; submitit bypasses it, ALF
      exits cleanly before SIGKILL, so SLURM records COMPLETED.  Indicator:
      "Bypassing signal SIGTERM".  Not applied to FAILED to avoid
      misclassifying preempted-then-requeued jobs, which also log the SIGTERM
      bypass but correctly stay FAILED.
    """
    cached = _submitit_timeout_cache.get(jobid)
    if cached is None:
        text: str | None = None
        log_path = Path(submit_dir) / f"{jobid}_0_log.out"
        # Suppressing the read covers the missing file, so no separate exists().
        with contextlib.suppress(OSError):
            text = log_path.read_text(errors="replace")
        if text is None:
            # The log has not appeared yet — on a networked filesystem it can lag
            # the job's state change.  Caching "not timed out" now would pin that
            # answer for the session and misreport a real TIMEOUT, so report the
            # default without caching and look again next refresh.
            return False
        cached = (
            "this job is timed-out" in text,
            "Bypassing signal SIGTERM" in text,
        )
        _submitit_timeout_cache[jobid] = cached
    timed_out, sigterm_bypassed = cached
    return timed_out or (status == "COMPLETED" and sigterm_bypassed)


_TERMINAL_STATES: frozenset[str] = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "DEADLINE",
        "NODE_FAIL",
        "PREEMPTED",
    }
)


_bin_cache: dict[Any, int] = {}
# Keys read once while their job was already in a terminal state. Such a count
# can never change again, so it is served from _bin_cache without touching the
# filesystem. Callers monitoring many finished simulations would otherwise
# re-open every data.h5 on every refresh.
_bin_final: set[Any] = set()

# (st_mtime_ns, st_size) of the data.h5 each cached count was read from, used to
# skip the h5py open when the file has not changed since.
_bin_stat: dict[Any, tuple[int, int]] = {}

# HDF5 open failures that mean "ALF is part-way through writing this file", not
# "this file is broken".  ALF rewrites data.h5 in place, so a reader can catch it
# after the superblock records a new end-of-file but before the data is flushed --
# hence "truncated file: eof = ... < stored_eof = ...".  The condition clears
# itself as soon as the write completes.
_MIDWRITE_MARKERS = (
    "truncated file",
    "file signature not found",
    "unable to read superblock",
    "bad object header version number",
    "unable to lock file",
    "resource temporarily unavailable",
)

# Backoff between re-open attempts.  ALF's write window is short, so one or two
# retries usually turn a mid-write into a good read rather than a stale row.
_H5_RETRY_DELAYS = (0.05, 0.15)

# Consecutive failed reads of one file before a mid-write stops being treated as
# transient and gets reported.  At a 30s refresh this is minutes of failure, by
# which point the file is genuinely damaged rather than being written.
_MIDWRITE_LOG_AFTER = 5

_bin_read_failures: dict[Any, int] = {}


def _is_midwrite_error(exc: BaseException | str) -> bool:
    """True if *exc* looks like a read that raced ALF's writer.

    Accepts a string as well as an exception: a worker process's mid-write
    verdict has to cross back to the caller through
    :func:`_read_bin_count`'s return value, since the exception itself is not
    always picklable and does not need to survive the trip -- only whether it
    matched one of these markers does.
    """
    text = (exc if isinstance(exc, str) else str(exc)).lower()
    return any(marker in text for marker in _MIDWRITE_MARKERS)


# Filesystems record mtimes coarsely -- NFS commonly only to the second -- so a
# change landing in the same tick as a reading is invisible to it.  Caching such
# a reading would pin a stale answer until something else moved the mtime past
# the tick, so a reading is only trusted once its mtime has settled.
_MTIME_SETTLE_NS = 2_000_000_000


def _mtime_settled(mtime_ns: int) -> bool:
    """True if *mtime_ns* is far enough in the past to be a safe cache key."""
    return time.time_ns() - mtime_ns > _MTIME_SETTLE_NS


# h5py wraps every call into the HDF5 C library in a single process-wide lock
# (h5py._objects.phil) -- confirmed empirically: N threads each holding it for
# 50ms behave identically to N sequential 50ms calls, regardless of how many
# threads there are. So fanning bin-count reads out across _get_io_pool's
# threads (as Campaign.status() does) never actually overlaps the blocking
# h5py.File() open itself, only the Python-level dispatch around it. Only a
# separate OS process gets its own independent HDF5 library instance -- and
# hence its own phil -- so this pool is what actually lets two data.h5 opens
# progress at once. Deliberately smaller than _MAX_IO_WORKERS: each worker
# imports h5py/numpy into its own address space, which costs real memory an
# idle thread would not, and (unlike the thread pool) is opt-in per call
# rather than the default -- see _bin_count's ``use_process_pool``.
_process_pool: ProcessPoolExecutor | None = None
_process_pool_lock = threading.Lock()


def _process_pool_size() -> int:
    """Worker count for the h5py read pool, capped to any SLURM allocation.

    A bare process count defaults to the whole node's cores, not the cgroup a SLURM
    task was actually granted, and a status check run as its own job (e.g.
    ``reconcile`` on a timer) must not oversubscribe that allocation the way
    a handful of extra processes importing h5py/numpy each could.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    budget = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 4)
    return min(8, max(2, budget))


def _get_process_pool() -> ProcessPoolExecutor:
    """The shared process pool for h5py reads, created on first use."""
    global _process_pool
    with _process_pool_lock:
        if _process_pool is None:
            _process_pool = ProcessPoolExecutor(max_workers=_process_pool_size())
        return _process_pool


def _stat_sig(filename: str) -> tuple[int, int] | None:
    """``(st_mtime_ns, st_size)`` of *filename*, or None if it cannot be stat'd."""
    try:
        st = os.stat(filename)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _read_bin_count(
    filename: str, counting_obs: str
) -> tuple[int, bool, str | None, tuple[int, int] | None]:
    """Open *filename* and read its bin count, retrying through a mid-write race.

    Pure function of its arguments -- no cache, no module state read or
    written -- so it is safe to run inside a worker process via
    :func:`_get_process_pool`. The caller owns everything stateful: caching,
    the stale-value fallback, and the failure-count bookkeeping, none of which
    a worker process could share back with the caller's copy of those module
    dicts anyway.

    The file's (mtime, size) is taken here and handed back rather than stat'd by
    the caller, so that on a batch a grid's worth of stats runs across the pool
    instead of serially in one process -- on a networked filesystem a stat is
    milliseconds, which at campaign scale is minutes. Taken *before* the read
    for the reason :func:`_bin_count` documents: a write landing in between then
    leaves a stale signature that forces a re-read next time, which is the safe
    direction to be wrong in.
    """
    import h5py

    sig = _stat_sig(filename)
    N_bins = 0
    read_ok = False
    last_exc: BaseException | None = None
    for attempt in range(len(_H5_RETRY_DELAYS) + 1):
        try:
            # POSIX file locking stalls (or errors) on networked filesystems, and
            # a read-only probe gets nothing from it.
            with h5py.File(filename, "r", locking=False) as f:
                if counting_obs in f:
                    N_bins = f[counting_obs + "/obser"].shape[0]
            read_ok = True
            break
        except FileNotFoundError:
            break
        except (OSError, KeyError) as e:
            last_exc = e
            # Only a mid-write is worth retrying, and only while attempts remain.
            if not _is_midwrite_error(e) or attempt == len(_H5_RETRY_DELAYS):
                break
            time.sleep(_H5_RETRY_DELAYS[attempt])

    return N_bins, read_ok, repr(last_exc) if last_exc is not None else None, sig


def _bin_count(
    sim: Simulation,
    counting_obs: str = "Ener_scal",
    refresh: bool = False,
    data_dir: str | None = None,
    final: bool = False,
    force: bool = False,
    use_process_pool: bool = False,
) -> int:
    """
    Counts bins for a given observable in simulation data, with caching.
    Args:
        sim: Simulation instance.
        counting_obs: Observable name.
        refresh: Whether to refresh cache.
        data_dir: Directory holding ``data.h5`` to read instead of
            ``sim.sim_dir`` — used to count bins in a single ``Temp_i/``
            realisation of a PARALLEL_PARAMS job.
        final: Whether the job has reached a terminal state. The file is still
            read once (the last bins may have landed since the previous
            refresh), but the result is then frozen and served from cache.
        force: Skip the (mtime, size) short-circuit and re-read the file even if
            it looks unchanged.
        use_process_pool: Run the actual read in :func:`_get_process_pool`
            instead of this process. Worth it only when many chains are being
            probed at once (h5py serializes every call in-process regardless
            of thread count -- see that pool's docstring), so this defaults
            off: a lone call, like a worker job checking its own bin count,
            would pay a process pool's startup cost for nothing.
    Returns:
        Number of bins.
    """
    filename = os.path.join(
        data_dir if data_dir is not None else sim.sim_dir, "data.h5"
    )
    key = (filename, counting_obs)

    if key in _bin_final:
        return _bin_cache.get(key, 0)

    if (key in _bin_cache) and (not refresh):
        return _bin_cache[key]

    # ALF rewrites data.h5 as a whole, so an unchanged (mtime, size) means
    # unchanged content: a stat is far cheaper than letting h5py parse the file
    # structure, and a running job appends a bin far less often than the monitor
    # polls.  Stat *before* reading — a write landing between the two then leaves
    # a stale signature that forces a re-read next time, whereas stat-after would
    # pair the new signature with the old count and never re-read.
    stat_sig: tuple[int, int] | None = None
    try:
        st = os.stat(filename)
        stat_sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        pass
    if (
        not force
        and stat_sig is not None
        and key in _bin_cache
        and _bin_stat.get(key) == stat_sig
    ):
        return _bin_cache[key]

    if use_process_pool:
        read = (
            _get_process_pool().submit(_read_bin_count, filename, counting_obs).result()
        )
    else:
        read = _read_bin_count(filename, counting_obs)

    return _absorb_bin_read(key, filename, read, final)


def _absorb_bin_read(
    key: tuple[str, str],
    filename: str,
    read: tuple[int, bool, str | None, tuple[int, int] | None],
    final: bool,
) -> int:
    """Fold one :func:`_read_bin_count` result into the module caches.

    Split out of :func:`_bin_count` so the batched path
    (:func:`_bin_counts`) inherits the same caching, stale-value fallback and
    failure bookkeeping instead of reimplementing them. The signature travels
    in ``read`` because the reader takes it (see :func:`_read_bin_count`).
    """
    N_bins, read_ok, error_text, stat_sig = read

    if error_text is not None and not read_ok:
        fails = _bin_read_failures.get(key, 0) + 1
        _bin_read_failures[key] = fails
        # Racing ALF's writer is expected and self-correcting, so stay quiet
        # about it; a file that keeps failing is a real problem worth surfacing.
        if _is_midwrite_error(error_text) and fails < _MIDWRITE_LOG_AFTER:
            logger.debug(
                "%s is mid-write (attempt %d), keeping the cached bin count: %s",
                filename,
                fails,
                error_text,
            )
        else:
            logger.error(f"Error reading {filename}: {error_text}")
        # Keep the last known good value rather than caching a spurious 0 that
        # would be shown briefly on the next refresh.
        return _bin_cache.get(key, 0)

    _bin_read_failures.pop(key, None)

    # Don't let a transient 0 overwrite a previously-seen non-zero count —
    # ALF truncates and rewrites data.h5 between bins, so a 0 mid-write is
    # not meaningful and would cause the progress bar to flicker.  The stat
    # signature is deliberately not recorded here, so the next refresh re-reads
    # rather than caching the mid-write state.
    if N_bins == 0 and _bin_cache.get(key, 0) > 0:
        return _bin_cache[key]

    _bin_cache[key] = N_bins
    if read_ok and stat_sig is not None and _mtime_settled(stat_sig[0]):
        _bin_stat[key] = stat_sig
    # Only freeze a count that came from an actual read: a terminal job whose
    # data.h5 is missing or unreadable may still appear once the filesystem
    # catches up, or once a truncated file is repaired.
    if final and read_ok:
        _bin_final.add(key)
    return N_bins


# Files per task handed to the process pool by _bin_counts. One file per task
# makes every read cost a full pickle/IPC round trip, which on a campaign-sized
# batch dominates the h5py open it was meant to overlap; a chunk amortises that
# over many reads while staying small enough that the workers finish together.
_BIN_BATCH_CHUNK = 64


def _bin_counts(
    filenames: list[str],
    counting_obs: str = "Ener_scal",
    final: bool = False,
    force: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> list[int]:
    """Bin counts for many ``data.h5`` paths at once, in caller order.

    The batched counterpart of :func:`_bin_count`, for a caller holding
    thousands of paths (:meth:`py_alf.campaign.Campaign.status`). It differs
    only in *dispatch*: the cheap (mtime, size) short-circuit runs here, and
    whatever survives it goes to :func:`_get_process_pool` as one chunked
    ``map`` rather than one blocking ``submit``/``result`` per file. Reading
    one file at a time through the pool costs an IPC round trip per read and
    serialises on the executor's single work queue -- measured on a
    24k-chain campaign, that fan-out achieved no concurrency at all.

    ``on_progress(n)`` is called with how many paths a step settled, so a caller
    can drive a bar over a batch big enough to be worth watching.
    """
    keys = [(name, counting_obs) for name in filenames]
    counts: list[int | None] = [None] * len(filenames)
    pending: list[int] = []

    for i, key in enumerate(keys):
        if key in _bin_final:
            counts[i] = _bin_cache.get(key, 0)
            continue
        # Stat only when there is a previous reading for the stat to validate.
        # _bin_stat lives in this process, so a freshly started CLI holds none
        # and every such stat is a guaranteed miss -- on a networked filesystem
        # a stat is milliseconds, and a grid's worth of them cost more than they
        # could ever save. The chains that do have one (a second pass within a
        # run, e.g. reconcile's launch after its status) still short-circuit,
        # and the reads themselves take their own signature inside the pool.
        if not force and key in _bin_cache:
            # Both sides return None when absent, so an unrecorded signature and
            # a vanished file would otherwise compare equal and serve the cache
            # without ever looking at the disk.
            recorded = _bin_stat.get(key)
            if recorded is not None and recorded == _stat_sig(filenames[i]):
                counts[i] = _bin_cache[key]
                continue
        pending.append(i)

    # Everything settled by the caches above cost nothing, so report it in one
    # step: a bar that inched through them would misrepresent where the time is.
    if on_progress is not None and len(pending) < len(filenames):
        on_progress(len(filenames) - len(pending))

    if pending:
        reads = _get_process_pool().map(
            _read_bin_count,
            [filenames[i] for i in pending],
            repeat(counting_obs, len(pending)),
            chunksize=max(
                1, min(_BIN_BATCH_CHUNK, len(pending) // _process_pool_size())
            ),
        )
        for i, read in zip(pending, reads):
            counts[i] = _absorb_bin_read(keys[i], filenames[i], read, final)
            if on_progress is not None:
                on_progress(1)

    return [0 if c is None else c for c in counts]


def _find_job_log(
    jobid: str,
    root_dir: list[str] | None = None,
    submit_dir: str | Path | None = None,
) -> Path | None:
    if root_dir is None:
        root_dir = ["."]
    if jobid is None:
        logger.info("No job ID provided for logfile search.")
        return None

    # submitit names logs as {jobid}_{task_id}_log.out; task_id is always 0.
    if submit_dir is not None:
        submitit_log = Path(submit_dir) / f"{jobid}_0_log.out"
        if submitit_log.exists():
            return submitit_log

    # Legacy SLURM-template naming: job-{jobid}.log (underscores → dashes).
    all_matches = []
    for dir in root_dir:
        pattern = f"job-{jobid.replace('_', '-')}.log"
        all_matches.extend(Path(dir).rglob(pattern))
    if not all_matches:
        logger.error(f"Could not find logfile for job {jobid} in {root_dir}")
        return None
    if len(all_matches) != 1:
        logger.warning(f"Multiple logfiles found for job {jobid} in {root_dir}")
    return all_matches[0]


Simulation.bin_count = _bin_count
