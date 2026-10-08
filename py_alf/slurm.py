"""SLURM job state, submitit timeout detection and job logs."""

from __future__ import annotations

import contextlib
import getpass
import logging
import subprocess
from pathlib import Path

from submitit.core.utils import JobPaths

logger = logging.getLogger(__name__)


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


def _normalise_state(raw: str) -> str:
    """Return a canonical SLURM state from a raw sacct or squeue value.

    sacct truncates state strings to its column width and appends ``+``, e.g.
    ``CANCELLED+`` (cancelled by uid) or ``OUT_OF_ME+`` (out of memory).
    This strips the truncation marker and restores full names.
    """
    state = raw.split()[0].rstrip("+")
    if state == "OUT_OF_ME":
        return "OUT_OF_MEMORY"
    return state


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


def _job_states_sacct(
    jobids: list[str],
) -> dict[str, dict[str, str | None]]:
    """:func:`job_states` from ``sacct`` alone, for jobs no longer queued."""
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
                state = _normalise_state(parts[1])
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


def job_states(jobids: list[str]) -> dict[str, dict[str, str | None]]:
    """``jobid -> {'status', 'runtime', 'nodelist'}`` for many jobs at once.

    Asks ``squeue`` first and ``sacct`` for jobs that have left the queue.
    ``nodelist`` is None unless the job is running.
    """
    if not jobids:
        return {}

    # A job in a terminal state never changes again, so it is served from cache
    # rather than paid for with another sacct call.
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
                    "status": _normalise_state(raw_state),
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
            sacct_statuses = _job_states_sacct(need_sacct)
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
        sacct_statuses = _job_states_sacct(jobids)
        for jid in jobids:
            status_map[jid] = sacct_statuses.get(
                jid, {"status": "UNKNOWN", "runtime": None, "nodelist": None}
            )

    for jid, entry in status_map.items():
        if entry.get("status") in TERMINAL_STATES:
            _terminal_status_cache[jid] = dict(entry)

    status_map.update(cached)
    return status_map


_submitit_timeout_cache: dict[str, tuple[bool, bool]] = {}

# Terminal SLURM states are immutable, so they are cached per job ID and never
# re-queried.  Keyed by the full task ID ("12345" or "12345_7").
_terminal_status_cache: dict[str, dict[str, str | None]] = {}


def is_timeout(jobid: str, folder: str | Path, status: str = "FAILED") -> bool:
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
        log_path = job_log(jobid, folder)
        # Suppressing the read covers the missing file, so no separate exists().
        with contextlib.suppress(OSError):
            text = log_path.read_text(errors="replace")
        if text is None:
            # On a networked filesystem the log can lag the job's state, so
            # answer without caching and look again next time.
            return False
        cached = (
            "this job is timed-out" in text,
            "Bypassing signal SIGTERM" in text,
        )
        _submitit_timeout_cache[jobid] = cached
    timed_out, sigterm_bypassed = cached
    return timed_out or (status == "COMPLETED" and sigterm_bypassed)


# Job states meaning "this chain is still being worked on, leave it alone".
ACTIVE_STATES = frozenset({"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING"})

TERMINAL_STATES: frozenset[str] = frozenset(
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


def job_log(jobid: str, folder: str | Path, stream: str = "out") -> Path:
    """Path of submitit's ``stdout`` (or ``err``) log for *jobid* in *folder*.

    *folder* is the ``submit_dir`` the job was submitted with; a ``%A``/``%j``
    template is resolved for this job.
    """
    paths = JobPaths(folder, job_id=jobid)
    return paths.stdout if stream == "out" else paths.stderr


def cancel(job_ids: list[str]) -> None:
    """``scancel`` the given jobs or whole arrays."""
    subprocess.run(["scancel", *job_ids], check=True)


def queued_arrays() -> set[str]:
    """Array ids (job ids for single jobs) of this user's jobs SLURM still holds.

    Raises if ``squeue`` fails, so an unanswered query is never read as an
    empty queue.
    """
    result = subprocess.run(
        ["squeue", "-h", "-u", getpass.getuser(), "-o", "%F"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}
