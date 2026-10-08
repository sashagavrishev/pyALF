"""What one array task does on the compute node.

The decisions that depend on progress -- is this chain already finished? how
much time does it still need? -- are made here, when the task starts and the
bins on disk are current, rather than at submission time when they would all be
stale by the time the task ran.

That also makes a requeued attempt cheap: one whose chain has meanwhile reached
its target returns in seconds instead of burning an allocation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import submitit

from ..execute import exec_alf_binary
from .ledger import DEFAULT_COUNTING_OBS, segment_dir
from .policy import SegmentPolicy

# ALF's own mutex against two processes sharing a sim_dir, and the marker this
# worker writes next to it to record which task the mutex belongs to.
RUNNING_FILE = "RUNNING"
RUNNING_OWNER_FILE = "RUNNING.owner"


@dataclass(frozen=True)
class SegmentPlan:
    """Everything a task needs to size and run its own segment.

    Attached to the ``Simulation`` before submission and pickled with it, so the
    node needs no access to the ledger.
    """

    target_bins: int
    hours_per_bin: float  # a-priori estimate, superseded by measurement
    partition_rules: dict
    policy: SegmentPolicy
    # CPU_MAX the array was submitted with, and so the wall time SLURM actually
    # allocated. The node may ask for less, never more: a larger CPU_MAX would
    # let ALF run past the point where SLURM kills it, losing the graceful
    # truncation this whole mechanism depends on.
    cpu_max_ceiling: float = 0.0
    chain_id: str = ""
    counting_obs: str = DEFAULT_COUNTING_OBS


def _count_bins(sim, counting_obs: str = DEFAULT_COUNTING_OBS) -> int:
    """Bins currently in this chain's ``data.h5`` (0 when it does not exist)."""
    return sim.bin_count(counting_obs)


def _claim_running(sim_dir: Path, job_id: str) -> None:
    """Record which task is about to let ALF create ``RUNNING`` here."""
    (sim_dir / RUNNING_OWNER_FILE).write_text(job_id)


def _clear_own_running(sim_dir: Path, job_id: str) -> bool:
    """Remove a ``RUNNING`` this very task left behind, and only that.

    ALF hard-aborts when ``RUNNING`` already exists, so a task killed mid-bin
    would poison its own requeue. The file is a mutex against two ALF processes
    in one directory, so it is cleared only when the owner marker names this
    same job id -- which a requeued attempt reuses, making the leftover provably
    ours. A missing or foreign marker is left alone for ALF to abort on.
    """
    running = sim_dir / RUNNING_FILE
    if not running.exists():
        return False
    try:
        owner = (sim_dir / RUNNING_OWNER_FILE).read_text().strip()
    except OSError:
        return False
    if owner != job_id:
        return False
    running.unlink()
    return True


def measured_hours_per_bin(sim_dir: str | Path) -> float | None:
    """Seconds-per-bin from this chain's own history, in hours.

    Prefers the most recent segment that actually produced bins, so the estimate
    tracks the machine the chain is really running on rather than a cost model.
    Returns ``None`` until a segment has produced at least one bin.
    """
    best: tuple[str, float] | None = None
    for path in sorted(segment_dir(sim_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        gained = record.get("bins_after", 0) - record.get("bins_before", 0)
        elapsed = record.get("elapsed_s", 0)
        if gained > 0 and elapsed > 0:
            stamp = record.get("finished_at", "")
            if best is None or stamp >= best[0]:
                best = (stamp, elapsed / gained / 3600)
    return best[1] if best else None


def run_segment(sim) -> None:
    """Run one checkpoint-restart segment of ``sim``'s Markov chain."""
    plan: SegmentPlan = sim.segment_plan
    sim_dir = Path(sim.sim_dir)
    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        and f"{os.environ['SLURM_ARRAY_JOB_ID']}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
    ) or os.environ.get("SLURM_JOB_ID", f"local-{os.getpid()}")

    if _clear_own_running(sim_dir, job_id):
        print(
            f"[segment] {sim_dir.name}: cleared RUNNING left by this "
            f"job's interrupted attempt ({job_id})."
        )

    bins_before = _count_bins(sim, plan.counting_obs)
    if bins_before >= plan.target_bins:
        print(
            f"[segment] {sim_dir.name}: {bins_before}/{plan.target_bins} "
            "bins already present, nothing to do."
        )
        return

    remaining = plan.target_bins - bins_before
    hours_per_bin = measured_hours_per_bin(sim_dir) or plan.hours_per_bin
    cpu_max = plan.policy.cpu_max(remaining, hours_per_bin, plan.partition_rules)
    if plan.cpu_max_ceiling:
        cpu_max = min(cpu_max, plan.cpu_max_ceiling)

    # Both bounds, so ALF stops at whichever comes first: CPU_MAX keeps the job
    # inside its allocation, NBin stops it exactly at the target when the target
    # is reached first. ALF's bin loop counts from 1 for each run and appends, so
    # NBin = remaining lands on bins_before + remaining. Both are generic VAR_QMC
    # parameters and never enter directory_name, so varying them per segment
    # cannot move this chain's sim_dir.
    sim.sim_dict = {
        **sim.sim_dict,
        "CPU_MAX": float(cpu_max),
        "NBin": int(remaining),
    }

    print(
        f"[segment] {sim_dir.name}: {bins_before}/{plan.target_bins} bins, "
        f"{remaining} to go at {hours_per_bin * 60:.2f} min/bin "
        f"-> CPU_MAX={cpu_max:.3f} h, NBin={remaining}"
    )

    # Prep here rather than at submission: this renames confout_* -> confin_*,
    # and a requeued attempt must pick up the checkpoint the interrupted one
    # left behind, not the one that existed when the array was submitted.
    started = time.time()
    sim.run(only_prep=True)
    _claim_running(sim_dir, job_id)
    exec_alf_binary(sim)
    elapsed = time.time() - started

    bins_after = _count_bins(sim, plan.counting_obs)
    record = {
        "job_id": job_id,
        "chain_id": plan.chain_id,
        # What this segment's bins were actually computed with, not what HEAD
        # is now -- lets a later archival pass tell old-commit bins apart from
        # current ones without trusting anything but the segment itself.
        "alf_commit": getattr(sim.alf_src, "commit", lambda: None)(),
        "cpu_max": float(cpu_max),
        "hours_per_bin_used": hours_per_bin,
        "bins_before": bins_before,
        "bins_after": bins_after,
        "elapsed_s": round(elapsed, 1),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = segment_dir(sim_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Job id and timestamp, not the job id alone: SLURM reuses a job id when it
    # requeues, so a requeued attempt would otherwise overwrite the record of
    # the attempt it is replacing, losing that segment's timing measurement.
    stamp = record["finished_at"].replace(":", "").replace("-", "")
    (out_dir / f"{job_id}-{stamp}.json").write_text(json.dumps(record, indent=2))

    print(
        f"[segment] {sim_dir.name}: {bins_before} -> {bins_after} bins "
        f"in {elapsed / 3600:.2f} h"
    )


def _checkpoint_segment(sim) -> submitit.helpers.DelayedSubmission:
    """Hand submitit an identical call to requeue when this task is interrupted.

    It carries no state because there is none to carry: ALF flushes ``data.h5``
    and ``confout_0.h5`` every bin, so the checkpoint already exists on disk and
    the requeued call simply re-reads it and resumes.

    This is a safety net, not the primary mechanism. The signal arrives shortly
    before the wall limit with ALF mid-bin, whereas ``CPU_MAX`` normally has ALF
    stop cleanly at a bin boundary well before that. It covers preemption, bad
    time estimates and node contention.
    """
    return submitit.helpers.DelayedSubmission(run_segment, sim)


# submitit looks this attribute up on the submitted function itself, so it has
# to exist after a plain import on the node.
run_segment.checkpoint = _checkpoint_segment
