"""A campaign: a grid of independent Markov chains driven to a bin target.

The caller declares *what* to compute -- a set of chains and how many bins each
needs -- and this module works out how to get there within a queue that will not
run a job for longer than two days.

One SLURM array is submitted per group of chains, and three layers then carry
them to the target:

1. ``CPU_MAX`` (see :mod:`py_alf.campaign.policy`) -- ALF stops itself cleanly
   at a bin boundary inside the partition's wall-time limit, having flushed both
   ``data.h5`` and its checkpoint;
2. submitit's checkpoint/requeue (see :func:`py_alf.campaign.worker.run_segment`)
   -- an automatic retry when the task is preempted or the time estimate was
   wrong, bounded by ``slurm_max_num_timeout``;
3. :func:`Campaign.reconcile` -- repairs what neither covers: cancellation, node
   failure, an exhausted retry budget.

The third exists because the first two can fail in ways SLURM will not report
honestly -- under submitit a wall-clock stop is recorded as ``FAILED`` or
``CANCELLED``, never ``TIMEOUT``, and is indistinguishable by exit code from a
genuine crash. So progress is judged by *bins on disk*, never by job state.

Nothing here knows about any particular model or machine: the caller supplies
the chains, a configured :class:`~py_alf.cluster_submission.ClusterSubmitter`,
where the ledger lives, the partition limits, and -- optionally -- a cost model
saying what a bin of *its* Hamiltonian costs.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..alf_source import ALF_source
from ..cluster_submission import (
    ClusterSubmitter,
    _bin_cache,
    _bin_counts,
    _get_slurm_status_bulk,
    _is_submitit_timeout,
    _map_io,
)
from ..simulation import Simulation
from .chain import Chain
from .ledger import DEFAULT_COUNTING_OBS, Ledger, chain_point
from .policy import SegmentPolicy
from .worker import SegmentPlan, measured_hours_per_bin, run_segment

# Job states meaning "this chain is still being worked on, leave it alone".
ACTIVE_STATES = frozenset({"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "COMPLETING"})

# Reports progress out of a long scan: ``(n_settled, phase)``. A campaign can
# hold tens of thousands of chains, so a caller driving one from a terminal
# needs to see movement -- but which chains are cheap is this layer's business,
# not the caller's, hence a hook rather than an exposed work breakdown.
ProgressFn = Callable[[int, str], None]

# Hours per bin assumed for a chain that has neither measured itself yet nor
# been given a ``cost_model``. Deliberately model-free -- only the caller knows
# what its Hamiltonian costs -- and only ever sizes the first segment, which the
# worker's own measurement then supersedes.
DEFAULT_HOURS_PER_BIN = 1.0


@dataclass
class ChainStatus:
    """Plain data describing one chain's progress."""

    chain_id: str
    sim_dir: str
    # Grid coordinates as the launcher recorded them (a disorder seed, a sweep
    # value); the core reads none of them, it only carries them back out.
    point: dict[str, Any]
    bins: int
    target_bins: int
    segments: int
    active_job: str | None
    last_state: str | None
    timed_out: bool
    verdict: str  # done | active | resumable | suspect | unstarted

    @property
    def complete(self) -> bool:
        return self.bins >= self.target_bins


@dataclass
class Campaign:
    """A named grid of chains, its target, and the policy that paces it.

    ``submitter`` is used as given, so the memory request, wckey and executor
    are the caller's to decide; only the per-array job name and requeue budget
    are overridden here. ``partition_rules`` are the same limits that submitter
    was built with, which the policy needs in raw form.
    """

    name: str
    chains: list[Chain]
    target_bins: int
    submitter: ClusterSubmitter
    ledger_path: Path
    partition_rules: dict[str, Any] = field(default_factory=dict)
    policy: SegmentPolicy = field(default_factory=SegmentPolicy)
    job_name_prefix: str | None = None
    hours_per_bin: dict[str, float] = field(default_factory=dict)
    # Observable whose bin count measures progress, and the a-priori
    # hours-per-bin estimate for a chain that has not measured itself yet
    # (``None`` falls back to :data:`DEFAULT_HOURS_PER_BIN`).
    counting_obs: str = DEFAULT_COUNTING_OBS
    cost_model: Callable[[dict], float] | None = None
    # Labels recorded in the ledger for provenance; nothing is resolved from
    # them, since ``ledger_path`` already says where this campaign lives.
    experiment: str = ""
    env_name: str = ""

    # --- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        name: str,
        chains: list[Chain],
        target_bins: int,
        **kwargs,
    ) -> Campaign:
        if not chains:
            raise SystemExit("Campaign has no chains; check the parameter grid.")
        return cls(
            name=name,
            chains=list(chains),
            target_bins=target_bins,
            **kwargs,
        )

    @classmethod
    def from_ledger(
        cls,
        ledger_path: str | Path,
        submitter: ClusterSubmitter,
        partition_rules: dict[str, Any] | None = None,
        *,
        alf_src: ALF_source | None = None,
        machine: str = "GNU",
        default_ham_name: str | None = None,
        cost_model: Callable[[dict], float] | None = None,
        **kwargs,
    ) -> Campaign:
        """Reopen a campaign from its ledger alone.

        The ledger stores each chain's parameters, seeds and ``sim_dir``, which
        is enough to rebuild the ``Simulation`` objects -- so ``status`` and
        ``reconcile`` work from a cron job or a fresh shell without replaying the
        launcher's command line. ``sim_root`` is recovered from the recorded
        ``sim_dir`` so the rebuilt chain resolves to the very same directory.

        ``machine`` and ``alf_src`` say where to find the binary those rebuilt
        simulations would run; ``default_ham_name`` names the model for ledgers
        written before the chains recorded it themselves.
        """
        ledger = Ledger.load(ledger_path)
        alf_src = alf_src if alf_src is not None else ALF_source()
        chains = []
        for cid, record in ledger.chains.items():
            ham_name = record.get("ham_name") or default_ham_name
            if not ham_name:
                raise SystemExit(
                    f"{ledger_path}: chain {cid} does not record its Hamiltonian; "
                    "pass default_ham_name to name the model it was launched for."
                )
            sim = Simulation(
                alf_src,
                ham_name,
                dict(record["params"]),
                mc_seed=record["mc_seed"],
                machine=machine,
                sim_root=str(Path(record["sim_dir"]).parent),
                hdf5=True,
            )
            chains.append(
                Chain(
                    chain_id=cid,
                    sim=sim,
                    mc_seed=record["mc_seed"],
                    target_bins=record.get("target_bins", ledger.target_bins),
                    point=chain_point(record),
                    init_config=record.get("init_config"),
                    array_key=record.get("array_key", ""),
                )
            )
        return cls(
            name=ledger.name,
            chains=chains,
            target_bins=ledger.target_bins,
            submitter=submitter,
            ledger_path=Path(ledger_path),
            partition_rules=partition_rules or {},
            policy=SegmentPolicy(**ledger.data.get("policy", {})),
            counting_obs=ledger.counting_obs,
            cost_model=cost_model,
            experiment=ledger.data.get("experiment", ""),
            env_name=ledger.data.get("env", ""),
            **kwargs,
        )

    def groups(self) -> dict[str, list[Chain]]:
        """Chains bundled into SLURM arrays.

        One array per ``array_key`` -- by convention one parameter point, whose
        chains share a cost and therefore a wall-time request, since submitit
        applies a single set of SLURM parameters to a whole array.
        """
        out: dict[str, list[Chain]] = {}
        for chain in self.chains:
            out.setdefault(chain.array_key, []).append(chain)
        return out

    # --- estimation ---------------------------------------------------------

    def hours_per_bin_for(self, chain: Chain) -> float:
        """Best available seconds-per-bin estimate for ``chain``, in hours.

        Measurement from the chain's own history wins; failing that an explicit
        per-array override; failing that the injected cost model, and only then
        a flat guess.
        """
        measured = measured_hours_per_bin(chain.sim_dir)
        if measured:
            return measured
        if chain.array_key in self.hours_per_bin:
            return self.hours_per_bin[chain.array_key]
        if self.cost_model is None:
            return DEFAULT_HOURS_PER_BIN
        return self.cost_model(chain.sim.sim_dict)

    def bins_on_disk(self, chain: Chain) -> int:
        """Bins currently in one chain's ``data.h5``."""
        return self.bins_on_disk_many([chain])[0]

    def bins_on_disk_many(self, chains: list[Chain]) -> list[int]:
        """Bins in each chain's ``data.h5``, read as one batch.

        Batched rather than one call per chain: h5py serializes every call
        within one process (see ``_get_process_pool``'s docstring), so the reads
        have to be spread over separate OS processes to overlap at all -- and
        handing that pool one file per task makes each read cost an IPC round
        trip instead, which on a campaign-sized grid is the whole cost. See
        :func:`~py_alf.cluster_submission._bin_counts`.
        """
        return _bin_counts(
            [os.path.join(c.sim_dir, "data.h5") for c in chains], self.counting_obs
        )

    # --- launching ----------------------------------------------------------

    def launch(
        self,
        segments: int | None = None,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> Ledger:
        """Queue the campaign: one SLURM array per group of chains.

        ``segments`` is the number of *attempts* a task may make: the first run
        plus the requeues submitit is allowed to perform. Chains that are
        already complete, or that still have an active job, are left alone -- so
        relaunching a campaign tops up only what needs it.
        """
        ledger = Ledger.load_or_new(
            self.ledger_path,
            name=self.name,
            target_bins=self.target_bins,
            policy=vars(self.policy),
            counting_obs=self.counting_obs,
            experiment=self.experiment,
            env_name=self.env_name,
        )
        ledger.upsert_chains(self.chains)

        rules = self.partition_rules
        for key, chains in self.groups().items():
            runnable = self._runnable(chains)
            if not runnable:
                if verbose:
                    print(f"[{key or self.name}] nothing to submit.")
                continue

            plans = [
                (chain, bins, self.hours_per_bin_for(chain)) for chain, bins in runnable
            ]
            # One array shares one wall-time request, so the array asks for the
            # neediest chain's budget; every worker then trims its own CPU_MAX
            # down from that ceiling.
            ceiling = max(
                self.policy.cpu_max(self.target_bins - bins, hpb, rules)
                for _, bins, hpb in plans
            )
            attempts = segments or max(
                self.policy.segments_needed(self.target_bins - bins, hpb, rules)
                for _, bins, hpb in plans
            )

            if verbose or dry_run:
                print(
                    f"[{key or self.name}] {len(runnable)} chain(s), "
                    f"{attempts} attempt(s), CPU_MAX ceiling {ceiling:.2f} h"
                )
            if dry_run:
                for chain, bins, hpb in plans:
                    print(
                        f"    {chain.chain_id} {Path(chain.sim_dir).name} "
                        f"bins={bins}/{self.target_bins} hpb={hpb * 60:.2f} min"
                    )
                continue

            self._submit_array(key, plans, ceiling, attempts, rules, ledger, verbose)

        if not dry_run:
            ledger.save()
            if verbose:
                print(f"Ledger: {ledger.path}")
        return ledger

    def _runnable(self, chains: list[Chain]) -> list[tuple[Chain, int]]:
        """Chains short of the target, each with the bin count that decided it.

        Chains still held by SLURM are dropped by ``ClusterSubmitter.submit``
        itself, which reads each ``jobid.txt`` and skips PENDING/RUNNING jobs.

        The count is returned rather than recomputed by the caller: sizing the
        array's budget needs the same number, and re-reading the whole grid to
        get it would double the launch's filesystem cost.
        """
        bins = self.bins_on_disk_many(chains)
        return [
            (c, b) for c, b in zip(chains, bins, strict=True) if b < self.target_bins
        ]

    def _submit_array(
        self,
        key: str,
        plans: list[tuple[Chain, int, float]],
        ceiling: float,
        attempts: int,
        rules: dict,
        ledger: Ledger,
        verbose: bool,
    ) -> None:
        # Per-array overrides of the shared submitter: submitit's requeue
        # countdown starts at ``attempts`` and is decremented once per
        # *timed-out* requeue (a preemption requeue does not decrement), and an
        # explicit name keeps one stable, unsuffixed job name per array.
        job_properties: dict[str, Any] = {"slurm_max_num_timeout": max(1, attempts)}
        job_name = self._job_name(key)
        if job_name is not None:
            job_properties["name"] = job_name
        chains = [c for c, _, _ in plans]

        for chain, _, hpb in plans:
            chain.sim.sim_dict = {**chain.sim.sim_dict, "CPU_MAX": float(ceiling)}
            chain.sim.segment_plan = SegmentPlan(
                index=0,
                target_bins=self.target_bins,
                hours_per_bin=hpb,
                partition_rules=rules,
                policy=self.policy,
                init_config=chain.init_config,
                chain_id=chain.chain_id,
                cpu_max_ceiling=float(ceiling),
                counting_obs=self.counting_obs,
            )

        jobs = self.submitter.submit(
            sims=[c.sim for c in chains],
            job_properties=job_properties,
            runner=run_segment,
            prep=False,
            # A campaign runs unattended, and ``submit`` has just confirmed via
            # sacct that no job holds these directories, so a leftover RUNNING
            # is debris from an interrupted attempt rather than a live process.
            stale_running="remove",
        )
        if not jobs:
            return

        # ``submit`` drops chains whose job is still active, so the returned
        # list need not line up with ``chains``. It writes each submitted job's
        # id into that chain's own jobid.txt, which is the reliable pairing.
        submitted = {job.job_id for job in jobs}
        for chain in chains:
            jobid_file = Path(chain.sim_dir) / "jobid.txt"
            job_id = jobid_file.read_text().strip() if jobid_file.exists() else None
            if job_id not in submitted:
                continue
            ledger.add_segment(
                chain.chain_id,
                {
                    "index": 0,
                    "job_id": job_id,
                    "cpu_max_planned": float(ceiling),
                },
            )
        if verbose:
            print(f"    array {jobs[0].job_id.split('_')[0]}")

    def _job_name(self, key: str) -> str | None:
        if self.job_name_prefix is None:
            return None
        return f"{self.job_name_prefix}_{key}" if key else self.job_name_prefix

    # --- inspection ---------------------------------------------------------

    def _resolve_bins(
        self,
        ledger: Ledger,
        states: dict[str, dict[str, str | None]],
        deep: bool = False,
        on_progress: ProgressFn | None = None,
    ) -> dict[str, int]:
        """Every chain's bin count, opening as few ``data.h5`` files as possible.

        Reading the whole grid is what makes a status check expensive, and most
        of those reads answer a question that is already settled. Three tiers,
        cheapest first:

        1. a chain the ledger records at its target is finished and is trusted
           outright -- bins only ever increase, and the worker stops on the
           target rather than past it;
        2. a chain with no running job whose cached count was itself taken while
           it was idle, at the segment list it still has, keeps that count --
           nothing writes to an idle chain's file, and no job has started since
           (:meth:`~py_alf.campaign.ledger.Ledger.bins_still_stand`). This is
           the tier that carries a mid-campaign grid, where almost nothing has
           reached its target yet;
        3. a chain with no running job is however many bins its last segment
           reported writing (:func:`py_alf.campaign.worker.run_segment` records
           ``bins_after``); nothing has touched the file since it ended;
        4. anything else -- a live job, or a chain whose worker record is
           missing because it died before writing one -- is read from disk, as
           one batch.

        ``deep`` skips the first two tiers, for when the data is suspected to
        have changed underneath the ledger (files restored, a chain re-run by
        hand, a count written by an older version).
        """
        by_id = {c.chain_id: c for c in self.chains}
        known: dict[str, int] = {}
        needs_read: list[str] = []

        for chain_id, record in ledger.chains.items():
            target = record.get("target_bins", ledger.target_bins)
            cached = record.get("bins")
            if not deep and isinstance(cached, int) and cached >= target:
                known[chain_id] = cached
                continue
            if not deep and not _has_active_job(record, states):
                if ledger.bins_still_stand(record) and isinstance(cached, int):
                    known[chain_id] = cached
                    continue
                reported = _bins_reported_by_worker(record)
                if reported is not None:
                    known[chain_id] = reported
                    continue
            needs_read.append(chain_id)

        # The tiers above are pure dict lookups, so credit them in one step
        # rather than pretending they took measurable time.
        if on_progress is not None:
            on_progress(len(known), "cached")

        if needs_read:
            counts = _bin_counts(
                [
                    str(Path(ledger.chains[cid]["sim_dir"]) / "data.h5")
                    for cid in needs_read
                ],
                self.counting_obs,
                on_progress=(
                    None if on_progress is None else lambda n: on_progress(n, "reading")
                ),
            )
            known.update(zip(needs_read, counts, strict=True))
            # A chain whose Simulation was rebuilt shares Simulation.bin_count's
            # cache, so keep that consistent with what was just read rather than
            # letting a later call re-open the same file.
            for chain_id in needs_read:
                chain = by_id.get(chain_id)
                if chain is not None:
                    _bin_cache[
                        (os.path.join(chain.sim_dir, "data.h5"), self.counting_obs)
                    ] = known[chain_id]

        return known

    def status(
        self,
        ledger: Ledger | None = None,
        deep: bool = False,
        persist: bool = True,
        on_progress: ProgressFn | None = None,
    ) -> list[ChainStatus]:
        """Per-chain progress and a verdict, judged by bins rather than job state.

        ``deep`` re-reads every chain's ``data.h5`` instead of trusting the
        ledger's cached counts (see :meth:`_resolve_bins`). ``persist`` writes
        the counts back, which is what makes the next check cheap; pass False
        for a caller that must not touch the ledger.

        ``on_progress(n, phase)`` is called as chains are resolved: ``n`` is how
        many were settled by this step and ``phase`` names what is being done,
        so a caller can drive a progress bar without knowing the tiers. Every
        chain is reported exactly once, so the counts sum to the grid size.
        Rendering is the caller's business -- this layer has no opinion about
        terminals, and none of the reporting happens unless a hook is passed.
        """
        ledger = ledger or Ledger.load(self.ledger_path)
        if on_progress is not None:
            # No count: the scan is a fixed pass over the unfinished chains, and
            # naming it is what keeps the bar from looking hung during it.
            # Phase names stay terse -- they are a bar label, and a long one
            # leaves tqdm no width to draw the bar itself in.
            on_progress(0, "scanning")
        absorbed = ledger.absorb_segment_records(skip_finished=not deep)

        all_jobs = [
            s["job_id"]
            for record in ledger.chains.values()
            for s in record.get("segments", [])
            if s.get("job_id")
        ]
        if on_progress is not None and all_jobs:
            on_progress(0, "slurm")
        states = _get_slurm_status_bulk(all_jobs) if all_jobs else {}
        submit_dir = self.submitter.submit_dir
        bins_by_id = self._resolve_bins(
            ledger, states, deep=deep, on_progress=on_progress
        )
        # Only a count taken while nothing was writing may be reused next time,
        # so the caching side has to know which chains those were.
        settled = {
            chain_id
            for chain_id, record in ledger.chains.items()
            if not _has_active_job(record, states)
        }
        moved = ledger.record_bins(bins_by_id, settled=settled)
        if persist and (moved or absorbed):
            ledger.save()

        def _chain_status(item: tuple[str, dict]) -> ChainStatus:
            chain_id, record = item
            bins = bins_by_id.get(chain_id, 0)
            segments = record.get("segments", [])
            active = next(
                (
                    s["job_id"]
                    for s in segments
                    if (states.get(s.get("job_id")) or {}).get("status")
                    in ACTIVE_STATES
                ),
                None,
            )
            last_state = None
            timed_out = False
            if segments and segments[-1].get("job_id"):
                jid = segments[-1]["job_id"]
                last_state = (states.get(jid) or {}).get("status")
                if last_state in {"FAILED", "COMPLETED"} and submit_dir is not None:
                    timed_out = _is_submitit_timeout(jid, submit_dir, status=last_state)
                    if timed_out:
                        last_state = "TIMEOUT"

            if bins >= ledger.target_bins:
                verdict = "done"
            elif active:
                verdict = "active"
            elif not segments:
                verdict = "unstarted"
            elif bins > 0:
                verdict = "resumable"
            else:
                verdict = "suspect"

            return ChainStatus(
                chain_id=chain_id,
                sim_dir=record["sim_dir"],
                point=chain_point(record),
                bins=bins,
                target_bins=ledger.target_bins,
                segments=len(segments),
                active_job=active,
                last_state=last_state,
                timed_out=timed_out,
                verdict=verdict,
            )

        # The bin counts are already resolved; what is left per chain is a
        # submitit log read, and only for one whose last segment just ended.
        # Those are independent filesystem probes, so they still go through the
        # shared I/O pool -- on a networked filesystem it is the per-probe
        # latency, not CPU, that a campaign of thousands of chains pays for.
        return _map_io(_chain_status, list(ledger.chains.items()))

    # --- repair -------------------------------------------------------------

    def reconcile(
        self,
        submit: bool = True,
        force: bool = False,
        segments: int | None = None,
        verbose: bool = True,
    ) -> list[ChainStatus]:
        """Resubmit chains that neither ``CPU_MAX`` nor a requeue carried home.

        Covers a chain that was cancelled, lost its node, or ran out of requeue
        budget. ``suspect`` chains -- short *and* holding no bins at
        all -- are reported but not resubmitted without ``force``: a chain that
        never produced a bin is far more likely to be crashing than timing out,
        and blindly requeueing it would loop.
        """
        ledger = Ledger.load(self.ledger_path)
        # status() absorbs the worker records and saves the ledger itself, so
        # doing either here would only scan every chain's segment directory a
        # second time for what the first pass already folded in.
        statuses = self.status(ledger)

        wanted = {"resumable", "unstarted"} | ({"suspect"} if force else set())
        needy = {s.chain_id for s in statuses if s.verdict in wanted}
        suspects = [s for s in statuses if s.verdict == "suspect"]

        if verbose:
            counts: dict[str, int] = {}
            for s in statuses:
                counts[s.verdict] = counts.get(s.verdict, 0) + 1
            print("  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
            for s in suspects:
                print(
                    f"  suspect {s.chain_id} {Path(s.sim_dir).name}: 0 bins after "
                    f"{s.segments} segment(s), last state {s.last_state}"
                    + ("" if force else " -- not resubmitted (use --force)")
                )

        if not submit or not needy:
            return statuses

        subset = replace(self, chains=[c for c in self.chains if c.chain_id in needy])
        subset.launch(segments=segments, verbose=verbose)
        return statuses

    # --- chaining -----------------------------------------------------------

    def then(
        self,
        command: list[str],
        depends: str = "afterany",
        ledger: Ledger | None = None,
        dry_run: bool = False,
    ) -> str | None:
        """Queue ``command`` (an ``sbatch`` argument list) after the campaign.

        Gated on every chain's most recent segment, so an analysis stage can be
        submitted at the same time as the simulations it consumes. ``afterany``,
        not ``afterok``: a chain that stopped on the wall clock has still
        produced usable bins.
        """
        ledger = ledger or Ledger.load(self.ledger_path)
        job_ids = sorted({j.split("_")[0] for j in ledger.last_segment_job_ids()})
        if not job_ids:
            raise SystemExit("Campaign has no submitted segments to depend on.")

        cmd = [
            "sbatch",
            "--parsable",
            f"--dependency={depends}:{':'.join(job_ids)}",
            *command,
        ]
        if dry_run:
            print(" ".join(cmd))
            return None

        job_id = subprocess.run(
            cmd, check=True, capture_output=True, text=True
        ).stdout.strip()
        ledger.add_followup(
            {"job_id": job_id, "command": command, "dependency": depends}
        )
        ledger.save()
        return job_id


def _has_active_job(record: dict[str, Any], states: dict[str, dict]) -> bool:
    """True if any of this chain's segments is still queued or running."""
    return any(
        (states.get(s.get("job_id")) or {}).get("status") in ACTIVE_STATES
        for s in record.get("segments", [])
    )


def _bins_reported_by_worker(record: dict[str, Any]) -> int | None:
    """Highest ``bins_after`` this chain's workers recorded, or None if none did.

    Only meaningful for a chain with nothing running: the worker writes this
    after ALF has flushed, so for an idle chain it *is* what stands on disk, and
    reading the file back would just confirm it. The maximum rather than the
    last: segments are held in submission order, and a requeued attempt that
    crashed early can record fewer bins than one that already succeeded.
    """
    reported = [
        s["bins_after"]
        for s in record.get("segments", [])
        if isinstance(s.get("bins_after"), int)
    ]
    return max(reported) if reported else None
