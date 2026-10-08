"""Launch, status and reconcile for a grid of chains; see :mod:`py_alf.campaign`."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .._io import map_io
from ..alf_source import ALF_source
from ..bins import read_bin_counts
from ..cluster_submission import ClusterSubmitter
from ..simulation import Simulation
from ..slurm import ACTIVE_STATES, is_timeout, job_states
from .chain import Chain
from .ledger import DEFAULT_COUNTING_OBS, Ledger
from .policy import SegmentPolicy
from .worker import SegmentPlan, measured_hours_per_bin, run_segment

# Progress hook for a long scan: ``(n_settled, phase)``.
ProgressFn = Callable[[int, str], None]

# Hours per bin for a chain with neither a measurement nor a cost model; it only
# sizes that chain's first segment.
DEFAULT_HOURS_PER_BIN = 1.0


@dataclass
class ChainStatus:
    """Plain data describing one chain's progress."""

    chain_id: str
    sim_dir: str
    # The launcher's grid coordinates, carried through unread.
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

    ``submitter`` is used as given, so the memory request, partition limits and
    executor are the caller's to decide; only the per-array job name and
    requeue budget are set here.
    """

    name: str
    chains: list[Chain]
    target_bins: int
    submitter: ClusterSubmitter
    ledger_path: Path
    policy: SegmentPolicy = field(default_factory=SegmentPolicy)
    job_name_prefix: str | None = None
    # The observable whose bins count as progress, and the hours-per-bin
    # estimate for a chain that has not measured itself yet.
    counting_obs: str = DEFAULT_COUNTING_OBS
    cost_model: Callable[[dict], float] | None = None
    # Provenance labels recorded in the ledger; nothing is resolved from them.
    experiment: str = ""
    env_name: str = ""
    # Each array's submitit files go to ``jobs_dir/<array_key>/<array id>/``;
    # None leaves them in the submitter's own ``submit_dir``.
    jobs_dir: Path | None = None

    # --- construction -------------------------------------------------------

    @classmethod
    def from_ledger(
        cls,
        ledger_path: str | Path,
        submitter: ClusterSubmitter,
        *,
        alf_src: ALF_source | None = None,
        machine: str = "GNU",
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
        simulations would run.
        """
        ledger = Ledger.load(ledger_path)
        alf_src = alf_src if alf_src is not None else ALF_source()
        chains = []
        for cid, record in ledger.chains.items():
            sim = Simulation(
                alf_src,
                record["ham_name"],
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
                    target_bins=record["target_bins"],
                    point=record["point"],
                    array_key=record["array_key"],
                )
            )
        return cls(
            name=ledger.name,
            chains=chains,
            target_bins=ledger.target_bins,
            submitter=submitter,
            ledger_path=Path(ledger_path),
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
        """Best hours-per-bin estimate for ``chain``.

        Measurement from the chain's own history wins; failing that the
        injected cost model, and only then a flat guess.
        """
        measured = measured_hours_per_bin(chain.sim_dir)
        if measured:
            return measured
        if self.cost_model is None:
            return DEFAULT_HOURS_PER_BIN
        return self.cost_model(chain.sim.sim_dict)

    # --- launching ----------------------------------------------------------

    def launch(
        self,
        segments: int | None = None,
        dry_run: bool = False,
        verbose: bool = True,
        bins: dict[str, int] | None = None,
    ) -> Ledger:
        """Queue the campaign: one SLURM array per group of chains.

        ``segments`` is the number of *attempts* a task may make: the first run
        plus the requeues submitit is allowed to perform. Chains that are
        already complete, or that still have an active job, are left alone -- so
        relaunching a campaign tops up only what needs it. ``bins`` holds counts
        already resolved by :meth:`status`, by chain id; other chains are read.
        """
        if not self.chains:
            raise SystemExit("Campaign has no chains; check the parameter grid.")
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
        active = self._active_chain_ids(ledger)

        rules = self.submitter.partition_rules or {}
        for key, chains in self.groups().items():
            idle = [c for c in chains if c.chain_id not in active]
            runnable = self._runnable(idle, bins or {})
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

    def _active_chain_ids(self, ledger: Ledger) -> set[str]:
        """Chains with a segment SLURM still holds, from one bulk query."""
        if self.submitter.executor != "slurm":
            return set()
        records = {c.chain_id: ledger.chains[c.chain_id] for c in self.chains}
        jobs = [
            s["job_id"]
            for record in records.values()
            for s in record.get("segments", [])
            if s.get("job_id")
        ]
        if not jobs:
            return set()
        states = job_states(jobs)
        return {
            cid for cid, record in records.items() if _has_active_job(record, states)
        }

    def _runnable(
        self, chains: list[Chain], known: dict[str, int]
    ) -> list[tuple[Chain, int]]:
        """Chains short of the target, each with the bin count that decided it.

        The count is returned rather than recomputed by the caller: sizing the
        array's budget needs the same number, and re-reading the whole grid to
        get it would double the launch's filesystem cost.
        """
        unread = [c for c in chains if c.chain_id not in known]
        read = read_bin_counts(
            [os.path.join(c.sim_dir, "data.h5") for c in unread], self.counting_obs
        )
        counts = {**known, **dict(zip((c.chain_id for c in unread), read, strict=True))}
        bins = [counts[c.chain_id] for c in chains]
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
        # submitit's requeue budget counts timed-out requeues only; a
        # preemption requeue does not use it up.
        job_properties: dict[str, Any] = {}
        job_name = self._job_name(key)
        if job_name is not None:
            job_properties["name"] = job_name
        chains = [c for c, _, _ in plans]

        for chain, _, hpb in plans:
            chain.sim.sim_dict = {**chain.sim.sim_dict, "CPU_MAX": float(ceiling)}
            chain.sim.segment_plan = SegmentPlan(
                target_bins=self.target_bins,
                hours_per_bin=hpb,
                partition_rules=rules,
                policy=self.policy,
                chain_id=chain.chain_id,
                cpu_max_ceiling=float(ceiling),
                counting_obs=self.counting_obs,
            )

        jobs = self.submitter.submit(
            sims=[c.sim for c in chains],
            job_properties=job_properties,
            max_requeues=max(1, attempts),
            submit_dir=None if self.jobs_dir is None else self.jobs_dir / key / "%A",
            runner=run_segment,
            prep=False,
            # launch has already left out every chain with an active job, so a
            # leftover RUNNING is debris from an interrupted attempt.
            skip_active=False,
            stale_running="remove",
        )
        if not jobs:
            return

        # Pair each chain with its job through the jobid.txt ``submit`` wrote
        # into the chain's own directory.
        submitted = {job.job_id for job in jobs}
        for chain in chains:
            jobid_file = Path(chain.sim_dir) / "jobid.txt"
            job_id = jobid_file.read_text().strip() if jobid_file.exists() else None
            if job_id not in submitted:
                continue
            ledger.add_segment(
                chain.chain_id,
                {
                    "job_id": job_id,
                    "cpu_max_planned": float(ceiling),
                },
            )
        if verbose:
            print(f"    array {jobs[0].job_id.split('_')[0]}")

    def job_folder(self, array_key: str, job_id: str) -> Path:
        """Folder holding submitit's files for *job_id*, from array *array_key*."""
        if self.jobs_dir is None:
            return Path(self.submitter.submit_dir)
        return self.jobs_dir / array_key / job_id.split("_")[0]

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
        """Every chain's bin count, from the ledger where it is settled, else from disk.

        A chain the ledger records at its target is finished, and an idle chain
        holds the ``bins_after`` its last segment reported, since nothing has
        written to it since. Live chains, and idle ones whose worker died before
        reporting, are read as one batch. ``deep`` reads every chain, for when
        the data may have changed underneath the ledger.
        """
        known: dict[str, int] = {}
        needs_read: list[str] = []

        for chain_id, record in ledger.chains.items():
            target = record.get("target_bins", ledger.target_bins)
            cached = record.get("bins")
            if not deep and isinstance(cached, int) and cached >= target:
                known[chain_id] = cached
                continue
            if not deep and not _has_active_job(record, states):
                reported = _bins_reported_by_worker(record)
                if reported is not None:
                    known[chain_id] = reported
                    continue
            needs_read.append(chain_id)

        # Settled from the ledger, so reported in one step.
        if on_progress is not None:
            on_progress(len(known), "cached")

        if needs_read:
            counts = read_bin_counts(
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

        return known

    def status(
        self,
        ledger: Ledger | None = None,
        deep: bool = False,
        persist: bool = True,
        on_progress: ProgressFn | None = None,
    ) -> list[ChainStatus]:
        """Per-chain progress and a verdict, judged by bins rather than job state.

        ``deep`` reads every ``data.h5`` instead of trusting the ledger's
        counts. ``persist`` writes the counts back to the ledger.
        ``on_progress(n, phase)`` reports each chain exactly once, as it is
        settled.
        """
        ledger = ledger or Ledger.load(self.ledger_path)
        if on_progress is not None:
            # Names the scan of segment records, which settles no chain itself.
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
        states = job_states(all_jobs) if all_jobs else {}
        bins_by_id = self._resolve_bins(
            ledger, states, deep=deep, on_progress=on_progress
        )
        moved = ledger.record_bins(bins_by_id)
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
                if last_state in {"FAILED", "COMPLETED"}:
                    folder = self.job_folder(record.get("array_key", ""), jid)
                    timed_out = is_timeout(jid, folder, status=last_state)
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
                point=record["point"],
                bins=bins,
                target_bins=ledger.target_bins,
                segments=len(segments),
                active_job=active,
                last_state=last_state,
                timed_out=timed_out,
                verdict=verdict,
            )

        # What is left per chain is at most one submitit log read, an
        # independent filesystem probe, so they share the I/O pool.
        return map_io(_chain_status, list(ledger.chains.items()))

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
        budget. ``suspect`` chains, short and holding no bins at all, are only
        resubmitted with ``force``: they are more likely crashing than slow.
        """
        ledger = Ledger.load(self.ledger_path)
        # status() absorbs the worker records and saves the ledger itself.
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
        subset.launch(
            segments=segments,
            verbose=verbose,
            bins={s.chain_id: s.bins for s in statuses},
        )
        return statuses


def _has_active_job(record: dict[str, Any], states: dict[str, dict]) -> bool:
    """True if any of this chain's segments is still queued or running."""
    return any(
        (states.get(s.get("job_id")) or {}).get("status") in ACTIVE_STATES
        for s in record.get("segments", [])
    )


def _bins_reported_by_worker(record: dict[str, Any]) -> int | None:
    """Highest ``bins_after`` this chain's workers recorded, or None if none did.

    For an idle chain this is what stands on disk. The maximum, because a
    requeued attempt that crashed early can record fewer bins than an earlier one.
    """
    reported = [
        s["bins_after"]
        for s in record.get("segments", [])
        if isinstance(s.get("bins_after"), int)
    ]
    return max(reported) if reported else None
