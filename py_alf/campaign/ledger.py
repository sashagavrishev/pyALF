"""Durable record of a campaign: which chain is which, and what has run.

The ledger is the campaign's index. It maps every ``chain_id`` to its
Monte-Carlo seed, ``sim_dir``, grid coordinates (``point``) and job history,
which is what lets a later process -- ``reconcile``, an analysis script --
pick up a run it did not submit.

Writes are driver-side and atomic (temp file + ``os.replace``): a crashed or
concurrently-running driver can never leave a half-written index. Workers never
touch this file. Each writes its own segment record under its own ``sim_dir``
instead (:data:`SEGMENT_SUBDIR`), so hundreds of concurrent array tasks never
contend, and :meth:`Ledger.absorb_segment_records` folds them in afterwards.
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import Any

from .._io import map_io

SEGMENT_SUBDIR = "segments"
LEDGER_VERSION = 1

# Observable whose bin count measures progress, recorded per campaign.
DEFAULT_COUNTING_OBS = "Ener_scal"


def segment_dir(sim_dir: str | Path) -> Path:
    """Directory holding one chain's worker-written segment records."""
    return Path(sim_dir) / SEGMENT_SUBDIR


def ledger_path(data_dir: str | Path, name: str) -> Path:
    """Where the campaign ``name`` keeps its ledger under an experiment's data.

    Beside the simulation output rather than next to the launcher, so a campaign
    stays with the data it produced when either is moved.
    """
    return Path(data_dir) / "campaigns" / f"{name}.json"


class Ledger:
    """Read/modify/write access to one campaign's index."""

    def __init__(self, path: Path, data: dict[str, Any]):
        self.path = Path(path)
        self.data = data

    # --- construction -------------------------------------------------------

    @classmethod
    def new(
        cls,
        path: Path,
        *,
        name: str,
        target_bins: int,
        policy: dict[str, Any],
        counting_obs: str = DEFAULT_COUNTING_OBS,
        experiment: str = "",
        env_name: str = "",
    ) -> Ledger:
        return cls(
            path,
            {
                "version": LEDGER_VERSION,
                "name": name,
                "experiment": experiment,
                "env": env_name,
                "target_bins": target_bins,
                "counting_obs": counting_obs,
                "policy": policy,
                "created": datetime.now().isoformat(timespec="seconds"),
                "updated": None,
                "chains": {},
                "followups": [],
            },
        )

    @classmethod
    def load(cls, path: Path) -> Ledger:
        path = Path(path)
        if not path.exists():
            raise SystemExit(
                f"No campaign ledger at {path}. Launch the campaign first, or check "
                "--name / --env."
            )
        data = json.loads(path.read_text())
        if data.get("version") != LEDGER_VERSION:
            raise SystemExit(
                f"{path} has ledger version {data.get('version')}, expected "
                f"{LEDGER_VERSION}."
            )
        return cls(path, data)

    @classmethod
    def load_or_new(cls, path: Path, **kwargs) -> Ledger:
        """Reopen an existing campaign, or start one. Re-launching is additive."""
        if Path(path).exists():
            return cls.load(path)
        return cls.new(Path(path), **kwargs)

    # --- mutation -----------------------------------------------------------

    def upsert_chains(self, chains) -> None:
        """Add chains, preserving the job history of any already present.

        Re-launching a campaign with a wider grid therefore extends it rather
        than resetting the chains that have already done work.
        """
        for chain in chains:
            existing = self.data["chains"].get(chain.chain_id, {})
            record = chain.to_record()
            record["segments"] = existing.get("segments", [])
            self.data["chains"][chain.chain_id] = record

    def add_segment(self, chain_id: str, segment: dict[str, Any]) -> None:
        """Append a submitted segment to a chain's history."""
        self.data["chains"][chain_id].setdefault("segments", []).append(segment)

    def record_bins(
        self, counts: dict[str, int], settled: Collection[str] = ()
    ) -> bool:
        """Cache each chain's measured bin count. True if anything changed.

        This is what stops a status check paying for the whole grid every time.
        A chain that has reached its target is finished -- ALF only ever appends
        bins, and the worker sizes its last segment to land exactly on the
        target -- so once that count is written here, no later run needs to open
        that ``data.h5`` again.

        Only a *higher* count is recorded. A read that raced ALF's writer, or a
        directory that has yet to appear on a lagging filesystem, comes back low
        or zero, and letting that overwrite a good value would make the campaign
        look like it had gone backwards.

        ``settled`` names the chains that had no job running when this count was
        taken, and is what lets an *unfinished* chain be cached too: nothing
        writes to an idle chain's ``data.h5``, so its count still stands next
        time. Recorded as ``bins_segments``, the chain's segment count at the
        moment of reading, because "idle then and idle now" is not enough on its
        own -- a job could have run and finished in between. Submitting one adds
        a segment, so a segment count that still matches means nothing has run
        since. A chain read while a job *was* running gets no marker at all: its
        file was being appended to as it was read, and that count is a snapshot,
        not a resting value.
        """
        changed = False
        for chain_id, bins in counts.items():
            record = self.data["chains"].get(chain_id)
            if record is None:
                continue
            if bins > record.get("bins", -1):
                record["bins"] = int(bins)
                changed = True
            marker = len(record.get("segments", [])) if chain_id in settled else None
            if marker != record.get("bins_segments"):
                if marker is None:
                    record.pop("bins_segments", None)
                else:
                    record["bins_segments"] = marker
                changed = True
        return changed

    def bins_still_stand(self, record: dict[str, Any]) -> bool:
        """True if this chain's cached count can be reused without reading.

        Only ever true for a count :meth:`record_bins` took while the chain was
        idle, and only while its segment list is the one it was taken at. The
        caller still has to establish that no job is running *now* -- this
        answers the other half, that none has run since.
        """
        marker = record.get("bins_segments")
        return marker is not None and marker == len(record.get("segments", []))

    def add_followup(self, record: dict[str, Any]) -> None:
        """Record a job chained after the campaign (e.g. an analysis stage)."""
        self.data.setdefault("followups", []).append(record)

    def absorb_segment_records(self, skip_finished: bool = False) -> int:
        """Merge worker-written segment records into the ledger.

        Matches on ``job_id`` and fills in what only the node knew: bins before
        and after, elapsed time, and the ``CPU_MAX`` it actually chose. Returns
        the number of records folded in.

        ``scontrol requeue`` reuses the job id, so several worker records can
        match one submitted segment. Records are read in filename order, which
        is job-then-timestamp, so the ledger ends up holding the latest
        attempt; the per-chain record files keep them all, which is what
        :func:`~py_alf.campaign.worker.measured_hours_per_bin` reads.

        Each chain's ``segment_dir`` scan is a directory listing plus however
        many small JSON reads it turns up, independent of every other chain's
        -- the same shape of filesystem probe :class:`~py_alf.campaign.campaign.Campaign`
        already fans out, so a campaign with thousands of chains does not pay
        for this scan one chain at a time.

        ``skip_finished`` drops the chains :meth:`record_bins` has already seen
        reach their target: no further segment will ever run for one, so its
        directory can only hold records that were folded in on an earlier pass.
        Late in a campaign that is nearly the whole grid, and the listings it
        avoids are the bulk of what a status check has left to pay for.
        """
        records = list(self.data["chains"].values())
        if skip_finished:
            records = [
                r
                for r in records
                if r.get("bins", -1) < r.get("target_bins", self.target_bins)
            ]

        def _absorb_one(record: dict[str, Any]) -> int:
            by_job = {
                s.get("job_id"): s
                for s in record.get("segments", [])
                if s.get("job_id")
            }
            merged = 0
            for path in sorted(segment_dir(record["sim_dir"]).glob("*.json")):
                try:
                    worker = json.loads(path.read_text())
                except (OSError, json.JSONDecodeError):
                    continue  # still being written, or truncated by a hard kill
                target = by_job.get(worker.get("job_id"))
                if target is None:
                    target = dict(worker)
                    record.setdefault("segments", []).append(target)
                    if worker.get("job_id"):
                        by_job[worker["job_id"]] = target
                else:
                    target.update(worker)
                merged += 1
            return merged

        return sum(map_io(_absorb_one, records))

    def save(self) -> Path:
        """Atomically write the ledger."""
        self.data["updated"] = datetime.now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        os.replace(tmp, self.path)
        return self.path

    # --- lookup -------------------------------------------------------------

    @property
    def name(self) -> str:
        return str(self.data.get("name", ""))

    @property
    def target_bins(self) -> int:
        return int(self.data["target_bins"])

    @property
    def counting_obs(self) -> str:
        return str(self.data["counting_obs"])

    @property
    def chains(self) -> dict[str, dict[str, Any]]:
        return self.data["chains"]
