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

    def record_bins(self, counts: dict[str, int]) -> bool:
        """Cache each chain's bin count. True if anything changed.

        Only a higher count is recorded: a read that raced ALF's writer can come
        back low, and bins never decrease.
        """
        changed = False
        for chain_id, bins in counts.items():
            record = self.data["chains"].get(chain_id)
            if record is not None and bins > record.get("bins", -1):
                record["bins"] = int(bins)
                changed = True
        return changed

    def add_followup(self, record: dict[str, Any]) -> None:
        """Record a job chained after the campaign (e.g. an analysis stage)."""
        self.data.setdefault("followups", []).append(record)

    def absorb_segment_records(self, skip_finished: bool = False) -> int:
        """Merge worker-written segment records into the ledger, by ``job_id``.

        Fills in what only the node knew (bins before and after, elapsed time,
        the ``CPU_MAX`` it chose) and returns how many records were folded in.
        A requeue reuses the job id; records are read in time order, so the
        latest attempt wins. ``skip_finished`` skips chains already at their
        target, which can gain no new records.
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
