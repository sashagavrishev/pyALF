"""One Markov chain: one ``Simulation``, the unit a campaign schedules.

The caller builds the ``Simulation`` objects, so a campaign runs in exactly the
directories that caller's analysis reads.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from ..simulation import Simulation

# Run controls change how long a job runs, never what it computes, and never
# enter sim_dir, so they are left out of a chain's recorded parameters.
RUN_CONTROL_KEYS = ("CPU_MAX", "NBin")


def chain_id(sim: Simulation, mc_seed: int) -> str:
    """Stable short identifier for the chain ``sim`` runs with ``mc_seed``.

    Hashes the directory name, a canonical rendering of the parameters, so the
    id survives moving the data root.
    """
    key = f"{sim.ham_name}|{os.path.basename(sim.sim_dir)}|{mc_seed}"
    return hashlib.blake2b(key.encode(), digest_size=6).hexdigest()


@dataclass
class Chain:
    """One Markov chain: a target, a place to run, and where it sits in a grid."""

    chain_id: str
    sim: Simulation
    target_bins: int
    # Free-form coordinates of this chain in the caller's grid. Whatever makes
    # the chain distinct beyond its Markov seed goes here -- a disorder seed, a
    # sweep value -- so the core needs no field per experiment shape.
    point: dict[str, Any] = field(default_factory=dict)
    # Chains sharing an array_key are submitted as one SLURM array, so they must
    # share a cost: by convention one key per parameter point.
    array_key: str = ""

    @classmethod
    def from_sim(
        cls,
        sim: Simulation,
        target_bins: int,
        point: dict[str, Any] | None = None,
        array_key: str = "",
    ) -> Chain:
        """The chain ``sim`` runs, identified by its directory and Monte-Carlo seed."""
        return cls(
            chain_id=chain_id(sim, sim.mc_seed),
            sim=sim,
            target_bins=target_bins,
            point=dict(point or {}),
            array_key=array_key,
        )

    @property
    def sim_dir(self) -> str:
        return self.sim.sim_dir

    @property
    def mc_seed(self) -> int:
        return self.sim.mc_seed

    def to_record(self) -> dict[str, Any]:
        """Ledger representation (JSON-safe, excludes the Simulation object)."""
        return {
            "sim_dir": self.sim_dir,
            # Recorded so ``Campaign.from_ledger`` can rebuild the Simulation
            # without knowing which model the campaign was launched for.
            "ham_name": self.sim.ham_name,
            "mc_seed": self.mc_seed,
            "target_bins": self.target_bins,
            "point": dict(self.point),
            "array_key": self.array_key,
            "params": {
                k: v for k, v in self.sim.sim_dict.items() if k not in RUN_CONTROL_KEYS
            },
        }
