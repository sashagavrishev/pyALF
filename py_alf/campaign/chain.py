"""One Markov chain: the unit a campaign schedules.

A *chain* is one ``Simulation`` -- one set of parameters, one Monte-Carlo seed,
one ``sim_dir`` -- driven to a bin target by however many jobs that takes. What
makes a chain physically distinct (a disorder realisation, a point of a
parameter grid) is the caller's business: it builds the ``Simulation`` objects
and hands them over, so a campaign resolves exactly the directories that
caller's analysis already looks in.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

from ..simulation import Simulation

# Run controls: they change how long a job runs, never what it computes, and
# (verified against ALF's parameter list) never appear in sim_dir. Excluded from
# chain_id so a chain keeps its identity across segments with different budgets.
RUN_CONTROL_KEYS = ("CPU_MAX", "NBin")


def chain_id(sim: Simulation, mc_seed: int) -> str:
    """Stable short identifier for the chain that ``sim`` runs.

    Hashes the *directory name* rather than the parameter dict: that name is
    already a canonical rendering of every Hamiltonian parameter, so the id
    inherits its uniqueness while staying independent of the data root --
    relocating the data does not renumber anything. ``mc_seed`` is folded in so
    two chains sharing a parameter set but running different Markov chains stay
    distinguishable.
    """
    key = f"{sim.ham_name}|{os.path.basename(sim.sim_dir)}|{mc_seed}"
    return hashlib.blake2b(key.encode(), digest_size=6).hexdigest()


@dataclass
class Chain:
    """One Markov chain: a target, a place to run, and where it sits in a grid."""

    chain_id: str
    sim: Simulation
    mc_seed: int
    target_bins: int
    # Free-form coordinates of this chain in the caller's grid. Whatever makes
    # the chain distinct beyond its Markov seed goes here -- a disorder seed, a
    # sweep value -- so the core needs no field per experiment shape.
    point: dict[str, Any] = field(default_factory=dict)
    # Chains sharing an array_key are submitted as one SLURM array, so they must
    # share a cost: by convention one key per parameter point.
    array_key: str = ""

    @property
    def sim_dir(self) -> str:
        return self.sim.sim_dir

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
