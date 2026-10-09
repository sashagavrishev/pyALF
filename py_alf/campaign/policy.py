"""How long each checkpoint-restart segment may run: its ``CPU_MAX``.

ALF breaks its bin loop once fewer than about 1.5 bins still fit in ``CPU_MAX``
(``control_mod.F90:make_truncation``), with ``data.h5`` and ``confout_0`` flushed,
so a segment ends cleanly on its budget. Capping ``CPU_MAX`` below
``max_partition``'s limit turns a long chain into jobs that each fit that queue,
and a chain with little left asks for less and lands in a smaller partition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentPolicy:
    """Wall-time budget rules shared by every chain in a campaign.

    ``max_partition`` is the largest queue a segment may target. ``margin`` keeps
    ``CPU_MAX`` below its limit, so the submitter's ``--time`` still leaves ALF
    room to stop and write. ``safety`` inflates the estimate, since asking too
    little costs a whole extra segment and asking too much only priority.
    ``max_segments`` bounds the requeue budget.
    """

    max_partition: str = "medium"
    margin: float = 0.95
    safety: float = 1.3
    max_segments: int = 6
    # Floor on a segment's budget: make_truncation needs a few bins to measure,
    # and a shorter run is dwarfed by its queue wait.
    min_hours: float = 0.25
    # Chains expected to need less than this share a task, run one after another
    # up to this budget; 0 gives every chain a task of its own.
    pack_hours: float = 0.0
    # Charged per packed chain for prep, ALF start-up and opening data.h5.
    pack_overhead_hours: float = 1 / 60
    # Tasks per SLURM array, below the cluster's MaxArraySize.
    max_array_tasks: int = 1000

    def pack_budget(self, partition_rules: dict) -> float:
        """Hours a pack may fill: ``pack_hours``, but never past :meth:`max_hours`,
        or the task's request would outgrow the partition."""
        return min(self.pack_hours, self.max_hours(partition_rules))

    def chain_hours(self, remaining_bins: int, hours_per_bin: float) -> float:
        """Expected hours to finish a chain, with the safety factor."""
        return remaining_bins * hours_per_bin * self.safety

    def max_hours(self, partition_rules: dict) -> float:
        """Largest ``CPU_MAX`` any segment may ask for, in hours.

        Unbounded where there is no queue to satisfy (the ``local`` and
        ``debug`` executors define no partitions): a segment there simply runs
        until the chain is done, which is one segment.
        """
        if not partition_rules:
            return float("inf")
        try:
            limit = float(partition_rules[self.max_partition])
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"SegmentPolicy.max_partition={self.max_partition!r} is not in this "
                f"environment's partition_rules ({sorted(partition_rules)})"
            ) from exc
        return limit * self.margin

    def cpu_max(
        self, remaining_bins: int, hours_per_bin: float, partition_rules: dict
    ) -> float:
        """``CPU_MAX`` (hours) for a segment with ``remaining_bins`` still to do.

        Sized to *finish* the chain when that fits the partition cap, and pinned
        at the cap otherwise -- in which case the chain simply needs another
        segment, which is the whole point of the mechanism.
        """
        want = remaining_bins * hours_per_bin * self.safety
        return min(max(want, self.min_hours), self.max_hours(partition_rules))

    def segments_needed(
        self, total_bins: int, hours_per_bin: float, partition_rules: dict
    ) -> int:
        """How many segments the *initial* estimate says the target will take.

        Used only to size the requeue budget; the worker re-measures on every
        attempt, and :meth:`~py_alf.campaign.campaign.Campaign.reconcile`
        covers any shortfall, so this being wrong is recoverable either way.
        """
        cap = self.max_hours(partition_rules)
        total_hours = total_bins * hours_per_bin * self.safety
        return max(1, min(self.max_segments, math.ceil(total_hours / cap)))
