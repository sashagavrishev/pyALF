"""Checkpoint-restart campaigns: grids of Markov chains driven to a bin target.

Public surface::

    from py_alf.campaign import (
        Campaign, Chain, ChainStatus, Ledger, SegmentPolicy, chain_id, ledger_path
    )

A caller builds one :class:`Chain` per Simulation (:meth:`Chain.from_sim`), wraps
them in a :class:`Campaign` with a configured ``ClusterSubmitter``, and calls
``launch()``, which submits one SLURM array per ``array_key``. Three layers then
carry each chain to its target:

1. ``CPU_MAX`` (:mod:`~py_alf.campaign.policy`): ALF stops at a bin boundary
   inside the partition limit, with ``data.h5`` and its checkpoint flushed;
2. submitit's requeue (:func:`~py_alf.campaign.worker.run_segment`): retries a
   task the wall clock or a preemption cut short, up to the array's budget;
3. :meth:`Campaign.reconcile`: resubmits what neither delivered, such as a
   cancellation, a node failure or an exhausted budget.

Under submitit a wall-clock stop is recorded as ``FAILED`` or ``CANCELLED``,
never ``TIMEOUT``, so progress is judged by bins on disk, not by job state.
"""

from .campaign import Campaign, ChainStatus
from .chain import Chain, chain_id
from .ledger import Ledger, ledger_path
from .policy import SegmentPolicy

__all__ = [
    "Campaign",
    "Chain",
    "ChainStatus",
    "Ledger",
    "SegmentPolicy",
    "chain_id",
    "ledger_path",
]
