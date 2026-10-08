"""Checkpoint-restart campaigns: grids of Markov chains driven to a bin target.

Public surface::

    from py_alf.campaign import (
        Campaign, Chain, ChainStatus, Ledger, SegmentPolicy, chain_id, ledger_path
    )

A caller builds one :class:`~py_alf.campaign.chain.Chain` per Simulation it
wants driven to a target, wraps them in a
:class:`~py_alf.campaign.campaign.Campaign` together with a configured
``ClusterSubmitter``, and calls ``launch()``, which submits one SLURM array per
group of chains. Getting from there to the target is three layers: ``CPU_MAX``
stops ALF cleanly inside the partition limit, submitit requeues a task the wall
clock or a preemption cut short, and ``reconcile`` resubmits whatever neither
delivered. See :mod:`py_alf.campaign.campaign`.
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
