"""Checkpoint-restart campaigns: grids of Markov chains driven to a bin target.

Public surface::

    from py_alf.campaign import Campaign, Chain, SegmentPolicy, ledger_path

A caller builds one :class:`~py_alf.campaign.chain.Chain` per Simulation it
wants driven to a target, wraps them in a
:class:`~py_alf.campaign.campaign.Campaign` together with a configured
``ClusterSubmitter``, and calls ``launch()``, which submits one SLURM array per
group of chains. Getting from there to the target is three layers: ``CPU_MAX``
stops ALF cleanly inside the partition limit, submitit requeues a task the wall
clock or a preemption cut short, and ``reconcile`` resubmits whatever neither
delivered. See :mod:`py_alf.campaign.campaign`.
"""

from .campaign import (
    ACTIVE_STATES,
    DEFAULT_HOURS_PER_BIN,
    Campaign,
    ChainStatus,
)
from .chain import Chain, chain_id
from .ledger import (
    DEFAULT_COUNTING_OBS,
    Ledger,
    ledger_path,
    segment_dir,
)
from .policy import SegmentPolicy
from .worker import SegmentPlan, measured_hours_per_bin, run_segment

__all__ = [
    "ACTIVE_STATES",
    "DEFAULT_COUNTING_OBS",
    "DEFAULT_HOURS_PER_BIN",
    "Campaign",
    "Chain",
    "ChainStatus",
    "Ledger",
    "SegmentPlan",
    "SegmentPolicy",
    "chain_id",
    "ledger_path",
    "measured_hours_per_bin",
    "run_segment",
    "segment_dir",
]
