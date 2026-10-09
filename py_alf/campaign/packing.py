"""Grouping short chains into tasks that each run several chains in turn."""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Sequence
from typing import TypeVar

T = TypeVar("T")


def pack(items: Sequence[tuple[Hashable, T, float]], capacity: float) -> list[list[T]]:
    """Group ``(key, item, hours)`` into packs of at most ``capacity`` hours.

    Worst-fit decreasing: items go largest first into the least loaded pack,
    and a new pack opens only when even that one cannot take them. Ties break
    on ``key``, so the same input packs the same way. An item larger than
    ``capacity`` gets a pack of its own.
    """
    ordered = sorted(items, key=lambda kit: (-kit[2], kit[0]))
    packs: list[list[T]] = []
    loads: list[tuple[float, int]] = []  # (hours, pack index), least loaded first
    room = capacity * (1 + 1e-9)  # a sum of 0.1s must not overflow 2.0 by rounding
    for _key, item, hours in ordered:
        if loads and loads[0][0] + hours <= room:
            load, i = heapq.heappop(loads)
            packs[i].append(item)
            heapq.heappush(loads, (load + hours, i))
        else:
            packs.append([item])
            heapq.heappush(loads, (hours, len(packs) - 1))
    return packs
