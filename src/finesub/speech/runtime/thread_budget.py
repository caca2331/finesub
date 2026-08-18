"""Keep concurrent shards from oversubscribing the CPU's OpenMP teams."""

from __future__ import annotations

import contextlib

import torch

from ...reporting import current_reporter


@contextlib.contextmanager
def bounded_intra_op_threads(concurrency: int):
    """Split the torch intra-op thread budget across ``concurrency`` workers.

    Every shard thread that enters a torch CPU op spins up its own OpenMP team
    of ``torch.get_num_threads()`` workers, so N shards ask the machine for N
    times the budget. torch ships its own libiomp, the separation stage leaves
    another native pool behind in the same process, and an oversubscribed team
    that spins past ``KMP_BLOCKTIME`` and then sleeps matches the dual-shard
    stall in tmp/mt8g-8gb-multithread-handoff.md exactly: process alive, CPU
    near zero, GPU idle, VRAM held, no checkpoint movement for many minutes.

    This is a preventive measure against that hypothesis, not a proven fix --
    the stall was never reproduced. It is cheap and reversible: only the number
    of threads changes, never the work, and aligned output was verified
    unchanged across thread counts (docs/wt-parallelism.md).
    """

    workers = max(1, int(concurrency))
    previous = torch.get_num_threads()
    budget = max(1, previous // workers)
    if workers <= 1 or budget == previous:
        yield
        return

    torch.set_num_threads(budget)
    current_reporter().debug(
        "intra-op thread budget",
        {"from": previous, "to": budget, "shards": workers},
    )
    try:
        yield
    finally:
        torch.set_num_threads(previous)
