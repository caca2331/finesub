"""The torch intra-op thread budget: split across concurrent shards, or widened for one CPU-bound model pass."""

from __future__ import annotations

import contextlib
import os

import torch

from ...reporting import current_reporter

#: Share of the logical CPUs one CPU-bound model pass may use (owner 2026-11-11: the rest is left to other work).
WIDE_SHARE = 0.75


def _logical_cpus() -> int:
    """The logical CPUs this process may run on: its CPU affinity where the OS reports one (Linux cpusets), else all
    of them. A CPU quota (Docker `--cpus`) and Windows affinity are not seen."""
    affinity = getattr(os, "sched_getaffinity", None)
    return len(affinity(0)) if affinity else (os.cpu_count() or 1)


@contextlib.contextmanager
def widened_intra_op_threads():
    """Raise the torch intra-op threads to `WIDE_SHARE` of the logical CPUs for the body, never below what they are.

    torch starts at the physical core count. On the forced aligner's CPU pass (fp32 matrix products) 12 or 16
    threads on an 8-core / 16-thread CPU were 15–23% faster than 8 with bit-identical output; 12 matched 16
    (`docs/joint-asr.md`); other topologies (hybrid cores, many cores) were not measured. Without SMT the share is
    below the default and nothing changes. A set `OMP_NUM_THREADS` or `MKL_NUM_THREADS` is the user's choice and is
    left alone. The setting is process-wide: torch ops running alongside (another batch lane) get the same budget
    until the body ends. The count is put back only if it is still ours: Silero's load sets and restores it too
    (`silero_ghost.py`), and restoring over a change made meanwhile could leave the process on its one thread. (The reverse
    interleaving can leave the widened count in place, which only gives later torch ops more threads.)
    """

    previous = torch.get_num_threads()
    target = int(_logical_cpus() * WIDE_SHARE)
    if os.environ.get("OMP_NUM_THREADS") or os.environ.get("MKL_NUM_THREADS") or target <= previous:
        yield
        return

    torch.set_num_threads(target)
    current_reporter().debug("intra-op threads widened", {"from": previous, "to": target})
    try:
        yield
    finally:
        if torch.get_num_threads() == target:
            torch.set_num_threads(previous)


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
