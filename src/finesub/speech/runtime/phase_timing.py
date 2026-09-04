"""Where the ASR stage's wall clock actually goes.

`docs/plans/crispasr-followups.md` -> A3.2 makes this one measurement the blocker for
three separate decisions: whether upper-layer batching (A1) is worth wiring,
what stage overlap would be worth, and whether the Qwen referee's marginal cost
(A4) can be closed. Until now `vad-asr` reported a single opaque
`asr_align_sec` covering encode, decode, the Python-side refine pass, the
rescue ladder, the inline referee and DP re-segmentation together -- so any
estimate of any of those was a guess.

Two properties make this safe to leave switched on in production code:

* **Inactive is free.** With no collector installed, `phase()` does one
  thread-local lookup and returns a shared no-op. It is not gated behind a
  flag, because a profiler that has to be enabled is a profiler that is off
  when the interesting run happens.
* **It cannot change results.** Nothing here touches the values flowing
  through; it only reads the clock around them.

**Inclusive and exclusive are both recorded, deliberately.** The rescue ladder
*contains* decodes, so an exclusive-only accounting would report rescue as
nearly free (it is mostly waiting on the decodes it triggers) and an
inclusive-only accounting would double-count those decodes against both. The
question "what would removing rescue save" needs the inclusive number; the
question "what is this run made of" needs the exclusive one, which partitions
the total. Recording one and deriving the other is impossible, so both are kept.

⚠ **Recursive phases inflate `inclusive`.** A phase entered inside itself adds
the same span to `inclusive` once per level. The phases in this code base are
not recursive; if one ever becomes so, read `exclusive` and `calls` instead.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

__all__ = ["PhaseStat", "collect", "merge", "phase", "active"]


@dataclass
class PhaseStat:
    """One phase's totals across a collection scope."""

    calls: int = 0
    inclusive_s: float = 0.0
    exclusive_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "calls": self.calls,
            "inclusive_s": round(self.inclusive_s, 4),
            "exclusive_s": round(self.exclusive_s, 4),
        }


@dataclass
class _Frame:
    name: str
    started: float
    children_s: float = 0.0


@dataclass
class _Collector:
    stats: dict[str, PhaseStat] = field(default_factory=dict)
    stack: list[_Frame] = field(default_factory=list)


_local = threading.local()


def active() -> bool:
    return getattr(_local, "collector", None) is not None


@contextmanager
def collect(into: dict[str, PhaseStat] | None = None) -> Iterator[dict[str, PhaseStat]]:
    """Accumulate phase timings for one stage run, on this thread.

    Thread-local like `transcribe.collecting_stats`, and for the same reason:
    the stage leases a model per worker, so a process-wide accumulator would
    interleave two workers' phases into one meaningless total. The ASR stage
    runs a single worker today, but that is a resource decision, not a promise.

    `into` continues an existing table instead of starting a fresh one, and a
    phase seen in both scopes **adds up**. That is the point: the Qwen referee
    can run twice in one stage -- once inline for a language-vote redecode,
    once at the tail for verification -- and merging the two tables with
    `dict.update` would silently drop the first one's calls and seconds,
    leaving the A3/A4 telemetry describing only part of the stage.
    """

    previous = getattr(_local, "collector", None)
    collector = _Collector(stats=into if into is not None else {})
    _local.collector = collector
    try:
        yield collector.stats
    finally:
        # An exception mid-phase leaves frames on the stack. They are dropped
        # rather than attributed: a partial span is not a measurement.
        _local.collector = previous


def merge(into: dict[str, PhaseStat], stats: dict[str, PhaseStat]) -> None:
    """Add a table collected on another thread into `into`, phase by phase.

    `collect(into=...)` cannot do this for a helper thread: the collector is
    thread-local, so the thread keeps its own table and hands it over when it
    is joined. Same rule as `into=` -- a phase seen in both **adds up**.
    """

    for name, stat in stats.items():
        target = into.setdefault(name, PhaseStat())
        target.calls += stat.calls
        target.inclusive_s += stat.inclusive_s
        target.exclusive_s += stat.exclusive_s


class _NoOp:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NO_OP = _NoOp()


class _Span:
    """Times one phase and folds it into its parent's child total."""

    __slots__ = ("_collector", "_name")

    def __init__(self, collector: _Collector, name: str) -> None:
        self._collector = collector
        self._name = name

    def __enter__(self) -> None:
        self._collector.stack.append(_Frame(self._name, time.perf_counter()))

    def __exit__(self, *exc: object) -> None:
        collector = self._collector
        if not collector.stack:
            return
        frame = collector.stack.pop()
        elapsed = time.perf_counter() - frame.started
        stat = collector.stats.get(frame.name)
        if stat is None:
            stat = collector.stats[frame.name] = PhaseStat()
        stat.calls += 1
        stat.inclusive_s += elapsed
        stat.exclusive_s += elapsed - frame.children_s
        if collector.stack:
            collector.stack[-1].children_s += elapsed


def phase(name: str):
    """Time `name`, or do nothing at all when no collector is installed."""

    collector = getattr(_local, "collector", None)
    if collector is None:
        return _NO_OP
    return _Span(collector, name)


def as_dict(stats: dict[str, PhaseStat]) -> dict[str, dict[str, float]]:
    """Serialisable form, ordered by how much wall clock each phase owns."""

    return {
        name: stat.as_dict()
        for name, stat in sorted(
            stats.items(), key=lambda item: item[1].exclusive_s, reverse=True
        )
    }
