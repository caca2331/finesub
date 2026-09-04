"""The measurement protocol, so a benchmark cannot quietly cheat.

`docs/plans/crispasr-followups.md` -> P4 fixes the discipline *before* the machine's
baselines are re-taken, because the 2026-08-27 hardware change invalidated all
of them and an undisciplined re-take would just mint new numbers nobody can
trust. Every rule below exists because someone shipped a wrong conclusion
without it:

* **Discard the first run, warm up by shape.** Windows/WDDM promotes the GPU to
  P0 only after seconds of sustained activity, and it actively throttles bursty
  "sub-100 ms kernel group" work -- which is exactly the shape of chunked ASR
  decoding and chunked separator inference.
* **Report absolute ms next to any RTF.** An RTF alone hides both the clip
  length and the load time.
* **Alternate A,B,A,B and take the median of per-round ratios.** Running all of
  A then all of B compares a cold card against a hot one.
* **Scale check.** A run that is 5x longer must do ~5x the work. The tell for a
  fake win is "the 55 s clip finished as fast as the 11 s clip".
* **A non-zero exit or an empty transcript is a FAIL, never a timing.** The
  canonical disaster is a backend that failed to load, exited in 0.5 s, and
  minted a "102x realtime" record.
* **Declare the aggregate.** `min-of-N` flatters CPU runs (no scheduler noise
  floor to beat) and `median` flatters GPU runs (absorbs the occasional WDDM
  stall). Picking one silently is picking the conclusion.

Nothing here imports `finesub`; it is a measurement protocol, not a pipeline
stage. This is a development tool, maintained on demand.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal, Sequence

Aggregate = Literal["median", "min"]


class BenchmarkFailure(RuntimeError):
    """A run that must not be reported as a timing."""


@dataclass(frozen=True)
class RunConfig:
    """The eight things a timing is meaningless without.

    P4 item 6. Every field is required -- there is no default that would let a
    report omit one and still look complete.
    """

    backend: str
    quantization: str
    device: str
    threads: int
    thermal_state: Literal["cold", "warm"]
    load_excluded: bool
    clip_seconds: float
    aggregate: Aggregate
    #: Free-form, but this is where "a video was playing on the desktop" goes.
    notes: str = ""

    def __post_init__(self) -> None:
        if self.clip_seconds <= 0:
            raise ValueError("clip_seconds must be positive")
        if self.aggregate not in ("median", "min"):
            raise ValueError("aggregate must be declared as 'median' or 'min'")


@dataclass
class Timing:
    """One measured quantity, kept with the samples it was reduced from."""

    label: str
    samples_ms: list[float]
    aggregate: Aggregate
    discarded_ms: list[float] = field(default_factory=list)

    @property
    def value_ms(self) -> float:
        if not self.samples_ms:
            raise BenchmarkFailure(f"{self.label}: no samples survived")
        if self.aggregate == "min":
            return min(self.samples_ms)
        return statistics.median(self.samples_ms)

    @property
    def spread_ms(self) -> float:
        """Max - min. A spread near the value itself means the run is noise."""

        return max(self.samples_ms) - min(self.samples_ms) if self.samples_ms else 0.0

    def rtf(self, clip_seconds: float) -> float:
        return (self.value_ms / 1000.0) / clip_seconds

    def render(self, clip_seconds: float) -> str:
        """Absolute ms first; RTF is never allowed to travel alone."""

        return (
            f"{self.label}: {self.value_ms:8.1f} ms"
            f"  (spread {self.spread_ms:6.1f} ms over n={len(self.samples_ms)})"
            f"  RTF {self.rtf(clip_seconds):.4f}"
        )


def repeat(
    fn: Callable[[], object],
    *,
    warmup: int = 1,
    repeats: int = 3,
    label: str,
    aggregate: Aggregate,
    validate: Callable[[object], None] | None = None,
) -> Timing:
    """Run `fn` warmup+repeats times, keeping only the post-warmup samples.

    `validate` is where "an empty transcript is a FAIL" lives: raise from it and
    the run is refused rather than timed.
    """

    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    discarded: list[float] = []
    kept: list[float] = []
    for index in range(warmup + repeats):
        started = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if validate is not None:
            validate(result)
        (discarded if index < warmup else kept).append(elapsed_ms)
    return Timing(label=label, samples_ms=kept, aggregate=aggregate, discarded_ms=discarded)


def paired_ratio(
    a: Callable[[], object],
    b: Callable[[], object],
    *,
    rounds: int = 4,
    labels: tuple[str, str] = ("a", "b"),
) -> dict[str, object]:
    """Compare two arms with the within-round order effect *cancelled*, not averaged.

    Two biases, and they need different treatment:

    * **Thermal drift across the run.** Handled by taking the ratio *inside*
      one round: `median(all_a) / median(all_b)` would measure the ordering as
      much as the code.
    * **The second caller's advantage inside a round.** The call that runs
      second inherits whatever the first one warmed -- caches, clocks,
      allocator state. Alternating AB/BA gives each arm that advantage half the
      time, but **pooling all the ratios and taking the median does not remove
      it**: with an odd `rounds` the majority order simply wins the median, and
      even with equal counts the median of a mixed distribution is not the
      unbiased estimate.

    Model the advantage as a multiplicative factor `k` on whichever arm goes
    second. Then

        ratio_ab = k * (ta / tb)        ratio_ba = (1 / k) * (ta / tb)

    so the **geometric mean of the two per-order medians** is `ta / tb` exactly
    -- `k` cancels. That is what `ratio` reports. `order_effect` reports the
    recovered `k = sqrt(ratio_ab / ratio_ba)`, so the bias is a number you can
    read rather than one the aggregate quietly absorbed.

    `rounds` must be **even**: unequal AB/BA counts reintroduce the bias the
    geometric mean is there to remove, so the function refuses rather than
    returning a number that looks fine.
    """

    if rounds < 2 or rounds % 2:
        raise ValueError(
            f"rounds must be an even number >= 2 (got {rounds}); unequal AB/BA "
            "counts leave the order effect in the result"
        )

    a_ms: list[float] = []
    b_ms: list[float] = []
    ab_ratios: list[float] = []
    ba_ratios: list[float] = []
    for index in range(rounds):
        a_first = index % 2 == 0
        first_call, second_call = (a, b) if a_first else (b, a)

        started = time.perf_counter()
        first_call()
        first = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        second_call()
        second = (time.perf_counter() - started) * 1000.0

        a_elapsed, b_elapsed = (first, second) if a_first else (second, first)
        a_ms.append(a_elapsed)
        b_ms.append(b_elapsed)
        ratio = a_elapsed / b_elapsed if b_elapsed else float("inf")
        (ab_ratios if a_first else ba_ratios).append(ratio)

    ab_median = statistics.median(ab_ratios)
    ba_median = statistics.median(ba_ratios)
    combined = math.sqrt(ab_median * ba_median) if ab_median > 0 and ba_median > 0 else float("nan")
    order_effect = (
        math.sqrt(ab_median / ba_median) if ba_median > 0 and ab_median > 0 else float("nan")
    )
    return {
        "labels": list(labels),
        f"{labels[0]}_ms": a_ms,
        f"{labels[1]}_ms": b_ms,
        "ab_ratios": ab_ratios,
        "ba_ratios": ba_ratios,
        "ratio_ab_median": ab_median,
        "ratio_ba_median": ba_median,
        # The estimate to quote. Geometric, not arithmetic: the bias is
        # multiplicative, so only the geometric mean cancels it.
        "ratio": combined,
        # k > 1 means going second was worth something. Reported so a run can
        # say how large the effect it just removed was.
        "order_effect": order_effect,
    }


def scale_check(
    points: Sequence[tuple[float, float]],
    *,
    tolerance: float = 0.35,
    quantity: str = "work",
) -> dict[str, object]:
    """Assert the work done grows with the input, roughly proportionally.

    `points` is (input_seconds, observed_quantity). The failure this catches is
    not a slow run but a *fake fast* one: a 5x longer clip that produced the
    same amount of output was not transcribed, it was truncated or dropped.
    """

    if len(points) < 2:
        raise ValueError("scale_check needs at least two points")
    base_seconds, base_value = points[0]
    if base_seconds <= 0 or base_value <= 0:
        raise BenchmarkFailure(f"scale_check: degenerate base point {points[0]!r}")

    rows: list[dict[str, float]] = []
    worst = 0.0
    for seconds, value in points:
        expected_ratio = seconds / base_seconds
        actual_ratio = value / base_value
        deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        worst = max(worst, deviation)
        rows.append(
            {
                "seconds": seconds,
                quantity: value,
                "expected_x": expected_ratio,
                "actual_x": actual_ratio,
                "deviation": deviation,
            }
        )
    return {"points": rows, "worst_deviation": worst, "passed": worst <= tolerance}


def refuse_empty(result: object, *, what: str) -> None:
    """The 102x-realtime guard: nothing produced means nothing was measured."""

    if result is None:
        raise BenchmarkFailure(f"{what}: produced None -- this is a FAIL, not a timing")
    if isinstance(result, (list, tuple, dict, str)) and len(result) == 0:
        raise BenchmarkFailure(f"{what}: produced nothing -- this is a FAIL, not a timing")


def environment() -> dict[str, object]:
    """Machine facts that belong in every report."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def render_report(config: RunConfig, timings: Sequence[Timing], extra: dict[str, object] | None = None) -> str:
    lines = ["=" * 78, "CONFIG", "=" * 78]
    for key, value in asdict(config).items():
        lines.append(f"  {key:<16} {value}")
    lines.append("")
    lines.append("=" * 78)
    lines.append(f"TIMINGS  (aggregate = {config.aggregate})")
    lines.append("=" * 78)
    for timing in timings:
        lines.append("  " + timing.render(config.clip_seconds))
        if timing.discarded_ms:
            discarded = ", ".join(f"{value:.1f}" for value in timing.discarded_ms)
            lines.append(f"      discarded warmup: {discarded} ms")
    if extra:
        lines.append("")
        lines.append("=" * 78)
        lines.append("CONTEXT")
        lines.append("=" * 78)
        lines.append(json.dumps(extra, indent=2, ensure_ascii=False, default=str))
    return "\n".join(lines)
