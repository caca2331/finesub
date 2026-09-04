"""Guards for the measurement protocol itself.

`docs/bench-baselines.md` claims the P4 rules are "enforced in code, not
suggested in a comment". That claim is only worth something if the enforcement
is itself checked -- the A8 magnitude moved by 8 percentage points when
`paired_ratio`'s fixed A-then-B order was fixed, which is exactly the kind of
bias a tool can hide inside a plausible-looking number.

Lives beside its tool, per CLAUDE.md's rule that `tools/` tests are not
collected by the default suite. Run explicitly:

    python -m pytest tools/bench/test_discipline.py -o addopts=""
"""

from __future__ import annotations

import math
import statistics

import pytest

from tools.bench import discipline


# --- the eight required fields ----------------------------------------------


def _config(**overrides):
    base = dict(
        backend="fw-refine",
        quantization="float16",
        device="cuda",
        threads=1,
        thermal_state="warm",
        load_excluded=False,
        clip_seconds=30.0,
        aggregate="median",
    )
    base.update(overrides)
    return discipline.RunConfig(**base)


def test_the_aggregate_must_be_declared() -> None:
    """min-of-N flatters CPU, median flatters GPU: picking silently picks the
    conclusion."""

    with pytest.raises(ValueError, match="aggregate"):
        _config(aggregate="mean")


def test_a_zero_length_clip_is_refused() -> None:
    with pytest.raises(ValueError, match="clip_seconds"):
        _config(clip_seconds=0.0)


# --- paired_ratio: the bias this file exists for -----------------------------


def test_the_within_round_order_alternates() -> None:
    """Always running A first leaves B a permanent second-caller advantage."""

    calls: list[str] = []
    result = discipline.paired_ratio(
        lambda: calls.append("a"),
        lambda: calls.append("b"),
        rounds=4,
        labels=("a", "b"),
    )

    assert calls == ["a", "b", "b", "a", "a", "b", "b", "a"]
    assert len(result["ab_ratios"]) == len(result["ba_ratios"]) == 2


def test_an_odd_round_count_is_refused() -> None:
    """Unequal AB/BA counts put the order effect straight back in.

    21 rounds is 11 AB and 10 BA, so a pooled median can be decided by the
    majority order. Refusing beats returning a number that looks fine.
    """

    with pytest.raises(ValueError, match="even"):
        discipline.paired_ratio(lambda: None, lambda: None, rounds=21)


def test_a_known_second_caller_bias_is_recovered_not_averaged() -> None:
    """The contract: inject a known multiplicative bias, get the true ratio back.

    Arm A nominally takes 2x arm B. Whichever arm runs *second* in a round is
    made 1.5x faster -- the shape of a real second-caller advantage (warm
    caches, boosted clocks). A pooled median cannot remove that; the geometric
    mean of the two per-order medians cancels it exactly.
    """

    import time as _time

    nominal = {"a": 0.040, "b": 0.020}
    boost = 1.5
    position = {"n": 0}

    def arm(name: str):
        def run() -> None:
            position["n"] += 1
            is_second = position["n"] % 2 == 0
            _time.sleep(nominal[name] / (boost if is_second else 1.0))

        return run

    result = discipline.paired_ratio(arm("a"), arm("b"), rounds=6, labels=("a", "b"))

    # Each order is biased, in opposite directions, by roughly the boost.
    assert result["ratio_ab_median"] > result["ratio_ba_median"]
    assert result["ratio_ab_median"] == pytest.approx(2.0 * boost, rel=0.20)
    assert result["ratio_ba_median"] == pytest.approx(2.0 / boost, rel=0.20)

    # The combined estimate recovers the truth, and names the bias it removed.
    assert result["ratio"] == pytest.approx(2.0, rel=0.12)
    assert result["order_effect"] == pytest.approx(boost, rel=0.20)


def test_the_combined_estimate_beats_a_pooled_median() -> None:
    """Why the geometric mean, spelled out as an assertion.

    With an odd split the pooled median lands on the majority order. This
    reconstructs that pooled number from the same data and shows it is further
    from the truth than the geometric mean.
    """

    ab, ba = [3.0, 3.0, 3.0], [1.0 / 3.0, 1.0 / 3.0]  # true ratio 1.0, k = 3
    pooled = statistics.median(ab + ba)
    combined = math.sqrt(statistics.median(ab) * statistics.median(ba))

    assert combined == pytest.approx(1.0)
    assert abs(pooled - 1.0) > abs(combined - 1.0)


def test_the_ratio_is_taken_inside_the_round() -> None:
    """One ratio per round, kept per order so neither can hide in a pool."""

    result = discipline.paired_ratio(
        lambda: None, lambda: None, rounds=4, labels=("a", "b")
    )

    assert len(result["ab_ratios"]) + len(result["ba_ratios"]) == 4


# --- the FAIL rules ----------------------------------------------------------


def test_an_empty_result_is_a_failure_not_a_timing() -> None:
    """The 102x-realtime guard: a backend that died fast is not a fast backend."""

    with pytest.raises(discipline.BenchmarkFailure):
        discipline.refuse_empty([], what="transcribe")
    with pytest.raises(discipline.BenchmarkFailure):
        discipline.refuse_empty(None, what="transcribe")


def test_repeat_discards_the_warmup_but_keeps_it_visible() -> None:
    """"How slow was the first run" is data, not noise to be swallowed."""

    seen: list[int] = []

    def work() -> object:
        seen.append(1)
        return ["result"]

    timing = discipline.repeat(
        work, warmup=2, repeats=3, label="x", aggregate="median"
    )

    assert len(seen) == 5
    assert len(timing.samples_ms) == 3
    assert len(timing.discarded_ms) == 2


def test_repeat_propagates_a_validation_failure() -> None:
    with pytest.raises(discipline.BenchmarkFailure):
        discipline.repeat(
            lambda: [],
            warmup=0,
            repeats=1,
            label="x",
            aggregate="median",
            validate=lambda result: discipline.refuse_empty(result, what="x"),
        )


def test_a_timing_with_no_samples_refuses_to_produce_a_value() -> None:
    timing = discipline.Timing(label="x", samples_ms=[], aggregate="median")
    with pytest.raises(discipline.BenchmarkFailure):
        _ = timing.value_ms


def test_rtf_never_travels_without_absolute_ms() -> None:
    timing = discipline.Timing(label="x", samples_ms=[500.0], aggregate="median")
    rendered = timing.render(clip_seconds=10.0)

    assert "500.0 ms" in rendered
    assert "RTF" in rendered


# --- scale check -------------------------------------------------------------


def test_scale_check_catches_the_fake_fast_run() -> None:
    """A 5x longer clip that produced the same output was not transcribed."""

    result = discipline.scale_check(
        [(11.0, 100.0), (55.0, 105.0)], quantity="words"
    )
    assert result["passed"] is False


def test_scale_check_passes_when_work_grows_with_the_input() -> None:
    result = discipline.scale_check(
        [(11.0, 100.0), (55.0, 500.0)], quantity="words"
    )
    assert result["passed"] is True


def test_scale_check_refuses_a_degenerate_baseline() -> None:
    with pytest.raises(discipline.BenchmarkFailure):
        discipline.scale_check([(11.0, 0.0), (55.0, 500.0)])
