"""The stage-split accumulator, including the two ways it could lie.

`phase_timing` is what turned `asr_align_sec` from one opaque number into the
A3.2 breakdown, and three separate decisions now rest on it (A1's value, stage
overlap, A4's referee cost). So the guards here are about the arithmetic being
trustworthy rather than merely present:

* inclusive and exclusive must both be right, because the rescue ladder
  *contains* decodes -- exclusive alone would report rescue as nearly free,
  inclusive alone would count those decodes against two phases at once;
* a second scope must **add to** the first, because the Qwen referee can run
  twice in one stage (inline for a language redecode, then at the tail) and
  overwriting would silently discard the first run's numbers.
"""

from __future__ import annotations

import time

from finesub.speech.runtime import phase_timing


def _busy(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def test_phase_is_a_noop_when_nothing_is_collecting() -> None:
    """Not gated behind a flag, so it must cost nothing when unused."""

    assert not phase_timing.active()
    with phase_timing.phase("asr.encode"):
        pass
    assert not phase_timing.active()


def test_a_phase_records_its_calls() -> None:
    with phase_timing.collect() as stats:
        for _ in range(3):
            with phase_timing.phase("asr.encode"):
                _busy(0.002)

    assert stats["asr.encode"].calls == 3
    assert stats["asr.encode"].inclusive_s > 0
    assert stats["asr.encode"].exclusive_s > 0


def test_a_nested_phase_is_charged_to_the_child_not_the_parent() -> None:
    """The rescue-vs-decode question in miniature.

    The parent's inclusive time covers the child; its exclusive time must not,
    or "what would removing the rescue save" and "what is this run made of"
    would be the same number, and neither would be right.
    """

    with phase_timing.collect() as stats:
        with phase_timing.phase("asr.rescue_coverage"):
            _busy(0.005)
            with phase_timing.phase("asr.decode"):
                _busy(0.02)

    parent = stats["asr.rescue_coverage"]
    child = stats["asr.decode"]
    assert parent.inclusive_s >= child.inclusive_s
    assert parent.exclusive_s < parent.inclusive_s
    # The parent's own work is what is left after the child's span.
    assert parent.exclusive_s < child.exclusive_s


def test_a_second_scope_accumulates_instead_of_replacing() -> None:
    """`into=` is what keeps the referee's two runs from erasing each other."""

    with phase_timing.collect() as stats:
        with phase_timing.phase("qwen.load"):
            _busy(0.002)
    first = stats["qwen.load"].inclusive_s

    with phase_timing.collect(into=stats):
        with phase_timing.phase("qwen.load"):
            _busy(0.002)
        with phase_timing.phase("qwen.infer"):
            _busy(0.002)

    assert stats["qwen.load"].calls == 2, "the inline run must not be discarded"
    assert stats["qwen.load"].inclusive_s > first
    assert stats["qwen.infer"].calls == 1


def test_a_fresh_scope_does_not_inherit_the_previous_one() -> None:
    with phase_timing.collect() as first:
        with phase_timing.phase("asr.encode"):
            pass
    with phase_timing.collect() as second:
        pass

    assert first["asr.encode"].calls == 1
    assert second == {}


def test_an_exception_mid_phase_does_not_leave_the_collector_installed() -> None:
    class _Boom(Exception):
        pass

    try:
        with phase_timing.collect():
            with phase_timing.phase("asr.decode"):
                raise _Boom
    except _Boom:
        pass

    assert not phase_timing.active(), "a failed stage must not leak its collector"


def test_as_dict_orders_by_exclusive_time() -> None:
    with phase_timing.collect() as stats:
        with phase_timing.phase("small"):
            _busy(0.002)
        with phase_timing.phase("large"):
            _busy(0.03)

    rendered = phase_timing.as_dict(stats)
    assert list(rendered) == ["large", "small"]
    assert rendered["large"]["calls"] == 1
