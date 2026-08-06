from __future__ import annotations

from tools.wt_refine_port.teacher_force_probe import (
    CapturedSegment,
    WordTiming,
    compare_word_timings,
    plan_window,
)


def test_plan_window_clamps_expansion_to_audio() -> None:
    segment = CapturedSegment(3, 0.4, 9.8, "hello", (1, 2))
    window = plan_window(segment, audio_duration=10.0, refine_sec=1.0)
    assert (window.start, window.end, window.tokens) == (0.0, 10.0, (1, 2))


def test_compare_word_timings_reports_frame_parity() -> None:
    reference = [WordTiming("a", 1.0, 1.2), WordTiming("b", 1.2, 1.5)]
    candidate = [WordTiming("a", 1.02, 1.2), WordTiming("b", 1.2, 1.52)]
    result = compare_word_timings(candidate, reference)
    assert result["same_text"] is True
    assert result["start_within_20ms"] is True
    assert result["end_within_20ms"] is True


def test_compare_word_timings_does_not_hide_text_mismatch() -> None:
    result = compare_word_timings(
        [WordTiming("candidate", 0.0, 1.0)],
        [WordTiming("reference", 0.0, 1.0)],
    )
    assert result["same_text"] is False
    assert "start_max_sec" not in result
