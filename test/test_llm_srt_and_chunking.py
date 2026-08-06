from __future__ import annotations

import pytest

from llm.chunking import (
    SubtitleSegment,
    overlap_count_for_boundary,
    plan_correction_windows,
    preceding_context_for_boundary,
    render_segments_as_csv,
    split_window_in_half,
)
from asr_playground.subtitles.model import (
    parse_srt,
    validate_srt_text,
)


class TinyCounter:
    source = "tiny-test"

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 10)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(1, int(seconds))


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "第一句。"),
        SubtitleSegment("2", 1.2, 2.0, "第二句还没完"),
        SubtitleSegment("3", 2.0, 3.0, "现在结束。"),
        SubtitleSegment("4", 3.2, 4.0, "第四句。"),
        SubtitleSegment("5", 4.2, 5.0, "第五句。"),
    ]


def _sparse_segments() -> list[SubtitleSegment]:
    # 40s apart: nothing falls within the 30s overlap lookback.
    return [
        SubtitleSegment("1", 0.0, 1.0, "第一句。"),
        SubtitleSegment("2", 40.0, 41.0, "第二句还没完"),
        SubtitleSegment("3", 80.0, 81.0, "现在结束。"),
        SubtitleSegment("4", 120.0, 121.0, "第四句。"),
        SubtitleSegment("5", 160.0, 161.0, "第五句。"),
    ]


def test_overlap_count_for_boundary_is_purely_content_driven() -> None:
    dense = [
        SubtitleSegment(str(i + 1), i * 3.0, i * 3.0 + 1.5, "句。") for i in range(20)
    ]
    # Cut at index 15 (start 45.0): indices 5..14 start within the last 30s.
    assert overlap_count_for_boundary(dense, 15) == 10
    # v13: no floor — a gap boundary correctly yields zero overlap (stitching
    # must not span a >30s hole; continuity is the preceding block's job).
    assert overlap_count_for_boundary(_sparse_segments(), 3) == 0
    assert overlap_count_for_boundary(dense, 0) == 0


def test_preceding_context_is_fixed_count_and_crosses_gaps() -> None:
    dense = [
        SubtitleSegment(str(i + 1), i * 3.0, i * 3.0 + 1.5, "句。") for i in range(20)
    ]
    # Fixed lookback of 10, regardless of timing.
    assert [seg.id for seg in preceding_context_for_boundary(dense, 15)] == [
        str(i) for i in range(6, 16)
    ]
    assert preceding_context_for_boundary(dense, 0) == []
    # Deliberately no gap-stop: sparse (40s apart) segments are still taken.
    sparse = _sparse_segments()
    assert [seg.id for seg in preceding_context_for_boundary(sparse, 3)] == [
        "1",
        "2",
        "3",
    ]


def test_render_segments_as_csv_allows_negative_starts_for_preceding() -> None:
    preceding = [
        SubtitleSegment("8", 47.6, 48.9, "前文一"),
        SubtitleSegment("9", 49.4, 50.4, "前文二"),
    ]
    # Clip 0s at 60.0: preceding lines land before it -> negative local times.
    csv_text = render_segments_as_csv(
        preceding, window_start=60.0, allow_negative_start=True
    )
    assert csv_text.splitlines() == [
        "8|-12.4|1.3|0.5|前文一",
        "9|-10.6|1.0|0.0|前文二",
    ]
    # Default rendering still clamps (input CSV never shows negatives).
    clamped = render_segments_as_csv(preceding, window_start=60.0)
    assert clamped.splitlines()[0].startswith("8|0.0|")


def test_plan_correction_windows_single_window_when_budget_fits() -> None:
    windows = plan_correction_windows(
        _segments(), counter=TinyCounter(), audio_duration=30.0
    )

    assert len(windows) == 1
    window = windows[0]
    assert window.chunk_id == "0001"
    assert window.overlap_segments == []
    assert window.boundary_reason == "final_window"
    # Single window contains the global first and last segments: 60s pads,
    # clamped to [0, audio_duration].
    assert window.clip_start == 0.0
    assert window.clip_end == 30.0


def test_plan_correction_windows_even_split_with_dynamic_overlap() -> None:
    segments = [
        SubtitleSegment(
            str(i + 1), i * 3.0, i * 3.0 + 1.5, "这是很长的一句字幕文本用来撑大体积。"
        )
        for i in range(45)
    ]

    windows = plan_correction_windows(
        segments,
        planning_output_limit=600,
        counter=TinyCounter(),
        audio_duration=200.0,
    )

    assert len(windows) >= 2
    # Full coverage of every source id in order.
    covered = {seg.id for window in windows for seg in window.segments}
    assert covered == {seg.id for seg in segments}
    segment_index = {seg.id: idx for idx, seg in enumerate(segments)}
    core_counts = []
    for idx, window in enumerate(windows):
        overlap = window.overlap_segments
        if idx == 0:
            assert overlap == []
            assert window.preceding_segments == []
        else:
            # Content-driven overlap: segments within the last 30s (10 at 3s
            # spacing), physically shared with the previous window's tail.
            assert len(overlap) == 10
            prev_tail = windows[idx - 1].segments[-len(overlap) :]
            assert [seg.id for seg in overlap] == [seg.id for seg in prev_tail]
            assert window.segments[: len(overlap)] == overlap
            # Read-only preceding context: the 10 raw lines right before the
            # window start (overlap included in the window).
            start = segment_index[window.segments[0].id]
            expected = [seg.id for seg in segments[max(0, start - 10) : start]]
            assert [seg.id for seg in window.preceding_segments] == expected
        core_counts.append(len(window.segments) - len(window.overlap_segments))
        if idx < len(windows) - 1:
            assert window.boundary_reason in {
                "even_sentence_or_gap_boundary",
                "forced_even_boundary",
            }
            # Every line ends with sentence punctuation, so cuts snap cleanly.
            assert window.boundary_reason == "even_sentence_or_gap_boundary"
        else:
            assert window.boundary_reason == "final_window"
        # Budget derives from the clip duration (padding included).
        assert window.clip_end > window.clip_start
        assert window.budget.estimated_output_tokens <= 600
    # Near-even split of the 45 core segments.
    assert sum(core_counts) == 45
    assert max(core_counts) - min(core_counts) <= 6
    # Interior boundaries use 5s pads; edge windows use 60s pads (clamped to
    # [0, audio_duration]).
    assert windows[0].clip_start == 0.0
    assert windows[-1].clip_end == pytest.approx(
        min(200.0, segments[-1].end + 60.0)
    )
    middle = windows[1]
    assert middle.clip_start == pytest.approx(middle.segments[0].start - 5.0)


def test_plan_correction_windows_forces_boundary_outside_radius() -> None:
    segments = [
        SubtitleSegment(
            str(i + 1),
            i * 3.0,
            i * 3.0 + 2.5,  # 0.5s gaps: below the reasonable-boundary threshold
            "第五句有标点。" if i == 5 else "这一句还没有结束继续说下去没有标点",
        )
        for i in range(44)
    ]

    windows = plan_correction_windows(
        segments,
        planning_output_limit=800,
        counter=TinyCounter(),
        audio_duration=200.0,
    )

    assert len(windows) >= 2
    assert windows[0].boundary_reason == "forced_even_boundary"


def test_plan_correction_windows_replans_until_budget_fits() -> None:
    class HeavyAudioCounter(TinyCounter):
        # The k estimate prices audio at the fixed 32 tok/s, but validation
        # asks the counter: overprice audio so the initial k fails the input
        # limit and the planner must re-place cuts with more windows.
        def count_audio_seconds(self, seconds: float) -> int:
            return max(1, int(seconds * 1_200))

    segments = [
        SubtitleSegment(
            str(i + 1), i * 3.0, i * 3.0 + 1.5, "这是很长的一句字幕文本用来撑大体积。"
        )
        for i in range(45)
    ]

    plan_report: dict = {}
    windows = plan_correction_windows(
        segments,
        counter=HeavyAudioCounter(),
        audio_duration=200.0,
        report_sink=plan_report,
    )

    assert len(windows) >= 2
    assert all(window.budget.input_tokens <= 194_000 for window in windows)
    # The sink reports the budget-driven replanning for the task report.
    assert plan_report["planned_windows"] == len(windows)
    assert plan_report["replan_attempts"] >= 1
    assert plan_report["planned_windows"] > plan_report["estimated_windows"]
    assert plan_report["last_over_budget_error"]


def test_plan_correction_windows_rejects_oversized_single_segment() -> None:
    segments = [SubtitleSegment("1", 0.0, 5.0, "长" * 120_000)]

    with pytest.raises(ValueError, match="cannot fit"):
        plan_correction_windows(segments, counter=TinyCounter())


def _window_subtitle_tokens(window) -> int:
    """Real asr_result CSV tokens of a planned window (core + overlap)."""
    from llm.chunking import estimate_window_budget

    budget = estimate_window_budget(
        window.segments,
        audio_seconds=window.clip_end - window.clip_start,
        counter=TinyCounter(),
    )
    return budget.subtitle_input_tokens


def test_plan_correction_windows_caps_window_subtitle_tokens() -> None:
    # ~80-char lines: TinyCounter (len//10) prices each CSV row around 8-9
    # tokens, so 200 dense segments far exceed a 1k <asr_result> cap.
    segments = [
        SubtitleSegment(
            str(i + 1), i * 3.0, i * 3.0 + 1.5, "这是一段很长的字幕文本。" * 6
        )
        for i in range(200)
    ]

    windows = plan_correction_windows(
        segments,
        counter=TinyCounter(),
        audio_duration=650.0,
        max_window_subtitle_tokens=1000,
    )

    assert len(windows) >= 2
    covered = {seg.id for window in windows for seg in window.segments}
    assert covered == {seg.id for seg in segments}
    for window in windows:
        assert _window_subtitle_tokens(window) <= 1000, (
            f"window {window.chunk_id} <asr_result> exceeds the cap"
        )


def test_plan_correction_windows_cap_defaults_to_limits_field() -> None:
    from dataclasses import replace

    from llm.config import DEFAULT_LIMITS

    segments = [
        SubtitleSegment(
            str(i + 1), i * 3.0, i * 3.0 + 1.5, "这是一段很长的字幕文本。" * 6
        )
        for i in range(200)
    ]

    windows = plan_correction_windows(
        segments,
        counter=TinyCounter(),
        audio_duration=650.0,
        limits=replace(DEFAULT_LIMITS, max_window_subtitle_tokens=800),
    )

    assert len(windows) >= 2
    for window in windows:
        assert _window_subtitle_tokens(window) <= 800


def _cap_binding_segments() -> list[SubtitleSegment]:
    """Big enough that the default 10k cap binds but the output coefficient
    alone would still allow one window -- the only regime where "cap unset"
    and "cap disabled" are distinguishable."""
    return [
        SubtitleSegment(
            str(i + 1), i * 3.0, i * 3.0 + 2.5, "这是一段足够长的字幕文本用于撑大窗口。" * 12
        )
        for i in range(400)
    ]


def test_plan_correction_windows_cap_zero_disables() -> None:
    from llm.config import DEFAULT_LIMITS

    segments = _cap_binding_segments()
    default_cap = plan_correction_windows(
        segments, counter=TinyCounter(), audio_duration=1200.0
    )
    zero_cap = plan_correction_windows(
        segments,
        counter=TinyCounter(),
        audio_duration=1200.0,
        max_window_subtitle_tokens=0,
    )

    # Unset (-> limits default) and disabled must be different windowings here,
    # or this fixture cannot tell the two apart and the test proves nothing.
    assert len(zero_cap) == 1
    assert len(default_cap) > 1
    for window in default_cap:
        assert _window_subtitle_tokens(window) <= DEFAULT_LIMITS.max_window_subtitle_tokens


def test_research_context_key_separates_unset_cap_from_disabled_cap(tmp_path) -> None:
    """Unset (-> limits default) and 0 (no cap) plan different windows, so they
    must not produce the same research-context key -- otherwise a config edit
    reuses a context whose window ids no longer line up."""
    from llm.profiles import DEFAULT_PROFILE
    from llm.research import planning_metadata

    segments = _cap_binding_segments()
    unset = plan_correction_windows(
        segments, counter=TinyCounter(), audio_duration=1200.0
    )
    disabled = plan_correction_windows(
        segments,
        counter=TinyCounter(),
        audio_duration=1200.0,
        max_window_subtitle_tokens=0,
    )
    assert len(unset) != len(disabled)  # the fixture really distinguishes them

    stable = tmp_path / "stable.json"
    stable.write_text("{}", encoding="utf-8")
    common = dict(
        stable_json=stable,
        extra_info="",
        enable_web_search=False,
        search_rounds=0,
        collect_task_feedback=False,
        audio_duration=1200.0,
    )
    meta_unset = planning_metadata(
        DEFAULT_PROFILE, max_window_subtitle_tokens=None, **common
    )
    meta_disabled = planning_metadata(
        DEFAULT_PROFILE, max_window_subtitle_tokens=0, **common
    )
    assert meta_unset != meta_disabled
    assert meta_disabled["max_window_subtitle_tokens"] == 0
    assert meta_unset["max_window_subtitle_tokens"] > 0


def test_plan_correction_windows_single_segment_over_cap_raises() -> None:
    segments = [SubtitleSegment("1", 0.0, 5.0, "长" * 200_000)]

    with pytest.raises(ValueError, match="cannot fit"):
        plan_correction_windows(
            segments, counter=TinyCounter(), max_window_subtitle_tokens=100
        )


def test_split_window_in_half_prefers_sentence_boundary_and_overlaps() -> None:
    segments = _sparse_segments()
    window = plan_correction_windows(
        segments, counter=TinyCounter(), audio_duration=300.0
    )[0]
    assert len(window.segments) == 5

    halves = split_window_in_half(
        window,
        counter=TinyCounter(),
        global_first_id="1",
        global_last_id="5",
        audio_duration=200.0,
    )

    assert halves is not None
    first, second = halves
    assert first.chunk_id == f"{window.chunk_id}-a"
    assert second.chunk_id == f"{window.chunk_id}-b"
    # Split lands on the sentence boundary at segment 3 ("现在结束。").
    assert first.segments[-1].id == "3"
    assert first.boundary_reason == "split_retry_first_half"
    assert second.boundary_reason == "split_retry_second_half"
    # Sparse timing (40s apart): content-driven overlap is zero at the split —
    # the halves partition the window cleanly and still cover all of it.
    assert second.segments[0].id == "4"
    assert first.segments[0].id == window.segments[0].id
    assert second.segments[-1].id == window.segments[-1].id
    assert second.overlap_segments == []
    # Preceding context: -a inherits the parent's (empty here); -b looks back
    # across the parent's own tail.
    assert first.preceding_segments == list(window.preceding_segments)
    assert [seg.id for seg in second.preceding_segments] == ["1", "2", "3"]
    # Clip pads: the first half still contains the global first segment (60s
    # lead pad clamps to 0); the second half is interior at its start (5s pad)
    # and contains the global last segment (60s trail pad clamps to duration).
    assert first.clip_start == 0.0
    assert first.clip_end == pytest.approx(first.segments[-1].end + 5.0)
    assert second.clip_start == pytest.approx(second.segments[0].start - 5.0)
    # 161.0 + 60 = 221 exceeds the 200s audio: clamped.
    assert second.clip_end == 200.0


def test_split_window_in_half_can_recurse_and_stops_at_single_segment() -> None:
    segments = _sparse_segments()
    window = plan_correction_windows(segments, counter=TinyCounter())[0]

    halves = split_window_in_half(window, counter=TinyCounter())
    assert halves is not None
    first, _second = halves
    deeper = split_window_in_half(first, counter=TinyCounter())
    assert deeper is not None
    assert deeper[0].chunk_id == f"{first.chunk_id}-a"

    single = deeper[0]
    while len(single.segments) > 1:
        result = split_window_in_half(single, counter=TinyCounter())
        assert result is not None
        single = result[0]
    assert split_window_in_half(single, counter=TinyCounter()) is None


def test_srt_parser_accepts_dense_segments_without_blank_lines() -> None:
    dense = (
        "1\n00:00:00,000 --> 00:00:01,000\n第一句\n"
        "2\n00:00:01,000 --> 00:00:02,000\n第二句\n"
        "3\n00:00:02,000 --> 00:00:03,000\n第三句\n"
    )

    segments = parse_srt(dense)

    assert [segment.text for segment in segments] == ["第一句", "第二句", "第三句"]
    assert validate_srt_text(dense).ok


def test_srt_validation_rejects_bad_timing() -> None:
    bad = "1\n00:00:02,000 --> 00:00:01,000\n坏\n"

    result = validate_srt_text(bad)

    assert not result.ok
    assert any("end must be greater" in error for error in result.errors)


def test_srt_validation_uses_weighted_display_length() -> None:
    latin = "1\n00:00:00,000 --> 00:00:01,000\n" + "a" * 30 + "\n"
    cjk = "1\n00:00:00,000 --> 00:00:01,000\n" + "中" * 17 + "\n"

    assert not validate_srt_text(latin, max_line_chars=16).warnings
    result = validate_srt_text(cjk, max_line_chars=16)
    assert any(
        "17 weighted characters; limit is 16" in warning
        for warning in result.warnings
    )
