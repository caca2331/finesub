from __future__ import annotations

import pytest

from asr_playground.subtitles.postprocess import postprocess_srt_text
from asr_playground.subtitles.model import parse_srt


def _segment_times(text: str) -> list[tuple[float, float]]:
    return [(segment.start, segment.end) for segment in parse_srt(text)]


def test_profile_one_extends_every_end_by_fixed_pad() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:00,300\n中\n\n"
        "2\n00:00:02,000 --> 00:00:02,300\n中中\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\n较长原轴\n\n"
        "4\n00:00:06,000 --> 00:00:06,200\n贴紧\n\n"
        "5\n00:00:06,300 --> 00:00:07,000\n下一条\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1)

    assert _segment_times(rendered) == [
        (0.0, 0.6),
        (2.0, 2.6),
        (4.0, 5.3),
        (6.0, 6.3),
        (6.3, 7.3),
    ]
    assert report.applied_profiles == (1,)
    assert report.duration_extended == 5
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 0


def test_profile_one_caps_extension_at_next_start_and_closes_flash_gap() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:00,300\n贴紧下一条\n\n"
        "2\n00:00:00,500 --> 00:00:02,000\n先延后闪轴\n\n"
        "3\n00:00:02,400 --> 00:00:03,450\n闭合闪轴\n"
    )

    rendered, report = postprocess_srt_text(source, profile=1)
    segments = parse_srt(rendered)

    # Gap to next is 0.2s < 0.3s pad → capped at next start (no leftover flash).
    assert segments[0].end == 0.5
    # +0.3s pad leaves a 0.1s gap (< 0.3s flash threshold) → closed to next start.
    assert segments[1].end == 2.4
    assert report.duration_extended == 3
    assert report.flash_extended == 1


def test_profile_two_cleans_text_without_changing_timeline() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "  你好，世界。  \n\n"
        "2\n"
        "00:00:01,100 --> 00:00:01,600\n"
        "第　二句\n"
    )

    rendered, report = postprocess_srt_text(source, profile=2)
    segments = parse_srt(rendered)

    assert _segment_times(rendered) == [(0.0, 0.5), (1.1, 1.6)]
    assert [segment.text for segment in segments] == ["你好 世界", "第 二句"]
    assert report.applied_profiles == (2,)
    assert report.duration_extended == 0
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 3
    assert report.trimmed_lines == 1


def test_profile_zero_is_profile_one_then_profile_two() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "  你好，世界。  \n\n"
        "2\n"
        "00:00:01,100 --> 00:00:01,600\n"
        "第　二句\n"
    )

    combined, report = postprocess_srt_text(source, profile=0)
    duration_only, _ = postprocess_srt_text(source, profile=1)
    composed, _ = postprocess_srt_text(duration_only, profile=2)

    assert combined == composed
    assert report.applied_profiles == (3, 4, 1, 2)
    # Seg1: +0.3 → 0.8; leftover gap to 1.1 is 0.3 (not < 0.3) so no flash.
    # Seg2: +0.3 → 1.9.
    assert report.duration_extended == 2
    assert report.flash_extended == 0
    assert report.punctuation_replacements == 3
    assert report.trimmed_lines == 1


def test_profile_minus_one_is_noop_render() -> None:
    source = (
        "1\n"
        "00:00:00,000 --> 00:00:00,500\n"
        "你好，世界。\n"
    )

    rendered, report = postprocess_srt_text(source, profile=-1)

    segment = parse_srt(rendered)[0]
    assert segment.end == 0.5
    assert segment.text == "你好，世界。"
    assert report.profile == -1
    assert report.applied_profiles == ()


def test_unsupported_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected one of -1, 0, 1, 2, 3, 4"):
        postprocess_srt_text(
            "1\n00:00:00,000 --> 00:00:00,500\n字幕\n",
            profile=5,
        )


def test_profile_four_trims_overlaps_and_warns(capsys) -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:02,000\n压到下一句\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n被压\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\n正常贴紧\n"
    )

    rendered, report = postprocess_srt_text(source, profile=4)

    assert _segment_times(rendered) == [(0.0, 1.5), (1.5, 3.0), (3.0, 4.0)]
    assert report.applied_profiles == (4,)
    assert report.overlaps_fixed == 1
    assert report.duration_extended == 0
    warning = capsys.readouterr().err
    assert "1 overlapping subtitle cue(s)" in warning
    assert "00:00:02,000" in warning


def test_profile_four_is_silent_and_idempotent_without_overlap(capsys) -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:01,000\n甲\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n乙\n"
    )

    rendered, report = postprocess_srt_text(source, profile=4)

    assert _segment_times(rendered) == [(0.0, 1.0), (1.0, 2.0)]
    assert report.overlaps_fixed == 0
    assert capsys.readouterr().err == ""


def test_profile_four_collapses_a_cue_starting_after_the_next_one() -> None:
    source = (
        "1\n00:00:05,000 --> 00:00:06,000\n乱序\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n更早\n"
    )

    rendered, report = postprocess_srt_text(source, profile=4)

    # The end can never precede its own start, so the cue collapses instead.
    assert _segment_times(rendered) == [(5.0, 5.0), (1.0, 2.0)]
    assert report.overlaps_fixed == 1


def test_profile_zero_resolves_overlap_before_extending_ends() -> None:
    source = (
        "1\n00:00:00,000 --> 00:00:02,000\n压到下一句\n\n"
        "2\n00:00:01,500 --> 00:00:03,000\n被压\n"
    )

    rendered, report = postprocess_srt_text(source, profile=0)

    # Overlap first (end 2.0 -> 1.5), then the +0.3s pad capped at the next
    # start leaves it there; the last cue keeps the full pad.
    assert _segment_times(rendered) == [(0.0, 1.5), (1.5, 3.3)]
    assert report.applied_profiles == (3, 4, 1, 2)
    assert report.overlaps_fixed == 1
    # Only cue 2 could actually move: cue 1 is already capped at the next start.
    assert report.duration_extended == 1
