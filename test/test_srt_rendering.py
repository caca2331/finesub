from __future__ import annotations

import pytest

from asr_playground.subtitles import rendering as to_srt


def test_format_srt_time_clamps_negative_values() -> None:
    assert to_srt.format_srt_time(-1.2) == "00:00:00,000"


def test_format_srt_time_rounds_milliseconds_and_hours() -> None:
    assert to_srt.format_srt_time(3661.2345) == "01:01:01,234"
    assert to_srt.format_srt_time(1.9996) == "00:00:02,000"


def test_render_segment_srt_skips_invalid_segments() -> None:
    segments = [
        {"start": 0.0, "end": 1.5, "text": "hello"},
        {"start": 3.0, "end": 2.0, "text": "skip"},
        {"start": 2.0, "end": 3.0, "text": ""},
    ]

    assert to_srt.render_segment_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "hello\n"
        "\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,000\n"
        "\"\"\n"
    )


def test_render_word_srt_requires_word_timestamps() -> None:
    with pytest.raises(ValueError, match="No segments contain words"):
        to_srt.render_word_srt([{"start": 0.0, "end": 1.0, "text": "hello"}])


def test_render_word_srt_outputs_word_segments() -> None:
    segments = [
        {
            "words": [
                {"start": 0.0, "end": 0.4, "word": "hello"},
                {"start": 0.4, "end": 0.9, "word": "world"},
            ]
        }
    ]

    assert to_srt.render_word_srt(segments) == (
        "1\n"
        "00:00:00,000 --> 00:00:00,400\n"
        "hello\n"
        "\n"
        "2\n"
        "00:00:00,400 --> 00:00:00,900\n"
        "world\n"
    )


def _times(segments: list[dict]) -> list[tuple[float, float, str]]:
    return to_srt.resolve_overlaps(to_srt._timed(segments))[0]


def test_overlapping_cues_truncate_the_earlier_cue_and_keep_every_line() -> None:
    """A hallucinated run swallowing real lines must not delete them (out/ bed: 43 such cues,
    largest 27 s, all surviving `asr_stabilize` profile 0)."""

    rows = _times(
        [
            {"start": 47.5, "end": 76.6, "text": "hallucination"},
            {"start": 54.3, "end": 56.1, "text": "real line"},
            {"start": 80.0, "end": 81.0, "text": "after"},
        ]
    )

    assert rows == [
        (47.5, 54.3, "hallucination"),
        (54.3, 56.1, "real line"),
        (80.0, 81.0, "after"),
    ]


def test_cues_sharing_a_start_shift_the_later_one_instead_of_squeezing_the_earlier() -> None:
    assert _times(
        [{"start": 10.0, "end": 20.0, "text": "long"}, {"start": 10.0, "end": 12.0, "text": "short"}]
    ) == [(10.0, 12.0, "short"), (12.0, 20.0, "long")]


def test_a_cue_with_nowhere_left_to_go_is_dropped() -> None:
    assert _times(
        [{"start": 0.0, "end": 10.0, "text": "a"}, {"start": 0.0, "end": 10.0, "text": "b"}]
    ) == [(0.0, 10.0, "a")]


def test_chained_overlaps_resolve_pairwise() -> None:
    assert _times(
        [
            {"start": 0.0, "end": 10.0, "text": "a"},
            {"start": 2.0, "end": 12.0, "text": "b"},
            {"start": 4.0, "end": 14.0, "text": "c"},
        ]
    ) == [(0.0, 2.0, "a"), (2.0, 4.0, "b"), (4.0, 14.0, "c")]


def test_non_overlapping_input_renders_unchanged() -> None:
    assert to_srt.render_segment_srt(
        [{"start": 1.0, "end": 2.0, "text": "x"}, {"start": 3.0, "end": 4.0, "text": "y"}]
    ) == ("1\n00:00:01,000 --> 00:00:02,000\nx\n\n2\n00:00:03,000 --> 00:00:04,000\ny\n")
