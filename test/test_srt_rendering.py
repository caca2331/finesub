from __future__ import annotations

import pytest

import to_srt


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
