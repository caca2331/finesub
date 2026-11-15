from __future__ import annotations

import json

import pytest

from finesub.speech.preprocessing.energy import render_non_speech_srt
from finesub.subtitles import rendering as to_srt
from finesub.subtitles.model import SrtSegment, format_srt_timestamp, parse_srt, render_srt, validate_srt_text
from finesub.subtitles.postprocess import postprocess_srt_text


def test_format_srt_timestamp_clamps_negative_values() -> None:
    assert format_srt_timestamp(-1.2) == "00:00:00,000"


def test_format_srt_timestamp_rounds_milliseconds_and_hours() -> None:
    assert format_srt_timestamp(3661.2345) == "01:01:01,234"
    assert format_srt_timestamp(1.9996) == "00:00:02,000"


# --- what the writer writes, `parse_srt` reads back ----------------------------------------------------------


def _round_trip(text: str) -> list[tuple[float, float, str]]:
    assert validate_srt_text(text).errors == [], text
    return [(cue.start, cue.end, cue.text) for cue in parse_srt(text)]


def test_a_line_the_punctuation_step_empties_does_not_split_the_cue() -> None:
    """A blank line ends a cue in SRT: "。\n你好" through profile 2 used to come out as a cue with no text."""
    out, _ = postprocess_srt_text(render_srt([SrtSegment(1, 0.0, 1.0, "。\n你好"), SrtSegment(2, 2.0, 3.0, "　\n。")]),
                                  profile=2)
    assert _round_trip(out) == [(0.0, 1.0, "你好"), (2.0, 3.0, '""')]


def test_all_whitespace_text_is_written_like_no_text() -> None:
    # `""`, as an empty text has always been written (whether to show it at all is another question)
    text = to_srt.render_segment_srt([{"start": 0.0, "end": 1.0, "text": "  "}, {"start": 2.0, "end": 3.0}])
    assert _round_trip(text) == [(0.0, 1.0, '""'), (2.0, 3.0, '""')]
    assert _round_trip(render_srt([SrtSegment(1, 0.0, 1.0, " \n\t")])) == [(0.0, 1.0, '""')]


def test_a_span_that_is_not_a_finite_number_is_not_written() -> None:
    nan, inf = float("nan"), float("inf")
    segments = [{"start": nan, "end": 1.0, "text": "a"}, {"start": 0.0, "end": nan, "text": "b"},
                {"start": 0.0, "end": inf, "text": "c"}, {"start": 2.0, "end": 3.0, "text": "d"}]
    assert _round_trip(to_srt.render_segment_srt(segments)) == [(2.0, 3.0, "d")]
    words = [{"words": [{"start": nan, "end": 0.5, "word": "x"}, {"start": 0.5, "end": 0.9, "word": "y"}]}]
    assert _round_trip(to_srt.render_word_srt(words)) == [(0.5, 0.9, "y")]


def test_a_span_shorter_than_a_millisecond_stays_positive_once_rounded() -> None:
    text = to_srt.render_segment_srt([{"start": 5.0001, "end": 5.0004, "text": "a"}])
    assert _round_trip(text) == [(5.0, 5.001, "a")]
    assert _round_trip(render_non_speech_srt([(1.0002, 1.0003)])) == [(1.0, 1.001, '""')]


# --- `--empty-cues`: a line with no text in the rendered SRT ---------------------------------------------------


def _convert(tmp_path, segments, **kwargs) -> list[tuple[float, float, str]]:
    stable = tmp_path / "talk-stable.json"
    stable.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    written = to_srt.convert_json_to_srt(stable, output_path=tmp_path / "talk-raw.srt", validate=False, **kwargs)
    return _round_trip(written.path.read_text(encoding="utf-8"))


EMPTY_BETWEEN = [{"start": 0.0, "end": 2.0, "text": "a"}, {"start": 1.5, "end": 3.0, "text": " "},
                 {"start": 4.0, "end": 5.0, "text": "b"}]


def test_a_line_without_text_is_left_out_by_default_and_cuts_nothing_short(tmp_path) -> None:
    # Left out before the overlaps are resolved: the line before it keeps its whole span.
    assert _convert(tmp_path, EMPTY_BETWEEN) == [(0.0, 2.0, "a"), (4.0, 5.0, "b")]
    assert _convert(tmp_path, EMPTY_BETWEEN, empty_cues=False) == [(0.0, 2.0, "a"), (4.0, 5.0, "b")]


def test_with_empty_cues_a_line_without_text_is_a_quoted_cue(tmp_path) -> None:
    assert _convert(tmp_path, EMPTY_BETWEEN, empty_cues=True) == [(0.0, 1.5, "a"), (1.5, 3.0, '""'), (4.0, 5.0, "b")]


def test_the_word_writer_leaves_out_words_without_text_unless_asked(tmp_path) -> None:
    segments = [{"start": 0.0, "end": 1.0, "text": "x",
                 "words": [{"start": 0.0, "end": 0.5, "word": "x"}, {"start": 0.5, "end": 1.0, "word": ""}]}]
    assert _convert(tmp_path, segments, word=True) == [(0.0, 0.5, "x")]
    assert _convert(tmp_path, segments, word=True, empty_cues=True) == [(0.0, 0.5, "x"), (0.5, 1.0, '""')]


def test_config_toml_turns_empty_cues_on(tmp_path, monkeypatch) -> None:
    from finesub import config as app_config

    config = tmp_path / "config.toml"
    config.write_text("[postprocess]\nempty_cues = true\n", encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config))
    app_config.clear_config_cache()
    assert len(_convert(tmp_path, EMPTY_BETWEEN)) == 3
    assert len(_convert(tmp_path, EMPTY_BETWEEN, empty_cues=False)) == 2  # the flag wins


def test_an_inverted_span_is_left_for_the_validator_to_name() -> None:
    # not a rounding artefact: an upstream defect, written as it is
    text = render_srt([SrtSegment(1, 2.0, 1.0, "a")])
    assert "Segment 1 end must be greater than start." in validate_srt_text(text).errors


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
    return [row[:3] for row in to_srt.resolve_overlaps(to_srt._timed(segments))[0]]


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


def test_only_a_line_on_the_overlap_lane_stays_over_the_next_one() -> None:
    """A collapsed row stretched over the next one (docs/asr-stabilize.md, "坍缩行") is shown with it; any other
    local overlap is still a defect to trim."""

    rows = _times(
        [
            {"start": 10.0, "end": 12.0, "text": "stretched", "overlap_lane": True},
            {"start": 10.5, "end": 11.0, "text": "backchannel"},
            {"start": 11.2, "end": 14.0, "text": "next"},
            {"start": 13.0, "end": 15.0, "text": "whisper overlap"},
        ]
    )

    assert rows == [
        (10.0, 12.0, "stretched"),
        (10.5, 11.0, "backchannel"),
        (11.2, 13.0, "next"),
        (13.0, 15.0, "whisper overlap"),
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


def test_a_cue_nested_in_the_gap_a_shifted_cue_jumped_is_kept() -> None:
    """A long cue sharing its start with a short one is shifted past it; a cue starting inside that short one
    still has room before the shifted cue's new start, and used to be resolved against it and dropped."""

    assert _times(
        [
            {"start": 0.0, "end": 3.0, "text": "B"},
            {"start": 0.0, "end": 10.0, "text": "A"},
            {"start": 1.0, "end": 4.0, "text": "C"},
        ]
    ) == [(0.0, 1.0, "B"), (1.0, 3.0, "C"), (3.0, 10.0, "A")]
