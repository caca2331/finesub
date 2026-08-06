from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from asr_playground.speech.recognition import transcribe as asr_align
from asr_playground.speech.recognition import checkpoint as checkpoint_store
from asr_playground.speech.recognition import segments as recognition_segments
from asr_playground.text import (
    cleanup_asr_words_for_fallback,
    collapse_repeating_pattern,
    collapse_repeating_segment_words,
    count_word_units,
    detect_abnormal_asr_words,
    detect_repeating_group_cycle,
)


def test_normalize_vad_segments_filters_and_clamps_ranges() -> None:
    raw = [
        {"start": -1.0, "end": 1.5, "conf": "0.2"},
        {"start": 2.0, "end": 2.0},
        {"start": "bad", "end": 3.0},
        {"start": 4.0, "end": 10.0},
    ]

    assert asr_align.normalize_vad_segments(raw, 5.0) == [
        {"start": 0.0, "end": 1.5},
        {"start": 4.0, "end": 5.0},
    ]


def test_combined_audio_keeps_real_gap_audio_and_pads_long_gaps() -> None:
    sr = 10
    group = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.1, "end": 2.1},
        {"start": 3.0, "end": 4.0},
        {"start": 6.0, "end": 7.0},
    ]
    audio = np.ones(70, dtype=np.float32)

    combined, offsets = asr_align.build_combined_audio(audio, sr, group, 0.3)

    # Inserted per gap: real audio min(gap, 0.7) + adaptive silence
    # min(0.1 + 0.2*gap, 0.8) -> 0.1+0.12 / 0.7+0.28 / 0.7+0.5; the
    # group-tail real pad is empty (audio ends at 7.0s) but the 0.3s tail
    # silence is still appended.
    assert len(combined) / sr == pytest.approx(6.7)
    assert [offset[3] for offset in offsets] == pytest.approx([0.0, 1.2, 3.2, 5.4])
    # The duration estimate covers intervals + gaps (tail pad excluded).
    assert asr_align.combined_group_duration(group, gap_sec=0.3) == pytest.approx(6.4)
    # Gap silences: 1 sample after gap 1, 3 after gap 2, 5 after gap 3,
    # plus the 3-sample tail silence (12 zero samples total).
    assert combined[11:12] == pytest.approx(np.zeros(1, dtype=np.float32))
    assert combined[29:32] == pytest.approx(np.zeros(3, dtype=np.float32))
    assert combined[49:54] == pytest.approx(np.zeros(5, dtype=np.float32))
    assert float(combined.sum()) == pytest.approx(55.0)


def test_combined_audio_pads_group_tail_with_real_audio_and_silence() -> None:
    sr = 10
    group = [{"start": 0.0, "end": 1.0}]
    audio = np.ones(25, dtype=np.float32)

    combined, offsets = asr_align.build_combined_audio(audio, sr, group, 0.3)

    # Tail: 0.7s real audio (default limit) + 0.3s silence.
    assert len(combined) / sr == pytest.approx(2.0)
    assert offsets == [(0, 0.0, 1.0, 0.0, 1.0)]
    assert combined[17:20] == pytest.approx(np.zeros(3, dtype=np.float32))
    assert float(combined.sum()) == pytest.approx(17.0)


def test_combined_audio_group_tail_capped_by_next_interval_gap() -> None:
    sr = 10
    group = [{"start": 0.0, "end": 1.0}]
    audio = np.ones(25, dtype=np.float32)

    combined, _offsets = asr_align.build_combined_audio(
        audio, sr, group, 0.3, tail_real_limit_sec=0.2
    )

    # Tail: 0.2s real audio (capped by the gap to the next interval) + 0.3s.
    assert len(combined) / sr == pytest.approx(1.5)
    assert float(combined.sum()) == pytest.approx(12.0)


@pytest.mark.parametrize(
    ("original_gap", "expected"),
    # real min(gap, 0.7) + silence min(0.1 + 0.2*gap, 0.8), negative gap
    # clamped to zero; the 0.8 silence cap engages from gap 3.5s.
    [(-0.2, 0.1), (0.1, 0.22), (0.7, 0.94), (1.0, 1.0), (2.0, 1.2), (4.0, 1.5)],
)
def test_inserted_gap_duration_keeps_real_tail_plus_silence(
    original_gap: float,
    expected: float,
) -> None:
    left = {"start": 0.0, "end": 1.0}
    right = {"start": 1.0 + original_gap, "end": 2.0 + original_gap}

    assert asr_align.synthetic_gap_seconds(left, right) == pytest.approx(expected)


def test_combined_time_maps_long_gap_real_audio_one_to_one() -> None:
    # Original gap 5s: inserted region = 0.7s real audio + 0.3s silence.
    offsets = [(0, 0.0, 1.0, 0.0, 1.0), (1, 6.0, 7.0, 2.0, 3.0)]

    assert asr_align._combined_time_to_original(1.5, offsets) == pytest.approx(1.5)
    assert asr_align._combined_time_to_original(1.7, offsets) == pytest.approx(1.7)
    assert asr_align._combined_time_to_original(1.85, offsets) == pytest.approx(3.85)
    assert asr_align._combined_time_to_original(2.5, offsets) == pytest.approx(6.5)


def test_clamp_segment_overlaps_pulls_back_extended_tail() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 2.4,
            "words": [{"start": 0.0, "end": 2.4, "word": "a", "space_before": False}],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "words": [{"start": 2.0, "end": 3.0, "word": "b", "space_before": False}],
        },
    ]

    out = recognition_segments.clamp_segment_overlaps(segments)

    assert out[0]["end"] == pytest.approx(2.0)
    assert out[0]["words"][0]["end"] == pytest.approx(2.0)
    assert segments[0]["end"] == pytest.approx(2.4)
    assert out[1] == segments[1]


def test_clamp_merges_orphaned_words_into_last_surviving_word() -> None:
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "研究 リソース",
            "words": [
                {"start": 0.0, "end": 1.0, "word": "研究", "space_before": False,
                 "confidence": 0.9},
                {"start": 1.4, "end": 3.0, "word": "リソース", "space_before": True,
                 "confidence": 0.4},
            ],
        },
        {
            "start": 1.2,
            "end": 2.0,
            "words": [{"start": 1.2, "end": 2.0, "word": "え", "space_before": False}],
        },
    ]

    out = recognition_segments.clamp_segment_overlaps(segments)

    # The word beyond the clamped end merges into the last surviving word
    # instead of remaining as a zero-duration leftover.
    assert out[0]["end"] == pytest.approx(1.2)
    assert [w["word"] for w in out[0]["words"]] == ["研究 リソース"]
    assert out[0]["words"][0]["start"] == pytest.approx(0.0)
    assert out[0]["words"][0]["end"] == pytest.approx(1.0)
    assert out[0]["words"][0]["confidence"] == pytest.approx(0.4)


def test_clamp_collapsed_segment_merges_as_prefix_and_is_dropped() -> None:
    segments = [
        {
            "start": 1.0,
            "end": 1.4,
            "text": "お",
            "words": [
                {"start": 1.0, "end": 1.4, "word": "お", "space_before": False,
                 "confidence": 0.3},
            ],
        },
        {
            "start": 1.0,
            "end": 2.0,
            "text": "はよう",
            "words": [
                {"start": 1.0, "end": 2.0, "word": "はよう", "space_before": False,
                 "confidence": 0.8},
            ],
        },
    ]

    out = recognition_segments.clamp_segment_overlaps(segments)

    # Fully collapsed segment: its text prefixes the next segment's first
    # word and the empty leftover segment is dropped.
    assert len(out) == 1
    assert out[0]["words"][0]["word"] == "おはよう"
    assert out[0]["text"] == "おはよう"
    assert out[0]["words"][0]["start"] == pytest.approx(1.0)
    assert out[0]["words"][0]["end"] == pytest.approx(2.0)
    assert out[0]["words"][0]["confidence"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "char,expected",
    [
        ("。", "sentence"), ("．", "sentence"), ("｡", "sentence"),
        (".", "sentence"), ("！", "sentence"), ("?", "sentence"),
        ("…", "sentence"), ("‼", "sentence"),
        ("、", "clause"), ("，", "clause"), (",", "clause"),
        ("；", "clause"), ("：", "clause"), ("､", "clause"),
        ("「", "opening"), ("『", "opening"), ("（", "opening"),
        ("(", "opening"), ("【", "opening"), ("《", "opening"),
        ("」", "closing"), ("』", "closing"), ("）", "closing"),
        (")", "closing"), ("】", "closing"), ("》", "closing"),
        ("・", "other"), ("—", "other"), ("‘", "opening"),
        ("’", "closing"), ("‥", "sentence"),
        ("あ", "none"), ("A", "none"), ("3", "none"), (" ", "none"),
        ("", "none"),
    ],
)
def test_punct_class_covers_direction_semantics(char: str, expected: str) -> None:
    from asr_playground.text import punct_class

    assert punct_class(char) == expected


def test_zero_length_segment_gets_minimal_duration() -> None:
    segments = [
        {
            "start": 1.0,
            "end": 1.0,
            "text": "お",
            "words": [{"start": 1.0, "end": 1.0, "word": "お", "space_before": False}],
        },
        {
            "start": 2.0,
            "end": 3.0,
            "words": [{"start": 2.0, "end": 3.0, "word": "次", "space_before": False}],
        },
    ]

    out = recognition_segments.extend_zero_length_segments(segments)

    assert out[0]["end"] == pytest.approx(1.01)
    assert out[0]["words"][0]["end"] == pytest.approx(1.01)
    # far-away neighbor untouched
    assert out[1] == segments[1]
    # input not mutated
    assert segments[0]["end"] == pytest.approx(1.0)


def test_zero_length_extension_squeezes_next_segment_start() -> None:
    segments = [
        {
            "start": 1.0,
            "end": 1.0,
            "text": "お",
            "words": [{"start": 1.0, "end": 1.0, "word": "お", "space_before": False}],
        },
        {
            "start": 1.0,
            "end": 1.0,
            "text": "は",
            "words": [{"start": 1.0, "end": 1.0, "word": "は", "space_before": False}],
        },
        {
            "start": 1.005,
            "end": 2.0,
            "words": [{"start": 1.005, "end": 2.0, "word": "次", "space_before": False}],
        },
    ]

    out = recognition_segments.extend_zero_length_segments(segments)

    # chain of coincident zero-length segments resolves sequentially
    assert (out[0]["start"], out[0]["end"]) == (pytest.approx(1.0), pytest.approx(1.01))
    assert (out[1]["start"], out[1]["end"]) == (pytest.approx(1.01), pytest.approx(1.02))
    assert out[1]["words"][0]["start"] == pytest.approx(1.01)
    assert out[1]["words"][0]["end"] == pytest.approx(1.02)
    # squeezed real segment: start (and word start) pushed later, end kept
    assert out[2]["start"] == pytest.approx(1.02)
    assert out[2]["words"][0]["start"] == pytest.approx(1.02)
    assert out[2]["end"] == pytest.approx(2.0)


def test_repeating_token_is_collapsed() -> None:
    assert collapse_repeating_pattern("hahahahahaha", detect_more_than=4, keep_repeats=2) == (
        "haha",
        True,
    )


def test_segment_repeat_cleanup_includes_punctuation_and_merges_touched_words() -> None:
    words = [
        {
            "start": 1.0,
            "end": 1.5,
            "word": "え、" * 4,
            "space_before": False,
            "confidence": 0.8,
        },
        {
            "start": 1.5,
            "end": 2.0,
            "word": "え、" * 6 + "え",
            "space_before": False,
            "confidence": 0.6,
        },
    ]

    assert collapse_repeating_segment_words(words) == [
        {
            "start": 1.0,
            "end": 2.0,
            "word": "え、" * 5 + "え",
            "space_before": False,
            "confidence": 0.6,
        }
    ]


def test_segment_repeat_cleanup_does_not_ignore_different_punctuation() -> None:
    text = "え、え。え！え？え；え：え…え・"
    words = [{"start": 0.0, "end": 1.0, "word": text, "space_before": False}]

    assert collapse_repeating_segment_words(words) == words


def test_segment_repeat_cleanup_recursively_handles_distinct_regions() -> None:
    words = [
        {"start": 0.0, "end": 1.0, "word": "ha" * 9, "space_before": False},
        {"start": 1.0, "end": 2.0, "word": "middle", "space_before": True},
        {"start": 2.0, "end": 3.0, "word": "yo!" * 8, "space_before": True},
    ]

    cleaned = cleanup_asr_words_for_fallback(
        words,
        segment_start=0.0,
        segment_end=3.0,
    )

    assert [word["word"] for word in cleaned] == ["ha" * 5, "middle", "yo!" * 5]
    assert [word["space_before"] for word in cleaned] == [False, True, True]


def test_asr_confidence_fields_survive_mapping_and_finalization() -> None:
    group = [{"start": 10.0, "end": 12.0, "conf": 0.8}]
    result = {
        "segments": [
            {
                "text": " hello world",
                "confidence": 0.875,
                "no_speech_prob": 0.125,
                "alignment_events": [
                    {
                        "type": "disfluency_candidate",
                        "original_start": 0.2,
                        "refined_start": 0.4,
                    },
                    {"type": "unfinished", "token_count": 12},
                ],
                "words": [
                    {"text": "hello", "start": 0.2, "end": 0.8, "confidence": 0.75},
                    {"text": "world", "start": 0.8, "end": 1.5, "confidence": 0.625},
                ],
            }
        ]
    }

    words, asr_segments = asr_align._map_asr_result_to_intervals(
        result,
        group,
        [(0, 10.0, 12.0, 0.0, 2.0)],
    )

    finalized = asr_align._finalize_group_candidate(
        group,
        words,
        asr_segments,
        np.zeros(0, dtype=np.float32),
        16000,
        lang="en",
    )

    assert len(finalized) == 1
    assert finalized[0]["confidence"] == pytest.approx(0.875)
    assert finalized[0]["no_speech_prob"] == pytest.approx(0.125)
    assert "vad_conf" not in finalized[0]
    assert "conf" not in finalized[0]
    assert finalized[0]["alignment_events"] == [
        {
            "type": "disfluency_candidate",
            "original_start": pytest.approx(10.2),
            "refined_start": pytest.approx(10.4),
        },
        {"type": "unfinished", "token_count": 12},
    ]
    assert [word["confidence"] for word in finalized[0]["words"]] == pytest.approx(
        [0.75, 0.625]
    )


def test_missing_asr_confidence_fields_remain_optional() -> None:
    group = [{"start": 0.0, "end": 1.0}]
    result = {
        "segments": [
            {
                "text": "hello",
                "words": [{"text": "hello", "start": 0.0, "end": 0.5}],
            }
        ]
    }

    words, asr_segments = asr_align._map_asr_result_to_intervals(
        result,
        group,
        [(0, 0.0, 1.0, 0.0, 1.0)],
    )
    finalized = asr_align._finalize_group_candidate(
        group,
        words,
        asr_segments,
        np.zeros(0, dtype=np.float32),
        16000,
        lang="en",
    )

    assert "no_speech_prob" not in finalized[0]
    assert "confidence" not in finalized[0]
    assert "confidence" not in finalized[0]["words"][0]


def test_round_floats_keeps_no_speech_prob_precision() -> None:
    rounded = asr_align.round_floats(
        {
            "start": 1.23456,
            "no_speech_prob": 0.0004321,
            "words": [{"confidence": 0.87654, "no_speech_prob": 0.0001234}],
        }
    )

    assert rounded["start"] == pytest.approx(1.235)
    assert rounded["no_speech_prob"] == pytest.approx(0.000432)
    assert rounded["words"][0]["confidence"] == pytest.approx(0.877)
    assert rounded["words"][0]["no_speech_prob"] == pytest.approx(0.000123)


def test_transcribe_kwargs_keep_decoding_on_the_one_pass_trace() -> None:
    """Beam search and temperature fallback both leave the one-pass trace."""

    kwargs = asr_align._build_transcribe_kwargs(language="en")

    assert kwargs["beam_size"] is None
    assert kwargs["best_of"] is None
    assert kwargs["temperature"] == 0.0
    assert kwargs["language"] == "en"


def test_refine_backend_sets_checkpoint_feature_defaults() -> None:
    class Model:
        def __init__(self) -> None:
            self.options = None

        def transcribe_wt(self, audio, **options):
            self.options = options
            return {"segments": [], "language": "en"}

    model = Model()
    asr_align._transcribe_group_candidate(
        model,
        [{"start": 0.0, "end": 1.0}],
        np.zeros(16000, dtype=np.float32),
        16000,
        0.3,
        language="en",
    )

    assert model.options["detect_disfluencies"] is True
    assert model.options["collect_refine_signals"] is True
    assert model.options["collect_attention_signals"] is False


def test_asr_metadata_records_refine_precision() -> None:
    metadata = asr_align.asr_align_metadata(
        model="large-v3-turbo",
        device="cuda",
        language="ja",
        gap_sec=0.3,
    )

    assert metadata["refine_sec"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", 6.0),
        ("字幕测试", 4.0),
        ("abc日本123", 6.0),
        ("abc123", 3.0),
        ("123456", 1.0),
        ("don't-stop", 3.0),
    ],
)
def test_count_word_units_handles_latin_and_cjk_text(text: str, expected: float) -> None:
    assert count_word_units(text) == pytest.approx(expected)


def test_group_cycle_detects_local_repeat_at_thirty_two_units() -> None:
    motif = "天地玄黄宇宙洪荒"

    issue = detect_repeating_group_cycle(f"正常前缀{motif * 4}正常后缀")

    assert issue is not None
    assert "count=4" in issue
    assert "motif_units=8" in issue
    assert "span_units=32" in issue


def test_group_cycle_ignores_repeat_below_thirty_two_units() -> None:
    assert detect_repeating_group_cycle("天地玄黄宇宙洪" * 4) is None


def test_repeat_detection_caps_pattern_length_at_100_chars() -> None:
    capped = "".join(chr(0x4E00 + i) for i in range(100))
    over = "".join(chr(0x4E00 + i) for i in range(101))

    assert collapse_repeating_pattern(capped * 8) == (capped * 5, True)
    assert collapse_repeating_pattern(over * 8) == (over * 8, False)

    assert detect_repeating_group_cycle(capped * 4) is not None
    assert detect_repeating_group_cycle(over * 4) is None

    words = [{"start": 0.0, "end": 1.0, "word": over * 8, "space_before": False}]
    assert collapse_repeating_segment_words(words) == words


def test_group_cycle_spans_multiple_interval_word_lists() -> None:
    motif = "天地玄黄宇宙洪荒"
    per_interval_words = [
        [{"start": float(i), "end": float(i) + 0.5, "word": motif}]
        for i in range(4)
    ]

    issues = detect_abnormal_asr_words(per_interval_words)

    assert any(issue.startswith("repeating_group_cycle ") for issue in issues)


def test_whisper_segment_stays_whole_across_interval_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    motif = "天地玄黄宇宙洪荒"
    result = {
        "language": "ja",
        "segments": [
            {
                "text": f"前{motif * 4}後",
                "words": [
                    {"text": "前", "start": 0.1, "end": 0.2},
                    {"text": motif * 4, "start": 1.2, "end": 1.8},
                    {"text": "後", "start": 2.1, "end": 2.3},
                ],
            }
        ],
    }
    fake_whisper = types.SimpleNamespace(transcribe=lambda *_args, **_kwargs: result)
    monkeypatch.setitem(sys.modules, "whisper_timestamped", fake_whisper)

    words, segments, lang, issues, uses_auto = (
        asr_align._transcribe_group_candidate(
            _wt_model(lambda *_args, **_kwargs: result),
            [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}],
            np.zeros(30, dtype=np.float32),
            10,
            0.3,
            language=None,
        )
    )

    assert lang == "ja"
    assert uses_auto is True
    # The whisper segment is kept whole in its dominant interval: the gap
    # word keeps real (proportionally mapped) coordinates instead of being
    # merged into a neighbor.
    dominant_words = words[0]
    assert [w["word"] for w in dominant_words] == ["前", motif * 4, "後"]
    gap_word = dominant_words[1]
    assert 1.0 <= gap_word["start"] <= gap_word["end"] <= 2.0
    assert segments[1] == []
    assert any(issue.startswith("repeating_group_cycle ") for issue in issues)


def test_interval_fallback_applies_short_group_language_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcribe_languages: list[str | None] = []
    abnormal_result = {
        "language": "en",
        "segments": [
            {
                "text": "え" * 32,
                "words": [{"text": "え" * 32, "start": 0.1, "end": 0.5}],
            }
        ],
    }
    results = iter(
        [
            abnormal_result,
            {
                "language": "ja",
                "segments": [
                    {
                        "text": "正常",
                        "words": [{"text": "正常", "start": 0.1, "end": 0.5}],
                    }
                ],
            },
            {
                "language": "en",
                "segments": [
                    {
                        "text": "正常",
                        "words": [{"text": "正常", "start": 0.1, "end": 0.5}],
                    }
                ],
            },
        ]
    )

    def fake_transcribe(_model, _audio, **kwargs):
        transcribe_languages.append(kwargs.get("language"))
        return next(results)

    def fake_finalize(group, *_args, lang: str, **_kwargs):
        return [
            {
                "start": float(group[0]["start"]),
                "end": float(group[-1]["end"]),
                "lang": lang,
            }
        ]

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", fake_finalize)
    history: list[str] = []

    # align_group 只处理到第一个隔离区间为止，剩余交还调用方；这里要看的是
    # 剩余窗口复用历史语言，所以用完整消化的包装。
    aligned = asr_align._align_group_consume_all(
        _wt_model(fake_transcribe),
        [{"start": 0.0, "end": 6.0}, {"start": 7.0, "end": 13.0}],
        np.zeros(130, dtype=np.float32),
        10,
        0.3,
        language=None,
        auto_language_history=history,
    )

    # initial (auto) -> isolated interval (auto, no history yet) -> remainder
    # window, which is short enough to reuse the 'ja' the isolated decode wrote.
    assert transcribe_languages == [None, None, "ja"]
    assert [segment["lang"] for segment in aligned] == ["ja", "ja"]
    assert history == ["ja"]


def test_asr_metadata_records_updated_long_word_unit_threshold() -> None:
    metadata = asr_align.asr_align_metadata(
        model="large-v3-turbo",
        device="cuda",
        language="ja",
        gap_sec=0.3,
    )

    assert metadata["long_word_words"] == 15


def test_long_word_token_uses_fifteen_unit_boundary() -> None:
    def issues(text: str) -> list[str]:
        return detect_abnormal_asr_words(
            [[{"start": 0.0, "end": 1.0, "word": text}]]
        )

    below_threshold = issues("一二三四五六七八九十天地玄黄")
    at_threshold = issues("一二三四五六七八九十天地玄黄宇")

    assert not any(issue.startswith("long_word_token=") for issue in below_threshold)
    assert any(issue.startswith("long_word_token=units(15)") for issue in at_threshold)


def test_short_auto_language_group_reuses_recent_mode() -> None:
    group = [{"start": 0.0, "end": 10.0}]
    history = ["en", "ja", "ja", "en", "ja", "en", "ja", "ja", "en", "ja"]

    selected, remains_auto = asr_align._language_for_group(
        None,
        group,
        gap_sec=0.3,
        auto_language_history=history,
    )

    assert selected == "ja"
    assert remains_auto is False


def test_long_group_keeps_auto_language_detection() -> None:
    selected, remains_auto = asr_align._language_for_group(
        None,
        [{"start": 0.0, "end": 10.001}],
        gap_sec=0.3,
        auto_language_history=["ja"] * 10,
    )

    assert selected is None
    assert remains_auto is True


def test_auto_language_mode_tie_prefers_most_recent_language() -> None:
    history = ["ja", "en"] * 5

    assert asr_align._most_frequent_recent_language(history) == "en"


def test_auto_language_recording_stages_entries_until_group_boundary() -> None:
    history = ["ja"] * 9
    segments = [{"lang": "en"}, {"lang": "en"}, {"lang": "None"}]

    asr_align._record_auto_detected_segment_languages(history, segments)

    # Staged, not trimmed: the group boundary decides what survives.
    assert history == ["ja"] * 9 + ["en", "en"]


def test_group_language_collapses_to_single_entry() -> None:
    history = ["ja"] * 9
    mark = len(history)
    # One hallucination-heavy group emitting far more segments than the window.
    asr_align._record_auto_detected_segment_languages(
        history, [{"lang": "ko"} for _ in range(22)]
    )

    asr_align._collapse_group_language_entries(history, mark)

    assert history == ["ja"] * 9 + ["ko"]
    # The window still speaks ja, so short groups keep reusing ja.
    assert asr_align._most_frequent_recent_language(history) == "ja"


def test_group_language_collapse_keeps_window_at_ten_groups() -> None:
    history = ["ja"] * asr_align.AUTO_LANGUAGE_HISTORY_GROUPS
    mark = len(history)
    asr_align._record_auto_detected_segment_languages(history, [{"lang": "ko"}])

    asr_align._collapse_group_language_entries(history, mark)

    assert len(history) == asr_align.AUTO_LANGUAGE_HISTORY_GROUPS
    assert history == ["ja"] * 9 + ["ko"]


def test_group_language_collapse_uses_group_majority() -> None:
    history: list[str] = []
    asr_align._record_auto_detected_segment_languages(
        history, [{"lang": "ja"}, {"lang": "ko"}, {"lang": "ja"}, {"lang": "None"}]
    )

    asr_align._collapse_group_language_entries(history, 0)

    assert history == ["ja"]


def test_group_language_collapse_records_nothing_without_detection() -> None:
    history = ["ja"]

    asr_align._collapse_group_language_entries(history, 1)

    assert history == ["ja"]


def test_group_boundary_collapses_only_under_auto_detection(monkeypatch) -> None:
    def fake_decode(*args, **kwargs):
        history = kwargs.get("auto_language_history")
        if history is not None:
            history.extend(["ko"] * 22)
        return [], []

    monkeypatch.setattr(asr_align, "align_group", fake_decode)
    monkeypatch.setattr(asr_align, "_rescue_low_coverage", lambda *a, **k: [])
    group = [{"start": 0.0, "end": 1.0}]

    auto_history: list[str] = []
    asr_align._align_intervals_group(
        group, None, 16000, model=object(), gap_sec=0.3,
        language=None, auto_language_history=auto_history,
    )
    assert auto_history == ["ko"]

    configured_history: list[str] = []
    asr_align._align_intervals_group(
        group, None, 16000, model=object(), gap_sec=0.3,
        language="ja", auto_language_history=configured_history,
    )
    # Untouched: with an explicit language the history is never consulted, and
    # the real decode paths record nothing (only this fake does).
    assert configured_history == ["ko"] * 22


def _wt_model(transcribe):
    """A model shaped the way the recognition core calls it.

    The backend owns the call now (``model.transcribe_wt(audio, **kwargs)``),
    so tests hand in a model instead of patching a whisper module. Fakes keep
    their ``(model, audio, **kwargs)`` signature; the model slot goes unused.
    """

    return types.SimpleNamespace(
        transcribe_wt=lambda audio, **kwargs: transcribe(None, audio, **kwargs)
    )


def _checkpoint_intervals() -> list[dict[str, float]]:
    return [{"start": float(i * 10), "end": float(i * 10 + 1)} for i in range(4)]


def _one_interval_per_group(monkeypatch) -> None:
    """Pin grouping to one interval per iteration so group counts are exact."""

    monkeypatch.setattr(
        asr_align, "build_alignment_groups", lambda remaining, **_: [[remaining[0]]]
    )


def _run_align_with_checkpoint(monkeypatch, tmp_path, *, fail_at=None, gap_sec=0.3):
    intervals = _checkpoint_intervals()
    calls: list[float] = []

    def fake_group(group, audio, sr, **kwargs):
        start = float(group[0]["start"])
        calls.append(start)
        if fail_at is not None and start == fail_at:
            raise RuntimeError("boom")
        return [{"start": start, "end": float(group[-1]["end"]), "lang": "ja"}], []

    _one_interval_per_group(monkeypatch)
    monkeypatch.setattr(asr_align, "_align_intervals_group", fake_group)
    monkeypatch.setattr(asr_align, "_build_recall_temp_groups", lambda *a, **k: [])
    checkpoint = checkpoint_store.path_for_output(tmp_path / "x-aligned.json")

    def run():
        return asr_align.align_segments(
            intervals,
            None,
            16000,
            model=object(),
            gap_sec=gap_sec,
            language="ja",
            checkpoint_path=checkpoint,
            checkpoint_key=checkpoint_store.build_key(
                model_name="large-v3-turbo",
                language="ja",
                gap_sec=gap_sec,
                audio_path=tmp_path / "audio.wav",
            ),
        )

    return run, calls, checkpoint


def test_reused_language_group_does_not_enter_history(monkeypatch) -> None:
    """A short group that borrows the history's language must not vote in it.

    Otherwise the reused language keeps re-electing itself and the window can
    never recover from one bad detection."""

    def fake_transcribe(_model, _audio, **kwargs):
        # Language was assigned from history, not detected by whisper.
        assert kwargs.get("language") == "ja"
        return {
            "language": "ja",
            "segments": [
                {"text": "正常", "words": [{"text": "正常", "start": 0.1, "end": 0.5}]}
            ],
        }

    def fake_finalize(group, *_args, lang: str, **_kwargs):
        return [{"start": float(group[0]["start"]), "end": float(group[-1]["end"]), "lang": lang}]

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", fake_finalize)
    history = ["ja"] * asr_align.AUTO_LANGUAGE_HISTORY_GROUPS

    asr_align._align_intervals_group(
        [{"start": 0.0, "end": 5.0}],  # <= AUTO_LANGUAGE_SHORT_GROUP_SEC
        np.zeros(100, dtype=np.float32),
        10,
        model=_wt_model(fake_transcribe),
        gap_sec=0.3,
        language=None,
        auto_language_history=history,
    )

    assert history == ["ja"] * asr_align.AUTO_LANGUAGE_HISTORY_GROUPS


def test_asr_checkpoint_path_does_not_collide_with_aligned_output() -> None:
    path = checkpoint_store.path_for_output("out/foo/foo-aligned.json")

    assert path.name == "foo-aligned.partial.json"


def test_asr_checkpoint_cleared_after_successful_run(monkeypatch, tmp_path) -> None:
    run, calls, checkpoint = _run_align_with_checkpoint(monkeypatch, tmp_path)

    segments = run()

    assert [seg["start"] for seg in segments] == [0.0, 10.0, 20.0, 30.0]
    assert calls == [0.0, 10.0, 20.0, 30.0]
    assert not checkpoint.exists()


def test_asr_checkpoint_resumes_after_crash(monkeypatch, tmp_path) -> None:
    run, calls, checkpoint = _run_align_with_checkpoint(
        monkeypatch, tmp_path, fail_at=20.0
    )
    with pytest.raises(RuntimeError):
        run()

    # Two groups survived the crash; the partial holds them plus the cursor.
    assert calls == [0.0, 10.0, 20.0]
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["processed_intervals"] == 2
    assert [seg["start"] for seg in saved["segments"]] == [0.0, 10.0]

    run2, calls2, _ = _run_align_with_checkpoint(monkeypatch, tmp_path)
    segments = run2()

    # Only the unfinished groups are re-decoded, and the output is complete.
    assert calls2 == [20.0, 30.0]
    assert [seg["start"] for seg in segments] == [0.0, 10.0, 20.0, 30.0]
    assert not checkpoint.exists()


def test_asr_checkpoint_ignored_when_parameters_change(monkeypatch, tmp_path) -> None:
    run, _, checkpoint = _run_align_with_checkpoint(monkeypatch, tmp_path, fail_at=20.0)
    with pytest.raises(RuntimeError):
        run()
    assert checkpoint.exists()

    # A different gap_sec is a different alignment: the partial must be dropped.
    run2, calls2, _ = _run_align_with_checkpoint(monkeypatch, tmp_path, gap_sec=0.5)
    segments = run2()

    assert calls2 == [0.0, 10.0, 20.0, 30.0]
    assert len(segments) == 4


def test_asr_checkpoint_ignores_legacy_schema(monkeypatch, tmp_path) -> None:
    run, calls, checkpoint = _run_align_with_checkpoint(monkeypatch, tmp_path)
    intervals = _checkpoint_intervals()
    key = checkpoint_store.build_key(
        model_name="large-v3-turbo",
        language="ja",
        gap_sec=0.3,
        audio_path=tmp_path / "audio.wav",
    )
    key["intervals"] = checkpoint_store.intervals_digest(intervals)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "version": checkpoint_store.SCHEMA_VERSION - 1,
                "fingerprint": key,
                "processed_intervals": 1,
                "group_idx": 1,
                "segments": [{"start": 0.0, "end": 1.0, "text": "legacy"}],
            }
        ),
        encoding="utf-8",
    )

    segments = run()

    assert calls == [0.0, 10.0, 20.0, 30.0]
    assert [segment["start"] for segment in segments] == [0.0, 10.0, 20.0, 30.0]


def test_asr_checkpoint_ignores_corrupt_partial(monkeypatch, tmp_path) -> None:
    run, calls, checkpoint = _run_align_with_checkpoint(monkeypatch, tmp_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("{not json", encoding="utf-8")

    segments = run()

    assert calls == [0.0, 10.0, 20.0, 30.0]
    assert len(segments) == 4


def test_align_segments_without_checkpoint_writes_nothing(monkeypatch, tmp_path) -> None:
    intervals = _checkpoint_intervals()
    _one_interval_per_group(monkeypatch)
    monkeypatch.setattr(
        asr_align,
        "_align_intervals_group",
        lambda group, audio, sr, **k: (
            [{"start": float(group[0]["start"]), "end": 1.0}], []
        ),
    )
    monkeypatch.setattr(asr_align, "_build_recall_temp_groups", lambda *a, **k: [])

    segments = asr_align.align_segments(
        intervals, None, 16000, model=object(), gap_sec=0.3, language="ja"
    )

    assert len(segments) == 4
    assert list(tmp_path.iterdir()) == []


def test_coverage_shortfall_exempts_short_batches() -> None:
    intervals = [{"start": 0.0, "end": 3.0}]

    # 0.6 * 3s - 2s < 0: small batches never trigger, even with no output.
    assert asr_align._coverage_shortfall(intervals, []) is None


def test_coverage_shortfall_detects_window_skip() -> None:
    intervals = [{"start": 0.0, "end": 10.0}, {"start": 11.0, "end": 21.0}]
    segments = [{"start": 0.0, "end": 4.0, "text": "x"}]

    shortfall = asr_align._coverage_shortfall(intervals, segments)

    assert shortfall is not None
    speech, covered, required = shortfall
    assert speech == pytest.approx(20.0)
    assert covered == pytest.approx(4.0)
    assert required == pytest.approx(10.0)


def test_covered_speech_ignores_segment_time_outside_intervals() -> None:
    intervals = [{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 30.0}]
    # One segment spanning the inter-interval gap: only the 6s + 6s inside
    # the intervals count as covered speech.
    segments = [{"start": 4.0, "end": 26.0, "text": "x"}]

    assert asr_align._covered_speech_seconds(intervals, segments) == pytest.approx(12.0)


def test_recall_temp_groups_skip_sliver_complements() -> None:
    intervals = [{"start": 0.0, "end": 10.0}]
    # Complements: 0-6 (real) and 9.97-10.0 (sliver below RECALL_COMPLEMENT_MIN_SEC).
    spans = [(6.0, 9.97)]

    groups = asr_align._build_recall_temp_groups(intervals, segment_spans=spans)

    assert groups == [[{"start": 0.0, "end": 6.0}]]


def test_recall_sliver_complements_do_not_count_toward_threshold() -> None:
    intervals = [{"start": 0.0, "end": 10.0}]
    # Complements: 0-4.9 (real, below the 5s recall threshold alone) and
    # 5.05-5.25 (0.2s sliver). Without the sliver filter the chain would
    # reach 5.1s and trigger recall.
    spans = [(4.9, 5.05), (5.25, 10.0)]

    assert asr_align._build_recall_temp_groups(intervals, segment_spans=spans) == []


def test_recall_tail_limit_bounded_by_covered_span_and_next_interval() -> None:
    temp_group = [{"start": 1.0, "end": 2.0}]

    # Covered span starting exactly at the chain end: no real-audio tail.
    assert asr_align._recall_tail_limit_sec(temp_group, [(2.0, 5.0)], []) == 0.0
    # Span covering the chain end: no tail either.
    assert asr_align._recall_tail_limit_sec(temp_group, [(1.5, 5.0)], []) == 0.0
    # Nearby span start bounds the tail.
    assert asr_align._recall_tail_limit_sec(
        temp_group, [(2.3, 5.0)], []
    ) == pytest.approx(0.3)
    # Next interval start bounds the tail; earlier starts are ignored.
    assert asr_align._recall_tail_limit_sec(
        temp_group, [], [0.5, 2.4]
    ) == pytest.approx(0.4)
    # No obstacle: full gap-audio allowance.
    assert asr_align._recall_tail_limit_sec(temp_group, [], []) == pytest.approx(
        asr_align.GAP_KEEP_REAL_MAX_SEC
    )


def _full_span_finalize(group, _words, _asr_segments, *_args, lang, **_kwargs):
    return [
        {
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"]),
            "text": "T",
            "lang": lang,
        }
    ]


def test_low_coverage_rescue_accepts_clean_beam_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [{"start": 0.0, "end": 10.0}, {"start": 11.0, "end": 21.0}]
    original = [{"start": 0.0, "end": 2.0, "text": "短", "lang": "ja"}]
    decode_options_seen = []

    def fake_transcribe_candidate(_model, g, *_args, **kwargs):
        decode_options_seen.append(kwargs.get("decode_options"))
        return [[] for _ in g], [[] for _ in g], "ja", [], False

    monkeypatch.setattr(
        asr_align, "_transcribe_group_candidate", fake_transcribe_candidate
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", _full_span_finalize)

    rescued = asr_align._rescue_low_coverage(
        object(), group, original, None, 10, 0.3, language="ja"
    )

    assert decode_options_seen == [asr_align._rescue_decode_options()]
    assert rescued == [{"start": 0.0, "end": 21.0, "text": "T", "lang": "ja"}]


def test_low_coverage_rescue_peels_until_rear_window_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval_a = {"start": 0.0, "end": 10.0}
    interval_b = {"start": 11.0, "end": 21.0}
    interval_c = {"start": 22.0, "end": 32.0}
    group = [interval_a, interval_b, interval_c]
    original = [{"start": 0.0, "end": 2.0, "text": "短", "lang": "ja"}]
    windows = []

    def fake_transcribe_candidate(_model, g, *_args, **_kwargs):
        # Beam rescue attempt: abnormal, so the ladder moves on to splitting.
        return [[] for _ in g], [[] for _ in g], "ja", ["fake_issue"], False

    def fake_align_group(_model, g, *_args, **_kwargs):
        windows.append([float(item["start"]) for item in g])
        if g == [interval_b, interval_c]:
            # First rear window: still coverage-low, forcing another peel.
            return [], []
        return [
            {
                "start": float(g[0]["start"]),
                "end": float(g[-1]["end"]),
                "text": "T",
                "lang": "ja",
            }
        ], []

    monkeypatch.setattr(
        asr_align, "_transcribe_group_candidate", fake_transcribe_candidate
    )
    monkeypatch.setattr(asr_align, "align_group", fake_align_group)

    rescued = asr_align._rescue_low_coverage(
        object(), group, original, None, 10, 0.3, language="ja"
    )

    assert windows == [[0.0], [11.0, 22.0], [11.0], [22.0]]
    assert [seg["start"] for seg in rescued] == [0.0, 11.0, 22.0]


def test_low_coverage_rescue_keeps_original_when_not_improved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [{"start": 0.0, "end": 10.0}, {"start": 11.0, "end": 21.0}]
    original = [{"start": 0.0, "end": 2.0, "text": "短", "lang": "ja"}]

    def fake_transcribe_candidate(_model, g, *_args, **_kwargs):
        return [[] for _ in g], [[] for _ in g], "ja", ["fake_issue"], False

    def fake_align_group(*_args, **_kwargs):
        return [], []

    monkeypatch.setattr(
        asr_align, "_transcribe_group_candidate", fake_transcribe_candidate
    )
    monkeypatch.setattr(asr_align, "align_group", fake_align_group)

    rescued = asr_align._rescue_low_coverage(
        object(), group, original, None, 10, 0.3, language="ja"
    )

    assert rescued == original


def test_align_group_isolates_without_beam_rescue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ladder no longer tries a whole-group beam decode before isolation.

    Beam lost to isolation on every measured axis (it collapsed on degenerate
    audio just like greedy, and dropped content on top), so an abnormal result
    now goes straight to interval isolation with greedy decoding throughout."""

    beam_sizes = []
    abnormal_result = {
        "language": "ja",
        "segments": [
            {
                "text": "え" * 32,
                "words": [{"text": "え" * 32, "start": 0.1, "end": 0.5}],
            }
        ],
    }
    clean_result = {
        "language": "ja",
        "segments": [
            {
                "text": "正常 正常",
                "words": [
                    {"text": "正常", "start": 0.1, "end": 0.5},
                    {"text": "正常", "start": 7.1, "end": 7.5},
                ],
            }
        ],
    }
    results = iter([abnormal_result, clean_result, clean_result, clean_result])

    def fake_transcribe(_model, _audio, **kwargs):
        beam_sizes.append(kwargs.get("beam_size"))
        return next(results)

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", _full_span_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(fake_transcribe),
        [{"start": 0.0, "end": 6.0}, {"start": 7.0, "end": 13.0}],
        np.zeros(130, dtype=np.float32),
        10,
        0.3,
        language="ja",
    )

    # Greedy throughout: no rescue decode in the ladder requests beam search.
    assert asr_align.ASR_RESCUE_BEAM_SIZE not in beam_sizes
    assert set(beam_sizes) == {None}
    assert aligned


def test_asr_metadata_records_coverage_rescue_tunables() -> None:
    metadata = asr_align.asr_align_metadata(
        model="large-v3-turbo",
        device="cuda",
        language="ja",
        gap_sec=0.3,
    )

    assert metadata["asr_coverage_min_ratio"] == asr_align.ASR_COVERAGE_MIN_RATIO
    assert (
        metadata["asr_coverage_tolerance_sec"] == asr_align.ASR_COVERAGE_TOLERANCE_SEC
    )
    assert metadata["asr_rescue_beam_size"] == asr_align.ASR_RESCUE_BEAM_SIZE
    assert metadata["recall_complement_min_sec"] == asr_align.RECALL_COMPLEMENT_MIN_SEC


def test_collapse_word_stack_detects_consecutive_near_zero_words() -> None:
    # Confirmed collapse shape (collapse re-align experiment, kaguya60#187):
    # a run of frame-quantized ~20ms words piled into a near-zero span.
    stacked = [
        {"start": 8.00 + 0.02 * i, "end": 8.02 + 0.02 * i, "word": w}
        for i, w in enumerate(["いや、", "絶", "対"])
    ]
    tail = [{"start": 8.5, "end": 8.9, "word": "そこ"}]

    issues = detect_abnormal_asr_words([stacked + tail])

    assert any(issue.startswith("collapse_word_stack count=3") for issue in issues)


def test_collapse_word_stack_ignores_isolated_near_zero_words() -> None:
    # Fast-but-correct speech shape (yui#87): near-zero words appear only in
    # isolation between normally timed words — must not trigger.
    words = [
        {"start": 364.83, "end": 364.85, "word": "え、"},
        {"start": 364.85, "end": 364.93, "word": "どう"},
        {"start": 364.93, "end": 365.27, "word": "いうこと?"},
        {"start": 365.27, "end": 365.29, "word": "どう"},
        {"start": 365.29, "end": 365.43, "word": "いうこと?"},
        {"start": 365.43, "end": 365.53, "word": "や"},
        {"start": 365.53, "end": 365.77, "word": "ばい!"},
    ]

    issues = detect_abnormal_asr_words([words])

    assert not any(issue.startswith("collapse_word_stack") for issue in issues)


def test_collapse_word_stack_requires_run_of_three() -> None:
    two_stacked = [
        {"start": 1.00, "end": 1.02, "word": "あ"},
        {"start": 1.02, "end": 1.04, "word": "い"},
        {"start": 1.10, "end": 1.50, "word": "う"},
    ]

    issues = detect_abnormal_asr_words([two_stacked])

    assert not any(issue.startswith("collapse_word_stack") for issue in issues)


def test_align_group_isolates_abnormal_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Gap between intervals is exactly 1.0s (0.7 kept real + 0.3 silence), so
    # the combined timeline maps back to the original one identically.
    group = [
        {"start": 0.0, "end": 6.0},
        {"start": 7.0, "end": 13.0},
        {"start": 14.0, "end": 20.0},
    ]
    clean_word = {"text": "正常", "start": 0.1, "end": 0.5}
    stack_words = [
        {"text": w, "start": 8.00 + 0.02 * i, "end": 8.02 + 0.02 * i}
        for i, w in enumerate(["私", "た", "ち", "は"])
    ]
    abnormal_group_result = {
        "language": "ja",
        "segments": [
            {"text": "正常", "words": [clean_word]},
            {"text": "私たちは", "words": stack_words},
            {"text": "正常", "words": [{"text": "正常", "start": 15.0, "end": 15.4}]},
        ],
    }
    clean_single_result = {
        "language": "ja",
        "segments": [{"text": "正常", "words": [clean_word]}],
    }
    results = iter(
        [
            abnormal_group_result,  # initial greedy decode of the full group
            clean_single_result,  # clean-front window [interval 0]
            clean_single_result,  # isolated abnormal interval 1
            clean_single_result,  # remainder window [interval 2]
        ]
    )
    beam_sizes = []

    def fake_transcribe(_model, _audio, **kwargs):
        beam_sizes.append(kwargs.get("beam_size"))
        return next(results)

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", _full_span_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(fake_transcribe),
        group,
        np.zeros(220, dtype=np.float32),
        10,
        0.3,
        language="ja",
    )

    # initial + 前窗 + 隔离区间；第三个 interval 作为未消费尾部交还，
    # 由主循环与后续内容重新分组，不在这里解码。
    assert beam_sizes == [None, None, None]
    assert _unconsumed == [group[2]]
    assert aligned == [
        {"start": 0.0, "end": 6.0, "text": "T", "lang": "ja"},
        {"start": 7.0, "end": 13.0, "text": "T", "lang": "ja"},
    ]


def test_asr_metadata_records_collapse_stack_tunables() -> None:
    metadata = asr_align.asr_align_metadata(
        model="large-v3-turbo",
        device="cuda",
        language="ja",
        gap_sec=0.3,
    )

    assert metadata["collapse_stack_word_sec"] == asr_align.COLLAPSE_STACK_WORD_SEC
    assert metadata["collapse_stack_min_run"] == asr_align.COLLAPSE_STACK_MIN_RUN


def test_disfluency_marker_is_exempt_from_word_rules() -> None:
    # A long [*] span is a disfluency candidate (resolved by word_starts),
    # not a stretched word; it must not trigger the rescue ladder. Observed
    # live on BV1dwjP6LECU: long_word_duration=5.35s token='[*]'.
    words = [
        {"start": 564.2, "end": 569.6, "word": "[*]"},
        {"start": 569.6, "end": 570.1, "word": "断"},
    ]
    assert asr_align.detect_abnormal_asr_words([words]) == []
    # The same span with real text still fires.
    stretched = [dict(words[0], word="断")]
    assert any(
        issue.startswith("long_word_duration")
        for issue in asr_align.detect_abnormal_asr_words([stretched])
    )


def test_known_phrase_stack_only_predicate() -> None:
    phrase_stack = [
        {"start": 8.00, "end": 8.02, "word": "ご視聴"},
        {"start": 8.02, "end": 8.04, "word": "ありがとう"},
        {"start": 8.04, "end": 8.06, "word": "ございました"},
    ]
    issues = ["collapse_word_stack count=3 span=8.000-8.060 token='x'"]

    assert asr_align._is_known_phrase_stack_only([phrase_stack], issues)
    # 短语重复堆叠也算
    assert asr_align._is_known_phrase_stack_only(
        [phrase_stack + [dict(w, start=w["start"] + 0.06, end=w["end"] + 0.06) for w in phrase_stack]],
        issues,
    )
    # 真话 + 零宽短语尾也算：堆叠部分是短语、剩余词自身干净（400 窗审计中
    # 纯短语异常窗全部是这个形态），重解只会给健康部分引入方差。
    real_plus_tail = [
        {"start": 6.00, "end": 7.10, "word": "今全部"},
        {"start": 7.10, "end": 7.90, "word": "喋ったね"},
    ] + phrase_stack
    assert asr_align._is_known_phrase_stack_only([real_plus_tail], issues)
    # 截断片段（掉了「ご」）也算
    truncated = [
        {"start": 8.00, "end": 8.02, "word": "視聴"},
        {"start": 8.02, "end": 8.04, "word": "ありがとう"},
        {"start": 8.04, "end": 8.06, "word": "ございました"},
    ]
    assert asr_align._is_known_phrase_stack_only([truncated], issues)

    # 堆叠含非短语文本 / 剩余词自身异常 / 非 stack issue / 碎片过短 → 不早退
    alien_stack = phrase_stack + [{"start": 8.06, "end": 8.08, "word": "本当に"}]
    assert not asr_align._is_known_phrase_stack_only([alien_stack], issues)
    bad_remainder = [
        {"start": 5.0 + i * 0.3, "end": 5.25 + i * 0.3, "word": "ぺそ"}
        for i in range(9)
    ] + phrase_stack
    assert not asr_align._is_known_phrase_stack_only([bad_remainder], issues)
    assert not asr_align._is_known_phrase_stack_only(
        [phrase_stack], issues + ["repeating_token token='x'"]
    )
    tiny_fragment = [
        {"start": 8.00, "end": 8.02, "word": "あり"},
        {"start": 8.02, "end": 8.04, "word": "が"},
        {"start": 8.04, "end": 8.06, "word": "とう"},
    ]
    assert not asr_align._is_known_phrase_stack_only([tiny_fragment], issues)


def test_align_group_skips_ladder_for_phrase_only_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [{"start": 0.0, "end": 6.0}, {"start": 7.0, "end": 13.0}]
    phrase_result = {
        "language": "ja",
        "segments": [
            {"text": "正常", "words": [{"text": "正常", "start": 0.1, "end": 0.5}]},
            {
                "text": "ご視聴ありがとうございました",
                "words": [
                    {"text": "ご視聴", "start": 8.00, "end": 8.02},
                    {"text": "ありがとう", "start": 8.02, "end": 8.04},
                    {"text": "ございました", "start": 8.04, "end": 8.06},
                ],
            },
        ],
    }
    calls = []

    def fake_transcribe(_model, _audio, **kwargs):
        calls.append(kwargs.get("beam_size"))
        return phrase_result

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", _full_span_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(fake_transcribe),
        group,
        np.zeros(150, dtype=np.float32),
        10,
        0.3,
        language="ja",
    )

    assert calls == [None]  # 单次贪心解码后早退，无重试/beam
    assert aligned == [{"start": 0.0, "end": 13.0, "text": "T", "lang": "ja"}]


def test_isolation_front_falls_back_to_clean_slice_on_degenerate_redecode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    group = [
        {"start": 0.0, "end": 6.0},
        {"start": 7.0, "end": 13.0},
        {"start": 14.0, "end": 20.0},
    ]
    stack_words = [
        {"text": w, "start": 8.00 + 0.02 * i, "end": 8.02 + 0.02 * i}
        for i, w in enumerate(["私", "た", "ち", "は"])
    ]
    abnormal_group_result = {
        "language": "ja",
        "segments": [
            # 前窗候选词覆盖大半个 interval（覆盖率闸门要求切片真的有内容；
            # 拆成两个词避免触发 long_word_duration 异常）
            {
                "text": "正常 正常",
                "words": [
                    {"text": "正常", "start": 0.1, "end": 3.0},
                    {"text": "正常", "start": 3.0, "end": 5.9},
                ],
            },
            {"text": "私たちは", "words": stack_words},
            {"text": "正常", "words": [{"text": "正常", "start": 15.0, "end": 15.4}]},
        ],
    }
    # 前窗重解码退化成复读堆叠（lang=en 作为来源标记）
    degenerate_front_result = {
        "language": "en",
        "segments": [
            {
                "text": "へへへ",
                "words": [
                    {"text": "へ", "start": 0.10 + 0.02 * i, "end": 0.12 + 0.02 * i}
                    for i in range(4)
                ],
            }
        ],
    }
    clean_single_result = {
        "language": "ja",
        "segments": [{"text": "正常", "words": [{"text": "正常", "start": 0.1, "end": 0.5}]}],
    }
    results = iter(
        [
            abnormal_group_result,  # initial full-group decode
            abnormal_group_result,  # group beam (still bad)
            degenerate_front_result,  # front window re-decode degenerates
            clean_single_result,  # isolated abnormal interval
            clean_single_result,  # remainder window
        ]
    )

    def fake_transcribe(_model, _audio, **_kwargs):
        return next(results)

    finalized_word_texts = []

    def recording_finalize(group_arg, words, _asr_segments, *args, lang, **kwargs):
        finalized_word_texts.append(
            "".join(str(w.get("word") or "") for ws in words for w in ws)
        )
        return _full_span_finalize(group_arg, words, _asr_segments, *args, lang=lang, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=fake_transcribe),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", recording_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(fake_transcribe),
        group,
        np.zeros(220, dtype=np.float32),
        10,
        0.3,
        language="ja",
    )

    # 前窗保留候选切片（词为 正常正常），而不是退化重解码（へへへ 堆叠）
    assert finalized_word_texts[0] == "正常正常"
    assert "へ" not in finalized_word_texts[0]
    # 隔离区间之后的 interval 作为未消费尾部交还，不在本次 align_group 内解码。
    assert [seg["start"] for seg in aligned] == [0.0, 7.0]
    assert _unconsumed == [group[2]]


def test_isolation_front_hollow_slice_decodes_interval_by_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候选把前窗语音整段吸附到异常 interval（前窗词表为空）时，覆盖率闸门
    必须拒绝空切片，降级为逐 interval 重解恢复内容。"""
    group = [
        {"start": 0.0, "end": 6.0},
        {"start": 7.0, "end": 13.0},
        {"start": 14.0, "end": 20.0},
    ]
    stack_words = [
        {"text": w, "start": 8.00 + 0.02 * i, "end": 8.02 + 0.02 * i}
        for i, w in enumerate(["私", "た", "ち", "は"])
    ]
    hollow_front_result = {
        "language": "ja",
        "segments": [
            # 前窗 interval 无词；异常 stack 挂在 interval 1
            {"text": "私たちは", "words": stack_words},
            {"text": "正常", "words": [{"text": "正常", "start": 15.0, "end": 15.4}]},
        ],
    }
    degenerate_front_result = {
        "language": "ja",
        "segments": [
            {
                "text": "へへへ",
                "words": [
                    {"text": "へ", "start": 0.10 + 0.02 * i, "end": 0.12 + 0.02 * i}
                    for i in range(4)
                ],
            }
        ],
    }
    recovered_front_result = {
        "language": "ja",
        "segments": [{"text": "救出", "words": [{"text": "救出", "start": 0.1, "end": 5.9}]}],
    }
    clean_single_result = {
        "language": "ja",
        "segments": [{"text": "正常", "words": [{"text": "正常", "start": 0.1, "end": 0.5}]}],
    }
    results = iter(
        [
            hollow_front_result,  # initial full-group decode
            degenerate_front_result,  # front window re-decode degenerates
            recovered_front_result,  # front interval 0 decoded alone
            clean_single_result,  # isolated abnormal interval
            clean_single_result,  # remainder window
        ]
    )

    finalized_word_texts = []

    def recording_finalize(group_arg, words, _asr_segments, *args, lang, **kwargs):
        finalized_word_texts.append(
            "".join(str(w.get("word") or "") for ws in words for w in ws)
        )
        return _full_span_finalize(group_arg, words, _asr_segments, *args, lang=lang, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=lambda _m, _a, **_k: next(results)),
    )
    monkeypatch.setattr(asr_align, "_finalize_group_candidate", recording_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(lambda _m, _a, **_k: next(results)),
        group,
        np.zeros(220, dtype=np.float32),
        10,
        0.3,
        language="ja",
    )

    # 空切片被拒；前窗 interval 单独重解的 救出 进入结果
    assert "救出" in finalized_word_texts
    assert all("へ" not in text for text in finalized_word_texts)
    # 隔离区间之后的 interval 作为未消费尾部交还，不在本次 align_group 内解码。
    assert [seg["start"] for seg in aligned] == [0.0, 7.0]
    assert _unconsumed == [group[2]]


def _motif_words(t0: float, reps: int = 5, step: float = 0.45, key: str = "word"):
    out, t = [], t0
    for _ in range(reps):
        for ch in "私たちは":
            out.append({key: ch, "start": round(t, 2), "end": round(t + 0.1, 2)})
            t += step
    return out


def test_per_interval_check_misses_cross_interval_collapse() -> None:
    """repeating_group_cycle 要 32 units 才触发；一条横跨两个 interval 的循环
    每个 interval 各占 20 units，逐 interval 判定全部「干净」，合起来才是坍缩。"""

    per_interval = [_motif_words(0.1), _motif_words(12.1)]

    assert asr_align._first_abnormal_interval_index(per_interval) is None
    assert all(not detect_abnormal_asr_words([words]) for words in per_interval)

    joined = detect_abnormal_asr_words(per_interval)
    assert any(issue.startswith("repeating_group_cycle") for issue in joined)


def test_isolation_rejects_front_slice_with_cross_interval_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """前窗重解失败时，不得沿用「逐 interval 干净、整体是坍缩」的候选切片。

    两个 interval 各挂一条 5 次 motif 的 whisper segment（各 20 units，单独看
    都干净），第三个 interval 单独异常把 k 推到 2，于是前两个成为「干净前窗」；
    前窗重解退化后，只有对切片整体重新判定才能拒绝它。"""

    group = [
        {"start": 0.0, "end": 10.0},
        {"start": 12.0, "end": 22.0},
        {"start": 24.0, "end": 30.0},
    ]
    # 关键：两段 motif 必须分属不同 whisper segment，否则整段会被挂到同一个
    # interval 上，该 interval 自己就超阈值、k 会退到 0（前窗为空）。
    diluted_result = {
        "language": "ja",
        "segments": [
            {"text": "私たちは" * 5, "words": _motif_words(0.1, key="text")},
            {"text": "私たちは" * 5, "words": _motif_words(12.1, key="text")},
            {"text": "あ" * 20, "words": [{"text": "あ" * 20, "start": 25.0, "end": 25.4}]},
        ],
    }
    degenerate = {
        "language": "ja",
        "segments": [
            {
                "text": "へへへへ",
                "words": [
                    {"text": "へ", "start": round(0.10 + 0.02 * i, 3),
                     "end": round(0.12 + 0.02 * i, 3)}
                    for i in range(4)
                ],
            }
        ],
    }
    clean = {
        "language": "ja",
        "segments": [{"text": "救出", "words": [{"text": "救出", "start": 0.1, "end": 3.0}]}],
    }
    results = iter([diluted_result, degenerate] + [clean] * 10)
    monkeypatch.setitem(
        sys.modules,
        "whisper_timestamped",
        types.SimpleNamespace(transcribe=lambda *a, **k: next(results)),
    )

    def joining_finalize(part, words, segs, audio, sr, *, lang, **kwargs):
        text = "".join(
            str(w.get("word") or w.get("text") or "") for ws in words for w in ws
        )
        return [{"start": float(part[0]["start"]), "end": float(part[-1]["end"]),
                 "text": text, "lang": lang}]

    monkeypatch.setattr(asr_align, "_finalize_group_candidate", joining_finalize)

    aligned, _unconsumed = asr_align.align_group(
        _wt_model(lambda *a, **k: next(results)), group, np.zeros(320, dtype=np.float32), 10, 0.3, language="ja"
    )

    # 断言看返回的 segment，不看 finalize 调用记录：候选切片为了做覆盖率判定
    # 必然会被 finalize 一次，那不代表它被采用。
    assert not any("私たちは私たちは" in str(seg.get("text") or "") for seg in aligned)


def test_group_audio_seconds_matches_the_audio_that_gets_built() -> None:
    """The planner's fit measure has to equal what build_combined_audio makes,
    or 'this group fits one encoder window' is not a claim about anything."""

    sr = 100
    audio = np.ones(3000, dtype=np.float32)  # 30s, so no tail runs off the end
    group = [
        {"start": 1.0, "end": 3.0},
        {"start": 4.0, "end": 6.0},
    ]
    successor = {"start": 8.0, "end": 9.0}

    combined, _offsets = asr_align.build_combined_audio(
        audio,
        sr,
        group,
        0.3,
        tail_real_limit_sec=min(
            asr_align.GAP_KEEP_REAL_MAX_SEC,
            successor["start"] - group[-1]["end"],
        ),
    )

    assert asr_align.combined_group_audio_seconds(
        group, successor, gap_sec=0.3
    ) == pytest.approx(len(combined) / sr)


def test_group_tail_is_bounded_by_the_successor_gap() -> None:
    group = [{"start": 0.0, "end": 1.0}]

    # Wide gap: the pad saturates at GAP_KEEP_REAL_MAX_SEC.
    assert asr_align.group_tail_seconds(
        group, {"start": 9.0}, gap_sec=0.3
    ) == pytest.approx(asr_align.GAP_KEEP_REAL_MAX_SEC + 0.3)
    # Tight gap: real pad may not bleed into the next group's speech.
    assert asr_align.group_tail_seconds(
        group, {"start": 1.2}, gap_sec=0.3
    ) == pytest.approx(0.5)
    # End of file: nothing bounds it but the constant.
    assert asr_align.group_tail_seconds(
        group, None, gap_sec=0.3
    ) == pytest.approx(asr_align.GAP_KEEP_REAL_MAX_SEC + 0.3)


def test_grouping_counts_the_tail_pad_against_the_target() -> None:
    """A group whose content lands just under the target no longer spills past
    the encoder window once the pad is added."""

    # Two 14.6s intervals with a 2s gap: content 29.2 + inserts, tail 1.0.
    segments = [
        {"start": 0.0, "end": 14.6},
        {"start": 16.6, "end": 31.2},
        {"start": 40.0, "end": 50.0},
    ]
    groups = asr_align.build_alignment_groups(
        segments, gap_sec=0.3, group_target_sec=30.0, min_group_length=5.0
    )

    flat = [item for group in groups for item in group]
    position = 0
    for group in groups:
        position += len(group)
        successor = flat[position] if position < len(flat) else None
        assert asr_align.combined_group_audio_seconds(
            group, successor, gap_sec=0.3
        ) <= 30.0


def _ghost_segment(
    text: str,
    at: float,
    words: list[str] | None = None,
    *,
    events: bool = True,
) -> dict:
    parts = words if words is not None else [text]
    segment = {
        "start": at,
        "end": at,
        "text": text,
        "words": [
            {"start": at, "end": at, "word": part, "confidence": 0.6}
            for part in parts
        ],
    }
    if events:
        segment["alignment_events"] = [
            {"type": "zero_duration_chunk_tail", "word": parts[-1]}
        ]
    return segment


def _real_segment(text: str, start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [{"start": start, "end": end, "word": text, "confidence": 0.9}],
    }


def test_ghost_duplicate_of_neighbor_is_dropped() -> None:
    segments = [
        _real_segment("乙女心", 12.8, 13.4),
        _real_segment("満載って感じですけど", 15.2, 16.4),
        _ghost_segment("乙女", 15.3),
    ]
    out, dropped = recognition_segments.drop_ghost_duplicate_segments(segments)
    assert [seg["text"] for seg in out] == ["乙女心", "満載って感じですけど"]
    assert len(dropped) == 1 and "乙女" in dropped[0]


def test_ghost_pair_echoing_one_real_segment_is_fully_dropped() -> None:
    segments = [
        _real_segment("どうしてロザリンまで", 1098.5, 1099.7),
        _ghost_segment("どうしてロザリンまで", 1101.3, ["どうして", "ロザリン", "まで"]),
        _ghost_segment("どうしてロザリンまで", 1101.3, ["どうして", "ロザリン", "まで"]),
    ]
    out, dropped = recognition_segments.drop_ghost_duplicate_segments(segments)
    assert [seg["text"] for seg in out] == ["どうしてロザリンまで"]
    assert len(dropped) == 2


def test_ghosts_without_a_real_duplicate_are_kept() -> None:
    # A novel ghost still goes through the normal abnormality ladder, two
    # identical ghosts must not confirm each other, and single-character
    # keys or far-away duplicates are not evidence.
    segments = [
        _real_segment("いっぱい見せてくれてありがとう", 215.5, 217.3),
        _ghost_segment("お疲れ様でした", 218.4),
        _ghost_segment("楽しかった", 219.0),
        _ghost_segment("楽しかった", 219.0),
        _real_segment("よかった", 230.0, 231.0),
        _ghost_segment("よ", 231.5),
    ]
    out, dropped = recognition_segments.drop_ghost_duplicate_segments(segments)
    assert dropped == []
    assert len(out) == len(segments)


def test_ghost_duplicate_outside_context_window_is_kept() -> None:
    segments = [
        _real_segment("同じ言葉", 100.0, 101.0),
        _ghost_segment("同じ言葉", 110.0),
    ]
    out, dropped = recognition_segments.drop_ghost_duplicate_segments(segments)
    assert dropped == []
    assert len(out) == 2


def test_ghost_without_decode_evidence_is_kept() -> None:
    # A quantized real repeat (twice-shouted call, sung refrain) matches the
    # span+duplicate checks but carries no squeeze event; it must survive.
    segments = [
        _real_segment("おい!", 100.0, 100.6),
        _ghost_segment("おい!", 101.2, events=False),
    ]
    out, dropped = recognition_segments.drop_ghost_duplicate_segments(segments)
    assert dropped == []
    assert len(out) == 2
