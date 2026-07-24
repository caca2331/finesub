from __future__ import annotations

import json
from pathlib import Path

import pytest

import asr_stabilize


def _word(
    text: str,
    start: float,
    end: float,
    confidence: float = 0.9,
    *,
    space_before: bool = False,
) -> dict[str, object]:
    return {
        "start": start,
        "end": end,
        "word": text,
        "space_before": space_before,
        "confidence": confidence,
    }


def _segment(
    words: list[dict[str, object]],
    *,
    confidence: float = 0.9,
    energy: float = 5.0,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, object]:
    return {
        "start": words[0]["start"] if start is None else start,
        "end": words[-1]["end"] if end is None else end,
        "text": "".join(str(word["word"]) for word in words),
        "words": words,
        "confidence": confidence,
        "vad_weighted_energy_db": energy,
    }


def _payload(*segments: dict[str, object]) -> dict[str, object]:
    return {
        "segments": list(segments),
        "metadata": {"asr_align": {"model": "test"}},
        "unknown": {"keep": True},
    }


def test_profile_1_drops_segment_when_phrase_consumes_all_words() -> None:
    segment = _segment(
        [
            _word("ご", 0.0, 0.1),
            _word("視", 0.1, 0.2),
            _word("聴", 0.2, 0.3),
            _word("ありがとうございました。", 0.3, 1.0),
        ]
    )

    result, report = asr_stabilize.stabilize_payload(_payload(segment), profile=1)

    assert result["segments"] == []
    assert report.phrase_occurrences_removed == 1
    assert report.emptied_segments == 1


def test_profile_1_ignores_phrase_spanning_more_than_five_words() -> None:
    pieces = ["ご", "視", "聴", "ありがとう", "ござい", "ました"]
    words = [_word(text, index, index + 1) for index, text in enumerate(pieces)]
    segment = _segment(words)

    result, report = asr_stabilize.stabilize_payload(_payload(segment), profile=1)

    assert result["segments"] == [segment]
    assert report.phrase_occurrences_removed == 0


def test_profile_1_keeps_partial_word_without_joining_punctuation_and_shrinks_start() -> None:
    segment = _segment(
        [
            _word("ご", 1.0, 1.2),
            _word("視", 1.2, 1.4),
            _word("聴", 1.4, 1.6),
            _word("ありがとうございました!ではまた", 1.6, 3.0),
        ],
        energy=-12.0,
    )

    result, _report = asr_stabilize.stabilize_payload(_payload(segment), profile=1)
    updated = result["segments"][0]

    assert updated["text"] == "ではまた"
    assert [word["word"] for word in updated["words"]] == ["ではまた"]
    assert updated["start"] == pytest.approx(1.6)
    assert updated["end"] == pytest.approx(3.0)
    assert updated["vad_weighted_energy_db"] == -12.0


def test_profile_1_only_shrinks_an_emptied_outer_word() -> None:
    prefix = _word("前文。", 0.0, 1.0)
    suffix = _word("後文", 5.0, 6.0)
    phrase_words = [
        _word("ご", 1.0, 2.0),
        _word("視", 2.0, 3.0),
        _word("聴", 3.0, 4.0),
        _word("ありがとうございました!", 4.0, 5.0),
    ]

    middle, _ = asr_stabilize.stabilize_payload(
        _payload(_segment([prefix, *phrase_words, suffix])), profile=1
    )
    trailing, _ = asr_stabilize.stabilize_payload(
        _payload(_segment([prefix, *phrase_words])), profile=1
    )

    assert middle["segments"][0]["text"] == "前文。後文"
    assert middle["segments"][0]["start"] == pytest.approx(0.0)
    assert middle["segments"][0]["end"] == pytest.approx(6.0)
    assert trailing["segments"][0]["text"] == "前文。"
    assert trailing["segments"][0]["end"] == pytest.approx(1.0)


def test_profile_1_removes_multiple_eligible_occurrences() -> None:
    target = asr_stabilize.COMMON_HALLUCINATION_TEXT
    words = [_word(f"{target}!中間{target}。末尾", 0.0, 2.0)]

    result, report = asr_stabilize.stabilize_payload(
        _payload(_segment(words)), profile=1
    )

    assert result["segments"][0]["text"] == "中間末尾"
    assert report.phrase_occurrences_removed == 2


def test_profile_1_leaves_segments_without_words_unchanged() -> None:
    segment = {
        "start": 0.0,
        "end": 1.0,
        "text": asr_stabilize.COMMON_HALLUCINATION_TEXT,
    }

    result, _report = asr_stabilize.stabilize_payload(_payload(segment), profile=1)

    assert result["segments"] == [segment]


@pytest.mark.parametrize(
    ("segment", "expected"),
    [
        (
            _segment([_word("速" * 23, 0.0, 1.0)]),
            [asr_stabilize.TAG_TIME_DRIFT],
        ),
        (
            _segment([_word("長い文章", 0.0, 0.2)], energy=-21.0),
            [
                asr_stabilize.TAG_HIGHLY_SUSPECTED_HALLUCINATION,
                asr_stabilize.TAG_TIME_DRIFT,
            ],
        ),
        (
            _segment([_word("え?", 0.0, 0.05)], energy=-21.0),
            [
                asr_stabilize.TAG_HIGHLY_SUSPECTED_HALLUCINATION,
                asr_stabilize.TAG_TIME_DRIFT,
            ],
        ),
        (
            _segment(
                [_word("長文", 0.0, 1.0, confidence=0.2)],
                confidence=0.2,
                energy=-1.0,
            ),
            [
                asr_stabilize.TAG_HIGHLY_SUSPECTED_HALLUCINATION,
                asr_stabilize.TAG_TIME_DRIFT,
            ],
        ),
        (
            _segment(
                [_word("え?", 0.0, 1.0, confidence=0.2)],
                confidence=0.2,
                energy=1.0,
            ),
            [
                asr_stabilize.TAG_HIGHLY_SUSPECTED_FILLER,
                asr_stabilize.TAG_TIME_DRIFT,
            ],
        ),
    ],
)
def test_profile_2_assigns_expected_tags(
    segment: dict[str, object], expected: list[str]
) -> None:
    result, _report = asr_stabilize.stabilize_payload(_payload(segment), profile=2)

    assert result["segments"][0]["tags"] == expected


def test_profile_2_uses_strict_thresholds_and_missing_metrics_do_not_classify() -> None:
    boundary = _segment(
        [_word("字" * 22, 0.0, 1.0, confidence=0.3)],
        confidence=0.3,
        energy=0.0,
    )
    boundary_minus_twenty = _segment(
        [_word("長い文章", 0.0, 1.0)], energy=-20.0
    )
    missing_energy = _segment(
        [_word("え?", 0.0, 1.0, confidence=0.2)],
        confidence=0.2,
    )
    missing_energy.pop("vad_weighted_energy_db")

    result, _report = asr_stabilize.stabilize_payload(
        _payload(boundary, boundary_minus_twenty, missing_energy), profile=2
    )

    assert "tags" not in result["segments"][0]
    assert result["segments"][1]["tags"] == [asr_stabilize.TAG_TIME_DRIFT]
    assert result["segments"][2]["tags"] == [asr_stabilize.TAG_TIME_DRIFT]


def test_weighted_word_confidence_uses_each_words_weighted_length() -> None:
    segment = _segment(
        [_word("日", 0.0, 0.5, confidence=0.1), _word("A", 0.5, 1.0, confidence=0.5)]
    )

    assert asr_stabilize.weighted_word_confidence(segment) == pytest.approx(
        (0.1 * 1.0 + 0.5 * 0.5) / 1.5
    )


def test_profile_0_runs_cleanup_then_tags_and_discards_suspicious_segments() -> None:
    phrase = _segment(
        [_word(asr_stabilize.COMMON_HALLUCINATION_TEXT, 0.0, 1.0)], energy=-30.0
    )
    filler = _segment(
        [_word("え?", 2.0, 3.0, confidence=0.1)], confidence=0.1, energy=2.0
    )
    drift = _segment([_word("速" * 23, 4.0, 5.0)])

    result, report = asr_stabilize.stabilize_payload(
        _payload(phrase, filler, drift), profile=0
    )

    assert len(result["segments"]) == 1
    assert result["segments"][0]["tags"] == [asr_stabilize.TAG_TIME_DRIFT]
    assert report.emptied_segments == 1
    assert report.suspicious_segments_dropped == 1


def test_profile_minus_one_is_byte_identical_and_other_profiles_preserve_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip-aligned.json"
    original = b'{\r\n  "segments": [],\r\n  "metadata": {"x": 1},\r\n  "unknown": true\r\n}\r\n'
    source.write_bytes(original)
    output = tmp_path / "clip-stable.json"

    written, report = asr_stabilize.stabilize_json_file(
        source, output_path=output, profile=-1
    )

    assert written == output.resolve()
    assert output.read_bytes() == original
    assert report.applied_profiles == ()

    payload = _payload(_segment([_word("正常", 0.0, 1.0)]))
    result, _ = asr_stabilize.stabilize_payload(payload, profile=2)
    assert result["metadata"] == payload["metadata"]
    assert result["unknown"] == payload["unknown"]


def test_default_output_path_replaces_aligned_suffix() -> None:
    assert asr_stabilize.default_output_path(Path("clip-aligned.json")) == Path(
        "clip-stable.json"
    )
    assert asr_stabilize.default_output_path(Path("clip.json")) == Path(
        "clip-stable.json"
    )


def test_unsupported_profile_and_invalid_schema_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported ASR stabilize profile"):
        asr_stabilize.stabilize_payload({"segments": []}, profile=4)
    with pytest.raises(ValueError, match="segments"):
        asr_stabilize.stabilize_payload({"metadata": {}}, profile=0)
