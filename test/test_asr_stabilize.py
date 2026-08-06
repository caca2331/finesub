from __future__ import annotations

import json
from pathlib import Path

import pytest

from asr_playground.speech.postprocessing import stabilization as asr_stabilize


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
    payload["segments"][0]["alignment_events"] = [
        {"type": "disfluency_candidate", "refined_start": 0.2}
    ]
    result, _ = asr_stabilize.stabilize_payload(payload, profile=2)
    assert result["metadata"] == payload["metadata"]
    assert result["unknown"] == payload["unknown"]
    assert result["segments"][0]["alignment_events"] == [
        {"type": "disfluency_candidate", "refined_start": 0.2}
    ]


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


def test_closing_phrase_ghost_is_tagged_and_dropped() -> None:
    # それではまた。 squeezed into 0.28s (>20 chars/s) — the yui-mod family
    # that survived every other leg (positive energy, mid confidence).
    ghost = _segment(
        [_word("それではまた。", 10.0, 10.28)], confidence=0.203, energy=0.7
    )
    tagged, report = asr_stabilize.stabilize_payload(_payload(ghost), profile=2)
    assert asr_stabilize.TAG_PHRASE_GHOST in tagged["segments"][0]["tags"]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 1

    stabilized, report = asr_stabilize.stabilize_payload(_payload(ghost), profile=0)
    assert stabilized["segments"] == []
    assert report.suspicious_segments_dropped == 1


def test_closing_phrase_at_normal_speed_is_kept() -> None:
    # Human-kept real occurrences (H6dTZf9QFTY): おわり as a PV voice line at
    # conf 0.24 and an end-of-stream thanks — rate is the only discriminator.
    real_owari = _segment(
        [_word("おわり", 10.0, 10.88)], confidence=0.24, energy=5.0
    )
    real_thanks = _segment(
        [_word("ありがとうございました", 20.0, 21.05)], confidence=0.999, energy=0.7
    )
    payload = _payload(real_owari, real_thanks)
    stabilized, report = asr_stabilize.stabilize_payload(payload, profile=0)
    assert [seg["text"] for seg in stabilized["segments"]] == [
        real_owari["text"],
        real_thanks["text"],
    ]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 0


def test_closing_phrase_inside_longer_sentence_is_never_a_ghost() -> None:
    # Even at an absurd squeeze, a longer sentence containing the phrase is
    # outside the whole-segment bound (collapse handling owns those).
    sentence = _segment(
        [_word("お手伝いしてくれてありがとうございました皆様", 5.0, 5.2)],
        confidence=0.7,
        energy=3.0,
    )
    tagged, _ = asr_stabilize.stabilize_payload(_payload(sentence), profile=2)
    assert asr_stabilize.TAG_PHRASE_GHOST not in tagged["segments"][0].get(
        "tags", []
    )


def test_closing_phrase_ghost_allows_trailing_fragment_chars() -> None:
    # 聴ありがとうございました (a clipped ご視聴 tail over silence) stays
    # within the +2 char bound and is squeezed — dropped.
    ghost = _segment(
        [_word("聴ありがとうございました", 100.0, 100.16)], confidence=0.73, energy=2.0
    )
    stabilized, _ = asr_stabilize.stabilize_payload(_payload(ghost), profile=0)
    assert stabilized["segments"] == []


def test_lang_switch_hallucination_is_tagged_but_never_dropped() -> None:
    ja = _segment(
        [_word("日本語のセグメントがたくさんあって全体としては日本語配信の書き起こしですこの調子で本編の会話がずっと続いていきます", 0.0, 2.0)]
    )
    suspicious = _segment(
        [_word("Thank you very much.", 3.0, 8.0)], confidence=0.4
    )
    payload = _payload(ja, suspicious)

    tagged, report = asr_stabilize.stabilize_payload(payload, profile=2)
    assert (
        asr_stabilize.TAG_LANG_SWITCH_HALLUCINATION
        in tagged["segments"][1]["tags"]
    )
    assert report.tag_counts[asr_stabilize.TAG_LANG_SWITCH_HALLUCINATION] == 1

    # Observation-only: wide-corpus review found real English lyrics/dubs and
    # translation-mode renderings of real speech among the matches, so
    # profile 0 keeps the segment, tag intact.
    stabilized, report = asr_stabilize.stabilize_payload(payload, profile=0)
    assert [seg["text"] for seg in stabilized["segments"]] == [
        ja["text"],
        suspicious["text"],
    ]
    assert report.suspicious_segments_dropped == 0
    assert (
        asr_stabilize.TAG_LANG_SWITCH_HALLUCINATION
        in stabilized["segments"][1]["tags"]
    )


def test_lang_switch_requires_low_confidence_and_enough_letters() -> None:
    ja = _segment(
        [_word("日本語のセグメントがたくさんあって全体としては日本語配信の書き起こしですこの調子で本編の会話がずっと続いていきます", 0.0, 2.0)]
    )
    confident = _segment([_word("Thank you very much.", 3.0, 8.0)], confidence=0.9)
    short = _segment([_word("Yes!", 9.0, 9.5)], confidence=0.1)
    result, _ = asr_stabilize.stabilize_payload(
        _payload(ja, confident, short), profile=0
    )
    assert [seg["text"] for seg in result["segments"]] == [
        ja["text"],
        confident["text"],
        short["text"],
    ]


def test_lang_switch_gate_stays_off_for_latin_and_bilingual_runs() -> None:
    english = _segment(
        [_word("This entire run is English speech throughout.", 0.0, 3.0)],
        confidence=0.4,
    )
    also_english = _segment(
        [_word("So low confidence alone must not drop anything.", 4.0, 7.0)],
        confidence=0.3,
    )
    result, report = asr_stabilize.stabilize_payload(
        _payload(english, also_english), profile=0
    )
    assert len(result["segments"]) == 2
    assert report.tag_counts[asr_stabilize.TAG_LANG_SWITCH_HALLUCINATION] == 0


def test_very_low_energy_drop_exempts_highly_confident_words() -> None:
    # Audited failure mode: real speech whose timeline collapsed gets its
    # energy sampled in silence; the decoder's per-word confidence is the
    # counter-evidence (drift victims measured 0.92-0.99).
    drifted = _segment(
        [_word("これ", 0.0, 0.02, confidence=0.95)], confidence=0.92, energy=-30.0
    )
    result, _ = asr_stabilize.stabilize_payload(_payload(drifted), profile=0)
    assert [seg["text"] for seg in result["segments"]] == [drifted["text"]]
    assert result["segments"][0]["tags"] == [asr_stabilize.TAG_TIME_DRIFT]


def test_very_low_energy_exemption_stops_at_the_silence_floor() -> None:
    # At the -100 dB measurement floor there is no audio at all; confident
    # hallucinations there (kaguya 音楽×5) must stay droppable.
    floored = _segment(
        [_word("音楽", 0.0, 1.0, confidence=0.99)], confidence=0.99, energy=-100.0
    )
    result, report = asr_stabilize.stabilize_payload(_payload(floored), profile=0)
    assert result["segments"] == []
    assert report.suspicious_segments_dropped == 1
