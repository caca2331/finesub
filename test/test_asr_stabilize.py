from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.speech.postprocessing import stabilization as asr_stabilize


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


def _noise_leg_victim(**extra: object) -> dict[str, object]:
    """A short low-confidence segment at low energy: the shape the noise leg
    drops, and therefore the shape the second-model veto rescues."""

    segment = _segment(
        [_word("どうも", 10.0, 10.5, confidence=0.2)], confidence=0.2, energy=-18.0
    )
    segment.update(extra)
    return segment


def test_a_segment_the_second_model_rescued_says_so_in_its_tags() -> None:
    """The veto used to erase its own evidence.

    Clearing the two flags left a rescued segment indistinguishable from one
    that was never suspected, which is why the veto's error rate stayed
    invisible until someone reconstructed it by counterfactual replay: of 49
    archived rescues, 8 were wrong (docs/crispasr-followups.md). The segment is
    still kept -- nothing about the decision changed -- it just records why.
    """

    dropped, report = asr_stabilize.stabilize_payload(
        _payload(_noise_leg_victim()), profile=0
    )
    assert dropped["segments"] == [], "without evidence the noise leg drops it"

    kept, report = asr_stabilize.stabilize_payload(
        _payload(_noise_leg_victim(qwen_verify={"text": "嗯。"})), profile=0
    )
    assert [seg["text"] for seg in kept["segments"]] == ["どうも"]
    assert asr_stabilize.TAG_SECOND_MODEL_VETO in kept["segments"][0]["tags"]
    assert report.tag_counts[asr_stabilize.TAG_SECOND_MODEL_VETO] == 1


def test_the_veto_tag_is_observational_and_never_drops_anything() -> None:
    """Same discipline as 语言切换幻觉: it marks, it does not delete."""

    assert asr_stabilize.TAG_SECOND_MODEL_VETO in asr_stabilize.TAG_ORDER
    kept, _ = asr_stabilize.stabilize_payload(
        _payload(_noise_leg_victim(qwen_verify={"text": "嗯。"})), profile=0
    )
    assert len(kept["segments"]) == 1


def test_an_unsuspected_segment_is_not_tagged_just_for_having_evidence() -> None:
    """Most segments carrying `qwen_verify` were never in danger.

    Tagging on "evidence exists" instead of "evidence changed the outcome"
    would put the mark on thousands of healthy segments and make it useless --
    on the archive the correct predicate fires on 49 of 8752.
    """

    healthy = _segment(
        [_word("おはよう", 3.0, 4.0, confidence=0.95)], confidence=0.95, energy=6.0
    )
    healthy["qwen_verify"] = {"text": "おはよう"}
    kept, report = asr_stabilize.stabilize_payload(_payload(healthy), profile=0)
    assert kept["segments"][0].get("tags", []) == []
    assert report.tag_counts[asr_stabilize.TAG_SECOND_MODEL_VETO] == 0


def test_the_tier_field_name_matches_the_module_that_owns_it() -> None:
    """`stabilization` spells the field itself instead of importing it.

    It has to: this is the torch-free postprocessing path, while
    `preprocessing/energy.py` — which owns the name — imports torch at module
    level. A silent copy is exactly the drift this repo has been bitten by, so
    the two spellings are pinned together here, where paying for torch is fine.
    """

    from finesub.speech.preprocessing import energy as vad_energy

    assert (
        asr_stabilize.SEGMENT_LEVEL_TIER_FIELD == vad_energy.SEGMENT_LEVEL_TIER_FIELD
    )


def test_the_veto_level_floor_is_off_unless_asked_for() -> None:
    """Default off is a decision, not an oversight: the veto is right 41 times
    out of 49, and the floor's measured cost includes one real line."""

    assert asr_stabilize.resolve_veto_level_floor() is False
    assert asr_stabilize.resolve_veto_level_floor(True) is True

    quiet = _noise_leg_victim(qwen_verify={"text": "嗯。"}, vad_level_tier="suspect")
    kept, _ = asr_stabilize.stabilize_payload(_payload(quiet), profile=0)
    assert [seg["text"] for seg in kept["segments"]] == ["どうも"]


def test_the_floor_lets_the_noise_leg_through_on_a_quiet_span() -> None:
    quiet = _noise_leg_victim(qwen_verify={"text": "嗯。"}, vad_level_tier="suspect")
    dropped, report = asr_stabilize.stabilize_payload(
        _payload(quiet), profile=0, veto_level_floor=True
    )
    assert dropped["segments"] == []
    # The veto never fired, so it must not claim it did.
    assert report.tag_counts[asr_stabilize.TAG_SECOND_MODEL_VETO] == 0


def test_the_floor_leaves_loud_spans_to_the_veto() -> None:
    """It is a floor, not a new drop rule: without a tier nothing changes.

    Older artifacts carry no `vad_level_tier` at all, so this is also what
    stops the switch from behaving differently on them than on fresh runs --
    absent field means the floor cannot fire.
    """

    loud = _noise_leg_victim(qwen_verify={"text": "嗯。"})
    kept, report = asr_stabilize.stabilize_payload(
        _payload(loud), profile=0, veto_level_floor=True
    )
    assert [seg["text"] for seg in kept["segments"]] == ["どうも"]
    assert report.tag_counts[asr_stabilize.TAG_SECOND_MODEL_VETO] == 1


def _residue_thank_you(text: str = "Thank you.", **extra: object) -> dict[str, object]:
    """The shape that escaped in the P1 fallback run: a stretched English
    boilerplate line at residue energy, confident enough to hit the
    very-low-energy exemption (measured 0.952 vs 0.632 for the flagged ones)."""

    segment = _segment(
        [_word(text, 100.0, 111.6, confidence=0.95)], confidence=0.95, energy=-63.0
    )
    segment.update(extra)
    return segment


def test_english_closing_phrase_drops_only_on_second_model_evidence() -> None:
    without = _residue_thank_you()
    kept, report = asr_stabilize.stabilize_payload(_payload(without), profile=0)
    assert [seg["text"] for seg in kept["segments"]] == ["Thank you."]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 0

    verified = _residue_thank_you(qwen_verify={"text": ""})
    dropped, report = asr_stabilize.stabilize_payload(_payload(verified), profile=0)
    assert dropped["segments"] == []
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 1


def test_english_closing_phrase_is_never_dropped_on_rate_alone() -> None:
    # 20 chars/sec is a physical impossibility in CJK but NORMAL fast English
    # (~200 wpm), so the offline rate leg must not reach the Latin family --
    # otherwise a hurried real "Thank you." is deleted with no evidence.
    squeezed = _segment(
        [_word("Thank you.", 5.0, 5.3, confidence=0.95)], confidence=0.95, energy=5.0
    )
    kept, report = asr_stabilize.stabilize_payload(_payload(squeezed), profile=0)
    assert [seg["text"] for seg in kept["segments"]] == ["Thank you."]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 0


def test_english_closing_phrase_survives_when_the_second_model_hears_it() -> None:
    real = _residue_thank_you(qwen_verify={"text": "Thank you"})
    kept, report = asr_stabilize.stabilize_payload(_payload(real), profile=0)
    assert [seg["text"] for seg in kept["segments"]] == ["Thank you."]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 0


def test_split_boilerplate_covers_the_verb_half_but_not_the_pronoun_half() -> None:
    # The re-segmentation splits the hallucinated line at the word boundary
    # (25/25 pairs exactly contiguous). "you." stays uncovered on purpose:
    # three characters would match real pronouns.
    verb = _residue_thank_you("Thank", qwen_verify={"text": ""})
    pronoun = _residue_thank_you("you.", qwen_verify={"text": ""})
    result, _ = asr_stabilize.stabilize_payload(_payload(verb, pronoun), profile=0)
    assert [seg["text"] for seg in result["segments"]] == ["you."]


def test_english_boilerplate_bypasses_the_very_low_energy_exemption() -> None:
    # The exemption itself is unchanged -- it protects drift victims. What
    # changed is that the phrase list its comment leans on now covers English.
    exempt = _residue_thank_you(qwen_verify={"text": ""})
    assert (
        asr_stabilize.weighted_word_confidence(exempt)
        > asr_stabilize.VERY_LOW_ENERGY_DROP_WORD_CONFIDENCE_EXEMPT
    )
    assert exempt["vad_weighted_energy_db"] > asr_stabilize.VERY_LOW_ENERGY_EXEMPT_FLOOR_DB
    result, _ = asr_stabilize.stabilize_payload(_payload(exempt), profile=0)
    assert result["segments"] == []


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
