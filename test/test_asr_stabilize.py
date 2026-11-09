from __future__ import annotations

from finesub import settings

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
    # Every entry the input carried survives. Stabilize may add its own --
    # this fixture has no `audio_duration`, so the overrun dimension goes
    # unchecked and says so.
    assert result["metadata"].items() >= payload["metadata"].items()
    assert result["metadata"]["asr_stabilize"] == {"audio_end_unchecked": True}
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

    assert settings.VETO_LEVEL_FLOOR.resolve() is False
    assert settings.VETO_LEVEL_FLOOR.resolve(True) is True

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


def test_closing_phrase_survives_when_the_second_model_hears_other_words() -> None:
    """Same reading as the noise-leg veto: anything heard keeps the line. On
    short spans Qwen writes real thanks in the wrong words or language (MAI2
    heard "Thank you." where Qwen wrote this)."""

    misheard = _residue_thank_you(qwen_verify={"text": "Tinggi.", "language": "Indonesian"})
    kept, report = asr_stabilize.stabilize_payload(_payload(misheard), profile=0)
    assert [seg["text"] for seg in kept["segments"]] == ["Thank you."]
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 0


@pytest.mark.parametrize("heard", ["", "Tinggi."])
def test_the_floor_drops_a_quiet_stock_phrase_whatever_the_referee_wrote(heard: str) -> None:
    """One reading for both consumers: under the floor what Qwen heard on a
    quiet span counts as nothing -- the noise legs are not withdrawn, and the
    stock phrase drops, which is the direction the floor exists for."""

    quiet = _residue_thank_you(qwen_verify={"text": heard}, vad_level_tier="suspect")
    dropped, report = asr_stabilize.stabilize_payload(
        _payload(quiet), profile=0, veto_level_floor=True
    )
    assert dropped["segments"] == []
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 1


def test_punctuation_only_evidence_reads_as_heard_nothing() -> None:
    """The reading normalizes before it asks: "。" is no speech, for the
    stock-phrase leg as for the noise-leg veto."""

    ghost = _residue_thank_you(qwen_verify={"text": "。"})
    dropped, report = asr_stabilize.stabilize_payload(_payload(ghost), profile=0)
    assert dropped["segments"] == []
    assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 1


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


def test_stable_output_normalizes_a_timeline_the_aligned_input_had_wrong() -> None:
    # A stable JSON can be rebuilt from an aligned artifact produced before the
    # recognition-stage fixes -- or by hand -- so the contract has to hold on
    # this write path too. Downstream cannot tell the difference: it just drops
    # `end <= start` without saying so.
    collapsed = _segment([_word("お", 1.0, 1.0)], start=1.0, end=1.0)
    stranded = _segment([_word("あ", 2.0, 3.4)], start=2.0, end=3.0)
    following = _segment([_word("い", 3.1, 4.0)])

    result, report = asr_stabilize.stabilize_payload(
        _payload(collapsed, stranded, following), profile=0
    )

    spans = [(seg["start"], seg["end"]) for seg in result["segments"]]
    assert spans == [(1.0, 1.01), (2.0, 3.1), (3.1, 4.0)]
    for segment in result["segments"]:
        for word in segment["words"]:
            assert segment["start"] <= word["start"]
            assert word["end"] <= segment["end"]
    kinds = [
        record["kind"]
        for record in result["metadata"]["asr_stabilize"]["segment_time_repairs"]
    ]
    assert kinds == ["contain", "zero-span"]
    assert report.segment_time_repairs == 2


# --- the rebuild path's audio-end dimension (P1) ---------------------------


def test_rebuild_clamps_a_segment_past_the_recorded_audio_duration() -> None:
    payload = _payload(_segment([_word("末尾", 269.0, 272.68)]))
    payload["metadata"]["asr_align"]["audio_duration"] = 270.814

    result, report = asr_stabilize.stabilize_payload(payload, profile=2)

    assert result["segments"][-1]["end"] == pytest.approx(270.814)
    assert float(result["segments"][-1]["words"][-1]["end"]) <= 270.814
    repairs = result["metadata"]["asr_stabilize"]["segment_time_repairs"]
    assert [r["kind"] for r in repairs] == ["overrun"]
    assert report.segment_time_repairs == 1
    assert "audio_end_unchecked" not in result["metadata"]["asr_stabilize"]


def test_rebuild_without_a_recorded_duration_says_it_did_not_check() -> None:
    """"Did not overrun" and "nobody measured" must not look alike.

    Measured on six delivered packs: one carries nothing but `segments`, the
    other five carry `asr_align` but predate the duration field. Repairing
    those would need the audio, which this path does not have -- so it says
    so instead of staying silent.
    """

    payload = _payload(_segment([_word("末尾", 269.0, 272.68)]))

    result, report = asr_stabilize.stabilize_payload(payload, profile=2)

    assert result["segments"][-1]["end"] == pytest.approx(272.68)
    assert result["metadata"]["asr_stabilize"]["audio_end_unchecked"] is True
    assert report.segment_time_repairs == 0


def test_the_clamped_last_cue_survives_rendering(tmp_path: Path) -> None:
    """Clamping the stable JSON alone is not enough.

    The render pass pads every cue end by 0.3 s, and the last cue has no next
    start to stop it -- so a segment clamped exactly to the audio end comes
    back out past it unless the pad is bounded too. This is the one assertion
    that covers both halves; each half has its own unit tests elsewhere.
    """

    from finesub.subtitles import rendering
    from finesub.subtitles.model import parse_srt
    from finesub.subtitles.postprocess import (
        TIMELINE_POSTPROCESS_PROFILES,
        postprocess_srt_file,
    )

    audio_end = 270.814
    payload = _payload(_segment([_word("末尾", 269.0, 272.68)]))
    payload["metadata"]["asr_align"]["audio_duration"] = audio_end
    stable, _ = asr_stabilize.stabilize_payload(payload, profile=2)

    stable_path = tmp_path / "clip-stable.json"
    stable_path.write_text(json.dumps(stable), encoding="utf-8")
    srt_path = rendering.convert_json_to_srt(
        stable_path, output_path=tmp_path / "clip.srt", validate=False
    ).path
    for profile in TIMELINE_POSTPROCESS_PROFILES:
        postprocess_srt_file(
            srt_path, profile=profile, validate=False, audio_end=audio_end
        )

    cues = parse_srt(srt_path.read_text(encoding="utf-8"))
    assert cues[-1].end == pytest.approx(audio_end)


# --- what the FINAL segment set leaves unclaimed (P3) ----------------------


def test_audio_end_repair_keeps_every_tail_cue_renderable() -> None:
    from finesub.subtitles.rendering import render_segment_srt
    from finesub.subtitles.model import parse_srt

    payload = _payload(
        _segment([_word("first", 8.0, 10.0)]),
        _segment([_word("second", 11.0, 12.0)]),
        _segment([_word("third", 13.0, 14.0)]),
    )
    payload["metadata"] = {"asr_align": {"audio_duration": 10.0}}
    stable, _ = asr_stabilize.stabilize_payload(payload, profile=2)
    cues = parse_srt(render_segment_srt(stable["segments"]))

    assert [cue.text for cue in cues] == ["first", "second", "third"]
    assert all(0 <= cue.start < cue.end <= 10.0 for cue in cues)
    assert all(left.end <= right.start for left, right in zip(cues, cues[1:]))
    from finesub.speech.recognition.segments import normalize_segment_timeline

    again, repairs = normalize_segment_timeline(stable["segments"], audio_end=10.0)
    assert again == stable["segments"]
    assert repairs == []


def _timeline(*intervals: tuple[float, float]) -> dict[str, object]:
    return {
        "intervals": [{"start": start, "end": end} for start, end in intervals],
        "pause_hints": {"scorer": [], "padding": []},
    }


def test_decode_coverage_can_be_clean_while_the_stable_still_has_a_gap() -> None:
    """The two measurements answer different questions, and must disagree here.

    The decode stage checks coverage per group, before stabilize runs. A group
    can pass that check and still lose a line afterwards -- which is why
    recording only the decode-side number would leave the same blind spot the
    whole exercise is about.
    """

    from finesub.speech.recognition import transcribe

    intervals = [{"start": 0.0, "end": 20.0}]
    decoded = [
        _segment([_word("前半", 0.0, 10.0)]),
        _segment([_word("ご視聴ありがとうございました。", 12.0, 20.0)], energy=-30.0),
    ]

    # Decode side: the group is fully covered, nothing to report.
    assert transcribe._coverage_shortfall(intervals, decoded) is None

    payload = _payload(*decoded)
    payload["vad_timeline"] = _timeline((0.0, 20.0))
    result, _ = asr_stabilize.stabilize_payload(payload, profile=0)

    # ...and stabilize dropped the hallucinated tail, so the viewer lost 8 s.
    assert len(result["segments"]) == 1
    gaps = result["metadata"]["asr_stabilize"]["speech_gaps"]
    assert gaps["uncovered_sec"] == pytest.approx(10.0)
    assert gaps["pieces"] == 1
    assert gaps["spans"] == [[10.0, 20.0]]


def test_fragmented_loss_is_visible_even_though_no_piece_is_probe_worthy() -> None:
    """The form that fell between the two existing mechanisms.

    Several 1-2 s pieces: too small for the referee's `GAP_MIN_SEC`, too small
    to move a 30 s group's coverage ratio. Their total is what says something
    went missing.
    """

    from finesub.speech.verification import qwen_referee

    intervals = [{"start": 0.0, "end": 12.0}]
    kept = [
        _segment([_word("a", 0.0, 3.0)]),
        _segment([_word("b", 4.5, 6.0)]),
        _segment([_word("c", 8.0, 12.0)]),
    ]

    assert qwen_referee.collect_gaps(intervals, kept) == []

    payload = _payload(*kept)
    payload["vad_timeline"] = _timeline((0.0, 12.0))
    result, _ = asr_stabilize.stabilize_payload(payload, profile=2)

    gaps = result["metadata"]["asr_stabilize"]["speech_gaps"]
    assert gaps["pieces"] == 2
    assert gaps["uncovered_sec"] == pytest.approx(3.5)
    assert gaps["longest_piece_sec"] == pytest.approx(2.0)


def test_a_payload_without_a_vad_timeline_reports_nothing_rather_than_zero() -> None:
    """"No gaps" and "no yardstick" are different answers."""

    payload = _payload(_segment([_word("孤立", 0.0, 1.0)]))

    result, _ = asr_stabilize.stabilize_payload(payload, profile=2)

    stabilize_meta = result["metadata"].get("asr_stabilize") or {}
    assert "speech_gaps" not in stabilize_meta


# --- the two feedback-pack windows, as shapes (P6) -------------------------
# The timing relations are the part worth keeping; the text is synthetic on
# purpose. Turning private subtitles into a string does not make them ours,
# and every assertion below is about geometry, not words.
#
# Measured from the delivered artifact (docs/reports/2026-09-07-...):
# the #33 window lost two whole VAD intervals, 1.505 s and 1.925 s, with real
# silence -- an interval boundary -- between them.

CASE_33_INTERVALS = (
    (236.625, 237.930),
    (240.775, 242.280),
    (243.995, 245.920),
    (246.215, 247.560),
)
CASE_33_UNCLAIMED = ((240.775, 242.280), (243.995, 245.920))


def test_the_case_33_shape_is_two_whole_intervals_nobody_claimed() -> None:
    """Both existing mechanisms are blind to it, and the summary is not.

    ⚠ The pieces sit in DIFFERENT VAD intervals, each filling its own
    completely. That is why "add up the pieces inside one interval" -- the
    first idea for widening the referee's criterion -- would still miss this
    window (bench-baselines §26).
    """

    from finesub.speech.recognition.segments import speech_gap_summary
    from finesub.speech.verification import qwen_referee

    intervals = [
        {"start": start, "end": end} for start, end in CASE_33_INTERVALS
    ]
    claimed = [
        _segment([_word("あ", start + 0.05, end - 0.05)])
        for start, end in CASE_33_INTERVALS
        if (start, end) not in CASE_33_UNCLAIMED
    ]

    # No single uncovered run reaches GAP_MIN_SEC, so the referee probes none.
    assert qwen_referee.collect_gaps(intervals, claimed) == []

    summary = speech_gap_summary(intervals, claimed)
    # Two whole intervals, plus a 0.05 s edge either side of each claimed one.
    assert summary["pieces"] == 6
    assert summary["uncovered_sec"] == pytest.approx(1.505 + 1.925 + 0.2, abs=1e-3)
    assert summary["longest_piece_sec"] == pytest.approx(1.925, abs=1e-3)
    assert [240.775, 242.28] in summary["spans"]
    assert [243.995, 245.92] in summary["spans"]


def test_the_case_22_shape_is_not_a_gap_at_all() -> None:
    """#22 lost no time -- one cue carried the neighbour's text.

    Pinned as a boundary: a coverage measurement cannot see this class of
    defect, so nothing here should be read as covering it. The check that
    would is a content-level correspondence one, which does not exist yet.
    """

    from finesub.speech.recognition.segments import speech_gap_summary

    intervals = [{"start": 125.9, "end": 134.0}]
    # Same spans either way; only which text sits on which cue differs.
    correct = [
        _segment([_word("甲", 125.9, 129.0)]),
        _segment([_word("乙", 129.0, 134.0)]),
    ]
    misassigned = [
        _segment([_word("乙", 125.9, 129.0)]),
        _segment([_word("丙", 129.0, 134.0)]),
    ]

    assert speech_gap_summary(intervals, correct) == speech_gap_summary(
        intervals, misassigned
    )
    assert speech_gap_summary(intervals, misassigned)["uncovered_sec"] == 0.0


def _lang(segment: dict[str, object], lang: str) -> dict[str, object]:
    return {**segment, "lang": lang}


def test_mandarin_segments_are_written_in_simplified() -> None:
    zh = _lang(_segment([_word("然後", 0.0, 0.4), _word("我們", 0.4, 0.8), _word("說", 0.8, 1.0)]), "zh")
    simplified = _lang(_segment([_word("这个", 2.0, 2.4), _word("温暖", 2.4, 2.8)]), "zh")
    result, report = asr_stabilize.stabilize_payload(_payload(zh, simplified), profile=1)

    first, second = result["segments"]
    assert first["text"] == "然后我们说"
    assert [w["word"] for w in first["words"]] == ["然后", "我们", "说"]
    assert second["text"] == "这个温暖"
    assert report.simplified_segments == 1
    assert report.to_dict()["simplified_segments"] == 1


def test_only_mandarin_segments_are_converted() -> None:
    """Japanese kanji have simplified look-alikes (見 -> 见) and Cantonese is left as heard."""

    ja = _lang(_segment([_word("見て", 0.0, 0.4), _word("時間", 0.4, 0.8)]), "ja")
    yue = _lang(_segment([_word("點解", 2.0, 2.4)]), "yue")
    unlabelled = _segment([_word("這個", 4.0, 4.4)])
    result, report = asr_stabilize.stabilize_payload(_payload(ja, yue, unlabelled), profile=1)

    assert [s["text"] for s in result["segments"]] == ["見て時間", "點解", "這個"]
    assert report.simplified_segments == 0


def test_unlabelled_rows_follow_the_episode_language() -> None:
    """A row the gap fill or the recall repair inserted carries no `lang`: in a Mandarin episode (the rows'
    `lang`, or what ASR was told) it is converted too."""

    zh = _lang(_segment([_word("然後", 0.0, 0.4)]), "zh")
    unlabelled = _segment([_word("這個", 2.0, 2.4)])
    result, _ = asr_stabilize.stabilize_payload(_payload(zh, unlabelled), profile=1)
    assert [s["text"] for s in result["segments"]] == ["然后", "这个"]

    told = {**_payload(_segment([_word("這個", 0.0, 0.4)])), "metadata": {"asr_align": {"language": "zh"}}}
    result, _ = asr_stabilize.stabilize_payload(told, profile=1)
    assert [s["text"] for s in result["segments"]] == ["这个"]

    # a group whose language went undetected is written `"None"` (transcribe): unset too
    undetected = _lang(_segment([_word("這個", 2.0, 2.4)]), "None")
    result, _ = asr_stabilize.stabilize_payload(_payload(zh, undetected), profile=1)
    assert [s["text"] for s in result["segments"]] == ["然后", "这个"]


def test_verbatim_profile_writes_the_aligned_bytes(tmp_path: Path) -> None:
    """Profile -1 (and joint transcription, which uses it) copies the aligned file: no conversion."""

    source = tmp_path / "x-aligned.json"
    source.write_text(
        json.dumps(_payload(_lang(_segment([_word("這個", 0.0, 0.4)]), "zh")), ensure_ascii=False),
        encoding="utf-8",
    )
    output, _ = asr_stabilize.stabilize_json_file(source, profile=-1)

    assert output.read_bytes() == source.read_bytes()
