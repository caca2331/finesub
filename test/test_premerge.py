from __future__ import annotations

import pytest

from asr_stabilize import stabilize_payload
from premerge import (
    PREMERGE_RULES_VERSION,
    PREMERGE_WORD_TAG,
    premerge_metadata,
    premerge_segments,
)
from segment_split import SPLIT_TAG


def _seg(sid, start, end, text, words=None, confidence=None):
    entry = {"id": sid, "start": start, "end": end, "text": text}
    if words is not None:
        entry["words"] = words
    if confidence is not None:
        entry["confidence"] = confidence
    return entry


def test_e1_small_kana_rejoin_and_merge_semantics() -> None:
    segments = [
        _seg(
            "1",
            0.0,
            1.9,
            "ごめんごめ",
            words=[{"word": "ごめんごめ", "space_before": False}],
            confidence=0.9,
        ),
        _seg(
            "2",
            2.4,
            3.7,
            "ん、怖がらないで",
            words=[{"word": "ん", "space_before": False}],
            confidence=0.7,
        ),
    ]
    merged, report = premerge_segments(segments)

    assert len(merged) == 1
    entry = merged[0]
    assert entry["id"] == "1"
    assert entry["text"] == "ごめんごめん、怖がらないで"
    assert entry["start"] == 0.0 and entry["end"] == 3.7
    assert [w["word"] for w in entry["words"]] == ["ごめんごめ", "ん"]
    # The junction position survives as a word tag on the absorbed side.
    assert entry["words"][0].get(PREMERGE_WORD_TAG) is None
    assert entry["words"][1][PREMERGE_WORD_TAG] is True
    assert entry["confidence"] == 0.7  # min of the pair
    assert entry["premerge_sources"] == ["1", "2"]
    assert report["rejoined"] == 1
    event = report["events"][0]
    assert event["detail"] == "E1-small-kana-start"
    assert event["merged_sources"] == ["1", "2"]
    assert event["merged_span_sec"] == 3.7


def test_e1_allows_wide_gap_and_sokuon_start() -> None:
    # Real corpus shape: 私はず / っと at a 0.2s+ gap.
    segments = [
        _seg("1", 0.0, 1.0, "私はず"),
        _seg("2", 1.21, 2.0, "っと考えてた"),
    ]
    merged, _ = premerge_segments(segments)
    assert len(merged) == 1 and merged[0]["text"] == "私はずっと考えてた"

    # But not beyond the 1.0s allowance.
    far = [
        _seg("1", 0.0, 1.0, "私はず"),
        _seg("2", 2.1, 3.0, "っと考えてた"),
    ]
    merged, _ = premerge_segments(far)
    assert len(merged) == 2


def test_quotative_and_backchannel_are_not_evidence() -> None:
    quotative = [
        _seg("1", 0.0, 1.0, "くれてありがとう"),
        _seg("2", 1.38, 2.0, "っていう点"),
    ]
    backchannel = [
        _seg("1", 0.0, 1.0, "はいルミ"),
        _seg("2", 1.1, 2.0, "ん?"),
    ]
    for segments in (quotative, backchannel):
        merged, report = premerge_segments(segments)
        assert len(merged) == 2
        assert report["events"] == []


def test_e2_single_kana_rejoin_but_not_interjections() -> None:
    # おわ / り with a wide gap is the canonical missed word cut from the
    # v1 evaluation.
    word_cut = [
        _seg("1", 0.0, 0.4, "おわ"),
        _seg("2", 1.23, 1.5, "り"),
    ]
    merged, report = premerge_segments(word_cut)
    assert len(merged) == 1
    assert merged[0]["text"] == "おわり"
    assert report["events"][0]["detail"] == "E2-single-kana"

    interjection = [
        _seg("1", 0.0, 1.0, "どうかな"),
        _seg("2", 1.1, 1.5, "あ"),
    ]
    merged, _ = premerge_segments(interjection)
    assert len(merged) == 2


def test_weak_junctions_never_merge() -> None:
    # v1's failure mode: unpunctuated sentence boundary at gap=0.
    sentences = [
        _seg("1", 0.0, 3.0, "さあ彼女の冒険ノートを開いてみよう"),
        _seg("2", 3.0, 6.0, "今日のページに綴られるのは"),
    ]
    merged, report = premerge_segments(sentences)
    assert len(merged) == 2
    assert report["events"] == []
    assert report["skipped_no_evidence"] >= 1


def test_rejoin_vetoes_and_shape_guard_are_audited() -> None:
    vetoed = [
        _seg("1", 0.0, 1.0, "そうなんです"),
        _seg("2", 1.1, 2.0, "っす"),
    ]
    merged, report = premerge_segments(vetoed)
    assert len(merged) == 2
    assert report["rejected"][0]["detail"] == "vetoed:left-terminal-form"

    # Merged span over 7s: evidence fires but the shape guard blocks.
    long_span = [
        _seg("1", 0.0, 6.5, "ここまでの流れをまとめるとやっぱりこれがず"),
        _seg("2", 6.6, 7.8, "っと続いてた"),
    ]
    merged, report = premerge_segments(long_span)
    assert len(merged) == 2
    assert report["rejected"][0]["detail"] == "blocked:over-max-duration"


def test_filler_attachment_is_direction_typed() -> None:
    forward = [
        _seg("1", 0.0, 1.0, "えっと、"),
        _seg("2", 1.1, 2.0, "メメみたいな"),
    ]
    merged, report = premerge_segments(forward)
    assert len(merged) == 1
    assert merged[0]["text"] == "えっと、メメみたいな"
    assert merged[0]["id"] == "1"
    assert report["events"][0]["detail"] == "filler-forward"

    backward = [
        _seg("1", 0.0, 1.0, "座っておしゃべりしようよ"),
        _seg("2", 1.1, 2.0, "ね"),
    ]
    merged, report = premerge_segments(backward)
    assert len(merged) == 1
    assert report["events"][0]["detail"] == "filler-backward"

    # A leading filler never attaches backward onto preceding content.
    leading_at_end = [
        _seg("1", 0.0, 1.0, "やばい今の"),
        _seg("2", 1.1, 2.0, "ちょっと"),
    ]
    # Response words are not fillers at all.
    response = [
        _seg("1", 0.0, 1.0, "してくれただけ"),
        _seg("2", 1.1, 2.0, "うん"),
    ]
    # Filler + filler stays apart.
    fillers = [
        _seg("1", 0.0, 1.0, "あの"),
        _seg("2", 1.1, 2.0, "えっと"),
    ]
    # Over the filler gap allowance.
    slow = [
        _seg("1", 0.0, 1.0, "えっと"),
        _seg("2", 1.25, 2.0, "メメみたいな"),
    ]
    for segments in (leading_at_end, response, fillers, slow):
        merged, report = premerge_segments(segments)
        assert len(merged) == 2
        assert report["filler_attached"] == 0


def test_premerge_metadata_carries_rules_version() -> None:
    meta = premerge_metadata()
    assert meta["rules_version"] == PREMERGE_RULES_VERSION
    assert meta["rejoin_max_gap_sec"] == 1.0
    assert meta["filler_max_gap_sec"] == 0.2
    assert meta["max_merged_duration_sec"] == 7.0
    assert meta["max_merged_weighted_chars"] == 36.0
    assert "うん" not in meta["leading_fillers"]
    assert "でも" not in meta["leading_fillers"]


def test_split_tag_structurally_blocks_premerge() -> None:
    segments = [
        _seg("1", 0.0, 0.4, "おわ"),
        {**_seg("2", 1.2, 1.5, "り"), "tags": [SPLIT_TAG]},
    ]
    merged, report = premerge_segments(segments)
    assert len(merged) == 2
    assert report["events"] == []


def test_merge_unions_segment_tags_except_positional_split_tag() -> None:
    segments = [
        {**_seg("1", 0.0, 0.4, "おわ"), "tags": ["时间漂移"]},
        {**_seg("2", 1.2, 1.5, "り"), "tags": ["高度疑似幻觉"]},
    ]
    merged, _ = premerge_segments(segments)
    assert len(merged) == 1
    assert merged[0]["tags"] == ["时间漂移", "高度疑似幻觉"]


def test_stabilize_profile_0_applies_premerge_last_with_metadata() -> None:
    payload = {
        "segments": [
            {
                "start": 0.0,
                "end": 0.4,
                "text": "おわ",
                "words": [{"word": "おわ", "start": 0.0, "end": 0.4,
                           "space_before": False}],
            },
            {
                "start": 1.2,
                "end": 1.5,
                "text": "り",
                "words": [{"word": "り", "start": 1.2, "end": 1.5,
                           "space_before": False}],
            },
            {
                "start": 3.0,
                "end": 4.0,
                "text": "つぎの文です。",
                "words": [{"word": "つぎの文です。", "start": 3.0, "end": 4.0,
                           "space_before": False}],
            },
        ]
    }
    result, report = stabilize_payload(payload, profile=0)

    segments = result["segments"]
    assert [seg["text"] for seg in segments] == ["おわり", "つぎの文です。"]
    assert segments[0]["words"][1][PREMERGE_WORD_TAG] is True
    assert "id" not in segments[0]  # synthetic audit ids stay internal
    assert segments[0]["premerge_sources"] == [1, 2]
    assert report.premerge_rejoined == 1
    assert report.applied_profiles == (1, 3, 2)
    assert result["metadata"]["premerge"]["rules_version"] == (
        PREMERGE_RULES_VERSION
    )

    # Profile 3 alone also premerges; profile 2 alone does not.
    result3, report3 = stabilize_payload(payload, profile=3)
    assert len(result3["segments"]) == 2
    assert report3.premerge_rejoined == 1
    result2, _ = stabilize_payload(payload, profile=2)
    assert len(result2["segments"]) == 3
    assert "premerge" not in (result2.get("metadata") or {})
