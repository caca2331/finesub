from __future__ import annotations

import pytest

import segment_split as sp


def _word(text, start, end, space_before=False, confidence=None):
    w = {"word": text, "start": start, "end": end, "space_before": space_before}
    if confidence is not None:
        w["confidence"] = confidence
    return w


def _seg(start, end, words, **extra):
    seg = {
        "start": start,
        "end": end,
        "words": words,
        "text": "".join(str(w["word"]) for w in words),
    }
    seg.update(extra)
    return seg


INTERVALS = [
    {"start": 0.0, "end": 4.0},
    {"start": 6.0, "end": 10.0},   # gap 2.0s > 0.7 -> artificial zone [4.7, 6.0]
    {"start": 10.4, "end": 14.0},  # gap 0.4s -> no artificial zone
]


def test_build_zones_only_for_gaps_beyond_kept_real_audio() -> None:
    spans = [(0.0, 4.0), (6.0, 10.0), (10.4, 14.0)]

    zones = sp.build_zones(spans)

    assert zones == [(pytest.approx(4.7), pytest.approx(6.0), 0)]


def test_no_split_segment_passes_through_unchanged() -> None:
    seg = _seg(0.0, 3.0, [_word("こんにちは", 0.0, 1.5), _word("元気", 1.6, 3.0)])

    out = sp.split_segments([seg], INTERVALS)

    assert len(out) == 1
    assert out[0] is seg  # bit-identical pass-through


def test_split_at_vad_gap_and_metric_inheritance() -> None:
    # 12s / many chars segment spanning the 2s VAD gap: must split there.
    words = (
        [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35, confidence=0.9) for i, ch in
         enumerate("今日は本当に良い天気です")]
        + [_word(ch, 6.1 + i * 0.35, 6.45 + i * 0.35, confidence=0.8) for i, ch in
           enumerate("明日も晴れると良いですね")]
    )
    seg = _seg(0.2, 10.3, words, lang="ja", confidence=0.85, no_speech_prob=0.001)

    out = sp.split_segments([seg], INTERVALS)

    assert len(out) == 2
    assert out[0]["end"] <= 4.7 + 1e-6
    assert out[1]["start"] >= 4.7 - 1e-6
    for piece in out:
        assert piece["lang"] == "ja"
        assert piece["confidence"] == 0.85
        assert piece["no_speech_prob"] == 0.001
        assert piece["words"][0]["confidence"] in (0.9, 0.8)
    # Split provenance: every piece after the first carries the segment tag.
    assert "tags" not in out[0]
    assert out[1]["tags"] == [sp.SPLIT_TAG]


def test_case3_bridge_word_defaults_right_with_uncertain_tag() -> None:
    # A word bridging the whole artificial zone with CJK glue on both sides
    # (the wt stretch pathology: 真|ん中): pass 2 can't decide, the case-3
    # default anchors it RIGHT, and the receiving piece is tagged uncertain.
    words = (
        [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35) for i, ch in
         enumerate("今日は本当に良い天気です")]
        + [_word("真", 4.45, 6.1)]  # bridges the zone [4.7, 6.0]
        + [_word(ch, 6.1 + i * 0.35, 6.45 + i * 0.35) for i, ch in
           enumerate("ん中にあるタイマーです")]
    )
    seg = _seg(0.2, 10.0, words)

    out = sp.split_segments([seg], INTERVALS)

    assert len(out) >= 2
    right = next(p for p in out if str(p["text"]).startswith("真"))
    first = right["words"][0]
    assert first["word"] == "真"
    assert first["split_adjust_case"] == "case3"
    assert first["split_anchor"] == "right"
    assert first["split_anchor_source"] == "default"
    assert sp.SPLIT_TAG in right["tags"]
    assert sp.ANCHOR_UNCERTAIN_TAG in right["tags"]
    # The bridge word folded into the right piece's lead-in.
    assert first["start"] >= 6.0 - sp.SPLIT_LEAD_IN_SEC - 1e-6
    assert first["raw_start"] == pytest.approx(4.45)


def test_gap_word_glued_right_moves_with_next_piece() -> None:
    # Artificial-zone word with a space before it and no separator after:
    # anchors right, cut lands on its left, coordinates fold into lead-in.
    words = (
        [_word(ch, 0.2 + i * 0.3, 0.5 + i * 0.3) for i, ch in
         enumerate("これはとても長い文章です")]
        + [_word("ア", 5.0, 5.4, space_before=True)]  # inside zone [4.7, 6.0]
        + [_word(ch, 6.1 + i * 0.3, 6.4 + i * 0.3) for i, ch in
           enumerate("テスと言いました本当です")]
    )
    seg = _seg(0.2, 10.0, words)

    out = sp.split_segments([seg], INTERVALS)

    assert len(out) >= 2
    right = next(p for p in out if "ア" in str(p["text"]))
    assert str(right["text"]).startswith("ア")
    gap_word = right["words"][0]
    assert gap_word["start"] >= 6.0 - sp.SPLIT_LEAD_IN_SEC - 1e-6
    assert gap_word["end"] == pytest.approx(6.0)
    # Adjustment provenance: the moved zone word keeps its case, anchor and
    # raw times; the space before ア separates it from the left, so the glue
    # pass (not the default) anchored it right.
    assert gap_word["split_adjust_case"] == "case1"
    assert gap_word["split_anchor"] == "right"
    assert gap_word["split_anchor_source"] == "glue"
    assert gap_word["raw_start"] == pytest.approx(5.0)
    assert gap_word["raw_end"] == pytest.approx(5.4)
    # Untouched real words carry no provenance fields.
    left = out[0]
    assert all(
        "split_adjust_case" not in w and "raw_start" not in w
        for w in left["words"]
    )


def test_boundary_banned_between_gap_word_and_its_anchor() -> None:
    spans = [(0.0, 4.0), (6.0, 10.0)]
    zones = sp.build_zones(spans)
    words = [
        _word("あ", 3.0, 4.0),
        _word("ん", 5.0, 5.2, space_before=True),  # zone word, glued right
        _word("次", 6.1, 6.5),
    ]
    adj = sp.adjust_words(words, spans, zones)
    boundaries = sp.score_boundaries(adj, spans, sp.DEFAULT_SPLIT_PARAMS)

    assert adj[1].anchor == 1  # anchored right
    assert not boundaries[0].banned  # cut left of the prefix allowed
    assert boundaries[1].banned  # never severed from its anchor side


def test_piece_times_clamped_to_segment_bounds() -> None:
    # Zero-duration leftover word beyond segment end must not stretch pieces.
    words = [
        _word(ch, 0.2 + i * 0.5, 0.7 + i * 0.5) for i, ch in enumerate("六字分内容")
    ] + [_word("尾", 3.6, 3.6)]
    seg = _seg(0.2, 3.2, words)

    out = sp.split_segments([seg], INTERVALS)

    for piece in out:
        assert float(piece["start"]) >= 0.2 - 1e-9
        assert float(piece["end"]) <= 3.2 + 1e-9


def test_split_params_metadata_lists_all_tunables() -> None:
    meta = sp.split_params_metadata()

    assert meta["no_gap_penalty"] == sp.DEFAULT_SPLIT_PARAMS.no_gap_penalty
    assert meta["g_knee"] == sp.DEFAULT_SPLIT_PARAMS.g_knee
    assert meta["dur_ok"] == [0.6, 8.0]
    assert meta["lead_in_sec"] == sp.SPLIT_LEAD_IN_SEC
