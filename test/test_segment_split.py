from __future__ import annotations

import pytest

from asr_playground.speech.postprocessing import segmentation as sp


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
    # A segment the DP leaves whole keeps its own coordinates: the gap-word
    # adjustment stays virtual. Only the seam marker is added.
    assert {k: v for k, v in out[0].items() if k != "words"} == {
        k: v for k, v in seg.items() if k != "words"
    }
    assert [
        {k: v for k, v in w.items() if k != sp.WHISPER_SEGMENT_WORD_TAG}
        for w in out[0]["words"]
    ] == seg["words"]
    assert out[0]["words"][0][sp.WHISPER_SEGMENT_WORD_TAG] is True


def test_text_only_segment_gets_one_marked_synthetic_word() -> None:
    segment = {"start": 0.2, "end": 1.8, "text": "字幕だけ", "lang": "ja"}

    out = sp.split_segments([segment], INTERVALS)

    assert len(out) == 1
    assert out[0]["text"] == "字幕だけ"
    assert out[0]["lang"] == "ja"
    assert out[0]["words"] == [
        {
            "word": "字幕だけ",
            "start": 0.2,
            "end": 1.8,
            sp.SYNTHETIC_WORD_KEY: True,
        }
    ]
    assert "words" not in segment


def test_text_only_segment_is_preserved_among_dp_eligible_segments() -> None:
    text_only = {"start": 0.2, "end": 0.8, "text": "先頭"}
    normal = _seg(
        1.0,
        3.0,
        [_word("通常", 1.0, 1.8), _word("字幕", 2.0, 3.0)],
    )

    out = sp.split_segments([text_only, normal], INTERVALS)

    assert [segment["text"] for segment in out] == ["先頭", "通常字幕"]
    assert out[0]["words"][0][sp.SYNTHETIC_WORD_KEY] is True


def test_split_at_vad_gap_and_metric_inheritance() -> None:
    # 12s / many chars segment spanning the 2s VAD gap: must split there.
    words = (
        [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35, confidence=0.9) for i, ch in
         enumerate("今日は本当に良い天気です")]
        + [_word(ch, 6.1 + i * 0.35, 6.45 + i * 0.35, confidence=0.8) for i, ch in
           enumerate("明日も晴れると良いですね")]
    )
    seg = _seg(0.2, 10.3, words, lang="ja", confidence=0.85, no_speech_prob=0.001)
    seg["alignment_events"] = [
        {"type": "alignment_stack", "start": 1.0, "end": 1.2},
        {"type": "disfluency_candidate", "refined_start": 7.0},
    ]

    out = sp.split_segments([seg], INTERVALS)

    assert len(out) == 2
    assert out[0]["end"] <= 4.7 + 1e-6
    assert out[1]["start"] >= 4.7 - 1e-6
    for piece in out:
        assert piece["lang"] == "ja"
        assert piece["confidence"] == 0.85
        assert piece["no_speech_prob"] == 0.001
        assert piece["words"][0]["confidence"] in (0.9, 0.8)
    # Provenance is inverted: the piece the DP cut inside a Whisper segment is
    # the one that gets tagged; the seam marker sits on the source's first word.
    assert "tags" not in out[0]
    assert out[1]["tags"] == [sp.MID_SEGMENT_TAG]
    assert out[0]["words"][0][sp.WHISPER_SEGMENT_WORD_TAG] is True
    assert sp.WHISPER_SEGMENT_WORD_TAG not in out[1]["words"][0]
    assert out[0]["alignment_events"] == [
        {"type": "alignment_stack", "start": 1.0, "end": 1.2}
    ]
    assert out[1]["alignment_events"] == [
        {"type": "disfluency_candidate", "refined_start": 7.0}
    ]
    assert sum(len(piece.get("alignment_events") or []) for piece in out) == 2


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
    assert sp.MID_SEGMENT_TAG in right["tags"]
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


def _pause_boundary(left_end, right_start, params=None):
    """Score one boundary with both words inside interval 0 (no VAD gap)."""
    spans = [(0.0, 14.0)]
    words = [_word("あ", 0.5, left_end), _word("い", right_start, right_start + 0.2)]
    adj = sp.adjust_words(words, spans, sp.build_zones(spans))
    return sp.boundary_score(adj[0], adj[1], spans, params or sp.DEFAULT_SPLIT_PARAMS)


def test_non_vad_gap_zero_pause_keeps_the_old_penalty() -> None:
    p = sp.DEFAULT_SPLIT_PARAMS
    b = _pause_boundary(2.0, 2.0)

    assert b.non_vad_gap
    # Degenerates exactly to the old step penalty: a*(t + penalty) + base.
    assert b.b == pytest.approx(p.a * (b.t + p.non_vad_gap_penalty) + p.base)


def test_non_vad_gap_word_pause_waives_the_penalty() -> None:
    p = sp.DEFAULT_SPLIT_PARAMS
    b = _pause_boundary(2.0, 2.3)

    assert b.non_vad_gap
    # pause = 0.3 -> shape = -0.09, strictly cheaper than the zero-pause cost.
    assert b.b == pytest.approx(p.a * (b.t - 0.3 * 0.3) + p.base)
    assert b.b < _pause_boundary(2.0, 2.0).b


def test_non_vad_gap_pause_is_clamped_at_the_floor() -> None:
    p = sp.DEFAULT_SPLIT_PARAMS
    b = _pause_boundary(2.0, 6.0)  # pause 4.0s, -pause^2 = -16 without the floor

    assert b.b == pytest.approx(p.a * (b.t - 1.0) + p.base)


def test_vad_gap_uses_g_score_even_when_the_gap_measures_zero() -> None:
    # Abutting intervals: the words sit in different intervals but the silence
    # between them is 0. The predicate is "is this a VAD gap", not "g > 0", so
    # this must NOT fall back to the word pause (which is 0 here and would
    # otherwise collect the penalty).
    spans = [(0.0, 4.0), (4.0, 8.0)]
    words = [_word("あ", 3.5, 4.0), _word("い", 4.0, 4.5)]
    adj = sp.adjust_words(words, spans, sp.build_zones(spans))
    assert adj[1].anchor > adj[0].anchor

    b = sp.boundary_score(adj[0], adj[1], spans, sp.DEFAULT_SPLIT_PARAMS)

    p = sp.DEFAULT_SPLIT_PARAMS
    assert not b.non_vad_gap
    assert b.g == pytest.approx(0.0)
    assert b.b == pytest.approx(p.a * b.t + p.base)   # no penalty


def test_max_piece_sec_does_not_change_a_normal_split() -> None:
    import dataclasses

    words = [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35) for i, ch in
             enumerate("今日は本当に良い天気ですね明日も晴れると良いです")]
    seg = _seg(0.2, 9.0, words)
    unbounded = dataclasses.replace(sp.DEFAULT_SPLIT_PARAMS, max_piece_sec=1e9)

    bounded = sp.split_segments([seg], INTERVALS)
    loose = sp.split_segments([seg], INTERVALS, params=unbounded)

    assert [p["text"] for p in bounded] == [p["text"] for p in loose]


def test_cross_segment_piece_inherits_by_word_weight_and_union_clamp() -> None:
    # Two adjacent ASR segments inside one VAD interval with no pause between
    # them: the seam bonus is not enough to keep a cut this cheap, so the DP
    # merges them into one piece spanning both sources.
    left = _seg(10.5, 11.2, [_word(ch, 10.5 + i * 0.2, 10.7 + i * 0.2) for i, ch in
                             enumerate("あいう")],
                lang="ja", confidence=0.9, no_speech_prob=0.02)
    right = _seg(11.1, 11.6, [_word(ch, 11.1 + i * 0.2, 11.3 + i * 0.2) for i, ch in
                              enumerate("えお")],
                 lang="ja", confidence=0.4, no_speech_prob=0.06)

    out = sp.split_segments([left, right], INTERVALS)

    merged = next(p for p in out if p["text"] == "あいうえお")
    # 3 words at 0.9 and 2 at 0.4 -> weighted mean, not min/max/first.
    assert merged["confidence"] == pytest.approx((3 * 0.9 + 2 * 0.4) / 5)
    assert merged["no_speech_prob"] == pytest.approx((3 * 0.02 + 2 * 0.06) / 5)
    assert merged["lang"] == "ja"
    # Union clamp: the piece may span both sources' bounds, not just one's.
    assert merged["start"] >= 10.5 - 1e-9
    assert merged["end"] <= 11.6 + 1e-9
    # Both source segments' first words stay marked, so the swallowed seam is
    # still recoverable from the piece's interior.
    assert [i for i, w in enumerate(merged["words"])
            if w.get(sp.WHISPER_SEGMENT_WORD_TAG)] == [0, 3]


def test_single_source_piece_inherits_exactly_as_before() -> None:
    seg = _seg(0.2, 3.2, [_word(ch, 0.2 + i * 0.5, 0.7 + i * 0.5) for i, ch in
                          enumerate("六字分内容")],
               lang="ja", confidence=0.85, no_speech_prob=0.001)

    out = sp.split_segments([seg], INTERVALS)

    for piece in out:
        assert piece["lang"] == "ja"
        assert piece["confidence"] == 0.85
        assert piece["no_speech_prob"] == 0.001


def test_regroup_restores_the_original_segmentation_and_coordinates() -> None:
    words = (
        [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35) for i, ch in
         enumerate("今日は本当に良い天気です")]
        + [_word("真", 4.45, 6.1)]
        + [_word(ch, 6.1 + i * 0.35, 6.45 + i * 0.35) for i, ch in
           enumerate("ん中にあるタイマーです")]
    )
    seg = _seg(0.2, 10.0, words)

    out = sp.split_segments([seg], INTERVALS)
    regrouped = sp.regroup_by_whisper_segments(out)

    assert len(regrouped) == 1
    assert regrouped[0]["text"] == seg["text"]
    assert [
        (w["word"], pytest.approx(w["start"]), pytest.approx(w["end"]))
        for w in regrouped[0]["words"]
    ] == [(w["word"], pytest.approx(w["start"]), pytest.approx(w["end"])) for w in words]


def test_split_is_idempotent_on_its_own_output() -> None:
    words = (
        [_word(ch, 0.2 + i * 0.35, 0.55 + i * 0.35) for i, ch in
         enumerate("今日は本当に良い天気です")]
        + [_word("真", 4.45, 6.1)]
        + [_word(ch, 6.1 + i * 0.35, 6.45 + i * 0.35) for i, ch in
           enumerate("ん中にあるタイマーです")]
    )
    seg = _seg(0.2, 10.0, words, lang="ja", confidence=0.7)

    once = sp.split_segments([seg], INTERVALS)
    twice = sp.split_segments(once, INTERVALS)

    assert twice == once


def test_split_params_metadata_lists_all_tunables() -> None:
    meta = sp.split_params_metadata()

    assert meta["non_vad_gap_penalty"] == sp.DEFAULT_SPLIT_PARAMS.non_vad_gap_penalty
    assert meta["whisper_segment_bonus"] == sp.DEFAULT_SPLIT_PARAMS.whisper_segment_bonus
    assert meta["max_piece_sec"] == sp.DEFAULT_SPLIT_PARAMS.max_piece_sec
    assert meta["g_knee"] == sp.DEFAULT_SPLIT_PARAMS.g_knee
    assert meta["dur_ok"] == [0.6, 8.0]
    assert meta["lead_in_sec"] == sp.SPLIT_LEAD_IN_SEC
