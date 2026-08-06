from tools.wt_refine_validation.window_score import (
    annotate_clip_scripts,
    detector_flags,
    latin_ratio,
)


def _row(*, issues=(), events=(), coverage_low=False, segments=(), clip_cjk=True):
    return {
        "clip": "c",
        "group_index": 0,
        "group_issues": list(issues),
        "events": list(events),
        "coverage_low": coverage_low,
        "segments": list(segments),
        "_clip_dominant_cjk": clip_cjk,
    }


def test_existing_rules_parse_issue_prefixes():
    flags = detector_flags(
        _row(issues=["collapse_word_stack count=3 span=1-2", "repeating_word_run count=8"])
    )
    assert flags["E:collapse_word_stack"]
    assert flags["E:repeating_word_run"]
    assert not flags["E:long_word_duration"]


def test_decode_limit_signature_needs_unfinished_plus_bulk():
    unfinished = {"type": "unfinished", "token_count": 223}
    small_rep = {"type": "decoder_repetition", "token_count": 12, "repeat_count": 6}
    big_rep = {"type": "decoder_repetition", "token_count": 220, "repeat_count": 55}
    assert not detector_flags(_row(events=[unfinished, small_rep]))[
        "S:decode_limit_signature"
    ]
    flags = detector_flags(_row(events=[unfinished, big_rep]))
    assert flags["S:decode_limit_signature"]
    assert flags["S:decoder_repetition_big"]
    assert not detector_flags(_row(events=[big_rep]))["S:decode_limit_signature"]


def test_alignment_stack_thresholds():
    small = {"type": "alignment_stack", "token_count": 3, "tokens_per_active_frame": 3.0}
    big = {"type": "alignment_stack", "token_count": 3, "tokens_per_active_frame": 4.0}
    assert not detector_flags(_row(events=[small]))["S:alignment_stack_big"]
    assert detector_flags(_row(events=[big]))["S:alignment_stack_big"]


def test_lang_switch_needs_cjk_clip_latin_segment_and_low_conf():
    segment = {"start": 1.0, "end": 3.0, "text": "Thank you very much.", "confidence": 0.3}
    assert detector_flags(_row(segments=[segment]))["S:lang_switch_lowconf"]
    assert not detector_flags(_row(segments=[segment], clip_cjk=False))[
        "S:lang_switch_lowconf"
    ]
    confident = dict(segment, confidence=0.9)
    assert not detector_flags(_row(segments=[confident]))["S:lang_switch_lowconf"]


def test_annotate_clip_scripts_marks_cjk_majority():
    rows = [
        _row(segments=[{"text": "これは日本語のセグメントです"}]),
        _row(segments=[{"text": "short en"}]),
    ]
    for row in rows:
        row.pop("_clip_dominant_cjk")
    annotate_clip_scripts(rows)
    assert rows[0]["_clip_dominant_cjk"] is True
    assert latin_ratio("abcあいう") == 0.5
