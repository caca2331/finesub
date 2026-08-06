import json

from tools.wt_refine_validation.artifact_survey import (
    discover_pairs,
    match_stable,
    survey_pair,
)


def _segment(start, end, text, *, events=None, words=None):
    return {
        "start": start,
        "end": end,
        "text": text,
        "confidence": 0.9,
        "words": words
        if words is not None
        else [{"word": text, "start": start, "end": end, "confidence": 0.9}],
        **({"alignment_events": events} if events else {}),
    }


def test_match_stable_prefers_best_overlap_and_reports_drop():
    stable = [
        {"start": 0.0, "end": 1.0, "text": "a"},
        {"start": 1.0, "end": 3.0, "text": "b"},
    ]
    assert match_stable({"start": 1.2, "end": 2.8}, stable) is stable[1]
    assert match_stable({"start": 10.0, "end": 11.0}, stable) is None


def test_survey_pair_skips_artifacts_without_signals(tmp_path):
    aligned = tmp_path / "x-aligned.json"
    aligned.write_text(
        json.dumps({"segments": [_segment(0.0, 1.0, "ok")]}), encoding="utf-8"
    )
    assert survey_pair(aligned, None) is None


def test_survey_pair_cross_tabulates_outcomes(tmp_path):
    events = [{"type": "unfinished", "token_count": 223}]
    aligned = tmp_path / "x-aligned.json"
    stable = tmp_path / "x-stable.json"
    aligned.write_text(
        json.dumps(
            {
                "segments": [
                    _segment(0.0, 1.0, "clean"),
                    _segment(2.0, 3.0, "signal", events=events),
                    _segment(5.0, 6.0, "gone"),
                ]
            }
        ),
        encoding="utf-8",
    )
    stable.write_text(
        json.dumps(
            {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "clean"},
                    {"start": 2.0, "end": 3.0, "text": "signal", "tags": ["时间漂移"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    surveyed = survey_pair(aligned, stable)
    assert surveyed["stats"] == {
        "segments": 3,
        "segments_with_events": 1,
        "stable_tagged": 1,
        "stable_dropped": 1,
    }
    by_index = {row["index"]: row for row in surveyed["evidence_rows"]}
    assert by_index[1]["stable_outcome"] == "tagged"
    assert by_index[1]["timeline_tags_only"] is True
    assert by_index[1]["event_brief"] == ["unfin(223)"]
    assert by_index[2]["stable_outcome"] == "dropped"


def test_discover_pairs_skips_wt_baselines(tmp_path):
    (tmp_path / "wt-aligned.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fw-refine-aligned.json").write_text("{}", encoding="utf-8")
    (tmp_path / "fw-refine-stable.json").write_text("{}", encoding="utf-8")
    pairs = discover_pairs([tmp_path])
    assert len(pairs) == 1
    assert pairs[0][0].name == "fw-refine-aligned.json"
    assert pairs[0][1].name == "fw-refine-stable.json"
