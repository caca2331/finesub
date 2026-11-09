"""Per-line confidence marks (`marks.py`) and where they surface: stable, the `spk` cell, the per-speaker exports.

No model is loaded: a small sidecar is built by hand. Whether the port reproduces the fitted experiment line by line
is checked on real material by `tools/diarization_explore/marks_port_check.py`, not here.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from finesub.speech.diarization import marks
from finesub.speech.diarization.assign import assign_all
from finesub.speech.diarization.evidence import ClusterEvidence, evidence_path, timeline_key
from finesub.speech.diarization.export import FALLBACK_TRACK, group_by_speaker
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel, same_speaker, speaker_cell
from finesub.speech.diarization.segments import TrackSegment
from finesub.speech.diarization.stage import DiarizationResult, write_rows, write_timeline
from finesub.subtitles.stable_rows import overlay_for, read_overlay, write_overlay

FRAMES = 1100  # 22 s on the 20 ms grid
TIMELINE = (TrackSegment(0, 0.0, 8.0), TrackSegment(1, 10.0, 20.0))
LINES = [
    {"start": 1.0, "end": 4.0, "text": "こんにちは世界です"},     # 0: s0
    {"start": 5.0, "end": 5.5, "text": "はい"},                 # 1: s0, back-channel
    {"start": 8.2, "end": 9.0, "text": "えっと"},               # 2: missed; slot 0 still speaks
    {"start": 7.0, "end": 11.0, "text": "それでどうなったの"},     # 3: s0 and s1 a quarter each
    {"start": 12.0, "end": 16.0, "text": "長い文章を話しています"},  # 4: s1
    {"start": 20.5, "end": 21.5, "text": "ふふ"},               # 5: missed; nothing speaks
    {"start": 30.0, "end": 31.0, "text": "範囲外"},              # 6: past the end of the grid
]


def sidecar(**overrides):
    """Slot 0 speaks 0-9 s and is clustered to track 0; slot 1 speaks 10-20 s, track 1."""
    from finesub.speech.diarization.nemotron3_backend import windows

    grid = np.zeros((FRAMES, 8), dtype=np.uint8)
    grid[0:450, 0] = 1
    grid[500:1000, 1] = 1
    chunks = windows(grid.astype(np.float32)).shape[0]
    hard = np.zeros((chunks, 8), dtype=np.int16)
    hard[:, 1] = 1
    soft = np.full((chunks, 8, 2), np.nan, dtype=np.float32)
    soft[:, 0] = (2.0, 1.0)
    soft[:, 1] = (1.0, 2.0)
    native = np.zeros((FRAMES, 2), dtype=np.int8)
    native[0:400, 0] = 1
    native[500:1000, 1] = 1
    fields = dict(soft=soft, centroids=np.zeros((2, 4), dtype=np.float32), native_activity=native, native_step=0.02,
                  centroid_of_track=np.array([0, 1], dtype=np.int16), slot_grid=grid,
                  slot_probs=grid.astype(np.float32) * 0.9, hard=hard, timeline=timeline_key(TIMELINE))
    fields.update(overrides)
    return ClusterEvidence(**fields)


def computed(evidence=None, **constants):
    speakers = assign_all(LINES, TIMELINE)
    return speakers, marks.row_marks(LINES, TIMELINE, speakers, evidence or sidecar())


@pytest.fixture
def everything_low(monkeypatch):
    """Every scorable line is `low` and names its best other candidate: isolates the mechanics from the fit."""
    monkeypatch.setattr(marks, "P_WRONG", 0.0)
    monkeypatch.setattr(marks, "ALTERNATIVE", 0.0)


# --- marks.py -------------------------------------------------------------------------------------------

def test_the_fitted_score_runs_on_every_scorable_line():
    speakers, marked = computed()
    assert len(marked) == len(LINES)
    for mark in marked:
        assert mark is None or mark["tier"] in ("low", "filled")
        if mark and mark["tier"] == "low":
            assert 0.0 <= mark["p_wrong"] <= 1.0


def test_back_channel_lines_are_never_low(everything_low):
    _, marked = computed()
    assert marked[0]["tier"] == "low" and marked[4]["tier"] == "low"
    assert marked[1] is None, "two characters: cheap to delete, and the reference often leaves them out"


def test_the_alternative_is_the_best_other_candidate(everything_low):
    _, marked = computed()
    assert marked[3]["alternative"] == 1, "s1 covers a quarter of the line"
    assert "alternative" not in marked[0], "nothing else is near 1-4 s"


def test_a_missed_line_takes_its_pure_slots_track():
    speakers, marked = computed()
    assert speakers[2].track is None
    assert marked[2] == {"tier": "filled", "track": 0, "by": "slot"}


def test_a_missed_line_with_no_activity_takes_the_nearest_labelled_lines_track():
    _, marked = computed()
    assert marked[5] == {"tier": "filled", "track": 1, "by": "nearest"}, "argmax over zeros would have picked slot 0"


def test_an_impure_slot_falls_back_to_the_nearest_line(monkeypatch):
    monkeypatch.setattr(marks, "FILL_PURITY", 1.01)
    _, marked = computed()
    assert marked[2]["by"] == "nearest"


def test_a_filled_line_never_feeds_another_fill():
    lines = [{"start": 1.0, "end": 4.0, "text": "こんにちは世界です"}, {"start": 8.2, "end": 9.0, "text": "えっと"},
             {"start": 20.5, "end": 21.5, "text": "ふふ"}]
    speakers = assign_all(lines, TIMELINE)
    marked = marks.row_marks(lines, TIMELINE, speakers, sidecar())
    assert marked[2] == {"tier": "filled", "track": 0, "by": "nearest"}, "nearest acoustic line is line 0, not filled line 1"


def test_a_line_past_the_grid_gets_no_mark():
    _, marked = computed()
    assert marked[6] is None


@pytest.mark.parametrize("missing", ["slot_grid", "slot_probs", "hard"])
def test_evidence_without_the_slot_fields_carries_no_marks(missing):
    assert computed(sidecar(**{missing: None}))[1] is None


def test_mismatched_shapes_carry_no_marks():
    assert computed(sidecar(slot_probs=np.zeros((FRAMES - 3, 8), dtype=np.float32)))[1] is None


def test_resample_is_the_overlap_weighted_mean():
    rng = np.random.default_rng(0)
    probs = rng.random((101, 3)).astype(np.float32)
    frames = 52  # runs past the end of the input
    expected = np.zeros((frames, 3), dtype=np.float32)
    for t in range(frames):
        lo, hi = t * 2.0, (t + 1) * 2.0
        idx = np.arange(int(math.floor(lo)), min(int(math.ceil(hi)), len(probs)))
        if len(idx):
            w = np.minimum(idx + 1, hi) - np.maximum(idx, lo)
            expected[t] = (probs[idx] * w[:, None]).sum(axis=0) / w.sum()
    assert np.array_equal(marks.resample(probs, 0.01, frames), expected)


def test_apply_sets_review_from_the_marks_alone(everything_low):
    speakers, marked = computed()
    stale = [speaker.__class__(**{**speaker.__dict__, "review": True}) if speaker.track is not None else speaker
             for speaker in speakers]
    applied = marks.apply(stale, marked)
    assert applied[0].review and applied[0].extras["mark"]["tier"] == "low"
    assert not applied[1].review and "mark" not in applied[1].extras, "assign's coverage flag does not survive"
    assert applied[2].track is None and applied[2].extras["mark"]["track"] == 0, "a filled line keeps `track` empty"


# --- -diar-rows.json ---------------------------------------------------------------------------------------

def write_run(tmp_path, evidence, backend="nemotron3"):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": LINES}, ensure_ascii=False), encoding="utf-8")
    diar = write_timeline(DiarizationResult(tracks=TIMELINE, frame_seconds=0.02, overlap=0.0, backend=backend),
                          tmp_path / "x-diar.json")
    if evidence is not None:
        evidence.save(evidence_path(diar))
    return stable, diar


def written(stable):
    """`-diar-rows.json`'s run record, and its lines as entries."""
    overlay = read_overlay(stable)
    return overlay["speaker"], overlay["rows"]


def warnings_said(monkeypatch):
    from finesub import reporting
    import finesub.speech.diarization.stage as stage_module

    said = []

    class Recorder(reporting.NullReporter):
        def warning(self, code, message, *, impact="", action=""):
            said.append(code)

    monkeypatch.setattr(stage_module, "current_reporter", lambda: Recorder())
    return said


def test_the_rows_carry_the_marks_and_say_they_were_computed(tmp_path, everything_low):
    stable, diar = write_run(tmp_path, sidecar())
    write_rows(stable, diar)
    record, segments = written(stable)
    assert record["marks"]["fitted_on"] == "ja"
    assert segments[0]["speaker"]["mark"]["tier"] == "low" and segments[0]["speaker"]["review"] is True
    assert segments[2]["speaker"]["mark"] == {"tier": "filled", "track": 0, "by": "slot"}
    assert "track" not in segments[2]["speaker"], "the third state: a speaker object without a track"
    assert "speaker" not in segments[6]


def test_without_the_sidecar_nothing_is_marked_and_the_run_says_so(tmp_path, monkeypatch):
    said = warnings_said(monkeypatch)
    stable, diar = write_run(tmp_path, None)
    write_rows(stable, diar)
    record, segments = written(stable)
    assert record["marks"] is None
    assert not any("mark" in (segment.get("speaker") or {}) for segment in segments)
    assert said == ["speaker-marks-unavailable"]


def test_a_stale_sidecar_counts_as_absent(tmp_path, monkeypatch):
    warnings_said(monkeypatch)
    stable, diar = write_run(tmp_path, sidecar(timeline=(5, 9)))
    write_rows(stable, diar)
    assert written(stable)[0]["marks"] is None


def test_a_research_backend_gets_no_marks_and_no_warning(tmp_path, monkeypatch):
    said = warnings_said(monkeypatch)
    stable, diar = write_run(tmp_path, None, backend="diarizen")
    write_rows(stable, diar)
    assert written(stable)[0]["marks"] is None
    assert said == []


def test_writing_again_after_the_sidecar_is_deleted_drops_the_marks(tmp_path, monkeypatch, everything_low):
    warnings_said(monkeypatch)
    stable, diar = write_run(tmp_path, sidecar())
    write_rows(stable, diar)
    evidence_path(diar).unlink()
    write_rows(stable, diar)
    record, segments = written(stable)
    assert record["marks"] is None
    assert "speaker" not in segments[2], "the filled line goes back to unlabelled"
    assert not any("mark" in (segment.get("speaker") or {}) for segment in segments)


def test_the_sidecar_round_trips_the_slot_fields(tmp_path):
    path = sidecar().save(tmp_path / "x-diar-evidence.npz")
    loaded = ClusterEvidence.load(path)
    assert loaded.slot_grid.dtype == np.uint8 and loaded.slot_probs.dtype == np.float32
    assert np.array_equal(loaded.hard, sidecar().hard)


def test_a_sidecar_without_slot_fields_loads_them_as_none(tmp_path):
    loaded = ClusterEvidence.load(sidecar(slot_grid=None, slot_probs=None, hard=None).save(tmp_path / "e.npz"))
    assert loaded.slot_grid is None and loaded.slot_probs is None and loaded.hard is None


# --- rendering and the other readers ------------------------------------------------------------------------

def low(track, tracks=None, alternative=None, coverage=None):
    mark = {"tier": "low", "p_wrong": 0.2}
    if alternative is not None:
        mark["alternative"] = alternative
    extras = {"tracks": tracks or [track], "mark": mark}
    if coverage:
        extras["coverage"] = coverage
    return SegmentSpeaker(track=track, review=True, extras=extras)


def plain(track, tracks=None, coverage=None):
    extras = {"tracks": tracks or [track]}
    if coverage:
        extras["coverage"] = coverage
    return SegmentSpeaker(track=track, extras=extras)


def filled(track):
    return SegmentSpeaker(extras={"mark": {"tier": "filled", "track": track, "by": "slot"}})


def test_low_reads_as_a_question_mark_and_its_alternative():
    assert speaker_cell([low(3)]) == "s3?"
    assert speaker_cell([low(3, alternative=5)]) == "s3?(s5)"
    assert speaker_cell([low(3, [3, 1], alternative=5)]) == "s3?(s5)+s1"


def test_an_alternative_already_on_the_line_is_not_repeated():
    assert speaker_cell([low(3, [3, 5], alternative=5)]) == "s3?+s5"


def test_a_filled_line_reads_with_a_star():
    assert speaker_cell([filled(4)]) == "s4*"


def test_a_merged_line_drops_its_filled_sources_when_any_is_acoustic():
    assert speaker_cell([plain(0, coverage={"0": 0.9}), filled(4)], [2.0, 1.0], review_margin=None) == "s0"


def test_a_merged_line_shows_its_filled_track_beside_sources_with_no_track():
    # Exports file this line on s4; an empty cell would hide where it went.
    assert speaker_cell([SegmentSpeaker(), filled(4)], [2.0, 1.0], review_margin=None) == "s4*"


def test_a_merged_line_of_filled_sources_orders_them_by_time():
    assert speaker_cell([filled(2), filled(4), filled(4)], [3.0, 2.0, 2.0]) == "s4*+s2*"


def test_a_merged_line_follows_its_sources_flags_with_marks():
    # Coverage alone would call this line close (0.5 against 0.5); with marks the sources' own flags decide.
    sources = [plain(0, coverage={"0": 1.0}), plain(1, coverage={"1": 1.0})]
    assert speaker_cell(sources, [1.0, 1.0], review_margin=0.15) == "s0~+s1"
    assert speaker_cell(sources, [1.0, 1.0], review_margin=None) == "s0+s1"
    flagged = [low(0, coverage={"0": 1.0}, alternative=5), plain(1, coverage={"1": 1.0})]
    assert speaker_cell(flagged, [2.0, 1.0], review_margin=None) == "s0?(s5)+s1"


def test_a_runner_up_only_follows_the_low_flag():
    # Coverage decides the flag when a margin is passed (a run without marks); the line is not close,
    # so no `?` -- and the grammar puts `(s5)` only after one.
    sources = [low(0, coverage={"0": 1.0}, alternative=5), plain(1, coverage={"1": 1.0})]
    assert speaker_cell(sources, [2.0, 1.0], review_margin=0.15) == "s0+s1"


def test_a_runner_up_stays_off_a_line_another_source_leads():
    # s5 is the low source's rival to s0; once s1 leads the merged line, `s1?(s5)` would offer s5 against s1.
    sources = [low(0, coverage={"0": 1.0}, alternative=5), plain(1, coverage={"1": 1.0})]
    assert speaker_cell(sources, [1.0, 2.0], review_margin=None) == "s1?+s0"


def test_low_sources_that_disagree_on_the_alternative_show_none():
    sources = [low(0, coverage={"0": 1.0}, alternative=5), low(0, coverage={"0": 1.0}, alternative=6)]
    assert speaker_cell(sources, [1.0, 1.0], review_margin=None) == "s0?"


def test_a_named_line_shows_no_alternative():
    named = SegmentSpeaker(labels=(SpeakerLabel(role="guest", conf="high", name="Aki", src="human"),), track=3,
                           review=True, extras={"tracks": [3], "mark": {"tier": "low", "alternative": 5}})
    assert speaker_cell([named]) == "Aki"


def test_a_filled_line_goes_to_its_track_in_the_exports():
    lines = [{"start": 0.0, "end": 1.0, "text": "a"}, {"start": 1.0, "end": 2.0, "text": "b"},
             {"start": 2.0, "end": 3.0, "text": "c"}]
    groups = group_by_speaker(lines, [plain(0), filled(4), SegmentSpeaker()])
    assert [line["text"] for line in groups["speaker4"]] == ["b"]
    assert [line["text"] for line in groups[FALLBACK_TRACK]] == ["c"]


def test_a_filled_line_is_not_an_acoustic_label_to_other_readers():
    assert same_speaker(filled(4), plain(4)) is None


def test_the_product_cell_follows_the_marks_when_the_rows_say_they_were_computed(tmp_path):
    from finesub.llm.stages.correction.run import _speaker_cells

    def row(index, track, coverage):
        return {"id": str(index), "start": float(index), "end": index + 1.0, "text": "x",
                "speaker": {"track": track, "tracks": [track], "block": 0, "labels": [], "coverage": coverage}}

    segments = [row(1, 0, {"0": 1.0}), row(2, 1, {"1": 1.0})]
    lines = [{key: value for key, value in segment.items() if key != "speaker"} for segment in segments]
    for marks_block, expected in ((None, "s0~+s1"), (marks.metadata(), "s0+s1")):
        path = tmp_path / "clip-stable.json"
        path.write_text(json.dumps({"metadata": {}, "segments": lines}), encoding="utf-8")
        write_overlay(path, overlay_for(lines, [{"speaker": s["speaker"]} for s in segments],
                                        speaker={"marks": marks_block}))
        cell = _speaker_cells(path)
        assert cell(["1", "2"]) == expected, "with marks a merged line is not re-decided from coverage"


def test_a_file_whose_only_labels_are_filled_still_gets_its_split_tracks(tmp_path):
    import json
    from types import SimpleNamespace

    from finesub.stages import _run_speaker_export_stage

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [
        {"start": 0.0, "end": 1.0, "text": "a", "speaker": filled(4).to_dict()},
        {"start": 1.0, "end": 2.0, "text": "b", "speaker": filled(4).to_dict()}]}), encoding="utf-8")
    seen = []
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="srt", empty_cues=False,
                              announce=lambda *args, **kwargs: seen.append(args))
    assert (tmp_path / "clip-speakers" / "clip-speakers-speaker4.srt").exists()
    assert ("speaker-tracks", "skipped") not in [args[:2] for args in seen]
