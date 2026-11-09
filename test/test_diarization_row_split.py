"""Acoustic row splitting (`speech/diarization/row_split.py`) and where its lines go.

The rule's readings (two rulers, three tiers) are reproduced by `tools/diarization_explore/row_split.py parity`
on local data; these pin the parts a refactor could break without any reading moving: the placement and
gating rules, the halves, the language and evidence gates, and the two files a `--diarize` run keeps.
"""

from __future__ import annotations

import json

import pytest

from finesub.speech.diarization import row_split
from finesub.speech.diarization.row_split import Cut, find_cuts
from finesub.speech.diarization.segments import TrackSegment
from finesub.speech.postprocessing.segmentation import Boundary


def boundary(b: float, *, banned: bool = False) -> Boundary:
    return Boundary(banned, 0.0, 0.0, b)


def words(*spans):
    """Words `w0`, `w1`, ... (each four characters) over the given (start, end) spans."""
    return [{"word": f"ことば{k}", "start": a, "end": b} for k, (a, b) in enumerate(spans)]


def row(*spans):
    ws = words(*spans)
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "words": ws, "text": "".join(w["word"] for w in ws)}


# --- placement -------------------------------------------------------------------------------------------------

def test_the_lowest_boundary_in_range_wins_and_banned_ones_never_do():
    bounds = [boundary(3.0), boundary(-1.0, banned=True), boundary(1.0), boundary(3.0)]
    assert row_split._place(bounds, 2, (1, 4)) == 3


def test_ties_go_to_the_acoustic_cut_then_left():
    bounds = [boundary(1.0), boundary(3.0), boundary(1.0), boundary(1.0)]
    assert row_split._place(bounds, 2, (1, 4)) == 1  # 1 and 3 are equally near 2: the left one
    assert row_split._place(bounds, 3, (1, 4)) == 3


def test_a_range_with_every_boundary_banned_is_not_cut():
    banned = boundary(0.0, banned=True)
    assert row_split._place([boundary(0.0), banned, banned], 2, (2, 3)) is None
    assert row_split._place([banned, banned, banned], 2, (2, 3)) is None


def test_the_word_left_of_the_cut_goes_right_when_only_its_left_side_is_a_break():
    # boundaries: before word 1 = index 0, before word 2 = index 1
    assert row_split._glue([boundary(1.0), boundary(3.0)], 2) == 1
    assert row_split._glue([boundary(3.0), boundary(3.0)], 2) == 2   # no break before it
    assert row_split._glue([boundary(1.0), boundary(1.0)], 2) == 2   # a break after it too
    assert row_split._glue([boundary(1.0, banned=True), boundary(3.0)], 2) == 2


# --- detection and gates ----------------------------------------------------------------------------------------

SPANS = ((0.0, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 1.6))
#: slot 0 speaks the first two words, slot 1 the last two; the vbx tracks agree
SLOTS = [TrackSegment(0, 0.0, 0.8), TrackSegment(1, 0.8, 1.6)]
TRACKS = [TrackSegment(5, 0.0, 0.8), TrackSegment(6, 0.8, 1.6)]


def cuts(segment, *, slots=SLOTS, tracks=TRACKS, bounds=None):
    bounds = bounds or [[boundary(3.0), boundary(1.0), boundary(3.0)]]
    return find_cuts([segment], slots=slots, tracks=tracks, boundaries=bounds)


def test_a_clean_change_on_a_break_is_cut():
    assert cuts(row(*SPANS)) == [Cut(0, 2)]


def test_no_cut_on_a_bare_boundary():
    assert cuts(row(*SPANS), bounds=[[boundary(3.0), boundary(3.0), boundary(3.0)]]) == []


def test_no_cut_when_the_halves_lead_one_vbx_track():
    assert cuts(row(*SPANS), tracks=[TrackSegment(5, 0.0, 1.6)]) == []


def test_no_cut_without_a_clean_second_slot():
    overlapping = [TrackSegment(0, 0.0, 1.6), TrackSegment(1, 0.8, 1.6)]
    assert cuts(row(*SPANS), slots=overlapping) == []


def test_no_cut_when_a_side_is_too_short():
    short = row((0.0, 0.05), (0.05, 0.1), (0.8, 1.2), (1.2, 1.6))
    assert cuts(short, slots=[TrackSegment(0, 0.0, 0.1), TrackSegment(1, 0.8, 1.6)]) == []


# --- gates on the whole file ------------------------------------------------------------------------------------

class Evidence:
    native_step = 0.02
    slot_grid = [[1, 0]] * 40 + [[0, 1]] * 40


def payload(language="ja"):
    return {"metadata": {"asr_align": {"language": language, "segment_split": {"length_scale": 1.0}}},
            "vad_timeline": {"intervals": [{"start": 0.0, "end": 0.8}, {"start": 0.81, "end": 1.6}]},
            "segments": [row(*SPANS)]}


def test_a_language_without_a_reading_is_not_cut():
    assert row_split.cuts_for(payload("en"), TRACKS, Evidence()) == []


def test_without_nemotron_slots_nothing_is_cut():
    assert row_split.cuts_for(payload(), TRACKS, None) == []


def test_the_whole_file_path_cuts_at_the_vad_gap():
    assert row_split.cuts_for(payload(), TRACKS, Evidence()) == [Cut(0, 2)]


def test_an_auto_detected_japanese_run_is_cut_by_each_line_s_language():
    """The default `--language` is auto, which records `"None"` as the run's language; the lines' own
    `lang` decides."""
    auto = payload("None")
    auto["segments"][0]["lang"] = "ja"
    assert row_split.cuts_for(auto, TRACKS, Evidence()) == [Cut(0, 2)]
    auto["segments"][0]["lang"] = "en"
    assert row_split.cuts_for(auto, TRACKS, Evidence()) == []


# --- in -diar-rows.json ---------------------------------------------------------------------------------------

def _diarized(tmp_path, *, evidence=True):
    """A stable, its timeline and (optionally) the evidence that cuts its one line before word 2."""
    import numpy as np

    from finesub.speech.diarization.evidence import ClusterEvidence, timeline_key
    from finesub.speech.diarization.stage import DiarizationResult, write_timeline

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(payload()), encoding="utf-8")
    timeline = write_timeline(DiarizationResult(tracks=tuple(TRACKS), frame_seconds=0.08, overlap=0.0),
                              tmp_path / "clip-diar.json")
    if evidence:
        ClusterEvidence(soft=np.zeros((0, 0, 0)), centroids=np.zeros((0, 0)), native_activity=np.zeros((0, 0)),
                        native_step=0.02, timeline=timeline_key(TRACKS),
                        slot_grid=np.asarray(Evidence.slot_grid, dtype=np.uint8)).save(tmp_path / "clip-diar-evidence.npz")
    return stable, timeline


def split_of(stable):
    """`-diar-rows.json`'s split record and the first line's entry."""
    from finesub.subtitles.stable_rows import read_overlay

    overlay = read_overlay(stable)
    return overlay["split"], overlay["rows"][0]


def stage(tmp_path, diarize_split=True):
    """`_run_diarization_stage` on the task in `tmp_path`; returns what it announced."""
    from finesub.stages import _run_diarization_stage, default_pipeline_paths

    seen = []
    _run_diarization_stage(default_pipeline_paths(tmp_path / "clip.wav", tmp_path / "clip.srt"), diarize="acoustics",
                           device=None, gpu_tier="entry", announce=lambda stage, status, **k: seen.append(status),
                           diarize_split=diarize_split)
    return seen


def test_a_cut_goes_on_the_line_s_entry_and_the_stable_is_untouched(tmp_path):
    from finesub.speech.diarization.stage import write_rows

    stable, timeline = _diarized(tmp_path)
    before = stable.read_bytes()
    write_rows(stable, timeline, split_choice=True)
    record, entry = split_of(stable)
    assert stable.read_bytes() == before
    assert entry["split"]["at"] == 2
    assert [s["track"] for s in entry["split"]["speakers"]] == [5, 6]
    assert record == {"chosen": True, "version": 1, "cuts": 1}


def test_without_evidence_nothing_is_cut(tmp_path):
    from finesub.speech.diarization.stage import write_rows

    stable, timeline = _diarized(tmp_path, evidence=False)
    write_rows(stable, timeline)
    record, entry = split_of(stable)
    assert "split" not in entry
    assert record == {"chosen": False, "version": 1, "cuts": 0}


@pytest.mark.parametrize("made", [None, "-raw.srt", "-translated.srt", ".srt", "-annotated.csv", "-speaker-judge.json",
                                  ".llm-artifacts/correction-windows.jsonl"])
@pytest.mark.parametrize("diarize_split", [True, False])
def test_the_choice_follows_the_switch_unless_products_already_count_the_old_lines(tmp_path, made, diarize_split):
    from finesub.stages import default_pipeline_paths, split_choice

    paths = default_pipeline_paths(tmp_path / "clip.wav", tmp_path / "clip.srt")
    if made:
        (tmp_path / f"clip{made}").parent.mkdir(exist_ok=True)
        (tmp_path / f"clip{made}").write_text("", encoding="utf-8")
    assert split_choice(paths, diarize_split) is (diarize_split and made is None)


@pytest.mark.parametrize("diarize_split", [True, False])
def test_a_diarize_run_writes_the_chosen_raw_srt_from_a_cached_timeline(tmp_path, diarize_split):
    """End to end through `run_pipeline` to raw-srt: a stable and a timeline with evidence already on disk."""
    from finesub import pipeline

    source = tmp_path / "clip.wav"
    source.write_bytes(b"fake")
    out = tmp_path / "out"
    out.mkdir()
    stable, _ = _diarized(out)
    stable_bytes = stable.read_bytes()

    def run(split):
        pipeline.run_pipeline(source, output_path=out / "clip.srt", diarize="acoustics", diarize_split=split)

    def cues():
        return (out / "clip-raw.srt").read_text(encoding="utf-8").count("-->")

    run(diarize_split)
    record, entry = split_of(stable)
    assert entry["split"]["at"] == 2 and record["chosen"] is diarize_split
    assert cues() == (2 if diarize_split else 1)
    assert sorted(p.name for p in out.glob("clip-raw*.srt")) == ["clip-raw.srt"], "one raw SRT"
    assert stable.read_bytes() == stable_bytes, "the stable is never rewritten"

    # A rerun with the switch flipped keeps the task's lines, also once the raw SRT (removable) is gone.
    def products():  # the run record (-metadata.json) is rewritten by every run
        return {p.name: p.read_bytes() for p in out.iterdir() if p.is_file() and p.name != "clip-metadata.json"}

    before = products()
    run(not diarize_split)
    assert products() == before
    (out / "clip-raw.srt").unlink()
    run(not diarize_split)
    assert cues() == (2 if diarize_split else 1)


def test_rows_are_regrouped_only_when_they_carry_the_seam_tags(monkeypatch):
    """As in `split_segments`: untagged rows are taken as the ASR's own segments. Regrouping them would fold
    the whole file into one segment and score its edges as inner boundaries."""
    from finesub.speech.postprocessing import segmentation

    calls = []
    real = segmentation.regroup_by_whisper_segments
    monkeypatch.setattr(segmentation, "regroup_by_whisper_segments", lambda rows: calls.append(1) or real(rows))
    rows = [row((0.0, 0.4), (0.4, 0.8)), row((0.8, 1.2), (1.2, 1.6))]
    intervals = [{"start": 0.0, "end": 1.6}]
    assert [len(b) for b in segmentation.row_boundaries(rows, intervals)] == [1, 1]
    assert calls == []
    rows[0]["words"][0]["whisper_segment_start"] = True
    assert [len(b) for b in segmentation.row_boundaries(rows, intervals)] == [1, 1]
    assert calls == [1]


def test_the_rows_file_is_reused_whatever_the_evidence_or_the_switch_says_later(tmp_path):
    """Owner 2026-10-14/15: rerun jitter, or evidence deleted since, is no reason to change which lines a task's
    products count: once written, `-diar-rows.json` is reused as it is."""
    _diarized(tmp_path)
    assert stage(tmp_path) == ["running"]
    rows_bytes = (tmp_path / "clip-diar-rows.json").read_bytes()
    (tmp_path / "clip-diar-evidence.npz").unlink()
    assert stage(tmp_path, diarize_split=False) == ["reused"]
    assert (tmp_path / "clip-diar-rows.json").read_bytes() == rows_bytes


def test_a_deleted_rows_file_is_decided_again_from_the_switch(tmp_path):
    """The manual's way to switch versions without recomputing the timeline: delete -diar-rows.json, -raw.srt
    and the LLM products, run again. A product left behind still counts the old lines."""
    import shutil

    stable, _ = _diarized(tmp_path)
    artifacts = tmp_path / "clip.llm-artifacts"
    artifacts.mkdir()
    (artifacts / "correction-windows.jsonl").write_text("", encoding="utf-8")
    stage(tmp_path)
    assert split_of(stable)[0]["chosen"] is False, "a correction under way counts the old lines"
    (tmp_path / "clip-diar-rows.json").unlink()
    shutil.rmtree(artifacts)
    stage(tmp_path)
    assert split_of(stable)[0]["chosen"] is True


def test_a_freshly_computed_timeline_writes_the_rows_again(tmp_path, monkeypatch):
    """Owner 2026-10-14: a rerun that asked for a new answer (the timeline deleted and computed again)
    replaces the old decision."""
    import shutil

    from finesub.speech.diarization import stage as diarization
    from finesub.speech.diarization.stage import DiarizationResult

    stable, timeline = _diarized(tmp_path)
    kept = tmp_path / "kept.npz"
    shutil.move(tmp_path / "clip-diar-evidence.npz", kept)
    stage(tmp_path)  # e.g. a backend without slots: nothing cut
    assert split_of(stable)[0]["cuts"] == 0
    timeline.unlink()

    def fresh(*args, evidence_path=None, **kwargs):
        shutil.copy(kept, evidence_path)
        return DiarizationResult(tracks=tuple(TRACKS), frame_seconds=0.08, overlap=0.0)

    monkeypatch.setattr(diarization, "run_diarization", fresh)
    monkeypatch.setattr("finesub.stages._diarization_device", lambda device, tier: "cpu")
    assert stage(tmp_path) == ["running"]
    record, entry = split_of(stable)
    assert entry["split"]["at"] == 2 and record["chosen"] is True


def test_a_stable_labelled_before_the_rows_file_carries_its_labels_over_once(tmp_path):
    """⚠ Compatibility, to delete at 1.0. Owner 2026-10-15: a 0.6.0 stable (labels on its segments, run record in
    metadata.speaker) is read with those labels, and the stage writes them to -diar-rows.json as they are --
    no relabel (it would lose the marks once the evidence is gone, and every judge window keyed on them) --
    with the row splitting decided as for a new task: not chosen while products count the old lines, chosen
    once they are deleted (the manual's way to switch versions)."""
    from finesub.llm.stages.correction.run import _speaker_cells
    from finesub.speech.diarization import marks
    from finesub.subtitles.stable_rows import read_overlay, read_rows

    stable, timeline = _diarized(tmp_path)
    old = json.loads(stable.read_text(encoding="utf-8"))
    speaker = {"labels": [], "block": 0, "track": 5, "tracks": [5, 6], "coverage": {"5": 0.55, "6": 0.45}}
    old["segments"][0]["speaker"] = speaker
    hints = json.loads(timeline.read_text(encoding="utf-8"))["metadata"]["hints"]  # 0.6.0 copied them
    old["metadata"]["speaker"] = {"model": "x", "marks": marks.metadata(), "hints": hints}
    stable.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    before = stable.read_bytes()

    assert read_rows(stable)[0]["speaker"] == speaker, "read before any run writes the file"
    assert _speaker_cells(stable)(["1"]) == "s5+s6", "marks computed: the cell follows them, not the coverage"
    (tmp_path / "clip-raw.srt").write_text("", encoding="utf-8")  # an earlier run's products
    assert stage(tmp_path) == ["reused"]
    overlay = read_overlay(stable)
    assert (tmp_path / "clip-diar-rows.json").exists() and stable.read_bytes() == before
    assert overlay["rows"][0]["speaker"] == speaker and overlay["rows"][0]["split"]["at"] == 2
    assert overlay["speaker"] == old["metadata"]["speaker"]
    assert overlay["split"] == {"chosen": False, "version": 1, "cuts": 1}
    assert stage(tmp_path) == ["reused"]
    (tmp_path / "clip-diar-rows.json").unlink()
    (tmp_path / "clip-raw.srt").unlink()
    stage(tmp_path)
    overlay = read_overlay(stable)
    assert overlay["split"]["chosen"] is True and overlay["rows"][0]["speaker"] == speaker
    (tmp_path / "clip-diar-rows.json").write_text("{broken", encoding="utf-8")
    assert all("speaker" not in line for line in read_rows(stable)), "a broken rows file never falls back"


def test_old_labels_from_another_timeline_are_not_carried_over(tmp_path):
    """⚠ Compatibility, to delete at 1.0. A 0.6.0 task whose -diar.json was computed again since: the stable's own
    labels belong to a partition that is gone, so they are neither read nor carried; the rows are written from
    the timeline that is there."""
    from finesub.subtitles.stable_rows import read_overlay, read_rows

    stable, _ = _diarized(tmp_path)
    old = json.loads(stable.read_text(encoding="utf-8"))
    old["segments"][0]["speaker"] = {"labels": [], "block": 0, "track": 9, "tracks": [9], "coverage": {"9": 1.0}}
    old["metadata"]["speaker"] = {"model": "x", "marks": None, "hints": {"co_activity": {}, "fourth_track_share": 0.5}}
    stable.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    assert "speaker" not in read_rows(stable)[0]
    assert stage(tmp_path) == ["running"]
    assert read_overlay(stable)["rows"][0]["speaker"]["track"] != 9
