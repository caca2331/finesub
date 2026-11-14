"""Reading the speaker judge's sidecar back: the checks, the new-track numbering, the `spk` cell.

The sidecar outlives what it was judged on, so every reader goes through `judgments.read_judgments`;
these pin what it refuses and what it keeps (diar-marks plan 5.5, 「sidecar 与下游」).
"""

from __future__ import annotations

import json

import pytest

from finesub.llm.stages.correction.run import _speaker_cells
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization.labels import speaker_of
from finesub.speech.diarization.product import product


def plain(track, *others):
    coverage = {str(track): 0.9, **{str(other): 0.3 for other in others}}
    return {"track": track, "tracks": [track, *others], "block": 0, "coverage": coverage,
            "labels": [{"role": "unknown", "conf": "high"}]}


def write_stable(tmp_path, speakers):
    segments = []
    for index, speaker in enumerate(speakers, start=1):
        row = {"id": str(index), "start": float(index), "end": index + 1.0, "text": f"line {index}"}
        if speaker is not None:
            row["speaker"] = speaker
        segments.append(row)
    path = tmp_path / "clip-stable.json"
    path.write_text(json.dumps({"segments": segments}), encoding="utf-8")
    return path


def write_sidecar(stable, windows, **overrides):
    """`windows`: (first id, last id, results[, status]); keys are computed the way the judge writes them."""
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    ids = jd.segment_ids(segments)
    tracks = jd.profile_tracks([speaker_of(s) for s in segments])
    entries = []
    for first, last, results, *status in windows:
        low, high = ids.index(first), ids.index(last) + 1
        entries.append({"id": f"0001.{len(entries) + 1}", "first": first, "last": last,
                        "key": jd.window_key(segments[low:high], ids[low:high], tracks),
                        "status": status[0] if status else "success", "results": results})
    payload = {"version": jd.SIDECAR_VERSION, "route": "separate", "prompt_version": "t",
               "rows_hash": jd.rows_hash(segments), "windows": entries, **overrides}
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def read(stable):
    warnings = []
    return jd.read_judgments(stable, warn=warnings.append), warnings


def test_the_sidecar_sits_beside_the_stable(tmp_path):
    assert jd.sidecar_path(tmp_path / "ep1-stable.json") == tmp_path / "ep1-speaker-judge.json"


def test_no_sidecar_is_the_acoustic_labels(tmp_path):
    assert read(write_stable(tmp_path, [plain(0)])) == (None, [])


def test_a_move_and_a_new_track_and_an_abstention_are_read(tmp_path):
    stable = write_stable(tmp_path, [plain(0), plain(1), plain(0), plain(1)])
    write_sidecar(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}]),
                           ("3", "4", [{"row": "3", "label": "甲", "from": 0},
                                       {"row": "4", "abstain": True, "from": 1}])])
    judged, warnings = read(stable)
    assert judged.moved == {"1": 1, "3": 2} and judged.new_tracks == (2,) and judged.abstained == {"4"}
    assert warnings == []


def test_new_tracks_start_past_every_track_the_stable_names(tmp_path):
    """A filled line's `mark.track` and a low mark's `alternative` can name a track that leads no line,
    and ids need not be contiguous: s9 here is only an alternative, so the first new track is s10."""
    low = {**plain(0), "review": True, "mark": {"tier": "low", "alternative": 9}}
    filled = {"block": 0, "labels": [], "mark": {"tier": "filled", "track": 7}}
    stable = write_stable(tmp_path, [low, filled, plain(3)])
    write_sidecar(stable, [("1", "3", [{"row": "3", "label": "甲", "from": 3}])])
    assert read(stable)[0].moved == {"3": 10}


def test_numbering_follows_window_order_then_first_appearance(tmp_path):
    """Each window's labels get their own numbers, so the same label in two windows is two tracks; the
    numbers come from the sidecar's window order, whatever order the windows finished in."""
    stable = write_stable(tmp_path, [plain(0)] * 6)
    write_sidecar(stable, [("1", "3", [{"row": "1", "label": "乙", "from": 0}, {"row": "2", "label": "甲", "from": 0},
                                       {"row": "3", "label": "乙", "from": 0}]),
                           ("4", "6", [{"row": "5", "label": "甲", "from": 0}])])
    assert read(stable)[0].moved == {"1": 1, "2": 2, "3": 1, "5": 3}


def test_a_window_that_failed_or_is_pending_keeps_the_acoustic_labels(tmp_path):
    stable = write_stable(tmp_path, [plain(0), plain(0)])
    write_sidecar(stable, [("1", "1", [{"row": "1", "track": 1, "from": 0}], "failed"),
                           ("2", "2", [{"row": "2", "track": 1, "from": 0}], "pending")])
    assert read(stable)[0].moved == {}


def test_a_window_whose_rows_changed_is_dropped_with_a_warning(tmp_path):
    """Re-running diarization renumbers tracks: the old verdict must not land on the new ids. (A change
    to the set of tracks the profiles draw on voids every window; here only row 2's label moves.)"""
    stable = write_stable(tmp_path, [plain(0), plain(1), plain(0), plain(1)])
    write_sidecar(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}]), ("3", "4", [{"row": "3", "track": 1, "from": 0}])])
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"][1]["speaker"] = plain(0)
    stable.write_text(json.dumps(payload), encoding="utf-8")
    judged, warnings = read(stable)
    assert judged.moved == {"3": 1} and len(warnings) == 1


def test_a_result_whose_from_no_longer_holds_is_dropped(tmp_path):
    filled = {"block": 0, "labels": [], "mark": {"tier": "filled", "track": 1}}
    stable = write_stable(tmp_path, [plain(0), None, filled, plain(1)])
    write_sidecar(stable, [("1", "4", [{"row": "1", "track": 1, "from": 1},
                                       {"row": "2", "track": 0, "from": None},
                                       {"row": "3", "track": 0, "from": {"filled": 1}},
                                       {"row": "4", "track": 0, "from": None}])])
    assert read(stable)[0].moved == {"2": 0, "3": 0}


def test_a_retranscribed_stable_drops_the_whole_sidecar(tmp_path):
    stable = write_stable(tmp_path, [plain(0), plain(1)])
    write_sidecar(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}])])
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"].append({"id": "3", "start": 3.0, "end": 4.0, "text": "more", "speaker": plain(0)})
    stable.write_text(json.dumps(payload), encoding="utf-8")
    judged, warnings = read(stable)
    assert judged is None and "another transcript" in warnings[0]


def test_windows_that_do_not_tile_the_file_are_damage(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    for windows in ([("1", "2", [])], [("1", "2", []), ("2", "4", [])], [("2", "4", []), ("1", "1", [])]):
        write_sidecar(stable, windows)
        judged, warnings = read(stable)
        assert judged is None and "damaged" in warnings[0], windows


def test_the_judged_speaker_leads_its_track_alone(tmp_path):
    low = {**plain(0, 1), "review": True, "mark": {"tier": "low", "alternative": 1}}
    stable = write_stable(tmp_path, [low, plain(1)])
    write_sidecar(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    speakers = [speaker_of(s) for s in segments]
    judged = product(segments, read(stable)[0]).speakers()
    assert judged["1"].track == 1 and not judged["1"].review and "mark" not in judged["1"].extras
    assert judged["2"] == speakers[1]


# --- the `spk` cell ------------------------------------------------------------------------------


def cells(stable, tmp_path):
    return _speaker_cells(stable)


def test_a_moved_line_shows_both_sides(tmp_path):
    low = {**plain(3), "review": True, "mark": {"tier": "low", "alternative": 5}}
    stable = write_stable(tmp_path, [low, plain(7), plain(5), None])
    write_sidecar(stable, [("1", "4", [{"row": "1", "track": 7, "from": 3}, {"row": "4", "track": 7, "from": None}])])
    cell = cells(stable, tmp_path)
    assert cell(["1"]) == "s7<s3?(s5)"
    assert cell(["2"]) == "s7"
    assert cell(["4"]) == "s7<", "an unassigned source moved: the acoustic side is empty"


def test_a_merged_line_ranks_its_tracks_by_time_and_keeps_the_acoustic_cell(tmp_path):
    stable = write_stable(tmp_path, [plain(3), plain(3), plain(3)])
    payload = json.loads(stable.read_text(encoding="utf-8"))
    payload["segments"][2]["end"] = 10.0  # source 3 is the longest
    stable.write_text(json.dumps(payload), encoding="utf-8")
    write_sidecar(stable, [("1", "3", [{"row": "1", "track": 7, "from": 3}])])
    assert cells(stable, tmp_path)(["1", "3"]) == "s3+s7<s3"


# --- the split-track export ----------------------------------------------------------------------


def test_the_export_follows_the_sidecar_and_leaves_no_stale_file(tmp_path):
    from types import SimpleNamespace

    from finesub.stages import _run_speaker_export_stage

    stable = write_stable(tmp_path, [plain(0), plain(1), plain(0)])
    folder = tmp_path / "clip-speakers"
    folder.mkdir()
    (folder / "clip-speakers-speaker9.srt").write_text("old", encoding="utf-8")
    (folder / "clip-speakers.ass").write_text("old", encoding="utf-8")
    write_sidecar(stable, [("1", "3", [{"row": "1", "track": 1, "from": 0}, {"row": "3", "label": "甲", "from": 0}])])
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="srt", empty_cues=False, announce=lambda *a, **k: None)
    assert sorted(p.name for p in folder.iterdir()) == ["clip-speakers-speaker1.srt", "clip-speakers-speaker2.srt"]
    assert "line 1" in (folder / "clip-speakers-speaker1.srt").read_text(encoding="utf-8")
    assert "line 3" in (folder / "clip-speakers-speaker2.srt").read_text(encoding="utf-8")


def test_rejudging_one_window_keeps_the_others_numbers_when_its_count_holds(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_sidecar(stable, [("1", "2", [{"row": "1", "label": "甲", "from": 0}]), ("3", "4", [{"row": "3", "label": "乙", "from": 0}])])
    first = read(stable)[0].moved
    write_sidecar(stable, [("1", "2", [{"row": "2", "label": "丙", "from": 0}]), ("3", "4", [{"row": "3", "label": "乙", "from": 0}])])
    assert read(stable)[0].moved["3"] == first["3"]


def test_moved_filled_and_overlapping_sources_render_their_acoustic_cell(tmp_path):
    filled = {"block": 0, "labels": [], "mark": {"tier": "filled", "track": 4}}
    stable = write_stable(tmp_path, [filled, plain(5, 4), plain(7)])
    write_sidecar(stable, [("1", "3", [{"row": "1", "track": 7, "from": {"filled": 4}}, {"row": "2", "track": 7, "from": 5}])])
    cell = cells(stable, tmp_path)
    assert cell(["1"]) == "s7<s4*"
    assert cell(["2"]) == "s7<s5+s4"


def test_a_carried_track_whose_window_no_longer_holds_falls_back(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_sidecar(stable, [("1", "2", [{"row": "1", "label": "甲", "from": 0}], "failed"),
                           ("3", "4", [{"row": "3", "carried": {"window": "0001.1", "label": "甲"}, "from": 0}])])
    assert read(stable)[0].moved == {}


# --- a window answered in parts ------------------------------------------------------------------


def write_parted(stable, parts, *, window="0001"):
    """One window halved: (first id, last id, results[, status]) for each half."""
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    ids = jd.segment_ids(segments)
    tracks = jd.profile_tracks([speaker_of(s) for s in segments])
    entries = []
    for half, (first, last, results, *status) in enumerate(parts):
        low, high = ids.index(first), ids.index(last) + 1
        entries.append({"id": jd.part_id(window, half), "first": first, "last": last,
                        "key": jd.window_key(segments[low:high], ids[low:high], tracks),
                        "status": status[0] if status else "success", "results": results})
    for entry in entries:  # a carried track names the verdict that opened it, as the judge writes it
        for result in entry["results"]:
            if "carried" in result and "of" not in result["carried"]:
                opener = next(other for other in entries if other["id"] == result["carried"]["window"])
                result["carried"]["of"] = jd.verdict_of(opener)
    payload = {"version": jd.SIDECAR_VERSION, "route": "separate", "rows_hash": jd.rows_hash(segments),
               "windows": [{"id": window, "chunk": window, "first": ids[0], "last": ids[-1], "parts": entries}]}
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_a_window_cut_into_parts_reads_each_part(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}]),
                          ("3", "4", [{"row": "4", "track": 1, "from": 0}], "failed")])
    judged, _ = read(stable)
    assert judged.moved == {"1": 1}, "a failed part keeps its rows on the acoustic labels"
    assert not jd.covers(stable), "a failed part is owed"


def test_the_same_label_in_two_parts_is_two_new_tracks(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", [{"row": "1", "label": "甲", "from": 0}]),
                          ("3", "4", [{"row": "3", "label": "甲", "from": 0}])])
    assert read(stable)[0].moved == {"1": 1, "3": 2}


def test_a_part_may_carry_on_the_part_before_it(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", [{"row": "1", "label": "甲", "from": 0}]),
                          ("3", "4", [{"row": "3", "carried": {"window": "0001-a", "label": "甲"}, "from": 0}])])
    assert read(stable)[0].moved == {"1": 1, "3": 1}


def test_a_carried_track_drops_when_its_opener_was_judged_again(tmp_path):
    """The opener judged anew on the same rows may give the same label to another person: the carry it
    named is gone, so that row falls back rather than joining whoever holds the label now."""
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", [{"row": "1", "label": "甲", "from": 0}]),
                          ("3", "4", [{"row": "3", "carried": {"window": "0001-a", "label": "甲"}, "from": 0}])])
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][0]["parts"][0]["results"] = [{"row": "2", "label": "甲", "from": 0}]
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert read(stable)[0].moved == {"2": 1}


@pytest.mark.parametrize("results", [[None], [{"row": "1", "track": "s1", "from": 0}], "x",
                                     [{"row": "1", "carried": {"window": "0001-a"}, "from": 0}]])
def test_a_damaged_but_parsable_unit_falls_back_and_is_owed(tmp_path, results):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", []), ("3", "4", [])])
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    payload["windows"][0]["parts"][0]["results"] = results
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert read(stable)[0].moved == {} and not jd.covers(stable)


def test_parts_that_do_not_tile_their_window_are_damage(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", []), ("4", "4", [])])
    judged, warnings = read(stable)
    assert judged is None and "damaged" in warnings[0]


def test_a_version_one_sidecar_reads_as_absent(tmp_path):
    stable = write_stable(tmp_path, [plain(0), plain(1)])
    write_sidecar(stable, [("1", "2", [{"row": "1", "track": 1, "from": 0}])], version=1)
    judged, warnings = read(stable)
    assert judged is None and "version" in warnings[0]
    assert not jd.covers(stable)


def test_a_successful_unit_without_results_is_owed(tmp_path):
    stable = write_stable(tmp_path, [plain(0)] * 4)
    write_parted(stable, [("1", "2", []), ("3", "4", [])])
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    del payload["windows"][0]["parts"][0]["results"]
    jd.sidecar_path(stable).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert not jd.covers(stable)
