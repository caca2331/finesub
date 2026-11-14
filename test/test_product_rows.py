"""The product layer (`speech.diarization.product`): the judge's rows with what a judge that heard the audio
split, filled and inserted, and the readers that see it -- the correction's lines, pass 2's grouping, the `spk`
column, the per-speaker export and the knowledge harvest."""

from __future__ import annotations

import json

from finesub.speech.diarization import judgments as jd
from finesub.llm.prompt_constants import Thresholds
from finesub.speech.diarization import product as pr

from .test_speaker_judge_shape import shaping
from .test_speaker_judge_stage import FakeClient, run, speaker


def heard_stable(tmp_path, *, acts=None):
    """Six rows alternating s1 / s0, 1 s apart, two timed words each, judged in one unit that split row 2 (its
    second half on s1), filled row 4 at its head, inserted a line on s1 after row 4 and one after row 5 that has
    no room (row 6 starts 0.3 s after row 5 ends)."""
    stable = tmp_path / "clip-stable.json"
    lines = []
    for n in range(1, 7):
        start = 3.0 * n if n < 6 else 17.3
        lines.append({"id": str(n), "start": start, "end": start + 2, "text": f"line {n}", "speaker": speaker(n % 2),
                      "words": [{"word": "line ", "start": start, "end": start + 1},
                                {"word": str(n), "start": start + 1, "end": start + 2}]})
    stable.write_text(json.dumps({"segments": lines}), encoding="utf-8")
    run(stable, FakeClient(judge=shaping(acts or {})), tmp_path, cut=((1, 6),), shape=jd.SHAPE_TRACKED)
    path = jd.sidecar_path(stable)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["windows"][0]
    entry["splits"] = [{"row": "2", "prefix": "line", "at": 1, "from": 0, "track": 1}]
    entry["fills"] = [{"row": "4", "text": "そう line 4", "head": 7, "tail": 0}]
    entry["inserts"] = [{"row": "4", "start": 14.1, "end": 14.9, "text": "ほんと", "track": 1},
                        {"row": "5", "start": 17.0, "end": 17.2, "text": "うん", "track": 0}]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return stable


def test_the_product_layer_applies_the_splits_fills_and_shown_inserts(tmp_path):
    stable = heard_stable(tmp_path)
    layer = pr.read_product(stable, warn=lambda message: None)
    assert [(line.id, line.source, line.part, line.heard) for line in layer.lines] == [
        ("1", "1", pr.WHOLE, False), ("2", "2", pr.FIRST, False), ("2b", "2", pr.SECOND, False),
        ("3", "3", pr.WHOLE, False), ("4", "4", pr.WHOLE, True), ("4i", "4", pr.INSERTED, True),
        ("5", "5", pr.WHOLE, False), ("6", "6", pr.WHOLE, False)], "row 5's insert is deferred: no line yet"
    rows = {line.id: line.row for line in layer.lines}
    assert (rows["2"]["text"], rows["2"]["end"], rows["2b"]["text"], rows["2b"]["start"]) == ("line", 7.0, "2", 7.0)
    assert (rows["4"]["text"], rows["4"]["start"], rows["4"]["end"]) == ("そう line 4", 11.0, 14.0)
    assert "words" not in rows["4"], "its words no longer spell its text"
    assert (rows["4i"]["text"], rows["4i"]["start"], rows["4i"]["end"]) == ("ほんと", 14.1, 14.9)
    speakers = layer.speakers()
    assert [speakers[key].track for key in ("1", "2", "2b", "3", "4", "4i")] == [1, 0, 1, 1, 0, 1]
    assert set(layer.judged) == {"2b", "4i"}, "the verdicts moved nobody: only the lines the judge made are decided"


def test_the_sidecar_still_holds_and_the_same_sidecar_gives_the_same_lines(tmp_path):
    stable = heard_stable(tmp_path)
    first = pr.read_product(stable, warn=lambda message: None)
    assert jd.covers(stable, shape=jd.SHAPE_TRACKED), "the judge's layer is untouched"
    assert pr.read_product(stable, warn=lambda message: None) == first


def test_without_a_sidecar_the_lines_are_the_rows_with_explicit_ids(tmp_path):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"start": 0.0, "end": 1.0, "text": "a"},
                                               {"start": 1.0, "end": 2.0, "text": "b"}]}), encoding="utf-8")
    layer = pr.read_product(stable)
    assert [(line.id, line.part) for line in layer.lines] == [("1", pr.WHOLE), ("2", pr.WHOLE)]
    assert layer.speakers() is None


def test_after_the_judge_each_window_spells_its_lines_as_the_product_lines(tmp_path):
    from finesub.llm.chunking import load_product_segments, load_segments_from_stable_json, product_windows

    from .test_llm_grouping import window

    stable = heard_stable(tmp_path)
    planned = load_segments_from_stable_json(stable)
    assert [segment.id for segment in planned] == ["1", "2", "3", "4", "5", "6"], "planning reads the judge's rows"
    segments, layer = load_product_segments(stable)
    first, second = product_windows([window("0001", planned[:3]), window("0002", planned[4:], overlap=planned[3:4])],
                                    segments, layer)
    assert [segment.id for segment in first.segments] == ["1", "2", "2b", "3"]
    assert [segment.id for segment in second.overlap_segments] == ["4", "4i"]
    assert [segment.id for segment in second.segments] == ["4", "4i", "5", "6"]
    assert [segment.seam.value for segment in first.segments][2] == "换人", "a second half follows a change"


def test_spelling_twice_changes_nothing_and_the_clip_grows_to_hold_the_lines(tmp_path):
    from dataclasses import replace

    from finesub.llm.chunking import load_product_segments, load_segments_from_stable_json, product_windows

    from .test_llm_grouping import window

    stable = heard_stable(tmp_path)
    planned = load_segments_from_stable_json(stable)
    segments, layer = load_product_segments(stable)
    # row 4 grew from 12.0 to 11.0 at its head, and its inserted line ends at 14.9
    planned_window = replace(window("0001", planned[3:4]), clip_start=11.5, clip_end=14.5)
    once = product_windows([planned_window], segments, layer)
    assert (once[0].clip_start, once[0].clip_end) == (11.0, 14.9)
    assert product_windows(once, segments, layer) == once
    assert product_windows([window("0002", planned[3:4])], segments, layer)[0].clip_end == 0.0, "no clip, none made"


def test_a_direct_call_without_a_plan_corrects_the_product_lines(tmp_path, monkeypatch):
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.correction import execute_correction_windows
    from finesub.llm.stages.correction import run as correction_run

    from .test_llm_correction_translation import FakeTokenCounter

    stable = heard_stable(tmp_path)
    real, spelled = correction_run.product_windows, []

    def spy(windows, segments, layer):
        out = real(windows, segments, layer)
        spelled.extend(out)
        return out

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):  # the client is built once the windows are settled
        raise Stop

    monkeypatch.setattr(correction_run, "product_windows", spy)
    monkeypatch.setattr(correction_run, "RoleClient", stop)
    try:
        execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt",
                                   token_counter=FakeTokenCounter(), profile=resolve_profile("text", "none", "quality"), length_scale=1.0)
    except Stop:
        pass
    assert [line.id for window in spelled for line in window.segments] == ["1", "2", "2b", "3", "4", "4i", "5", "6"]


def test_pass_1s_acts_carry_over_to_the_product_lines(tmp_path):
    from finesub.llm.chunking import load_product_segments, load_segments_from_stable_json, product_windows
    from finesub.llm.stages import three_pass

    from .test_llm_grouping import window

    # q3 joins row 3 to the line before it, now row 2's second half (both s1); q5's join would cross row 4's
    # inserted line, so row 5 stands alone; q6 is dropped as ever
    stable = heard_stable(tmp_path, acts={"q3": "join", "q5": "join", "q6": "drop"})
    segments, layer = load_product_segments(stable)
    windows = product_windows([window("0001", load_segments_from_stable_json(stable))], segments, layer)
    out, grouping, record = three_pass.pass2_plan(windows, stable, jd.SHAPE_TRACKED, warn=lambda message: None, thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1",), ("2",), ("2b", "3"), ("4",), ("4i",), ("5",))
    assert grouping.dropped == ("6",) and record["without_pass1"] == []
    assert [line.id for line in out[0].segments] == ["1", "2", "2b", "4", "4i", "5"]


def test_the_spk_cell_reads_the_product_lines_and_keeps_a_cell_whose_line_is_gone(tmp_path):
    from finesub.llm.output_protocol import respeak_annotated_csv
    from finesub.llm.stages.correction.run import _speaker_cells

    stable = heard_stable(tmp_path)
    cell = _speaker_cells(stable)
    assert cell(["2b"]) == "s1<s0", "the second half on the judge's track, the row's acoustic cell beside it"
    assert cell(["4i"]) == "s1<", "the acoustic layer never saw an inserted line"
    assert cell(["3"]) == "s1"
    jd.sidecar_path(stable).unlink()  # the judge, run again, left row 2 whole
    cell = _speaker_cells(stable)
    assert cell(["2b"]) is None
    header = "type|position|start|end|gap|corrected_text|translation|conf|char_count|spk|note"
    annotated = f"{header}\nsub|2b|7.0|8.0|0.0|2|2|high|1|s1<s0|\n"
    assert respeak_annotated_csv(annotated, cell) == annotated


def test_the_per_speaker_export_writes_the_product_lines(tmp_path):
    from types import SimpleNamespace

    from finesub.stages import _run_speaker_export_stage

    stable = heard_stable(tmp_path)
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="srt", empty_cues=False,
                              announce=lambda *args, **kwargs: None)
    track = (tmp_path / "clip-speakers" / "clip-speakers-speaker1.srt").read_text(encoding="utf-8")
    assert "00:00:07,000 --> 00:00:08,000\n2\n" in track and "ほんと" in track and "そう line 4" not in track, \
        "row 2's second half and the inserted line on s1; the filled row 4 stays on s0"


def test_a_three_pass_task_corrects_the_product_lines(tmp_path, monkeypatch):
    import finesub.llm.correction_translation as orchestration
    from finesub.llm.stages import three_pass

    from .test_speaker_judge_stage import fast_whole_file, task_request

    monkeypatch.setattr(three_pass, "grouped_prompts", lambda profile: True)
    stable = heard_stable(tmp_path)
    fast_whole_file(monkeypatch, orchestration, rows=6)
    monkeypatch.setattr(orchestration, "RoleClient", lambda **kwargs: FakeClient(judge=shaping({})))
    seen = {}

    def fake_execute(**kwargs):
        seen["ids"] = [line.id for window in kwargs["plan"].windows for line in window.segments]
        out = kwargs["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_name(f"{out.stem}-translated.srt").write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    orchestration.run_correction_task(task_request(tmp_path, stable, mode="execute"))
    assert seen["ids"] == ["1", "2", "2b", "3", "4", "4i", "5", "6"], "the judge's unit was reused, its lines applied"


def test_an_inserted_line_stands_alone_even_when_its_row_is_dropped():
    from finesub.llm.grouping import line_acts

    lines = [pr.Line({"id": key}, source, part) for key, source, part in (
        ("3", "3", pr.WHOLE), ("4", "4", pr.WHOLE), ("4i", "4", pr.INSERTED), ("5", "5", pr.WHOLE),
        ("6", "6", pr.FIRST), ("6b", "6", pr.SECOND), ("7", "7", pr.WHOLE))]
    shaped = jd.ShapeActs({"4": "drop", "5": "may_join", "6": "join", "7": "join"}, {"4": "w"}, frozenset({"6"}),
                          frozenset())
    acts, missing = line_acts(shaped, pr.Product(tuple(lines)))
    assert acts == {"4": "drop", "6": "join", "7": "join"}, "5 reads nothing over 4i; 7 joins 6b"
    assert missing == {"6", "6b"}
    assert line_acts(None, pr.Product(tuple(lines))) == ({}, None)


def test_a_deferred_line_stops_a_join_only_across_another_person(tmp_path):
    from finesub.llm.grouping import line_acts
    from finesub.speech.diarization.labels import UNLABELLED

    lines = tuple(pr.Line({"id": key, "speaker": speaker(1)}, key, pr.WHOLE) for key in "1234")
    shaped = jd.ShapeActs({"2": "join", "3": "may_join", "4": "drop"}, {"4": "w"}, frozenset(), frozenset())
    # s0 was heard between rows 1 and 2, s1 (the same person) between 2 and 3, s0 between 3 and 4
    deferred = {"1": jd.decided(0, UNLABELLED), "2": jd.decided(1, UNLABELLED), "3": jd.decided(0, UNLABELLED)}
    acts, _ = line_acts(shaped, pr.Product(lines, deferred=deferred))
    assert acts == {"3": "may_join", "4": "drop"}, "a join would hide s0's turn; a drop stands across anything"
    bare = tuple(pr.Line({"id": key}, key, pr.WHOLE) for key in "1234")
    assert line_acts(shaped, pr.Product(bare, deferred=deferred))[0] == shaped.acts, "no speakers to tell by"
    layer = pr.read_product(heard_stable(tmp_path), warn=lambda message: None)
    assert layer.deferred["5"].track == 0, "row 5's insert has no room: it is the layer's deferred line"


def test_a_join_over_a_dropped_line_still_cannot_cross_an_inserted_line():
    from finesub.llm.grouping import line_acts
    from finesub.speech.diarization.labels import UNLABELLED

    def line(key, source, part=pr.WHOLE):
        return pr.Line({"id": key, "speaker": speaker(0)}, source, part)

    shaped = jd.ShapeActs({"3": "drop", "4": "join"}, {"3": "えー"}, frozenset(), frozenset())
    shown = (line("1", "1"), line("2", "2"), line("2i", "2", pr.INSERTED), line("3", "3"), line("4", "4"))
    assert line_acts(shaped, pr.Product(shown))[0] == {"3": "drop"}, "row 4 would join the inserted line"
    rows = (line("1", "1"), line("2", "2"), line("3", "3"), line("4", "4"))
    after_kept = {"2": jd.decided(1, UNLABELLED)}  # s1 heard after row 2, then row 3 dropped
    assert line_acts(shaped, pr.Product(rows, deferred=after_kept))[0] == {"3": "drop"}
    on_dropped = {"3": jd.decided(1, UNLABELLED)}  # s1 heard after the dropped row: compared with row 2, kept
    assert line_acts(shaped, pr.Product(rows, deferred=on_dropped))[0] == {"3": "drop"}
    same = {"3": jd.decided(0, UNLABELLED)}
    assert line_acts(shaped, pr.Product(rows, deferred=same))[0] == {"3": "drop", "4": "join"}


def relabel(stable, **edits):
    path = jd.sidecar_path(stable)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["windows"][0].update(edits)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_a_second_half_or_an_inserted_line_on_a_new_label_gets_its_own_track(tmp_path):
    from types import SimpleNamespace

    from finesub.llm.stages.correction.run import _speaker_cells
    from finesub.llm.stages.speaker_judge.task import _new_tracks
    from finesub.stages import _run_speaker_export_stage

    stable = heard_stable(tmp_path)
    relabel(stable, splits=[{"row": "2", "prefix": "line", "at": 1, "from": 0, "label": "乙"}],
            inserts=[{"row": "4", "start": 14.1, "end": 14.9, "text": "ほんと", "label": "丙"},
                     {"row": "5", "start": 17.0, "end": 17.2, "text": "うん", "label": "丁"}])
    judged = jd.read_judgments(stable)
    assert judged.new_tracks == (2, 3, 4) and _new_tracks(judged) == ["s2", "s3"], "丁's line is deferred"
    assert _speaker_cells(stable)(["2b"]) == "s2<s0"
    _run_speaker_export_stage(SimpleNamespace(stable_json=stable), speaker_tracks="srt", empty_cues=False,
                              announce=lambda *args, **kwargs: None)
    written = sorted(path.name for path in (tmp_path / "clip-speakers").glob("*.srt"))
    assert written == [f"clip-speakers-speaker{n}.srt" for n in range(4)]


def test_the_knowledge_harvest_marks_the_lines_the_judge_heard():
    from finesub.llm.chunking import SubtitleSegment
    from finesub.llm.knowledge.materials import HEARD_MARK, render_raw_csv
    from finesub.llm.prompts import build_knowledge_update_messages

    rows = render_raw_csv([SubtitleSegment("4", 11.0, 14.0, "そう line 4"), SubtitleSegment("5", 15.0, 17.0, "line 5")],
                          {"4"}).splitlines()[1:]
    assert rows[0].endswith(f"|{HEARD_MARK}そう line 4") and HEARD_MARK not in rows[1]
    plain, heard = (build_knowledge_update_messages(refined=False, task_summary="t", window_packs="p", heard=flag)[0]
                    ["content"] for flag in (False, True))
    assert HEARD_MARK not in plain and HEARD_MARK in heard
