"""Overlapping lines (`docs/joint-asr.md`): lines of one `line_speaker` never overlap, lines of different
ones may, and every pass that keeps lines apart does so per group key (`subtitles/line_groups.py`). Lines without
the key -- every local task but its overlap lane -- are one group, so those passes behave exactly as before."""

from __future__ import annotations

from finesub.speech.recognition import segments as segment_ops
from finesub.subtitles import line_groups, time_order


def line(start, end, text, speaker=None, words=None):
    out = {"start": start, "end": end, "text": text,
           "words": words if words is not None else [{"start": start, "end": end, "word": text}]}
    if speaker is not None:
        out[line_groups.LINE_SPEAKER] = speaker
    return out


def test_no_key_is_one_group_and_a_keyless_line_among_keyed_ones_is_its_own():
    assert line_groups.groups([None, None, None]) == [[0, 1, 2]]
    assert line_groups.groups(["A", None, "B", "A", None]) == [[0, 3], [1], [2], [4]]


def test_in_order_is_by_start_then_key():
    items = [line(2.0, 3.0, "c", "B"), line(1.0, 2.0, "b", "B"), line(1.0, 2.0, "a", "A")]
    assert [s["text"] for s in line_groups.in_order(items)] == ["a", "b", "c"]


def test_grouped_hands_a_keyless_list_through_untouched():
    items = [line(2.0, 3.0, "late"), line(0.0, 1.0, "early")]
    assert line_groups.grouped(items, lambda group: group) == items  # not reordered


def test_the_timeline_contract_runs_per_speaker():
    # B's backchannel sits inside A's long line, and B's word sticks out past B's end: B is repaired (recorded
    # at its place in the input), the overlap with A is not a defect.
    a = line(0.0, 6.0, "long", "A")
    b = line(2.0, 2.4, "un", "B", words=[{"start": 2.0, "end": 2.8, "word": "un"}])
    a2 = line(6.5, 8.0, "more", "A")
    out, repairs = segment_ops.normalize_segment_timeline([a, b, a2])
    assert [(s["text"], s["start"], s["end"]) for s in out] == [("long", 0.0, 6.0), ("un", 2.0, 2.8),
                                                                 ("more", 6.5, 8.0)]
    assert [(r["kind"], r["index"]) for r in repairs] == [("contain", 1)]  # B's position in the input


def test_the_timeline_contract_without_keys_is_the_old_one():
    a = line(0.0, 6.0, "long")
    b = line(2.0, 2.4, "un", words=[{"start": 2.0, "end": 2.8, "word": "un"}])
    out, repairs = segment_ops.normalize_segment_timeline([a, b])
    assert [r["kind"] for r in repairs] == ["contain", "overlap"]  # recorded as it always was
    assert out[1]["end"] == 2.8


def test_the_order_check_runs_per_speaker():
    a = line(0.0, 6.0, "long", "A", words=[{"start": 0.0, "end": 3.0, "word": "x"},
                                           {"start": 3.0, "end": 6.0, "word": "y"}])
    b = line(2.0, 2.4, "un", "B")
    assert time_order.first_backward([a, b], using="words") is None
    same = time_order.first_backward([a, line(2.0, 2.4, "un", "A")], using="words")
    assert same is not None and same.index == 1
    keyless = [{k: v for k, v in s.items() if k != line_groups.LINE_SPEAKER} for s in (a, b)]
    assert time_order.first_backward(keyless, using="words").index == 1


# --- rendering and postprocessing ----------------------------------------------------------------------------

import json  # noqa: E402

import pytest  # noqa: E402

from finesub.reporting import reporting_to  # noqa: E402
from finesub.subtitles import postprocess, rendering  # noqa: E402
from finesub.subtitles.model import format_srt_timestamp, parse_srt, validate_srt_text  # noqa: E402

TIMELINE = postprocess.TimelineParams(end_extend_seconds=0.3, flash_gap_seconds=0.2)

#: A (S01) 0-6 s, B's backchannel (S02) 2-3 s inside it, A again 6.5-8 s, then A2 overlapping A's own line.
CLOUD = [line(0.0, 6.0, "long", "S01"), line(2.0, 3.0, "un", "S02"), line(6.5, 8.0, "more", "S01"),
         line(7.5, 9.0, "own", "S01")]


def _srt(cues):
    return "".join(f"{i}\n{format_srt_timestamp(a)} --> {format_srt_timestamp(b)}\n{t}\n\n"
                   for i, (a, b, t) in enumerate(cues, start=1))


def _spans(text):
    return [(c.start, c.end, c.text) for c in parse_srt(text)]


def test_the_span_writer_keeps_two_speakers_and_truncates_one(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": CLOUD}), encoding="utf-8")
    written = rendering.convert_json_to_srt(stable, output_path=tmp_path / "x.srt", validate=False)
    assert written.keys == ["S01", "S02", "S01", "S01"]
    assert _spans(written.path.read_text(encoding="utf-8")) == [
        (0.0, 6.0, "long"), (2.0, 3.0, "un"), (6.5, 7.5, "more"), (7.5, 9.0, "own")]


def test_a_cue_resolve_overlaps_drops_takes_its_key_with_it():
    rows = [(0.0, 2.0, "a", "S01"), (0.0, 1.0, "gone", "S01"), (0.5, 1.5, "b", "S02")]
    fixed, found = rendering.resolve_overlaps(rows)
    assert [(r[2], r[3]) for r in fixed] == [("a", "S01"), ("b", "S02")] and found == 1


def test_a_local_task_has_no_keys(tmp_path):
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": [line(0.0, 1.0, "a"), line(2.0, 3.0, "b")]}), encoding="utf-8")
    assert rendering.convert_json_to_srt(stable, output_path=tmp_path / "x.srt", validate=False).keys is None


def test_word_cues_of_two_speakers_come_in_order_by_start():
    segs = [line(0.0, 6.0, "long", "S01", words=[{"start": 0.0, "end": 3.0, "word": "x"},
                                                  {"start": 3.0, "end": 6.0, "word": "y"}]),
            line(2.0, 2.5, "un", "S02")]
    assert [c[2] for c in rendering.word_cues(segs)] == ["x", "un", "y"]


@pytest.mark.parametrize("profile", [0, 4])
def test_postprocess_keeps_two_speakers_overlapping_and_truncates_one(profile):
    text = _srt([(0.0, 6.0, "long"), (2.0, 3.0, "un"), (6.5, 8.0, "more"), (7.5, 9.0, "own")])
    keys = ["S01", "S02", "S01", "S01"]
    out, report = postprocess.postprocess_srt_text(text, profile=profile, keys=keys, timeline=TIMELINE)
    spans = _spans(out)
    assert spans[1][:2] == (2.0, 3.3 if profile == 0 else 3.0)  # B is padded, never cut against A
    assert spans[0][1] >= 6.0 and spans[2][1] == 7.5  # A's own overlap is cut, the other speaker's is not
    assert report.overlaps_fixed == 1


def test_postprocess_without_keys_truncates_as_it_always_did():
    text = _srt([(0.0, 6.0, "long"), (2.0, 3.0, "un")])
    out, report = postprocess.postprocess_srt_text(text, profile=4)
    assert _spans(out)[0][1] == 2.0 and report.overlaps_fixed == 1


def test_the_pad_never_makes_an_overlap_across_speakers_that_was_not_there():
    # A ends at 5.0; B starts 5.1; A's pad must stop at B's start, B's pad at A2's.
    text = _srt([(0.0, 5.0, "a"), (5.1, 6.0, "b"), (6.1, 7.0, "a2")])
    out, _ = postprocess.postprocess_srt_text(text, profile=1, keys=["S01", "S02", "S01"], timeline=TIMELINE)
    assert [s[1] for s in _spans(out)] == [5.1, 6.1, 7.3]


def test_a_keyless_cue_among_keyed_ones_keeps_its_overlap_and_unknown_is_one_speaker():
    text = _srt([(0.0, 4.0, "a"), (1.0, 2.0, "inserted"), (5.0, 8.0, "u1"), (6.0, 7.0, "u2")])
    out, _ = postprocess.postprocess_srt_text(text, profile=4, keys=["S01", None, "unknown", "unknown"])
    assert [s[1] for s in _spans(out)] == [4.0, 2.0, 6.0, 7.0]


def test_the_audio_end_bound_runs_per_speaker():
    # Two speakers' cues past the end: each gets the last millisecond, neither is pushed back by the other.
    text = _srt([(10.2, 11.0, "a"), (10.4, 11.0, "b")])
    out, _ = postprocess.postprocess_srt_text(text, profile=1, keys=["S01", "S02"], timeline=TIMELINE,
                                               audio_end=10.0)
    assert _spans(out) == [(9.999, 10.0, "a"), (9.999, 10.0, "b")]
    out, _ = postprocess.postprocess_srt_text(text, profile=1, timeline=TIMELINE, audio_end=10.0)
    assert _spans(out)[0][0] < 9.999  # one speaker: the earlier cue makes room


def test_the_validator_reports_an_overlap_only_within_a_speaker():
    text = _srt([(0.0, 6.0, "long"), (2.0, 3.0, "un")])
    overlap = "Segment 2 starts before the previous segment ends."
    assert overlap in validate_srt_text(text).errors
    assert overlap not in validate_srt_text(text, keys=["S01", "S02"]).errors
    assert overlap in validate_srt_text(text, keys=["S01", "S01"]).errors


def test_postprocess_moves_no_cue_so_the_keys_still_match_the_result():
    # Out of order by start (a model's insert, a judge-filled time): sorting would hand B's key to the insert.
    text = _srt([(0.0, 10.0, "a"), (9.0, 9.8, "inserted"), (2.0, 3.0, "b"), (11.0, 12.0, "a2")])
    keys = ["S01", None, "S02", "S01"]
    out, _ = postprocess.postprocess_srt_text(text, profile=0, keys=keys, timeline=TIMELINE)
    assert [s[2] for s in _spans(out)] == ["a", "inserted", "b", "a2"]
    assert not [e for e in validate_srt_text(out, keys=keys).errors if "previous segment ends" in e]


def test_the_pad_reads_the_next_start_whatever_the_cue_order():
    # A judge split puts A's second half right after A; the backchannel B, covered by A, starts earlier but comes
    # after it: B's pad must still stop at Ab's start.
    text = _srt([(0.0, 2.5, "a"), (2.6, 10.0, "ab"), (2.0, 2.4, "b")])
    out, _ = postprocess.postprocess_srt_text(text, profile=1, keys=["S01", "S01", "S02"], timeline=TIMELINE)
    assert _spans(out)[2][1] == 2.6


def test_keys_that_do_not_match_the_cues_are_an_error():
    with pytest.raises(ValueError):
        postprocess.postprocess_srt_text(_srt([(0.0, 1.0, "a")]), profile=4, keys=["S01", "S02"])


# --- the LLM stage's writers ---------------------------------------------------------------------------------

from finesub.llm import chunking  # noqa: E402
from finesub.llm.output_protocol import KIND_INSERT, KIND_SUB, TranslatedCsvSegment  # noqa: E402
from finesub.llm.stages.correction.run import product_keys  # noqa: E402


def _row(ids, kind=KIND_SUB, start=0.0, end=1.0):
    return TranslatedCsvSegment(source_ids=tuple(ids), start=start, end=end, corrected_text="x", translation="x",
                                kind=kind)


def test_a_product_row_is_keyed_by_its_first_source_and_an_inserted_one_by_none():
    lines = chunking._subtitle_segments(CLOUD[:3], None)
    assert [s.group_key for s in lines] == ["S01", "S02", "S01"]
    assert chunking.segment_keys(lines) == ["S01", "S02", "S01"]
    rows = [_row(["1"]), _row(["2", "3"]), _row(["4.0", "1.0"], kind=KIND_INSERT)]
    assert product_keys(rows, lines) == ["S01", "S02", None]
    assert product_keys(rows, chunking._subtitle_segments([line(0.0, 1.0, "a")], None)) is None


def _finished(tmp_path, annotated_rows, cues):
    from finesub.llm.correction_translation import _finished_keys
    from finesub.llm.output_protocol import ANNOTATED_CSV_HEADER

    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": CLOUD[:3]}), encoding="utf-8")
    out = tmp_path / "x.srt"
    (tmp_path / "x-annotated.csv").write_text(
        "\n".join([ANNOTATED_CSV_HEADER, *annotated_rows]) + "\n", encoding="utf-8")
    translated = tmp_path / "x-translated.srt"
    translated.write_text(_srt(cues), encoding="utf-8")
    seen = []

    class Reporter:
        def __getattr__(self, name):
            return lambda *a, **k: seen.append((name, a)) if name == "warning" else None

    with reporting_to(Reporter()):
        keys = _finished_keys(stable, out, translated)
    return keys, seen


def test_finishing_from_disk_reads_the_keys_off_the_annotated_rows(tmp_path):
    rows = ["sub|1|6.0|0.0|a|a|||||", "sub|2|1.0|0.0|b|b|||||", "insert|4.0,1.0|1.0|0.0|c|c|||||"]
    keys, seen = _finished(tmp_path, rows, [(0.0, 6.0, "a"), (2.0, 3.0, "b"), (4.0, 5.0, "c")])
    assert keys == ["S01", "S02", None] and seen == []


def test_finishing_from_disk_warns_and_drops_the_keys_when_the_rows_do_not_match(tmp_path):
    keys, seen = _finished(tmp_path, ["sub|1|6.0|0.0|a|a|||||"], [(0.0, 6.0, "a"), (2.0, 3.0, "b")])
    assert keys is None and [s[1][0] for s in seen] == ["final-srt-keys"]
    keys, seen = _finished(tmp_path, ["sub|9|6.0|0.0|a|a|||||"], [(0.0, 6.0, "a")])
    assert keys is None and len(seen) == 1


def test_a_row_with_an_empty_translation_is_in_the_corrected_products_only(tmp_path, monkeypatch):
    """A correct-only task's line that is only filler (its `polished` cell empty): in `-corrected.srt` and
    `-annotated.csv`, not in `-translated.srt` or the result, and finishing from disk keys the cues without it.
    Both text columns come out in simplified forms, whatever the model wrote."""
    from dataclasses import replace as replaced
    from finesub.llm.correction_translation import _finished_keys
    from finesub.llm.stages.correction.run import _write_products
    from finesub.subtitles.model import parse_srt

    lines = chunking._subtitle_segments(CLOUD[:3], None)
    monkeypatch.setattr(chunking, "load_product_segments", lambda *_a, **_k: (lines, None))
    rendered = [replaced(_row(["1"]), corrected_text="陳世驤說", translation="陳世驤說"),
                replaced(_row(["2"], start=1.0, end=2.0), corrected_text="呃", translation=""),
                _row(["3"], start=2.0, end=3.0)]
    out = tmp_path / "x.srt"
    products = _write_products(rendered, out, "", None, lines, postprocess_profile=None, audio_end=None,
                               simplified=True)

    assert len(parse_srt(products.corrected_text)) == 3
    assert len(parse_srt(products.translated_text)) == 2
    annotated = (tmp_path / "x-annotated.csv").read_text(encoding="utf-8")
    assert "|呃||" in annotated
    assert "陈世骧说" in products.corrected_text and "陈世骧说" in products.translated_text
    assert "陳" not in annotated
    assert _finished_keys(tmp_path / "x-stable.json", out, products.translated) == ["S01", "S01"]


def test_finishing_from_disk_gives_the_keys_the_run_wrote_with(tmp_path, monkeypatch):
    """The two-step final SRT (`--stage translated-srt`, then `final-srt`) is keyed like the one-pass one, for a
    split row's second half (`<id>b`, the row's fields) and a line the judge heard missing (`<id>i`, no key)."""
    from finesub.llm.correction_translation import _finished_keys
    from finesub.llm.output_protocol import render_annotated_csv, render_translated_segments_as_srt

    product = [line(0.0, 6.0, "a", "S01"), dict(line(1.0, 2.0, "half", "S02"), id="2b"),
               {"id": "3i", "start": 2.5, "end": 3.0, "text": "heard", "words": []}, line(7.0, 8.0, "d", "S01")]
    lines = chunking._subtitle_segments(product, None)
    monkeypatch.setattr(chunking, "load_product_segments", lambda *_a, **_k: (lines, None))
    rendered = [_row([lines[0].id], start=0.0, end=6.0), _row(["2b"], start=1.0, end=2.0),
                _row(["3i"], start=2.5, end=3.0), _row(["8.0", "0.5"], kind=KIND_INSERT, start=8.0, end=8.5),
                _row([lines[3].id], start=7.0, end=8.0)]
    (tmp_path / "x-annotated.csv").write_text(render_annotated_csv(rendered), encoding="utf-8")
    translated = tmp_path / "x-translated.srt"
    translated.write_text(render_translated_segments_as_srt(rendered), encoding="utf-8")
    one_pass = product_keys(rendered, lines)
    assert one_pass == ["S01", "S02", None, None, "S01"]
    assert _finished_keys(tmp_path / "x-stable.json", tmp_path / "x.srt", translated) == one_pass


# --- the raw SRT writers, end to end through both timeline file passes ---------------------------------------

#: CLOUD's raw SRT after the timeline half of profile 0 (`[postprocess]` defaults): B stays on screen with A and
#: is padded, A's own overlap is cut, A's flash gap before its next line closed.
RAW = [(0.0, 6.5, "long"), (2.0, 3.3, "un"), (6.5, 7.5, "more"), (7.5, 9.3, "own")]


def test_the_raw_srt_stage_keeps_two_speakers_through_both_timeline_passes(tmp_path):
    from finesub import stages

    paths = stages.default_pipeline_paths(tmp_path / "x.wav", tmp_path / "x.srt")
    paths.stable_json.write_text(json.dumps({"segments": CLOUD}), encoding="utf-8")
    stages._write_raw_srt(paths, word=False, postprocess_profile=0, empty_cues=False)
    assert _spans(paths.raw_srt.read_text(encoding="utf-8")) == RAW


def test_the_correction_stage_rewrites_the_raw_srt_with_the_keys(tmp_path, monkeypatch):
    from finesub.llm.stages.correction import run

    class Planned(Exception):
        pass

    def stop(**_kwargs):
        raise Planned  # the raw SRT is written before the windows are planned

    monkeypatch.setattr(run, "resolve_correction_plan", stop)
    stable = tmp_path / "x-stable.json"
    stable.write_text(json.dumps({"segments": CLOUD}), encoding="utf-8")
    with pytest.raises(Planned):
        run.execute_correction_windows(stable_json=stable, output_path=tmp_path / "x.srt", length_scale=1.0)
    assert _spans((tmp_path / "x-raw.srt").read_text(encoding="utf-8")) == RAW


def test_each_overlap_lane_is_a_group_and_every_other_local_line_one_more():
    lane = line_groups.key_of({"overlap_lane": line_groups.lane_id((10.0, 10.24))})
    other = line_groups.key_of({"overlap_lane": line_groups.lane_id((10.0, 10.48))})
    assert lane.startswith(line_groups.LANE) and lane != other
    assert line_groups.key_of({"overlap_lane": "10.0-10.24", "line_speaker": "S01"}) == "S01"
    assert line_groups.groups([None, lane, None, None, lane, other]) == [[0, 2, 3], [1, 4], [5]]
    assert not line_groups.speakers([None, lane, other]) and line_groups.speakers([None, lane, "S01"])
    # with a speaker anywhere a line without a key is its own group again
    assert line_groups.groups([None, lane, "S01", None]) == [[0], [1], [2], [3]]


def test_pass_3_reads_only_the_rows_in_the_subtitle(tmp_path, monkeypatch):
    """A correct-only task's filler line (empty `polished`) has no cue: pass 3 reads the other rows against the
    cues and leaves it as it was."""
    from dataclasses import replace as replaced
    from types import SimpleNamespace
    from finesub.llm.stages.correction import run as correction_run
    from finesub.llm.stages.proofread import prompt as proofread_prompt
    from finesub.llm.stages.proofread import run as proofread_run

    lines = chunking._subtitle_segments(CLOUD[:3], None)
    rendered = [_row(["1"]), replaced(_row(["2"], start=1.0, end=2.0), corrected_text="呃", translation=""),
                _row(["3"], start=2.0, end=3.0)]
    passed = correction_run._write_products(rendered, tmp_path / "x.srt", "", None, lines, postprocess_profile=None,
                                            audio_end=None, simplified=True)
    seen = {}

    def fake_run(segments, read, cut, **_kwargs):
        seen["rows"], seen["lines"] = len(segments), len(read)
        return SimpleNamespace(segments=[replaced(row, translation="校过") for row in segments], status="done",
                               windows=1, reused=0, failed=0, stopped=False, changes=[], usage={}, older_prompt={})

    monkeypatch.setattr(proofread_run, "run_proofread", fake_run)
    monkeypatch.setattr(proofread_prompt, "windows_of", lambda ids, windows: [])
    monkeypatch.setattr(correction_run, "_settled_style_block", lambda run: "")
    monkeypatch.setattr(correction_run, "_window_terms", lambda run, ids: {})
    monkeypatch.setattr(correction_run, "append_task_artifact", lambda *a, **k: None)
    run = SimpleNamespace(resume=False, max_retries_per_window=0, context_pack=None, client=None, task_id="t",
                          profile=SimpleNamespace(continuity="parallel"), parallel_window_limit=1)
    out = correction_run._proofread_rows(run, rendered, passed, [], [], title="", extra_style="", artifact_dir=tmp_path)

    assert (seen["rows"], seen["lines"]) == (2, 2)
    assert [row.translation for row in out] == ["校过", "", "校过"]
