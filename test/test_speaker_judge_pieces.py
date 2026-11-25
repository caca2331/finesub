"""Each piece's own voice and fixed piece names: the voice comparison's version 3 (each
row's and label's print, `speech.joint.voices`), the merge gate on a cut track's or an opened track's own rows
(`judgments.merges`), the voice section and the voice-change hint (`run.piece_voices`, `run.kept_rows`), `s3-1`
names and the two-step reading of a unit's cuts (`people_io.read_people`), and an unidentified track's profile line
bringing what the table recorded on it. The contract tests the design asked for."""

from __future__ import annotations

import json

import numpy as np
import pytest

from finesub.llm.stages.speaker_judge import answers as A
from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge import run as R
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization import track_people as T
from finesub.speech.joint import voices as V

from .test_speaker_judge_cuts import _stable, cut, cut_unit, seen, staged
from .test_speaker_judge_merges import fake_embed, judged, rows_of, write_sidecar, write_stable
from .test_speaker_judge_track_people import COCO, EMA, unit, whys

X, Y, XY = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]


def write_voices(stable, rows, labels):
    """A version 3 comparison: `rows` a vector per row (None: no print), each row 4 clean seconds; `labels` label ->
    vector; the pairs from the label vectors."""
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    pairs = [{"a": a, "b": b, "cos": round(float(np.dot(labels[a], labels[b])), 3), "overlap": 0.0}
             for a in sorted(labels) for b in sorted(labels) if a < b]
    payload = {"version": V.VERSION, "key": V.identity(segments), "tiers": {"high": 0.85, "mid": 0.75}, 
               "labels": {label: {"talk": 4.0, "printed": True, "vector": vector} for label, vector in labels.items()},
               "pairs": pairs, "rows": [None if vector is None else {"vector": vector, "seconds": 4.0} for vector in rows]}
    V.voices_path(stable).write_text(json.dumps(payload), encoding="utf-8")
    return payload


# --- the voice comparison, version 3 -------------------------------------------------------------------------


def test_compare_keeps_each_label_s_and_row_s_print_and_a_label_is_its_rows_weighted():
    sr = 16000
    pcm = np.zeros(sr * 30, np.int16)
    pcm[0:sr * 4] = 3000
    pcm[sr * 5:sr * 9] = 1000
    pcm[sr * 10:sr * 14] = 3000
    lines = [{"start": 0.0, "end": 4.0, "line_speaker": "A"}, {"start": 5.0, "end": 9.0, "line_speaker": "A"},
             {"start": 10.0, "end": 14.0, "line_speaker": "C"}, {"start": 15.0, "end": 16.0, "line_speaker": "D"},
             {"start": 20.0, "end": 24.0, "line_speaker": "unknown"}]
    out = V.compare(lines, pcm, fake_embed)
    assert len(out["rows"]) == len(lines) and out["rows"][4] is None
    assert out["rows"][0]["seconds"] == pytest.approx(4.0) and len(out["rows"][0]["vector"]) == 3
    assert "vector" not in out["labels"]["D"] and V.label_print(out, "D") is None
    assert V.piece(out, [3]) is None  # one second: under relink.MIN_LABEL
    assert np.allclose(V.piece(out, [0, 1]), V.label_print(out, "A"), atol=1e-3)
    assert V.cosine(V.piece(out, [0]), V.label_print(out, "C")) > 0.99


def test_an_older_comparison_is_not_read(tmp_path):
    stable = write_stable(tmp_path, ["A", "B"])
    payload = write_voices(stable, [X, Y], {"A": X, "B": Y})
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    assert V.read(stable, segments) is not None
    V.voices_path(stable).write_text(json.dumps({**payload, "version": 2}), encoding="utf-8")
    assert V.read(stable, segments) is None


# --- the merge gate ------------------------------------------------------------------------------------------


def _cut_named_coco(stable):
    """Row 7 on, s1 is a piece named Coco (its new track 2); s0 is called Coco in the last unit."""
    write_sidecar(stable, [("1", "4", {}, [], unit(1, stage=[0, 1], writes=[("t:1", EMA)])),
                           ("5", "8", {}, [], {**cut_unit(2, track=1, row="7",
                                                          writes=[{"who": "t:0002:切s1自q7", "to": COCO,
                                                                   "why": "q7 被叫ココ"}]), "stage": [0, 1]}),
                           ("9", "12", {}, [], unit(3, stage=[0, 1], writes=[("t:0", COCO)]))])


def test_a_cut_s_piece_merges_by_its_own_rows_voice_not_its_label_s(tmp_path):
    stable = _stable(tmp_path)  # A: rows 1 3 6 10 (s0); B: the rest (s1); the piece: B's rows 7 on
    _cut_named_coco(stable)
    a_rows = {0, 2, 5, 9}
    write_voices(stable, [X if k in a_rows else Y if k < 6 else X for k in range(12)], {"A": X, "B": Y})
    assert judged(stable).merged in ({2: 0}, {0: 2})  # the piece sounds like s0, its label B does not
    write_voices(stable, [X if k in a_rows else Y for k in range(12)], {"A": X, "B": X})
    result = judged(stable)
    assert result.merged == {} and (0, 2, "voice") in {(min(a, b), max(a, b), why) for a, b, why in result.refused}


def test_the_cut_track_s_first_piece_is_its_own_rows_and_a_short_piece_passes(tmp_path):
    stable = _stable(tmp_path)
    cutter = {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}
    write_sidecar(stable, [("1", "4", {}, [], unit(1, stage=[0, 1], writes=[("t:1", EMA), ("t:0", EMA)])),
                           ("5", "8", {}, [], cutter), ("9", "12", {}, [], seen(cutter, "0002", 1, "7"))])
    a_rows = {0, 2, 5, 9}
    rows = [X if k in a_rows else Y for k in range(12)]
    write_voices(stable, rows, {"A": X, "B": X})  # the labels alone would merge s0 and s1
    assert (0, 1, "voice") in {(min(a, b), max(a, b), why) for a, b, why in judged(stable).refused}
    write_voices(stable, [None if k in {1, 3, 4} else vector for k, vector in enumerate(rows)], {"A": X, "B": Y})
    assert judged(stable).merged in ({1: 0}, {0: 1})  # s1's first piece has no print: the voice gate lets it by


def test_two_untouched_labels_still_read_their_stored_pair(tmp_path):
    stable = write_stable(tmp_path, ["A", "B", "A", "B"])
    write_sidecar(stable, [("1", "4", {}, [], unit(1, stage=[0, 1], writes=[("t:1", "t:0")]))])
    write_voices(stable, [Y, Y, Y, Y], {"A": X, "B": Y})  # the rows would say one voice; the labels' pair says not
    assert judged(stable).merged == {}


# --- what the judge is shown of voices -----------------------------------------------------------------------


def _three_units(tmp_path):
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, [{"row": "2", "from": 1, "track": 0}], unit(1, stage=[0, 1])),
                           ("5", "8", {}, [], {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}),
                           ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    return stable, segments, found, R.cuts_before(segments, found, found[2], table)


def test_kept_rows_start_at_the_latest_piece_and_leave_out_rows_moved_away(tmp_path):
    _, segments, found, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    assert kept == {0: [0, 2, 5], 1: [6, 7]}  # s1: rows 7 and 8 (its piece); row 2 went to s0, not s0's own


def test_the_voice_section_compares_a_cut_track_as_its_latest_piece_and_names_it(tmp_path):
    stable, segments, found, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    a_rows = {0, 2, 5, 9}
    voices = write_voices(stable, [X if k in a_rows else Y if k < 6 else X for k in range(12)], {"A": X, "B": Y})
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    prints, names, changed = R.piece_voices(rows, window, voices, [0, 1], [], state, kept)
    assert set(prints) == {1} and names == {1: "s1（第2段 q7 起的一截）"} and not changed
    text = P.voice_lines(rows, [0, 1], voices, None, prints, names)
    assert "- s0 与 s1（第2段 q7 起的一截）：高" in text and "切开过的轨比的是它最近切下的那一截" in text
    assert "- s0 与 s1" not in P.voice_lines(rows, [0, 1], voices, None, {}, {})  # its label B: far from A


def test_a_track_whose_rows_here_sound_unlike_its_kept_rows_is_said_to(tmp_path):
    stable, segments, found, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    voices = write_voices(stable, [XY if k in {9} else X if k < 8 else Y for k in range(12)], {"A": X, "B": X})
    _, _, changed = R.piece_voices(rows, window, voices, [0, 1], [], state, kept)
    # s1 here (rows 9 11 12) is Y, its piece before (rows 7 8) X: from its first row (v24); s0 here is 0.8 like before
    assert changed == {1: R.VoiceHint(8, R.VOICE_CHANGE)}
    text = P.profiles(rows, window, voices=voices, merging=True, people=True, stage={0, 1},
                      changed={1: P.voice_hint(1, rows[8].id)}).text
    assert text.count("s1 从 q9 附近起，声音和它前面的行不像") == 1
    assert "s1：" in text.split("s1 从 q9 附近起")[0].splitlines()[-1]
    off = R.piece_voices(rows, window, voices, [0], [1], state, kept)[2]
    assert off == {}  # off stage: never said


# --- names and the two-step reading --------------------------------------------------------------------------


def test_two_cuts_in_a_unit_may_name_each_other_s_earlier_piece_whatever_their_order():
    table = staged(stage=(3, 4))
    rows_on = {3: ["5", "7", "8"], 4: ["6", "9"]}
    for names in ({"s3 自q7": "s4-1（q7 自称僕，和表里 s4-1 一致）", "s4 自q9": "s3-1（q9 自称あてし）"},
                  {"s4 自q9": "s3-1（q9 自称あてし）", "s3 自q7": "s4-1（q7 自称僕，和表里 s4-1 一致）"}):
        got = cut(table, names, rows_on=rows_on)
        assert sorted((c["track"], c["row"]) for c in got["cuts"]) == [(3, "7"), (4, "9")]
        assert {(w["who"], w["to"]) for w in got["writes"]} == {("t:u9:切s3自q7", "t:4"), ("t:u9:切s4自q9", "t:3")}


def test_a_refused_right_side_leaves_both_cuts_standing():
    table = staged(stage=(3, 4))
    got = cut(table, {"s3 自q7": "p1", "s4 自q9": "s3-1（q9 自称あてし）"}, rows_on={3: ["5", "7", "8"], 4: ["6", "9"]})
    assert len(got["cuts"]) == 2 and [(w["who"], w["to"]) for w in got["writes"]] == [("t:u9:切s4自q9", "t:3")]
    assert whys(got) == ["no evidence" + PIO.CUT_STANDS]


def test_a_dash_and_a_number_is_a_piece_or_refused_never_the_track():
    second, third = cut_unit(2, row="7"), cut_unit(3, row="12")
    table = staged(unit(1, stage=[3, 4]), second, third)
    for dash in "-－ー—–‐":
        got = cut(table, {"s4": f"s3{dash}2（q21 接着 s3-2 那一截说）"}, rows_on={3: ["20"], 4: ["21"]})
        assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:4", "t:u2:切s3自q7")], dash
    for name in ("s3-3", "s3-0", "s3前", "p1-1"):
        got = cut(table, {"s4": f"{name}（q21 接着说）"}, rows_on={3: ["20"], 4: ["21"]})
        assert not got["writes"] and whys(got), name


def test_an_earlier_piece_answered_in_judge_is_sent_back_with_where_it_belongs(tmp_path):
    stable = write_stable(tmp_path, ["A", "B", "A"])
    rows = rows_of(stable)
    ids = [f"q{row.id}" for row in rows]
    reply = "<judge>\n" + "\n".join(json.dumps({"id": key, "track": "s1-1" if key == ids[1] else "keep"})
                                    for key in ids) + "\n</judge>"
    with pytest.raises(A.AnswerError) as error:
        A.parse_judge(reply, ids, [0, 1])
    assert "earlier pieces (s3-1) are written in <tracks>" in str(error.value)


def test_pieces_keep_their_numbers_after_another_cut():
    first = cut_unit(2, row="7")
    table = staged(unit(1, stage=[3, 4]), first)
    assert table.handle("t:3") == "s3-1"
    later = staged(unit(1, stage=[3, 4]), first, cut_unit(3, row="12"))
    assert [later.handle(key) for key in later.pieces[3]] == ["s3-1", "s3-2", "s3"]


# --- an unidentified track's profile line --------------------------------------------------------------------


def test_an_unidentified_track_s_profile_brings_its_aliases_and_notes():
    table = staged(unit(1, stage=[3, 4], people=[{"op": "call", "key": "t:3", "text": "氷上"},
                                                 {"op": "add", "key": "t:3", "text": "自称「私」，使用敬语"}]))
    short, long = PIO.status(table, "t:3")
    assert short == "" and long == "还没认出；人物表里记着它被称为 氷上，〔第1段〕自称「私」，使用敬语"
    assert PIO.status(table, "t:4") == ("", "还没认出")
    linked = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": "t:4", "why": "q7 接着 s4 说"}],
                         people=[{"op": "call", "key": "t:3", "text": "氷上"}]))
    assert PIO.status(linked, "t:3")[1].endswith("；还没认出，s3 自己在人物表里记着被称为 氷上")
    named = staged(unit(1, stage=[3, 4], writes=[{"who": "t:4", "to": EMA, "why": "q2 被叫エマ"},
                                                {"who": "t:3", "to": "t:4", "why": "q7 接着 s4 说"}],
                        people=[{"op": "call", "key": "t:3", "text": "氷上"}]))
    assert "还没认出" not in PIO.status(named, "t:3")[1]  # linked to someone identified: no clues


# --- review round 1 ------------------------------------------------------------------------------------------


def test_a_voice_file_the_gate_cannot_read_is_said_not_passed_over(tmp_path):
    stable = _stable(tmp_path)
    _cut_named_coco(stable)
    payload = write_voices(stable, [X] * 12, {"A": X, "B": Y})
    V.voices_path(stable).write_text(json.dumps({**payload, "version": 2}), encoding="utf-8")
    said = []
    jd.read_judgments(stable, warn=said.append)
    assert any("pass no voice gate" in message for message in said)
    V.voices_path(stable).unlink()
    said.clear()
    jd.read_judgments(stable, warn=said.append)
    assert not said  # no file: nothing to say


def test_a_cut_saying_its_piece_is_the_one_it_was_cut_from_stands_unidentified():
    table = staged(stage=(3, 4))
    got = cut(table, {"s3 自q7": "s3-1（q7 自称僕）"})
    assert got["cuts"] == [{"track": 3, "row": "7"}] and not got["writes"]
    assert whys(got) == ["the new piece is said to be the one it was cut from" + PIO.CUT_STANDS]
    again = staged(unit(1, stage=[3, 4]), cut_unit(2, row="5"), stage=(3, 4))  # s3-1 is t:3, s3-2 the piece at q5
    got = cut(again, {"s3 自q7": "s3-1（q7 又被叫エマ，和 s3-1 一样）"})
    assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:u9:切s3自q7", "t:3")]  # back to the first person
