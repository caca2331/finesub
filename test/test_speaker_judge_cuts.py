"""A track cut where its speaker changes: `s3 自q91 = 右` in `<tracks>`
(`speaker_judge.people_io.read_people`), the pieces in the person table (`speech.diarization.track_people`), where
the rows land (`judgments.resolve`, `run.cut_results`), what the units after it are shown (`run.cuts_before`,
`run.moved_out`, the profiles' row counts) and the forced cut. The contract tests the design's peer review asked for."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge import run as R
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization import track_people as T

from .test_speaker_judge_merges import judged, rows_of, write_sidecar, write_stable
from .test_speaker_judge_track_people import EMA, COCO, table_after, unit, whys

CUT = "t:u9:切s3自q7"


def staged(*units, stage=(3, 4)):
    table = table_after(*units)
    table.stage(stage)
    return table


def cut(table, tracks, body=None, rows_on=None, guard=None, unit_id="u9", segment=9):
    return PIO.read_people(tracks, body, table, unit_id, {}, (), segment,
                           rows_on={3: ["5", "7", "8"], 4: ["6"]} if rows_on is None else rows_on, guard=guard)


# --- what a unit writes --------------------------------------------------------------------------------------


def test_a_cut_opens_a_piece_that_s3_means_from_then_on_wherever_the_line_stands():
    table = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]))
    got = cut(table, {"s4": "s3（q6 接着 s3 说）", "s3 自q7": "p2（q7 被叫ココ）"}, "s3 被称为：ココちゃん")
    assert got["cuts"] == [{"track": 3, "row": "7"}]
    assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:4", CUT), (CUT, COCO)]
    assert got["people"] == [{"op": "call", "key": CUT, "text": "ココちゃん"}]


def test_a_cut_must_start_at_a_marked_row_of_a_track_on_stage_once():
    table = staged()
    for tracks, why in [({"s3 自q6": "p1（q6 被叫エマ）"}, "not a ▶ row of this track here"),
                        ({"s5 自q9": "p1（q9 被叫エマ）"}, "track not on stage"),
                        ({"s3 自q7": "p1（q7 被叫エマ）", "s3 自q8": "p2（q8 被叫ココ）"}, "cut twice here"),
                        ({"s3-1 自q7": "p1（q7 被叫エマ）"}, "a cut is `sK 自qN` with sK a track on stage"),
                        ({"@甲 自q7": "p1（q7 被叫エマ）"}, "a cut is `sK 自qN` with sK a track on stage")]:
        got = cut(table, tracks)
        assert why in whys(got), tracks
        assert all(w["who"] != "t:3" for w in got["writes"]), tracks  # never read as the whole track
    assert "cuts" not in cut(table, {"s3 自q6": "p1（q6 被叫エマ）"})


def test_a_whole_write_against_the_moves_before_is_a_cut_at_the_first_row_here():
    table = staged()
    got = cut(table, {"s4": "s3（q6 接着说）", "s3": "p1（q7 被叫エマ）"}, "s3 补：自称僕", guard={3: {EMA}})
    assert got["cuts"] == [{"track": 3, "row": "5", "forced": True}]
    forced = "t:u9:切s3自q5"
    assert {(w["who"], w["to"]) for w in got["writes"]} == {(forced, EMA), ("t:4", forced)}
    assert got["people"][0]["key"] == forced
    assert "cuts" not in cut(table, {"s3": "p2（q7 被叫ココ）"}, guard={3: {EMA}})  # another candidate
    assert "cuts" not in cut(table, {"s3": "p1（q7 被叫エマ）"}, guard={4: {EMA}})  # another track
    assert "cuts" not in cut(table, {"s3": "p1"}, guard={3: {EMA}})  # no evidence: not written at all
    already = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]))
    assert "cuts" not in cut(already, {"s3": "p1（q7 被叫エマ）"}, guard={3: {EMA}})  # the moves gathered Ema


def test_the_guard_reads_a_track_through_another_as_the_table_stood_before_this_unit():
    table = staged(unit(1, stage=[3, 4], writes=[{"who": "t:4", "to": EMA, "why": "q2 被叫エマ"}]))
    assert cut(table, {"s3": "s4（同一方）"}, guard={3: {EMA}})["cuts"][0]["forced"]
    fresh = staged()  # s4 is Ema only by this unit's own line: not a forced cut
    assert "cuts" not in cut(fresh, {"s4": "p1（q6 被叫エマ）", "s3": "s4（同一方）"}, guard={3: {EMA}})


# --- the table -----------------------------------------------------------------------------------------------


def cut_unit(segment=2, track=3, row="7", writes=(), uid=None, **more):
    entry = unit(segment, stage=[track, 4], writes=writes, **more)
    entry["cuts"] = [{"track": track, "row": row}]
    return entry


def test_pieces_are_called_s3_1_s3_2_s3_and_the_stage_follows_the_latest():
    first = unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}])
    second = cut_unit(2, row="7", writes=[{"who": "t:u2:切s3自q7", "to": COCO, "why": "q7 被叫ココ"}])
    third = cut_unit(3, row="12")
    table = table_after(first, second, third)
    assert table.pieces[3] == ["t:3", "t:u2:切s3自q7", "t:u3:切s3自q12"]
    assert [table.handle(key) for key in table.pieces[3]] == ["s3-1", "s3-2", "s3"]
    table.stage([3])
    assert table.onstage == {"t:u3:切s3自q12"} and table.current(3) == "t:u3:切s3自q12"
    assert table.land("t:3") == EMA and table.land("t:u2:切s3自q7") == COCO
    assert table.history.get("t:u3:切s3自q12") is None and table.history["t:3"] == {EMA}
    text = PIO.render_table(table)
    assert "- s3-1〔到第2段 q7 之前〕（第1段认作：q2 被叫エマ）" in text
    assert "- s3-2〔到第3段 q12 之前〕（第2段认作：q7 被叫ココ）" in text


def test_the_earlier_piece_can_be_written_and_a_rejudged_cut_unit_drops_what_named_its_piece():
    second = cut_unit(2, row="7")
    table = table_after(unit(1, stage=[3, 4]), second)
    table.stage([3, 4])
    got = cut(table, {"s3-1": "p1（表里 s3-1 被称为エマ）", "s4": "s3（同一方）"}, rows_on={3: ["9"], 4: ["10"]})
    assert {(w["who"], w["to"]) for w in got["writes"]} == {("t:3", EMA), ("t:4", "t:u2:切s3自q7")}
    later = unit(3, stage=[3, 4], writes=got["writes"], seen=got.get("people_seen"))
    assert table_after(unit(1, stage=[3, 4]), second, later).land("t:4") == "t:u2:切s3自q7"
    again = {**second, "cuts": [{"track": 3, "row": "8"}]}
    assert table_after(unit(1, stage=[3, 4]), again, later).land("t:4") == "t:4"
    old = json.dumps({name: unit(1).get(name) for name in ("opened_people", "writes", "people", "persons")},
                     ensure_ascii=False, sort_keys=True)
    assert T.people_of(unit(1)) == hashlib.sha256(old.encode("utf-8")).hexdigest()[:16]  # no cuts: hashed without


# --- where the rows go ---------------------------------------------------------------------------------------


def _stable(tmp_path):
    # rows 1-4 one unit, 5-8 the cutting one, 9-12 a later one; A is s0, B is s1
    return write_stable(tmp_path, ["A", "B", "A", "B", "B", "A", "B", "B", "B", "A", "B", "B"])


def seen(cutter, unit_id, track, row, **more):
    """A later unit's entry fields when it was shown `cutter`'s cut (`unit_id` is the cutter's id), as the run
    records it under a relayed person table (`cuts_seen`)."""
    return {"cuts_seen": {T.opened_key(unit_id, T.cut_label(track, row)): T.people_of(cutter)}, **more}


def test_rows_of_the_track_from_the_cut_on_go_to_its_new_track_unless_a_unit_moved_them(tmp_path):
    stable = _stable(tmp_path)
    cutting = {"cuts": [{"track": 1, "row": "7"}]}
    write_sidecar(stable, [("1", "4", {}, []),
                           ("5", "8", {}, [{"row": "6", "from": 0, "label": T.cut_label(1, "7")}], cutting),
                           ("9", "12", {}, [{"row": "11", "from": 1, "track": 0}], seen(cutting, "0002", 1, "7"))])
    result = judged(stable)
    assert result.new_tracks == (2,)
    assert {row: result.moved[row] for row in sorted(result.moved, key=int)} == {"6": 2, "7": 2, "8": 2, "9": 2,
                                                                               "11": 0, "12": 2}


def test_a_cut_reaches_a_later_unit_s_rows_only_when_that_unit_saw_it(tmp_path):
    # owner 2026-11-25: under `parallel`, or with the cutting unit judged again after the later ones, a later unit
    # never saw the cut (here it wrote s1 whole to Ema); what it kept stays with s1 as it judged, while the cutting
    # unit's own rows from the cut go to the piece
    stable = _stable(tmp_path)
    cutting = {"cuts": [{"track": 1, "row": "7"}]}
    blind = unit(3, stage=[0, 1], writes=[{"who": "t:1", "to": EMA, "why": "q9 被叫エマ"}])
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], cutting), ("9", "12", {}, [], blind)])
    result = judged(stable)
    assert [result.moved.get(row) for row in ("7", "8", "9", "11", "12")] == [2, 2, None, None, None]
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], cutting),
                           ("9", "12", {}, [], {**blind, **seen(cutting, "0002", 1, "7")})])
    assert [judged(stable).moved.get(row) for row in ("9", "11", "12")] == [2, 2, 2]
    other = {"cuts": [{"track": 1, "row": "8"}]}  # the cutter judged again, cutting elsewhere: not what it saw
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], other),
                           ("9", "12", {}, [], {**blind, **seen(cutting, "0002", 1, "7")})])
    assert [judged(stable).moved.get(row) for row in ("7", "8", "9")] == [None, 2, None]
    rewritten = {**cutting, "writes": [{"who": "t:1", "to": EMA, "why": "q6 被叫エマ"}]}  # same cut, other writes
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], rewritten),
                           ("9", "12", {}, [], {**blind, **seen(cutting, "0002", 1, "7")})])
    assert [judged(stable).moved.get(row) for row in ("7", "8", "9")] == [2, 2, None]


def test_a_unit_records_only_each_track_s_latest_piece_it_saw():
    # cut twice before it: its `s1` means the latest piece; were that cutter judged again, its rows must not fall
    # to the earlier piece, which it was told is someone else
    cuts = R.CutState(pieces={1: [(6, T.opened_key("0002", T.cut_label(1, "7"))),
                                  (10, T.opened_key("0003", T.cut_label(1, "11")))]})
    table = SimpleNamespace(created={"0002": "a" * 16, "0003": "b" * 16})
    assert R.cuts_seen(cuts, table) == {T.opened_key("0003", T.cut_label(1, "11")): "b" * 16}


def test_a_cut_again_wins_from_its_row_and_a_rejudged_cutter_drops_its_carried_answers(tmp_path):
    stable = _stable(tmp_path)
    first = {"cuts": [{"track": 1, "row": "5"}]}
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], first),
                           ("9", "12", {}, [], seen(first, "0002", 1, "5", cuts=[{"track": 1, "row": "11"}]))])
    result = judged(stable)
    assert [result.moved[row] for row in ("5", "8", "9", "11", "12")] == [2, 2, 2, 3, 3]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    of = jd.verdict_of(payload["windows"][1])
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], first),
                           ("9", "12", {}, [{"row": "10", "from": 0, "carried": {"window": "0002",
                                                                                  "label": T.cut_label(1, "5"),
                                                                                  "of": of}}])])
    assert judged(stable).moved["10"] == 2
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [], {"cuts": [{"track": 1, "row": "7"}]}),
                           ("9", "12", {}, [{"row": "10", "from": 0, "carried": {"window": "0002",
                                                                                  "label": T.cut_label(1, "5"),
                                                                                  "of": of}}])])
    assert "10" not in judged(stable).moved


def test_the_cutting_unit_s_answers_naming_the_track_from_its_row_on_go_to_the_piece():
    results = [{"row": "5", "from": 0, "track": 1}, {"row": "8", "from": 0, "track": 1},
               {"row": "6", "from": 1, "track": 0}]
    got = R.cut_results(results, [{"track": 1, "row": "7"}], {str(at): at for at in range(1, 13)})
    assert got == [{"row": "5", "from": 0, "track": 1}, {"row": "8", "from": 0, "label": T.cut_label(1, "7")},
                   {"row": "6", "from": 1, "track": 0}]
    # an audio judge's split second half and insert name their speaker the same way (`answer_fields`)
    heard = [{"row": "8", "prefix": "あの", "at": 1, "from": 0, "track": 1}, {"row": "9", "text": "うん", "track": 1}]
    assert [x.get("label") for x in R.cut_results(heard, [{"track": 1, "row": "7"}],
                                                  {str(at): at for at in range(1, 13)})] == [T.cut_label(1, "7")] * 2


def test_an_audio_judge_cannot_name_a_cut_s_piece_by_its_inner_label():
    from finesub.llm.stages.speaker_judge.answers import _speaker

    assert _speaker("@切s1自q7", {0, 1}) is None and _speaker("@甲", {0, 1}).label == "甲"


def test_the_units_after_see_the_piece_its_moves_and_the_guard(tmp_path):
    stable = _stable(tmp_path)
    entries = [unit(1, stage=[0, 1], persons={"@p1": EMA}),
               {**cut_unit(2, track=1, row="7", persons={"@p1": EMA}), "stage": [0, 1]}]
    write_sidecar(stable, [("1", "4", {}, [{"row": "2", "from": 1, "label": "p1"}], entries[0]),
                           ("5", "8", {}, [{"row": "6", "from": 0, "label": "p1"}], entries[1]),
                           ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    state = R.cuts_before(segments, found, found[2], table)
    assert state.start == {1: 6} and state.since == {1: (2, "7")}
    assert state.carry[1] == ("0002", T.cut_label(1, "7"), jd.verdict_of(payload["windows"][1]))
    number = state.number[1]
    assert state.guard == {0: {EMA}}  # s1's row 2 went to Ema before the cut: not this piece's
    assert R.moved_out(segments, found, found[2], state).get(1) is None
    assert R.opened_people(segments, found, found[2], table)[number] == "t:0002:切s1自q7"
    # what the run and the session replay judge the third unit with (`run.relayed`): the same relay, numbered 3
    relay = R.relayed(segments, found, found[2], P.rows_of(segments), (P.Candidate("Ema"), P.Candidate("Coco")), True)
    assert relay["segment"] == 3 and relay["cuts"].since == {1: (2, "7")}
    assert relay["people_keys"][number] == "t:0002:切s1自q7" and relay["table"].pieces == table.pieces
    # what it records of them: the piece, with its cutting unit as it stood (what the reader matches)
    assert R.cuts_seen(relay["cuts"], relay["table"]) == seen(entries[1], "0002", 1, "7")["cuts_seen"]
    alone = R.relayed(segments, found, found[2], P.rows_of(segments), (), False)
    assert alone["segment"] == 3 and alone["table"] is None and alone["carried"] == []


def test_the_profile_says_the_track_s_rows_before_here_and_after(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", "w0:A", "w0:B", "w0:A"])
    rows = rows_of(stable)
    window = JudgeWindow("0002", "0002", 2, 4)
    text = P.profiles(rows, window, merging=True, people=True, stage={0},
                      spread={0: (1, 2, 1, (2, "3"))}).text
    assert "转写里这条轨从第2段 q3 起前面各段 1 行、这一段的 ▶ 行里 2 行、后面各段 1 行" in text
    plain = P.profiles(rows, window, merging=True, people=True, stage={0}, spread={0: (1, 2, 1, None)}).text
    assert "转写里这条轨前面各段 1 行、这一段的 ▶ 行里 2 行、后面各段 1 行" in plain


# --- the merge -----------------------------------------------------------------------------------------------


def test_the_piece_votes_its_own_name_and_the_earlier_piece_keeps_its_history(tmp_path):
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, [], unit(1, stage=[0, 1], writes=[("t:1", EMA)])),
                           ("5", "8", {}, [], {**cut_unit(2, track=1, row="7",
                                                          writes=[{"who": "t:0002:切s1自q7", "to": COCO,
                                                                   "why": "q7 被叫ココ"}],
                                                          opened=()), "stage": [0, 1]}),
                           ("9", "12", {}, [], unit(3, stage=[0, 1], writes=[("t:0", COCO)]))])
    result = judged(stable)
    assert result.merged in ({2: 0}, {0: 2}) and [(m.by, m.name) for m in result.merges] == [("name", "Coco")]
    assert 1 not in result.merged and 1 not in result.merged.values()  # Ema's piece: no one else is Ema


# --- review round 1 ------------------------------------------------------------------------------------------


def test_a_cut_whose_right_side_is_refused_still_stands_with_its_piece_unidentified():
    table = staged()
    for right in ("p1", "s4", "p9（q7 被叫）", "s4-1（q7 自称僕）"):
        got = cut(table, {"s3 自q7": right})
        assert got["cuts"] == [{"track": 3, "row": "7"}], right
        assert all(PIO.CUT_STANDS in why for why in whys(got)) and not got["writes"], right
    assert cut(table, {"s3 自q7": "?"})["cuts"] == [{"track": 3, "row": "7"}]


def test_earlier_pieces_parse_by_their_fixed_number_and_the_latest_has_none():
    second, third = cut_unit(2, row="7"), cut_unit(3, row="12")
    table = table_after(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]), second, third)
    table.stage([3, 4])
    got = cut(table, {"s3-1": "p1（表里 s3-1 被称为エマ）", "s3ー2": "p2（q8 被叫ココ）", "s3-3": "p1（q9 被叫エマ）",
                      "s3前": "p1（q9 被叫エマ）"}, rows_on={3: ["20"], 4: ["21"]})
    assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:3", EMA), ("t:u2:切s3自q7", COCO)]
    assert any(why.startswith("no such piece") for why in whys(got))
    assert any("s3-1, s3-2" in why for why in whys(got))  # v22's name: refused, never read as s3


def test_an_unidentified_piece_votes_nothing_and_takes_back_nothing(tmp_path):
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, [], unit(1, stage=[0, 1], writes=[("t:1", EMA), ("t:0", EMA)])),
                           ("5", "8", {}, [], {**cut_unit(2, track=1, row="7", writes=[("t:0002:切s1自q7",
                                                                                        "t:0002:切s1自q7")]),
                                              "stage": [0, 1]}),
                           ("9", "12", {}, [])])
    result = judged(stable)
    assert set(result.merged.items()) in ({(1, 0)}, {(0, 1)}) and 2 not in result.merged
    table = T.replay((), [("0001", unit(1, stage=[0, 1])), ("0002", cut_unit(2, track=1, row="7",
                                                                           writes=[("t:0002:切s1自q7", "t:0002:切s1自q7")]))])
    assert not table.history.get("t:0002:切s1自q7")


def test_abstained_rows_after_the_cut_go_to_the_piece_and_the_overlap_gate_sees_it(tmp_path):
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, []), ("5", "8", {}, [{"row": "8", "from": 1, "abstain": True}],
                                                 {"cuts": [{"track": 1, "row": "7"}]}), ("9", "12", {}, [])])
    result = judged(stable)
    assert result.moved["8"] == 2 and "8" in result.abstained
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    spans, _ = jd.talk_spans(segments, result)
    assert (14.0, 15.0) in spans[2] and (14.0, 15.0) not in spans[1]  # row 7 talks on the piece


def test_a_later_unit_s_answer_naming_the_track_is_written_as_the_piece(tmp_path):
    from finesub.llm.stages.speaker_judge.answers import Verdict, results

    stable = _stable(tmp_path)
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    verdicts = {f"q{rows[at].id}": Verdict("keep") for at in range(8, 12)}
    verdicts["q10"] = Verdict("track", "接着说", track=1)
    got = results(rows, window, verdicts, {1: ("0002", T.cut_label(1, "7"), "abc")})
    assert got == [{"row": "10", "from": 0, "why": "接着说",
                    "carried": {"window": "0002", "label": T.cut_label(1, "7"), "of": "abc"}}]


def test_a_move_to_an_earlier_piece_is_said_by_that_piece(tmp_path):
    stable = _stable(tmp_path)
    entries = [unit(1, stage=[0, 1], writes=[{"who": "t:1", "to": EMA, "why": "q2 被叫エマ"}]),
               {**cut_unit(2, track=1, row="7", writes=[{"who": "t:0002:切s1自q7", "to": COCO, "why": "q7 被叫ココ"}]),
                "stage": [0, 1]}]
    write_sidecar(stable, [("1", "4", {}, [{"row": "1", "from": 0, "track": 1}, {"row": "3", "from": 0, "track": 1}],
                            entries[0]), ("5", "8", {}, [], entries[1]), ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    state = R.cuts_before(segments, found, found[2], table)
    moved = R.moved_out(segments, found, found[2], state)
    (aside, key), = state.aside.items()
    assert key == "t:1" and moved[0][2] == [(aside, 2)]
    names = PIO.profile_names(table, {aside: key, 1: table.current(1)})
    assert names[aside][0] == "认作 p1 Ema" and names[1][0] == "认作 p2 Coco"


def test_a_move_to_a_track_s_latest_piece_is_said_as_that_track(tmp_path):
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, []),
                           ("5", "8", {}, [{"row": "6", "from": 0, "label": T.cut_label(1, "7")}],
                            {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}),
                           ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    state = R.cuts_before(segments, found, found[2], table)
    assert R.moved_out(segments, found, found[2], state)[0][2] == [(1, 1)] and not state.aside


def test_the_cutting_unit_may_name_the_earlier_piece_and_link_a_new_track_to_it():
    table = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]))
    got = PIO.read_people({"@乙": "s3-1（q8 自称僕，和表里 s3-1 一致）", "s3 自q7": "p2（q7 被叫ココ）",
                           "s3-1": "p1（q5 被叫エマ）"}, None, table, "u9", {}, ("乙",), 9,
                          rows_on={3: ["5", "7", "8"], 4: ["6"]})
    assert got["cuts"] == [{"track": 3, "row": "7"}]
    pairs = [(w["who"], w["to"]) for w in got["writes"]]
    assert (CUT, COCO) in pairs and ("t:3", EMA) in pairs, pairs
    assert [w["to"] for w in got["writes"] if w["who"].endswith("乙")] in (["t:3"], [EMA]), (pairs, whys(got))
