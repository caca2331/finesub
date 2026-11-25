"""Linking back what belongs together and changing back to an earlier piece: a cut track's earlier pieces' own
voice (`run.old_rows`, `run.piece_voices`) in the voice section and in a hint (`VoiceHint.like`); the section listing
`high` first (`prompt.voice_lines`); the lines that change back to, link from or take back to an earlier piece
(`people_io.read_people`). The contract tests the design asked for."""

from __future__ import annotations

import json

from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge import run as R
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import judgments as jd
from finesub.speech.diarization import track_people as T

from .test_speaker_judge_cuts import _stable, cut, cut_unit, staged
from .test_speaker_judge_merges import rows_of, write_sidecar, write_stable
from .test_speaker_judge_pieces import write_voices
from .test_speaker_judge_track_people import unit, whys

X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def voices(*vectors, seconds=None):
    """A comparison holding only rows: a vector per row, 4 clean seconds each unless `seconds` says otherwise."""
    seconds = seconds or {}
    return {"tiers": {"high": 0.85, "mid": 0.75}, "rows": [{"vector": vector, "seconds": seconds.get(k, 4.0)} for k, vector in enumerate(vectors)]}


# --- the hint ------------------------------------------------------------------------------------------------


def test_the_profile_line_names_the_earlier_pieces():
    assert P.voice_hint(1, "170") == "；s1 从 q170 附近起，声音和它前面的行不像"
    assert P.voice_hint(1, "170", like=["s1-1", "s1-2"]) \
        == "；s1 从 q170 附近起，声音和它前面的行不像，和旧截 s1-1、s1-2 的声音像"


# --- earlier pieces' voices ----------------------------------------------------------------------------------


def _three_units(tmp_path, cut_entry=None):
    """s0: rows 1 3 6 10 (A); s1: the rest (B). Unit 1 moves row 2 to s0; unit 2 cuts s1 at q7; unit 3 is judged."""
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, [{"row": "2", "from": 1, "track": 0}], unit(1, stage=[0, 1])),
                           ("5", "8", {}, [], cut_entry or {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}),
                           ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    state = R.cuts_before(segments, found, found[2], table)
    return stable, segments, found, table, state


def test_an_earlier_piece_is_its_own_rows_the_units_before_left_on_it(tmp_path):
    _, segments, found, table, state = _three_units(tmp_path)
    assert R.old_rows(segments, found, found[2], state) == {"t:1": [3, 4]}  # row 2 went to s0; 7 on: the latest


def _old_voices(stable, latest=X, s0=Y, overlap=0.0):
    """Rows: s0's (0 2 5 9) `s0`, s1's earlier piece (3 4) Y, its latest piece (6 7) `latest`, s1 here (8 10 11) Y."""
    by_row = {0: s0, 2: s0, 5: s0, 9: s0, 1: Y, 3: Y, 4: Y, 6: latest, 7: latest, 8: Y, 10: Y, 11: Y}
    payload = write_voices(stable, [by_row[k] for k in range(12)], {"A": s0, "B": X})
    payload["pairs"][0]["overlap"] = overlap
    return payload


def test_an_earlier_piece_is_compared_with_the_tracks_here_and_listed_only_high(tmp_path):
    stable, segments, found, table, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    old = R.old_rows(segments, found, found[2], state)
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    prints, names, _ = R.piece_voices(rows, window, _old_voices(stable), [0, 1], [], state, kept, old, table)
    assert names["t:1"] == "s1-1（到第2段 q7 之前的一截）"
    text = P.voice_lines(rows, [0, 1], _old_voices(stable), None, prints, names)
    assert "- s1-1（到第2段 q7 之前的一截） 与 s0：高" in text
    assert "s1-1（到第2段 q7 之前的一截） 与 s1" not in text  # its latest piece is X
    mid = _old_voices(stable, s0=[0.8, 0.6, 0.0])  # 0.6 like the earlier piece: not listed
    prints = R.piece_voices(rows, window, mid, [0, 1], [], state, kept, old, table)[0]
    assert "s1-1" not in P.voice_lines(rows, [0, 1], mid, None, prints, names)


def test_the_overlap_gate_spares_an_earlier_piece_and_its_own_track_s_latest(tmp_path):
    stable, segments, found, table, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    old = R.old_rows(segments, found, found[2], state)
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    both = _old_voices(stable, latest=Y, overlap=10.0)  # A and B talk at once; s1's latest piece sounds as before
    prints, names, _ = R.piece_voices(rows, window, both, [0, 1], [], state, kept, old, table)
    text = P.voice_lines(rows, [0, 1], both, None, prints, names)
    assert "- s1-1（到第2段 q7 之前的一截） 与 s1（第2段 q7 起的一截）：高" in text
    assert "与 s0" not in text


def test_an_earlier_piece_too_short_is_left_out(tmp_path):
    stable, segments, found, table, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    old = R.old_rows(segments, found, found[2], state)
    rows = rows_of(stable)
    payload = _old_voices(stable)
    payload["rows"][3] = None
    payload["rows"][4]["seconds"] = 2.0
    prints, names, _ = R.piece_voices(rows, JudgeWindow("0003", "0003", 8, 12), payload, [0, 1], [], state, kept,
                                      old, table)
    assert "t:1" not in prints and "t:1" not in names


def test_a_hint_on_a_cut_track_names_the_earlier_pieces_its_rows_sound_like(tmp_path):
    stable, segments, found, table, state = _three_units(tmp_path)
    kept = R.kept_rows(segments, found, found[2], state)
    old = R.old_rows(segments, found, found[2], state)
    rows = rows_of(stable)
    window = JudgeWindow("0003", "0003", 8, 12)
    _, _, changed = R.piece_voices(rows, window, _old_voices(stable), [0, 1], [], state, kept, old, table)
    assert changed[1] == R.VoiceHint(8, R.VOICE_CHANGE, like=("t:1",))  # s1 here is Y: its latest X, s1-1 Y
    unlike = _old_voices(stable)
    for k in (3, 4):
        unlike["rows"][k]["vector"] = Z
    _, _, changed = R.piece_voices(rows, window, unlike, [0, 1], [], state, kept, old, table)
    assert changed[1] == R.VoiceHint(8, R.VOICE_CHANGE)


def test_the_people_section_lists_high_first_keeping_each_tier_s_order(tmp_path):
    stable = write_stable(tmp_path, ["A", "B", "C"])
    rows = rows_of(stable)
    labels = {"A": X, "B": [0.8, 0.6, 0.0], "C": [0.9, 0.436, 0.0]}  # A-B mid, A-C and B-C high
    payload = write_voices(stable, [labels[k] for k in "ABC"], labels)
    people = P.voice_lines(rows, [0, 1, 2], payload, None, {}, {})
    assert [line for line in people.splitlines() if "：" in line and line.startswith("- s")][:3] \
        == ["- s0 与 s2：高", "- s1 与 s2：高", "- s0 与 s1：中"]


# --- lines on earlier pieces ---------------------------------------------------------------------------------


def _cut_once():
    return staged(unit(1, stage=[3, 4]), cut_unit(2, row="7"))  # s3: s3-1 (t:3), then s3 from unit 2's q7


def test_a_cut_may_say_the_new_speaker_is_an_earlier_piece_but_not_the_one_it_was_cut_from():
    table = _cut_once()
    back = cut(table, {"s3 自q8": "s3-1（q8 自称あてし，和表里 s3-1 一致）"})
    assert back["cuts"] == [{"track": 3, "row": "8"}]
    assert [(w["who"], w["to"]) for w in back["writes"]] == [(T.cut_key("u9", 3, "8"), "t:3")]
    same = cut(table, {"s3 自q8": "s3-2（q8 自称あてし）"})  # the piece from q7: the one it was cut from
    assert same["cuts"] == [{"track": 3, "row": "8"}] and not same["writes"]
    assert whys(same) == ["the new piece is said to be the one it was cut from" + PIO.CUT_STANDS]


def test_an_earlier_piece_may_be_linked_and_the_latest_taken_back_to_it():
    table = _cut_once()
    linked = cut(table, {"s3-1": "s4（q6 s4 接着 s3-1 讲迷路的经过，自称あてし）"})
    assert [(w["who"], w["to"]) for w in linked["writes"]] == [("t:3", "t:4")]
    undone = cut(table, {"s3": "s3-1（q7 起仍自称あてし，被叫エマ）"})
    assert [(w["who"], w["to"]) for w in undone["writes"]] == [("t:u2:切s3自q7", "t:3")]


def test_the_script_s_cut_drops_nothing_written_on_an_earlier_piece():
    table = _cut_once()
    got = PIO.read_people({"s3-1": "s4（q6 s4 接着 s3-1 说，自称あてし）"}, None, table, "u9", {}, (), 9,
                          rows_on={3: ["5", "7", "8"], 4: ["6"]}, voice_cuts={3: "7"})
    assert got["cuts"] == [{"track": 3, "row": "7", "voice_guard": True}]
    assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:3", "t:4")]



def _cut_twice(tmp_path):
    """s1 cut at q4 (unit 1) and q7 (unit 2): s1-1 is its row 2, s1-2 its rows 4 5, the latest its rows 7 8."""
    stable = _stable(tmp_path)
    write_sidecar(stable, [("1", "4", {}, [], {**cut_unit(1, track=1, row="4"), "stage": [0, 1]}),
                           ("5", "8", {}, [], {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}),
                           ("9", "12", {}, [])])
    segments = json.loads(stable.read_text(encoding="utf-8"))["segments"]
    payload = json.loads(jd.sidecar_path(stable).read_text(encoding="utf-8"))
    found = jd.units(payload["windows"], jd.segment_ids(segments))
    table = T.replay(("Ema", "Coco"), [(other.id, other.entry) for other in found[:2]])
    return stable, segments, found, table, R.cuts_before(segments, found, found[2], table)


def test_a_track_cut_twice_gives_each_earlier_piece_its_own_rows_and_name(tmp_path):
    stable, segments, found, table, state = _cut_twice(tmp_path)
    old = R.old_rows(segments, found, found[2], state)
    middle = table.pieces[1][1]
    assert old == {"t:1": [1], middle: [3, 4]}
    by_row = {k: X for k in range(12)}
    by_row.update({1: [0.0, 0.9, 0.436], 3: Y, 4: Y, 8: Y, 10: Y, 11: Y})  # s1-1 0.9 like s1 here, s1-2 1.0
    payload = write_voices(stable, [by_row[k] for k in range(12)], {"A": X, "B": X})
    kept = R.kept_rows(segments, found, found[2], state)
    prints, names, changed = R.piece_voices(rows_of(stable), JudgeWindow("0003", "0003", 8, 12), payload, [0, 1], [],
                                            state, kept, old, table)
    assert names[middle] == "s1-2（到第2段 q7 之前的一截）" and names["t:1"] == "s1-1（到第1段 q4 之前的一截）"
    assert changed[1].like == (middle, "t:1")  # both high, the most alike first


def test_every_kind_of_cut_leaves_an_earlier_piece(tmp_path):
    for flag in ({}, {"forced": True}, {"voice_guard": True}):
        entry = {**cut_unit(2, track=1, row="7"), "stage": [0, 1]}
        entry["cuts"] = [{"track": 1, "row": "7", **flag}]
        _, segments, found, _, state = _three_units(tmp_path, entry)
        assert R.old_rows(segments, found, found[2], state) == {"t:1": [3, 4]}, flag
