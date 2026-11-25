"""Where a track's voice changes in a unit and the script's own cut when the reply leaves it whole:
`run.voice_split`, `run.voice_guards`, `people_io.read_people(voice_cuts=...)`, and a linked track's profile line
bringing what the table recorded on the last track of its links (`people_io.status`). The contract tests the design
asked for."""

from __future__ import annotations

from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import run as R
from finesub.llm.stages.speaker_judge.answers import Verdict
from finesub.llm.stages.speaker_judge.windows import JudgeWindow

from .test_speaker_judge_cuts import staged
from .test_speaker_judge_merges import rows_of, write_stable
from .test_speaker_judge_track_people import COCO, EMA, unit, whys

X, Y, Z = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]


def voices(*vectors, seconds=None):
    """A comparison holding only rows: a vector per row, 4 clean seconds each unless `seconds` says otherwise."""
    seconds = seconds or {}
    return {"tiers": {"high": 0.85, "mid": 0.75}, "rows": [{"vector": vector, "seconds": seconds.get(k, 4.0)} for k, vector in enumerate(vectors)]}


# --- where the voice changes ---------------------------------------------------------------------------------


def test_a_change_inside_the_unit_is_found_at_its_row_under_the_lower_line():
    found = R.voice_split(voices(X, X, X, X, Y, Y), [2, 3, 4, 5], [0, 1])
    assert found == R.VoiceHint(4, R.VOICE_SPLIT)  # the whole unit against before is 0.71: v23 said nothing


def test_a_unit_wholly_someone_else_s_is_found_at_its_first_row_under_the_v23_line():
    assert R.voice_split(voices(X, X, Y, Y, Y), [2, 3, 4], [0, 1]) == R.VoiceHint(2, R.VOICE_CHANGE)


def test_one_voice_throughout_and_a_track_first_heard_here_say_nothing():
    assert R.voice_split(voices(X, X, X, X), [2, 3], [0, 1]) is None
    assert R.voice_split(voices(X, X, Y, Y), [0, 1, 2, 3], []) is None


def test_a_unit_wholly_changed_keeps_the_v23_line_where_its_lowest_row_is():
    # the whole unit against before is 0.27, from its second row on 0.0: the lowest is there, the line stays 0.6
    found = R.voice_split(voices(X, [0.6, 0.8, 0.0], Z, Z), [1, 2, 3], [0])
    assert found == R.VoiceHint(2, R.VOICE_CHANGE)


def test_a_side_too_short_to_print_is_not_compared_and_equals_go_to_the_earliest():
    assert R.voice_split(voices(X, X, Y, seconds={2: 1.0}), [1, 2], [0]) is None  # the tail alone: one second
    assert R.voice_split(voices(X, Y, Z), [1, 2], [0]) == R.VoiceHint(1, R.VOICE_CHANGE)  # both rows 0.0


# --- the script's own cut ------------------------------------------------------------------------------------


def _unit(tmp_path):
    """s0 throughout: rows 1-2 before the unit (kept), rows 3-6 in it; rows 5 and 6 another voice."""
    stable = write_stable(tmp_path, ["A"] * 6)
    rows = rows_of(stable)
    window = JudgeWindow("0002", "0002", 2, 6)
    prints = voices(X, X, X, X, Y, Y)
    hints = {0: R.voice_split(prints, [2, 3, 4, 5], [0, 1])}
    return rows, window, prints, hints


def _verdicts(rows, **moved):
    return {f"q{row.id}": moved.get(f"q{row.id}", Verdict("keep")) for row in rows}


def test_a_hinted_track_left_whole_is_cut_at_the_hinted_row(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    assert R.voice_guards(rows, window, prints, hints, _verdicts(rows), set(), {0: [0, 1]}) == {0: rows[4].id}
    abstained = _verdicts(rows, q5=Verdict("abstain"))
    assert R.voice_guards(rows, window, prints, hints, abstained, set(), {0: [0, 1]}) == {0: rows[4].id}


def test_a_hinted_track_the_reply_cut_anywhere_is_not_cut_again(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    assert R.voice_guards(rows, window, prints, hints, _verdicts(rows), {0}, {0: [0, 1]}) == {}


def test_abstained_rows_stay_and_are_measured(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    alike = voices(X, X, X, X, Y, X)  # the row after the hinted one sounds as before: abstained, it is measured
    away = _verdicts(rows, q5=Verdict("track", track=1), q6=Verdict("abstain"))
    assert R.voice_guards(rows, window, alike, hints, away, set(), {0: [0, 1]}) == {}
    gone = _verdicts(rows, q5=Verdict("track", track=1), q6=Verdict("track", track=1))
    assert R.voice_guards(rows, window, alike, hints, gone, set(), {0: [0, 1]}) == {}


def test_the_cut_falls_on_the_first_row_from_the_hint_that_stays(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    one_moved = _verdicts(rows, q5=Verdict("track", track=1))
    assert R.voice_guards(rows, window, prints, hints, one_moved, set(), {0: [0, 1]}) == {0: rows[5].id}


def test_changed_rows_moved_away_leave_nothing_to_cut(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    moved = _verdicts(rows, q5=Verdict("new", label="甲"), q6=Verdict("track", track=1))
    assert R.voice_guards(rows, window, prints, hints, moved, set(), {0: [0, 1]}) == {}
    short = voices(X, X, X, X, Y, Y, seconds={5: 1.0})  # what stays from the hint is too short to print
    assert R.voice_guards(rows, window, short, hints, _verdicts(rows, q5=Verdict("track", track=1)), set(),
                          {0: [0, 1]}) == {}


def test_what_stays_after_judging_is_measured_again_under_the_hint_s_line(tmp_path):
    rows, window, prints, hints = _unit(tmp_path)
    alike = voices(X, X, X, X, Y, X)  # the hinted row moved, the row after it sounds as before
    moved = _verdicts(rows, q5=Verdict("track", track=1))
    assert R.voice_guards(rows, window, alike, hints, moved, set(), {0: [0, 1]}) == {}


# --- what a guarded unit's lines become ----------------------------------------------------------------------


ROWS_ON = {3: ["5", "7", "8"], 4: ["6"]}


def _read(table, names, body=None, voice_cuts=None):
    return PIO.read_people(names, body, table, "u9", {}, (), 9, rows_on=ROWS_ON, voice_cuts=voice_cuts)


def test_a_guarded_track_is_cut_and_every_line_on_it_this_unit_is_dropped():
    table = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]))
    got = _read(table, {"s3": "p2（q7 被叫ココ）", "s4": "s3（q6 接着 s3 说）"}, "s3 被称为：ココちゃん\ns4 补：自称僕",
                voice_cuts={3: "7"})
    assert got["cuts"] == [{"track": 3, "row": "7", "voice_guard": True}]
    assert not got["writes"] and [op["key"] for op in got["people"]] == ["t:4"]
    assert whys(got) == [PIO.VOICE_GUARDED] * 3


def test_a_line_on_the_guarded_track_s_earlier_piece_stands():
    table = staged(unit(1, stage=[3, 4]))
    got = _read(table, {"s4": "s3-1（q6 自称あてし，和表里 s3-1 一致）"}, voice_cuts={3: "7"})
    assert [(w["who"], w["to"]) for w in got["writes"]] == [("t:4", "t:3")]


def test_the_reply_s_own_cut_wins_and_a_refused_one_is_no_cut():
    table = staged(unit(1, stage=[3, 4]))
    got = _read(table, {"s3 自q8": "?"}, voice_cuts={3: "7"})
    assert got["cuts"] == [{"track": 3, "row": "8"}]
    refused = _read(table, {"s3 自q6": "?"}, voice_cuts={3: "7"})  # q6 is s4's: the cut does not stand
    assert refused["cuts"] == [{"track": 3, "row": "7", "voice_guard": True}]
    assert whys(refused) == ["not a ▶ row of this track here"]  # set aside once


def test_v22_s_forced_cut_holds_the_script_s_cut_back():
    table = staged(unit(1, stage=[3, 4], writes=[{"who": "t:3", "to": EMA, "why": "q2 被叫エマ"}]))
    got = PIO.read_people({"s3": "p2（q7 被叫ココ）"}, None, table, "u9", {}, (), 9, rows_on=ROWS_ON,
                          guard={3: {COCO}}, voice_cuts={3: "7"})
    # the moves before put s3's rows on Coco: v22 cuts at its first row, the voice adds nothing
    assert got["cuts"] == [{"track": 3, "row": "5", "forced": True}]


def test_another_track_s_cut_naming_the_guarded_track_stands_without_its_right_side():
    table = staged(unit(1, stage=[3, 4]))
    got = _read(table, {"s4 自q6": "s3（q6 接着 s3 说）"}, voice_cuts={3: "7"})
    assert {(c["track"], c["row"]) for c in got["cuts"]} == {(4, "6"), (3, "7")} and not got["writes"]
    assert whys(got) == [PIO.VOICE_GUARDED + PIO.CUT_STANDS]


# --- a linked track's profile line ---------------------------------------------------------------------------


def test_a_track_linked_to_one_identified_brings_its_own_and_that_track_s_clues():
    table = staged(unit(1, stage=[3, 4, 5],
                        writes=[{"who": "t:4", "to": EMA, "why": "q2 被叫エマ"},
                                {"who": "t:3", "to": "t:4", "why": "q7 接着 s4 说"},
                                {"who": "t:5", "to": "t:3", "why": "q9 接着 s3 说"}],
                        people=[{"op": "call", "key": "t:4", "text": "兄貴"},
                                {"op": "call", "key": "t:3", "text": "シナズさん"}]), stage=(3, 4, 5))
    assert PIO.status(table, "t:3")[1].endswith("；s3 自己在人物表里记着被称为 シナズさん；s4 在人物表里记着被称为 兄貴")
    assert PIO.status(table, "t:5")[1].endswith("；s4 在人物表里记着被称为 兄貴")  # two links on: the last track's


def test_a_chain_ending_at_a_track_no_one_is_yet_brings_its_own_and_the_end_s_clues():
    table = staged(unit(1, stage=[3, 4, 5],
                        writes=[{"who": "t:3", "to": "t:4", "why": "q7 接着 s4 说"},
                                {"who": "t:5", "to": "t:3", "why": "q9 接着 s3 说"}],
                        people=[{"op": "call", "key": "t:4", "text": "兄貴"},
                                {"op": "call", "key": "t:5", "text": "シナズさん"}]), stage=(3, 4, 5))
    assert PIO.status(table, "t:5")[1].endswith("；还没认出，s5 自己在人物表里记着被称为 シナズさん；s4 在人物表里记着被称为 兄貴")
