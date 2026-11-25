"""The person table where every track is a person and is identified on evidence: its replay (`speech.diarization.track_people`), what a unit writes and is shown
(`speaker_judge.people_io`), the prompt with it, and the merge it votes (`judgments.merges`). The contract tests the
design's peer review asked for."""

from __future__ import annotations

from finesub.llm.stages.speaker_judge import people_io as PIO
from finesub.llm.stages.speaker_judge import prompt as P
from finesub.llm.stages.speaker_judge.windows import JudgeWindow
from finesub.speech.diarization import track_people as T

from .test_speaker_judge_merges import judged, rows_of, write_sidecar, write_stable

V27 = "speaker-judge-v27"
EMA, COCO = "c:Ema", "c:Coco"


def unit(segment, stage=(), writes=(), people=(), persons=None, opened=(), seen=None, version=V27):
    """A unit's entry; a write is `(who, to)` or the dict `read_people` records; its mentions as `read_people`
    records them."""
    writes = [{"who": w[0], "to": w[1]} if isinstance(w, tuple) else w for w in writes]
    mentions = list(dict.fromkeys([*(key for w in writes for key in (w["who"], w["to"], w.get("via")) if key),
                                   *(persons or {}).values(), *(op["key"] for op in people)]))
    return {"prompt_version": version, "segment": segment, "stage": list(stage), "writes": writes,
            "people": list(people), "persons": persons or {}, "mentions": mentions,
            **({"opened_people": list(opened)} if opened else {}), **({"people_seen": seen} if seen else {})}


def table_after(*units, cast=("Ema", "Coco"), others=None):
    return T.replay(cast, [(f"u{at}", entry) for at, entry in enumerate(units, 1)], others)


def write(table, tracks, body=None, implied=None, labels=(), unit_id="u9", segment=9):
    """What a unit with these `<tracks>` lines (and `<people>` body) records, against `table` as it saw it (no ▶ rows
    to cut from: the cuts are `test_speaker_judge_cuts`')."""
    return PIO.read_people(tracks, body, table, unit_id, implied or {}, labels, segment, rows_on={})


def whys(got):
    return [item["why"] for item in got.get("people_ignored", ())]


# --- replay --------------------------------------------------------------------------------------------------


def test_every_track_on_stage_gets_a_person_after_the_candidates_and_is_called_by_its_number():
    table = table_after(unit(1, stage=[5, 3], opened=["甲"]))
    assert [table.handle(key) for key in table.people] == ["p1", "p2", "s3", "s5", "第1段@甲"]
    assert table.land("t:3") == "t:3"


def test_a_candidate_brings_its_other_names():
    table = table_after(cast=("Ema",), others={"Ema": ("エマ", "桜羽エマ")})
    assert PIO.render_table(table) == "- p1 Ema（别名：エマ、桜羽エマ）"


def test_a_person_opened_by_a_unit_reads_only_while_that_unit_holds_what_it_wrote():
    first = unit(1, stage=[3], opened=["甲"])
    later = unit(2, stage=[4], writes=[("t:4", "t:u1:甲")], seen={"t:u1:甲": T.people_of(first)})
    assert table_after(first, later).land("t:4") == "t:u1:甲"
    again = {**first, "people": [{"op": "add", "key": "t:u1:甲", "text": "店员"}]}  # judged again: other answers
    table = table_after(again, later)
    assert table.land("t:4") == "t:4" and "u2" in table.persons


def test_a_dropped_write_falls_back_to_the_last_one_that_still_reads():
    first = unit(1, stage=[3], opened=["甲"])
    table = table_after(first, unit(2, stage=[4], writes=[("t:4", EMA)]),
                        unit(3, stage=[4], writes=[("t:4", "t:u1:甲")], seen={"t:u1:甲": "stale"}))
    assert table.land("t:4") == EMA


def test_units_before_v27_are_not_replayed():
    """`dev`'s v11 units are reused on resume but write nothing into the table."""
    table = table_after(unit(1, stage=[0], writes=[("t:0", EMA)], version="speaker-judge-v11"),
                        unit(2, stage=[1], writes=[("t:1", COCO)]))
    assert "t:0" not in table.people and table.land("t:1") == COCO and "u1" not in table.persons


def test_taking_back_counts_only_from_a_candidate():
    table = table_after(unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                        unit(2, stage=[0, 1], writes=[("t:0", "t:0"), ("t:1", "t:1")]))
    assert table.history == {"t:0": {EMA, T.TAKEN_BACK}}


# --- what a unit writes --------------------------------------------------------------------------------------


def test_a_write_needs_evidence_and_records_it():
    table = table_after(unit(1, stage=[3, 7]))
    table.stage([3, 7])
    assert write(table, {"s7": "s3（q2 接着 s3 说）", "s3": "p1"})["writes"] == [
        {"who": "t:7", "to": "t:3", "why": "q2 接着 s3 说"}]
    got = write(table, {"s3": "p1", "s7": "s3", "@甲": "p1"})
    assert got["writes"] == [] and whys(got) == ["no evidence", "no evidence", "@label not opened in <judge>"]


def test_evidence_in_either_bracket_with_brackets_inside_cut_at_40():
    table = table_after()
    table.stage([3, 4, 5, 6])
    got = write(table, {"s3": "p1 (qA 被叫「エマ（ちゃん）」)", "s4": "p1 Ema（qB 自我介绍「エマです」",
                        "s5": "p2 Coco", "s6": "p2：" + "长" * 50})
    assert [(w["who"], w["to"], w.get("why", "")[:8]) for w in got["writes"]] == [
        ("t:3", EMA, "qA 被叫「エマ"), ("t:4", EMA, "qB 自我介绍「"), ("t:6", COCO, "长" * 8)]
    assert got["writes"][0]["why"] == "qA 被叫「エマ(ちゃん)」" and got["writes"][2]["cut"] == 50
    assert len(got["writes"][2]["why"]) == T.WHY_CHARS and whys(got) == ["no evidence"]


def test_a_candidate_s_name_or_other_name_alone_is_no_evidence():
    table = table_after(others={"Ema": ("エマ",)})
    table.stage([3, 4, 5])
    got = write(table, {"s3": "p1（Ema）", "s4": "p1（「エマ」）", "s5": "p1 Ema（qA 被叫エマ）"})
    assert [w["who"] for w in got["writes"]] == ["t:5"] and whys(got) == ["no evidence", "no evidence"]


def test_a_write_records_where_its_target_landed_then_and_the_track_it_went_through():
    seen = table_after(unit(1, stage=[3, 7]), unit(2, stage=[3], writes=[("t:3", EMA)]))
    seen.stage([3, 7])
    assert write(seen, {"s7": "s3（q5 接着 s3 说）"})["writes"] == [
        {"who": "t:7", "to": EMA, "via": "t:3", "why": "q5 接着 s3 说"}]
    later = table_after(unit(1, stage=[3, 7]), unit(2, stage=[3], writes=[("t:3", EMA)]),
                        unit(3, stage=[7], writes=[{"who": "t:7", "to": EMA, "via": "t:3"}]),
                        unit(4, stage=[3], writes=[("t:3", COCO)]))
    assert later.land("t:7") == EMA and later.land("t:3") == COCO and later.via["t:7"] == "t:3"


def test_question_mark_or_its_own_number_takes_back_without_evidence():
    table = table_after(unit(1, stage=[0], writes=[("t:0", EMA)]))
    table.stage([0, 1])
    for back in ("?", "？", "s0", "?（qC 被叫ココ）"):
        assert [(w["who"], w["to"]) for w in write(table, {"s0": back})["writes"]] == [("t:0", "t:0")]
    assert write(table, {"s0": "p1（qA 被叫エマ）", "s1": "s0（qB 接着 s0 说）"})["writes"][1] == {
        "who": "t:1", "to": EMA, "via": "t:0", "why": "qB 接着 s0 说"}


def test_a_link_to_a_track_already_under_it_changes_nothing():
    a, b, c, d = "t:1", "t:2", "t:3", "t:4"
    table = table_after(unit(1, stage=[1, 2, 3, 4], writes=[(a, b)]), unit(2, stage=[2], writes=[(b, c)]),
                        unit(3, stage=[3], writes=[(c, d)]))
    assert table.land(a) == d
    table.stage([1, 4])
    got = write(table, {"s4": "s1（q1 一问一答同一方）"})
    assert got["writes"] == [] and whys(got) == ["already one person"]


def test_the_sides_a_track_on_stage_a_listed_track_a_label_of_its_own_or_an_earlier_one():
    table = table_after(unit(1, stage=[3], opened=["甲"], people=[{"op": "add", "key": "t:3", "text": "关西腔"},
                                                                 {"op": "add", "key": "t:u1:甲", "text": "店员"}]),
                        unit(2, stage=[8]))
    table.stage([5])
    got = write(table, {"s5": "p1（qA 被叫エマ）", "s3": "p2（qB 被叫ココ）", "第1段@甲": "s5（qC 一问一答同一方）",
                        "@乙": "第1段@甲（qD 自称店长）", "第9段@丙": "p2（qE 被叫ココ）", "p1": "p2（x）",
                        "s8": "p1（x）", "s9": "p1（x）", "@p2": "p2", "@丁": "p1（x）", "s5 ": "p9（x）",
                        "第2段@甲": "p1（x）"}, labels=["乙", "丙", "p2"])
    assert [(w["who"], w["to"]) for w in got["writes"]] == [
        ("t:5", EMA), ("t:3", COCO), ("t:u1:甲", EMA), ("t:u9:乙", EMA), ("t:u9:丙", COCO)]
    assert got["opened_people"] == ["乙", "丙"]
    assert whys(got) == ["a candidate points at no one", "track neither on stage nor listed",
                         "track neither on stage nor listed", "@label not opened in <judge>", "not a candidate",
                         "no such listed track"]


def test_people_lines_alias_and_note_on_a_track_and_a_candidate_only_without_a_track_here():
    table = table_after(unit(1, stage=[3], people=[{"op": "call", "key": "t:3", "text": "エマ"}]))
    table.stage([3, 5])
    body = "\n".join(["s3 被称为：「エマ」", "s3 被称为：エマちゃん", "s3 被稱為：エマちゃん", "s5 補：" + "长" * 150,
                      "p2 补：只被提到", "p1 补：不收", "@甲 被称为：店长", "@乙 补：不收", "s9 补：不收",
                      "s3 叫：旧写法", "s3 重写：旧写法", "乱写"])
    got = write(table, {"s3": "p1（qA 被叫エマ）"}, body, labels=["甲"])
    assert [(op["op"], op["key"], op["text"][:5]) for op in got["people"]] == [
        ("call", "t:3", "エマちゃん"), ("add", "t:5", "长" * 5), ("add", COCO, "只被提到"), ("call", "t:u9:甲", "店长")]
    assert got["people"][1]["cut"] == 150 and len(got["people"][1]["text"]) == 100
    assert whys(got) == ["alias already there", "alias already there", "on a track here: written on the track",
                         "@label not opened in <judge>", "track neither on stage nor listed", "does not read",
                         "does not read", "does not read"]


def test_pn_answers_only_a_candidate():
    table = table_after(unit(1, stage=[3, 5], writes=[("t:3", EMA)]))
    table.stage([3, 5])
    assert PIO.pickable(table) == {1: EMA, 2: COCO}
    text, implied = PIO.answers_with_people('{"id":"q1","track":"p1"}\n{"id":"q2","track":"p3"}', table)
    assert '"track":"@p1"' in text and '"track":"p3"' in text and implied == {"@p1": EMA}


# --- what a unit is shown ------------------------------------------------------------------------------------


def test_the_table_hangs_tracks_under_whom_they_were_identified_as_with_their_evidence():
    table = table_after(
        unit(1, stage=[3, 5, 7], people=[{"op": "add", "key": "t:7", "text": "关西腔"}],
             writes=[{"who": "t:7", "to": "t:5", "why": "q3 接着说"}]),
        unit(2, stage=[3], writes=[{"who": "t:3", "to": EMA, "why": "q8 被叫エマ"}],
             people=[{"op": "call", "key": "t:3", "text": "エマ"}]),
        unit(3, stage=[9], writes=[{"who": "t:9", "to": EMA, "via": "t:3", "why": "q12 接着 s3 说"}]),
        unit(4, stage=[11]))
    table.stage([11])
    assert PIO.render_table(table).splitlines() == [
        "- p1 Ema（台词里叫：エマ）",
        "  - s3（第2段认作：q8 被叫エマ）（被称为：エマ）",
        "  - s9（第3段经 s3 认作：q12 接着 s3 说）",
        "- p2 Coco",
        "- s5（还没认出）",
        "  - s7（第1段连上：q3 接着说）：〔第1段〕关西腔",
    ]
    assert PIO.profile_names(table, {3: "t:3", 9: "t:9", 7: "t:7", 11: "t:11", 12: EMA}) == {
        3: ("认作 p1 Ema", "前面认作 p1 Ema（第2段：q8 被叫エマ）"),
        9: ("认作 p1 Ema", "前面经 s3 认作 p1 Ema（第3段：q12 接着 s3 说）"),
        7: ("连到 s5", "前面连到 s5（第1段：q3 接着说）；还没认出，s7 自己在人物表里记着〔第1段〕关西腔"), 11: ("", "还没认出"),
        12: ("认作 p1 Ema", "")}
    named = table_after(unit(1, stage=[5, 7], writes=[("t:7", "t:5")]), unit(2, stage=[5], writes=[("t:5", EMA)]))
    assert PIO.status(named, "t:7") == ("认作 p1 Ema", "前面连到 s5（第1段），s5 前面认作 p1 Ema")


def test_a_track_shows_its_latest_notes_up_to_the_limit_and_voiced_tracks_are_listed():
    table = table_after(unit(1, stage=[3], people=[{"op": "add", "key": "t:3", "text": "旧" * 200}]),
                        unit(2, stage=[3], people=[{"op": "add", "key": "t:3", "text": "新" * 200}]),
                        unit(3, stage=[4]))
    table.voiced = {"t:4"}
    assert PIO.render_table(table).splitlines()[2:] == ["- s4（还没认出）", "- s3（还没认出）：〔第2段〕" + "新" * 200]
    assert len(table.people["t:3"].notes) == 2


def test_the_prompt_has_the_table_each_track_says_what_it_was_identified_as_and_voices_off_stage(tmp_path):
    stable = write_stable(tmp_path, ["w0:A", "w0:B", "w0:A", *["w0:B", "w0:A"] * 20, "w1:C"])
    rows = rows_of(stable)
    window = JudgeWindow("0001", "0001", 0, 3)
    table = table_after(unit(1, stage=[0, 2], writes=[{"who": "t:0", "to": EMA, "why": "q1 被叫エマ"}]))
    stage = P.stage_tracks(rows, window, marks=False)
    table.stage(stage)
    voices = {"tiers": {"high": 0.85, "mid": 0.75}, "labels": {}, "pairs": [{"a": "w0:A", "b": "w1:C", "cos": 0.9, "overlap": 0.0}]}
    earlier = [2]
    voiced = [b for _, b, _ in P.voice_pairs(rows, sorted(stage), voices, earlier) if b not in stage]
    assert voiced == [2]
    table.voiced = {"t:2"}
    profile = P.profiles(rows, window, voices=voices, merging=True, people=True, stage=stage,
                         relayed=PIO.profile_names(table, {track: f"t:{track}" for track in range(3)}),
                         off_stage={2: PIO.brief(table, "t:2")})
    system, user, _ = P.judge_prompt(rows, window, title="t", summary="", cast=(), marks=False, profile=profile,
                                     shape=True, people=PIO.table_section(table, 2))
    assert "；前面认作 p1 Ema（第1段：q1 被叫エマ）。" in user and "；还没认出。" in user
    assert "- s0 与 s2（不在这一段，还没认出）：高" in user
    assert "- p1 Ema\n  - s0（第1段认作：q1 被叫エマ）\n- p2 Coco\n- s2（还没认出）" in user
    assert "例句" not in system + user and "`<people>`" in system and "被称为" in system and "重写" not in system
    assert "把两条整轨连起来（`s7 = s6（凭据）`）" in system


# --- the merge -----------------------------------------------------------------------------------------------


def sidecar(tmp_path, labels, *units):
    stable = write_stable(tmp_path, labels)
    rows = [str(at) for at in range(1, len(labels) + 1)]
    per = max(1, len(rows) // len(units))
    spans = [(rows[at * per], rows[len(rows) - 1 if at == len(units) - 1 else (at + 1) * per - 1])
             for at in range(len(units))]
    write_sidecar(stable, [(first, last, {}, [], entry) for (first, last), entry in zip(spans, units)])
    return stable


def fresh(tmp_path, name):
    (tmp_path / name).mkdir()
    return tmp_path / name


def test_two_tracks_landing_on_one_candidate_merge_by_name(tmp_path):
    stable = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                     unit(2, stage=[2, 3], writes=[("t:2", EMA)]))
    result = judged(stable)
    assert result.merged == {2: 0} and [(m.by, m.name) for m in result.merges] == [("name", "Ema")]


def test_taken_back_from_a_candidate_never_votes_again_but_never_identified_still_does(tmp_path):
    back = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                   unit(2, stage=[0, 2], writes=[("t:0", "t:0"), ("t:2", EMA)]),
                   unit(3, stage=[0, 3], writes=[("t:0", EMA)]))
    result = judged(back)
    assert 0 not in result.merged and (0, 0, "names") in result.refused
    never = sidecar(fresh(tmp_path, "x"), ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", "t:0")]),
                    unit(2, stage=[0, 2], writes=[("t:2", EMA)]), unit(3, stage=[0, 3], writes=[("t:0", EMA)]))
    assert judged(never).merged == {2: 0}


def test_a_refused_track_on_a_candidate_neither_votes_nor_links_but_one_on_a_track_links(tmp_path):
    twice = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                    unit(2, stage=[0, 2], writes=[("t:0", COCO), ("t:2", COCO)]))
    result = judged(twice)
    assert result.merged == {} and (0, 0, "names") in result.refused
    linked = sidecar(fresh(tmp_path, "x"), ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                     unit(2, stage=[0, 2], writes=[("t:0", "t:0")]),
                     unit(3, stage=[0, 2], writes=[("t:0", "t:2")]))
    result = judged(linked)
    assert result.merged == {2: 0} and {m.by for m in result.merges} == {"link"}


def test_a_candidate_reached_through_another_track_counts_for_the_veto(tmp_path):
    stable = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA)]),
                     unit(2, stage=[0, 2], writes=[("t:0", "t:2")]), unit(3, stage=[2, 3], writes=[("t:2", COCO)]))
    result = judged(stable)
    assert (0, 0, "names") in result.refused and 0 not in result.merged


def test_tracks_landing_on_one_unidentified_person_link_with_its_own_track(tmp_path):
    stable = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1, 2], writes=[("t:0", "t:2")]),
                     unit(2, stage=[1, 3], writes=[("t:1", "t:2")]))
    result = judged(stable)
    assert set(result.merged) | set(result.merged.values()) == {0, 1, 2}
    assert {m.by for m in result.merges} == {"link"}


def test_one_unit_writing_two_tracks_to_two_candidates_votes_against_a_later_link(tmp_path):
    stable = sidecar(tmp_path, ["A", "B", "C", "D"], unit(1, stage=[0, 1], writes=[("t:0", EMA), ("t:1", COCO)]),
                     unit(2, stage=[0, 1], writes=[("t:0", "t:0"), ("t:1", "t:0")]))
    assert judged(stable).merged == {}  # one link for, one vote against
    linked = sidecar(fresh(tmp_path, "x"), ["A", "B", "C", "D"], unit(1, stage=[0, 1]),
                     unit(2, stage=[0, 1], writes=[("t:1", "t:0")]))
    assert judged(linked).merged == {1: 0}


def test_a_pn_track_votes_its_candidate(tmp_path):
    """Row 4 (on s3) answered `p1`: a new track for Ema, merged with s0, which landed on Ema."""
    stable = write_stable(tmp_path, ["A", "B", "C", "D"])
    write_sidecar(stable, [("1", "2", {}, [], unit(1, stage=[0, 1], writes=[("t:0", EMA)])),
                           ("3", "4", {}, [{"row": "4", "label": "p1", "from": 3}],
                            unit(2, stage=[2, 3], persons={"@p1": EMA}))])
    result = judged(stable)
    assert result.merged == {4: 0} and [(m.by, m.name) for m in result.merges] == [("name", "Ema")]
