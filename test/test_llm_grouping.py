"""Three passes: pass 1's shape turned into the grouped lines pass 2 reads, and pass 2's rows mapped back to
the original lines (`finesub.llm.grouping`)."""

from __future__ import annotations

import pytest

from finesub.llm.chunking import Seam, SubtitleSegment, SubtitleWindow, load_segments_from_stable_json
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.client import LLMCallResult
from finesub.llm.grouping import Grouping, expand_sources, group_lines, grouped_windows, joined_text, pass2_windows
from finesub.llm.output_protocol import TranslatedCsvSegment
from finesub.llm.token_budget import CorrectionBudget
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.correction import execute_correction_windows

from .conftest import setattr_correction
from .test_llm_speaker_turns import FakeTokenCounter, grouped_reply, labelled_stable, speaker


def lines(*spans, text="あ"):
    """Lines 1..n from (start, end) spans, all with the same short text unless a span gives its own."""
    out = []
    for number, span in enumerate(spans, start=1):
        start, end, *own = span
        out.append(SubtitleSegment(str(number), float(start), float(end), own[0] if own else f"{text}{number}",
                                   seam=Seam.NONE))
    return out


def by_track(*tracks):
    """`different` for lines whose speakers are `tracks` in order."""
    track = {str(number): value for number, value in enumerate(tracks, start=1)}
    return lambda a, b: track[a] != track[b]


def test_a_join_makes_one_line_and_a_may_join_a_mergeable_seam():
    segments = lines((0, 1), (1.2, 2), (3, 4), (4.5, 5))
    grouping = group_lines(segments, {"2": "join", "4": "may_join"}, different=by_track(1, 1, 1, 1), thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1", "2"), ("3",), ("4",))
    assert grouping.seams == {"1": Seam.NONE, "3": Seam.NONE, "4": Seam.MAY}


def test_a_join_never_crosses_two_speakers_and_the_seam_reads_a_change():
    segments = lines((0, 1), (1.2, 2), (2.5, 3))
    grouping = group_lines(segments, {"2": "join", "3": "may_join"}, different=by_track(1, 2, 2), thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1",), ("2",), ("3",))
    assert grouping.refused == (("2", "speaker"),)
    assert grouping.seams == {"1": Seam.NONE, "2": Seam.CHANGE, "3": Seam.MAY}


def test_a_join_past_four_seconds_stays_a_line_of_its_own_marked_mergeable_within_seven():
    segments = lines((0, 2), (2.5, 4.5), (8, 10))
    grouping = group_lines(segments, {"2": "join", "3": "join"}, different=by_track(1, 1, 1), thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1",), ("2",), ("3",))
    assert grouping.refused == (("2", "seconds"), ("3", "seconds"))
    # 1 + 2 span 4.5 s: mergeable; 2 + 3 span 7.5 s: past the absolute seconds, not even mergeable
    assert grouping.seams["2"] is Seam.MAY and grouping.seams["3"] is Seam.NONE


def test_three_lines_join_only_around_a_lone_filler():
    filler = lines((0, 0.5), (0.6, 0.8, "えー"), (0.9, 1.5), (1.6, 2))
    assert group_lines(filler, {"2": "join", "3": "join", "4": "join"},
                       different=by_track(1, 1, 1, 1), thresholds=Thresholds.at(1.0)).groups == (("1", "2", "3"), ("4",))
    words = lines((0, 0.5), (0.6, 0.8, "それでね"), (0.9, 1.5))
    grouping = group_lines(words, {"2": "join", "3": "join"}, different=by_track(1, 1, 1), thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1", "2"), ("3",))
    assert grouping.refused == (("3", "rows"),) and grouping.seams["3"] is Seam.MAY


def test_a_dropped_line_leaves_and_the_next_join_goes_to_the_kept_line_before_it():
    segments = lines((0, 1), (1.1, 1.3, "えっと"), (1.5, 2.5))
    grouping = group_lines(segments, {"2": "drop", "3": "join"}, different=by_track(1, 2, 1), thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1", "3"),)
    assert grouping.dropped == ("2",)


def test_nothing_joins_or_reads_mergeable_across_a_window_s_first_core_line():
    segments = lines((0, 1), (1.2, 2), (2.2, 3))
    grouping = group_lines(segments, {"2": "join", "3": "may_join"}, different=by_track(1, 1, 1), breaks={"2"}, thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1",), ("2",), ("3",))
    assert grouping.refused == (("2", "window"),)
    assert grouping.seams == {"1": Seam.NONE, "2": Seam.NONE, "3": Seam.MAY}
    # a change of speaker at the window's first core line still reads one: the overlap comes before it
    assert group_lines(segments, {}, different=by_track(1, 2, 2), breaks={"2"}, thresholds=Thresholds.at(1.0)).seams["2"] is Seam.CHANGE


def test_a_dropped_first_core_line_hands_the_window_s_edge_to_the_next_kept_line():
    segments = lines((0, 1), (1.1, 1.3, "えー"), (1.5, 2.5), (2.6, 3))
    grouping = group_lines(segments, {"2": "drop", "3": "join", "4": "may_join"}, different=by_track(1, 1, 1, 1),
                           breaks={"1", "2"}, thresholds=Thresholds.at(1.0))
    assert grouping.groups == (("1",), ("3",), ("4",))
    assert grouping.refused == (("3", "window"),)
    assert grouping.seams == {"1": Seam.NONE, "3": Seam.NONE, "4": Seam.MAY}
    assert group_lines(segments, {"2": "drop", "3": "may_join"}, different=by_track(1, 1, 1, 1),
                       breaks={"1", "2"}, thresholds=Thresholds.at(1.0)).seams["3"] is Seam.NONE


def window(chunk_id, segments, overlap=(), preceding=()):
    return SubtitleWindow(chunk_id, [*overlap, *segments], list(overlap), "even", CorrectionBudget(100, 50, 100, "test"),
                          preceding_segments=list(preceding), length_scale=1.0)


def test_grouped_windows_join_the_lines_and_leave_out_groups_cut_by_a_window_s_edge():
    segments = lines((0, 1), (1.2, 2), (2.2, 3), (3.2, 4), (4.2, 5))
    grouping = Grouping((("1", "2"), ("3",), ("4", "5")), {"1": Seam.NONE, "3": Seam.MAY, "4": Seam.CHANGE})
    first = window("0001", segments[:3])
    # the second window's overlap starts inside the group 1 + 2: that group is left out there
    second = window("0002", segments[3:], overlap=segments[1:3], preceding=segments[:1])
    one, two = grouped_windows([first, second], grouping)
    assert [(s.id, s.text, s.seam) for s in one.segments] == [("1", "あ1あ2", Seam.NONE), ("3", "あ3", Seam.MAY)]
    assert one.segments[0].end == 2
    assert [s.id for s in two.overlap_segments] == ["3"]
    assert [s.id for s in two.segments] == ["3", "4"]
    assert two.preceding_segments == []


def test_the_read_only_lines_are_grouped_and_a_window_left_with_no_core_line_is_left_out():
    segments = lines((0, 1), (1.2, 2), (2.2, 2.4, "えー"), (3, 4))
    grouping = group_lines(segments, {"2": "join", "3": "drop"}, different=by_track(1, 1, 1, 1), breaks={"3", "4"}, thresholds=Thresholds.at(1.0))
    first = window("0001", segments[:2])
    emptied = window("0002", segments[2:3], overlap=segments[1:2], preceding=segments[:1])
    last = window("0003", segments[3:], preceding=segments[:3])
    one, three = grouped_windows([first, emptied, last], grouping)
    assert [s.id for s in one.segments] == ["1"] and three.chunk_id == "0003"
    assert [(s.id, s.text) for s in three.preceding_segments] == [("1", "あ1あ2")]


def test_a_group_crossing_a_window_s_first_core_line_is_refused():
    segments = lines((0, 1), (1.2, 2))
    grouping = Grouping((("1", "2"),), {"1": Seam.NONE})
    with pytest.raises(ValueError, match="crosses"):
        grouped_windows([window("0001", segments[:1]), window("0002", segments[1:], overlap=segments[:1])], grouping)


def test_a_window_with_a_core_line_lacking_pass_1_keeps_the_single_pass_prompt_whole():
    segments = lines((0, 1), (1.2, 2), (2.2, 3), (3.2, 4), (4.2, 5), (5.2, 6))
    windows = [window("0001", segments[:3]), window("0002", segments[3:], overlap=segments[2:3])]
    acts = {"2": "join", "3": "may_join", "5": "join"}
    same = by_track(*[1] * 6)
    out, grouping = pass2_windows(windows, segments, acts, {"6"}, different=same, thresholds=Thresholds.at(1.0))
    first, second = out
    assert first.grouped and [line.id for line in first.segments] == ["1", "3"]
    # 0002 lacks pass 1 for line 6: none of its acts are used, so 5 stays a line of its own
    assert second is windows[1] and not second.grouped
    assert grouping.groups == (("1", "2"), ("3",), ("4",), ("5",), ("6",))
    # no usable sidecar: every window keeps the single pass prompt
    out, grouping = pass2_windows(windows, segments, acts, None, different=same, thresholds=Thresholds.at(1.0))
    assert out == windows and grouping.groups == tuple((line.id,) for line in segments)


def test_a_window_on_the_single_pass_prompt_after_a_grouped_one_leaves_its_overlap_rows_whole():
    from finesub.llm.output_protocol import CsvValidationResult
    from finesub.llm.stages.correction.context import CorrectionRun

    segments = lines((0, 1), (1.2, 2), (2.2, 3), (3.2, 4), (4.2, 5), (5.2, 6))
    windows = [window("0001", segments[:3]), window("0002", segments[3:], overlap=segments[1:3])]
    out, grouping = pass2_windows(windows, segments, {"3": "join"}, {"5"}, different=by_track(*[1] * 6), thresholds=Thresholds.at(1.0))
    first, second = out
    assert [line.id for line in first.segments] == ["1", "2"] and not second.grouped

    def row(*ids):
        return TranslatedCsvSegment(ids, float(ids[0]), float(ids[-1]) + 1, "x", "x")

    run = CorrectionRun.__new__(CorrectionRun)  # only what commit_window touches
    run.rendered_segments, run.grouping, run.progress, run.carried = [], grouping, None, None
    run.commit_window(first, CsvValidationResult(True, [row("1"), row("2")], [], []), "")
    # the grouped window lands as the original lines its rows stand for
    assert [segment.source_ids for segment in run.rendered_segments] == [("1",), ("2", "3")]
    # the overlap (lines 2, 3) is the grouped window's core: its row over both original lines stays whole
    run.commit_window(second, CsvValidationResult(True, [row("2"), row("3"), row("4", "5"), row("6")], [], []), "")
    assert [segment.source_ids for segment in run.rendered_segments] == [("1",), ("2", "3"), ("4", "5"), ("6",)]


def test_texts_join_with_a_space_only_between_letters_of_a_spaced_script():
    assert joined_text(["I think", "so."]) == "I think so."
    assert joined_text(["そうですね", "はい"]) == "そうですねはい"
    assert joined_text(["OK", "です"]) == "OKです"
    assert joined_text(["well,", "yes"]) == "well,yes"
    assert joined_text(["round", "2"]) == "round 2"
    assert joined_text(["第１", "２回"]) == "第１２回"
    assert joined_text(["ｿｳ", "ﾃﾞｽ"]) == "ｿｳﾃﾞｽ"  # half-width kana: not a spaced script


def test_rows_name_the_original_lines_again():
    grouping = Grouping((("1", "2"), ("4",), ("5", "6")), {})
    rows = [TranslatedCsvSegment(("1",), 0, 1, "a", "a"), TranslatedCsvSegment(("4", "5"), 1, 2, "b", "b")]
    assert [row.source_ids for row in expand_sources(rows, grouping)] == [("1", "2"), ("4", "5", "6")]


def test_a_run_reads_the_grouped_lines_and_writes_the_original_ones(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1), speaker(1), speaker(1), speaker(2), speaker(2)],
                             texts=["一", "二", "えっと", "四", "五"])
    segments = load_segments_from_stable_json(stable)
    windows, grouping = pass2_windows([window("0001", segments)], segments, {"2": "join", "3": "drop", "5": "may_join"},
                                      frozenset(), different=by_track(1, 1, 1, 2, 2), thresholds=Thresholds.at(1.0))
    seen: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            seen.append(messages[-1]["content"])
            return LLMCallResult(content=grouped_reply("sub|1|一二|一二|high|", "sub|2,3|四五|四五|high|"),
                                 role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
                               max_retries_per_window=1, max_replacements_per_window=0,
                               profile=resolve_profile("text", "none", "quality"), task_artifact_dir=tmp_path / "art",
                               windows_override=windows, grouping=grouping, length_scale=1.0)
    rows = [line.split("|") for line in seen[0].splitlines() if line[:2] in ("1|", "2|", "3|")]
    assert [(row[0], row[4], row[5]) for row in rows] == [("1", "", "一二"), ("2", "换人", "四"), ("3", "可并", "五")]
    annotated = (tmp_path / "out-annotated.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert [row.split("|")[1] for row in annotated] == ["1,2", "4,5"]
    import json  # the window record names the grouped prompt set, which a replay resolves back
    record = json.loads((tmp_path / "art" / "correction-windows.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["variant"] == "capableC-grouped"


def test_two_windows_stitch_with_a_grouped_line_in_the_overlap(tmp_path, monkeypatch):
    stable = labelled_stable(tmp_path, [speaker(1)] * 5, texts=["一", "二", "三", "四", "五"])
    segments = load_segments_from_stable_json(stable)
    grouping = group_lines(segments, {"3": "join"}, different=by_track(1, 1, 1, 1, 1), breaks={"1", "4"}, thresholds=Thresholds.at(1.0))
    windows = [window("0001", segments[:3]), window("0002", segments[3:], overlap=segments[1:3])]
    replies = iter([grouped_reply("sub|1|一|一|high|", "sub|2|二三|二三|high|"),
                    grouped_reply("sub|1|二三|二三|high|", "sub|2,3|四五|四五|high|")])

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(content=next(replies), role=LLMRole.AUDIO_MULTIMODAL, model="fake",
                                 fallback_used=False, raw_response={"candidates": [{"finishReason": "STOP"}]})

    setattr_correction(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(stable_json=stable, output_path=tmp_path / "out.srt", token_counter=FakeTokenCounter(),
                               max_retries_per_window=1, max_replacements_per_window=0,
                               profile=resolve_profile("text", "none", "quality"), task_artifact_dir=tmp_path / "art",
                               windows_override=grouped_windows(windows, grouping), grouping=grouping, length_scale=1.0)
    annotated = (tmp_path / "out-annotated.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert [row.split("|")[1] for row in annotated] == ["1", "2,3", "4,5"]
