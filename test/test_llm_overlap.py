"""Overlapping lines in the LLM layer (`docs/llm_design_notes.md`「`同时` / `换回` 的决策记录」): what reads the group key and the
times -- covered, the 同时 / 换回 seams, cut points, window and clip ends, merged lines -- and that without keys
(a local task without the overlap lane) none of it moves."""

from __future__ import annotations

from finesub.llm import chunking
from finesub.llm.prompt_constants import Thresholds
from finesub.llm.chunking import Seam, SubtitleSegment, SubtitleWindow
from finesub.media.clips import compute_clip_range
from finesub.subtitles import line_groups


def seg(i, start, end, key=None, text="x", seam=Seam.CHANGE):
    return SubtitleSegment(id=str(i), start=start, end=end, text=text, seam=seam, group_key=key)


def seams(lines):
    return [line.seam.value for line in chunking.turn_seams(lines)][1:]


def test_a_backchannel_is_together_and_the_long_line_goes_back():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.5, 12, "A")]
    assert seams(lines) == ["同时", "换回"]


def test_three_people_nested():
    # A 0-10, B 2-6, C 3-4, A again at 10.5: B and C are together, the last line goes back
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 6, "B"), seg(3, 3, 4, "C"), seg(4, 10.5, 12, "A")]
    assert seams(lines) == ["同时", "同时", "换回"]


def test_a_chain_is_together_only():
    lines = [seg(1, 0, 5, "A"), seg(2, 4, 8, "B"), seg(3, 7, 9, "A")]
    assert seams(lines) == ["同时", "同时"]


def test_the_same_person_twice_inside_a_long_line_marks_the_second_only_as_it_was():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 4, 5, "B", seam=Seam.NONE)]
    assert seams(lines) == ["同时", ""]


def test_a_question_and_answer_is_not_going_back():
    lines = [seg(1, 0, 2, "A"), seg(2, 2.5, 4, "B"), seg(3, 4.5, 6, "A")]
    assert seams(lines) == ["换人", "换人"]


def test_without_keys_nothing_changes():
    lines = [seg(1, 0, 10), seg(2, 2, 3), seg(3, 10.5, 12)]
    assert seams(lines) == ["换人", "换人"]
    assert line_groups.covered(lines) == [False, False, False]
    assert line_groups.run_end(lines) == 12


def test_a_line_in_the_gap_between_the_judges_halves_is_not_together():
    # the judge cut A 0-10 at its pause: its second half follows it in the product (`diarization.product`),
    # before B, which falls in the pause
    lines = [seg(1, 0, 2, "A"), seg("1b", 3, 10, "A", seam=Seam.NONE), seg(2, 2.5, 2.8, "B")]
    assert line_groups.covered(lines) == [False, False, False]
    assert seams(lines)[1] != "同时"
    # nor does A going on after the second half go back: B ended before that half began
    assert seams(lines + [seg(3, 10.5, 12, "A")])[2] == "换人"


def test_no_cut_parts_two_lines_said_at_once_whatever_the_list_order():
    # the judge's second half Ab precedes B in the list though B starts first; Ab and B overlap
    lines = [seg(1, 0, 2, "A", text="a。"), seg("1b", 3, 10, "A", text="ab。", seam=Seam.NONE),
             seg(2, 2.5, 9, "B", text="b。"), seg(3, 10.5, 12, "A", text="c。")]
    parted = line_groups.parted(lines)
    assert parted == [False, True, False, False]
    assert not chunking.is_reasonable_boundary(lines, 1, parted=parted)  # Ab | B would part them
    assert chunking.is_reasonable_boundary(lines, 2, parted=parted)
    assert line_groups.parted([seg(s.id, s.start, s.end, None, text=s.text) for s in lines]) == [False] * 4


def test_a_keyless_line_among_keyed_ones_is_its_own_speaker():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, None)]
    assert seams(lines) == ["同时"]


def test_a_sentence_end_before_a_covered_line_is_no_cut():
    lines = [seg(1, 0, 10, "A", text="長い。"), seg(2, 2, 3, "B", text="うん。"), seg(3, 10.5, 12, "A")]
    parted = line_groups.parted(lines)
    assert not chunking.is_reasonable_boundary(lines, 0, parted=parted)  # before B, which A covers
    assert chunking.is_reasonable_boundary(lines, 1, parted=parted)  # before A's next line
    assert chunking.is_reasonable_boundary(lines, 0)  # without the flags: the old rule


def test_window_and_clip_end_at_the_latest_end_of_another_speaker():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B")]
    window = SubtitleWindow(chunk_id="0001", segments=lines, overlap_segments=[], boundary_reason="",
                            budget=None, length_scale=1.0)
    assert window.end == 10
    assert compute_clip_range(lines, global_first_id="x", global_last_id="y", audio_duration=100) == (0.0, 15.0)
    local = [seg(1, 0, 10), seg(2, 2, 3)]
    assert compute_clip_range(local, global_first_id="x", global_last_id="y", audio_duration=100) == (0.0, 8.0)


def test_a_merge_across_together_ends_at_the_latest_end_of_another_speaker():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B")]
    assert line_groups.run_end(lines) == 10
    assert line_groups.run_end([seg(1, 0, 4, "A"), seg(2, 3, 5, "A")]) == 5


def test_overlapping_marks_both_sides():
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 11, 12, "A"), seg(4, 13, 14, "B")]
    assert line_groups.overlapping(lines) == [True, True, False, False]


def test_pass1_never_joins_two_speakers_talking_at_once():
    from finesub.llm.stages.three_pass import _different

    class Layer:
        def speakers(self):
            return {}

    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.5, 12, "B")]
    different = _different(Layer(), lines)
    assert different("1", "2") and not different("2", "3") and not different("1", "3")
    assert not _different(Layer(), [seg(1, 0, 10), seg(2, 2, 3)])("1", "2")  # local: the labels only


def test_the_judge_halves_never_cut_before_a_covered_row():
    from finesub.llm.stages.speaker_judge.windows import halves

    rows = [seg(1, 0, 2, "A", text="a。"), seg(2, 2.5, 10, "A", text="b。"), seg(3, 3, 4, "B", text="c。"),
            seg(4, 10.5, 12, "A", text="d。")]
    assert halves(rows, 0, 4) == [(0, 3), (3, 4)]  # the middle cut would fall before B, whom A covers
    local = [seg(r.id, r.start, r.end, None, text=r.text) for r in rows]
    assert halves(local, 0, 4) == [(0, 2), (2, 4)]


def test_every_window_of_a_keyed_task_explains_together_and_back():
    from finesub.llm.prompts import build_correction_csv_messages

    def system(lines):
        lines = chunking.turn_seams(lines)
        window = SubtitleWindow(chunk_id="0001", segments=lines, overlap_segments=[], boundary_reason="", budget=None, length_scale=1.0)
        return build_correction_csv_messages(window=window)[0]["content"]

    # one system prompt per task: a window without overlap explains them as well (the prefix cache)
    nested = system([seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.5, 12, "A")])
    plain = system([seg(1, 0, 2, "A"), seg(2, 2.5, 4, "B")])
    assert nested == plain
    assert "标着「同时」的行也是换人" in plain and "标着「换回」的行也是换人" in plain
    assert "换人、同时、换回标记或空|" in plain and "`turn` 列标着「换人」「同时」或「换回」的行不与上一行合并" in plain
    assert "「换人」「同时」或「换回」是转写模型给的线索" in plain
    local = system([seg(1, 0, 2), seg(2, 2.5, 4)])
    assert "标着「同时」" not in local and "换人标记或空|" in local and "`turn` 列标着「换人」的行不与上一行合并" in local
    assert "「换人」是声学层给的线索" in local
    # a local window holding an overlap lane line explains 同时 / 换回, but its labels are still the acoustic layer's
    lane = system([seg(1, 0, 10, line_groups.LANE), seg(2, 2, 3), seg(3, 10.5, 12)])
    assert "标着「同时」的行也是换人" in lane and "「换人」「同时」或「换回」是声学层给的线索" in lane
    assert "转写模型" not in lane
    # a window edge between the lane line and the row it covers: the covered row's 同时 is still explained
    covered = [seg(2, 2, 3, seam=Seam.TOGETHER), seg(3, 10.5, 12)]
    window = SubtitleWindow(chunk_id="0002", segments=covered, overlap_segments=[], boundary_reason="", budget=None, length_scale=1.0)
    alone = build_correction_csv_messages(window=window)[0]["content"]
    assert "标着「同时」的行也是换人" in alone and "转写模型" not in alone


def test_the_judge_reads_a_cloud_tasks_tracks_as_its_transcriptions_speakers():
    from dataclasses import replace

    from finesub.llm.stages.speaker_judge import prompt as judge
    from finesub.llm.stages.speaker_judge.windows import JudgeWindow

    from .test_speaker_judge_stage import segments_of, speaker

    rows = judge.rows_of(segments_of([(speaker(i % 2), 2, f"台词{i}") for i in range(1, 5)]))
    window = JudgeWindow("0001", "0001", 0, 4)

    def system(rows):
        profile = judge.profiles(rows, window)
        return judge.judge_prompt(rows, window, title="t", summary="", cast=(), marks=False, profile=profile,
                                  shape=True, audio=judge.audio_layout(rows, window, profile.samples))[0]

    local = system(rows)
    cloud = system([replace(row, group_key=f"S{row.speaker.track}") for row in rows])
    assert judge.label_source(rows) == "acoustic"
    assert "声学层" in local and "转写模型" not in local
    assert "声学层" not in cloud and "转写模型常把**一个人拆到几条轨**" in cloud
    assert "`s5+s4`" in local and "`s5+s4`" not in cloud


# --- merges across 同时 / 换回 (contract: 「LLM」 第三条) ----------------------------------------------------------

def _validate(lines, *rows, grouped=False):
    from dataclasses import replace

    from finesub.llm.output_protocol import validate_correction_window_output
    from finesub.llm.prompt_variants import resolve_variant
    from finesub.llm.routing.config import CapabilityTier

    from .test_llm_speaker_turns import reply, window_of

    window = replace(window_of(lines), grouped=grouped)
    return validate_correction_window_output(reply(*rows), window,
                                             variant=resolve_variant("capableC", CapabilityTier.CAPABLE))


def test_a_merge_across_together_or_back_is_a_turn_violation_and_its_note_names_the_value():
    lines = chunking.turn_seams([seg(1, 0, 5, "A", text="长。"), seg(2, 2, 3, "B", text="嗯。"),
                                 seg(3, 5.5, 7, "A", text="接着。")])
    assert [line.seam for line in lines[1:]] == [Seam.TOGETHER, Seam.BACK]
    together = _validate(lines, "sub|1,2|长。嗯。|长嗯|high|", "sub|3|接着。|接着|high|")
    assert together.ok and together.turn_violation_rows == (("1", "2"),)
    assert together.turn_notes == ("顺带：position 为 1,2 的合并行里，源 2 的 turn 列标了同时，拆回各自的行",)
    back = _validate(lines, "sub|1|长。|长|high|", "sub|2,3|嗯。接着。|嗯接着|high|")
    assert back.ok and back.turn_violation_rows == (("2", "3"),)
    assert back.turn_notes == ("顺带：position 为 2,3 的合并行里，源 3 的 turn 列标了换回，拆回各自的行",)
    # pass 2: only 可并 may merge, so 同时 separates there too
    grouped = _validate(lines, "sub|1,2|长。嗯。|长嗯|high|", "sub|3|接着。|接着|high|", grouped=True)
    assert grouped.ok and grouped.turn_violation_rows == (("1", "2"),)


def test_a_violating_merge_of_two_overlapping_lines_ends_at_the_latest_end_and_keys_by_its_first_source():
    from finesub.llm.stages.correction.run import product_keys

    lines = chunking.turn_seams([seg(1, 0, 5, "A", text="长。"), seg(2, 2, 3, "B", text="嗯。"),
                                 seg(3, 5.5, 7, "A", text="接着。")])
    result = _validate(lines, "sub|1|长。|长|high|", "sub|2,3|嗯。接着。|嗯接着|high|")
    merged = result.segments[1]
    assert (merged.start, merged.end) == (2, 7)
    result = _validate([seg(1, 0, 4, "B", text="先。"), seg(2, 4.5, 10, "A", text="长。"),
                        seg(3, 5, 6, "B", text="嗯。", seam=Seam.TOGETHER)],
                       "sub|1|先。|先|high|", "sub|2,3|长。嗯。|长嗯|high|")
    merged = result.segments[1]
    assert (merged.start, merged.end) == (4.5, 10)  # the last source is B's 5-6; A's line ends later
    assert product_keys(result.segments, [seg(1, 0, 4, "B"), seg(2, 4.5, 10, "A"), seg(3, 5, 6, "B")]) == \
        ["B", "A"]


# --- pass 1's grouping (contract: 「LLM」 第二条) ----------------------------------------------------------------

def _keyed_different(lines):
    by_id = {line.id: line for line in lines}

    def different(a, b):
        x, y = by_id[a], by_id[b]
        return x.group_key != y.group_key
    return different


def test_group_lines_never_marks_two_speakers_talking_at_once_mergeable():
    from finesub.llm.grouping import group_lines
    from finesub.llm.stages.three_pass import _different

    class Layer:
        def speakers(self):
            return {}

    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.5, 12, "A")]
    joined = group_lines(lines, {"2": "join"}, different=_different(Layer(), lines), thresholds=Thresholds.at(1.0))
    assert joined.groups == (("1",), ("2",), ("3",)) and joined.refused == (("2", "speaker"),)
    mergeable = group_lines(lines, {"2": "may_join"}, different=_different(Layer(), lines), thresholds=Thresholds.at(1.0))
    assert mergeable.seams["2"] is Seam.TOGETHER  # a change, never 可并


def test_a_line_pass1_dropped_counts_for_neither_together_nor_back():
    from finesub.llm.grouping import group_lines

    # A 0-5, B 4-8 (dropped), C 7-9: with B, C starts while B talks; without it, nobody does
    lines = [seg(1, 0, 5, "A"), seg(2, 4, 8, "B"), seg(3, 7, 9, "C")]
    kept = group_lines(lines, {}, different=_keyed_different(lines), thresholds=Thresholds.at(1.0))
    assert kept.seams["3"] is Seam.TOGETHER
    dropped = group_lines(lines, {"2": "drop"}, different=_keyed_different(lines), thresholds=Thresholds.at(1.0))
    assert dropped.dropped == ("2",) and dropped.seams["3"] is Seam.CHANGE
    # A 0-10, B 2-3, X 10.2-10.4 (after A ended), A2 at 10.5: with X, A2 is a plain change; X dropped, only
    # B's interruption is left between A and A2, so A2 goes back
    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.2, 10.4, "X"), seg(4, 10.5, 12, "A")]
    assert group_lines(lines, {}, different=_keyed_different(lines), thresholds=Thresholds.at(1.0)).seams["4"] is Seam.CHANGE
    assert group_lines(lines, {"3": "drop"}, different=_keyed_different(lines), thresholds=Thresholds.at(1.0)).seams["4"] is Seam.BACK


# --- cut points (contract: 「LLM」 切点) ------------------------------------------------------------------------

def test_window_cuts_skip_covered_lines_and_two_lines_nested_in_a_long_one():
    # A's long line covers B's two lines: neither the cut before B1 nor the one between B1 and B2 is a cut
    lines = [seg(1, 0, 1, "A", text="a。"), seg(2, 1.5, 10, "A", text="long。"), seg(3, 2, 3, "B", text="b1。"),
             seg(4, 4, 5, "B", text="b2。", seam=Seam.NONE), seg(5, 10.5, 11, "A", text="c。"),
             seg(6, 11.5, 12, "A", text="d。")]
    prefix = list(range(len(lines) + 1))
    cuts, reasons = chunking._place_cuts(lines, prefix, 2)
    assert cuts == [4] and reasons[4] == "even_sentence_or_gap_boundary"
    parted = line_groups.parted(lines)
    assert not chunking.is_reasonable_boundary(lines, 1, parted=parted)  # before B1
    assert not chunking.is_reasonable_boundary(lines, 2, parted=parted)  # between B1 and B2
    local = [seg(s.id, s.start, s.end, None, text=s.text) for s in lines]
    assert chunking._place_cuts(local, prefix, 2)[0] == [3]


def test_a_window_cut_is_forced_when_every_candidate_is_covered():
    lines = [seg(1, 0, 20, "A", text="a。")] + [seg(i, 2 * i, 2 * i + 1, "B", text="b。") for i in range(2, 7)]
    cuts, reasons = chunking._place_cuts(lines, list(range(len(lines) + 1)), 2)
    assert cuts == [3] and reasons[3] == "forced_even_boundary"


def test_a_window_split_in_half_never_starts_its_second_half_on_a_covered_line():
    from .test_llm_speaker_turns import FakeTokenCounter, window_of

    lines = [seg(1, 0, 2, "A", text="a。"), seg(2, 2.5, 10, "A", text="b。"), seg(3, 3, 4, "B", text="c。"),
             seg(4, 10.5, 12, "A", text="d。")]
    first, _second = chunking.split_window_in_half(window_of(lines), counter=FakeTokenCounter(),
                                                   overlap_window_seconds=0.0)
    assert [s.id for s in first.segments] == ["1", "2", "3"]  # the middle cut would fall before B
    local = [seg(s.id, s.start, s.end, None, text=s.text) for s in lines]
    first, _second = chunking.split_window_in_half(window_of(local), counter=FakeTokenCounter(),
                                                   overlap_window_seconds=0.0)
    assert [s.id for s in first.segments] == ["1", "2"]

    halves = chunking.split_window_in_half(window_of(local), counter=FakeTokenCounter(), overlap_window_seconds=0.0)
    assert [chunking.render_window_segments_as_csv(half).splitlines()[0].split("|")[3] for half in halves] == ["", ""], (
        "a second half's first line has no line before it in its window")


def test_a_different_speaker_talking_over_a_line_reads_no_pause():
    product = chunking.turn_seams([seg(1, 0, 10, "A"), seg(2, 2, 3, "B"), seg(3, 10.5, 12, "A")])
    window = SubtitleWindow(chunk_id="0001", segments=product, overlap_segments=[], boundary_reason="",
                            budget=None, length_scale=1.0)
    gaps = [row.split("|")[3] for row in chunking.render_window_segments_as_csv(window).splitlines()]
    # B starts inside A; A's next line counts from the line right before it (B), as H counted B's own gap
    assert gaps == ["", "0.0", "7.5"]


def test_a_line_together_only_after_a_judge_fill_still_shows_at_a_window_head():
    # A ended at 3.0 when the windows were planned (the cut before B was fine); the judge's fill made it 0-5,
    # so B (4-6) now starts while A talks and opens its window as 同时
    product = chunking.turn_seams([seg(1, 0, 5, "A", text="补全了。"), seg(2, 4, 6, "B", text="嗯。"),
                                   seg(3, 6.5, 8, "A", text="接着。")])
    window = SubtitleWindow(chunk_id="0002", segments=product[1:], overlap_segments=[], boundary_reason="",
                            budget=None, preceding_segments=product[:1], length_scale=1.0)
    cells = [row.split("|")[4] for row in chunking.render_window_segments_as_csv(window).splitlines()]
    assert cells == ["同时", "换回"]

# --- the judge's audio (contract: 「LLM」 窗终点与样本) ----------------------------------------------------------

def test_the_judges_recording_ends_at_the_latest_end_and_its_samples_are_one_voice():
    from finesub.llm.stages.speaker_judge import prompt as judge
    from finesub.llm.stages.speaker_judge.windows import JudgeWindow

    from .test_speaker_judge_stage import speaker

    # (start, end, track, key): S2's 2-4.5 overlaps S1's 0-3; the window's last row (21-23) sits inside S1's 20-30
    spec = [(0, 3, 1, "S1"), (2, 4.5, 2, "S2"), (10, 13, 1, "S1"), (14, 17, 2, "S2"), (20, 30, 1, "S1"),
            (21, 23, 2, "S2")]

    def rows(keyed):
        return judge.rows_of([{"id": str(i), "start": float(s), "end": float(e), "text": f"台词{i}",
                               "speaker": speaker(t), **({"line_speaker": k} if keyed else {})}
                              for i, (s, e, t, k) in enumerate(spec, start=1)])

    window = JudgeWindow("0001", "0001", 0, len(spec))
    cloud, local = rows(True), rows(False)
    assert judge.profiles(cloud, window).samples == {1: [2], 2: [3]}
    assert judge.profiles(local, window).samples == {1: [0, 2], 2: [1, 3]}
    samples = judge.profiles(cloud, window).samples
    assert judge.audio_layout(cloud, window, samples).pieces[-1][1] == 30 + judge.EDGE_PAD
    assert judge.audio_layout(local, window, samples).pieces[-1][1] == 23 + judge.EDGE_PAD


# --- planning estimates (contract: 「LLM」 估算) ----------------------------------------------------------------

def test_window_estimates_take_the_latest_end_when_lines_overlap():
    from finesub.llm.routing.config import ModelLimits
    from finesub.llm.routing.profiles import TranslationProfile, resolve_profile

    from .test_llm_speaker_turns import FakeTokenCounter

    class Recording(FakeTokenCounter):
        seconds: list = []

        def count_audio_seconds(self, seconds):
            self.seconds.append(seconds)
            return super().count_audio_seconds(seconds)

    lines = [seg(1, 0, 10, "A"), seg(2, 2, 3, "B")]
    counter = Recording()
    chunking.estimate_window_budget(lines, counter=counter, profile=resolve_profile("audio"))
    chunking.estimate_window_budget([seg(1, 0, 10), seg(2, 2, 3)], counter=counter, profile=resolve_profile("audio"))
    assert counter.seconds == [10, 3]

    def count(lines):
        # 100 text tokens; the overlap estimate is 100 x 5 s / span: over the output room at a span of 3 s
        return chunking._estimate_window_count(
            lines, total_text_tokens=100, total_mass=100.0, context_tokens=0, prompt_tokens=0,
            overlap_window_seconds=5.0, limits=ModelLimits(prompt_input_limit=10**9),
            planning_limits=ModelLimits(output_limit=150),
            profile=TranslationProfile(correction_media="text", retrieval="none", difficulty="quality",
                                       output_coefficient=1.0))
    assert (count(lines), count([seg(1, 0, 10), seg(2, 2, 3)])) == (1, 100)


# --- the agent side's window (N1) ------------------------------------------------------------------------------

def test_the_agent_sides_window_keeps_each_lines_speaker():
    from finesub.llm.agent.agent_validators import deserialize_window, serialize_window

    from .test_llm_speaker_turns import window_of

    window = window_of(chunking.turn_seams([seg(1, 0, 10, "A"), seg(2, 2, 3, "B")]))
    back = deserialize_window(serialize_window(window))
    assert [s.group_key for s in back.segments] == ["A", "B"]
    assert back.end == window.end == 10  # without the keys the window would end at B's 3


def test_a_single_pass_seam_is_a_change_when_two_people_talk_at_once_whatever_the_labels():
    # The judge put B's backchannel on A's track: the labels say one person, the keys and times say two. Only
    # the overlap is evidence against the judge; whether A's next line goes back after B is the labels' call.
    from finesub.speech.diarization.labels import SegmentSpeaker

    rows = [{"id": "1", "start": 0.0, "end": 10.0, "text": "長い", "line_speaker": "S01"},
            {"id": "2", "start": 2.0, "end": 3.0, "text": "うん", "line_speaker": "S02"},
            {"id": "3", "start": 10.5, "end": 12.0, "text": "続き", "line_speaker": "S01"}]
    same = {row["id"]: SegmentSpeaker(track=0) for row in rows}
    assert [s.seam.value for s in chunking._subtitle_segments(rows, same)] == ["", "同时", ""]
    local = [{k: v for k, v in row.items() if k != "line_speaker"} for row in rows]
    assert [s.seam.value for s in chunking._subtitle_segments(local, same)] == ["", "", ""]


def test_a_local_line_on_the_overlap_lane_is_together_and_the_judge_still_reads_acoustic_tracks():
    # The collapse repair stretched row 1 over row 2 (docs/asr-stabilize.md, "坍缩行"); every other local line
    # stays one group, so rows 2 and 3 are kept apart as before.
    from finesub.llm.stages.speaker_judge import prompt as judge
    from finesub.speech.diarization.labels import SegmentSpeaker

    rows = [{"id": "1", "start": 0.0, "end": 10.0, "text": "長い", "overlap_lane": "0.0-0.1"},
            {"id": "2", "start": 2.0, "end": 3.0, "text": "うん"},
            {"id": "3", "start": 10.5, "end": 12.0, "text": "続き"}]
    same = {row["id"]: SegmentSpeaker(track=0) for row in rows}
    lines = chunking._subtitle_segments(rows, same)
    assert [s.seam.value for s in lines] == ["", "同时", ""]
    assert chunking.segment_keys(lines) == [f"{line_groups.LANE}:0.0-0.1", None, None]
    assert judge.label_source(judge.rows_of(rows)) == "acoustic"


def test_the_absolute_merge_gate_measures_the_cue_a_merge_across_together_becomes():
    # A's long line wraps B's backchannel: 1,2 ends at A's 10 s, not B's 3 s, so the merged cue is 10 s long
    lines = chunking.turn_seams([seg(1, 0, 10, "A", text="长。"), seg(2, 2, 3, "B", text="嗯。"),
                                 seg(3, 10.5, 12, "A", text="接着。")])
    result = _validate(lines, "sub|1,2|长。嗯。|长嗯|high|", "sub|3|接着。|接着|high|")
    assert not result.ok
    assert any("10.0s" in error and "absolute limit" in error for error in result.errors)
