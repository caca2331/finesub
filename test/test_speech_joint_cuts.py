"""Where the joint-ASR backends cut a source: at Silero pauses, the model's own breaks lowering the bar
(`speech/joint/cuts.py`)."""

from __future__ import annotations

import pytest

from finesub.speech.joint import cuts as C


def test_pauses_are_the_silences_between_speech_spans():
    assert C.pauses_of([(0, 10), (12, 20), (15, 25), (30, 40)]) == [(11.0, 2), (27.5, 5)]


def test_a_pause_qualifies_as_deep_as_its_distance_or_as_deep_as_the_bar_takes_to_fall_to_it():
    assert C.depth(25.0, 2.0, 30.0) == 5.0  # NEED0: at once
    assert C.depth(25.0, 1.0, 30.0) == 20.0  # the bar halves 20 s out
    assert C.depth(25.0, 0.5, 30.0) == 60.0
    assert C.depth(25.0, 3.0, 30.0) == 20.0  # over L_TRUST: counted as LONG_COUNTS
    # the model breaks there too: a quarter of the length
    assert C.depth(25.0, 0.5, 30.0, [(24.9, 25.1)]) == 5.0
    assert round(C.depth(25.0, 0.3, 30.0, [(25.4, 26.0)]), 2) == 13.33  # within AGREE_TOL of the pause [24.85, 25.15]
    assert round(C.depth(25.0, 0.3, 30.0, [(25.46, 26.0)]), 2) == 113.33  # not


def test_a_cut_takes_the_pause_that_qualifies_nearest_else_its_nominal_time():
    pauses = [(10.0, 0.5), (14.0, 1.5), (18.0, 1.5), (30.0, 9.0)]  # 30: out of range
    assert C.pause_in(pauses, 5.0, 20.0, 20.0) == 18.0  # equal depths: the one nearer the nominal time
    assert C.pause_in(pauses, 5.0, 20.0, 5.0) == 14.0
    assert C.pause_in(pauses, 20.0, 29.0, 25.0) == 25.0
    assert C.pause_in([], 0.0, 10.0, 9.87) == 9.87  # no pause: the nominal time untouched, not to 0.1 s
    # a near pause that qualifies beats a longer one further back
    assert C.pause_in([(5.0, 2.0), (25.0, 1.0)], 0.0, 30.0, 30.0) == 25.0
    # ... and a short one the model breaks at beats both
    assert C.pause_in([(5.0, 2.0), (25.0, 1.0), (28.0, 0.3)], 0.0, 30.0, 30.0, [(27.9, 28.0)]) == 28.0
    assert C.pause_in([(5.0, 2.0), (25.0, 1.0), (28.0, 0.3)], 0.0, 30.0, 30.0) == 25.0
    # none qualifies within the range's 10 s: the relatively best, the longest as counted (5 s counts as 1 s)
    assert C.pause_in([(3.0, 5.0), (4.0, 1.2), (8.0, 0.3)], 0.0, 10.0, 10.0) == 4.0
    assert C.pause_in([(12.34, 1.0)], 10.0, 15.0, 15.0) == 12.3  # to 0.1 s, as the calls are named
    # rounding never takes it out of its range: the 0.1 s step on the inside
    assert C.pause_in([(14.97, 1.0)], 10.0, 15.0, 10.0) == 14.9
    assert C.pause_in([(10.03, 1.0)], 10.04, 15.0, 15.0) == 15.0  # out of range anyway: the nominal time
    assert C.pause_in([(10.07, 1.0)], 10.06, 15.0, 15.0) == 10.1


def test_windows_end_at_a_pause_before_the_limit_and_start_at_one_60_to_90_s_before_that_end():
    pauses = [(500.0, 2.0), (530.0, 0.5), (425.0, 1.0), (950.0, 1.0)]
    assert C.windows(0.0, 1200.0, 540.0, pauses, reach=60.0) == [(0.0, 500.0), (425.0, 950.0), (890.0, 1200.0)]


def test_without_pauses_windows_are_the_fixed_cuts():
    assert C.windows(0.0, 1200.0, 540.0, [], reach=60.0) == [(0.0, 540.0), (480.0, 1020.0), (960.0, 1200.0)]
    assert C.windows(0.0, 300.0, 540.0, []) == [(0.0, 300.0)]


def test_a_cached_window_in_the_allowed_ranges_is_taken_so_fixed_cuts_replay():
    pauses = [(500.0, 2.0), (950.0, 1.0)]
    fixed = [(0.0, 540.0), (480.0, 1020.0), (960.0, 1200.0)]
    assert C.windows(0.0, 1200.0, 540.0, pauses, reach=60.0, cached_reach=60.0, known=fixed) == fixed
    # one out of range (an end 100 s short of the limit) is not this rule's window
    assert C.windows(0.0, 1200.0, 540.0, pauses, reach=60.0, cached_reach=60.0, known=[(0.0, 440.0)])[0] == (0.0, 500.0)


def test_a_window_cut_by_the_longest_pause_up_to_120_s_back_still_replays():
    pauses = [(530.0, 2.0)]
    assert C.windows(0.0, 1200.0, 540.0, pauses)[0] == (0.0, 530.0)  # planned: within REACH
    rule = dict(cached_reach=C.CACHED_REACH)
    assert C.windows(0.0, 1200.0, 540.0, pauses, known=[(0.0, 430.0)], **rule)[0] == (0.0, 430.0)  # CACHED_REACH
    assert C.windows(0.0, 1200.0, 540.0, pauses, known=[(0.0, 410.0)], **rule)[0] == (0.0, 530.0)
    assert C.windows(0.0, 1200.0, 540.0, pauses, known=[(0.0, 430.0)])[0] == (0.0, 530.0)  # not asked: REACH


def test_the_next_window_starts_where_the_previous_ones_output_breaks_too():
    pauses = [(430.0, 2.0), (448.0, 0.4)]
    assert C.window(0.0, 1200.0, 540.0, pauses, 510.0) == ((430.0, 970.0), False)  # 448: 80 s deep, 430: 20
    words = [{"text": "です。", "start": 447.0, "end": 447.8}, {"text": "次", "start": 448.3, "end": 448.6}]
    assert C.window(0.0, 1200.0, 540.0, pauses, 510.0, breaks=C.word_breaks(words)) == ((448.0, 988.0), False)
    assert C.window(0.0, 1200.0, 540.0, pauses) == ((0.0, 540.0), False)


def test_a_planned_window_ends_at_the_source_end_not_within_a_names_rounding_of_it():
    assert C.windows(900.0, 1800.08, 900.0, [], reach=100.0) == [(900.0, 1800.0), (1740.0, 1800.08)]
    # a cached one is named to 0.1 s: within that of the end it is the last
    assert C.windows(900.0, 1800.08, 900.0, [], reach=100.0, known=[(900.0, 1800.1)]) == [(900.0, 1800.1)]


def test_windows_that_could_make_no_headway_are_refused():
    with pytest.raises(ValueError):
        C.windows(0.0, 1000.0, 200.0, [], reach=120.0)
    with pytest.raises(ValueError):  # a cached window may end cached_reach s back
        C.windows(0.0, 1000.0, 200.0, [], reach=30.0, cached_reach=120.0, known=[(0.0, 100.0)])
    assert C.windows(0.0, 400.0, 200.0, [], reach=30.0)[0] == (0.0, 200.0)
    # codex acceptance review: with no cached_reach asked for, a cached window is looked for within reach
    assert C.windows(0.0, 400.0, 200.0, [], reach=30.0, known=[(0.0, 180.0)])[0] == (0.0, 180.0)


def test_the_model_breaks_between_segments_and_at_marks_or_a_change_of_speaker():
    segs = [{"start": 0.0, "end": 10.0}, {"start": 5.0, "end": 12.0}, {"start": 11.0, "end": 13.0},
            {"start": 14.0, "end": 20.0}]
    assert C.segment_breaks(segs) == [(13.0, 14.0)]  # 5-12 overlaps 0-10: the latest end so far
    words = [("こんばんは。", 0.0, 1.0, "a"), ("今日", 1.5, 2.0, "a"), ("は", 2.0, 2.2, "a"), ("、", 2.2, 2.25, "a"),
             ("雨", 2.6, 3.0, "a"), ("です", 3.0, 3.4, "b"), ("」", 3.4, 3.4, "b"), ("って", 3.9, 4.0, "b"),
             ("はい", 4.0, 4.3, "b"), ("。", 4.3, 4.3, "b"), ("」", 4.3, 4.3, "b"), ("次", 4.8, 5.0, "b")]
    words = [{"text": x, "start": s, "end": e, "speaker": k} for x, s, e, k in words]
    # a mark on the word, a mark's own token (its time not used), a new speaker; a closer's own token breaks only
    # after a mark (codex acceptance review: 「です」って goes on)
    assert C.word_breaks(words) == [(1.0, 1.5), (2.2, 2.6), (3.0, 3.0), (4.3, 4.8)]


def test_an_edge_by_the_words_goes_out_to_the_middle_of_its_pause_never_in():
    assert C.outward(10.0, [(10.4, 1.0)], True) == 10.4
    assert C.outward(10.6, [(10.4, 1.0)], True) == 10.6  # the middle is inside
    assert C.outward(10.6, [(10.4, 1.0)], False) == 10.4
    assert C.outward(12.0, [(10.4, 1.0)], True) == 12.0  # in no pause
    # codex acceptance review: judged by the middle itself, 10.05 (inside), not its name 10.1
    assert C.outward(10.07, [(10.05, 0.1)], True) == 10.07
    assert C.outward(10.07, [(10.05, 0.1)], False) == 10.07  # 10.05 is out, its name 10.1 is in
    assert C.outward(10.06, [(10.08, 0.2)], True) == 10.1


def test_a_cached_call_by_the_words_keeps_the_edges_it_was_asked_from():
    pauses = [(99.7, 1.0)]
    assert C.widened(100.04, 200.0, pauses, 0.0, 600.0) == (99.7, 200.0)
    # named before 2026-11-22 at the edges themselves: those, unrounded (the answer's times count from them)
    assert C.widened(100.04, 200.03, pauses, 0.0, 600.0, [(100.0, 200.0)]) == (100.04, 200.03)
    assert C.widened(100.04, 200.0, pauses, 0.0, 600.0, [(99.7, 200.0)]) == (99.7, 200.0)
    assert C.widened(100.04, 200.0, pauses, 99.9, 600.0) == (99.9, 200.0)  # inside [lo, hi]
    # a move that keeps the 0.1 s name is none (codex acceptance review 2: clamped to 100.0 at the piece's start,
    # a call named 100.0 would be asked from 100.0 now and from 100.04 before)
    assert C.widened(100.04, 200.0, [(99.8, 1.0)], 100.0, 600.0) == (100.04, 200.0)
    assert C.widened(100.04, 200.0, [(99.8, 1.0)], 100.0, 600.0, [(100.0, 200.0)]) == (100.04, 200.0)
    assert C.widened(10.06, 20.0, [(10.08, 0.2)], 0.0, 600.0) == (10.06, 20.0)


def test_a_pad_goes_out_to_a_pause_never_in():
    pauses = [(36.0, 1.0), (44.0, 3.0), (113.0, 0.8), (106.0, 2.0)]  # 44 and 106: inside the pad, not taken
    assert C.padded(50.0, 100.0, 10.0, pauses, 300.0) == (36.0, 113.0)
    assert C.padded(50.0, 100.0, 10.0, [], 300.0) == (40.0, 110.0)
    assert C.padded(5.0, 295.0, 10.0, pauses, 300.0) == (0.0, 300.0)  # the source's ends
    assert C.stretch(1.0) == C.MIN_STRETCH and C.stretch(10.0) == 5.0


def test_a_pad_stretches_out_only_as_far_as_the_longest_span_allows():
    pauses = [(85.0, 1.0), (614.0, 1.0)]
    assert C.padded(100.0, 600.0, 10.0, pauses, 2000.0, longest=540.0) == (85.0, 614.0)
    assert C.padded(100.0, 600.0, 10.0, pauses, 2000.0, longest=525.0) == (90.0, 610.0)  # 2.5 s each way: none
    assert C.padded(100.0, 615.0, 10.0, pauses, 2000.0, longest=535.0) == (90.0, 625.0)  # the pads, no more


def test_a_cached_pad_in_range_is_taken():
    pauses = [(36.0, 1.0)]
    assert C.padded(50.0, 100.0, 10.0, pauses, 300.0, known=[(40.0, 110.0)]) == (40.0, 110.0)
    assert C.padded(50.0, 100.0, 10.0, pauses, 300.0, known=[(42.0, 110.0)]) == (36.0, 110.0)  # inside the pad


def test_call_names_give_back_their_spans():
    names = ["w-0.0-540.0", "w-480.0-1020.0", "r-15.0-85.0", "identity", "w-x-y"]
    assert C.spans_named(names, "w") == [(0.0, 540.0), (480.0, 1020.0)]
    assert C.spans_named(names, "r") == [(15.0, 85.0)]
