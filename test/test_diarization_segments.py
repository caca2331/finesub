"""Frame activity -> segments (P11).

The first test is the one that matters: two people talking at once must come out
as two segments. Every published smoothing recipe for this problem is a
single-label state machine, and pressing one onto a 4-sigmoid output silently
hands the overlap to whoever wins -- destroying the one capability this line has
that clustering never did.
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.segments import (
    SmoothingParameters,
    TrackSegment,
    pair_co_activity,
    segment_overlap_ratio,
    segments_from_activity,
    track_seconds,
    union_seconds,
)


def frames(pattern: str, tracks: int = 2) -> list[list[float]]:
    """`"AB"` -> one frame with tracks 0 and 1 active; `"."` -> silence."""
    rows = []
    for character in pattern.split():
        row = [0.0] * tracks
        for letter in character:
            if letter != ".":
                row[ord(letter) - ord("A")] = 1.0
        rows.append(row)
    return rows


def test_two_speakers_at_once_both_survive():
    # Twenty frames (1.6s) of A and B talking over each other.
    activity = frames(" ".join(["AB"] * 20))
    found = segments_from_activity(activity)
    assert sorted(segment.track for segment in found) == [0, 1], "the overlap must not collapse"
    assert all(segment.duration == pytest.approx(1.6) for segment in found)


def test_a_gap_shorter_than_min_gap_is_bridged():
    activity = frames(" ".join(["A"] * 10 + ["."] * 1 + ["A"] * 10))
    found = segments_from_activity(activity)
    assert len(found) == 1
    assert found[0].start == pytest.approx(0.0)
    assert found[0].end == pytest.approx(21 * 0.08)


def test_a_gap_longer_than_min_gap_is_honoured():
    activity = frames(" ".join(["A"] * 10 + ["."] * 5 + ["A"] * 10))
    found = segments_from_activity(activity)
    assert len(found) == 2


def test_bridging_happens_before_dropping_short_segments():
    # Two 2-frame bursts either side of a 1-frame gap. Each is below
    # min_duration on its own; together they clear it. Dropping first would
    # lose the speech entirely.
    activity = frames(" ".join(["A"] * 2 + ["."] * 1 + ["A"] * 2))
    found = segments_from_activity(activity)
    assert len(found) == 1
    assert found[0].duration == pytest.approx(5 * 0.08)


def test_a_backchannel_shorter_than_min_duration_disappears_under_the_bridge():
    # ⚠ Recorded because it is a real loss, not because it is right: B's single
    # frame is dropped as too short, and A's gap -- now empty -- is bridged over
    # it. A 0.08s 「うん」 leaves no trace at all.
    #
    # It is bounded: only a turn under `min_duration` (0.24s) can vanish this
    # way, and a backchannel that brief rarely has a subtitle line of its own.
    # Both numbers are provisional (plan 6), and if calibration raises `min_gap`
    # past `min_duration` then *surviving* turns start disappearing too, which
    # would need a guard in `_bridge`.
    activity = frames(" ".join(["A"] * 10 + ["B"] * 1 + ["A"] * 10))
    found = segments_from_activity(activity)
    assert [segment.track for segment in found] == [0]
    assert found[0].duration == pytest.approx(21 * 0.08), "A bridged straight over it"


def test_genuine_simultaneous_speech_overlaps():
    activity = frames(" ".join(["A"] * 5 + ["AB"] * 10 + ["A"] * 5))
    found = segments_from_activity(activity)
    assert sorted(segment.track for segment in found) == [0, 1]
    assert segment_overlap_ratio(found) > 0.4


def test_a_lone_short_burst_is_dropped():
    activity = frames(" ".join(["A"] * 2 + ["."] * 10))
    assert segments_from_activity(activity) == []


def test_hysteresis_keeps_a_wobbling_track_in_one_piece():
    # Dips to 0.45: below `enter`, above `leave`, so the segment continues.
    activity = [[0.9, 0.0]] * 5 + [[0.45, 0.0]] * 3 + [[0.9, 0.0]] * 5
    found = segments_from_activity(activity)
    assert len(found) == 1
    assert found[0].duration == pytest.approx(13 * 0.08)


def test_a_track_that_never_reaches_enter_produces_nothing():
    activity = [[0.49, 0.0]] * 50
    assert segments_from_activity(activity) == []


def test_segments_come_back_in_time_order():
    activity = frames(" ".join(["A"] * 10 + ["."] * 5 + ["B"] * 10))
    found = segments_from_activity(activity)
    assert [segment.track for segment in found] == [0, 1]
    assert found[0].start < found[1].start


def test_overlap_ratio_is_zero_for_one_speaker_and_positive_for_two():
    solo = segments_from_activity(frames(" ".join(["A"] * 20)))
    assert segment_overlap_ratio(solo) == 0.0
    both = segments_from_activity(frames(" ".join(["AB"] * 20)))
    assert segment_overlap_ratio(both) == pytest.approx(1.0)


def test_overlap_ratio_counts_only_the_overlapping_part():
    activity = frames(" ".join(["A"] * 10 + ["AB"] * 10 + ["A"] * 10))
    ratio = segment_overlap_ratio(segments_from_activity(activity))
    assert ratio == pytest.approx(10 / 30, abs=0.02)


def test_parameters_reject_an_impossible_hysteresis():
    with pytest.raises(ValueError):
        SmoothingParameters(enter=0.4, leave=0.6)


def test_empty_input_is_not_an_error():
    assert segments_from_activity([]) == []


# --- the per-pair veto signal (plan 5l.11) -------------------------------------


def test_co_activity_is_measured_against_the_shorter_track():
    # s1 speaks 2 s, all of it over s0 -- a short track that only ever talks over a
    # long one is still clearly a second mouth.
    tracks = [TrackSegment(0, 0.0, 10.0), TrackSegment(1, 4.0, 6.0)]
    assert pair_co_activity(tracks) == {(0, 1): pytest.approx(1.0)}


def test_tracks_that_never_meet_have_zero_co_activity():
    tracks = [TrackSegment(0, 0.0, 2.0), TrackSegment(1, 2.0, 4.0), TrackSegment(0, 4.0, 6.0)]
    assert pair_co_activity(tracks) == {(0, 1): 0.0}


def test_co_activity_sums_every_meeting_of_the_two_tracks():
    tracks = [
        TrackSegment(0, 0.0, 4.0),
        TrackSegment(1, 3.0, 5.0),
        TrackSegment(0, 6.0, 8.0),
        TrackSegment(1, 7.5, 9.0),
    ]
    # s0 6 s, s1 3.5 s; they meet for 1.0 + 0.5 s
    assert pair_co_activity(tracks)[(0, 1)] == pytest.approx(1.5 / 3.5)
    assert track_seconds(tracks) == {0: pytest.approx(6.0), 1: pytest.approx(3.5)}
    # speech: 0-5 and 6-9
    assert union_seconds(tracks) == pytest.approx(8.0)


def test_union_seconds_of_nothing_is_zero():
    assert union_seconds([]) == 0.0
