"""MOSS Pro's key filling (`speech/joint/fill.py`) and which cached call a rerun takes as a piece's main call."""

from __future__ import annotations

import pytest

from finesub.speech.joint import cuts as C
from finesub.speech.joint import fill as F
from finesub.speech.joint.mosspro import main_call

K = F.Key


def spans(*pieces):
    return [(p.start, p.end, p.key) for p in pieces]


def test_a_source_that_fits_goes_whole_into_the_key_with_the_least_room_that_holds_it():
    assert spans(*F.plan(1000, [(0, 0)], [K(3000), K(1500), K(4000)], [])) == [(0, 1000, 1)]
    assert spans(*F.plan(1000, [(0, 0)], [K(1500), K(1500)], [])) == [(0, 1000, 0)]  # a tie: .env order


def test_one_that_fits_none_is_cut_once_least_room_first_with_the_minimum_overlap():
    plan = F.plan(2800, [(0, 0)], [K(2000), K(1000), K(2000)], [])
    assert spans(*plan) == [(0, 1000, 1), (940, 2800, 0)]  # the last piece: the tightest key that holds it


def test_cut_and_start_go_on_pauses_when_that_costs_nothing():
    pauses = [(1490.0, 2.0), (1420.0, 2.0), (1499.0, 0.4), (1460.0, 3.0)]  # 1460: past REACH; 1499: too short so near
    plan = F.plan(2000, [(0, 0)], [K(1500), K(1500)], pauses)
    assert spans(*plan) == [(0, 1490.0, 0), (1420.0, 2000, 1)]


def test_a_seam_off_a_pause_ranks_after_every_pause_however_short():
    plan = F.plan(2000, [(0, 0)], [K(1500), K(1500)], [(1470.5, 0.05)])  # qualifies only 780 s deep
    assert spans(*plan)[0] == (0, 1470.5, 0)


def test_seams_compare_how_many_are_off_a_pause_then_the_summed_depth_then_the_later_cut():
    # A: the cut at the limit (off), the start on the pause at 1430 (depth 10); B: the cut on the pause at 1475
    # (depth 25), its start range 1385-1415 without a pause (off). One off each: A's 10 < B's 25.
    plan = F.plan(2000, [(0, 0)], [K(1500), K(1500)], [(1430.0, 2.0), (1475.0, 2.0)])
    assert spans(*plan) == [(0, 1500.0, 0), (1430.0, 2000, 1)]
    # a pause at 1400 puts B's start on one too (depth 15): no seam off a pause wins over one
    plan = F.plan(2000, [(0, 0)], [K(1500), K(1500)], [(1430.0, 2.0), (1475.0, 2.0), (1400.0, 2.0)])
    assert spans(*plan) == [(0, 1475.0, 0), (1400.0, 2000, 1)]
    # equal sums: the later cut
    plan = F.plan(2000, [(0, 0)], [K(1500), K(1500)], [(1480.0, 2.0), (1405.0, 2.0), (1490.0, 2.0), (1415.0, 2.0)])
    assert spans(*plan) == [(0, 1490.0, 0), (1415.0, 2000, 1)]


def test_a_pause_that_would_leave_the_next_piece_without_room_is_not_taken():
    # codex's case: the longest pause makes the second piece 1275 s, past 1200 s of room
    plan = F.plan(2300, [(0, 0)], [K(1200), K(1200)], [(1100.0, 5.0), (1190.0, 0.5)])
    assert spans(*plan) == [(0, 1190.0, 0), (1100.0, 2300, 1)]
    assert all(p.end - p.start <= 1200 for p in plan)


def test_free_hours_fill_before_the_paid_key_and_its_rest_goes_on_it_uncut():
    plan = F.plan(3000, [(0, 0)], [K(1000), K(1000), K(1000, paid=True)], [])
    assert spans(*plan) == [(0, 1000, 0), (940, 1940, 1), (1880, 3000, 2)]
    assert [p.billed for p in plan] == [0, 0, 120]
    # one key's free hours less: the paid key still takes the rest in one call, no extra cut
    assert spans(*F.plan(3000, [(0, 0)], [K(1000), K(1000, paid=True)], [])) == [(0, 1000, 0), (940, 3000, 1)]


def test_more_cuts_win_when_they_save_billed_seconds():
    plan = F.plan(2000, [(0, 0)], [K(1500), K(0, paid=True)], [])
    assert spans(*plan) == [(0, 1500, 0), (1440, 2000, 1)]


def test_hard_cut_wins_over_a_pause_that_would_bill_more():
    plan = F.plan(2000, [(0, 0)], [K(1500), K(0, paid=True)], [(1450.0, 3.0)])
    assert spans(*plan)[0] == (0, 1500, 0)


def test_the_paid_keys_free_room_takes_a_cut_piece_when_that_saves_the_balance():
    # codex's case: the free key's hour and the paid key's whole rest would bill 260 s
    plan = F.plan(5000, [(0, 0)], [K(4100), K(1200, paid=True)], [])
    assert spans(*plan) == [(0, 1200, 1), (1140, 4740, 0), (4680, 5000, 0)]
    assert sum(p.billed for p in plan) == 0


@pytest.mark.parametrize("keys, expected", [
    ([K(4100), K(0, paid=True)], [(0, 3500.0, 0), (3440.0, 4040.0, 0), (3980.0, 5000, 1)]),  # 1020 s billed, not 1460
    ([K(4100)], [(0, 3500.0, 0), (3440.0, 4040.0, 0)]),  # free mode: transcribed to 4040, not 3600
])
def test_a_cut_that_leaves_the_key_ten_minutes_is_tried(keys, expected):
    # codex's case: the three longest pauses each leave the key under MIN_ROOM
    pauses = [(3590.0, 4.0), (3580.0, 3.0), (3570.0, 2.0), (3500.0, 1.0)]
    assert spans(*F.plan(5000, [(0, 0)], keys, pauses)) == expected


@pytest.mark.parametrize("keys, expected", [
    ([K(1000), K(4100), K(0, paid=True)], [(0, 1000, 0), (940, 4440, 1), (4380, 4980, 1), (4920, 6000, 2)]),
    ([K(1000), K(4100)], [(0, 1000, 0), (940, 4440, 1), (4380, 4980, 1)]),
])
def test_where_one_cut_lands_does_not_hide_a_later_keys_ten_minute_cut(keys, expected):
    # codex's round-2 case: a pause at 900 would move key b's window onto the pause at 4330; the ten-minute limit
    # makes b's cut not depend on where a's lands (billed 1080 s, not 1520; free mode reaches 4980 s, not 4540)
    pauses = [(900.0, 1.0), (970.0, 4.0), (980.0, 3.0), (990.0, 2.0), (995.0, 0.1), (4330.0, 1.0)]
    assert spans(*F.plan(6000, [(0, 0)], keys, pauses)) == expected


def test_a_key_used_twice_still_lets_the_cut_go_on_a_pause():
    # keys compare by their room as planning began: the cut moving key a's later room does not outrank the seam
    plan = F.plan(7200, [(0, 0)], [K(4235), K(4235)], [(3580.0, 2.0), (3510.0, 1.5)])
    assert spans(*plan) == [(0, 3580.0, 0), (3510.0, 7110.0, 1), (7050.0, 7200, 0)]


@pytest.mark.parametrize("total, pauses", [
    (4000, [(3481.0, 1.0), (3570.0, 4.0), (3580.0, 3.0), (3590.0, 2.0)]),  # codex's round-3 case: 3570 s billed before
    (4000, []),
])
def test_a_billed_piece_ends_where_the_rest_fits_a_key_too_small_to_cut(total, pauses):
    # the rest from 3401 is 599 s, the small key's room: 3461 s billed (owner 2026-11-10: a hard cut is fine there)
    assert spans(*F.plan(total, [(0, 0)], [K(599), K(0, paid=True)], pauses)) == [(0, 3461.0, 1), (3401.0, 4000, 0)]


def test_a_rest_under_the_hour_is_billed_only_up_to_where_a_free_key_takes_the_rest():
    assert spans(*F.plan(3000, [(0, 0)], [K(599), K(0, paid=True)], [(2930.0, 1.0)])) == [(0, 2461.0, 1), (2401.0, 3000, 0)]


def test_free_pieces_after_a_billed_one_cost_the_same_put_first():
    # codex's round-4 case: billed [0,3481] then 1122 s and 517 s free; the same 3481 s billed with the free hour first
    plan = F.plan(5000, [(0, 0)], [K(1122), K(517), K(0, paid=True)],
                  [(3481.0, 1.0), (3587.0, 4.0), (3592.0, 3.0), (3597.0, 2.0)])
    assert spans(*plan) == [(0, 1122, 0), (1062.0, 4543.0, 2), (4483.0, 5000, 1)]
    assert sum(p.billed for p in plan) == 3481


def test_the_paid_keys_small_free_room_takes_no_free_cut_piece():
    # codex's case: ending at the pause at 580 would be a free, non-final piece on 590 s of room
    assert spans(*F.plan(700, [(0, 0)], [K(200), K(590, paid=True)], [(580.0, 1.0)])) == [(0, 700, 1)]


@pytest.mark.parametrize("total, paid_room, pauses, cheapest, billed", [
    # the rest fits key a from 560: before, at, after the paid key's room; cheapest by every pause (brute.py)
    (700, 550, [(591.0, 1.0), (610.0, 4.0), (620.0, 3.0), (630.0, 2.0)], 10, 10),
    (700, 560, [(591.0, 1.0), (610.0, 4.0), (620.0, 3.0), (630.0, 2.0)], 31, 140),  # the pauses: past REACH
    (700, 590, [(591.0, 1.0), (610.0, 4.0), (620.0, 3.0), (630.0, 2.0)], 1, 110),
    (700, 560, [(591.0, 1.0), (680.0, 2.0)], 31, 120),  # one within REACH of the end
    (711, 590, [(591.0, 1.0), (710.6, 4.0), (710.7, 3.0), (710.8, 2.0)], 1, 121),
])
def test_a_billed_piece_on_a_paid_key_with_little_free_room_can_miss_the_cheapest(
        total, paid_room, pauses, cheapest, billed):
    # A known miss (codex): the cheapest end can be a pause just past the paid key's small free room, not among the
    # pauses tried when it lies more than REACH s before the limit; the plan bills up to a pause nearer, or the rest.
    plan = F.plan(total, [(0, 0)], [K(200), K(paid_room, paid=True)], pauses)
    assert sum(p.billed for p in plan) == billed >= cheapest
    assert all(p.billed or p.end == total or [K(200), K(paid_room)][p.key].room >= F.MIN_ROOM for p in plan)


@pytest.mark.parametrize("room", [689, 690, 691, 3599, 3600, 3601, 4199, 4200, 4201])
@pytest.mark.parametrize("paid", [False, True])
def test_every_piece_keeps_the_rules_around_the_room_boundaries(room, paid):
    keys = [K(room), K(1500), *([K(300, paid=True)] if paid else [])]
    pauses = [(float(m), 1.0 + m % 3) for m in range(97, 9000, 211)]
    plan = F.plan(9000, [(0, 0)], keys, pauses)
    rooms = [k.room for k in keys]
    for i, p in enumerate(plan):
        used = p.end - p.start
        assert used <= F.PART_SECONDS
        assert p.billed == max(0.0, used - rooms[p.key]) if keys[p.key].paid else p.billed == 0 and used <= rooms[p.key]
        if p.end < 9000:  # a cut piece: billed, or on a key with ten minutes; the next starts 60-90 s back
            assert p.billed or rooms[p.key] >= F.MIN_ROOM
            nxt = plan[i + 1] if i + 1 < len(plan) else None
            assert nxt is None or p.end - F.OVERLAP[1] <= nxt.start <= p.end - F.OVERLAP[0]
            assert nxt is not None or not paid  # only without a paid key may a plan stop short
        rooms[p.key] = max(0.0, rooms[p.key] - used)


def test_pieces_the_balance_pays_come_after_the_free_ones():
    plan = F.plan(4 * 3600, [(0, 0)], [K(2000), K(3000), K(500, paid=True)], [])
    assert [p.key for p in plan[:2]] == [0, 1] and all(p.key == 2 for p in plan[2:])
    assert all(p.end - p.start <= F.PART_SECONDS for p in plan)  # the hour includes the overlap


@pytest.mark.parametrize("room, expected", [
    (F.MIN_ROOM - 1, [(0, 1000, 1)]),  # cut into by nobody: the 1000 s key fills, then the day stops
    (F.MIN_ROOM, [(0, 600, 0), (540, 1540, 1)]),
])
def test_room_under_ten_minutes_takes_no_cut_piece(room, expected):
    assert spans(*F.plan(1700, [(0, 0)], [K(room), K(1000)], [])) == expected


def test_room_under_ten_minutes_still_takes_a_short_remainder_whole():
    assert spans(*F.plan(300, [(0, 0)], [K(400), K(4000)], [])) == [(0, 300, 0)]


def test_without_a_paid_key_the_free_hours_fill_and_the_plan_stops_short():
    plan = F.plan(3000, [(0, 0)], [K(1000), K(1500)], [])
    assert spans(*plan) == [(0, 1000, 0), (940, 2440, 1)]
    assert F.plan(3000, [(0, 0)], [K(100), K(200)], []) == []


def test_a_long_source_is_cut_on_a_pause_within_the_hour():
    plan = F.plan(5000, [(0, 0)], [K(0, paid=True)], [(3580.0, 2.0), (3550.0, 2.0)])  # 3550: past REACH
    assert spans(*plan) == [(0, 3580.0, 0), (3520.0, 5000, 0)]


def test_next_starts_are_the_minimum_overlap_and_pauses_up_to_ninety_seconds_back():
    pauses = [(905.0, 9.0), (912.0, 1.0), (925.0, 2.0), (930.0, 3.0), (945.0, 9.0)]  # 905 and 945: out of range
    # the best TRIED_STARTS by depth: 925 at 15 s; 930's 3 s counts as 1 s, so 20 s; 912 at 28 s
    assert F.next_starts(1000, pauses) == [(940.0, None), (925.0, 15.0), (930.0, 20.0)]
    # the previous piece's segments break at 930: a quarter of the length qualifies there at once
    segs = [{"start": 900.0, "end": 929.0}, {"start": 931.0, "end": 990.0}]
    breaks = C.segment_breaks(segs)
    assert F.next_starts(1000, pauses, breaks) == [(940.0, None), (930.0, 10.0), (925.0, 15.0)]


# --- which cached call is a piece's main call ---------------------------------------------------------------


def test_main_call_of_the_first_piece_starts_at_zero_and_ends_latest():
    names = ["s-0.0-1200.0", "s-5.0-1200.0", "s-0.0-300.0", "s-0.0-1200.0-again", "s-0.0-1300.0-retry", "identity"]
    assert main_call(names, 0.0, first=True) == (0.0, 1200.0)


def test_main_call_of_a_later_piece_starts_inside_the_overlap_and_ends_past_the_previous_piece():
    names = ["s-0.0-1200.0", "s-1135.2-2300.0", "s-1150.0-1200.0", "s-1160.0-1400.0", "s-1500.0-2300.0"]
    assert main_call(names, 1200.0, first=False) == (1135.2, 2300.0)


def test_main_call_ending_with_its_tail_continuation_takes_the_earlier_start():
    names = ["s-1140.0-2300.0", "s-1145.0-2300.0"]  # same end; the second is out of range anyway
    assert main_call(names, 1200.0, first=False) == (1140.0, 2300.0)
    assert main_call(["s-1110.0-2300.0", "s-1140.0-2300.0"], 1200.0, first=False) == (1110.0, 2300.0)


def test_main_call_inside_the_previous_piece_is_not_taken():
    assert main_call(["s-1140.0-1200.0"], 1200.0, first=False) is None


def test_old_fixed_hour_windows_are_still_found():
    names = ["s-0.0-3600.0", "s-3540.0-5400.0"]
    assert main_call(names, 0.0, first=True) == (0.0, 3600.0)
    assert main_call(names, 3600.0, first=False) == (3540.0, 5400.0)
