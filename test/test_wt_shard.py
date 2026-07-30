from __future__ import annotations

import pytest

from asr_playground.speech.recognition import sharding as ws


def _iv(start, end):
    return {"start": float(start), "end": float(end)}


def _groups(spec, *, gap=1.0):
    """Build groups from per-group interval counts, each interval 10s of speech
    separated by ``gap`` (and by ``gap`` across group boundaries too)."""

    groups = []
    clock = 0.0
    for count in spec:
        group = []
        for _ in range(count):
            group.append(_iv(clock, clock + 10.0))
            clock += 10.0 + gap
        groups.append(group)
    return groups


def _plan(groups, workers, **kw):
    return ws.plan_wt_shards(groups, max_workers=workers, **kw)


# --------------------------------------------------------------- worker count

@pytest.mark.parametrize(
    "total, expected",
    [
        (0.0, 1),
        (149.9, 1),
        (150.0, 2),
        (299.9, 2),
        (300.0, 3),
        (450.0, 4),
        (100000.0, 667),
    ],
)
def test_duration_worker_limit_steps_every_threshold(total, expected) -> None:
    assert ws.duration_worker_limit(total) == expected


def test_threshold_constant_is_the_documented_one() -> None:
    assert ws.WORKER_VAD_THRESHOLD_SEC == 150.0


def test_worker_count_is_capped_by_profile() -> None:
    groups = _groups([3] * 40)          # 1200s speech -> ladder allows 5
    assert ws.duration_worker_limit(ws.interval_seconds([i for g in groups for i in g])) >= 5

    assert _plan(groups, 2).workers == 2
    assert _plan(groups, 4).workers == 4


def test_short_file_uses_one_worker_despite_large_profile() -> None:
    groups = _groups([2, 2, 2])         # 60s of speech

    plan = _plan(groups, 4)

    assert plan.workers == 1
    assert plan.shards[0].successor_interval_index is None


def test_long_wall_clock_but_little_speech_stays_single_worker() -> None:
    # Two 5s intervals separated by an hour of silence: wall clock is huge,
    # effective speech is 10s. Gaps must not buy workers.
    groups = [[_iv(0.0, 5.0)], [_iv(3600.0, 3605.0)]]

    plan = _plan(groups, 4)

    assert plan.total_vad_seconds == pytest.approx(10.0)
    assert plan.workers == 1


def test_worker_count_falls_back_to_group_count() -> None:
    # Plenty of speech (900s) but only two groups to hand out.
    groups = _groups([45, 45])

    plan = _plan(groups, 4)

    assert plan.workers == 2
    assert [s.group_end_index - s.group_start_index for s in plan.shards] == [1, 1]


def test_no_groups_yields_empty_plan() -> None:
    plan = _plan([], 4)

    assert plan.workers == 0
    assert plan.shards == []


# ------------------------------------------------------------------ balancing

def test_shards_partition_groups_and_intervals_contiguously() -> None:
    groups = _groups([3, 4, 2, 5, 1, 6])

    plan = _plan(groups, 3)

    assert plan.shards[0].group_start_index == 0
    assert plan.shards[-1].group_end_index == len(groups)
    total_intervals = sum(len(g) for g in groups)
    assert plan.shards[-1].interval_end_index == total_intervals
    for left, right in zip(plan.shards, plan.shards[1:]):
        assert left.group_end_index == right.group_start_index
        assert left.interval_end_index == right.interval_start_index
        # The successor is the next shard's first interval, never the file end.
        assert left.successor_interval_index == right.interval_start_index
    assert plan.shards[-1].successor_interval_index is None


# Balancing tests drive the worker count through max_workers and neutralise the
# duration ladder, so a failure points at the boundary search and not at the
# fragmentation guard.
_NO_LADDER = {"threshold_sec": 1e-6}


def test_balance_uses_speech_not_silence() -> None:
    # Groups 0-1 are tightly packed; groups 2-3 hold the same speech spread over
    # a wall-clock span ~50x larger. Balancing on span would cut much earlier.
    tight = [
        [_iv(base + i * 11.0, base + i * 11.0 + 10.0) for i in range(10)]
        for base in (0.0, 200.0)
    ]
    sparse = [
        [_iv(base + i * 500.0, base + i * 500.0 + 10.0) for i in range(10)]
        for base in (1000.0, 7000.0)
    ]

    plan = ws.plan_wt_shards(tight + sparse, max_workers=2, **_NO_LADDER)

    assert plan.workers == 2
    assert plan.shards[0].group_end_index == 2      # speech-balanced, not span
    assert plan.shards[0].vad_seconds == pytest.approx(plan.shards[1].vad_seconds)


def test_boundary_prefers_the_larger_real_gap_on_a_tie() -> None:
    # Three equal groups, two workers. Cutting after group 0 or after group 1
    # is equally unbalanced by speech, so the bigger real gap decides.
    span = 10.0
    g0 = [_iv(0.0, span)]
    g1 = [_iv(span + 0.2, 2 * span + 0.2)]          # small gap before g1
    g2 = [_iv(2 * span + 30.0, 3 * span + 30.0)]    # big gap before g2

    plan = ws.plan_wt_shards([g0, g1, g2], max_workers=2, **_NO_LADDER)

    # target = 15s; |10-15| == |20-15|, so the 30s gap wins over the 0.2s one.
    assert plan.shards[0].group_end_index == 2


def test_tie_falls_back_to_the_earlier_boundary() -> None:
    # Same deviation tie, but now both candidate boundaries carry equal gaps,
    # so determinism comes from preferring the earlier one.
    groups = [[_iv(i * 20.0, i * 20.0 + 10.0)] for i in range(3)]

    plan = ws.plan_wt_shards(groups, max_workers=2, **_NO_LADDER)

    assert plan.shards[0].group_end_index == 1


def test_every_worker_keeps_at_least_one_group() -> None:
    groups = _groups([90, 1, 1])       # first group dominates the speech budget

    plan = _plan(groups, 3)

    assert plan.workers == 3
    for shard in plan.shards:
        assert shard.group_end_index > shard.group_start_index
    # The giant group cannot be split, so shard 0 keeps all of it.
    assert plan.shards[0].group_end_index == 1


def test_long_group_is_never_split() -> None:
    # One enormous group plus small ones: shards may be very unbalanced, but a
    # boundary must never land inside a group.
    groups = _groups([100, 1, 1, 1])
    starts = {0}
    cursor = 0
    for group in groups:
        cursor += len(group)
        starts.add(cursor)

    plan = _plan(groups, 4)

    for shard in plan.shards:
        assert shard.interval_start_index in starts
        assert shard.interval_end_index in starts


def test_plan_is_deterministic() -> None:
    groups = _groups([3, 7, 2, 9, 4, 1, 6, 5])

    first = _plan(groups, 3)
    second = _plan(groups, 3)

    assert first == second


def test_metadata_snapshot_records_workers_and_bounds() -> None:
    groups = _groups([10, 10, 10])

    meta = _plan(groups, 2).metadata()

    assert meta["wt_workers"] == 2
    assert len(meta["shards"]) == 2
    assert meta["shards"][0]["groups"][0] == 0
    assert meta["shards"][-1]["intervals"][1] == sum(len(g) for g in groups)
