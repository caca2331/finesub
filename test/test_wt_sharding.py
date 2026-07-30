"""Execution-side tests for WT sharding (docs/wt-parallelism.md).

No model and no GPU: `align_segments` is monkeypatched so these exercise the
sharding contract itself -- successor plumbing, interval ownership, model
exclusivity, determinism under out-of-order completion.
"""

from __future__ import annotations

import contextlib
import threading
import time

import pytest

from asr_playground.speech.recognition import transcribe as asr_align
from asr_playground.speech.recognition import sharding as ws
from asr_playground.speech.recognition import checkpoint as checkpoint_store


def _iv(start, end):
    return {"start": float(start), "end": float(end)}


# ------------------------------------------------------- successor plumbing

def test_next_interval_start_prefers_the_real_next_interval() -> None:
    remaining = [_iv(0, 1), _iv(2, 3), _iv(4, 5)]

    assert asr_align._next_interval_start(remaining, 1, None) == pytest.approx(2.0)
    assert asr_align._next_interval_start(remaining, 2, None) == pytest.approx(4.0)


def test_next_interval_start_uses_successor_at_the_shard_tail() -> None:
    remaining = [_iv(0, 1), _iv(2, 3)]

    # Whole list consumed: without a successor this is the end of the audio,
    # with one it is a shard boundary and the next shard's interval bounds it.
    assert asr_align._next_interval_start(remaining, 2, None) is None
    assert asr_align._next_interval_start(remaining, 2, 9.5) == pytest.approx(9.5)


# ------------------------------------------------------- ownership merging

def _seg(interval_id, start, end, text):
    return {
        "start": float(start),
        "end": float(end),
        "text": text,
        ws.INTERVAL_ID_KEY: interval_id,
    }


def test_merge_drops_segments_outside_the_owning_shard() -> None:
    # Shard 0 owns intervals [0,2); its padding produced a segment attributed to
    # interval 2, which belongs to shard 1 and must be discarded in favour of
    # shard 1's own output for that interval.
    left = [_seg(0, 0.0, 1.0, "a"), _seg(1, 2.0, 3.0, "b"), _seg(2, 4.0, 5.0, "leak")]
    right = [_seg(2, 4.0, 5.0, "real"), _seg(3, 6.0, 7.0, "d")]

    merged = ws.merge_shard_segments([(0, 2, left), (2, 4, right)])

    assert [s["text"] for s in merged] == ["a", "b", "real", "d"]


def test_merge_is_independent_of_worker_completion_order() -> None:
    left = [_seg(0, 0.0, 1.0, "a"), _seg(1, 2.0, 3.0, "b")]
    right = [_seg(2, 4.0, 5.0, "c")]

    forward = ws.merge_shard_segments([(0, 2, left), (2, 3, right)])
    reversed_ = ws.merge_shard_segments([(2, 3, right), (0, 2, left)])

    assert forward == reversed_


def test_merge_orders_multiple_segments_inside_one_interval() -> None:
    shard = [_seg(0, 3.0, 4.0, "late"), _seg(0, 1.0, 2.0, "early")]

    merged = ws.merge_shard_segments([(0, 1, shard)])

    assert [s["text"] for s in merged] == ["early", "late"]


def test_merge_rejects_segment_without_interval_ownership() -> None:
    with pytest.raises(ValueError, match="missing interval ownership"):
        ws.merge_shard_segments(
            [(0, 1, [{"start": 0.0, "end": 1.0, "text": "legacy"}])]
        )


def test_interval_ids_are_tagged_and_stripped() -> None:
    intervals = [_iv(0, 1), _iv(2, 3)]

    tagged = ws.tag_interval_ids(intervals)
    assert [item[ws.INTERVAL_ID_KEY] for item in tagged] == [0, 1]
    assert ws.INTERVAL_ID_KEY not in intervals[0]      # input untouched

    stripped = ws.strip_interval_ids([_seg(0, 0.0, 1.0, "a")])
    assert ws.INTERVAL_ID_KEY not in stripped[0]
    assert stripped[0]["text"] == "a"


# ------------------------------------------------------------ checkpointing

def test_single_shard_keeps_the_unsharded_partial_name() -> None:
    plain = checkpoint_store.path_for_output("out/x/x-aligned.json")

    assert ws.checkpoint_path(
        "out/x/x-aligned.json", 0, shard_count=1
    ) == plain
    assert ws.checkpoint_path(
        "out/x/x-aligned.json", 0, shard_count=3
    ) != plain
    assert "shard-002" in str(
        ws.checkpoint_path("out/x/x-aligned.json", 2, shard_count=3)
    )


# ------------------------------------------------- sharded execution driver

class _FakePool:
    """Hands out distinct models and asserts none is ever used concurrently."""

    def __init__(self, size):
        self._free = [f"model-{i}" for i in range(size)]
        self._busy = set()
        self._lock = threading.Lock()
        self.max_concurrent = 0

    @contextlib.contextmanager
    def lease(self):
        with self._lock:
            assert self._free, "pool handed out more models than it has"
            model = self._free.pop()
            assert model not in self._busy, "model leased twice at once"
            self._busy.add(model)
            self.max_concurrent = max(self.max_concurrent, len(self._busy))
        try:
            yield model
        finally:
            with self._lock:
                self._busy.discard(model)
                self._free.append(model)


class _Loader:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def captured(monkeypatch):
    """Replace align_segments with a recorder that emits one segment per
    interval, tagged with that interval's id."""

    calls = []

    def fake_align(intervals, audio, sr, *, model, gap_sec, language,
                   audio_loader=None, checkpoint_path=None, checkpoint_key=None,
                   successor_start=None):
        calls.append(
            {
                "ids": [item[ws.INTERVAL_ID_KEY] for item in intervals],
                "successor_start": successor_start,
                "model": model,
                "checkpoint_path": checkpoint_path,
                "checkpoint_key": checkpoint_key,
                "loader": audio_loader,
            }
        )
        # Emit into the neighbour too, so ownership filtering has something to do.
        out = [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": f"t{item[ws.INTERVAL_ID_KEY]}",
                ws.INTERVAL_ID_KEY: item[ws.INTERVAL_ID_KEY],
            }
            for item in intervals
        ]
        out.append(
            {
                "start": 0.0,
                "end": 0.1,
                "text": "spill",
                ws.INTERVAL_ID_KEY: intervals[-1][ws.INTERVAL_ID_KEY] + 1,
            }
        )
        return out

    monkeypatch.setattr(asr_align, "align_segments", fake_align)
    return calls


def _run(intervals, groups, workers, captured_loaders=None, **kw):
    plan = ws.plan_wt_shards(groups, max_workers=workers, threshold_sec=1e-6)
    pool = _FakePool(max(1, plan.workers))

    def factory():
        loader = _Loader()
        if captured_loaders is not None:
            captured_loaders.append(loader)
        return loader

    segments = asr_align.align_segments_sharded(
        intervals,
        None,
        16000,
        plan=plan,
        model_pool=pool,
        gap_sec=0.3,
        language="ja",
        audio_loader_factory=factory,
        **kw,
    )
    return plan, pool, segments


def test_sharded_run_covers_every_interval_exactly_once(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]

    _plan, _pool, segments = _run(intervals, groups, 3)

    assert [s["text"] for s in segments] == [f"t{i}" for i in range(6)]
    assert all(ws.INTERVAL_ID_KEY not in s for s in segments)


def test_each_shard_receives_its_successor_start(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]

    plan, _pool, _segments = _run(intervals, groups, 3)

    by_first_id = {tuple(call["ids"])[0]: call for call in captured}
    for shard in plan.shards:
        call = by_first_id[shard.interval_start_index]
        if shard.successor_interval_index is None:
            assert call["successor_start"] is None
        else:
            expected = intervals[shard.successor_interval_index]["start"]
            assert call["successor_start"] == pytest.approx(expected)


def test_each_shard_gets_a_private_loader_that_is_released(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]
    loaders = []

    plan, _pool, _segments = _run(intervals, groups, 3, captured_loaders=loaders)

    assert plan.workers == 3
    assert len(loaders) == 3
    assert all(loader.closed for loader in loaders)          # released, not leaked
    assert len({id(call["loader"]) for call in captured}) == 3


def test_concurrent_shards_hold_distinct_models(monkeypatch) -> None:
    # Shards that merely run back-to-back may legitimately reuse one idle model,
    # so force real overlap: no shard returns until all three are inside.
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]
    barrier = threading.Barrier(3, timeout=10)
    seen = []
    lock = threading.Lock()

    def fake_align(intervals_, audio, sr, *, model, **kw):
        with lock:
            seen.append(model)
        barrier.wait()
        return []

    monkeypatch.setattr(asr_align, "align_segments", fake_align)

    _plan, pool, _segments = _run(intervals, groups, 3)

    assert len(set(seen)) == 3          # _FakePool also asserts no double-lease
    assert pool.max_concurrent == 3


def test_single_worker_runs_inline_and_keeps_the_plain_checkpoint(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(4)]
    intervals = [item for group in groups for item in group]

    _plan, _pool, segments = _run(
        intervals, groups, 1,
        aligned_output="out/x/x-aligned.json",
        checkpoint_key={"model": "m"},
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["successor_start"] is None
    assert call["checkpoint_path"] == checkpoint_store.path_for_output(
        "out/x/x-aligned.json"
    )
    # Single-shard keys stay exactly as the caller built them.
    assert call["checkpoint_key"] == {"model": "m"}
    assert [s["text"] for s in segments] == [f"t{i}" for i in range(4)]


def test_multi_shard_checkpoints_are_per_shard_and_fingerprinted(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]

    _run(
        intervals, groups, 3,
        aligned_output="out/x/x-aligned.json",
        checkpoint_key={"model": "m"},
    )

    paths = {str(call["checkpoint_path"]) for call in captured}
    assert len(paths) == 3
    assert all("shard-" in path for path in paths)
    for call in captured:
        assert call["checkpoint_key"]["model"] == "m"
        assert "shard" in call["checkpoint_key"]
        assert "shard_intervals" in call["checkpoint_key"]


def test_result_is_stable_when_workers_finish_out_of_order(monkeypatch) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]

    def fake_align(intervals_, audio, sr, *, model, gap_sec, language,
                   audio_loader=None, checkpoint_path=None, checkpoint_key=None,
                   successor_start=None):
        # Earlier shards finish last.
        time.sleep(0.02 * (5 - intervals_[0][ws.INTERVAL_ID_KEY]))
        return [
            {
                "start": float(item["start"]),
                "end": float(item["end"]),
                "text": f"t{item[ws.INTERVAL_ID_KEY]}",
                ws.INTERVAL_ID_KEY: item[ws.INTERVAL_ID_KEY],
            }
            for item in intervals_
        ]

    monkeypatch.setattr(asr_align, "align_segments", fake_align)

    _plan, _pool, segments = _run(intervals, groups, 3)

    assert [s["text"] for s in segments] == [f"t{i}" for i in range(6)]


def test_shard_failure_propagates(monkeypatch) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(4)]
    intervals = [item for group in groups for item in group]

    def fake_align(intervals_, *a, **kw):
        if intervals_[0][ws.INTERVAL_ID_KEY] != 0:
            raise RuntimeError("shard blew up")
        return []

    monkeypatch.setattr(asr_align, "align_segments", fake_align)

    with pytest.raises(RuntimeError, match="shard blew up"):
        _run(intervals, groups, 2)


def test_empty_input_returns_empty(captured) -> None:
    plan = ws.plan_wt_shards([], max_workers=4)

    assert asr_align.align_segments_sharded(
        [], None, 16000,
        plan=plan,
        model_pool=_FakePool(1),
        gap_sec=0.3,
        language=None,
        audio_loader_factory=_Loader,
    ) == []


# --------------------------------------------------------------- model pool

def test_model_pool_loads_lazily_reuses_and_caps(monkeypatch) -> None:
    from asr_playground.speech.runtime import model_pool

    built = []
    monkeypatch.setattr(
        model_pool,
        "load_whisper_model_serialized",
        lambda whisper, name, *, device, checkpoint=None: (
            built.append(name) or f"m{len(built)}"
        ),
    )
    pool = model_pool.WtModelPool(object(), "turbo", device="cpu", size=2)

    assert pool.loaded == 0                      # nothing built until leased
    with pool.lease() as first:
        assert pool.loaded == 1
    with pool.lease() as second:
        assert second is first                   # idle model reused, not rebuilt
    assert pool.loaded == 1

    with pool.lease():
        with pool.lease():
            assert pool.loaded == 2              # second built only under overlap
    assert len(built) == 2


def test_model_pool_blocks_past_its_size(monkeypatch) -> None:
    from asr_playground.speech.runtime import model_pool

    monkeypatch.setattr(
        model_pool,
        "load_whisper_model_serialized",
        lambda whisper, name, *, device, checkpoint=None: object(),
    )
    pool = model_pool.WtModelPool(object(), "turbo", device="cpu", size=1)
    entered = threading.Event()
    acquired_second = threading.Event()

    def waiter():
        with pool.lease():
            acquired_second.set()

    with pool.lease():
        thread = threading.Thread(target=waiter, daemon=True)
        thread.start()
        entered.set()
        assert not acquired_second.wait(timeout=0.2)   # blocked while size is 1
    thread.join(timeout=5)
    assert acquired_second.is_set()                   # released -> unblocked


def test_model_pool_load_failure_frees_its_slot(monkeypatch) -> None:
    from asr_playground.speech.runtime import model_pool

    def boom(whisper, name, *, device, checkpoint=None):
        raise RuntimeError("no such model")

    monkeypatch.setattr(model_pool, "load_whisper_model_serialized", boom)
    pool = model_pool.WtModelPool(object(), "nope", device="cpu", size=1)

    for _ in range(3):
        # A failed load must not permanently consume the slot, or a retry
        # would deadlock instead of reporting the real error.
        with pytest.raises(RuntimeError, match="no such model"):
            with pool.lease():
                pass
    assert pool.loaded == 0


def test_warm_builds_every_instance_from_one_shared_checkpoint(monkeypatch) -> None:
    from asr_playground.speech.runtime import model_pool

    shared = ({"dims": {}}, b"heads")
    monkeypatch.setattr(
        model_pool,
        "read_shared_checkpoint",
        lambda name, *, device: shared,
    )
    seen = []
    monkeypatch.setattr(
        model_pool,
        "load_whisper_model_serialized",
        lambda whisper, name, *, device, checkpoint=None: (
            seen.append(checkpoint) or object()
        ),
    )

    pool = model_pool.WtModelPool(object(), "turbo", device="cuda", size=3)
    pool.warm()

    # All three exist before any shard runs, so no build lands on a shard's
    # critical path, and each was built from the same already-read checkpoint.
    assert pool.loaded == 3
    assert seen == [shared, shared, shared]

    with pool.lease(), pool.lease(), pool.lease():
        assert pool.loaded == 3  # leasing them all builds nothing further


def test_warm_survives_a_checkpoint_it_cannot_pre_read(monkeypatch) -> None:
    from asr_playground.speech.runtime import model_pool

    monkeypatch.setattr(
        model_pool, "read_shared_checkpoint", lambda name, *, device: None
    )
    seen = []
    monkeypatch.setattr(
        model_pool,
        "load_whisper_model_serialized",
        lambda whisper, name, *, device, checkpoint=None: (
            seen.append(checkpoint) or object()
        ),
    )

    pool = model_pool.WtModelPool(object(), "local.pt", device="cuda", size=2)
    pool.warm()

    # No shared checkpoint (CPU run, local path, unreadable cache) still warms;
    # each build just falls back to the stock loader.
    assert pool.loaded == 2
    assert seen == [None, None]


def test_read_shared_checkpoint_declines_cpu_and_unofficial_names() -> None:
    from asr_playground.speech.runtime import model_pool

    # Neither call may touch the network or the cache.
    assert model_pool.read_shared_checkpoint("large-v3-turbo", device="cpu") is None
    assert model_pool.read_shared_checkpoint("./local.pt", device="cuda") is None


# ------------------------------------------------- the load-bearing invariant

def _spans(groups):
    return [[(round(i["start"], 6), round(i["end"], 6)) for i in g] for g in groups]


@pytest.mark.parametrize("seed", range(12))
def test_shard_slices_regroup_exactly_like_the_full_file(seed) -> None:
    """Shards cut only between initial groups, and `align_segments` recomputes
    groups from its own slice. If that recomputation disagreed with the initial
    grouping, shard boundaries would silently stop being semantic boundaries --
    the premise the whole design rests on. Assert it directly.
    """

    import random

    rng = random.Random(seed)
    intervals = []
    clock = 0.0
    for _ in range(120):
        length = rng.uniform(0.4, 9.0)
        intervals.append(_iv(clock, clock + length))
        clock += length + rng.choice([0.05, 0.2, 0.6, 1.5, 4.0])

    gap_sec = 0.3
    initial = asr_align.build_alignment_groups(intervals, gap_sec=gap_sec)
    plan = ws.plan_wt_shards(initial, max_workers=4, threshold_sec=1e-6)
    assert plan.workers >= 2, "test needs a genuinely sharded plan"

    rebuilt = []
    for shard in plan.shards:
        slice_ = intervals[shard.interval_start_index:shard.interval_end_index]
        rebuilt.extend(asr_align.build_alignment_groups(slice_, gap_sec=gap_sec))

    assert _spans(rebuilt) == _spans(initial)


def test_single_shard_passes_the_original_intervals_through(captured) -> None:
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(4)]
    intervals = [item for group in groups for item in group]

    _run(intervals, groups, 1)

    # Tagging must add the id and nothing else, or the unsharded path would
    # no longer be the unsharded path.
    seen = captured[0]
    assert seen["ids"] == [0, 1, 2, 3]
    assert seen["successor_start"] is None


def test_stale_shard_partials_are_swept(tmp_path, captured) -> None:
    aligned = tmp_path / "x-aligned.json"
    for index in (0, 1, 2, 3, 4):
        ws.checkpoint_path(aligned, index, shard_count=5).write_text(
            "{}", encoding="utf-8"
        )
    groups = [[_iv(i * 10, i * 10 + 5)] for i in range(6)]
    intervals = [item for group in groups for item in group]

    _run(intervals, groups, 3, aligned_output=aligned, checkpoint_key={"model": "m"})

    left = sorted(p.name for p in tmp_path.glob("*.partial.shard-*.json"))
    # The 3 live shards keep their files (the fake align never clears them);
    # the two from the wider previous plan are gone.
    assert left == [
        ws.checkpoint_path(aligned, i, shard_count=3).name
        for i in range(3)
    ]
