"""Planning and execution for single-file WT sharding.

The planning functions remain pure and cheap to test. Execution receives the
recognition callable and model pool explicitly, so this module does not import
the larger transcribe service or create a circular dependency.

Two rules shape a plan:

* **Worker count** grows with *effective speech*, not wall-clock length, and
  only one worker per ``WORKER_VAD_THRESHOLD_SEC``. That threshold is a
  fragmentation guard: a short file split to the profile's limit would pay
  model warm-up, boundary tails, per-shard language re-detection and recall
  overlap on every piece, for very little decode to amortise them over.
* **Shard boundaries only fall between initial groups**, so they inherit
  ``build_alignment_groups``' semantic-pause policy instead of inventing a
  second segmentation. A long uninterrupted speech run therefore produces
  unbalanced shards on purpose: semantic integrity outranks equal load.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import checkpoint

# One extra worker allowed per this much effective speech (see module docstring).
# Halved from 300s once the per-shard fixed cost fell: the pool now builds every
# model before shard work starts, from one shared checkpoint, so a shard no
# longer opens by waiting out a model load (measured wait=40.3s at worst,
# now 0.0s). Also a prerequisite for batch running one file at a time -- at
# 300s most short files would have regressed to a single worker.
WORKER_VAD_THRESHOLD_SEC = 150.0

# Deviation/gap comparisons are rounded here before ordering, so ties break by
# the documented rule rather than by float noise.
_TIE_DIGITS = 9
INTERVAL_ID_KEY = "_interval_id"


@dataclass(frozen=True)
class WtShard:
    """One worker's contiguous slice of the initial groups."""

    shard_id: int
    group_start_index: int              # inclusive, index into initial groups
    group_end_index: int                # exclusive
    interval_start_index: int           # inclusive, global interval index
    interval_end_index: int             # exclusive
    vad_seconds: float
    successor_interval_index: Optional[int]  # first interval of the next shard

    @property
    def interval_count(self) -> int:
        return self.interval_end_index - self.interval_start_index


@dataclass(frozen=True)
class WtShardPlan:
    workers: int
    total_vad_seconds: float
    shards: List[WtShard]

    def metadata(self) -> Dict[str, object]:
        """Snapshot for ``metadata.asr_align`` -- the artifact must be able to
        say what concurrency produced it (docs/wt-parallelism.md)."""

        return {
            "wt_workers": self.workers,
            "total_vad_seconds": round(self.total_vad_seconds, 3),
            "shards": [
                {
                    "id": shard.shard_id,
                    "groups": [shard.group_start_index, shard.group_end_index],
                    "intervals": [shard.interval_start_index, shard.interval_end_index],
                    "vad_seconds": round(shard.vad_seconds, 3),
                }
                for shard in self.shards
            ],
        }


def interval_seconds(intervals: Sequence[Dict[str, object]]) -> float:
    """Effective speech time: interval interiors only, never gaps or file span."""

    return sum(
        max(0.0, float(item.get("end", 0.0)) - float(item.get("start", 0.0)))
        for item in intervals
    )


def duration_worker_limit(
    total_vad_seconds: float,
    *,
    threshold_sec: float = WORKER_VAD_THRESHOLD_SEC,
) -> int:
    if threshold_sec <= 0:
        return 1
    return int(max(0.0, total_vad_seconds) // threshold_sec) + 1


def plan_wt_shards(
    groups: Sequence[Sequence[Dict[str, object]]],
    *,
    max_workers: int,
    threshold_sec: float = WORKER_VAD_THRESHOLD_SEC,
) -> WtShardPlan:
    """Split ``groups`` (the initial ``build_alignment_groups`` output) into a
    deterministic list of shards. ``max_workers`` is the profile ceiling."""

    usable = [group for group in groups if group]
    if not usable:
        return WtShardPlan(workers=0, total_vad_seconds=0.0, shards=[])

    group_vad = [interval_seconds(group) for group in usable]
    total_vad = sum(group_vad)

    # Interval index range per group: groups partition the interval list in order.
    group_interval_start: List[int] = []
    cursor = 0
    for group in usable:
        group_interval_start.append(cursor)
        cursor += len(group)
    total_intervals = cursor

    workers = max(
        1,
        min(
            int(max_workers),
            duration_worker_limit(total_vad, threshold_sec=threshold_sec),
            len(usable),
        ),
    )

    # Real gap across each inter-group boundary, used only to break ties.
    boundary_gap: Dict[int, float] = {}
    for index in range(1, len(usable)):
        left_end = float(usable[index - 1][-1].get("end", 0.0))
        right_start = float(usable[index][0].get("start", 0.0))
        boundary_gap[index] = max(0.0, right_start - left_end)

    cuts: List[int] = []          # group indices where a new shard starts
    start_group = 0
    remaining_vad = total_vad
    for shard_index in range(workers - 1):
        remaining_workers = workers - shard_index
        target = remaining_vad / remaining_workers
        # Leave one whole group for each worker still to come.
        highest = len(usable) - (remaining_workers - 1)
        best_end: Optional[int] = None
        best_key = None
        accumulated = 0.0
        for end_group in range(start_group + 1, highest + 1):
            accumulated += group_vad[end_group - 1]
            key = (
                round(abs(accumulated - target), _TIE_DIGITS),
                -round(boundary_gap.get(end_group, 0.0), _TIE_DIGITS),
                end_group,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_end = end_group
        assert best_end is not None  # highest >= start_group + 1 by construction
        cuts.append(best_end)
        remaining_vad -= sum(group_vad[start_group:best_end])
        start_group = best_end

    starts = [0] + cuts
    ends = cuts + [len(usable)]
    shards: List[WtShard] = []
    for shard_id, (group_start, group_end) in enumerate(zip(starts, ends)):
        interval_start = group_interval_start[group_start]
        interval_end = (
            group_interval_start[group_end] if group_end < len(usable) else total_intervals
        )
        shards.append(
            WtShard(
                shard_id=shard_id,
                group_start_index=group_start,
                group_end_index=group_end,
                interval_start_index=interval_start,
                interval_end_index=interval_end,
                vad_seconds=sum(group_vad[group_start:group_end]),
                successor_interval_index=(
                    interval_end if group_end < len(usable) else None
                ),
            )
        )
    return WtShardPlan(workers=len(shards), total_vad_seconds=total_vad, shards=shards)


def tag_interval_ids(
    intervals: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Copy intervals and attach their global ownership index."""

    return [
        {**interval, INTERVAL_ID_KEY: index}
        for index, interval in enumerate(intervals)
    ]


def strip_interval_ids(
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    return [
        {key: value for key, value in segment.items() if key != INTERVAL_ID_KEY}
        for segment in segments
    ]


def merge_shard_segments(
    shard_outputs: Sequence[Tuple[int, int, List[Dict[str, object]]]],
) -> List[Dict[str, object]]:
    """Merge per-shard segments according to interval ownership."""

    owned: List[Tuple[int, float, float, Dict[str, object]]] = []
    for interval_start, interval_end, segments in shard_outputs:
        for segment in segments:
            interval_id = segment.get(INTERVAL_ID_KEY)
            if interval_id is None:
                raise ValueError(
                    "sharded ASR segment is missing interval ownership; "
                    "discard the incompatible checkpoint and rerun"
                )
            index = int(interval_id)
            if not (interval_start <= index < interval_end):
                continue
            owned.append(
                (
                    index,
                    float(segment.get("start", 0.0)),
                    float(segment.get("end", 0.0)),
                    segment,
                )
            )
    owned.sort(key=lambda row: (row[0], row[1], row[2]))
    return [row[3] for row in owned]


def _suffixed_partial(aligned_output: str | Path, shard_id: int) -> Path:
    return Path(aligned_output).with_suffix(f".partial.shard-{shard_id:03d}.json")


def checkpoint_path(
    aligned_output: str | Path, shard_id: int, *, shard_count: int
) -> Path:
    """Return the checkpoint path for one shard."""

    if shard_count <= 1:
        return checkpoint.path_for_output(aligned_output)
    return _suffixed_partial(aligned_output, shard_id)


def align_segments_sharded(
    intervals: List[Dict[str, object]],
    audio: object | None,
    sr: int,
    *,
    plan: WtShardPlan,
    model_pool,
    gap_sec: float,
    language: Optional[str],
    audio_loader_factory,
    align_segments_fn: Callable[..., List[Dict[str, object]]],
    aligned_output: Optional[str | Path] = None,
    checkpoint_key: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    """Run one recognition worker per shard and merge by interval ownership."""

    if not intervals or not plan.shards:
        return []

    tagged = tag_interval_ids(intervals)

    def run_shard(shard: WtShard) -> List[Dict[str, object]]:
        shard_intervals = tagged[
            shard.interval_start_index : shard.interval_end_index
        ]
        successor_start = (
            float(tagged[shard.successor_interval_index].get("start", 0.0))
            if shard.successor_interval_index is not None
            else None
        )
        partial = (
            checkpoint_path(
                aligned_output, shard.shard_id, shard_count=len(plan.shards)
            )
            if aligned_output is not None
            else None
        )
        key = dict(checkpoint_key or {})
        if key and len(plan.shards) > 1:
            key["shard"] = [shard.shard_id, plan.workers]
            key["shard_intervals"] = [
                shard.interval_start_index,
                shard.interval_end_index,
            ]
        queued = time.perf_counter()
        with model_pool.lease() as model:
            leased = time.perf_counter()
            loader = audio_loader_factory()
            try:
                return align_segments_fn(
                    shard_intervals,
                    audio,
                    sr,
                    model=model,
                    gap_sec=gap_sec,
                    language=language,
                    audio_loader=loader,
                    checkpoint_path=partial,
                    checkpoint_key=key or None,
                    successor_start=successor_start,
                )
            finally:
                close = getattr(loader, "close", None)
                if callable(close):
                    close()
                if len(plan.shards) > 1:
                    print(
                        f"Info: shard {shard.shard_id} done "
                        f"(intervals={shard.interval_count}, "
                        f"speech={shard.vad_seconds:.1f}s, "
                        f"wait={leased - queued:.1f}s, "
                        f"align={time.perf_counter() - leased:.1f}s)",
                        file=sys.stderr,
                    )

    if len(plan.shards) == 1:
        results = [run_shard(plan.shards[0])]
    else:
        with ThreadPoolExecutor(max_workers=len(plan.shards)) as pool:
            futures = [pool.submit(run_shard, shard) for shard in plan.shards]
            results = [future.result() for future in futures]

    if aligned_output is not None:
        sweep_stale_partials(aligned_output, keep=len(plan.shards))

    merged = merge_shard_segments(
        [
            (shard.interval_start_index, shard.interval_end_index, segments)
            for shard, segments in zip(plan.shards, results)
        ]
    )
    return strip_interval_ids(merged)


def sweep_stale_partials(aligned_output: str | Path, *, keep: int) -> None:
    """Remove partials left by a previous, differently-shaped plan."""

    target = Path(aligned_output)
    pattern = target.with_suffix(".partial.shard-*.json").name
    live = {_suffixed_partial(target, index).name for index in range(keep)}
    try:
        for stale in target.parent.glob(pattern):
            if stale.name not in live:
                stale.unlink(missing_ok=True)
    except OSError:
        pass
