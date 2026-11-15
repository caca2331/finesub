from __future__ import annotations

import numpy as np
import pytest

from finesub.speech.preprocessing.separator import separation
from finesub.speech.preprocessing.separator.separation import (
    WORKER_DURATION_THRESHOLD_SEC,
    plan_separation_blocks,
    separator_worker_limit,
)

SR = 44100


def cores(blocks, total_frames):
    edges = [block.block_start for block in blocks] + [total_frames]
    return [edges[i + 1] - edges[i] for i in range(len(blocks))]


@pytest.mark.parametrize(
    "duration_sec, expected",
    [(0.0, 1), (299.0, 1), (300.0, 2), (899.0, 3), (2669.0, 9)],
)
def test_worker_limit_grows_one_step_per_threshold(duration_sec, expected) -> None:
    assert separator_worker_limit(duration_sec) == expected


def test_worker_limit_collapses_when_the_ladder_is_disabled() -> None:
    assert separator_worker_limit(9999.0, threshold_sec=0.0) == 1


def test_blocks_are_equal_and_a_whole_multiple_of_the_workers() -> None:
    total = 2669 * SR
    blocks = plan_separation_blocks(
        total, SR, workers=2, max_core_seconds=600.0, pad_samples=10 * SR
    )

    assert len(blocks) == 6            # 2669s / (600s x 2) -> 3 rounds x 2 workers
    assert len(blocks) % 2 == 0        # every worker gets the same number
    lengths = cores(blocks, total)
    assert max(lengths) - min(lengths) <= 1     # equal but for integer rounding
    assert max(lengths) <= 600 * SR             # and inside the core limit


def test_blocks_tile_the_timeline_without_gaps_or_overlap() -> None:
    total = 1000 * SR
    blocks = plan_separation_blocks(
        total, SR, workers=3, max_core_seconds=600.0, pad_samples=5 * SR
    )

    assert blocks[0].block_start == 0
    starts = [block.block_start for block in blocks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    # Cores must cover the whole file: the last core ends at total_frames.
    assert sum(cores(blocks, total)) == total


def test_pad_is_clamped_at_both_ends() -> None:
    total = 900 * SR
    pad = 10 * SR
    blocks = plan_separation_blocks(
        total, SR, workers=3, max_core_seconds=600.0, pad_samples=pad
    )

    assert blocks[0].read_start == 0                  # no pad before the file
    assert blocks[-1].read_end == total               # nor past its end
    for block in blocks[1:]:
        assert block.read_start == block.block_start - pad


def test_single_worker_still_respects_the_core_limit() -> None:
    total = 2669 * SR
    blocks = plan_separation_blocks(
        total, SR, workers=1, max_core_seconds=600.0, pad_samples=0
    )

    assert len(blocks) == 5
    assert max(cores(blocks, total)) <= 600 * SR


def test_short_file_yields_one_block_per_worker() -> None:
    total = 200 * SR
    blocks = plan_separation_blocks(
        total, SR, workers=1, max_core_seconds=600.0, pad_samples=0
    )
    assert len(blocks) == 1


def test_the_ladder_keeps_cores_above_the_implied_floor() -> None:
    # No separate minimum-core knob exists because the ladder already implies
    # one: at one round the core is duration/workers, bounded below by
    # 300k/(k+1), smallest at k=1 -> 150s. Sampled around and between the steps,
    # including the exact threshold multiples where the bound is tight.
    for duration_sec in (300, 301, 599, 600, 601, 900, 1200, 2669, 7200):
        workers = separator_worker_limit(float(duration_sec))
        total = duration_sec * SR
        blocks = plan_separation_blocks(
            total, SR, workers=workers, max_core_seconds=600.0, pad_samples=0
        )
        shortest = min(cores(blocks, total)) / SR
        assert shortest >= 150.0, (duration_sec, workers, shortest)


def test_threshold_constant_is_the_documented_one() -> None:
    assert WORKER_DURATION_THRESHOLD_SEC == 300.0


@pytest.mark.parametrize(
    "seconds, workers, core_seconds, pad_seconds",
    [
        (30, 1, 5, 10),      # cores shorter than the pad (separator CLI `--block-seconds 5 --pad-seconds 10`)
        (16, 1, 15, 10),     # a core limit above the pad still splits into two 8 s cores
        (2669, 2, 600, 10),  # the pipeline's shape
    ],
)
def test_merged_blocks_tile_the_source_exactly(seconds, workers, core_seconds, pad_seconds) -> None:
    rate = 100
    total = seconds * rate
    blocks = plan_separation_blocks(
        total, rate, workers=workers, max_core_seconds=core_seconds, pad_samples=pad_seconds * rate
    )

    class Merge:
        samplerate, channels = rate, 1
        written: list[np.ndarray] = []

        def write(self, frames) -> None:
            self.written.append(frames[:, 0])

    merge = Merge()
    for block in blocks:
        # A stem whose every frame is its source frame number, so a repeat or a shift shows.
        stem = np.arange(block.read_start, block.read_end, dtype=np.float64)[np.newaxis, :]
        separation._append_separated_block(
            out_file=merge, stem=stem, stem_rate=rate, output_path=None, merge_format="flac",
            block=block, source_rate=rate, chunk_frames=4096,
        )
    assert np.array_equal(np.concatenate(merge.written), np.arange(total))
