from __future__ import annotations

import numpy as np
import pytest
import torch

from test.compare_vad_srt import compare_interval_sets, normalize_intervals
from asr_playground.speech.preprocessing.energy import (
    _score_to_non_speech_intervals,
    invert_intervals,
)


def _energy_track(labels: list[str]) -> dict:
    """Build scorer inputs from per-frame 'loud'/'quiet' labels at a 10ms hop."""
    arr = np.array(labels)
    energy = np.where(arr == "quiet", -60.0, -10.0).astype(np.float32)
    noise = np.full(len(labels), -40.0, dtype=np.float32)
    starts = (np.arange(len(labels)) * 0.01).astype(np.float32)
    ends = (starts + 0.025).astype(np.float32)
    return {
        "energy_db": torch.from_numpy(energy),
        "noise_floor_db": torch.from_numpy(noise),
        "frame_dbfs": torch.from_numpy(energy.copy()),
        "frame_starts": torch.from_numpy(starts),
        "frame_ends": torch.from_numpy(ends),
        "duration_sec": float(ends[-1]) if len(labels) else 0.0,
    }


def test_score_detects_a_sustained_quiet_span() -> None:
    # 50 loud, 50 quiet, 50 loud: the quiet span (>=40 accumulated score at
    # +2/frame) is emitted once, bounded to the quiet region only.
    track = _energy_track(["loud"] * 50 + ["quiet"] * 50 + ["loud"] * 50)
    intervals = _score_to_non_speech_intervals(**track, enter_margin_db=6.0, weighted=True)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert start == pytest.approx(0.50, abs=1e-3)
    assert end == pytest.approx(1.015, abs=1e-3)


def test_score_ignores_quiet_span_below_confirmation() -> None:
    # 10 quiet frames only reach score 20 (< 40), so no interval is confirmed.
    track = _energy_track(["loud"] * 20 + ["quiet"] * 10 + ["loud"] * 20)
    assert _score_to_non_speech_intervals(**track, enter_margin_db=6.0, weighted=True) == []


def test_score_handles_empty_and_all_quiet_tracks() -> None:
    assert _score_to_non_speech_intervals(**_energy_track([]), enter_margin_db=6.0, weighted=True) == []
    # A track that is quiet throughout is one non-speech interval to the end.
    all_quiet = _score_to_non_speech_intervals(
        **_energy_track(["quiet"] * 60), enter_margin_db=6.0, weighted=True
    )
    assert len(all_quiet) == 1
    assert all_quiet[0][0] == pytest.approx(0.0, abs=1e-3)


def test_invert_intervals_clamps_and_fills_gaps() -> None:
    assert invert_intervals([(-1.0, 1.0), (2.5, 4.0)], 3.0) == [(1.0, 2.5)]


def test_normalize_intervals_merges_overlaps_and_touching_ranges() -> None:
    assert normalize_intervals([(2.0, 3.0), (0.0, 1.0), (1.0, 2.5)]) == [
        (0.0, 3.0)
    ]


def test_compare_interval_sets_reports_jaccard_metrics() -> None:
    result = compare_interval_sets(
        [(0.0, 2.0), (4.0, 6.0)],
        [(1.0, 5.0)],
        label_a="vad",
        label_b="ref",
    )

    assert result.intersection_sec == pytest.approx(2.0)
    assert result.union_sec == pytest.approx(6.0)
    assert result.jaccard_similarity == pytest.approx(1.0 / 3.0)
    assert result.jaccard_distance == pytest.approx(2.0 / 3.0)
