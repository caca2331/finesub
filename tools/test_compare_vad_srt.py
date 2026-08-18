"""Tests for the SRT interval comparison tool.

Run from the repository root: `python -m pytest tools/test_compare_vad_srt.py`.
Not collected by the default suite (`tools/` is maintained on demand).
"""

from __future__ import annotations

import pytest

from tools.compare_vad_srt import compare_interval_sets, normalize_intervals


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
