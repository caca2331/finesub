"""Freeze whisper-timestamped 1.15.9's token-frame DTW behavior.

The CTranslate2 tests copy the paths emitted here as constants. Keeping the
oracle outside the C++ test means the production implementation cannot
accidentally validate itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import dtw
import numpy as np


@dataclass(frozen=True)
class OracleCase:
    name: str
    weights: tuple[tuple[float, ...], ...]
    allow_empty_subwords: bool = True
    encourage_early: bool = False


CASES = (
    OracleCase(
        name="wt_default_symmetric1",
        weights=(
            (0.9, 0.1, 0.0),
            (0.8, 0.2, 0.1),
            (0.1, 0.7, 0.9),
        ),
    ),
    OracleCase(
        name="wt_strict_no_empty_subwords",
        weights=(
            (0.9, 0.1, 0.0),
            (0.8, 0.2, 0.1),
            (0.1, 0.7, 0.9),
        ),
        allow_empty_subwords=False,
        encourage_early=True,
    ),
)


def _strict_step_pattern():
    return dtw.stepPattern.StepPattern(
        dtw.stepPattern._c(
            1,
            1,
            1,
            -1,
            1,
            0,
            0,
            1,
            2,
            0,
            1,
            -1,
            2,
            0,
            0,
            1,
        )
    )


def alignment_path(
    weights: Sequence[Sequence[float]],
    *,
    allow_empty_subwords: bool = True,
    encourage_early: bool = False,
) -> list[tuple[int, int]]:
    """Return WT's token/frame path for already-normalized attention weights."""

    costs = -np.asarray(weights, dtype=np.float64)
    if costs.ndim != 2 or not costs.size:
        raise ValueError("weights must be a non-empty 2D matrix")
    if encourage_early:
        costs[0, 0] = costs.min()
    pattern = (
        dtw.stepPattern.symmetric1
        if allow_empty_subwords
        else _strict_step_pattern()
    )
    result = dtw.dtw(costs, step_pattern=pattern)
    return [
        (int(token_index), int(frame_index))
        for token_index, frame_index in zip(result.index1s, result.index2s)
    ]


def render_cases() -> list[dict[str, object]]:
    return [
        {
            "name": case.name,
            "allow_empty_subwords": case.allow_empty_subwords,
            "encourage_early": case.encourage_early,
            "path": alignment_path(
                case.weights,
                allow_empty_subwords=case.allow_empty_subwords,
                encourage_early=case.encourage_early,
            ),
        }
        for case in CASES
    ]


def main() -> int:
    print(json.dumps(render_cases(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
