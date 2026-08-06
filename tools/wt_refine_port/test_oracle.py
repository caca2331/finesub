from __future__ import annotations

import pytest

from tools.wt_refine_port.oracle import CASES, alignment_path, render_cases


def test_frozen_oracle_paths() -> None:
    assert render_cases() == [
        {
            "name": "wt_default_symmetric1",
            "allow_empty_subwords": True,
            "encourage_early": False,
            "path": [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
        },
        {
            "name": "wt_strict_no_empty_subwords",
            "allow_empty_subwords": False,
            "encourage_early": True,
            "path": [(0, 0), (1, 1), (2, 2)],
        },
    ]


def test_early_cell_is_orthogonal_to_step_pattern() -> None:
    case = CASES[0]
    assert alignment_path(case.weights, encourage_early=False) == alignment_path(
        case.weights, encourage_early=True
    )


@pytest.mark.parametrize("weights", [[], [[]], [1.0, 2.0]])
def test_rejects_invalid_matrices(weights) -> None:
    with pytest.raises(ValueError, match="non-empty 2D"):
        alignment_path(weights)
