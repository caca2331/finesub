from __future__ import annotations

import pytest
import torch

from asr_playground.speech.runtime.thread_budget import bounded_intra_op_threads


def test_budget_is_split_across_shards_and_restored() -> None:
    baseline = torch.get_num_threads()
    if baseline < 2:
        pytest.skip("needs at least two intra-op threads to split")

    with bounded_intra_op_threads(2):
        assert torch.get_num_threads() == max(1, baseline // 2)
    assert torch.get_num_threads() == baseline


def test_single_shard_leaves_the_budget_alone() -> None:
    baseline = torch.get_num_threads()
    with bounded_intra_op_threads(1):
        assert torch.get_num_threads() == baseline
    assert torch.get_num_threads() == baseline


def test_budget_never_drops_below_one() -> None:
    baseline = torch.get_num_threads()
    with bounded_intra_op_threads(baseline * 8):
        assert torch.get_num_threads() >= 1
    assert torch.get_num_threads() == baseline


def test_budget_is_restored_when_the_body_raises() -> None:
    baseline = torch.get_num_threads()
    if baseline < 2:
        pytest.skip("needs at least two intra-op threads to split")

    with pytest.raises(RuntimeError, match="boom"):
        with bounded_intra_op_threads(2):
            raise RuntimeError("boom")
    # A failed shard run must not leave the process throttled for later stages.
    assert torch.get_num_threads() == baseline
