from __future__ import annotations

import pytest
import torch

from finesub.speech.runtime.thread_budget import bounded_intra_op_threads


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


def _widen_setup(monkeypatch, logical: int) -> None:
    from finesub.speech.runtime import thread_budget

    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.setattr(thread_budget, "_logical_cpus", lambda: logical)


def test_widened_takes_three_quarters_of_the_logical_cpus_and_restores(monkeypatch) -> None:
    from finesub.speech.runtime.thread_budget import widened_intra_op_threads

    baseline = torch.get_num_threads()
    if baseline < 2:
        pytest.skip("needs at least two intra-op threads")
    _widen_setup(monkeypatch, baseline * 2)  # SMT: torch's default is half the logical CPUs

    with pytest.raises(RuntimeError, match="boom"):
        with widened_intra_op_threads():
            assert torch.get_num_threads() == int(baseline * 2 * 0.75)
            raise RuntimeError("boom")
    assert torch.get_num_threads() == baseline


@pytest.mark.parametrize("why", ["no-smt", "OMP_NUM_THREADS", "MKL_NUM_THREADS"])
def test_widened_never_lowers_and_leaves_a_set_thread_count_alone(monkeypatch, why: str) -> None:
    from finesub.speech.runtime.thread_budget import widened_intra_op_threads

    baseline = torch.get_num_threads()
    _widen_setup(monkeypatch, baseline if why == "no-smt" else baseline * 4)
    if why != "no-smt":
        monkeypatch.setenv(why, str(baseline))

    with widened_intra_op_threads():
        assert torch.get_num_threads() == baseline
    assert torch.get_num_threads() == baseline


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_the_aligner_widens_the_threads_only_on_the_cpu(monkeypatch, device: str) -> None:
    import types

    import numpy as np

    from finesub.speech.joint import aligner

    baseline = torch.get_num_threads()
    if baseline < 2:
        pytest.skip("needs at least two intra-op threads")
    _widen_setup(monkeypatch, baseline * 2)
    model = types.SimpleNamespace(device=types.SimpleNamespace(type=device))
    monkeypatch.setattr(aligner, "_model", lambda: (model, None))
    monkeypatch.setattr(aligner, "_kept", lambda unit: unit)
    seen = []
    monkeypatch.setattr(aligner, "_aligned", lambda *a: seen.append(torch.get_num_threads()) or {})

    seg = {"start": 0.0, "end": 1.0, "text": "abc"}
    aligner.aligned_units([seg], all_segments=[seg], pcm16k=np.zeros(16000, dtype=np.int16))
    assert seen == [int(baseline * 2 * 0.75) if device == "cpu" else baseline]
    assert torch.get_num_threads() == baseline


def test_widened_leaves_a_count_changed_meanwhile(monkeypatch) -> None:
    from finesub.speech.runtime.thread_budget import widened_intra_op_threads

    baseline = torch.get_num_threads()
    if baseline < 2:
        pytest.skip("needs at least two intra-op threads")
    _widen_setup(monkeypatch, baseline * 2)
    try:
        with widened_intra_op_threads():
            torch.set_num_threads(1)  # another lane's Silero load, mid-way
        assert torch.get_num_threads() == 1  # its own restore is the one that counts
    finally:
        torch.set_num_threads(baseline)
