from __future__ import annotations

import threading
import time

import pytest

from finesub.speech.runtime.gpu_stage_gate import GpuStageGate


def test_gpu_stage_gate_allows_same_family_and_blocks_other_family() -> None:
    gate = GpuStageGate()
    first_separator = gate.acquire("separator")
    second_separator = gate.acquire("separator")
    wt_acquired = threading.Event()
    release_wt = threading.Event()

    def acquire_wt() -> None:
        lease = gate.acquire("wt")
        wt_acquired.set()
        release_wt.wait(timeout=2)
        lease.release()

    thread = threading.Thread(target=acquire_wt)
    thread.start()
    assert not wt_acquired.wait(timeout=0.05)

    first_separator.release()
    assert not wt_acquired.wait(timeout=0.05)
    second_separator.release()
    assert wt_acquired.wait(timeout=1)

    release_wt.set()
    thread.join(timeout=1)
    assert not thread.is_alive()


def _acquire_in_thread(gate: GpuStageGate, family: str, order: list[str]) -> tuple[threading.Thread, threading.Event]:
    """Acquire `family` on a thread, note it in `order`, hold until the returned event is set."""

    release = threading.Event()

    def run() -> None:
        lease = gate.acquire(family)
        order.append(family)
        release.wait(timeout=2)
        lease.release()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, release


def test_a_queued_family_is_not_starved_by_more_of_the_running_one() -> None:
    """Once another family waits, a newcomer of the running family waits behind it."""

    gate = GpuStageGate()
    running = gate.acquire("separator")
    order: list[str] = []
    wt, release_wt = _acquire_in_thread(gate, "wt", order)
    deadline = time.monotonic() + 2
    while gate._next_family != "wt" and time.monotonic() < deadline:  # wt is queued
        time.sleep(0.005)
    late, release_late = _acquire_in_thread(gate, "separator", order)
    try:
        time.sleep(0.05)
        assert order == []
    finally:
        running.release()
        release_wt.set()
        release_late.set()
        wt.join(timeout=2)
        late.join(timeout=2)
    assert order == ["wt", "separator"]


def test_a_disabled_acquire_neither_waits_nor_counts() -> None:
    gate = GpuStageGate()
    held = gate.acquire("separator")
    passed = threading.Event()

    def bypass() -> None:
        gate.acquire("wt", enabled=False).release()
        passed.set()

    threading.Thread(target=bypass, daemon=True).start()
    try:
        assert passed.wait(timeout=1), "a disabled acquire waited for another family"
    finally:
        held.release()
    gate.acquire("wt").release()  # the bypass left nothing counted or queued


def test_releasing_a_family_that_does_not_hold_the_gate_is_an_error_and_a_lease_releases_once() -> None:
    gate = GpuStageGate()
    lease = gate.acquire("separator")
    with pytest.raises(RuntimeError, match="'wt'"):
        gate.release("wt")
    lease.release()
    lease.release()  # the second is a no-op, not a mismatch
    gate.acquire("wt").release()
