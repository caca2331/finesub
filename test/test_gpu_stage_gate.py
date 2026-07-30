from __future__ import annotations

import threading

from asr_playground.speech.runtime.gpu_stage_gate import GpuStageGate


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
