from __future__ import annotations

import threading
import time

from desktop.backend.resources.gpus import Gpu, GpuProbe, parse_nvidia_smi


def test_a_multi_gpu_machine_is_read_off_nvidia_smi() -> None:
    listing = "\n".join(
        [
            "0, NVIDIA GeForce RTX 5060 Ti, 16311",
            "1, NVIDIA GeForce RTX 4090, 24564",
        ]
    )

    assert parse_nvidia_smi(listing) == (
        Gpu(index=0, name="NVIDIA GeForce RTX 5060 Ti", memory_mb=16311),
        Gpu(index=1, name="NVIDIA GeForce RTX 4090", memory_mb=24564),
    )


def test_lines_that_are_not_devices_are_dropped() -> None:
    # Driver errors and stray headers come back on the same stream; a machine
    # with no usable GPU has to end up with an empty list, not a fake device.
    listing = "\n".join(
        [
            "index, name, memory.total",
            "NVIDIA-SMI has failed because it couldn't communicate",
            "",
            "0, NVIDIA GeForce RTX 5060 Ti, 16311",
            "1, , 8192",
        ]
    )

    assert parse_nvidia_smi(listing) == (
        Gpu(index=0, name="NVIDIA GeForce RTX 5060 Ti", memory_mb=16311),
    )


def test_the_probe_answers_before_it_has_looked() -> None:
    # The whole point: nothing in the interface may block on nvidia-smi, so the
    # snapshot is readable while the probe is still out.
    started = threading.Event()
    release = threading.Event()

    def slow_query() -> tuple[Gpu, ...]:
        started.set()
        release.wait(5)
        return (Gpu(index=0, name="NVIDIA GeForce RTX 5060 Ti", memory_mb=16311),)

    probe = GpuProbe(query=slow_query)
    probe.start()
    assert started.wait(5)

    scanning = probe.snapshot()
    assert scanning.state == "scanning"
    assert scanning.devices == ()

    release.set()
    for _ in range(500):
        if probe.snapshot().state != "scanning":
            break
        time.sleep(0.01)
    assert probe.snapshot().state == "ready"
    assert probe.snapshot().find(0) is not None


def test_a_machine_without_a_driver_reports_unavailable() -> None:
    probe = GpuProbe(query=lambda: ())
    probe.start()
    for _ in range(500):
        if probe.snapshot().state != "scanning":
            break
        time.sleep(0.01)

    assert probe.snapshot().state == "unavailable"
    assert probe.snapshot().devices == ()


def test_a_failing_probe_is_not_a_crash() -> None:
    def broken() -> tuple[Gpu, ...]:
        raise OSError("nvidia-smi went missing")

    probe = GpuProbe(query=broken)
    probe.start()
    for _ in range(500):
        if probe.snapshot().state != "scanning":
            break
        time.sleep(0.01)

    assert probe.snapshot().state == "unavailable"
