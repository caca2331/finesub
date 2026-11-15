from __future__ import annotations

import io

import pytest
import torch

from finesub.reporting import TerminalReporter, reporting_to
from finesub.speech.runtime import device as device_module
from finesub.speech.runtime import resource_usage
from finesub.speech.runtime.resources import get_resource_profile

GIB = 1024**3


@pytest.fixture
def scripted_memory(monkeypatch: pytest.MonkeyPatch):
    """Drive the sampler from a scripted sequence of current-RSS readings."""

    def install(readings: list[int | None]):
        remaining = list(readings)

        def read() -> int | None:
            return remaining.pop(0) if remaining else None

        monkeypatch.setattr(resource_usage, "_current_process_memory_bytes", read)

    return install


def test_sampler_reports_the_stage_maximum(scripted_memory) -> None:
    scripted_memory([100, 500, 200, 300])
    sampler = resource_usage.StageMemorySampler()  # consumes the first reading
    sampler._observe()
    sampler._observe()
    assert sampler.stop() == 500


def test_sampler_is_inert_where_current_rss_is_unreadable(scripted_memory) -> None:
    scripted_memory([None])
    sampler = resource_usage.StageMemorySampler()
    sampler.start()
    assert sampler._thread is None  # nothing to sample, so no thread is spawned
    assert sampler.stop() is None


def test_sampler_thread_tracks_growth() -> None:
    sampler = resource_usage.start_stage_memory_sampling(interval_sec=0.01)
    try:
        peak = sampler.stop()
    finally:
        sampler.stop()
    # Real process memory is unknown but must be a positive number on Windows
    # and Linux; other platforms degrade to None rather than lying.
    assert peak is None or peak > 0


def _run_print(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_peak: int | None,
    process_peak: int,
    profile,
) -> str:
    """Render one usage report at verbose level and return what was shown.

    Usage now goes through the reporter rather than `print`, so the figures
    only appear where someone asked for them -- verbose here.
    """

    monkeypatch.setattr(resource_usage, "_peak_gpu_memory_bytes", lambda device: 0)
    monkeypatch.setattr(
        resource_usage, "_peak_process_memory_bytes", lambda: process_peak
    )

    class _Sampler:
        def stop(self):
            return stage_peak

    sampler = None if stage_peak is None else _Sampler()
    stream = io.StringIO()
    reporter = TerminalReporter(stream, level="verbose", isatty=False)
    with reporting_to(reporter):
        resource_usage.print_peak_resource_usage(None, profile, sampler=sampler)
    return stream.getvalue()


def test_budget_check_uses_the_stage_peak_not_the_process_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_resource_profile("standard")
    # An earlier stage in this process peaked over the limit; this stage did not.
    shown = _run_print(
        monkeypatch,
        stage_peak=profile.ram_limit_bytes - 1,
        process_peak=profile.ram_limit_bytes + 10**9,
        profile=profile,
    )
    assert "peak_mem exceeds" not in shown
    # The contaminated figure stays visible so the stage number is not mistaken
    # for the whole run.
    assert "peak_mem_process=" in shown


def test_stage_peak_still_fails_its_own_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_resource_profile("standard")
    shown = _run_print(
        monkeypatch,
        stage_peak=profile.ram_limit_bytes + 1,
        process_peak=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    assert "peak_mem exceeds" in shown
    # Equal figures add no information, so the extra field is suppressed.
    assert "peak_mem_process=" not in shown


def test_without_a_sampler_the_process_peak_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = get_resource_profile("standard")
    shown = _run_print(
        monkeypatch,
        stage_peak=None,
        process_peak=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    assert "peak_mem exceeds" in shown
    assert "peak_mem_process=" not in shown


def test_usage_within_budget_stays_out_of_normal_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling numbers are not progress; only going over earns a line."""

    profile = get_resource_profile("standard")
    monkeypatch.setattr(resource_usage, "_peak_gpu_memory_bytes", lambda device: 0)
    monkeypatch.setattr(
        resource_usage,
        "_peak_process_memory_bytes",
        lambda: profile.ram_limit_bytes - 1,
    )
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level="normal", isatty=False)):
        resource_usage.print_peak_resource_usage(None, profile, sampler=None)

    assert stream.getvalue() == ""


# ---- GPU peak readings, whichever vendor owns the card ----------------------


@pytest.fixture
def xpu_allocator(monkeypatch: pytest.MonkeyPatch):
    """A ``torch.xpu`` whose peak figures are readable, like the Intel build's."""

    class _Xpu:
        def __init__(self) -> None:
            self.reset = 0

        def device_count(self) -> int:
            return 1

        def max_memory_allocated(self, index=None) -> int:
            return 3 * GIB

        def max_memory_reserved(self, index=None) -> int:
            return 5 * GIB

        def reset_peak_memory_stats(self, index=None) -> None:
            self.reset += 1

    module = _Xpu()
    monkeypatch.setattr(torch.version, "xpu", "20250200", raising=False)
    monkeypatch.setattr(torch, "xpu", module, raising=False)
    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    return module


def test_the_peak_reading_comes_from_the_placement_family(xpu_allocator) -> None:
    """An Intel stage must be measured on its allocator, not on a CUDA zero.

    The figures come from `device.peak_gpu_memory_bytes` -- here reached
    through the same module, because ``resource_usage`` may not touch
    ``torch.xpu`` itself (the source guard, see ``docs/testing.md``).
    """

    assert resource_usage._peak_gpu_memory_bytes("xpu") == 5 * GIB

    resource_usage.reset_peak_gpu_memory_stats_for_run("xpu")
    assert xpu_allocator.reset == 1

    # An indexed request addresses the same figure; the index is the backend's.
    assert resource_usage._peak_gpu_memory_bytes("xpu:0") == 5 * GIB


def test_the_usage_report_shows_the_intel_figure(xpu_allocator, monkeypatch) -> None:
    """`peak_gmem` must not read "0.00 B" on a stage that actually used the card."""

    monkeypatch.setattr(resource_usage, "_peak_process_memory_bytes", lambda: GIB)
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level="verbose", isatty=False)):
        resource_usage.print_peak_resource_usage("xpu", None, sampler=None)

    assert "peak_gmem=5.00 GB" in stream.getvalue()


def test_a_cpu_run_says_zero_without_asking_any_driver(monkeypatch) -> None:
    """No accelerator is a real answer (`0`), not "unknown" (`N/A`)."""

    monkeypatch.setattr(device_module, "gpu_device", lambda: "")
    monkeypatch.setattr(resource_usage, "_peak_process_memory_bytes", lambda: GIB)
    stream = io.StringIO()
    with reporting_to(TerminalReporter(stream, level="verbose", isatty=False)):
        resource_usage.print_peak_resource_usage(None, None, sampler=None)

    assert "peak_gmem=0.00 B" in stream.getvalue()
    assert resource_usage._peak_gpu_memory_bytes(None) == 0
