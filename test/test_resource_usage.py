from __future__ import annotations

import pytest

from asr_playground.speech.runtime import resource_usage
from asr_playground.speech.runtime.resources import get_resource_profile


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
):
    monkeypatch.setattr(resource_usage, "_peak_gpu_memory_bytes", lambda device: 0)
    monkeypatch.setattr(
        resource_usage, "_peak_process_memory_bytes", lambda: process_peak
    )

    class _Sampler:
        def stop(self):
            return stage_peak

    sampler = None if stage_peak is None else _Sampler()
    resource_usage.print_peak_resource_usage(None, profile, sampler=sampler)


def test_budget_check_uses_the_stage_peak_not_the_process_peak(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    profile = get_resource_profile(8)
    # An earlier stage in this process peaked over the limit; this stage did not.
    _run_print(
        monkeypatch,
        stage_peak=profile.ram_limit_bytes - 1,
        process_peak=profile.ram_limit_bytes + 10**9,
        profile=profile,
    )
    captured = capsys.readouterr()
    assert "peak_mem exceeds" not in captured.err
    # The contaminated figure stays visible so the stage number is not mistaken
    # for the whole run.
    assert "peak_mem_process:" in captured.out


def test_stage_peak_still_fails_its_own_budget(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    profile = get_resource_profile(8)
    _run_print(
        monkeypatch,
        stage_peak=profile.ram_limit_bytes + 1,
        process_peak=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    captured = capsys.readouterr()
    assert "peak_mem exceeds" in captured.err
    # Equal figures add no information, so the extra line is suppressed.
    assert "peak_mem_process:" not in captured.out


def test_without_a_sampler_the_process_peak_is_used(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    profile = get_resource_profile(8)
    _run_print(
        monkeypatch,
        stage_peak=None,
        process_peak=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    captured = capsys.readouterr()
    assert "peak_mem exceeds" in captured.err
    assert "peak_mem_process:" not in captured.out
