from __future__ import annotations

import pytest

from asr_playground.speech.runtime.resources import (
    BYTES_PER_GIB,
    DEFAULT_GPU_BUDGET_GB,
    GPU_SYSTEM_RESERVE_GB,
    RAM_BUDGET_GB,
    RESOURCE_PROFILES,
    get_resource_profile,
    gpu_budget_choices,
    resource_limit_violations,
)


def test_gpu_budget_profiles_reserve_system_memory() -> None:
    assert gpu_budget_choices() == (4, 8, 12, 16)
    for budget_gb, profile in RESOURCE_PROFILES.items():
        assert profile.gpu_budget_gb == budget_gb
        assert profile.gpu_system_reserve_gb == GPU_SYSTEM_RESERVE_GB
        assert profile.usable_gpu_gb == pytest.approx(budget_gb - 1.0)
        assert profile.gpu_limit_bytes == int((budget_gb - 1.0) * BYTES_PER_GIB)
        assert profile.ram_budget_gb == RAM_BUDGET_GB
        assert profile.ram_limit_bytes == RAM_BUDGET_GB * BYTES_PER_GIB


def test_separator_instances_scale_once_per_4gb() -> None:
    """ASR runs one worker regardless of budget; only separation scales."""

    profiles = [RESOURCE_PROFILES[item] for item in gpu_budget_choices()]
    assert [profile.vocal_separator_instances for profile in profiles] == [
        1,
        2,
        3,
        4,
    ]
    assert [profile.vocal_separation_batch_size for profile in profiles] == [1, 1, 1, 1]


def test_default_profile_is_4gb_budget() -> None:
    assert get_resource_profile().gpu_budget_gb == DEFAULT_GPU_BUDGET_GB
    assert get_resource_profile().vocal_separator_instances == 1
    assert get_resource_profile().vocal_separation_batch_size == 1


def test_unknown_gpu_budget_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported GPU budget"):
        get_resource_profile(10)


def test_resource_limit_violations_cover_gpu_and_ram_caps() -> None:
    profile = get_resource_profile(8)
    assert resource_limit_violations(
        peak_gpu_bytes=profile.gpu_limit_bytes,
        peak_ram_bytes=profile.ram_limit_bytes,
        profile=profile,
    ) == []
    violations = resource_limit_violations(
        peak_gpu_bytes=profile.gpu_limit_bytes + 1,
        peak_ram_bytes=profile.ram_limit_bytes + 1,
        profile=profile,
    )
    assert len(violations) == 2
    assert "peak_gmem exceeds" in violations[0]
    assert "peak_mem exceeds" in violations[1]
