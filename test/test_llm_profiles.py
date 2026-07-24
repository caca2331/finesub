from __future__ import annotations

import pytest

from llm.config import LLMRole, default_role_configs
from llm.profiles import (
    DEFAULT_PROFILE,
    expected_output_tokens,
    max_window_csv_tokens,
    resolve_profile,
    video_tokens_per_second,
    window_output_budget,
)


def test_resolve_profile_strict_preset_traits() -> None:
    # (route, level) -> (native_search, external_injection, use_audio,
    #                    use_video, thinking_override, coefficient)
    expectations = {
        ("text", "low"): (False, False, False, False, "low", 2.0),
        ("text", "med"): (False, False, False, False, "", 3.5),
        ("text", "high"): (True, False, False, False, "", 4.5),
        ("mm", "low"): (False, True, False, False, "", 4.5),
        ("mm", "med"): (False, True, True, False, "", 5.0),
        ("mm", "high"): (False, True, True, True, "", 6.0),
    }
    for (route, level), expected in expectations.items():
        profile = resolve_profile(route, level)
        assert profile.profile_id == f"{route}-{level}"
        assert (
            profile.native_search,
            profile.external_injection,
            profile.use_audio,
            profile.use_video,
            profile.thinking_override,
            profile.output_coefficient,
        ) == expected


def test_default_profile_is_mm_med_current_behavior() -> None:
    assert DEFAULT_PROFILE.profile_id == "mm-med"
    assert DEFAULT_PROFILE.output_scale == 1.0


def test_resolve_profile_rejects_unknown_and_bad_scale() -> None:
    with pytest.raises(ValueError):
        resolve_profile("audio", "med")
    with pytest.raises(ValueError):
        resolve_profile("mm", "extreme")
    with pytest.raises(ValueError):
        resolve_profile("mm", "med", output_scale=0)


def test_expected_output_tokens_scales_with_k_and_coefficient() -> None:
    mm_med = resolve_profile("mm", "med")
    assert expected_output_tokens(mm_med, 1_000) == 5_000
    assert expected_output_tokens(mm_med.with_output_scale(1.3), 1_000) == 6_500
    text_low = resolve_profile("text", "low")
    assert expected_output_tokens(text_low, 1_000) == 2_000
    assert expected_output_tokens(text_low, 0) == 0


def test_window_output_budgets() -> None:
    # 0.9 x 65,536 - 5,000 and 0.8 x 65,536 - 10,000.
    assert window_output_budget() == 53_982
    assert window_output_budget(fast=True) == 42_428


def test_max_window_csv_tokens_matches_design_table() -> None:
    normal = {
        ("text", "low"): 26_991,
        ("text", "med"): 15_423,
        ("text", "high"): 11_996,
        ("mm", "low"): 11_996,
        ("mm", "med"): 10_796,
        ("mm", "high"): 8_997,
    }
    fast = {
        ("text", "low"): 21_214,
        ("text", "med"): 12_122,
        ("text", "high"): 9_428,
        ("mm", "low"): 9_428,
        ("mm", "med"): 8_485,
        ("mm", "high"): 7_071,
    }
    for key, expected in normal.items():
        assert max_window_csv_tokens(resolve_profile(*key)) == expected
    for key, expected in fast.items():
        assert max_window_csv_tokens(resolve_profile(*key), fast=True) == expected
    # k > 1 shrinks the cap.
    scaled = resolve_profile("mm", "med", output_scale=1.25)
    assert max_window_csv_tokens(scaled) == int(53_982 / (1.25 * 5.0))


def test_video_token_rate_low_and_high_resolution() -> None:
    assert video_tokens_per_second() == pytest.approx(17.75)
    assert video_tokens_per_second(high_resolution=True) == pytest.approx(67.25)


def test_internet_capable_role_config() -> None:
    configs = default_role_configs()
    role = configs[LLMRole.INTERNET_CAPABLE]
    assert role.native_search_tool == "google_search"
    assert role.thinking_level == "medium"
    assert len(role.endpoint_chain) >= 1
    # Other roles never enable native search.
    for other, config in configs.items():
        if other is not LLMRole.INTERNET_CAPABLE:
            assert config.native_search_tool == ""
