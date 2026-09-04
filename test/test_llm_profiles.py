from __future__ import annotations

import pytest

from finesub.llm.routing.config import LLMRole, default_role_configs
from finesub.llm.routing.profiles import (
    DEFAULT_PROFILE,
    LEGACY_PRESET_VECTORS,
    SwitchConflictError,
    parse_profile_id,
    expected_output_tokens,
    max_window_csv_tokens,
    resolve_profile,
    video_tokens_per_second,
    window_output_budget,
)


def test_switch_vector_derives_every_trait() -> None:
    # switches -> (native_search, external_injection, correction audio/video,
    #              coefficient). The six rows are the retired presets, so
    #              this doubles as the equivalence check.
    # efficiency's coefficient discount is gone: difficulty only
    # picks the variant (thinking is the preset knob), so c is
    # difficulty-free -- 2.0 -> 3.5.
    expectations = {
        ("text", "none", "efficiency"): (False, False, False, False, 3.5),
        ("text", "none", "quality"): (False, False, False, False, 3.5),
        ("text", "native", "quality"): (True, False, False, False, 4.5),
        ("text", "local", "quality"): (False, True, False, False, 4.5),
        ("audio", "local", "quality"): (False, True, True, False, 5.0),
        ("video", "local", "quality"): (False, True, True, True, 6.0),
    }
    for switches, expected in expectations.items():
        profile = resolve_profile(*switches)
        media, retrieval, difficulty = switches
        # The convenience knob sets both per-task switches (plan v2 D20).
        assert profile.profile_id == (
            f"correction_media={media},planning_media={media},"
            f"retrieval={retrieval},difficulty={difficulty},"
            "continuity=serial"
        )
        assert (
            profile.native_search,
            profile.external_injection,
            profile.correction_use_audio,
            profile.correction_use_video,
            profile.output_coefficient,
        ) == expected
        assert profile.planning_media == profile.correction_media


def test_per_task_media_overrides_split_the_axes() -> None:
    profile = resolve_profile(
        "video", "local", "quality", correction_media="text"
    )
    assert profile.correction_media == "text"
    assert profile.planning_media == "video"
    assert profile.uses_media and profile.uses_video
    assert not profile.correction_use_audio
    assert profile.planning_use_video
    # The output coefficient budgets the correction window only.
    assert profile.output_coefficient == resolve_profile(
        "text", "local", "quality"
    ).output_coefficient
    assert profile.profile_id == (
        "correction_media=text,planning_media=video,retrieval=local,"
        "difficulty=quality,continuity=serial"
    )
    assert parse_profile_id(profile.profile_id) == profile


def test_parse_profile_id_reads_pre_split_vectors() -> None:
    # Fingerprints and replay fixtures frozen before D20 carry the single
    # ``media`` axis; it maps onto both switches.
    parsed = parse_profile_id(
        "media=audio,retrieval=local,difficulty=quality,continuity=serial"
    )
    assert parsed == resolve_profile("audio", "local", "quality")


def test_legacy_preset_names_still_parse_for_old_artifacts() -> None:
    for name, switches in LEGACY_PRESET_VECTORS.items():
        assert parse_profile_id(name) == resolve_profile(*switches)
    # Round-trip through the canonical form.
    for profile in (resolve_profile(*s) for s in LEGACY_PRESET_VECTORS.values()):
        assert parse_profile_id(profile.profile_id) == profile


def test_default_profile_keeps_the_harness_default_run_shape() -> None:
    # The retired "mm-med"; pipeline.py asks for media=video explicitly.
    assert DEFAULT_PROFILE.profile_id == (
        "correction_media=audio,planning_media=audio,retrieval=local,"
        "difficulty=quality,continuity=serial"
    )
    assert DEFAULT_PROFILE.output_scale == 1.0


def test_efficiency_pins_the_other_axes_and_errors_on_conflict() -> None:
    # efficiency's low thinking now lives on the default preset's knob
    # ([presets.default.thinking]), not on the profile.
    assert resolve_profile("text", "none", "efficiency").difficulty == "efficiency"
    for conflicting in (("audio", "none"), ("text", "local"), ("video", "native")):
        with pytest.raises(SwitchConflictError):
            resolve_profile(*conflicting, "efficiency")


def test_resolve_profile_rejects_unknown_and_bad_scale() -> None:
    with pytest.raises(ValueError):
        resolve_profile("mm", "local", "quality")
    with pytest.raises(ValueError):
        resolve_profile("audio", "web", "quality")
    with pytest.raises(ValueError):
        resolve_profile("audio", "local", "extreme")
    with pytest.raises(ValueError):
        resolve_profile("audio", "local", "quality", output_scale=0)


def test_expected_output_tokens_scales_with_k_and_coefficient() -> None:
    mm_med = resolve_profile("audio", "local", "quality")
    assert expected_output_tokens(mm_med, 1_000) == 5_000
    assert expected_output_tokens(mm_med.with_output_scale(1.3), 1_000) == 6_500
    text_low = resolve_profile("text", "none", "efficiency")
    assert expected_output_tokens(text_low, 1_000) == 3_500
    assert expected_output_tokens(text_low, 0) == 0


def test_window_output_budgets() -> None:
    # 0.9 x 65,536 - 5,000 and 0.8 x 65,536 - 10,000.
    assert window_output_budget() == 53_982
    assert window_output_budget(fast=True) == 42_428


def test_max_window_csv_tokens_matches_design_table() -> None:
    normal = {
        ("text", "none", "efficiency"): 15_423,
        ("text", "none", "quality"): 15_423,
        ("text", "native", "quality"): 11_996,
        ("text", "local", "quality"): 11_996,
        ("audio", "local", "quality"): 10_796,
        ("video", "local", "quality"): 8_997,
    }
    fast = {
        ("text", "none", "efficiency"): 12_122,
        ("text", "none", "quality"): 12_122,
        ("text", "native", "quality"): 9_428,
        ("text", "local", "quality"): 9_428,
        ("audio", "local", "quality"): 8_485,
        ("video", "local", "quality"): 7_071,
    }
    for key, expected in normal.items():
        assert max_window_csv_tokens(resolve_profile(*key)) == expected
    for key, expected in fast.items():
        assert max_window_csv_tokens(resolve_profile(*key), fast=True) == expected
    # k > 1 shrinks the cap.
    scaled = resolve_profile("audio", "local", "quality", output_scale=1.25)
    assert max_window_csv_tokens(scaled) == int(53_982 / (1.25 * 5.0))


def test_video_token_rate_low_and_high_resolution() -> None:
    assert video_tokens_per_second() == pytest.approx(17.75)
    assert video_tokens_per_second(high_resolution=True) == pytest.approx(67.25)


def test_native_search_is_a_per_call_capability_not_a_role() -> None:
    """Native search filters the bound group per call (plan v2 D4).

    No role/config carries the tool: it is a per-call request, and the group
    is filtered by what each member can actually do. In the correction group
    that leaves the paid full-Flash members alone -- the free members
    cannot ground, and there is no separate native chain to fall back to.
    """

    from finesub.llm.routing.capabilities import endpoint_supports

    configs = default_role_configs()
    for config in configs.values():
        assert config.native_search_tool == ""

    correction_chain = configs[LLMRole.AUDIO_MULTIMODAL].endpoint_chain
    assert [
        endpoint.target_id
        for endpoint in correction_chain
        if endpoint_supports(endpoint, needs_native_search=True)
    ] == ["gemini-paid-3_7-flash", "gemini-paid-3_8-flash"]

    # The packaged free-tier native group stays opt-in: 2.5 Flash is below the
    # correction/knowledge floor, so it is bound only on purpose.
    from finesub.llm.routing.model_routes import default_model_routes

    routes = default_model_routes()
    group = routes.model_groups["gemini-native-search"]
    assert all(
        routes.target_fact(target_id).supports_native_search
        and routes.target_profile(target_id).native_search_tool == "google_search"
        for target_id in group.target_ids
    )
    assert "gemini-native-search" not in set(
        routes.presets["default"].bindings.values()
    )


def test_difficulty_picks_the_variant_through_the_cell() -> None:
    """The old effective_tier ceiling is cell data now (plan v2 D2/D3):
    correction intermediate/efficiency cells carry basicB; quality carries capableC."""

    from finesub.llm.routing.config import CapabilityTier, role_config_for
    from finesub.llm.prompt_variants import resolve_variant

    assert role_config_for("correction-mm", "quality").variant == "capableC"
    assert role_config_for("correction-mm", "intermediate").variant == "basicB"
    assert role_config_for("correction-mm", "efficiency").variant == "basicB"
    # An explicit name always wins over the tier default (prompt-iteration A/B).
    assert resolve_variant("capableC", CapabilityTier.BASIC).name == "capableC"
    assert resolve_variant(None, CapabilityTier.BASIC).name == "basicB"
