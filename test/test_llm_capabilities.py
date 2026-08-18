"""Capability table -> role chain join: startup check and runtime filtering."""

from __future__ import annotations

import pytest

from finesub import config as app_config
from finesub.llm.routing import api_keys
from finesub.llm.routing import capabilities
from finesub.llm.routing.capabilities import (
    CapabilityUnavailableError,
    ChainRequirement,
    endpoint_supports,
    required_chains,
    usable_endpoints,
    validate_profile_capabilities,
)
from finesub.llm.client import LLMCallResult, RoleClient, UploadedFileRef
from finesub.llm.routing.config import (
    GEMINI_25_FLASH,
    GEMINI_36_FLASH,
    GEMINI_37_FLASH,
    GEMINI_FREE_TIER,
    GEMINI_PAID_TIER,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
)
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.rate_limit import ModelRateLimiter

GEMMA = ModelEndpoint(
    GEMINI_FREE_TIER, "gemini/gemma-4-31b-it", native_search_tool="google_search"
)
FLASH_36 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_36_FLASH)
FLASH_25 = ModelEndpoint(
    GEMINI_FREE_TIER, GEMINI_25_FLASH, native_search_tool="google_search"
)


def test_endpoint_supports_reads_each_bit_independently() -> None:
    # Splitting supports_video_audio matters here: Gemma carries a native
    # search tool but no media at all.
    assert endpoint_supports(GEMMA, needs_native_search=True)
    assert not endpoint_supports(GEMMA, needs_audio=True)
    assert not endpoint_supports(GEMMA, needs_video=True)

    assert endpoint_supports(FLASH_36, needs_audio=True, needs_video=True)
    assert not endpoint_supports(FLASH_36, needs_native_search=True)

    assert endpoint_supports(FLASH_25, needs_audio=True, needs_native_search=True)
    # A native call needs the *target's* tool too, not just the fact bit --
    # a plain paid 3.7 endpoint still must not serve native calls without
    # serve native calls without the tool wired.
    toolless_native_fact = ModelEndpoint(GEMINI_PAID_TIER, GEMINI_37_FLASH)
    assert not endpoint_supports(toolless_native_fact, needs_native_search=True)


def test_unknown_endpoint_is_not_tightened_without_evidence() -> None:
    unknown = ModelEndpoint(
        GEMINI_PAID_TIER,
        "gemini/not-in-the-catalog",
        native_search_tool="google_search",
    )
    assert endpoint_supports(unknown, needs_audio=True, needs_native_search=True)


def test_required_chains_track_what_each_profile_actually_attaches() -> None:
    labels = lambda profile: [c.label for c in required_chains(profile)]

    # text routes: no query round, no media chain.
    text = required_chains(resolve_profile("text", "none", "quality"))
    assert len(text) == 1
    assert not text[0].needs_audio and not text[0].needs_video

    # text-high needs the native bit on the correction requirement. Since the
    # 2026-08-12 binding fix exactly one bound member can serve it -- paid
    # 3.7, whose target declares the search tool; the free members are
    # filtered out because they cannot ground at all.
    native = required_chains(resolve_profile("text", "native", "quality"))[0]
    assert native.needs_native_search
    assert [endpoint.target_id for endpoint in usable_endpoints(native)] == [
        "gemini-paid-3_7-flash"
    ]

    # mm-low: harness injection means a query round, but still no media.
    assert any("查询轮" in label for label in labels(resolve_profile("text", "local", "quality")))
    assert not any("fast round 1" in label for label in labels(resolve_profile("text", "local", "quality")))

    # video run: both switches default to video, so the query round asks for
    # the video bit too (plan v2 D20 -- it reads the shared mp4 clip).
    high = {c.label: c for c in required_chains(resolve_profile("video", "local", "quality"))}
    correction = next(c for label, c in high.items() if "纠错窗" in label)
    query = next(c for label, c in high.items() if "查询轮" in label)
    assert correction.needs_video and correction.needs_audio
    assert query.needs_audio and query.needs_video

    # ...unless planning is dialed back to audio: the pre-D20 shape, opt-in.
    dialed = {
        c.label: c
        for c in required_chains(
            resolve_profile("video", "local", "quality", planning_media="audio")
        )
    }
    query_dialed = next(c for label, c in dialed.items() if "查询轮" in label)
    assert query_dialed.needs_audio and not query_dialed.needs_video

    # correction_media=text keeps a media-less correction chain while the
    # query round still carries its clip.
    text_corr = {
        c.label: c
        for c in required_chains(
            resolve_profile("audio", "local", "quality", correction_media="text")
        )
    }
    correction_text = next(c for label, c in text_corr.items() if "纠错窗" in label)
    query_media = next(c for label, c in text_corr.items() if "查询轮" in label)
    assert not correction_text.needs_audio and not correction_text.needs_video
    assert query_media.needs_audio


def test_fast_shape_replaces_the_local_query_chain() -> None:
    profile = resolve_profile("video", "local", "quality")

    normal = {chain.label for chain in required_chains(profile)}
    fast = {
        chain.label
        for chain in required_chains(profile, fast_enabled=True)
    }

    assert any("查询轮" in label for label in normal)
    assert "fast round 1" not in normal
    assert not any("查询轮" in label for label in fast)
    assert "fast round 1" in fast


def test_fast_without_local_retrieval_has_no_fused_round1_chain() -> None:
    # Native/none fast runs use the ordinary text research stage (when one is
    # needed), not the media-bearing fast round 1.
    for retrieval in ("native", "none"):
        labels = {
            chain.label
            for chain in required_chains(
                resolve_profile("video", retrieval, "quality"),
                fast_enabled=True,
            )
        }
        assert "fast round 1" not in labels
        assert not any("查询轮" in label for label in labels)


def test_validation_checks_only_the_planned_local_shape(monkeypatch) -> None:
    # The normal query needs planning_media=audio, while both correction and
    # fused fast R1 follow correction_media=text. A text-only binding therefore
    # proves that fast validation omits the mutually exclusive query chain.
    profile = resolve_profile(
        "text", "local", "quality", planning_media="audio"
    )
    configs = {
        "correction-text": RoleModelConfig(
            role=LLMRole.AUDIO_MULTIMODAL,
            endpoint_chain=(GEMMA,),
            test_endpoint=GEMMA,
        ),
        "planning-mm": RoleModelConfig(
            role=LLMRole.LIGHTWEIGHT_MULTIMODAL,
            endpoint_chain=(GEMMA,),
            test_endpoint=GEMMA,
        ),
    }
    monkeypatch.setattr(
        "finesub.llm.routing.capabilities.role_config_for",
        lambda task_group, difficulty="quality", **kwargs: configs[task_group],
    )

    validate_profile_capabilities(profile, fast_enabled=True)
    with pytest.raises(CapabilityUnavailableError, match="查询轮"):
        validate_profile_capabilities(profile, fast_enabled=False)


@pytest.mark.parametrize(
    "media,retrieval,difficulty",
    [
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("text", "local", "quality"),
        ("audio", "local", "quality"),
        ("video", "local", "quality"),
        # Native works on the shipped preset since the 2026-08-12 binding fix:
        # paid 3.7 declares the search tool, so the bound groups have a
        # grounding member.
        ("text", "native", "quality"),
    ],
)
def test_every_shipped_profile_has_a_usable_chain(
    media: str, retrieval: str, difficulty: str
) -> None:
    validate_profile_capabilities(resolve_profile(media, retrieval, difficulty))


def test_native_retrieval_fails_fast_without_a_native_capable_binding(
    monkeypatch,
) -> None:
    """When nothing bound can ground, startup validation errors rather than
    downgrading (plan v2 D4: no fallback chain, no silent downgrade), and the
    message points at the fix -- the packaged free-tier native group."""

    real = capabilities.endpoint_supports

    def no_grounding(endpoint, **kwargs):
        if kwargs.get("needs_native_search"):
            return False
        return real(endpoint, **kwargs)

    monkeypatch.setattr(capabilities, "endpoint_supports", no_grounding)
    with pytest.raises(CapabilityUnavailableError) as excinfo:
        validate_profile_capabilities(resolve_profile("text", "native", "quality"))
    message = str(excinfo.value)
    assert "native" in message
    assert "gemini-native-search" in message


def test_validation_fails_fast_when_a_pool_is_switched_off(monkeypatch) -> None:
    monkeypatch.setattr(api_keys, "provider_tier_enabled", lambda _tier: False)
    with pytest.raises(CapabilityUnavailableError) as excinfo:
        validate_profile_capabilities(resolve_profile("video", "local", "quality"))
    message = str(excinfo.value)
    assert "纠错窗" in message
    assert "audio" in message and "video" in message


def test_agent_only_capability_check_uses_effective_policy(
    tmp_path, monkeypatch
) -> None:
    from finesub.llm.routing.model_routes import default_model_routes

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[llm]\nexecution_policy = "agent-only"\npreset = "agy"\n', encoding="utf-8"
    )
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config_path))
    app_config.clear_config_cache()
    default_model_routes.cache_clear()
    monkeypatch.setattr(api_keys, "provider_tier_enabled", lambda _tier: False)

    try:
        validate_profile_capabilities(resolve_profile("text", "none", "quality"))
        # Packaged Agy is subscription-backed and needs no API-key tier, so the
        # agy preset still has something to call once every tier is disabled --
        # its text cells reach Opus, its media cells the Gemini agy fronts.
        validate_profile_capabilities(resolve_profile("audio", "local", "quality"))
    finally:
        app_config.clear_config_cache()
        default_model_routes.cache_clear()


def test_validation_names_the_chain_that_is_short(monkeypatch) -> None:
    """A binding that serves correction but not the query round still fails."""

    configs = {
        "correction-mm": RoleModelConfig(
            role=LLMRole.AUDIO_MULTIMODAL,
            endpoint_chain=(FLASH_36,),
            test_endpoint=FLASH_36,
        ),
        "planning-mm": RoleModelConfig(
            role=LLMRole.LIGHTWEIGHT_MULTIMODAL,
            endpoint_chain=(GEMMA,),
            test_endpoint=GEMMA,
        ),
    }
    monkeypatch.setattr(
        "finesub.llm.routing.capabilities.role_config_for",
        lambda task_group, difficulty="quality", **kwargs: configs[task_group],
    )
    with pytest.raises(CapabilityUnavailableError) as excinfo:
        validate_profile_capabilities(resolve_profile("audio", "local", "quality"))
    message = str(excinfo.value)
    assert "查询轮" in message
    assert "纠错窗" not in message


def test_test_profile_skips_the_check(monkeypatch) -> None:
    monkeypatch.setattr(api_keys, "provider_tier_enabled", lambda _tier: False)
    validate_profile_capabilities(resolve_profile("video", "local", "quality"), test_profile=True)


def test_runtime_chain_skips_endpoints_that_cannot_take_the_attachment(monkeypatch) -> None:
    """A media call must not fall back onto a text-only endpoint."""

    seen: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        seen.append(model)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        role_configs={
            LLMRole.AUDIO_MULTIMODAL: RoleModelConfig(
                role=LLMRole.AUDIO_MULTIMODAL,
                # Gemma first: it must be skipped for an audio attachment.
                endpoint_chain=(GEMMA, FLASH_36),
                test_endpoint=FLASH_36,
            )
        },
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        file_ref=UploadedFileRef(
            file_id="f", filename="clip.aac", mime_type="audio/aac"
        ),
    )
    assert isinstance(result, LLMCallResult)
    assert seen == [GEMINI_36_FLASH]


def test_runtime_chain_keeps_text_only_endpoints_without_an_attachment(monkeypatch) -> None:
    seen: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        seen.append(model)
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        role_configs={
            LLMRole.LIGHTWEIGHT: RoleModelConfig(
                role=LLMRole.LIGHTWEIGHT,
                endpoint_chain=(GEMMA, FLASH_36),
                test_endpoint=FLASH_36,
            )
        },
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    client.complete(
        LLMRole.LIGHTWEIGHT,
        [{"role": "user", "content": "hi"}],
        max_tokens=32_768,
    )
    assert seen == ["gemini/gemma-4-31b-it"]


def test_chain_requirement_lists_the_bits_it_wants() -> None:
    requirement = ChainRequirement(
        label="x", endpoints=(FLASH_36,), needs_audio=True, needs_native_search=True
    )
    assert requirement.missing_capabilities() == ("audio", "native_search")


def test_native_high_no_longer_warns_about_a_forced_basic_prompt() -> None:
    """The "native pool is basic-only" warning died with the v1 native chain
    (plan v2 D4): the variant comes from the cell, so difficulty=quality gets
    capableC on whatever native-capable model the user binds."""

    from finesub.llm.routing.capabilities import profile_warnings

    assert not any(
        "不会提升" in m
        for m in profile_warnings(resolve_profile("text", "native", "quality"))
    )


def test_uncalibrated_vectors_warn_and_the_six_retired_presets_do_not() -> None:
    from finesub.llm.routing.capabilities import profile_warnings

    for switches in (
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("text", "local", "quality"),
        ("audio", "local", "quality"),
        ("video", "local", "quality"),
    ):
        assert not any(
            "未标定" in m for m in profile_warnings(resolve_profile(*switches))
        ), switches
    assert any(
        "未标定" in m
        for m in profile_warnings(resolve_profile("audio", "none", "quality"))
    )


def test_repositioned_vectors_warn_that_their_old_calibration_is_stale() -> None:
    """efficiency and media=video changed output-side conditions.

    efficiency forces basicB where the measured text-low may have answered with a
    capable prompt; video's reasoning depth dropped high -> medium. Their c
    keeps the old value until P6 re-measures, and must say so on every run.
    """

    from finesub.llm.routing.capabilities import profile_warnings

    for switches in (("text", "none", "efficiency"), ("video", "local", "quality")):
        assert any(
            "未重新标定" in m for m in profile_warnings(resolve_profile(*switches))
        ), switches
    for switches in (("audio", "local", "quality"), ("text", "none", "quality")):
        assert not any(
            "未重新标定" in m for m in profile_warnings(resolve_profile(*switches))
        ), switches
