from __future__ import annotations

from pathlib import Path

import pytest

from finesub.llm import provider_transports
from finesub.llm.client import (
    LLMCallResult,
    RoleClient,
    extract_token_distribution,
    is_prompt_blocked,
)
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.model_router import FailureKind, ModelRouter, classify_failure
from finesub.llm.routing.model_routes import load_model_routes
from finesub.llm.rate_limit import ModelRateLimiter


# Facts in the catalog, composition in config.toml (owner decision
# 2026-08-12). These two halves are what a user actually writes.
USER_CATALOG = """fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens|thinking|quality_score
ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|32768|high,medium,low|60
claude|anthro|anthropic|https://api.anthropic.com|claude-sonnet-5|200000|64000|xhigh,medium,low|50
"""


def _catalog(tmp_path, text: str = USER_CATALOG):
    from finesub.llm.routing.model_catalog import load_model_catalog, merge_catalogs

    path = tmp_path / "model_catalog.psv"
    path.write_text(text, encoding="utf-8")
    return merge_catalogs(
        load_model_catalog(), load_model_catalog(path, self_reported=True)
    )


def _user_config() -> dict:
    return {
        "preset": "mine",
        "model_groups": {"my-corr": {"targets": ["ds-flash"]}},
        "presets": {
            "mine": {
                "name": "mine",
                "bindings": {"correction-text/quality": "my-corr"},
            }
        },
    }


def test_openai_compat_payload_omits_sampling_and_maps_reasoning(monkeypatch, tmp_path) -> None:
    """D18: no temperature/top_p ever; the pre-mapped thinking value is sent
    as reasoning_effort verbatim."""

    seen = {}

    def fake_post(url, *, headers, payload, timeout):
        seen.update(url=url, headers=dict(headers), payload=dict(payload))
        return {
            "choices": [
                {
                    "message": {
                        "content": "答案",
                        "reasoning_content": "内心戏",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 40,
                "prompt_cache_miss_tokens": 60,
                "reasoning_tokens": 10,
            },
        }

    monkeypatch.setattr(provider_transports, "_post_json", fake_post)
    response = provider_transports.openai_compat_generate(
        base_url="https://api.deepseek.com",
        api_key="sk-test",
        model="deepseek-v4-flash",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=32_768,
        thinking_level="medium",
        timeout=30.0,
    )

    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["payload"]["reasoning_effort"] == "medium"
    assert "temperature" not in seen["payload"]
    assert "top_p" not in seen["payload"]
    # Raw chain of thought must never surface as visible output.
    assert "reasoning_content" not in response["choices"][0]["message"]
    dist = extract_token_distribution(response)
    assert dist["cached_input_tokens"] == 40
    assert dist["thinking_tokens"] == 10
    assert dist["output_tokens"] == 20  # completion minus reasoning


def test_openai_compat_empty_thinking_sends_no_parameter(monkeypatch, tmp_path) -> None:
    """A fact declaring thinking=none maps every level to "" -- and an empty
    value must not put reasoning_effort on the wire (vLLM/Ollama may 400 on
    unknown fields)."""

    monkeypatch.setattr(
        provider_transports,
        "_post_json",
        lambda url, *, headers, payload, timeout: (
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
            if "reasoning_effort" not in payload
            else (_ for _ in ()).throw(AssertionError("must not send effort"))
        ),
    )
    provider_transports.openai_compat_generate(
        base_url="https://x",
        api_key="k",
        model="no-think",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=100,
        thinking_level="",
        timeout=5.0,
    )


def test_anthropic_payload_and_normalization(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_post(url, *, headers, payload, timeout):
        seen.update(url=url, headers=dict(headers), payload=dict(payload))
        return {
            "content": [
                {"type": "thinking", "thinking": "…"},
                {"type": "text", "text": "答案"},
            ],
            "stop_reason": "max_tokens",
            "usage": {
                "input_tokens": 80,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 5,
                "output_tokens": 50,
            },
        }

    monkeypatch.setattr(provider_transports, "_post_json", fake_post)
    response = provider_transports.anthropic_generate(
        base_url="https://api.anthropic.com",
        api_key="sk-ant",
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "hi"},
        ],
        max_tokens=64_000,
        thinking_level="xhigh",
        timeout=30.0,
    )

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant"
    assert seen["headers"]["anthropic-version"]
    # system is a top-level parameter, not a message (Messages API contract).
    assert seen["payload"]["system"] == "规则"
    assert all(m["role"] != "system" for m in seen["payload"]["messages"])
    assert seen["payload"]["max_tokens"] == 64_000
    # The pre-mapped effort word goes to output_config.effort (owner
    # decision 2026-08-11), not thinking.budget_tokens.
    assert seen["payload"]["output_config"] == {"effort": "xhigh"}
    assert "thinking" not in seen["payload"]
    assert "temperature" not in seen["payload"]

    assert response["choices"][0]["message"]["content"] == "答案"
    assert response["choices"][0]["finish_reason"] == "length"
    dist = extract_token_distribution(response)
    assert dist["prompt_tokens"] == 105  # input + cache read + cache write
    assert dist["cached_input_tokens"] == 20


def test_anthropic_refusal_counts_as_blocked_prompt(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        provider_transports,
        "_post_json",
        lambda url, *, headers, payload, timeout: {
            "content": [],
            "stop_reason": "refusal",
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    )
    response = provider_transports.anthropic_generate(
        base_url="https://api.anthropic.com",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1000,
        thinking_level="",
        timeout=5.0,
    )
    assert is_prompt_blocked("", response)


def test_quota_classification_for_non_gemini_providers() -> None:
    # DeepSeek insufficient balance is a 402; OpenAI reports an exhausted
    # account as 429 insufficient_quota. Both must advance the group (quota),
    # not spin retries or hard-fail as permanent.
    assert classify_failure(
        RuntimeError("HTTP 402 Insufficient Balance")
    ) is FailureKind.QUOTA
    assert classify_failure(
        RuntimeError('HTTP 429 {"error": {"type": "insufficient_quota"}}')
    ) is FailureKind.QUOTA
    assert classify_failure(
        RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
    ) is FailureKind.RATE_LIMIT


def test_correction_planning_limits_follow_the_group_envelope(
    monkeypatch, tmp_path
) -> None:
    """Window planning budgets against the bound group's minimum
    -- prompt + output must fit the smallest member's context window."""

    import finesub.llm.routing.model_routes as model_routes_module
    from finesub.llm.routing.config import DEFAULT_LIMITS
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.routing.capabilities import correction_planning_limits

    routes = load_model_routes(user_config=_user_config(), catalog=_catalog(tmp_path))
    monkeypatch.setattr(
        model_routes_module, "default_model_routes", lambda: routes
    )

    limits = correction_planning_limits(resolve_profile("text", "none", "quality"))
    assert limits.output_limit == 32_768
    assert limits.prompt_input_limit == 128_000 - 32_768 - DEFAULT_LIMITS.safety_margin
    assert limits.context_limit == 128_000

    # Difficulty fallback (2026-08-11): the user preset binds only high, so
    # intermediate reuses the same custom group -- and the same envelope.
    assert (
        correction_planning_limits(resolve_profile("text", "none", "intermediate")).output_limit
        == 32_768
    )
    # A cell resolving to a whole-Gemini group keeps the defaults untouched
    # (correction-mm falls through to the default preset here).
    assert (
        correction_planning_limits(resolve_profile("audio", "local", "quality"))
        is DEFAULT_LIMITS
    )


def test_client_dispatches_custom_provider_through_transport(monkeypatch, tmp_path) -> None:
    routes = load_model_routes(user_config=_user_config(), catalog=_catalog(tmp_path))
    seen = {}

    def fake_transport(**kwargs):
        seen.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    monkeypatch.setattr(
        "finesub.llm.provider_transports.openai_compat_generate", fake_transport
    )
    monkeypatch.setenv("FINESUB_KEY_DEEPSEEK", "sk-deepseek-test")
    config = role_config_for("correction-text", "quality", routes=routes)
    client = RoleClient(
        router=ModelRouter(routes),
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
        role_configs={LLMRole.AUDIO_MULTIMODAL: config},
    )

    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        max_tokens=32_768,
        seed=7,
    )

    assert result.content == "ok"
    assert result.target_id == "ds-flash"
    assert seen["base_url"] == "https://api.deepseek.com"
    assert seen["api_key"] == "sk-deepseek-test"
    assert seen["model"] == "deepseek-v4-flash"
    # The cell's abstract medium request arrives mapped through the fact's
    # declared thinking spec ("high,medium,low" identity here).
    assert seen["thinking_level"] == "medium"
    # The provider-agnostic re-roll: seed rides the prompt tail (D18/§10.2).
    assert seen["messages"][-1]["content"].endswith("(seed=7)")


# --- Custom providers are first-class at runtime, not just at load time -----
#
# Every check below used to consult the *packaged* catalog alone, where a
# user-declared model simply does not exist. Two failure shapes came out of
# that (fixed 2026-08-12) and both are pinned here: a hard ValueError from the
# rate limiter on the production path, and an audio call that a text-only
# endpoint silently accepted.

_CUSTOM_CONFIG = """
[llm]
preset = "mine"

[llm.model_groups.my-corr]
targets = ["ds-flash"]

[llm.presets.mine]
name = "mine"
[llm.presets.mine.bindings]
"correction-mm/quality" = "my-corr"
"""

_CUSTOM_CATALOG = """fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|rpm|tpm
ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|30|1000000
"""


@pytest.fixture
def custom_provider_config(tmp_path: Path, monkeypatch):
    """Point the whole process at a data root declaring one custom model.

    Both halves, as a user writes them: the fact in the data root's
    ``model_catalog.psv``, the composition in ``config.toml`` next to it.
    """

    from finesub import config as app_config
    from finesub.llm.routing.model_catalog import default_model_catalog
    from finesub.llm.routing.model_routes import default_model_routes

    (tmp_path / "config.toml").write_text(_CUSTOM_CONFIG, encoding="utf-8")
    (tmp_path / "model_catalog.psv").write_text(_CUSTOM_CATALOG, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    monkeypatch.setenv("FINESUB_MODEL_CATALOG", str(tmp_path / "model_catalog.psv"))
    app_config.clear_config_cache()
    default_model_catalog.cache_clear()
    default_model_routes.cache_clear()
    yield tmp_path
    app_config.clear_config_cache()
    default_model_catalog.cache_clear()
    default_model_routes.cache_clear()


def test_rate_limiter_reads_user_declared_limits(custom_provider_config) -> None:
    """The limiter is enabled on every production path; resolving limits from
    the packaged catalog alone raised ValueError for a custom provider, i.e.
    the whole feature failed on first call. It must read the declared rpm/tpm
    -- including from the bare (tier, model) endpoint that ``chat_complete``
    reconstructs internally, which carries no fact id."""

    from finesub.llm.routing.config import ModelEndpoint

    limiter = ModelRateLimiter(
        enabled=True, state_path=custom_provider_config / "rl.json"
    )
    limits = limiter.limits_for(ModelEndpoint("deepseek", "deepseek-v4-flash"))

    assert limits.effective_rpm == 27  # 30 * 0.9 safety factor
    assert limits.effective_tpm == 900_000
    assert limiter.reserve(ModelEndpoint("deepseek", "deepseek-v4-flash"), 100)


def test_text_only_custom_model_is_filtered_out_of_media_calls(
    custom_provider_config,
) -> None:
    """Media stays Gemini-only in this cut (D10), so a catalog row that says
    nothing about media is text-only. Resolving it against the packaged
    catalog returned "unknown", which *passes*, and the transport then dropped
    the attachment while flattening messages to text -- a silent quality loss
    of exactly the kind v2 set out to remove."""

    from finesub.llm.routing.capabilities import endpoint_supports
    from finesub.llm.routing.config import ModelEndpoint

    endpoint = ModelEndpoint(
        provider_tier="deepseek",
        api_model_id="deepseek-v4-flash",
        target_id="ds-flash",
        fact_id="ds-flash",
        backend="openai_compat",
    )

    assert endpoint_supports(endpoint) is True
    assert endpoint_supports(endpoint, needs_audio=True) is False
    assert endpoint_supports(endpoint, needs_video=True) is False


def test_binding_warnings_follow_the_active_preset(
    custom_provider_config, capsys
) -> None:
    """§5.4's warnings exist for user presets; emitting them for the hard-coded
    default (which is warning-free by test) made the mechanism dead code."""

    from finesub.llm.routing.capabilities import validate_profile_capabilities
    from finesub.llm.routing.profiles import resolve_profile

    validate_profile_capabilities(
        resolve_profile("text", "none", "quality"), test_profile=False
    )

    warnings = capsys.readouterr().err
    assert "ds-flash" in warnings
    assert "低于下限" in warnings  # floor: default 50 < 70
    assert "规划包络" in warnings  # envelope: 128k < 194k baseline


def test_catalog_row_thinking_defaults_to_the_identity_mapping(tmp_path) -> None:
    """A row that says nothing about thinking gets the abstract levels
    verbatim (all three dialects spell them the same); ``false`` opts out."""

    from finesub.llm.routing.model_catalog import thinking_value_for

    catalog = _catalog(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|thinking\n"
        "ident|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|\n"
        "off|anthro|anthropic|https://api.anthropic.com|claude-sonnet-5|200000|false\n",
    )
    facts = {entry.fact_id: entry for entry in catalog}

    assert thinking_value_for(facts["ident"], "high") == "high"
    assert thinking_value_for(facts["ident"], "medium") == "medium"
    assert facts["off"].thinking_levels is None
    assert thinking_value_for(facts["off"], "high") == ""


def test_duplicate_provider_model_pair_is_rejected(tmp_path) -> None:
    """(provider, model) is the rate-limit accounting key and the fallback
    fact lookup, so two rows on one endpoint would share a bucket."""

    with pytest.raises(ValueError, match="duplicate provider/model fact"):
        _catalog(
            tmp_path,
            "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens\n"
            "a|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000\n"
            "b|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000\n",
        )


def test_token_scale_corrects_the_local_estimate_only(monkeypatch, tmp_path) -> None:
    """Plan v2 D14: the 3-tier counter speaks Gemini's vocabulary, so another
    model's estimate is systematically off. ``token_scale`` corrects it where
    estimates are actually used -- the input-limit check and the TPM
    reservation -- and nowhere else: reported usage stays the provider's."""

    from finesub.llm.routing.model_router import ModelRouter

    catalog = _catalog(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|token_scale\n"
        "ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|1.5\n",
    )
    routes = load_model_routes(user_config=_user_config(), catalog=catalog)
    monkeypatch.setattr("finesub.llm.client.estimate_call_input_tokens", lambda *a, **k: 1_000)
    seen = {}

    def fake_transport(**kwargs):
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1_800, "completion_tokens": 2},
        }

    def fake_reserve(endpoint, estimated_input_tokens, **kwargs):
        seen["reserved"] = estimated_input_tokens
        return None

    monkeypatch.setattr(
        "finesub.llm.provider_transports.openai_compat_generate", fake_transport
    )
    monkeypatch.setenv("FINESUB_KEY_DEEPSEEK", "sk-test")
    limiter = ModelRateLimiter(enabled=True, state_path=tmp_path / "rl.json")
    monkeypatch.setattr(limiter, "reserve", fake_reserve)
    client = RoleClient(
        router=ModelRouter(routes),
        max_retries=0,
        rate_limiter=limiter,
        role_configs={
            LLMRole.AUDIO_MULTIMODAL: role_config_for(
                "correction-text", "quality", routes=routes
            )
        },
    )

    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL, [{"role": "user", "content": "hi"}]
    )

    accepted = result.route_decision["candidates"][0]
    assert accepted["estimated_input_tokens"] == 1_500  # 1000 * 1.5
    assert accepted["unscaled_estimate_tokens"] == 1_000
    assert seen["reserved"] == 1_500
    # The calibration sample compares the provider's number against the
    # *unscaled* estimate, so it measures the real bias (1800/1000).
    assert accepted["estimate_calibration"]["ratio"] == 1.8
    assert accepted["estimate_calibration"]["token_scale"] == 1.5


def test_calibration_reference_suggests_only_with_enough_samples() -> None:
    """Advisory by design (plan §5.6): token_scale moves window geometry, so
    adopting it is an explicit config change, and a thin sample is noise."""

    from finesub.llm.task_report import _calibration_lines

    def sample(ratio: float, scale: float = 1.0) -> dict:
        return {"fact_id": "ds", "ratio": ratio, "token_scale": scale}

    thin = _calibration_lines({"ds": [sample(1.8)] * 3})
    assert any("too few samples" in line for line in thin)

    drifting = _calibration_lines({"ds": [sample(1.8)] * 6})
    assert any("consider token_scale = 1.80" in line for line in drifting)

    settled = _calibration_lines({"ds": [sample(1.52, 1.5)] * 6})
    assert any("keep" in line for line in settled)

    assert _calibration_lines({}) == ["- No calibration samples were retained."]


def test_planning_envelope_converts_through_token_scale(monkeypatch, tmp_path) -> None:
    """Planning counts in local estimate tokens, ``max_input_tokens`` is in the
    provider's, and ``token_scale`` is the bridge.

    Without the conversion the planner and the dispatch check disagreed: a
    window planned to fit was rejected as ``input_limit`` on arrival, every
    window, leaving no candidate. The regression shape is a big-context /
    small-output model, where the envelope sits close to the real ceiling.
    """

    import finesub.llm.routing.model_routes as model_routes_module
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.routing.capabilities import correction_planning_limits

    catalog = _catalog(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens|token_scale\n"
        "ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|200000|8000|1.5\n",
    )
    routes = load_model_routes(user_config=_user_config(), catalog=catalog)
    monkeypatch.setattr(model_routes_module, "default_model_routes", lambda: routes)

    limits = correction_planning_limits(resolve_profile("text", "none", "quality"))

    # (200000 - 8000 - 1000) / 1.5, i.e. the provider budget in local units.
    assert limits.prompt_input_limit == 127_333
    # ...and that is what the dispatch check will accept once scaled back up.
    assert limits.prompt_input_limit * 1.5 <= 200_000
    assert limits.context_limit == int(200_000 / 1.5)


def test_planning_envelope_never_relaxes_on_a_sub_unit_scale(
    monkeypatch, tmp_path
) -> None:
    """A fact declaring token_scale < 1 says the local counter over-estimates
    for it. Estimates are a safety bound, so that must not widen the window --
    the scale is clamped at 1.0 (the plan's "dangerous direction")."""

    import finesub.llm.routing.model_routes as model_routes_module
    from finesub.llm.routing.config import DEFAULT_LIMITS
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.routing.capabilities import correction_planning_limits

    catalog = _catalog(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|token_scale\n"
        "ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|194000|0.5\n",
    )
    routes = load_model_routes(user_config=_user_config(), catalog=catalog)
    monkeypatch.setattr(model_routes_module, "default_model_routes", lambda: routes)

    limits = correction_planning_limits(resolve_profile("text", "none", "quality"))

    assert routes.group_estimate_scale("my-corr") == 1.0
    assert limits.prompt_input_limit <= DEFAULT_LIMITS.prompt_input_limit


def test_provider_rows_must_agree_on_key_env(tmp_path) -> None:
    """The provider owns the key, so two rows disagreeing about it is not a
    row-order question: the second row's key would simply never be used while
    the file looks right."""

    with pytest.raises(ValueError, match="already declared"):
        load_model_routes(
            catalog=_catalog(
                tmp_path,
                "fact_id|provider_tier|provider_kind|base_url|key_env|api_model_id|max_input_tokens\n"
                "a|deepseek|openai_compat|https://api.deepseek.com|KEY_A|model-a|128000\n"
                "b|deepseek|openai_compat|https://api.deepseek.com|KEY_B|model-b|128000\n",
            )
        )


def test_text_dialect_row_cannot_claim_media(tmp_path) -> None:
    """Media is Gemini-only in this cut (D10) and the text transports flatten
    messages to plain strings, so a row claiming audio on one of them would
    pass capability filtering and then have its clip silently dropped.

    The old ``[llm.models]`` surface hard-coded ``supports_audio=False``; the
    catalog column is free-form, so the invariant is declared here instead.
    """

    from finesub.llm.routing.model_catalog import load_model_catalog

    path = tmp_path / "model_catalog.psv"
    path.write_text(
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|supports_audio\n"
        "ds|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|128000|true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supports_audio=true is not available"):
        load_model_catalog(path)


def test_knowledge_chunks_are_sized_against_the_bound_group(
    monkeypatch, tmp_path
) -> None:
    """D13's envelope reached the correction loop but not the knowledge stage:
    chunks built at the packaged 194k ceiling for a group whose smallest member
    holds far less are skipped at dispatch as ``input_limit`` -- the model the
    user bound never answers, and a single-member group fails outright."""

    import finesub.llm.routing.model_routes as model_routes_module
    from finesub.llm.routing.config import DEFAULT_LIMITS, planning_limits_for

    catalog = _catalog(
        tmp_path,
        "fact_id|provider_tier|provider_kind|base_url|api_model_id|max_input_tokens|max_output_tokens\n"
        "ds-flash|deepseek|openai_compat|https://api.deepseek.com|deepseek-v4-flash|32000|8000\n",
    )
    user = _user_config()
    user["presets"]["mine"]["bindings"]["knowledge/quality"] = "my-corr"
    routes = load_model_routes(user_config=user, catalog=catalog)
    monkeypatch.setattr(model_routes_module, "default_model_routes", lambda: routes)

    limits = planning_limits_for("knowledge")

    assert limits.prompt_input_limit == 32_000 - 8_000 - DEFAULT_LIMITS.safety_margin
    assert limits.prompt_input_limit < DEFAULT_LIMITS.prompt_input_limit
    # A whole-Gemini knowledge binding is untouched.
    assert planning_limits_for("knowledge", routes=load_model_routes()) is DEFAULT_LIMITS
