from __future__ import annotations

from pathlib import Path

import pytest

from finesub.llm import provider_transports
from finesub.llm.http import StreamProgress
from finesub.llm.client import (
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
USER_CATALOG = """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
max_output_tokens = 32768
thinking = ["high", "medium", "low"]
capability = "low"

[models.claude]
provider_tier = "anthro"
provider_kind = "anthropic"
base_url = "https://api.anthropic.com"
api_model_id = "claude-sonnet-5"
max_input_tokens = 200000
max_output_tokens = 64000
thinking = ["xhigh", "medium", "low"]
capability = "low"
"""


def _catalog(tmp_path, text: str = USER_CATALOG):
    from finesub.llm.routing.model_catalog import load_effective_catalog

    path = tmp_path / "model_catalog.toml"
    path.write_text(text, encoding="utf-8")
    return load_effective_catalog(path)


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

    def fake_stream(url, *, headers, payload, assembler, timeout):
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

    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)
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
        progress=StreamProgress(),
    )

    assert seen["url"] == "https://api.deepseek.com/chat/completions"
    # Streamed, with usage: without include_usage the stream carries none.
    assert seen["payload"]["stream"] is True
    assert seen["payload"]["stream_options"] == {"include_usage": True}
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
        "_stream_json",
        lambda url, *, headers, payload, assembler, timeout: (
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
        progress=StreamProgress(),
    )


STREAMS = Path(__file__).parent / "fixtures" / "llm_streams"


def _feeding(lines, seen=None):
    """A `_stream_json` stand-in that feeds `lines` to the real assembler."""

    def fake_stream(url, *, headers, payload, assembler, timeout):
        if seen is not None:
            seen.update(url=url, headers=dict(headers), payload=dict(payload))
        for line in lines:
            assembler.feed_line(line)
        return assembler.finish()

    return fake_stream


def test_anthropic_payload_and_normalization(monkeypatch, tmp_path) -> None:
    seen = {}
    lines = (STREAMS / "anthropic-omitted-thinking.sse").read_text(encoding="utf-8").split("\n")
    monkeypatch.setattr(provider_transports, "_stream_json", _feeding(lines, seen))
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
        progress=StreamProgress(),
    )

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-ant"
    assert seen["headers"]["anthropic-version"]
    assert seen["payload"]["stream"] is True
    # system is a top-level parameter, not a message (Messages API contract),
    # with the breakpoint every call of the stage reads after it.
    assert seen["payload"]["system"] == [
        {"type": "text", "text": "规则", "cache_control": {"type": "ephemeral"}}
    ]
    assert all(m["role"] != "system" for m in seen["payload"]["messages"])
    # No next turn announced: the last message is not written to the cache.
    assert seen["payload"]["messages"][-1] == {"role": "user", "content": "hi"}
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
    assert dist["cache_write_input_tokens"] == 5
    assert dist["uncached_input_tokens"] == 80
    # message_delta's cumulative count replaces message_start's placeholder.
    assert response["usage"]["completion_tokens"] == 50


def test_anthropic_refusal_counts_as_blocked_prompt(monkeypatch, tmp_path) -> None:
    lines = [
        'data: {"type":"message_start","message":{"id":"m","model":"m","usage":{"input_tokens":10,"output_tokens":0}}}',
        "",
        'data: {"type":"message_delta","delta":{"stop_reason":"refusal"},"usage":{"output_tokens":0}}',
        "",
        'data: {"type":"message_stop"}',
        "",
    ]
    monkeypatch.setattr(provider_transports, "_stream_json", _feeding(lines))
    response = provider_transports.anthropic_generate(
        base_url="https://api.anthropic.com",
        api_key="k",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1000,
        thinking_level="",
        timeout=5.0,
        progress=StreamProgress(),
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
    # The row declares no `context_window`, which means "input and output are
    # independent pools" -- so the envelope is the declared input limit, not
    # input minus output. A single-pool endpoint has to say so with the
    # column; see docs/manual/model-routing.md, 「模型事实表(catalog)」.
    assert limits.prompt_input_limit == 128_000

    # Difficulty fallback (2026-08-11): the user preset binds only high, so
    # intermediate reuses the same custom group -- and the same envelope.
    assert (
        correction_planning_limits(resolve_profile("text", "none", "intermediate")).output_limit
        == 32_768
    )
    # A cell resolving to a whole-Gemini group keeps the defaults untouched
    # (correction-mm falls through to the default preset here). Equality, not
    # identity: `planning_limits_for` used to short-circuit and hand back the
    # singleton whenever the group cleared the harness ceilings, and that
    # early return is what this rewrite removed. The values still match --
    # free Gemini declares no `context_window`, so its envelope is its own
    # `max_input_tokens`, which is where the default came from.
    assert (
        correction_planning_limits(resolve_profile("audio", "local", "quality"))
        == DEFAULT_LIMITS
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


def test_client_tells_the_api_call_a_next_turn_will_follow(monkeypatch, tmp_path) -> None:
    """The request turn and a preparation t1 announce their next turn; the
    transport whose cache is opt-in (Anthropic) needs to know before it sends."""

    from finesub.llm import llm_runtime

    routes = load_model_routes(user_config=_user_config(), catalog=_catalog(tmp_path))
    seen = {}

    def fake_chat_complete(messages, **kwargs):
        seen.update(kwargs)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
        }

    monkeypatch.setattr(llm_runtime, "chat_complete", fake_chat_complete)
    client = RoleClient(
        router=ModelRouter(routes),
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
        role_configs={
            LLMRole.AUDIO_MULTIMODAL: role_config_for("correction-text", "quality", routes=routes)
        },
    )

    client.complete(LLMRole.AUDIO_MULTIMODAL, [{"role": "user", "content": "hi"}], followup_expected=True)
    assert seen["followup_expected"] is True
    client.complete(LLMRole.AUDIO_MULTIMODAL, [{"role": "user", "content": "hi"}])
    assert seen["followup_expected"] is False


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

_CUSTOM_CATALOG = """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
rpm = 30
tpm = 1000000
"""


@pytest.fixture
def custom_provider_config(tmp_path: Path, monkeypatch):
    """Point the whole process at a data root declaring one custom model.

    Both halves, as a user writes them: the fact in the data root's
    ``model_catalog.toml``, the composition in ``config.toml`` next to it.
    """

    from finesub import config as app_config
    from finesub.llm.routing.model_catalog import default_model_catalog
    from finesub.llm.routing.model_routes import default_model_routes

    (tmp_path / "config.toml").write_text(_CUSTOM_CONFIG, encoding="utf-8")
    (tmp_path / "model_catalog.toml").write_text(_CUSTOM_CATALOG, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(tmp_path / "config.toml"))
    monkeypatch.setenv("FINESUB_MODEL_CATALOG", str(tmp_path / "model_catalog.toml"))
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
    custom_provider_config, reported
) -> None:
    """§5.4's warnings exist for user presets; emitting them for the hard-coded
    default (which is warning-free by test) made the mechanism dead code."""

    from finesub.llm.routing.capabilities import validate_profile_capabilities
    from finesub.llm.routing.profiles import resolve_profile

    validate_profile_capabilities(
        resolve_profile("text", "none", "quality"), test_profile=False
    )

    warnings = reported.joined()
    assert "ds-flash" in warnings
    assert "规划包络" in warnings  # envelope: 128k < 194k baseline
    # `ds-flash` states no `capability`, and since 2026-09-04 that is a
    # note rather than a floor warning: an unstated score is no claim, so it
    # passes every floor and says so once in the log.
    assert "低于下限" not in warnings
    notes = " | ".join(message for message, _fields in reported.debugs)
    assert "未声明能力档" in notes and "ds-flash" in notes


def test_catalog_row_thinking_defaults_to_the_identity_mapping(tmp_path) -> None:
    """A row that says nothing about thinking gets the abstract levels
    verbatim (all three dialects spell them the same); ``false`` opts out."""

    from finesub.llm.routing.model_catalog import thinking_value_for

    catalog = _catalog(
        tmp_path,
        """
[models.ident]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000

[models.off]
provider_tier = "anthro"
provider_kind = "anthropic"
base_url = "https://api.anthropic.com"
api_model_id = "claude-sonnet-5"
max_input_tokens = 200000
thinking = false
""",
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
            """
[models.a]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000

[models.b]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
""",
        )


def test_token_scale_corrects_the_local_estimate_only(monkeypatch, tmp_path) -> None:
    """Plan v2 D14: the 3-tier counter speaks Gemini's vocabulary, so another
    model's estimate is systematically off. ``token_scale`` corrects it where
    estimates are actually used -- the input-limit check and the TPM
    reservation -- and nowhere else: reported usage stays the provider's."""

    from finesub.llm.routing.model_router import ModelRouter

    catalog = _catalog(
        tmp_path,
        """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
token_scale = 1.5
""",
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
        """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 200000
max_output_tokens = 8000
token_scale = 1.5
""",
    )
    routes = load_model_routes(user_config=_user_config(), catalog=catalog)
    monkeypatch.setattr(model_routes_module, "default_model_routes", lambda: routes)

    limits = correction_planning_limits(resolve_profile("text", "none", "quality"))

    # 200000 / 1.5: independent pools (no `context_window`), converted into
    # the local estimate units the planner counts in.
    assert limits.prompt_input_limit == 133_333
    # ...and that is what the dispatch check will accept once scaled back up.
    assert limits.prompt_input_limit * 1.5 <= 200_000


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
        """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 194000
token_scale = 0.5
""",
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
                """
[models.a]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
key_env = "KEY_A"
api_model_id = "model-a"
max_input_tokens = 128000

[models.b]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
key_env = "KEY_B"
api_model_id = "model-b"
max_input_tokens = 128000
""",
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

    path = tmp_path / "model_catalog.toml"
    path.write_text(
        """
[models.ds]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
supports_audio = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="supports_audio = true is not available"):
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
        """
[models.ds-flash]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 32000
max_output_tokens = 8000
""",
    )
    user = _user_config()
    user["presets"]["mine"]["bindings"]["knowledge/quality"] = "my-corr"
    routes = load_model_routes(user_config=user, catalog=catalog)
    monkeypatch.setattr(model_routes_module, "default_model_routes", lambda: routes)

    limits = planning_limits_for("knowledge")

    # Independent pools again (no `context_window` in the row).
    assert limits.prompt_input_limit == 32_000
    assert limits.prompt_input_limit < DEFAULT_LIMITS.prompt_input_limit
    # A whole-Gemini knowledge binding is untouched (equality, not identity --
    # see the note in test_correction_planning_limits_follow_the_group_envelope).
    assert planning_limits_for("knowledge", routes=load_model_routes()) == DEFAULT_LIMITS
