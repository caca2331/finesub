from __future__ import annotations

from pathlib import Path

import pytest

from llm.client import (
    LiteLLMRoleClient,
    UploadedFileRef,
    _as_tiered,
    _to_plain_response,
    _upload_gemini_file_rest,
    extract_finish_reason,
    extract_token_distribution,
    is_likely_output_limited,
    is_prompt_blocked,
    sum_token_distributions,
)
from llm.config import (
    GEMINI_FREE_TIER,
    CapabilityTier,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
)
from llm.llm_runtime import _convert_content_parts, _thinking_config
from llm.rate_limit import ModelRateLimiter


class FakeResponse:
    def __init__(self, *, headers=None, payload=None, status_code: int = 200) -> None:
        self.headers = headers or {}
        self._payload = payload or {}
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class FakeHttpClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __init__(self, *, timeout: float, probe_not_ready_times: int = 0) -> None:
        self.timeout = timeout
        self.posts = []
        self.gets = []
        self.probe_not_ready_times = probe_not_ready_times
        self.probe_calls = 0

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if ":countTokens" in url:
            self.probe_calls += 1
            if self.probe_calls <= self.probe_not_ready_times:
                return FakeResponse(status_code=400)
            return FakeResponse(payload={"totalTokens": 4242})
        if len(self.posts) == 1:
            return FakeResponse(headers={"x-goog-upload-url": "https://upload.test/session"})
        return FakeResponse(
            payload={
                "file": {
                    "name": "files/yui",
                    "uri": "https://generativelanguage.googleapis.com/v1beta/files/yui",
                    "mimeType": "audio/mpeg",
                    "state": "ACTIVE",
                }
            }
        )

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return FakeResponse(payload={"name": "files/yui", "state": "ACTIVE"})


def test_upload_gemini_file_rest_uses_resumable_protocol(tmp_path) -> None:
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake audio")
    clients = []

    def client_factory(**kwargs):
        client = FakeHttpClient(**kwargs)
        clients.append(client)
        return client

    ref = _upload_gemini_file_rest(
        Path(audio),
        api_key="test-key",
        client_factory=client_factory,
        sleep_func=lambda _: None,
    )

    client = clients[0]
    assert ref.file_id == "https://generativelanguage.googleapis.com/v1beta/files/yui"
    assert ref.mime_type == "audio/mpeg"
    assert client.posts[0][1]["headers"]["X-Goog-Upload-Protocol"] == "resumable"
    assert client.posts[1][1]["headers"]["X-Goog-Upload-Command"] == "upload, finalize"
    assert client.posts[1][1]["content"] == b"fake audio"
    # ACTIVE alone is not trusted: the free countTokens probe must confirm the
    # media is actually countable before the ref is returned.
    assert client.probe_calls == 1
    assert ":countTokens" in client.posts[-1][0]


def test_is_prompt_blocked_matches_content_filter_signature() -> None:
    # HTTP 200 + empty content + finish_reason=content_filter is Gemini's
    # promptFeedback.blockReason=PROHIBITED_CONTENT signature (2026-07-11).
    blocked = {"choices": [{"finish_reason": "content_filter", "message": {}}]}
    normal = {"choices": [{"finish_reason": "stop", "message": {}}]}

    assert extract_finish_reason(blocked) == "content_filter"
    assert is_prompt_blocked(None, blocked)
    assert is_prompt_blocked("", blocked)
    # Non-empty content means the call produced output — not a prompt block.
    assert not is_prompt_blocked("some text", blocked)
    # Ordinary empty responses (e.g. transient) are not classified as blocked.
    assert not is_prompt_blocked("", normal)
    assert not is_prompt_blocked("", {})


def test_upload_gemini_file_rest_waits_until_media_countable(tmp_path) -> None:
    # generateContent right after upload has returned empty output while the
    # file was ACTIVE but not yet sampled; the probe must poll through that.
    audio = tmp_path / "clip.mp4"
    audio.write_bytes(b"fake video")
    clients = []

    def client_factory(**kwargs):
        client = FakeHttpClient(probe_not_ready_times=3, **kwargs)
        clients.append(client)
        return client

    ref = _upload_gemini_file_rest(
        Path(audio),
        api_key="test-key",
        client_factory=client_factory,
        sleep_func=lambda _: None,
    )

    assert ref.file_id.endswith("/files/yui")
    assert clients[0].probe_calls == 4  # 3 not-ready responses + 1 success


def test_complete_routes_file_and_thinking_through_rest(monkeypatch) -> None:
    captured: dict = {}

    def fake_chat_complete(
        messages,
        *,
        model,
        thinking_budget=None,
        thinking_level=None,
        temperature=0.0,
        seed=None,
        max_tokens=None,
        retries=2,
        native_search_tool=None,
        **kwargs,
    ):
        captured["messages"] = messages
        captured["model"] = model
        captured["thinking_level"] = thinking_level
        captured["thinking_budget"] = thinking_budget
        captured["temperature"] = temperature
        captured["seed"] = seed
        captured["max_tokens"] = max_tokens
        captured["retries"] = retries
        return {
            "choices": [
                {"message": {"content": "<translated>ok</translated>"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
            "id": "resp-1",
        }

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)

    client = LiteLLMRoleClient(
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    ref = UploadedFileRef(
        file_id="https://generativelanguage.googleapis.com/v1beta/files/x",
        filename="x.flac",
        mime_type="audio/flac",
    )
    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        max_tokens=512,
        file_ref=ref,
    )

    assert result.content == "<translated>ok</translated>"
    assert result.model == "gemini/gemini-3.6-flash"
    assert result.fallback_used is False
    # AUDIO_MULTIMODAL role config carries thinking_level="medium".
    assert captured["thinking_level"] == "medium"
    assert captured["temperature"] == 1.0
    assert captured["seed"] is None
    assert captured["max_tokens"] == 512
    assert captured["retries"] == 0
    # The audio file is attached to the last user message as an OpenAI file part.
    user_content = captured["messages"][-1]["content"]
    file_parts = [
        part
        for part in user_content
        if isinstance(part, dict) and part.get("type") == "file"
    ]
    assert file_parts and file_parts[0]["file"]["file_id"] == ref.file_id
    assert file_parts[0]["file"]["format"] == "audio/flac"
    # Raw response is normalized to a plain dict for downstream artifacts.
    assert isinstance(result.raw_response, dict)


def test_complete_text_only_passes_medium_thinking_level(monkeypatch) -> None:
    captured: dict = {}

    def fake_chat_complete(messages, *, model, thinking_level=None, **kwargs):
        captured["model"] = model
        captured["thinking_level"] = thinking_level
        captured["has_file"] = any(
            isinstance(part, dict) and part.get("type") == "file"
            for msg in messages
            for part in (msg["content"] if isinstance(msg["content"], list) else [])
        )
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)

    client = LiteLLMRoleClient(rate_limiter=ModelRateLimiter(enabled=False))
    client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert captured["model"] == "gemini/gemini-3.5-flash"
    assert captured["thinking_level"] == "medium"
    assert captured["has_file"] is False


def test_complete_falls_back_on_quota_error(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_chat_complete(messages, *, model, provider_tier=None, **kwargs):
        calls.append(model)
        # Both capable free models are exhausted; the loop falls back to the
        # first BASIC endpoint (3.5 Flash Lite).
        if model in ("gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"):
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    client = LiteLLMRoleClient(max_retries=0, rate_limiter=limiter)
    result = client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert calls[0] == "gemini/gemini-3.5-flash"
    assert result.model == "gemini/gemini-3.5-flash-lite"
    assert result.fallback_used is True
    # Fixed-list callers still get the answering endpoint's tier reported.
    assert result.capability_tier is CapabilityTier.BASIC
    assert not limiter.is_daily_exhausted(
        ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    )


def test_as_tiered_wraps_lists_and_passes_factories() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    wrapped = _as_tiered(msgs)
    assert wrapped(CapabilityTier.CAPABLE) is msgs
    assert wrapped(CapabilityTier.BASIC) is msgs

    def factory(tier):
        return [{"role": "user", "content": tier.value}]

    assert _as_tiered(factory) is factory


def test_complete_assembles_prompt_for_fallback_tier(monkeypatch) -> None:
    # Both capable free models (cap 7) are rate-limited; the loop lands on the
    # first flash-lite (cap 5), which must receive the BASIC assembly and report
    # its tier back. The CAPABLE assembly is memoized across 3.5/3.6 Flash.
    factory_calls: list = []

    def factory(tier):
        factory_calls.append(tier)
        return [{"role": "user", "content": f"prompt-{tier.value}"}]

    seen: list[tuple[str, str]] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        seen.append((model, messages[0]["content"]))
        if model in ("gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"):
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    client = LiteLLMRoleClient(
        max_retries=0, rate_limiter=ModelRateLimiter(enabled=False)
    )
    result = client.complete(LLMRole.GENERAL_CAPABLE, factory)

    assert result.model == "gemini/gemini-3.5-flash-lite"
    assert result.fallback_used is True
    assert result.capability_tier is CapabilityTier.BASIC
    assert factory_calls == [CapabilityTier.CAPABLE, CapabilityTier.BASIC]
    assert seen == [
        ("gemini/gemini-3.5-flash", "prompt-capable"),
        ("gemini/gemini-3.6-flash", "prompt-capable"),
        ("gemini/gemini-3.5-flash-lite", "prompt-basic"),
    ]


def test_complete_memoizes_assembly_per_tier(monkeypatch) -> None:
    # All free endpoints fail; paid 3.5 Flash answers and must reuse the CAPABLE
    # assembly memoized from the free capable models — one factory call per
    # tier, not per endpoint.
    factory_calls: list = []

    def factory(tier):
        factory_calls.append(tier)
        return [{"role": "user", "content": tier.value}]

    def fake_chat_complete(messages, *, model, provider_tier=None, **kwargs):
        if provider_tier == GEMINI_FREE_TIER:
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    client = LiteLLMRoleClient(
        max_retries=0, rate_limiter=ModelRateLimiter(enabled=False)
    )
    result = client.complete(LLMRole.GENERAL_CAPABLE, factory)

    assert result.model == "gemini/gemini-3.5-flash"
    assert result.capability_tier is CapabilityTier.CAPABLE
    assert factory_calls == [CapabilityTier.CAPABLE, CapabilityTier.BASIC]


def test_complete_defaults_to_capable_without_catalog_entry(monkeypatch) -> None:
    endpoint = ModelEndpoint(GEMINI_FREE_TIER, "gemini/unknown-experimental")
    configs = {
        LLMRole.GENERAL_CAPABLE: RoleModelConfig(
            role=LLMRole.GENERAL_CAPABLE,
            endpoint_chain=(endpoint,),
            test_endpoint=endpoint,
        )
    }
    tiers: list = []

    def factory(tier):
        tiers.append(tier)
        return [{"role": "user", "content": "hi"}]

    def fake_chat_complete(messages, **kwargs):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    client = LiteLLMRoleClient(
        role_configs=configs,
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    result = client.complete(LLMRole.GENERAL_CAPABLE, factory)

    assert tiers == [CapabilityTier.CAPABLE]
    assert result.capability_tier is CapabilityTier.CAPABLE


def test_complete_daily_quota_records_strike_but_does_not_immediately_lock(
    monkeypatch, tmp_path
) -> None:
    # A single explicit per-day 429 must NOT lock the day — Gemini's free-tier
    # PerDay signal flickers, so locking requires the sustained strike gate.
    def fake_chat_complete(messages, *, model, **kwargs):
        if model == "gemini/gemini-3.5-flash":
            raise RuntimeError("quota exceeded: generate requests per day")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    client = LiteLLMRoleClient(max_retries=0, rate_limiter=limiter)
    client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert not limiter.is_daily_exhausted(
        ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    )


def test_rate_limiter_locks_daily_only_after_sustained_strikes(tmp_path) -> None:
    from llm.rate_limit import DAILY_STRIKE_SPAN_SECONDS

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    # Three hits inside a burst (<5 min span) do NOT lock.
    assert not limiter.note_daily_quota_hit(ep, now=1000.0)
    assert not limiter.note_daily_quota_hit(ep, now=1005.0)
    assert not limiter.note_daily_quota_hit(ep, now=1010.0)
    assert not limiter.is_daily_exhausted(ep)

    # A success clears the streak (strikes must be consecutive failures).
    limiter.reset_daily_strikes(ep)

    # Three hits spread over >=5 min DO lock.
    assert not limiter.note_daily_quota_hit(ep, now=2000.0)
    assert not limiter.note_daily_quota_hit(ep, now=2150.0)
    assert limiter.note_daily_quota_hit(
        ep, now=2000.0 + DAILY_STRIKE_SPAN_SECONDS
    )
    assert limiter.is_daily_exhausted(ep)


def test_chat_complete_passes_messages_through_unmodified(monkeypatch) -> None:
    """v17: every prompt template mandates the opening <reasoning> block, so
    the runtime never rewrites messages (the old non-thinking-model injection
    is gone)."""

    from llm import llm_runtime

    captured: dict = {}

    def fake_completion(**kwargs):
        captured["messages"] = kwargs["messages"]
        captured["model"] = kwargs["model"]
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    messages = [{"role": "user", "content": "hi"}]
    llm_runtime.chat_complete(
        messages,
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        thinking_level="medium",
        retries=0,
    )

    assert captured["model"] == "gemini/gemini-3.1-flash-lite"
    assert captured["messages"] == messages


def test_chat_complete_does_not_retry_invalid_request(monkeypatch) -> None:
    from llm import llm_runtime

    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        raise ValueError("invalid prompt schema")

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    with pytest.raises(ValueError, match="invalid prompt schema"):
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.1-flash-lite",
            retries=5,
        )
    assert calls == 1


def test_chat_complete_records_api_attempts(monkeypatch) -> None:
    from llm import llm_runtime

    calls = {"count": 0}

    class RateLimitError(RuntimeError):
        status_code = 429

    def fake_completion(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RateLimitError("HTTP 429 too many requests")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    response = llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=1,
    )

    attempts = response["_harness_api_attempts"]
    assert [attempt["return_code"] for attempt in attempts] == ["429", "200"]
    assert attempts[0]["provider_tier"] == "GEMINI_FREE"
    assert attempts[0]["api_key_name"] == "free-main"
    assert attempts[1]["call_number_for_api_key_and_model"] == 2


def test_chat_complete_sets_fifteen_minute_timeout(monkeypatch) -> None:
    from llm import llm_runtime

    captured: dict = {}

    def fake_completion(**kwargs):
        captured["timeout"] = kwargs["timeout"]
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=0,
    )

    assert captured["timeout"] == 15 * 60


def test_chat_complete_aborts_after_two_consecutive_timeouts(monkeypatch) -> None:
    from llm import llm_runtime

    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("request timed out")

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with pytest.raises(TimeoutError) as raised:
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.5-flash-lite",
            retries=7,
        )

    assert calls == 2
    assert getattr(raised.value, "_harness_consecutive_timeout_abort") is True
    attempts = getattr(raised.value, "_harness_api_attempts")
    assert [item["return_code"] for item in attempts] == [
        "NO_RESPONSE_TIMEOUT",
        "NO_RESPONSE_TIMEOUT",
    ]


def test_complete_does_not_fallback_after_consecutive_timeout_abort(monkeypatch) -> None:
    calls: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        calls.append(model)
        exc = TimeoutError("request timed out twice")
        setattr(exc, "_harness_consecutive_timeout_abort", True)
        raise exc

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    client = LiteLLMRoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    with pytest.raises(TimeoutError):
        client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert calls == ["gemini/gemini-3.5-flash"]


def test_thinking_config_prefers_level_for_gemini3() -> None:
    assert _thinking_config("gemini/gemini-3.5-flash", 800, "medium") == {
        "thinkingLevel": "medium"
    }
    assert _thinking_config("gemini/gemini-3.5-flash", 1600, None) == {
        "thinkingLevel": "high"
    }
    assert _thinking_config("gemini/gemini-3.1-flash-lite", 800, None) == {
        "thinkingLevel": "low"
    }
    assert _thinking_config("gemini/gemini-3.1-flash-lite", 0, None) == {
        "thinkingLevel": "minimal"
    }
    # Gemini 2.5 has no thinkingLevel; budget drives thinkingBudget directly.
    assert _thinking_config("gemini/gemini-2.5-flash", 800, "medium") == {
        "thinkingBudget": 800
    }
    assert _thinking_config("gemini/gemini-2.5-flash", None, None) == {}


def test_convert_content_parts_maps_video_file_block() -> None:
    block = {
        "type": "file",
        "file": {
            "file_id": "files/abc",
            "format": "video/mp4",
            "detail": "low",
            "video_metadata": {"fps": 0.25},
        },
    }
    part = _convert_content_parts([block])[0]
    assert part["fileData"] == {"fileUri": "files/abc", "mimeType": "video/mp4"}
    assert part["videoMetadata"] == {"fps": 0.25}
    # detail="low" must survive as per-part mediaResolution.level so mm-high
    # clips keep the planned low-resolution frame-token billing.
    assert part["mediaResolution"] == {"level": "MEDIA_RESOLUTION_LOW"}


def test_convert_content_parts_audio_has_no_media_resolution() -> None:
    block = {"type": "file", "file": {"file_id": "files/a", "format": "audio/wav"}}
    part = _convert_content_parts([block])[0]
    assert part["fileData"] == {"fileUri": "files/a", "mimeType": "audio/wav"}
    assert "mediaResolution" not in part
    assert "videoMetadata" not in part


def test_to_plain_response_normalizes_pydantic_like() -> None:
    class FakeModelResponse:
        def model_dump(self):
            return {"usage": {"prompt_tokens": 5}}

    assert _to_plain_response(FakeModelResponse()) == {"usage": {"prompt_tokens": 5}}
    assert _to_plain_response({"a": 1}) == {"a": 1}

    class NoDump:
        pass

    assert _to_plain_response(NoDump()) == {}


def test_is_likely_output_limited_uses_usage_metadata() -> None:
    assert is_likely_output_limited(
        {"usageMetadata": {"candidatesTokenCount": 65_500}}, max_tokens=65_536
    )
    assert is_likely_output_limited(
        {"usageMetadata": {"candidatesTokenCount": 32_000, "thoughtsTokenCount": 33_500}},
        max_tokens=65_536,
    )
    assert not is_likely_output_limited(
        {"usageMetadata": {"candidatesTokenCount": 1_000}}, max_tokens=65_536
    )
    assert not is_likely_output_limited({}, max_tokens=65_536)


def test_extract_token_distribution_splits_modalities_and_thinking() -> None:
    rest = {
        "usageMetadata": {
            "promptTokenCount": 89_196,
            "promptTokensDetails": [
                {"modality": "AUDIO", "tokenCount": 65_971},
                {"modality": "TEXT", "tokenCount": 23_225},
            ],
            "candidatesTokenCount": 7_023,
            "thoughtsTokenCount": 33_543,
            "totalTokenCount": 129_762,
        }
    }
    dist = extract_token_distribution(rest)
    assert dist == {
        "prompt_tokens": 89_196,
        "prompt_text_tokens": 23_225,
        "prompt_audio_tokens": 65_971,
        "uncached_input_tokens": 89_196,
        "cached_input_tokens": 0,
        "total_input_tokens": 89_196,
        "thinking_tokens": 33_543,
        "output_tokens": 7_023,
        "total_output_tokens": 40_566,
        "total_tokens": 129_762,
    }

    cached_rest = {
        "usageMetadata": {
            "promptTokenCount": 10_000,
            "cachedContentTokenCount": 4_000,
            "candidatesTokenCount": 100,
            "thoughtsTokenCount": 50,
            "totalTokenCount": 10_150,
        }
    }
    cached_dist = extract_token_distribution(cached_rest)
    assert cached_dist["cached_input_tokens"] == 4_000
    assert cached_dist["uncached_input_tokens"] == 6_000
    assert cached_dist["total_output_tokens"] == 150

    # OpenAI/litellm shape: completion_tokens includes reasoning tokens, and the
    # audio split arrives as prompt_tokens_details.audio_tokens (dict).
    litellm_style = {
        "usage": {
            "prompt_tokens": 58,
            "completion_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 143},
            "prompt_tokens_details": {"audio_tokens": 26, "text_tokens": 32},
            "total_tokens": 208,
        }
    }
    dist2 = extract_token_distribution(litellm_style)
    assert dist2["prompt_audio_tokens"] == 26
    assert dist2["prompt_text_tokens"] == 58 - 26
    assert dist2["thinking_tokens"] == 143
    assert dist2["output_tokens"] == 7

    # Plain OpenAI usage without a modality split keeps all prompt tokens as text.
    openai_style = {
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 4_436,
            "completion_tokens_details": {"reasoning_tokens": 4_307},
            "total_tokens": 14_436,
        }
    }
    dist3 = extract_token_distribution(openai_style)
    assert dist3["prompt_text_tokens"] == 10_000
    assert dist3["thinking_tokens"] == 4_307
    assert dist3["output_tokens"] == 129
    assert extract_token_distribution({})["total_tokens"] == 0

    totals = sum_token_distributions([dist, dist2, dist3])
    assert totals["prompt_audio_tokens"] == 65_971 + 26
    assert totals["uncached_input_tokens"] == 89_196 + 58 + 10_000
    assert totals["call_count"] == 3


def test_llm_exchange_metadata_includes_token_breakdown() -> None:
    from llm.client import LLMCallResult, llm_exchange_metadata

    result = LLMCallResult(
        content="ok",
        role=LLMRole.GENERAL_CAPABLE,
        model="gemini/gemini-3.5-flash",
        fallback_used=False,
        raw_response={
            "usageMetadata": {
                "promptTokenCount": 100,
                "cachedContentTokenCount": 20,
                "candidatesTokenCount": 10,
                "thoughtsTokenCount": 5,
            }
        },
        api_key_label="free-main",
        thinking_level="medium",
        thinking_budget=800,
    )
    meta = llm_exchange_metadata(result, attempt=0)
    assert meta["prompt_version"]
    from llm.prompt_compose import PROMPT_VERSION

    assert meta["prompt_version"] == PROMPT_VERSION
    assert meta["thinking_level"] == "medium"
    assert meta["input_tokens"] == "80 / 20 / 100 (uncached / cached / total)"
    assert meta["output_tokens_breakdown"] == "10 / 5 / 15 (visible / thinking / total)"
    assert "provider_tier" not in meta
    assert "model" not in meta
    assert "api_key" not in meta
    assert "uncached_input_tokens" not in meta
    assert "cached_input_tokens" not in meta
    assert "total_input_tokens" not in meta
    assert "thinking_tokens" not in meta
    assert "output_tokens" not in meta
    assert "total_output_tokens" not in meta


# ---------------------------------------------------------------------------
# 429 retry-after parser
# ---------------------------------------------------------------------------


def test_parse_retry_after_seconds_gemini_retry_delay() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    # Gemini structured retryDelay in error JSON text
    exc = RuntimeError(
        'RateLimitError: Quota exceeded. "retryDelay": "51.980s". '
        'quotaId "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"'
    )
    assert parse_retry_after_seconds(exc) == pytest.approx(51.98)


def test_parse_retry_after_seconds_gemini_please_retry_in() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError(
        "RateLimitError: Quota exceeded for metric generate_content_free_tier_"
        "requests, limit 20. Please retry in 12.5s."
    )
    assert parse_retry_after_seconds(exc) == pytest.approx(12.5)


def test_parse_retry_after_seconds_openai_retry_after_header() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError("Error code: 429 - Retry-After: 30")
    assert parse_retry_after_seconds(exc) == pytest.approx(30.0)


def test_parse_retry_after_seconds_anthropic_retry_after() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError('{"type":"error","retry_after":45,"error":{"type":"rate_limit_error"}}')
    assert parse_retry_after_seconds(exc) == pytest.approx(45.0)


def test_parse_retry_after_seconds_generic_wait() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    assert parse_retry_after_seconds(
        RuntimeError("rate limited, wait 60 seconds before retrying")
    ) == pytest.approx(60.0)
    assert parse_retry_after_seconds(
        RuntimeError("try again in 25s")
    ) == pytest.approx(25.0)


def test_parse_retry_after_seconds_no_hint_returns_zero() -> None:
    from llm.rate_limit import parse_retry_after_seconds

    assert parse_retry_after_seconds(RuntimeError("HTTP 429 too many requests")) == 0.0
    assert parse_retry_after_seconds(RuntimeError("HTTP 503 unavailable")) == 0.0


# ---------------------------------------------------------------------------
# key_id_for_secret
# ---------------------------------------------------------------------------


def test_key_id_for_secret_is_stable_and_non_reversible() -> None:
    from llm.rate_limit import key_id_for_secret

    kid = key_id_for_secret("AIzaSyD-test-key-12345")
    assert kid.startswith("sha256:")
    assert len(kid) == len("sha256:") + 12
    # Deterministic
    assert key_id_for_secret("AIzaSyD-test-key-12345") == kid
    # Different keys produce different ids
    assert key_id_for_secret("AIzaSyD-other-key") != kid
    # Raw key not recoverable from the id
    assert "AIzaSyD" not in kid


# ---------------------------------------------------------------------------
# Per-key daily accounting
# ---------------------------------------------------------------------------


def test_per_key_daily_accounting_isolates_keys(tmp_path) -> None:
    """A daily lock on key A must not poison key B for the same endpoint."""
    from llm.rate_limit import DAILY_STRIKE_SPAN_SECONDS

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    # Sustained strikes on key-a lock only key-a.
    assert not limiter.note_daily_quota_hit(ep, key_id="key-a", now=1000.0)
    assert not limiter.note_daily_quota_hit(ep, key_id="key-a", now=1150.0)
    assert limiter.note_daily_quota_hit(
        ep, key_id="key-a", now=1000.0 + DAILY_STRIKE_SPAN_SECONDS
    )
    assert limiter.is_daily_exhausted(ep, key_id="key-a")
    # key-b is unaffected.
    assert not limiter.is_daily_exhausted(ep, key_id="key-b")
    # Endpoint-level (key_id="") is also unaffected.
    assert not limiter.is_daily_exhausted(ep)


def test_per_key_strikes_reset_on_success(tmp_path) -> None:
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    limiter.note_daily_quota_hit(ep, key_id="k1", now=100.0)
    limiter.note_daily_quota_hit(ep, key_id="k1", now=200.0)
    # Success resets the streak.
    limiter.reset_daily_strikes(ep, key_id="k1")
    # Two more hits alone don't reach the threshold of 3.
    assert not limiter.note_daily_quota_hit(ep, key_id="k1", now=500.0)
    assert not limiter.note_daily_quota_hit(ep, key_id="k1", now=900.0)
    assert not limiter.is_daily_exhausted(ep, key_id="k1")


# ---------------------------------------------------------------------------
# Backoff formula in chat_complete
# ---------------------------------------------------------------------------


def test_chat_complete_backoff_uses_provider_hint_clamped(monkeypatch) -> None:
    """Backoff = min(max(exponential, provider_hint), 300) + 1."""
    from llm import llm_runtime

    sleeps: list[float] = []
    calls = {"count": 0}

    class RateLimitError(RuntimeError):
        status_code = 429

    def fake_completion(**kwargs):
        calls["count"] += 1
        if calls["count"] <= 3:
            raise RateLimitError(
                'HTTP 429 "retryDelay": "20s" quotaId "PerMinute"'
            )
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda s: sleeps.append(s))

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=5,
    )

    # attempt 0: max(0.5, 20) = 20 -> min(20, 300) + 1 = 21
    # attempt 1: max(1.0, 20) = 20 -> min(20, 300) + 1 = 21
    # attempt 2: max(2.0, 20) = 20 -> min(20, 300) + 1 = 21
    assert sleeps == [21.0, 21.0, 21.0]


def test_chat_complete_backoff_caps_at_300_plus_1(monkeypatch) -> None:
    from llm import llm_runtime

    sleeps: list[float] = []
    calls = {"count": 0}

    class RateLimitError(RuntimeError):
        status_code = 429

    def fake_completion(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RateLimitError("HTTP 429 retry after 9999 seconds")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}"},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda s: sleeps.append(s))

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=2,
    )

    # Provider says 9999s but cap is 300: min(max(0.5, 9999), 300) + 1 = 301
    assert sleeps == [301.0]
