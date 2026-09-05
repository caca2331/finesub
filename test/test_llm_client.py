from __future__ import annotations

from pathlib import Path

from datetime import datetime, timedelta, timezone

import pytest

from finesub.llm.client import (
    RoleClient,
    _as_tiered,
    _to_plain_response,
    extract_finish_reason,
    extract_token_distribution,
    is_likely_output_limited,
    is_prompt_blocked,
    sum_token_distributions,
)
from finesub.llm.media_upload import (
    UploadCancelled,
    UploadedFileRef,
    _upload_gemini_file_rest,
)
from finesub.llm.routing.config import (
    GEMINI_FREE_TIER,
    GEMINI_PAID_TIER,
    CapabilityTier,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
)
from finesub.llm.llm_runtime import _convert_content_parts, _thinking_config
from finesub.llm.rate_limit import (
    COMBO_COOLDOWN_TTL_SECONDS,
    ComboCooldownPhase,
    ModelRateLimiter,
    endpoint_key,
)


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

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)

    client = RoleClient(
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
    assert result.model == "gemini/gemini-3.7-flash"
    assert result.fallback_used is False
    # AUDIO_MULTIMODAL role config carries the *abstract* thinking_level
    # "medium", and 3.7 Flash maps identity, so "medium" is what goes out.
    # The non-identity mapping is guarded where it is declared, in
    # test_llm_config_and_budget's catalog tests.
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

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)

    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))
    client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert captured["model"] == "gemini/gemini-3.6-flash"
    # research/high thinks high (owner tuning 2026-08-12).
    assert captured["thinking_level"] == "high"
    assert captured["has_file"] is False


def test_complete_falls_back_on_quota_error(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def fake_chat_complete(messages, *, model, provider_tier=None, **kwargs):
        calls.append((provider_tier, model))
        # The free tier is exhausted; the loop falls through to the paid tail.
        if provider_tier == GEMINI_FREE_TIER:
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    client = RoleClient(max_retries=0, rate_limiter=limiter)
    result = client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert calls[0] == (GEMINI_FREE_TIER, "gemini/gemini-3.6-flash")
    assert result.target_id == "gemini-paid-3_7-flash"
    assert result.fallback_used is True
    # v2 (D2): the reported tier follows the served *variant* -- the research
    # cell is single-template ("").
    assert result.capability_tier is CapabilityTier.CAPABLE
    assert result.variant == ""
    assert not limiter.is_daily_exhausted(
        ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    )


def test_as_tiered_wraps_lists_and_passes_factories() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    wrapped = _as_tiered(msgs)
    assert wrapped("capableC") is msgs
    assert wrapped("basicB") is msgs

    def factory(variant):
        return [{"role": "user", "content": variant}]

    assert _as_tiered(factory) is factory


def test_complete_assembles_prompt_per_entry_variant_override(monkeypatch) -> None:
    # Plan v2 D3: a model-group entry may override the cell's variant; the
    # factory is called with the candidate's effective variant name and the
    # answering candidate reports it back.
    strong = ModelEndpoint(
        GEMINI_FREE_TIER, "gemini/gemini-3.6-flash", target_id="strong"
    )
    lite = ModelEndpoint(
        GEMINI_FREE_TIER, "gemini/gemini-3.5-flash-lite", target_id="lite"
    )
    configs = {
        LLMRole.AUDIO_MULTIMODAL: RoleModelConfig(
            role=LLMRole.AUDIO_MULTIMODAL,
            endpoint_chain=(strong, lite),
            test_endpoint=strong,
            variant="capableC",
            variant_overrides={"lite": "basicB"},
        )
    }
    factory_calls: list = []

    def factory(variant):
        factory_calls.append(variant)
        return [{"role": "user", "content": f"prompt-{variant}"}]

    seen: list[tuple[str, str]] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        seen.append((model, messages[0]["content"]))
        if model == "gemini/gemini-3.6-flash":
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        role_configs=configs,
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    result = client.complete(LLMRole.AUDIO_MULTIMODAL, factory)

    assert result.model == "gemini/gemini-3.5-flash-lite"
    assert result.fallback_used is True
    assert result.variant == "basicB"
    assert result.capability_tier is CapabilityTier.BASIC
    assert factory_calls == ["capableC", "basicB"]
    assert seen == [
        ("gemini/gemini-3.6-flash", "prompt-capableC"),
        ("gemini/gemini-3.5-flash-lite", "prompt-basicB"),
    ]


def test_complete_memoizes_assembly_per_variant(monkeypatch) -> None:
    # All free endpoints fail; paid 3.7 Flash answers and must reuse the one
    # assembly memoized across the group -- one factory call per variant
    # (the research cell is single-template), not per endpoint.
    factory_calls: list = []

    def factory(variant):
        factory_calls.append(variant)
        return [{"role": "user", "content": variant or "single"}]

    def fake_chat_complete(messages, *, model, provider_tier=None, **kwargs):
        if provider_tier == GEMINI_FREE_TIER:
            raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        max_retries=0, rate_limiter=ModelRateLimiter(enabled=False)
    )
    result = client.complete(LLMRole.GENERAL_CAPABLE, factory)

    # The paid tail of the research group leads with 3.8 Flash.
    assert result.model == "gemini/gemini-3.7-flash"
    assert result.capability_tier is CapabilityTier.CAPABLE
    assert factory_calls == [""]


def test_complete_explicit_unverified_fixture_defaults_to_capable(monkeypatch) -> None:
    endpoint = ModelEndpoint(
        GEMINI_FREE_TIER, "gemini/unknown-experimental", unverified=True
    )
    configs = {
        LLMRole.GENERAL_CAPABLE: RoleModelConfig(
            role=LLMRole.GENERAL_CAPABLE,
            endpoint_chain=(endpoint,),
            test_endpoint=endpoint,
        )
    }
    variants: list = []

    def factory(variant):
        variants.append(variant)
        return [{"role": "user", "content": "hi"}]

    def fake_chat_complete(messages, **kwargs):
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(
        role_configs=configs,
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    result = client.complete(LLMRole.GENERAL_CAPABLE, factory)

    assert variants == [""]
    assert result.capability_tier is CapabilityTier.CAPABLE


def test_complete_rejects_unknown_production_endpoint() -> None:
    endpoint = ModelEndpoint(GEMINI_FREE_TIER, "gemini/unknown-experimental")
    client = RoleClient(
        role_configs={
            LLMRole.GENERAL_CAPABLE: RoleModelConfig(
                role=LLMRole.GENERAL_CAPABLE,
                endpoint_chain=(endpoint,),
                test_endpoint=endpoint,
            )
        },
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
    )

    with pytest.raises(ValueError, match="no verified runtime fact"):
        client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])


def test_complete_daily_quota_records_strike_but_does_not_immediately_lock(
    monkeypatch, tmp_path
) -> None:
    # A single explicit per-day 429 must NOT lock the day — Gemini's free-tier
    # PerDay signal flickers, so locking requires the sustained strike gate.
    def fake_chat_complete(messages, *, model, **kwargs):
        if model == "gemini/gemini-3.5-flash":
            raise RuntimeError("quota exceeded: generate requests per day")
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    client = RoleClient(max_retries=0, rate_limiter=limiter)
    client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert not limiter.is_daily_exhausted(
        ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    )


def test_rate_limiter_locks_daily_only_after_sustained_strikes(tmp_path) -> None:
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    # One or two hits do NOT lock.
    assert not limiter.note_daily_quota_hit(ep, now=1000.0)
    assert not limiter.note_daily_quota_hit(ep, now=1005.0)
    assert not limiter.is_daily_exhausted(ep)

    # A success clears the streak (strikes must be consecutive failures).
    limiter.reset_daily_strikes(ep)

    # Three consecutive hits lock (no minimum time-span gate).
    assert not limiter.note_daily_quota_hit(ep, now=2000.0)
    assert not limiter.note_daily_quota_hit(ep, now=2001.0)
    assert limiter.note_daily_quota_hit(ep, now=2002.0)
    assert limiter.is_daily_exhausted(ep)


def test_chat_complete_passes_messages_through_unmodified(monkeypatch) -> None:
    """v17: every prompt template mandates the opening <reasoning> block, so
    the runtime never rewrites messages (the old non-thinking-model injection
    is gone)."""

    from finesub.llm import llm_runtime

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


def test_the_gemini_rest_endpoint_honours_a_configured_base(monkeypatch) -> None:
    """The two custom transports have always taken a base URL; this one did not.

    That asymmetry is the whole change: a machine that cannot reach Google's
    host directly could point every provider at a mirror except the one the
    packaged models use.
    """

    from finesub.llm import llm_runtime

    captured: dict = {}

    def fake_completion(**kwargs):
        captured["api_base"] = kwargs["api_base"]
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    def run(env_map):
        monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.1-flash-lite",
            thinking_level="medium",
            retries=0,
        )

    keys = {"GEMINI_FREE": "{free-main:key1}"}
    run(keys)
    assert captured["api_base"] == llm_runtime.GEMINI_API_BASE

    run({**keys, "GEMINI_BASE_URL": "  https://mirror.example/v1beta  "})
    assert captured["api_base"] == "https://mirror.example/v1beta"

    # An empty value is not a configuration; it must not blank the endpoint.
    run({**keys, "GEMINI_BASE_URL": ""})
    assert captured["api_base"] == llm_runtime.GEMINI_API_BASE


def test_the_gemini_rest_url_takes_the_base_it_is_given(monkeypatch) -> None:
    from finesub.llm import llm_runtime

    seen: dict = {}

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"candidates": []}

        @staticmethod
        def raise_for_status():
            return None

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            seen["url"] = url
            return _Response()

    # Patched at the module's own seam, not on `httpx` itself. Reaching
    # into `httpx.Client` used to work only because every module shared the
    # one module object -- it silently rerouted the whole process, this test
    # included in whatever else was running.
    monkeypatch.setattr(llm_runtime, "llm_http_client", _Client)
    llm_runtime._gemini_generate_content(
        model="gemini/gemini-3.1-flash-lite",
        messages=[{"role": "user", "content": "hi"}],
        api_key="key",
        temperature=0.0,
        safety_settings=None,
        thinking_config={},
        max_tokens=None,
        tools=None,
        timeout=1.0,
        # A trailing slash is what a person pasting a base URL actually types.
        api_base="https://mirror.example/v1beta/",
    )

    assert seen["url"] == (
        "https://mirror.example/v1beta/models/"
        "gemini-3.1-flash-lite:generateContent"
    )


def test_chat_complete_uses_configured_free_pool_order(monkeypatch) -> None:
    from finesub.llm.routing import api_keys
    from finesub.llm import llm_runtime

    captured: dict = {}

    def fake_completion(**kwargs):
        captured["api_key"] = kwargs["api_key"]
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{main:key1,spare:key2}"},
    )
    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda path=None: {"pools": {"gemini_free": ["spare", "main"]}},
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    response = llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=0,
    )

    assert captured["api_key"] == "key2"
    assert response["_harness_api_key_label"] == "spare"
    assert response["_harness_key_id"] == "spare"


def test_role_client_skips_disabled_free_provider(monkeypatch) -> None:
    from finesub.llm.routing import api_keys

    calls: list[str] = []

    def fake_chat_complete(messages, *, provider_tier, model, **kwargs):
        calls.append(provider_tier)
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
            "_harness_key_id": "paid-main",
        }

    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda path=None: {"providers": {"gemini_free": False}},
    )
    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
    )

    assert result.content == "ok"
    assert calls == [GEMINI_PAID_TIER]


def test_chat_complete_does_not_retry_invalid_request(monkeypatch) -> None:
    from finesub.llm import llm_runtime

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
    from finesub.llm import llm_runtime

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


def test_every_api_attempt_also_says_one_line_in_the_run_log(monkeypatch) -> None:
    """The ledger travels with the artifacts; the log is what a user sends.

    Same facts, said out loud -- and never the prompt or the answer, which are
    written per call under the task's `exchanges/`.
    """

    from finesub.llm import llm_runtime
    from finesub.reporting import NullReporter, reporting_to

    lines: list[dict] = []

    class _Debug(NullReporter):
        def debug(self, message, fields=None) -> None:
            if message == "llm api call":
                lines.append(dict(fields or {}))

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
        llm_runtime, "_read_dotenv", lambda: {"GEMINI_FREE": "{free-main:secret1}"}
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with reporting_to(_Debug()):
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.1-flash-lite",
            retries=1,
        )

    assert [entry["code"] for entry in lines] == ["429", "200"]
    # The endpoint's own words, not a sentence of ours: a bare 429 does not
    # separate "this key is spent today" from "slow down".
    assert lines[0]["why"] == "HTTP 429 too many requests"
    assert "why" not in lines[1], "a success has nothing to explain"
    assert lines[0]["model"] == "gemini/gemini-3.1-flash-lite"
    # A key *label*, never the key itself: this file is written to be sent.
    assert lines[0]["key"] == "free-main"
    assert not any("secret1" in str(value) for entry in lines for value in entry.values())
    assert not any("hi" == str(value) for entry in lines for value in entry.values())


def test_a_credentialed_url_in_an_endpoint_error_is_redacted(monkeypatch) -> None:
    """`[llm] proxy` and a custom `base_url` are the user's own addresses.

    `https://user:token@host` is an ordinary way to write one, so an httpx
    error quoting the request URL can carry a credential into the one file
    this log exists to be sent as.
    """

    from finesub.llm import llm_runtime
    from finesub.reporting import NullReporter, reporting_to

    lines: list[dict] = []

    class _Debug(NullReporter):
        def debug(self, message, fields=None) -> None:
            if message == "llm api call":
                lines.append(dict(fields or {}))

    def fail(**kwargs):
        raise RuntimeError(
            "ConnectError for https://carl:hunter2@proxy.example/v1/models"
        )

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime, "_read_dotenv", lambda: {"GEMINI_FREE": "{free-main:key1}"}
    )
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fail)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with reporting_to(_Debug()):
        with pytest.raises(Exception):
            llm_runtime.chat_complete(
                [{"role": "user", "content": "hi"}],
                provider_tier=GEMINI_FREE_TIER,
                model="gemini/gemini-3.1-flash-lite",
                retries=0,
            )

    assert lines
    assert all("hunter2" not in entry["why"] for entry in lines)
    assert all("carl" not in entry["why"] for entry in lines)
    # The host survives -- that is the part worth reading.
    assert all("proxy.example" in entry["why"] for entry in lines)


def test_chat_complete_sets_fifteen_minute_timeout(monkeypatch) -> None:
    from finesub.llm import llm_runtime

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
    from finesub.llm import llm_runtime

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

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    with pytest.raises(TimeoutError):
        client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert calls == ["gemini/gemini-3.6-flash"]


def test_thinking_config_prefers_level_for_gemini3() -> None:
    """...and asks for the thought summary wherever there is thinking to summarise.

    Thought summaries cost nothing extra (the thinking tokens are billed
    either way) and `extract_message_content` drops the parts they arrive in,
    so the answer is unaffected. Without them a thinking model's deliberation
    is a token count with no content behind it -- which is exactly what left
    the 2026-08-25 prompt iteration unable to say what a 2x difference in
    thinking was actually about.
    """

    assert _thinking_config("gemini/gemini-3.5-flash", 800, "medium") == {
        "thinkingLevel": "medium",
        "includeThoughts": True,
    }
    assert _thinking_config("gemini/gemini-3.5-flash", 1600, None) == {
        "thinkingLevel": "high",
        "includeThoughts": True,
    }
    assert _thinking_config("gemini/gemini-3.1-flash-lite", 800, None) == {
        "thinkingLevel": "low",
        "includeThoughts": True,
    }
    # Nothing to summarise when thinking is off; asking would be noise.
    assert _thinking_config("gemini/gemini-3.1-flash-lite", 0, None) == {
        "thinkingLevel": "minimal"
    }
    # Gemini 2.5 has no thinkingLevel; budget drives thinkingBudget directly.
    assert _thinking_config("gemini/gemini-2.5-flash", 800, "medium") == {
        "thinkingBudget": 800,
        "includeThoughts": True,
    }
    assert _thinking_config("gemini/gemini-2.5-flash", None, None) == {}


def test_thought_parts_are_kept_out_of_the_answer_and_readable_on_their_own() -> None:
    """The two extractors partition the parts; neither sees the other's."""

    from finesub.llm.llm_runtime import extract_message_content, extract_thought_text

    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "counting the characters...", "thought": True},
                        {"text": "sub|1|1.0|0.2|hi|你好|high|2|"},
                    ]
                }
            }
        ]
    }
    assert extract_message_content(response) == "sub|1|1.0|0.2|hi|你好|high|2|"
    assert extract_thought_text(response) == "counting the characters..."
    assert extract_thought_text({"candidates": []}) == ""


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
    from finesub.llm.client import LLMCallResult, llm_exchange_metadata

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
    from finesub.llm.prompt_compose import PROMPT_VERSION

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
    from finesub.llm.rate_limit import parse_retry_after_seconds

    # Gemini structured retryDelay in error JSON text
    exc = RuntimeError(
        'RateLimitError: Quota exceeded. "retryDelay": "51.980s". '
        'quotaId "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"'
    )
    assert parse_retry_after_seconds(exc) == pytest.approx(51.98)


def test_parse_retry_after_seconds_gemini_please_retry_in() -> None:
    from finesub.llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError(
        "RateLimitError: Quota exceeded for metric generate_content_free_tier_"
        "requests, limit 20. Please retry in 12.5s."
    )
    assert parse_retry_after_seconds(exc) == pytest.approx(12.5)


def test_parse_retry_after_seconds_openai_retry_after_header() -> None:
    from finesub.llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError("Error code: 429 - Retry-After: 30")
    assert parse_retry_after_seconds(exc) == pytest.approx(30.0)


def test_parse_retry_after_seconds_anthropic_retry_after() -> None:
    from finesub.llm.rate_limit import parse_retry_after_seconds

    exc = RuntimeError('{"type":"error","retry_after":45,"error":{"type":"rate_limit_error"}}')
    assert parse_retry_after_seconds(exc) == pytest.approx(45.0)


def test_parse_retry_after_seconds_generic_wait() -> None:
    from finesub.llm.rate_limit import parse_retry_after_seconds

    assert parse_retry_after_seconds(
        RuntimeError("rate limited, wait 60 seconds before retrying")
    ) == pytest.approx(60.0)
    assert parse_retry_after_seconds(
        RuntimeError("try again in 25s")
    ) == pytest.approx(25.0)


def test_parse_retry_after_seconds_no_hint_returns_zero() -> None:
    from finesub.llm.rate_limit import parse_retry_after_seconds

    assert parse_retry_after_seconds(RuntimeError("HTTP 429 too many requests")) == 0.0
    assert parse_retry_after_seconds(RuntimeError("HTTP 503 unavailable")) == 0.0


# ---------------------------------------------------------------------------
# key_id_for_secret
# ---------------------------------------------------------------------------


def test_key_id_for_secret_is_stable_and_non_reversible() -> None:
    from finesub.llm.rate_limit import key_id_for_secret

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
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=False)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    # Three consecutive strikes on key-a lock only key-a.
    assert not limiter.note_daily_quota_hit(ep, key_id="key-a", now=1000.0)
    assert not limiter.note_daily_quota_hit(ep, key_id="key-a", now=1001.0)
    assert limiter.note_daily_quota_hit(ep, key_id="key-a", now=1002.0)
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


def test_chat_complete_sticky_retries_count_toward_rpm(monkeypatch, tmp_path) -> None:
    """Each HTTP attempt (including failures) increments the RPM window."""
    from finesub.llm import llm_runtime
    from finesub.llm.rate_limit import ModelRateLimiter, endpoint_key

    calls = {"count": 0}

    class TransientError(RuntimeError):
        status_code = 503

    def fake_completion(**kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise TransientError("503 high demand / unavailable")
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 10},
        }

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{free-main:key1}"}

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=3,
        rate_limiter=limiter,
        estimated_input_tokens=100,
    )

    assert calls["count"] == 3
    key_id = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id
    bucket = limiter._bucket(endpoint_key(ep, key_id))
    assert len(bucket.request_times) == 3
    # TPM pre-reserve only on the first attempt.
    assert len(bucket.token_events) == 1
    assert bucket.token_events[0].tokens == 100


def test_chat_complete_backoff_uses_provider_hint_clamped(monkeypatch) -> None:
    """Backoff = min(max(exponential, provider_hint), 300) + 1."""
    from finesub.llm import llm_runtime

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

    # attempt 0: max(4.0, 20) = 20 -> min(20, 300) + 1 = 21
    # attempt 1: max(8.0, 20) = 20 -> min(20, 300) + 1 = 21
    # attempt 2: max(16.0, 20) = 20 -> min(20, 300) + 1 = 21
    assert sleeps == [21.0, 21.0, 21.0]


def test_chat_complete_backoff_caps_at_300_plus_1(monkeypatch) -> None:
    from finesub.llm import llm_runtime

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

    # Provider says 9999s but cap is 300: min(max(4.0, 9999), 300) + 1 = 301
    assert sleeps == [301.0]


# ---------------------------------------------------------------------------
# Combo cooldown (tier + model + key transient failure window)
# ---------------------------------------------------------------------------


def test_combo_cooldown_phase_transitions(tmp_path) -> None:
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    base = datetime(2026, 7, 29, 12, 0, 0, tzinfo=timezone.utc)

    limiter.note_combo_exhausted(ep, key_id="k1", now=base)
    assert (
        limiter.combo_cooldown_phase(ep, key_id="k1", now=base + timedelta(minutes=5))
        is ComboCooldownPhase.SKIP
    )
    assert (
        limiter.combo_cooldown_phase(ep, key_id="k1", now=base + timedelta(minutes=25))
        is ComboCooldownPhase.PROBE
    )
    assert limiter.effective_sticky_retries(
        ep, key_id="k1", default_retries=3, now=base + timedelta(minutes=25)
    ) == 0
    limiter.clear_combo_cooldown(ep, key_id="k1")
    assert (
        limiter.combo_cooldown_phase(ep, key_id="k1", now=base + timedelta(minutes=25))
        is ComboCooldownPhase.NONE
    )
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)
    assert (
        limiter.combo_cooldown_phase(
            ep, key_id="k1", now=base + timedelta(seconds=COMBO_COOLDOWN_TTL_SECONDS)
        )
        is ComboCooldownPhase.NONE
    )


def test_chat_complete_skips_combo_in_skip_phase(monkeypatch, tmp_path) -> None:
    from finesub.llm import llm_runtime

    used_keys: list[str] = []

    def fake_completion(**kwargs):
        used_keys.append(kwargs["api_key"])
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
        }

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{key-a:secret-a,key-b:secret-b}"}
    key_a = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id

    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    limiter.note_combo_exhausted(ep, key_id=key_a, now=started)

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=0,
        rate_limiter=limiter,
        estimated_input_tokens=10,
    )

    assert used_keys == ["secret-b"]


def test_a_pinned_call_moves_to_the_next_key_once_the_first_is_locked(
    monkeypatch, tmp_path
) -> None:
    """Pinning means one key per call, not key #1 forever (owner, 2026-08-19).

    A media call pins because the Files object belongs to the uploading key's
    project. Until this was fixed the pin truncated the pool *before* the
    daily-exhaustion skip could look at it, so once key-a locked the call
    reported "every key skipped" while key-b sat there with a full quota --
    for media that meant the run lost the pool entirely.
    `client._uploaded_media_ref` picks its upload key by the same rule, so
    both sides land in the same project.
    """

    from finesub.llm import llm_runtime

    used_keys: list[str] = []

    def fake_completion(**kwargs):
        used_keys.append(kwargs["api_key"])
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
        }

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{key-a:secret-a,key-b:secret-b}"}
    key_a = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id
    limiter.mark_daily_exhausted(ep, key_id=key_a)

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=0,
        rate_limiter=limiter,
        estimated_input_tokens=10,
        pin_first_key=True,
    )

    assert used_keys == ["secret-b"]


def test_a_media_call_pins_to_its_named_owner_not_the_first_key(
    monkeypatch, tmp_path
) -> None:
    """The ref's canonical named id must select that exact project."""

    from finesub.llm import llm_runtime

    used_keys: list[str] = []

    def fake_completion(**kwargs):
        used_keys.append(kwargs["api_key"])
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
        }

    env_map = {"GEMINI_FREE": "{key-a:secret-a,key-b:secret-b}"}
    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=0,
        rate_limiter=ModelRateLimiter(
            state_path=tmp_path / ".state", enabled=False
        ),
        estimated_input_tokens=10,
        pin_first_key=True,
        pin_key_id="key-b",
    )

    assert used_keys == ["secret-b"]


def test_a_pinned_call_and_its_upload_choose_the_same_key(
    monkeypatch, tmp_path
) -> None:
    """The one invariant media pinning rests on, at both ends.

    The Files object belongs to the uploading key's project, so the upload and
    the call that reads it must pick the same key by the same rule -- the
    daily lock, and nothing else. When they disagree the call 403s, the
    failure path forgets the ref, the re-upload picks the same wrong key
    again, and the window burns its whole retry budget in that loop. A combo
    cooldown must therefore *not* move a pinned call: it is soft and per-combo,
    while `_first_key_for_tier` (the upload side) only ever honours the lock.
    """

    from finesub.llm import llm_runtime

    used_keys: list[str] = []

    def fake_completion(**kwargs):
        used_keys.append(kwargs["api_key"])
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
        }

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    model = "gemini/gemini-3.1-flash-lite"
    ep = ModelEndpoint(GEMINI_FREE_TIER, model)
    env_map = {"GEMINI_FREE": "{key-a:secret-a,key-b:secret-b}"}
    key_a = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    def _upload_key() -> str:
        key, _ = llm_runtime._first_key_for_tier(
            GEMINI_FREE_TIER, env_map, rate_limiter=limiter, model=model
        )
        return key

    def _call_key() -> str:
        used_keys.clear()
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model=model,
            retries=0,
            rate_limiter=limiter,
            estimated_input_tokens=10,
            pin_first_key=True,
        )
        return used_keys[-1]

    assert _upload_key() == _call_key() == "secret-a"

    # A combo cooldown on key-a is soft: neither side moves.
    limiter.note_combo_exhausted(
        ep, key_id=key_a, now=datetime.now(timezone.utc) - timedelta(minutes=5)
    )
    assert _upload_key() == _call_key() == "secret-a"

    # The daily lock is hard: both move, together.
    limiter.mark_daily_exhausted(ep, key_id=key_a)
    assert _upload_key() == _call_key() == "secret-b"


def test_a_pinned_call_never_rotates_keys_in_place(monkeypatch, tmp_path) -> None:
    """A pinned call carries a file only this key's project can read.

    So rotating would trade a quota error for a guaranteed 403. Pinning is a
    media constraint and nothing else -- `continuity=parallel` no longer pins
    (owner ruling 2026-08-19: parallel behaves like serial about keys, and the
    occasional two-keys-in-flight overlap is accepted rather than designed
    out) -- and an ordinary call still walks the pool when a key spends its
    retry budget on quota errors.
    """

    from finesub.llm import llm_runtime

    used_keys: list[str] = []

    def fake_completion(**kwargs):
        used_keys.append(kwargs["api_key"])
        raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED quota exceeded")

    env_map = {"GEMINI_FREE": "{key-a:secret-a,key-b:secret-b}"}
    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    def _run(tag, **extra):
        # A limiter per run: the quota errors below push key-a into a combo
        # cooldown, and a shared limiter would make the second run skip key-a
        # for that reason instead of the one under test.
        limiter = ModelRateLimiter(
            state_path=tmp_path / f".state-{tag}", enabled=True
        )
        used_keys.clear()
        with pytest.raises(Exception):
            llm_runtime.chat_complete(
                [{"role": "user", "content": "hi"}],
                provider_tier=GEMINI_FREE_TIER,
                model="gemini/gemini-3.1-flash-lite",
                retries=0,
                rate_limiter=limiter,
                estimated_input_tokens=10,
                **extra,
            )
        return list(used_keys)

    assert _run("pinned", pin_first_key=True) == ["secret-a"]
    # The unpinned path is unchanged: it still walks the pool.
    assert _run("free") == ["secret-a", "secret-b"]


def test_chat_complete_probe_success_clears_combo_cooldown(monkeypatch, tmp_path) -> None:
    from finesub.llm import llm_runtime

    calls = {"count": 0}

    def fake_completion(**kwargs):
        calls["count"] += 1
        return {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            "usageMetadata": {"promptTokenCount": 1},
        }

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{free-main:key1}"}
    key_id = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id
    limiter.note_combo_exhausted(
        ep,
        key_id=key_id,
        now=datetime.now(timezone.utc) - timedelta(minutes=25),
    )

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(
        llm_runtime,
        "time",
        type(
            "T",
            (),
            {
                "monotonic": staticmethod(lambda: 0.0),
                "sleep": staticmethod(lambda _: None),
                "time": staticmethod(lambda: 0.0),
            },
        )(),
    )

    llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.1-flash-lite",
        retries=3,
        rate_limiter=limiter,
        estimated_input_tokens=10,
    )

    assert calls["count"] == 1
    assert endpoint_key(ep, key_id) not in limiter._combo_cooldowns


def test_chat_complete_sticky_exhaustion_starts_combo_cooldown(monkeypatch, tmp_path) -> None:
    from finesub.llm import llm_runtime

    class TransientError(RuntimeError):
        status_code = 503

    def fake_completion(**kwargs):
        raise TransientError("503 unavailable")

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{free-main:key1}"}
    key_id = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with pytest.raises(TransientError):
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.1-flash-lite",
            retries=1,
            rate_limiter=limiter,
            estimated_input_tokens=10,
        )

    assert endpoint_key(ep, key_id) in limiter._combo_cooldowns


def test_chat_complete_probe_failure_restarts_combo_cooldown(monkeypatch, tmp_path) -> None:
    from finesub.llm import llm_runtime

    class TransientError(RuntimeError):
        status_code = 503

    def fake_completion(**kwargs):
        raise TransientError("503 unavailable")

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.1-flash-lite")
    env_map = {"GEMINI_FREE": "{free-main:key1}"}
    key_id = llm_runtime._get_key_entries(GEMINI_FREE_TIER, env_map)[0].key_id
    old_start = datetime.now(timezone.utc) - timedelta(minutes=25)
    limiter.note_combo_exhausted(ep, key_id=key_id, now=old_start)
    old_stamp = limiter._combo_cooldowns[endpoint_key(ep, key_id)]

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: env_map)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_completion)

    with pytest.raises(TransientError):
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier=GEMINI_FREE_TIER,
            model="gemini/gemini-3.1-flash-lite",
            retries=3,
            rate_limiter=limiter,
            estimated_input_tokens=10,
        )

    new_stamp = limiter._combo_cooldowns[endpoint_key(ep, key_id)]
    assert new_stamp != old_stamp
    assert (
        limiter.combo_cooldown_phase(ep, key_id=key_id)
        is ComboCooldownPhase.SKIP
    )


def test_a_deterministic_400_is_not_retried_because_it_contains_digits() -> None:
    """The markers were matched unbounded against the whole error body.

    `GeminiAPIError.__str__` embeds the provider's JSON, which routinely
    carries token counts. "The input token count (215000)" contains "500", so
    a permanent 400 was classified retryable: three sticky retries against the
    identical oversized body, then the same again on each of the chain's five
    endpoints -- up to twenty doomed calls and about two minutes of backoff,
    against a free tier whose entire daily budget is twenty requests. `4290` in
    an id did the same for the quota predicate.
    """
    from finesub.llm.client import is_quota_or_rate_limit_error, is_retryable_provider_error
    from finesub.llm.llm_runtime import GeminiAPIError

    oversized = GeminiAPIError(
        'Gemini API error (HTTP 400): {"message":"The input token count '
        '(215000) exceeds the maximum number of tokens allowed (194000)."}',
        400,
    )
    digits_in_an_id = GeminiAPIError(
        "Gemini API error (HTTP 400): invalid argument: fileUri 4290 bad", 400
    )

    assert not is_retryable_provider_error(oversized)
    assert not is_quota_or_rate_limit_error(oversized)
    assert not is_retryable_provider_error(digits_in_an_id)
    assert not is_quota_or_rate_limit_error(digits_in_an_id)

    # The real thing still classifies, from the status rather than the prose.
    assert is_retryable_provider_error(GeminiAPIError("busy", 503))
    assert is_quota_or_rate_limit_error(GeminiAPIError("slow down", 429))
    assert is_retryable_provider_error(GeminiAPIError("slow down", 429))


def test_a_transport_failure_without_a_status_is_still_retryable() -> None:
    """Statusless transports keep the text fallback -- just not as the primary."""
    from finesub.llm.client import is_retryable_provider_error

    assert is_retryable_provider_error(TimeoutError("request timed out"))
    assert is_retryable_provider_error(RuntimeError("service temporarily down"))
    assert not is_retryable_provider_error(ValueError("bad argument"))


def test_only_one_concurrent_caller_claims_a_combo_probe(tmp_path) -> None:
    """PROBE is one call, not one call per parallel window.

    Reading the phase is a pure read, so with `continuity=parallel` every
    concurrent window saw PROBE at the same instant and each fired its own --
    four probes of something meant to be tested with one, four requests off a
    free tier's daily allowance, and four restarts of the 20-minute skip.
    """

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    base = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    probe_at = base + timedelta(minutes=25)
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)

    claims = [
        limiter.claim_combo_probe(ep, key_id="k1", now=probe_at) for _ in range(4)
    ]

    assert claims.count(True) == 1
    # Everyone else still reads PROBE; the caller is what must treat it as SKIP.
    assert (
        limiter.combo_cooldown_phase(ep, key_id="k1", now=probe_at)
        is ComboCooldownPhase.PROBE
    )


def test_a_probe_claim_is_scoped_to_its_cooldown_window(tmp_path) -> None:
    """A failed probe restarts the window, which must re-open the claim."""

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    base = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)
    assert limiter.claim_combo_probe(ep, key_id="k1", now=base + timedelta(minutes=25))

    # The probe failed: the window restarts, so the next one is claimable again.
    restart = base + timedelta(minutes=25)
    limiter.note_combo_exhausted(ep, key_id="k1", now=restart)
    assert limiter.claim_combo_probe(ep, key_id="k1", now=restart + timedelta(minutes=25))

    # A cleared cooldown leaves no claim behind either.
    limiter.clear_combo_cooldown(ep, key_id="k1")
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)
    assert limiter.claim_combo_probe(ep, key_id="k1", now=base + timedelta(minutes=25))


def test_claiming_outside_the_probe_phase_never_succeeds(tmp_path) -> None:
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    base = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)

    # No cooldown at all.
    assert not limiter.claim_combo_probe(ep, key_id="k1", now=base)
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)
    # SKIP window.
    assert not limiter.claim_combo_probe(ep, key_id="k1", now=base + timedelta(minutes=5))
    # Past the TTL the cooldown is gone.
    assert not limiter.claim_combo_probe(ep, key_id="k1", now=base + timedelta(minutes=150))


def test_probe_claim_holds_under_real_thread_contention(tmp_path) -> None:
    """The whole point is concurrency, so contend for it with real threads."""

    import threading

    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    ep = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")
    base = datetime(2026, 8, 15, 12, 0, 0, tzinfo=timezone.utc)
    probe_at = base + timedelta(minutes=25)
    limiter.note_combo_exhausted(ep, key_id="k1", now=base)

    start = threading.Barrier(8)
    won: list[bool] = []
    won_lock = threading.Lock()

    def contend() -> None:
        start.wait()
        claimed = limiter.claim_combo_probe(ep, key_id="k1", now=probe_at)
        with won_lock:
            won.append(claimed)

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(won) == 8
    assert won.count(True) == 1


def test_a_dead_chain_reports_every_candidate_not_just_the_last_one() -> None:
    """The last link's exception is not the reason the call failed.

    The 2026-08-15 605-subtitle run died with "Provider GEMINI_PAID is disabled
    or has no selected API key" -- true about the final candidate and unrelated
    to the cause, which was six correction retries draining every free tier
    ahead of it. The summary is appended and the type is preserved, because
    failure classification is isinstance-based and the retry-after parsers scan
    the original text.
    """

    from finesub.llm.routing.api_keys import ProviderUnavailableError
    from finesub.llm.client import _append_chain_summary

    exc = ProviderUnavailableError("Provider GEMINI_PAID has no selected API key.")
    _append_chain_summary(
        exc,
        {
            "candidates": [
                {"target_id": "gemini-free-3_7-flash", "failure_kind": "quota"},
                {"target_id": "gemini-free-3_6-flash", "failure_kind": "quota"},
                {"target_id": "gemini-paid-3_7-flash", "failure_kind": "unavailable"},
            ]
        },
    )

    text = str(exc)
    assert text.startswith("Provider GEMINI_PAID has no selected API key.")
    assert "gemini-free-3_7-flash=quota" in text
    assert "gemini-free-3_6-flash=quota" in text
    assert "gemini-paid-3_7-flash=unavailable" in text
    assert isinstance(exc, ProviderUnavailableError)


def test_a_single_candidate_chain_is_left_unannotated() -> None:
    """Nothing to summarise when there was no chain to exhaust."""

    from finesub.llm.routing.api_keys import ProviderUnavailableError
    from finesub.llm.client import _append_chain_summary

    exc = ProviderUnavailableError("boom")
    _append_chain_summary(exc, {"candidates": [{"target_id": "only", "failure_kind": "x"}]})
    assert str(exc) == "boom"


def test_a_grounded_reply_lands_in_the_retrieval_ledger(monkeypatch) -> None:
    """`retrieval=native` over `google_search` used to leave no trace.

    The reply carries the queries the model ran and the pages that grounded its
    answer; nothing read them, so a Gemini native round looked identical to a
    backend that reports nothing -- and "native retrieval is unauditable" was
    partly a statement about this parser.
    """

    def fake_chat_complete(messages, *, model, **kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
            # Every success path through `chat_complete` attaches these; the row
            # the grounding hangs off is the attempt that answered.
            "_harness_api_attempts": [{"return_code": "200", "model": model}],
            "candidates": [
                {
                    "groundingMetadata": {
                        "webSearchQueries": ["虚拟主播 本名"],
                        "groundingChunks": [
                            {"web": {"uri": "https://a.test/x", "title": "A"}},
                            {"web": {"uri": "https://b.test/y", "title": "B"}},
                            {"web": {"uri": "https://a.test/x", "title": "A"}},
                        ],
                    }
                }
            ],
        }

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    events = result.execution_attempts[-1]["search_events"]
    assert len(events) == 1
    assert events[0]["queries"] == ["虚拟主播 本名"]
    assert events[0]["query"] == "虚拟主播 本名"
    assert events[0]["urls"] == ["https://a.test/x", "https://b.test/y"]


def test_an_ungrounded_reply_adds_no_ledger_row(monkeypatch) -> None:
    """Absent or empty grounding metadata must not manufacture an empty row --
    downstream reads "has a URL row" as "this backend can be corroborated"."""

    def fake_chat_complete(messages, *, model, **kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {},
            "_harness_api_attempts": [{"return_code": "200", "model": model}],
            "candidates": [{"finishReason": "STOP", "groundingMetadata": {}}],
        }

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}]
    )

    assert all(
        "search_events" not in attempt for attempt in result.execution_attempts
    )


def _capture_dispatch(monkeypatch) -> dict:
    captured: dict = {}

    def fake_chat_complete(messages, *, model, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    return captured


def test_a_repair_call_shows_the_model_its_own_answer_and_the_errors(
    monkeypatch,
) -> None:
    """A stateless endpoint gets the repair context as two extra turns.

    The previous answer is an assistant turn rather than text quoted inside the
    user turn: that is what makes it "what you wrote" to the model instead of a
    document someone handed it, and it costs nothing to re-quote.
    """

    captured = _capture_dispatch(monkeypatch)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
    )

    sent = captured["messages"]
    assert sent[0] == {"role": "user", "content": "hi"}
    assert sent[1] == {"role": "assistant", "content": "sub|1|wrong"}
    assert sent[2]["role"] == "user"
    assert "Row 1 references unknown source id 3." in sent[2]["content"]


def test_repair_context_needs_both_an_answer_and_a_reason(monkeypatch) -> None:
    """Half of it is only prompt weight: an error with no output to fix, or an
    output with no stated problem, changes nothing about what to write."""

    captured = _capture_dispatch(monkeypatch)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
    )
    assert captured["messages"] == [{"role": "user", "content": "hi"}]

    client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        validation_errors=["   ", ""],
    )
    assert captured["messages"] == [{"role": "user", "content": "hi"}]


def test_a_repair_call_keeps_the_clip_on_the_turn_that_states_the_task(
    monkeypatch,
) -> None:
    """The attachment belongs to the window description, not to the request to
    fix it -- appending the repair turns after the attach must not move it."""

    captured = _capture_dispatch(monkeypatch)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))
    ref = UploadedFileRef(
        file_id="https://generativelanguage.googleapis.com/v1beta/files/x",
        filename="x.flac",
        mime_type="audio/flac",
    )

    client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "window"}],
        file_ref=ref,
        previous_output="sub|1|wrong",
        validation_errors=["Translated missing source id(s): 2."],
    )

    sent = captured["messages"]
    assert sent[-1]["role"] == "user"
    assert isinstance(sent[-1]["content"], str)  # the repair turn, text only
    assert any(
        isinstance(part, dict) and part.get("type") == "file"
        for part in sent[0]["content"]
    )


def test_a_candidate_that_cannot_hold_the_repair_context_still_gets_its_retry(
    monkeypatch,
) -> None:
    """Dropping the aid beats losing the attempt.

    Repair context is added to a window that was already planned to fit. If it
    tips that window over the target's input limit, the candidate falls back to
    the blind retry every retry used to be -- skipping it could empty the chain
    and turn a recoverable validation failure into a dead run.
    """

    captured = _capture_dispatch(monkeypatch)

    def estimate(messages, **kwargs):
        has_repair = any(msg.get("role") == "assistant" for msg in messages)
        return 2_000_000 if has_repair else 10

    monkeypatch.setattr("finesub.llm.client.estimate_call_input_tokens", estimate)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.GENERAL_CAPABLE,
        [{"role": "user", "content": "hi"}],
        previous_output="sub|1|wrong",
        validation_errors=["Row 1 references unknown source id 3."],
    )

    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    accepted = [
        row
        for row in result.route_decision["candidates"]
        if row.get("decision") == "accepted"
    ]
    assert accepted[0]["repair_context"] == "dropped_input_limit"
    # ⚠ The output request has to be recomputed from what actually goes out.
    # It was clamped against the *rejected* 2,000,000-token estimate until
    # 2026-09-04, so this call -- one word of prompt, the whole window free --
    # went to the provider asking for a single token, and the blind retry the
    # drop exists to enable came back truncated.
    assert captured["kwargs"]["max_tokens"] > 1
    assert result.requested_output_tokens == captured["kwargs"]["max_tokens"]


# --- Files API upload retries -------------------------------------------------
#
# The retry budget is the upload's own; it is classified by httpx exception
# type, because the failure seen in the field carries no retryable-looking text.

import threading

import httpx

from finesub.llm import client as client_module
from finesub.llm import media_upload


def _status_error(status: int, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://upload.test/session?upload_id=SECRET")
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


WINERROR_10054 = httpx.ReadError("[Errno 10054] 远程主机强迫关闭了一个现有的连接。")


class ScriptedHttpClient:
    """Resumable-protocol fake whose failures are scripted per request index.

    ``failures`` maps the 0-based index of a *post* (start / finalize /
    countTokens, in call order) to the exception that call raises.
    """

    def __init__(self, *, timeout, failures: dict[int, BaseException] | None = None) -> None:
        self.timeout = timeout
        self.failures = failures or {}
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.sessions = 0
        # Scripted countTokens replies, consumed in order; then always ready.
        self.probe_responses: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def post(self, url, **kwargs):
        index = len(self.posts)
        self.posts.append((url, kwargs))
        if index in self.failures:
            failure = self.failures[index]
            raise failure() if callable(failure) else failure
        if ":countTokens" in url:
            return self.probe_responses.pop(0) if self.probe_responses else FakeResponse(
                payload={"totalTokens": 4242}
            )
        if kwargs.get("headers", {}).get("X-Goog-Upload-Command") == "start":
            self.sessions += 1
            return FakeResponse(
                headers={"x-goog-upload-url": f"https://upload.test/session-{self.sessions}"}
            )
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
        self.gets.append(url)
        return FakeResponse(payload={"name": "files/yui", "state": "ACTIVE"})


def _upload(tmp_path, monkeypatch, *, failures=None, cancel=None, max_attempts=3):
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake audio")
    reads = {"n": 0}
    real_read = Path.read_bytes

    def counted_read(self):
        reads["n"] += 1
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", counted_read)
    warnings: list[tuple[str, str]] = []

    class Reporter:
        @staticmethod
        def warning(code, message, **kwargs):
            warnings.append((code, message))

    # The upload lives in `media_upload` now, so that is where both lookups
    # happen -- patching `client_module` here left the retry warnings unseen.
    monkeypatch.setattr(media_upload, "current_reporter", lambda: Reporter())
    monkeypatch.setattr(media_upload.random, "uniform", lambda a, b: 0.0)
    sleeps: list[float] = []
    holder: dict = {}

    def factory(**kwargs):
        holder["client"] = ScriptedHttpClient(failures=failures, **kwargs)
        return holder["client"]

    def run():
        return _upload_gemini_file_rest(
            audio,
            api_key="test-key",
            client_factory=factory,
            sleep_func=sleeps.append,
            cancel=cancel,
            max_attempts=max_attempts,
        )

    return run, holder, sleeps, warnings, reads


def test_upload_without_failures_uses_one_session_and_one_send(tmp_path, monkeypatch) -> None:
    run, holder, sleeps, warnings, reads = _upload(tmp_path, monkeypatch)
    ref = run()
    assert ref.file_id.endswith("/files/yui")
    assert holder["client"].sessions == 1
    assert sleeps == [] and warnings == []
    assert isinstance(holder["client"].timeout, httpx.Timeout)


def test_a_reset_during_finalize_is_retried_with_a_fresh_session(tmp_path, monkeypatch) -> None:
    # The incident: finalize (post #1) dies with WinError 10054.
    run, holder, sleeps, warnings, reads = _upload(
        tmp_path, monkeypatch, failures={1: WINERROR_10054}
    )
    ref = run()
    assert ref.file_id.endswith("/files/yui")
    client = holder["client"]
    assert client.sessions == 2
    # start, finalize(fail), start, finalize, countTokens
    commands = [
        p[1].get("headers", {}).get("X-Goog-Upload-Command") for p in client.posts[:4]
    ]
    assert commands == ["start", "upload, finalize", "start", "upload, finalize"]
    assert client.posts[3][0] == "https://upload.test/session-2"
    assert sleeps == [1.0]
    assert reads["n"] == 1
    assert warnings[0][0] == "gemini-upload-retry"
    assert "ReadError" in warnings[0][1] and "10054" in warnings[0][1]


def test_a_connect_error_at_start_is_retried(tmp_path, monkeypatch) -> None:
    run, holder, sleeps, *_ = _upload(
        tmp_path, monkeypatch, failures={0: httpx.ConnectError("refused")}
    )
    run()
    assert holder["client"].sessions == 1  # the failed start never opened one
    assert len(holder["client"].posts) == 4
    assert sleeps == [1.0]


def test_the_budget_is_three_attempts_and_the_last_error_surfaces(tmp_path, monkeypatch) -> None:
    failures = {1: WINERROR_10054, 3: httpx.WriteError("x"), 5: httpx.ReadError("final")}
    run, holder, sleeps, warnings, _ = _upload(tmp_path, monkeypatch, failures=failures)
    with pytest.raises(httpx.ReadError, match="final"):
        run()
    assert holder["client"].sessions == 3
    assert sleeps == [1.0, 3.0]
    assert [code for code, _ in warnings] == [
        "gemini-upload-retry",
        "gemini-upload-retry",
        "gemini-upload-failed",
    ]
    assert "3 attempts" in warnings[-1][1]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_deterministic_client_errors_are_not_retried(tmp_path, monkeypatch, status) -> None:
    run, holder, sleeps, warnings, _ = _upload(
        tmp_path, monkeypatch, failures={1: _status_error(status)}
    )
    with pytest.raises(httpx.HTTPStatusError):
        run()
    assert holder["client"].sessions == 1
    assert sleeps == [] and warnings == []


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried_without_quoting_the_session_url(
    tmp_path, monkeypatch, status
) -> None:
    run, holder, sleeps, warnings, _ = _upload(
        tmp_path, monkeypatch, failures={1: _status_error(status)}
    )
    run()
    assert holder["client"].sessions == 2
    assert f"HTTP {status}" in warnings[0][1]
    assert "SECRET" not in warnings[0][1] and "upload.test" not in warnings[0][1]
    assert "test-key" not in warnings[0][1]


def test_retry_after_is_honoured_when_sane_and_capped_otherwise(tmp_path, monkeypatch) -> None:
    run, holder, sleeps, *_ = _upload(
        tmp_path, monkeypatch, failures={1: _status_error(429, {"retry-after": "7"})}
    )
    run()
    assert sleeps == [7.0]

    run, holder, sleeps, *_ = _upload(
        tmp_path, monkeypatch, failures={1: _status_error(429, {"retry-after": "86400"})}
    )
    run()
    assert sleeps == [1.0]

    run, holder, sleeps, *_ = _upload(
        tmp_path, monkeypatch, failures={1: _status_error(503, {"retry-after": "soon"})}
    )
    run()
    assert sleeps == [1.0]


def test_a_poll_hiccup_after_finalize_does_not_upload_again(tmp_path, monkeypatch) -> None:
    # countTokens is post #2; it fails once on the wire. The file is already
    # on the server, so only the probe is retried.
    run, holder, sleeps, warnings, _ = _upload(
        tmp_path, monkeypatch, failures={2: httpx.ReadError("blip")}
    )
    run()
    client = holder["client"]
    assert client.sessions == 1
    assert sum(":countTokens" in url for url, _ in client.posts) == 2
    assert warnings[0][0] == "gemini-upload-retry" and "token_poll" in warnings[0][1]


def test_a_cancelled_owner_stops_the_retry_at_the_attempt_boundary(tmp_path, monkeypatch) -> None:
    cancel = threading.Event()

    def reset_then_cancel():
        # The owner shuts down while this attempt is failing: the backoff
        # wait returns at once and the second attempt never goes on the wire.
        cancel.set()
        return WINERROR_10054

    run, holder, sleeps, warnings, _ = _upload(
        tmp_path,
        monkeypatch,
        failures={1: reset_then_cancel, 3: WINERROR_10054},
        cancel=cancel,
    )
    with pytest.raises(UploadCancelled):
        run()
    assert holder["client"].sessions == 1
    assert len(holder["client"].posts) == 2
    assert sleeps == []  # the wait went through the event, not sleep_func


class _StatusResponse(FakeResponse):
    """A FakeResponse whose raise_for_status behaves like httpx's."""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _status_error(self.status_code)


def test_a_transient_probe_status_retries_the_probe_only(tmp_path, monkeypatch) -> None:
    run, holder, sleeps, warnings, _ = _upload(tmp_path, monkeypatch)

    def factory_with_probe(**kwargs):
        client = ScriptedHttpClient(**kwargs)
        client.probe_responses = [_StatusResponse(status_code=429)]
        holder["client"] = client
        return client

    ref = _upload_gemini_file_rest(
        tmp_path / "clip.mp3",
        api_key="test-key",
        client_factory=factory_with_probe,
        sleep_func=sleeps.append,
    )
    assert ref.file_id.endswith("/files/yui")
    client = holder["client"]
    assert client.sessions == 1
    assert sum(":countTokens" in url for url, _ in client.posts) == 2
    assert warnings[0][0] == "gemini-upload-retry" and "HTTP 429" in warnings[0][1]
    assert sleeps == [1.0]


def test_a_deterministic_probe_error_fails_at_once_instead_of_polling(
    tmp_path, monkeypatch
) -> None:
    run, holder, sleeps, warnings, _ = _upload(tmp_path, monkeypatch)

    def factory_with_probe(**kwargs):
        client = ScriptedHttpClient(**kwargs)
        client.probe_responses = [_StatusResponse(status_code=401)]
        holder["client"] = client
        return client

    with pytest.raises(httpx.HTTPStatusError):
        _upload_gemini_file_rest(
            tmp_path / "clip.mp3",
            api_key="test-key",
            client_factory=factory_with_probe,
            sleep_func=sleeps.append,
        )
    assert sum(":countTokens" in url for url, _ in holder["client"].posts) == 1
    assert sleeps == [] and warnings == []


def test_a_not_ready_probe_is_still_a_wait_not_a_failure(tmp_path, monkeypatch) -> None:
    run, holder, sleeps, warnings, _ = _upload(tmp_path, monkeypatch)

    def factory_with_probe(**kwargs):
        client = ScriptedHttpClient(**kwargs)
        client.probe_responses = [_StatusResponse(status_code=400)]
        holder["client"] = client
        return client

    _upload_gemini_file_rest(
        tmp_path / "clip.mp3",
        api_key="test-key",
        client_factory=factory_with_probe,
        sleep_func=sleeps.append,
    )
    assert sum(":countTokens" in url for url, _ in holder["client"].posts) == 2
    assert sleeps == [2.0] and warnings == []


def test_cancel_during_a_polling_wait_returns_at_once(tmp_path, monkeypatch) -> None:
    cancel = threading.Event()
    run, holder, sleeps, warnings, _ = _upload(tmp_path, monkeypatch, cancel=cancel)

    class CancelOnFirstProbe(ScriptedHttpClient):
        def post(self, url, **kwargs):
            if ":countTokens" in url and not cancel.is_set():
                # Owner shuts down while the media is still not sampled: the
                # 2s polling wait must return at once, not run out.
                cancel.set()
                self.probe_responses = [_StatusResponse(status_code=400)]
            return super().post(url, **kwargs)

    def factory_with_probe(**kwargs):
        holder["client"] = CancelOnFirstProbe(**kwargs)
        return holder["client"]

    with pytest.raises(UploadCancelled, match="token_poll"):
        _upload_gemini_file_rest(
            tmp_path / "clip.mp3",
            api_key="test-key",
            client_factory=factory_with_probe,
            sleep_func=sleeps.append,
            cancel=cancel,
        )
    assert sleeps == []


def test_agent_transport_usage_never_trips_the_output_limit_heuristic() -> None:
    """Agent CLIs report SESSION-CUMULATIVE usage in the result event; comparing
    it against a per-turn cap misreads healthy calls as truncated and triggers a
    needless split-in-half rerun (docs/report 2026-08-28 §2.2: 61,446/65,536
    observed with zero truncated turns). The agent-shaped raw_response — the
    dict the LOCAL dispatch branch builds with an "agent" key — is exempt; the
    REST path keeps the proximity heuristic."""

    from finesub.llm.client import is_agent_transport_response, is_likely_output_limited
    from finesub.llm.stages.correction.metadata import _output_limit_check

    agent_response = {
        "usage": {"completion_tokens": 61_446},
        "agent": {"capsule_id": "x", "events": []},
    }
    assert is_agent_transport_response(agent_response)
    assert not is_likely_output_limited(agent_response, max_tokens=65_536)

    rest_response = {"usage": {"completion_tokens": 65_530}}
    assert not is_agent_transport_response(rest_response)
    assert is_likely_output_limited(rest_response, max_tokens=65_536)

    check = _output_limit_check(dict(agent_response, usage={"completion_tokens": 65_530}), 65_536, 100)
    assert check["limited"] is False
    assert check["basis"] == "agent_session_cumulative_usage_not_comparable"
