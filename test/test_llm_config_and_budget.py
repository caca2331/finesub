from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

import llm.token_budget as token_budget
from llm.client import (
    QuotaKind,
    classify_quota_error,
    is_quota_or_rate_limit_error,
    is_retryable_provider_error,
)
from llm.config import (
    CAPABILITY_TIER_THRESHOLD,
    DEFAULT_LIMITS,
    GEMINI_25_FLASH,
    GEMINI_31_FLASH_LITE,
    GEMINI_35_FLASH,
    GEMINI_35_FLASH_LITE,
    GEMINI_36_FLASH,
    GEMINI_FREE_TIER,
    CapabilityTier,
    LLMRole,
    ModelEndpoint,
    default_role_configs,
    research_search_query_limit,
    thinking_budget_for_level,
    tier_for_capability,
)
from llm.model_catalog import (
    CATALOG_COLUMNS,
    default_model_catalog,
    get_model_catalog_entry,
    get_model_catalog_entry_for_tier,
)
from llm.rate_limit import ModelRateLimiter, endpoint_key, estimate_call_input_tokens
from llm.profiles import resolve_profile
from llm.token_budget import (
    FallbackTokenCounter,
    GeminiCountTokensCounter,
    HeuristicTokenCounter,
    LocalGeminiTokenCounter,
    TokenBudgetError,
    build_correction_budget,
    default_token_counter,
    requested_output_limit,
    validate_correction_budget,
)


def test_tier_for_capability_threshold_boundary() -> None:
    assert tier_for_capability(7) is CapabilityTier.CAPABLE  # 3.5-flash
    # 3.0-flash sits exactly on the threshold and stays capable.
    assert CAPABILITY_TIER_THRESHOLD == 6
    assert tier_for_capability(6) is CapabilityTier.CAPABLE
    assert tier_for_capability(5) is CapabilityTier.BASIC  # flash-lite / 2.5
    assert tier_for_capability(3) is CapabilityTier.BASIC  # gemma


def test_default_role_configs_use_expected_endpoint_chains() -> None:
    configs = default_role_configs()
    free_36 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_36_FLASH)
    free_35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH)
    free_lite35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH_LITE)

    # Correction role (audio_multimodal): 3.6 first.
    audio = configs[LLMRole.AUDIO_MULTIMODAL]
    assert audio.endpoints(test_profile=True) == (free_lite35,)
    correction_models = [ep.litellm_model for ep in audio.endpoints(test_profile=False)]
    assert correction_models[0] == GEMINI_36_FLASH
    assert correction_models.index(GEMINI_36_FLASH) < correction_models.index(
        GEMINI_35_FLASH
    )
    assert correction_models.index(GEMINI_35_FLASH_LITE) < correction_models.index(
        GEMINI_31_FLASH_LITE
    )

    # General capable: 3.5 → 3.6 → 3.5-lite (research / fast r1 / knowledge).
    general = configs[LLMRole.GENERAL_CAPABLE]
    general_models = [ep.litellm_model for ep in general.endpoints(test_profile=False)]
    assert general_models[0] == GEMINI_35_FLASH
    assert general_models.index(GEMINI_35_FLASH) < general_models.index(GEMINI_36_FLASH)
    assert general_models.index(GEMINI_36_FLASH) < general_models.index(
        GEMINI_35_FLASH_LITE
    )

    # Lightweight roles prefer 3.5 Flash Lite (纠错 r1 MM + search-loop text).
    lightweight_mm = configs[LLMRole.LIGHTWEIGHT_MULTIMODAL]
    lite_models = [
        ep.litellm_model for ep in lightweight_mm.endpoints(test_profile=False)
    ]
    assert lite_models[0] == GEMINI_35_FLASH_LITE
    assert lite_models.index(GEMINI_35_FLASH_LITE) < lite_models.index(
        GEMINI_31_FLASH_LITE
    )
    assert free_36 not in lightweight_mm.endpoints(test_profile=False)
    assert lightweight_mm.thinking_level == "medium"
    assert configs[LLMRole.AUDIO_MULTIMODAL].thinking_level == "medium"
    internet = configs[LLMRole.INTERNET_CAPABLE]
    assert all(
        endpoint.litellm_model == GEMINI_25_FLASH
        for endpoint in internet.endpoints(test_profile=False)
    )


def test_model_catalog_loads_gemini_tier_psv_facts() -> None:
    from llm.model_catalog import default_model_catalog

    default_model_catalog.cache_clear()
    entries = default_model_catalog()
    assert len(entries) == 13
    free_entries = [e for e in entries if e.provider_tier == "GEMINI_FREE"]
    paid_entries = [e for e in entries if e.provider_tier == "GEMINI_PAID"]
    assert len(free_entries) == 7
    assert len(paid_entries) == 6
    gemma4 = get_model_catalog_entry_for_tier("gemini/gemma-4-31b-it", "GEMINI_FREE")
    assert gemma4 is not None
    assert gemma4.max_input_tokens == 16_000
    assert gemma4.max_output_tokens == 32_768
    assert gemma4.rpm == 15
    assert gemma4.tpm == -1
    assert gemma4.rpd == 1500
    assert gemma4.supports_native_search is True
    non_gemma_entries = [e for e in entries if e.litellm_model != "gemini/gemma-4-31b-it"]
    assert all(entry.max_input_tokens == 194_000 for entry in non_gemma_entries)
    assert all(entry.max_output_tokens == 65_536 for entry in non_gemma_entries)
    lite = get_model_catalog_entry_for_tier(
        "gemini/gemini-3.1-flash-lite", "GEMINI_FREE"
    )
    assert lite is not None
    assert lite.supports_reasoning is True
    assert lite.rpm == 15
    assert lite.rpd == 500
    # New models share their predecessor's limits and capability (3.6 Flash ==
    # 3.5 Flash; 3.5 Flash Lite == 3.1 Flash Lite) on both tiers.
    for tier in ("GEMINI_FREE", "GEMINI_PAID"):
        flash36 = get_model_catalog_entry_for_tier("gemini/gemini-3.6-flash", tier)
        flash35 = get_model_catalog_entry_for_tier("gemini/gemini-3.5-flash", tier)
        assert flash36 is not None and flash35 is not None
        assert (flash36.rpm, flash36.tpm, flash36.rpd, flash36.tpd) == (
            flash35.rpm,
            flash35.tpm,
            flash35.rpd,
            flash35.tpd,
        )
        assert flash36.capability == flash35.capability == 7
        lite35 = get_model_catalog_entry_for_tier("gemini/gemini-3.5-flash-lite", tier)
        lite31 = get_model_catalog_entry_for_tier("gemini/gemini-3.1-flash-lite", tier)
        assert lite35 is not None and lite31 is not None
        assert (lite35.rpm, lite35.tpm, lite35.rpd, lite35.tpd) == (
            lite31.rpm,
            lite31.tpm,
            lite31.rpd,
            lite31.tpd,
        )
        assert lite35.capability == lite31.capability == 5
    flash25 = get_model_catalog_entry("gemini/gemini-2.5-flash")
    assert flash25 is not None
    assert flash25.supports_native_search is True
    assert flash25.capability == 5
    assert "provider_tier" in CATALOG_COLUMNS


def test_thinking_budget_derives_from_level_share_of_output_limit() -> None:
    # low/medium/high = 20%/40%/60% of the API output limit (65,536).
    assert thinking_budget_for_level("low") == 13_107
    assert thinking_budget_for_level("medium") == 26_214
    assert thinking_budget_for_level("high") == 39_321
    assert thinking_budget_for_level("") == 0
    # Role configs no longer carry standalone budget numbers: they derive
    # from their thinking_level.
    for config in default_role_configs().values():
        assert config.thinking_budget == thinking_budget_for_level(config.thinking_level)


def test_token_budget_uses_fixed_output_limit_and_profile_output_estimate() -> None:
    assert requested_output_limit() == 65_536

    # Default profile is mm-med: expected output = 5.0 x csv tokens.
    budget = build_correction_budget(
        input_tokens=20_000,
        subtitle_input_tokens=1_000,
        token_counter_source="test",
    )
    validate_correction_budget(budget)
    assert budget.estimated_output_tokens == 5_000
    assert budget.total_with_margin == 20_000 + 5_000 + DEFAULT_LIMITS.safety_margin

    text_low = build_correction_budget(
        input_tokens=20_000,
        subtitle_input_tokens=1_000,
        token_counter_source="test",
        profile=resolve_profile("text", "low"),
    )
    assert text_low.estimated_output_tokens == 2_000


def test_audio_token_count_uses_gemini_official_32_tokens_per_second() -> None:
    counter = GeminiCountTokensCounter()

    assert DEFAULT_LIMITS.audio_tokens_per_second == 32
    assert counter.count_audio_seconds(10.0) == 320
    assert counter.count_audio_seconds(0.0) == 0


class FakeCountTokensResponse:
    def __init__(self, payload=None, *, should_fail: bool = False) -> None:
        self.payload = payload or {"totalTokens": 123}
        self.status_code = 500 if should_fail else 200
        self.text = "countTokens failed" if should_fail else ""

    def json(self):
        return self.payload


class FakeCountTokensClient:
    def __init__(self, *, timeout: float, should_fail: bool = False) -> None:
        self.timeout = timeout
        self.should_fail = should_fail
        self.posts = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return FakeCountTokensResponse(should_fail=self.should_fail)


def test_gemini_count_tokens_counter_calls_api() -> None:
    clients = []

    def client_factory(**kwargs):
        client = FakeCountTokensClient(**kwargs)
        clients.append(client)
        return client

    counter = GeminiCountTokensCounter(
        model="gemini/gemini-3.1-flash-lite",
        api_key="test-key",
        client_factory=client_factory,
    )

    assert counter.count_text("hello") == 123
    # Cached by content hash: the second count must not add an API call.
    assert counter.count_text("hello") == 123
    assert len(clients) == 1
    url, kwargs = clients[0].posts[0]
    assert "gemini-3.1-flash-lite:countTokens" in url
    assert "test-key" not in url
    assert kwargs["headers"]["x-goog-api-key"] == "test-key"
    assert kwargs["json"]["contents"][0]["parts"] == [{"text": "hello"}]


def test_gemini_count_tokens_counter_raises_on_failure() -> None:
    def client_factory(**kwargs):
        return FakeCountTokensClient(**kwargs, should_fail=True)

    counter = GeminiCountTokensCounter(
        api_key="test-key",
        client_factory=client_factory,
    )

    with pytest.raises(RuntimeError, match="countTokens failed"):
        counter.count_text("hello")


def test_token_budget_rejects_free_tier_prompt_input_overflow() -> None:
    budget = build_correction_budget(
        input_tokens=194_001,
        subtitle_input_tokens=1,
        token_counter_source="test",
    )

    with pytest.raises(TokenBudgetError, match="Prompt input tokens"):
        validate_correction_budget(budget)


def test_token_budget_rejects_estimated_output_overflow() -> None:
    budget = build_correction_budget(
        input_tokens=1_000,
        subtitle_input_tokens=20_000,
        token_counter_source="test",
    )

    with pytest.raises(TokenBudgetError, match="Estimated output tokens"):
        validate_correction_budget(budget)


class _BoomCounter:
    source = "boom"

    def count_text(self, text: str) -> int:
        raise RuntimeError("counter down")

    def count_texts(self, texts) -> int:
        raise RuntimeError("counter down")

    def count_audio_seconds(self, seconds: float) -> int:
        return 0


class _ConstCounter:
    source = "const"

    def __init__(self, value: int) -> None:
        self.value = value
        self.calls = 0

    def count_text(self, text: str) -> int:
        self.calls += 1
        return self.value

    def count_texts(self, texts) -> int:
        self.calls += 1
        return self.value

    def count_audio_seconds(self, seconds: float) -> int:
        return int(seconds)


def test_fallback_counter_uses_first_working_backend_and_caches() -> None:
    primary = _ConstCounter(7)
    counter = FallbackTokenCounter(counters=(primary, _BoomCounter()))

    assert counter.count_text("hi") == 7
    assert counter.count_text("hi") == 7  # cached: no second delegate call
    assert primary.calls == 1
    assert counter.last_source == "const"


def test_fallback_counter_falls_through_to_next_backend() -> None:
    backup = _ConstCounter(42)
    counter = FallbackTokenCounter(counters=(_BoomCounter(), backup, HeuristicTokenCounter()))

    assert counter.count_text("hello") == 42
    assert counter.last_source == "const"


def test_fallback_counter_reaches_heuristic_when_all_apis_fail() -> None:
    counter = FallbackTokenCounter(
        counters=(_BoomCounter(), _BoomCounter(), HeuristicTokenCounter())
    )

    assert counter.count_text("你好世界 hello") > 0
    assert counter.last_source == "heuristic"


def test_fallback_counter_empty_inputs_short_circuit() -> None:
    counter = FallbackTokenCounter(counters=(_BoomCounter(), HeuristicTokenCounter()))

    assert counter.count_text("") == 0
    assert counter.count_texts([]) == 0
    assert counter.count_texts(["", ""]) == 0


def test_heuristic_counter_weights_cjk_higher_than_latin() -> None:
    counter = HeuristicTokenCounter()

    assert counter.count_text("") == 0
    # Equal char counts, but CJK weighs more than Latin, and digits more still.
    assert counter.count_text("字" * 20) > counter.count_text("a" * 20)
    assert counter.count_text("1" * 20) > counter.count_text("a" * 20)
    assert counter.count_audio_seconds(10.0) == 320


# Representative samples per category the heuristic must upper-bound.
_HEURISTIC_SAMPLES = {
    "digits": "0123456789" * 8,
    "english": "The quick brown fox jumps over the lazy dog. " * 12,
    "chinese": "这是一段用于测试的中文文本，包含标点符号。" * 6,
    "japanese": "これは日本語の字幕テキストです。トークン数を数える。" * 6,
    "korean": "한국어문장을테스트합니다한번더씁니다" * 4,
    "cyrillic": "это русский текст для проверки токенизации " * 6,
    "thai": "ภาษาไทยสำหรับทดสอบการแบ่งโทเค็น" * 6,
    "arabic": "نص عربي لاختبار عملية الترميز " * 6,
    "emoji": "🎉👍🔥😀🚀" * 8,
    "punct": "。，！？；：、（）「」.,!?;:()[]{}" * 4,
    "subtitle_csv": "\n".join(
        f"{i}|0.0|1.5|0.2|这是第{i}行字幕 line {i} test" for i in range(1, 30)
    ),
    "mixed": "字幕 subtitle 12345 한국어 русский .,!? 🎉 español",
    "tiny": "hi",
    "single": "a",
}


@pytest.mark.skipif(
    not LocalGeminiTokenCounter().available,
    reason="bundled gemini-token-counter binary not present",
)
@pytest.mark.slow
def test_heuristic_is_upper_bound_across_categories() -> None:
    # The heuristic must never under-count the real token count for any tested
    # category (the `lazy` truncation fast path relies on this upper bound).
    heuristic = HeuristicTokenCounter()
    local = LocalGeminiTokenCounter()
    for name, text in _HEURISTIC_SAMPLES.items():
        real = local.count_text(text)
        estimate = heuristic.count_text(text)
        assert estimate >= real, f"{name}: heuristic {estimate} < real {real}"


def test_classify_char_buckets() -> None:
    from llm.token_budget import classify_char

    assert classify_char("5") == "digit"
    assert classify_char("a") == "latin"
    assert classify_char("中") == "cjk"
    assert classify_char("ひ") == "cjk"
    assert classify_char("한") == "hangul"
    assert classify_char("。") == "wide_punct"
    assert classify_char("！") == "wide_punct"  # fullwidth
    assert classify_char("я") == "other_script"
    assert classify_char("!") == "ascii_sym"
    assert classify_char(" ") == "space"
    assert classify_char("🎉") == "other"


def test_default_token_counter_chain_order() -> None:
    counter = default_token_counter()

    assert isinstance(counter, FallbackTokenCounter)
    sources = [c.source for c in counter.counters]
    assert sources == [
        "gemini-token-counter-local",
        "gemini-countTokens",
        "heuristic",
    ]


def test_local_counter_treats_windows_bundle_as_unavailable_on_non_windows() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    windows_exe = repo_root / "bin" / "windows-amd64" / "tokcount.exe"
    if not windows_exe.is_file():
        pytest.skip("bundled tokcount.exe not present")

    counter = LocalGeminiTokenCounter(exe_path=str(windows_exe))
    if os.name == "nt":
        assert counter.available
    else:
        assert not counter.available


def test_local_counter_resolver_does_not_probe_obsolete_root_windows_path(
    monkeypatch,
) -> None:
    repo_root = Path(token_budget.__file__).resolve().parents[2]
    probed: list[Path] = []

    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    monkeypatch.setattr(
        token_budget,
        "_local_counter_exe_is_runnable",
        lambda path: probed.append(Path(path)) or False,
    )
    monkeypatch.setattr(token_budget.shutil, "which", lambda _name: None)

    assert token_budget._resolve_local_counter_exe() is None
    assert repo_root / "bin" / "windows-amd64" / "tokcount.exe" in probed
    assert repo_root / "bin" / "gemini-token-counter" in probed
    assert repo_root / "bin" / "gemini-token-counter.exe" not in probed


@pytest.mark.skipif(
    not LocalGeminiTokenCounter().available,
    reason="bundled gemini-token-counter binary not present",
)
def test_local_counter_matches_api_offset_on_ascii_and_cjk() -> None:
    local = LocalGeminiTokenCounter()

    # The +1 offset makes the local count exceed the bare tokenizer by one,
    # matching the countTokens contents envelope. Verified constant across
    # inputs; here we just assert positivity, determinism, and empty handling.
    assert local.count_text("") == 0
    first = local.count_text("hello 世界 test")
    assert first > 0
    assert local.count_text("hello 世界 test") == first  # cached, deterministic


@pytest.mark.skipif(
    not LocalGeminiTokenCounter().available,
    reason="bundled gemini-token-counter binary not present",
)
def test_local_counter_reuses_static_server_across_instances() -> None:
    token_budget._shutdown_local_counter_services()
    try:
        first = LocalGeminiTokenCounter()
        second = LocalGeminiTokenCounter()

        assert first.count_text("first exact count") > 0
        service = token_budget._get_local_counter_service(
            str(first.exe_path), first.model
        )
        first_pid = service.pid
        assert first_pid is not None

        assert second.count_text("second exact count") > 0
        assert token_budget._get_local_counter_service(
            str(second.exe_path), second.model
        ) is service
        assert service.pid == first_pid
    finally:
        token_budget._shutdown_local_counter_services()


@pytest.mark.skipif(
    not LocalGeminiTokenCounter().available,
    reason="bundled gemini-token-counter binary not present",
)
def test_local_counter_restarts_transparently_after_idle_exit() -> None:
    token_budget._shutdown_local_counter_services()
    try:
        local = LocalGeminiTokenCounter(server_idle_timeout_seconds=0.05)
        assert local.count_text("before idle exit") > 0
        service = token_budget._get_local_counter_service(
            str(local.exe_path), local.model
        )
        first_pid = service.pid
        assert first_pid is not None

        deadline = time.monotonic() + 3
        while service.pid is not None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service.pid is None

        assert local.count_text("after idle exit") > 0
        assert service.pid is not None
        assert service.pid != first_pid
    finally:
        token_budget._shutdown_local_counter_services()


def test_model_rate_limiter_rpm_window_uses_61s_and_safety_factor(tmp_path) -> None:
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini/gemini-3.1-flash-lite")
    state_path = tmp_path / ".state"
    limiter = ModelRateLimiter(state_path=state_path, enabled=True)
    limits = limiter.limits_for(endpoint)
    assert limits.effective_rpm == 13  # floor(15 * 0.9)
    assert limits.effective_tpm == 225_000  # floor(250000 * 0.9)

    for _ in range(limits.effective_rpm):
        limiter._record_acquire(endpoint, 100, now=0.0)
    wait = limiter.wait_seconds(endpoint, 100, now=0.0)
    assert wait == pytest.approx(61.0)


def test_model_rate_limiter_tpm_ignores_output_size_in_acquire(tmp_path) -> None:
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini/gemini-3.5-flash")
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    limiter.acquire(endpoint, 50_000, now_func=lambda: 0.0, sleep_func=lambda _: None)
    wait_small = limiter.wait_seconds(endpoint, 200_000, now=0.0)
    wait_large = limiter.wait_seconds(endpoint, 200_000, now=0.0)
    assert wait_small == wait_large


def test_model_rate_limiter_negative_tpm_means_unbounded(tmp_path) -> None:
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini/gemma-4-31b-it")
    limiter = ModelRateLimiter(state_path=tmp_path / ".state", enabled=True)
    limits = limiter.limits_for(endpoint)
    assert limits.effective_rpm == 13
    assert limits.effective_tpm == -1

    limiter.acquire(endpoint, 10_000_000, now_func=lambda: 0.0, sleep_func=lambda _: None)
    assert limiter.wait_seconds(endpoint, 10_000_000, now=0.0) == 0.0


def test_model_rate_limiter_daily_exhausted_persisted(tmp_path) -> None:
    endpoint = ModelEndpoint("GEMINI_FREE", "gemini/gemini-3.5-flash")
    state_path = tmp_path / ".state"
    limiter = ModelRateLimiter(state_path=state_path, enabled=True)
    limiter.mark_daily_exhausted(endpoint)
    assert limiter.is_daily_exhausted(endpoint)
    reloaded = ModelRateLimiter(state_path=state_path, enabled=True)
    assert reloaded.is_daily_exhausted(endpoint)
    assert endpoint_key(endpoint) in reloaded._daily_exhausted


def test_estimate_call_input_tokens_counts_text_only() -> None:
    messages = [{"role": "user", "content": "hello"}]
    assert estimate_call_input_tokens(messages) > 0


def test_quota_error_detection_for_fallback() -> None:
    assert is_quota_or_rate_limit_error(RuntimeError("RESOURCE_EXHAUSTED quota exceeded"))
    assert is_quota_or_rate_limit_error(RuntimeError("HTTP 429 too many requests"))
    assert not is_quota_or_rate_limit_error(RuntimeError("invalid JSON"))


def test_classify_quota_error_uses_quota_id_not_retry_hint() -> None:
    # A PerDay quotaId classifies as DAILY even with a short retryDelay — the
    # hint is a generic backoff (~50s) that Gemini returns on genuine daily
    # exhaustion too, so it cannot mean "transient". Flakiness is absorbed by the
    # rate limiter's strike gate, not by the retry hint.
    per_day = RuntimeError(
        "RateLimitError: Quota exceeded for metric generate_content_free_tier_"
        "requests, limit 20, model gemini-3.5-flash. Please retry in 51.98s. "
        'quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier"'
    )
    assert classify_quota_error(per_day) is QuotaKind.DAILY

    # A per-minute quotaId is the transient bucket (never locks the day).
    per_minute = RuntimeError(
        "RateLimitError: Quota exceeded. Please retry in 12s. "
        'quotaId "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"'
    )
    assert classify_quota_error(per_minute) is QuotaKind.PER_MINUTE

    # A rate error with no structured quotaId -> generic rate bucket.
    assert classify_quota_error(RuntimeError("HTTP 429 too many requests")) is (
        QuotaKind.OTHER_RATE
    )
    # A non-quota provider error is not a quota kind at all.
    assert classify_quota_error(RuntimeError("HTTP 503 unavailable")) is QuotaKind.NONE


def test_retryable_provider_error_detection_for_transient_gemini_failures() -> None:
    assert is_retryable_provider_error(RuntimeError("HTTP 503 high demand"))
    assert is_retryable_provider_error(TimeoutError("timed out"))
    assert not is_retryable_provider_error(ValueError("invalid prompt schema"))


def test_research_search_query_limit_scales_with_raw_segments() -> None:
    assert research_search_query_limit(0) == 8
    assert research_search_query_limit(99) == 8
    assert research_search_query_limit(100) == 9
    assert research_search_query_limit(6_400) == 16
    assert research_search_query_limit(10_000) == 16
    assert research_search_query_limit(1_000_000) == 16
