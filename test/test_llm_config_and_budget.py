from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

from finesub import paths
import finesub.llm.token_budget as token_budget
from finesub.llm.client import (
    QuotaKind,
    classify_quota_error,
    is_quota_or_rate_limit_error,
    is_retryable_provider_error,
)
from finesub.llm.routing.config import (
    DEFAULT_LIMITS,
    GEMINI_25_FLASH,
    GEMINI_31_FLASH_LITE,
    GEMINI_35_FLASH,
    GEMINI_35_FLASH_LITE,
    GEMINI_38_FLASH,
    GEMINI_37_FLASH,
    GEMINI_36_FLASH,
    GEMINI_FREE_TIER,
    LLMRole,
    ModelEndpoint,
    default_role_configs,
    research_search_query_limit,
    thinking_budget_for_level,
)
from finesub.llm.routing.model_catalog import (
    CATALOG_COLUMNS,
    default_model_catalog,
    get_model_catalog_entry,
    get_model_catalog_entry_for_tier,
)
from finesub.llm.rate_limit import ModelRateLimiter, endpoint_key, estimate_call_input_tokens
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.token_budget import (
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


def test_default_role_configs_use_expected_endpoint_chains() -> None:
    configs = default_role_configs()
    free_36 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_36_FLASH)
    free_35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH)
    free_lite35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH_LITE)

    # Correction role (audio_multimodal): 3.8 → 3.7 → 3.6 → 3.5. The lites left
    # the high cell (no silent quality downgrade); they live in /intermediate.
    audio = configs[LLMRole.AUDIO_MULTIMODAL]
    assert [
        (ep.provider_tier, ep.api_model_id)
        for ep in audio.endpoints(test_profile=True)
    ] == [(free_lite35.provider_tier, free_lite35.api_model_id)]
    correction_models = [ep.api_model_id for ep in audio.endpoints(test_profile=False)]
    assert correction_models[:4] == [
        GEMINI_37_FLASH,
        GEMINI_38_FLASH,
        GEMINI_36_FLASH,
        GEMINI_35_FLASH,
    ]
    assert GEMINI_35_FLASH_LITE not in correction_models
    from finesub.llm.routing.config import role_config_for

    # /intermediate is the lite cell, free before paid. 3.1 Flash Lite left the
    # chain on 2026-08-14: two lites of the same generation ahead of the paid
    # key bought nothing but a second way to be rate-limited.
    med_models = [
        ep.api_model_id
        for ep in role_config_for("correction-mm", "intermediate").endpoints(
            test_profile=False
        )
    ]
    assert med_models == [GEMINI_35_FLASH_LITE, GEMINI_35_FLASH_LITE]
    assert GEMINI_31_FLASH_LITE not in med_models

    # General capable (research): free 3.6 → 3.5 → 3.8 → 3.7 → paid 3.8 → 3.7. No lite fallback
    # since the 2026-08-11 acceptance edit -- floor-quality work stops
    # instead of degrading.
    general = configs[LLMRole.GENERAL_CAPABLE]
    general_models = [ep.api_model_id for ep in general.endpoints(test_profile=False)]
    assert general_models == [
        GEMINI_36_FLASH,
        GEMINI_35_FLASH,
        GEMINI_37_FLASH,
        GEMINI_38_FLASH,
        GEMINI_37_FLASH,
        GEMINI_38_FLASH,
    ]
    assert GEMINI_35_FLASH_LITE not in general_models

    # Lightweight roles prefer 3.5 Flash Lite (纠错 r1 MM + search-loop text);
    # 3.1-lite left the group in the same acceptance edit.
    lightweight_mm = configs[LLMRole.LIGHTWEIGHT_MULTIMODAL]
    lite_models = [
        ep.api_model_id for ep in lightweight_mm.endpoints(test_profile=False)
    ]
    assert lite_models == [GEMINI_35_FLASH_LITE, GEMINI_35_FLASH_LITE]
    assert GEMINI_31_FLASH_LITE not in lite_models
    assert GEMINI_36_FLASH not in lite_models
    assert lightweight_mm.thinking_level == "medium"
    assert configs[LLMRole.AUDIO_MULTIMODAL].thinking_level == "medium"
    # Native search is a per-call capability now (plan v2 D4): the packaged
    # native-capable group exists but is not bound by the default preset.
    from finesub.llm.routing.model_routes import default_model_routes

    routes = default_model_routes()
    native_group = routes.model_groups["gemini-native-search"]
    # Free entry point is 2.5 Flash; the paid fallback is the full-Flash native
    # targets, 3.7 ahead of 3.8 like every other list (owner 2026-09-03).
    assert [
        routes.target_fact(target_id).api_model_id
        for target_id in native_group.target_ids
    ] == [GEMINI_25_FLASH, GEMINI_37_FLASH, GEMINI_38_FLASH]
    assert all(
        routes.target_profile(target_id).native_search_tool == "google_search"
        for target_id in native_group.target_ids
    )


def test_model_catalog_loads_gemini_tier_psv_facts() -> None:
    from finesub.llm.routing.model_catalog import default_model_catalog

    default_model_catalog.cache_clear()
    entries = default_model_catalog()
    free_entries = [e for e in entries if e.provider_tier == "GEMINI_FREE"]
    paid_entries = [e for e in entries if e.provider_tier == "GEMINI_PAID"]
    local_entries = [e for e in entries if e.provider_tier == "LOCAL_CODEX"]
    # 3.0 Flash Preview left the roster in the 2026-08-11 acceptance edit;
    # 3.8 Flash joined both tiers on release (2026-09-02); free keeps 3.8, 3.7,
    # 3.6 and 3.5.
    assert len(free_entries) == 8
    assert len(paid_entries) == 3
    assert [(entry.fact_id, entry.api_model_id) for entry in local_entries] == [
        ("local-codex-gpt-5_6-luna", "gpt-5.6-luna"),
        ("local-codex-gpt-5_6-terra", "gpt-5.6-terra"),
        ("local-codex-gpt-5_6-sol", "gpt-5.6-sol"),
    ]
    # By fact_id, not by position: this list is one people insert into, and an
    # index would quietly start checking the neighbour's row instead of failing.
    codex = {entry.fact_id: entry for entry in local_entries}
    assert codex["local-codex-gpt-5_6-luna"].thinking_levels == (
        "xhigh",
        "high",
        "medium",
    )
    assert codex["local-codex-gpt-5_6-luna"].quality_score == 70
    # Terra takes luna's ladder because its own default reasoning level is
    # `medium` too (sol's is `low`). The score is the owner's call (2026-09-03):
    # between Sonnet 5 (77) and Opus 5 (88), which also keeps it between its own
    # siblings luna (70) and sol (90) -- the vendor catalog orders those three by
    # `priority`, which supports the ordering and not the number. Unmeasured.
    assert codex["local-codex-gpt-5_6-terra"].thinking_levels == (
        "xhigh",
        "high",
        "medium",
    )
    assert codex["local-codex-gpt-5_6-terra"].quality_score == 82
    sonnet = next(
        e for e in entries if e.fact_id == "local-claude-sonnet-5"
    ).quality_score
    opus = next(e for e in entries if e.fact_id == "local-claude-opus-5").quality_score
    assert sonnet < codex["local-codex-gpt-5_6-terra"].quality_score < opus
    assert codex["local-codex-gpt-5_6-sol"].thinking_levels == ("high", "high", "low")
    assert codex["local-codex-gpt-5_6-sol"].quality_score == 90
    agy = get_model_catalog_entry_for_tier(
        "gemini-3.8-flash", "LOCAL_AGY"
    )
    assert agy is not None
    assert agy.fact_id == "local-agy-gemini-3_8-flash"
    # 3.8 Flash caps its top level instead of mapping identity (2026-09-03):
    # at the same nominal effort it emitted ~1.5x the reasoning 3.7 did, which
    # shares the 65536 output cap with the subtitles themselves. Only the high
    # cell moves: the first shape of this change also pushed abstract medium
    # down to `low`, and a measured 5-pair comparison showed correction (which
    # sits at abstract medium) then emitted zero reasoning tokens in 4 of 5
    # windows -- `low` hands the decision to the model rather than trimming it.
    assert agy.thinking_levels == ("medium", "medium", "low")
    # …and the mapping is what a caller actually gets back, per abstract level.
    from finesub.llm.routing.model_catalog import thinking_value_for

    assert [thinking_value_for(agy, level) for level in ("high", "medium", "low")] == [
        "medium",
        "medium",
        "low",
    ]
    assert agy.supports_audio is True
    assert agy.supports_video is True
    # True since 2026-08-15: agy's own `search_web` is entitled by the second
    # (native) project. The *fact* says the model can ground; whether a given
    # call may is the target's declared tool, which only the native target has.
    assert agy.supports_native_search is True
    assert agy.video_high_resolution_only is True
    assert agy.quality_score == 75
    gemma4 = get_model_catalog_entry_for_tier("gemini/gemma-4-31b-it", "GEMINI_FREE")
    assert gemma4 is not None
    assert gemma4.max_input_tokens == 16_000
    assert gemma4.max_output_tokens == 32_768
    assert gemma4.rpm == 15
    assert gemma4.tpm == -1
    assert gemma4.rpd == 1500
    assert gemma4.supports_native_search is True
    non_gemma_entries = [
        e
        for e in entries
        if e.api_model_id != "gemini/gemma-4-31b-it"
        and e.provider_tier != "LOCAL_WORKBUDDY"
    ]
    # Not one number: the catalog states each vendor's real ceiling —
    # DeepSeek 256k, Opus/Sonnet 128k, Haiku 64k (owner-confirmed; the old
    # uniform 65,536 OVERSTATED Haiku, which mattered because max_output is
    # the truncation denominator). What the guard is for is a row that
    # forgot to say anything.
    assert all(entry.max_output_tokens >= 64_000 for entry in non_gemma_entries)
    # WorkBuddy's ceilings are genuinely below that floor and are exempted
    # rather than rounded up: each was read off the CLI's own `modelUsage`
    # (2026-09-04). Pinned by value so the exemption cannot become a place a
    # forgotten row hides -- CATALOG_WINDOWS carries the same three numbers.
    assert {
        e.fact_id: e.max_output_tokens
        for e in entries
        if e.provider_tier == "LOCAL_WORKBUDDY"
    } == {
        "local-workbuddy-hy3": 64_000,
        "local-workbuddy-hy4": 64_000,
        "local-workbuddy-glm-5_3-flash": 32_000,
        "local-workbuddy-deepseek-v4-flash": 50_000,
        "local-workbuddy-deepseek-v4-pro": 50_000,
    }
    lite = get_model_catalog_entry_for_tier(
        "gemini/gemini-3.1-flash-lite", "GEMINI_FREE"
    )
    assert lite is not None
    # Packaged Gemini rows carry the identity thinking mapping.
    assert lite.thinking_levels == ("high", "medium", "low")
    assert lite.rpm == 15
    assert lite.rpd == 500
    # New models share their predecessor's limits (3.7 == 3.6 == 3.5 Flash;
    # 3.5 Flash Lite == 3.1 Flash Lite). Older capable models remain as
    # free-tier fallback rows.
    flash37 = get_model_catalog_entry_for_tier("gemini/gemini-3.7-flash", "GEMINI_FREE")
    flash36 = get_model_catalog_entry_for_tier("gemini/gemini-3.6-flash", "GEMINI_FREE")
    flash35 = get_model_catalog_entry_for_tier("gemini/gemini-3.5-flash", "GEMINI_FREE")
    assert flash37 is not None and flash36 is not None and flash35 is not None
    assert (flash37.rpm, flash37.tpm, flash37.rpd, flash37.tpd) == (
        flash36.rpm,
        flash36.tpm,
        flash36.rpd,
        flash36.tpd,
    )
    assert (flash36.rpm, flash36.tpm, flash36.rpd, flash36.tpd) == (
        flash35.rpm,
        flash35.tpm,
        flash35.rpd,
        flash35.tpd,
    )
    lite35 = get_model_catalog_entry_for_tier(
        "gemini/gemini-3.5-flash-lite", "GEMINI_FREE"
    )
    lite31 = get_model_catalog_entry_for_tier(
        "gemini/gemini-3.1-flash-lite", "GEMINI_FREE"
    )
    assert lite35 is not None and lite31 is not None
    assert (lite35.rpm, lite35.tpm, lite35.rpd, lite35.tpd) == (
        lite31.rpm,
        lite31.tpm,
        lite31.rpd,
        lite31.tpd,
    )
    for paid in paid_entries:
        assert (paid.rpm, paid.tpm, paid.rpd) == (1_000, 4_000_000, -1)
    flash25 = get_model_catalog_entry("gemini/gemini-2.5-flash")
    assert flash25 is not None
    assert flash25.supports_native_search is True
    assert all(entry.fact_id for entry in entries)
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


#: Every packaged row's three window numbers, and the input envelope they imply.
#: Written out rather than recomputed from the file: this table *is* what
#: `docs/plans/model-window-limits-plan.md` §4 was audited against, so it has to fail
#: when the file changes rather than follow it.
#:
#: A blank `context_window` in the file means "input and output are independent
#: pools" and is stored as their sum, which is why every such row's envelope is
#: exactly its `max_input_tokens` -- the joint constraint cannot bind there.
CATALOG_WINDOWS = {
    # single pool: the answer is spent out of the same budget as the prompt
    "local-codex-gpt-5_6-luna": (272_000, 272_000, 65_536, 206_464),
    "local-codex-gpt-5_6-terra": (272_000, 272_000, 65_536, 206_464),
    "local-codex-gpt-5_6-sol": (272_000, 272_000, 65_536, 206_464),
    "local-claude-opus-5": (1_000_000, 1_000_000, 128_000, 872_000),
    "local-claude-sonnet-5": (1_000_000, 1_000_000, 128_000, 872_000),
    "local-claude-haiku-4_5": (200_000, 200_000, 64_000, 136_000),
    "local-agy-gemini-3_8-flash": (1_048_576, 1_048_576, 65_536, 983_040),
    "local-agy-gemini-3_7-flash": (1_048_576, 1_048_576, 65_536, 983_040),
    "gemini-paid-3_8-flash": (1_048_576, 1_048_576, 65_536, 983_040),
    "gemini-paid-3_7-flash": (1_048_576, 1_048_576, 65_536, 983_040),
    "gemini-paid-3_5-flash-lite": (1_048_576, 1_048_576, 65_536, 983_040),
    "local-dsh-deepseek-v4-flash": (1_000_000, 1_000_000, 256_000, 744_000),
    "local-dsh-deepseek-v4-pro": (1_000_000, 1_000_000, 256_000, 744_000),
    # Owner-supplied (2026-09-04), and they are NOT the CLI's self-report:
    # `modelUsage.contextWindow` answers 1000000 for every model on this tier,
    # which `hy3` shows to be a placeholder -- there the CLI says 192000/64000
    # and the owner's numbers agree exactly. The rest take the owner's.
    "local-workbuddy-hy3": (192_000, 192_000, 64_000, 128_000),
    "local-workbuddy-hy4": (300_000, 300_000, 64_000, 236_000),
    # 32000, not 64000: the CLI's own `modelUsage` said so from the first
    # probe and the owner confirmed it (2026-09-04). It matters beyond
    # planning -- the worker is told this number, and a wrong one is why a
    # `high`-effort window reasoned past the ceiling instead of pausing
    # (docs/llm_local_agent.md §12.1.5).
    "local-workbuddy-glm-5_3-flash": (300_000, 300_000, 32_000, 268_000),
    "local-workbuddy-deepseek-v4-flash": (300_000, 300_000, 50_000, 250_000),
    "local-workbuddy-deepseek-v4-pro": (300_000, 300_000, 50_000, 250_000),
    # independent pools: `context_window` left blank in the file
    "local-agy-opus-4_6": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_8-flash": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_7-flash": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_6-flash": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_5-flash": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_5-flash-lite": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-3_1-flash-lite": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-2_5-flash": (259_536, 194_000, 65_536, 194_000),
    "gemini-free-gemma-4-31b": (48_768, 16_000, 32_768, 16_000),
    "local-conversational-agent": (259_536, 194_000, 65_536, 194_000),
}


def test_every_catalog_row_states_its_window_and_derives_its_envelope() -> None:
    """The three numbers per row, and what the planner makes of them.

    `context_window` exists because `max_input_tokens` used to answer two
    questions at once -- "how big is the pool" for the planning envelope and
    "what will the API accept" at dispatch -- and on a single-pool provider
    those want different values.
    """

    default_model_catalog.cache_clear()
    entries = {entry.fact_id: entry for entry in default_model_catalog()}

    assert set(entries) == set(CATALOG_WINDOWS), "a catalog row was added or removed"
    for fact_id, (ctx, max_in, max_out, envelope) in CATALOG_WINDOWS.items():
        entry = entries[fact_id]
        assert (
            entry.context_window,
            entry.max_input_tokens,
            entry.max_output_tokens,
        ) == (ctx, max_in, max_out), fact_id
        # The loader's invariant, restated where a reader of the table sees it.
        assert entry.context_window >= max(
            entry.max_input_tokens, entry.max_output_tokens
        ), fact_id
        assert (
            min(entry.max_input_tokens, entry.context_window - entry.max_output_tokens)
            == envelope
        ), fact_id


def test_a_context_window_below_either_half_is_a_declaration_error(
    tmp_path: Path,
) -> None:
    # Not clamped: a pool smaller than the halves it must hold is someone
    # mistyping a number, and planning against a window the provider does not
    # have is the failure this column was added to prevent.
    from finesub.llm.routing.model_catalog import (
        CATALOG_FILENAME,
        load_model_catalog,
    )

    override = tmp_path / CATALOG_FILENAME
    override.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens"
        "|context_window\n"
        "bad|GEMINI_FREE|gemini/x|200000|65536|100000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_window"):
        load_model_catalog(override)


def test_an_explicit_zero_context_window_is_not_read_as_blank(tmp_path: Path) -> None:
    # Blank is a property of the *cell*, not of the value. Reading `0` as absent
    # would hand the row the widest pool a typo can ask for -- the opposite of
    # what the person typing a zero meant.
    from finesub.llm.routing.model_catalog import (
        CATALOG_FILENAME,
        load_model_catalog,
    )

    override = tmp_path / CATALOG_FILENAME
    override.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens"
        "|context_window\n"
        "zero|GEMINI_FREE|gemini/x|200000|65536|0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="context_window"):
        load_model_catalog(override)


def test_a_non_positive_output_limit_is_a_declaration_error(tmp_path: Path) -> None:
    # `max_output_tokens` is a denominator twice over: the envelope subtracts it
    # from the pool and the truncation check divides by it. Zero would make the
    # first give back the whole window and the second meaningless, both far from
    # the row that caused it.
    from finesub.llm.routing.model_catalog import (
        CATALOG_FILENAME,
        load_model_catalog,
    )

    override = tmp_path / CATALOG_FILENAME
    override.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens\n"
        "empty-output|GEMINI_FREE|gemini/x|200000|0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_output_tokens"):
        load_model_catalog(override)


def test_a_blank_context_window_means_two_independent_pools(tmp_path: Path) -> None:
    from finesub.llm.routing.model_catalog import (
        CATALOG_FILENAME,
        load_model_catalog,
    )

    override = tmp_path / CATALOG_FILENAME
    override.write_text(
        "fact_id|provider_tier|api_model_id|max_input_tokens|max_output_tokens\n"
        "split|GEMINI_FREE|gemini/x|200000|65536\n",
        encoding="utf-8",
    )

    (entry,) = load_model_catalog(override)

    assert entry.context_window == 200_000 + 65_536
    # Which is the point: the joint constraint gives back the input limit
    # untouched, so a two-pool provider needs no cell at all.
    assert (
        min(entry.max_input_tokens, entry.context_window - entry.max_output_tokens)
        == 200_000
    )


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

    # efficiency lost its coefficient discount: c is 3.5 like quality.
    text_low = build_correction_budget(
        input_tokens=20_000,
        subtitle_input_tokens=1_000,
        token_counter_source="test",
        profile=resolve_profile("text", "none", "efficiency"),
    )
    assert text_low.estimated_output_tokens == 3_500


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
    reason="bundled tokcount binary not present",
)
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
    from finesub.llm.token_budget import classify_char

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
        "tokcount-local",
        "gemini-countTokens",
        "heuristic",
    ]


def test_agent_only_token_counter_never_includes_api(monkeypatch) -> None:
    from finesub.llm.routing.execution_policy import ExecutionSettings

    monkeypatch.setattr(
        "finesub.llm.routing.execution_policy.load_execution_settings",
        lambda: ExecutionSettings(policy_id="agent-only"),
    )

    sources = [counter.source for counter in default_token_counter().counters]

    assert sources == ["tokcount-local", "heuristic"]


def test_injected_agent_only_settings_override_global_counter_policy(
    monkeypatch,
) -> None:
    from finesub.llm.routing.execution_policy import ExecutionSettings

    monkeypatch.setattr(
        "finesub.llm.routing.execution_policy.load_execution_settings",
        lambda: ExecutionSettings(policy_id="api-only"),
    )

    sources = [
        counter.source
        for counter in default_token_counter(
            execution_settings=ExecutionSettings(policy_id="agent-only")
        ).counters
    ]

    assert sources == ["tokcount-local", "heuristic"]


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


@pytest.mark.requires_main_checkout
def test_local_counter_resolver_does_not_probe_obsolete_root_windows_path(
    monkeypatch,
) -> None:
    # The resolver under test, not a parent count off `token_budget.__file__`:
    # that was `parents[2]` while the module sat at `src/llm/`, and the 2026-08
    # move to `src/finesub/llm/` silently made it point at `src/`. Nothing said
    # so for a whole branch -- `requires_main_checkout` skips this in the
    # worktree where the rename was done.
    repo_root = paths.resolve_checkout_root()
    assert repo_root is not None, "this test only means anything in a checkout"
    probed: list[Path] = []

    monkeypatch.delenv("GEMINI_TOKEN_COUNTER_EXE", raising=False)
    monkeypatch.setattr(
        token_budget,
        "_local_counter_exe_is_runnable",
        lambda path: probed.append(Path(path)) or False,
    )
    monkeypatch.setattr(
        token_budget.token_counter, "find_on_path", lambda: None
    )

    assert token_budget._resolve_local_counter_exe() is None
    assert probed == [repo_root / "bin" / "windows-amd64" / "tokcount.exe"]


@pytest.mark.skipif(
    not LocalGeminiTokenCounter().available,
    reason="bundled tokcount binary not present",
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
    reason="bundled tokcount binary not present",
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
    reason="bundled tokcount binary not present",
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
        limiter.reserve(endpoint, 100, now_func=lambda: 0.0)
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


def test_daily_strikes_do_not_survive_into_a_new_pacific_day(tmp_path) -> None:
    """The streak only ever cleared on success or on locking.

    Timestamps were appended and never read, so two isolated flickers on one
    day plus a single 429 days later -- against a fresh daily quota -- added up
    to a lock for the whole of that later day.
    """
    from finesub.llm.routing.config import GEMINI_FREE_TIER, ModelEndpoint
    from finesub.llm.rate_limit import ModelRateLimiter

    limiter = ModelRateLimiter(state_path=tmp_path / "state.json")
    endpoint = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.6-flash")

    monday = 1_000_000.0
    assert not limiter.note_daily_quota_hit(endpoint, key_id="k", now=monday)
    assert not limiter.note_daily_quota_hit(endpoint, key_id="k", now=monday + 60)

    # Four days later: a brand-new quota, so the first hit must not lock.
    thursday = monday + 4 * 24 * 3600
    assert not limiter.note_daily_quota_hit(endpoint, key_id="k", now=thursday)
    assert not limiter.is_daily_exhausted(endpoint, key_id="k")


def test_three_strikes_within_one_day_still_lock(tmp_path) -> None:
    """The gate itself is unchanged -- only stale days are dropped."""
    from finesub.llm.routing.config import GEMINI_FREE_TIER, ModelEndpoint
    from finesub.llm.rate_limit import ModelRateLimiter

    limiter = ModelRateLimiter(state_path=tmp_path / "state.json")
    endpoint = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.6-flash")

    now = 1_000_000.0
    assert not limiter.note_daily_quota_hit(endpoint, key_id="k", now=now)
    assert not limiter.note_daily_quota_hit(endpoint, key_id="k", now=now + 30)
    assert limiter.note_daily_quota_hit(endpoint, key_id="k", now=now + 90)
    assert limiter.is_daily_exhausted(endpoint, key_id="k")


def test_one_process_does_not_erase_another_processs_daily_lock(tmp_path) -> None:
    """The section was assigned wholesale from a snapshot taken at __init__.

    A limiter is `lru_cache`d for the life of the process, so a desktop app
    that started before a batch run would overwrite the lock that run had
    recorded -- and both would then keep spending retries on a dead key.
    """
    from finesub.llm.routing.config import GEMINI_FREE_TIER, ModelEndpoint
    from finesub.llm.rate_limit import ModelRateLimiter

    state = tmp_path / "state.json"
    endpoint = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.6-flash")
    other = ModelEndpoint(GEMINI_FREE_TIER, "gemini/gemini-3.5-flash")

    # A long-lived process, booted first.
    early = ModelRateLimiter(state_path=state)
    # A second process locks a key for the day.
    later = ModelRateLimiter(state_path=state)
    later.mark_daily_exhausted(endpoint, key_id="k1")

    # The first one writes something unrelated afterwards.
    early.note_daily_quota_hit(other, key_id="k2", now=1_000_000.0)

    fresh = ModelRateLimiter(state_path=state)
    assert fresh.is_daily_exhausted(endpoint, key_id="k1"), (
        "the other process's lock must survive"
    )


def test_the_counter_reports_which_backend_actually_answered() -> None:
    """Artifacts recorded the chain, not the result.

    A plan produced entirely by the heuristic -- whose weights are deliberate
    upper bounds -- was indistinguishable from an exact one, while the recorded
    planning metadata matched either way. The degradation is silent by design;
    the record should not be.
    """
    from finesub.llm.token_budget import FallbackTokenCounter, HeuristicTokenCounter

    class _Boom:
        source = "exact-but-broken"

        def count_text(self, text: str) -> int:
            raise RuntimeError("no binary here")

    counter = FallbackTokenCounter(counters=(_Boom(), HeuristicTokenCounter()))
    assert "+" in counter.source, "before any call, the chain is all we know"

    counter.count_text("字幕一行")

    assert counter.source == "heuristic"
    assert counter.last_source == "heuristic"


def test_the_output_reserve_is_not_a_cap() -> None:
    """Request and reserve are two numbers, and only one goes on the wire.

    `SESSION_OUTPUT_MAX_TOKENS` used to be sent as `max_tokens`, so a
    non-correction round that needed more than it was truncated by a harness
    constant nobody had calibrated. Since 2026-09-04 it is a *planning reserve*:
    it decides whether a candidate's input still fits beside the answer on a
    single-pool model, and the request fills that candidate's own ceiling.
    """

    from finesub.llm.client import _output_budget
    from finesub.llm.routing.config import SESSION_OUTPUT_MAX_TOKENS
    from finesub.llm.routing.model_catalog import get_model_catalog_entry_for_tier

    entry = get_model_catalog_entry_for_tier("gemini/gemini-3.7-flash", "GEMINI_FREE")
    assert entry is not None

    # The non-correction shape: reserve given, request unrationed.
    request, reserve = _output_budget(None, SESSION_OUTPUT_MAX_TOKENS, entry)
    assert request == entry.max_output_tokens
    assert reserve == SESSION_OUTPUT_MAX_TOKENS
    assert request > reserve, "the reserve must not become the cap again"

    # Neither given: both collapse onto the candidate's ceiling.
    assert _output_budget(None, None, entry) == (
        entry.max_output_tokens,
        entry.max_output_tokens,
    )
    # Only `max_tokens`: exactly the old behaviour, which is what keeps an
    # unconverted caller honest.
    assert _output_budget(4096, None, entry) == (4096, 4096)
    # No catalog row to consult (stub configs, tests): the documented default.
    assert _output_budget(None, None, None)[0] == DEFAULT_LIMITS.output_limit


def test_a_single_pool_request_is_clamped_to_what_context_is_left() -> None:
    """Unrationed still means "what fits" -- and only ever on the API path.

    The reserve check can pass a candidate whose own ceiling no longer fits
    beside this prompt; asking anyway is a provider 400, not a shorter answer.
    A row whose `context_window` is the blank-cell stand-in
    (`max_input + max_output`) cannot be clamped by construction, because its
    two halves are metered separately.
    """

    from dataclasses import replace

    from finesub.llm.client import _clamp_request_to_context, _output_budget
    from finesub.llm.routing.model_catalog import get_model_catalog_entry_for_tier

    single_pool = get_model_catalog_entry_for_tier("claude-opus-5", "LOCAL_CLAUDE")
    assert single_pool is not None
    assert single_pool.context_window == 1_000_000

    request, _reserve = _output_budget(None, None, single_pool)
    assert request == single_pool.max_output_tokens, "the budget itself is unclamped"

    # Room to spare: the ceiling is asked for in full.
    assert _clamp_request_to_context(request, single_pool, 10_000) == (
        single_pool.max_output_tokens
    )
    # Tight: the request drops to what is left, never below 1.
    assert _clamp_request_to_context(request, single_pool, 950_000) == 50_000
    assert _clamp_request_to_context(request, single_pool, 1_000_000) == 1
    # No row to consult, or nothing estimated: nothing to clamp against.
    assert _clamp_request_to_context(request, None, 950_000) == request
    assert _clamp_request_to_context(request, single_pool, 0) == request

    # Split pools (blank `context_window` -> max_input + max_output): the clamp
    # can never bite, whatever the input.
    split = replace(
        single_pool,
        max_input_tokens=194_000,
        max_output_tokens=65_536,
        context_window=194_000 + 65_536,
    )
    assert _clamp_request_to_context(65_536, split, 194_000) == 65_536


def test_the_session_reserve_is_decimal_and_below_every_shipped_ceiling() -> None:
    """32,000 -- decimal like `WINDOW_REFUSE_OUTPUT`, not 32Ki.

    It is compared against catalog columns that state vendor numbers, where a
    power of two only ever coincides by accident. It also has to stay at or
    under the smallest output ceiling any bound group declares, or the reserve
    alone would disqualify a healthy candidate.
    """

    from finesub.llm.routing.config import SESSION_OUTPUT_MAX_TOKENS
    from finesub.llm.routing.model_routes import default_model_routes

    assert SESSION_OUTPUT_MAX_TOKENS == 32_000

    routes = default_model_routes()
    smallest = min(
        routes.target_fact(target_id).max_output_tokens
        for group_id in routes.model_groups
        for target_id in routes.model_groups[group_id].target_ids
    )
    assert SESSION_OUTPUT_MAX_TOKENS <= smallest, (
        f"the reserve ({SESSION_OUTPUT_MAX_TOKENS}) exceeds the smallest output "
        f"ceiling in any model group ({smallest}); it would skip that candidate "
        "on the reserve alone"
    )
