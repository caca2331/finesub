"""Model role configuration for LLM subtitle correction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, Tuple


GEMINI_36_FLASH = "gemini/gemini-3.6-flash"
GEMINI_35_FLASH = "gemini/gemini-3.5-flash"
GEMINI_35_FLASH_LITE = "gemini/gemini-3.5-flash-lite"
GEMINI_31_FLASH_LITE = "gemini/gemini-3.1-flash-lite"
GEMINI_25_FLASH = "gemini/gemini-2.5-flash"
GEMINI_FREE_TIER = "GEMINI_FREE"
GEMINI_PAID_TIER = "GEMINI_PAID"


class LLMRole(str, Enum):
    # Correction windows (and fast-mode correction after r1): prefer 3.6 Flash.
    AUDIO_MULTIMODAL = "audio_multimodal"
    # Research r1/r2, fast round 1, post-task knowledge update, and other
    # non-correction work: prefer 3.5 Flash, then 3.6, then 3.5 Flash Lite.
    GENERAL_CAPABLE = "general_capable"
    # Search-loop judge ("查询"): prefer 3.5 Flash Lite (text).
    LIGHTWEIGHT = "lightweight"
    # Correction query round ("纠错 r1"): same 3.5-lite chain, multimodal role.
    LIGHTWEIGHT_MULTIMODAL = "lightweight_multimodal"
    # Pool for models with a built-in web-search tool (text-high route). The
    # Gemini free tier rejects google_search (429), so this role is meant to be
    # configured with a third-party/paid searchable model.
    INTERNET_CAPABLE = "internet_capable"


# Prompt tier derived from the answering model's catalog ``capability`` column:
# >=6 (3.5/3.0-flash) is CAPABLE, below (flash-lite/2.5-flash, gemma) is BASIC.
# The correction prompt is assembled per tier inside the endpoint loop so a
# fallback model never receives a prompt written for a stronger one: CAPABLE
# gets the judgment-based merge fragments, BASIC the conservative 1:1 variant
# (docs/llm_harness_behavior.md, docs/llm_prompts.md).
CAPABILITY_TIER_THRESHOLD = 6


class CapabilityTier(str, Enum):
    CAPABLE = "capable"
    BASIC = "basic"


def tier_for_capability(capability: int) -> CapabilityTier:
    return (
        CapabilityTier.CAPABLE
        if capability >= CAPABILITY_TIER_THRESHOLD
        else CapabilityTier.BASIC
    )


@dataclass(frozen=True)
class ModelLimits:
    context_limit: int = 256_000
    prompt_input_limit: int = 194_000
    output_limit: int = 65_536
    safety_margin: int = 1_000
    audio_tokens_per_second: int = 32


@dataclass(frozen=True)
class RateLimitPolicy:
    safety_factor: float = 0.9
    window_seconds: float = 61.0


@dataclass(frozen=True)
class ModelEndpoint:
    provider_tier: str
    litellm_model: str


@dataclass(frozen=True)
class RoleModelConfig:
    role: LLMRole
    endpoint_chain: Tuple[ModelEndpoint, ...]
    test_endpoint: ModelEndpoint
    # REST-path thinking level for gemini-3.x ("low"/"medium"/"high"; "" keeps
    # the model default).
    thinking_level: str = ""
    # Token-count thinking budget for models without thinkingLevel. 0 derives
    # it from thinking_level (see thinking_budget_for_level); budgets are no
    # longer maintained as standalone numbers.
    thinking_budget: int = 0
    # Provider-native web-search tool to enable on generation calls (e.g.
    # "google_search", "web_search"). Empty disables; only the
    # internet_capable role sets this, and the test profile never enables it.
    native_search_tool: str = ""

    def __post_init__(self) -> None:
        if self.thinking_budget <= 0 and self.thinking_level:
            object.__setattr__(
                self,
                "thinking_budget",
                thinking_budget_for_level(self.thinking_level),
            )

    def endpoints(self, *, test_profile: bool = False) -> Tuple[ModelEndpoint, ...]:
        if test_profile:
            return (self.test_endpoint,)
        return self.endpoint_chain


DEFAULT_LIMITS = ModelLimits()

# Thinking budgets (token counts, for models controlled by budget rather than
# thinkingLevel) derive from the level as a share of the API output limit:
# low/medium/high = 20%/40%/60%. Not maintained as standalone numbers anymore.
THINKING_BUDGET_RATIO_BY_LEVEL = {"low": 0.2, "medium": 0.4, "high": 0.6}


def thinking_budget_for_level(
    level: str, *, output_limit: int = DEFAULT_LIMITS.output_limit
) -> int:
    ratio = THINKING_BUDGET_RATIO_BY_LEVEL.get((level or "").strip().lower())
    if ratio is None:
        return 0
    return int(output_limit * ratio)

# Adjacent correction windows physically re-include the previous window's tail
# for stitching redundancy: all segments starting within the last
# OVERLAP_WINDOW_SECONDS before the boundary (purely content-driven, v13 —
# a >=30s gap correctly yields zero overlap; continuity is the read-only
# preceding-context block's job, not the overlap's).
OVERLAP_WINDOW_SECONDS = 30.0
# Read-only raw ASR lines injected before each window (background only, never
# translated). Fixed count, no gap-stop: after a hard gap the new window is
# exactly where cold-start risk peaks, and the negative timestamps let the
# model see how far back the context is and weigh it accordingly.
PRECEDING_CONTEXT_MAX_SEGMENTS = 10

# Hard caps on locally executed search queries (protects the Tavily free quota;
# prompts state the same caps and extra queries are dropped by the harness).
DEFAULT_RESEARCH_SEARCH_QUERIES = 8
MAX_RESEARCH_SEARCH_QUERIES = 16
MAX_WINDOW_SEARCH_QUERIES = 8

# Knowledge-entry pass-through (v17): a session may keep up to
# KB_TRANSFER_MAX_ENTRIES already-injected entries for the next step's
# injection set; transfers plus that step's new requests share
# KB_WINDOW_TOTAL_ENTRIES (transfers win when the total overflows).
KB_TRANSFER_MAX_ENTRIES = 8
KB_WINDOW_NEW_REQUEST_MAX_ENTRIES = 8
KB_WINDOW_TOTAL_ENTRIES = 12

# Unified token budgets for harness-injected blocks (search results, extract
# results, knowledge entries). One rendered unit — a single query's results, a
# single extracted URL, or a single knowledge entry — is capped at
# INJECTION_SECTION_MAX_TOKENS; a whole injected block is capped by
# injection_block_token_limit(unit_cap), where unit_cap is the round's query
# (or entry) cap. Knowledge entries use the same numbers as queries by design.
INJECTION_SECTION_MAX_TOKENS = 4_000
INJECTION_BLOCK_BASE_TOKENS = 4_000
INJECTION_BLOCK_PER_UNIT_TOKENS = 2_000


def injection_block_token_limit(unit_cap: int) -> int:
    """Whole-block token budget for a round whose unit cap is ``unit_cap``."""

    return max(0, int(unit_cap)) * INJECTION_BLOCK_PER_UNIT_TOKENS + INJECTION_BLOCK_BASE_TOKENS


# Local keyword pre-injection: at most this many knowledge entries matched from
# the user note's keys/aliases are injected into research/fast round 1 (or, on
# the text route, into every correction window).
KB_PREINJECT_MAX_ENTRIES = 8

# Cumulative next_advice ledger cap across all windows; the rendered ledger is
# front-truncated (oldest windows dropped) to this budget at injection time.
ADVICE_LEDGER_MAX_TOKENS = 8_000

# Multi-round search loop: total search rounds (round 0 emitted by the main
# conversation plus follow-up rounds emitted by the lightweight loop model).
# Follow-up rounds get half the round-0 query cap.
DEFAULT_RESEARCH_SEARCH_ROUNDS = 3
SEARCH_LOOP_FOLLOWUP_DIVISOR = 2

# Window planning reserve for everything in a correction call that is not the
# planned window payload (CSV + media): the static system prompt (~4k measured
# + style/mistakes headroom), user scaffolding, general/window context,
# accumulated advice (<=8k), query-round notes, the search-results block
# (<=injection_block_token_limit(8)=20k) and the knowledge-entry block (<=28k).
# Worst case sums to ~69k; windows are output-formula-bound in practice, so
# the extra reserve almost never changes the window count.
WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS = 72_000

# Harness-side caps on injected prompt fragments and per-call output ceilings.
# All units are tokens (counted with the caller's token counter, falling back
# to default_token_counter); the numeric values carried over 1:1 from the old
# char-based caps, which slightly relaxes them for CJK-heavy text.
ANALYSIS_NOTES_MAX_TOKENS = 1_500
# One <task_update_feedback> block (correction window or research final round).
TASK_FEEDBACK_MAX_TOKENS = 4_000
# Fast-mode round 1 doubles as the correction round's main background, so its
# notes cap is wider than the research round-1 cap.
FAST_ANALYSIS_NOTES_MAX_TOKENS = 2_000
EVIDENCE_PACK_MAX_TOKENS = 20_000
PROGRESS_UPDATE_MAX_TOKENS = 2_000
WINDOW_NOTES_MAX_TOKENS = 800
NEXT_ADVICE_MAX_TOKENS = 800
# Default per-call output ceiling for every non-correction session (v17: the
# mandatory opening <reasoning> block shares this budget); the correction
# round alone keeps the full DEFAULT_LIMITS.output_limit.
SESSION_OUTPUT_MAX_TOKENS = 32_768
QUERY_ROUND_MAX_TOKENS = SESSION_OUTPUT_MAX_TOKENS
SEARCH_LOOP_MAX_TOKENS = SESSION_OUTPUT_MAX_TOKENS
SEARCH_LOOP_THINKING_LEVEL = "medium"
SEARCH_LOOP_THINKING_BUDGET = thinking_budget_for_level(SEARCH_LOOP_THINKING_LEVEL)


def research_search_query_limit(raw_segment_count: int) -> int:
    """Dynamic background-research query cap from the raw subtitle size."""

    count = max(0, int(raw_segment_count))
    return min(
        MAX_RESEARCH_SEARCH_QUERIES,
        DEFAULT_RESEARCH_SEARCH_QUERIES + int(math.sqrt(count) // 10),
    )


def followup_search_query_limit(round0_limit: int) -> int:
    """Per-round query cap for search-loop follow-up rounds (half of round 0)."""

    return max(1, math.ceil(max(0, int(round0_limit)) / SEARCH_LOOP_FOLLOWUP_DIVISOR))


def default_role_configs() -> Dict[LLMRole, RoleModelConfig]:
    free_36 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_36_FLASH)
    free_35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH)
    free_lite35 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_35_FLASH_LITE)
    free_lite = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_31_FLASH_LITE)
    paid_36 = ModelEndpoint(GEMINI_PAID_TIER, GEMINI_36_FLASH)
    paid_35 = ModelEndpoint(GEMINI_PAID_TIER, GEMINI_35_FLASH)
    paid_lite35 = ModelEndpoint(GEMINI_PAID_TIER, GEMINI_35_FLASH_LITE)
    free_25 = ModelEndpoint(GEMINI_FREE_TIER, GEMINI_25_FLASH)
    paid_25 = ModelEndpoint(GEMINI_PAID_TIER, GEMINI_25_FLASH)
    # Correction / fast "r2" (the actual subtitle window): 3.6 first.
    correction_chain = (free_36, free_35, free_lite35, free_lite, paid_36)
    # Research / fast r1 / knowledge update / etc.: 3.5 → 3.6 → 3.5-lite.
    general_chain = (free_35, free_36, free_lite35, free_lite, paid_35)
    # Shared by 纠错 r1 (lightweight_multimodal) and search-loop judge
    # (lightweight): 3.5-lite first.
    lite_chain = (free_lite35, free_lite, paid_lite35)
    internet_chain = (free_25, paid_25)
    return {
        LLMRole.AUDIO_MULTIMODAL: RoleModelConfig(
            role=LLMRole.AUDIO_MULTIMODAL,
            endpoint_chain=correction_chain,
            test_endpoint=free_lite35,
            thinking_level="medium",
        ),
        LLMRole.GENERAL_CAPABLE: RoleModelConfig(
            role=LLMRole.GENERAL_CAPABLE,
            endpoint_chain=general_chain,
            test_endpoint=free_lite35,
            thinking_level="medium",
        ),
        LLMRole.LIGHTWEIGHT: RoleModelConfig(
            role=LLMRole.LIGHTWEIGHT,
            endpoint_chain=lite_chain,
            test_endpoint=free_lite35,
            thinking_level="medium",
        ),
        LLMRole.LIGHTWEIGHT_MULTIMODAL: RoleModelConfig(
            role=LLMRole.LIGHTWEIGHT_MULTIMODAL,
            endpoint_chain=lite_chain,
            test_endpoint=free_lite35,
            thinking_level="medium",
        ),
        LLMRole.INTERNET_CAPABLE: RoleModelConfig(
            role=LLMRole.INTERNET_CAPABLE,
            endpoint_chain=internet_chain,
            test_endpoint=free_25,
            thinking_level="medium",
            native_search_tool="google_search",
        ),
    }
