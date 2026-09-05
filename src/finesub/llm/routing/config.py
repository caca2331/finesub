"""Model role configuration for LLM subtitle correction.

Bottom of the routing layers: routes, execution policy and the router are all
built on this module, and it imports none of them. The three helpers below
that read a *resolved* route back are where the direction inverts, so they
import inside the function -- naming those modules at the top would make
`finesub.llm.routing.config` unimportable. Each site is marked `# Inverted layer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Mapping, Tuple


GEMINI_38_FLASH = "gemini/gemini-3.8-flash"
GEMINI_37_FLASH = "gemini/gemini-3.7-flash"
GEMINI_36_FLASH = "gemini/gemini-3.6-flash"
GEMINI_35_FLASH = "gemini/gemini-3.5-flash"
GEMINI_35_FLASH_LITE = "gemini/gemini-3.5-flash-lite"
GEMINI_31_FLASH_LITE = "gemini/gemini-3.1-flash-lite"
GEMINI_25_FLASH = "gemini/gemini-2.5-flash"
GEMINI_FREE_TIER = "GEMINI_FREE"
GEMINI_PAID_TIER = "GEMINI_PAID"


class LLMRole(str, Enum):
    # Correction windows (and fast-mode correction after r1): 3.8 → 3.7 → 3.6 → 3.5.
    AUDIO_MULTIMODAL = "audio_multimodal"
    # Research r1/r2, fast round 1, post-task knowledge update, and other
    # non-correction work: prefer 3.6 Flash, then 3.5, then 3.8 → 3.7.
    GENERAL_CAPABLE = "general_capable"
    # Search-loop judge ("查询"): prefer 3.5 Flash Lite (text).
    LIGHTWEIGHT = "lightweight"
    # Correction query round ("纠错 r1"): same 3.5-lite chain, multimodal role.
    LIGHTWEIGHT_MULTIMODAL = "lightweight_multimodal"


# Prompt tier comes directly from the answering endpoint's catalog fact.
# The correction prompt is assembled per tier inside the endpoint loop so a
# fallback model never receives a prompt written for a stronger one: CAPABLE
# gets the judgment-based merge fragments, BASIC the conservative 1:1 variant
# (docs/llm_harness_behavior.md, docs/llm_prompts.md).
class CapabilityTier(str, Enum):
    CAPABLE = "capable"
    BASIC = "basic"

@dataclass(frozen=True)
class ModelLimits:
    #: What a call may send and ask for. **Not ceilings**: they are the values
    #: used when there is no route declaration to consult (stub configs, tests,
    #: the artifact metadata dump). Every real group derives both from its own
    #: catalog rows in `planning_limits_for`, which no longer clamps them --
    #: a `context_limit` and a `safety_margin` used to live here too, and both
    #: are gone: with the input envelope derived as `context_window - output`,
    #: `input + expected_output <= context_window` holds by construction, so
    #: the check they served could never fire.
    prompt_input_limit: int = 194_000
    output_limit: int = 65_536
    audio_tokens_per_second: int = 32
    # Planning envelope for video attachments. True when any candidate that
    # can answer the routed media cell is forced onto the high-resolution frame
    # tier (currently the local Agy transport).
    video_high_resolution: bool = False
    # Quality guardrail: the largest per-window <asr_result> CSV (core +
    # overlap rows) allowed, in tokens. Beyond this, translation quality drops
    # even when the model output would still fit; 0 disables the cap.
    max_window_subtitle_tokens: int = 10_000


@dataclass(frozen=True)
class RateLimitPolicy:
    safety_factor: float = 0.9
    window_seconds: float = 61.0


@dataclass(frozen=True)
class ModelEndpoint:
    provider_tier: str
    api_model_id: str
    target_id: str = ""
    fact_id: str = ""
    backend: str = "gemini_rest"
    native_search_tool: str = ""
    # Custom test fixtures may explicitly opt into the former permissive
    # behavior. Packaged production targets always have a verified fact id.
    unverified: bool = False


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
    # "google_search", "web_search"). Empty disables. Native search is a
    # *capability* requested per call (plan v2 D4): the router filters the
    # bound group by ``supports_native_search`` and the client turns the
    # target's own tool on. The field stays for custom test fixtures.
    native_search_tool: str = ""
    # v2 cell identity (plan §5.4-§6): the bound model group and the cell's
    # default prompt variant. ``model_group_id`` being set routes the plan
    # through the group path; empty falls back to the adapter path (custom
    # endpoint_chain fixtures).
    model_group_id: str = ""
    task_group_id: str = ""
    difficulty: str = "quality"
    variant: str = ""
    variant_overrides: Mapping[str, str] = field(default_factory=dict)
    # How long one headless agent conversation lives for this cell -- the
    # four-tier knob of docs/llm_local_agent.md §12.1 (api / per-window /
    # resume / pseudo-conversational), owner-set per task group with
    # difficulty fallback. Only a local agent reads it; API backends have no
    # session to reuse.
    #
    # Read by `client.py`'s `_run_local_agent` (2026-08-19): per-window is the
    # default and today's behaviour, `resume` is the experiment switch for the
    # production-size reuse A/B (docs/llm_local_agent_experiments.md §3.5) and
    # degrades to `api` on drivers without session reuse. The durable task
    # runtime still has no production call site -- docs/llm_local_agent.md
    # §12.1.1.
    agent_session_mode: str = ""

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


def effective_window_subtitle_cap(
    value: int | None, limits: ModelLimits = DEFAULT_LIMITS
) -> int:
    """The <asr_result> cap the planner will actually apply.

    ``None`` means "take the limits default"; ``0`` means "no cap". Those are
    different windowings, so anything that records the cap -- above all the
    research-context cache key -- has to resolve it first. Recording the raw
    ``None`` as ``0`` would let an unset config reuse a context planned with the
    cap disabled, and the window ids would silently disagree.
    """

    return int(limits.max_window_subtitle_tokens if value is None else value)


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
# + style headroom), user scaffolding, general context, the window's
# research notes (<=8k), accumulated advice (<=8k), query-round notes, the search-results block
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
# Research notes injected into one correction window. A re-shaped window can
# select several older notes at once, so this caps the concatenation, not one
# note; it is part of the planning reserve accounted above.
WINDOW_CONTEXT_MAX_TOKENS = 8_000
# How much of a shared context a non-correction session is expected to spend on
# its answer -- **a planning reserve, not a cap** (owner 2026-09-04). It is
# passed as `output_reserve`, so it only decides whether a candidate's input
# still fits beside the answer on a single-pool model; the request itself fills
# that candidate's own `max_output_tokens`. It used to be sent as `max_tokens`,
# which silently truncated any non-correction round that needed more than this
# and made the number a quality knob nobody had calibrated.
#
# 32,000 rather than 32,768: a decimal reserve, for the same reason
# `WINDOW_REFUSE_OUTPUT` is decimal -- these are compared against catalog
# columns that state vendor numbers, and a power of two only ever coincides
# with one by accident.
SESSION_OUTPUT_MAX_TOKENS = 32_000
QUERY_ROUND_MAX_TOKENS = SESSION_OUTPUT_MAX_TOKENS
SEARCH_LOOP_MAX_TOKENS = SESSION_OUTPUT_MAX_TOKENS
# (The search-judge thinking level now comes from the preset knob like every
# other session; the old per-call constants are gone.)


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


# Each role's default task group, used when a call site passes only the role
# (mostly tests and role-shaped helpers). Production call sites pass the task
# group explicitly -- notably GENERAL_CAPABLE serves research, knowledge
# update *and* the fast fusion round, which are different task groups now.
ROLE_DEFAULT_TASK_GROUP: Dict[LLMRole, str] = {
    LLMRole.AUDIO_MULTIMODAL: "correction-mm",
    LLMRole.GENERAL_CAPABLE: "research",
    LLMRole.LIGHTWEIGHT: "search_judge",
    LLMRole.LIGHTWEIGHT_MULTIMODAL: "planning-mm",
}
_TASK_GROUP_DEFAULT_ROLE = {
    "correction-mm": LLMRole.AUDIO_MULTIMODAL,
    "correction-text": LLMRole.AUDIO_MULTIMODAL,
    "planning-mm": LLMRole.LIGHTWEIGHT_MULTIMODAL,
    "planning-text": LLMRole.LIGHTWEIGHT_MULTIMODAL,
    "research": LLMRole.GENERAL_CAPABLE,
    "search_judge": LLMRole.LIGHTWEIGHT,
    "knowledge": LLMRole.GENERAL_CAPABLE,
}


def role_config_for(
    task_group_id: str,
    difficulty: str = "quality",
    *,
    role: LLMRole | None = None,
    preset_id: str | None = None,
    routes=None,
) -> RoleModelConfig:
    """Executable config for one (task group, difficulty) cell (model-routing v2).

    The active preset (``config.toml [llm].preset``, default "default") binds
    the cell to a model group; the cell supplies the default prompt variant
    and thinking level; the preset supplies the test target. The role is only
    a label on artifacts now.
    """

    # Inverted layer (see module docstring).
    from .model_routes import default_model_routes

    routes = routes if routes is not None else default_model_routes()
    preset_id = preset_id or routes.active_preset_id
    group, cell = routes.resolve_binding(preset_id, task_group_id, difficulty)

    def endpoint(target_id: str) -> ModelEndpoint:
        target = routes.targets[target_id]
        fact = routes.target_fact(target_id)
        profile = routes.target_profile(target_id)
        return ModelEndpoint(
            provider_tier=fact.provider_tier,
            api_model_id=fact.api_model_id,
            target_id=target.id,
            fact_id=target.fact_id,
            backend=target.backend,
            native_search_tool=profile.native_search_tool,
        )

    return RoleModelConfig(
        role=role or _TASK_GROUP_DEFAULT_ROLE[task_group_id],
        endpoint_chain=tuple(
            endpoint(target_id) for target_id in group.target_ids
        ),
        test_endpoint=endpoint(routes.presets[preset_id].test_target_id),
        # The preset-level thinking knob (difficulty fallback, medium default);
        # each candidate's catalog mapping translates it at call time.
        thinking_level=routes.resolve_thinking(
            preset_id, task_group_id, difficulty
        ),
        agent_session_mode=routes.resolve_agent_session(
            preset_id, task_group_id, difficulty
        ),
        variant=cell.variant,
        variant_overrides=group.variant_overrides,
        model_group_id=group.id,
        task_group_id=task_group_id,
        difficulty=difficulty,
    )


def _cell_forces_high_resolution_video(
    task_group_id: str, difficulty: str, routes
) -> bool:
    """Whether any candidate for this media cell forces the expensive tier.

    Deliberately not memoized. The answer depends on the active execution
    policy as well as the routing tables, and the policy comes from
    ``config.toml``, which can be re-read inside one process -- a cache keyed
    on the routing digest alone answers a stale policy's question. Expanding
    the chain is a pure in-memory plan with no probing, and the callers are
    per-stage (three in ``correction_translation``, one each in ``research``
    and ``stages.correction``), not per-window, so there is nothing here worth
    trading correctness for.
    """

    # Inverted layer (see module docstring).
    from .execution_policy import load_execution_settings
    from .model_router import ModelRouter

    planned = ModelRouter(
        routes=routes, policy_id=load_execution_settings().policy_id
    ).plan(role_config_for(task_group_id, difficulty, routes=routes))
    return any(
        candidate.fact.supports_video and candidate.fact.video_high_resolution_only
        for candidate in planned.candidates
    )


def planning_limits_for(
    task_group_id: str,
    difficulty: str = "quality",
    *,
    routes=None,
    output_reserve: int | None = None,
) -> ModelLimits:
    """Planning envelope for one cell's bound model group (plan v2 D13).

    Planning happens before "who answers" is known, so it plans against the
    group minimum: prompt input capped so prompt + output still fit the
    smallest member's context, output capped at the smallest output ceiling.
    Whole-Gemini groups resolve to :data:`DEFAULT_LIMITS` unchanged.

    **Unit conversion matters here.** The planner counts in *local estimate*
    tokens while ``max_input_tokens`` is in the provider's, and ``token_scale``
    is the bridge: dispatch checks ``estimate * scale <= max_input``, so the
    planner's budget is ``max_input / scale``. Skipping the conversion makes
    planning and dispatch disagree -- a unit planned to fit is then rejected
    as ``input_limit`` on arrival, with no candidate left. The scale is clamped
    at 1.0: a fact declaring < 1 says the local counter over-estimates for it,
    and relaxing a *safety* bound on that basis is the dangerous direction.

    A *route declaration* problem is not swallowed -- planning against a group
    that cannot serve the call fails every call far from its cause. Only
    "there is no declaration to consult" (stub configs in tests, where the
    caller pins its own endpoints) falls back to the defaults.
    """

    from dataclasses import replace as _replace

    # Inverted layer (see module docstring).
    from .model_routes import ModelRouteConfigError, default_model_routes

    if routes is None:
        try:
            routes = default_model_routes()
        except (ModelRouteConfigError, OSError):
            return DEFAULT_LIMITS
    try:
        group, _cell = routes.resolve_binding(
            routes.active_preset_id, task_group_id, difficulty
        )
        input_ceiling, min_output = routes.group_planning_envelope(
            group.id, output_reserve=output_reserve
        )
        scale = routes.group_estimate_scale(group.id)
    except KeyError:
        return DEFAULT_LIMITS
    video_high_resolution = (
        _cell_forces_high_resolution_video(task_group_id, difficulty, routes)
        if task_group_id.endswith("-mm")
        else False
    )

    # Both numbers come from the catalog and nothing here caps them. The
    # harness used to hold its own ceilings (194,000 input / 65,536 output) and
    # return them verbatim whenever the group cleared both, which meant a
    # declared window above them never reached the planner at all. The envelope
    # is the group's own answer now; `group_planning_envelope` has already
    # subtracted the output from each member's context window, so the only work
    # left is the unit conversion into the local estimate the planner counts in.
    return _replace(
        DEFAULT_LIMITS,
        prompt_input_limit=max(1, int(input_ceiling / scale)),
        output_limit=max(1, min_output),
        video_high_resolution=video_high_resolution,
    )


def default_role_configs() -> Dict[LLMRole, RoleModelConfig]:
    """Role-indexed view over the default preset's ``high`` cells.

    Compatibility shim for role-only call sites; sessions that know their
    task group and difficulty call :func:`role_config_for` directly.
    """

    return {
        role: role_config_for(task_group_id, "quality", role=role)
        for role, task_group_id in ROLE_DEFAULT_TASK_GROUP.items()
    }
