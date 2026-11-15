"""Model-capability requirements per task chain, and the startup check.

Two layers, documented in docs/llm_harness_behavior.md:

- the **capability table** (``model_catalog.toml``) says what a model can do;
- the **task-group chains** (``model_routes.toml`` bindings) say who does which job.

This module joins them: for a given profile it lists every chain the run will
actually use together with the capability bits that chain needs, then fails
fast if any of them has no usable endpoint left. The correction chain is the
media-bearing one; ``retrieval=native`` adds the preparation task's chain,
which must contain a member that can run the native search tool.
"""

from __future__ import annotations

import math
import os

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence, Tuple

from finesub.reporting import current_reporter
from finesub import api_keys
from . import execution_policy, model_router, model_routes
# Bound by name as well as by module: `test_llm_model_router` replaces
# *this module's* view of the fact table to make one target video-incapable,
# which is a narrower thing to say than editing the table for everybody.
from .model_routes import (
    DEFAULT_AGENT_SESSION_MODE,
    TOOL_SESSION_MODES,
    runtime_fact_for,
)
from .config import (
    ModelEndpoint,
    ModelLimits,
    effective_window_subtitle_cap,
    planning_limits_for,
    role_config_for,
)
from .profiles import TranslationProfile


@dataclass(frozen=True)
class ChainRequirement:
    """One role chain plus the capability bits this run needs from it."""

    label: str
    endpoints: Tuple[ModelEndpoint, ...]
    needs_audio: bool = False
    needs_video: bool = False
    needs_native_search: bool = False

    def missing_capabilities(self) -> Tuple[str, ...]:
        names = []
        if self.needs_audio:
            names.append("audio")
        if self.needs_video:
            names.append("video")
        if self.needs_native_search:
            names.append("native_search")
        return tuple(names)


def endpoint_supports(
    endpoint: ModelEndpoint,
    *,
    needs_audio: bool = False,
    needs_video: bool = False,
    needs_native_search: bool = False,
) -> bool:
    """Whether the runtime facts say this endpoint carries every requested bit.

    The fact comes from the **merged** table (``runtime_fact_for``), not the
    packaged catalog: a user-declared HTTP model is text-only by construction
    (media is supported only by packaged Gemini REST and local Agy targets), and resolving it against the
    packaged rows alone returned "unknown" -- which passes -- so a text
    transport used to accept an audio call and silently drop the attachment.

    An endpoint neither table knows still passes: the harness does not tighten
    behaviour without evidence (test fixtures rely on this).

    Native search additionally requires the *target's* tool declaration --
    a fact may say the model could ground while this particular target has no
    search tool wired; accepting it would send a native call out without the
    tool.
    """

    if needs_native_search and not endpoint.native_search_tool:
        return False
    entry = runtime_fact_for(endpoint)
    if entry is None:
        return True
    if needs_audio and not entry.supports_audio:
        return False
    if needs_video and not entry.supports_video:
        return False
    if needs_native_search and not entry.supports_native_search:
        return False
    return True


def usable_endpoints(requirement: ChainRequirement) -> List[ModelEndpoint]:
    """Endpoints of the chain that are both enabled and capable enough."""

    return [
        endpoint
        for endpoint in requirement.endpoints
        if _endpoint_enabled(endpoint)
        and endpoint_supports(
            endpoint,
            needs_audio=requirement.needs_audio,
            needs_video=requirement.needs_video,
            needs_native_search=requirement.needs_native_search,
        )
    ]


def _endpoint_enabled(endpoint: ModelEndpoint) -> bool:
    return endpoint.backend == "local_agent" or api_keys.provider_tier_enabled(
        endpoint.provider_tier
    )


def _effective_endpoints(
    config, *, native_search: bool = False
) -> Tuple[ModelEndpoint, ...]:
    settings = execution_policy.load_execution_settings()
    return tuple(
        candidate.endpoint
        for candidate in model_router.ModelRouter(policy_id=settings.policy_id)
        .plan(config, native_search=native_search)
        .candidates
    )


def _strategy_chain(profile: TranslationProfile, task_group: str) -> ChainRequirement:
    if task_group == "correction":
        return required_chains(profile)[0]
    if task_group == "research":
        return ChainRequirement(
            label="研究 (research)",
            endpoints=_effective_endpoints(
                role_config_for("research", profile.difficulty),
                native_search=profile.native_search,
            ),
            needs_native_search=profile.native_search,
        )
    raise ValueError(f"unknown task group for a strategy: {task_group!r}")


def chain_backends(profile: TranslationProfile, *, task_group: str = "correction") -> List[str]:
    """``backend`` of every usable endpoint of the chain serving ``task_group``.

    ``correction``: the window chain of :func:`required_chains` (the fast
    whole-input window shares its cell). ``research``: the preparation task's
    chain under the same native-search requirement. What the execution
    strategy (docs/llm_harness_behavior.md, 「开关组合 → 实际会跑的会话」) reads.
    """

    return [endpoint.backend for endpoint in usable_endpoints(_strategy_chain(profile, task_group))]


def chain_serves_tool_session(
    profile: TranslationProfile, *, task_group: str = "correction"
) -> bool:
    """Whether this task's agent calls would actually get the harness tools.

    An agentic task fetches for itself, which only exists on the **tool
    session** transport. ``agent_transports.agent_transport_for`` derives that
    transport from ``(session mode, driver, media)``, so all three inputs are
    asked here:

    * a call carrying a media part is always ``capsule`` -- the tool protocol
      is text-only (docs/llm_agent_tool_protocol.md §7);
    * ``api`` and ``resume`` are ``capsule`` by definition;
    * ``per-window`` is a tool session only for a driver whose CLI takes a
      per-invocation MCP server, and falls back to ``capsule`` otherwise --
      or when the dev-only ``FINESUB_AGENT_TRANSPORT=capsule`` pins it there.

    All three leave an "acquire it yourself" prompt in front of an executor
    with no tools while the harness has already stood down its own request
    turn -- ``retrieval=local`` silently becomes ``none``. ⚠ The third one used
    to be left to the transport's own fallback warning on the grounds that it
    "changes cost, not capability". That was true before this strategy existed:
    the same prompt simply travelled a slower road. Now the strategy decides
    *who fetches*, so the fallback destroys retrieval and the warning cannot
    undo it -- the answer has to be known before the prompt is written.
    """

    if task_group == "correction":
        if profile.correction_media != "text":
            return False
        routing_group = correction_task_group(profile)
    elif task_group == "research":
        routing_group = "research"
    else:
        raise ValueError(f"unknown task group for a strategy: {task_group!r}")
    mode = getattr(role_config_for(routing_group, profile.difficulty), "agent_session_mode", "")
    mode = mode or DEFAULT_AGENT_SESSION_MODE
    if mode not in TOOL_SESSION_MODES:
        return False
    from ..agent.agent_transports import AGENT_TRANSPORT_OVERRIDE_ENV

    # The dev-only transport pin moves a per-window call onto capsule, which
    # has no tools either (pseudo-conversational is a tool session whatever
    # the pin says).
    if mode == "per-window" and os.environ.get(AGENT_TRANSPORT_OVERRIDE_ENV, "").strip() == "capsule":
        return False
    # Every agent that may answer, not just the first: a fallback down the
    # chain lands on the same "you fetch" prompt. Non-agent members do not
    # reach here -- a chain with one of those is already `batched`.
    return all(
        _driver_takes_mcp_server(endpoint.provider_tier, endpoint.api_model_id)
        for endpoint in usable_endpoints(_strategy_chain(profile, task_group))
        if endpoint.backend == "local_agent"
    )


@lru_cache(maxsize=None)
def _driver_takes_mcp_server(provider_tier: str, model: str) -> bool:
    """Does this agent target's CLI accept a per-invocation MCP server?

    The one input of the transport rule that needs a probe -- a ``--help`` run,
    which the driver caches and which this run is going to make anyway before
    its first agent call. Memoized on the target because the strategy is
    resolved once per task and the answer is a property of the installed CLI.

    Anything that goes wrong -- no driver registered for the tier, the CLI
    missing, the probe raising -- answers "no", which is the side that keeps
    the harness fetching for the model.
    """

    try:
        driver = execution_policy.driver_for_provider_tier(
            execution_policy.load_execution_settings(),
            provider_tier=provider_tier,
            model=model,
        )
        return bool(getattr(driver.probe(), "supports_mcp_config", False))
    except Exception:  # noqa: BLE001 -- any failure means "cannot promise tools"
        return False


def correction_task_group(profile: TranslationProfile) -> str:
    """The correction window's task group; ``correction_media`` selects it
    (plan v2 D20 -- the switch *is* the selector)."""

    return "correction-mm" if profile.correction_media != "text" else "correction-text"


def correction_planning_limits(profile: TranslationProfile) -> ModelLimits:
    """Window-planning envelope for the correction cell (plan v2 D13).

    Thin wrapper: the correction window's task group comes from
    ``correction_media``, the envelope itself is the generic
    :func:`finesub.llm.routing.config.planning_limits_for` -- every stage that sizes an input
    against "whoever answers" needs the same conversion.

    It lives here rather than beside the correction loop because the research
    stage plans the *same* windows and must therefore use the *same* envelope;
    importing the loop from ``research`` would close an import cycle.
    """

    task_group = correction_task_group(profile)
    # Set aside what the answer is expected to take, not what a call may ask
    # for (owner 2026-09-04). The largest window this planner will ever emit is
    # the quality cap, so `output_scale x c x cap` bounds every window's
    # expected output -- closed form, because the cap is an independent
    # constant rather than a function of the envelope being computed, so there
    # is no fixpoint to iterate towards.
    #
    # `max_tokens` on the wire still asks for the full declared ceiling: the
    # answer keeps all its headroom, and only the *input* side is freed. The
    # two stay compatible because the output cap binds a window long before the
    # input ceiling does (pinned in `test_llm_model_routes`).
    baseline = planning_limits_for(task_group, profile.difficulty)
    cap = effective_window_subtitle_cap(None, baseline)
    if cap <= 0:
        return baseline
    reserve = math.ceil(profile.output_scale * profile.output_coefficient * cap)
    return planning_limits_for(task_group, profile.difficulty, output_reserve=reserve)


def correction_planning_envelope_description(profile: TranslationProfile) -> str:
    """Human-readable members that constrain the correction planning envelope."""

    try:
        routes = model_routes.default_model_routes()
        group, _cell = routes.resolve_binding(
            routes.active_preset_id,
            correction_task_group(profile),
            profile.difficulty,
        )
        facts = [
            (target_id, routes.target_fact(target_id))
            for target_id in group.target_ids
        ]
    except (KeyError, model_routes.ModelRouteConfigError, OSError):
        return f"{correction_task_group(profile)}/{profile.difficulty} (default limits)"
    # Same two passes as `group_planning_envelope`: the output is settled for
    # the whole group first, then each member says how much input it can still
    # take. Which of the two bounds is worth naming -- "input" sends the
    # reader to `max_input_tokens`, "context" to a pool the answer has to share.
    min_output = min(fact.max_output_tokens for _, fact in facts)
    ceilings = {
        target_id: min(fact.max_input_tokens, fact.context_window - min_output)
        for target_id, fact in facts
    }
    min_ceiling = min(ceilings.values())
    max_scale = max(1.0, *(float(fact.token_scale or 1.0) for _, fact in facts))
    members: list[str] = []
    for target_id, fact in facts:
        limits: list[str] = []
        if ceilings[target_id] == min_ceiling:
            limits.append(
                "context"
                if fact.context_window - min_output < fact.max_input_tokens
                else "input"
            )
        if fact.max_output_tokens == min_output:
            limits.append("output")
        if float(fact.token_scale or 1.0) == max_scale and max_scale > 1.0:
            limits.append("token_scale")
        if limits:
            members.append(f"{target_id}[{'+'.join(limits)}]")
    return f"group={group.id}; limiting_members={','.join(members) or 'none'}"


def required_chains(profile: TranslationProfile) -> List[ChainRequirement]:
    """Every chain this profile will actually exercise, with its capability bits.

    Media requirements mirror what the call sites really attach: the
    correction window (request turn and delivery alike, fast or not) carries
    the ``correction_media`` clip. Text-only chains (the non-native
    preparation task, knowledge update) need no capability bits and are left
    out.
    """

    correction_group = correction_task_group(profile)
    correction_config = role_config_for(correction_group, profile.difficulty)
    chains: List[ChainRequirement] = [
        ChainRequirement(
            label="纠错窗 (correction window)",
            endpoints=_effective_endpoints(
                correction_config, native_search=profile.native_search
            ),
            needs_audio=profile.correction_use_audio,
            needs_video=profile.correction_use_video,
            needs_native_search=profile.native_search,
        )
    ]
    if profile.native_search:
        # The preparation task runs the model's own search tool under
        # retrieval=native; like the correction window, the bound research
        # group must contain a native-capable member (no fallback chain).
        chains.append(
            ChainRequirement(
                label="准备任务 (native)",
                endpoints=_effective_endpoints(
                    role_config_for("research", profile.difficulty),
                    native_search=True,
                ),
                needs_native_search=True,
            )
        )
    return chains


class CapabilityUnavailableError(RuntimeError):
    """No endpoint in some required chain can serve the requested switches."""


def _describe(requirement: ChainRequirement) -> str:
    wanted = "+".join(requirement.missing_capabilities()) or "（无能力要求）"
    enabled = [
        endpoint
        for endpoint in requirement.endpoints
        if _endpoint_enabled(endpoint)
    ]
    disabled_tiers = sorted(
        {
            endpoint.provider_tier
            for endpoint in requirement.endpoints
            if not _endpoint_enabled(endpoint)
        }
    )
    parts = [f"{requirement.label} 需要 {wanted}"]
    if not enabled:
        parts.append("该链的 provider tier 全部未启用")
    else:
        parts.append(
            "已启用但能力不足: "
            + ", ".join(f"{e.provider_tier}|{e.api_model_id}" for e in enabled)
        )
    if disabled_tiers:
        parts.append("被关闭的 tier: " + ", ".join(disabled_tiers))
    return "；".join(parts)


def validate_profile_capabilities(
    profile: TranslationProfile,
    *,
    test_profile: bool = False,
) -> None:
    """Fail fast when a switch combination has no endpoint able to serve it.

    ``test_profile`` runs pin a single stub endpoint and never enable native
    search, so the check does not apply.
    """

    if test_profile:
        return
    unmet: Sequence[ChainRequirement] = [
        requirement
        for requirement in required_chains(profile)
        if not usable_endpoints(requirement)
    ]
    if unmet:
        hints = [
            (
                "retrieval=native 需要绑定的模型组里有 supports_native_search "
                "的成员：出厂预设靠付费 3.8 Flash 接地，免费档只有 2.5 Flash "
                "能联网（低于纠错/知识下限，需显式绑定打包的 "
                "gemini-native-search 组）——v2 起没有独立的 native 链，"
                "也不做静默降级"
            )
            for requirement in unmet
            if requirement.needs_native_search
        ][:1]
        raise CapabilityUnavailableError(
            "所选开关组合没有可用 endpoint：\n"
            + "\n".join(f"  - {_describe(requirement)}" for requirement in unmet)
            + ("".join(f"\n  提示: {hint}" for hint in hints))
        )
    for message in profile_warnings(profile):
        current_reporter().warning("routing-profile", message)
    # Binding-time advisory warnings for the **active** preset (model-routing v2).
    # Hard-coding "default" here made the whole mechanism dead code: the
    # shipped default is warning-free by test, and a user preset -- the only
    # thing these warnings exist for -- was never inspected.
    routes = model_routes.default_model_routes()
    if routes.override_notices:
        # Same-id override is intentional, a typo looks identical -- say it once.
        current_reporter().warning(
            "routing-override",
            "config.toml 覆盖了打包声明的 " + "、".join(routes.override_notices),
            action="同 id 覆盖是有意为之时可忽略；拼错了看起来一模一样",
        )
    for message in [*routes.preset_binding_warnings(routes.active_preset_id),
                    *routes.quality_cap_warnings(routes.active_preset_id, research_chunks=profile.retrieval != "native")]:
        current_reporter().warning("routing-preset", message)
    # Notes, not warnings: `debug` is the channel that always reaches the log
    # file and only reaches the terminal under `--verbose`, which is what "for
    # the record" means here (docs/reporting.md keeps the event set at eight,
    # so a blank `capability` does not get a ninth).
    for message in routes.preset_binding_notes(routes.active_preset_id):
        current_reporter().debug(message)


def profile_warnings(profile: TranslationProfile) -> List[str]:
    """Non-fatal notices about a switch vector that runs but may surprise.

    "The run proceeds and produces something reasonable, but not what the
    switches literally promise" is exactly what a warning is for -- rejecting
    it would remove a usable combination.
    """

    messages: List[str] = []
    # (The old "native pool is basic-tier only" warning died with the v1
    # native-search chain: the variant now comes from the task-group cell, so
    # difficulty=quality gets capableC on whichever native-capable model the
    # user bound -- plan v2 D4.) The output coefficient is not calibrated per
    # switch vector and says nothing at run time (owner 2026-09-30).
    if profile.continuity == "parallel":
        messages.append(
            "continuity=parallel："
            "窗口间无 advice 台账与词条透传链，会话级词条集在屏障处一次定死。"
        )
    return messages


#: The window a model group must be able to take in, and to write out, before
#: a run is willing to plan against it. Warn below the first pair, refuse below
#: the second (owner decision 2026-09-03).
#:
#: Decimal on purpose, not 64Ki/32Ki: `local-claude-haiku` declares exactly
#: 64000 output, and the owner's ruling is that it passes -- one power of two
#: here would put a healthy model into permanent warning.
#:
#: The input floor moved 194000 -> 192000 for the same reason (owner
#: 2026-09-04): `local-codebuddy-hy3` declares exactly 192000 in, measured off
#: the CLI's own `modelUsage`, and it is the cheapest healthy row on that tier.
#: 194000 was never a vendor number either -- it is the free-Gemini planning
#: baseline that `ENVELOPE_BASELINE_INPUT` still carries, and that one is a
#: different question (how many windows a group costs), so it did not move.
WINDOW_WARN_INPUT = 192_000
WINDOW_WARN_OUTPUT = 64_000
WINDOW_REFUSE_INPUT = 96_000
WINDOW_REFUSE_OUTPUT = 32_000


class ModelWindowTooSmallError(RuntimeError):
    """A bound model group has a member too small to plan a correction against."""


def _window_complaints(minimum: int, floor: int, refuse: int, side: str) -> str:
    if minimum < refuse:
        return f"{side} {minimum} < {refuse}"
    if minimum < floor:
        return f"{side} {minimum} < {floor}"
    return ""


def check_model_group_windows(
    routes: model_routes.ModelRouteCatalog | None = None,
) -> None:
    """Warn or refuse on a bound group whose smallest member is too small.

    ✱ **The catalog's two columns, not the planning envelope.** The gate asks
    about the *shape* of a member -- long enough in, long enough out -- while
    `group_planning_envelope` answers the different question of what a window
    may be planned at when the joint budget is what binds. Reading the envelope
    here would fail `local-claude-haiku` on its input (200000 - 64000 =
    136000) for having a context window rather than for being small, which the
    owner ruled against on 2026-09-03: "总量受限，但缩放的形状健康".

    ✱ **Groups, not the catalog.** `gemini-free-gemma-4-31b` declares 16000
    input and would trip the refusal on sight, but it is a search target that
    belongs to no model group and answers no correction window. Only what the
    active preset can actually dispatch to is inspected.

    Called before ASR rather than from the correction stage: refusing after the
    expensive half of a run has already happened costs the user the whole run
    to learn about a configuration mistake.
    """

    if routes is None:
        try:
            routes = model_routes.default_model_routes()
        except (model_routes.ModelRouteConfigError, OSError):
            # A route table that will not load is not this check's error to
            # report; the router says it far more precisely, and saying it
            # twice in two voices helps nobody.
            return
    warnings: List[str] = []
    refusals: List[str] = []
    for group_id in routes.reachable_group_ids():
        min_input, min_output = routes.group_declared_minima(group_id)
        complaints = [
            complaint
            for complaint in (
                _window_complaints(
                    min_input, WINDOW_WARN_INPUT, WINDOW_REFUSE_INPUT, "最大输入"
                ),
                _window_complaints(
                    min_output, WINDOW_WARN_OUTPUT, WINDOW_REFUSE_OUTPUT, "最大输出"
                ),
            )
            if complaint
        ]
        if not complaints:
            continue
        line = f"模型组 {group_id}：{'、'.join(complaints)}"
        if min_input < WINDOW_REFUSE_INPUT or min_output < WINDOW_REFUSE_OUTPUT:
            refusals.append(line)
        else:
            warnings.append(line)
    if warnings:
        # One warning listing every group, not one per group: a single small
        # member usually sits in several bound groups at once (the shipped
        # presets share their Gemini targets), and four near-identical lines
        # say nothing the first one did not.
        current_reporter().warning(
            "model-window-small",
            "；".join(warnings),
            impact="窗口会被切得更小，纠错质量与合并判断都会受影响",
            action=f"换用最大输入 ≥{WINDOW_WARN_INPUT}、最大输出 ≥{WINDOW_WARN_OUTPUT} 的模型，"
            "或接受更碎的窗口",
        )
    if refusals:
        raise ModelWindowTooSmallError(
            "绑定的模型组里有成员窗口过小，无法用于纠错翻译：\n"
            + "\n".join(f"  - {line}" for line in refusals)
            + f"\n  下限：最大输入 {WINDOW_REFUSE_INPUT}、最大输出 {WINDOW_REFUSE_OUTPUT}"
        )
