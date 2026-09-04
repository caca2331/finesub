"""Model-capability requirements per role chain, and the startup check.

Two layers, documented in docs/llm_harness_behavior.md:

- the **capability table** (``model_catalog.psv``) says what a model can do;
- the **role chains** (``config.default_role_configs``) say who does which job.

This module joins them: for a given profile it lists every chain the run will
actually use together with the capability bits that chain needs, then fails
fast if any of them has no usable endpoint left. Checking only the correction
chain would miss "correction pool supports it, query-round pool does not".
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import List, Sequence, Tuple

from finesub.reporting import current_reporter
from . import api_keys, execution_policy, model_router, model_routes
# Bound by name as well as by module: `test_llm_model_router` replaces
# *this module's* view of the fact table to make one target video-incapable,
# which is a narrower thing to say than editing the table for everybody.
from .model_routes import runtime_fact_for
from .config import (
    ModelEndpoint,
    ModelLimits,
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


def correction_task_group(profile: TranslationProfile) -> str:
    """The correction window's task group; ``correction_media`` selects it
    (plan v2 D20 -- the switch *is* the selector)."""

    return "correction-mm" if profile.correction_media != "text" else "correction-text"


def planning_task_group(profile: TranslationProfile) -> str:
    return "planning-mm" if profile.planning_media != "text" else "planning-text"


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

    return planning_limits_for(correction_task_group(profile), profile.difficulty)


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


def required_chains(
    profile: TranslationProfile, *, fast_enabled: bool = False
) -> List[ChainRequirement]:
    """Every chain this profile will actually exercise, with its capability bits.

    Media requirements mirror what the call sites really attach: the
    correction window carries the ``correction_media`` clip, the ordinary
    local query round carries the ``planning_media`` clip, and its mutually
    exclusive fast round 1 carries the correction clip. Text-only chains
    (search-loop judge, non-native research rounds, knowledge update) need no
    capability bits and are left out.
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
        # Research R2 runs the model's own search tool under retrieval=native;
        # like the correction window, the bound research group must contain a
        # native-capable member (no fallback chain since D4).
        chains.append(
            ChainRequirement(
                label="研究 R2 (native)",
                endpoints=_effective_endpoints(
                    role_config_for("research", profile.difficulty),
                    native_search=True,
                ),
                needs_native_search=True,
            )
        )
    if profile.external_injection and not fast_enabled:
        chains.append(
            ChainRequirement(
                label="每窗查询轮 (query round)",
                endpoints=_effective_endpoints(
                    role_config_for(planning_task_group(profile), profile.difficulty)
                ),
                needs_audio=profile.planning_use_audio,
                needs_video=profile.planning_use_video,
            )
        )
    if profile.external_injection and fast_enabled:
        # Fast round 1 carries the correction window's clip and belongs with
        # the correction task group, not research (model-routing v2 -- do not
        # "tidy" it back onto research: with media it shares the correction
        # cell's quality bar and clip).
        chains.append(
            ChainRequirement(
                label="fast round 1",
                endpoints=_effective_endpoints(correction_config),
                needs_audio=profile.correction_use_audio,
                needs_video=profile.correction_use_video,
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
    fast_enabled: bool = False,
    test_profile: bool = False,
) -> None:
    """Fail fast when a switch combination has no endpoint able to serve it.

    ``fast_enabled`` selects the planned local-retrieval shape: ordinary query
    round or fused fast round 1. ``test_profile`` runs pin a single stub
    endpoint and never enable native search, so the check does not apply.
    """

    if test_profile:
        return
    unmet: Sequence[ChainRequirement] = [
        requirement
        for requirement in required_chains(profile, fast_enabled=fast_enabled)
        if not usable_endpoints(requirement)
    ]
    if unmet:
        hints = [
            (
                "retrieval=native 需要绑定的模型组里有 supports_native_search "
                "的成员：出厂预设靠付费 3.8 / 3.7 Flash 接地，免费档只有 2.5 Flash "
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
    for message in routes.preset_binding_warnings(routes.active_preset_id):
        current_reporter().warning("routing-preset", message)


# The six vectors the retired route/level presets mapped to. Everything else is
# expressible but has never been measured -- neither the c coefficient nor the
# output quality (calibration follow-ups live in docs/llm_followups.md).
# Keys are (correction_media, retrieval, difficulty) since D20: the output
# coefficient follows the correction window's media, so the calibration key
# does too. The entries migrated 1:1 -- pre-split runs had both switches equal.
CALIBRATED_VECTORS = frozenset(
    {
        ("text", "none", "efficiency"),
        ("text", "none", "quality"),
        ("text", "native", "quality"),
        ("text", "local", "quality"),
        ("audio", "local", "quality"),
        ("video", "local", "quality"),
    }
)

# Vectors whose measured c predates a repositioning that changed their output
# side: the old measurement no longer calibrates them. They keep the
# old value until P6 re-measures, and say so on every run.
RECALIBRATION_PENDING: dict[Tuple[str, str, str], str] = {
    ("text", "none", "efficiency"): (
        "efficiency 取消输出系数减免（model-routing v2：c 从实测时的 2.0 提到 3.5，"
        "窗口容量降至 57%），且该档仍强制 basicB 变体"
    ),
    ("video", "local", "quality"): (
        "reasoning 措辞深度由 high 降为 medium（旧 mm-high 实测时为 high）"
    ),
}


def profile_warnings(profile: TranslationProfile) -> List[str]:
    """Non-fatal notices about a switch vector that runs but may surprise.

    Both cases are "the run proceeds and produces something reasonable, but not
    what the switches literally promise", which is exactly what a warning is
    for -- rejecting them would remove usable combinations over a catalog fact
    that a future model could change.
    """

    messages: List[str] = []
    # (The old "native pool is basic-tier only" warning died with the v1
    # native-search chain: the variant now comes from the task-group cell, so
    # difficulty=quality gets capableC on whichever native-capable model the
    # user bound -- plan v2 D4.)
    vector = (profile.correction_media, profile.retrieval, profile.difficulty)
    if vector not in CALIBRATED_VECTORS:
        messages.append(
            f"开关组合 {profile.profile_id} 未标定：输出预算系数 "
            f"c={profile.output_coefficient} 是按轴推定值，质量也无实测（计划 P6）。"
        )
    if profile.continuity == "parallel":
        messages.append(
            "continuity=parallel：并发上限与风控形态未标定（计划 P7d，默认取保守值）；"
            "窗口间无 advice 台账与词条透传链，会话级词条集在屏障处一次定死。"
        )
    stale_reason = RECALIBRATION_PENDING.get(vector)
    if stale_reason is not None:
        messages.append(
            f"开关组合 {profile.profile_id} 未重新标定：c="
            f"{profile.output_coefficient} 沿用重构前实测值，但本次重构改变了"
            f"该档的输出侧条件——{stale_reason}——旧标定已失效，P6 重测前按现值运行。"
        )
    return messages


#: The window a model group must be able to take in, and to write out, before
#: a run is willing to plan against it. Warn below the first pair, refuse below
#: the second (owner decision 2026-09-03).
#:
#: Decimal on purpose, not 64Ki/32Ki: `local-claude-haiku-4_5` declares exactly
#: 64000 output, and the owner's ruling is that it passes -- one power of two
#: here would put a healthy model into permanent warning.
WINDOW_WARN_INPUT = 194_000
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
    here would fail `local-claude-haiku-4_5` on its input (200000 - 64000 =
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
