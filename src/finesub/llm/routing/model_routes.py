"""Declarative execution targets, model pools, and ordered route chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Any, Mapping

from .model_catalog import (
    LOCAL_AGENT_KIND,
    PACKAGED_PROVIDER_KINDS,
    ModelCatalogEntry,
    catalog_by_fact_id,
    default_model_catalog,
    get_model_catalog_entry_by_fact,
    get_model_catalog_entry_for_tier,
    quality_floor_warnings,
)


ROUTES_FILENAME = "model_routes.toml"
# Model-routing v2 first cut: two text-only API backends join the packaged pair. They
# exist only for user-declared providers (config.toml [llm.providers]).
# A worker a person is already running, which claims work rather than being
# called. Its own backend rather than another local-agent tier, because the
# direction is what differs: every `backend == "local_agent"` site launches a
# CLI, and a tier that must never be launched would have had to be remembered
# at each of them. An unknown backend is refused everywhere by construction.
CONVERSATIONAL_BACKEND = "conversational_agent"
ALLOWED_BACKENDS = frozenset(
    {"gemini_rest", "local_agent", "openai_compat", "anthropic", CONVERSATIONAL_BACKEND}
)
CUSTOM_PROVIDER_KINDS = frozenset({"openai_compat", "anthropic"})


#: Which execution profile a local-agent tier's auto-built targets get. Only
#: dsh is listed, and that is the whole point: the other three CLIs *are* their
#: account, so their model lists are the vendor's and belong in the packaged
#: catalog. dsh is a harness around whatever provider its owner configured in
#: `$DSH_HOME/settings.yaml`, so the models it can reach are unknowable when
#: this package is built -- a person adds one by writing a row, the same way
#: they add an OpenAI-compatible endpoint.
AUTO_TARGET_LOCAL_AGENT_PROFILES = {
    "LOCAL_DSH": ("dsh-default", "dsh-web-search"),
}


def _auto_target_profile(
    fact: ModelCatalogEntry, profiles: dict[str, ExecutionProfile]
) -> str | None:
    """The profile an unclaimed catalog row's own target should use.

    ``None`` means "do not build one": a packaged-dialect fact with no target
    is a fact the catalog records without wiring up.

    For a local agent the profile is a *driver* property rather than a model
    one -- it is what declares the CLI's own search tool -- so it cannot be
    read off the row. It is taken from the tier instead, with the row's
    ``supports_native_search`` choosing between the tier's two: a person who
    said their model can search gets the target that entitles the tool, and
    one who did not gets the plain one. Tiers absent from the table keep the
    old behaviour of not being auto-wired at all.
    """

    if fact.provider_kind == LOCAL_AGENT_KIND:
        pair = AUTO_TARGET_LOCAL_AGENT_PROFILES.get(fact.provider_tier)
        if pair is None:
            return None
        plain, searching = pair
        wanted = searching if fact.supports_native_search else plain
        if wanted not in profiles:
            raise ModelRouteConfigError(
                f"model_catalog {fact.fact_id!r}: provider tier "
                f"{fact.provider_tier!r} auto-builds targets on execution "
                f"profile {wanted!r}, which the packaged routes do not declare"
            )
        return wanted
    if fact.provider_kind not in CUSTOM_PROVIDER_KINDS:
        return None
    if "custom-default" not in profiles:
        profiles["custom-default"] = ExecutionProfile("custom-default", "")
    return "custom-default"
BACKEND_PROVIDER_TIERS = {
    "gemini_rest": frozenset({"GEMINI_FREE", "GEMINI_PAID"}),
    # One backend, one driver per tier: the tier is what selects the CLI.
    "local_agent": frozenset({"LOCAL_CODEX", "LOCAL_CLAUDE", "LOCAL_AGY", "LOCAL_DSH"}),
    CONVERSATIONAL_BACKEND: frozenset({"LOCAL_CONVERSATIONAL"}),
}
# D8: the v1 chains' step-wise fallback sets were all identical, so groups are
# flat ordered lists and every member shares this uniform fallback policy (the
# final step's ``[]`` was documentary).
STANDARD_FALLBACK = frozenset(
    {"unavailable", "quota", "rate_limit", "timeout", "transient"}
)

# The gate that forbids nothing. Safe as a default only because a policy can no
# longer *add* an agent: the packaged `default` preset names none, so this is
# api-only in practice until somebody binds a group that says otherwise.
DEFAULT_EXECUTION_POLICY = "agent-text-preferred"


class ModelRouteConfigError(ValueError):
    """The packaged route declaration is internally inconsistent."""


@dataclass(frozen=True)
class ProviderSpec:
    """One first-class API provider (model-routing v2).

    The packaged Gemini tiers and the local agent are providers too -- the
    (tier, model, key_id) rate-ledger semantics are unchanged, this is mostly
    the rename the plan promised. Custom providers add a base URL and a
    single-key env name whose value flows through the same encrypted ``.env``
    path as every other key (``secrets.py``); never a separate plaintext store.
    """

    id: str
    kind: str  # gemini | local_agent | openai_compat | anthropic
    base_url: str = ""
    key_env: str = ""


# Derived from the catalog's tier -> dialect map so the two cannot drift.
PACKAGED_PROVIDERS: Mapping[str, ProviderSpec] = MappingProxyType(
    {
        tier: ProviderSpec(tier, kind)
        for tier, kind in PACKAGED_PROVIDER_KINDS.items()
    }
)


def default_key_env(provider_id: str) -> str:
    normalized = "".join(
        ch if ch.isalnum() else "_" for ch in provider_id.upper()
    ).strip("_")
    return f"FINESUB_KEY_{normalized}"


@dataclass(frozen=True)
class ExecutionProfile:
    id: str
    native_search_tool: str = ""


@dataclass(frozen=True)
class ExecutionTarget:
    id: str
    backend: str
    fact_id: str
    enabled_by: str
    execution_profile: str


@dataclass(frozen=True)
class RoutePolicy:
    """Which backends may answer at all. A gate, never a source.

    Policies once also prepended model groups onto every cell, which meant the
    bound group said one thing and the policy silently prefixed another. Agents
    now take part by being members of a bound group, so what is left is the one
    guarantee group membership cannot express: "no local CLI whatever a group
    lists", or "not one token of API quota".
    """

    id: str
    allowed_backends: frozenset[str]


# --- v2 declarative layer (docs/manual/model-routing.md) --------------------

# The seven task groups are harness-defined: sessions map onto them (§6), so
# the set is closed -- users compose model groups and presets, not new tasks.
TASK_GROUP_IDS = (
    "correction-mm",
    "correction-text",
    "planning-mm",
    "planning-text",
    "research",
    "search_judge",
    "knowledge",
)
DIFFICULTIES = ("quality", "intermediate", "efficiency")
# Within-preset difficulty fallback (owner decision 2026-08-11): an unbound
# cell walks up toward high before crossing over to the default preset.
DIFFICULTY_FALLBACK = {
    "quality": ("quality",),
    "intermediate": ("intermediate", "quality"),
    "efficiency": ("efficiency", "intermediate", "quality"),
}
# The four named prompt variants plus "" (the single-template sessions).
ALLOWED_VARIANTS = frozenset({"", "capableB", "capableC", "basicA", "basicB"})
# §5.3: the envelope warning baseline is today's free-Gemini ceiling, not the
# in-group spread -- a small-context member is worth flagging even in a group
# of equals. It survived the 2026-09 window-limit rewrite unchanged because
# free Gemini's envelope still lands on it exactly: it declares no context
# window, so the pool is `194000 + 65536` and the ceiling is `max_input`.
ENVELOPE_BASELINE_INPUT = 194_000
ENVELOPE_BASELINE_OUTPUT = 32_768


@dataclass(frozen=True)
class ModelGroup:
    id: str
    target_ids: tuple[str, ...]
    # Per-entry variant override (D3): prompt iteration / user groups only;
    # the default preset never sets it.
    variant_overrides: Mapping[str, str] = MappingProxyType({})


# The namespace a binding's quick model selection lands in. Declared group ids
# may not use it, so a synthesised group can never shadow or be shadowed by one
# somebody wrote down.
SINGLE_TARGET_GROUP_PREFIX = "target:"


def _wrap_single_target(
    target_id: str, model_groups: dict[str, ModelGroup]
) -> str:
    """Give one target the shape the rest of routing expects: a group.

    Everything downstream -- the plan, the trace, the resume digest, the floor
    warning -- reads a group, so wrapping here means quick selection is a
    parsing convenience and not a second code path with its own bugs.
    """

    group_id = f"{SINGLE_TARGET_GROUP_PREFIX}{target_id}"
    model_groups.setdefault(
        group_id, ModelGroup(group_id, (target_id,), MappingProxyType({}))
    )
    return group_id


# Where a preferred-target overlay used to land. Still reserved so a declared
# group can never squat a namespace synthesised groups have ever used.
PREFERRED_GROUP_PREFIX = "preferred:"

# Process-local routing override (kb-followups plan B): `--llm-model` installs
# it once per CLI entry, BEFORE any routing consumption, and every consumer —
# planning envelopes, capability preflight, research/correction's internal
# RoleClient() constructions, resume identity — reads it through the memoized
# `default_model_routes()`. Lifecycle: entry points call install
# unconditionally (empty → clears), so repeated `main()` calls in one
# interpreter never leak the previous run's override.
_RUNTIME_PREFERRED: dict[str, str] = {}


def install_runtime_preferred(overlay: Mapping[str, str] | None) -> None:
    """Install (or clear, with None/empty) the process-local `--llm-model`
    override. Values follow `[llm.preferred_targets]` semantics — a model
    group or a route target — validated at the next route load."""

    global _RUNTIME_PREFERRED
    _RUNTIME_PREFERRED = {str(k).strip(): str(v).strip() for k, v in (overlay or {}).items()}
    default_model_routes.cache_clear()


def runtime_preferred() -> dict[str, str]:
    return dict(_RUNTIME_PREFERRED)


def parse_llm_model_args(values: list[str] | None) -> dict[str, str]:
    """`--llm-model [task-group=]value` (repeatable) → overlay table.

    A bare value is the `default` key; later occurrences of the same key win.
    Name validation happens at route load, where the tables live."""

    if isinstance(values, str):  # a lone value, not argparse's append list
        values = [values]
    overlay: dict[str, str] = {}
    for raw in values or []:
        raw = str(raw).strip()
        key, sep, value = raw.partition("=")
        if not sep:
            key, value = "default", raw
        key, value = key.strip(), value.strip()
        if not key or not value:
            raise ModelRouteConfigError(
                f"--llm-model expects [任务组=]模型组或target，got {raw!r}"
            )
        overlay[key] = value
    return overlay


def _preferred_bindings(
    table: Mapping[str, Any],
    targets: Mapping[str, ExecutionTarget],
    model_groups: dict[str, ModelGroup],
    *,
    source: str,
) -> dict[str, str]:
    """One preferred table resolved to `{task group or "default": group id}`.

    A value may name a **model group** (the cell rebinds to it wholesale) or a
    **target** (the cell rebinds to a synthesised single-member group). Both
    REPLACE the bound chain — no fallback behind the pin (owner decision
    2026-08-28): a call the pinned model cannot serve fails loudly instead of
    silently routing elsewhere; `-mm` load-time member checks still run on the
    rebound bindings and warn at startup."""

    resolved: dict[str, str] = {}
    for key, value in table.items():
        name = str(key).strip()
        if name != "default" and name not in TASK_GROUP_IDS:
            raise ModelRouteConfigError(
                f"{source}.{name} names no task group; known: "
                f"default, {', '.join(sorted(TASK_GROUP_IDS))}"
            )
        if not isinstance(value, str) or not value.strip():
            raise ModelRouteConfigError(f"{source}.{name} must name a model group or target")
        ident = value.strip()
        in_groups = ident in model_groups
        in_targets = ident in targets
        if in_groups and in_targets:
            raise ModelRouteConfigError(
                f"{source}.{name} = {ident!r} is ambiguous: it names both a "
                "model group and a target — rename one"
            )
        if in_groups:
            resolved[name] = ident
        elif in_targets:
            resolved[name] = _wrap_single_target(ident, model_groups)
        else:
            raise ModelRouteConfigError(
                f"{source}.{name} names no known target or model group: {ident!r}; "
                f"known groups: {', '.join(sorted(g for g in model_groups if ':' not in g))}"
            )
    return resolved


def _with_preferred_bindings(
    preset: Preset,
    runtime: Mapping[str, str],
    config: Mapping[str, str],
    model_groups: dict[str, ModelGroup],
) -> Preset:
    """Rebind cells by the four-step chain
    `CLI[组] > CLI[default] > config[组] > config[default]` — resolved per
    task group, never dict-merged: a bare `--llm-model X` overrides a
    config-side per-group pin (the user said "this run, use X"), while the
    config's other task-group pins stay live where the CLI is silent."""

    bindings = dict(preset.bindings)
    for (task_group_id, difficulty), _group_id in preset.bindings.items():
        bound = (
            runtime.get(task_group_id)
            or runtime.get("default")
            or config.get(task_group_id)
            or config.get("default")
        )
        if bound is None:
            continue
        bindings[(task_group_id, difficulty)] = bound
    return replace(preset, bindings=MappingProxyType(bindings))


@dataclass(frozen=True)
class TaskGroupCell:
    variant: str


@dataclass(frozen=True)
class TaskGroup:
    id: str
    floor_score: int
    cells: Mapping[str, TaskGroupCell]
    # ``reference_model`` is not stored: it is derived from the default
    # preset's high binding (``ModelRouteCatalog.task_group_reference_model``).


# The thinking knob's global default: an entirely unfilled preset thinks
# medium everywhere (owner design 2026-08-11).
DEFAULT_THINKING_LEVEL = "medium"

# How long one headless agent conversation lives, in harness LLM sessions
# (owner decision 2026-08-14; four-tier redefinition 2026-08-19, the table is
# docs/llm_local_agent.md §12.1):
#   api                  one conversation per call -- the full-replay baseline,
#                        no session to retire, one retry layer by definition
#   per-window           one conversation per window: a window's whole repair
#                        chain resumes the conversation that wrote the output
#   resume               one provider conversation resumed across the windows
#                        of a run (per worker lane under parallel continuity)
#   pseudo-conversational several harness sessions inside one agent invocation
# The vendors differ enough here -- cache thresholds, inheritance and idle TTLs
# are all vendor-specific (docs/llm_local_agent_experiments.md §3.2) -- that this
# is set by the owner rather than tuned per install. A user who wants a
# non-preset agent uses conversational mode instead of turning knobs here.
AGENT_SESSION_MODES = ("api", "per-window", "resume", "pseudo-conversational")
# An entirely unfilled preset uses per-window: that is what production has done
# since the in-window repair wiring (2026-08-17), so the default must name it
# rather than the first tuple entry (the tuple is ordered by session lifetime,
# and its first entry -- the `api` baseline -- would be a behavior regression).
DEFAULT_AGENT_SESSION_MODE = "per-window"


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    test_target_id: str
    bindings: Mapping[tuple[str, str], str]
    # Per-cell thinking knob ("task_group/difficulty" -> low|medium|high),
    # resolved with the same difficulty-first fallback as bindings; entirely
    # unfilled cells default to medium. The knob is the abstract level; each
    # model's catalog `thinking` mapping translates it per candidate.
    thinking: Mapping[tuple[str, str], str] = MappingProxyType({})
    # Per-cell agent session mode, same key shape and same fallback as
    # ``thinking``; entirely unfilled cells use DEFAULT_AGENT_SESSION_MODE.
    agent_session: Mapping[tuple[str, str], str] = MappingProxyType({})


# Fact fields that only change how we evaluate or display a model, never what
# a call selects, prompts, or budgets (model-routing v2). They feed advisory_digest
# (artifacts only) and stay out of routing_identity_digest (resume keys), so
# editing a score or a display name does not invalidate checkpoints.
ADVISORY_FACT_FIELDS = frozenset({"display_name", "quality_score", "self_reported"})


@dataclass(frozen=True)
class ModelRouteCatalog:
    routing_identity_digest: str
    advisory_digest: str
    facts: Mapping[str, ModelCatalogEntry]
    execution_profiles: Mapping[str, ExecutionProfile]
    targets: Mapping[str, ExecutionTarget]
    policies: Mapping[str, RoutePolicy]
    model_groups: Mapping[str, ModelGroup] = MappingProxyType({})
    task_groups: Mapping[str, TaskGroup] = MappingProxyType({})
    presets: Mapping[str, Preset] = MappingProxyType({})
    providers: Mapping[str, ProviderSpec] = PACKAGED_PROVIDERS
    # User-declared facts (config.toml [llm.models]): unverified numbers the
    # user typed in. Artifacts label them so they never mix silently with
    # measured facts; fail-closed is relaxed for them (plan §10).
    self_reported_fact_ids: frozenset[str] = frozenset()
    active_preset_id: str = "default"
    # Packaged ids the user's config.toml replaced. Announced once at startup:
    # same-id override is intentional (2026-08-12) but a typo looks identical.
    override_notices: tuple[str, ...] = ()

    def provider_for_target(self, target_id: str) -> ProviderSpec:
        return self.providers[self.target_fact(target_id).provider_tier]

    def binds_local_agent(self) -> bool:
        """Can any cell this run may reach dispatch to a local agent CLI?

        Not answerable from the execution policy any more: a policy only
        subtracts backends, so `agent-text-preferred` over a preset of Gemini
        targets reaches no agent at all. Media handling asks this before
        deciding to defer a clip upload -- deferring under a plan that will
        certainly answer from the API just moves the upload later.
        """

        return any(
            self.targets[target_id].backend == "local_agent"
            for preset_id in {"default", self.active_preset_id}
            for group_id in self.presets[preset_id].bindings.values()
            for target_id in self.model_groups[group_id].target_ids
        )

    def resolve_binding(
        self, preset_id: str, task_group_id: str, difficulty: str
    ) -> tuple[ModelGroup, TaskGroupCell]:
        """The model group + cell serving one (task group, difficulty).

        Difficulty fallback first (owner decision 2026-08-11): an unbound
        intermediate/efficiency cell reuses the next-higher bound cell *of the same
        preset* -- difficulty only changes the prompt unless a preset says
        otherwise, so switching it must not silently swap in another
        preset's group. Only when the preset binds nothing for the task
        group at all does resolution cross over to the default preset
        (users override just the cells they care about, §5.5). The returned
        cell is always the *requested* difficulty's (variant/thinking stay).
        """

        preset = self.presets[preset_id]
        for fallback_difficulty in DIFFICULTY_FALLBACK[difficulty]:
            group_id = preset.bindings.get((task_group_id, fallback_difficulty))
            if group_id is not None:
                return (
                    self.model_groups[group_id],
                    self.task_groups[task_group_id].cells[difficulty],
                )
        if preset_id != "default":
            group, _cell = self.resolve_binding(
                "default", task_group_id, difficulty
            )
            return group, self.task_groups[task_group_id].cells[difficulty]
        raise ModelRouteConfigError(
            f"No binding for {task_group_id}/{difficulty} in preset "
            f"{preset_id!r} (and no default fallback)"
        )

    def resolve_thinking(
        self, preset_id: str, task_group_id: str, difficulty: str
    ) -> str:
        """The thinking knob for one cell (owner design 2026-08-11).

        Same resolution shape as bindings: difficulty-first within the
        preset, then the default preset, then the global medium default. The
        result is the abstract level; the answering model's catalog
        ``thinking`` mapping turns it into a provider value per candidate.
        """

        for lookup_id in dict.fromkeys((preset_id, "default")):
            preset = self.presets[lookup_id]
            for fallback_difficulty in DIFFICULTY_FALLBACK[difficulty]:
                level = preset.thinking.get((task_group_id, fallback_difficulty))
                if level is not None:
                    return level
        return DEFAULT_THINKING_LEVEL

    def resolve_agent_session(
        self, preset_id: str, task_group_id: str, difficulty: str
    ) -> str:
        """The agent session mode for one cell (owner decision 2026-08-14).

        Same resolution as ``thinking``: difficulty-first within the preset,
        then the default preset, then the global default -- so a table that
        only pins ``quality`` still answers for the cheaper difficulties.
        """

        for lookup_id in dict.fromkeys((preset_id, "default")):
            preset = self.presets[lookup_id]
            for fallback_difficulty in DIFFICULTY_FALLBACK[difficulty]:
                mode = preset.agent_session.get((task_group_id, fallback_difficulty))
                if mode is not None:
                    return mode
        return DEFAULT_AGENT_SESSION_MODE

    def reachable_group_ids(self) -> tuple[str, ...]:
        """Every model group a run under the active preset could dispatch to.

        The active preset *and* the default one, because a task group the
        active preset binds nothing for falls back across to the default
        (`resolve_binding`). Deliberately not "every group in the file": a
        packaged `model_routes.toml` also carries the `agy-*` / `dsh-*` groups
        for presets this run never selects, and a check that walked those would
        answer for somebody else's configuration.
        """

        return tuple(
            dict.fromkeys(
                group_id
                for preset_id in dict.fromkeys(("default", self.active_preset_id))
                for group_id in self.presets[preset_id].bindings.values()
            )
        )

    def group_declared_minima(self, group_id: str) -> tuple[int, int]:
        """(min max_input_tokens, min max_output_tokens) over the group.

        The catalog's two columns as declared, with no arithmetic between them
        -- which is what makes this a different question from
        `group_planning_envelope` below, not a cheaper version of it.

        Planning has to be conservative about the *joint* budget, so the
        envelope subtracts the answer from each member's context window. This
        one asks about **shape**: can this member take a long enough input,
        and can it write a long enough answer. A model whose two limits are
        both ample while their sum exceeds its context window is a normal,
        healthy model -- `local-claude-haiku-4_5` (200000 / 64000 in a 200000
        window) is the worked example, and the owner's 2026-09-03 ruling that
        it passes is what these two methods are kept apart for.
        """

        facts = [
            self.target_fact(target_id)
            for target_id in self.model_groups[group_id].target_ids
        ]
        return (
            min(fact.max_input_tokens for fact in facts),
            min(fact.max_output_tokens for fact in facts),
        )

    def group_planning_envelope(self, group_id: str) -> tuple[int, int]:
        """(input ceiling, min max_output) over the group (D13).

        Window planning happens before "who answers" is known, so it must be
        conservative; each candidate is still hard-checked against its own
        limits at dispatch (v1 behaviour, unchanged).

        **Two passes, not a per-column min.** The output is settled first,
        because every member will be asked for that many tokens; only then does
        each member say how much input it can still take -- its own
        `max_input_tokens`, or what is left of its context window once the
        answer is reserved. Taking the min of `context_window` and of
        `max_output_tokens` separately and subtracting would *overestimate* in a
        heterogeneous group: a member with `ctx=200000/out=64000` (ceiling
        136000) beside one with `ctx=210000/out=128000` (ceiling 82000) yields
        `200000 - 64000 = 136000`, 54000 above what the second member can
        actually take -- the one direction a planning envelope may not err in.
        """

        group = self.model_groups[group_id]
        facts = [self.target_fact(target_id) for target_id in group.target_ids]
        min_output = min(fact.max_output_tokens for fact in facts)
        input_ceiling = min(
            min(fact.max_input_tokens, fact.context_window - min_output)
            for fact in facts
        )
        return (max(1, input_ceiling), min_output)

    def group_estimate_scale(self, group_id: str) -> float:
        """The group's conservative ``token_scale`` for window planning.

        The largest member's scale, floored at 1.0: planning must hold for
        whichever candidate answers, and a member declaring < 1 (the local
        counter over-estimates for it) must not be allowed to *relax* a bound
        the other members still need.
        """

        group = self.model_groups[group_id]
        return max(
            1.0,
            *(
                float(self.target_fact(target_id).token_scale or 1.0)
                for target_id in group.target_ids
            ),
        )

    def preset_binding_warnings(self, preset_id: str) -> list[str]:
        """Binding-time advisory warnings for one preset (§5.3/§5.4/§8).

        Warnings, not errors: they flag configurations that run but deserve a
        look -- a member under the task group's floor, a member that drags the
        planning envelope down, an -mm cell whose member cannot hear. The
        shipped default preset must return [] (tested).
        """

        preset = self.presets[preset_id]
        messages: list[str] = []
        seen_groups: set[str] = set()
        for task_group_id in TASK_GROUP_IDS:
            for difficulty in DIFFICULTIES:
                group, _cell = self.resolve_binding(
                    preset_id, task_group_id, difficulty
                )
                task_group = self.task_groups[task_group_id]
                facts = [
                    self.target_fact(target_id) for target_id in group.target_ids
                ]
                # The floor guards the task's *nominal* quality, so it checks
                # the quality cell only (owner decision 2026-08-11): binding a
                # cheaper group into intermediate/efficiency IS the declared downgrade
                # tier -- warning about it would be noise.
                if difficulty == "quality":
                    messages.extend(
                        quality_floor_warnings(
                            facts,
                            floor_score=task_group.floor_score,
                            reference_model=self.task_group_reference_model(
                                task_group_id, difficulty
                            ),
                            owner=f"{task_group_id}/{difficulty}",
                        )
                    )
                if task_group_id.endswith("-mm"):
                    # Report the *count*, not a boolean. Warning per text-only
                    # member made the shipped `agy` preset print six lines a run
                    # for a deliberate pattern (Opus first so text windows
                    # prefer it, media windows skip past it for free). But
                    # collapsing that to "does anyone here hear?" threw away the
                    # thing worth knowing: a four-member cell whose media calls
                    # have exactly one real candidate looks perfectly healthy,
                    # and its chain is one failure deep.
                    deaf = [fact.fact_id for fact in facts if not fact.supports_audio]
                    hearing = len(facts) - len(deaf)
                    if hearing == 0:
                        messages.append(
                            f"{task_group_id}/{difficulty}: 组内没有成员支持音频"
                            f"（{', '.join(deaf)}），带媒体的调用会在能力过滤后无候选可用"
                        )
                    elif hearing == 1 and deaf:
                        messages.append(
                            f"{task_group_id}/{difficulty}: {len(facts)} 个成员里只有 1 个"
                            f"支持音频，带媒体的调用链长为 1"
                            f"（纯文本成员：{', '.join(deaf)}）"
                        )
                if group.id not in seen_groups:
                    seen_groups.add(group.id)
                    input_ceiling, min_output = self.group_planning_envelope(
                        group.id
                    )
                    # By ceiling, not by `max_input_tokens`: a member can be the
                    # one that binds because its context window has to hold the
                    # answer too, while its declared input limit is the largest
                    # in the group.
                    limiting = min(
                        facts,
                        key=lambda fact: min(
                            fact.max_input_tokens, fact.context_window - min_output
                        ),
                    ).fact_id
                    if (
                        input_ceiling < ENVELOPE_BASELINE_INPUT
                        or min_output < ENVELOPE_BASELINE_OUTPUT
                    ):
                        ratio = max(
                            1.0, ENVELOPE_BASELINE_INPUT / max(1, input_ceiling)
                        )
                        messages.append(
                            f"模型组 {group.id}: 规划包络由 {limiting} 决定 "
                            f"({input_ceiling}/{min_output} tokens，基线 "
                            f"{ENVELOPE_BASELINE_INPUT}/{ENVELOPE_BASELINE_OUTPUT})，"
                            f"窗口数约 ×{ratio:.1f}"
                        )
        return messages

    def target_fact(self, target_id: str) -> ModelCatalogEntry:
        target = self.targets[target_id]
        return self.facts[target.fact_id]

    def target_profile(self, target_id: str) -> ExecutionProfile:
        target = self.targets[target_id]
        return self.execution_profiles[target.execution_profile]

    def fact_for_endpoint(
        self, provider_tier: str, api_model_id: str, fact_id: str = ""
    ) -> ModelCatalogEntry | None:
        """The runtime fact behind a dispatched endpoint, user models included.

        Fact id first (the router always carries one), then the
        (provider, model) pair -- which is what the rate limiter reconstructs
        deep inside ``chat_complete``, where only the tier and model name
        survive. Both resolve against the *merged* fact table, so a
        user-declared model is no longer invisible to capability filtering and
        limit lookups (the packaged catalog alone cannot see it).
        """

        if fact_id:
            fact = self.facts.get(fact_id)
            if fact is not None:
                return fact
        for fact in self.facts.values():
            if (
                fact.provider_tier == provider_tier
                and fact.api_model_id == api_model_id
            ):
                return fact
        return None

    def task_group_reference_model(
        self, task_group_id: str, difficulty: str = "quality"
    ) -> str:
        """Display name of the model a cell is calibrated to.

        Derived, not declared (owner decision 2026-08-12): the first member of
        the group the **default** preset binds to that cell -- the model the
        expectations were written for, by construction. Deriving it keeps one
        truth when the roster changes and keeps a display-only string out of
        the routing identity.

        Per cell rather than per task group, because a preset may bind a
        different group per difficulty: correction's ``quality`` is calibrated to
        the leading full Flash while its ``intermediate`` (the declared downgrade tier) is
        calibrated to 3.5 Flash Lite. The floor warning still only fires on
        ``quality``, so that is the default.
        """

        group, _cell = self.resolve_binding("default", task_group_id, difficulty)
        if not group.target_ids:
            return ""
        return self.target_fact(group.target_ids[0]).display_name


def _routes_path() -> Path:
    return Path(__file__).resolve().with_name(ROUTES_FILENAME)


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, Mapping):
        raise ModelRouteConfigError(f"[{name}] must be a TOML table")
    return value


def _required_string(data: Mapping[str, Any], key: str, owner: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelRouteConfigError(f"{owner}.{key} must be a non-empty string")
    return value.strip()


def load_model_routes(
    path: str | Path | None = None,
    *,
    user_config: Mapping[str, Any] | None = None,
    catalog: Any | None = None,
) -> ModelRouteCatalog:
    """Load the packaged declaration over a model catalog.

    Two inputs, two jobs (owner decision 2026-08-12):

    - the **catalog** supplies the facts -- capabilities, limits, the endpoint
      dialect and URL. Rows the packaged targets do not claim become targets of
      their own, so adding a model is one line in the override catalog.
    - ``user_config`` (the ``[llm]`` table of ``config.toml``) supplies the
      **composition**: model groups, presets, the active ``preset``. Everything
      is named tables -- no array-of-tables, so the scalar-only config writer
      keeps working on the same file.

    ``catalog`` defaults to the merged packaged+override catalog; tests pass an
    explicit sequence of entries to stay hermetic.
    """

    route_path = Path(path).expanduser() if path is not None else _routes_path()
    route_bytes = route_path.read_bytes()
    data = tomllib.loads(route_bytes.decode("utf-8"))
    user = user_config or {}
    if not isinstance(user, Mapping):
        raise ModelRouteConfigError("[llm] must be a TOML table")

    facts = catalog_by_fact_id(
        default_model_catalog() if catalog is None else catalog
    )
    profiles: dict[str, ExecutionProfile] = {}
    for profile_id, raw in _table(data, "execution_profiles").items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(
                f"execution_profiles.{profile_id} must be a table"
            )
        tool = raw.get("native_search_tool", "")
        if not isinstance(tool, str):
            raise ModelRouteConfigError(
                f"execution_profiles.{profile_id}.native_search_tool must be a string"
            )
        profiles[profile_id] = ExecutionProfile(profile_id, tool.strip())

    targets: dict[str, ExecutionTarget] = {}
    for target_id, raw in _table(data, "targets").items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(f"targets.{target_id} must be a table")
        target = ExecutionTarget(
            id=target_id,
            backend=_required_string(raw, "backend", f"targets.{target_id}"),
            fact_id=_required_string(raw, "fact", f"targets.{target_id}"),
            enabled_by=_required_string(raw, "enabled_by", f"targets.{target_id}"),
            execution_profile=_required_string(
                raw, "execution_profile", f"targets.{target_id}"
            ),
        )
        if target.backend not in ALLOWED_BACKENDS:
            raise ModelRouteConfigError(
                f"Target {target_id!r} has unknown backend {target.backend!r}"
            )
        if target.fact_id not in facts:
            raise ModelRouteConfigError(
                f"Target {target_id!r} references missing fact {target.fact_id!r}"
            )
        if target.execution_profile not in profiles:
            raise ModelRouteConfigError(
                f"Target {target_id!r} references missing execution profile "
                f"{target.execution_profile!r}"
            )
        fact = facts[target.fact_id]
        profile = profiles[target.execution_profile]
        expected_tiers = BACKEND_PROVIDER_TIERS.get(target.backend)
        if expected_tiers is None:
            # openai_compat / anthropic exist only for user-declared providers,
            # which build their targets below rather than through this table.
            raise ModelRouteConfigError(
                f"Target {target_id!r} declares backend {target.backend!r}, "
                "which only user-declared providers ([llm.providers]) may use"
            )
        if fact.provider_tier not in expected_tiers:
            raise ModelRouteConfigError(
                f"Target {target_id!r} backend {target.backend!r} cannot use fact "
                f"provider tier {fact.provider_tier!r}; expected one of "
                f"{sorted(expected_tiers)!r}"
            )
        if target.backend == "gemini_rest" and target.enabled_by != fact.provider_tier:
            raise ModelRouteConfigError(
                f"Gemini target {target_id!r} enabled_by {target.enabled_by!r} "
                f"does not match fact tier {fact.provider_tier!r}"
            )
        if target.backend == "local_agent" and target.enabled_by != "local_agent":
            raise ModelRouteConfigError(
                f"Local-agent target {target_id!r} must use enabled_by='local_agent'"
            )
        allowed_tools = {
            "gemini_rest": {"", "google_search"},
            # One per vendor, spelled the vendor's way: Codex and Claude Code
            # both call it `web_search`, agy calls it `search_web`. The name is
            # what the driver entitles and what the event stream reports, so it
            # cannot be normalised away here.
            "local_agent": {"", "web_search", "search_web"},
            # The harness does not configure a tool set it never launches: what
            # an attached agent can reach is whatever its own host granted it.
            CONVERSATIONAL_BACKEND: {""},
        }[target.backend]
        if profile.native_search_tool not in allowed_tools:
            raise ModelRouteConfigError(
                f"Target {target_id!r} backend {target.backend!r} does not support "
                f"native_search_tool={profile.native_search_tool!r}; expected one of "
                f"{sorted(allowed_tools)!r}"
            )
        if profile.native_search_tool and not fact.supports_native_search:
            raise ModelRouteConfigError(
                f"Target {target_id!r} enables native search tool "
                f"{profile.native_search_tool!r}, but its runtime fact does not support it"
            )
        targets[target_id] = target

    if not profiles or not targets:
        raise ModelRouteConfigError(f"{route_path} has an empty required table")

    # --- Providers and targets derived from the catalog ---------------------
    #
    # Facts belong to the catalog, composition to config.toml (owner decision
    # 2026-08-12): the endpoint dialect and URL are facts, so they are catalog
    # columns, and ``[llm.providers]`` / ``[llm.models]`` are gone. Every
    # catalog row that no packaged target already claims becomes a target of
    # its own, which is what lets a user add a model by writing one line in
    # the override catalog.
    providers: dict[str, ProviderSpec] = dict(PACKAGED_PROVIDERS)
    self_reported = {
        fact.fact_id for fact in facts.values() if fact.self_reported
    }
    claimed_fact_ids = {target.fact_id for target in targets.values()}
    for fact_id, fact in facts.items():
        declared = providers.get(fact.provider_tier)
        if declared is None:
            if fact.provider_kind not in CUSTOM_PROVIDER_KINDS:
                raise ModelRouteConfigError(
                    f"model_catalog {fact_id!r}: provider_kind="
                    f"{fact.provider_kind!r} is a packaged dialect, but "
                    f"{fact.provider_tier!r} is not a packaged provider tier"
                )
            providers[fact.provider_tier] = ProviderSpec(
                fact.provider_tier,
                fact.provider_kind,
                fact.base_url,
                fact.key_env or default_key_env(fact.provider_tier),
            )
        elif (
            fact.provider_kind != declared.kind
            or (fact.base_url and fact.base_url != declared.base_url)
            or (fact.key_env and fact.key_env != declared.key_env)
        ):
            # Rows sharing a provider tier describe the same endpoint, and the
            # provider is what owns the key. Letting any of the three disagree
            # would make the answer depend on row order -- and for key_env that
            # is silent: the second row's key is simply never used, while the
            # config file looks exactly right.
            raise ModelRouteConfigError(
                f"model_catalog {fact_id!r}: provider {fact.provider_tier!r} is "
                f"already declared as {declared.kind} at "
                f"{declared.base_url or '(packaged)'} with key_env "
                f"{declared.key_env or '(packaged pool)'}"
            )
        if fact_id in claimed_fact_ids:
            continue
        profile_id = _auto_target_profile(fact, profiles)
        if profile_id is None:
            # A packaged-dialect fact with no target is simply not wired up
            # (the catalog may record facts for experiments -- plan D3).
            continue
        if fact_id in targets:
            raise ModelRouteConfigError(
                f"model_catalog {fact_id!r} collides with a declared target id"
            )
        targets[fact_id] = ExecutionTarget(
            id=fact_id,
            backend=(
                "local_agent"
                if fact.provider_kind == LOCAL_AGENT_KIND
                else fact.provider_kind
            ),
            fact_id=fact_id,
            enabled_by=(
                "local_agent"
                if fact.provider_kind == LOCAL_AGENT_KIND
                else fact.provider_tier
            ),
            execution_profile=profile_id,
        )
    for section in ("providers", "models"):
        if _table(user, section):
            raise ModelRouteConfigError(
                f"[llm.{section}] moved into the model catalog (2026-08-12): "
                "declare the model as a row in the data root's "
                "model_catalog.psv (provider_kind/base_url/key_env are columns "
                "now) and keep config.toml for model groups and presets"
            )

    # Same-id means "mine wins", symmetric with the catalog's fact_id override
    # (owner decision 2026-08-12): one mental model for both files instead of
    # "your catalog row overrides, your group name is a collision error".
    # What that costs is the typo guard, so every override is announced once
    # at startup rather than happening silently.
    override_notices: list[str] = []
    model_groups: dict[str, ModelGroup] = {}
    raw_model_groups = dict(_table(data, "model_groups"))
    for group_id, raw in _table(user, "model_groups").items():
        if group_id in raw_model_groups:
            override_notices.append(f"模型组 {group_id}")
        raw_model_groups[group_id] = raw
    for group_id, raw in raw_model_groups.items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(f"model_groups.{group_id} must be a table")
        if group_id.startswith(SINGLE_TARGET_GROUP_PREFIX):
            raise ModelRouteConfigError(
                f"model_groups.{group_id}: the {SINGLE_TARGET_GROUP_PREFIX!r} "
                "prefix is reserved for bindings that name a single target"
            )
        if group_id.startswith(PREFERRED_GROUP_PREFIX):
            raise ModelRouteConfigError(
                f"model_groups.{group_id}: the {PREFERRED_GROUP_PREFIX!r} "
                "prefix is reserved for preferred-target overlays"
            )
        members = raw.get("targets")
        if not isinstance(members, list) or not members or any(
            not isinstance(member, str) or not member for member in members
        ):
            raise ModelRouteConfigError(
                f"model_groups.{group_id}.targets must be a non-empty string array"
            )
        if len(set(members)) != len(members):
            raise ModelRouteConfigError(
                f"Model group {group_id!r} contains duplicate targets"
            )
        missing = [member for member in members if member not in targets]
        if missing:
            raise ModelRouteConfigError(
                f"Model group {group_id!r} references missing targets: "
                f"{', '.join(missing)}"
            )
        raw_overrides = raw.get("variant_overrides", {})
        if not isinstance(raw_overrides, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in raw_overrides.items()
        ):
            raise ModelRouteConfigError(
                f"model_groups.{group_id}.variant_overrides must map targets "
                "to variant names"
            )
        bad_override_targets = sorted(set(raw_overrides) - set(members))
        bad_override_variants = sorted(
            value
            for value in raw_overrides.values()
            if value not in ALLOWED_VARIANTS or not value
        )
        if bad_override_targets or bad_override_variants:
            raise ModelRouteConfigError(
                f"Model group {group_id!r} has invalid variant overrides: "
                f"targets={bad_override_targets}, variants={bad_override_variants}"
            )
        # A conversational worker cannot be a step in a fallback chain, in
        # either direction. Falling back *to* it would mean calling something
        # that only answers if a person happens to have an agent attached right
        # now; falling back *from* it would mean taking over a task that agent
        # may be halfway through. Alone in its own group, "which group is bound"
        # stays a real choice rather than a silent step in someone else's list.
        conversational = [
            member
            for member in members
            if targets[member].backend == CONVERSATIONAL_BACKEND
        ]
        if conversational and len(members) > 1:
            raise ModelRouteConfigError(
                f"Model group {group_id!r} mixes the conversational target "
                f"{conversational[0]!r} with other members; a conversational "
                "worker claims work rather than being called, so it cannot be "
                "a fallback step -- give it a group of its own"
            )
        model_groups[group_id] = ModelGroup(
            group_id,
            tuple(members),
            MappingProxyType(dict(raw_overrides)),
        )

    task_groups: dict[str, TaskGroup] = {}
    raw_task_groups = _table(data, "task_groups")
    if set(raw_task_groups) != set(TASK_GROUP_IDS):
        raise ModelRouteConfigError(
            f"[task_groups] must define exactly {sorted(TASK_GROUP_IDS)}; "
            f"got {sorted(raw_task_groups)}"
        )
    for group_id, raw in raw_task_groups.items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(f"task_groups.{group_id} must be a table")
        floor = raw.get("floor_score")
        if isinstance(floor, bool) or not isinstance(floor, int) or not 0 <= floor <= 100:
            raise ModelRouteConfigError(
                f"task_groups.{group_id}.floor_score must be an integer in [0, 100]"
            )
        raw_cells = raw.get("cells")
        if not isinstance(raw_cells, Mapping) or set(raw_cells) != set(DIFFICULTIES):
            raise ModelRouteConfigError(
                f"task_groups.{group_id}.cells must define exactly "
                f"{sorted(DIFFICULTIES)}"
            )
        cells: dict[str, TaskGroupCell] = {}
        for difficulty, raw_cell in raw_cells.items():
            owner = f"task_groups.{group_id}.cells.{difficulty}"
            if not isinstance(raw_cell, Mapping):
                raise ModelRouteConfigError(f"{owner} must be a table")
            variant = raw_cell.get("variant", "")
            if variant not in ALLOWED_VARIANTS:
                raise ModelRouteConfigError(
                    f"{owner}.variant must be one of {sorted(ALLOWED_VARIANTS)}"
                )
            # Thinking left the cell (2026-08-11): it is a preset-level knob
            # now ([presets.<id>.thinking]).
            cells[difficulty] = TaskGroupCell(variant)
        task_groups[group_id] = TaskGroup(
            group_id, floor, MappingProxyType(cells)
        )

    presets: dict[str, Preset] = {}
    raw_presets = dict(_table(data, "presets"))
    if "default" not in raw_presets:
        raise ModelRouteConfigError("[presets] must define the 'default' preset")
    for preset_id, raw in _table(user, "presets").items():
        if preset_id in raw_presets:
            # Overriding "default" is allowed and safe: the completeness check
            # below runs on whatever ends up as the default preset.
            override_notices.append(f"预设 {preset_id}")
        raw_presets[preset_id] = raw
    default_test_target = (
        str(raw_presets["default"].get("test_target") or "").strip()
        if isinstance(raw_presets["default"], Mapping)
        else ""
    )
    for preset_id, raw in raw_presets.items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(f"presets.{preset_id} must be a table")
        name = _required_string(raw, "name", f"presets.{preset_id}")
        # User presets may omit test_target and inherit the default's (§5.5).
        if preset_id != "default" and not str(raw.get("test_target") or "").strip():
            raw = {**raw, "test_target": default_test_target}
        test_target = _required_string(raw, "test_target", f"presets.{preset_id}")
        raw_bindings = raw.get("bindings")
        if not isinstance(raw_bindings, Mapping) or not raw_bindings:
            raise ModelRouteConfigError(
                f"presets.{preset_id}.bindings must be a non-empty table"
            )
        bindings: dict[tuple[str, str], str] = {}
        for key, group_id in raw_bindings.items():
            task_group_id, sep, difficulty = str(key).partition("/")
            if (
                not sep
                or task_group_id not in task_groups
                or difficulty not in DIFFICULTIES
            ):
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.bindings key {key!r} must be "
                    "'<task_group>/<difficulty>'"
                )
            if not isinstance(group_id, str) or not group_id:
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.bindings[{key!r}] must name a model "
                    "group or a target"
                )
            if group_id not in model_groups:
                if group_id not in targets:
                    raise ModelRouteConfigError(
                        f"presets.{preset_id}.bindings[{key!r}] names neither a "
                        f"model group nor a target: {group_id!r}"
                    )
                # Quick model selection: "just use this one model here" is the
                # common case, and writing a one-member group for it is pure
                # ceremony. Groups resolve first, so a group id never becomes
                # ambiguous by somebody adding a target of the same name.
                group_id = _wrap_single_target(group_id, model_groups)
            bindings[(task_group_id, difficulty)] = group_id
        if preset_id == "default":
            # Difficulty fallback walks up to high, so binding every group's
            # high cell is what guarantees no switch combination is stranded
            # (§5.5 as amended by the 2026-08-11 fallback decision).
            missing_groups = [
                group_id
                for group_id in TASK_GROUP_IDS
                if (group_id, "quality") not in bindings
            ]
            if missing_groups:
                raise ModelRouteConfigError(
                    "Default preset must bind every task group's high cell; "
                    f"missing: {', '.join(missing_groups)}"
                )
        raw_thinking = raw.get("thinking", {})
        if not isinstance(raw_thinking, Mapping):
            raise ModelRouteConfigError(
                f"presets.{preset_id}.thinking must be a table"
            )
        thinking: dict[tuple[str, str], str] = {}
        for key, level in raw_thinking.items():
            task_group_id, sep, difficulty = str(key).partition("/")
            if (
                not sep
                or task_group_id not in task_groups
                or difficulty not in DIFFICULTIES
            ):
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.thinking key {key!r} must be "
                    "'<task_group>/<difficulty>'"
                )
            if level not in ("low", "medium", "high"):
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.thinking[{key!r}] must be "
                    "low/medium/quality"
                )
            thinking[(task_group_id, difficulty)] = level
        raw_agent_session = raw.get("agent_session", {})
        if not isinstance(raw_agent_session, Mapping):
            raise ModelRouteConfigError(
                f"presets.{preset_id}.agent_session must be a table"
            )
        agent_session: dict[tuple[str, str], str] = {}
        for key, mode in raw_agent_session.items():
            task_group_id, sep, difficulty = str(key).partition("/")
            if (
                not sep
                or task_group_id not in task_groups
                or difficulty not in DIFFICULTIES
            ):
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.agent_session key {key!r} must be "
                    "'<task_group>/<difficulty>'"
                )
            if mode not in AGENT_SESSION_MODES:
                raise ModelRouteConfigError(
                    f"presets.{preset_id}.agent_session[{key!r}] must be one of "
                    f"{list(AGENT_SESSION_MODES)}"
                )
            agent_session[(task_group_id, difficulty)] = mode
        presets[preset_id] = Preset(
            preset_id,
            name,
            test_target,
            MappingProxyType(bindings),
            MappingProxyType(thinking),
            MappingProxyType(agent_session),
        )

    policies: dict[str, RoutePolicy] = {}
    for policy_id, raw in _table(data, "policies").items():
        if not isinstance(raw, Mapping):
            raise ModelRouteConfigError(f"policies.{policy_id} must be a table")
        raw_backends = raw.get("allowed_backends")
        if (
            not isinstance(raw_backends, list)
            or not raw_backends
            or any(not isinstance(item, str) for item in raw_backends)
        ):
            raise ModelRouteConfigError(
                f"policies.{policy_id}.allowed_backends must be a non-empty "
                "string array"
            )
        unknown_backends = set(raw_backends) - ALLOWED_BACKENDS
        if unknown_backends:
            raise ModelRouteConfigError(
                f"Policy {policy_id!r} allows unknown backends: "
                f"{sorted(unknown_backends)}"
            )
        if len(set(raw_backends)) != len(raw_backends):
            raise ModelRouteConfigError(
                f"Policy {policy_id!r} contains duplicate allowed_backends"
            )
        # A policy is a gate, not a source: it can forbid a backend a bound
        # group names, never add one. There is deliberately nothing else to
        # parse -- the overlay keys that used to live here put "which models
        # answer" in two places at once. Whether the *active* policy leaves the
        # *active* preset with a usable candidate is checked at startup by
        # `capabilities.validate_profile_capabilities`, which is the only place
        # that knows both.
        policies[policy_id] = RoutePolicy(policy_id, frozenset(raw_backends))
    if not policies:
        raise ModelRouteConfigError(f"{route_path} has no route policies")

    active_preset_id = str(user.get("preset") or "default").strip() or "default"
    if active_preset_id not in presets:
        raise ModelRouteConfigError(
            f"llm.preset = {active_preset_id!r} names no known preset; "
            f"known: {sorted(presets)}"
        )

    # Both presets, not just the active one: a cell the active preset leaves
    # unbound falls back to the default preset's, and a preferred model that
    # stopped applying at exactly those cells would be the harder bug to see.
    config_preferred = _preferred_bindings(
        _table(user, "preferred_targets"), targets, model_groups,
        source="llm.preferred_targets",
    )
    runtime_preferred_map = _preferred_bindings(
        _RUNTIME_PREFERRED, targets, model_groups, source="--llm-model",
    )
    # The test-target reachability check runs on the DECLARED composition,
    # before any preferred pin overlays the bindings: the test endpoint is
    # resolved directly from its target id (`routing/config.py`), never through
    # the bound groups, so an exclusive pin covering every cell must not fail
    # this composition-sanity check.
    declared_group_ids = set(presets[active_preset_id].bindings.values()) | set(
        presets["default"].bindings.values()
    )
    # Checking only the presets this run can actually use: a packaged preset
    # you never selected must not fail your config.
    reachable_targets = {
        target_id
        for group_id in declared_group_ids
        for target_id in model_groups[group_id].target_ids
    }
    for preset_id in sorted({"default", active_preset_id}):
        test_target = presets[preset_id].test_target_id
        if test_target not in reachable_targets:
            # v1's per-chain test_target moved to the preset (§5.5): it is the
            # only guarantee test_profile runs never hit a real chain.
            raise ModelRouteConfigError(
                f"presets.{preset_id}.test_target {test_target!r} is not in any "
                "bound model group"
            )

    if config_preferred or runtime_preferred_map:
        for preset_id in sorted({"default", active_preset_id}):
            presets[preset_id] = _with_preferred_bindings(
                presets[preset_id], runtime_preferred_map, config_preferred, model_groups
            )

    # What the active run can actually reach: the active preset's bindings and
    # the default preset's (per-cell fallback), pins included — these feed the
    # routing identity, so a changed pin invalidates stale resumes by design.
    referenced_group_ids = set(presets[active_preset_id].bindings.values()) | set(
        presets["default"].bindings.values()
    )
    routing_target_ids = {
        target_id
        for group_id in referenced_group_ids
        for target_id in model_groups[group_id].target_ids
    } | {
        presets[preset_id].test_target_id
        for preset_id in ("default", active_preset_id)
    }
    referenced_fact_ids = {
        targets[target_id].fact_id for target_id in routing_target_ids
    }

    # Split digest (model-routing v2): the routing identity covers everything that
    # can change what a call selects, prompts, or budgets; the advisory digest
    # covers evaluation/display-only fields. The criterion is one sentence:
    # changes what this run produces -> routing; changes how we judge it ->
    # advisory. Only the former may enter resume keys. Self-reported facts and
    # user groups/presets that the active run cannot reach are advisory too --
    # adding an unrelated model must not invalidate checkpoints.
    def _snapshot(fact_ids, field_filter) -> bytes:
        return json.dumps(
            [
                {
                    key: value
                    for key, value in asdict(facts[fact_id]).items()
                    if field_filter(key)
                }
                for fact_id in sorted(fact_ids)
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    routing_fact_ids = [
        fact_id
        for fact_id in facts
        if fact_id not in self_reported or fact_id in referenced_fact_ids
    ]
    # The packaged declaration enters as *parsed routing structure*, never as
    # file bytes (fixed 2026-08-12): hashing the text put advisory fields --
    # ``floor_score``, the preset display name, even a comment -- into every
    # resume key, so re-tuning a floor silently discarded every completed
    # window. Same criterion as the fact fields above.
    packaged_identity = json.dumps(
        {
            # Only what a reachable target uses: declaring an unrelated model
            # (which mints the shared "custom-default" profile) must not move
            # the identity.
            "execution_profiles": {
                profile_id: profiles[profile_id].native_search_tool
                for profile_id in sorted(
                    {targets[target_id].execution_profile for target_id in routing_target_ids}
                )
            },
            "targets": {
                target_id: [
                    targets[target_id].backend,
                    targets[target_id].fact_id,
                    targets[target_id].enabled_by,
                    targets[target_id].execution_profile,
                ]
                for target_id in sorted(routing_target_ids)
            },
            "task_cells": {
                f"{group_id}/{difficulty}": task_groups[group_id].cells[difficulty].variant
                for group_id in sorted(task_groups)
                for difficulty in DIFFICULTIES
            },
            "policies": {
                policy_id: {
                    "allowed_backends": sorted(policy.allowed_backends),
                }
                for policy_id, policy in sorted(policies.items())
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    user_identity = json.dumps(
        {
            "active_preset": active_preset_id,
            "presets": {
                preset_id: {
                    "test_target": presets[preset_id].test_target_id,
                    "bindings": {
                        f"{group}/{difficulty}": bound
                        for (group, difficulty), bound in sorted(
                            presets[preset_id].bindings.items()
                        )
                    },
                    "thinking": {
                        f"{group}/{difficulty}": level
                        for (group, difficulty), level in sorted(
                            presets[preset_id].thinking.items()
                        )
                    },
                    # The session tier decides the call form (capsule vs tool
                    # session, one conversation per call vs per window), so a
                    # checkpoint written under one tier must not resume under
                    # another (owner decision 2026-08-22, identity hole 1).
                    "agent_session": {
                        f"{group}/{difficulty}": mode
                        for (group, difficulty), mode in sorted(
                            presets[preset_id].agent_session.items()
                        )
                    },
                }
                for preset_id in sorted({"default", active_preset_id})
            },
            "groups": {
                group_id: {
                    "targets": list(model_groups[group_id].target_ids),
                    "variant_overrides": dict(
                        sorted(model_groups[group_id].variant_overrides.items())
                    ),
                }
                for group_id in sorted(referenced_group_ids)
            },
            "providers": {
                provider_id: [
                    providers[provider_id].kind,
                    providers[provider_id].base_url,
                    providers[provider_id].key_env,
                ]
                for provider_id in sorted(
                    facts[fact_id].provider_tier for fact_id in referenced_fact_ids
                )
                if provider_id in providers
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    routing_identity_digest = hashlib.sha256(
        b"packaged-routes\0"
        + packaged_identity
        + b"\0model-facts\0"
        + _snapshot(routing_fact_ids, lambda key: key not in ADVISORY_FACT_FIELDS)
        + b"\0user-routes\0"
        + user_identity
    ).hexdigest()
    advisory_digest = hashlib.sha256(
        b"model-facts-advisory\0"
        + _snapshot(
            facts, lambda key: key == "fact_id" or key in ADVISORY_FACT_FIELDS
        )
        + b"\0advisory-routes\0"
        + json.dumps(
            {
                # Judgement-only knobs: they phrase warnings, never a decision.
                "floors": {
                    group_id: task_groups[group_id].floor_score
                    for group_id in sorted(task_groups)
                },
                "preset_names": {
                    preset_id: presets[preset_id].name
                    for preset_id in sorted(presets)
                },
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ModelRouteCatalog(
        routing_identity_digest,
        advisory_digest,
        MappingProxyType(dict(facts)),
        MappingProxyType(profiles),
        MappingProxyType(targets),
        MappingProxyType(policies),
        MappingProxyType(model_groups),
        MappingProxyType(task_groups),
        MappingProxyType(presets),
        MappingProxyType(providers),
        frozenset(self_reported),
        active_preset_id,
        tuple(override_notices),
    )


@lru_cache(maxsize=1)
def default_model_routes() -> ModelRouteCatalog:
    from finesub.config import read_config_with_path

    data, _path = read_config_with_path()
    user = data.get("llm", {}) if isinstance(data, Mapping) else {}
    return load_model_routes(user_config=user if isinstance(user, Mapping) else {})


def runtime_fact_for(endpoint: Any) -> ModelCatalogEntry | None:
    """Resolve any dispatched endpoint's runtime fact (merged table first).

    The single fact source for capability filtering and rate limits. A broken
    *route declaration* (a bad ``[llm.model_groups]``, say) falls back to the
    catalog rather than turning every limit lookup into a crash. A broken
    **catalog** deliberately does not: the fallback reads the same file, so its
    parse error surfaces -- facts are what these callers are asking about, and
    guessing at them is worse than stopping. An endpoint neither table knows
    still returns ``None``, which callers read as "no evidence" as before.
    """

    fact_id = str(getattr(endpoint, "fact_id", "") or "")
    provider_tier = str(getattr(endpoint, "provider_tier", "") or "")
    api_model_id = str(getattr(endpoint, "api_model_id", "") or "")
    try:
        routes = default_model_routes()
    except ModelRouteConfigError:
        routes = None
    if routes is not None:
        fact = routes.fact_for_endpoint(provider_tier, api_model_id, fact_id)
        if fact is not None:
            return fact
    if fact_id:
        fact = get_model_catalog_entry_by_fact(fact_id)
        if fact is not None:
            return fact
    return get_model_catalog_entry_for_tier(api_model_id, provider_tier)
