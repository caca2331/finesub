"""Validated runtime selection for API and local-agent execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from finesub import settings

from ..agent.local_agent import (
    AgyDriverConfig,
    AgyLocalAgentDriver,
    CAPSULE_SCHEMA_VERSION,
    ClaudeCodeDriverConfig,
    ClaudeCodeLocalAgentDriver,
    CodexDriverConfig,
    CodexLocalAgentDriver,
    DshDriverConfig,
    DshLocalAgentDriver,
    CodeBuddyDriverConfig,
    CodeBuddyLocalAgentDriver,
    local_agent_execution_profiles,
)
from .model_routes import default_model_routes

if TYPE_CHECKING:
    from .model_routes import ModelRouteCatalog


# Provider tiers whose facts run through a local agent CLI. The tier picks
# the driver, so a new agent vendor is a tier plus a driver, not a branch in
# every caller.
LOCAL_CODEX_TIER = "LOCAL_CODEX"
LOCAL_CLAUDE_TIER = "LOCAL_CLAUDE"
LOCAL_AGY_TIER = "LOCAL_AGY"
LOCAL_DSH_TIER = "LOCAL_DSH"
LOCAL_CODEBUDDY_TIER = "LOCAL_CODEBUDDY"


@dataclass(frozen=True)
class ExecutionSettings:
    # Every default is the declaration's (`finesub.settings`), not a copy.
    # Mixed by default (owner decision 2026-08-15): whatever a bound group
    # names may answer. A preset with no agent members behaves as before.
    policy_id: str = settings.EXECUTION_POLICY.default
    local_agent_timeout_seconds: int = settings.LOCAL_AGENT_TIMEOUT_SECONDS.default
    local_agent_allow_unisolated_user_config: bool = (
        settings.LOCAL_AGENT_ALLOW_UNISOLATED_USER_CONFIG.default
    )
    local_agent_service_tier: str = settings.LOCAL_AGENT_SERVICE_TIER.default
    # Debugging only, and outside execution identity: it decides what is
    # left on disk, not what a call does.
    local_agent_keep_episodes: bool = settings.LOCAL_AGENT_KEEP_EPISODES.default
    # Also outside execution identity: it decides when a CLI is ended after
    # its work is accepted, not what the work is.
    local_agent_exit_on_accept: str = settings.LOCAL_AGENT_EXIT_ON_ACCEPT.default
    # Empty means: use the active cell's abstract thinking level after the
    # selected model fact maps it. A non-empty value is an explicit global
    # compatibility override for every local Codex call.
    local_agent_reasoning_effort: str = settings.LOCAL_AGENT_REASONING_EFFORT.default
    # The PHYSICAL ceiling: how many agent CLI processes this machine and
    # subscription run at once (task-parallelism plan W4). One number for
    # every driver -- it budgets the machine, not a vendor -- and the shared
    # slot pool (§1.1) makes it a process-wide fact, not a per-client one.
    # Distinct from `--llm-parallel-windows` (one task's willingness) and the
    # batch's `max_parallel_tasks` (admission).
    local_agent_max_parallel: int = settings.LOCAL_AGENT_MAX_PARALLEL.default
    # No transport switch here on purpose (owner decision 2026-08-22): the
    # transport derives from the cell's `agent_session_mode` and the driver's
    # probe (`agent_transports.agent_transport_for`); `FINESUB_AGENT_TRANSPORT`
    # is the dev-only override.

    def exit_on_accept_for(self, *, rebuilds_usage: bool) -> bool:
        """`auto` ends the CLI on accept only where the driver can rebuild the
        call's usage from per-request events; `on` / `off` force it."""

        mode = self.local_agent_exit_on_accept
        return mode == "on" or (mode == "auto" and rebuilds_usage)

    def codex_driver_config(self, *, model: str) -> CodexDriverConfig:
        overrides: list[str] = []
        if self.local_agent_service_tier:
            overrides.append(f'service_tier="{self.local_agent_service_tier}"')
        if self.local_agent_reasoning_effort:
            overrides.append(
                f'model_reasoning_effort="{self.local_agent_reasoning_effort}"'
            )
        return CodexDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            allow_unisolated_user_config=self.local_agent_allow_unisolated_user_config,
            config_overrides=tuple(overrides),
            max_parallel=self.local_agent_max_parallel,
            keep_episodes=self.local_agent_keep_episodes,
            exit_on_accept=self.exit_on_accept_for(rebuilds_usage=False),
        )

    def claude_code_driver_config(self, *, model: str) -> ClaudeCodeDriverConfig:
        """Claude Code has no config-override channel; effort is a flag.

        `local_agent_reasoning_effort` keeps the same meaning it has for
        Codex -- empty means "use the cell's level after the model fact maps
        it", non-empty is a deliberate global override.
        """

        return ClaudeCodeDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            allow_unisolated_user_config=self.local_agent_allow_unisolated_user_config,
            effort=self.local_agent_reasoning_effort,
            max_parallel=self.local_agent_max_parallel,
            keep_episodes=self.local_agent_keep_episodes,
            exit_on_accept=self.exit_on_accept_for(rebuilds_usage=True),
        )

    def codebuddy_driver_config(self, *, model: str) -> CodeBuddyDriverConfig:
        """CodeBuddy's config; effort is a flag, as it is for Claude Code.

        `local_agent_allow_unisolated_user_config` is not passed on: this CLI
        has no isolated mode to opt out of, so the switch has nothing to turn
        off here and `required_capabilities` never lists the two bits it
        relaxes.
        """

        return CodeBuddyDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            effort=self.local_agent_reasoning_effort,
            max_parallel=self.local_agent_max_parallel,
            output_ceiling_hint=_codebuddy_ceiling_hint(model),
            fallback_model=_codebuddy_fallback_model(model),
            keep_episodes=self.local_agent_keep_episodes,
            exit_on_accept=self.exit_on_accept_for(rebuilds_usage=True),
        )

    def agy_driver_config(self, *, model: str) -> AgyDriverConfig:
        return AgyDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            effort=self.local_agent_reasoning_effort,
            max_parallel=self.local_agent_max_parallel,
            keep_episodes=self.local_agent_keep_episodes,
            exit_on_accept=self.exit_on_accept_for(rebuilds_usage=True),
        )

    def dsh_driver_config(self, *, model: str) -> DshDriverConfig:
        """dsh's config.

        `local_agent_reasoning_effort` keeps the meaning it has everywhere
        else -- empty means "use the cell's level after the model fact maps
        it". dsh has no flag for it: the driver puts it on whichever model
        plugin owns the route, through the same per-call patch overlay that
        carries the MCP server.
        """

        return DshDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            allow_unisolated_user_config=self.local_agent_allow_unisolated_user_config,
            effort=self.local_agent_reasoning_effort,
            max_parallel=self.local_agent_max_parallel,
            keep_episodes=self.local_agent_keep_episodes,
            exit_on_accept=self.exit_on_accept_for(rebuilds_usage=False),
        )

    def driver_config_for(self, *, provider_tier: str, model: str):
        """The driver config a local-agent fact's provider tier calls for."""

        tier = normalized_tier(provider_tier)
        if tier == LOCAL_CLAUDE_TIER:
            return self.claude_code_driver_config(model=model)
        if tier == LOCAL_CODEBUDDY_TIER:
            return self.codebuddy_driver_config(model=model)
        if tier == LOCAL_AGY_TIER:
            return self.agy_driver_config(model=model)
        if tier == LOCAL_DSH_TIER:
            return self.dsh_driver_config(model=model)
        return self.codex_driver_config(model=model)


#: Which allowance a CodeBuddy paid twin draws on.
#:
#: The twin (`hy3-x`) has no catalog row of its own, and
#: must not get one: every local-agent model auto-builds a routable
#: target (`model_routes._wire_unclaimed_catalog_target`), which would put a
#: model that spends credits into the routing tables. So the single thing the
#: harness needs to know about it -- which allowance a call that reached one
#: actually spent -- is declared here, beside the function that makes reaching
#: one possible at all.
CODEBUDDY_FALLBACK_QUOTA_POOL = "CODEBUDDY_CREDITS"


def fallback_quota_pool_for_tier(provider_tier: str) -> str:
    """The allowance a paid twin on this tier draws on, or "" for none.

    Empty is the answer for every tier but CodeBuddy, and it means "this tier
    has no twin to attribute anything to" -- so a caller that cannot recognise
    an answering model keeps whatever pool it already had.
    """

    if normalized_tier(provider_tier) == LOCAL_CODEBUDDY_TIER:
        return CODEBUDDY_FALLBACK_QUOTA_POOL
    return ""


def _codebuddy_fallback_model(model: str) -> str:
    """The paid twin this row falls back to, or "" for none.

    ⚠ Reaching a non-empty value here is what makes a call able to spend
    credits, so it is read from the catalog row and from nowhere else.
    """

    from .model_catalog import get_model_catalog_entry_for_tier

    entry = get_model_catalog_entry_for_tier(model, LOCAL_CODEBUDDY_TIER)
    if entry is None:
        return ""
    return str(entry.fallback_model or "")


def _codebuddy_ceiling_hint(model: str) -> int:
    """This model's real per-turn output ceiling, or 0 to say nothing.

    One row answers both halves: `hint_output_ceiling` says *whether* the
    worker is told, `max_output_tokens` says *what*. Never written down twice,
    because a number that does not match reality is measurably worse than
    silence (`codebuddy_output_ceiling_clause` records the measurement). A
    model with no row falls back to silence rather than to a guess.
    """

    from .model_catalog import get_model_catalog_entry_for_tier

    entry = get_model_catalog_entry_for_tier(model, LOCAL_CODEBUDDY_TIER)
    if entry is None or not entry.hint_output_ceiling:
        return 0
    return int(entry.max_output_tokens)


def normalized_tier(provider_tier: str) -> str:
    """One spelling of a provider tier, for keys and lookups."""

    return (provider_tier or "").strip().upper()


def driver_for_provider_tier(
    settings: ExecutionSettings, *, provider_tier: str, model: str
):
    """Build the local-agent driver a provider tier names.

    One place decides tier -> driver, so the readiness pre-filter, dispatch
    and the tests all reach the same CLI for the same candidate.

    Unknown tiers fail closed. Defaulting to Codex would mean a typo in a
    catalog row silently runs the task on the wrong vendor's CLI -- and with a
    third agent tier arriving, "wrong vendor" stops being hypothetical. A
    refusal is cheap; a task answered by an unintended backend is not.
    """

    drivers = {
        LOCAL_CODEX_TIER: CodexLocalAgentDriver,
        LOCAL_CLAUDE_TIER: ClaudeCodeLocalAgentDriver,
        LOCAL_AGY_TIER: AgyLocalAgentDriver,
        LOCAL_DSH_TIER: DshLocalAgentDriver,
        LOCAL_CODEBUDDY_TIER: CodeBuddyLocalAgentDriver,
    }
    tier = normalized_tier(provider_tier)
    if tier not in drivers:
        raise ValueError(
            f"No local-agent driver is registered for provider tier "
            f"{provider_tier!r}; known tiers are {sorted(drivers)}"
        )
    return drivers[tier](
        settings.driver_config_for(provider_tier=tier, model=model)
    )


def default_agent_slot_budgets(settings: ExecutionSettings | None = None) -> tuple:
    """The in-flight budgets a task with agent demand books against.

    Task-parallelism plan W4: a task whose routing can reach a local agent
    reserves one mandatory-lane slot ON EVERY vendor pool the chain can reach
    -- which pool a call actually lands on is decided per attempt, so a
    reservation on only the first pool covers nothing when the route falls
    through to a second vendor (reviewer 2026-08-30 P1-1). "Can reach" is
    decided from the catalog (every policy-allowed ``local_agent`` target in
    any model group), which over-approximates on purpose: an unused
    reservation only makes fan-out conservative, while a missing one would
    let optional claims starve a task's mandatory lane (invariant I1).
    Pure-API setups (no agent group, or a policy that forbids the backend)
    get () and stay entirely outside the budgets. Pools are per vendor
    (`_shared_in_flight_pool` keys on the driver id), so the tuple is one
    budget per distinct vendor, typically one."""

    effective = settings or load_execution_settings()
    routes = default_model_routes()
    policy = routes.policies[effective.policy_id]
    if "local_agent" not in policy.allowed_backends:
        return ()
    budgets: list = []
    for group_id in sorted(routes.model_groups):
        for target_id in routes.model_groups[group_id].target_ids:
            target = routes.targets[target_id]
            if target.backend != "local_agent":
                continue
            fact = routes.target_fact(target_id)
            try:
                driver = driver_for_provider_tier(
                    effective,
                    provider_tier=fact.provider_tier,
                    model=fact.api_model_id,
                )
            except ValueError:
                continue
            if not any(budget is driver._in_flight for budget in budgets):
                budgets.append(driver._in_flight)
    return tuple(budgets)


def load_execution_settings() -> ExecutionSettings:
    """`[llm]`'s execution keys, from one read of config.toml.

    Types, bounds and defaults are the declarations' (`finesub.settings`);
    what is left here is what only this domain knows: the policy has to exist
    in the route table, and the two agent overrides are case-insensitive words
    from a closed list.

    `local_agent_timeout_seconds` is the one number that says how long a local
    agent may spend on one call; the lease TTL is derived from it
    (`lease_ttl_for`), so raising it raises how long we hold the task too and
    nothing has to be kept in sync by hand. It is *not* how long a
    conversational run waits for somebody to join: that is a fixed hang guard,
    and the answer to "I will be back in three hours" is the two-step run in
    docs/manual/agent.md, not a bigger number here.
    """

    config = settings.read()
    location = f" in {config.path}" if config.path else ""
    policy_id = settings.EXECUTION_POLICY.resolve(config=config).strip()
    routes = default_model_routes()
    if policy_id not in routes.policies:
        raise ValueError(
            f"llm.execution_policy must be one of {sorted(routes.policies)}{location}"
        )
    service_tier = settings.LOCAL_AGENT_SERVICE_TIER.resolve(config=config).strip().lower()
    if service_tier not in {"", "fast", "flex"}:
        raise ValueError(
            f"llm.local_agent_service_tier must be '', 'fast', or 'flex'{location}"
        )
    exit_on_accept = settings.LOCAL_AGENT_EXIT_ON_ACCEPT.resolve(config=config).strip().lower()
    if exit_on_accept not in {"auto", "on", "off"}:
        raise ValueError(f"llm.local_agent_exit_on_accept must be 'auto', 'on', or 'off'{location}")
    reasoning = settings.LOCAL_AGENT_REASONING_EFFORT.resolve(config=config).strip().lower()
    if reasoning not in {"", "low", "medium", "high", "xhigh"}:
        raise ValueError(
            "llm.local_agent_reasoning_effort must be empty or "
            "low/medium/high/xhigh"
            f"{location}"
        )
    return ExecutionSettings(
        policy_id=policy_id,
        local_agent_timeout_seconds=settings.LOCAL_AGENT_TIMEOUT_SECONDS.resolve(config=config),
        local_agent_allow_unisolated_user_config=(
            settings.LOCAL_AGENT_ALLOW_UNISOLATED_USER_CONFIG.resolve(config=config)
        ),
        local_agent_service_tier=service_tier,
        local_agent_reasoning_effort=reasoning,
        local_agent_max_parallel=settings.LOCAL_AGENT_MAX_PARALLEL.resolve(config=config),
        local_agent_keep_episodes=settings.LOCAL_AGENT_KEEP_EPISODES.resolve(config=config),
        local_agent_exit_on_accept=exit_on_accept,
    )


def execution_identity(
    settings: ExecutionSettings | None = None,
    *,
    routes: ModelRouteCatalog | None = None,
) -> dict[str, Any]:
    effective = settings or load_execution_settings()
    effective_routes = routes or default_model_routes()
    # Resume keys embed this dict verbatim, so only the routing identity may
    # appear here; the advisory digest lives in artifacts (route decision
    # trace), never in checkpoint identity (model-routing v2).
    return {
        "policy_id": effective.policy_id,
        "routing_identity_digest": effective_routes.routing_identity_digest,
        "capsule_schema": CAPSULE_SCHEMA_VERSION,
        "local_agent_reasoning_effort": effective.local_agent_reasoning_effort,
        "local_agent_service_tier": effective.local_agent_service_tier,
        "local_agent_allow_unisolated_user_config": (
            effective.local_agent_allow_unisolated_user_config
        ),
        "local_agent_timeout_seconds": effective.local_agent_timeout_seconds,
        # The transport a call actually takes (capsule vs tool session) is
        # deliberately absent (owner decision 2026-08-22): it follows the
        # session tier -- which is routing identity -- and the machine's
        # probe, which is not a contract. Both transports consume the same
        # prompt under the same validator; which one served a window is in
        # the route decision trace, advisory only.
        "local_agent_drivers": local_agent_execution_profiles(),
    }
