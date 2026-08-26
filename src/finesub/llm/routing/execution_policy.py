"""Validated runtime selection for API and local-agent execution backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from finesub.config import read_config_with_path

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
    local_agent_execution_profiles,
)
from .model_routes import DEFAULT_EXECUTION_POLICY, default_model_routes

if TYPE_CHECKING:
    from .model_routes import ModelRouteCatalog


# Provider tiers whose facts run through a local agent CLI. The tier picks
# the driver, so a new agent vendor is a tier plus a driver, not a branch in
# every caller.
LOCAL_CODEX_TIER = "LOCAL_CODEX"
LOCAL_CLAUDE_TIER = "LOCAL_CLAUDE"
LOCAL_AGY_TIER = "LOCAL_AGY"
LOCAL_DSH_TIER = "LOCAL_DSH"


@dataclass(frozen=True)
class ExecutionSettings:
    # Mixed by default (owner decision 2026-08-15): whatever a bound group
    # names may answer. A preset with no agent members behaves as before.
    policy_id: str = DEFAULT_EXECUTION_POLICY
    local_agent_timeout_seconds: int = 1680
    local_agent_allow_unisolated_user_config: bool = False
    local_agent_service_tier: str = ""
    # Empty means: use the active cell's abstract thinking level after the
    # selected model fact maps it. A non-empty value is an explicit global
    # compatibility override for every local Codex call.
    local_agent_reasoning_effort: str = ""
    # No transport switch here on purpose (owner decision 2026-08-22): the
    # transport derives from the cell's `agent_session_mode` and the driver's
    # probe (`agent_transports.agent_transport_for`); `FINESUB_AGENT_TRANSPORT`
    # is the dev-only override.

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
        )

    def agy_driver_config(self, *, model: str) -> AgyDriverConfig:
        return AgyDriverConfig(
            model=model,
            timeout_seconds=self.local_agent_timeout_seconds,
            effort=self.local_agent_reasoning_effort,
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
        )

    def driver_config_for(self, *, provider_tier: str, model: str):
        """The driver config a local-agent fact's provider tier calls for."""

        tier = normalized_tier(provider_tier)
        if tier == LOCAL_CLAUDE_TIER:
            return self.claude_code_driver_config(model=model)
        if tier == LOCAL_AGY_TIER:
            return self.agy_driver_config(model=model)
        if tier == LOCAL_DSH_TIER:
            return self.dsh_driver_config(model=model)
        return self.codex_driver_config(model=model)


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


def _llm_table() -> tuple[Mapping[str, Any], str]:
    data, path = read_config_with_path()
    raw = data.get("llm", {})
    location = f" in {path}" if path else ""
    if not isinstance(raw, Mapping):
        raise ValueError(f"[llm] must be a TOML table{location}")
    return raw, location


def _string(
    table: Mapping[str, Any], key: str, default: str, location: str
) -> str:
    value = table.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"llm.{key} must be a string{location}")
    return value.strip()


def _bool(
    table: Mapping[str, Any], key: str, default: bool, location: str
) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"llm.{key} must be true/false{location}")
    return value


def _int(
    table: Mapping[str, Any], key: str, default: int, location: str
) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"llm.{key} must be an integer{location}")
    return value


def load_execution_settings() -> ExecutionSettings:
    table, location = _llm_table()
    policy_id = _string(table, "execution_policy", DEFAULT_EXECUTION_POLICY, location)
    routes = default_model_routes()
    if policy_id not in routes.policies:
        raise ValueError(
            f"llm.execution_policy must be one of {sorted(routes.policies)}{location}"
        )
    # 28 minutes: the one number that says how long a local agent may spend on
    # one call. The lease TTL is derived from it (`lease_ttl_for`), so raising
    # it raises how long we hold the task too and nothing has to be kept in
    # sync by hand. No ceiling -- how long is too long for a call is exactly
    # the thing only the person running it can know. It is *not* how long a
    # conversational run waits for somebody to join: that is a fixed hang
    # guard, and the answer to "I will be back in three hours" is the two-step
    # run in docs/manual/agent.md, not a bigger number here.
    timeout = _int(table, "local_agent_timeout_seconds", 1680, location)
    if timeout < 10:
        raise ValueError(
            f"llm.local_agent_timeout_seconds must be at least 10{location}"
        )
    service_tier = _string(table, "local_agent_service_tier", "", location).lower()
    if service_tier not in {"", "fast", "flex"}:
        raise ValueError(
            f"llm.local_agent_service_tier must be '', 'fast', or 'flex'{location}"
        )
    reasoning = _string(
        table, "local_agent_reasoning_effort", "", location
    ).lower()
    if reasoning not in {"", "low", "medium", "high", "xhigh"}:
        raise ValueError(
            "llm.local_agent_reasoning_effort must be empty or "
            "low/medium/high/xhigh"
            f"{location}"
        )
    return ExecutionSettings(
        policy_id=policy_id,
        local_agent_timeout_seconds=timeout,
        local_agent_allow_unisolated_user_config=_bool(
            table, "local_agent_allow_unisolated_user_config", False, location
        ),
        local_agent_service_tier=service_tier,
        local_agent_reasoning_effort=reasoning,
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
