"""Runtime route planning and provider failure classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import httpx

from ..agent import agent_quota
from . import api_keys, execution_policy
from .config import ModelEndpoint, RoleModelConfig
from .model_catalog import (
    ModelCatalogEntry,
    get_model_catalog_entry_by_fact,
    get_model_catalog_entry_for_tier,
)
from .model_routes import (
    CONVERSATIONAL_BACKEND,
    DEFAULT_EXECUTION_POLICY,
    STANDARD_FALLBACK,
    ModelRouteCatalog,
    default_model_routes,
)


class FailureKind(str, Enum):
    UNAVAILABLE = "unavailable"
    QUOTA = "quota"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    CONTRACT_EXHAUSTED = "contract_exhausted"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class RouteCandidate:
    target_id: str
    group_id: str
    # Which failure kinds may walk to the next candidate. Uniform today (the
    # policy prepends that varied it are gone), but the *client* still consults
    # it per candidate: a refusal or a bad request must stop the chain rather
    # than pay for the same answer from every model in the group.
    fallback_on: frozenset[str]
    endpoint: ModelEndpoint
    fact: ModelCatalogEntry | None


@dataclass(frozen=True)
class RoutePlan:
    policy_id: str
    task_group_id: str
    difficulty: str
    model_group_id: str
    native_search: bool
    routing_identity_digest: str
    # Artifact-only companion (model-routing v2): advisory facts never enter resume
    # keys, but the trace still records which advisory snapshot was in force.
    advisory_digest: str
    candidates: tuple[RouteCandidate, ...]

    # The capsule/task label local agents see; the task group *is* the task.
    @property
    def task(self) -> str:
        return self.task_group_id

    def decision_trace(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "task_group_id": self.task_group_id,
            "difficulty": self.difficulty,
            "model_group_id": self.model_group_id,
            "native_search": self.native_search,
            "routing_identity_digest": self.routing_identity_digest,
            "advisory_digest": self.advisory_digest,
            "effective_chain": [
                {
                    "target_id": candidate.target_id,
                    "group_id": candidate.group_id,
                    "fallback_on": sorted(candidate.fallback_on),
                    "backend": candidate.endpoint.backend,
                    "provider_tier": candidate.endpoint.provider_tier,
                    "model": candidate.endpoint.api_model_id,
                    "fact": asdict(candidate.fact) if candidate.fact is not None else None,
                }
                for candidate in self.candidates
            ],
            "candidates": [
                {
                    "target_id": candidate.target_id,
                    "group_id": candidate.group_id,
                    "decision": "pending",
                }
                for candidate in self.candidates
            ],
        }


class ModelRouter:
    """Build immutable execution plans from packaged route declarations."""

    def __init__(
        self,
        routes: ModelRouteCatalog | None = None,
        *,
        policy_id: str = DEFAULT_EXECUTION_POLICY,
    ) -> None:
        self.routes = routes or default_model_routes()
        if policy_id not in self.routes.policies:
            raise ValueError(
                f"Unknown LLM execution policy {policy_id!r}; expected one of "
                f"{sorted(self.routes.policies)}"
            )
        self.policy_id = policy_id

    def plan(
        self,
        config: RoleModelConfig,
        *,
        test_profile: bool = False,
        native_search: bool = False,
    ) -> RoutePlan:
        if config.model_group_id:
            return self._group_plan(
                config, test_profile=test_profile, native_search=native_search
            )
        return self._adapter_plan(config, test_profile=test_profile)

    def _group_plan(
        self, config: RoleModelConfig, *, test_profile: bool, native_search: bool
    ) -> RoutePlan:
        """Expand one cell's bound model group under the active policy.

        The bound group in its declared order, and nothing else. Policies used
        to *prepend* agent groups onto every cell, which meant "which models
        answer" was decided in two places at once; agents now take part by
        being members of a bound group, and the policy is purely a backend
        gate -- it can forbid a backend a group names, never add one it does
        not. Native search stays a per-call capability (D4) filtered in the
        client's candidate loop, so the plan does not branch on it either.
        """

        policy = self.routes.policies[self.policy_id]
        candidates: list[RouteCandidate] = []
        if test_profile:
            # The preset-level test target is the only guarantee test runs
            # never hit a real chain (§5.5).
            member_lists = [(config.test_endpoint.target_id, "test")]
        else:
            member_lists = [
                (target_id, config.model_group_id)
                for target_id in self.routes.model_groups[
                    config.model_group_id
                ].target_ids
            ]
        for target_id, group_id in member_lists:
            target = self.routes.targets[target_id]
            if target.backend not in policy.allowed_backends:
                continue
            fact = self.routes.target_fact(target_id)
            profile = self.routes.target_profile(target_id)
            candidates.append(
                RouteCandidate(
                    target_id=target_id,
                    group_id=group_id,
                    fallback_on=frozenset(STANDARD_FALLBACK),
                    endpoint=ModelEndpoint(
                        provider_tier=fact.provider_tier,
                        api_model_id=fact.api_model_id,
                        target_id=target.id,
                        fact_id=target.fact_id,
                        backend=target.backend,
                        native_search_tool=profile.native_search_tool,
                    ),
                    fact=fact,
                )
            )
        return RoutePlan(
            policy_id=self.policy_id,
            task_group_id=config.task_group_id,
            difficulty=config.difficulty,
            model_group_id=config.model_group_id,
            native_search=native_search,
            routing_identity_digest=self.routes.routing_identity_digest,
            advisory_digest=self.routes.advisory_digest,
            candidates=tuple(candidates),
        )

    def _adapter_plan(
        self, config: RoleModelConfig, *, test_profile: bool
    ) -> RoutePlan:
        endpoints = config.endpoints(test_profile=test_profile)
        policy = self.routes.policies[self.policy_id]
        candidates: list[RouteCandidate] = []
        for index, endpoint in enumerate(endpoints):
            if endpoint.backend not in policy.allowed_backends:
                continue
            fact = (
                get_model_catalog_entry_by_fact(endpoint.fact_id)
                if endpoint.fact_id
                else get_model_catalog_entry_for_tier(
                    endpoint.api_model_id, endpoint.provider_tier
                )
            )
            if fact is not None and (
                endpoint.provider_tier != fact.provider_tier
                or endpoint.api_model_id != fact.api_model_id
            ):
                raise ValueError(
                    f"Custom endpoint fact_id {endpoint.fact_id!r} resolves to "
                    f"{fact.provider_tier}/{fact.api_model_id}, not "
                    f"{endpoint.provider_tier}/{endpoint.api_model_id}."
                )
            if fact is None and not endpoint.unverified:
                raise ValueError(
                    "Custom production endpoint has no verified runtime fact: "
                    f"{endpoint.provider_tier}/{endpoint.api_model_id}. "
                    "Tests must opt in with unverified=True."
                )
            candidates.append(
                RouteCandidate(
                    target_id=endpoint.target_id or f"custom-{index}",
                    group_id="custom",
                    fallback_on=frozenset(
                        {
                            FailureKind.UNAVAILABLE.value,
                            FailureKind.QUOTA.value,
                            FailureKind.RATE_LIMIT.value,
                            FailureKind.TIMEOUT.value,
                            FailureKind.TRANSIENT.value,
                        }
                    ),
                    endpoint=endpoint,
                    fact=fact,
                )
            )
        return RoutePlan(
            policy_id=self.policy_id,
            task_group_id=config.task_group_id or "custom",
            difficulty=config.difficulty,
            model_group_id="custom",
            native_search=False,
            routing_identity_digest="custom",
            advisory_digest="custom",
            candidates=tuple(candidates),
        )


def classify_failure(exc: BaseException) -> FailureKind:
    """Map a dispatched execution failure to the route policy vocabulary."""

    # Deferred on purpose, and the one direction this package may not
    # take at import time: `finesub.llm.client` is built on the router, so
    # naming it up top would close the loop.
    from ..client import (
        QuotaKind,
        classify_quota_error,
        is_retryable_provider_error,
        provider_status_code,
    )

    if isinstance(exc, api_keys.ProviderUnavailableError):
        return FailureKind.UNAVAILABLE
    declared_kind = str(getattr(exc, "route_failure_kind", "") or "")
    if declared_kind in {item.value for item in FailureKind}:
        return FailureKind(declared_kind)
    status = provider_status_code(exc)
    text_lower = f"{type(exc).__name__}: {exc}".lower()
    if status is not None:
        if status == 429:
            quota_kind = classify_quota_error(exc)
            # OpenAI reports an exhausted account as a 429 with
            # ``insufficient_quota`` (not a rate limit); retrying is pointless
            # so the group must advance (provider survey, 2026-08-11).
            if quota_kind is QuotaKind.DAILY or "insufficient_quota" in text_lower:
                return FailureKind.QUOTA
            return FailureKind.RATE_LIMIT
        if status == 402:
            # DeepSeek: insufficient balance. Dead account, not a bad request.
            return FailureKind.QUOTA
        if status < 500:
            return FailureKind.PERMANENT
        text = f"{type(exc).__name__}: {exc}".lower()
        return (
            FailureKind.TIMEOUT
            if isinstance(exc, (TimeoutError, httpx.TimeoutException))
            or any(marker in text for marker in ("timeout", "timed out"))
            else FailureKind.TRANSIENT
        )
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or any(
        marker in text for marker in ("timeout", "timed out")
    ):
        return FailureKind.TIMEOUT
    quota_kind = classify_quota_error(exc)
    if quota_kind is QuotaKind.DAILY:
        return FailureKind.QUOTA
    if quota_kind is not QuotaKind.NONE:
        return FailureKind.RATE_LIMIT
    return FailureKind.TRANSIENT if is_retryable_provider_error(exc) else FailureKind.PERMANENT


@lru_cache(maxsize=16)
def _default_local_agent_available(
    model: str, provider_tier: str = "", native_search: bool = False
) -> bool:
    """Whether the installed agent CLI can serve this model.

    Probed once per model and cached: the probe shells out three times and the
    answer cannot change inside a run. Returning ``True`` unconditionally made
    startup capability validation pass on a machine whose Codex CLI is missing
    or too old for the isolation flags this implementation requires -- the run
    then failed at the first real call, i.e. after the clip cutting and
    uploads that "fail fast before auxiliary work" exists to avoid.

    A probe that itself explodes counts as unavailable rather than
    propagating: this is the *pre-filter*, and dispatch reports the real
    reason.
    """

    from ..agent.local_agent import driver_readiness

    try:
        settings = execution_policy.load_execution_settings()
        driver = execution_policy.driver_for_provider_tier(
            settings, provider_tier=provider_tier, model=model
        )
    except Exception:  # pragma: no cover - defensive, see docstring
        return False
    ready, _detail = driver_readiness(driver, native_search=native_search)
    return ready


def candidate_quota_pool(candidate: RouteCandidate) -> str:
    """Which subscription allowance this candidate draws on.

    Falls back to the provider tier when the candidate carries no verified
    fact -- an unverified custom endpoint. The tier is the pool's default
    anyway, so the fallback books the freeze exactly where the catalog would
    have without a `quota_pool` column.
    """

    fact = candidate.fact
    return (
        fact.effective_quota_pool
        if fact is not None
        else candidate.endpoint.provider_tier
    )


def provider_enabled(
    candidate: RouteCandidate,
    *,
    agent_ready: Any | None = None,
    native_search: bool = False,
) -> bool:
    """Pre-filter: can this candidate's provider serve a call at all?

    ``agent_ready`` lets a client answer for its *own* driver (an injected or
    factory-built one), instead of probing whatever CLI happens to be on PATH.
    Without it the check falls back to the installed CLI.
    """

    if candidate.endpoint.backend == CONVERSATIONAL_BACKEND:
        # Nothing to probe and nothing to launch: this worker is a person's
        # own agent, which claims work when it is running rather than being
        # dialled. Whether one is attached is not knowable here and changes
        # during a run, so the calling path always says no -- the task queue,
        # not the route chain, is where it takes part.
        return False
    if candidate.endpoint.backend == "local_agent":
        model = candidate.endpoint.api_model_id
        tier = candidate.endpoint.provider_tier
        # An exhausted allowance disqualifies every target that draws on it,
        # which is the whole reason exhaustion is booked per pool: otherwise
        # the chain walks from one spent model straight into its neighbour on
        # the same plan, paying a full CLI launch to be told the same thing.
        # The pool is usually the tier, but Antigravity meters its Gemini and
        # Opus models separately -- freezing them together would take a working
        # model out of service because its neighbour ran dry.
        if agent_quota.default_ledger().is_frozen(candidate_quota_pool(candidate)):
            return False
        # The provider tier picks the driver, so readiness has to be asked of
        # the CLI this candidate would actually use -- otherwise one vendor's
        # missing CLI would disqualify the other's targets, or advertise them.
        if agent_ready is not None:
            return bool(agent_ready(model, tier, native_search))
        return _default_local_agent_available(model, tier, native_search)
    return api_keys.provider_tier_enabled(candidate.endpoint.provider_tier)
