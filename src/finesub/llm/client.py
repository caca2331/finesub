"""Runtime client helpers for role-based LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import mimetypes
from pathlib import Path
import re
import sys
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
)

import httpx

from .routing.config import (
    ROLE_DEFAULT_TASK_GROUP,
    CapabilityTier,
    LLMRole,
    RoleModelConfig,
    default_role_configs,
    role_config_for,
)
from .routing.capabilities import endpoint_supports
from .routing.model_catalog import thinking_value_for
from .routing.profiles import VIDEO_SAMPLE_FPS
from .rate_limit import ModelRateLimiter, estimate_call_input_tokens
from .agent import agent_quota
from .routing import api_keys

if TYPE_CHECKING:
    from .routing.execution_policy import ExecutionSettings
    from .agent.local_agent import LocalAgentDriver
    from .routing.model_router import ModelRouter

VALIDATION_BASE_TEMPERATURE = 1.0
VALIDATION_TEMPERATURE_STEP = 0.01
_VALIDATION_SEED_BASE = 1_730_001


@dataclass(frozen=True)
class UploadedFileRef:
    file_id: str
    filename: str
    mime_type: str
    local_path: str = ""
    # The local clip already owns agy's required 0.25 fps sampling, so the
    # driver may copy it into the capsule instead of encoding it again.
    agy_prepared: bool = False
    # Exact media duration when the caller cut this clip and therefore knows it.
    # Window clips are raw ADTS, which carries no container duration: ffprobe
    # has to guess a bitrate from the opening frames, and a window that starts
    # on quiet audio has been measured at 22x its real length. That guess feeds
    # the input-token estimate, so it can strike every candidate off the chain
    # as ``input_limit``. 0.0 means "unknown, fall back to probing".
    duration_seconds: float = 0.0

    @property
    def is_audio(self) -> bool:
        return self.mime_type.strip().lower().startswith("audio/")

    @property
    def is_video(self) -> bool:
        return self.mime_type.strip().lower().startswith("video/")


def local_media_file_ref(
    path: str | Path, *, agy_prepared_video: bool = False
) -> UploadedFileRef:
    """A media reference that needs no Gemini Files API upload.

    ``agent-only`` uses this form so clipping does not accidentally require a
    Gemini API key before the local agy candidate is even considered.
    """

    local = Path(path).expanduser().resolve()
    mime_type = mimetypes.guess_type(local.name)[0] or "application/octet-stream"
    return UploadedFileRef(
        file_id="",
        filename=local.name,
        mime_type=mime_type,
        local_path=str(local),
        agy_prepared=bool(agy_prepared_video and mime_type.startswith("video/")),
    )


def with_media_duration(ref: UploadedFileRef, seconds: float) -> UploadedFileRef:
    """Record the exact duration of a clip the caller cut itself."""

    seconds = float(seconds or 0.0)
    if seconds <= 0 or ref.duration_seconds == seconds:
        return ref
    return replace(ref, duration_seconds=seconds)


def _record_vendor_error(attempts: List[Dict[str, Any]]) -> None:
    """Copy the CLI's own error prose into the failed attempt row.

    The row already points at the kept capsule through ``evidence_locator``,
    and that is where the vendor text lives. But capsules sit under a machine
    temp root that ``finesub agent-clean`` -- or the OS -- may reclaim, and
    then the artifact holds a pointer to nothing and a reader is left with
    ``outcome: failed`` plus a return code. Reading the text once, at failure
    time, makes the artifact answer "what did the vendor actually say" on its
    own. Silence here is normal: a transport that failed before writing events
    has nothing to quote.
    """

    from .agent.agent_paths import vendor_error_from_attempts

    for attempt in attempts:
        if attempt.get("vendor_error"):
            continue
        text = vendor_error_from_attempts([attempt])
        if text:
            attempt["vendor_error"] = text


def mark_agy_prepared_video(ref: UploadedFileRef) -> UploadedFileRef:
    if not ref.is_video or ref.agy_prepared:
        return ref
    return replace(ref, agy_prepared=True)


# Policies that let a local agent answer at all. Necessary but no longer
# sufficient: a policy only subtracts backends, so the bound groups have to
# name an agent too -- see ``binds_local_agent``.
AGENT_CAPABLE_POLICY_IDS = frozenset({"agent-only", "agent-text-preferred"})


def window_media_ref(
    path: str | Path,
    *,
    execution_settings: Any | None = None,
    routes: Any | None = None,
) -> UploadedFileRef:
    """The reference one window clip should be carried by under this policy.

    Correction and fast mode need exactly this decision, and writing it out in
    both places meant the next policy id -- or the next change to the
    agy-prepared rule -- had two sites to get right, with "silently uploads to
    Gemini under an agent policy" as the failure mode.

    ``routes`` must be the same catalog the client's router will plan from --
    pass ``client.router.routes``. Falling back to the global one would let the
    table that decides *how the clip is carried* disagree with the table that
    decides *who answers*, and then a clip is either uploaded for an agent that
    never needed it or kept local for a chain that only has API candidates.
    """

    from .routing.model_routes import DEFAULT_EXECUTION_POLICY, default_model_routes

    local = Path(path)
    policy_id = str(
        getattr(execution_settings, "policy_id", "") or DEFAULT_EXECUTION_POLICY
    )
    agent_reachable = (
        policy_id in AGENT_CAPABLE_POLICY_IDS
        and (routes or default_model_routes()).binds_local_agent()
    )
    ref = (
        local_media_file_ref(local)
        if agent_reachable
        else upload_gemini_file(local)
    )
    if ref.is_video:
        # Clips are cut at agy's required sampling rate whoever answers, so
        # this is a fact about the file rather than a routing decision.
        ref = mark_agy_prepared_video(ref)
    return ref


@dataclass(frozen=True)
class LLMCallResult:
    content: str
    role: LLMRole
    model: str
    fallback_used: bool
    raw_response: Mapping[str, Any]
    # Prompt tier of the served variant; kept for records. ``variant`` is the
    # prompt-variant name the answering candidate really received -- callers
    # re-assemble the exact sent messages from it when the prompt was passed
    # as a factory.
    capability_tier: CapabilityTier = CapabilityTier.CAPABLE
    variant: str = ""
    api_key_label: str = ""
    thinking_level: str = ""
    thinking_budget: int = 0
    api_attempts: List[Mapping[str, Any]] = field(default_factory=list)
    execution_attempts: List[Mapping[str, Any]] = field(default_factory=list)
    target_id: str = ""
    backend: str = "gemini_rest"
    route_decision: Mapping[str, Any] = field(default_factory=dict)


class LLMIPRiskError(RuntimeError):
    """Provider response suggests the client IP/proxy is risk-blocked."""


def provider_status_code(exc: BaseException) -> Optional[int]:
    """The HTTP status the provider actually returned, when it is knowable.

    Structured first: `GeminiAPIError` carries `status_code`, and httpx errors
    carry `response.status_code`. Only if neither exists do we look at the text,
    and then only at a *word-bounded* 4xx/5xx -- see the warning below.
    """

    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status:
        return status
    match = re.search(r"\b([45]\d\d)\b", f"{type(exc).__name__}: {exc}")
    return int(match.group(1)) if match else None


def is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    """Whether this is the provider saying "not now" rather than "not ever".

    Status first. The substring pass below stays as a fallback for transports
    that surface no status at all, but it must never be the primary signal:
    these markers were matched *unbounded* against the whole stringified
    exception, which for `GeminiAPIError` embeds the full JSON error body. A
    deterministic 400 reading "The input token count (215000) exceeds the
    maximum" was therefore classified as retryable, because "215000" contains
    "500" -- and `4290` in any id or count made it look like a 429.
    """

    status = provider_status_code(exc)
    if status is not None:
        return status == 429
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "quota",
        "rate limit",
        "ratelimit",
        "resource_exhausted",
        "too many requests",
    ]
    return any(marker in text for marker in markers)


def is_daily_quota_error(exc: BaseException) -> bool:
    """True only for provider messages that identify a per-day quota."""

    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "daily quota",
        "daily limit",
        "per day",
        "requests/day",
        "requests per day",
        "request per day",
        "generate_requests_per_day",
        "generaterequestsperday",
        "rpd limit",
        "rpd quota",
    ]
    return any(marker in text for marker in markers)


class QuotaKind(str, Enum):
    """Classification of a provider quota / rate-limit error."""

    NONE = "none"
    OTHER_RATE = "other_rate"
    PER_MINUTE = "per_minute"
    DAILY = "daily"


_QUOTA_ID_RE = re.compile(r"quotaid[\"'\s:]+[\"']?([A-Za-z0-9_-]+)", re.IGNORECASE)


def classify_quota_error(exc: BaseException) -> QuotaKind:
    """Classify a provider error from its structured ``quotaId``.

    Gemini 429s embed ``"quotaId": "...PerDay..."`` / ``"...PerMinute..."`` in
    the error JSON. We classify off that quotaId, NOT the ``retryDelay`` hint —
    Google returns a small generic backoff (~20-60s) even for a genuine daily
    exhaustion, so the retry hint cannot distinguish daily from per-minute — a
    genuine daily lock is confirmed downstream by the rate limiter's strike gate
    instead. Falls back to loose text heuristics only when no quotaId is present.
    """

    text = f"{type(exc).__name__}: {exc}"
    ids = [m.group(1).lower() for m in _QUOTA_ID_RE.finditer(text)]
    if ids:
        if any("perday" in q for q in ids):
            return QuotaKind.DAILY
        if any("perminute" in q for q in ids):
            return QuotaKind.PER_MINUTE
        return QuotaKind.OTHER_RATE
    if is_daily_quota_error(exc):
        return QuotaKind.DAILY
    if is_quota_or_rate_limit_error(exc):
        return QuotaKind.OTHER_RATE
    return QuotaKind.NONE


def is_likely_ip_risk_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    # Avoid bare "proxy"/"risk"/"blocked": many unrelated errors contain
    # those words.
    markers = [
        "unsupported location",
        "user location is not supported",
        "location is not supported",
        "ip address",
        "via a proxy",
        "using a proxy",
        "vpn",
        "abuse",
        "abusive",
        "suspicious",
        "unusual traffic",
        "unusual activity",
        "ip risk",
        "risk-blocked",
        "forbidden region",
        "not available in your country",
        "requests from this location",
    ]
    return any(marker in text for marker in markers)


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Whether re-sending the identical request could plausibly work.

    Status-driven, which is what `docs/llm_harness_behavior.md` already
    describes ("参数/鉴权等不可重试 4xx 立即上抛"). A 4xx other than 429 is the
    provider rejecting *this request*: retrying sends the same bytes to the
    same endpoint for the same answer, then the chain tries the next endpoint
    and does it again. With five endpoints and three sticky retries that was up
    to twenty doomed calls and two minutes of backoff for one deterministic
    400 -- on a free tier whose whole daily budget is twenty requests.
    """

    if is_quota_or_rate_limit_error(exc):
        return True
    status = provider_status_code(exc)
    if status is not None:
        return status >= 500
    # No status anywhere: transport-level failures, which are worth a retry.
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "timed out",
    ]
    return any(marker in text for marker in markers)


@lru_cache(maxsize=2)
def _shared_rate_limiter(enabled: bool) -> ModelRateLimiter:
    """One process-wide limiter shared by research/correction/update clients."""

    return ModelRateLimiter(enabled=enabled)


def _append_chain_summary(exc: BaseException, route_decision: Mapping[str, Any]) -> None:
    """Say what the whole chain did, not just what its last link said.

    The last candidate's exception used to surface verbatim, so a run whose
    free tiers had been exhausted by repeated retries died reporting
    "Provider GEMINI_PAID is disabled or has no selected API key" -- true about
    the final link and unrelated to why the call failed. The 2026-08-15
    605-subtitle run is the case: six correction attempts drained the free
    tiers, the chain reached a paid tier with no key, and that is all the user
    saw.

    The message is *appended* rather than replaced, and the exception keeps its
    type: `route_failure_kind` classification is `isinstance`-based and the
    retry-after parsers scan the original text.
    """

    candidates = route_decision.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return
    parts: list[str] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        target = str(item.get("target_id") or "?")
        detail = str(
            item.get("failure_kind")
            or item.get("reason")
            or item.get("outcome")
            or item.get("decision")
            or "unknown"
        )
        parts.append(f"{target}={detail}")
    if len(parts) < 2:
        return
    summary = "; ".join(parts)
    original = str(exc)
    exc.args = (
        f"{original} [chain exhausted: {summary}]",
        *tuple(exc.args[1:]),
    )


def validation_retry_sampling_kwargs(attempt: int) -> Dict[str, Any]:
    """Sampling controls for logical validation/parse retries.

    Attempt 0 starts at the normal temperature. After each validation failure,
    callers retry with the next attempt number, lowering temperature and
    changing seed. The next independent successful workflow naturally starts
    again at attempt 0.
    """

    attempt = max(0, int(attempt))
    temperature = max(
        0.0,
        round(VALIDATION_BASE_TEMPERATURE - VALIDATION_TEMPERATURE_STEP * attempt, 2),
    )
    return {
        "temperature": temperature,
        "seed": _VALIDATION_SEED_BASE + attempt,
    }


def attach_file_to_messages(
    messages: List[Dict[str, Any]],
    file_ref: UploadedFileRef,
) -> List[Dict[str, Any]]:
    if not messages:
        raise ValueError("messages cannot be empty")
    updated = [dict(message) for message in messages]
    user_idx = next(
        (idx for idx in range(len(updated) - 1, -1, -1) if updated[idx].get("role") == "user"),
        len(updated) - 1,
    )
    original = updated[user_idx].get("content", "")
    if isinstance(original, list):
        content = list(original)
    else:
        content = [{"type": "text", "text": str(original)}]
    file_block: Dict[str, Any] = {
        "file_id": file_ref.file_id,
        "filename": file_ref.filename,
        "format": file_ref.mime_type,
    }
    if file_ref.local_path:
        file_block["local_path"] = file_ref.local_path
    if file_ref.agy_prepared:
        file_block["agy_prepared"] = True
    if file_ref.mime_type.startswith("video/"):
        # mm-high clips: low sample rate and low media resolution keep the
        # billed frame tokens at the planned 71 tok/frame x 0.25 fps.
        # video_metadata.fps is mapped to videoMetadata.fps in the REST call.
        file_block["detail"] = "low"
        file_block["video_metadata"] = {"fps": VIDEO_SAMPLE_FPS}
    content.append({"type": "file", "file": file_block})
    updated[user_idx]["content"] = content
    return updated


# Factory assembling messages for a prompt-variant name; ``complete`` accepts
# one in place of a fixed message list so the prompt can follow the candidate
# that actually answers (only the correction call site passes a factory today).
# The variant comes from the cell (plan v2 D2/D3), optionally overridden per
# model-group entry; "" means the session's single template.
TieredMessages = Callable[[str], List[Dict[str, Any]]]


def _as_tiered(messages: List[Dict[str, Any]] | TieredMessages) -> TieredMessages:
    return messages if callable(messages) else (lambda _variant: messages)


def variant_capability_tier(variant: str) -> CapabilityTier:
    """The tier a variant name targets, for reporting/records.

    Variant names bake their tier in (capableB/C, basicA/B); "" -- the
    single-template sessions -- reports CAPABLE.
    """

    return (
        CapabilityTier.BASIC
        if (variant or "").startswith("basic")
        else CapabilityTier.CAPABLE
    )


def _grounding_search_events(
    response: Any, *, tool: str
) -> list[Mapping[str, Any]]:
    """Gemini's own record of a grounded call, in the retrieval ledger's shape.

    `retrieval=native` over `google_search` left no trace: the reply carries the
    queries the model ran and the pages that grounded its answer, and nothing
    read them. A Gemini native round therefore looked identical to a backend
    that reports nothing at all, and "native retrieval is unauditable" was in
    part a statement about this parser.

    One row per call rather than per query: Gemini says which queries it ran and
    which pages grounded the answer, never which page answered which query.
    Splitting would assert an attribution the API does not make -- and multiply
    the result count by the query count.
    """

    from .web_search import (
        gemini_grounding_chunks,
        gemini_grounding_metadata,
        gemini_grounding_queries,
    )

    if not isinstance(response, Mapping):
        return []
    metadata = gemini_grounding_metadata(response)
    if not metadata:
        return []
    queries = [
        text
        for text in (item.strip() for item in gemini_grounding_queries(metadata))
        if text
    ]
    urls: list[str] = []
    for chunk in gemini_grounding_chunks(metadata):
        url = str(chunk.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)
    if not queries and not urls:
        return []
    return [
        {
            "tool": tool or "google_search",
            # Exact whenever the model ran a single query, which is the usual
            # case; joined rather than dropped when it ran several.
            "query": "; ".join(queries),
            "queries": queries,
            "urls": urls,
        }
    ]


def _api_rows_as_execution_attempts(
    attempts: Sequence[Mapping[str, Any]],
    *,
    target_id: str,
    backend: str,
    model: str,
) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for item in attempts:
        row = dict(item)
        row.setdefault("target_id", target_id)
        row.setdefault("backend", backend)
        row.setdefault("reported_model", row.get("model") or model)
        if "duration_ms" not in row and row.get("elapsed_sec") is not None:
            try:
                row["duration_ms"] = int(float(row["elapsed_sec"]) * 1000)
            except (TypeError, ValueError):
                pass
        rows.append(row)
    return rows


class RoleClient:
    def __init__(
        self,
        *,
        role_configs: Mapping[LLMRole, RoleModelConfig] | None = None,
        rate_limiter: ModelRateLimiter | None = None,
        test_profile: bool = False,
        # Sticky same-key retries only. Kept low because even 5xx responses
        # appear to consume Gemini daily quota (observed 2026-07-29).
        max_retries: int = 3,
        # Parallel window dispatch pins every call to the pool's first key --
        # concurrency and key rotation are mutually exclusive (plan A.2): a
        # media-less call rotating keys while a dozen requests are in flight
        # is exactly the multi-active-key risk shape the design avoids.
        pin_all_keys: bool = False,
        router: ModelRouter | None = None,
        execution_settings: ExecutionSettings | None = None,
        local_agent_driver: LocalAgentDriver | None = None,
        # (provider_tier, model): the tier is what picks the CLI, so a
        # factory that only saw the model could not honour the same rule
        # the default registry does.
        local_agent_driver_factory: (
            Callable[[str, str], LocalAgentDriver] | None
        ) = None,
    ) -> None:
        # Explicitly injected configs (test fixtures pinning stub endpoints)
        # win over cell resolution even when a task group is passed.
        self._injected_roles = frozenset(role_configs or ())
        self.role_configs = dict(role_configs or default_role_configs())
        self._cell_configs: Dict[tuple[str, str], RoleModelConfig] = {}
        self.test_profile = test_profile
        self.max_retries = int(max_retries)
        self.pin_all_keys = bool(pin_all_keys)
        if execution_settings is None:
            from .routing.execution_policy import ExecutionSettings, load_execution_settings

            execution_settings = (
                load_execution_settings()
                if router is None
                else ExecutionSettings(policy_id=router.policy_id)
            )
        if router is None:
            from .routing.model_router import ModelRouter

            router = ModelRouter(policy_id=execution_settings.policy_id)
        elif router.policy_id != execution_settings.policy_id:
            raise ValueError(
                f"Router policy {router.policy_id!r} does not match execution settings "
                f"{execution_settings.policy_id!r}"
            )
        self.router = router
        self.execution_settings = execution_settings
        from .routing.execution_policy import execution_identity

        self.execution_identity = execution_identity(
            execution_settings, routes=router.routes
        )
        # Keyed by (provider_tier, model): the tier is what picks the CLI, so
        # two tiers sharing a model id must not share a driver. ``None`` as the
        # tier is the injected-fake wildcard.
        self._local_agent_drivers: Dict[tuple[str | None, str], Any] = {}
        self._local_agent_driver_factory = local_agent_driver_factory
        # video->audio safety-net warnings, once per target for this client
        # (one client per session), not per window (model-routing v2).
        self._media_downgrade_warned: set[str] = set()
        # (provider_tier, absolute clip path) -> the Files API object standing
        # in for it. Client-scoped, not call-scoped: under an agent policy the
        # clip is referenced locally and only uploaded when an API candidate
        # answers, and a per-call cache re-uploaded the same window for its
        # query round, its correction round and every retry.
        self._remote_media_refs: Dict[tuple[str, str], UploadedFileRef] = {}
        # (provider_tier, model, repair-chain key) -> the agent conversation
        # that answered that chain's first attempt. A repair is a follow-up
        # question about output the model just wrote, so resuming that
        # conversation is both cheaper (no re-sending the window) and
        # materially different to the model: in its own session the previous
        # answer is what it wrote, in a fresh one it is a text someone handed
        # it. agy declines the fresh-session form outright, which is why
        # production repairs were blind for that backend.
        self._agent_repair_conversations: Dict[tuple[str | None, str, str], str] = {}
        if local_agent_driver is not None:
            injected_model = local_agent_driver.config.model
            self._local_agent_drivers[(None, injected_model)] = local_agent_driver
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = _shared_rate_limiter(enabled=not test_profile)

    def _local_driver_for_model(self, model: str, provider_tier: str = ""):
        from .routing.execution_policy import driver_for_provider_tier, normalized_tier

        tier = normalized_tier(provider_tier)
        local_driver = (
            self._local_agent_drivers.get((tier, model))
            or self._local_agent_drivers.get((None, model))
            or self._local_agent_drivers.get((None, ""))
        )
        if local_driver is None:
            local_driver = (
                self._local_agent_driver_factory(tier, model)
                if self._local_agent_driver_factory is not None
                else driver_for_provider_tier(
                    self.execution_settings,
                    provider_tier=tier,
                    model=model,
                )
            )
            self._local_agent_drivers[(tier, model)] = local_driver
        configured_model = local_driver.config.model
        if configured_model and configured_model != model:
            raise ValueError(
                f"Local driver model {configured_model!r} does not match "
                f"route fact {model!r}"
            )
        return local_driver

    def _uploaded_media_ref(
        self, ref: UploadedFileRef, *, provider_tier: str
    ) -> UploadedFileRef:
        """A Files API object for a locally-referenced clip, uploaded once.

        Files expire on the service side (48h), and a run long enough to hit
        that would otherwise fail on a stale handle it cached itself, so a
        rejected reuse falls back to uploading again rather than to an error.
        """

        if not ref.local_path:
            raise RuntimeError(
                "Gemini media fallback has neither file_id nor local_path"
            )
        key = (provider_tier, str(Path(ref.local_path).resolve()))
        cached = self._remote_media_refs.get(key)
        if cached is not None:
            return cached
        remote = upload_gemini_file(
            ref.local_path, api_key=_first_gemini_api_key(provider_tier)
        )
        self._remote_media_refs[key] = remote
        return remote

    def _agent_failure(
        self,
        exc: BaseException,
        *,
        endpoint: Any,
        plan: Any,
        quota_pool: str,
        target_id: str = "",
        attempts: List[Mapping[str, Any]] | None = None,
    ):
        """Ask whether a failed agent call means the subscription is spent.

        The probe runs through the same driver the call used, so it is the
        subscription's own answer rather than a guess from the error text --
        which matters because none of the three CLIs can be asked how much
        quota is left.
        """

        tier = endpoint.provider_tier

        def record(row: Mapping[str, Any]) -> None:
            if attempts is None:
                return
            # A probe is a real call against a real subscription. It belongs in
            # the same ledger as every other call, or a run's own artifacts
            # cannot account for what it spent.
            attempts.append(
                {
                    **dict(row),
                    "target_id": f"{target_id} (quota probe)".strip(),
                    "probe": "quota",
                }
            )

        def ping() -> None:
            driver = self._local_driver_for_model(endpoint.api_model_id, tier)
            try:
                probe_result = driver.run(
                    agent_quota.QUOTA_PING_MESSAGES,
                    task="quota-probe",
                    profile_id=f"policy={plan.policy_id};probe=quota",
                    reasoning_effort="low",
                )
            except BaseException as probe_failure:
                for row in (
                    getattr(probe_failure, "_harness_execution_attempts", None) or []
                ):
                    if isinstance(row, Mapping):
                        record(row)
                raise
            record(probe_result.execution_attempt)

        try:
            return agent_quota.evaluate_agent_failure(
                pool=quota_pool,
                exc=exc,
                ping=ping,
                warn=lambda message: print(f"Warning: {message}", file=sys.stderr),
            )
        except Exception:
            # Diagnosis must never replace the failure it was diagnosing.
            return exc

    def _forget_uploaded_media_ref(
        self, ref: UploadedFileRef, *, provider_tier: str
    ) -> None:
        if ref.local_path:
            self._remote_media_refs.pop(
                (provider_tier, str(Path(ref.local_path).resolve())), None
            )

    def _local_agent_ready(
        self,
        model: str,
        provider_tier: str = "",
        native_search: bool = False,
    ) -> bool:
        """Whether *this client's* agent driver can serve ``model``.

        The pre-filter must ask the driver the call would actually use --
        an injected fake in tests, the factory-built one otherwise -- not the
        CLI that happens to be installed.
        """

        try:
            driver = self._local_driver_for_model(model, provider_tier)
            probe = driver.probe()
        except Exception:
            return False
        return driver.meets_requirements(
            probe, native_search=native_search
        )

    def _run_local_agent(
        self,
        driver: "LocalAgentDriver",
        messages: Sequence[Mapping[str, Any]],
        *,
        repair_session_key: str,
        provider_tier: str,
        model: str,
        **call_kwargs: Any,
    ) -> tuple[Any, List[Mapping[str, Any]]]:
        """Run one agent call, resuming this repair chain's conversation.

        The durable task runtime already does this properly, but production
        does not go through it (docs/llm_local_agent.md §12.1.1), so a repair
        used to mean a *fresh* agent process handed its own previous answer as
        plain text. Codex and Claude accept that; agy declines it and retried
        blind. Keeping the handle for the length of one window's attempt chain
        makes the repair a follow-up turn instead, for every backend.

        Deliberately not cross-window reuse: each chain has its own key, so the
        A/B that found agy reuse a net loss between *independent* tasks does
        not apply (docs/llm_followups.md, "validation 失败改为修复轮").

        Returns the result and any execution attempts that were discarded on
        the way to it, so a rebuilt conversation still shows up as two spawns.
        """

        from .agent.local_agent import (
            LocalAgentError,
            LocalAgentPolicyViolationError,
            LocalAgentQuotaError,
        )
        from .routing.execution_policy import normalized_tier

        previous_output = str(call_kwargs.get("previous_output") or "")
        errors = tuple(call_kwargs.get("validation_errors") or ())
        reuse = False
        if repair_session_key:
            try:
                reuse = bool(driver.probe().supports_session_reuse)
            except Exception:
                reuse = False
        if not reuse:
            # An `assignment` scope on a driver without reliable reuse fails
            # before the spawn rather than degrading, so it is opt-in per
            # driver and everything else keeps the full-replay behaviour.
            return driver.run(messages, **call_kwargs), []
        # Keyed like `_local_agent_drivers`, and for the same reason: a session
        # id belongs to the CLI that issued it, and two provider tiers sharing
        # a model id do not share a driver. Keying on the chain alone handed a
        # later attempt -- which re-routes freely, and may land on another
        # vendor or the other tier -- a conversation it does not own.
        chain = (normalized_tier(provider_tier), model, repair_session_key)
        if not previous_output and not errors:
            # First attempt of a chain never inherits a handle: the key is
            # reused across windows of a run, and a stale conversation would
            # answer with the previous window still in context.
            self._agent_repair_conversations.pop(chain, None)
        handle = self._agent_repair_conversations.get(chain, "")
        reuse_kwargs = dict(
            call_kwargs,
            session_scope="assignment",
            conversation_key=repair_session_key,
        )
        discarded: List[Mapping[str, Any]] = []
        try:
            result = driver.run(
                messages, conversation_handle=handle, **reuse_kwargs
            )
        except LocalAgentError as exc:
            # Only a turn that actually tried to resume may rebuild -- the same
            # rule the durable worker follows -- and only for a failure a fresh
            # conversation could plausibly fix. A spent subscription or a
            # refused policy is not one: retrying burns a second spawn and
            # delays the freeze that exists to stop us walking into it.
            if not handle or isinstance(
                exc, (LocalAgentQuotaError, LocalAgentPolicyViolationError)
            ):
                raise
            discarded = list(getattr(exc, "_harness_execution_attempts", []) or [])
            self._agent_repair_conversations.pop(chain, None)
            result = driver.run(messages, conversation_handle="", **reuse_kwargs)
        if getattr(result, "conversation_handle", ""):
            self._agent_repair_conversations[chain] = result.conversation_handle
        return result, discarded

    def _config_for(
        self, role: LLMRole, task_group: str = "", difficulty: str = ""
    ) -> RoleModelConfig:
        """The cell config serving one call (model-routing v2).

        Production call sites pass their task group (and the run's
        difficulty); role-only calls fall back to the role's default cell.
        Explicitly injected ``role_configs`` always win -- they are test
        fixtures pinning stub endpoints.
        """

        if role in self._injected_roles:
            return self.role_configs[role]
        if not task_group and not difficulty:
            return self.role_configs[role]
        task_group = task_group or ROLE_DEFAULT_TASK_GROUP[role]
        difficulty = difficulty or "quality"
        key = (task_group, difficulty)
        if key not in self._cell_configs:
            self._cell_configs[key] = role_config_for(
                task_group, difficulty, role=role, routes=self.router.routes
            )
        return self._cell_configs[key]

    def ensure_eligible_target(
        self,
        role: LLMRole,
        *,
        needs_audio: bool = False,
        needs_video: bool = False,
        native_search: bool = False,
        task_group: str = "",
        difficulty: str = "",
    ) -> None:
        """Fail before auxiliary work when no routed target can serve a call."""

        from .routing.model_router import provider_enabled

        config = self._config_for(role, task_group, difficulty)
        plan = self.router.plan(
            config,
            test_profile=self.test_profile,
            native_search=bool(native_search and not self.test_profile),
        )
        for candidate in plan.candidates:
            if not provider_enabled(
                candidate,
                agent_ready=self._local_agent_ready,
                native_search=bool(native_search and not self.test_profile),
            ):
                continue
            if (
                candidate.endpoint.backend == "gemini_rest"
                and self.rate_limiter.is_daily_exhausted(candidate.endpoint)
            ):
                continue
            if endpoint_supports(
                candidate.endpoint,
                needs_audio=needs_audio,
                needs_video=needs_video,
                needs_native_search=bool(native_search and not self.test_profile),
            ):
                return
        wanted = ", ".join(
            name
            for name, required in (
                ("audio", needs_audio),
                ("video", needs_video),
                ("native_search", native_search and not self.test_profile),
            )
            if required
        ) or "requested call"
        raise RuntimeError(
            f"No eligible target for {plan.task_group_id}/{plan.difficulty} "
            f"before auxiliary work (requires {wanted})"
        )

    def complete(
        self,
        role: LLMRole,
        messages: List[Dict[str, Any]] | TieredMessages,
        *,
        max_tokens: int = 65_536,
        temperature: float = VALIDATION_BASE_TEMPERATURE,
        seed: int | None = None,
        file_ref: UploadedFileRef | None = None,
        fallback_audio_ref: Callable[[], UploadedFileRef | None] | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        native_search: bool = False,
        task_group: str = "",
        difficulty: str = "",
        previous_output: str = "",
        validation_errors: Sequence[str] = (),
        # Names the chain of attempts answering one unit of work, so an agent
        # repair can resume the conversation that produced the output it is
        # repairing. Empty keeps the full-replay behaviour.
        repair_session_key: str = "",
    ) -> LLMCallResult:
        """Call the role's routed endpoint chain.

        ``messages`` is either a fixed list or a :data:`TieredMessages` factory;
        with a factory the prompt is assembled for the *variant* of the
        candidate about to answer (cell default or per-entry override), so a
        fallback never receives a prompt written for a different variant. The
        variant's tier is reported back via ``LLMCallResult.capability_tier``.

        ``fallback_audio_ref`` is the one-level media ladder (model-routing v2):
        when ``file_ref`` is a video and a candidate can hear but not watch,
        the candidate receives this audio clip instead of being skipped. It is
        a safety net, not a mechanism -- normal configurations put
        video-capable models where video is asked for -- so triggering it
        warns (once per target) and marks the decision trace.

        ``native_search`` asks for the model's own web-search tool. It is a
        per-call capability (plan v2 D4): the plan swaps in the policy's
        native prepends and the candidate loop filters the bound group by
        ``supports_native_search`` -- an empty result is an error, never a
        silent downgrade.

        ``previous_output`` + ``validation_errors`` turn a retry into a repair
        round. The representation is the transport's business, which is why it
        is a call argument and not two more messages from the caller: a
        stateless endpoint receives them as an assistant/user turn pair, while
        a local agent receives them as capsule inputs -- and an agent resuming
        its own session already has the output in context, so it is handed the
        errors alone. Both are optional and are dropped silently when a
        candidate cannot fit them.
        """

        from . import llm_runtime
        from .prompt_compose import compose_repair_turns

        config = self._config_for(role, task_group, difficulty)
        call_thinking_budget = (
            config.thinking_budget if thinking_budget is None else thinking_budget
        )
        call_thinking_level = (
            config.thinking_level if thinking_level is None else thinking_level
        )
        native_search_requested = bool(native_search and not self.test_profile)
        tiered = _as_tiered(messages)
        repair_errors = [
            text for text in (str(item).strip() for item in validation_errors) if text
        ]
        repair_turns = compose_repair_turns(previous_output, repair_errors)
        composed: Dict[tuple[str, str, bool], List[Dict[str, Any]]] = {}
        estimates: Dict[tuple[str, str, bool, bool], int] = {}

        def ref_key(ref: UploadedFileRef | None) -> str:
            if ref is None:
                return ""
            return ref.file_id or ref.local_path

        def messages_for(
            variant: str, ref: UploadedFileRef | None, *, repair: bool = False
        ) -> List[Dict[str, Any]]:
            # Lazy per-(variant, attachment, repair) assembly: built at most
            # once, and only when the endpoint loop actually reaches that
            # combination (the attachment can differ per candidate under the
            # media ladder, and repair context can be dropped per candidate).
            key = (variant, ref_key(ref), repair)
            if key not in composed:
                base = tiered(variant)
                # Attach first: the clip belongs to the turn that states the
                # task, not to the repair request appended after it.
                base = attach_file_to_messages(base, ref) if ref else base
                composed[key] = base + repair_turns if repair else base
            return composed[key]

        def estimate_for(
            variant: str,
            ref: UploadedFileRef | None,
            *,
            high_resolution_video: bool,
            repair: bool = False,
        ) -> int:
            key = (variant, ref_key(ref), high_resolution_video, repair)
            if key not in estimates:
                estimates[key] = estimate_call_input_tokens(
                    messages_for(variant, ref, repair=repair),
                    file_ref=ref,
                    execution_settings=self.execution_settings,
                    high_resolution_video=high_resolution_video,
                )
            return estimates[key]

        last_exc: BaseException | None = None
        accumulated_api_attempts: List[Mapping[str, Any]] = []
        accumulated_execution_attempts: List[Mapping[str, Any]] = []
        from .routing.model_router import (
            candidate_quota_pool,
            classify_failure,
            provider_enabled,
        )

        plan = self.router.plan(
            config,
            test_profile=self.test_profile,
            native_search=native_search_requested,
        )
        route_decision = plan.decision_trace()
        if repair_turns:
            route_decision["repair_context"] = {
                "previous_output_chars": len(previous_output),
                "validation_errors": len(repair_errors),
            }

        def mark_unreached(start: int, reason: str) -> None:
            for pending in route_decision["candidates"][start:]:
                if pending["decision"] == "pending":
                    pending.update(decision="skipped", reason=reason)

        for idx, candidate in enumerate(plan.candidates):
            endpoint = candidate.endpoint
            decision: Dict[str, Any] = route_decision["candidates"][idx]
            if endpoint.backend not in {
                "gemini_rest",
                "local_agent",
                "openai_compat",
                "anthropic",
            }:
                decision.update(decision="skipped", reason="backend_unavailable")
                continue
            if not provider_enabled(
                candidate,
                agent_ready=self._local_agent_ready,
                native_search=native_search_requested,
            ):
                decision.update(decision="skipped", reason="provider_disabled")
                continue
            if (
                endpoint.backend == "gemini_rest"
                and self.rate_limiter.is_daily_exhausted(endpoint)
            ):
                decision.update(decision="skipped", reason="daily_exhausted")
                continue
            catalog_entry = candidate.fact
            # Capability filtering: drop endpoints the catalog says cannot
            # serve this call. Media bits come from what is actually attached,
            # so a text-only fallback never receives an audio/video part.
            candidate_ref = file_ref
            media_downgrade = ""
            if not endpoint_supports(
                endpoint,
                needs_audio=file_ref is not None and file_ref.is_audio,
                needs_video=file_ref is not None and file_ref.is_video,
                needs_native_search=native_search_requested,
            ):
                # One-level media ladder (model-routing v2): a candidate that can
                # hear but not watch gets the audio clip instead of being
                # skipped. There is no further rung -- audio never degrades
                # to text, and native search never degrades at all.
                downgradable = (
                    file_ref is not None
                    and file_ref.is_video
                    and fallback_audio_ref is not None
                    and endpoint_supports(
                        endpoint,
                        needs_audio=True,
                        needs_native_search=native_search_requested,
                    )
                )
                audio_ref = fallback_audio_ref() if downgradable else None
                if audio_ref is None:
                    decision.update(
                        decision="skipped", reason="capability_mismatch"
                    )
                    continue
                candidate_ref = audio_ref
                media_downgrade = "video->audio"
                if candidate.target_id not in self._media_downgrade_warned:
                    self._media_downgrade_warned.add(candidate.target_id)
                    print(
                        f"Warning: target {candidate.target_id} 不支持视频，"
                        "本会话对它按 video->audio 阶梯降一级发送音频剪辑"
                        "（安全网；正常配置应为该格配备有视频能力的模型）。",
                        file=sys.stderr,
                    )
            # Variant ownership (plan v2 D2/D3): the cell's default, unless
            # the model-group entry overrides it. The tier is derived from
            # the variant name for reporting only.
            candidate_variant = config.variant_overrides.get(
                candidate.target_id, config.variant
            )
            tier = variant_capability_tier(candidate_variant)
            # Whether this candidate gets the repair context at all, in whatever
            # form. The estimate below has to include it either way: an agent
            # receives the same text as capsule inputs, so measuring the bare
            # messages would under-count exactly the calls that grew.
            repair_enabled = bool(repair_turns)
            high_resolution_video = bool(
                candidate_ref is not None
                and candidate_ref.is_video
                and catalog_entry is not None
                and catalog_entry.video_high_resolution_only
            )
            estimated_input = estimate_for(
                candidate_variant,
                candidate_ref,
                high_resolution_video=high_resolution_video,
                repair=repair_enabled,
            )
            # Per-fact estimate correction (plan v2 D14). The 3-tier counter
            # speaks Gemini's vocabulary, so another model's estimate carries a
            # systematic bias; ``token_scale`` corrects it for the two things
            # estimates are for -- the input-limit check and the TPM
            # reservation. Reported usage is never scaled: reports and the
            # calibration suggestion read the API's own numbers.
            unscaled_estimate = estimated_input
            token_scale = (
                float(getattr(catalog_entry, "token_scale", 1.0) or 1.0)
                if catalog_entry is not None
                else 1.0
            )
            if token_scale != 1.0:
                estimated_input = int(round(estimated_input * token_scale))
            if catalog_entry is not None:
                if max_tokens > catalog_entry.max_output_tokens:
                    decision.update(
                        decision="skipped",
                        reason="output_limit",
                        requested=max_tokens,
                        limit=catalog_entry.max_output_tokens,
                    )
                    continue
                if (
                    estimated_input > catalog_entry.max_input_tokens
                    and repair_enabled
                ):
                    # The repair context is an aid, not the task. A window that
                    # only fits without it still gets its retry -- a blind one,
                    # which is what every retry was before this existed.
                    repair_enabled = False
                    estimated_input = estimate_for(
                        candidate_variant,
                        candidate_ref,
                        high_resolution_video=high_resolution_video,
                        repair=False,
                    )
                    unscaled_estimate = estimated_input
                    if token_scale != 1.0:
                        estimated_input = int(round(estimated_input * token_scale))
                    decision["repair_context"] = "dropped_input_limit"
                if estimated_input > catalog_entry.max_input_tokens:
                    decision.update(
                        decision="skipped",
                        reason="input_limit",
                        estimated=estimated_input,
                        limit=catalog_entry.max_input_tokens,
                        capability_tier=tier.value,
                    )
                    continue
            native_search_tool = (
                endpoint.native_search_tool if native_search_requested else ""
            )
            # An agent takes the repair context as capsule inputs, so putting it
            # in the messages too would hand it the same thing twice.
            repair_in_messages = repair_enabled and endpoint.backend != "local_agent"
            call_messages_base = messages_for(
                candidate_variant, candidate_ref, repair=repair_in_messages
            )
            decision.update(
                decision="accepted",
                estimated_input_tokens=estimated_input,
                capability_tier=tier.value,
            )
            if token_scale != 1.0:
                decision["token_scale"] = token_scale
                decision["unscaled_estimate_tokens"] = unscaled_estimate
            if media_downgrade:
                # Artifact-visible record of the safety net firing (§8-3).
                decision["media_downgrade"] = media_downgrade
            if high_resolution_video:
                decision["video_resolution_tier"] = "high_only"
            # Per-model thinking mapping (owner design 2026-08-11): translate
            # the cell's abstract high/medium/low request through the selected
            # fact before dispatch. Local Codex uses the same mapping as API
            # transports; an explicitly configured global local-agent effort
            # remains a compatibility override. Facts unknown to the catalog
            # (test fixtures) pass the level through unchanged.
            if catalog_entry is None:
                mapped_thinking = call_thinking_level
                mapped_budget = call_thinking_budget
            else:
                mapped_thinking = thinking_value_for(
                    catalog_entry, call_thinking_level
                )
                mapped_budget = (
                    0
                    if catalog_entry.thinking_levels is None
                    else call_thinking_budget
                )
            try:
                dispatch_messages = call_messages_base
                if (
                    endpoint.backend == "gemini_rest"
                    and candidate_ref is not None
                    and not candidate_ref.file_id
                ):
                    if not candidate_ref.local_path:
                        raise RuntimeError(
                            "Gemini media fallback has neither file_id nor local_path"
                        )
                    remote_ref = self._uploaded_media_ref(
                        candidate_ref, provider_tier=endpoint.provider_tier
                    )
                    dispatch_messages = messages_for(
                        candidate_variant, remote_ref, repair=repair_in_messages
                    )
                if endpoint.backend == "local_agent":
                    # A model that declares `thinking = false` takes no such
                    # parameter at all, and the global override must not force
                    # one on it: agy rejects `--effort` for its Claude models
                    # *before* the call, and that hard failure classifies as
                    # transient -- two of them and the probe freezes the whole
                    # allowance for two hours over a flag.
                    takes_thinking = (
                        candidate.fact is None
                        or candidate.fact.thinking_levels is not None
                    )
                    agent_thinking = (
                        (
                            self.execution_settings.local_agent_reasoning_effort
                            or mapped_thinking
                        )
                        if takes_thinking
                        else ""
                    )
                    local_driver = self._local_driver_for_model(
                        endpoint.api_model_id, endpoint.provider_tier
                    )
                    agent_result, rebuilt_from = self._run_local_agent(
                        local_driver,
                        dispatch_messages,
                        repair_session_key=repair_session_key,
                        provider_tier=endpoint.provider_tier,
                        model=endpoint.api_model_id,
                        task=plan.task,
                        native_search=native_search_requested,
                        profile_id=(
                            f"policy={plan.policy_id};target={candidate.target_id};"
                            f"route={plan.routing_identity_digest}"
                        ),
                        reasoning_effort=agent_thinking,
                        previous_output=previous_output if repair_enabled else "",
                        validation_errors=repair_errors if repair_enabled else (),
                    )
                    # A call that worked is the cheapest possible proof the
                    # subscription is alive, and it clears any freeze.
                    agent_quota.default_ledger().note_success(
                        candidate_quota_pool(candidate)
                    )
                    # The driver already extracted the search rows from its own
                    # event dialect; recomputing them here would be a second
                    # copy of that filter, free to drift from the first.
                    execution_attempt = dict(agent_result.execution_attempt)
                    execution_attempt["target_id"] = candidate.target_id
                    all_execution_attempts = [
                        *accumulated_execution_attempts,
                        # A resume that went cold spawned too, and an audit that
                        # cannot see it reads one call where two happened.
                        *(
                            {**dict(item), "target_id": candidate.target_id}
                            for item in rebuilt_from
                        ),
                        execution_attempt,
                    ]
                    local_usage = dict(agent_result.usage)
                    reasoning_tokens = int(
                        local_usage.get("reasoning_output_tokens", 0)
                        or local_usage.get("reasoning_tokens", 0)
                        or 0
                    )
                    output_tokens = int(local_usage.get("output_tokens", 0) or 0)
                    input_tokens = int(local_usage.get("input_tokens", 0) or 0)
                    cached_tokens = int(
                        local_usage.get("cached_input_tokens", 0) or 0
                    )
                    total_tokens = int(
                        local_usage.get("total_tokens", 0)
                        or input_tokens + output_tokens
                    )
                    raw_response = {
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "prompt_tokens_details": {
                                "cached_tokens": cached_tokens,
                            },
                            "completion_tokens": output_tokens,
                            "completion_tokens_details": {
                                "reasoning_tokens": reasoning_tokens,
                            },
                            "total_tokens": total_tokens,
                        },
                        "agent": {
                            "capsule_id": agent_result.episode_id,
                            "events": list(agent_result.normalized_events),
                        },
                    }
                    decision.update(outcome="success")
                    mark_unreached(idx + 1, "not_reached_after_success")
                    return LLMCallResult(
                        content=agent_result.content,
                        role=role,
                        model=agent_result.reported_model,
                        fallback_used=idx > 0,
                        raw_response=raw_response,
                        capability_tier=tier,
                        variant=candidate_variant,
                        thinking_level=agent_thinking,
                        thinking_budget=0,
                        api_attempts=list(accumulated_api_attempts),
                        execution_attempts=all_execution_attempts,
                        target_id=candidate.target_id,
                        backend=endpoint.backend,
                        route_decision=route_decision,
                    )
                call_kwargs: Dict[str, Any] = {
                    "provider_tier": endpoint.provider_tier,
                    "model": endpoint.api_model_id,
                    "thinking_budget": mapped_budget,
                    "thinking_level": mapped_thinking,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "retries": self.max_retries,
                    "native_search_tool": native_search_tool or None,
                }
                # User-declared providers route through the thin text
                # transports (model-routing v2); packaged Gemini tiers keep the
                # hand-written REST path.
                provider = self.router.routes.providers.get(endpoint.provider_tier)
                if provider is not None and provider.kind in (
                    "openai_compat",
                    "anthropic",
                ):
                    call_kwargs["provider_spec"] = {
                        "kind": provider.kind,
                        "base_url": provider.base_url,
                        "key_env": provider.key_env,
                    }
                call_kwargs.update({
                    # An uploaded file is project-scoped to the first key; a
                    # rotated key would 403 on it, so pin media calls.
                    "pin_first_key": file_ref is not None or self.pin_all_keys,
                    # Per-key rate limiting (RPM/TPM + daily) lives inside
                    # chat_complete where the answering key is known.
                    "rate_limiter": self.rate_limiter,
                    "estimated_input_tokens": estimated_input,
                })
                if seed is not None:
                    call_kwargs["seed"] = seed
                response = llm_runtime.chat_complete(dispatch_messages, **call_kwargs)
                plain = _to_plain_response(response)
                actual_input = extract_token_distribution(plain)["total_input_tokens"]
                if actual_input > 0 and unscaled_estimate > 0:
                    # Calibration reference (model-routing v2): the provider's own
                    # number over our *unscaled* local estimate. Collected per
                    # fact so the task report can suggest a ``token_scale``;
                    # never written back automatically -- the scale moves the
                    # window geometry, so adopting it is an explicit config
                    # change the user makes.
                    decision["estimate_calibration"] = {
                        "fact_id": endpoint.fact_id or endpoint.api_model_id,
                        "provider_tier": endpoint.provider_tier,
                        "estimated_unscaled": unscaled_estimate,
                        "actual": actual_input,
                        "ratio": round(actual_input / unscaled_estimate, 4),
                        "token_scale": token_scale,
                    }
                if actual_input <= 0:
                    actual_input = estimated_input
                answering_key_id = str(plain.pop("_harness_key_id", "") or "")
                # Popped unconditionally so the ticket object never lands in
                # raw_response artifacts; with it, settle refines this call's
                # own TPM reservation instead of whichever event is newest.
                rate_ticket = plain.pop("_harness_rate_ticket", None)
                self.rate_limiter.settle(
                    endpoint,
                    actual_input_tokens=actual_input,
                    estimated_input_tokens=estimated_input,
                    key_id=answering_key_id,
                    ticket=rate_ticket,
                )
                content = llm_runtime.extract_message_content(plain)
                api_key_label = str(plain.pop("_harness_api_key_label", "") or "")
                api_attempts = list(plain.pop("_harness_api_attempts", []) or [])
                all_api_attempts = [*accumulated_api_attempts, *api_attempts]
                all_execution_attempts = [
                    *accumulated_execution_attempts,
                    *_api_rows_as_execution_attempts(
                        api_attempts,
                        target_id=candidate.target_id,
                        backend=endpoint.backend,
                        model=endpoint.api_model_id,
                    ),
                ]
                # Whatever the provider reports it grounded on, whether or not
                # this call asked for search: grounding nobody requested is the
                # kind of thing an audit should see, not the kind it should
                # have to infer. It belongs to the attempt that answered.
                grounding = _grounding_search_events(
                    plain, tool=native_search_tool
                )
                if grounding and all_execution_attempts:
                    answering = dict(all_execution_attempts[-1])
                    answering["search_events"] = grounding
                    all_execution_attempts[-1] = answering
                decision.update(outcome="success")
                mark_unreached(idx + 1, "not_reached_after_success")
                return LLMCallResult(
                    content=content,
                    role=role,
                    model=endpoint.api_model_id,
                    fallback_used=idx > 0,
                    raw_response=plain,
                    capability_tier=tier,
                    variant=candidate_variant,
                    api_key_label=api_key_label,
                    # What was actually sent, i.e. after the fact's thinking
                    # mapping. Identical to the abstract knob for every
                    # identity-mapped model; a model that remaps (or declares
                    # ``false``) would otherwise be misreported in artifacts.
                    thinking_level=mapped_thinking or "",
                    thinking_budget=int(call_thinking_budget or 0),
                    api_attempts=all_api_attempts,
                    execution_attempts=all_execution_attempts,
                    target_id=candidate.target_id,
                    backend=endpoint.backend,
                    route_decision=route_decision,
                )
            except Exception as exc:  # pragma: no cover - network/provider behavior
                if endpoint.backend == "local_agent":
                    exc = self._agent_failure(
                        exc,
                        endpoint=endpoint,
                        plan=plan,
                        quota_pool=candidate_quota_pool(candidate),
                        target_id=candidate.target_id,
                        attempts=accumulated_execution_attempts,
                    )
                last_exc = exc
                if endpoint.backend == "gemini_rest" and candidate_ref is not None:
                    # The cached Files API object may be the thing that failed
                    # (they expire after 48h), and we cannot tell that apart
                    # from an ordinary provider error. Dropping it costs one
                    # re-upload; keeping a dead handle costs every later call.
                    self._forget_uploaded_media_ref(
                        candidate_ref, provider_tier=endpoint.provider_tier
                    )
                new_api_attempts = list(
                    getattr(exc, "_harness_api_attempts", []) or []
                )
                new_execution_attempts = list(
                    getattr(exc, "_harness_execution_attempts", []) or []
                )
                if endpoint.backend == "local_agent":
                    new_execution_attempts = [
                        {**dict(item), "target_id": candidate.target_id}
                        for item in new_execution_attempts
                    ]
                    _record_vendor_error(new_execution_attempts)
                elif not new_execution_attempts:
                    new_execution_attempts = _api_rows_as_execution_attempts(
                        new_api_attempts,
                        target_id=candidate.target_id,
                        backend=endpoint.backend,
                        model=endpoint.api_model_id,
                    )
                accumulated_api_attempts.extend(new_api_attempts)
                accumulated_execution_attempts.extend(
                    new_execution_attempts or new_api_attempts
                )
                if getattr(exc, "_harness_consecutive_timeout_abort", False):
                    decision.update(outcome="failed", failure_kind="timeout")
                    mark_unreached(idx + 1, "timeout_abort")
                    setattr(exc, "_harness_route_decision", route_decision)
                    setattr(
                        exc,
                        "_harness_execution_attempts",
                        list(accumulated_execution_attempts),
                    )
                    raise
                if isinstance(exc, LLMIPRiskError) or is_likely_ip_risk_error(exc):
                    decision.update(outcome="failed", failure_kind="permanent")
                    error = LLMIPRiskError(
                        "LLM IP risk warning: provider response suggests this "
                        f"IP/proxy was risk-blocked: {exc}"
                    )
                    mark_unreached(idx + 1, "permanent_failure")
                    setattr(
                        error,
                        "_harness_api_attempts",
                        list(getattr(exc, "_harness_api_attempts", []) or []),
                    )
                    setattr(error, "_harness_route_decision", route_decision)
                    setattr(
                        error,
                        "_harness_execution_attempts",
                        list(accumulated_execution_attempts),
                    )
                    raise error from None
                failure = classify_failure(exc)
                decision.update(outcome="failed", failure_kind=failure.value)
                setattr(exc, "_harness_route_decision", route_decision)
                setattr(
                    exc,
                    "_harness_execution_attempts",
                    list(accumulated_execution_attempts),
                )
                if failure.value in candidate.fallback_on:
                    continue
                mark_unreached(idx + 1, "fallback_not_allowed")
                raise
        if last_exc is not None:
            setattr(last_exc, "_harness_route_decision", route_decision)
            setattr(
                last_exc,
                "_harness_execution_attempts",
                list(accumulated_execution_attempts),
            )
            _append_chain_summary(last_exc, route_decision)
            raise last_exc
        reason_counts: Dict[str, int] = {}
        for item in route_decision["candidates"]:
            reason = str(item.get("reason") or item.get("failure_kind") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        # `reason x count`, not `reason=count`: these are candidate counts, and
        # "input_limit=2" was read as "the input limit is 2" by its own author
        # while diagnosing a real run.
        summary = ", ".join(
            f"{reason}x{count}" for reason, count in sorted(reason_counts.items())
        )
        error = RuntimeError(
            f"No eligible target for role {role.value}"
            + (f" ({summary})." if summary else ".")
        )
        setattr(error, "_harness_route_decision", route_decision)
        setattr(error, "_harness_execution_attempts", list(accumulated_execution_attempts))
        raise error


def _to_plain_response(response: Any) -> Mapping[str, Any]:
    """Normalize a response into a plain dict for artifacts.

    Downstream helpers (``extract_token_distribution``,
    ``_provider_reference_metadata``, ``_response_finish_reason``) expect a
    ``Mapping``/``dict`` in the Gemini REST response shape. The runtime now
    returns a plain dict directly; this function is kept for safety and to
    preserve harness metadata attributes.
    """

    label = getattr(response, "_harness_api_key_label", "")
    if not label and isinstance(response, Mapping):
        label = response.get("_harness_api_key_label", "")
    key_id = getattr(response, "_harness_key_id", "")
    if not key_id and isinstance(response, Mapping):
        key_id = response.get("_harness_key_id", "")
    attempts = getattr(response, "_harness_api_attempts", None)
    if attempts is None and isinstance(response, Mapping):
        attempts = response.get("_harness_api_attempts")

    dumped: Dict[str, Any] = {}
    if isinstance(response, Mapping):
        dumped = dict(response)
    else:
        for attr in ("model_dump", "dict"):
            method = getattr(response, attr, None)
            if callable(method):
                try:
                    res = method()
                    if isinstance(res, Mapping):
                        dumped = dict(res)
                        break
                except Exception:  # pragma: no cover - defensive
                    continue
    if label:
        dumped["_harness_api_key_label"] = label
    if key_id:
        dumped["_harness_key_id"] = key_id
    if attempts:
        dumped["_harness_api_attempts"] = list(attempts)
    return dumped


def _first_gemini_api_key(provider_tier: str | None = None) -> str:
    from . import llm_runtime

    env_map = llm_runtime._read_dotenv()
    if provider_tier is None:
        entry, _tier = api_keys.first_enabled_gemini_entry(env_map)
        return entry.key
    key, _ = llm_runtime._first_key_for_tier(provider_tier, env_map)
    return key


def extract_token_distribution(response: Any) -> Dict[str, int]:
    """Full prompt/thinking/output token split for reports and exchange headers.

    Handles raw Gemini REST responses (``usageMetadata`` with per-modality
    ``promptTokensDetails``) and OpenAI-style litellm ``usage`` (where
    ``completion_tokens`` includes reasoning tokens). Missing fields are 0.

    Input breakdown:
    - ``uncached_input_tokens`` — billable fresh prompt tokens
    - ``cached_input_tokens`` — prompt tokens served from context cache
    - ``total_input_tokens`` — full prompt side (uncached + cached)

    Output breakdown:
    - ``output_tokens`` — visible completion, excluding thinking
    - ``thinking_tokens`` — internal reasoning tokens
    - ``total_output_tokens`` — output + thinking
    """

    dist = {
        "prompt_tokens": 0,
        "prompt_text_tokens": 0,
        "prompt_audio_tokens": 0,
        "uncached_input_tokens": 0,
        "cached_input_tokens": 0,
        "total_input_tokens": 0,
        "thinking_tokens": 0,
        "output_tokens": 0,
        "total_output_tokens": 0,
        "total_tokens": 0,
    }
    usage: Any = None
    if isinstance(response, Mapping):
        usage = response.get("usageMetadata") or response.get("usage_metadata")
        if usage is None:
            usage = response.get("usage")
    else:
        usage = getattr(response, "usage", None)
    if usage is None:
        return dist

    def _get(*keys: str) -> Any:
        for key in keys:
            if isinstance(usage, Mapping):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            if value is not None:
                return value
        return None

    def _int(*keys: str) -> int:
        value = _get(*keys)
        return int(value) if isinstance(value, (int, float)) else 0

    dist["prompt_tokens"] = _int("promptTokenCount", "prompt_token_count", "prompt_tokens")
    dist["total_tokens"] = _int("totalTokenCount", "total_token_count", "total_tokens")
    candidates = _get("candidatesTokenCount", "candidates_token_count")
    if isinstance(candidates, (int, float)):
        # Gemini REST shape: thoughts are reported separately from candidates.
        dist["output_tokens"] = int(candidates)
        dist["thinking_tokens"] = _int("thoughtsTokenCount", "thoughts_token_count")
    else:
        # OpenAI/litellm shape: completion_tokens includes reasoning tokens.
        completion = _int("completion_tokens")
        details = _get("completion_tokens_details")
        reasoning = None
        if isinstance(details, Mapping):
            reasoning = details.get("reasoning_tokens")
        elif details is not None:
            reasoning = getattr(details, "reasoning_tokens", None)
        thinking = int(reasoning) if isinstance(reasoning, (int, float)) else 0
        dist["thinking_tokens"] = thinking
        dist["output_tokens"] = max(0, completion - thinking)

    audio = 0
    modality_details = _get("promptTokensDetails", "prompt_tokens_details")
    if isinstance(modality_details, Sequence) and not isinstance(modality_details, (str, bytes)):
        # Raw Gemini REST shape: list of {modality, tokenCount}.
        for detail in modality_details:
            if not isinstance(detail, Mapping):
                continue
            modality = str(detail.get("modality", "")).upper()
            count = detail.get("tokenCount") or detail.get("token_count")
            if modality == "AUDIO" and isinstance(count, (int, float)):
                audio += int(count)
    elif modality_details is not None:
        # litellm shape: PromptTokensDetails(Wrapper) exposes an audio_tokens field
        # (dict after model_dump, object otherwise).
        if isinstance(modality_details, Mapping):
            count = modality_details.get("audio_tokens") or modality_details.get("audioTokens")
        else:
            count = getattr(modality_details, "audio_tokens", None)
        if isinstance(count, (int, float)):
            audio += int(count)
    dist["prompt_audio_tokens"] = audio
    dist["prompt_text_tokens"] = max(0, dist["prompt_tokens"] - audio)

    cached = _int(
        "cachedContentTokenCount",
        "cached_content_token_count",
        "cache_read_input_tokens",
    )
    if cached == 0:
        if isinstance(modality_details, Mapping):
            nested = modality_details.get("cached_tokens") or modality_details.get(
                "cached_content_token_count"
            )
            if isinstance(nested, (int, float)):
                cached = int(nested)
        elif modality_details is not None:
            nested = getattr(modality_details, "cached_tokens", None)
            if isinstance(nested, (int, float)):
                cached = int(nested)
    dist["cached_input_tokens"] = max(0, cached)
    dist["total_input_tokens"] = dist["prompt_tokens"]
    dist["uncached_input_tokens"] = max(0, dist["prompt_tokens"] - dist["cached_input_tokens"])
    dist["total_output_tokens"] = dist["output_tokens"] + dist["thinking_tokens"]
    return dist


def sum_token_distributions(distributions: Iterable[Mapping[str, int]]) -> Dict[str, int]:
    """Sum token distribution dicts (union of keys) for report totals."""

    totals: Dict[str, int] = {}
    count = 0
    for dist in distributions:
        count += 1
        for key, value in dist.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + int(value)
    totals["call_count"] = count
    return totals


def is_likely_output_limited(
    response: Any,
    *,
    max_tokens: int,
    margin: int = 100,
) -> bool:
    """True when output+thinking tokens land within ``margin`` of the cap.

    Catches truncation that the provider does not flag with a MAX_TOKENS finish
    reason (thinking tokens count against the same budget).
    """

    dist = extract_token_distribution(response)
    total = dist["output_tokens"] + dist["thinking_tokens"]
    return total > 0 and total >= max_tokens - margin


def upload_gemini_file(path: str | Path, *, api_key: str | None = None) -> UploadedFileRef:
    file_path = Path(path).expanduser().resolve()
    return _upload_gemini_file_rest(file_path, api_key=api_key or _first_gemini_api_key())


class GeminiPromptBlockedError(RuntimeError):
    """The prompt tripped a non-configurable Gemini prompt classifier.

    Signature: HTTP 200, empty content, finish_reason=content_filter,
    promptFeedback.blockReason=PROHIBITED_CONTENT (safety_settings cannot
    disable it). Observed false-positive on ordinary VTuber material where the
    trigger was compositional — injected web-extract text plus the rest of the
    prompt. Deterministic for the exact prompt, so retrying unchanged is
    pointless; callers should drop optional injected blocks and rebuild.
    """


def extract_finish_reason(raw_response: Any) -> str:
    if not isinstance(raw_response, dict):
        return ""
    candidates = raw_response.get("choices") or raw_response.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    return str(
        candidates[0].get("finish_reason") or candidates[0].get("finishReason") or ""
    )


def is_prompt_blocked(content: str | None, raw_response: Any) -> bool:
    """True when a call returned nothing because the prompt was filter-blocked."""

    if (content or "").strip():
        return False
    reason = extract_finish_reason(raw_response).lower()
    # ``refusal`` is Anthropic's stop_reason for a declined reply (HTTP 200);
    # with empty content it is this same "prompt was blocked" shape.
    if reason in {"content_filter", "safety", "prohibited_content", "blocklist", "refusal"}:
        return True
    if not isinstance(raw_response, Mapping):
        return False
    feedback = raw_response.get("promptFeedback") or raw_response.get("prompt_feedback")
    if not isinstance(feedback, Mapping):
        return False
    block_reason = str(
        feedback.get("blockReason") or feedback.get("block_reason") or ""
    ).strip()
    return bool(block_reason)


# The Files API can report ACTIVE while the media is still being prepared for
# sampling: generateContent issued seconds after a video upload has returned
# HTTP 200 with zero output and text-only billing (observed 2026-07-11).
# countTokens is free (auth-only), so it doubles as an exact readiness probe:
# once the file's media tokens are actually counted, generateContent sees the
# media too.
GEMINI_MEDIA_PROBE_MODEL = "gemini-3.1-flash-lite"


def _wait_for_media_tokens(
    client: Any,
    *,
    api_base: str,
    auth_header: Mapping[str, str],
    file_uri: str,
    mime_type: str,
    sleep_func: Callable[[float], None],
    poll_interval_seconds: float,
    max_poll_attempts: int,
) -> None:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"fileData": {"fileUri": file_uri, "mimeType": mime_type}}],
            }
        ]
    }
    url = f"{api_base}/models/{GEMINI_MEDIA_PROBE_MODEL}:countTokens"
    for _ in range(max_poll_attempts):
        response = client.post(
            url,
            headers={**auth_header, "Content-Type": "application/json"},
            json=body,
        )
        if response.status_code == 200:
            try:
                total = int(response.json().get("totalTokens") or 0)
            except (TypeError, ValueError, AttributeError):
                total = 0
            if total > 0:
                return
        sleep_func(poll_interval_seconds)
    raise TimeoutError(
        f"Gemini media never became countable after upload: {file_uri}"
    )


def _upload_gemini_file_rest(
    file_path: Path,
    *,
    api_key: str,
    client_factory: Callable[..., Any] = httpx.Client,
    sleep_func: Callable[[float], None] = time.sleep,
    poll_interval_seconds: float = 2.0,
    max_poll_attempts: int = 60,
    probe_poll_attempts: int = 150,
) -> UploadedFileRef:
    """Upload media to Gemini Files API using the resumable REST protocol."""

    data = file_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    upload_base = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    api_base = "https://generativelanguage.googleapis.com/v1beta"
    auth_header = {"x-goog-api-key": api_key}
    with client_factory(timeout=600.0) as client:
        start = client.post(
            upload_base,
            headers={
                **auth_header,
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(data)),
                "X-Goog-Upload-Header-Content-Type": mime_type,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": file_path.name}},
        )
        start.raise_for_status()
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError("Gemini Files API did not return x-goog-upload-url.")

        finalize = client.post(
            upload_url,
            headers={
                "Content-Length": str(len(data)),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=data,
        )
        finalize.raise_for_status()
        file_obj = finalize.json().get("file", {})
        name = file_obj.get("name")
        if not name:
            raise RuntimeError("Gemini Files API response did not include file.name.")

        for _ in range(max_poll_attempts):
            state = str(file_obj.get("state") or "").upper()
            if state in {"", "ACTIVE"}:
                break
            if state == "FAILED":
                raise RuntimeError(f"Gemini file processing failed: {file_obj}")
            sleep_func(poll_interval_seconds)
            status = client.get(f"{api_base}/{name}", headers=auth_header)
            status.raise_for_status()
            file_obj = status.json()
        else:
            raise TimeoutError(f"Gemini file did not become ACTIVE: {name}")

        file_id = file_obj.get("uri") or name
        final_mime = file_obj.get("mimeType") or mime_type
        if final_mime.split("/", 1)[0] in {"audio", "video"}:
            _wait_for_media_tokens(
                client,
                api_base=api_base,
                auth_header=auth_header,
                file_uri=file_id,
                mime_type=final_mime,
                sleep_func=sleep_func,
                poll_interval_seconds=poll_interval_seconds,
                max_poll_attempts=probe_poll_attempts,
            )

    return UploadedFileRef(
        file_id=file_id,
        filename=file_path.name,
        mime_type=final_mime,
        local_path=str(file_path),
    )


from .exchange_metadata import llm_exchange_metadata  # re-export for callers/tests
