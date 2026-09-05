"""Runtime client helpers for role-based LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import json
import mimetypes
from pathlib import Path
import random
import re
import sys
import threading
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

from finesub.reporting import current_reporter
from .http import llm_http_client
from .routing.config import (
    DEFAULT_LIMITS,
    ROLE_DEFAULT_TASK_GROUP,
    CapabilityTier,
    LLMRole,
    ModelEndpoint,
    RoleModelConfig,
    default_role_configs,
    role_config_for,
)
from .routing.capabilities import endpoint_supports
from .routing.execution_policy import normalized_tier
from .routing.model_catalog import thinking_value_for
from .routing.model_routes import CONVERSATIONAL_BACKEND
from .routing.profiles import VIDEO_SAMPLE_FPS
from .exchange_metadata import AGENT_SESSION_USAGE_FILENAME
from .media_upload import UploadedFileRef, upload_gemini_file
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
    #: The `max_tokens` this call actually asked for -- the answering
    #: candidate's own ceiling once callers stopped rationing it (2026-09-04).
    #: Anything comparing the reply against "the cap" has to use this, not the
    #: planning reserve, or a healthy long answer reads as truncated.
    requested_output_tokens: int = 0
    route_decision: Mapping[str, Any] = field(default_factory=dict)
    # Admission gate D, answer C (docs/llm_local_agent.md §7): a call that
    # depended on implicit provider history -- it resumed a conversation whose
    # earlier turns are not part of any hashable input -- must not seed a
    # reusable L1 checkpoint. Its output is as usable as any other; only the
    # "replay this exact call from its hash" claim is void, so commit sites
    # skip the store and a resume re-sends that one call instead.
    resumable: bool = True
    # Set only by the task-runtime agent path (docs/llm_agent_tool_protocol.md
    # §1), where tier-1 repairs happen inside one call: how many repair
    # rounds the runtime ran, and whether it gave up -- in which case
    # ``content`` is the last rejected output and the caller's own tier-1
    # budget for this chain is spent.
    agent_repair_rounds: int = 0
    repair_exhausted: bool = False


def _agent_tool_documents(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Split the harness's messages into the two documents a tool session reads.

    System messages are the protocol (the output contract); everything else is
    the payload (the material to process). Both roads that read these
    documents are text-only -- the harness MCP server cannot serve a media
    part through `read_context`, and a person's own agent is handed files it
    reads itself -- so a call carrying media belongs on the capsule
    transport, and this is where that is enforced rather than left to the
    catalog's capability columns being right.
    """

    from .agent.agent_transports import AgentRuntimeCallError

    def text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, Mapping):
            if isinstance(content.get("text"), str) and not any(
                key in content for key in ("file_path", "local_path", "file_id", "inline_data")
            ):
                return str(content["text"])
            raise AgentRuntimeCallError(
                "the agent task protocol is text-only (harness MCP server and "
                "`finesub agent-join` alike); this call carries a media part, "
                "which belongs on the capsule transport"
            )
        if isinstance(content, Sequence):
            return "\n".join(text_of(item) for item in content)
        raise AgentRuntimeCallError(f"unsupported message content: {type(content).__name__}")

    protocol: List[str] = []
    payload: List[str] = []
    for message in messages:
        role = str(message.get("role") or "user")
        text = text_of(message.get("content"))
        if role == "system":
            protocol.append(text)
        elif role == "user":
            payload.append(text)
        else:
            payload.append(f"<{role}>\n{text}\n</{role}>")
    return "\n\n".join(protocol).strip(), "\n\n".join(payload).strip()


def _agent_task_inputs(
    messages: Sequence[Mapping[str, Any]],
    *,
    validator_spec: Mapping[str, Any] | None,
    variant: str,
    capability_tier: CapabilityTier,
    max_repair_attempts: int,
    call_kwargs: Mapping[str, Any],
    retrieval: str,
) -> Dict[str, Any]:
    """What one harness call becomes as a runtime task, whichever session
    form carries it (a single-task tool session or a pseudo-conversational
    session): documents, input hash, validator, metadata, retrieval mode."""

    import hashlib

    from .agent.agent_transports import AgentRuntimeCallError
    from .agent.agent_validators import runtime_validators

    validator_id = str((validator_spec or {}).get("id") or "accept")
    try:
        validators = runtime_validators(validator_id)
    except KeyError as exc:
        raise AgentRuntimeCallError(str(exc)) from exc
    max_repairs = max(0, int(max_repair_attempts))
    session_type = str(call_kwargs.get("task") or "session")
    # The kb extras ride the manifest metadata, never the driver kwargs:
    # `driver.run` has an explicit keyword signature and no use for them.
    run_kwargs = {
        key: value
        for key, value in call_kwargs.items()
        if key
        not in {
            "previous_output",
            "validation_errors",
            "kb_handle_bindings",
            "kb_tools",
            "knowledge_root",
            "knowledge_identity",
            "kb_signal_task",
            "kb_signal_window",
        }
    }
    # An argument of its own, never a driver kwarg: `driver.run` has an
    # explicit keyword signature (the capsule transport spreads `call_kwargs`
    # straight into it) and takes only the derived `native_search`.
    retrieval = str(retrieval or "none")
    if retrieval not in {"none", "local", "native"}:
        raise AgentRuntimeCallError(f"Unknown retrieval mode: {retrieval!r}")
    serialized = json.dumps(
        # `retrieval` is hashed although it is not a driver argument: `local`
        # and `none` share a `native_search` of False and differ only in
        # whether the harness offers its web tools, so leaving it out would
        # let two calls with different tool surfaces share one identity.
        {"messages": list(messages), "call": run_kwargs, "retrieval": retrieval},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    protocol_text, payload_text = _agent_tool_documents(messages)
    knowledge_binding = _agent_knowledge_binding(call_kwargs)
    metadata = {
        "profile_id": str(call_kwargs.get("profile_id") or ""),
        "validator": dict((validator_spec or {}).get("params") or {}),
        "variant": variant or "",
        "capability_tier": capability_tier.value,
        "max_repair_attempts": max_repairs,
    }
    if knowledge_binding is not None:
        metadata["knowledge_identity"] = knowledge_binding["identity"]
        metadata["kb_root"] = knowledge_binding["root"]
        metadata["kb_tools"] = knowledge_binding["entitlement"]
        # Signal identity for the exposure ledger (plan §5.1): the runtime's
        # own task_id is a constant ("call") on per-call assignments, so the
        # caller names the run task and window its events should join with.
        for key in ("kb_signal_task", "kb_signal_window"):
            value = str(call_kwargs.get(key) or "")
            if value:
                metadata[key] = value
    else:
        explicit_identity = str(call_kwargs.get("knowledge_identity") or "")
        if explicit_identity:
            metadata["knowledge_identity"] = explicit_identity
    handle_bindings = call_kwargs.get("kb_handle_bindings")
    if handle_bindings:
        metadata["kb_handle_bindings"] = [dict(binding) for binding in handle_bindings]
    return {
        "validator_id": validator_id,
        "validators": validators,
        "max_repairs": max_repairs,
        "session_type": session_type,
        "run_kwargs": run_kwargs,
        "input_hash": "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "metadata": metadata,
        "knowledge": knowledge_binding,
        "protocol_text": protocol_text,
        "payload_text": payload_text,
        # The switch's own three states, passed through rather than squeezed
        # into a boolean: `local` is harness-executed retrieval (the proxied
        # web tools, budgeted by the runtime ledger), `native` is the
        # provider's own search tool, `none` is neither. Collapsing them here
        # -- every truthy `native_search` became `local` -- is what made
        # `retrieval=native` run on the harness proxy on every agent call,
        # against what routing/profiles.py defines the switch to mean.
        "retrieval_mode": retrieval,
    }


def _agent_knowledge_binding(call_kwargs: Mapping[str, Any]) -> Dict[str, Any] | None:
    """Tool-session knowledge binding (knowledge-node plan §4.3, 4b/4c).

    Authorization is the caller's explicit grant — ``kb_tools`` = ``read`` or
    ``propose`` (the §4.3 matrix, stamped into the manifest for the server to
    check). The generation pin only supplies the *default* root/revision for
    grants that name neither; a pin alone must never hand an unrelated task
    (a search judge) the tools and the gate. A standalone knowledge update
    (no pin: the module CLI's own runs are pinned, but reference_ingest and
    direct callers are not) passes ``knowledge_root`` + ``knowledge_identity``
    explicitly and gets the full binding. Conversational tasks never get the
    block: their control protocol has no kb tools yet.
    """

    import hashlib

    from .knowledge.base import (
        active_generation_pins,
        kb_index_block_text,
        knowledge_root_path,
    )

    entitlement = str(call_kwargs.get("kb_tools") or "")
    if entitlement not in ("read", "propose"):
        return None
    root = str(call_kwargs.get("knowledge_root") or "")
    identity = str(call_kwargs.get("knowledge_identity") or "")
    if root:
        root = str(knowledge_root_path(root))
        if not identity.startswith("rev:"):
            return None  # an explicit root must come with its revision
    else:
        pins = active_generation_pins()
        if len(pins) != 1:
            return None
        root, pin_rev = next(iter(pins.items()))
        if not identity.startswith("rev:"):
            identity = f"rev:{pin_rev}"
    rev = int(identity.split(":")[1])
    text = kb_index_block_text(root, rev)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {
        "root": root,
        "identity": f"rev:{rev}",
        "entitlement": entitlement,
        "block": {"kind": "kb_index", "ref": "kb_index", "digest": digest},
    }


def agent_usage_payload(usage: Mapping[str, Any] | None) -> Dict[str, Any]:
    """A local agent's usage in the OpenAI-ish shape the reports read.

    One conversion for both books: the per-call `raw_response` a window
    writes, and the per-session totals a pseudo-conversational run reports
    once its CLI has left (`write_agent_session_usage`).
    """

    local = dict(usage or {})

    def _int(*keys: str) -> int:
        for key in keys:
            value = local.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return int(value)
        return 0

    input_tokens = _int("input_tokens")
    output_tokens = _int("output_tokens")
    return {
        "prompt_tokens": input_tokens,
        "prompt_tokens_details": {"cached_tokens": _int("cached_input_tokens")},
        "completion_tokens": output_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": _int("reasoning_output_tokens", "reasoning_tokens")
        },
        "total_tokens": _int("total_tokens") or input_tokens + output_tokens,
    }


def write_agent_session_usage(artifact_dir: Path | str, registry: Any) -> Path | None:
    """Book a run's pseudo-conversational sessions where the report reads them.

    Usage is metered per CLI invocation, not per task, so the per-window
    exchange records of such a run carry no tokens at all (docs
    /llm_local_agent.md §12.1.3). The session totals are written here once
    the registry has closed -- which is when a session's CLI has actually
    left -- and the task report adds them to the per-provider table without
    touching the per-window call counts.
    """

    rows = list(registry.usage_rows())
    root = Path(artifact_dir).expanduser().resolve()
    path = root / AGENT_SESSION_USAGE_FILENAME
    if not rows:
        # Nothing to book, and an artifact directory is reused across runs:
        # leaving the last one's file behind would have the report count
        # tokens this run never spent (switching a cell off `pseudo` does
        # exactly that).
        path.unlink(missing_ok=True)
        return None
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "sessions": [
            {
                "provider_tier": str(row.get("provider_tier") or ""),
                "model": str(row.get("model") or ""),
                "lane": int(row.get("lane") or 0),
                "mode": str(row.get("mode") or ""),
                "label": str(row.get("label") or ""),
                "usage": extract_token_distribution(
                    {"usage": agent_usage_payload(row.get("usage"))}
                ),
            }
            for row in rows
        ]
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


# How many fresh CLI sessions a tool-protocol call may start after the agent
# left without submitting (docs/llm_agent_tool_protocol.md §4: a premature
# stop is a transport fault, not a replacement). Small: a second silent exit
# is the route chain's problem.
AGENT_PREMATURE_STOP_RETRIES = 1


def _harness_request_id(session_id: str, operation: str) -> str:
    """``H(caller_session_id, caller_sequence)`` for the harness itself.

    The harness makes each of its runtime calls once per CLI session, so the
    session id plus the operation name *is* the sequence: deterministic, so a
    retry of the same logical call replays, and no UUID is minted (docs §2).
    """

    import hashlib

    return "harness-" + hashlib.sha256(f"{session_id}:{operation}".encode("utf-8")).hexdigest()[:32]


def _reset_agent_context(
    runtime: Any, assignment_id: str, record: Mapping[str, Any], *, session_id: str, reason: str
) -> None:
    """A first premature stop: same lease, new context (docs §0-3)."""

    from .agent.agent_task_runtime import StaleLeaseError

    if not record.get("lease_owner"):
        return
    try:
        runtime.reset_conversation(
            assignment_id=assignment_id,
            task_id=str(record["task_id"]),
            worker_id=str(record["lease_owner"]),
            lease_generation=int(record["lease_generation"]),
            request_id=_harness_request_id(session_id, "reset"),
            reason=reason,
        )
    except StaleLeaseError:
        # The lease went away underneath (TTL): the next claim starts fresh anyway.
        return


def _retire_agent_task(
    runtime: Any, assignment_id: str, record: Mapping[str, Any], *, session_id: str, reason: str
) -> None:
    """Retire the task a finished CLI session left leased.

    The harness acts for the process that is gone -- the one exception to
    "the harness holds no lease" (docs §0-1). A lease already gone (the
    session submitted and was accepted, or the TTL reclaimed it) is nothing
    to retire; any other runtime error is real and propagates.
    """

    from .agent.agent_task_runtime import StaleLeaseError

    if not record.get("lease_owner"):
        return
    try:
        runtime.retire_task(
            assignment_id=assignment_id,
            task_id=str(record["task_id"]),
            worker_id=str(record["lease_owner"]),
            lease_generation=int(record["lease_generation"]),
            request_id=_harness_request_id(session_id, "retire"),
            reason=reason,
        )
    except StaleLeaseError:
        return


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
        # The readiness grade (missing / broken / unusable CLI) is what turns
        # "provider_disabled" into an answer to "why did agy not run today".
        if item.get("detail"):
            detail = f"{detail} ({item['detail']})"
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
        router: ModelRouter | None = None,
        execution_settings: ExecutionSettings | None = None,
        local_agent_driver: LocalAgentDriver | None = None,
        # (provider_tier, model): the tier is what picks the CLI, so a
        # factory that only saw the model could not honour the same rule
        # the default registry does.
        local_agent_driver_factory: (
            Callable[[str, str], LocalAgentDriver] | None
        ) = None,
        # Where agent tool sessions put their per-call assignments. The
        # default derives it from the driver's episode domain; tests inject a
        # temp directory because their fake drivers have no domain.
        agent_assignment_root: Path | None = None,
    ) -> None:
        # Explicitly injected configs (test fixtures pinning stub endpoints)
        # win over cell resolution even when a task group is passed.
        self._agent_assignment_root = agent_assignment_root
        self._injected_roles = frozenset(role_configs or ())
        self.role_configs = dict(role_configs or default_role_configs())
        self._cell_configs: Dict[tuple[str, str], RoleModelConfig] = {}
        self.test_profile = test_profile
        self.max_retries = int(max_retries)
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
        # Guards the dict above. Parallel lanes hold *disjoint* keys (a lane id
        # under `resume`, a window's chain key under `per-window`), so they
        # never contend for an entry -- but they do mutate one dict, and a
        # tier-2 retirement scans it. Without this, a replacement round
        # concurrent with another lane's first call raises "dictionary changed
        # size during iteration".
        self._agent_conversation_lock = threading.Lock()
        # Lane identity is issued by the run's `LaneOrdinalPool`
        # (`run_context`, task-parallelism plan W1) -- see `_agent_lane_id`.
        # Why the last readiness check refused each (tier, model): the route
        # decision trace copies it next to `provider_disabled`, so a report
        # can say which CLI was missing or broken that day.
        self._agent_readiness_detail: Dict[tuple[str, str], str] = {}
        # Pseudo-conversational sessions opened outside any run scope; see
        # `_agent_session_host` and `close`.
        self._own_agent_sessions: Any = None
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

    def _key_is_locked(self, key_id: str, *, provider_tier: str, model: str) -> bool:
        """Is this key daily-locked for this endpoint (so it cannot serve)?"""

        limiter = self.rate_limiter
        if limiter is None or not key_id or not model:
            return False
        return bool(
            limiter.is_daily_exhausted(
                ModelEndpoint(provider_tier, model), key_id=key_id
            )
        )

    @staticmethod
    def _key_is_in_tier(key_id: str, *, provider_tier: str) -> bool:
        """Whether the candidate tier still contains this canonical key id."""

        if not key_id or not provider_tier:
            return False
        from . import llm_runtime

        entries = llm_runtime._get_key_entries(
            provider_tier, llm_runtime._read_dotenv()
        )
        return any(entry.key_id == key_id for entry in entries)

    def _media_owner_can_serve(
        self, ref: UploadedFileRef, *, provider_tier: str, model: str
    ) -> bool:
        """Whether this candidate can read the ref without re-uploading it."""

        return bool(
            ref.api_provider_tier == provider_tier
            and self._key_is_in_tier(ref.api_key_id, provider_tier=provider_tier)
            and not self._key_is_locked(
                ref.api_key_id, provider_tier=provider_tier, model=model
            )
        )

    def _dispatchable_media_ref(
        self, ref: UploadedFileRef, *, provider_tier: str, model: str
    ) -> UploadedFileRef:
        """The media ref this candidate can actually read, uploading if needed.

        Three cases, and the third is the one that kept biting. A local-only
        ref uploads. A ref whose owning key can still serve passes through. A
        ref owned by a key this endpoint has locked is *unusable* -- the file
        lives in that key's project, and the call would 403 -- so it is
        re-uploaded under a key that can serve, and the caller pins to that.

        Recognising the third case is why `api_key_id` exists. It used to be
        invisible: the eagerly uploaded ref (`window_media_ref`, which cannot
        know the tier yet, let alone the model) already carries a `file_id`,
        so nothing here looked at it again -- every candidate 403'd in turn and
        the whole call died with the pool sitting there unlocked.
        """

        if not ref.file_id:
            if not ref.local_path:
                raise RuntimeError(
                    "Gemini media fallback has neither file_id nor local_path"
                )
            return self._uploaded_media_ref(
                ref, provider_tier=provider_tier, model=model
            )
        if self._media_owner_can_serve(
            ref, provider_tier=provider_tier, model=model
        ):
            return ref
        if not ref.local_path:
            # Nothing to re-upload from: let the call 403 and the chain move
            # on, which is strictly better than raising something new here.
            return ref
        self._forget_uploaded_media_ref(ref, provider_tier=provider_tier)
        return self._uploaded_media_ref(
            ref, provider_tier=provider_tier, model=model
        )

    def _uploaded_media_ref(
        self, ref: UploadedFileRef, *, provider_tier: str, model: str = ""
    ) -> UploadedFileRef:
        """A Files API object for a locally-referenced clip, uploaded once.

        Files expire on the service side (48h), and a run long enough to hit
        that would otherwise fail on a stale handle it cached itself, so a
        rejected reuse falls back to uploading again rather than to an error.

        The uploading key is recorded on the returned ref, and the call pins to
        exactly that (`_dispatchable_media_ref` -> ``pin_key_id``), so the two
        sides no longer derive it separately -- see the note on
        ``UploadedFileRef.api_key_id``. The cache is re-checked on every hit
        for the same reason: it is keyed by (tier, file), while a lock is per
        (tier, model, key), so an entry can be good for one model and
        unreachable for the next.
        """

        if not ref.local_path:
            raise RuntimeError(
                "Gemini media fallback has neither file_id nor local_path"
            )
        key = (provider_tier, str(Path(ref.local_path).resolve()))
        cached = self._remote_media_refs.get(key)
        if cached is not None and self._media_owner_can_serve(
            cached, provider_tier=provider_tier, model=model
        ):
            return cached
        if cached is not None:
            # The cache is keyed by (tier, file) but a lock is per (tier,
            # model, key), so a cached object can be fine for one model and
            # unreachable for the next. Serving it anyway is a guaranteed 403.
            self._remote_media_refs.pop(key, None)
        upload_key = _first_gemini_api_key(
            provider_tier, rate_limiter=self.rate_limiter, model=model
        )
        remote = upload_gemini_file(ref.local_path, api_key=upload_key)
        # ``upload_gemini_file(api_key=...)`` only sees the secret, so it can
        # provide the anonymous hash but not a named pool entry's canonical
        # id. This caller has the tier and stamps the identity from that pool.
        remote = replace(
            remote,
            api_key_id=_canonical_gemini_key_id(provider_tier, upload_key),
            api_provider_tier=provider_tier,
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
                warn=lambda message: current_reporter().warning(
                    "agent-quota", message
                ),
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
        CLI that happens to be installed. Refusals are graded and warned
        about by `driver_readiness` (docs/llm_local_agent.md §11); the
        detail is kept for the route decision trace.
        """

        from .agent.local_agent import driver_readiness

        key = (normalized_tier(provider_tier), model)
        try:
            driver = self._local_driver_for_model(model, provider_tier)
        except Exception as exc:  # noqa: BLE001 -- an unroutable tier is a refusal, not a crash
            self._agent_readiness_detail[key] = f"no driver: {exc}"
            return False
        ready, detail = driver_readiness(driver, native_search=native_search)
        if ready:
            self._agent_readiness_detail.pop(key, None)
        else:
            self._agent_readiness_detail[key] = detail
        return ready

    def _run_local_agent(
        self,
        driver: "LocalAgentDriver",
        messages: Sequence[Mapping[str, Any]],
        *,
        repair_session_key: str,
        provider_tier: str,
        model: str,
        session_mode: str = "",
        **call_kwargs: Any,
    ) -> tuple[Any, List[Mapping[str, Any]], bool]:
        """Run one agent call under the cell's session mode (four tiers).

        ``session_mode`` is the cell's ``agent_session_mode`` knob
        (docs/llm_local_agent.md §12.1): ``api`` never reuses a conversation,
        ``per-window`` (the default) resumes one window's repair chain,
        ``resume`` carries one conversation across the run's windows -- per
        worker lane (`_conversation_key`), so each parallel lane owns its own
        conversation and two lanes never race one handle. Any mode degrades to
        the ``api`` behaviour when the driver's probe lacks
        ``supports_session_reuse``: an ``assignment`` scope on such a driver
        fails before the spawn rather than degrading, so reuse is opt-in per
        driver.

        The per-window tier is why this exists at all: a repair used to mean a
        *fresh* agent process handed its own previous answer as plain text.
        Codex and Claude accept that; agy declines it and retried blind.
        Keeping the handle for the length of one window's attempt chain makes
        the repair a follow-up turn instead, for every backend
        (docs/llm_followups.md, "validation 失败改为修复轮").

        This is the narrow path: what is left to it since the transport
        became a function of the session tier is `api`, `resume` and every
        call carrying media (docs/llm_local_agent.md §12.1). A tool session
        does the same with durable state instead of this dict, which is a
        conversation cache and, under gate D's answer C, *only* a performance
        cache -- never identity.

        Returns the result, any execution attempts that were discarded on the
        way to it (so a rebuilt conversation still shows up as two spawns), and
        whether the call inherited a live handle -- i.e. depended on implicit
        provider history, which makes it non-resumable for L1 (gate D answer
        C, docs/llm_local_agent.md §7).
        """

        from .agent.local_agent import (
            LocalAgentError,
            LocalAgentPolicyViolationError,
            LocalAgentQuotaError,
        )
        from .routing.execution_policy import normalized_tier

        # An empty mode (injected test configs) means the routing default.
        conversation_key = self._conversation_key(session_mode, repair_session_key)
        previous_output = str(call_kwargs.get("previous_output") or "")
        errors = tuple(call_kwargs.get("validation_errors") or ())
        reuse = False
        if conversation_key:
            try:
                reuse = bool(driver.probe().supports_session_reuse)
            except Exception:
                reuse = False
        if not reuse:
            return driver.run(messages, **call_kwargs), [], False
        # Keyed like `_local_agent_drivers`, and for the same reason: a session
        # id belongs to the CLI that issued it, and two provider tiers sharing
        # a model id do not share a driver. Keying on the chain alone handed a
        # later attempt -- which re-routes freely, and may land on another
        # vendor or the other tier -- a conversation it does not own.
        chain = (normalized_tier(provider_tier), model, conversation_key)
        # A tier-2 replacement retires this key's conversation for *every*
        # candidate, not just this one, so that happens in `complete` before
        # the candidate loop -- see `_retire_agent_conversations`.
        if session_mode != "resume" and not previous_output and not errors:
            # First attempt of a per-window chain never inherits a handle: the
            # key is reused across windows of a run, and a stale conversation
            # would answer with the previous window still in context. Resume
            # mode wants exactly that inheritance, and its dict entry starts
            # empty anyway (the cache lives one client instance = one run).
            with self._agent_conversation_lock:
                self._agent_repair_conversations.pop(chain, None)
        with self._agent_conversation_lock:
            handle = self._agent_repair_conversations.get(chain, "")
        inherited_history = bool(handle)
        reuse_kwargs = dict(
            call_kwargs,
            session_scope="assignment",
            conversation_key=conversation_key,
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
            with self._agent_conversation_lock:
                self._agent_repair_conversations.pop(chain, None)
            inherited_history = False
            result = driver.run(messages, conversation_handle="", **reuse_kwargs)
        if getattr(result, "conversation_handle", ""):
            # Never held across `driver.run`: the lock protects the dict, not
            # the read-run-write sequence. Two calls racing one chain key would
            # be two lanes sharing a conversation, which the key scheme rules
            # out by construction.
            with self._agent_conversation_lock:
                self._agent_repair_conversations[chain] = result.conversation_handle
        return result, discarded, inherited_history

    def _agent_runtime_root(self, driver: "LocalAgentDriver") -> Path:
        """Where this client's task-runtime assignments live.

        Next to the driver's capsules by default -- the same domain the
        episode evidence already uses, so relocation and cleanup see both.
        """

        from .agent.agent_transports import AgentRuntimeCallError

        if self._agent_assignment_root is not None:
            return Path(self._agent_assignment_root)
        capsules = getattr(driver, "capsules", None)
        if capsules is None:
            raise AgentRuntimeCallError(
                "an agent tool session needs a driver with an episode domain "
                "or an explicit agent_assignment_root"
            )
        return Path(capsules.resolve_location().parent) / "assignments"

    def _run_agent_tool_call(
        self,
        driver: "LocalAgentDriver",
        messages: Sequence[Mapping[str, Any]],
        *,
        validator_spec: Mapping[str, Any] | None,
        variant: str,
        capability_tier: CapabilityTier,
        max_repair_attempts: int,
        retrieval: str = "none",
        **call_kwargs: Any,
    ) -> tuple[Any, List[Mapping[str, Any]], bool, int, bool]:
        """Run one agent call as a tool session on a single-task assignment.

        docs/llm_agent_tool_protocol.md §1: the agent takes the task, reads
        the protocol and payload through the harness MCP server and submits;
        the capsule carries only the bootstrap. The runtime owns the tier-1
        repair loop (``max_repair_attempts`` distinct rejected answers, each
        ``submit`` judged by the validator ``validator_spec`` names) and the
        accepted artifact is the answer. Tier 2 -- a replacement on a fresh
        conversation -- stays the caller's: a new `complete`, re-routed, with
        a new assignment. One CLI invocation is one task scope.

        Returns the result, earlier sessions' execution attempts (premature
        stops), whether the output inherited hidden history (never, here),
        how many repair rounds ran, and whether the budget ran out -- in
        which case the result carries the last *rejected* output.
        """

        import uuid

        from finesub_bootstrap.fsops import remove_tree

        from .agent.agent_task_runtime import (
            AgentTaskRuntime,
            AgentTaskSpec,
            lease_ttl_for,
        )

        inputs = _agent_task_inputs(
            messages,
            validator_spec=validator_spec,
            variant=variant,
            capability_tier=capability_tier,
            max_repair_attempts=max_repair_attempts,
            call_kwargs=call_kwargs,
            retrieval=retrieval,
        )
        validator_id = inputs["validator_id"]
        validators = inputs["validators"]
        max_repairs = inputs["max_repairs"]
        session_type = inputs["session_type"]
        run_kwargs = inputs["run_kwargs"]
        input_hash = inputs["input_hash"]
        metadata = inputs["metadata"]
        retrieval_mode = inputs["retrieval_mode"]
        assignment_id = f"call-{uuid.uuid4().hex}"
        protocol_documents = {session_type: inputs["protocol_text"]}
        context_documents = {"payload": inputs["payload_text"]}
        knowledge_binding = inputs["knowledge"]
        required_blocks = (
            {"kind": "protocol", "digest": "@protocol"},
            {"kind": "payload", "digest": "@context"},
            *((dict(knowledge_binding["block"]),) if knowledge_binding else ()),
        )
        root = self._agent_runtime_root(driver) / assignment_id
        runtime = AgentTaskRuntime.start_assignment(
            root,
            assignment_id=assignment_id,
            worker_goal="answer the harness call",
            tasks=[
                AgentTaskSpec(
                    task_id="call",
                    session_type=session_type,
                    input_hash=input_hash,
                    goal="answer the harness call",
                    validator_id=validator_id,
                    protocol_key=session_type,
                    context_key="payload",
                    retrieval_mode=retrieval_mode,
                    metadata=metadata,
                    required_blocks=required_blocks,
                )
            ],
            session_scope="task",
            protocol_documents=protocol_documents,
            context_documents=context_documents,
            execution_identity=dict(self.execution_identity),
            validators=validators,
            # Our deadline is this driver's, plus the margin. Derived rather
            # than left at the module default so the two can never drift --
            # `_assert_lease_outlives_one_call` is what used to catch the
            # drift, and now has nothing to catch.
            lease_ttl_seconds=lease_ttl_for(
                float(getattr(getattr(driver, "config", None), "timeout_seconds", 0) or 0)
                or self.execution_settings.local_agent_timeout_seconds
            ),
        )
        from .agent.agent_mcp_server import KB_TOOL_NAMES, TOOL_NAMES, WEB_TOOL_NAMES

        return self._run_agent_tool_session(
            runtime,
            driver,
            root=root,
            assignment_id=assignment_id,
            input_hash=input_hash,
            max_repairs=max_repairs,
            run_kwargs=run_kwargs,
            remove_tree=remove_tree,
            retrieval_mode=retrieval_mode,
            tools=[
                *TOOL_NAMES,
                *(WEB_TOOL_NAMES if retrieval_mode == "local" else ()),
                *(KB_TOOL_NAMES if knowledge_binding else ()),
            ],
            knowledge_root=knowledge_binding["root"] if knowledge_binding else "",
        )

    def _agent_session_host(
        self,
        driver: "LocalAgentDriver",
        *,
        provider_tier: str,
        model: str,
        native_search: bool,
    ) -> Any:
        """The pseudo-conversational session serving this (tier, model, lane).

        Keyed for the length of the run (docs/llm_followups.md, second-round
        decision 4): every role or task group bound to the same agent model
        shares one CLI session; parallel lanes get one each. The run scope
        (`agent_session_scope`) owns the registry; a client outside any scope
        gets a private one that is closed at interpreter exit at the latest.
        """

        import uuid

        from .agent.agent_session_host import AgentSessionHost

        registry = self._session_registry()
        key = (normalized_tier(provider_tier), model, self._agent_lane_id(), "pseudo-conversational")

        def build() -> AgentSessionHost:
            root = self._agent_runtime_root(driver) / f"session-{uuid.uuid4().hex[:12]}"
            return AgentSessionHost(
                driver,
                root=root,
                execution_identity=dict(self.execution_identity),
                task_timeout_seconds=float(
                    getattr(getattr(driver, "config", None), "timeout_seconds", 0)
                    or self.execution_settings.local_agent_timeout_seconds
                ),
                label=f"{key[0].lower()}/{model}/lane{key[2]}",
                native_search=native_search,
            )

        # One CLI per key at a time. A session's built-in tools are fixed at
        # launch, so a call needing the other retrieval entitlement replaces
        # it rather than running beside it: keying the two apart would leave
        # both holding one of the driver's `max_parallel` slots for the length
        # of the run, and at `max_parallel=1` the second could never start
        # while the first waits for the run to end.
        return registry.host_for(
            key, build, reuse_if=lambda host: host.native_search == native_search
        )

    def close(self) -> None:
        """End the agent sessions this client opened outside a run scope."""

        registry = self._own_agent_sessions
        self._own_agent_sessions = None
        if registry is not None:
            registry.close()

    def _run_agent_session_call(
        self,
        driver: "LocalAgentDriver",
        messages: Sequence[Mapping[str, Any]],
        *,
        provider_tier: str,
        model: str,
        validator_spec: Mapping[str, Any] | None,
        variant: str,
        capability_tier: CapabilityTier,
        max_repair_attempts: int,
        fresh_session: bool,
        retrieval: str = "none",
        **call_kwargs: Any,
    ) -> tuple[Any, List[Mapping[str, Any]], bool, int, bool]:
        """One harness call as one task of the run's pseudo-conversational
        session (docs/llm_local_agent.md §12.1.3)."""

        inputs = _agent_task_inputs(
            messages,
            validator_spec=validator_spec,
            variant=variant,
            capability_tier=capability_tier,
            max_repair_attempts=max_repair_attempts,
            call_kwargs=call_kwargs,
            retrieval=retrieval,
        )
        host = self._agent_session_host(
            driver,
            provider_tier=provider_tier,
            model=model,
            native_search=inputs["retrieval_mode"] == "native",
        )
        return host.run_task(
            session_type=inputs["session_type"],
            input_hash=inputs["input_hash"],
            validator_id=inputs["validator_id"],
            metadata=inputs["metadata"],
            protocol_text=inputs["protocol_text"],
            payload_text=inputs["payload_text"],
            retrieval_mode=inputs["retrieval_mode"],
            # `native_search` rides along: one pseudo-conversational CLI serves
            # many tasks and can only be entitled at launch (docs §2), so the
            # session takes the mode of the task that starts it.
            run_kwargs=dict(inputs["run_kwargs"]),
            max_repairs=inputs["max_repairs"],
            fresh_session=fresh_session,
            extra_required_blocks=(
                (inputs["knowledge"]["block"],) if inputs["knowledge"] else ()
            ),
        )

    def _conversational_queue(self, target_id: str) -> Any:
        """The run's queue for a person's own agent, one per target."""

        from .agent.agent_paths import (
            conversational_assignment_parent,
            resolve_agent_episode_location,
        )
        from .agent.agent_session_host import ConversationalQueue

        registry = self._session_registry()
        key = ("CONVERSATIONAL", target_id, 0, "conversational")

        def build() -> ConversationalQueue:
            location = resolve_agent_episode_location()
            explicit = self._agent_assignment_root
            return ConversationalQueue(
                parent=(
                    Path(explicit)
                    if explicit is not None
                    else conversational_assignment_parent(location)
                ),
                activity_root=location.activity_root,
                execution_identity=dict(self.execution_identity),
                # How long one stretch of work may take -- the same number
                # every other agent transport answers to; the lease follows
                # from it. How long we wait for somebody to join is not a
                # setting and is not this (`CONVERSATIONAL_JOIN_WAIT_SECONDS`).
                call_timeout_seconds=float(
                    self.execution_settings.local_agent_timeout_seconds
                ),
                label=target_id,
            )

        return registry.host_for(key, build)

    def _run_conversational_call(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        target_id: str,
        validator_spec: Mapping[str, Any] | None,
        variant: str,
        capability_tier: CapabilityTier,
        max_repair_attempts: int,
        retrieval: str = "none",
        **call_kwargs: Any,
    ) -> tuple[Any, List[Mapping[str, Any]], bool, int, bool]:
        """One harness call as one task for a person's own agent
        (docs/llm_local_agent.md §12.1.4)."""

        inputs = _agent_task_inputs(
            messages,
            validator_spec=validator_spec,
            variant=variant,
            capability_tier=capability_tier,
            max_repair_attempts=max_repair_attempts,
            call_kwargs=call_kwargs,
            retrieval=retrieval,
        )
        protocol_text = inputs["protocol_text"]
        if str(inputs["session_type"]).startswith("correction"):
            from .prompts import conversational_correction_effort

            # Appended, not composed in: the shared contract is what every
            # backend answers to, and this note is about how much effort one
            # transport should spend on one column. Deliberately outside
            # `input_hash` too -- it changes no requirement, so a window
            # already committed must not be invalidated by it.
            protocol_text = (
                protocol_text.rstrip()
                + "\n\n"
                + conversational_correction_effort()
                + "\n"
            )
        return self._conversational_queue(target_id).run_task(
            session_type=inputs["session_type"],
            input_hash=inputs["input_hash"],
            validator_id=inputs["validator_id"],
            metadata=inputs["metadata"],
            protocol_text=protocol_text,
            payload_text=inputs["payload_text"],
            retrieval_mode=inputs["retrieval_mode"],
            max_repairs=inputs["max_repairs"],
        )

    def _run_agent_tool_session(
        self,
        runtime: Any,
        driver: "LocalAgentDriver",
        *,
        root: Path,
        assignment_id: str,
        input_hash: str,
        max_repairs: int,
        run_kwargs: Mapping[str, Any],
        remove_tree: Callable[[Path], None],
        retrieval_mode: str = "none",
        tools: Sequence[str] = (),
        knowledge_root: str = "",
    ) -> tuple[Any, List[Mapping[str, Any]], bool, int, bool]:
        """One CLI invocation that takes, reads and submits its task itself.

        The harness's part is small by design (docs/llm_agent_tool_protocol.md
        §4): declare the server, hand the agent the bootstrap, and
        when the process is gone read the runtime -- `accepted` is the only
        completion; the final assistant text carries no artifact. An agent
        that leaves without ever submitting is a premature stop: one fresh
        CLI on the same worker (the lease is renewed, not re-claimed) before
        the call fails to the route chain.
        """

        import copy
        from dataclasses import is_dataclass
        import os
        import sys
        import uuid

        from types import SimpleNamespace

        from .agent.agent_mcp_server import TOOL_NAMES
        from .agent.agent_session_host import (
            capsule_retained,
            fold_proxied_retrieval,
            write_audit_bundle,
        )
        from .agent.agent_transports import AgentRuntimeCallError
        from .agent.local_agent import LocalAgentError
        from .prompts import agent_tool_worker_bootstrap

        worker_id = "worker-1"
        tool_names = list(tools) or list(TOOL_NAMES)
        bootstrap = [
            {
                "role": "user",
                "content": agent_tool_worker_bootstrap(
                    assignment_id=assignment_id,
                    worker_id=worker_id,
                    native_search=retrieval_mode == "native",
                ),
            }
        ]
        attempts: List[Mapping[str, Any]] = []
        result: Any = None
        record: Mapping[str, Any] = {}
        for attempt in range(1 + AGENT_PREMATURE_STOP_RETRIES):
            session_id = uuid.uuid4().hex
            env = {
                "FINESUB_MCP_ROOT": str(root),
                "FINESUB_MCP_ASSIGNMENT": assignment_id,
                "FINESUB_MCP_WORKER": worker_id,
                "FINESUB_MCP_SESSION": session_id,
                "FINESUB_MCP_LOG": str(root / "control" / "mcp-frames.jsonl"),
            }
            if knowledge_root:
                env["FINESUB_MCP_KNOWLEDGE_ROOT"] = knowledge_root
            # The server is a Python process: it must find this package the
            # way the harness did. A packaged install has it on the
            # interpreter's path; a checkout carries it in PYTHONPATH -- made
            # absolute, because the CLI spawns the server in its own cwd.
            from .agent.agent_session_host import (
                absolute_pythonpath,
                mcp_block_files,
                mcp_page_chars,
            )


            if os.environ.get("PYTHONPATH"):
                env["PYTHONPATH"] = absolute_pythonpath(os.environ["PYTHONPATH"])
            env["FINESUB_MCP_PAGE_CHARS"] = os.environ.get("FINESUB_MCP_PAGE_CHARS", "") or str(
                mcp_page_chars(driver)
            )
            block_files = mcp_block_files(driver)
            if block_files:
                env["FINESUB_MCP_BLOCK_FILES"] = "1"
            mcp_server = {
                "command": sys.executable,
                "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
                "env": env,
                "tools": tool_names,
                "view_roots": [str(root)] if block_files else [],
            }
            driver_error: BaseException | None = None

            def task_accepted() -> bool:
                return (
                    runtime.task_record(assignment_id=assignment_id, task_id="call")["status"]
                    == "accepted"
                )

            try:
                result = driver.run(
                    bootstrap,
                    session_scope="task",
                    mcp_server=mcp_server,
                    completion=task_accepted,
                    **run_kwargs,
                )
            except LocalAgentError as exc:
                # The runtime, not the CLI's exit, says whether the task is
                # done (docs §0-3). A session that submitted and was accepted
                # and *then* tripped over something -- agy's hook denying a
                # stray native tool flips its whole result to ERROR -- still
                # produced the artifact; only a session that never got there
                # is the driver's failure to report.
                driver_error = exc
                failed_attempt = dict(
                    (getattr(exc, "_harness_execution_attempts", None) or [{}])[-1]
                )
                result = SimpleNamespace(
                    content="",
                    reported_model=str(failed_attempt.get("reported_model") or ""),
                    episode_id=str(failed_attempt.get("capsule_id") or ""),
                    execution_attempt=failed_attempt,
                    normalized_events=(),
                    usage=dict(failed_attempt.get("usage") or {}),
                    conversation_handle="",
                    turn_identity="",
                )
            record = runtime.task_record(assignment_id=assignment_id, task_id="call")
            if record["status"] == "accepted":
                if driver_error is not None:
                    current_reporter().warning(
                        "agent-session-error-after-accept",
                        f"agent session {session_id} was accepted, then failed: {driver_error}",
                    )
                break
            if driver_error is not None:
                # Not accepted and the driver failed: whatever lease the dead
                # CLI still holds is retired before the error goes up, so the
                # task is never left `leased` until the TTL notices -- and the
                # audit bundle is written with the state *after* that.
                _retire_agent_task(
                    runtime, assignment_id, record, session_id=session_id,
                    reason=f"driver error: {type(driver_error).__name__}",
                )
                record = runtime.task_record(assignment_id=assignment_id, task_id="call")
                write_audit_bundle(
                    runtime,
                    result=result,
                    root=root,
                    assignment_id=assignment_id,
                    worker_id=worker_id,
                    record=record,
                    accepted_text="",
                    error=f"{type(driver_error).__name__}: {driver_error}",
                )
                raise driver_error
            submitted = bool(record.get("last_candidate"))
            if submitted or attempt == AGENT_PREMATURE_STOP_RETRIES:
                break
            # Premature stop, first time (docs §0-3): the session keeps its
            # lease and generation -- this is a transport fault, not a
            # replacement -- but its context is gone, so the conversation is
            # reset with the lease held, which also clears the pull ledger.
            # The fresh CLI on the same worker id resumes the same lease and
            # has to be handed its blocks again.
            _reset_agent_context(
                runtime, assignment_id, record, session_id=session_id, reason="premature stop"
            )
            record = runtime.task_record(assignment_id=assignment_id, task_id="call")
            attempts.append(
                {**dict(result.execution_attempt), "premature_stop": True}
            )
            current_reporter().warning(
                "agent-premature-stop",
                f"agent session {session_id} ended without a submit; retrying once",
            )
        assert result is not None
        if record["status"] != "accepted":
            # Whatever is left -- an exhausted repair chain, the premature
            # cap -- is this session's end: retire it (second tier) so the
            # task is re-queued with no lease rather than `repairing` under a
            # lease nobody holds.
            _retire_agent_task(
                runtime, assignment_id, record, session_id=session_id, reason="session ended"
            )
            record = runtime.task_record(assignment_id=assignment_id, task_id="call")

        def with_content(text: str) -> Any:
            if is_dataclass(result):
                return replace(result, content=text)
            clone = copy.copy(result)
            clone.content = text
            return clone

        # Provenance: the searches of a tool session ran through the harness's
        # own proxy, so their URLs are in the runtime ledger and not in the
        # driver's events -- which is the only place downstream looks. Fold
        # them in before the result leaves, so a proxied round carries the same
        # evidence a native one does.
        result = fold_proxied_retrieval(
            runtime, result, assignment_id=assignment_id, task_id="call"
        )

        accepted_text = ""
        if record["status"] == "accepted" and record["accepted_artifact_ref"]:
            artifact = json.loads(runtime.read_artifact(record["accepted_artifact_ref"]))
            if artifact.get("input_hash") != input_hash:
                raise AgentRuntimeCallError("accepted artifact does not match the call")
            accepted_text = str(artifact.get("artifact") or "")
        audited = write_audit_bundle(
            runtime,
            result=result,
            root=root,
            assignment_id=assignment_id,
            worker_id=worker_id,
            record=record,
            accepted_text=accepted_text,
        )
        if accepted_text:
            # A clean run has no capsule left to be audited into -- the driver
            # prunes it, which is the retention rule the bundle follows -- so
            # "not written" only means "lost" when there was somewhere to
            # write. Keeping every successful call's root was the reverse of
            # what that rule says, and left this run's whole text on disk.
            if audited or not capsule_retained(result):
                try:
                    remove_tree(root)
                except OSError:
                    pass
            else:
                # The runtime root is now the only evidence of this session,
                # so it stays (docs §0-4).
                current_reporter().warning(
                    "agent-audit-bundle",
                    f"keeping assignment root {root} because its audit bundle was not written",
                )
            return with_content(accepted_text), attempts, False, 0, False
        last_candidate = record.get("last_candidate")
        if isinstance(last_candidate, str) and last_candidate:
            # The agent submitted, was refused, and left: the chain is spent
            # from the caller's point of view (tier 2 is its replacement).
            return with_content(last_candidate), attempts, False, max_repairs, True
        error = AgentRuntimeCallError(
            "agent session ended without an accepted submit "
            f"(task status {record['status']!r}; "
            f"{'; '.join(record['validation_errors']) or 'no submit seen'})"
        )
        setattr(error, "_harness_execution_attempts", [*attempts, dict(result.execution_attempt)])
        raise error

    def _assert_pseudo_session_possible(self, plan: Any) -> None:
        """At least one agent candidate of the chain can run a tool session."""

        from .agent.agent_transports import AgentRuntimeCallError

        seen: list[str] = []
        for candidate in plan.candidates:
            endpoint = candidate.endpoint
            if endpoint.backend != "local_agent":
                continue
            try:
                driver = self._local_driver_for_model(endpoint.api_model_id, endpoint.provider_tier)
                probe = driver.probe()
            except Exception as exc:  # noqa: BLE001 -- reported below
                seen.append(f"{candidate.target_id}: {type(exc).__name__}: {exc}")
                continue
            if getattr(probe, "supports_mcp_config", False):
                return
            seen.append(f"{candidate.target_id}: no per-invocation MCP support")
        raise AgentRuntimeCallError(
            "agent session mode 'pseudo-conversational' needs an agent target whose "
            "CLI takes a per-invocation MCP server; none in this chain does"
            + (f" ({'; '.join(seen)})" if seen else " (the chain has no agent target)")
        )

    def routes_to_conversational(
        self, role: "LLMRole", task_group: str = "", difficulty: str = ""
    ) -> bool:
        """Does this cell's bound group name a conversational target?

        Task-parallelism plan W6: a person's own agent serves one assignment
        queue -- fan-out just queues N tasks behind however many agents joined
        (usually one), same wall clock as serial but with the advice ledger
        stripped from the prompts. The correction stage asks this before
        honouring ``continuity=parallel``.
        """

        config = self._config_for(role, task_group, difficulty)
        if not config.model_group_id:
            return False
        routes = self.router.routes
        group = routes.model_groups.get(config.model_group_id)
        if group is None:
            return False
        return any(
            routes.targets[target_id].backend == "conversational_agent"
            for target_id in group.target_ids
        )

    def _session_registry(self) -> Any:
        """The run's session registry, or this client's private fallback."""

        from .agent.agent_session_host import current_registry, private_registry

        registry = current_registry()
        if registry is None:
            if self._own_agent_sessions is None:
                self._own_agent_sessions = private_registry()
            registry = self._own_agent_sessions
        return registry

    def _agent_lane_id(self) -> int:
        """The calling worker's lane ordinal within the current run.

        Issued by the run's `LaneOrdinalPool` (task-parallelism plan W1): the
        ordinal belongs to the run, not to a thread or a pool. A phase's pool
        workers lease 1..N through `run_context.bind_llm_worker` and hand the
        numbers back when the phase ends, so the next phase's fresh threads
        lease the same set and land on the same conversations and pseudo
        hosts. A thread nobody bound (the serial path, a bare client) leases
        on first use and keeps its ordinal -- lent to a pool while the thread
        parks on its futures, reacquired here on its next use.

        Deliberately not `threading.get_ident()` as *identity*: the OS
        recycles a thread id once its thread exits, and a run opens pools in
        sequence, so a recycled id would hand a later worker a finished
        worker's conversation. The ident is only the pool's lease-owner tag.
        """

        from .run_context import lane_ordinal_for_thread

        return lane_ordinal_for_thread(self._session_registry().lanes)

    def _conversation_key(self, session_mode: str, repair_session_key: str) -> str:
        """Which conversation this call belongs to, per the cell's session mode.

        ``resume`` keys on the worker lane: one conversation per lane for the
        whole run, shared by every harness session that lane executes. The
        conversation is LANE-scoped, not window-scoped: under parallel
        dispatch a lane serves whichever windows the pool hands it, so its
        history holds several windows' rounds, and a window's correction may
        ride a lane whose query history is another window's -- window->lane
        affinity is NOT guaranteed (reviewer 2026-08-30 P2-3; the dynamic
        window->lane binding is the same A/B noise plan §5 records). It never
        spans runs -- the cache lives on the client instance, and a correction
        run builds its own (`stages/correction/run.py`). Under
        `continuity=parallel` the two phases run in separate pools, but each
        phase leases the same lane ordinals from the run's pool
        (task-parallelism plan W1), so a lane's conversation does carry from
        the query pool into the correction pool within one client.
        """

        from .agent.agent_transports import session_scope_for_mode

        if session_scope_for_mode(session_mode or "per-window") != "assignment":
            return ""
        if session_mode == "resume":
            return f"resume-run:{self._agent_lane_id()}"
        return repair_session_key

    def _retire_agent_conversations(self, conversation_key: str) -> None:
        """Drop every cached handle for one conversation key.

        Tier 2 of the retry budget retires the session chain, and *which*
        candidate answers is decided per attempt: a replacement round may be
        answered by an API endpoint or by a second agent target, leaving the
        first agent's handle in the cache for a later repair round to inherit
        -- the degenerate conversation the replacement exists to escape. So
        the retirement is by key, across every (tier, model) that holds one.
        """

        if not conversation_key:
            return
        with self._agent_conversation_lock:
            for chain in [
                chain
                for chain in self._agent_repair_conversations
                if chain[2] == conversation_key
            ]:
                self._agent_repair_conversations.pop(chain, None)

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
        refused = "; ".join(
            f"{candidate.target_id}: {detail}"
            for candidate in plan.candidates
            for detail in (
                self._agent_readiness_detail.get(
                    (
                        normalized_tier(candidate.endpoint.provider_tier),
                        candidate.endpoint.api_model_id,
                    )
                ),
            )
            if detail
        )
        raise RuntimeError(
            f"No eligible target for {plan.task_group_id}/{plan.difficulty} "
            f"before auxiliary work (requires {wanted})"
            + (f" [agent CLIs refused: {refused}]" if refused else "")
        )

    def complete(
        self,
        role: LLMRole,
        messages: List[Dict[str, Any]] | TieredMessages,
        *,
        max_tokens: int | None = None,
        output_reserve: int | None = None,
        temperature: float = VALIDATION_BASE_TEMPERATURE,
        seed: int | None = None,
        file_ref: UploadedFileRef | None = None,
        fallback_audio_ref: Callable[[], UploadedFileRef | None] | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
        # The switch itself (`routing/profiles.py`): none / local / native.
        # It used to arrive as a `native_search` boolean, which cost the
        # difference between `local` and `none` and left the agent path
        # reconstructing a mode it could not know.
        retrieval: str = "none",
        task_group: str = "",
        difficulty: str = "",
        previous_output: str = "",
        validation_errors: Sequence[str] = (),
        # Names the chain of attempts answering one unit of work, so an agent
        # repair can resume the conversation that produced the output it is
        # repairing. Empty keeps the full-replay behaviour.
        repair_session_key: str = "",
        # Tier 2 of the two-tier retry budget (docs/llm_followups.md
        # "两档重试"): the caller declares this call a *replacement* -- the
        # previous session chain is presumed degenerate, so every cached
        # conversation for this call's key is retired before routing, whichever
        # candidate holds it.
        fresh_session: bool = False,
        # The caller's output validator, for backends that can run the tier-1
        # repair loop themselves (local agents on the tool-session
        # transport): ``{"id": <agent_validators id>,
        # "params": {...}}``, JSON-serializable so the process serving
        # `submit` -- the harness, or the MCP server the CLI spawned -- can
        # resolve the same function. API backends ignore it: their repairs
        # are the caller's loop, one `complete` per attempt, as before.
        validator_spec: Mapping[str, Any] | None = None,
        max_repair_attempts: int = 0,
        # Extra key/values folded into an agent-backed call's task kwargs
        # (JSON-serializable; API backends never see them). The knowledge
        # update passes its per-chunk `knowledge_identity` (working_rev),
        # `kb_validate` admission and the prompt's `kb_handle_bindings`
        # through here (knowledge-node plan §6.5, step 4c).
        agent_task_extras: Mapping[str, Any] | None = None,
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
        from .agent.agent_transports import agent_transport_for, session_scope_for_mode
        from .prompt_compose import compose_repair_turns

        config = self._config_for(role, task_group, difficulty)
        if config.agent_session_mode:
            # Validate the cell's agent session mode eagerly, before routing:
            # inside the candidate loop an unknown mode would read as one
            # failed candidate, and the call could silently fall through to an
            # API backend instead of saying the setting is wrong.
            session_scope_for_mode(config.agent_session_mode)
        if fresh_session:
            # Tier 2 retires the session chain *before* routing, because which
            # candidate answers is decided per attempt: retiring only inside
            # the local-agent branch would leave a handle behind whenever the
            # replacement round happens to be answered by an API endpoint or a
            # second agent target, and a later repair round could route back
            # into the degenerate conversation this replacement is escaping.
            self._retire_agent_conversations(
                self._conversation_key(
                    config.agent_session_mode, repair_session_key
                )
            )
        call_thinking_budget = (
            config.thinking_budget if thinking_budget is None else thinking_budget
        )
        call_thinking_level = (
            config.thinking_level if thinking_level is None else thinking_level
        )
        # The test profile pins one cheap target and buys no retrieval of any
        # kind, so it collapses the switch rather than only its native state.
        retrieval_requested = "none" if self.test_profile else str(retrieval or "none")
        native_search_requested = retrieval_requested == "native"
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
        if config.agent_session_mode == "pseudo-conversational" and file_ref is None:
            # The tier was set to get one run-long session; a chain that
            # cannot provide it must fail here, not slide into an API
            # candidate as if the tier were unset (docs §12.1.3).
            self._assert_pseudo_session_possible(plan)
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
                CONVERSATIONAL_BACKEND,
                "openai_compat",
                "anthropic",
            }:
                decision.update(decision="skipped", reason="backend_unavailable")
                continue
            # A conversational target is never "enabled" (nobody can probe a
            # person's agent); its calls are queued for that agent instead
            # (docs/llm_local_agent.md §12.1.4).
            if endpoint.backend != CONVERSATIONAL_BACKEND and not provider_enabled(
                candidate,
                agent_ready=self._local_agent_ready,
                native_search=native_search_requested,
            ):
                decision.update(decision="skipped", reason="provider_disabled")
                detail = self._agent_readiness_detail.get(
                    (normalized_tier(endpoint.provider_tier), endpoint.api_model_id)
                )
                if endpoint.backend == "local_agent" and detail:
                    decision["detail"] = detail
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
                    current_reporter().warning(
                        "media-downgraded",
                        f"target {candidate.target_id} 不支持视频，本会话对它按 "
                        "video->audio 阶梯降一级发送音频剪辑",
                        impact="这是安全网；正常配置应为该格配备有视频能力的模型",
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
            request_max_tokens, reserve = _output_budget(max_tokens, output_reserve, catalog_entry)
            if catalog_entry is not None:
                if reserve > catalog_entry.max_output_tokens:
                    decision.update(
                        decision="skipped",
                        reason="output_limit",
                        requested=reserve,
                        limit=catalog_entry.max_output_tokens,
                    )
                    continue
                over_input = estimated_input > catalog_entry.max_input_tokens
                # Same question, the other ceiling: on a single-pool provider a
                # request can clear `max_input_tokens` and still not leave room
                # for the answer. Dropping the repair context helps exactly as
                # much there, so both busted ceilings take the same way out --
                # checking only the first one would skip a candidate that the
                # blind retry could still have reached.
                over_context = (
                    estimated_input + reserve > catalog_entry.context_window
                )
                if (over_input or over_context) and repair_enabled:
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
                    decision["repair_context"] = (
                        "dropped_input_limit" if over_input else "dropped_context_limit"
                    )
                # Re-clamped against the input that is really sent -- see
                # `_clamp_request_to_context` for what the old ordering asked for.
                request_max_tokens = _clamp_request_to_context(
                    request_max_tokens, catalog_entry, estimated_input
                )
                if estimated_input > catalog_entry.max_input_tokens:
                    decision.update(
                        decision="skipped",
                        reason="input_limit",
                        estimated=estimated_input,
                        limit=catalog_entry.max_input_tokens,
                        capability_tier=tier.value,
                    )
                    continue
                # Single-pool providers (Codex, Anthropic) spend one budget on
                # both halves, so a request can clear each ceiling and still not
                # fit. Planning already reserves the answer -- the envelope is
                # `context_window - output_limit` -- but planning only knows the
                # group's minimum, and this candidate may be the tighter one.
                if estimated_input + reserve > catalog_entry.context_window:
                    decision.update(
                        decision="skipped",
                        reason="context_limit",
                        estimated=estimated_input + reserve,
                        limit=catalog_entry.context_window,
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
                dispatch_key_id = ""
                if endpoint.backend == "gemini_rest" and candidate_ref is not None:
                    remote_ref = self._dispatchable_media_ref(
                        candidate_ref,
                        provider_tier=endpoint.provider_tier,
                        model=endpoint.api_model_id,
                    )
                    dispatch_key_id = remote_ref.api_key_id
                    if remote_ref is not candidate_ref:
                        dispatch_messages = messages_for(
                            candidate_variant, remote_ref, repair=repair_in_messages
                        )
                if endpoint.backend in {"local_agent", CONVERSATIONAL_BACKEND}:
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
                        agent_call_kwargs = dict(
                            task=plan.task,
                            native_search=native_search_requested,
                            profile_id=(
                                f"policy={plan.policy_id};"
                                f"target={candidate.target_id};"
                                f"route={plan.routing_identity_digest}"
                            ),
                            reasoning_effort=agent_thinking,
                            previous_output=(
                                previous_output if repair_enabled else ""
                            ),
                            validation_errors=(
                                repair_errors if repair_enabled else ()
                            ),
                        )
                        agent_repair_rounds = 0
                        repair_exhausted = False
                        # transport = f(tier, driver capability, media) -- docs
                        # /llm_local_agent.md §12.1. The probe is cached: the
                        # readiness pre-filter already ran it for this candidate.
                        try:
                            agent_probe = local_driver.probe()
                        except Exception:  # noqa: BLE001 -- readiness already graded it
                            agent_probe = None
                        agent_transport = agent_transport_for(
                            config.agent_session_mode,
                            agent_probe,
                            has_media=candidate_ref is not None,
                            driver_id=str(getattr(local_driver, "driver_id", "")),
                        )
                        decision["agent_transport"] = agent_transport
                        if (
                            agent_transport == "tool-session"
                            and config.agent_session_mode == "pseudo-conversational"
                        ):
                            (
                                agent_result,
                                rebuilt_from,
                                inherited_history,
                                agent_repair_rounds,
                                repair_exhausted,
                            ) = self._run_agent_session_call(
                                local_driver,
                                dispatch_messages,
                                provider_tier=endpoint.provider_tier,
                                model=endpoint.api_model_id,
                                validator_spec=validator_spec,
                                variant=candidate_variant,
                                capability_tier=tier,
                                max_repair_attempts=max_repair_attempts,
                                fresh_session=fresh_session,
                                retrieval=retrieval_requested,
                                # Tool-session only: the extras become manifest
                                # metadata; a capsule driver's explicit `run`
                                # signature must never see them.
                                **dict(agent_task_extras or {}),
                                **agent_call_kwargs,
                            )
                        elif agent_transport == "tool-session":
                            (
                                agent_result,
                                rebuilt_from,
                                inherited_history,
                                agent_repair_rounds,
                                repair_exhausted,
                            ) = self._run_agent_tool_call(
                                local_driver,
                                dispatch_messages,
                                validator_spec=validator_spec,
                                variant=candidate_variant,
                                capability_tier=tier,
                                max_repair_attempts=max_repair_attempts,
                                retrieval=retrieval_requested,
                                **dict(agent_task_extras or {}),
                                **agent_call_kwargs,
                            )
                        else:
                            agent_result, rebuilt_from, inherited_history = (
                                self._run_local_agent(
                                    local_driver,
                                    dispatch_messages,
                                    repair_session_key=repair_session_key,
                                    provider_tier=endpoint.provider_tier,
                                    model=endpoint.api_model_id,
                                    session_mode=config.agent_session_mode,
                                    **agent_call_kwargs,
                                )
                            )
                    else:
                        # A person's agent: no driver, no thinking knob, no
                        # transport choice -- the queue is the transport.
                        agent_thinking = ""
                        agent_transport = "conversational"
                        decision["agent_transport"] = "conversational"
                        (
                            agent_result,
                            rebuilt_from,
                            inherited_history,
                            agent_repair_rounds,
                            repair_exhausted,
                        ) = self._run_conversational_call(
                            dispatch_messages,
                            target_id=candidate.target_id,
                            validator_spec=validator_spec,
                            variant=candidate_variant,
                            capability_tier=tier,
                            max_repair_attempts=max_repair_attempts,
                            retrieval=retrieval_requested,
                            task=plan.task,
                            native_search=native_search_requested,
                            profile_id=(
                                f"policy={plan.policy_id};"
                                f"target={candidate.target_id};"
                                f"route={plan.routing_identity_digest}"
                            ),
                        )
                    # A call that worked is the cheapest possible proof the
                    # subscription is alive, and it clears any freeze.
                    if endpoint.backend == "local_agent":
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
                    raw_response = {
                        "usage": agent_usage_payload(agent_result.usage),
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
                        fallback_used=_agent_fallback_used(idx, all_execution_attempts),
                        raw_response=raw_response,
                        capability_tier=tier,
                        variant=candidate_variant,
                        thinking_level=agent_thinking,
                        thinking_budget=0,
                        api_attempts=list(accumulated_api_attempts),
                        execution_attempts=all_execution_attempts,
                        target_id=candidate.target_id,
                        backend=endpoint.backend,
                        # 0: an agent call sends no `max_tokens` at all, so a
                        # number here would invent a request never made.
                        requested_output_tokens=0,
                        route_decision=route_decision,
                        resumable=not inherited_history,
                        agent_repair_rounds=agent_repair_rounds,
                        repair_exhausted=repair_exhausted,
                    )
                call_kwargs: Dict[str, Any] = {
                    "provider_tier": endpoint.provider_tier,
                    "model": endpoint.api_model_id,
                    "thinking_budget": mapped_budget,
                    "thinking_level": mapped_thinking,
                    "temperature": temperature,
                    "max_tokens": request_max_tokens,
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
                    # Media only: the Files object belongs to the uploading
                    # key's project and any other key 403s on it. Parallel
                    # dispatch used to force this on every call too -- owner
                    # ruling 2026-08-19 reverted that, see the pinning note in
                    # `llm_runtime.chat_complete`.
                    #
                    # Keyed off *this candidate's* ref, not the caller's: the
                    # media ladder can drop the file for a candidate that
                    # cannot take it, and pinning a call that sends no file
                    # only narrows the pool for nothing.
                    "pin_first_key": candidate_ref is not None,
                    # Which key, read off the file rather than derived again.
                    # Empty when the ref predates the field, and then the
                    # first-unlocked fallback applies.
                    "pin_key_id": dispatch_key_id,
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
                    requested_output_tokens=int(request_max_tokens),
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


def _first_gemini_api_key(
    provider_tier: str | None = None,
    *,
    rate_limiter: Any = None,
    model: str = "",
) -> str:
    from . import llm_runtime

    env_map = llm_runtime._read_dotenv()
    if provider_tier is None:
        entry, _tier = api_keys.first_enabled_gemini_entry(env_map)
        return entry.key
    key, _ = llm_runtime._first_key_for_tier(
        provider_tier, env_map, rate_limiter=rate_limiter, model=model
    )
    return key


def _canonical_gemini_key_id(provider_tier: str, api_key: str) -> str:
    """Return the same key id routing/rate-limit accounting uses for a secret."""

    from . import llm_runtime
    from .rate_limit import key_id_for_secret

    env_map = llm_runtime._read_dotenv()
    for entry in llm_runtime._get_key_entries(provider_tier, env_map):
        if entry.key == api_key:
            return entry.key_id
    # A caller-supplied key outside the configured pool has no stable name.
    return key_id_for_secret(api_key)


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


def _agent_fallback_used(
    idx: int, attempts: Sequence[Mapping[str, Any]]
) -> bool:
    """Whether this answer came from somewhere other than first choice.

    Two ways to get there and the audit needs both. `idx > 0` is the harness
    walking its own candidate chain. `configured_model` on an attempt is a CLI
    that swapped the model *inside* one session -- WorkBuddy's paid fallback --
    which never moves `idx`; counting only the index printed "No fallback was
    recorded in retained artifacts" for a call that had just been billed to a
    paid line, which is the one summary someone checks after a surprise bill.
    """

    return idx > 0 or any(item.get("configured_model") for item in attempts)


def _output_budget(
    max_tokens: int | None,
    output_reserve: int | None,
    catalog_entry: Any,
) -> tuple[int, int]:
    """(what to request, what to set aside) for one candidate.

    Two numbers since 2026-09-04, and they answer different questions. The
    **request** is how much room the answer is allowed, and it is not rationed:
    absent an explicit `max_tokens` it fills the candidate's own declared
    ceiling. The **reserve** is how much of a shared context planning keeps
    free for that answer, and it is an *estimate* -- a caller that knows how
    big the answer should be says so (`SESSION_OUTPUT_MAX_TOKENS` for the
    non-correction rounds) and the input side is freed by the difference.

    Passing neither collapses both onto the candidate's ceiling; passing only
    `max_tokens` reproduces the old behaviour exactly, which is what keeps
    every caller that has not been converted honest.

    Neither number is clamped here -- see `_clamp_request_to_context`, which
    the caller applies once the input side has stopped moving.
    """

    request = max_tokens
    if request is None:
        request = (
            catalog_entry.max_output_tokens
            if catalog_entry is not None
            else DEFAULT_LIMITS.output_limit
        )
    reserve = output_reserve if output_reserve is not None else request
    return int(request), int(reserve)


def _clamp_request_to_context(
    request: int, catalog_entry: Any, estimated_input: int
) -> int:
    """Cut the request down to the context this prompt actually leaves free.

    ✱ **Unrationed still means "what fits"** (owner 2026-09-04). On a
    single-pool row the request is capped at whatever context is left beside
    this prompt -- not to save output, but because asking for tokens that
    cannot exist is a provider 400 rather than a shorter answer, and the
    reserve check above would have let the candidate through. A row whose
    `context_window` is the blank-cell stand-in (`max_input + max_output`)
    cannot be clamped by construction, which is exactly right: its two halves
    are metered separately.

    ⚠ **Apply this against the input that is really sent.** It was folded
    into `_output_budget` until 2026-09-04 and therefore ran before the
    oversized repair context could be dropped, so a candidate rescued by that
    drop went out asking for the sliver its *rejected* estimate had left -- 1
    token in the case that motivated the drop. Separating the two makes the
    ordering a property of the call site rather than a comment.
    """

    if catalog_entry is None or estimated_input <= 0:
        return int(request)
    return int(max(1, min(request, catalog_entry.context_window - estimated_input)))


def is_likely_output_limited(
    response: Any,
    *,
    max_tokens: int,
    margin: int = 100,
) -> bool:
    """True when output+thinking tokens land within ``margin`` of the cap.

    Catches truncation that the provider does not flag with a MAX_TOKENS finish
    reason (thinking tokens count against the same budget).

    Agent-transport responses are exempt: their usage is the CLI result
    event's SESSION-CUMULATIVE total across every tool turn (Claude Code and
    agy both report it that way), not this turn's output — comparing it
    against a per-turn cap flags healthy calls as truncated and triggers a
    needless split-in-half rerun (observed 61,446/65,536 on a run where no
    turn was cut; docs/report 2026-08-28 §2.2). Truncation on that path is
    detected by explicit signals (error results, output validation), never by
    this proximity heuristic.
    """

    if is_agent_transport_response(response):
        return False
    dist = extract_token_distribution(response)
    total = dist["output_tokens"] + dist["thinking_tokens"]
    return total > 0 and total >= max_tokens - margin


def is_agent_transport_response(response: Any) -> bool:
    """Whether ``raw_response`` came through the local-agent transport.

    The agent success path always builds ``{"usage": …, "agent": {…}}``
    (`_dispatch`'s LOCAL branch); REST paths return the provider's own
    payload, which never carries an ``agent`` key."""

    return isinstance(response, Mapping) and "agent" in response


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


from .exchange_metadata import llm_exchange_metadata  # re-export for callers/tests
