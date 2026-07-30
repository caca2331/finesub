"""Runtime client helpers for role-based LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import mimetypes
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import httpx

from .config import (
    CapabilityTier,
    LLMRole,
    RoleModelConfig,
    default_role_configs,
    tier_for_capability,
)
from .model_catalog import get_model_catalog_entry_for_tier
from .profiles import VIDEO_SAMPLE_FPS
from .rate_limit import ModelRateLimiter, estimate_call_input_tokens
from . import api_keys

VALIDATION_BASE_TEMPERATURE = 1.0
VALIDATION_TEMPERATURE_STEP = 0.01
_VALIDATION_SEED_BASE = 1_730_001


@dataclass(frozen=True)
class UploadedFileRef:
    file_id: str
    filename: str
    mime_type: str
    local_path: str = ""


@dataclass(frozen=True)
class LLMCallResult:
    content: str
    role: LLMRole
    model: str
    fallback_used: bool
    raw_response: Mapping[str, Any]
    # Prompt tier of the endpoint that actually answered; callers re-assemble
    # the exact sent messages from it when the prompt was passed as a factory.
    capability_tier: CapabilityTier = CapabilityTier.CAPABLE
    api_key_label: str = ""
    thinking_level: str = ""
    thinking_budget: int = 0
    api_attempts: List[Mapping[str, Any]] = field(default_factory=list)


class LLMIPRiskError(RuntimeError):
    """Provider response suggests the client IP/proxy is risk-blocked."""


def is_quota_or_rate_limit_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "quota",
        "rate limit",
        "ratelimit",
        "resource_exhausted",
        "too many requests",
        "429",
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
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = [
        "503",
        "500",
        "502",
        "504",
        "unavailable",
        "high demand",
        "temporarily",
        "timeout",
        "timed out",
    ]
    return is_quota_or_rate_limit_error(exc) or any(marker in text for marker in markers)


@lru_cache(maxsize=2)
def _shared_rate_limiter(enabled: bool) -> ModelRateLimiter:
    """One process-wide limiter shared by research/correction/update clients."""

    return ModelRateLimiter(enabled=enabled)


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
    if file_ref.mime_type.startswith("video/"):
        # mm-high clips: low sample rate and low media resolution keep the
        # billed frame tokens at the planned 71 tok/frame x 0.25 fps.
        # video_metadata.fps is mapped to videoMetadata.fps in the REST call.
        file_block["detail"] = "low"
        file_block["video_metadata"] = {"fps": VIDEO_SAMPLE_FPS}
    content.append({"type": "file", "file": file_block})
    updated[user_idx]["content"] = content
    return updated


# Factory assembling messages for a capability tier; ``complete`` accepts one
# in place of a fixed message list so the prompt can follow the endpoint that
# actually answers (only the correction call site passes a factory today).
TieredMessages = Callable[[CapabilityTier], List[Dict[str, Any]]]


def _as_tiered(messages: List[Dict[str, Any]] | TieredMessages) -> TieredMessages:
    return messages if callable(messages) else (lambda _tier: messages)


class LiteLLMRoleClient:
    def __init__(
        self,
        *,
        role_configs: Mapping[LLMRole, RoleModelConfig] | None = None,
        rate_limiter: ModelRateLimiter | None = None,
        test_profile: bool = False,
        # Sticky same-key retries only. Kept low because even 5xx responses
        # appear to consume Gemini daily quota (observed 2026-07-29).
        max_retries: int = 3,
    ) -> None:
        self.role_configs = dict(role_configs or default_role_configs())
        self.test_profile = test_profile
        self.max_retries = int(max_retries)
        if rate_limiter is not None:
            self.rate_limiter = rate_limiter
        else:
            self.rate_limiter = _shared_rate_limiter(enabled=not test_profile)

    def complete(
        self,
        role: LLMRole,
        messages: List[Dict[str, Any]] | TieredMessages,
        *,
        max_tokens: int = 65_536,
        temperature: float = VALIDATION_BASE_TEMPERATURE,
        seed: int | None = None,
        file_ref: UploadedFileRef | None = None,
        thinking_budget: int | None = None,
        thinking_level: str | None = None,
    ) -> LLMCallResult:
        """Call the role's endpoint chain through the Gemini REST path.

        ``messages`` is either a fixed list or a :data:`TieredMessages` factory;
        with a factory the prompt is assembled for the capability tier of the
        endpoint about to answer, so a cross-tier fallback never receives a
        prompt written for a stronger model. The chosen tier is reported back
        via ``LLMCallResult.capability_tier``.
        """

        from . import llm_runtime

        config = self.role_configs[role]
        call_thinking_budget = (
            config.thinking_budget if thinking_budget is None else thinking_budget
        )
        call_thinking_level = (
            config.thinking_level if thinking_level is None else thinking_level
        )
        native_search_tool = "" if self.test_profile else config.native_search_tool
        tiered = _as_tiered(messages)
        composed: Dict[CapabilityTier, List[Dict[str, Any]]] = {}
        estimates: Dict[CapabilityTier, int] = {}

        def messages_for(tier: CapabilityTier) -> List[Dict[str, Any]]:
            # Lazy per-tier assembly: a tier's messages are built at most once,
            # and only when the endpoint loop actually reaches that tier.
            if tier not in composed:
                base = tiered(tier)
                composed[tier] = (
                    attach_file_to_messages(base, file_ref) if file_ref else base
                )
                estimates[tier] = estimate_call_input_tokens(
                    composed[tier], file_ref=file_ref
                )
            return composed[tier]

        last_exc: BaseException | None = None
        accumulated_attempts: List[Mapping[str, Any]] = []
        endpoints = config.endpoints(test_profile=self.test_profile)
        for idx, endpoint in enumerate(endpoints):
            if not api_keys.provider_tier_enabled(endpoint.provider_tier):
                continue
            if self.rate_limiter.is_daily_exhausted(endpoint):
                continue
            catalog_entry = get_model_catalog_entry_for_tier(
                endpoint.litellm_model, endpoint.provider_tier
            )
            if (
                native_search_tool
                and catalog_entry is not None
                and not catalog_entry.supports_native_search
            ):
                continue
            # No catalog entry -> CAPABLE: don't tighten the prompt without
            # evidence (same stance as the native-search non-special-casing).
            tier = (
                tier_for_capability(catalog_entry.capability)
                if catalog_entry is not None
                else CapabilityTier.CAPABLE
            )
            call_messages_base = messages_for(tier)
            estimated_input = estimates[tier]
            try:
                call_kwargs: Dict[str, Any] = {
                    "provider_tier": endpoint.provider_tier,
                    "model": endpoint.litellm_model,
                    "thinking_budget": call_thinking_budget,
                    "thinking_level": call_thinking_level,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "retries": self.max_retries,
                    "native_search_tool": native_search_tool or None,
                    # An uploaded file is project-scoped to the first key; a
                    # rotated key would 403 on it, so pin media calls.
                    "pin_first_key": file_ref is not None,
                    # Per-key rate limiting (RPM/TPM + daily) lives inside
                    # chat_complete where the answering key is known.
                    "rate_limiter": self.rate_limiter,
                    "estimated_input_tokens": estimated_input,
                }
                if seed is not None:
                    call_kwargs["seed"] = seed
                response = llm_runtime.chat_complete(call_messages_base, **call_kwargs)
                plain = _to_plain_response(response)
                actual_input = extract_token_distribution(plain)["total_input_tokens"]
                if actual_input <= 0:
                    actual_input = estimated_input
                answering_key_id = str(plain.pop("_harness_key_id", "") or "")
                self.rate_limiter.settle(
                    endpoint,
                    actual_input_tokens=actual_input,
                    estimated_input_tokens=estimated_input,
                    key_id=answering_key_id,
                )
                content = llm_runtime.extract_message_content(plain)
                api_key_label = str(plain.pop("_harness_api_key_label", "") or "")
                api_attempts = list(plain.pop("_harness_api_attempts", []) or [])
                all_attempts = [*accumulated_attempts, *api_attempts]
                return LLMCallResult(
                    content=content,
                    role=role,
                    model=endpoint.litellm_model,
                    fallback_used=idx > 0,
                    raw_response=plain,
                    capability_tier=tier,
                    api_key_label=api_key_label,
                    thinking_level=call_thinking_level or "",
                    thinking_budget=int(call_thinking_budget or 0),
                    api_attempts=all_attempts,
                )
            except Exception as exc:  # pragma: no cover - network/provider behavior
                last_exc = exc
                accumulated_attempts.extend(
                    list(getattr(exc, "_harness_api_attempts", []) or [])
                )
                if isinstance(exc, api_keys.ProviderUnavailableError):
                    continue
                if getattr(exc, "_harness_consecutive_timeout_abort", False):
                    raise
                if isinstance(exc, LLMIPRiskError) or is_likely_ip_risk_error(exc):
                    raise LLMIPRiskError(
                        "LLM IP risk warning: provider response suggests this "
                        f"IP/proxy was risk-blocked: {exc}"
                    ) from None
                # Per-key daily strikes are recorded inside chat_complete; here
                # we just decide whether to try the next endpoint in the chain.
                if is_retryable_provider_error(exc):
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(
            f"All endpoints for role {role.value} are daily-exhausted or unavailable."
        )


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
    if reason in {"content_filter", "safety", "prohibited_content", "blocklist"}:
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
