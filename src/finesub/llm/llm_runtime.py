from __future__ import annotations

from dataclasses import dataclass
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from finesub import settings
from finesub.api_http import (
    HOST_BREAKER,
    Gate,
    HostUnreachableError,
    Route,
    current_route,
    is_connect_failure,
)
from finesub.reporting import current_reporter, redact_credentials
from . import prompt_cache
from .http import (
    HedgedRequest,
    StreamCancelled,
    StreamInterrupted,
    StreamProgress,
    hedge_on_silence,
    keep_partial_text,
    stream_within_deadline,
)
from .streaming import GeminiAssembler
from finesub import api_keys



DEFAULT_MODEL = "gemini/gemini-2.5-flash"
CONSECUTIVE_TIMEOUT_ABORT_COUNT = 2

# How long one generation call may take is sized from how much it is allowed to
# produce, not fixed. The whole call is the whole generation -- 64k tokens at
# 35 tok/s is ~30 minutes on its own, which the old flat 15-minute timeout cut
# off while the provider was still working (and, presumably, billing). Streaming
# shows the answer arriving, but not how long the rest will take, and a thinking
# model may send nothing at all before its answer.
#
# Two clocks, because "slower than it should be" and "dead" are different
# findings (owner decision 2026-10-23):
#
# - HARD: the whole generation at the floor rate, a deadline on the whole call
#   (`http.stream_within_deadline`), not an httpx timeout -- those bound each
#   read, and a trickling server never trips them. Past it the request is closed.
# - SOFT: a fraction of the hard one, counted as *silence* -- since the request
#   left or since its last byte. Past it a backup request is sent beside the
#   first, which stays open (`http.hedge_on_silence`): whichever answers first
#   is used and the other closed. Silence alone cannot tell a model thinking
#   without a word from a dead connection, so the first is never given up early.
#
# The budget is the call's *allowed* output (`maxOutputTokens` as sent, thinking
# included -- it counts against the same cap), never the planner's output
# estimate: an answer is allowed to exceed that estimate, and sizing the timeout
# from it would cut exactly those answers off mid-generation.
#
# The three numbers (overhead, floor rate, soft fraction) are `[llm]
# api_timeout_*` / `api_soft_timeout_fraction` in `finesub.settings`.
#: The budget when a call names no cap and the provider applies its own.
API_TIMEOUT_DEFAULT_OUTPUT_TOKENS = 65_536


@dataclass(frozen=True)
class ApiTimeouts:
    soft: float
    hard: float


def api_timeouts(max_tokens: Optional[int]) -> ApiTimeouts:
    config = settings.read()
    overhead = settings.API_TIMEOUT_OVERHEAD_SECONDS.resolve(config=config)
    floor_rate = settings.API_TIMEOUT_FLOOR_TOKENS_PER_SECOND.resolve(config=config)
    fraction = settings.API_SOFT_TIMEOUT_FRACTION.resolve(config=config)
    budget = max_tokens if max_tokens and max_tokens > 0 else API_TIMEOUT_DEFAULT_OUTPUT_TOKENS
    hard = overhead + budget / floor_rate
    return ApiTimeouts(soft=hard * fraction, hard=hard)


# Default safety thresholds false-positive on ordinary subtitle material (a
# crying VTuber reading in-game farewell dialogue was prompt-blocked with
# finish_reason=content_filter, zero output, media not billed). All calls in
# this repo transcribe/translate existing media, so relax every adjustable
# category explicitly.
GEMINI_SAFETY_SETTINGS = [
    {"category": category, "threshold": "BLOCK_NONE"}
    for category in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    )
]

def _read_dotenv() -> Dict[str, str]:
    return api_keys.read_dotenv()


def _get_key_list(env_name: str, env_map: Dict[str, str]) -> List[str]:
    return [entry.key for entry in _get_key_entries(env_name, env_map)]


def _get_key_entries(
    env_name: str,
    env_map: Dict[str, str],
) -> List[api_keys.ApiKeyEntry]:
    return api_keys.key_entries(env_name, env_map)


#: Ask for thought summaries whenever the model thinks at all. They cost
#: nothing extra -- the tokens are billed either way -- and without them a
#: thinking model's deliberation is a number with no content behind it, which
#: is exactly what stalled the 2026-08-25 prompt iteration: three arms differed
#: by 2x in thinking tokens and nothing could say what the difference was
#: about. `extract_message_content` already drops parts marked `thought`, so
#: nothing here can reach the answer.
INCLUDE_THOUGHTS = True


def _thinking_config(
    model_name: str,
    budget: Optional[int],
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Gemini REST ``thinkingConfig`` dict for generationConfig."""
    model = (model_name or "").lower()

    def thinking(config: Dict[str, Any]) -> Dict[str, Any]:
        if config and INCLUDE_THOUGHTS:
            config["includeThoughts"] = True
        return config

    if "gemini-3" in model:
        # Gemini 3.x thinking is controlled by thinkingLevel.
        if level:
            return thinking({"thinkingLevel": level})
        if budget is None:
            return {}
        if budget <= 0:
            # Minimal is still a thinking level, but there is nothing to
            # summarise; asking would only add a round trip's worth of noise.
            return {"thinkingLevel": "minimal"}
        elif budget <= 800:
            return thinking({"thinkingLevel": "low"})
        else:
            return thinking({"thinkingLevel": "high"})
    if "gemini-2.5" in model:
        # thinkingLevel does not exist on 2.5; only the token budget applies.
        if budget is None:
            return {}
        if "pro" in model and budget == 0:
            return {}
        if int(budget) <= 0:
            return {"thinkingBudget": int(budget)}
        return thinking({"thinkingBudget": int(budget)})
    return {}


def _first_key_for_tier(
    provider_tier: str,
    env_map: Dict[str, str],
    *,
    rate_limiter: Optional[Any] = None,
    model: str = "",
) -> Tuple[str, str]:
    """The key a pinned call will use: the first that is not daily-locked.

    Media uploads resolve their key through here so the upload and the call
    that reads the file land in the same project -- a Files object belongs to
    the uploading key, and any other key 403s on it. Passing the limiter is
    what keeps them in step once a key locks: without it the upload keeps
    going to key #1 while the call has moved on, and every attempt 403s
    forever instead of self-healing (the re-upload would go to key #1 again).

    Falls back to the plain first key when no limiter is available, or when
    every key is locked -- there is nothing better to return, and the call
    that follows will produce the real error.
    """

    entries = _get_key_entries(provider_tier, env_map)
    if not entries:
        raise api_keys.ProviderUnavailableError(
            f"Provider {provider_tier} is disabled or has no selected API key."
        )
    if rate_limiter is not None and model:
        from .routing.config import ModelEndpoint

        endpoint = ModelEndpoint(provider_tier, model)
        for entry in entries:
            if not rate_limiter.is_daily_exhausted(endpoint, key_id=entry.key_id):
                return entry.key, provider_tier
    return entries[0].key, provider_tier


def _attach_harness_meta(response: Any, *, api_key_label: str) -> Any:
    if isinstance(response, Mapping):
        response["_harness_api_key_label"] = api_key_label
        return response
    setattr(response, "_harness_api_key_label", api_key_label)
    return response


def _attach_api_attempts(response: Any, attempts: List[Dict[str, Any]]) -> Any:
    if isinstance(response, Mapping):
        response["_harness_api_attempts"] = list(attempts)
        return response
    setattr(response, "_harness_api_attempts", list(attempts))
    return response


def _attach_attempts_to_exception(exc: BaseException, attempts: List[Dict[str, Any]]) -> None:
    try:
        setattr(exc, "_harness_api_attempts", list(attempts))
    except Exception:  # pragma: no cover - defensive
        pass


def _status_code_from_exception(exc: BaseException) -> str:
    # Before the text rules: an interruption quotes the httpx error it wraps.
    if isinstance(exc, StreamCancelled):
        return "SUPERSEDED"
    if isinstance(exc, StreamInterrupted):
        return "STREAM_INTERRUPTED"
    # Before the text rules too: a `ConnectTimeout` says "timed out", but the
    # host never answered the handshake -- not a generation the deadline closed.
    if is_connect_failure(exc):
        return "NO_RESPONSE_CONNECT"
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return str(status)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return str(status)
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "NO_RESPONSE_TIMEOUT"
    if (
        "server disconnected" in lowered
        or "remote protocol" in lowered
        or "connection reset" in lowered
        or "connection closed" in lowered
    ):
        return "NO_RESPONSE_CONNECTION_CLOSED"
    match = re.search(r"\b([45]\d\d)\b", text)
    if match:
        return match.group(1)
    return type(exc).__name__


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _record_api_attempt(
    attempts: List[Dict[str, Any]],
    call_counts: Dict[Tuple[str, str], int],
    *,
    provider_tier: str,
    model_name: str,
    api_key_label: str,
    request: HedgedRequest[Any],
) -> None:
    """One HTTP request as it ended; `request.progress` was created as it
    departed, so its clock is the request's."""

    stream = request.progress
    error = request.error
    return_code = "200" if error is None else _status_code_from_exception(error)
    # A request closed because the other one answered did not fail.
    reason = "" if return_code in {"200", "SUPERSEDED"} else str(error)
    key = (api_key_label, model_name)
    call_counts[key] = call_counts.get(key, 0) + 1
    returned_at = _iso_now()
    elapsed = round(max(0.0, time.monotonic() - stream.started), 3)
    attempt: Dict[str, Any] = {
        "provider_tier": provider_tier,
        "model": model_name,
        "api_key_name": api_key_label,
        "call_number_for_api_key_and_model": call_counts[key],
        "return_code": return_code,
        "started_at": stream.started_wall.isoformat(timespec="milliseconds"),
        "returned_at": returned_at,
        "elapsed_sec": elapsed,
    }
    attempt.update(stream.record())
    if request.backup:
        attempt["backup"] = True
    if request.backup_sent_after is not None:
        attempt["backup_sent_sec"] = round(request.backup_sent_after, 3)
    kept = None
    if return_code not in {"200", "SUPERSEDED"} and stream.header_at is not None:
        # How much of the answer was lost with the attempt, and the text
        # itself for whoever investigates the failure.
        attempt["partial_chars"] = stream.output_chars
        kept = keep_partial_text(
            stream,
            {
                "model": model_name,
                "tier": provider_tier,
                "key": api_key_label,
                "code": return_code,
                "why": " ".join(redact_credentials(reason).split())[:500],
            },
        )
        if kept:
            attempt["partial_text_file"] = kept
    attempts.append(attempt)
    if return_code == "200" and (request.backup or request.backup_sent_after is not None):
        # The other half of the soft-timeout warning: how the pair ended.
        # Said at warning level because the first half was -- a reader who saw
        # "sent a backup" must be able to find out which one answered. What
        # became of the other one is its own attempt row.
        winner = "the backup" if request.backup else "the original request"
        current_reporter().warning(
            "llm-api-backup-outcome",
            f"{model_name}: {winner} answered, {elapsed:.0f}s after it was sent.",
        )
    # The same fact, said out loud. The ledger above travels with the artifacts
    # and is read afterwards; the run log is what a user sends when something
    # went wrong, and until now it recorded not one word about the API calls
    # the run is almost entirely made of.
    #
    # Status and a one-line description, never the prompt or the answer: the
    # full text of every exchange is already written per call under
    # `<stem>.llm-artifacts/exchanges/`, and a second copy would put tens of
    # megabytes into the one file that is meant to be small enough to send.
    # Those files render this very attempt list, which is what lets a line here
    # be matched to the exchange it belongs to.
    fields: Dict[str, Any] = {
        "model": model_name,
        "tier": provider_tier,
        # A label, never the key.
        "key": api_key_label,
        "code": return_code,
        "sec": f"{elapsed:.3f}",
        "n": call_counts[key],
    }
    if request.backup:
        fields["backup"] = "yes"
    if kept:
        fields["kept"] = kept
    if reason:
        # What the *endpoint* said, not a sentence of ours: a bare `429` does
        # not distinguish "this key is spent today" from "slow down", and that
        # difference is the first thing anyone reading the log wants.
        #
        # Redacted before trimming, because an httpx error quotes the request
        # URL and a user's own `[api] proxy` or custom `base_url` may be
        # written `https://user:token@host`. Keys themselves never appear:
        # every transport sends them as a header.
        fields["why"] = " ".join(redact_credentials(reason).split())[:200]
    current_reporter().debug("llm api call", fields)



def _native_search_tools(tool_name: str) -> List[Dict[str, Any]]:
    """Provider-native web-search tool spec for the Gemini REST tools array.

    Only calls that request the native-search capability enable this;
    unsupported providers/models
    reject the request and the task fails with the provider error.
    """

    normalized = tool_name.strip().lower()
    if normalized in {"google_search", "googlesearch"}:
        return [{"googleSearch": {}}]
    if normalized in {"web_search", "websearch"}:
        return [{"type": "web_search"}]
    raise ValueError(f"Unknown native search tool '{tool_name}'.")


# --------- Gemini REST direct call ---------

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

#: `.env` override for the endpoint above. The two custom transports have taken
#: a base URL since they existed; this path not having one was an asymmetry
#: rather than a decision -- and the one it hurt most is the case the download
#: routes already solve for everything else, a machine that cannot reach
#: Google's host directly. Also what lets an integration test point the REST
#: path at a local stand-in.
GEMINI_BASE_URL_VARIABLE = "GEMINI_BASE_URL"


def gemini_api_base(env_map: Mapping[str, str]) -> str:
    """Where Gemini generation calls go: the `GEMINI_BASE_URL` mirror when set, else Google.

    One answer for the request and for the host its reachability is booked on
    (`_HostAttempts`, `RoleClient._api_host`): booking a mirror's calls on
    Google would give every call through a working mirror a single attempt
    once Google itself is judged unreachable.
    """

    return (env_map.get(GEMINI_BASE_URL_VARIABLE) or "").strip() or GEMINI_API_BASE

# OpenAI-style ``detail`` -> Gemini per-part ``mediaResolution`` enum.
_MEDIA_RESOLUTION_BY_DETAIL = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}


class GeminiAPIError(Exception):
    """Error from the Gemini REST API with HTTP status context."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


def _convert_content_parts(content: Any) -> List[Dict[str, Any]]:
    """Convert a message's content (str or list of blocks) to Gemini parts."""
    if isinstance(content, str):
        return [{"text": content}]
    parts: List[Dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            parts.append({"text": block})
        elif isinstance(block, dict):
            block_type = block.get("type", "text")
            if block_type == "text":
                parts.append({"text": block.get("text", "")})
            elif block_type == "file":
                file_info = block.get("file", {})
                part: Dict[str, Any] = {
                    "fileData": {
                        "fileUri": file_info.get("file_id", ""),
                        "mimeType": file_info.get("format", ""),
                    }
                }
                video_meta = file_info.get("video_metadata")
                if video_meta:
                    part["videoMetadata"] = video_meta
                # detail -> per-part mediaResolution (Gemini 3+). Shape is
                # {"level": "MEDIA_RESOLUTION_*"} — a bare enum string 400s.
                # mm-high video clips rely on "low" to keep billed frame tokens
                # at the planned ~71 tok/frame; without it Gemini falls back.
                detail = _MEDIA_RESOLUTION_BY_DETAIL.get(file_info.get("detail", ""))
                if detail:
                    part["mediaResolution"] = {"level": detail}
                parts.append(part)
    return parts


def _messages_to_gemini_body(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split OpenAI-style messages into Gemini contents + systemInstruction parts."""
    system_parts: List[Dict[str, Any]] = []
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.extend(_convert_content_parts(content))
            continue
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": _convert_content_parts(content)})
    return contents, system_parts


def _gemini_generate_content(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    api_key: str,
    temperature: float,
    safety_settings: Optional[List[Dict[str, Any]]],
    thinking_config: Dict[str, Any],
    max_tokens: Optional[int],
    tools: Optional[List[Dict[str, Any]]],
    timeout: float,
    progress: StreamProgress,
    api_base: str = GEMINI_API_BASE,
) -> Dict[str, Any]:
    """Call the Gemini REST API directly, streamed.

    Returns the `generateContent` response shape (candidates, usageMetadata,
    promptFeedback), assembled from the SSE chunks by `GeminiAssembler`, so
    downstream consumers read what they always read. `timeout` is a deadline
    on the whole call.
    """
    model_id = model.split("/", 1)[-1] if "/" in model else model
    url = f"{api_base.rstrip('/')}/models/{model_id}:streamGenerateContent?alt=sse"

    contents, system_parts = _messages_to_gemini_body(messages)
    body: Dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}

    gen_config: Dict[str, Any] = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = int(max_tokens)
    if thinking_config:
        gen_config["thinkingConfig"] = thinking_config
    if gen_config:
        body["generationConfig"] = gen_config

    if safety_settings:
        body["safetySettings"] = safety_settings
    if tools:
        body["tools"] = tools

    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    return stream_within_deadline(
        url,
        headers=headers,
        json=body,
        deadline_seconds=timeout,
        assembler=GeminiAssembler(progress, _gemini_api_error),
        error_for_status=lambda resp: _gemini_api_error(resp.status_code, resp.text),
        progress=progress,
    )


def _gemini_api_error(status_code: int, body: str) -> GeminiAPIError:
    return GeminiAPIError(f"Gemini API error (HTTP {status_code}): {body}", status_code=status_code)


# Sticky-retry backoff: base×2^attempt, then max(with provider hint) and cap.
# Base raised from 0.5→4 (2026-07-29): even 5xx appears to consume Gemini
# daily quota, so burn fewer retries and wait longer between them.
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_CAP_SECONDS = 300.0


def describe_skipped_keys(
    env_name: str,
    *,
    daily: int,
    cooldown: int,
    probing: int,
    cooldown_retry_at: Optional[datetime] = None,
) -> str:
    """Say why no key was tried, keeping the two waits apart.

    One sentence naming both taught a reader to report "the daily quota is
    gone" when in fact two keys were twenty minutes into a back-off and the
    provider console showed five calls all day (observed 2026-09-03). The
    difference is what to do next: a daily lock holds until the quota's own
    reset, a cooldown for minutes -- so the cooldown says when.
    """

    reasons: List[str] = []
    if daily:
        reasons.append(f"{daily} exhausted for the day (until the quota's own reset)")
    if cooldown:
        until = ""
        if cooldown_retry_at is not None:
            moment = cooldown_retry_at.astimezone(timezone.utc).strftime("%H:%M")
            until = f", retryable from {moment} UTC"
        reasons.append(
            f"{cooldown} in a transient cooldown after recent errors{until}"
        )
    if probing:
        reasons.append(
            f"{probing} left to another call that is already probing the same "
            "cooldown"
        )
    detail = "; ".join(reasons) if reasons else "none is configured"
    return f"No API key for {env_name} could be tried: {detail}."


def _reserve_backup(
    rate_limiter: Any, endpoint: Any, key_id: str, estimated_input_tokens: int
) -> None:
    """A backup is one more request on the same key, sending the whole input
    again. Its reservation is never settled: the estimate stands for a request
    whose usage nobody reads."""

    if endpoint is not None:
        rate_limiter.acquire(endpoint, estimated_input_tokens, key_id=key_id)


def _generate_hedged(
    *,
    custom_kind: str,
    custom_base_url: str,
    api_key: str,
    key_id: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    temperature: float,
    thinking_level: Optional[str],
    thinking_cfg: Dict[str, Any],
    native_search_tool: Optional[str],
    search_request: Optional[Mapping[str, Any]] = None,
    env_map: Mapping[str, str],
    timeouts: ApiTimeouts,
    note_backup: Callable[[], None],
    record: Callable[[HedgedRequest[Any]], None],
    followup_expected: bool = False,
) -> Dict[str, Any]:
    """One generation on the chosen transport: a request closed at the hard
    deadline, and a backup beside it once the first has been silent for the
    soft one (`http.hedge_on_silence`). `record` sees every request as it
    ends; `note_backup` counts the backup towards the key's rate limits.

    On the two non-Gemini dialects the request first passes
    `prompt_cache.warm_prefix`: concurrent requests on a cold prefix wait for
    the first one to write it. The prefix is per key (caches live per
    account) and per system text. Gemini is left out -- its users are mostly
    on the free tier, where a cache hit saves nothing and waiting costs time.
    The caller has already counted this request towards RPM, so a follower
    released late may land outside the window it was counted in; with four
    parallel windows that stays far below any provider limit."""

    def send(progress: StreamProgress) -> Dict[str, Any]:
        if custom_kind:
            # D18: no sampling parameters for non-Gemini providers -- the
            # re-roll perturbation is already the prompt-tail seed text
            # `chat_complete` appends, which is provider-agnostic.
            from . import provider_transports

            if custom_kind == "openai_compat":
                return provider_transports.openai_compat_generate(
                    base_url=custom_base_url,
                    api_key=api_key,
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    thinking_level=thinking_level,
                    timeout=timeouts.hard,
                    progress=progress,
                    # Only on a call that searches; the name is just a marker.
                    search_request=search_request if native_search_tool else None,
                )
            return provider_transports.anthropic_generate(
                base_url=custom_base_url,
                api_key=api_key,
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                thinking_level=thinking_level,
                timeout=timeouts.hard,
                progress=progress,
                followup_expected=followup_expected,
                web_search=bool(native_search_tool),
            )
        return _gemini_generate_content(
            model=model_name,
            messages=messages,
            api_key=api_key,
            temperature=temperature,
            safety_settings=(
                GEMINI_SAFETY_SETTINGS if "gemini" in model_name.lower() else None
            ),
            thinking_config=thinking_cfg,
            max_tokens=max_tokens,
            tools=_native_search_tools(native_search_tool) if native_search_tool else None,
            timeout=timeouts.hard,
            progress=progress,
            api_base=gemini_api_base(env_map),
        )

    def on_backup(progress: StreamProgress) -> None:
        # Where the stream stands is the point: "no headers yet", "connected
        # but still thinking" and "still writing" call for different readings.
        current_reporter().warning(
            "llm-api-soft-timeout",
            f"{model_name}: nothing arrived for {timeouts.soft:.0f}s (soft timeout; "
            f"{progress.describe()}); sent a backup request and kept this one "
            f"open, each up to {timeouts.hard:.0f}s.",
        )
        note_backup()

    scope = (
        (custom_kind, custom_base_url, key_id, model_name, prompt_cache.prefix_key(messages))
        if custom_kind
        else ("",)  # no prefix: `warm_prefix` lets it through at once
    )
    with prompt_cache.warm_prefix(scope):
        return hedge_on_silence(
            send, silence_seconds=timeouts.soft, on_backup=on_backup, on_end=record
        )


def _with_seed_line(messages: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """No provider takes a native seed we use (Gemini REST has none; D18 sends
    no sampling parameters elsewhere), so a re-ask carries it as a trailing
    hint in the last user message -- which is why only a blind re-ask gets one
    (`client.validation_retry_sampling_kwargs`)."""

    seed_line = f"\n(seed={int(seed)})"
    seeded = list(messages)
    for i in range(len(seeded) - 1, -1, -1):
        if seeded[i].get("role") == "user":
            msg = dict(seeded[i])
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + seed_line
            elif isinstance(content, list):
                msg["content"] = list(content) + [{"type": "text", "text": seed_line}]
            seeded[i] = msg
            break
    return seeded


def chat_complete(
    messages: List[Dict[str, Any]],
    *,
    provider_tier: Optional[str] = None,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    retries: int = 3,
    native_search_tool: Optional[str] = None,
    # An OpenAI-compatible model's fields that turn its provider's search on
    # (catalog `search_request`); sent only when `native_search_tool` is set.
    search_request: Optional[Mapping[str, Any]] = None,
    pin_first_key: bool = False,
    # The exact key a media call must use: the one whose project owns the
    # uploaded file (`UploadedFileRef.api_key_id`). Empty falls back to the
    # first key this tier has that is not daily-locked.
    pin_key_id: str = "",
    rate_limiter: Optional[Any] = None,
    estimated_input_tokens: int = 0,
    # model-routing v2: a user-declared provider routes through the thin text
    # transports; None/gemini keeps the packaged Gemini REST path.
    # ``thinking_level`` arrives already mapped through the fact's declared
    # thinking spec ("" = send no thinking parameter).
    provider_spec: Optional[Mapping[str, str]] = None,
    # The caller will continue this conversation (a request turn, a
    # preparation t1): the transport keeps what was sent cacheable for the
    # next turn where that is opt-in (Anthropic).
    followup_expected: bool = False,
) -> Dict[str, Any]:
    env_map = _read_dotenv()
    # The tier comes from the endpoint the chain picked. A `profile=` argument
    # used to offer a second way to say the same thing, left over from the
    # pre-REST client; `client.complete` -- the only caller -- never set it.
    tier_name = provider_tier
    model_name = model or DEFAULT_MODEL

    if not tier_name:
        raise ValueError("chat_complete requires provider_tier or profile.")

    custom_kind = ""
    custom_base_url = ""
    if provider_spec is not None:
        kind = str(provider_spec.get("kind") or "")
        if kind in ("openai_compat", "anthropic"):
            custom_kind = kind
            custom_base_url = str(provider_spec.get("base_url") or "")

    # Try every key for the tier, rotating to the next one when the current key
    # is quota/rate limited (each free key is a separate project with its own
    # RPM/RPD). A quota error on one key does not waste the others' budget.
    # Custom providers resolve their (usually single) key from the declared
    # env name -- same encrypted ``.env`` path as every other key.
    key_source = (
        str(provider_spec.get("key_env") or "") if custom_kind else tier_name
    )
    key_entries = _get_key_entries(key_source, env_map)
    if not key_entries:
        raise api_keys.ProviderUnavailableError(
            f"Provider {tier_name} is disabled or has no selected API key."
        )
    # Pinning is a **media** constraint, and only that: the uploaded file is
    # scoped to the uploading key's project, so any other key 403s on it. A
    # pinned call therefore attempts exactly one key and never rotates in place
    # (the rotation at the bottom of the loop is skipped) -- rotating would
    # trade a quota error for a guaranteed 403.
    #
    # *Which* key: whichever one actually holds the file. The caller reads it
    # off the ref (`pin_key_id`) rather than deriving it a second time -- see
    # the note on `UploadedFileRef.api_key_id` for why that indirection is the
    # whole point. Without a pin id (a non-media pinned call, or a ref from
    # before this field existed) it falls back to the first key that is not
    # daily-locked; that is also what `_first_key_for_tier` gives the upload
    # side, so the two still agree by construction.
    #
    # It used to mean "key #1, always", and the truncation happened before the
    # exhaustion skip could look at it -- so once key #1 locked, the call
    # reported "every key skipped" while the rest of the pool sat there with
    # full quota.
    if pin_key_id:
        pinned = [entry for entry in key_entries if entry.key_id == pin_key_id]
        # Another key cannot read this file. If the pool changed between the
        # caller's owner check and dispatch, fail as unavailable instead of
        # making a guaranteed permanent 403 with a different project.
        if not pinned:
            raise api_keys.ProviderUnavailableError(
                f"Pinned media key is no longer selected for {tier_name}."
            )
        key_entries = pinned
    #
    # `continuity=parallel` used to pin every call as well, to keep only one
    # key in flight at a time. Owner ruling 2026-08-19 reverted that: parallel
    # behaves like serial about keys, and the occasional overlap -- one lane
    # rotating on a rate-limit while its siblings are still on the old key --
    # is accepted rather than designed out. What keeps it occasional is the
    # sticky-retry rule already in this loop: a 429 is retried in place on the
    # same key, and only a key that spends its whole retry budget on quota
    # errors is abandoned. See docs/llm_harness_routing.md.
    env_name = tier_name

    last_exc: Optional[Exception] = None
    thinking_cfg = _thinking_config(model_name, thinking_budget, thinking_level)
    api_attempts: List[Dict[str, Any]] = []
    call_counts: Dict[Tuple[str, str], int] = {}
    consecutive_timeouts = 0
    timeouts = api_timeouts(max_tokens)
    # No runtime-side reasoning injection for non-reasoning models: what a reply opens with is the
    # prompt templates' business.
    base_messages = messages if seed is None else _with_seed_line(messages, seed)

    reach = _HostAttempts(urlsplit(custom_base_url or gemini_api_base(env_map)).hostname or "", api_attempts,
                          lambda: _release_unanswered_probe(rate_limiter, _rl_endpoint, key_id, combo_phase))

    # Per-key daily tracking: build a ModelEndpoint for the rate limiter.
    _rl_endpoint = None
    if rate_limiter is not None:
        from .routing.config import ModelEndpoint

        _rl_endpoint = ModelEndpoint(env_name, model_name)

    # Why each key was passed over, so "nothing could be tried" can say which
    # of the two it was. They are hours apart in consequence: a daily lock
    # holds until the quota's own reset, a combo cooldown for twenty minutes.
    skipped_daily = 0
    skipped_cooldown = 0
    skipped_probing = 0
    cooldown_retry_at: Optional[datetime] = None

    for key_entry in key_entries:
        key = key_entry.key
        key_label = key_entry.label
        key_id = key_entry.key_id

        # Skip keys already locked as daily-exhausted (per-key accounting).
        if (
            _rl_endpoint is not None
            and rate_limiter.is_daily_exhausted(_rl_endpoint, key_id=key_id)
        ):
            skipped_daily += 1
            continue

        combo_phase = None
        if _rl_endpoint is not None:
            from .rate_limit import ComboCooldownPhase

            combo_phase = rate_limiter.combo_cooldown_phase(
                _rl_endpoint, key_id=key_id
            )
            # A *pinned* call does not let a cooldown move it. Only the daily
            # lock does -- which is exactly the rule `_first_key_for_tier` uses
            # to pick the upload key, and the two must agree or the call reads
            # a file that lives in another key's project. Letting a cooldown
            # move only the call gives 403, forget-and-re-upload, the same
            # cooldown, the same 403: a loop that eats the window's whole retry
            # budget. A cooldown is also the wrong reason to abandon a file --
            # it is a soft "this combo is sore, back off for minutes", while
            # the daily lock means the key cannot serve at all today.
            #
            # The PROBE claim goes the same way, and that costs something: N
            # concurrent pinned windows can each fire their own probe, which is
            # what `claim_combo_probe` exists to prevent. Bounded by
            # `--parallel-windows` and only in that phase -- cheaper than the
            # 403 loop, which is unbounded within a window's budget. The
            # cooldown still shapes `effective_sticky_retries` below; it just
            # does not move the key.
            if combo_phase is ComboCooldownPhase.SKIP and not pin_first_key:
                skipped_cooldown += 1
                retry_at = rate_limiter.combo_cooldown_retry_at(
                    _rl_endpoint, key_id=key_id
                )
                if retry_at is not None and (
                    cooldown_retry_at is None or retry_at < cooldown_retry_at
                ):
                    cooldown_retry_at = retry_at
                continue
            # PROBE spends one zero-retry call to test whether the combo came
            # back. Reading the phase is a pure read, so under
            # `continuity=parallel` every concurrent window would read PROBE at
            # once and fire its own. Losing the claim means behaving exactly
            # like SKIP: someone else is already asking the question.
            if (
                combo_phase is ComboCooldownPhase.PROBE
                and not pin_first_key
                and not rate_limiter.claim_combo_probe(
                    _rl_endpoint, key_id=key_id
                )
            ):
                skipped_probing += 1
                continue
            effective_retries = rate_limiter.effective_sticky_retries(
                _rl_endpoint, key_id=key_id, default_retries=retries
            )
        else:
            effective_retries = retries

        sticky_exhausted_retryable = False
        # The TPM ticket belongs to the logical call, not to one HTTP attempt:
        # it survives sticky retries so a success on attempt N still settles
        # its own attempt-0 reservation instead of a sibling's (the last-event
        # fallback under concurrency).
        rate_ticket = None
        for attempt in range(effective_retries + 1):
            reach.admit(attempt, last_exc)
            # Every HTTP attempt counts toward RPM. The first attempt also
            # pre-reserves TPM; sticky retries only note RPM (failed requests
            # still hit provider RPM — observed even on 5xx / 2026-07-29).
            if _rl_endpoint is not None:
                if attempt == 0 and estimated_input_tokens > 0:
                    rate_ticket = rate_limiter.acquire(
                        _rl_endpoint, estimated_input_tokens, key_id=key_id
                    )
                else:
                    rate_limiter.note_request(_rl_endpoint, key_id=key_id)

            # Wall-clock departure of this attempt, for the daily strike gate:
            # only attempts that departed after the previous strike was known
            # add independent evidence of exhaustion.
            departed_wall = time.time()
            try:
                response = _generate_hedged(
                    custom_kind=custom_kind,
                    custom_base_url=custom_base_url,
                    api_key=key,
                    key_id=key_id,
                    model_name=model_name,
                    messages=base_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking_level=thinking_level,
                    thinking_cfg=thinking_cfg,
                    native_search_tool=native_search_tool,
                    search_request=search_request,
                    env_map=env_map,
                    timeouts=timeouts,
                    note_backup=lambda: _reserve_backup(
                        rate_limiter, _rl_endpoint, key_id, estimated_input_tokens
                    ),
                    record=lambda request: _record_api_attempt(
                        api_attempts,
                        call_counts,
                        provider_tier=env_name,
                        model_name=model_name,
                        api_key_label=key_label,
                        request=request,
                    ),
                    # A seeded re-ask's opening is never re-sent with its seed:
                    # the next turn rebuilds it without one, so nothing would
                    # read what a breakpoint there writes.
                    followup_expected=followup_expected and seed is None,
                )
                _note_combo_answered(rate_limiter, _rl_endpoint, key_id)
                response = _attach_harness_meta(response, api_key_label=key_label)
                response["_harness_key_id"] = key_id
                if rate_ticket is not None:
                    # Carried to the caller so settle() can refine this very
                    # call's TPM reservation rather than the newest one.
                    response["_harness_rate_ticket"] = rate_ticket
                return _attach_api_attempts(response, api_attempts)
            except Exception as exc:  # pragma: no cover - network/remote errors
                last_exc = exc
                from .provider_transports import SearchTurnPaused

                if isinstance(exc, SearchTurnPaused):
                    # The server answered: a PROBE re-freezing here would skip
                    # this model's calls that do not search too.
                    _note_combo_answered(rate_limiter, _rl_endpoint, key_id)
                    _attach_attempts_to_exception(exc, api_attempts)
                    raise
                reach.give_up_if_unreachable(exc)
                # A retry starts the generation from zero, so only a request
                # the hard deadline closed counts towards giving up.
                if _status_code_from_exception(exc) == "NO_RESPONSE_TIMEOUT":
                    consecutive_timeouts += 1
                else:
                    consecutive_timeouts = 0
                if consecutive_timeouts >= CONSECUTIVE_TIMEOUT_ABORT_COUNT:
                    _attach_attempts_to_exception(exc, api_attempts)
                    try:
                        setattr(exc, "_harness_consecutive_timeout_abort", True)
                    except Exception:  # pragma: no cover - defensive
                        pass
                    raise
                from .client import (
                    classify_quota_error,
                    is_quota_or_rate_limit_error,
                    is_retryable_provider_error,
                    QuotaKind,
                )

                # Per-key daily strike: a PerDay 429 feeds the strike gate.
                # If the gate confirms sustained exhaustion, lock this key and
                # rotate immediately (don't burn retries on a dead key).
                if (
                    _rl_endpoint is not None
                    and classify_quota_error(exc) is QuotaKind.DAILY
                    and rate_limiter.note_daily_quota_hit(
                        _rl_endpoint, key_id=key_id, departed_at=departed_wall
                    )
                ):
                    break

                if attempt >= effective_retries or not is_retryable_provider_error(exc):
                    # Not connecting says nothing about quota or load (`_HostAttempts`).
                    if is_retryable_provider_error(exc) and not is_connect_failure(exc):
                        sticky_exhausted_retryable = True
                    break
                # Sticky: a quota/rate-limit 429 is retryable in place — keep
                # retrying the SAME key within its budget instead of rotating on
                # the first hit. Backoff respects the provider's suggested wait
                # (parsed from the error text) but is clamped to a sane cap.
                from .rate_limit import parse_retry_after_seconds

                exponential = _BACKOFF_BASE_SECONDS * (2**attempt)
                provider_hint = parse_retry_after_seconds(exc)
                sleep_seconds = min(max(exponential, provider_hint), _BACKOFF_CAP_SECONDS) + 1
                time.sleep(sleep_seconds)

        if _rl_endpoint is not None:
            from .rate_limit import ComboCooldownPhase

            if last_exc is not None and is_connect_failure(last_exc):
                reach.release_probe()
            elif combo_phase is ComboCooldownPhase.PROBE or sticky_exhausted_retryable:
                rate_limiter.note_combo_exhausted(_rl_endpoint, key_id=key_id)

        # Rotate to the next key only when this key spent its retries on a
        # quota/rate-limit error (a separate project may still have budget). A
        # non-quota failure (bad request, exhausted transient retries) won't be
        # fixed by another key, so stop. A *pinned* call never rotates either:
        # it carries an uploaded file that only this key's project can read,
        # so the next key would 403 rather than help (see the pinning note
        # above `key_entries`).
        if (
            pin_first_key
            or last_exc is None
            or not is_quota_or_rate_limit_error(last_exc)
        ):
            break
    if last_exc is None:
        # Every key was skipped before it could be tried. The typed error is
        # what tells the endpoint chain in client.py to move on -- don't let
        # that depend on the wording matching is_retryable_provider_error's
        # marker list. The wording still has to separate the reasons: one
        # sentence naming both taught a reader to say "daily quota is gone"
        # when in fact two keys were twenty minutes into a back-off and the
        # provider console showed five calls all day (observed 2026-09-03).
        raise api_keys.ProviderUnavailableError(
            describe_skipped_keys(
                env_name,
                daily=skipped_daily,
                cooldown=skipped_cooldown,
                probing=skipped_probing,
                cooldown_retry_at=cooldown_retry_at,
            )
        )
    _attach_attempts_to_exception(last_exc, api_attempts)
    raise last_exc


def _release_unanswered_probe(
    rate_limiter: Any, endpoint: Any, key_id: str, combo_phase: Any
) -> None:
    """A combo probe that never reached the server hands its claim back."""

    from .rate_limit import ComboCooldownPhase

    if endpoint is not None and combo_phase is ComboCooldownPhase.PROBE:
        rate_limiter.release_combo_probe(endpoint, key_id=key_id)


@dataclass
class _HostAttempts:
    """One call's view of its host's reachability (`api_http.HostBreaker`).

    Once the host is judged unreachable the call gets one attempt and no
    retries, and a connect failure ends it as `HostUnreachableError`: typed
    `unavailable`, so the candidate chain moves on, and worded so the reader
    learns which host and what to do about it. A combo probe it held never
    reached the server, so its claim is handed back. Not connecting says
    nothing about the combo's quota or load either, so it sets no sticky
    exhaustion and no cooldown: a persisted cooldown would outlive the network
    coming back.

    `release_probe` runs while one key is being tried; `chat_complete` passes a
    closure that reads that key's id and combo phase when it runs.
    """

    host: str
    api_attempts: List[Dict[str, Any]]
    release_probe: Callable[[], None]
    route: Optional[Route] = None
    once: bool = False

    def admit(self, attempt: int, last_exc: Optional[BaseException]) -> None:
        self.route = current_route()
        self.once = HOST_BREAKER.gate_required(self.host, self.route) is Gate.ONCE
        if self.once and attempt > 0:
            # Judged unreachable while this call was backing off.
            raise self._give_up(last_exc)

    def give_up_if_unreachable(self, exc: BaseException) -> None:
        if is_connect_failure(exc) and (self.once or HOST_BREAKER.unreachable(self.host, self.route)):
            raise self._give_up(exc) from exc

    def _give_up(self, cause: Optional[BaseException]) -> HostUnreachableError:
        self.release_probe()
        error = HostUnreachableError(self.host, self.route, cause)
        _attach_attempts_to_exception(error, self.api_attempts)
        return error


def _note_combo_answered(rate_limiter: Any, endpoint: Any, key_id: str) -> None:
    """The server answered: clear the key's daily strike streak and the combo cooldown."""

    if endpoint is not None:
        rate_limiter.reset_daily_strikes(endpoint, key_id=key_id)
        rate_limiter.clear_combo_cooldown(endpoint, key_id=key_id)


def extract_thought_text(response: Dict[str, Any]) -> str:
    """The model's own summary of how it got there, when it returned one.

    Only present with ``includeThoughts``; a summary, not the raw chain. The
    counterpart of `extract_message_content`, which drops exactly these parts.
    """

    try:
        parts = response["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError):
        return ""
    return "\n".join(
        str(p["text"])
        for p in parts
        if isinstance(p, dict) and p.get("thought") and p.get("text")
    ).strip()


def extract_message_content(response: Dict[str, Any]) -> str:
    # Gemini REST shape: candidates[0].content.parts[].text
    try:
        parts = response["candidates"][0]["content"]["parts"]
        texts = [
            p["text"]
            for p in parts
            if isinstance(p, dict) and "text" in p and not p.get("thought")
        ]
        if texts:
            return "".join(texts)
    except Exception:
        pass
    # Legacy OpenAI shapes (kept for safety / test fixtures)
    try:
        return response["choices"][0]["message"]["content"]
    except Exception:
        pass
    try:
        return response["choices"][0]["text"]
    except Exception:
        raise ValueError("Unexpected response format; missing message content.")
