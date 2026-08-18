from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

import httpx

from finesub.paths import resolve_env_file
from finesub_bootstrap import secrets
from .routing import api_keys



DEFAULT_MODEL = "gemini/gemini-2.5-flash"
LLM_API_TIMEOUT_SECONDS = 15 * 60
CONSECUTIVE_TIMEOUT_ABORT_COUNT = 2

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
    # secrets.read_env_file is the project's only .env parser (it owns the
    # decryption of protected values); ensure_protected is the safety net for
    # source checkouts that never run the desktop/CLI data migrations. It is
    # once-per-process -- this function has no cache and is called per lookup.
    env_path = resolve_env_file()
    if env_path is None:
        return {}
    secrets.ensure_protected(env_path)
    return secrets.read_env_file(env_path)


def _parse_key_list(value: str) -> List[str]:
    return api_keys.parse_key_list(value)


def _parse_key_map(value: str) -> List[Tuple[str, str]]:
    return api_keys.parse_key_map(value)


def _get_key_list(env_name: str, env_map: Dict[str, str]) -> List[str]:
    return [entry.key for entry in _get_key_entries(env_name, env_map)]


def _get_key_entries(
    env_name: str,
    env_map: Dict[str, str],
) -> List[api_keys.ApiKeyEntry]:
    pool_name = api_keys.pool_name_for_tier(env_name)
    if pool_name is not None:
        return api_keys.resolve_pool(pool_name, env_map)
    raw = os.getenv(env_name)
    if raw is None:
        raw = env_map.get(env_name, "")
    pairs = _parse_key_map(raw)
    if pairs:
        return [
            api_keys.ApiKeyEntry(name=name, key=key)
            for name, key in pairs
        ]
    return [
        api_keys.ApiKeyEntry(name="", key=key, named=False)
        for key in _parse_key_list(raw)
    ]


def _thinking_config(
    model_name: str,
    budget: Optional[int],
    level: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the Gemini REST ``thinkingConfig`` dict for generationConfig."""
    model = (model_name or "").lower()
    if "gemini-3" in model:
        # Gemini 3.x thinking is controlled by thinkingLevel.
        if level:
            return {"thinkingLevel": level}
        if budget is None:
            return {}
        if budget <= 0:
            return {"thinkingLevel": "minimal"}
        elif budget <= 800:
            return {"thinkingLevel": "low"}
        else:
            return {"thinkingLevel": "high"}
    if "gemini-2.5" in model:
        # thinkingLevel does not exist on 2.5; only the token budget applies.
        if budget is None:
            return {}
        if "pro" in model and budget == 0:
            return {}
        return {"thinkingBudget": int(budget)}
    return {}


def _first_key_for_tier(provider_tier: str, env_map: Dict[str, str]) -> Tuple[str, str]:
    entries = _get_key_entries(provider_tier, env_map)
    if not entries:
        raise api_keys.ProviderUnavailableError(
            f"Provider {provider_tier} is disabled or has no selected API key."
        )
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
    started_at: str,
    started_monotonic: float,
    return_code: str,
) -> None:
    key = (api_key_label, model_name)
    call_counts[key] = call_counts.get(key, 0) + 1
    returned_at = _iso_now()
    attempts.append(
        {
            "provider_tier": provider_tier,
            "model": model_name,
            "api_key_name": api_key_label,
            "call_number_for_api_key_and_model": call_counts[key],
            "return_code": return_code,
            "started_at": started_at,
            "returned_at": returned_at,
            "elapsed_sec": round(max(0.0, time.monotonic() - started_monotonic), 3),
        }
    )



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
) -> Dict[str, Any]:
    """Call the Gemini generateContent REST endpoint directly.

    Returns the raw JSON response dict (Gemini REST shape: candidates,
    usageMetadata, promptFeedback). Downstream consumers already handle
    this shape natively.
    """
    model_id = model.split("/", 1)[-1] if "/" in model else model
    url = f"{GEMINI_API_BASE}/models/{model_id}:generateContent"

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
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=body)

    if resp.status_code != 200:
        error_text = resp.text
        raise GeminiAPIError(
            f"Gemini API error (HTTP {resp.status_code}): {error_text}",
            status_code=resp.status_code,
        )
    return resp.json()


# Sticky-retry backoff: base×2^attempt, then max(with provider hint) and cap.
# Base raised from 0.5→4 (2026-07-29): even 5xx appears to consume Gemini
# daily quota, so burn fewer retries and wait longer between them.
_BACKOFF_BASE_SECONDS = 4.0
_BACKOFF_CAP_SECONDS = 300.0


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
    pin_first_key: bool = False,
    rate_limiter: Optional[Any] = None,
    estimated_input_tokens: int = 0,
    # model-routing v2: a user-declared provider routes through the thin text
    # transports; None/gemini keeps the packaged Gemini REST path.
    # ``thinking_level`` arrives already mapped through the fact's declared
    # thinking spec ("" = send no thinking parameter).
    provider_spec: Optional[Mapping[str, str]] = None,
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
    # Media calls upload the file under the first key's project; a rotated key
    # cannot access another project's file (403), so pin media to that key.
    if pin_first_key:
        key_entries = key_entries[:1]
    env_name = tier_name

    last_exc: Optional[Exception] = None
    is_gemini_model = "gemini" in model_name.lower()
    thinking_cfg = _thinking_config(model_name, thinking_budget, thinking_level)
    api_attempts: List[Dict[str, Any]] = []
    call_counts: Dict[Tuple[str, str], int] = {}
    consecutive_timeouts = 0
    # v17: every prompt template mandates an opening <reasoning> block, so the
    # old runtime-side injection for non-reasoning models is gone.
    base_messages = messages

    # Gemini REST has no native seed param; embed it as a trailing hint in
    # the last user message so validation retries get some determinism signal.
    if seed is not None:
        seed_line = f"\n(seed={int(seed)})"
        base_messages = list(messages)
        for i in range(len(base_messages) - 1, -1, -1):
            if base_messages[i].get("role") == "user":
                msg = dict(base_messages[i])
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = content + seed_line
                elif isinstance(content, list):
                    msg["content"] = list(content) + [{"type": "text", "text": seed_line}]
                base_messages[i] = msg
                break

    # Per-key daily tracking: build a ModelEndpoint for the rate limiter.
    _rl_endpoint = None
    if rate_limiter is not None:
        from .routing.config import ModelEndpoint

        _rl_endpoint = ModelEndpoint(env_name, model_name)

    for key_entry in key_entries:
        key = key_entry.key
        key_label = key_entry.label
        key_id = key_entry.key_id

        # Skip keys already locked as daily-exhausted (per-key accounting).
        if (
            _rl_endpoint is not None
            and rate_limiter.is_daily_exhausted(_rl_endpoint, key_id=key_id)
        ):
            continue

        combo_phase = None
        if _rl_endpoint is not None:
            from .rate_limit import ComboCooldownPhase

            combo_phase = rate_limiter.combo_cooldown_phase(
                _rl_endpoint, key_id=key_id
            )
            if combo_phase is ComboCooldownPhase.SKIP:
                continue
            # PROBE spends one zero-retry call to test whether the combo came
            # back. Reading the phase is a pure read, so under
            # `continuity=parallel` every concurrent window would read PROBE at
            # once and fire its own. Losing the claim means behaving exactly
            # like SKIP: someone else is already asking the question.
            if combo_phase is ComboCooldownPhase.PROBE and not (
                rate_limiter.claim_combo_probe(_rl_endpoint, key_id=key_id)
            ):
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

            started_at = ""
            started_monotonic = 0.0
            # Wall-clock departure of this attempt, for the daily strike gate:
            # only attempts that departed after the previous strike was known
            # add independent evidence of exhaustion.
            departed_wall = time.time()
            try:
                started_at = _iso_now()
                started_monotonic = time.monotonic()
                if custom_kind:
                    # D18: no sampling parameters for non-Gemini providers --
                    # the re-roll perturbation is already the prompt-tail seed
                    # text appended above, which is provider-agnostic.
                    from . import provider_transports

                    transport = (
                        provider_transports.openai_compat_generate
                        if custom_kind == "openai_compat"
                        else provider_transports.anthropic_generate
                    )
                    response = transport(
                        base_url=custom_base_url,
                        api_key=key,
                        model=model_name,
                        messages=base_messages,
                        max_tokens=max_tokens,
                        thinking_level=thinking_level,
                        timeout=LLM_API_TIMEOUT_SECONDS,
                    )
                else:
                    response = _gemini_generate_content(
                        model=model_name,
                        messages=base_messages,
                        api_key=key,
                        temperature=temperature,
                        safety_settings=GEMINI_SAFETY_SETTINGS if is_gemini_model else None,
                        thinking_config=thinking_cfg,
                        max_tokens=max_tokens,
                        tools=_native_search_tools(native_search_tool) if native_search_tool else None,
                        timeout=LLM_API_TIMEOUT_SECONDS,
                    )
                _record_api_attempt(
                    api_attempts,
                    call_counts,
                    provider_tier=env_name,
                    model_name=model_name,
                    api_key_label=key_label,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                    return_code="200",
                )
                # A success clears the per-key strike streak and combo cooldown.
                if _rl_endpoint is not None:
                    rate_limiter.reset_daily_strikes(_rl_endpoint, key_id=key_id)
                    rate_limiter.clear_combo_cooldown(_rl_endpoint, key_id=key_id)
                response = _attach_harness_meta(response, api_key_label=key_label)
                response["_harness_key_id"] = key_id
                if rate_ticket is not None:
                    # Carried to the caller so settle() can refine this very
                    # call's TPM reservation rather than the newest one.
                    response["_harness_rate_ticket"] = rate_ticket
                return _attach_api_attempts(response, api_attempts)
            except Exception as exc:  # pragma: no cover - network/remote errors
                last_exc = exc
                return_code = _status_code_from_exception(exc)
                if return_code == "NO_RESPONSE_TIMEOUT":
                    consecutive_timeouts += 1
                else:
                    consecutive_timeouts = 0
                if started_at:
                    _record_api_attempt(
                        api_attempts,
                        call_counts,
                        provider_tier=env_name,
                        model_name=model_name,
                        api_key_label=key_label,
                        started_at=started_at,
                        started_monotonic=started_monotonic,
                        return_code=return_code,
                    )
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
                    if is_retryable_provider_error(exc):
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

            if combo_phase is ComboCooldownPhase.PROBE or sticky_exhausted_retryable:
                rate_limiter.note_combo_exhausted(_rl_endpoint, key_id=key_id)

        # Rotate to the next key only when this key spent its retries on a
        # quota/rate-limit error (a separate project may still have budget). A
        # non-quota failure (bad request, exhausted transient retries) won't be
        # fixed by another key, so stop.
        if last_exc is None or not is_quota_or_rate_limit_error(last_exc):
            break
    if last_exc is None:
        # Every key was skipped before it could be tried (daily lock or combo
        # cooldown). The typed error is what tells the endpoint chain in
        # client.py to move on -- don't let that depend on the wording matching
        # is_retryable_provider_error's marker list.
        raise api_keys.ProviderUnavailableError(
            f"All API keys for {env_name} are daily-exhausted or in cooldown."
        )
    _attach_attempts_to_exception(last_exc, api_attempts)
    raise last_exc


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
