"""Thin text-only transports for user-declared API providers (model-routing v2).

Two dialects: OpenAI-compatible (DeepSeek / OpenRouter / vLLM / Ollama and
the official API all speak it) and the Anthropic Messages API. Gemini keeps
its own hand-written path in ``llm_runtime``. Deliberately *not* litellm
(D19): these ~200 lines keep the raw response body, which is exactly what the
failure classification and usage accounting need.

Both transports normalize responses into the OpenAI-ish shape the harness
already reads everywhere (``choices[0].message.content`` +
``choices[0].finish_reason`` + nested ``usage`` details):

- usage: OpenAI nests ``*_tokens_details``; DeepSeek reports flat
  ``prompt_cache_hit_tokens``/``reasoning_tokens``; Anthropic reports
  top-level ``cache_read_input_tokens`` and folds thinking into
  ``output_tokens``. All three land in ``prompt_tokens_details.cached_tokens``
  / ``completion_tokens_details.reasoning_tokens``.
- finish reasons: Anthropic ``end_turn``→``stop``, ``max_tokens``→``length``;
  ``refusal`` and everything else pass through verbatim (an Anthropic refusal
  is HTTP 200 + stop_reason, not an error).
- errors: non-2xx raises ``RuntimeError("HTTP <code> ...")`` so the existing
  keyword-based classification applies; DeepSeek's 402 (insufficient balance)
  and OpenAI's 429 ``insufficient_quota`` both classify as ``quota`` so the
  group advances instead of retrying a dead account.

Per D18 no sampling parameters are ever sent (reasoning models reject or
ignore them; the re-roll perturbation is the prompt-tail seed text, which is
provider-agnostic). ``thinking_level`` arrives *already mapped* through the
fact's declared thinking spec (owner design 2026-08-11: catalog/user models
carry ``false`` or an explicit high,med,low -> provider-value mapping):
OpenAI-compatible sends it as ``reasoning_effort`` verbatim; Anthropic sends
it as ``output_config.effort`` (low/medium/high/xhigh). An empty value sends
no thinking parameter at all (a fact declaring ``false`` is the declared way
to keep providers that 400 on unknown fields happy — vLLM/Ollama may reject
rather than ignore them, surveyed 2026-08-11, docs/provider-adapters.md).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import httpx

from .http import llm_http_client

ANTHROPIC_VERSION = "2023-06-01"


class ProviderHTTPError(RuntimeError):
    """Non-2xx from a user-declared provider, with the response kept.

    The message keeps the ``HTTP <code> <body>`` shape the existing
    keyword-based classification reads. Carrying the response matters for one
    thing the body cannot express: a ``Retry-After`` header. Without it the
    limiter falls back to fixed exponential backoff and can retry too early
    against a server that told us exactly how long to wait.
    """

    def __init__(self, message: str, response: httpx.Response) -> None:
        super().__init__(message)
        self.response = response
        self.status_code = response.status_code

    @property
    def retry_after_seconds(self) -> float | None:
        raw = self.response.headers.get("Retry-After")
        if not raw:
            return None
        try:
            # Only the delta-seconds form; an HTTP-date needs clock skew
            # handling we have no reason to take on yet.
            return max(0.0, float(raw.strip()))
        except ValueError:
            return None


def _post_json(
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    with llm_http_client(timeout=timeout) as client:
        response = client.post(url, headers=dict(headers), json=dict(payload))
    if response.status_code < 200 or response.status_code >= 300:
        snippet = response.text[:500]
        raise ProviderHTTPError(
            f"HTTP {response.status_code} {snippet}", response
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"HTTP {response.status_code} returned non-JSON body: "
            f"{response.text[:200]}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError("Provider returned a non-object JSON response")
    return data


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    return str(content or "")


def _plain_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten harness messages to plain text (these backends are text-only)."""

    return [
        {"role": str(msg.get("role", "user")), "content": _as_text(msg.get("content"))}
        for msg in messages
    ]


def openai_compat_generate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    thinking_level: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _plain_messages(messages),
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    level = (thinking_level or "").strip()
    if level:
        # Pre-mapped provider value (the fact's thinking spec).
        payload["reasoning_effort"] = level
    data = _post_json(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=timeout,
    )
    return _normalize_openai_response(data)


def _normalize_openai_response(data: Dict[str, Any]) -> Dict[str, Any]:
    usage = data.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if not isinstance(details, dict):
            details = {}
            usage["prompt_tokens_details"] = details
        # DeepSeek reports flat cache-hit/miss fields instead of the nested
        # details object (surveyed 2026-08-11).
        if "cached_tokens" not in details and isinstance(
            usage.get("prompt_cache_hit_tokens"), (int, float)
        ):
            details["cached_tokens"] = int(usage["prompt_cache_hit_tokens"])
        completion_details = usage.get("completion_tokens_details")
        if not isinstance(completion_details, dict):
            completion_details = {}
            usage["completion_tokens_details"] = completion_details
        if "reasoning_tokens" not in completion_details and isinstance(
            usage.get("reasoning_tokens"), (int, float)
        ):
            completion_details["reasoning_tokens"] = int(usage["reasoning_tokens"])
    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if isinstance(message, dict):
                # DeepSeek (`reasoning_content`) / vLLM (`reasoning`) return
                # the raw chain of thought next to the answer; the harness
                # must never treat it as visible output.
                message.pop("reasoning_content", None)
                message.pop("reasoning", None)
    return data


def anthropic_generate(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: Optional[int],
    thinking_level: Optional[str],
    timeout: float,
) -> Dict[str, Any]:
    plain = _plain_messages(messages)
    system_text = "\n\n".join(
        msg["content"] for msg in plain if msg["role"] == "system" and msg["content"]
    )
    turns = [msg for msg in plain if msg["role"] != "system"]
    if not turns or turns[0]["role"] != "user":
        # Messages API requires the first turn to be user; consecutive
        # same-role turns are merged server-side, so this is the only fix-up.
        turns = [{"role": "user", "content": ""}] + turns
    effective_max = int(max_tokens or 8192)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": effective_max,  # required by the Messages API
        "messages": turns,
    }
    if system_text:
        payload["system"] = system_text
    level = (thinking_level or "").strip()
    if level:
        # Pre-mapped effort word (low/medium/high/xhigh); the Anthropic
        # Messages API takes it as output_config.effort (owner decision
        # 2026-08-11 -- not thinking.budget_tokens).
        payload["output_config"] = {"effort": level}
    data = _post_json(
        base_url.rstrip("/") + "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout=timeout,
    )
    return _normalize_anthropic_response(data)


_ANTHROPIC_FINISH = {"end_turn": "stop", "max_tokens": "length"}


def _normalize_anthropic_response(data: Dict[str, Any]) -> Dict[str, Any]:
    blocks = data.get("content")
    text = ""
    if isinstance(blocks, list):
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
    stop_reason = str(data.get("stop_reason") or "")
    usage_in = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = int(usage_in.get("input_tokens") or 0)
    cache_read = int(usage_in.get("cache_read_input_tokens") or 0)
    cache_write = int(usage_in.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage_in.get("output_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_write
    return {
        "choices": [
            {
                "message": {"content": text},
                # ``refusal`` and other reasons pass through verbatim: an
                # Anthropic refusal is a 200 + stop_reason, and the harness's
                # blocked-prompt check knows the name.
                "finish_reason": _ANTHROPIC_FINISH.get(stop_reason, stop_reason),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cache_read},
            "completion_tokens": output_tokens,
            # Anthropic folds thinking into output_tokens without a separate
            # count; reasoning_tokens stays absent rather than guessed.
            "completion_tokens_details": {},
            "total_tokens": prompt_tokens + output_tokens,
        },
        "provider_raw": data,
    }
