"""Thin text-only transports for user-declared API providers (model-routing v2).

Two dialects: OpenAI-compatible (DeepSeek / OpenRouter / vLLM / Ollama and
the official API all speak it) and the Anthropic Messages API. Gemini keeps
its own hand-written path in ``llm_runtime``. Deliberately *not* litellm
(D19): these ~200 lines keep the raw response body, which is exactly what the
failure classification and usage accounting need.

Both stream (SSE, `http.stream_within_deadline`); `streaming` assembles the
events back into each dialect's non-streamed body, so everything below reads
the same shape it always did. Both normalize responses into the OpenAI-ish
shape the harness already reads everywhere (``choices[0].message.content`` +
``choices[0].finish_reason`` + nested ``usage`` details):

- usage: OpenAI nests ``*_tokens_details``; DeepSeek reports flat
  ``prompt_cache_hit_tokens``/``reasoning_tokens``; Anthropic reports
  top-level ``cache_read_input_tokens`` and folds thinking into
  ``output_tokens``, counting it under ``output_tokens_details.thinking_tokens``. All three land in ``prompt_tokens_details.cached_tokens``
  / ``completion_tokens_details.reasoning_tokens``.
- finish reasons: Anthropic ``end_turn``→``stop``, ``max_tokens``→``length``;
  ``refusal`` and everything else pass through verbatim (an Anthropic refusal
  is HTTP 200 + stop_reason, not an error).
- prompt caching (`prompt_cache`): the OpenAI-compatible request carries a
  ``prompt_cache_key`` per shared prefix (dropped for good on an endpoint that
  400s on it); the Anthropic request marks a breakpoint after the system and
  after the conversation's opening (the turns before the first reply) -- on
  the opening call itself only when the caller expects a next turn to re-send
  it. A call to Anthropic's own API that runs past `KEEPALIVE_SECONDS` keeps
  those entries alive with a ``max_tokens: 0`` request beside it, since the
  five-minute lifetime counts from the start of the request that wrote them.
- errors: non-2xx raises ``RuntimeError("HTTP <code> ...")`` so the existing
  keyword-based classification applies, and so does an error event inside a
  200 stream (``streaming.ProviderStreamError``, carrying the error's own
  status); DeepSeek's 402 (insufficient balance)
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

Search, on a call that asks for native retrieval (why: the catalog decision
record in docs/llm_design_notes.md): Anthropic gets its server-side
``web_search`` tool, capped at ``WEB_SEARCH_MAX_USES`` searches and never
resumed after a ``pause_turn``; an OpenAI-compatible endpoint gets the
model's own ``search_request`` fields merged into the body, since that dialect
has no common switch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

import httpx

from . import prompt_cache
from ..api_http import CONNECT_TIMEOUT_SECONDS
from ..reporting import current_reporter
from .http import StreamProgress, every_while, llm_http_client, stream_within_deadline
from .streaming import (
    AnthropicMessagesAssembler,
    OpenAIChatAssembler,
    ProviderStreamError,
    SSEAssembler,
)

ANTHROPIC_VERSION = "2023-06-01"
#: A cache entry lives five minutes from the start of the request that wrote or last read it, generation time
#: included (Anthropic's prompt caching docs). A call still running this long after it was sent gets a keep-alive
#: request beside it, and another every as many seconds, until it ends: the next request on its prefix then has at
#: least `300 - KEEPALIVE_SECONDS` seconds to start in.
KEEPALIVE_SECONDS = 240.0
#: The keep-alive answers as soon as the prefix is read; a slower one is not worth waiting on.
KEEPALIVE_TIMEOUT_SECONDS = 60.0
#: Only Anthropic's own API gets keep-alives: an Anthropic-compatible endpoint (DeepSeek, Kimi, ...) may bill a
#: ``max_tokens: 0`` request differently or keep no five-minute entries at all.
ANTHROPIC_API_HOST = "api.anthropic.com"
#: Searches one Anthropic call may run (owner 2026-11-09). A search past it
#: comes back as a `max_uses_exceeded` error result and the model answers
#: with what it has; it caps what one call spends on search and keeps the
#: server loop short, which makes a `pause_turn` (a long-running loop the
#: API paused) rarer, not impossible.
WEB_SEARCH_MAX_USES = 5
#: The basic tool version: the dynamic-filtering one is taken only by recent
#: Claude models, and Claude on Vertex and Azure has only this one.
ANTHROPIC_WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": WEB_SEARCH_MAX_USES,
}


class SearchTurnPaused(RuntimeError):
    """Anthropic's server-side search loop stopped the turn before an answer.

    Not resumed (the catalog decision record in docs/llm_design_notes.md): the
    call goes to the group's next member.
    It carries no status and its wording avoids every word the retry and quota
    classifiers read, so it is neither retried in place -- a retry would run
    the searches and bill them again, and a spent retry budget puts the model
    on a cool-down that skips its calls that do not search too -- nor taken
    for a quota or a network problem; `route_failure_kind` sends it on.
    """

    route_failure_kind = "transient"


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


def _stream_json(
    url: str,
    *,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    assembler: SSEAssembler,
    timeout: float,
) -> Dict[str, Any]:
    """`timeout` is the hard deadline on the whole call (`stream_within_deadline`)."""

    return stream_within_deadline(
        url,
        headers=headers,
        json=dict(payload),
        deadline_seconds=timeout,
        assembler=assembler,
        error_for_status=_http_error,
        progress=assembler.progress,
    )


def _http_error(response: httpx.Response) -> ProviderHTTPError:
    return ProviderHTTPError(f"HTTP {response.status_code} {response.text[:500]}", response)


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
    progress: StreamProgress,
    search_request: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": _plain_messages(messages),
        "stream": True,
        # Without it the stream carries no usage at all.
        "stream_options": {"include_usage": True},
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    level = (thinking_level or "").strip()
    if level:
        # Pre-mapped provider value (the fact's thinking spec).
        payload["reasoning_effort"] = level
    cache_key = (
        "" if prompt_cache.cache_key_rejected(base_url) else prompt_cache.prefix_key(messages)
    )
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    if search_request:
        # The loader refused any field that would overwrite one above.
        payload.update(search_request)

    def send() -> Dict[str, Any]:
        return _stream_json(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=payload,
            assembler=OpenAIChatAssembler(progress),
            timeout=timeout,
        )

    try:
        data = send()
    except ProviderHTTPError as exc:
        if not (cache_key and prompt_cache.is_cache_key_rejection(exc)):
            raise
        prompt_cache.note_cache_key_rejected(base_url)
        payload.pop("prompt_cache_key")
        data = send()
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
    progress: StreamProgress,
    followup_expected: bool = False,
    web_search: bool = False,
) -> Dict[str, Any]:
    plain = _plain_messages(messages)
    system_text = "\n\n".join(
        msg["content"] for msg in plain if msg["role"] == "system" and msg["content"]
    )
    turns: List[Dict[str, Any]] = [msg for msg in plain if msg["role"] != "system"]
    if not turns or turns[0]["role"] != "user":
        # Messages API requires the first turn to be user; consecutive
        # same-role turns are merged server-side, so this is the only fix-up.
        turns = [{"role": "user", "content": ""}] + turns
    # Anthropic reads an entry only from a breakpoint at or up to 20 blocks
    # after where it was written, so both sides of a conversation mark the same
    # place: the end of its opening (the turns before the first reply).
    marked = set()
    first_reply = next((i for i, turn in enumerate(turns) if turn["role"] == "assistant"), 0)
    if first_reply:
        # A later turn reads what the opening wrote (a request turn, a
        # preparation t1, a window's proofread). After an opening that wrote
        # nothing -- a window's first repair -- this writes it, and about half
        # the windows that need one repair need another, which reads it.
        marked.add(first_reply - 1)
    elif followup_expected:
        # The opening itself, when a next turn will re-send it. Not written on
        # every call: an entry costs 1.25x to write, and a repair round is too
        # rare a follow-up to pay for that on every window. A later turn has
        # nothing to mark: its next turn re-sends the opening, not this turn.
        marked.add(len(turns) - 1)
    marked = {index for index in marked if turns[index]["content"]}
    for index in marked:
        turns[index] = {"role": turns[index]["role"], "content": [_cached_text(turns[index]["content"])]}
    effective_max = int(max_tokens or 8192)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": effective_max,  # required by the Messages API
        "messages": turns,
        "stream": True,
    }
    if system_text:
        # The system is the same for every call of a stage, across windows
        # and runs: the breakpoint every call reads (a too-short system is
        # silently not cached, which is no worse than not asking).
        payload["system"] = [_cached_text(system_text)]
    level = (thinking_level or "").strip()
    if level:
        # Pre-mapped effort word (low/medium/high/xhigh); the Anthropic
        # Messages API takes it as output_config.effort (owner decision
        # 2026-08-11 -- not thinking.budget_tokens).
        payload["output_config"] = {"effort": level}
    if web_search:
        payload["tools"] = [dict(ANTHROPIC_WEB_SEARCH_TOOL)]
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
    }
    body = _keepalive_body(payload, max(marked, default=None))
    if body is not None and httpx.URL(url).host == ANTHROPIC_API_HOST:
        with every_while(KEEPALIVE_SECONDS, _KeepAlive(url, headers, body, progress, model)):
            data = _stream_json(url, headers=headers, payload=payload,
                                assembler=AnthropicMessagesAssembler(progress), timeout=timeout)
    else:
        data = _stream_json(url, headers=headers, payload=payload,
                            assembler=AnthropicMessagesAssembler(progress), timeout=timeout)
    if web_search:
        _check_search_reply(data)
    return _normalize_anthropic_response(data)


def _check_search_reply(data: Mapping[str, Any]) -> None:
    stop_reason = str(data.get("stop_reason") or "")
    if stop_reason == "pause_turn":
        raise SearchTurnPaused(
            "Anthropic server-side web search paused the turn before answering "
            "(stop_reason=pause_turn); FineSub does not resume it and hands the "
            "call to the next model in the group"
        )
    blocks = data.get("content") if isinstance(data.get("content"), list) else []
    handed_back = any(
        isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name") == "web_search"
        for block in blocks
    )
    if stop_reason == "tool_use" or handed_back:
        # The endpoint took the server tool for a client one: it speaks the
        # Messages API without running Anthropic's search. A rejected request,
        # like an endpoint that 400s on the unknown tool -- the call fails
        # rather than answering unsearched.
        raise ProviderStreamError(
            400,
            "this endpoint returned the web_search tool to the caller instead of "
            "running it server-side, so it does not provide Anthropic's search; "
            "set supports_native_search = false for this model in model_catalog.toml",
        )


def _anthropic_answer_text(blocks: List[Any]) -> str:
    """The answer: every text block except the one right before a search.

    Claude opens a search with a line like "Let me look that up", which is not
    answer text. A stretch of answer written right before a further search is
    dropped with it; that is rare, and the alternative -- keeping every
    preamble -- puts one into almost every searched reply.
    """

    kept = []
    for index, block in enumerate(blocks):
        if not (isinstance(block, dict) and block.get("type") == "text"):
            continue
        following = blocks[index + 1] if index + 1 < len(blocks) else None
        if isinstance(following, dict) and following.get("type") == "server_tool_use":
            continue
        kept.append(str(block.get("text", "")))
    return "".join(kept)


def _anthropic_search_events(blocks: List[Any]) -> List[Dict[str, Any]]:
    """One ledger row per search: its query and the result URLs.

    Anthropic pairs each result with its search (``tool_use_id``), so the rows
    say which page came back for which query -- finer than Gemini's one row
    per call.
    """

    urls_by_search: Dict[str, List[str]] = {}
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "web_search_tool_result":
            results = block.get("content") if isinstance(block.get("content"), list) else []
            urls_by_search[str(block.get("tool_use_id") or "")] = [
                str(item["url"]) for item in results if isinstance(item, dict) and item.get("url")
            ]
    events = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "server_tool_use":
            query_input = block.get("input") if isinstance(block.get("input"), dict) else {}
            query = str(query_input.get("query") or "").strip()
            events.append(
                {
                    "tool": "web_search",
                    "query": query,
                    "queries": [query] if query else [],
                    "urls": urls_by_search.get(str(block.get("id") or ""), []),
                }
            )
    return events


def _keepalive_body(payload: Mapping[str, Any], last_marked: Optional[int]) -> Optional[Dict[str, Any]]:
    """The keep-alive for `payload`: the same request up to its last cache breakpoint, asking for nothing.

    Everything that is part of the cached prefix stays as sent -- model, system, tools, ``output_config`` (effort
    is rendered into the prompt) and the breakpoints -- so the request reads the entry the call wrote and bills
    one cache read. The turns after the last breakpoint are dropped (they would bill as fresh input); with only
    the system marked, one placeholder user turn stands in, as Anthropic's pre-warming example does. ``stream`` is
    dropped: the API rejects ``max_tokens: 0`` with it. None when nothing is marked.
    """

    if last_marked is not None:
        messages = list(payload["messages"][: last_marked + 1])
    elif payload.get("system"):
        messages = [{"role": "user", "content": "."}]
    else:
        return None
    body = {key: value for key, value in payload.items() if key not in ("stream", "max_tokens", "messages")}
    body.update(max_tokens=0, messages=messages)
    return body


class _KeepAlive:
    """One keep-alive request per call (`http.every_while`); what it spent goes on the call's `StreamProgress`.

    It stays off every shared state a failed request touches -- the host reachability record, the key's strikes
    and rate-limit reservations: a keep-alive that fails costs the call nothing but the cache it was keeping. It
    stops (returns False) on a rejected request, or when the reply shows it did not just read the prefix -- a
    write, no read, or output: the entry was already gone, the endpoint keeps none, or the prefix differs from
    the call's -- and says so once.
    """

    def __init__(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any], progress: StreamProgress,
                 model: str) -> None:
        self.url, self.headers, self.body, self.progress, self.model = url, dict(headers), dict(body), progress, model

    def __call__(self) -> bool:
        if self.progress.cancelled:
            return False
        books = self.progress.keepalive
        books["requests"] = books.get("requests", 0) + 1
        try:
            timeout = httpx.Timeout(KEEPALIVE_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
            with llm_http_client(timeout=timeout) as client:
                response = client.post(self.url, headers=self.headers, json=self.body)
        except httpx.HTTPError:
            books["failed"] = books.get("failed", 0) + 1
            return True  # the network may be back for the next one
        if not 200 <= response.status_code < 300:
            books["failed"] = books.get("failed", 0) + 1
            if response.status_code == 429 or response.status_code >= 500:
                return True
            return self._stop(f"HTTP {response.status_code}")
        try:
            reply = response.json()
        except ValueError:
            reply = None
        usage = reply.get("usage") if isinstance(reply, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        read = int(usage.get("cache_read_input_tokens") or 0)
        written = int(usage.get("cache_creation_input_tokens") or 0)
        fresh = int(usage.get("input_tokens") or 0)
        for key, value in (("cached_input_tokens", read), ("cache_write_input_tokens", written),
                           ("uncached_input_tokens", fresh), ("total_input_tokens", read + written + fresh),
                           ("output_tokens", int(usage.get("output_tokens") or 0))):
            books[key] = books.get(key, 0) + value
        if written or not read or usage.get("output_tokens"):
            return self._stop(f"it read {read} cached tokens and wrote {written}")
        return True

    def _stop(self, why: str) -> bool:
        current_reporter().warning(
            "llm-cache-keepalive-stopped",
            f"{self.model}: the prompt-cache keep-alive did not just read the cached prefix ({why}): the entry "
            "expired, the endpoint keeps none, or the prefix is shorter than the model caches. No more keep-alives "
            "on this call; the next turn may write its prefix to the cache again.",
        )
        return False


def _cached_text(text: str) -> Dict[str, Any]:
    """A text block with a five-minute cache breakpoint after it."""

    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


_ANTHROPIC_FINISH = {"end_turn": "stop", "max_tokens": "length"}


def _normalize_anthropic_response(data: Dict[str, Any]) -> Dict[str, Any]:
    blocks = data.get("content") if isinstance(data.get("content"), list) else []
    text = _anthropic_answer_text(blocks)
    stop_reason = str(data.get("stop_reason") or "")
    usage_in = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    input_tokens = int(usage_in.get("input_tokens") or 0)
    cache_read = int(usage_in.get("cache_read_input_tokens") or 0)
    cache_write = int(usage_in.get("cache_creation_input_tokens") or 0)
    output_tokens = int(usage_in.get("output_tokens") or 0)
    # Thinking is folded into output_tokens; the count of it is under output_tokens_details (absent on a reply
    # that did not report it, which stays absent rather than guessed).
    details_in = usage_in.get("output_tokens_details") if isinstance(usage_in.get("output_tokens_details"), dict) else {}
    thinking = details_in.get("thinking_tokens")
    prompt_tokens = input_tokens + cache_read + cache_write
    server_tools = usage_in.get("server_tool_use") if isinstance(usage_in.get("server_tool_use"), dict) else {}
    normalized: Dict[str, Any] = {
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
            "prompt_tokens_details": {"cached_tokens": cache_read, "cache_write_tokens": cache_write},
            "completion_tokens": output_tokens,
            "completion_tokens_details": (
                {"reasoning_tokens": int(thinking)}
                if isinstance(thinking, (int, float)) and not isinstance(thinking, bool) else {}
            ),
            "total_tokens": prompt_tokens + output_tokens,
        },
        "provider_raw": data,
    }
    searches = int(server_tools.get("web_search_requests") or 0)
    if searches:
        # Billed per search, apart from tokens.
        normalized["usage"]["web_search_requests"] = searches
    events = _anthropic_search_events(blocks)
    if events:
        normalized["search_events"] = events
    return normalized
