"""SSE bodies of the streamed generation calls, assembled back into the
non-streamed response shape each dialect returns.

Downstream (content extraction, usage accounting, rate-limit settle, exchange
logs, the blocked-prompt check) reads the non-streamed shape and is not touched
by the switch to streaming: an assembler's `finish()` returns exactly that
shape, or raises. `http.stream_within_deadline` feeds it one body line at a
time and owns the clocks; the assembler owns the dialect.

Two rules hold for every dialect (docs/llm_design_notes.md, the streaming
decision record):

- **No terminal marker, no answer.** A stream that ends without it raises
  `StreamInterrupted` -- the half that arrived would otherwise read as a
  complete answer.
- **An error event after HTTP 200 carries its real status.** Retry and quota
  classification read the structured status code before the text
  (`client.provider_status_code`), so the raised error has the semantic code
  (529 for Anthropic's `overloaded_error`, ...) and no 2xx response attached.

Only visible answer text calls `progress.output`; thinking, reasoning and
metadata chunks are events that prove the connection is alive.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from finesub.reporting import current_reporter

from .http import StreamInterrupted, StreamProgress


class ProviderStreamError(RuntimeError):
    """An error event inside a 200 stream from a user-declared provider.

    The message keeps the ``HTTP <code> <body>`` shape the keyword-based
    classification reads, like `provider_transports.ProviderHTTPError`; unlike
    it there is no response to carry, so no ``Retry-After`` either.
    """

    retry_after_seconds: Optional[float] = None

    def __init__(self, status_code: int, body: Any) -> None:
        text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        super().__init__(f"HTTP {status_code} {text[:500]}")
        self.status_code = status_code


class SSEAssembler:
    """Server-sent events framing; subclasses consume whole events.

    Per the SSE spec, a line without a field prefix names an unknown field and
    is ignored -- which is what Gemini's error tail needs: after
    ``data: {"error": ...}`` it repeats the same JSON as bare multi-line text.
    An event still open when the body ends is dispatched rather than dropped, so
    a final error event without its blank line is still seen as an error.
    """

    def __init__(self, progress: StreamProgress) -> None:
        self.progress = progress
        self._data: List[str] = []
        self._event = ""

    def feed_line(self, line: str) -> None:
        if line == "":
            self._dispatch()
            return
        if line.startswith(":"):
            return
        field, _sep, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value

    def finish(self) -> Dict[str, Any]:
        self._dispatch()
        return self.result()

    def _dispatch(self) -> None:
        data, event = "\n".join(self._data), self._event or "message"
        had_data = bool(self._data)
        self._data, self._event = [], ""
        if not had_data:
            return
        self.progress.event()
        self.on_event(event, data)

    def on_event(self, event: str, data: str) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def result(self) -> Dict[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _json(data: str) -> Dict[str, Any]:
        try:
            parsed = json.loads(data)
        except ValueError as exc:
            raise StreamInterrupted(f"stream interrupted: unreadable event {data[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise StreamInterrupted(f"stream interrupted: non-object event {data[:200]!r}")
        return parsed


#: Error `code` / `type` strings an OpenAI-compatible stream may carry instead of
#: a number. Anything unrecognised is a 500: an error that arrives after a 200
#: is the server failing mid-answer, not the request being wrong.
_OPENAI_ERROR_STATUS = {
    "insufficient_quota": 429,
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "invalid_request_error": 400,
    "server_error": 500,
}


class OpenAIChatAssembler(SSEAssembler):
    """Chat Completions chunks (`stream_options.include_usage` on).

    Terminal: ``data: [DONE]``, or a normal close after a ``finish_reason`` *and*
    a usage chunk -- the usage chunk comes after the finish chunk, so a close
    between the two would otherwise book the call at zero tokens. Usage is the
    last non-null one anywhere: OpenAI puts it on a chunk with empty
    ``choices``, OpenRouter on the finish chunk, Gemini's compatible endpoint on
    every chunk, cumulatively. ``finish_reason`` likewise: OpenRouter repeats it.
    ⚠ With cumulative usage on every chunk (Gemini's endpoint) a close after the
    finish chunk but before ``[DONE]`` also passes: the text is whole, the usage
    may be a chunk short. Accepted: nothing is lost that a retry would recover.
    """

    def __init__(self, progress: StreamProgress) -> None:
        super().__init__(progress)
        self._text: List[str] = []
        self._finish_reason: Optional[str] = None
        self._usage: Optional[Dict[str, Any]] = None
        self._meta: Dict[str, Any] = {}
        self._done = False

    def on_event(self, event: str, data: str) -> None:
        if data.strip() == "[DONE]":
            self._done = True
            return
        chunk = self._json(data)
        error = chunk.get("error")
        if error:
            raise _openai_stream_error(error)
        for key in ("id", "model", "created"):
            if key in chunk and key not in self._meta:
                self._meta[key] = chunk[key]
        if isinstance(chunk.get("usage"), dict):
            self._usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str) and content:
                self._text.append(content)
                self.progress.output(content)
            if choice.get("finish_reason"):
                self._finish_reason = str(choice["finish_reason"])

    def result(self) -> Dict[str, Any]:
        if not (self._done or (self._finish_reason and self._usage is not None)):
            raise StreamInterrupted(
                "stream interrupted: ended without [DONE] "
                f"(finish_reason={self._finish_reason!r}, usage={'yes' if self._usage else 'no'})"
            )
        if self._finish_reason == "error":
            # OpenRouter marks a provider failure this way; the error event that
            # normally precedes it was missing.
            raise ProviderStreamError(500, "stream finished with finish_reason=error")
        data: Dict[str, Any] = {
            **self._meta,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(self._text)},
                    "finish_reason": self._finish_reason,
                }
            ],
        }
        if self._usage is not None:
            data["usage"] = self._usage
        return data


def _openai_stream_error(error: Any) -> ProviderStreamError:
    if not isinstance(error, dict):
        return ProviderStreamError(500, {"error": error})
    code = error.get("code")
    if isinstance(code, int) and 400 <= code < 600:
        status = code
    else:
        status = _OPENAI_ERROR_STATUS.get(str(code)) or _OPENAI_ERROR_STATUS.get(
            str(error.get("type")), 500
        )
    return ProviderStreamError(status, {"error": error})


#: Anthropic error `type` -> the HTTP status the same error has outside a
#: stream (platform.claude.com/docs/en/api/errors).
_ANTHROPIC_ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


class AnthropicMessagesAssembler(SSEAssembler):
    """Messages API events, rebuilt into the non-streamed Message object.

    Terminal: ``message_stop``. Usage is ``message_start``'s, updated by each
    ``message_delta`` (whose counts are cumulative). Only ``text_delta`` is
    answer text: with ``display: "omitted"`` -- the default on most current
    models -- a thinking block streams one empty ``thinking_delta`` and a
    signature, and neither is answer text.

    A call with the server-side search tool also yields ``server_tool_use``
    blocks (the query, streamed as ``input_json_delta``) and
    ``web_search_tool_result`` blocks (whole in ``content_block_start``), kept
    for the search ledger; so is a client ``tool_use`` block, which is how an
    endpoint that does not run the search hands it back. Thinking blocks are
    not kept: nothing re-sends this turn.
    """

    _KEPT_BLOCKS = frozenset({"text", "server_tool_use", "web_search_tool_result", "tool_use"})

    def __init__(self, progress: StreamProgress) -> None:
        super().__init__(progress)
        self._message: Dict[str, Any] = {}
        self._usage: Dict[str, Any] = {}
        # index -> (the block as started, its streamed text or input JSON pieces)
        self._blocks: Dict[int, tuple[Dict[str, Any], List[str]]] = {}
        self._stop_reason: Optional[str] = None
        self._stopped = False

    def on_event(self, event: str, data: str) -> None:
        body = self._json(data)
        kind = body.get("type") or event
        if kind == "error":
            raise _anthropic_stream_error(body.get("error"))
        if kind == "message_start":
            message = body.get("message") or {}
            self._message = {key: message[key] for key in ("id", "model") if key in message}
            if isinstance(message.get("usage"), dict):
                self._usage.update(message["usage"])
        elif kind == "content_block_start":
            block = dict(body.get("content_block") or {})
            if block.get("type") in self._KEPT_BLOCKS:
                first = str(block.pop("text", "") or "") if block.get("type") == "text" else ""
                self._blocks[int(body.get("index", 0))] = (block, [first] if first else [])
        elif kind == "content_block_delta":
            delta = body.get("delta") or {}
            index = int(body.get("index", 0))
            if delta.get("type") == "text_delta":
                text = str(delta.get("text") or "")
                self._blocks.setdefault(index, ({"type": "text"}, []))[1].append(text)
                if text:
                    self.progress.output(text)
            elif delta.get("type") == "input_json_delta" and index in self._blocks:
                self._blocks[index][1].append(str(delta.get("partial_json") or ""))
        elif kind == "message_delta":
            delta = body.get("delta") or {}
            if delta.get("stop_reason"):
                self._stop_reason = str(delta["stop_reason"])
            if isinstance(body.get("usage"), dict):
                self._usage.update(
                    {key: value for key, value in body["usage"].items() if value is not None}
                )
        elif kind == "message_stop":
            self._stopped = True

    def result(self) -> Dict[str, Any]:
        if not self._stopped:
            raise StreamInterrupted(
                f"stream interrupted: ended without message_stop (stop_reason={self._stop_reason!r})"
            )
        return {
            **self._message,
            "type": "message",
            "role": "assistant",
            "content": [self._block(index) for index in sorted(self._blocks)],
            "stop_reason": self._stop_reason,
            "usage": self._usage,
        }


    def _block(self, index: int) -> Dict[str, Any]:
        block, pieces = self._blocks[index]
        if block.get("type") == "text":
            return {**block, "text": "".join(pieces)}
        if pieces:
            # A tool block streams its input as JSON fragments after an empty
            # `input` in the start event.
            try:
                return {**block, "input": json.loads("".join(pieces))}
            except ValueError:
                return {**block, "input": {}}
        return block


def _anthropic_stream_error(error: Any) -> ProviderStreamError:
    kind = error.get("type") if isinstance(error, dict) else None
    return ProviderStreamError(
        _ANTHROPIC_ERROR_STATUS.get(str(kind), 500), {"type": "error", "error": error}
    )


#: How much of the already-received text a stream may repeat from its start
#: before it counts as the server restarting the generation. Gemma4 did this
#: three times in the 2026-10-23 probes, with first replayed chunks of 40 and
#: 71 characters (both in the fixtures) and 21 (p4-gemma-2, not kept);
#: an answer that coincidentally re-says its own opening 32 characters in
#: chunks lined up exactly with the original's is not a realistic case.
_REPLAY_CHARS = 32

#: Gemini `finishReason`s that stop the answer itself mid-generation; the
#: non-streamed reply carries no answer text for them.
_BLOCKED_FINISH = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"})


class GeminiAssembler(SSEAssembler):
    """`streamGenerateContent?alt=sse` chunks, rebuilt into one
    `generateContent` response.

    Terminal: a candidate's ``finishReason``, or ``promptFeedback.blockReason``
    (a blocked prompt has no candidates, hence no ``finishReason``). Top-level
    fields and each candidate's fields are the last value seen; parts are
    merged: adjacent text parts with the same ``thought`` flag join into one,
    and signature-only empty parts are dropped (nothing downstream reads
    ``thoughtSignature``, and Gemma4 joins parts with ``"\\n"``).

    `error_factory(code, body)` builds the exception for an in-stream error
    event, so the main path raises `GeminiAPIError` and Gemma4 its own key
    errors from the same assembler.
    """

    def __init__(self, progress: StreamProgress, error_factory: Callable[[int, str], Exception]) -> None:
        super().__init__(progress)
        self._error_factory = error_factory
        # Thread-local: the worker thread that feeds this has no reporter.
        self._reporter = current_reporter()
        self._top: Dict[str, Any] = {}
        self._candidates: Dict[int, Dict[str, Any]] = {}
        self._parts: Dict[int, List[Dict[str, Any]]] = {}
        self._text: Dict[int, str] = {}
        self._replay: Dict[int, int] = {}
        self._terminal = False
        self._grounding_warned = False

    def on_event(self, event: str, data: str) -> None:
        chunk = self._json(data)
        error = chunk.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            raise self._error_factory(code if isinstance(code, int) else 500, data)
        for key, value in chunk.items():
            if key != "candidates" and value is not None:
                self._top[key] = value
        feedback = chunk.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason"):
            self._terminal = True
        for candidate in chunk.get("candidates") or []:
            if isinstance(candidate, dict):
                self._candidate(candidate)

    def _candidate(self, candidate: Dict[str, Any]) -> None:
        index = int(candidate.get("index", 0))
        meta = self._candidates.setdefault(index, {})
        for key, value in candidate.items():
            if key == "content" or value is None:
                continue
            if key == "groundingMetadata" and "groundingMetadata" in meta:
                self._check_grounding(meta[key], value)
            meta[key] = value
        if candidate.get("finishReason"):
            self._terminal = True
        content = candidate.get("content") or {}
        if content.get("role"):
            meta["_role"] = content["role"]
        chunk_text = ""
        for part in content.get("parts") or []:
            if isinstance(part, dict):
                chunk_text += self._part(index, part)
        if chunk_text:
            self._watch_replay(index, chunk_text)
            self._text[index] = self._text.get(index, "") + chunk_text

    def _part(self, index: int, part: Dict[str, Any]) -> str:
        parts = self._parts.setdefault(index, [])
        clean = {key: value for key, value in part.items() if key != "thoughtSignature"}
        if "text" not in clean:
            if clean:
                parts.append(clean)
            return ""
        text = str(clean.get("text") or "")
        if not text:
            return ""
        thought = bool(clean.get("thought"))
        last = parts[-1] if parts else None
        if last is not None and "text" in last and bool(last.get("thought")) == thought:
            last["text"] += text
        else:
            parts.append(clean | {"text": text})
        if not thought:
            self.progress.output(text)
        return text

    def _watch_replay(self, index: int, text: str) -> None:
        """Gemma4 sometimes restarts its generation inside the same stream:
        the text begins again from the top, and appended it would say the
        opening twice. A chunk that matches the text so far from offset 0
        opens a cursor; matching on to `_REPLAY_CHARS` raises, a mismatch
        closes it."""

        seen = self._text.get(index, "")
        cursor = self._replay.pop(index, None)
        if cursor is None or seen[cursor : cursor + len(text)] != text:
            if not seen.startswith(text):
                return
            cursor = 0
        cursor += len(text)
        if cursor >= _REPLAY_CHARS:
            raise StreamInterrupted(
                f"stream interrupted: the server restarted its answer ({cursor} characters repeated from the start)"
            )
        self._replay[index] = cursor

    def _check_grounding(self, before: Any, after: Any) -> None:
        if self._grounding_warned or not (isinstance(before, dict) and isinstance(after, dict)):
            return
        for key in ("groundingChunks", "groundingSupports"):
            later = after.get(key) or []
            if any(item not in later for item in before.get(key) or []):
                self._grounding_warned = True
                self._reporter.warning(
                    "llm-stream-grounding-replaced",
                    f"A later chunk's groundingMetadata dropped earlier {key}; only the last "
                    "one is kept. The streamed grounding shape is unrecorded -- keep this stream "
                    "for a fixture.",
                )
                return

    def result(self) -> Dict[str, Any]:
        if not self._terminal:
            raise StreamInterrupted(
                "stream interrupted: ended without finishReason or blockReason"
            )
        data = dict(self._top)
        candidates = []
        for index in sorted(self._candidates):
            meta = dict(self._candidates[index])
            role = meta.pop("_role", "model")
            parts = self._parts.get(index, [])
            if str(meta.get("finishReason") or "").upper() in _BLOCKED_FINISH:
                # A non-streamed call blocked on its output returns no answer
                # text; streamed, the text before the block has already
                # arrived. Kept, a half answer would pass for a whole one and
                # skip the blocked-call handling (`client.is_prompt_blocked`).
                parts = [part for part in parts if part.get("thought") or "text" not in part]
            meta["content"] = {"parts": parts, "role": role}
            candidates.append(meta)
        if candidates:
            data["candidates"] = candidates
        return data
