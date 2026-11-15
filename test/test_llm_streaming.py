"""Streamed bodies assemble into the shape the non-streamed call returned.

The fixtures under `fixtures/llm_streams/` are real streams recorded by the
free-quota probes of 2026-10-23 (except the hand-written Anthropic one), with
long runs of identical events cut and signatures shortened; their first line
says where each came from. Hand-written streams are inline below and say which documented shape
they follow.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm import llm_runtime
from finesub.llm.client import (
    is_prompt_blocked,
    is_quota_or_rate_limit_error,
    is_retryable_provider_error,
    provider_status_code,
)
from finesub.llm.http import StreamInterrupted, StreamProgress
from finesub.llm.streaming import (
    AnthropicMessagesAssembler,
    GeminiAssembler,
    OpenAIChatAssembler,
    ProviderStreamError,
    SSEAssembler,
)

FIXTURES = Path(__file__).parent / "fixtures" / "llm_streams"


def _lines(name: str) -> list[str]:
    return (FIXTURES / name).read_text(encoding="utf-8").split("\n")


def _feed(assembler, lines):
    for line in lines:
        assembler.feed_line(line)
    return assembler.finish()


def _data_events(lines):
    return [
        json.loads(line[5:])
        for line in lines
        if line.startswith("data:") and line[5:].strip() != "[DONE]"
    ]


class _Recorder(SSEAssembler):
    def __init__(self) -> None:
        super().__init__(StreamProgress())
        self.events: list[tuple[str, str]] = []

    def on_event(self, event, data):
        self.events.append((event, data))

    def result(self):
        return {"events": self.events}


def test_sse_framing_follows_the_spec() -> None:
    lines = [
        ": keep-alive",
        "event: message_start",
        "data: {\"a\":",
        "data:  1}",
        "",
        # Gemini's error tail: bare text lines name unknown fields.
        "{",
        '  "error": {',
        "}",
        "",
        "data: last, never followed by a blank line",
    ]

    assert _feed(_Recorder(), lines)["events"] == [
        ("message_start", '{"a":\n 1}'),
        ("message", "last, never followed by a blank line"),
    ]


def test_openai_compat_stream_of_gemini_assembles_the_unary_shape() -> None:
    lines = _lines("gemini-openai-compat.sse")
    progress = StreamProgress()
    data = _feed(OpenAIChatAssembler(progress), lines)

    chunks = _data_events(lines)
    expected_text = "".join(
        choice["delta"].get("content") or ""
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    (choice,) = data["choices"]
    assert choice["message"]["content"] == expected_text
    assert choice["finish_reason"] == "stop"
    # Usage rides on every chunk, cumulatively: the last one is the call's,
    # and it matches the unary twin's prompt count (P0 ⑦).
    assert data["usage"] == chunks[-1]["usage"]
    assert data["usage"]["prompt_tokens"] == 88
    # The closing chunk carries only a thought signature: not answer text.
    assert progress.output_chars == len(expected_text)


def test_reasoning_only_stream_is_an_empty_answer_cut_at_length() -> None:
    """OpenRouter: reasoning in `delta.reasoning`, `finish_reason` repeated on
    the usage chunk, keep-alive comments. An answer that spent the whole cap
    thinking reaches downstream as empty + `length`, the existing truncation."""

    progress = StreamProgress()
    data = _feed(OpenAIChatAssembler(progress), _lines("openrouter-reasoning-length.sse"))

    (choice,) = data["choices"]
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 16384
    assert progress.first_output_at is None, "reasoning never starts the stall clock"


def _chunk(content=None, finish=None, usage=None) -> str:
    choices = []
    if content is not None or finish is not None:
        choices.append({"index": 0, "delta": {"content": content}, "finish_reason": finish})
    body = {"choices": choices}
    if usage is not None:
        body["usage"] = usage
    return "data: " + json.dumps(body)


def test_a_close_between_finish_and_usage_is_not_a_success() -> None:
    """The usage chunk follows the finish chunk; stopping between the two
    would book the call at zero tokens."""

    lines = [_chunk("hi"), "", _chunk(finish="stop"), ""]
    with pytest.raises(StreamInterrupted):
        _feed(OpenAIChatAssembler(StreamProgress()), lines)


def test_finish_and_usage_without_done_is_complete() -> None:
    lines = [_chunk("hi"), "", _chunk(finish="stop", usage={"prompt_tokens": 1}), ""]
    data = _feed(OpenAIChatAssembler(StreamProgress()), lines)
    assert data["choices"][0]["message"]["content"] == "hi"


def test_an_error_event_carries_its_status_not_the_200() -> None:
    """OpenRouter's documented mid-stream error: a top-level `error` on a 200."""

    lines = [
        _chunk("partial"),
        "",
        'data: {"error": {"code": "server_error", "message": "Provider disconnected unexpectedly"},'
        ' "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}',
        "",
    ]
    with pytest.raises(ProviderStreamError) as raised:
        _feed(OpenAIChatAssembler(StreamProgress()), lines)

    error = raised.value
    assert provider_status_code(error) == 500
    assert is_retryable_provider_error(error)
    assert llm_runtime._status_code_from_exception(error) == "500"
    assert error.retry_after_seconds is None


def test_an_exhausted_account_inside_a_stream_is_a_quota_error() -> None:
    lines = ['data: {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}}', ""]
    with pytest.raises(ProviderStreamError) as raised:
        _feed(OpenAIChatAssembler(StreamProgress()), lines)
    assert provider_status_code(raised.value) == 429
    assert is_quota_or_rate_limit_error(raised.value)


def test_omitted_thinking_never_starts_the_stall_clock() -> None:
    """Anthropic's default `display: "omitted"`: one empty `thinking_delta` and
    a signature per thinking block. Only the text block is answer."""

    progress = StreamProgress()
    data = _feed(AnthropicMessagesAssembler(progress), _lines("anthropic-omitted-thinking.sse"))

    assert data["content"] == [{"type": "text", "text": "答案"}]
    assert data["stop_reason"] == "max_tokens"
    assert data["usage"]["input_tokens"] == 80
    assert data["usage"]["output_tokens"] == 50
    assert progress.output_chars == 2
    # Every event, ping and the empty thinking delta included, proves liveness.
    assert progress.events == 12


def _anthropic_stream(*events: dict) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines += [f"event: {event['type']}", "data: " + json.dumps(event), ""]
    return lines


_MESSAGE_START = {"type": "message_start", "message": {"id": "m", "model": "x", "usage": {"input_tokens": 3}}}
_TEXT = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "half"}}


def test_anthropic_overloaded_mid_stream_is_a_retryable_529() -> None:
    lines = _anthropic_stream(
        _MESSAGE_START,
        _TEXT,
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    with pytest.raises(ProviderStreamError) as raised:
        _feed(AnthropicMessagesAssembler(StreamProgress()), lines)

    assert provider_status_code(raised.value) == 529
    assert is_retryable_provider_error(raised.value)
    assert llm_runtime._status_code_from_exception(raised.value) == "529"


def test_anthropic_rate_limit_mid_stream_is_a_429() -> None:
    lines = _anthropic_stream({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    with pytest.raises(ProviderStreamError) as raised:
        _feed(AnthropicMessagesAssembler(StreamProgress()), lines)
    assert provider_status_code(raised.value) == 429
    assert is_quota_or_rate_limit_error(raised.value)


def test_anthropic_stream_without_message_stop_is_interrupted() -> None:
    lines = _anthropic_stream(
        _MESSAGE_START,
        _TEXT,
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 9}},
    )
    with pytest.raises(StreamInterrupted):
        _feed(AnthropicMessagesAssembler(StreamProgress()), lines)


def _gemini(progress=None) -> GeminiAssembler:
    return GeminiAssembler(progress or StreamProgress(), llm_runtime._gemini_api_error)


def test_gemini_stream_assembles_the_generate_content_shape() -> None:
    lines = _lines("gemini-thinking.sse")
    progress = StreamProgress()
    data = _feed(_gemini(progress), lines)

    chunks = _data_events(lines)
    answer = "".join(
        part["text"]
        for chunk in chunks
        for part in chunk["candidates"][0]["content"]["parts"]
        if not part.get("thought")
    )
    (candidate,) = data["candidates"]
    parts = candidate["content"]["parts"]
    # Thought summaries merge into one part, the answer into another; the
    # closing signature-only part is gone.
    assert [bool(part.get("thought")) for part in parts] == [True, False]
    assert parts[1]["text"] == answer
    assert all("thoughtSignature" not in part for part in parts)
    assert candidate["finishReason"] == "STOP"
    assert llm_runtime.extract_message_content(data) == answer
    # What downstream reads beyond the text (stages/correction/metadata.py,
    # degeneration.py): the last chunk's values.
    assert data["usageMetadata"] == chunks[-1]["usageMetadata"]
    assert data["usageMetadata"]["thoughtsTokenCount"] == 32853
    assert data["modelVersion"] == "gemini-3.5-flash-lite"
    assert data["responseId"] == chunks[-1]["responseId"]
    assert progress.output_chars == len(answer)


def test_streamed_grounding_lines_up_with_the_assembled_text() -> None:
    """Recorded on gemini-2.5-flash: the whole groundingMetadata arrives once,
    on the finishReason chunk, and its support offsets index the full answer
    -- so the merged text must be byte-for-byte what was streamed."""

    from finesub.llm.web_search import gemini_grounding_chunks, gemini_grounding_metadata

    lines = _lines("gemini-grounded.sse")
    data = _feed(_gemini(), lines)

    grounding = gemini_grounding_metadata(data)
    assert grounding == _data_events(lines)[-1]["candidates"][0]["groundingMetadata"]
    assert len(gemini_grounding_chunks(grounding)) == 13
    text = llm_runtime.extract_message_content(data).encode("utf-8")
    for support in grounding["groundingSupports"]:
        segment = support["segment"]
        assert text[segment.get("startIndex", 0) : segment["endIndex"]].decode("utf-8") == segment["text"]


@pytest.mark.parametrize("name", ["gemma4-restart-503.sse", "gemma4-grounded-restart-503.sse"])
def test_gemma4_restarting_its_answer_is_an_interrupted_stream(name) -> None:
    """Recorded: Gemma4 began again from the top inside the same stream;
    appended, the opening would be said twice."""

    with pytest.raises(StreamInterrupted, match="restarted"):
        _feed(_gemini(), _lines(name))


def _gemini_chunks(*texts: str, finish: bool = True) -> list[str]:
    lines = []
    for i, text in enumerate(texts):
        candidate = {"content": {"parts": [{"text": text}], "role": "model"}, "index": 0}
        if finish and i == len(texts) - 1:
            candidate["finishReason"] = "STOP"
        lines += ["data: " + json.dumps({"candidates": [candidate]}), ""]
    return lines


def test_a_short_replayed_first_chunk_is_still_caught() -> None:
    """The second recorded restart began with a 21-character chunk; the
    cursor counts the replay across chunks."""

    opening = "The quick brown fox jumps over the lazy dog. "
    lines = _gemini_chunks(opening, "Then more.", opening[:21], opening[21:40], "x")
    with pytest.raises(StreamInterrupted, match="restarted"):
        _feed(_gemini(), lines)


def test_an_answer_that_merely_repeats_its_first_word_is_not_a_restart() -> None:
    data = _feed(_gemini(), _gemini_chunks("The answer is 42. ", "The", " end, and the answer stands."))
    assert llm_runtime.extract_message_content(data) == "The answer is 42. The end, and the answer stands."


def test_gemini_error_event_after_200_keeps_its_code() -> None:
    lines = _lines("gemma4-grounded-500.sse")
    with pytest.raises(llm_runtime.GeminiAPIError) as raised:
        _feed(_gemini(), lines)

    assert raised.value.status_code == 500
    assert provider_status_code(raised.value) == 500
    assert is_retryable_provider_error(raised.value)
    assert "Internal error encountered" in str(raised.value)


def test_a_daily_quota_inside_a_stream_classifies_as_daily() -> None:
    """The main path's in-stream 429 keeps the unary error text, so the
    structured `quotaId` still decides daily vs per-minute."""

    from finesub.llm.client import QuotaKind, classify_quota_error

    error = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}],
                }
            ],
        }
    }
    with pytest.raises(llm_runtime.GeminiAPIError) as raised:
        _feed(_gemini(), ["data: " + json.dumps(error), ""])

    assert str(raised.value).startswith("Gemini API error (HTTP 429): ")
    assert raised.value.status_code == 429
    assert classify_quota_error(raised.value) is QuotaKind.DAILY


def test_a_blocked_prompt_ends_the_stream_without_candidates() -> None:
    """Hand-written from the generateContent reference: a blocked prompt has
    `promptFeedback.blockReason` and no candidates, hence no finishReason."""

    lines = [
        'data: {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"},'
        ' "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12}, "modelVersion": "gemini-3.5-flash-lite"}',
        "",
    ]
    data = _feed(_gemini(), lines)
    assert "candidates" not in data
    assert is_prompt_blocked("", data)


@pytest.mark.parametrize("reason", ["SAFETY", "RECITATION"])
def test_an_answer_stopped_mid_generation_reaches_downstream_without_its_text(reason) -> None:
    """Streamed, the text before an output-side stop has already arrived; the
    non-streamed shape has none."""

    lines = _gemini_chunks("half of an answer that then got ", finish=False) + [
        'data: {"candidates": [{"content": {"parts": [{"text": "flag"}], "role": "model"},'
        f' "finishReason": "{reason}", "index": 0}}]}}',
        "",
    ]
    data = _feed(_gemini(), lines)

    assert data["candidates"][0]["content"]["parts"] == []
    assert data["candidates"][0]["finishReason"] == reason
    # SAFETY is a blocked call downstream; RECITATION an empty answer.
    assert is_prompt_blocked("", data) is (reason == "SAFETY")


def test_a_gemini_stream_without_finish_reason_is_interrupted() -> None:
    with pytest.raises(StreamInterrupted):
        _feed(_gemini(), _gemini_chunks("half an answer", finish=False))


def _grounded(text: str, chunks: list[str], supports: list[int], finish: bool = False) -> list[str]:
    candidate = {
        "content": {"parts": [{"text": text}], "role": "model"},
        "index": 0,
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": uri}} for uri in chunks],
            "groundingSupports": [{"groundingChunkIndices": [i]} for i in supports],
        },
    }
    if finish:
        candidate["finishReason"] = "STOP"
    return ["data: " + json.dumps({"candidates": [candidate]}), ""]


def test_grounding_that_grows_chunk_by_chunk_is_kept_quietly() -> None:
    from finesub.reporting import NullReporter, reporting_to

    warned: list[str] = []

    class _Warnings(NullReporter):
        def warning(self, code, message, *, impact="", action="") -> None:
            warned.append(code)

    lines = _grounded("A", ["u1"], [0]) + _grounded("B", ["u1", "u2"], [0, 1], finish=True)
    with reporting_to(_Warnings()):
        assembler = _gemini()
    data = _feed(assembler, lines)
    assert warned == []
    assert len(data["candidates"][0]["groundingMetadata"]["groundingChunks"]) == 2

    replaced = _grounded("A", ["u1"], [0]) + _grounded("B", ["u2"], [0], finish=True)
    with reporting_to(_Warnings()):
        assembler = _gemini()
    _feed(assembler, replaced)
    assert warned == ["llm-stream-grounding-replaced"]


def test_an_interrupted_stream_is_retryable_with_its_own_code() -> None:
    error = StreamInterrupted("stream interrupted: peer closed")
    assert is_retryable_provider_error(error)
    assert llm_runtime._status_code_from_exception(error) == "STREAM_INTERRUPTED"
