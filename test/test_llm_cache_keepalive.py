"""The Anthropic prompt-cache keep-alive (`provider_transports._KeepAlive`) and how the books read it.

A call to Anthropic's own API that runs past `KEEPALIVE_SECONDS` gets a ``max_tokens: 0`` request beside it, the
same up to its last breakpoint, so the entry it wrote outlives the five minutes counted from its start.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest

from finesub.llm import provider_transports, token_books
from finesub.llm.client import extract_token_distribution
from finesub.llm.http import StreamProgress

from .support.fakes import WarningCodeRecorder


STREAMS = Path(__file__).parent / "fixtures" / "llm_streams"
TICK = 0.02
SYSTEM = {"role": "system", "content": "规则" * 50}
OPENING = {"role": "user", "content": "窗口正文"}
READ = {"input_tokens": 3, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0, "output_tokens": 0}


class _Endpoint:
    """Stands in for `llm_http_client`: each keep-alive's body, answered from `replies` (the last one repeats)."""

    def __init__(self, *replies) -> None:
        self.bodies: list[dict] = []
        self.replies = list(replies) or [READ]
        self.posted = threading.Event()

    def __call__(self, timeout=None, **kwargs):
        endpoint = self

        class _Client:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, *, headers, json):
                endpoint.bodies.append(json)
                reply = endpoint.replies[min(len(endpoint.bodies), len(endpoint.replies)) - 1]
                endpoint.posted.set()
                if isinstance(reply, Exception):
                    raise reply
                if isinstance(reply, int):
                    return httpx.Response(reply, json={"error": {}})
                return httpx.Response(200, json={"content": [], "stop_reason": "max_tokens", "usage": reply})

        return _Client()


def _generate(monkeypatch, *, keepalives: int = 1, messages=None, base_url="https://api.anthropic.com",
              endpoint=None, fail=None, **kwargs):
    """`anthropic_generate` whose stream runs until `keepalives` keep-alives went out (or 1 s), then answers
    or raises `fail`."""

    endpoint = endpoint or _Endpoint()
    seen: dict = {}
    lines = (STREAMS / "anthropic-omitted-thinking.sse").read_text(encoding="utf-8").split("\n")

    def fake_stream(url, *, headers, payload, assembler, timeout):
        seen["payload"] = dict(payload)
        deadline = time.monotonic() + 1.0
        while len(endpoint.bodies) < keepalives and time.monotonic() < deadline:
            time.sleep(TICK / 4)
        if fail is not None:
            raise fail
        for line in lines:
            assembler.feed_line(line)
        return assembler.finish()

    monkeypatch.setattr(provider_transports, "KEEPALIVE_SECONDS", TICK)
    monkeypatch.setattr(provider_transports, "llm_http_client", endpoint)
    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)
    progress = StreamProgress()
    response = provider_transports.anthropic_generate(
        base_url=base_url, api_key="sk-ant", model="claude-haiku-5-5",
        messages=messages if messages is not None else [SYSTEM, OPENING],
        max_tokens=64_000, thinking_level="high", timeout=30.0, progress=progress, **kwargs,
    )
    return response, seen["payload"], endpoint, progress


def test_the_keepalive_is_the_call_up_to_its_last_breakpoint_asking_for_nothing(monkeypatch) -> None:
    turns = [SYSTEM, OPENING, {"role": "assistant", "content": "初稿"}, {"role": "user", "content": "校对"}]
    _, payload, endpoint, _ = _generate(monkeypatch, messages=turns, web_search=True)
    body = endpoint.bodies[0]
    # what is part of the cached prefix goes as sent: model, system, tools, effort and the marked opening
    for key in ("model", "system", "tools", "output_config"):
        assert body[key] == payload[key]
    assert body["messages"] == payload["messages"][:1]
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    # nothing to answer, and no stream: the API rejects max_tokens 0 with one
    assert body["max_tokens"] == 0 and "stream" not in body


def test_an_opening_the_next_turn_resends_is_kept_whole(monkeypatch) -> None:
    _, payload, endpoint, _ = _generate(monkeypatch, followup_expected=True)
    assert endpoint.bodies[0]["messages"] == payload["messages"]


def test_with_only_the_system_marked_a_placeholder_turn_stands_in(monkeypatch) -> None:
    _, payload, endpoint, _ = _generate(monkeypatch)
    # the window's text after the last breakpoint would bill as fresh input
    assert payload["messages"] == [OPENING]
    assert endpoint.bodies[0]["messages"] == [{"role": "user", "content": "."}]


def test_nothing_marked_sends_no_keepalive(monkeypatch) -> None:
    _, _, endpoint, progress = _generate(monkeypatch, keepalives=0, messages=[OPENING])
    time.sleep(TICK * 5)
    assert endpoint.bodies == [] and progress.keepalive == {}


def test_an_anthropic_compatible_endpoint_gets_none(monkeypatch) -> None:
    _, _, endpoint, _ = _generate(monkeypatch, keepalives=0, base_url="https://api.deepseek.com/anthropic")
    time.sleep(TICK * 5)
    assert endpoint.bodies == []


def test_keepalives_repeat_until_the_call_ends_and_stay_out_of_its_usage(monkeypatch) -> None:
    response, _, endpoint, progress = _generate(monkeypatch, keepalives=3)
    sent = len(endpoint.bodies)
    time.sleep(TICK * 5)
    assert sent >= 3 and len(endpoint.bodies) <= sent + 1  # one may have been on its way
    # the answer's usage is the answer's own: calibration and the rate limiter read it
    assert extract_token_distribution(response)["prompt_tokens"] == 105
    books = progress.keepalive
    # one counted on its way out may not have brought its usage back yet
    assert books["requests"] >= 3 and books["cached_input_tokens"] % 900 == 0 and not books.get("failed")
    assert 900 * (books["requests"] - 1) <= books["cached_input_tokens"] <= 900 * books["requests"]
    assert progress.record()["keepalive"] == books


@pytest.mark.parametrize("fail", [provider_transports.SearchTurnPaused("x"), TimeoutError("deadline")])
def test_a_call_that_fails_stops_its_keepalives_too(monkeypatch, fail) -> None:
    endpoint = _Endpoint()
    with pytest.raises(type(fail)):
        _generate(monkeypatch, keepalives=1, endpoint=endpoint, fail=fail)
    sent = len(endpoint.bodies)
    time.sleep(TICK * 5)
    assert len(endpoint.bodies) <= sent + 1


def test_a_failed_keepalive_is_counted_and_the_next_one_still_goes(monkeypatch) -> None:
    endpoint = _Endpoint(httpx.ConnectError("down"), 529, READ)
    response, _, _, progress = _generate(monkeypatch, keepalives=3, endpoint=endpoint)
    assert response["choices"][0]["message"]["content"] == "答案"
    assert progress.keepalive["failed"] == 2 and progress.keepalive["cached_input_tokens"] >= 900


@pytest.mark.parametrize("reply", [400, {**READ, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 900}])
def test_a_rejected_keepalive_or_one_that_did_not_read_stops_with_one_warning(monkeypatch, reply) -> None:
    from finesub.reporting import reporting_to

    warnings: list[str] = []

    endpoint = _Endpoint(reply)
    with reporting_to(WarningCodeRecorder(warnings)):
        _generate(monkeypatch, keepalives=3, endpoint=endpoint)
    assert len(endpoint.bodies) == 1 and warnings == ["llm-cache-keepalive-stopped"]


def test_a_later_turn_marks_only_the_opening_however_a_next_turn_is_announced(monkeypatch) -> None:
    # a request turn's fill or a repair re-sends past the opening; its next turn re-sends the opening alone
    turns = [SYSTEM, OPENING, {"role": "assistant", "content": "请求"}, {"role": "user", "content": "补的材料"}]
    _, payload, _, _ = _generate(monkeypatch, messages=turns, followup_expected=True)
    marked = [i for i, turn in enumerate(payload["messages"]) if isinstance(turn["content"], list)]
    assert marked == [0]


def test_the_books_charge_keepalives_to_the_call_and_to_a_failed_calls_attempts() -> None:
    keepalive = {"requests": 1, "cached_input_tokens": 900, "uncached_input_tokens": 3, "total_input_tokens": 903}
    usage = {"uncached_input_tokens": 10, "cached_input_tokens": 100, "total_input_tokens": 110, "output_tokens": 5}
    answered = {"kind": "correction_window_response", "payload": {
        "chunk_id": "0001", "usage": usage,
        "execution_attempts": [{"backend": "api", "return_code": "SUPERSEDED", "keepalive": keepalive},
                               {"backend": "api", "return_code": "200", "keepalive": keepalive}]}}
    [call] = token_books.book([answered])
    assert call.role == "call" and call.tokens["cached_input_tokens"] == 100 + 2 * 900
    assert call.tokens["total_input_tokens"] == 110 + 2 * 903 and call.tokens["output_tokens"] == 5
    failed = {"kind": "correction_window_call_error", "payload": {
        "chunk_id": "0001", "execution_attempts": [{"backend": "api", "return_code": "NO_RESPONSE_TIMEOUT",
                                                     "keepalive": keepalive}]}}
    [attempt] = token_books.book([failed])
    assert attempt.role == "unanswered" and attempt.tokens["cached_input_tokens"] == 900
    # answered by the group's next model: the first one's keep-alives stay on the first one
    fallback = {"kind": "correction_window_response", "payload": {
        "chunk_id": "0001", "usage": usage,
        "execution_attempts": [{"backend": "api", "target_id": "claude", "return_code": "NO_RESPONSE_TIMEOUT",
                                "keepalive": keepalive},
                               {"backend": "api", "target_id": "gemini", "return_code": "200"}]}}
    call, earlier = token_books.book([fallback])
    assert call.role == "call" and call.tokens["cached_input_tokens"] == 100
    assert earlier.role == "unanswered" and earlier.tokens["cached_input_tokens"] == 900


def test_a_reply_body_that_is_not_an_object_stops_nothing_but_itself(monkeypatch) -> None:
    class _List(_Endpoint):
        def __call__(self, timeout=None, **kwargs):
            client = super().__call__(timeout, **kwargs)
            post = client.post
            client.post = lambda url, *, headers, json: (post(url, headers=headers, json=json),
                                                         httpx.Response(200, json=[]))[1]
            return client

    response, _, endpoint, progress = _generate(monkeypatch, keepalives=2, endpoint=_List())
    assert response["choices"][0]["message"]["content"] == "答案" and len(endpoint.bodies) >= 1
