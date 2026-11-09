"""`api_http.HostBreaker`: one silent connect timeout per blocked host, not one per call.

A host the network will not let through (the Great Firewall's dropped SYN or reset handshake) used to cost every
call its own connect timeout, with nothing on the console. These pin the rules that replace that: the first failure
is said at once, a run lasting 30 s is the verdict, optional callers then skip it except for a periodic probe,
required callers try once, and any response clears it. The clock and the failures are fakes; only the CONNECT
tests open a socket, to a silent proxy on 127.0.0.1.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

import httpx
import pytest

from finesub import api_http
from finesub.api_http import (
    Gate,
    HostBreaker,
    HostUnreachableError,
    Route,
    current_route,
    is_connect_failure,
)
from finesub.llm import token_budget
from finesub.llm.token_budget import (
    FallbackTokenCounter,
    GeminiCountTokensCounter,
    HeuristicTokenCounter,
)

HOST = "generativelanguage.googleapis.com"
ROUTE = Route(key="direct", label="直连")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def warnings(monkeypatch) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(api_http, "warn_visibly", lambda code, message, **_: seen.append((code, message)))
    return seen


def _blocked() -> httpx.ConnectTimeout:
    return httpx.ConnectTimeout("timed out")


def _fail(breaker: HostBreaker, clock: Clock, *, after: float, took: float = 5.0) -> None:
    """A request sent `after` seconds from now that failed to connect `took` seconds later."""

    clock.now += after
    sent_at = clock.now
    clock.now += took
    breaker.note_connect_failure(HOST, ROUTE, _blocked(), sent_at=sent_at)


def test_connect_failures_are_told_apart_from_slow_answers() -> None:
    assert is_connect_failure(httpx.ConnectTimeout("x"))
    assert is_connect_failure(httpx.ConnectError("[Errno 10054] reset"))
    assert is_connect_failure(httpx.ProxyError("502 from proxy"))
    try:
        try:
            raise httpx.ConnectError("dns")
        except httpx.ConnectError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        assert is_connect_failure(wrapped)
    # The host answered the handshake: slow, not blocked.
    assert not is_connect_failure(httpx.ReadTimeout("x"))
    assert not is_connect_failure(httpx.PoolTimeout("x"))
    assert not is_connect_failure(RuntimeError("timed out"))  # by type, never by text


def test_first_failure_is_reported_at_once_and_verdict_needs_thirty_seconds(warnings) -> None:
    clock = Clock()
    breaker = HostBreaker(clock)

    _fail(breaker, clock, after=0, took=5)
    assert [code for code, _ in warnings] == ["host-connect-failed"]
    assert "直连" in warnings[0][1] and "ConnectTimeout" in warnings[0][1]
    assert not breaker.unreachable(HOST, ROUTE)

    _fail(breaker, clock, after=10, took=5)  # 20 s since the run began
    assert not breaker.unreachable(HOST, ROUTE)
    assert len(warnings) == 1  # one fact per run, not per failure

    _fail(breaker, clock, after=5, took=5)  # 30 s
    assert breaker.unreachable(HOST, ROUTE)
    assert [code for code, _ in warnings] == ["host-connect-failed", "host-unreachable"]
    assert breaker.gate_required(HOST, ROUTE) is Gate.ONCE


def test_failures_far_apart_do_not_add_up_before_the_verdict(warnings) -> None:
    clock = Clock()
    breaker = HostBreaker(clock)

    _fail(breaker, clock, after=0, took=1)
    _fail(breaker, clock, after=20 * 60, took=1)  # a second blip twenty minutes later

    assert not breaker.unreachable(HOST, ROUTE)
    assert [code for code, _ in warnings] == ["host-connect-failed", "host-connect-failed"]


def _judged(clock: Clock) -> HostBreaker:
    breaker = HostBreaker(clock)
    _fail(breaker, clock, after=0, took=15)
    _fail(breaker, clock, after=0, took=15)
    assert breaker.unreachable(HOST, ROUTE)
    return breaker


def _probe(breaker: HostBreaker, clock: Clock, *, took: float = 5.0) -> None:
    """One probe, granted now, that failed to connect `took` seconds later."""

    gate = breaker.gate_optional(HOST, ROUTE)
    assert gate is Gate.PROBE
    with breaker.probing(HOST, ROUTE, gate):
        sent_at = clock.now
        clock.now += took
        breaker.note_connect_failure(HOST, ROUTE, _blocked(), sent_at=sent_at)


def test_optional_callers_skip_and_probe_on_a_doubling_interval(warnings) -> None:
    clock = Clock()
    breaker = _judged(clock)

    assert breaker.gate_optional(HOST, ROUTE) is Gate.SKIP
    clock.now += 30
    _probe(breaker, clock, took=5)
    clock.now += 59
    assert breaker.gate_optional(HOST, ROUTE) is Gate.SKIP  # 59 s after the failed probe: the interval is now 60 s
    clock.now += 1
    assert breaker.gate_optional(HOST, ROUTE) is Gate.PROBE


def test_probe_interval_is_capped_at_ten_minutes(warnings) -> None:
    clock = Clock()
    breaker = _judged(clock)
    for _ in range(10):
        clock.now += api_http.PROBE_INTERVAL_MAX_SECONDS
        _probe(breaker, clock, took=0)
    clock.now += api_http.PROBE_INTERVAL_MAX_SECONDS
    assert breaker.gate_optional(HOST, ROUTE) is Gate.PROBE


def test_a_probe_holds_its_slot_until_its_holder_is_done(warnings) -> None:
    """A probe slower than the interval (several addresses, a slow CONNECT) must not let a second one out, and a
    required call failing meanwhile is not the probe ending."""

    clock = Clock()
    breaker = _judged(clock)
    clock.now += 30
    gate = breaker.gate_optional(HOST, ROUTE)
    assert gate is Gate.PROBE
    with pytest.raises(httpx.ReadTimeout):
        with breaker.probing(HOST, ROUTE, gate):
            assert breaker.gate_optional(HOST, ROUTE) is Gate.SKIP
            clock.now += 1
            other = clock.now
            clock.now += 1
            with pytest.raises(httpx.PoolTimeout):
                with api_http.track_reach(f"https://{HOST}/x", breaker=breaker, route=ROUTE):
                    raise httpx.PoolTimeout("pool")
            breaker.note_connect_failure(HOST, ROUTE, _blocked(), sent_at=other)
            clock.now += 600  # long past any due time
            assert breaker.gate_optional(HOST, ROUTE) is Gate.SKIP
            raise httpx.ReadTimeout("the probe itself ends in a read timeout")
    assert breaker.gate_optional(HOST, ROUTE) is Gate.PROBE
    assert breaker.unreachable(HOST, ROUTE)


def test_an_error_status_raised_inside_the_block_is_an_answer(warnings) -> None:
    """`raise_for_status` on a 503 leaves the block as an exception, but the host answered."""

    clock = Clock()
    breaker = _judged(clock)
    request = httpx.Request("POST", f"https://{HOST}/upload")
    with pytest.raises(httpx.HTTPStatusError):
        with api_http.track_reach(f"https://{HOST}/upload", breaker=breaker, route=ROUTE):
            httpx.Response(503, request=request).raise_for_status()
    assert not breaker.unreachable(HOST, ROUTE)


def test_the_verdict_outlives_a_quiet_spell(warnings) -> None:
    """The 60 s gap rule builds a run; it does not undo a verdict nobody tested."""

    clock = Clock()
    breaker = _judged(clock)
    clock.now += 11 * 60
    assert breaker.unreachable(HOST, ROUTE)
    assert breaker.gate_required(HOST, ROUTE) is Gate.ONCE


def test_any_response_clears_the_verdict(warnings) -> None:
    clock = Clock()
    breaker = _judged(clock)

    breaker.note_reached(HOST, ROUTE)

    assert not breaker.unreachable(HOST, ROUTE)
    assert breaker.gate_optional(HOST, ROUTE) is Gate.ALLOW
    assert warnings[-1][0] == "host-reachable"


def test_a_failure_sent_before_the_last_success_says_nothing_about_now(warnings) -> None:
    clock = Clock()
    breaker = HostBreaker(clock)
    early = clock.now
    clock.now += 40
    breaker.note_reached(HOST, ROUTE)  # a later-sent request got through
    breaker.note_connect_failure(HOST, ROUTE, _blocked(), sent_at=early)

    assert not breaker.unreachable(HOST, ROUTE)
    assert warnings == []


def test_a_new_route_is_a_new_host(warnings) -> None:
    clock = Clock()
    breaker = _judged(clock)
    via_proxy = Route(key="proxy:http://127.0.0.1:7890", label="经代理 http://127.0.0.1:7890")

    assert breaker.gate_optional(HOST, via_proxy) is Gate.ALLOW


def test_route_follows_what_httpx_reads(monkeypatch) -> None:
    monkeypatch.setattr(api_http, "configured_proxy", lambda: None)
    monkeypatch.setattr(api_http.urllib.request, "getproxies", lambda: {})
    assert current_route() == Route(key="direct", label="直连")

    # Switching on a VPN client's system proxy is what `getproxies` reports on Windows.
    monkeypatch.setattr(api_http.urllib.request, "getproxies", lambda: {"https": "http://127.0.0.1:7890"})
    system = current_route()
    assert system.key != "direct" and "127.0.0.1:7890" in system.label

    monkeypatch.setattr(api_http, "configured_proxy", lambda: "http://10.0.0.2:8080")
    assert current_route().label == "经代理 http://10.0.0.2:8080"
    monkeypatch.setattr(api_http, "configured_proxy", lambda: "direct")
    assert current_route().key == "direct"


def test_unreachable_error_declares_itself_unavailable_to_the_route_policy() -> None:
    from finesub.llm.routing.model_router import FailureKind, classify_failure

    error = HostUnreachableError(HOST, ROUTE, _blocked())

    assert classify_failure(error) is FailureKind.UNAVAILABLE
    assert HOST in str(error) and "[api] proxy" in str(error)


def test_warning_reaches_stderr_without_a_bound_reporter(capsys) -> None:
    api_http.warn_visibly("host-connect-failed", "连不上 x")
    assert "Warning: 连不上 x" in capsys.readouterr().err


# -- countTokens ----------------------------------------------------------------------------------------------------


class _BlockedClient:
    def __init__(self, calls: list[object], exc: Exception) -> None:
        self.calls = calls
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, **kwargs):
        self.calls.append(url)
        raise self.exc


def _count_tokens(breaker: HostBreaker, calls: list[object], exc: Exception, pauses: list[float]):
    return GeminiCountTokensCounter(
        api_key="k",
        client_factory=lambda timeout: _BlockedClient(calls, exc),
        breaker=breaker,
        sleep_func=pauses.append,
    )


def test_count_tokens_retries_a_connect_failure_once_then_gives_up(monkeypatch, warnings) -> None:
    monkeypatch.setattr(token_budget, "current_route", lambda: ROUTE)
    clock = Clock()
    breaker = HostBreaker(clock)
    calls: list[object] = []
    pauses: list[float] = []

    with pytest.raises(httpx.ConnectTimeout):
        _count_tokens(breaker, calls, _blocked(), pauses).count_text("hello")

    assert len(calls) == 2 and pauses == [1.0]
    assert [code for code, _ in warnings] == ["host-connect-failed"]


def test_count_tokens_sends_nothing_once_the_host_is_judged(monkeypatch, warnings) -> None:
    monkeypatch.setattr(token_budget, "current_route", lambda: ROUTE)
    clock = Clock()
    breaker = _judged(clock)
    calls: list[object] = []

    with pytest.raises(HostUnreachableError):
        _count_tokens(breaker, calls, _blocked(), []).count_text("hello")
    assert calls == []

    clock.now += 30  # the probe is due: exactly one request, no retry
    with pytest.raises(httpx.ConnectTimeout):
        _count_tokens(breaker, calls, _blocked(), []).count_text("again")
    assert len(calls) == 1


def test_count_tokens_read_timeout_is_not_a_connect_failure(monkeypatch, warnings) -> None:
    monkeypatch.setattr(token_budget, "current_route", lambda: ROUTE)
    breaker = HostBreaker(Clock())
    calls: list[object] = []

    with pytest.raises(httpx.ReadTimeout):
        _count_tokens(breaker, calls, httpx.ReadTimeout("slow"), []).count_text("hello")

    assert len(calls) == 1 and warnings == []


def test_count_tokens_uses_its_own_short_connect_timeout() -> None:
    timeout = GeminiCountTokensCounter().timeout
    assert timeout.connect == 5.0 and timeout.read == 15.0


def test_falling_back_to_the_heuristic_is_said_once_per_process(monkeypatch) -> None:
    said: list[str] = []
    monkeypatch.setattr(token_budget, "_HEURISTIC_WARNED", False)
    monkeypatch.setattr(token_budget, "warn_visibly", lambda code, message, **_: said.append(message))

    class Down:
        source = "gemini-countTokens"

        def count_text(self, text):
            raise RuntimeError("cannot reach host")

    counter = FallbackTokenCounter(counters=(Down(), HeuristicTokenCounter()))
    counter.count_text("one")
    counter.count_text("two")
    FallbackTokenCounter(counters=(Down(), HeuristicTokenCounter())).count_text("three")

    assert len(said) == 1 and "cannot reach host" in said[0]


# -- tokcount -------------------------------------------------------------------------------------------------------


def test_tokcount_that_cannot_start_is_left_alone_for_a_while(monkeypatch, tmp_path) -> None:
    starts: list[object] = []

    def popen(*args, **kwargs):
        starts.append(args)
        raise OSError("cannot start")

    monkeypatch.setattr(token_budget.subprocess, "Popen", popen)
    service = token_budget._LocalTokenCounterService(exe_path=str(tmp_path / "tokcount.exe"))

    with pytest.raises(RuntimeError, match="failed after restart"):
        service.request("x", timeout_seconds=1, idle_timeout_seconds=None)
    assert len(starts) == 2
    with pytest.raises(RuntimeError, match="not starting it again"):
        service.request("y", timeout_seconds=1, idle_timeout_seconds=None)
    assert len(starts) == 2


def test_tokcount_is_started_without_a_model_flag(monkeypatch, tmp_path) -> None:
    seen: list[list[str]] = []

    def popen(argv, **kwargs):
        seen.append(list(argv))
        raise OSError("stop here")

    monkeypatch.setattr(token_budget.subprocess, "Popen", popen)
    service = token_budget._LocalTokenCounterService(exe_path=str(tmp_path / "tokcount.exe"))
    with pytest.raises(RuntimeError):
        service.request("x", timeout_seconds=1, idle_timeout_seconds=None)

    assert seen[0][1:] == ["-server"]


# -- generation and uploads (required callers) ----------------------------------------------------------------------


def _gemini_once_key(monkeypatch, completion) -> None:
    from finesub.llm import llm_runtime

    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: {"GEMINI_FREE": "{free-main:key1}"})
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", completion)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(llm_runtime, "current_route", lambda: ROUTE)


def _judge_globally(warnings) -> None:
    """Put the process-wide breaker in the 'unreachable' state for the Gemini host."""

    breaker = api_http.HOST_BREAKER
    breaker.note_connect_failure(HOST, ROUTE, _blocked(), sent_at=breaker.now() - 40)
    assert breaker.unreachable(HOST, ROUTE)


def _chat(**kwargs):
    from finesub.llm import llm_runtime
    from finesub.llm.routing.config import GEMINI_FREE_TIER

    return llm_runtime.chat_complete(
        [{"role": "user", "content": "hi"}],
        provider_tier=GEMINI_FREE_TIER,
        model="gemini/gemini-3.5-flash-lite",
        **kwargs,
    )


def test_connect_timeouts_are_not_generation_timeouts(monkeypatch, warnings) -> None:
    """Two `ConnectTimeout`s used to read as two closed generations and abort the
    whole candidate chain; the host never answered, so they are retried and
    recorded as connect failures instead."""

    calls = 0

    def blocked(**kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out")

    _gemini_once_key(monkeypatch, blocked)

    with pytest.raises(httpx.ConnectTimeout) as raised:
        _chat(retries=2)

    assert calls == 3
    assert not getattr(raised.value, "_harness_consecutive_timeout_abort", False)
    codes = [item["return_code"] for item in getattr(raised.value, "_harness_api_attempts")]
    assert codes == ["NO_RESPONSE_CONNECT"] * 3


def test_a_judged_host_gets_one_attempt(monkeypatch, warnings) -> None:
    calls = 0

    def blocked(**kwargs):
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("[Errno 10054] reset")

    _gemini_once_key(monkeypatch, blocked)
    _judge_globally(warnings)

    with pytest.raises(HostUnreachableError) as raised:
        _chat(retries=3)

    assert calls == 1
    assert "[api] proxy" in str(raised.value)
    assert getattr(raised.value, "_harness_api_attempts")


def test_a_verdict_reached_while_backing_off_stops_the_retries(monkeypatch, warnings) -> None:
    calls = 0

    def blocked(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            _judge_globally(warnings)  # another caller's failures complete the run meanwhile
        raise httpx.ConnectError("refused")

    _gemini_once_key(monkeypatch, blocked)

    with pytest.raises(HostUnreachableError):
        _chat(retries=3)
    assert calls == 1


def test_an_answer_on_the_single_attempt_follows_the_usual_rules(monkeypatch, warnings) -> None:
    class Overloaded(RuntimeError):
        status_code = 503

    def answers(**kwargs):
        raise Overloaded("503 overloaded")

    _gemini_once_key(monkeypatch, answers)
    _judge_globally(warnings)

    with pytest.raises(Overloaded):
        _chat(retries=0)


def test_connect_failures_do_not_start_a_combo_cooldown(monkeypatch, warnings) -> None:
    from finesub.llm.rate_limit import ModelRateLimiter

    def blocked(**kwargs):
        raise httpx.ConnectError("refused")

    _gemini_once_key(monkeypatch, blocked)
    limiter = ModelRateLimiter(enabled=False)
    exhausted: list[object] = []
    monkeypatch.setattr(limiter, "note_combo_exhausted", lambda *a, **k: exhausted.append(a))

    with pytest.raises(httpx.ConnectError):
        _chat(retries=1, rate_limiter=limiter)

    assert exhausted == []


def test_a_probe_that_could_not_connect_hands_its_claim_back(monkeypatch, warnings) -> None:
    from finesub.llm.rate_limit import ComboCooldownPhase, ModelRateLimiter

    def blocked(**kwargs):
        raise httpx.ConnectError("refused")

    _gemini_once_key(monkeypatch, blocked)
    limiter = ModelRateLimiter(enabled=False)
    released: list[object] = []
    exhausted: list[object] = []
    monkeypatch.setattr(limiter, "combo_cooldown_phase", lambda *a, **k: ComboCooldownPhase.PROBE)
    monkeypatch.setattr(limiter, "claim_combo_probe", lambda *a, **k: True)
    monkeypatch.setattr(limiter, "release_combo_probe", lambda *a, **k: released.append(a))
    monkeypatch.setattr(limiter, "note_combo_exhausted", lambda *a, **k: exhausted.append(a))

    with pytest.raises(httpx.ConnectError):
        _chat(retries=3, rate_limiter=limiter)

    assert len(released) == 1 and exhausted == []


def test_a_chain_does_not_pay_one_host_twice(monkeypatch) -> None:
    from finesub.llm.client import LLMRole, RoleClient
    from finesub.llm.rate_limit import ModelRateLimiter

    calls: list[str] = []

    def fake_chat_complete(messages, *, model, **kwargs):
        calls.append(model)
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    with pytest.raises(httpx.ConnectTimeout) as raised:
        client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}])

    assert calls == ["gemini/gemini-3.6-flash"]
    reasons = [item.get("reason") for item in raised.value._harness_route_decision["candidates"]]
    assert "host_unreachable" in reasons


def test_an_upload_to_a_judged_host_is_tried_once(monkeypatch, warnings) -> None:
    from finesub.llm import media_upload

    monkeypatch.setattr(media_upload, "current_route", lambda: ROUTE)
    _judge_globally(warnings)
    calls = 0

    def blocked():
        nonlocal calls
        calls += 1
        raise httpx.ConnectTimeout("timed out")

    with pytest.raises(HostUnreachableError):
        media_upload._with_upload_retries(
            blocked, stage="upload", cancel=None, sleep_func=lambda _: None, max_attempts=3
        )
    assert calls == 1


def test_an_upload_answered_with_an_error_status_is_retried_as_usual(monkeypatch, warnings) -> None:
    from finesub.llm import media_upload

    monkeypatch.setattr(media_upload, "current_route", lambda: ROUTE)
    _judge_globally(warnings)
    calls = 0

    def overloaded_then_fine():
        nonlocal calls
        calls += 1
        if calls == 1:
            request = httpx.Request("POST", f"https://{media_upload.GEMINI_FILES_HOST}/upload")
            httpx.Response(503, request=request).raise_for_status()
        return "ok"

    assert media_upload._with_upload_retries(
        overloaded_then_fine, stage="upload", cancel=None, sleep_func=lambda _: None, max_attempts=3
    ) == "ok"
    assert calls == 2
    assert not api_http.HOST_BREAKER.unreachable(media_upload.GEMINI_FILES_HOST, ROUTE)


def test_calls_through_a_mirror_are_booked_on_the_mirror(monkeypatch, warnings) -> None:
    """Google judged unreachable says nothing about a `GEMINI_BASE_URL` mirror the
    calls actually go to: a 503 from it is retried as usual."""

    from finesub.llm import llm_runtime

    bases: list[str] = []

    class Overloaded(RuntimeError):
        status_code = 503

    def mirror(**kwargs):
        bases.append(kwargs["api_base"])
        if len(bases) == 1:
            raise Overloaded("503 overloaded")
        return {"choices": [{"message": {"content": "ok"}}]}

    _gemini_once_key(monkeypatch, mirror)
    monkeypatch.setattr(llm_runtime, "_read_dotenv", lambda: {
        "GEMINI_FREE": "{free-main:key1}", "GEMINI_BASE_URL": "https://mirror.example/v1beta",
    })
    _judge_globally(warnings)

    _chat(retries=2)

    assert bases == ["https://mirror.example/v1beta"] * 2


def test_a_proxy_password_stays_out_of_the_route_label(monkeypatch) -> None:
    monkeypatch.setattr(api_http, "configured_proxy", lambda: "http://user:secret@10.0.0.2:8080")
    configured = current_route()
    monkeypatch.setattr(api_http, "configured_proxy", lambda: None)
    monkeypatch.setattr(api_http.urllib.request, "getproxies", lambda: {"https": "http://user:secret@10.0.0.2:8080"})
    system = current_route()
    monkeypatch.setattr(api_http.urllib.request, "getproxies", lambda: {"https": "user:secret@10.0.0.2:8080"})
    bare = current_route()

    for route in (configured, system, bare):
        assert "secret" not in route.label and "10.0.0.2:8080" in route.label
        assert "secret" in route.key  # a changed password is still a changed route


# -- streamed requests (the hedged pair) ----------------------------------------------------------------------------


class _Reach:
    def __init__(self) -> None:
        self.failed_with: list[BaseException] = []

    def reached(self) -> None:
        pass

    def failed(self, exc: BaseException) -> None:
        self.failed_with.append(exc)


class _StreamClient:
    """`stream()` waits `delay` seconds (or until the request is cancelled), then raises `error`."""

    def __init__(self, error: BaseException, delay: float, progress) -> None:
        self.error, self.delay, self.progress = error, delay, progress

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def close(self) -> None:
        return None

    def stream(self, *_args, **_kwargs):
        import time

        end = time.monotonic() + self.delay
        while time.monotonic() < end and not self.progress.cancelled:
            time.sleep(0.01)
        raise self.error


def _stream(monkeypatch, *, delay: float, deadline: float, cancel_after: float | None = None) -> _Reach:
    import threading

    from finesub.llm import http as llm_http
    from finesub.llm.streaming import OpenAIChatAssembler

    reach = _Reach()
    monkeypatch.setattr(llm_http, "start_reach", lambda url: reach)
    progress = llm_http.StreamProgress()
    if cancel_after is not None:
        threading.Timer(cancel_after, lambda: setattr(progress, "cancelled", True)).start()
    try:
        llm_http.stream_within_deadline(
            "https://generativelanguage.googleapis.com/x",
            headers={},
            json={},
            deadline_seconds=deadline,
            assembler=OpenAIChatAssembler(progress),
            error_for_status=lambda response: RuntimeError(str(response.status_code)),
            progress=progress,
            client_factory=lambda **_: _StreamClient(httpx.ConnectTimeout("timed out"), delay, progress),
        )
    except BaseException:
        pass
    time.sleep(delay + 0.2)  # the worker records after the caller has been answered
    return reach


def test_a_streamed_connect_failure_is_recorded(monkeypatch) -> None:
    assert len(_stream(monkeypatch, delay=0.0, deadline=5).failed_with) == 1


def test_a_request_closed_by_its_pair_or_its_deadline_is_not_a_connect_failure(monkeypatch) -> None:
    assert _stream(monkeypatch, delay=0.5, deadline=5, cancel_after=0.1).failed_with == []
    assert _stream(monkeypatch, delay=0.5, deadline=0.1).failed_with == []


# -- the candidate chain keeps its media ----------------------------------------------------------------------------


def test_a_failure_that_never_connected_keeps_the_uploaded_media(monkeypatch) -> None:
    """The cached Files API object is dropped after a provider error (it may be
    the expired thing that failed); a request that never connected says nothing about it."""

    from finesub.llm.client import LLMRole, RoleClient
    from finesub.llm.media_upload import UploadedFileRef
    from finesub.llm.rate_limit import ModelRateLimiter

    def failing(error):
        def fake_chat_complete(messages, *, model, **kwargs):
            raise error

        return fake_chat_complete

    forgotten: list[object] = []
    ref = UploadedFileRef(file_id="files/abc", filename="w.aac", mime_type="audio/aac")
    for error, dropped in ((httpx.ConnectTimeout("timed out"), False), (RuntimeError("400 bad"), True)):
        forgotten.clear()
        monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", failing(error))
        client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))
        monkeypatch.setattr(client, "_forget_uploaded_media_ref", lambda ref, **kw: forgotten.append(ref))
        with pytest.raises(type(error)):
            client.complete(LLMRole.GENERAL_CAPABLE, [{"role": "user", "content": "hi"}], file_ref=ref)
        assert bool(forgotten) is dropped


# -- a proxy that never answers CONNECT -----------------------------------------------------------------------------


@contextmanager
def _silent_proxy() -> Iterator[str]:
    """A local proxy that accepts connections and never says a word; nothing leaves the machine."""

    import socket
    import threading

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(5)
    held: list[socket.socket] = []

    def accept() -> None:
        try:
            while True:
                held.append(server.accept()[0])
        except OSError:
            pass

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
    finally:
        server.close()
        for conn in held:
            conn.close()


def test_a_proxy_that_never_answers_connect_is_a_connect_failure_within_the_connect_timeout(monkeypatch) -> None:
    """Real sockets: httpcore waits for the CONNECT reply with the *read* timeout; `TunnelWatch` makes it the
    connect timeout and the failure a connect failure. If an httpcore release renames its trace events, this is
    what goes red."""

    with _silent_proxy() as proxy:
        monkeypatch.setattr(api_http, "configured_proxy", lambda: proxy)
        started = time.monotonic()
        with api_http.api_http_client(httpx.Timeout(30.0, connect=0.5)) as client:
            with pytest.raises(httpx.TimeoutException) as raised:
                client.post("https://generativelanguage.googleapis.com/v1beta/x", json={})
        assert time.monotonic() - started < 5
        assert is_connect_failure(raised.value)
        assert "代理没有回应 CONNECT" in api_http._describe(raised.value)


def test_the_tunnel_watch_shortens_only_the_connect_reply_and_puts_the_read_timeout_back() -> None:
    """The events as httpcore sends them: only `.started` carries the request."""

    from types import SimpleNamespace

    from finesub_bootstrap.http_client import TunnelWatch

    timeouts = {"connect": 5.0, "read": 600.0}
    connect = SimpleNamespace(method=b"CONNECT", extensions={"timeout": timeouts})
    real = SimpleNamespace(method=b"POST", extensions={"timeout": timeouts})
    seen: list[str] = []
    watch = TunnelWatch(lambda name, info: seen.append(name))

    # Through an `https://` proxy: its own TLS handshake comes first, then the CONNECT reply waits with the connect
    # timeout, then the real request with its own.
    watch("connection.connect_tcp.complete", {"return_value": object()})
    watch("connection.start_tls.started", {"ssl_context": None})
    watch("connection.start_tls.complete", {"return_value": object()})
    watch("http11.receive_response_headers.started", {"request": connect})
    assert timeouts["read"] == 5.0 and watch.pending
    watch("http11.receive_response_headers.complete", {"return_value": (b"HTTP/1.1", 200, b"OK", [], b"")})
    assert timeouts["read"] == 600.0 and not watch.pending
    watch("proxy.start_tls.started", {"ssl_context": None})
    watch("http11.receive_response_headers.started", {"request": real})
    watch("http11.receive_response_headers.complete", {"return_value": (b"HTTP/1.1", 200, b"OK", [], b"")})
    assert timeouts["read"] == 600.0 and not watch.pending

    # An unanswered CONNECT stays pending after it fails: that is what the timeout is classified by.
    watch("http11.receive_response_headers.started", {"request": connect})
    watch("http11.receive_response_headers.failed", {"exception": httpx.ReadTimeout("x")})
    assert timeouts["read"] == 600.0 and watch.pending
    assert len(seen) == 10  # the request's own trace (the stream deadline's) still sees every event


def test_after_the_tunnel_is_up_a_slow_answer_keeps_the_read_timeout(monkeypatch, tmp_path) -> None:
    """Real httpcore end to end: the proxy says 200 to CONNECT, then (playing the target) answers the real
    request later than the connect timeout. Only the CONNECT reply may be held to the connect timeout."""

    import datetime
    import ipaddress
    import socket
    import ssl
    import threading

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(1).not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                    x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    (tmp_path / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    (tmp_path / "key.pem").write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    target_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    target_tls.load_cert_chain(tmp_path / "cert.pem", tmp_path / "key.pem")

    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)

    def tunnel() -> None:
        conn, _ = server.accept()
        with conn:
            while b"\r\n\r\n" not in conn.recv(4096):
                pass
            conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            with target_tls.wrap_socket(conn, server_side=True) as tls:
                while b"\r\n\r\n" not in tls.recv(4096):
                    pass
                time.sleep(1.0)  # twice the connect timeout before the headers
                tls.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\n")
                time.sleep(1.0)  # and again before the body
                tls.sendall(b"done")

    threading.Thread(target=tunnel, daemon=True).start()
    monkeypatch.setattr(api_http, "configured_proxy", lambda: f"http://127.0.0.1:{server.getsockname()[1]}")
    try:
        client_tls = ssl.create_default_context(cafile=str(tmp_path / "cert.pem"))
        with api_http.api_http_client(httpx.Timeout(30.0, connect=0.5), verify=client_tls) as client:
            response = client.get("https://localhost/x")
        assert response.status_code == 200 and response.text == "done"
    finally:
        server.close()


def test_a_download_through_a_proxy_that_never_answers_connect_moves_on_within_the_connect_timeout() -> None:
    """The download family's client carries the same watch: its read timeout is 120 s."""

    from finesub_bootstrap.http_client import NetworkRoute, create_client, in_tunnel_setup

    with _silent_proxy() as proxy:
        started = time.monotonic()
        with create_client(NetworkRoute("proxy", proxy), timeout=httpx.Timeout(30.0, connect=0.5)) as client:
            with pytest.raises(httpx.TimeoutException) as raised:
                client.get("https://github.com/x")
        assert time.monotonic() - started < 5
        assert in_tunnel_setup(raised.value)


def test_a_socks_proxy_is_refused_with_what_to_use_instead(monkeypatch, tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[api]\nproxy = "socks5://user:secret@127.0.0.1:1080"\n', encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(config))
    with pytest.raises(ValueError) as raised:
        api_http.configured_proxy()
    assert "HTTP 端口" in str(raised.value) and "secret" not in str(raised.value)
