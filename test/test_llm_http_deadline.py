"""The hard deadline on an LLM API call holds against a real socket.

An httpx timeout bounds each read, not the request: a server that sends a byte
now and then never trips it. `http.stream_within_deadline` is what turns the hard
timeout into a deadline on the whole call, and a fake client cannot show that
it does -- the guarantee lives in what a shut-down socket does to a read that is
blocked in another thread. So most of these drive real requests at a tiny local
server that misbehaves on purpose. Everything is on 127.0.0.1 with an
ephemeral port.

The socket tests are **opt-in** (`pytest --run-network-mock`), like the other
tests that bind sockets (`docs/testing.md`); the ones above the server need no
socket and run by default -- among them the soft timeout's backup request
(`hedge_on_silence`) and the partial text a failed call leaves.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

import httpx
import pytest

from finesub import api_http
from finesub.llm import http as llm_http
from finesub.llm import llm_runtime


def _call(
    url: str,
    *,
    deadline: float,
    client_factory=llm_http.llm_http_client,
    progress: llm_http.StreamProgress | None = None,
):
    """One streamed call in the OpenAI dialect; the clock is the point."""

    from finesub.llm.streaming import OpenAIChatAssembler

    progress = progress or llm_http.StreamProgress()
    return llm_http.stream_within_deadline(
        url,
        headers={},
        json={},
        deadline_seconds=deadline,
        assembler=OpenAIChatAssembler(progress),
        error_for_status=lambda response: RuntimeError(f"HTTP {response.status_code}"),
        progress=progress,
        client_factory=client_factory,
    )


class _FailingClient:
    """Stands in for the httpx client: `stream()` raises `error` at once."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def close(self) -> None:
        return None

    def stream(self, *_args, **_kwargs):
        raise self._error


def test_a_read_timeout_at_the_deadline_is_reported_as_the_deadline() -> None:
    """The per-read timeout and the deadline share one budget, so which fires
    first is scheduling noise -- the exception type must not depend on it."""

    error = httpx.ReadTimeout("timed out")

    with pytest.raises(llm_http.DeadlineExceeded):
        _call("http://x", deadline=0, client_factory=lambda **_: _FailingClient(error))


class _Stream:
    """An httpcore network stream as the trace sees it."""

    def __init__(self, sock: object) -> None:
        self.sock = sock
        self.closed = False

    def get_extra_info(self, name: str):
        return self.sock if name == "socket" else None

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("route", ["connection", "proxy"])
def test_the_cut_reaches_the_tls_socket_not_the_detached_raw_one(monkeypatch, route) -> None:
    """Over HTTPS the TCP socket is detached by `wrap_socket` (fileno -1), so
    shutting it down does nothing -- and on Linux a close alone does not wake
    the blocked read. The TLS socket that replaces it is the one to shut."""

    shut: list[object] = []
    monkeypatch.setattr(llm_http, "_shut", shut.append)
    teardown = llm_http._Teardown(60)
    raw, tls = object(), object()

    teardown.trace(f"{route}.connect_tcp.complete", {"return_value": _Stream(raw)})
    teardown.trace(f"{route}.start_tls.complete", {"return_value": _Stream(tls)})
    teardown.cut()

    assert tls in shut


def test_a_stream_built_after_the_deadline_is_closed_not_leaked() -> None:
    teardown = llm_http._Teardown(0)
    stream = _Stream(None)

    with pytest.raises(llm_http.DeadlineExceeded):
        teardown.trace("connection.connect_tcp.complete", {"return_value": stream})
    assert stream.closed
    # ...and nothing more is sent.
    with pytest.raises(llm_http.DeadlineExceeded):
        teardown.trace("http11.send_request_headers.started", {})


class _BlockingClient:
    """`stream()` blocks until the client is closed -- i.e. until the cut."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def close(self) -> None:
        self.closed.set()

    def stream(self, *_args, **_kwargs):
        self.closed.wait(30)
        raise httpx.ReadError("closed")


def test_ctrl_c_is_not_held_up_by_the_deadline_wait() -> None:
    """A join with a timeout cannot be interrupted on Windows; waiting in
    slices is what lets Ctrl+C through -- and it must also stop the request."""

    import _thread

    client = _BlockingClient()
    threading.Timer(0.2, _thread.interrupt_main).start()

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _call("http://x", deadline=30, client_factory=lambda **_: client)

    assert time.monotonic() - started < 3
    assert client.closed.is_set(), "the interrupted request is torn down too"


def test_a_genuine_error_before_the_deadline_stays_itself() -> None:
    """A connect that fails at its own 15 s cap, long before a 46-minute
    deadline, is a connect failure -- not relabelled as the deadline."""

    error = httpx.ConnectTimeout("timed out")

    with pytest.raises(httpx.ConnectTimeout) as raised:
        _call("http://x", deadline=30, client_factory=lambda **_: _FailingClient(error))
    assert not isinstance(raised.value, llm_http.DeadlineExceeded)


class _Request:
    """A `hedge_on_silence` request the test drives: it sends bytes, answers or
    fails when told to, and ends early if the pair closes it."""

    def __init__(self) -> None:
        self.answer = threading.Event()
        self.result: object = None
        self.error: BaseException | None = None
        self.progress: llm_http.StreamProgress | None = None

    def __call__(self, progress: llm_http.StreamProgress):
        self.progress = progress
        while not self.answer.wait(0.02):
            if progress.cancelled:
                raise llm_http.StreamCancelled("closed")
        if self.error is not None:
            raise self.error
        return self.result


def _hedge(requests: list[_Request], *, silence: float = 0.2):
    """Run a pair over `requests` (primary, backup); returns the outcome, the
    backups noted and the requests as they ended."""

    sends = iter(requests)
    backups: list[float] = []
    ended: list[llm_http.HedgedRequest] = []
    try:
        outcome: object = llm_http.hedge_on_silence(
            lambda progress: next(sends)(progress),
            silence_seconds=silence,
            on_backup=lambda progress: backups.append(progress.silent_for(time.monotonic())),
            on_end=ended.append,
        )
    except Exception as exc:  # noqa: BLE001 - the outcome under test
        outcome = exc
    return outcome, backups, ended


def _answer_after(request: _Request, seconds: float, result: object = None, error=None) -> None:
    request.result, request.error = result, error
    threading.Timer(seconds, request.answer.set).start()


def test_an_answer_before_the_soft_timeout_sends_no_backup() -> None:
    primary = _Request()
    _answer_after(primary, 0.05, "first")

    outcome, backups, ended = _hedge([primary])

    assert outcome == "first" and backups == []
    assert [(item.backup, item.error) for item in ended] == [(False, None)]


def test_silence_sends_a_backup_and_the_first_answer_wins() -> None:
    """The silent first request is kept open; when the backup answers first it
    is closed, and both are reported -- the winner last, so the answer is the
    last thing recorded."""

    primary, backup = _Request(), _Request()
    _answer_after(backup, 0.1, "backup")

    outcome, backups, ended = _hedge([primary, backup])

    assert outcome == "backup"
    assert len(backups) == 1 and backups[0] >= 0.2
    assert [item.backup for item in ended] == [False, True]
    assert isinstance(ended[0].error, llm_http.StreamCancelled)
    assert ended[0].backup_sent_after is not None and ended[0].backup_sent_after >= 0.2


def test_the_first_request_still_wins_if_it_answers_after_the_backup_left() -> None:
    primary, backup = _Request(), _Request()
    _answer_after(primary, 0.35, "first")

    outcome, _backups, ended = _hedge([primary, backup])

    assert outcome == "first"
    assert [item.backup for item in ended] == [True, False]
    assert isinstance(ended[0].error, llm_http.StreamCancelled)


def test_a_backup_that_fails_leaves_the_first_request_to_answer() -> None:
    primary, backup = _Request(), _Request()
    _answer_after(backup, 0.25, error=RuntimeError("HTTP 429"))
    _answer_after(primary, 0.4, "first")

    outcome, _backups, ended = _hedge([primary, backup])

    assert outcome == "first"
    assert [(item.backup, str(item.error)) for item in ended] == [(True, "HTTP 429"), (False, "None")]


def test_two_answers_at_once_use_one_and_record_the_other_as_closed() -> None:
    primary, backup = _Request(), _Request()
    both = threading.Event()
    primary.answer = backup.answer = both
    primary.result, backup.result = "first", "backup"
    threading.Timer(0.3, both.set).start()

    outcome, _backups, ended = _hedge([primary, backup])

    loser, winner = ended
    assert outcome == winner.result and winner.error is None
    assert isinstance(loser.error, llm_http.StreamCancelled) and loser.result is None


def test_the_rate_limit_wait_before_a_backup_does_not_delay_the_first_answer() -> None:
    """`on_backup` runs on the backup's thread: while it waits, an answer on
    the first request is still taken at once."""

    primary, backup = _Request(), _Request()
    _answer_after(primary, 0.3, "first")
    sends = iter([primary, backup])

    started = time.monotonic()
    outcome = llm_http.hedge_on_silence(
        lambda progress: next(sends)(progress),
        silence_seconds=0.1,
        on_backup=lambda progress: time.sleep(0.6),
        on_end=lambda request: None,
    )

    assert outcome == "first"
    assert time.monotonic() - started < 0.55
    # ...and the backup, closed while it waited, never sends once it wakes.
    time.sleep(0.6)
    assert backup.progress is None


def test_ctrl_c_closes_both_requests_of_a_pair() -> None:
    import _thread

    primary, backup = _Request(), _Request()
    threading.Timer(0.35, _thread.interrupt_main).start()
    sends = iter([primary, backup])

    with pytest.raises(KeyboardInterrupt):
        llm_http.hedge_on_silence(
            lambda progress: next(sends)(progress),
            silence_seconds=0.1,
            on_backup=lambda progress: None,
            on_end=lambda request: None,
        )

    assert primary.progress is not None and primary.progress.cancelled
    assert backup.progress is not None and backup.progress.cancelled


def test_bytes_on_the_first_request_hold_the_backup_back() -> None:
    """Silence restarts with every byte: a request still writing is not silent."""

    primary = _Request()
    _answer_after(primary, 0.5, "first")

    def bytes_until_answered() -> None:
        while not primary.answer.is_set():
            if primary.progress is not None:
                primary.progress.line()
            time.sleep(0.05)

    threading.Thread(target=bytes_until_answered, daemon=True).start()
    outcome, backups, _ended = _hedge([primary])

    assert outcome == "first" and backups == []


def test_when_both_fail_the_backup_error_is_raised() -> None:
    """Even when the first request fails last: the backup is the newer word on
    the endpoint."""

    primary, backup = _Request(), _Request()
    _answer_after(primary, 0.45, error=RuntimeError("primary"))
    _answer_after(backup, 0.3, error=RuntimeError("backup"))

    outcome, _backups, ended = _hedge([primary, backup])

    assert str(outcome) == "backup"
    assert [str(item.error) for item in ended] == ["backup", "primary"]


def test_a_failure_before_the_soft_timeout_sends_no_backup() -> None:
    primary = _Request()
    _answer_after(primary, 0.05, error=RuntimeError("HTTP 500"))

    outcome, backups, _ended = _hedge([primary])

    assert str(outcome) == "HTTP 500" and backups == []


def test_a_failed_call_leaves_its_received_text_in_a_bounded_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(llm_http, "resolve_logs_dir", lambda: tmp_path)
    monkeypatch.setattr(llm_http, "RETAINED_PARTIAL_TEXTS", 3)
    names = []
    for index in range(5):
        progress = llm_http.StreamProgress()
        progress.output(f"half an answer {index}")
        names.append(keep := llm_http.keep_partial_text(progress, {"model": "m/x", "code": "500"}))
        assert keep is not None
        # Distinct modification times, whatever the filesystem's resolution.
        stamp = time.time() - 100 + index
        import os

        os.utime(tmp_path / llm_http.PARTIAL_TEXT_DIR / keep, (stamp, stamp))

    kept = sorted(path.name for path in (tmp_path / llm_http.PARTIAL_TEXT_DIR).iterdir())
    assert len(kept) == 3
    newest = (tmp_path / llm_http.PARTIAL_TEXT_DIR / names[-1]).read_text(encoding="utf-8")
    assert newest.startswith("model: m/x\ncode: 500\nstarted_at: ")
    assert "\nchars: 16\n\n" in newest
    assert newest.endswith("half an answer 4")
    # Nothing received, nothing written.
    assert llm_http.keep_partial_text(llm_http.StreamProgress(), {"model": "m"}) is None


GEMINI_OK = {
    "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, behaviour: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = behaviour
        self.requests = 0
        self.stop = threading.Event()
        self.client_gone = threading.Event()

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def handle_error(self, request, client_address) -> None:
        # The client hanging up mid-response is the point of these tests.
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        server: _Server = self.server  # type: ignore[assignment]
        server.requests += 1
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        getattr(self, f"_{server.behaviour}")(server)

    def _trickle(self, server: _Server) -> None:
        """Headers at once, then one byte of body every 50 ms, forever."""

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while not server.stop.is_set():
                self.wfile.write(b"1\r\n \r\n")
                self.wfile.flush()
                server.stop.wait(0.05)
        except OSError:
            server.client_gone.set()

    def _silent(self, server: _Server) -> None:
        """Take the request and never answer."""

        server.stop.wait(10)

    def _late(self, server: _Server) -> None:
        """Answer in full -- one Gemini chunk -- but only after 0.3 s."""

        server.stop.wait(0.3)
        self._sse_head()
        self._sse_write(f"data: {json.dumps(GEMINI_OK)}\n\n")
        self.wfile.write(b"0\r\n\r\n")

    def _sse_head(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse_write(self, text: str) -> None:
        body = text.encode("utf-8")
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()

    def _sse_stall(self, server: _Server) -> None:
        """Answer text at once, then silence."""

        self._sse_head()
        try:
            self._sse_write(_SSE_TEXT)
            server.stop.wait(10)
            # The first write after the peer hung up can still succeed.
            for _ in range(40):
                self._sse_write(": still here\n\n")
                time.sleep(0.05)
        except OSError:
            server.client_gone.set()

    def _sse_keepalive(self, server: _Server) -> None:
        """A comment every 50 ms, and never any answer text."""

        self._sse_head()
        try:
            while not server.stop.is_set():
                self._sse_write(": keep-alive\n\n")
                server.stop.wait(0.05)
        except OSError:
            server.client_gone.set()

    def _sse_no_done(self, server: _Server) -> None:
        """Answer text, then a clean end of body without the terminal marker."""

        self._sse_head()
        self._sse_write(_SSE_TEXT)
        self.wfile.write(b"0\r\n\r\n")

    def _sse_drop(self, server: _Server) -> None:
        """Answer text, then the connection dies mid-chunk."""

        self._sse_head()
        self._sse_write(_SSE_TEXT)
        self.wfile.write(b"40\r\ndata: {")
        self.wfile.flush()
        self.connection.shutdown(socket.SHUT_RDWR)

    def log_message(self, *_args) -> None:
        return


_SSE_TEXT = (
    'data: {"choices": [{"index": 0, "delta": {"content": "partial answer"}}]}\n\n'
)


@pytest.fixture
def serve(monkeypatch):
    # No proxy, whatever the machine or the checkout's config.toml says.
    monkeypatch.setattr(api_http, "configured_proxy", lambda: api_http.DIRECT)
    servers: list[_Server] = []

    def start(behaviour: str) -> _Server:
        server = _Server(behaviour)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop.set()
        server.shutdown()
        server.server_close()


@pytest.mark.network_mock
def test_a_trickling_response_is_cut_at_the_deadline(serve) -> None:
    server = serve("trickle")

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        _call(server.origin, deadline=0.4)
    elapsed = time.monotonic() - started

    # Before the fix this returned whenever the server chose to stop.
    assert 0.35 <= elapsed < 1.5
    # Torn down, not abandoned: the server's next write finds the peer gone.
    assert server.client_gone.wait(2.0)


@pytest.mark.network_mock
def test_a_silent_server_is_cut_at_the_deadline(serve) -> None:
    server = serve("silent")

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        _call(server.origin, deadline=0.4)
    assert time.monotonic() - started < 1.5


@pytest.mark.network_mock
def test_a_deadline_during_a_slow_lookup_sends_nothing(serve, monkeypatch) -> None:
    """The caller is back on time, and the request never leaves afterwards.

    A DNS lookup cannot be interrupted, so the worker is still in it when the
    deadline passes; once it comes back it must not connect-and-POST anyway --
    that would start a generation nobody is waiting for.
    """

    server = serve("silent")
    real_getaddrinfo = socket.getaddrinfo

    def slow_getaddrinfo(*args, **kwargs):
        time.sleep(0.4)
        return real_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_getaddrinfo)

    started = time.monotonic()
    with pytest.raises(llm_http.DeadlineExceeded):
        _call(server.origin, deadline=0.1)
    assert time.monotonic() - started < 0.35, "the caller does not wait out the lookup"

    time.sleep(0.8)  # well past the lookup finishing
    assert server.requests == 0


def _gemini_at(monkeypatch, server: _Server, *, soft: float, hard: float) -> None:
    monkeypatch.delenv("GEMINI_FREE", raising=False)
    monkeypatch.delenv("GEMINI_PAID", raising=False)
    monkeypatch.setattr(
        llm_runtime,
        "_read_dotenv",
        lambda: {"GEMINI_FREE": "{free-main:key1}", "GEMINI_BASE_URL": server.origin},
    )
    monkeypatch.setattr(
        llm_runtime,
        "api_timeouts",
        lambda max_tokens: llm_runtime.ApiTimeouts(soft, hard),
    )


@pytest.mark.network_mock
def test_two_hard_deadlines_take_the_existing_timeout_abort(serve, monkeypatch) -> None:
    """A cut call is a timeout to everything downstream of it."""

    server = serve("silent")
    # The soft timeout past the hard one: no backups, one request per attempt.
    _gemini_at(monkeypatch, server, soft=10.0, hard=0.3)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda _: None)

    with pytest.raises(llm_http.DeadlineExceeded) as raised:
        llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier="GEMINI_FREE",
            model="gemini/gemini-3.1-flash-lite",
            retries=7,
        )

    assert server.requests == 2
    assert getattr(raised.value, "_harness_consecutive_timeout_abort") is True
    attempts = getattr(raised.value, "_harness_api_attempts")
    assert [item["return_code"] for item in attempts] == ["NO_RESPONSE_TIMEOUT"] * 2


@pytest.mark.network_mock
def test_a_silent_first_request_that_answers_before_its_backup_is_used(serve, monkeypatch) -> None:
    from finesub.reporting import NullReporter, reporting_to

    warnings: list[str] = []

    class _Warnings(NullReporter):
        def warning(self, code, message, *, impact="", action="") -> None:
            warnings.append(code)

    server = serve("late")
    _gemini_at(monkeypatch, server, soft=0.1, hard=3.0)

    with reporting_to(_Warnings()):
        response = llm_runtime.chat_complete(
            [{"role": "user", "content": "hi"}],
            provider_tier="GEMINI_FREE",
            model="gemini/gemini-3.1-flash-lite",
            retries=0,
        )

    assert warnings == ["llm-api-soft-timeout", "llm-api-backup-outcome"]
    backup, first = response["_harness_api_attempts"]
    assert first["return_code"] == "200" and first["backup_sent_sec"] >= 0.1
    assert backup["return_code"] == "SUPERSEDED" and backup["backup"] is True
    assert llm_runtime.extract_message_content(response) == "ok"
    assert server.requests == 2


def _stream(server: _Server, *, deadline: float):
    progress = llm_http.StreamProgress()
    return progress, lambda: _call(server.origin, deadline=deadline, progress=progress)


@pytest.mark.network_mock
def test_a_cancelled_request_is_torn_down(serve) -> None:
    """What closes the losing request of a pair: the socket is shut, not left."""

    server = serve("sse_stall")
    progress, call = _stream(server, deadline=10.0)
    threading.Timer(0.3, progress.cancel).start()

    started = time.monotonic()
    with pytest.raises(llm_http.StreamCancelled):
        call()
    assert time.monotonic() - started < 2.0
    assert progress.output_text() == "partial answer"
    server.stop.set()
    assert server.client_gone.wait(2.0)


@pytest.mark.network_mock
def test_keep_alives_without_answer_text_wait_for_the_deadline(serve) -> None:
    """Bytes that are not answer text keep the call open up to the deadline,
    and are not counted as events."""

    server = serve("sse_keepalive")
    progress, call = _stream(server, deadline=0.8)

    with pytest.raises(llm_http.DeadlineExceeded):
        call()
    assert progress.header_at is not None and progress.first_output_at is None
    assert progress.record()["stream_events"] == 0


@pytest.mark.network_mock
def test_headers_that_never_come_wait_for_the_deadline(serve) -> None:
    server = serve("silent")
    progress, call = _stream(server, deadline=0.4)

    with pytest.raises(llm_http.DeadlineExceeded):
        call()
    assert progress.record() == {}


@pytest.mark.network_mock
@pytest.mark.parametrize("behaviour", ["sse_no_done", "sse_drop"])
def test_a_stream_that_ends_early_is_interrupted_not_answered(serve, behaviour) -> None:
    server = serve(behaviour)
    _progress, call = _stream(server, deadline=5.0)

    with pytest.raises(llm_http.StreamInterrupted):
        call()
