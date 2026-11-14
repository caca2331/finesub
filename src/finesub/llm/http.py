"""The one place an LLM-layer HTTP client is built.

Everything this package talks to over the network -- Gemini REST, the
OpenAI-compat / Anthropic transports, Exa / Gemma4 / Tavily, and the free
`countTokens` endpoint -- goes out through :func:`llm_http_client`, which takes
its route from `finesub.api_http` (the `[api] proxy` setting, shared with the
cloud ASR backends).

The clocks of a streamed generation call live here too: the hard deadline
(:func:`stream_within_deadline`) and the soft timeout's backup request
(:func:`hedge_on_silence`), with what a failed call leaves behind
(:func:`keep_partial_text`).

⚠ **Turning a proxy on changes the failure modes**, not just the route. Gemini
scores datacenter and shared-exit IPs, and answers a suspicious one with a
block that `client.LLMIPRiskError` recognises. A proxy that works for search
can still make correction calls fail this way.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import queue
import re
import socket
import threading
import time
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar

import httpx

from finesub_bootstrap.logs import prune as prune_logs

from ..api_http import CONNECT_TIMEOUT_SECONDS, api_http_client, start_reach
from ..paths import resolve_logs_dir
from ..reporting import bind_reporter, current_reporter

def llm_http_client(timeout: Any = None, **kwargs: Any) -> httpx.Client:
    """An `httpx.Client` for an LLM-layer API call.

    Signature-compatible with `httpx.Client` for the call sites that pass a
    bare `timeout=`, and with the `client_factory` seam the search and token
    clients already inject in tests.
    """

    return api_http_client(timeout, **kwargs)


class DeadlineExceeded(httpx.TimeoutException):
    """The whole request outlived its deadline and was torn down.

    A `TimeoutException`, and worded "timed out", on purpose: every retry and
    classification rule written for an httpx timeout applies to it unchanged
    (`llm_runtime._status_code_from_exception` reads it as NO_RESPONSE_TIMEOUT).
    """


class StreamCancelled(RuntimeError):
    """The caller closed this request: the other one of its pair answered first
    (:func:`hedge_on_silence`). Never leaves the pair -- the attempt records
    `SUPERSEDED`."""


class StreamInterrupted(RuntimeError):
    """The stream ended without its dialect's terminal marker.

    The connection dropped mid-body, closed cleanly too early, or the server
    restarted its answer inside the same stream. Never a success: a non-streamed
    body cannot arrive half-written, a stream can, and the half that did arrive
    would read as a complete answer. Retryable -- nothing about the request was
    wrong.
    """


class StreamProgress:
    """What has arrived on one streamed call, and the caller's handle on it.

    Written by the worker thread, read by the caller; single attribute writes
    and list appends, so no lock. Times are seconds since construction.
    "Output" is visible answer text only -- thinking, reasoning, keep-alives and
    usage-only chunks are bytes that prove the connection is alive, not answer.
    The text is kept so a failed call can leave what it had received
    (:func:`keep_partial_text`).
    """

    def __init__(self) -> None:
        self.started = time.monotonic()
        self.started_wall = datetime.now(timezone.utc)
        self.header_at: float | None = None
        self.first_output_at: float | None = None
        self.last_byte_at: float | None = None
        self.events = 0
        self.cancelled = False
        self._output: list[str] = []
        self.output_chars = 0

    def headers(self) -> None:
        self.header_at = self.last_byte_at = time.monotonic()

    def line(self) -> None:
        self.last_byte_at = time.monotonic()

    def event(self) -> None:
        self.events += 1

    def output(self, text: str) -> None:
        if not text:
            return
        if self.first_output_at is None:
            self.first_output_at = time.monotonic()
        self._output.append(text)
        self.output_chars += len(text)

    def output_text(self) -> str:
        return "".join(self._output)

    def silent_for(self, now: float) -> float:
        """Seconds since the last byte, or since the request left if none has
        arrived."""

        return now - (self.last_byte_at if self.last_byte_at is not None else self.started)

    def restart_clock(self) -> None:
        """For a request that waited (on a rate limit) after this was made:
        its clock is from leaving."""

        self.started = time.monotonic()
        self.started_wall = datetime.now(timezone.utc)

    def cancel(self) -> None:
        """Ask `stream_within_deadline` to close this request; it does within
        one wait slice."""

        self.cancelled = True

    def describe(self) -> str:
        if self.header_at is None:
            return "no response headers yet"
        if self.first_output_at is None:
            return "connected, no answer text yet"
        return f"answer streaming, {self.output_chars} characters so far"

    def record(self) -> dict[str, Any]:
        """The attempt fields; empty when nothing was streamed."""

        if self.header_at is None:
            return {}
        fields: dict[str, Any] = {
            "header_sec": round(self.header_at - self.started, 3),
            "stream_events": self.events,
        }
        if self.first_output_at is not None:
            fields["first_output_sec"] = round(self.first_output_at - self.started, 3)
        return fields


T = TypeVar("T")


class StreamAssembler(Protocol[T]):
    """Turns the body's lines into the dialect's non-streamed response shape."""

    def feed_line(self, line: str) -> None: ...

    def finish(self) -> T: ...




#: httpcore trace steps up to the moment the request leaves. The event name is
#: `<logger>.<step>.<phase>`, and the logger is `connection` or `proxy` (the
#: CONNECT tunnel's TLS) depending on the route, so steps are matched by suffix. Past the deadline each of them aborts the attempt: nothing is sent
#: once the deadline has passed -- sending late would start a generation nobody
#: is waiting for.
_CONNECT_STEPS = (".connect_tcp.", ".start_tls.")
_SEND_STEP = "http11.send_request_headers.started"
#: A step that just produced a stream. Its socket is what `cut()` shuts down,
#: and TLS replaces it: `wrap_socket` detaches the raw socket (fileno -1), so
#: the one worth holding is the latest.
_STREAM_DONE = (".connect_tcp.complete", ".start_tls.complete")

#: How long the caller waits on the worker at a time. On Windows a thread join
#: with a timeout is not interruptible, so Ctrl+C would otherwise wait out the
#: whole deadline -- which can be an hour and a half.
_JOIN_SLICE_SECONDS = 0.5


class _Teardown:
    """Everything needed to stop one request from outside, and the one clock
    that says whether it is too late.

    Expiry is read off the monotonic clock, never only off "has the teardown
    run": the two race (a timer thread can wake ~15 ms late on Windows, and a
    socket timeout does not), and the answer must not depend on which won.
    """

    def __init__(self, deadline_seconds: float) -> None:
        self.deadline_seconds = deadline_seconds
        self.deadline_at = time.monotonic() + max(0.0, deadline_seconds)
        self._lock = threading.Lock()
        self._cut = False
        self._sockets: list[socket.socket] = []
        self._client: httpx.Client | None = None

    def expired(self) -> bool:
        return self._cut or time.monotonic() >= self.deadline_at

    def error(self) -> DeadlineExceeded:
        return DeadlineExceeded(
            f"request timed out: exceeded its {self.deadline_seconds:.0f}s hard "
            "deadline and was closed"
        )

    def adopt_client(self, client: httpx.Client) -> None:
        with self._lock:
            self._client = client
            late = self._cut
        if late:
            client.close()

    def trace(self, name: str, info: Mapping[str, Any]) -> None:
        """httpcore's `trace` extension: learn the socket each time a step
        produces a stream, and refuse to go on once the deadline has passed."""

        stream = info.get("return_value") if name.endswith(_STREAM_DONE) else None
        if stream is not None:
            sock = stream.get_extra_info("socket")
            if sock is not None:
                with self._lock:
                    self._sockets.append(sock)
        before_send = name == _SEND_STEP or any(step in name for step in _CONNECT_STEPS)
        if before_send and self.expired():
            if stream is not None:
                # Built after the caller gave up (it was stuck in DNS or a
                # handshake): no pool owns it yet, so nobody else will close it.
                _shut(stream.get_extra_info("socket"))
                stream.close()
            raise self.error()

    def cut(self) -> None:
        with self._lock:
            self._cut = True
            sockets, client = list(self._sockets), self._client
        for sock in sockets:
            _shut(sock)
        if client is not None:
            client.close()


def _shut(sock: socket.socket | None) -> None:
    if sock is None:
        return
    try:
        # The base-class method, deliberately: `SSLSocket.shutdown` also drops
        # the TLS object from under the thread that is reading it.
        socket.socket.shutdown(sock, socket.SHUT_RDWR)
    except OSError:
        pass


def stream_within_deadline(
    url: str,
    *,
    headers: Mapping[str, str],
    json: Any,
    deadline_seconds: float,
    assembler: StreamAssembler[T],
    error_for_status: Callable[[httpx.Response], BaseException],
    progress: StreamProgress,
    client_factory: Callable[..., Any] = llm_http_client,
) -> T:
    """POST a streaming request and assemble its SSE body, within the hard
    deadline.

    ⚠ **An httpx timeout is not a deadline.** It bounds each network operation,
    so a server that sends a byte now and then -- a keep-alive, a trickling
    body -- never trips it, and the call can hold its slot indefinitely. The
    per-operation timeout stays (set to the deadline, connect capped at
    :data:`CONNECT_TIMEOUT_SECONDS`); the deadline itself is enforced here,
    three ways:

    * The request runs on a worker thread and the caller waits for it only
      until the deadline, in slices so Ctrl+C gets through, and is back on time
      even while the worker sits in something nothing can interrupt (a slow
      DNS lookup).
    * At the deadline the request is torn down, not merely left behind: the
      socket is shut down (that wakes a read blocked in another thread on every
      platform; a bare close does not on Linux) and the client closed. The
      socket is learnt from httpcore's `trace` extension as each step produces
      one -- the TCP socket, then the TLS socket that replaces it -- so this
      works before any response has arrived too. ⚠ During a TLS handshake there
      is nothing to shut down (the raw socket is already detached, the TLS one
      not yet returned); that window is bounded by the connect timeout.
    * The same trace refuses every step up to sending the request once the
      deadline has passed, so a worker still stuck in DNS when the caller gave
      up can never send it afterwards.

    `progress.cancel()` tears the request down the same way, as
    :class:`StreamCancelled`; silence is the caller's to judge
    (:func:`hedge_on_silence`), never a reason to close a request here.

    A non-2xx response is read whole and `error_for_status(response)` is raised,
    so each caller keeps its own HTTP error type. A body that breaks off, or ends
    without the dialect's terminal marker (the assembler's `finish` decides),
    raises :class:`StreamInterrupted`. Whichever clock decided the outcome is
    what the caller sees -- :class:`DeadlineExceeded` whenever the deadline did,
    even when the torn-down read (or a per-read timeout) raised something else
    first.
    """

    teardown = _Teardown(deadline_seconds)
    outcome: dict[str, Any] = {}
    timeout = httpx.Timeout(
        deadline_seconds, connect=min(CONNECT_TIMEOUT_SECONDS, deadline_seconds)
    )

    def attempt() -> None:
        # Reachability is recorded here, on the worker that sends, so the clock
        # starts when the request leaves rather than when a caller began queueing.
        reach = start_reach(url)
        try:
            with client_factory(timeout=timeout) as client:
                teardown.adopt_client(client)
                with client.stream(
                    "POST",
                    url,
                    headers=dict(headers),
                    json=json,
                    extensions={"trace": teardown.trace},
                ) as response:
                    progress.headers()
                    reach.reached()
                    if not 200 <= response.status_code < 300:
                        response.read()
                        raise error_for_status(response)
                    try:
                        for line in response.iter_lines():
                            progress.line()
                            assembler.feed_line(line)
                    except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                        raise StreamInterrupted(
                            f"stream interrupted: the connection broke mid-body ({exc})"
                        ) from exc
                    outcome["result"] = assembler.finish()
        except BaseException as exc:  # handed to the caller's thread
            outcome["error"] = exc
            # Torn down from outside -- the pair's other request answered, or the
            # deadline passed -- is not the host refusing a connection.
            if not progress.cancelled and not teardown.expired():
                reach.failed(exc)

    def cancelled() -> StreamCancelled:
        return StreamCancelled("closed by the caller: the other request of its pair answered first")

    if progress.cancelled:
        raise cancelled()
    worker = threading.Thread(target=attempt, name="llm-http-stream", daemon=True)
    worker.start()
    try:
        while worker.is_alive() and not progress.cancelled:
            remaining = teardown.deadline_at - time.monotonic()
            if remaining <= 0:
                break
            worker.join(min(remaining, _JOIN_SLICE_SECONDS))
    except BaseException:
        # Ctrl+C lands on this thread, not the worker's: stop the request too.
        teardown.cut()
        raise
    if worker.is_alive():
        teardown.cut()
        raise cancelled() if progress.cancelled else teardown.error()
    error = outcome.get("error")
    if error is not None:
        if isinstance(error, DeadlineExceeded) or teardown.expired():
            raise teardown.error() from error
        raise error
    return outcome["result"]


@dataclass
class HedgedRequest(Generic[T]):
    """One request of a :func:`hedge_on_silence` pair, as it ended."""

    progress: StreamProgress
    backup: bool
    result: T | None = None
    error: BaseException | None = None
    #: Seconds after this request left that its backup was started; primary only.
    backup_sent_after: float | None = None
    #: Past `on_backup`, i.e. about to send or sending.
    sending: bool = False


def hedge_on_silence(
    send: Callable[[StreamProgress], T],
    *,
    silence_seconds: float,
    on_backup: Callable[[StreamProgress], None],
    on_end: Callable[[HedgedRequest[T]], None],
) -> T:
    """Run `send` and, if nothing arrives on it for `silence_seconds`, run it a
    second time beside it without closing the first; the first to answer wins
    and the other is closed (owner decision 2026-10-23).

    Silence counts from the request leaving and restarts with every byte, so it
    cannot tell a model thinking without a word from a dead connection -- which
    is why the first request is kept, not replaced: if it was only thinking,
    its answer is still taken. Each request keeps its own hard deadline inside
    `send`. At most two: a silent backup gets no backup of its own.

    `on_backup(primary_progress)` runs on the backup's own thread before it
    sends, so a rate-limit wait there never holds up the watch on the first
    request; the backup's clock starts after it. `on_end(request)` runs on the
    calling thread once for every request that ran, in the order they ended --
    except that a winner always comes last, after the loser it closed (with
    :class:`StreamCancelled` as its error), so the answer is the last thing
    recorded. When both fail, the backup's error is raised: it is the newer word
    on the endpoint, and the primary's failure after the silence that started
    it says little more.
    """

    reporter = current_reporter()
    ended: queue.Queue[HedgedRequest[T]] = queue.Queue()

    def start(backup: bool) -> HedgedRequest[T]:
        request: HedgedRequest[T] = HedgedRequest(progress=StreamProgress(), backup=backup)

        def run() -> None:
            # Warnings raised here (on_backup's, an assembler's) go where the
            # caller's do: the reporter is bound per thread.
            bind_reporter(reporter)
            try:
                if backup:
                    on_backup(primary.progress)
                    request.progress.restart_clock()
                request.sending = True
                # Set before `sending` is read on the caller's thread, so a
                # backup recorded as never sent cannot send after all.
                if request.progress.cancelled:
                    raise StreamCancelled("closed before it was sent: the other request answered")
                request.result = send(request.progress)
            except BaseException as exc:  # handed to the caller's thread
                request.error = exc
            ended.put(request)

        threading.Thread(target=run, name="llm-http-hedge", daemon=True).start()
        return request

    primary = start(backup=False)
    running = [primary]
    last_failed: HedgedRequest[T] | None = None
    try:
        while running:
            wait = _JOIN_SLICE_SECONDS
            if primary.backup_sent_after is None and primary in running:
                until_backup = silence_seconds - primary.progress.silent_for(time.monotonic())
                if until_backup <= 0:
                    running.append(start(backup=True))
                    primary.backup_sent_after = time.monotonic() - primary.progress.started
                else:
                    wait = min(wait, until_backup)
            try:
                request = ended.get(timeout=wait)
            except queue.Empty:
                continue
            running.remove(request)
            if request.error is not None:
                on_end(request)
                last_failed = request if request.backup or last_failed is None else last_failed
                continue
            for loser in running:
                loser.progress.cancel()
            while running:
                # A backup still waiting in `on_backup` will see the cancel
                # before it sends anything: not worth waiting out its wait.
                for loser in [item for item in running if not item.sending]:
                    running.remove(loser)
                    loser.error = StreamCancelled("closed before it was sent: the other request answered")
                    on_end(loser)
                if not running:
                    break
                try:
                    loser = ended.get(timeout=_JOIN_SLICE_SECONDS)
                except queue.Empty:
                    continue
                running.remove(loser)
                if loser.error is None:
                    # Answered too, a moment too late to be used.
                    loser.result, loser.error = None, StreamCancelled(
                        "answered after the other request of its pair had won"
                    )
                on_end(loser)
            on_end(request)
            return request.result  # type: ignore[return-value]
    except BaseException:
        for request in running:
            request.progress.cancel()
        raise
    assert last_failed is not None and last_failed.error is not None
    raise last_failed.error


#: Failed calls whose received answer text is kept, newest first -- the same
#: bound as the local agent's failed episodes (`RETAINED_FAILED_EPISODES`):
#: evidence loses its value oldest-first.
RETAINED_PARTIAL_TEXTS = 20
#: Under the logs directory, beside the run logs it is matched against.
PARTIAL_TEXT_DIR = "api-partial"


def keep_partial_text(progress: StreamProgress, header: Mapping[str, Any]) -> str | None:
    """Write the answer text a failed call had received, for the logs.

    One file per call under `<logs>/api-partial/`, `header` and the request's
    start, duration and length as `key: value` lines above the text; the oldest past :data:`RETAINED_PARTIAL_TEXTS` are
    dropped. Returns the file name, or None when there was no text or nowhere
    to write. Best-effort: losing the evidence must not fail the call.
    """

    text = progress.output_text()
    logs = resolve_logs_dir()
    if not text or logs is None:
        return None
    directory = logs / PARTIAL_TEXT_DIR
    stamp = progress.started_wall.strftime("%Y%m%dT%H%M%S.%f")
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", str(header.get("model", "")))[:60]
    name = f"{stamp}-{model}.log"
    fields = {
        **header,
        "started_at": progress.started_wall.isoformat(timespec="milliseconds"),
        "failed_after_sec": round(time.monotonic() - progress.started, 3),
        "chars": progress.output_chars,
    }
    lines = [f"{key}: {value}" for key, value in fields.items()]
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(
            "\n".join(lines) + "\n\n" + text, encoding="utf-8", errors="replace"
        )
    except OSError:
        return None
    prune_logs(directory, keep=RETAINED_PARTIAL_TEXTS)
    return name
