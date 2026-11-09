"""The one place an HTTP client for a remote API is built: the LLM layer (`llm/http.llm_http_client`) and the
cloud ASR backends (`speech/joint/`), so "does this request use the proxy?" has one answer read off the code.

**Scope is API calls, deliberately.** The download family (`finesub_bootstrap.http_client`) has its own route
discovery -- Windows system proxy, environment, a liveness probe on local ports -- and is not touched by this
setting. That separation is why the proxy is resolved into an explicit `proxy=` argument rather than written into
`os.environ`: the environment is read by `apply_network_environment`, so an ambient proxy would silently reroute
downloads too.

The same two callers share the answer to "is this failure worth sending again?" (:func:`is_retryable_http_error`),
and every API caller shares the answer to "can this host be reached at all on this route?" (:class:`HostBreaker`,
:data:`HOST_BREAKER`): a blocked host -- the Great Firewall's dropped SYN or reset handshake -- otherwise costs every
call its own silent connect timeout.

Top-level rather than in `llm/` because `speech` may not import `llm` (`test_import_boundaries`).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import enum
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Iterator
import urllib.request
from urllib.parse import urlsplit

import httpx

from finesub_bootstrap.http_client import in_tunnel_setup, watch_tunnel

from . import settings
from .reporting import TerminalReporter, current_reporter, redact_credentials, reporter_delivers

#: `[api] proxy = "direct"` means "ignore the environment too". Without it, unsetting the key would still leave
#: `HTTPS_PROXY` in charge, and there would be no way to say "no proxy for API calls" on a machine that needs one
#: for everything else.
DIRECT = "direct"


def configured_proxy() -> str | None:
    """`[api] proxy` from config.toml (else its old name `[llm] proxy`): a URL, :data:`DIRECT`, or None.

    Read each time a client is built, not memoized: a client built after config.toml changed uses the new
    value, the same "edits take effect on the next read" contract as every other setting. The read itself is a
    cached, stat-invalidated lookup (`finesub.config`), negligible next to a request.
    """

    value = settings.PROXY.resolve()
    if value is None:
        value = settings.LEGACY_LLM_PROXY.resolve()
    value = (value or "").strip()
    if value.lower().startswith("socks"):
        # Not supported (owner 2026-11-09): httpx would need socksio, and its SOCKS handshake reads without a
        # timeout, so a proxy that stops answering would hang a call. Said here rather than as httpx's
        # "install socksio", which points the wrong way.
        raise ValueError(
            f"[api] proxy = {redact_credentials(value)!r}: SOCKS 代理不支持，填代理软件的 HTTP 端口（如 http://127.0.0.1:7890）"
        )
    return value or None


def client_kwargs() -> dict[str, Any]:
    """`proxy` / `trust_env` for one client, from the configured value.

    Three states, and the middle one is the reason this is not a bool:

    * key absent -- `trust_env=True`, i.e. `HTTPS_PROXY` still works. This is what the layer did before the
      setting existed, so adding it breaks nobody.
    * `proxy = "direct"` -- no proxy, and the environment is ignored.
    * `proxy = "<url>"` -- that proxy, and the environment is ignored. The file is more specific than the
      ambient environment, so it wins outright rather than being merged with it.
    """

    configured = configured_proxy()
    if configured is None:
        return {"trust_env": True}
    if configured.lower() == DIRECT:
        return {"proxy": None, "trust_env": False}
    return {"proxy": configured, "trust_env": False}


#: The connect phase of every remote API call that can set it apart (generation, uploads, cloud ASR). It bounds
#: only what happens before the server has said anything -- TCP, a proxy's CONNECT reply (`TunnelWatch`, shared
#: with the download family in `finesub_bootstrap.http_client`), TLS -- so no answer of the service's (429, 5xx, a long think, a slow upload) can be cut by it; what it decides
#: is how long a blocked host stays silent. That cost is paid per resolved address: at 45 s a blocked Gemini
#: upload sat silent for 169 s before its first line. A working route, proxy included, connects well inside 15 s;
#: one that needs longer is retried. Callers with something cheaper to fall back on use less (countTokens 5 s,
#: keyed web search 10 s).
CONNECT_TIMEOUT_SECONDS = 15.0


def api_http_client(timeout: Any = None, **kwargs: Any) -> httpx.Client:
    """An `httpx.Client` for a remote API call, routed by `[api] proxy`; its HTTPS requests carry a
    `TunnelWatch` (`finesub_bootstrap.http_client`)."""

    client = httpx.Client(timeout=timeout, **{**client_kwargs(), **kwargs})
    client.event_hooks["request"].append(watch_tunnel)
    return client


#: Failures where nothing about the request was wrong, so the identical request may well succeed: no response at
#: all (connect, DNS, proxy, a reset before the headers) or a timeout. Judged by *type*, never by message text --
#: `httpx.ReadError: [Errno 10054] 远程主机强迫关闭了一个现有的连接` carries none of the words a text match keys on,
#: and a text-only rule once made that a permanent failure that ended the whole LLM stage. The OpenAI and Anthropic
#: SDKs retry their whole connection-error family the same way. Left out on purpose: `UnsupportedProtocol` and
#: `LocalProtocolError` (a malformed request -- the same bytes fail the same way). A stream that broke off after
#: its headers is the LLM layer's `StreamInterrupted`, which its classifier adds, since `speech` may not import it.
RETRYABLE_TRANSPORT: tuple[type[BaseException], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    httpx.ProxyError,
    TimeoutError,
    ConnectionError,
)


def retryable_status(status: int) -> bool:
    """Whether an HTTP status says "not now" rather than "not this request": 408, 409, 429 and every 5xx (the set
    the OpenAI and Anthropic SDKs retry), and 425, which RFC 8470 says to send again. Any other 4xx is the service
    rejecting these bytes."""

    return status in (408, 409, 425, 429) or status >= 500


def is_retryable_http_error(exc: BaseException) -> bool:
    """By type, then by a status the exception *carries* (`response.status_code`, or a `status_code` attribute);
    never by its text. False for anything neither decides: a caller with its own fallback (the LLM layer reads
    quota wording off errors that have no status) runs it after this says no."""

    if isinstance(exc, RETRYABLE_TRANSPORT):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status > 0 and retryable_status(status)


def websocket_kwargs(host: str) -> dict[str, Any]:
    """The same three states as `client_kwargs`, as `websocket-client`'s connection options (35t Live): it reads
    `HTTPS_PROXY` itself when given no proxy, and a host on its no-proxy list goes direct."""

    from urllib.parse import unquote, urlsplit

    configured = configured_proxy()
    if configured is None:
        return {}
    if configured.lower() == DIRECT:
        return {"http_no_proxy": [host]}
    url = urlsplit(configured if "://" in configured else f"http://{configured}")
    out: dict[str, Any] = {"http_proxy_host": url.hostname, "http_proxy_port": url.port or 80, "proxy_type": "http"}
    if url.username:
        out["http_proxy_auth"] = (unquote(url.username), unquote(url.password or ""))
    return out


# -- reachability -------------------------------------------------------------------------------------------------

#: Failures that happened before any byte of a response: DNS, TCP connect, the TLS handshake, the proxy's CONNECT.
#: httpcore maps a socket timeout in any of those steps to `ConnectTimeout` and an `OSError` (a reset, a refused
#: port, an `SSLError`) to `ConnectError`, so every way the Great Firewall stops a request -- a poisoned answer, a
#: dropped SYN, a reset after the SNI -- lands here. A read timeout does not: the host answered the handshake.
CONNECT_FAILURES: tuple[type[BaseException], ...] = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError)

#: How long a run of connect failures must last before the host counts as unreachable. Long enough that a Wi-Fi
#: blip or one failed DNS lookup never qualifies -- by then the callers' own retries have already lost.
UNREACHABLE_AFTER_SECONDS = 30.0
#: Before the verdict, a failure this long after the previous one starts a new run instead of extending it.
STREAK_GAP_SECONDS = 60.0
#: After the verdict, optional callers send one probe per interval; it doubles on each failed probe, up to the cap.
PROBE_INTERVAL_SECONDS = 30.0
PROBE_INTERVAL_MAX_SECONDS = 600.0
#: How much of the exception text a warning quotes: enough to tell a certificate error, a dead local proxy and a
#: blocked host apart, which is the whole point of quoting it.
_DETAIL_CHARS = 120


def is_connect_failure(exc: BaseException) -> bool:
    """Whether `exc` (or what it wraps) failed before the host answered at all. By type, never by text -- plus the
    one timeout httpx types as a read: a proxy that never answered CONNECT (`TunnelWatch`)."""

    return any(isinstance(e, CONNECT_FAILURES) or in_tunnel_setup(e) for e in _chain(exc))


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """`exc` and what it wraps, explicit cause before implicit context."""

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        current = current.__cause__ or current.__context__


@dataclass(frozen=True)
class Route:
    """Which way a request leaves this machine: `key` tells routes apart, `label` is what a person reads."""

    key: str
    label: str


def _shown_proxy(url: str) -> str:
    # httpx reads an environment proxy written without a scheme as http://; so does the label, which is what
    # lets redact_credentials find the userinfo in `user:pass@host:port`.
    return redact_credentials(url if "://" in url else f"http://{url}")


def current_route() -> Route:
    """The route an `api_http_client` built now would take.

    Read from the same sources httpx reads: `[api] proxy` when set, else `urllib.request.getproxies()` -- the
    environment and, on Windows, the system proxy a VPN client switches on. A changed route is a new key, so
    turning a proxy on is retried at once instead of waiting out a verdict made without it. The label goes into
    warnings and error messages, so a proxy's userinfo is masked there; the key keeps the full address.
    """

    configured = configured_proxy()
    if configured is not None:
        if configured.lower() == DIRECT:
            return Route(key=DIRECT, label="直连")
        return Route(key=f"proxy:{configured}", label=f"经代理 {redact_credentials(configured)}")
    proxies = urllib.request.getproxies()
    if not proxies:
        return Route(key=DIRECT, label="直连")
    key = "env:" + ";".join(f"{name}={value}" for name, value in sorted(proxies.items()))
    shown = proxies.get("https") or proxies.get("all")
    return Route(key=key, label=f"经代理 {_shown_proxy(shown)}" if shown else "直连")


class Gate(enum.Enum):
    """What a caller may do with a host right now."""

    ALLOW = "allow"  # no verdict: as usual
    PROBE = "probe"  # verdict, optional caller: this one request checks whether the host is back
    SKIP = "skip"  # verdict, optional caller: use the fallback, send nothing
    ONCE = "once"  # verdict, required caller: one attempt, no retries


@dataclass
class _HostState:
    streak_start: float | None = None
    last_failure: float = -math.inf
    last_reached: float = -math.inf
    unreachable: bool = False
    probe_interval: float = PROBE_INTERVAL_SECONDS
    next_probe_at: float = 0.0
    probe_granted_at: float | None = None
    probe_in_flight: bool = False


class HostUnreachableError(RuntimeError):
    """A required call gave up on a host judged unreachable. Declares itself `unavailable` to the route policy,
    so a candidate chain moves on to the next provider instead of treating it as a timeout."""

    route_failure_kind = "unavailable"

    def __init__(self, host: str, route: Route, cause: BaseException | None = None) -> None:
        detail = _describe(cause) if cause is not None else "judged unreachable"
        super().__init__(
            f"cannot reach {host} ({route.label}): {detail}. "
            "若这台机器需要代理才能访问它，在 config.toml 设 [api] proxy；若已开代理，检查它是否在运行"
        )
        self.host = host
        self.route = route


def _describe(exc: BaseException) -> str:
    """The failure as a warning quotes it: type and the start of its text, or what a `ReadTimeout` from an
    unanswered CONNECT really was."""

    if any(in_tunnel_setup(e) for e in (exc, exc.__cause__, exc.__context__) if e is not None):
        return f"代理没有回应 CONNECT（{type(exc).__name__}）"
    return f"{type(exc).__name__}: {_detail(exc)}"


def _detail(exc: BaseException | None) -> str:
    text = " ".join(str(exc or "").split())
    return text[:_DETAIL_CHARS] + ("…" if len(text) > _DETAIL_CHARS else "")


def warn_visibly(code: str, message: str, *, action: str = "") -> None:
    """A warning that reaches a person even outside a bound reporter: silence is the failure this guards against."""

    reporter = current_reporter()
    if not reporter_delivers(reporter):
        reporter = TerminalReporter(sys.stderr)
    reporter.warning(code, message, action=action)


class HostBreaker:
    """Per process, per `(host, route)`: has this host stopped answering connects?

    Callers report every connect failure (:meth:`note_connect_failure`) and every response (:meth:`note_reached`),
    and ask before each attempt. The first failure of a run is reported at once, as a plain fact. A run of failures
    lasting :data:`UNREACHABLE_AFTER_SECONDS` is the verdict; from then on optional callers (token counting, web
    search) skip the host except for one probe per interval, and required callers (generation, uploads) still try
    -- they have nothing else to use -- but once, without retries, so the next call after the network comes back
    succeeds. Any response clears the verdict.

    Not persisted: one process runs a whole batch, so each process pays for the discovery once. A forked child
    starts clean. Quota and overload stay with the rate limiter's cooldowns; this only knows "connects or not".
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str], _HostState] = {}
        self._pid = os.getpid()

    def now(self) -> float:
        return self._clock()

    def reset(self) -> None:
        """Forget every host (tests share the process-wide breaker)."""

        with self._lock:
            self._states.clear()

    def _state(self, host: str, route: Route) -> _HostState:
        if self._pid != os.getpid():
            self._states.clear()
            self._pid = os.getpid()
        return self._states.setdefault((host, route.key), _HostState())

    def unreachable(self, host: str, route: Route) -> bool:
        with self._lock:
            return self._state(host, route).unreachable

    def gate_optional(self, host: str, route: Route) -> Gate:
        with self._lock:
            state = self._state(host, route)
            if not state.unreachable:
                return Gate.ALLOW
            now = self._clock()
            if now < state.next_probe_at or state.probe_in_flight:
                return Gate.SKIP
            # The next one is due an interval after this grant, and not before this one has ended (`probing`).
            state.probe_granted_at = now
            state.probe_in_flight = True
            state.next_probe_at = now + state.probe_interval
            return Gate.PROBE

    @contextmanager
    def probing(self, host: str, route: Route, gate: Gate) -> Iterator[None]:
        """Around the attempt a :meth:`gate_optional` answer was for: a probe holds its slot until the block ends,
        however it ends. Released by the holder, not inferred from request records: a probe may be several requests
        on other threads (a hedged pair), and a required call failing meanwhile is not the probe ending."""

        try:
            yield
        finally:
            if gate is Gate.PROBE:
                with self._lock:
                    self._state(host, route).probe_in_flight = False

    def gate_required(self, host: str, route: Route) -> Gate:
        return Gate.ONCE if self.unreachable(host, route) else Gate.ALLOW

    def note_reached(self, host: str, route: Route) -> None:
        with self._lock:
            state = self._state(host, route)
            was_unreachable = state.unreachable
            state.last_reached = self._clock()
            state.streak_start = None
            state.unreachable = False
            state.probe_interval = PROBE_INTERVAL_SECONDS
            state.next_probe_at = 0.0
            state.probe_granted_at = None
        if was_unreachable:
            warn_visibly("host-reachable", f"{host}（{route.label}）恢复连通")

    def note_connect_failure(self, host: str, route: Route, exc: BaseException, *, sent_at: float) -> None:
        """Count one connect failure of a request sent at `sent_at` (this breaker's clock)."""

        first = verdict = False
        with self._lock:
            state = self._state(host, route)
            if sent_at < state.last_reached:
                return  # a later-sent request got through: this one says nothing about now
            now = self._clock()
            if state.unreachable:
                if state.probe_granted_at is not None and sent_at >= state.probe_granted_at:
                    state.probe_interval = min(state.probe_interval * 2, PROBE_INTERVAL_MAX_SECONDS)
                    state.next_probe_at = now + state.probe_interval
                    state.probe_granted_at = None
                state.last_failure = now
                return
            if state.streak_start is None or now - state.last_failure > STREAK_GAP_SECONDS:
                state.streak_start = sent_at
                first = True
            state.last_failure = now
            if now - state.streak_start >= UNREACHABLE_AFTER_SECONDS:
                state.unreachable = verdict = True
                state.probe_interval = PROBE_INTERVAL_SECONDS
                state.next_probe_at = now + PROBE_INTERVAL_SECONDS
                state.probe_granted_at = None
        if first:
            warn_visibly("host-connect-failed", f"连不上 {host}（{route.label}）：{_describe(exc)}")
        if verdict:
            warn_visibly(
                "host-unreachable",
                f"{host}（{route.label}）连续 {UNREACHABLE_AFTER_SECONDS:.0f} 秒连不上，判定为不可达："
                "token 计数改用本地估算、联网检索跳过它，生成与上传每次只试一次、不再重试",
                action="若这台机器需要代理才能访问它，在 config.toml 设 [api] proxy；若已开代理，检查它是否在运行",
            )


#: The one breaker every API caller in this process shares.
HOST_BREAKER = HostBreaker()


class Reach:
    """One request's part in the reachability record; see :func:`track_reach`."""

    def __init__(self, breaker: HostBreaker, host: str, route: Route) -> None:
        self.breaker = breaker
        self.host = host
        self.route = route
        self.sent_at = breaker.now()
        self._reached = False

    def reached(self) -> None:
        """The host answered (headers arrived, whatever the status)."""

        if not self._reached:
            self._reached = True
            self.breaker.note_reached(self.host, self.route)

    def failed(self, exc: BaseException) -> None:
        """The request ended in `exc`; counted only if it never got past connecting."""

        if not self._reached and is_connect_failure(exc):
            self.breaker.note_connect_failure(self.host, self.route, exc, sent_at=self.sent_at)


def start_reach(url: str, *, breaker: HostBreaker | None = None, route: Route | None = None) -> Reach:
    """A :class:`Reach` for a request about to go to `url`, for callers that cannot wrap it in :func:`track_reach`."""

    return Reach(breaker or HOST_BREAKER, urlsplit(url).hostname or url, route or current_route())


@contextmanager
def track_reach(
    url: str, *, breaker: HostBreaker | None = None, route: Route | None = None
) -> Iterator[Reach]:
    """Record whether the request made inside the block reached `url`'s host.

    Wrap the send itself, nothing that waits before it (a rate limiter's queue would read as connect time). A
    connect failure leaving the block is counted; leaving it normally, or with an `httpx.HTTPStatusError`
    (`raise_for_status` on a 503 is still an answer), counts as reached. A streamed body calls
    :meth:`Reach.reached` itself as soon as the headers arrive, so an error raised after them is not mistaken
    for silence.
    """

    reach = start_reach(url, breaker=breaker, route=route)
    try:
        yield reach
    except BaseException as exc:
        if any(isinstance(e, httpx.HTTPStatusError) for e in _chain(exc)):
            reach.reached()
        else:
            reach.failed(exc)
        raise
    reach.reached()

