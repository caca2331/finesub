from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from typing import Any, Callable
from urllib.parse import urlparse, urlsplit, urlunsplit

import httpx

try:
    import winreg
except ImportError:  # pragma: no cover - Windows production code
    winreg = None  # type: ignore[assignment]


#: The connect phase of a download or a release lookup. It bounds only what happens before the server has said
#: anything (TCP, a proxy's CONNECT reply, TLS), never a slow transfer, which `read` governs; what it decides is
#: how long a blocked route stays silent before the next one in `network_routes()` is tried, and that is paid per
#: resolved address. A working route, proxy included, connects well inside 15 s. Same value and argument as the
#: API side's `finesub.api_http.CONNECT_TIMEOUT_SECONDS` (this package does not import `finesub`).
CONNECT_TIMEOUT_SEC = 15.0


@dataclass(frozen=True)
class NetworkRoute:
    label: str
    proxy: str | None


class NetworkConnectionError(httpx.ConnectError):
    """Raised after every usable proxy route and direct access have failed."""


def _normalise_proxy(value: str, scheme: str = "http") -> str:
    value = value.strip()
    if "://" not in value:
        value = f"{scheme}://{value}"
    return value


def _without_userinfo(proxy: str) -> str:
    """`http://user:pw@host:port` -> `http://***@host:port`, for a label.

    The label goes into error text and run logs, and from there into a
    feedback pack; the proxy itself keeps its credentials for the request.
    """

    parts = urlsplit(proxy)
    if "@" not in parts.netloc:
        return proxy
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit(parts._replace(netloc=f"***@{host}"))


def _windows_proxy() -> str | None:
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as key:
            enabled = int(winreg.QueryValueEx(key, "ProxyEnable")[0])
            server = str(winreg.QueryValueEx(key, "ProxyServer")[0]).strip()
    except (OSError, TypeError, ValueError):
        return None
    if not enabled or not server:
        return None

    if "=" not in server:
        return _normalise_proxy(server)

    entries: dict[str, str] = {}
    for item in server.split(";"):
        protocol, separator, address = item.partition("=")
        if separator and address.strip():
            entries[protocol.strip().lower()] = address.strip()
    # A SOCKS-only system proxy is not used (SOCKS is not supported): such a machine downloads directly.
    for protocol in ("https", "http"):
        address = entries.get(protocol)
        if address:
            return _normalise_proxy(address)
    return None


def _environment_proxies() -> list[str]:
    values: list[str] = []
    for name in (
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
    ):
        value = os.environ.get(name, "").strip()
        if value and value not in values:
            values.append(value)
    return values


def _local_proxy_is_listening(proxy: str) -> bool:
    parsed = urlparse(proxy)
    host = parsed.hostname
    try:
        port = parsed.port
    except ValueError:
        return False
    if not host or not port:
        return False
    if host.lower() not in {"127.0.0.1", "localhost", "::1"}:
        return True
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def network_routes() -> list[NetworkRoute]:
    """Return live proxy routes followed by a direct-access fallback."""
    routes: list[NetworkRoute] = []
    seen: set[str] = set()

    system_proxy = _windows_proxy()
    candidates = (
        [("Windows 系统代理", system_proxy)] if system_proxy else []
    ) + [("环境代理", value) for value in _environment_proxies()]

    for label, value in candidates:
        if value is None:
            continue
        proxy = _normalise_proxy(value)
        if proxy in seen or not _local_proxy_is_listening(proxy):
            continue
        seen.add(proxy)
        routes.append(NetworkRoute(f"{label} {_without_userinfo(proxy)}", proxy))

    routes.append(NetworkRoute("直连", None))
    return routes


def apply_network_environment(environment: dict[str, str]) -> None:
    """Make subprocess downloads follow the same live route as HTTPX."""
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        environment.pop(name, None)
    route = network_routes()[0]
    if route.proxy is None:
        return
    environment["HTTP_PROXY"] = route.proxy
    environment["HTTPS_PROXY"] = route.proxy
    environment["ALL_PROXY"] = route.proxy


def create_client(
    route: NetworkRoute,
    *,
    timeout: httpx.Timeout,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    return httpx.Client(
        proxy=route.proxy,
        trust_env=False,
        headers=headers,
        follow_redirects=True,
        timeout=timeout,
        event_hooks={"request": [watch_tunnel]},
    )


_TUNNEL_WATCH = "finesub_tunnel_watch"
_HEADERS_ENDED = ("http11.receive_response_headers.complete", "http11.receive_response_headers.failed")


class TunnelWatch:
    """An httpcore `trace` hook that keeps a proxy's CONNECT exchange in the connect phase.

    Through a proxy, an HTTPS request first sends `CONNECT host:443` and waits for the proxy's reply, and httpcore
    waits for that reply with the *read* timeout: a proxy that accepts the connection but never answers (hung, or
    one that dials the blocked upstream before replying) cost the whole read timeout -- a download's 120 s, a
    generation call's hard deadline -- and ended as a `ReadTimeout`, which no reachability rule counts. This waits
    for that one reply with the connect timeout instead and marks the request while it does, so
    :func:`in_tunnel_setup` sees the timeout for what it is.

    Keyed on the CONNECT request itself (httpcore traces it with the real request's extensions), not on the order
    of events: an `https://` proxy has its own TLS handshake before the CONNECT. If an httpcore release renames
    the event this stays inert and the old behaviour returns, which `test_llm_api_reachability.py` pins with a
    real socket.
    """

    def __init__(self, chained: Callable[[str, dict], Any] | None = None) -> None:
        self.pending = False
        self._chained = chained
        self._saved_read: tuple[dict, Any] | None = None

    def __call__(self, name: str, info: dict) -> None:
        # Only `.started` carries the request; `.complete` and `.failed` carry just the outcome, so the end of the
        # CONNECT reply is recognised by the state its start left.
        if name == "http11.receive_response_headers.started":
            if getattr(info.get("request"), "method", None) == b"CONNECT":
                self.pending = True
                timeouts = info["request"].extensions.get("timeout")
                if isinstance(timeouts, dict) and timeouts.get("connect") is not None:
                    # The CONNECT request shares the real request's extensions: put the read timeout back after.
                    self._saved_read = (timeouts, timeouts.get("read"))
                    timeouts["read"] = timeouts["connect"]
        elif self.pending and name in _HEADERS_ENDED:
            # Still pending after `.failed`: that is the timeout `in_tunnel_setup` is asked about.
            self.pending = name.endswith(".failed")
            if self._saved_read is not None:
                timeouts, read = self._saved_read
                timeouts["read"] = read
                self._saved_read = None
        if self._chained is not None:
            self._chained(name, info)


def watch_tunnel(request: httpx.Request) -> None:
    """An httpx request hook that gives every HTTPS request a :class:`TunnelWatch`."""

    if request.url.scheme == "https" and _TUNNEL_WATCH not in request.extensions:
        watch = TunnelWatch(request.extensions.get("trace"))
        request.extensions["trace"] = watch
        request.extensions[_TUNNEL_WATCH] = watch


def in_tunnel_setup(exc: BaseException) -> bool:
    """A timeout while a proxy's CONNECT reply was still awaited (:class:`TunnelWatch`)."""

    if not isinstance(exc, httpx.TimeoutException):
        return False
    request = getattr(exc, "_request", None)
    watch = request.extensions.get(_TUNNEL_WATCH) if request is not None else None
    return watch is not None and watch.pending


def is_connection_failure(error: BaseException) -> bool:
    return isinstance(error, httpx.TransportError)


def connection_error(attempts: list[tuple[str, BaseException]]) -> NetworkConnectionError:
    details = "；".join(f"{label}: {error}" for label, error in attempts)
    return NetworkConnectionError(f"所有连接方式均失败（{details}）")
