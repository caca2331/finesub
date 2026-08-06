from __future__ import annotations

from dataclasses import dataclass
import os
import socket
from urllib.parse import urlparse

import httpx

try:
    import winreg
except ImportError:  # pragma: no cover - Windows production code
    winreg = None  # type: ignore[assignment]


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
    for protocol in ("https", "http", "socks", "socks5"):
        address = entries.get(protocol)
        if address:
            scheme = "socks5" if protocol.startswith("socks") else "http"
            return _normalise_proxy(address, scheme)
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
        routes.append(NetworkRoute(f"{label} {proxy}", proxy))

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
    )


def is_connection_failure(error: BaseException) -> bool:
    return isinstance(error, httpx.TransportError)


def connection_error(attempts: list[tuple[str, BaseException]]) -> NetworkConnectionError:
    details = "；".join(f"{label}: {error}" for label, error in attempts)
    return NetworkConnectionError(f"所有连接方式均失败（{details}）")
