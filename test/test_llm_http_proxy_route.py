"""`[llm] proxy` actually routes an LLM API call, end to end.

`test_llm_http_client.py` pins what the setting *resolves to* and guards the
seam; neither would notice if httpx stopped honouring the argument, or if a
call site were handed the kwargs and dropped them. So this drives a real
request over a real socket through a real (tiny) forward proxy, and reads the
evidence off the request line: a proxied HTTP request carries the **absolute
URI** (`GET http://host:port/path`), a direct one carries only the path.

Two servers rather than one so the assertion is "which process received it",
not just "how was the line written". Everything is on 127.0.0.1 with an
ephemeral port; nothing leaves the machine.

**Opt-in** (`pytest --run-network-mock`). It is quick and hermetic, but binding
ports and starting server threads is not something every `pytest -q` should be
doing; run it when the route itself is what changed.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from finesub.llm import http as llm_http

pytestmark = pytest.mark.network_mock


class _Recorder(ThreadingHTTPServer):
    """An HTTP server that remembers the request lines it was given."""

    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.seen: list[tuple[str, str]] = []

    @property
    def origin(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _record(self) -> None:
        self.server.seen.append((self.command, self.path))  # type: ignore[attr-defined]
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        body = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _record
    do_POST = _record

    def log_message(self, *_args) -> None:  # keep pytest output clean
        return


@pytest.fixture
def servers():
    origin, proxy = _Recorder(), _Recorder()
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (origin, proxy)
    ]
    for thread in threads:
        thread.start()
    try:
        yield origin, proxy
    finally:
        for server in (origin, proxy):
            server.shutdown()
            server.server_close()


@pytest.fixture(autouse=True)
def _forget_config():
    llm_http.reset_proxy_cache()
    yield
    llm_http.reset_proxy_cache()


def _configure(tmp_path: Path, monkeypatch, proxy_value: str | None) -> None:
    from finesub import config as config_module

    body = "[llm]\n" + (f'proxy = "{proxy_value}"\n' if proxy_value else "")
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(path))
    config_module.clear_config_cache()
    llm_http.reset_proxy_cache()


def _post(url: str) -> dict:
    """A real LLM-layer call site, not the factory -- that is the point."""

    from finesub.llm.provider_transports import _post_json

    return _post_json(
        f"{url}/v1/chat",
        headers={"Authorization": "Bearer test"},
        payload={"hello": "world"},
        timeout=5.0,
    )


def test_a_configured_proxy_actually_carries_the_request(
    tmp_path, monkeypatch, servers
) -> None:
    origin, proxy = servers
    _configure(tmp_path, monkeypatch, proxy.origin)

    assert _post(origin.origin) == {"ok": True}

    # The proxy process received it; the origin was never contacted directly.
    assert origin.seen == []
    assert len(proxy.seen) == 1
    method, target = proxy.seen[0]
    assert method == "POST"
    # Absolute URI: the proxy is being asked to fetch it, which is the proof.
    assert target == f"{origin.origin}/v1/chat"


def test_direct_bypasses_the_proxy_and_the_environment(
    tmp_path, monkeypatch, servers
) -> None:
    """`proxy = "direct"` has to beat an ambient `HTTPS_PROXY`/`HTTP_PROXY`.

    That is its whole reason to exist: on a machine proxied at the environment
    level there would otherwise be no way to say "not for LLM calls".
    """

    origin, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy.origin)
    monkeypatch.setenv("ALL_PROXY", proxy.origin)
    _configure(tmp_path, monkeypatch, llm_http.DIRECT)

    assert _post(origin.origin) == {"ok": True}

    assert proxy.seen == []
    assert origin.seen == [("POST", "/v1/chat")]


def test_no_setting_lets_the_environment_decide(
    tmp_path, monkeypatch, servers
) -> None:
    """Absent means today's behaviour survives: `HTTP_PROXY` still routes.

    This is what makes adding the key a non-breaking change, so it is worth a
    real request rather than a kwargs assertion.
    """

    origin, proxy = servers
    monkeypatch.setenv("HTTP_PROXY", proxy.origin)
    _configure(tmp_path, monkeypatch, None)

    assert _post(origin.origin) == {"ok": True}

    assert origin.seen == []
    assert [method for method, _ in proxy.seen] == ["POST"]
