from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest


@pytest.fixture(autouse=True)
def managed_data_root(tmp_path_factory, monkeypatch) -> None:
    """Keep personal data out of the developer's real `%LOCALAPPDATA%`.

    `AppPaths` now derives the data root from the environment, so without this
    a test that builds paths would resolve to -- and eventually write into --
    the machine's own FineSub installation.
    """

    monkeypatch.setenv(
        "LOCALAPPDATA", str(tmp_path_factory.mktemp("LocalAppData"))
    )
    # The developer machine may opt out of .env protection globally
    # (FINESUB_ENV_PROTECT=0, a transition hatch); tests need the default.
    monkeypatch.delenv("FINESUB_ENV_PROTECT", raising=False)


@dataclass
class ServedAsset:
    body: bytes
    server: ThreadingHTTPServer
    thread: threading.Thread
    range_headers: list[str] = field(default_factory=list)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/asset"

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


@pytest.fixture
def serve_asset() -> Iterator[Callable[[bytes], ServedAsset]]:
    active: list[ServedAsset] = []

    def factory(body: bytes) -> ServedAsset:
        record: ServedAsset

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                range_header = self.headers.get("Range")
                if range_header:
                    record.range_headers.append(range_header)
                    start = int(range_header.removeprefix("bytes=").removesuffix("-"))
                    payload = body[start:]
                    self.send_response(206)
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(body) - 1}/{len(body)}",
                    )
                else:
                    payload = body
                    self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        record = ServedAsset(body=body, server=server, thread=thread)
        active.append(record)
        thread.start()
        return record

    yield factory

    for asset in active:
        asset.close()
