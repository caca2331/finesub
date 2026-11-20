"""Fixtures for the provisioning-layer tests.

`serve_asset` came from the desktop suite along with its only two users
(`test_downloader`, `test_resource_manager`); nothing over there serves an
asset any more. It stands up a real local HTTP server rather than patching the
client, because what these tests are about -- ranged resumption, digests,
partial bodies -- is what a real response does.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest


@dataclass
class ServedAsset:
    body: bytes
    server: ThreadingHTTPServer
    thread: threading.Thread
    range_headers: list[str] = field(default_factory=list)
    # Set to let a body held by `pause_after` carry on.
    release: threading.Event = field(default_factory=threading.Event)

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/asset"

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


@pytest.fixture
def serve_asset() -> Iterator[Callable[[bytes], ServedAsset]]:
    active: list[ServedAsset] = []

    def factory(body: bytes, *, pause_after: int = 0) -> ServedAsset:
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
                if not pause_after or record.release.is_set():
                    self.wfile.write(payload)
                    return
                # Held mid-file until the test releases it, so the client is
                # still downloading when the test kills it, however slowly the
                # test gets there -- killing the client mid-file is the only
                # way to observe what a half-finished download leaves behind.
                self.wfile.write(payload[:pause_after])
                self.wfile.flush()
                record.release.wait(timeout=60)
                try:
                    self.wfile.write(payload[pause_after:])
                except OSError:
                    return  # the client was killed meanwhile

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        # `shutdown()` waits out one poll; the default 0.5 s was paid in every user's teardown.
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        record = ServedAsset(body=body, server=server, thread=thread)
        active.append(record)
        thread.start()
        return record

    yield factory

    for asset in active:
        asset.close()
