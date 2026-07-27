from __future__ import annotations

import hashlib
from pathlib import Path

import httpx
import pytest

from desktop.backend.common.http_client import NetworkRoute
from desktop.backend.common.models import DownloadAsset
from desktop.backend.resources.downloader import DigestMismatch, download_asset


def _asset(url: str, body: bytes, *, sha256: str | None = None) -> DownloadAsset:
    return DownloadAsset(
        url=url,
        size=len(body),
        sha256=sha256 or hashlib.sha256(body).hexdigest(),
    )


def test_download_resumes_part_file_and_verifies_sha256(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = b"verified-resource-content"
    server = serve_asset(body)
    destination = tmp_path / "asset.zip"
    destination.with_suffix(".zip.part").write_bytes(body[:5])
    events = []

    result = download_asset(_asset(server.url, body), destination, events.append)

    assert result == destination
    assert result.read_bytes() == body
    assert server.range_headers == ["bytes=5-"]
    assert events[-1].downloaded == len(body)
    assert events[-1].total == len(body)


def test_download_removes_final_file_on_digest_mismatch(
    serve_asset,
    tmp_path: Path,
) -> None:
    body = b"tampered"
    server = serve_asset(body)
    destination = tmp_path / "asset.zip"

    with pytest.raises(DigestMismatch):
        download_asset(
            _asset(server.url, body, sha256="0" * 64),
            destination,
            lambda event: None,
        )

    assert not destination.exists()


def test_download_restarts_when_server_ignores_range(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = b"complete-body"
    destination = tmp_path / "asset.bin"
    destination.with_suffix(".bin.part").write_bytes(b"old")

    class Response:
        status_code = 200
        headers = {"Content-Length": str(len(body))}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def iter_bytes(self, chunk_size: int):
            yield body

        def raise_for_status(self) -> None:
            return None

    class Client:
        def stream(self, *args, **kwargs):
            return Response()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        "desktop.backend.resources.downloader.httpx.Client",
        lambda **kwargs: Client(),
    )

    result = download_asset(
        _asset("https://example.invalid/asset", body),
        destination,
        lambda event: None,
    )

    assert result.read_bytes() == body


def test_download_resumes_from_latest_partial_file_after_route_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    body = b"complete-body"
    destination = tmp_path / "asset.bin"
    requested_ranges: list[str | None] = []

    class Response:
        status_code = 206

        def __init__(self, route: str) -> None:
            self.route = route

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int):
            if self.route == "first":
                yield body[:4]
                raise httpx.ReadError("route disconnected")
            yield body[4:]

    class Client:
        def __init__(self, route: str) -> None:
            self.route = route

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def stream(self, method: str, url: str, *, headers: dict[str, str]):
            requested_ranges.append(headers.get("Range"))
            return Response(self.route)

    routes = [NetworkRoute("first", None), NetworkRoute("second", None)]
    monkeypatch.setattr(
        "desktop.backend.resources.downloader.network_routes",
        lambda: routes,
    )
    monkeypatch.setattr(
        "desktop.backend.resources.downloader.create_client",
        lambda route, **kwargs: Client(route.label),
    )

    result = download_asset(
        _asset("https://example.invalid/asset", body),
        destination,
        lambda event: None,
    )

    assert result.read_bytes() == body
    assert requested_ranges == [None, "bytes=4-"]
