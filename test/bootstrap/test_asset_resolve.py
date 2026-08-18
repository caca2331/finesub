from __future__ import annotations

import pytest

from finesub_bootstrap.asset_resolve import AssetResolutionError, resolve_asset
from finesub_bootstrap.http_client import NetworkRoute
from finesub_bootstrap.models import DownloadAsset, ResolvableAsset


DIGEST = "c0e252e6dcb2719907138fe6f01216d895cb442c3197833b1015f14e66f8b4b3"
ASSET_URL = (
    "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
    "ffmpeg-n9.0-latest-win64-lgpl-9.0.zip"
)


class _Response:
    def __init__(self, payload, *, status_code: int = 200, headers=None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError("raise_for_status should not decide these cases")

    def json(self):
        return self._payload


@pytest.fixture
def api(monkeypatch):
    """Serve one canned API response and record the endpoint that was asked for."""

    calls: list[str] = []

    def install(response: _Response) -> list[str]:
        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def get(self, endpoint: str):
                calls.append(endpoint)
                return response

        monkeypatch.setattr(
            "finesub_bootstrap.asset_resolve.network_routes",
            lambda: [NetworkRoute("直连", None)],
        )
        monkeypatch.setattr(
            "finesub_bootstrap.asset_resolve.create_client",
            lambda route, *, timeout, headers=None: Client(),
        )
        return calls

    return install


def _release(name: str, *, digest: str | None = DIGEST, size: int | None = 147007158):
    entry: dict = {"name": name}
    if digest is not None:
        entry["digest"] = f"sha256:{digest}"
    if size is not None:
        entry["size"] = size
    return _Response({"assets": [{"name": "unrelated.zip"}, entry]})


def test_a_pinned_asset_is_returned_untouched_without_asking_anyone(api) -> None:
    # Four of the five resources are pinned; resolution must be free for them,
    # not an API call each provisioning run.
    calls = api(_release("ffmpeg-n9.0-latest-win64-lgpl-9.0.zip"))
    pinned = DownloadAsset(url=ASSET_URL, size=10, sha256="a" * 64)

    assert resolve_asset(pinned) is pinned
    assert calls == []


def test_the_digest_and_size_come_from_the_release_api(api) -> None:
    calls = api(_release("ffmpeg-n9.0-latest-win64-lgpl-9.0.zip"))

    resolved = resolve_asset(
        ResolvableAsset(url=ASSET_URL, digest_from="github-release-api")
    )

    assert isinstance(resolved, DownloadAsset)
    assert resolved.url == ASSET_URL
    assert resolved.sha256 == DIGEST
    assert resolved.size == 147007158
    # `/releases/tags/latest`, never `/releases/latest`: BtbN's tag is literally
    # named "latest", which is not the same thing as "the newest release".
    assert calls == [
        "https://api.github.com/repos/BtbN/FFmpeg-Builds/releases/tags/latest"
    ]


def test_a_percent_encoded_tag_is_asked_for_by_its_real_name(api) -> None:
    # Our own CT2 wheel lives under `ct2-4.8.1+finesub0.4.0`, where the `+` is
    # `%2B` in the download URL but must not be `%252B` in the API path.
    calls = api(_release("ctranslate2.whl"))

    resolve_asset(
        ResolvableAsset(
            url=(
                "https://github.com/caca2331/finesub/releases/download/"
                "ct2-4.8.1%2Bfinesub0.4.0/ctranslate2.whl"
            ),
            digest_from="github-release-api",
        )
    )

    assert calls == [
        "https://api.github.com/repos/caca2331/finesub/releases/tags/"
        "ct2-4.8.1%2Bfinesub0.4.0"
    ]


def test_an_asset_missing_from_the_tag_is_named_in_the_error(api) -> None:
    api(_release("some-other-build.zip"))

    with pytest.raises(AssetResolutionError, match="ffmpeg-n9.0-latest"):
        resolve_asset(
            ResolvableAsset(url=ASSET_URL, digest_from="github-release-api")
        )


def test_an_asset_without_a_recorded_digest_refuses_rather_than_trusting_it(
    api,
) -> None:
    # GitHub only started recording digests recently, so an old asset can answer
    # without one. There is nothing safe to fall back to.
    api(_release("ffmpeg-n9.0-latest-win64-lgpl-9.0.zip", digest=None))

    with pytest.raises(AssetResolutionError, match="摘要"):
        resolve_asset(
            ResolvableAsset(url=ASSET_URL, digest_from="github-release-api")
        )


def test_a_spent_rate_limit_reads_as_come_back_later(api) -> None:
    api(
        _Response(
            {"message": "API rate limit exceeded"},
            status_code=403,
            headers={"x-ratelimit-remaining": "0"},
        )
    )

    with pytest.raises(AssetResolutionError, match="速率限制"):
        resolve_asset(
            ResolvableAsset(url=ASSET_URL, digest_from="github-release-api")
        )


def test_a_url_that_is_not_a_github_release_asset_is_a_manifest_error() -> None:
    with pytest.raises(AssetResolutionError, match="github-release-api"):
        resolve_asset(
            ResolvableAsset(
                url="https://example.invalid/downloads/tool.zip",
                digest_from="github-release-api",
            )
        )


def test_a_half_filled_asset_entry_is_a_validation_error() -> None:
    # `extra="forbid"` on both shapes is what discriminates them, so a pin with a
    # `digest_from` bolted on -- or a size with no digest -- must not validate as
    # either. Otherwise "which of the two is this" becomes a silent guess.
    from pydantic import ValidationError

    from finesub_bootstrap.models import ResourceSpec

    def spec(asset: dict) -> dict:
        return {
            "id": "ffmpeg",
            "version": "n9.0-latest",
            "destination": "runtime",
            "directory": "ffmpeg",
            "archive_type": "zip",
            "required_files": ["bin/ffmpeg.exe"],
            "asset": asset,
        }

    for asset in (
        {"url": ASSET_URL, "size": 1, "digest_from": "github-release-api"},
        {"url": ASSET_URL, "size": 1},
        {"url": ASSET_URL, "sha256": "a" * 64, "digest_from": "github-release-api"},
        {"url": ASSET_URL},
    ):
        with pytest.raises(ValidationError):
            ResourceSpec.model_validate(spec(asset))
