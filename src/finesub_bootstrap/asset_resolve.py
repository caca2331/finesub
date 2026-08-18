"""Turn a `ResolvableAsset` into a pinned `DownloadAsset` by asking the host.

Why this exists at all is written down on `ResolvableAsset`; the short version is
that some upstreams replace the bytes behind a stable URL on every build, so the
only way to both follow that URL and still verify what arrived is to ask the host
what it should hash to.

Two properties this deliberately keeps:

* The answer comes from the release API over its own TLS connection, not from the
  file transfer. A proxy that rewrote the download would have to rewrite the API
  response too, and the runtime resources do not go through the CN file mirror in
  the first place (only `model_fetch` does).
* Resolution happens once, above the downloader, producing an ordinary pinned
  asset. Everything downstream -- resume bounds, progress totals, the size and
  digest checks -- keeps working on a pinned asset and needs no notion of a
  moving one.

The trade this makes, and why only ffmpeg gets it, is
`docs/cli-bootstrap-logging-download-plan.md` 5.6.
"""

from __future__ import annotations

import re
from urllib.parse import quote, unquote, urlparse

import httpx

from finesub_bootstrap.http_client import (
    connection_error,
    create_client,
    is_connection_failure,
    network_routes,
)
from finesub_bootstrap.models import DownloadAsset, ResolvableAsset

#: `https://github.com/<owner>/<repo>/releases/download/<tag>/<name>`
_RELEASE_ASSET_PATH = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/releases/download/(?P<tag>[^/]+)/(?P<name>[^/]+)$"
)

_API_HOST = "https://api.github.com"
_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    "User-Agent": "finesub-bootstrap",
}


class AssetResolutionError(RuntimeError):
    """The host could not tell us what the current bytes should hash to."""


def resolve_asset(asset: DownloadAsset | ResolvableAsset) -> DownloadAsset:
    """Return `asset` pinned: unchanged if it already is, resolved if it is not."""

    if isinstance(asset, DownloadAsset):
        return asset

    owner, repo, tag, name = _parse_release_asset_url(asset.url)
    entry = _find_asset(owner, repo, tag, name)

    digest = str(entry.get("digest") or "")
    prefix = "sha256:"
    if not digest.startswith(prefix) or len(digest) != len(prefix) + 64:
        # GitHub only started recording asset digests recently, so an old asset
        # can answer without one. There is nothing safe to fall back to: naming
        # the gap beats installing an unverified executable.
        raise AssetResolutionError(
            f"GitHub 未提供 {name} 的 sha256 摘要，无法校验下载内容；"
            f"请改为在 manifest 中钉死该资产（tag {tag}）"
        )

    size = entry.get("size")
    if not isinstance(size, int) or size <= 0:
        raise AssetResolutionError(f"GitHub 返回的 {name} 大小无效：{size!r}")

    return DownloadAsset(
        url=asset.url,
        size=size,
        sha256=digest[len(prefix):],
    )


def _parse_release_asset_url(url: str) -> tuple[str, str, str, str]:
    parsed = urlparse(url)
    match = _RELEASE_ASSET_PATH.match(parsed.path)
    if parsed.netloc != "github.com" or match is None:
        raise AssetResolutionError(
            f'digest_from "github-release-api" needs a GitHub release asset URL, '
            f"got {url}"
        )
    return (
        match["owner"],
        match["repo"],
        unquote(match["tag"]),
        unquote(match["name"]),
    )


def _find_asset(owner: str, repo: str, tag: str, name: str) -> dict:
    # `/releases/tags/<tag>` and not `/releases/latest`: a tag literally named
    # "latest" -- which is what BtbN publishes -- is an ordinary tag, and the
    # `latest` endpoint means "newest release", which is not the same thing.
    endpoint = (
        f"{_API_HOST}/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
        f"/releases/tags/{quote(tag, safe='')}"
    )
    payload = _get_json(endpoint)
    for entry in payload.get("assets") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    raise AssetResolutionError(
        f"{owner}/{repo} 的 tag {tag} 下没有名为 {name} 的资产"
    )


def _get_json(endpoint: str) -> dict:
    timeout = httpx.Timeout(connect=20.0, read=30.0, write=30.0, pool=20.0)
    attempts: list[tuple[str, BaseException]] = []
    for route in network_routes():
        try:
            with create_client(route, timeout=timeout, headers=_HEADERS) as client:
                response = client.get(endpoint)
                _raise_for_rate_limit(response)
                response.raise_for_status()
                payload = response.json()
        except Exception as error:
            if not is_connection_failure(error):
                raise
            attempts.append((route.label, error))
            continue
        if not isinstance(payload, dict):
            raise AssetResolutionError(f"{endpoint} 返回了意外的内容")
        return payload
    raise connection_error(attempts)


def _raise_for_rate_limit(response: httpx.Response) -> None:
    # Unauthenticated API calls are 60/hour per IP. One install spends one, so
    # this is only reachable behind shared egress -- but there it is a plain
    # "come back later", and it must not read as a broken download.
    if response.status_code not in {403, 429}:
        return
    if response.headers.get("x-ratelimit-remaining") != "0":
        return
    raise AssetResolutionError(
        "GitHub API 速率限制已用尽（未认证为每小时 60 次，按 IP 计），"
        "请稍后重试；此调用只用于获取待下载文件的 sha256 摘要"
    )
