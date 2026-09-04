"""HTTP client half (urllib, stdlib-only) for the share protocol."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping


class RemoteError(RuntimeError):
    pass


#: A full snapshot of a large corpus is megabytes; 64 MB is generous headroom
#: and still a bound — the admission boundary cannot protect a client that a
#: hostile server OOMs before admission runs (round 14).
MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _request(
    url: str,
    *,
    method: str = "GET",
    body: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json; charset=utf-8")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise RemoteError(f"response from {url} exceeds {MAX_RESPONSE_BYTES} bytes")
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read(65536).decode("utf-8")).get("error", "")
        except Exception:
            detail = ""
        raise RemoteError(f"{exc.code} {url}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RemoteError(f"cannot reach {url}: {exc.reason}") from exc


def register(base_url: str) -> str:
    return str(_request(f"{base_url.rstrip('/')}/register", method="POST")["token"])


def fetch_snapshot(base_url: str) -> dict[str, Any]:
    return _request(f"{base_url.rstrip('/')}/snapshot")


def push_bundle(base_url: str, bundle: Mapping[str, Any], *, token: str) -> dict[str, Any]:
    return _request(
        f"{base_url.rstrip('/')}/push",
        method="POST",
        body=bundle,
        headers={"X-Share-Token": token},
    )


def push_status(base_url: str, queue_id: int, *, token: str) -> dict[str, Any]:
    return _request(
        f"{base_url.rstrip('/')}/push/{int(queue_id)}", headers={"X-Share-Token": token}
    )


# ---- maintainer endpoints (the review CLI's transport) --------------------


def lease_queue(
    base_url: str,
    *,
    maintainer_token: str,
    seconds: int = 900,
    limit: int = 10,
    queue_id: int | None = None,
) -> dict[str, Any]:
    query = f"lease_seconds={int(seconds)}&limit={int(limit)}"
    if queue_id is not None:
        query += f"&queue_id={int(queue_id)}"
    return _request(
        f"{base_url.rstrip('/')}/queue?{query}",
        headers={"X-Maintainer-Token": maintainer_token},
    )


def peek_queue(base_url: str, *, maintainer_token: str, queue_id: int | None = None) -> dict[str, Any]:
    query = f"?queue_id={int(queue_id)}" if queue_id is not None else ""
    return _request(
        f"{base_url.rstrip('/')}/queue/peek{query}",
        headers={"X-Maintainer-Token": maintainer_token},
    )


def release_item(base_url: str, *, maintainer_token: str, queue_id: int, lease_token: str) -> dict[str, Any]:
    return _request(
        f"{base_url.rstrip('/')}/release",
        method="POST",
        body={"queue_id": queue_id, "lease_token": lease_token},
        headers={"X-Maintainer-Token": maintainer_token},
    )


def post_verdict(
    base_url: str,
    *,
    maintainer_token: str,
    queue_id: int,
    lease_token: str,
    expected_version: int,
    verdict: str,
    reason: str = "",
    merge: Mapping[str, str] | None = None,
    evidence: list[Mapping[str, Any]] | None = None,
    override: str = "",
) -> dict[str, Any]:
    return _request(
        f"{base_url.rstrip('/')}/verdict",
        method="POST",
        body={
            "queue_id": queue_id,
            "lease_token": lease_token,
            "expected_version": expected_version,
            "verdict": verdict,
            "reason": reason,
            "merge": dict(merge or {}),
            "evidence": list(evidence or []),
            "override": override,
        },
        headers={"X-Maintainer-Token": maintainer_token},
    )
