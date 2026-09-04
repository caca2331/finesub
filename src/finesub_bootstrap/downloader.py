from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
from pathlib import Path
import time

import httpx

from finesub_bootstrap.fsops import replace_path

from finesub_bootstrap.locks import holding_lock
from finesub_bootstrap.models import DownloadAsset, DownloadProgress
from finesub_bootstrap.http_client import (
    connection_error,
    create_client,
    is_connection_failure,
    network_routes,
)


class DownloadError(RuntimeError):
    """Base class for verified download failures."""


class SizeMismatch(DownloadError):
    pass


class DigestMismatch(DownloadError):
    pass


class DownloadPaused(DownloadError):
    pass


ProgressCallback = Callable[[DownloadProgress], None]
PauseCheck = Callable[[], bool]


def _part_path(destination: Path) -> Path:
    return destination.with_suffix(f"{destination.suffix}.part")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expectation_path(part_path: Path) -> Path:
    return Path(f"{part_path}.expect")


def _discard_mismatched_part(part_path: Path, asset: DownloadAsset) -> None:
    """Drop a partial that was heading for different bytes than we want now.

    Resume is a supported feature, so a `.part` can outlive the session that
    began it -- and by then the target can have moved, either because a manifest
    pin was bumped or because the asset's digest is resolved afresh on every
    install (`asset_resolve`). Appending the tail of one build onto the head of
    another yields a file that can only fail the digest check after the whole
    download, so a partial is worth exactly nothing once the target changes.
    Recording what each partial is aimed at is what lets us tell.
    """

    expectation = _expectation_path(part_path)
    if part_path.is_file():
        recorded = (
            expectation.read_text(encoding="utf-8").strip()
            if expectation.is_file()
            else ""
        )
        if recorded != asset.sha256:
            part_path.unlink()
    else:
        expectation.unlink(missing_ok=True)
    expectation.write_text(asset.sha256, encoding="utf-8")


def download_asset(
    asset: DownloadAsset,
    destination: Path,
    progress: ProgressCallback,
    should_pause: PauseCheck | None = None,
) -> Path:
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The download cache is shared between front ends, so two of them can want
    # the same archive at once. Serializing them costs nothing (the loser finds
    # the finished file and returns) and is the only way to keep the resumable
    # `.part` file -- a per-process name would make pause/resume impossible.
    with holding_lock(
        destination.with_suffix(f"{destination.suffix}.lock"),
        waiting_message=(
            "Another FineSub process is downloading the same file; "
            "waiting for it to finish"
        ),
        should_pause=should_pause,
        on_pause=lambda: DownloadPaused("Download paused"),
    ):
        return _download_locked(asset, destination, progress, should_pause)


def _download_locked(
    asset: DownloadAsset,
    destination: Path,
    progress: ProgressCallback,
    should_pause: PauseCheck | None,
) -> Path:
    if destination.is_file() and _sha256(destination) == asset.sha256:
        # Already here: either a previous run got it, or the process we just
        # waited for did.
        progress(
            DownloadProgress(
                downloaded=asset.size,
                total=asset.size,
                bytes_per_second=0.0,
            )
        )
        return destination
    part_path = _part_path(destination)
    _discard_mismatched_part(part_path, asset)
    existing = part_path.stat().st_size if part_path.is_file() else 0
    if existing > asset.size:
        part_path.write_bytes(b"")
        existing = 0

    started = time.perf_counter()
    transferred = 0
    if should_pause is not None and should_pause():
        raise DownloadPaused("Download paused")
    if existing < asset.size:
        timeout = httpx.Timeout(connect=20.0, read=120.0, write=30.0, pool=20.0)
        attempts: list[tuple[str, BaseException]] = []
        for route in network_routes():
            resume_at = part_path.stat().st_size if part_path.is_file() else 0
            if resume_at > asset.size:
                part_path.write_bytes(b"")
                resume_at = 0
            headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
            try:
                with create_client(route, timeout=timeout) as client:
                    with client.stream("GET", asset.url, headers=headers) as response:
                        response.raise_for_status()
                        append = bool(resume_at and response.status_code == 206)
                        downloaded = resume_at if append else 0
                        mode = "ab" if append else "wb"
                        with part_path.open(mode) as target:
                            for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                                if should_pause is not None and should_pause():
                                    raise DownloadPaused("Download paused")
                                if not chunk:
                                    continue
                                target.write(chunk)
                                downloaded += len(chunk)
                                transferred += len(chunk)
                                elapsed = max(time.perf_counter() - started, 1e-6)
                                progress(
                                    DownloadProgress(
                                        downloaded=downloaded,
                                        total=asset.size,
                                        bytes_per_second=transferred / elapsed,
                                    )
                                )
                                if should_pause is not None and should_pause():
                                    raise DownloadPaused("Download paused")
                break
            except Exception as error:
                if not is_connection_failure(error):
                    raise
                attempts.append((route.label, error))
        else:
            raise connection_error(attempts)

    if should_pause is not None and should_pause():
        raise DownloadPaused("Download paused")
    actual_size = part_path.stat().st_size if part_path.is_file() else 0
    if actual_size != asset.size:
        raise SizeMismatch(
            f"Expected {asset.size} bytes for {asset.url}, received {actual_size}"
        )
    actual_digest = _sha256(part_path)
    if actual_digest != asset.sha256:
        quarantine = part_path.with_suffix(f"{part_path.suffix}.bad")
        # Bare on purpose: a digest mismatch is already the failure being
        # reported, and waiting on the rename would only delay saying so.
        os.replace(part_path, quarantine)
        _expectation_path(part_path).unlink(missing_ok=True)
        raise DigestMismatch(
            f"Expected SHA-256 {asset.sha256}, received {actual_digest}"
        )

    replace_path(part_path, destination)
    _expectation_path(part_path).unlink(missing_ok=True)
    elapsed = max(time.perf_counter() - started, 1e-6)
    progress(
        DownloadProgress(
            downloaded=asset.size,
            total=asset.size,
            bytes_per_second=transferred / elapsed,
        )
    )
    return destination
