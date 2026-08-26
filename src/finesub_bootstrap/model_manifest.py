"""What each production model is, precisely enough to verify a copy of it.

A public mirror is a faster way to fetch bytes we already know the shape of;
it is not a new root of trust. That only holds if the shape is written down
somewhere the mirror does not control -- here, shipped with the release.

The shipped table describes the three pipeline models, with sizes and digests
recorded from the official sources on 2026-08-10 and verified end to end in
the 2026-08-21 release drill. An **unlisted** model still behaves exactly as
it always did: the owning library downloads it and no extra verification is
claimed. Shipping invented hashes would be worse than shipping none, because
a verification that cannot pass gets switched off rather than fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

MANIFEST_NAME = "model-manifest.json"
MANIFEST_ENVIRONMENT = "FINESUB_MODEL_MANIFEST"


@dataclass(frozen=True, slots=True)
class ManifestFile:
    """One file of one model: what to fetch, and how to know it arrived.

    `sha256` may be empty, and that is a statement rather than an omission:
    the separator's `download_checks.json` is an upstream index on a moving
    branch, so pinning its digest would fail every machine on the day upstream
    edits it. Such a file is fetched once and not verified; everything with a
    digest is verified before it counts as present.
    """

    name: str
    url: str
    size: int
    sha256: str

    @property
    def is_verifiable(self) -> bool:
        return bool(self.sha256)


@dataclass(frozen=True, slots=True)
class ModelEntry:
    model_id: str
    #: Hugging Face repository and the revision pinned for it. A mirror
    #: resolving a moving `main` to different content is exactly what pinning
    #: prevents.
    repo: str = ""
    revision: str = ""
    files: tuple[ManifestFile, ...] = ()

    @property
    def is_verifiable(self) -> bool:
        return bool(self.files)


def manifest_path() -> Path:
    override = os.environ.get(MANIFEST_ENVIRONMENT, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent / MANIFEST_NAME


def load_manifest() -> dict[str, ModelEntry]:
    """Every listed model, or nothing -- never an exception."""

    try:
        body = json.loads(manifest_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(body, dict):
        return {}
    models = body.get("models")
    if not isinstance(models, dict):
        return {}
    return {
        model_id: _entry(model_id, spec)
        for model_id, spec in models.items()
        if isinstance(spec, dict)
    }


def _entry(model_id: str, spec: dict[str, Any]) -> ModelEntry:
    files = []
    for item in spec.get("files", []):
        if not isinstance(item, dict):
            continue
        name, url = item.get("name"), item.get("url")
        digest, size = item.get("sha256"), item.get("size")
        if not (name and url):
            continue
        files.append(
            ManifestFile(
                name=str(name),
                url=str(url),
                size=size if isinstance(size, int) else 0,
                sha256=str(digest or ""),
            )
        )
    return ModelEntry(
        model_id=model_id,
        repo=str(spec.get("repo") or ""),
        revision=str(spec.get("revision") or ""),
        files=tuple(files),
    )


def entry_for(model_id: str) -> ModelEntry | None:
    return load_manifest().get(model_id)


def file_matches(path: Path, expected: ManifestFile) -> bool:
    """Whether `path` is the file the manifest describes.

    A file with no pinned digest counts as present as soon as it exists --
    that is the whole contract for an upstream index on a moving branch.
    Checking a size or digest it never promised would re-download it forever.

    For the rest, size first: it rejects a truncated download for the price of
    a stat, and the digest of a several-hundred-MB checkpoint is not free.
    """

    if not expected.is_verifiable:
        return path.is_file()
    try:
        if path.stat().st_size != expected.size:
            return False
    except OSError:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected.sha256
