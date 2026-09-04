"""Verify a Hugging Face snapshot once, then answer from a marker.

The fixed-file half of the manifest has been verified since it landed
(`model_fetch.fetch_fixed_files` hashes each file and leaves a `.verified`
stamp beside it). The Hugging Face half never was: `snapshot_download` is the
library's business, and readiness was decided by three cheap facts about the
directory -- it exists, no blob is `.incomplete`, some revision is non-empty.
That accepts an interruption *between* two files, which is how a run could
report weights ready and then spend minutes fetching the rest.

Two constraints shape what follows.

**The question is asked often.** Every stage that needs a model asks "is it
there?" as it starts (the desktop used to ask on a timer). Hashing three
gigabytes to answer is not an option, so a full verification runs
once -- after a download this process performed -- and writes a marker; every
later answer compares the marker.

**The marker must not sit inside `snapshots/<revision>/`.** That directory
being non-empty is exactly what `model_caches._hf_repo_complete` reads as "a
revision landed"; a marker written there would make an interrupted, otherwise
empty snapshot look finished -- defeating the check this one exists to
strengthen. It goes at the repository level instead, where it also follows the
weights: move or delete the model and the marker goes with it.

A cache the machine already had stays readable without a marker. Absence means
"nobody verified this", not "this is broken" -- forcing a re-download of
weights that predate the mechanism would cost gigabytes to learn nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .model_manifest import ManifestFile, ModelEntry, file_matches


MARKER_NAME = ".finesub-verified.json"

#: The words a verification mismatch always carries in its message. A
#: verification that ran in a subprocess comes back as text, so classification
#: sometimes has only the message to go on -- `model_fetch.is_mirror_failure`
#: matches this exact phrase across that boundary.
MISMATCH_MARKER = "清单摘要对不上"


class VerificationMismatch(RuntimeError):
    """A downloaded snapshot did not match the manifest that pinned it.

    Raised by whoever verified the download. Distinct from other runtime
    errors because the fallback treats it as the *mirror's* failure: the HTTP
    layer saw a success, and only the manifest knows the bytes were wrong.
    """


def marker_path(hub: Path, cache_dir: str) -> Path:
    """Where the marker for one repository lives: beside its snapshots."""

    return hub / cache_dir / MARKER_NAME


def entry_digest(entry: ModelEntry) -> str:
    """What the marker promises: this repo, this revision, these files.

    Digesting the manifest entry rather than storing it means a manifest that
    moves -- a new revision, a re-pinned file -- invalidates the marker without
    anyone having to remember to bump a version.
    """

    payload = json.dumps(
        {
            "repo": entry.repo,
            "revision": entry.revision,
            "files": [
                {"name": item.name, "size": item.size, "sha256": item.sha256}
                for item in entry.files
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def marker_state(hub: Path, cache_dir: str, entry: ModelEntry) -> str:
    """One of ``"absent"``, ``"stale"``, ``"failed"``, ``"current"``.

    The caller decides what each means; they are not the same thing. Absent is
    a cache nobody verified, which is fine. Stale is a cache verified against a
    manifest that has since moved on, and failed is one whose last verification
    did not pass -- both mean "fetch it again". Stale wins over failed: a
    re-pinned manifest invalidates an old failure along with everything else.
    """

    try:
        record = json.loads(
            marker_path(hub, cache_dir).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return "absent"
    if not isinstance(record, dict):
        return "absent"
    if record.get("manifest") != entry_digest(entry):
        return "stale"
    return "failed" if record.get("failed") else "current"


def _snapshot_dir(hub: Path, cache_dir: str, entry: ModelEntry) -> Path | None:
    revisions = hub / cache_dir / "snapshots"
    if entry.revision:
        pinned = revisions / entry.revision
        return pinned if pinned.is_dir() else None
    existing = [path for path in revisions.glob("*") if path.is_dir()]
    return existing[0] if len(existing) == 1 else None


def pinned_snapshot_present(hub: Path, cache_dir: str, entry: ModelEntry) -> bool:
    """Whether the revision the manifest pins has actually landed.

    Non-empty, not merely present: the directory is created before files are
    linked into it. An entry that pins nothing is present by definition --
    there is no specific revision to demand.
    """

    if not entry.revision:
        return True
    pinned = hub / cache_dir / "snapshots" / entry.revision
    return pinned.is_dir() and any(pinned.iterdir())


def unverified_files(hub: Path, cache_dir: str, entry: ModelEntry) -> tuple[str, ...]:
    """Manifest files that are missing or do not match, hashing each one."""

    snapshot = _snapshot_dir(hub, cache_dir, entry)
    if snapshot is None:
        return tuple(item.name for item in entry.files) or ("<snapshot>",)
    failed: list[str] = []
    for item in entry.files:
        if not file_matches(snapshot / item.name, item):
            failed.append(item.name)
    return tuple(failed)


def write_marker(
    hub: Path, cache_dir: str, entry: ModelEntry, *, failed: tuple[str, ...] = ()
) -> None:
    """Record how this repository verified against the current manifest.

    Best effort, like the fixed-file stamp beside it: a cache root nobody can
    write to costs a verification next time, which is the behaviour that
    existed before this file.
    """

    record: dict[str, object] = {
        "manifest": entry_digest(entry),
        "repo": entry.repo,
        "revision": entry.revision,
    }
    if failed:
        record["failed"] = list(failed)
    try:
        marker_path(hub, cache_dir).write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass


def _discard_failed(snapshot: Path, names: tuple[str, ...]) -> None:
    """Remove files that failed verification, so a retry actually re-fetches.

    A snapshot entry may be a link into ``blobs/``; the corrupt bytes are the
    target's, so both go -- a blob left behind would be relinked as-is by the
    next download. Best effort: whatever cannot be removed is still fenced off
    by the failure marker.
    """

    for name in names:
        path = snapshot / name
        try:
            target = path.resolve() if path.is_symlink() else None
            path.unlink(missing_ok=True)
            if target is not None:
                target.unlink(missing_ok=True)
        except OSError:
            continue


def verify_and_mark(hub: Path, cache_dir: str, entry: ModelEntry) -> tuple[str, ...]:
    """Verify a freshly downloaded repository and mark it. Returns failures.

    Called by whoever performed the download -- mirror or official source, the
    two are the same risk once the bytes are on disk. An entry the manifest
    cannot verify (no pinned files) is marked without hashing: there is nothing
    to compare, and the marker still records which revision this is.

    A failure also leaves a mark -- otherwise the snapshot it leaves behind is
    non-empty, marker-less, and indistinguishable from a healthy pre-existing
    cache, which is exactly the disguise a corrupt model must not have. The
    failing files themselves are removed so the next attempt re-fetches them.
    """

    if not entry.is_verifiable:
        write_marker(hub, cache_dir, entry)
        return ()
    failed = unverified_files(hub, cache_dir, entry)
    if failed:
        snapshot = _snapshot_dir(hub, cache_dir, entry)
        if snapshot is not None:
            _discard_failed(snapshot, failed)
        write_marker(hub, cache_dir, entry, failed=failed)
        return failed
    write_marker(hub, cache_dir, entry)
    return ()


__all__ = [
    "MARKER_NAME",
    "MISMATCH_MARKER",
    "ManifestFile",
    "VerificationMismatch",
    "entry_digest",
    "marker_path",
    "marker_state",
    "pinned_snapshot_present",
    "unverified_files",
    "verify_and_mark",
    "write_marker",
]
