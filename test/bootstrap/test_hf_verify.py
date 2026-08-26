"""A Hugging Face snapshot is verified once and then answered from a marker."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from finesub_bootstrap import hf_verify, model_caches
from finesub_bootstrap.model_manifest import ManifestFile, ModelEntry


CACHE_DIR = "models--acme--weights"
REVISION = "0123456789abcdef"


def _entry(*files: ManifestFile) -> ModelEntry:
    return ModelEntry(
        model_id="whisper",
        repo="acme/weights",
        revision=REVISION,
        files=tuple(files),
    )


def _write(hub: Path, name: str, body: bytes) -> ManifestFile:
    snapshot = hub / CACHE_DIR / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / name).write_bytes(body)
    return ManifestFile(
        name=name,
        url="https://example.invalid/" + name,
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )


def test_a_verified_snapshot_is_marked_and_then_costs_nothing(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    entry = _entry(_write(hub, "model.bin", b"weights"))

    assert hf_verify.verify_and_mark(hub, CACHE_DIR, entry) == ()
    assert hf_verify.marker_state(hub, CACHE_DIR, entry) == "current"


def test_the_marker_sits_beside_the_snapshots_not_inside_one(tmp_path: Path) -> None:
    """Inside a revision it would make an interrupted download look finished.

    `model_caches._hf_repo_complete` reads "some revision directory is
    non-empty" as evidence that files landed. A marker written in there would
    supply that evidence by itself.
    """

    hub = tmp_path / "hub"
    entry = _entry(_write(hub, "model.bin", b"weights"))
    hf_verify.verify_and_mark(hub, CACHE_DIR, entry)

    marker = hf_verify.marker_path(hub, CACHE_DIR)
    assert marker.parent == hub / CACHE_DIR
    assert not list((hub / CACHE_DIR / "snapshots").rglob(hf_verify.MARKER_NAME))


def test_a_short_file_fails_verification_and_is_fenced_off(tmp_path: Path) -> None:
    """A failure leaves a mark and removes the corrupt file.

    Without the mark, the non-empty snapshot it leaves behind is
    indistinguishable from a healthy pre-existing cache; without the removal,
    the next download would relink the same bytes and fail the same way.
    """

    hub = tmp_path / "hub"
    good = _write(hub, "model.bin", b"weights")
    (hub / CACHE_DIR / "snapshots" / REVISION / "model.bin").write_bytes(b"weig")

    failed = hf_verify.verify_and_mark(hub, CACHE_DIR, _entry(good))

    assert failed == ("model.bin",)
    assert hf_verify.marker_state(hub, CACHE_DIR, _entry(good)) == "failed"
    assert not (hub / CACHE_DIR / "snapshots" / REVISION / "model.bin").exists()


def test_a_failed_symlinked_file_takes_its_blob_with_it(tmp_path: Path) -> None:
    """The other cache layout: removing the link alone would leave the bytes.

    `huggingface_hub` links a snapshot entry to `blobs/<digest>` wherever the
    platform lets it, and copies the file in only when it cannot (Windows
    without the symlink privilege -- it never hard-links). The copy layout is
    covered by the test above; this is the link one, where deleting just the
    snapshot entry would leave the corrupt blob for the next download to relink
    as-is, which is the failure `_discard_failed` exists to prevent.
    """

    hub = tmp_path / "hub"
    snapshot = hub / CACHE_DIR / "snapshots" / REVISION
    blobs = hub / CACHE_DIR / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)

    body = b"weights"
    digest = hashlib.sha256(body).hexdigest()
    blob = blobs / digest
    blob.write_bytes(body)
    link = snapshot / "model.bin"
    try:
        # Relative, the way huggingface_hub writes it -- an absolute link would
        # not exercise `_discard_failed`'s reliance on `Path.resolve()`
        # resolving against the link's own directory.
        link.symlink_to(os.path.relpath(blob, snapshot))
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"this platform cannot create symlinks: {error}")
    entry = _entry(
        ManifestFile(
            name="model.bin",
            url="https://example.invalid/model.bin",
            size=len(body),
            sha256=digest,
        )
    )
    assert hf_verify.verify_and_mark(hub, CACHE_DIR, entry) == ()

    blob.write_bytes(b"corrupted")

    assert hf_verify.verify_and_mark(hub, CACHE_DIR, entry) == ("model.bin",)
    assert hf_verify.marker_state(hub, CACHE_DIR, entry) == "failed"
    assert not link.exists() and not link.is_symlink()
    assert not blob.exists(), "the blob outlived the link it was reached through"


def test_a_failure_marker_clears_once_verification_passes(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    good = _write(hub, "model.bin", b"weights")
    (hub / CACHE_DIR / "snapshots" / REVISION / "model.bin").write_bytes(b"weig")
    hf_verify.verify_and_mark(hub, CACHE_DIR, _entry(good))

    _write(hub, "model.bin", b"weights")

    assert hf_verify.verify_and_mark(hub, CACHE_DIR, _entry(good)) == ()
    assert hf_verify.marker_state(hub, CACHE_DIR, _entry(good)) == "current"


def test_a_manifest_that_moved_on_makes_the_marker_stale(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    entry = _entry(_write(hub, "model.bin", b"weights"))
    hf_verify.verify_and_mark(hub, CACHE_DIR, entry)

    repinned = ModelEntry(
        model_id="whisper",
        repo=entry.repo,
        revision="fedcba9876543210",
        files=entry.files,
    )

    assert hf_verify.marker_state(hub, CACHE_DIR, repinned) == "stale"


def test_a_cache_nobody_verified_is_absent_not_broken(tmp_path: Path) -> None:
    """Weights that predate the mechanism keep working, unhashed.

    Treating "no marker" as damage would re-download gigabytes to learn
    nothing -- and every machine that had a cache before this existed would pay
    it once.
    """

    hub = tmp_path / "hub"
    entry = _entry(_write(hub, "model.bin", b"weights"))

    assert hf_verify.marker_state(hub, CACHE_DIR, entry) == "absent"


def test_an_unwritable_cache_costs_a_re_verification_not_a_failure(
    tmp_path: Path, monkeypatch
) -> None:
    hub = tmp_path / "hub"
    entry = _entry(_write(hub, "model.bin", b"weights"))

    def _refuse(*_args, **_kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_text", _refuse)

    assert hf_verify.verify_and_mark(hub, CACHE_DIR, entry) == ()


def test_readiness_ignores_a_missing_marker_but_not_a_stale_one(
    tmp_path: Path, monkeypatch
) -> None:
    """The readiness poll is where the three states turn into one decision."""

    models_root = tmp_path / "models"
    hub = models_root / "huggingface" / "hub"
    cache_dir = model_caches.WHISPER_CACHE_DIR
    snapshot = hub / cache_dir / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"weights")
    for other in (model_caches.QWEN_REFEREE_CACHE_DIR,):
        (hub / other / "snapshots" / REVISION).mkdir(parents=True)
        (hub / other / "snapshots" / REVISION / "x").write_bytes(b"x")
    separator_dir = models_root / "audio-separator"
    separator_dir.mkdir(parents=True)
    (separator_dir / model_caches.SEPARATOR_CHECKPOINT).write_bytes(b"ckpt")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    assert model_caches.missing_pipeline_models(models_root) == ()

    hf_verify.marker_path(hub, cache_dir).write_text(
        json.dumps({"manifest": "something-else"}), encoding="utf-8"
    )

    assert model_caches.missing_pipeline_models(models_root) == ("whisper",)


def test_readiness_rejects_a_cache_whose_verification_failed(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed verification must not be reported ready by the next poll."""

    models_root = tmp_path / "models"
    hub = models_root / "huggingface" / "hub"
    cache_dir = model_caches.WHISPER_CACHE_DIR
    snapshot = hub / cache_dir / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"weights")
    for other in (model_caches.QWEN_REFEREE_CACHE_DIR,):
        (hub / other / "snapshots" / REVISION).mkdir(parents=True)
        (hub / other / "snapshots" / REVISION / "x").write_bytes(b"x")
    separator_dir = models_root / "audio-separator"
    separator_dir.mkdir(parents=True)
    (separator_dir / model_caches.SEPARATOR_CHECKPOINT).write_bytes(b"ckpt")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    from finesub_bootstrap.model_manifest import entry_for

    real_entry = entry_for("whisper")
    assert real_entry is not None
    hf_verify.marker_path(hub, cache_dir).write_text(
        json.dumps(
            {"manifest": hf_verify.entry_digest(real_entry), "failed": ["model.bin"]}
        ),
        encoding="utf-8",
    )

    assert model_caches.missing_pipeline_models(models_root) == ("whisper",)
