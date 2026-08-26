"""The CLI's half of the model routing: fetch at the stage, mirror first."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from finesub_bootstrap import hf_verify, model_ensure, model_fetch
from finesub_bootstrap.model_manifest import ManifestFile, ModelEntry


CACHE_DIR = "models--acme--weights"
REVISION = "0123456789abcdef"
BODY = b"weights"


def _entry() -> ModelEntry:
    return ModelEntry(
        model_id="whisper",
        repo="acme/weights",
        revision=REVISION,
        files=(
            ManifestFile(
                name="model.bin",
                url="https://example.invalid/model.bin",
                size=len(BODY),
                sha256=hashlib.sha256(BODY).hexdigest(),
            ),
        ),
    )


@pytest.fixture
def cache(tmp_path: Path, monkeypatch) -> Path:
    """A hub directory the code under test will look at, and nothing else."""

    hub = tmp_path / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setattr(model_ensure, "entry_for", lambda _id: _entry())
    monkeypatch.setattr(
        model_ensure, "_PIPELINE_HF_CACHE_DIRS", {"whisper": CACHE_DIR}
    )
    return hub


def _land_the_weights(hub: Path) -> None:
    snapshot = hub / CACHE_DIR / "snapshots" / REVISION
    snapshot.mkdir(parents=True, exist_ok=True)
    (snapshot / "model.bin").write_bytes(BODY)


def test_weights_already_there_cost_a_directory_check(cache: Path, tmp_path) -> None:
    """The reason this can sit at the top of a stage."""

    _land_the_weights(cache)
    calls: list[str] = []

    model_ensure.ensure_hf_model(
        "whisper",
        data_root=tmp_path / "data",
        download=lambda model_id, environment: calls.append(model_id),
    )

    assert calls == []


def test_a_missing_model_is_fetched_and_then_marked(cache: Path, tmp_path) -> None:
    def fake_download(model_id: str, environment) -> None:
        _land_the_weights(cache)

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=fake_download
    )

    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "current"


def test_a_mirror_failure_earns_one_attempt_at_the_official_source(
    cache: Path, tmp_path, monkeypatch
) -> None:
    """And the second attempt must not carry the endpoint that just failed.

    huggingface_hub reads it at import, which is why each attempt is its own
    process; a fallback that kept the variable would retry the same host.
    """

    monkeypatch.setattr(
        model_fetch, "hf_endpoint_for", lambda *_args, **_kwargs: "https://mirror.invalid"
    )
    monkeypatch.delenv(model_fetch.HF_ENDPOINT, raising=False)
    seen: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        seen.append(environment.get(model_fetch.HF_ENDPOINT, ""))
        if len(seen) == 1:
            raise RuntimeError("connection reset by peer")
        _land_the_weights(cache)

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=fake_download
    )

    assert seen == ["https://mirror.invalid", ""]


def test_a_mirror_that_serves_wrong_bytes_earns_the_official_source(
    cache: Path, tmp_path, monkeypatch
) -> None:
    """Length right, content wrong: HTTP reports success, only the manifest
    knows. Verification therefore runs inside the fallback attempt, so a
    mismatch counts as the mirror's failure -- clean up, try the official
    source, and verify that result too."""

    monkeypatch.setattr(
        model_fetch, "hf_endpoint_for", lambda *_args, **_kwargs: "https://mirror.invalid"
    )
    monkeypatch.delenv(model_fetch.HF_ENDPOINT, raising=False)
    seen: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        seen.append(environment.get(model_fetch.HF_ENDPOINT, ""))
        snapshot = cache / CACHE_DIR / "snapshots" / REVISION
        snapshot.mkdir(parents=True, exist_ok=True)
        body = b"wrongbs" if len(seen) == 1 else BODY  # same length as BODY
        (snapshot / "model.bin").write_bytes(body)

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=fake_download
    )

    assert seen == ["https://mirror.invalid", ""]
    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "current"


def test_wrong_bytes_from_both_sources_still_fail_loudly(
    cache: Path, tmp_path, monkeypatch
) -> None:
    """The official source gets no more trust than the mirror did."""

    monkeypatch.setattr(
        model_fetch, "hf_endpoint_for", lambda *_args, **_kwargs: "https://mirror.invalid"
    )
    monkeypatch.delenv(model_fetch.HF_ENDPOINT, raising=False)
    attempts: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        attempts.append(environment.get(model_fetch.HF_ENDPOINT, ""))
        snapshot = cache / CACHE_DIR / "snapshots" / REVISION
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.bin").write_bytes(b"wrongbs")

    with pytest.raises(RuntimeError, match="校验失败"):
        model_ensure.ensure_hf_model(
            "whisper", data_root=tmp_path / "data", download=fake_download
        )

    assert attempts == ["https://mirror.invalid", ""]
    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "failed"


def test_a_local_failure_does_not_re_download_gigabytes_elsewhere(
    cache: Path, tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        model_fetch, "hf_endpoint_for", lambda *_args, **_kwargs: "https://mirror.invalid"
    )
    monkeypatch.delenv(model_fetch.HF_ENDPOINT, raising=False)
    attempts: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        attempts.append(model_id)
        raise RuntimeError("no space left on device")

    with pytest.raises(RuntimeError, match="no space left"):
        model_ensure.ensure_hf_model(
            "whisper", data_root=tmp_path / "data", download=fake_download
        )

    assert attempts == ["whisper"]


def test_a_download_that_does_not_verify_is_an_error_not_a_ready_model(
    cache: Path, tmp_path
) -> None:
    def fake_download(model_id: str, environment) -> None:
        snapshot = cache / CACHE_DIR / "snapshots" / REVISION
        snapshot.mkdir(parents=True, exist_ok=True)
        (snapshot / "model.bin").write_bytes(b"trunc")

    with pytest.raises(RuntimeError, match="校验失败"):
        model_ensure.ensure_hf_model(
            "whisper", data_root=tmp_path / "data", download=fake_download
        )

    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "failed"


def test_a_failed_verification_is_retried_not_reused(cache: Path, tmp_path) -> None:
    """The snapshot a failure leaves behind must not pass the fast path."""

    calls: list[str] = []

    def bad_then_good(model_id: str, environment) -> None:
        calls.append(model_id)
        snapshot = cache / CACHE_DIR / "snapshots" / REVISION
        snapshot.mkdir(parents=True, exist_ok=True)
        body = b"trunc" if len(calls) == 1 else BODY
        (snapshot / "model.bin").write_bytes(body)
        # A second manifest file that always lands whole: it keeps the snapshot
        # non-empty after the corrupt one is discarded, which is exactly the
        # state the fast path used to mistake for a ready model.
        (snapshot / "extra.txt").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="校验失败"):
        model_ensure.ensure_hf_model(
            "whisper", data_root=tmp_path / "data", download=bad_then_good
        )

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=bad_then_good
    )

    assert calls == ["whisper", "whisper"]
    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "current"


def test_a_stale_marker_forces_a_refetch(cache: Path, tmp_path) -> None:
    """A cache verified against a manifest that has since re-pinned is not ready."""

    _land_the_weights(cache)
    older = ModelEntry(
        model_id="whisper",
        repo="acme/weights",
        revision="fedcba9876543210",
        files=_entry().files,
    )
    hf_verify.write_marker(cache, CACHE_DIR, older)
    calls: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        calls.append(model_id)
        _land_the_weights(cache)

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=fake_download
    )

    assert calls == ["whisper"]
    assert hf_verify.marker_state(cache, CACHE_DIR, _entry()) == "current"


def test_a_snapshot_at_another_revision_is_not_the_pinned_model(
    cache: Path, tmp_path
) -> None:
    """The loaders are handed the pin; only that revision counts as present."""

    other = cache / CACHE_DIR / "snapshots" / "fedcba9876543210"
    other.mkdir(parents=True)
    (other / "model.bin").write_bytes(BODY)
    calls: list[str] = []

    def fake_download(model_id: str, environment) -> None:
        calls.append(model_id)
        _land_the_weights(cache)

    model_ensure.ensure_hf_model(
        "whisper", data_root=tmp_path / "data", download=fake_download
    )

    assert calls == ["whisper"]


def test_a_model_the_manifest_never_describes_keeps_its_old_behaviour(
    tmp_path, monkeypatch
) -> None:
    """Its own library fetches it, exactly as before -- no new failure mode."""

    monkeypatch.setattr(model_ensure, "entry_for", lambda _id: None)
    calls: list[str] = []

    model_ensure.ensure_hf_model(
        "whatever",
        data_root=tmp_path / "data",
        download=lambda model_id, environment: calls.append(model_id),
    )

    assert calls == []
