"""The CLI's half of the model routing: fetch at the stage, mirror first."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
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
    """A hub directory the code under test will look at, and nothing else.

    Both roots point at it: the variable a download subprocess would read, and
    the constant `huggingface_hub` froze at import, which is what a loader in
    this process reads. Setting only the first leaves `pinned_snapshot_loadable`
    looking at the machine's real cache -- green here, and about nothing.
    """

    hub = tmp_path / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    try:
        from huggingface_hub import constants

        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    except ImportError:  # [asr] not installed; only the write path is tested
        pass
    monkeypatch.setattr(model_ensure, "entry_for", lambda _id: _entry())
    monkeypatch.setattr(
        model_ensure, "_ENSURABLE_HF_CACHE_DIRS", {"whisper": CACHE_DIR}
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


def test_a_landed_snapshot_is_loadable_without_the_network(cache: Path) -> None:
    """What lets a loader pass `local_files_only=True` and skip the hub."""

    assert not model_ensure.pinned_snapshot_loadable("whisper")
    _land_the_weights(cache)
    assert model_ensure.pinned_snapshot_loadable("whisper")


def test_another_revision_is_not_the_pinned_one(cache: Path) -> None:
    """Loading offline at the pinned revision needs *that* snapshot present."""

    other = cache / CACHE_DIR / "snapshots" / "fedcba9876543210"
    other.mkdir(parents=True)
    (other / "model.bin").write_bytes(BODY)

    assert not model_ensure.pinned_snapshot_loadable("whisper")


def test_a_snapshot_missing_its_weights_is_not_loadable(cache: Path) -> None:
    """The state `_hf_repo_complete` accepts and a loader cannot use.

    A snapshot keeps its small files while the big one is deleted to reclaim
    the space (or was never linked). CTranslate2 then raises a bare
    `RuntimeError` -- indistinguishable from the CUDA failures that must never
    be retried -- so this has to be caught here, before `local_files_only` is
    ever passed, and not by an exception handler afterwards.
    """

    snapshot = cache / CACHE_DIR / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    assert not model_ensure.pinned_snapshot_loadable("whisper")

    (snapshot / "model.bin").write_bytes(BODY)
    assert model_ensure.pinned_snapshot_loadable("whisper")


def test_a_truncated_weight_file_is_not_loadable(cache: Path) -> None:
    """Present is not enough: an interrupted copy is the same failure."""

    _land_the_weights(cache)
    (cache / CACHE_DIR / "snapshots" / REVISION / "model.bin").write_bytes(
        BODY[:-1]
    )

    assert not model_ensure.pinned_snapshot_loadable("whisper")


def test_the_root_asked_about_is_the_loader_s_own(tmp_path, monkeypatch) -> None:
    """`ensure_hf_model` having returned is not the same answer.

    With nothing in the environment it writes into this install's managed
    cache, while the loader reads the conventional one -- and a
    `local_files_only` derived from the fetch instead of from the loader's root
    would fail a run that would otherwise have downloaded.
    """

    constants = pytest.importorskip(
        "huggingface_hub.constants", reason="[asr] extra not installed"
    )
    monkeypatch.setattr(model_ensure, "entry_for", lambda _id: _entry())
    monkeypatch.setattr(
        model_ensure, "_ENSURABLE_HF_CACHE_DIRS", {"whisper": CACHE_DIR}
    )
    conventional = tmp_path / "conventional" / "hub"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(conventional))
    _land_the_weights(tmp_path / "managed" / "huggingface" / "hub")

    assert not model_ensure.pinned_snapshot_loadable("whisper")

    _land_the_weights(conventional)
    assert model_ensure.pinned_snapshot_loadable("whisper")


_CACHE_ROOT_PROBE = """
import json, sys
from pathlib import Path
from huggingface_hub import constants
from finesub_bootstrap import model_ensure
# What a download will use: the constant put through the same normalisation
# every hub entry point applies to it before touching the disk.
hub = Path(constants.HF_HUB_CACHE).expanduser().resolve()
print(json.dumps([
    str(hub),
    str(model_ensure._loader_hub_dir()),
    str(model_ensure._hub_dir(None)),
]))
"""


@pytest.mark.parametrize(
    "variables",
    [
        pytest.param({}, id="nothing-set"),
        pytest.param({"HF_HUB_CACHE": r"C:\probe-cache"}, id="hf-hub-cache"),
        # The legacy spelling the library still honours, and `~` in either of
        # them: two rules a hand-written copy of the resolution did not have,
        # and each one silently sends the gate to a directory nobody reads.
        pytest.param(
            {"HUGGINGFACE_HUB_CACHE": r"C:\probe-legacy"}, id="legacy-variable"
        ),
        pytest.param({"HF_HUB_CACHE": "~/probe-cache"}, id="tilde-in-cache"),
        pytest.param({"HF_HOME": "~/probe-home"}, id="tilde-in-home"),
        # `constants.py` expands `~` before `$VAR`, so a variable that resolves
        # to a `~` path leaves a literal tilde in the constant for a download to
        # expand later. Taking the constant at face value pointed the gate at a
        # directory named `~` under the working directory.
        pytest.param(
            {
                "HF_HUB_CACHE": "%FINESUB_PROBE_ROOT%",
                "FINESUB_PROBE_ROOT": "~/probe-nested",
            },
            id="tilde-after-expansion",
        ),
    ],
)
def test_the_cache_root_is_the_one_the_hub_itself_resolved(variables) -> None:
    """`local_files_only` is only safe if this names the loader's own root.

    A subprocess because the library freezes its answer at import: the variables
    have to be set before it is loaded, which is also how a real run meets them.
    """

    pytest.importorskip("huggingface_hub", reason="[asr] extra not installed")

    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_HOME",
            "FINESUB_PROBE_ROOT",
        )
    }
    environment["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", _CACHE_ROOT_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        env={**environment, **variables},
    )

    assert result.returncode == 0, result.stderr[-2000:]
    hub_says, loader_says, fallback_says = json.loads(result.stdout)
    # Compared as paths: expanding `~` leaves the hub with a mixed separator
    # (``C:\Users\Carl/probe-cache``), which names the same directory.
    assert Path(loader_says) == Path(hub_says)
    # The hand-written ladder is what an install without `huggingface_hub`
    # falls back to, so it has to reach the same place.
    assert Path(fallback_says) == Path(hub_says)
