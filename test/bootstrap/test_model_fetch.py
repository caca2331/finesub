"""Choosing and falling back between model download endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from finesub_bootstrap import download_routes, model_fetch
from finesub_bootstrap.http_client import NetworkConnectionError
from finesub_bootstrap.model_manifest import ManifestFile, ModelEntry, file_matches


def _table(tmp_path: Path, monkeypatch, **values) -> None:
    table = tmp_path / "sources.json"
    table.write_text(json.dumps(values), encoding="utf-8")
    monkeypatch.setenv("FINESUB_DOWNLOAD_SOURCES", str(table))


def test_no_configured_mirror_leaves_the_environment_alone(
    tmp_path: Path, monkeypatch
) -> None:
    """The shipped table names one, so an empty table is how this is reached --
    the state a fork with its own sources, or a disabled class, is in."""

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch)
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert environment == {}


def test_the_shipped_endpoint_is_applied_for_the_cn_region(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("FINESUB_DOWNLOAD_SOURCES", raising=False)
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert environment["HF_ENDPOINT"].startswith("https://")


def test_a_configured_mirror_is_applied_for_the_cn_region(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert environment == {
        "HF_ENDPOINT": "https://mirror.example",
        # A mirror does not proxy Xet, and the failure is a 401 on the first
        # reconstruction request rather than something slower -- so the two
        # travel together or the route cannot install a model at all.
        "HF_HUB_DISABLE_XET": "1",
    }


def test_the_global_region_uses_the_official_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="global")

    assert environment == {}


def test_a_user_who_set_the_endpoint_is_not_overruled(
    tmp_path: Path, monkeypatch
) -> None:
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    monkeypatch.setenv("HF_ENDPOINT", "https://chosen.example")
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert "HF_ENDPOINT" not in environment, (
        "an explicit choice is not a region guess to override"
    )
    assert environment == {"HF_HUB_DISABLE_XET": "1"}, (
        "their endpoint hits the same Xet wall ours does -- that is a fact "
        "about the protocol, not a guess about where they should download from"
    )


def test_a_user_who_declared_their_own_xet_choice_keeps_it(
    tmp_path: Path, monkeypatch
) -> None:
    """A gateway that really does speak Xet is theirs to declare."""

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert environment == {"HF_ENDPOINT": "https://mirror.example"}


def test_the_official_endpoint_keeps_xet(tmp_path: Path, monkeypatch) -> None:
    """Xet is only broken *through a mirror*; the official host serves it."""

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="global")

    assert environment == {}


def test_a_degraded_class_falls_back_to_the_official_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    for _ in range(download_routes.FAILURE_LIMIT):
        download_routes.record_failure(tmp_path, "huggingface")
    environment: dict[str, str] = {}

    model_fetch.apply_hf_endpoint(environment, data_root=tmp_path, region="cn")

    assert environment == {}


def test_a_failed_mirror_is_retried_once_against_the_official_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    seen: list[str] = []

    def fetch(environment) -> None:
        endpoint = environment.get("HF_ENDPOINT", "")
        seen.append(endpoint)
        if endpoint:
            raise ConnectionError("mirror is down")

    model_fetch.fetch_with_fallback(
        fetch,
        base_environment={},
        data_root=tmp_path,
        region="cn",
        is_retryable=lambda error: True,
    )

    assert seen == ["https://mirror.example", ""]
    assert download_routes.failures(tmp_path, "huggingface") == 1


def test_the_official_attempt_gets_xet_back(tmp_path: Path, monkeypatch) -> None:
    """The ban belongs to the mirror, so it does not follow the fallback home."""

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    seen: list[tuple[str, str]] = []

    def fetch(environment) -> None:
        endpoint = environment.get("HF_ENDPOINT", "")
        seen.append((endpoint, environment.get("HF_HUB_DISABLE_XET", "")))
        if endpoint:
            raise ConnectionError("mirror is down")

    model_fetch.fetch_with_fallback(
        fetch,
        base_environment={"HF_HUB_DISABLE_XET": "1"},
        data_root=tmp_path,
        region="cn",
        is_retryable=lambda error: True,
    )

    assert seen == [("https://mirror.example", "1"), ("", "")]


def test_401_from_a_mirror_earns_the_official_retry(
    tmp_path: Path, monkeypatch
) -> None:
    """The regression for the Xet failure: an auth status must reach the fallback.

    It arrives as text rather than as an exception because every download runs
    in its own interpreter (`model_ensure._download`), so the `httpx.HTTPError`
    branch never sees it -- only the marker table does.
    """

    assert model_fetch.is_mirror_failure(RuntimeError("401 Unauthorized"))
    assert model_fetch.is_mirror_failure(RuntimeError("HTTP 403 Forbidden"))

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    seen: list[str] = []

    def fetch(environment) -> None:
        endpoint = environment.get("HF_ENDPOINT", "")
        seen.append(endpoint)
        if endpoint:
            raise RuntimeError(
                "401 Unauthorized: cas-server.xethub.hf.co refused the token"
            )

    model_fetch.fetch_with_fallback(
        fetch,
        base_environment={},
        data_root=tmp_path,
        region="cn",
        is_retryable=model_fetch.is_mirror_failure,
    )

    assert seen == ["https://mirror.example", ""]


def test_a_failure_that_is_not_the_mirror_is_not_retried_elsewhere(
    tmp_path: Path, monkeypatch
) -> None:
    """A full disk is not a reason to re-download several GB from a second host."""

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    attempts: list[str] = []

    def fetch(environment) -> None:
        attempts.append(environment.get("HF_ENDPOINT", ""))
        raise OSError("no space left on device")

    with pytest.raises(OSError):
        model_fetch.fetch_with_fallback(
            fetch,
            base_environment={},
            data_root=tmp_path,
            region="cn",
            is_retryable=lambda error: not isinstance(error, OSError),
        )

    assert attempts == ["https://mirror.example"]


def test_a_success_clears_the_failure_counter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    _table(tmp_path, monkeypatch, hfEndpoint="https://mirror.example")
    download_routes.record_failure(tmp_path, "huggingface")

    model_fetch.fetch_with_fallback(
        lambda environment: None,
        base_environment={},
        data_root=tmp_path,
        region="cn",
    )

    assert download_routes.failures(tmp_path, "huggingface") == 0


# -- fixed files ----------------------------------------------------------


def _entry(tmp_path: Path, body: bytes) -> ModelEntry:
    return ModelEntry(
        model_id="separator",
        files=(
            ManifestFile(
                name="model.ckpt",
                url="https://official.example/model.ckpt",
                size=len(body),
                sha256=hashlib.sha256(body).hexdigest(),
            ),
        ),
    )


def test_a_file_that_already_matches_is_not_downloaded_again(
    tmp_path: Path, monkeypatch
) -> None:
    body = b"weights"
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.ckpt").write_bytes(body)
    called: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: called.append(asset.url),
    )

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, body), directory, data_root=tmp_path, region="global"
    )

    assert called == []


def test_a_verified_file_is_not_hashed_again_on_the_next_run(
    tmp_path: Path, monkeypatch
) -> None:
    # The separator checkpoint is 639 MB and this runs before every separation.
    # The first full check leaves a stamp; the second must not read the file.
    body = b"weights"
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.ckpt").write_bytes(body)
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: None,
    )

    entry = _entry(tmp_path, body)
    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="global"
    )
    assert model_fetch.verified_stamp_path(directory / "model.ckpt").is_file()

    hashed: list[Path] = []
    monkeypatch.setattr(
        "finesub_bootstrap.model_manifest.file_matches",
        lambda path, expected: hashed.append(path) or True,
    )
    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="global"
    )

    assert hashed == []


def test_an_edited_file_falls_back_to_the_full_check(
    tmp_path: Path, monkeypatch
) -> None:
    # The stamp is an optimisation, never the verdict: a file that changed
    # under a valid stamp has to be caught and fetched again.
    body = b"weights"
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.ckpt").write_bytes(body)
    called: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: called.append(asset.url),
    )

    entry = _entry(tmp_path, body)
    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="global"
    )
    assert called == []

    (directory / "model.ckpt").write_bytes(b"something else entirely")
    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="global"
    )

    assert called == ["https://official.example/model.ckpt"]


def test_a_file_without_a_digest_is_never_stamped(
    tmp_path: Path, monkeypatch
) -> None:
    # An upstream index on a moving branch promises nothing to record, and a
    # stamp claiming otherwise would read as a verification that never happened.
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "index.json").write_text("{}", encoding="utf-8")
    entry = ModelEntry(
        model_id="separator",
        files=(
            ManifestFile(
                name="index.json",
                url="https://official.example/index.json",
                size=0,
                sha256="",
            ),
        ),
    )

    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="global"
    )

    assert not model_fetch.verified_stamp_path(directory / "index.json").exists()


def test_a_mismatched_file_is_fetched_again(tmp_path: Path, monkeypatch) -> None:
    body = b"weights"
    directory = tmp_path / "models"
    directory.mkdir()
    (directory / "model.ckpt").write_bytes(b"something else entirely")
    called: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: called.append(asset.url),
    )

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, body), directory, data_root=tmp_path, region="global"
    )

    assert called == ["https://official.example/model.ckpt"]


def test_the_cn_region_prefixes_the_configured_proxy(
    tmp_path: Path, monkeypatch
) -> None:
    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    called: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: called.append(asset.url),
    )

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, b"weights"),
        tmp_path / "models",
        data_root=tmp_path,
        region="cn",
    )

    assert called == ["https://proxy.example/https://official.example/model.ckpt"]


def test_a_proxy_failure_falls_back_to_the_official_url(
    tmp_path: Path, monkeypatch
) -> None:
    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    called: list[str] = []

    def download(asset, destination, progress):
        called.append(asset.url)
        if asset.url.startswith("https://proxy.example/"):
            # The shape download_asset really raises once every route failed.
            raise NetworkConnectionError("proxy is down")

    monkeypatch.setattr("finesub_bootstrap.downloader.download_asset", download)

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, b"weights"),
        tmp_path / "models",
        data_root=tmp_path,
        region="cn",
    )

    assert called == [
        "https://proxy.example/https://official.example/model.ckpt",
        "https://official.example/model.ckpt",
    ]
    assert download_routes.failures(tmp_path, "github") == 1


def test_a_status_from_the_proxy_also_falls_back(tmp_path: Path, monkeypatch) -> None:
    """A proxy answering 404 is exactly what the fallback exists for.

    `raise_for_status` reports it as an HTTPStatusError, which is not a
    transport error -- a filter that only looked for connection failures would
    let a dead proxy hard-fail the install.
    """

    import httpx

    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    called: list[str] = []

    def download(asset, destination, progress):
        called.append(asset.url)
        if asset.url.startswith("https://proxy.example/"):
            raise httpx.HTTPStatusError(
                "404", request=httpx.Request("GET", asset.url), response=httpx.Response(404)
            )

    monkeypatch.setattr("finesub_bootstrap.downloader.download_asset", download)

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, b"weights"),
        tmp_path / "models",
        data_root=tmp_path,
        region="cn",
    )

    assert len(called) == 2


def test_a_full_disk_is_not_retried_or_blamed_on_the_mirror(
    tmp_path: Path, monkeypatch
) -> None:
    """Retrying downloads several hundred MB again to fail the same way, and
    recording it would degrade a mirror that did nothing wrong."""

    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    called: list[str] = []

    def download(asset, destination, progress):
        called.append(asset.url)
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("finesub_bootstrap.downloader.download_asset", download)

    with pytest.raises(OSError):
        model_fetch.fetch_fixed_files(
            _entry(tmp_path, b"weights"),
            tmp_path / "models",
            data_root=tmp_path,
            region="cn",
        )

    assert len(called) == 1
    assert download_routes.failures(tmp_path, "github") == 0


def test_a_bad_digest_from_a_proxy_restarts_rather_than_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    """Splicing bytes from two sources produces something neither of them signed."""

    from finesub_bootstrap.downloader import DigestMismatch

    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    directory = tmp_path / "models"
    directory.mkdir()
    partial = directory / "model.ckpt"
    partial.write_bytes(b"half a file")
    existed: list[bool] = []

    def download(asset, destination, progress):
        if asset.url.startswith("https://proxy.example/"):
            raise DigestMismatch("wrong body")
        existed.append(destination.exists())

    monkeypatch.setattr("finesub_bootstrap.downloader.download_asset", download)

    model_fetch.fetch_fixed_files(
        _entry(tmp_path, b"weights"), directory, data_root=tmp_path, region="cn"
    )

    assert existed == [False], "the suspect body must be gone before the retry"


def test_an_unpinnable_index_counts_as_present_once_it_exists(
    tmp_path: Path,
) -> None:
    """The separator's `download_checks.json` lives on a moving branch.

    Pinning its digest would fail every machine the day upstream edits it, so
    it carries none -- and a check that demanded one would re-download it
    forever.
    """

    index = ManifestFile(
        name="download_checks.json",
        url="https://raw.githubusercontent.com/example/download_checks.json",
        size=0,
        sha256="",
    )
    target = tmp_path / "download_checks.json"

    assert file_matches(target, index) is False
    target.write_text("{}", encoding="utf-8")
    assert file_matches(target, index) is True


def test_a_file_without_a_digest_is_left_to_the_library(
    tmp_path: Path, monkeypatch
) -> None:
    """The verified downloader exists to refuse what this file cannot promise."""

    called: list[str] = []
    monkeypatch.setattr(
        "finesub_bootstrap.downloader.download_asset",
        lambda asset, destination, progress: called.append(asset.url),
    )
    entry = ModelEntry(
        model_id="separator",
        files=(
            ManifestFile(
                name="download_checks.json",
                url="https://raw.githubusercontent.com/example/checks.json",
                size=0,
                sha256="",
            ),
        ),
    )

    model_fetch.fetch_fixed_files(
        entry, tmp_path / "models", data_root=tmp_path, region="global"
    )

    assert called == []


def test_an_interception_page_from_a_proxy_is_not_accepted(
    tmp_path: Path, monkeypatch
) -> None:
    """A 200 carrying HTML is indistinguishable from success at the HTTP layer.

    Written to disk it would leave audio-separator to die inside json.load on
    a file it now believes it already has, with the mirror never blamed.
    """

    import httpx

    _table(tmp_path, monkeypatch, githubFileProxy="https://proxy.example/")
    directory = tmp_path / "models"
    served: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        served.append(str(request.url))
        if str(request.url).startswith("https://proxy.example/"):
            return httpx.Response(200, text="<html>blocked</html>")
        return httpx.Response(200, text='{"roformer_download_list": {}}')

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "finesub_bootstrap.http_client.create_client",
        lambda route, **kwargs: httpx.Client(transport=transport),
    )

    entry = ModelEntry(
        model_id="separator",
        files=(
            ManifestFile(
                name="download_checks.json",
                url="https://raw.githubusercontent.com/example/download_checks.json",
                size=0,
                sha256="",
            ),
        ),
    )
    model_fetch.fetch_fixed_files(
        entry, directory, data_root=tmp_path, region="cn"
    )

    assert len(served) == 2, "the official URL must be tried after a bad body"
    written = (directory / "download_checks.json").read_text(encoding="utf-8")
    assert written.startswith("{"), written
    assert download_routes.failures(tmp_path, "github") == 1


def test_a_cross_process_failure_is_judged_by_what_it_says(tmp_path: Path) -> None:
    """A download that ran in a subprocess comes back as words, not as the
    httpx exception -- so the judgement has to work on the message alone."""

    class PrefetchFailed(RuntimeError):
        """Stands in for whatever a subprocess wrapper raises: any exception
        type, carrying nothing but the text that crossed the process."""

    network = PrefetchFailed("模型下载失败：Connection reset by peer")
    local = PrefetchFailed("模型下载失败：OSError: No space left on device")
    unknown = PrefetchFailed("模型下载失败：退出码 1")
    # The subprocess verifies what it downloaded; a mirror that served wrong
    # bytes surfaces as this message, and only the message crosses back.
    mismatch = PrefetchFailed(
        "模型下载失败：whisper 下载后校验失败：model.bin（清单摘要对不上）"
    )

    assert model_fetch.is_mirror_failure(network)
    assert not model_fetch.is_mirror_failure(local)
    # Unrecognised is not blamed on the mirror: that would spend gigabytes and
    # disable a working host on evidence that never pointed at it.
    assert not model_fetch.is_mirror_failure(unknown)
    assert model_fetch.is_mirror_failure(mismatch)


def test_a_verification_mismatch_is_the_mirrors_failure() -> None:
    """In-process it arrives as the exception itself, not as words."""

    from finesub_bootstrap.hf_verify import VerificationMismatch

    assert model_fetch.is_mirror_failure(
        VerificationMismatch("whisper 下载后校验失败：model.bin（清单摘要对不上）")
    )


def test_file_matches_rejects_a_truncated_copy(tmp_path: Path) -> None:
    body = b"weights"
    target = tmp_path / "model.ckpt"
    target.write_bytes(body[:-1])
    wanted = ManifestFile(
        name="model.ckpt",
        url="https://official.example/model.ckpt",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )

    assert file_matches(target, wanted) is False


def test_the_shipped_manifest_describes_the_production_models() -> None:
    """Values were taken from the official sources, not from a local cache.

    The separator files were re-downloaded from GitHub and compared byte for
    byte; the Hugging Face digests come from the Hub's file-metadata API at the
    pinned revision.

    `whisper-ja` is the one entry no default run fetches -- a listed `--model`
    alternative. It is held to the same shape as the rest: an unverifiable
    alternative would be worse than an unlisted one, because it would claim
    routing and verification it cannot deliver.
    """

    from finesub_bootstrap.model_manifest import load_manifest

    manifest = load_manifest()
    assert set(manifest) == {"separator", "whisper", "whisper-ja", "qwen-referee"}

    for model_id in ("whisper", "whisper-ja", "qwen-referee"):
        entry = manifest[model_id]
        assert entry.repo, model_id
        # A commit hash, so it covers the whole tree -- that is what stops a
        # mirror resolving a moving branch to something else.
        assert len(entry.revision) == 40, model_id
        assert all(file.is_verifiable for file in entry.files), model_id

    separator = manifest["separator"]
    names = {file.name for file in separator.files}
    # All three, or a present checkpoint separates nothing: the index is what
    # `load_model` reads before it will use anything else.
    assert names == {
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "download_checks.json",
    }
    unpinnable = [f for f in separator.files if not f.is_verifiable]
    assert [f.name for f in unpinnable] == ["download_checks.json"]


def test_every_pinned_digest_is_a_sha256() -> None:
    """A truncated or mistyped digest fails only at download time, on a user's
    machine, after several GB."""

    from finesub_bootstrap.model_manifest import load_manifest

    for entry in load_manifest().values():
        for file in entry.files:
            if not file.is_verifiable:
                continue
            assert len(file.sha256) == 64, (entry.model_id, file.name)
            assert file.size > 0, (entry.model_id, file.name)
            int(file.sha256, 16)  # raises unless it is hex
