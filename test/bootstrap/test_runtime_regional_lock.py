"""Installing from a regional lock, and falling back to the canonical one."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from finesub_bootstrap import download_routes
from finesub_bootstrap.downloader import DownloadPaused
from finesub_bootstrap.environment import RuntimeEnvironment
from finesub_bootstrap.paths import AppPaths


def _app_source(root: Path) -> Path:
    source = root / "app-source"
    (source / "src" / "finesub").mkdir(parents=True)
    (source / "desktop" / "runtime").mkdir(parents=True)
    (source / "src" / "finesub" / "pipeline.py").write_text("X = 1\n", "utf-8")
    (source / "desktop" / "runtime" / "pylock.win-py312.toml").write_text(
        'lock-version = "1.0"\n', encoding="utf-8"
    )
    return source


def _runtime(tmp_path: Path, run) -> RuntimeEnvironment:
    paths = AppPaths.for_root(tmp_path / "root", data_root=tmp_path / "data")
    source = _app_source(tmp_path)
    uv = tmp_path / "uv.exe"
    uv.write_bytes(b"uv")
    return RuntimeEnvironment(
        paths=paths,
        app_source=source,
        runtime_lock=source / "desktop" / "runtime" / "pylock.win-py312.toml",
        uv_executable=lambda: uv,
        command_runner=run,
        runtime_validator=lambda _python: (True, ""),
    )


def _write_regional(runtime: RuntimeEnvironment) -> Path:
    regional = runtime.runtime_lock.with_name("pylock.win-py312.cn.toml")
    regional.write_text('lock-version = "1.0"\n', encoding="utf-8")
    return regional


def test_without_a_regional_lock_the_canonical_one_is_used(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)

    assert runtime.regional_lock() is None


def test_a_regional_lock_is_used_only_in_its_region(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    regional = _write_regional(runtime)

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "global")
    assert runtime.regional_lock() is None

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    assert runtime.regional_lock() == regional


def test_a_degraded_mirror_is_not_used_again(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)
    for _ in range(download_routes.FAILURE_LIMIT):
        download_routes.record_failure(runtime.paths.data_root, "pypi")

    assert runtime.regional_lock() is None


def test_the_marker_does_not_change_with_the_region(
    tmp_path: Path, monkeypatch
) -> None:
    """Otherwise moving between regions rebuilds a 5 GB environment that is
    already correct: the two locks describe the same files."""

    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "global")
    global_marker = json.dumps(runtime._marker(), sort_keys=True)
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    cn_marker = json.dumps(runtime._marker(), sort_keys=True)

    assert global_marker == cn_marker


def _install_calls(runtime: RuntimeEnvironment, failure=None) -> list[str]:
    used: list[str] = []

    def run(command, **kwargs):
        if command[1] == "pip":
            lock = command[command.index("--requirement") + 1]
            used.append(Path(lock).name)
            if failure is not None and lock.endswith(".cn.toml"):
                raise failure
        return subprocess.CompletedProcess(command, 0)

    runtime.command_runner = run
    staging_python = runtime.paths.runtime / "python.staging" / "python.exe"
    staging_python.parent.mkdir(parents=True, exist_ok=True)
    runtime._install_dependencies(
        Path("uv.exe"), staging_python, {}, log=None, should_pause=None
    )
    return used


def _uv_failure(output: str) -> subprocess.CalledProcessError:
    """The shape `_run` really raises.

    Its message is only "returned non-zero exit status 1"; everything that
    says *why* is in the captured output. A test that raises a RuntimeError
    carrying the reason in its message proves nothing about production.
    """

    return subprocess.CalledProcessError(1, ["uv", "pip", "install"], output=output)


def test_a_network_failure_retries_the_whole_install_against_the_official_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    used = _install_calls(
        runtime,
        failure=_uv_failure(
            "error: Failed to fetch numpy: connection reset by peer"
        ),
    )

    assert used == ["pylock.win-py312.cn.toml", "pylock.win-py312.toml"]
    assert download_routes.failures(runtime.paths.data_root, "pypi") == 1


def test_a_digest_mismatch_from_a_mirror_falls_back_to_the_official_source(
    tmp_path: Path, monkeypatch
) -> None:
    """From a mirror this is evidence of a bad mirror, not a bad file.

    The canonical lock carries the same digests, so retrying against it either
    succeeds with verified bytes or fails the same way -- and refusing to retry
    would leave a stale mirror as a hard install failure with no way out.
    """

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    used = _install_calls(
        runtime,
        failure=_uv_failure("error: Hash mismatch for numpy-2.4.6.whl"),
    )

    assert used == ["pylock.win-py312.cn.toml", "pylock.win-py312.toml"]
    assert download_routes.failures(runtime.paths.data_root, "pypi") == 1


def test_a_disk_failure_is_not_dressed_up_as_a_mirror_problem(
    tmp_path: Path, monkeypatch
) -> None:
    """A second host cannot make room on a full disk."""

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    with pytest.raises(subprocess.CalledProcessError):
        _install_calls(
            runtime,
            failure=_uv_failure("error: failed to write: No space left on device"),
        )


def test_a_failure_that_says_nothing_is_not_retried(
    tmp_path: Path, monkeypatch
) -> None:
    """Silence is not evidence that a different host would do better."""

    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    with pytest.raises(subprocess.CalledProcessError):
        _install_calls(runtime, failure=_uv_failure(""))


def test_a_pause_is_not_retried_elsewhere(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)

    with pytest.raises(DownloadPaused):
        _install_calls(runtime, failure=DownloadPaused("paused"))


def test_a_successful_regional_install_clears_the_counter(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("FINESUB_DOWNLOAD_REGION", "cn")
    runtime = _runtime(tmp_path, lambda command, **kwargs: None)
    _write_regional(runtime)
    download_routes.record_failure(runtime.paths.data_root, "pypi")

    used = _install_calls(runtime)

    assert used == ["pylock.win-py312.cn.toml"]
    assert download_routes.failures(runtime.paths.data_root, "pypi") == 0
