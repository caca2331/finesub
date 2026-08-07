from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from finesub_bootstrap import environment as environment_module
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.environment import (
    REQUIRED_RUNTIME_IMPORTS,
    RuntimeEnvironment,
    runtime_probe_source,
)
from finesub_bootstrap.downloader import DownloadPaused


def _write_app_source(root: Path) -> Path:
    source = root / "app-source"
    (source / "src" / "asr_playground").mkdir(parents=True)
    (source / "desktop" / "runtime").mkdir(parents=True)
    (source / "src" / "asr_playground" / "pipeline.py").write_text(
        "PIPELINE = True\n", "utf-8"
    )
    (source / "desktop" / "__init__.py").write_text("", "utf-8")
    (source / "desktop" / "runtime" / "pylock.win-py312.toml").write_text(
        'lock-version = "1.0"\n',
        encoding="utf-8",
    )
    (source / "pyproject.toml").write_text(
        "[project]\nname='finesub'\nversion='1.0.0'\n",
        encoding="utf-8",
    )
    return source


def _runtime_lock(app_source: Path) -> Path:
    return app_source / "desktop" / "runtime" / "pylock.win-py312.toml"



def _healthy_site_packages(python_executable: Path) -> Path:
    """A site-packages that passes the filesystem health check.

    status() no longer imports anything -- that cost 15s on the bridge thread
    -- so a fake environment now has to look right on disk instead.
    """

    from finesub_bootstrap.environment import REQUIRED_RUNTIME_PACKAGE_DIRS

    site_packages = python_executable.parent.parent / "Lib" / "site-packages"
    for name in REQUIRED_RUNTIME_PACKAGE_DIRS:
        (site_packages / name).mkdir(parents=True, exist_ok=True)
    (site_packages / "ctranslate2-4.8.1+wtrefine1.cu128.dist-info").mkdir(
        exist_ok=True
    )
    return site_packages


def test_runtime_install_activates_only_a_complete_environment(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(command)
        if command[1:3] == ["venv", str(paths.runtime / "python.staging")]:
            python = (
                paths.runtime
                / "python.staging"
                / "Scripts"
                / "python.exe"
            )
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            # uv would populate site-packages; status() reads it back, so the
            # fake has to leave something there to read.
            _healthy_site_packages(python)
        return subprocess.CompletedProcess(command, 0)

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=run,
        runtime_validator=lambda _python: (True, ""),
    )

    status = runtime.install()

    assert status.state == "ready"
    assert runtime.python_executable.is_file()
    marker = json.loads(
        (paths.runtime / "python" / "finesub-runtime.json").read_text("utf-8")
    )
    expected_lock_hash = hashlib.sha256(
        (
            app_source
            / "desktop"
            / "runtime"
            / "pylock.win-py312.toml"
        ).read_bytes()
    ).hexdigest()
    assert marker["schemaVersion"] == 2
    assert marker["runtimeLockHash"] == expected_lock_hash
    dependency_commands = [
        command for command in commands if command[1:3] == ["pip", "install"]
    ]
    assert dependency_commands == [
        [
            str(uv_executable),
            "pip",
            "install",
            "--python",
            str(
                paths.runtime
                / "python.staging"
                / "Scripts"
                / "python.exe"
            ),
            "--requirement",
            str(
                app_source
                / "desktop"
                / "runtime"
                / "pylock.win-py312.toml"
            ),
        ]
    ]


def test_runtime_install_failure_preserves_the_active_environment(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    active_python = paths.runtime / "python" / "Scripts" / "python.exe"
    active_python.parent.mkdir(parents=True)
    active_python.write_bytes(b"known-good")
    (paths.runtime / "python" / "finesub-runtime.json").write_text(
        '{"schemaVersion":1,"pythonVersion":"3.12","dependencyHash":"old"}',
        encoding="utf-8",
    )
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")

    def fail_install(command, **kwargs):
        if command[1] == "venv":
            staging_python = (
                paths.runtime
                / "python.staging"
                / "Scripts"
                / "python.exe"
            )
            staging_python.parent.mkdir(parents=True)
            staging_python.write_bytes(b"incomplete")
            return subprocess.CompletedProcess(command, 0)
        raise subprocess.CalledProcessError(1, command)

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=fail_install,
    )

    try:
        runtime.install()
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("failed dependency installation was accepted")

    assert active_python.read_bytes() == b"known-good"


def test_install_skips_a_runtime_that_became_ready_while_waiting(
    tmp_path: Path,
) -> None:
    # The cross-process install lock means another FineSub process may have
    # finished this exact install before we got the lock; rebuilding would
    # tear down a runtime that is already correct.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _healthy_site_packages(paths.runtime / "python" / "Scripts" / "python.exe")

    def refuse_to_run(command, **kwargs):
        raise AssertionError(f"a ready runtime must not be rebuilt: {command}")

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=refuse_to_run,
        runtime_validator=lambda _python: (True, ""),
    )
    runtime.marker_path.write_text(
        json.dumps(runtime._marker(), sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    assert runtime.install().state == "ready"


def _staging_builder(paths: AppPaths):
    """A command runner that populates python.staging like uv would."""

    def run(command, **kwargs):
        if command[1:3] == ["venv", str(paths.runtime / "python.staging")]:
            python = paths.runtime / "python.staging" / "Scripts" / "python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            _healthy_site_packages(python)
        return subprocess.CompletedProcess(command, 0)

    return run


def test_activation_waits_out_whoever_is_holding_the_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Windows denies a directory rename while anything holds a handle inside
    # the tree -- an antivirus scanning what uv just unpacked, a sync client.
    # Those windows are short, so the swap retries instead of failing an
    # install that is otherwise complete.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=_staging_builder(paths),
        runtime_validator=lambda _python: (True, ""),
    )
    real_replace = os.replace
    remaining = {"denials": 3}

    def flaky_replace(source, destination):
        if remaining["denials"]:
            remaining["denials"] -= 1
            raise PermissionError(5, "Access is denied")
        return real_replace(source, destination)

    naps: list[float] = []
    monkeypatch.setattr(environment_module.os, "replace", flaky_replace)
    monkeypatch.setattr(environment_module.time, "sleep", naps.append)

    assert runtime.install().state == "ready"
    assert runtime.python_executable.is_file()
    assert len(naps) == 3


def test_a_blocked_swap_explains_itself_and_keeps_the_built_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Everything but the rename succeeded, so the staging tree is a finished,
    # validated runtime. Throwing it away would cost the user another 2.8GB of
    # unpacking to retry a rename.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    active_python = paths.runtime / "python" / "Scripts" / "python.exe"
    active_python.parent.mkdir(parents=True)
    active_python.write_bytes(b"known-good")
    (paths.runtime / "python" / "finesub-runtime.json").write_text(
        '{"schemaVersion":1,"pythonVersion":"3.12","runtimeLockHash":"old"}',
        encoding="utf-8",
    )
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=_staging_builder(paths),
        runtime_validator=lambda _python: (True, ""),
    )
    real_replace = os.replace

    def refuse_the_new_environment(source, destination):
        if Path(source).name == "python.staging":
            raise PermissionError(5, "Access is denied")
        return real_replace(source, destination)

    monkeypatch.setattr(environment_module.os, "replace", refuse_the_new_environment)
    monkeypatch.setattr(environment_module.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="被占用"):
        runtime.install()

    # The environment that was working before the attempt still is.
    assert active_python.read_bytes() == b"known-good"
    staging = paths.runtime / "python.staging"
    assert (staging / "finesub-runtime.json").is_file()

    # Retrying once the directory is free is a rename, not a reinstall.
    monkeypatch.undo()
    monkeypatch.setattr(environment_module.time, "sleep", lambda _seconds: None)
    runtime.command_runner = lambda command, **kwargs: (_ for _ in ()).throw(
        AssertionError(f"a validated staging must not be rebuilt: {command}")
    )

    assert runtime.install().state == "ready"
    assert runtime.python_executable.read_bytes() == b"python"
    assert not os.path.lexists(staging)


def test_a_partial_staging_is_rebuilt_rather_than_activated(
    tmp_path: Path,
) -> None:
    # Only the marker says a staging tree finished; without it the leftovers
    # are whatever an interrupted install got to, and reusing them would
    # activate a half-installed environment.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    abandoned = paths.runtime / "python.staging" / "Scripts" / "python.exe"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_bytes(b"half-installed")
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=_staging_builder(paths),
        runtime_validator=lambda _python: (True, ""),
    )

    assert runtime.install().state == "ready"
    assert runtime.python_executable.read_bytes() == b"python"


def _link_directory(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``, however this platform allows it.

    Junctions first on Windows: users who move the runtime to another disk
    reach for `mklink /J` precisely because it needs no privileges, and unlike
    a symlink it is invisible to ``Path.is_symlink()``.
    """

    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        if result.returncode == 0:
            return
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - platform gate
        pytest.skip("this platform will not create directory links")


def _install_into(paths: AppPaths, app_source: Path, tmp_path: Path):
    uv_executable = tmp_path / "uv.exe"
    uv_executable.write_bytes(b"uv")
    return RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: uv_executable,
        command_runner=_staging_builder(paths),
        runtime_validator=lambda _python: (True, ""),
    )


def test_activation_clears_a_destination_that_only_looks_absent(
    tmp_path: Path,
) -> None:
    # A link whose target is gone reads as absent through Path.exists() while
    # still owning the name -- and Windows denies a rename onto a taken name,
    # so believing exists() leaves the install failing at the last step.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    moved_away = tmp_path / "runtime-on-another-disk"
    moved_away.mkdir()
    paths.runtime.mkdir(parents=True)
    _link_directory(paths.runtime / "python", moved_away)
    moved_away.rmdir()
    assert not (paths.runtime / "python").exists()
    assert os.path.lexists(paths.runtime / "python")
    runtime = _install_into(paths, app_source, tmp_path)

    assert runtime.install().state == "ready"
    assert runtime.python_executable.read_bytes() == b"python"


def test_activation_never_deletes_through_a_redirected_runtime(
    tmp_path: Path,
) -> None:
    # Someone who put the runtime on another disk left a link behind. Clearing
    # it out of the way must remove the link; walking into it would delete the
    # directory they redirected it to.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    elsewhere = tmp_path / "runtime-on-another-disk"
    elsewhere.mkdir()
    (elsewhere / "keep-me").write_text("not ours to delete", encoding="utf-8")
    paths.runtime.mkdir(parents=True)
    _link_directory(paths.runtime / "python", elsewhere)
    runtime = _install_into(paths, app_source, tmp_path)

    assert runtime.install().state == "ready"
    assert runtime.python_executable.read_bytes() == b"python"
    assert (elsewhere / "keep-me").is_file()


def _venv_with_base(python: Path, home: Path) -> Path:
    config = python.parent.parent / "pyvenv.cfg"
    config.write_text(
        f"home = {home}\nimplementation = CPython\nversion_info = 3.12.10\n",
        encoding="utf-8",
    )
    return config


def test_a_hand_moved_environment_repairs_its_own_base_pointer(
    tmp_path: Path,
) -> None:
    # Someone dragged the whole installation to another disk. The environment
    # is fine -- it carries no stdlib, and finds one through this single
    # absolute path -- so rewriting one line rescues several GB.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _healthy_site_packages(python)
    moved_base = paths.runtime / "python-builds" / "cpython-3.12.10-windows"
    moved_base.mkdir(parents=True)
    config = _venv_with_base(python, Path("D:/old-place/python-builds") / moved_base.name)
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    runtime.marker_path.write_text(json.dumps(runtime._marker()), encoding="utf-8")

    assert runtime.status().state == "ready"
    assert f"home = {moved_base}" in config.read_text("utf-8")


def test_a_healthy_environment_is_never_written_to_by_the_health_check(
    tmp_path: Path,
) -> None:
    # status() runs on every poll of the bridge thread; it must read, decide,
    # and only then write.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _healthy_site_packages(python)
    base = tmp_path / "system-python"
    base.mkdir()
    config = _venv_with_base(python, base)
    stamp = config.stat().st_mtime_ns
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    runtime.marker_path.write_text(json.dumps(runtime._marker()), encoding="utf-8")

    assert runtime.status().state == "ready"
    assert runtime.status().state == "ready"
    assert config.stat().st_mtime_ns == stamp


def test_a_base_interpreter_that_is_simply_gone_asks_for_a_reinstall(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _healthy_site_packages(python)
    _venv_with_base(python, tmp_path / "nowhere")
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    runtime.marker_path.write_text(json.dumps(runtime._marker()), encoding="utf-8")

    status = runtime.status()

    assert status.state == "missing"
    assert "基础解释器" in status.detail


def test_installing_warns_when_the_cache_is_on_another_volume(
    tmp_path: Path,
) -> None:
    # uv hardlinks wheels from its cache into the environment only within one
    # volume; split them and the same install silently costs ~5GB more.
    if os.name != "nt":
        pytest.skip("volume anchors are a Windows concept here")
    # Checked directly rather than through install(): the point is a second
    # volume, and a test cannot conjure one.
    paths = AppPaths.for_root(tmp_path / "root", big_data=Path("Z:/FineSub"))
    app_source = _write_app_source(tmp_path)
    runtime = _install_into(paths, app_source, tmp_path)
    messages: list[str] = []

    runtime._warn_if_cache_is_on_another_volume(messages.append)

    assert any("不在同一磁盘" in message for message in messages)

    same_volume = _install_into(
        AppPaths.for_root(tmp_path / "root"), app_source, tmp_path
    )
    quiet: list[str] = []
    same_volume._warn_if_cache_is_on_another_volume(quiet.append)

    assert quiet == []


def test_force_probe_runs_the_real_validator(tmp_path: Path) -> None:
    # The filesystem check cannot see damage inside a package directory; the
    # diagnostic path (`finesub doctor`) opts into the import probe instead.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    _healthy_site_packages(python)
    probes: list[int] = []
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
        runtime_validator=lambda _python: (probes.append(1), (True, ""))[1],
    )
    runtime.marker_path.write_text(
        json.dumps(runtime._marker()),
        encoding="utf-8",
    )

    assert runtime.status().state == "ready"
    assert probes == []
    assert runtime.status(force_probe=True).state == "ready"
    assert probes == [1]


def test_runtime_status_rejects_a_marker_when_a_package_went_missing(
    tmp_path: Path,
) -> None:
    # The marker says the install validated, but a package has since been
    # removed. status() catches that on the filesystem -- it must not import
    # anything, because it runs on the thread that draws the window.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    site_packages = _healthy_site_packages(python)
    shutil.rmtree(site_packages / "faster_whisper")
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    runtime.marker_path.write_text(
        json.dumps(runtime._marker()),
        encoding="utf-8",
    )

    status = runtime.status()

    assert status.state == "missing"
    assert "faster_whisper" in status.detail


def test_status_refuses_a_stock_ctranslate2_without_importing_it(
    tmp_path: Path,
) -> None:
    # Stock CT2 passes every path check; only the version tells them apart, and
    # dist-info carries it without loading the module.
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    site_packages = _healthy_site_packages(python)
    (site_packages / "ctranslate2-4.8.1+wtrefine1.cu128.dist-info").rmdir()
    (site_packages / "ctranslate2-4.8.1.dist-info").mkdir()
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    runtime.marker_path.write_text(
        json.dumps(runtime._marker()),
        encoding="utf-8",
    )

    status = runtime.status()

    assert status.state == "missing"
    assert "ctranslate2" in status.detail


def test_worker_context_uses_current_app_ffmpeg_and_private_model_caches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path / "home"))
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    python = paths.runtime / "python" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")
    ffmpeg_bin = paths.runtime / "ffmpeg" / "7.1" / "bin"
    ffmpeg_bin.mkdir(parents=True)

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
    )
    context = runtime.worker_context(
        ffmpeg_bin=ffmpeg_bin,
        extra_env={"GEMINI_FREE": "user-secret"},
    )

    python_path_parts = context.environment["PYTHONPATH"].split(
        ";" if __import__("os").name == "nt" else ":"
    )
    assert context.python_executable == python
    assert context.working_directory == app_source
    assert python_path_parts[:2] == [str(app_source), str(app_source / "src")]
    assert context.environment["PATH"].split(__import__("os").pathsep)[0] == str(
        ffmpeg_bin
    )
    assert context.environment["FINESUB_MODEL_DIR"] == str(paths.models)
    # Managed, because this machine's conventional HF cache holds none of the
    # repositories the pipeline uses (see test_model_caches for the other side).
    assert context.environment["HF_HOME"] == str(paths.models / "huggingface")
    assert context.environment["TORCH_HOME"] == str(paths.models / "torch")
    assert context.environment["GEMINI_FREE"] == "user-secret"


def test_development_runtime_uses_existing_interpreter_without_installing(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    development_python = tmp_path / "venv" / "Scripts" / "python.exe"
    development_python.parent.mkdir(parents=True)
    development_python.write_bytes(b"python")
    _healthy_site_packages(development_python)

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: (_ for _ in ()).throw(
            AssertionError("development runtime must not download uv")
        ),
        development_python=development_python,
        runtime_validator=lambda _python: (True, ""),
    )

    assert runtime.status().state == "ready"
    assert runtime.install().state == "ready"
    assert runtime.python_executable == development_python.resolve()


def test_development_runtime_rejects_missing_worker_dependency(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    development_python = tmp_path / "venv" / "Scripts" / "python.exe"
    development_python.parent.mkdir(parents=True)
    development_python.write_bytes(b"python")
    site_packages = _healthy_site_packages(development_python)
    shutil.rmtree(site_packages / "audio_separator")

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
        development_python=development_python,
    )

    status = runtime.status()

    assert status.state == "missing"
    assert "audio_separator" in status.detail


def test_pause_terminates_the_runtime_installer_process_tree(
    tmp_path: Path,
) -> None:
    paths = AppPaths.for_root(tmp_path / "root")
    app_source = _write_app_source(tmp_path)
    terminated: list[int] = []

    class FakeProcess:
        pid = 9876
        stdout = io.StringIO("")
        returncode: int | None = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 1
            return self.returncode

        def kill(self):
            self.returncode = 1

    process = FakeProcess()
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        runtime_lock=_runtime_lock(app_source),
        uv_executable=lambda: tmp_path / "uv.exe",
        process_factory=lambda *args, **kwargs: process,
        process_terminator=lambda child: terminated.append(child.pid),
    )
    pause_checks = iter((False, True))

    with pytest.raises(DownloadPaused):
        runtime._run(
            ["uv.exe", "pip", "install"],
            {},
            log=None,
            should_pause=lambda: next(pause_checks),
        )

    assert terminated == [9876]


def _run_probe(modules: tuple[str, ...], label: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", runtime_probe_source(modules, label)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_runtime_probe_accepts_an_environment_with_every_module() -> None:
    # Cheap stand-ins for the real list: what is under test is the probe's exit
    # protocol, not whether this interpreter happens to have the ASR stack.
    assert _run_probe(("json", "pathlib"), "wtrefine").returncode == 0


def test_runtime_probe_names_the_module_that_is_missing() -> None:
    result = _run_probe(("json", "asr_stack_not_installed"), "wtrefine")

    assert result.returncode == 1
    assert "asr_stack_not_installed" in result.stderr.strip().splitlines()[-1]


def test_runtime_probe_rejects_stock_ctranslate2(tmp_path: Path) -> None:
    # Stock CT2 imports fine and satisfies ctranslate2==4.8.1, so only the local
    # label separates it from the build fw-refine needs. Stand in for it with a
    # module on sys.path rather than requiring either real wheel.
    stub = tmp_path / "ctranslate2.py"
    stub.write_text("__version__ = '4.8.1'\n", encoding="utf-8")
    source = runtime_probe_source(("ctranslate2",), "wtrefine")
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )

    assert result.returncode == 1
    assert "stock build" in result.stderr


def test_runtime_probe_accepts_the_patched_ctranslate2(tmp_path: Path) -> None:
    stub = tmp_path / "ctranslate2.py"
    stub.write_text("__version__ = '4.8.1+wtrefine1.cu128'\n", encoding="utf-8")
    source = runtime_probe_source(("ctranslate2",), "wtrefine")
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )

    assert result.returncode == 0, result.stderr


def test_required_runtime_imports_cover_the_decode_chain() -> None:
    # The list started out as separator-only, which let a lock with no ASR
    # decoder at all pass validation. Each of these can go missing on its own.
    assert {
        "pydantic",
        "audio_separator.separator",
        "faster_whisper",
        "ctranslate2",
        "silero_vad",
    } <= set(REQUIRED_RUNTIME_IMPORTS)
