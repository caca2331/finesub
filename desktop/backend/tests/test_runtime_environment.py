from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess

import pytest

from desktop.backend.common.paths import AppPaths
from desktop.backend.runtime.environment import RuntimeEnvironment
from desktop.backend.resources.downloader import DownloadPaused


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
        return subprocess.CompletedProcess(command, 0)

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        uv_executable=lambda: uv_executable,
        command_runner=run,
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


def test_worker_context_uses_current_app_ffmpeg_and_private_model_caches(
    tmp_path: Path,
) -> None:
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

    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=app_source,
        uv_executable=lambda: (_ for _ in ()).throw(
            AssertionError("development runtime must not download uv")
        ),
        development_python=development_python,
    )

    assert runtime.status().state == "ready"
    assert runtime.install().state == "ready"
    assert runtime.python_executable == development_python.resolve()


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
