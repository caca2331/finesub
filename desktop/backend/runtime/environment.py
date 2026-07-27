from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any

from desktop.backend.common.http_client import apply_network_environment
from desktop.backend.common.models import ResourceStatus
from desktop.backend.common.paths import AppPaths
from desktop.backend.resources.downloader import DownloadPaused


CommandRunner = Callable[..., Any]
StageCallback = Callable[[str, str], None]
LogCallback = Callable[[str], None]
PauseCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class WorkerContext:
    python_executable: Path
    working_directory: Path
    environment: dict[str, str]


class RuntimeEnvironment:
    schema_version = 2

    def __init__(
        self,
        *,
        paths: AppPaths,
        app_source: Path,
        uv_executable: Callable[[], Path],
        command_runner: CommandRunner = subprocess.run,
        python_version: str = "3.12",
        development_python: Path | None = None,
    ) -> None:
        self.paths = paths
        self.app_source = app_source.expanduser().resolve()
        self.uv_executable = uv_executable
        self.command_runner = command_runner
        self.python_version = python_version
        self.development_python = (
            development_python.expanduser().resolve()
            if development_python is not None
            else None
        )
        self._system_python: Path | None = None
        self._system_python_checked = False

    @property
    def runtime_root(self) -> Path:
        return self.paths.runtime / "python"

    @property
    def python_executable(self) -> Path:
        if self.development_python is not None:
            return self.development_python
        return self.runtime_root / "Scripts" / "python.exe"

    @property
    def marker_path(self) -> Path:
        return self.runtime_root / "finesub-runtime.json"

    @property
    def runtime_lock(self) -> Path:
        return (
            self.app_source
            / "desktop"
            / "runtime"
            / "pylock.win-py312.toml"
        )

    def status(self) -> ResourceStatus:
        if self.development_python is not None:
            return self._status(
                "ready" if self.development_python.is_file() else "missing"
            )
        if not self.python_executable.is_file() or not self.marker_path.is_file():
            system_python = self.system_python()
            if system_python is not None:
                return self._status(
                    "missing",
                    f"已检测到系统 Python {self.python_version}：{system_python}；"
                    "将直接复用，只需安装 FineSub AI 依赖。",
                )
            return self._status("missing")
        try:
            marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self._status("missing")
        if marker != self._marker():
            return self._status(
                "missing",
                "应用依赖已变化，需要更新 Python 运行环境。",
            )
        return self._status("ready")

    def install(
        self,
        *,
        stage: StageCallback | None = None,
        log: LogCallback | None = None,
        should_pause: PauseCheck | None = None,
    ) -> ResourceStatus:
        if self.development_python is not None:
            if not self.development_python.is_file():
                raise FileNotFoundError(
                    f"Development Python was not found: {self.development_python}"
                )
            return self.status()
        uv = self.uv_executable().expanduser().resolve()
        if not uv.is_file():
            raise FileNotFoundError(f"uv bootstrap executable was not found: {uv}")

        self.paths.runtime.mkdir(parents=True, exist_ok=True)
        self.paths.cache.mkdir(parents=True, exist_ok=True)
        staging = self.paths.runtime / "python.staging"
        previous = self.paths.runtime / "python.previous"
        if staging.exists():
            shutil.rmtree(staging)

        environment = os.environ.copy()
        environment.update(
            {
                "UV_CACHE_DIR": str(self.paths.cache / "uv"),
                "UV_PYTHON_INSTALL_DIR": str(
                    self.paths.runtime / "python-builds"
                ),
                "PYTHONUTF8": "1",
                "UV_SYSTEM_PYTHON": "1",
            }
        )
        apply_network_environment(environment)
        try:
            base_python = self.system_python()
            if base_python is None:
                if stage is not None:
                    stage("installing_python", "正在安装 Python 3.12")
                self._run(
                    [
                        str(uv),
                        "python",
                        "install",
                        self.python_version,
                        "--no-bin",
                        "--no-registry",
                    ],
                    environment,
                    log=log,
                    should_pause=should_pause,
                )
                python_selector = self.python_version
            else:
                python_selector = str(base_python)
                if stage is not None:
                    stage(
                        "installing_python",
                        f"已检测到系统 Python {self.python_version}，跳过解释器下载",
                    )
                if log is not None:
                    log(f"Using system Python: {base_python}")
            if stage is not None:
                stage("creating_environment", "正在创建隔离运行环境")
            self._run(
                [
                    str(uv),
                    "venv",
                    str(staging),
                    "--python",
                    python_selector,
                ],
                environment,
                log=log,
                should_pause=should_pause,
            )
            staging_python = staging / "Scripts" / "python.exe"
            if not staging_python.is_file():
                raise FileNotFoundError(
                    "uv completed without creating the managed Python executable"
                )
            if stage is not None:
                stage("installing_dependencies", "正在安装 FineSub AI 依赖")
            self._run(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(staging_python),
                    "--requirement",
                    str(self.runtime_lock),
                ],
                environment,
                log=log,
                should_pause=should_pause,
            )
            (staging / "finesub-runtime.json").write_text(
                json.dumps(
                    self._marker(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
                newline="\n",
            )
            if should_pause is not None and should_pause():
                raise DownloadPaused("Python environment installation paused")
            if stage is not None:
                stage("activating", "正在校验并启用 Python 环境")
            self._activate(staging, previous)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise
        return self.status()

    def system_python(self) -> Path | None:
        if self.development_python is not None:
            return self.development_python if self.development_python.is_file() else None
        if self._system_python_checked:
            return self._system_python
        self._system_python_checked = True
        self._system_python = self._find_system_python()
        return self._system_python

    def worker_context(
        self,
        *,
        ffmpeg_bin: Path | None,
        extra_env: Mapping[str, str],
    ) -> WorkerContext:
        source_paths = [str(self.app_source), str(self.app_source / "src")]
        existing_python_path = os.environ.get("PYTHONPATH")
        if existing_python_path:
            source_paths.append(existing_python_path)
        environment = dict(extra_env)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(source_paths),
                "PYTHONUTF8": "1",
                "FINESUB_MODEL_DIR": str(self.paths.models),
                "HF_HOME": str(self.paths.models / "huggingface"),
                "TORCH_HOME": str(self.paths.models / "torch"),
                "UV_CACHE_DIR": str(self.paths.cache / "uv"),
            }
        )
        if ffmpeg_bin is not None:
            current_path = os.environ.get("PATH", "")
            environment["PATH"] = os.pathsep.join(
                part for part in (str(ffmpeg_bin), current_path) if part
            )
        return WorkerContext(
            python_executable=self.python_executable,
            working_directory=self.app_source,
            environment=environment,
        )

    def _run(
        self,
        command: list[str],
        environment: dict[str, str],
        *,
        log: LogCallback | None,
        should_pause: PauseCheck | None,
    ) -> None:
        if should_pause is not None and should_pause():
            raise DownloadPaused("Python environment installation paused")
        if self.command_runner is not subprocess.run:
            self.command_runner(
                command,
                cwd=self.app_source,
                env=environment,
                check=True,
            )
            if should_pause is not None and should_pause():
                raise DownloadPaused("Python environment installation paused")
            return

        process = subprocess.Popen(
            command,
            cwd=self.app_source,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if os.name == "nt"
                else 0
            ),
        )
        lines: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for output in process.stdout:
                lines.put(output.rstrip())

        reader = threading.Thread(
            target=read_output,
            name="finesub-runtime-installer-output",
            daemon=True,
        )
        reader.start()
        while process.poll() is None:
            self._drain_logs(lines, log)
            if should_pause is not None and should_pause():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                reader.join(timeout=1)
                self._drain_logs(lines, log)
                raise DownloadPaused("Python environment installation paused")
            time.sleep(0.1)
        reader.join(timeout=1)
        self._drain_logs(lines, log)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)

    def _find_system_python(self) -> Path | None:
        candidates: list[list[str]] = []
        launcher = shutil.which("py")
        if launcher:
            candidates.append(
                [
                    launcher,
                    f"-{self.python_version}",
                    "-c",
                    "import sys; print(sys.executable)",
                ]
            )
        for name in (f"python{self.python_version}", "python"):
            executable = shutil.which(name)
            if executable:
                candidates.append(
                    [
                        executable,
                        "-c",
                        "import sys; print(sys.executable)",
                    ]
                )

        expected = tuple(int(part) for part in self.python_version.split(".", 1))
        checked: set[str] = set()
        for command in candidates:
            key = os.path.normcase(command[0])
            if key in checked:
                continue
            checked.add(key)
            try:
                result = subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
                path = Path(result.stdout.strip()).expanduser().resolve()
                version = subprocess.run(
                    [
                        str(path),
                        "-c",
                        "import sys; print(sys.version_info[0], sys.version_info[1])",
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=5,
                    creationflags=(
                        getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        if os.name == "nt"
                        else 0
                    ),
                )
                actual = tuple(int(part) for part in version.stdout.split())
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            if path.is_file() and actual == expected:
                return path
        return None

    @staticmethod
    def _drain_logs(
        lines: queue.Queue[str],
        log: LogCallback | None,
    ) -> None:
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            if log is not None and line:
                log(line)

    def _activate(self, staging: Path, previous: Path) -> None:
        if previous.exists():
            shutil.rmtree(previous)
        had_active = self.runtime_root.exists()
        if had_active:
            os.replace(self.runtime_root, previous)
        try:
            os.replace(staging, self.runtime_root)
        except Exception:
            if had_active and previous.exists() and not self.runtime_root.exists():
                os.replace(previous, self.runtime_root)
            raise
        if previous.exists():
            shutil.rmtree(previous)

    def _marker(self) -> dict[str, object]:
        if not self.runtime_lock.is_file():
            raise FileNotFoundError(
                "FineSub desktop runtime lock was not found: "
                f"{self.runtime_lock}"
            )
        return {
            "schemaVersion": self.schema_version,
            "pythonVersion": self.python_version,
            "runtimeLockHash": hashlib.sha256(
                self.runtime_lock.read_bytes()
            ).hexdigest(),
        }

    def _status(self, state: str, detail: str = "") -> ResourceStatus:
        return ResourceStatus(
            id="uv",
            version=f"Python {self.python_version}",
            state=state,
            detail=detail,
        )
