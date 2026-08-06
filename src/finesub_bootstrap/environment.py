from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
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

from finesub_bootstrap.http_client import apply_network_environment
from finesub_bootstrap.model_caches import existing_hf_home
from finesub_bootstrap.models import ResourceStatus
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.processes import terminate_process_tree
from finesub_bootstrap.downloader import DownloadPaused


def shared_environment_overrides(paths: AppPaths) -> dict[str, str]:
    """Point the pipeline at the shared personal-data directory.

    The CLI and the desktop launch the same pipeline against the same
    ``user-data`` tree, so they have to agree on where it is. The knowledge
    base especially: left to resolve itself it walks up from the worker's
    source directory and lands in ``app/versions/<version>/knowledge``, which
    the next app update replaces -- silently taking the knowledge base with it.

    Only fills variables the user has not set themselves: an explicit
    environment always wins over the launcher's defaults.
    """

    overrides: dict[str, str] = {}
    env_file = paths.user_data / ".env"
    if "FINESUB_ENV_FILE" not in os.environ and env_file.is_file():
        overrides["FINESUB_ENV_FILE"] = str(env_file)
    config_file = paths.user_data / "config.toml"
    if "FINESUB_CONFIG_FILE" not in os.environ and config_file.is_file():
        overrides["FINESUB_CONFIG_FILE"] = str(config_file)
    if "FINESUB_KNOWLEDGE_ROOT" not in os.environ:
        overrides["FINESUB_KNOWLEDGE_ROOT"] = str(paths.user_data / "knowledge")
    # Cross-process limiter state (a single JSON file, despite the variable's
    # name). Left to resolve itself it lands either at the %LOCALAPPDATA%
    # default (wrong under a custom FINESUB_HOME) or, for the desktop worker,
    # inside the versioned app directory the next update orphans.
    if "FINESUB_STATE_DIR" not in os.environ:
        overrides["FINESUB_STATE_DIR"] = str(paths.cache / "state")
    return overrides


CommandRunner = Callable[..., Any]
StageCallback = Callable[[str, str], None]
LogCallback = Callable[[str], None]
PauseCheck = Callable[[], bool]
ProcessFactory = Callable[..., Any]
ProcessTerminator = Callable[[Any], None]
RuntimeValidator = Callable[[Path], tuple[bool, str]]


# One name per thing that can go missing independently: the worker's own IPC
# models, the separator stack (whose deps it pulls in transitively and has
# broken on before), the ASR decode chain, and the optional-by-CLI-default
# extras the pipeline reaches for. An environment that imports all of these can
# run a task end to end; one that cannot must be reported as needing repair
# rather than failing halfway through a job.
REQUIRED_RUNTIME_IMPORTS = (
    "pydantic",
    "audio_separator.separator",
    "beartype",
    "ml_collections",
    "faster_whisper",
    "ctranslate2",
    "silero_vad",
    "transformers",
)

# Stock CTranslate2 satisfies `import ctranslate2` and even the version pin, but
# cannot run fw-refine -- only the patched build emits the decoder trace. The
# lock installs the right one by hashed URL; this catches an environment that
# drifted off it. See docs/ct2-distribution.md.
REQUIRED_CTRANSLATE2_LOCAL_LABEL = "wtrefine"

# The same requirement expressed as directories under site-packages, for the
# checks that must not cost 15 seconds. Import names differ from distribution
# names, so these are the on-disk package directories, not the pip names.
REQUIRED_RUNTIME_PACKAGE_DIRS = (
    "pydantic",
    "audio_separator",
    "faster_whisper",
    "ctranslate2",
    "silero_vad",
    "transformers",
    "torch",
)


def runtime_probe_source(modules: tuple[str, ...], ctranslate2_label: str) -> str:
    """Build the `python -c` program that decides whether a runtime is usable.

    Kept separate from the subprocess call so it can be exercised against a
    real interpreter with a cheap module list, rather than only against an
    environment that already has the multi-gigabyte ASR stack installed.
    """

    return (
        "import importlib\n"
        "mods = {}\n"
        f"for name in {list(modules)!r}:\n"
        "    mods[name] = importlib.import_module(name)\n"
        "ct2 = mods.get('ctranslate2')\n"
        f"label = {ctranslate2_label!r}\n"
        "version = getattr(ct2, '__version__', '') if ct2 is not None else ''\n"
        "problem = (\n"
        "    f'ctranslate2 {version} is the stock build, not the patched one'\n"
        "    if ct2 is not None and label not in version\n"
        "    else None\n"
        ")\n"
        # None exits 0; any string -- including '' -- exits 1 and is printed.
        "raise SystemExit(problem)\n"
    )


@dataclass(frozen=True, slots=True)
class WorkerContext:
    python_executable: Path
    working_directory: Path
    environment: dict[str, str]


@contextmanager
def _holding_install_lock(
    lock_path: Path,
    *,
    log: LogCallback | None,
    should_pause: PauseCheck | None,
) -> Iterator[None]:
    """Serialize runtime installation across FineSub processes.

    The desktop app and the CLI shell can both decide the runtime needs
    (re)building; the staging swap must not run twice concurrently. Advisory
    byte lock on a sidecar file: waiting is polled so a pause request still
    gets through, and the sidecar itself is never deleted.
    """

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        announced = False
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if should_pause is not None and should_pause():
                    raise DownloadPaused(
                        "Python environment installation paused"
                    )
                if not announced and log is not None:
                    log(
                        "Another FineSub process is installing the runtime; "
                        "waiting for it to finish"
                    )
                announced = True
                time.sleep(0.5)
        try:
            yield
        finally:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        handle.close()


class RuntimeEnvironment:
    schema_version = 2

    def __init__(
        self,
        *,
        paths: AppPaths,
        app_source: Path,
        runtime_lock: Path,
        uv_executable: Callable[[], Path],
        command_runner: CommandRunner = subprocess.run,
        process_factory: ProcessFactory = subprocess.Popen,
        process_terminator: ProcessTerminator = terminate_process_tree,
        runtime_validator: RuntimeValidator | None = None,
        python_version: str = "3.12",
        development_python: Path | None = None,
    ) -> None:
        self.paths = paths
        self.app_source = app_source.expanduser().resolve()
        self.runtime_lock = runtime_lock.expanduser().resolve()
        self.uv_executable = uv_executable
        self.command_runner = command_runner
        self.process_factory = process_factory
        self.process_terminator = process_terminator
        self.runtime_validator = runtime_validator or self._validate_python
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

    def status(self, *, force_probe: bool = False) -> ResourceStatus:
        """Report whether the runtime is usable.

        ``force_probe`` replaces the instant filesystem health check with the
        real import probe (seconds, spawns the runtime Python) — the
        diagnostic path (`finesub doctor`) uses it so damage inside packages
        still gets caught.
        """

        if self.development_python is not None:
            if not self.development_python.is_file():
                return self._status("missing")
            healthy, detail = self._python_health(
                self.development_python,
                force_probe=force_probe,
            )
            if not healthy:
                return self._status(
                    "missing",
                    detail
                    or "开发 Python 环境缺少 FineSub 必需依赖。",
                )
            return self._status("ready")
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
        healthy, detail = self._active_runtime_health(force_probe=force_probe)
        if not healthy:
            return self._status(
                "missing",
                detail
                or "Python 运行环境缺少 FineSub 必需依赖，需要修复。",
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
        with _holding_install_lock(
            self.paths.runtime / ".install.lock",
            log=log,
            should_pause=should_pause,
        ):
            return self._install_locked(
                uv,
                stage=stage,
                log=log,
                should_pause=should_pause,
            )

    def _install_locked(
        self,
        uv: Path,
        *,
        stage: StageCallback | None,
        log: LogCallback | None,
        should_pause: PauseCheck | None,
    ) -> ResourceStatus:
        # Whoever held the lock before us may have finished the very install
        # we queued up for; redoing it would tear down a runtime that is
        # already correct (and possibly in use).
        current = self.status()
        if current.state == "ready":
            return current

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
            healthy, detail = self.runtime_validator(staging_python)
            if not healthy:
                raise RuntimeError(
                    detail
                    or "FineSub runtime dependency validation failed"
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
        extra_path_dirs: Sequence[Path] = (),
        extra_python_path: Sequence[Path] = (),
    ) -> WorkerContext:
        """Environment for the worker subprocess.

        PATH and PYTHONPATH are composed here rather than taken from
        ``extra_env``: this method owns them, and a caller that merely set them
        in ``extra_env`` would have them silently overwritten. Extra entries go
        through ``extra_path_dirs`` / ``extra_python_path`` instead -- managed
        tools that are found by execution (git) or by import (yt-dlp).
        """

        source_paths = [str(self.app_source), str(self.app_source / "src")]
        existing_python_path = os.environ.get("PYTHONPATH")
        if existing_python_path:
            source_paths.append(existing_python_path)
        # Appended, so the app's own modules still win any name clash.
        source_paths.extend(str(path) for path in extra_python_path)
        environment = dict(extra_env)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(source_paths),
                "PYTHONUTF8": "1",
                "FINESUB_MODEL_DIR": str(self.paths.models),
                # Weights this machine already has are not downloaded again;
                # see finesub_bootstrap.model_caches for why the granularity
                # differs between the separator and Hugging Face.
                "HF_HOME": str(
                    existing_hf_home(self.paths.models / "huggingface")
                ),
                "TORCH_HOME": str(self.paths.models / "torch"),
                "UV_CACHE_DIR": str(self.paths.cache / "uv"),
            }
        )
        prepended = [str(path) for path in (ffmpeg_bin, *extra_path_dirs) if path]
        if prepended:
            current_path = os.environ.get("PATH", "")
            environment["PATH"] = os.pathsep.join(
                part for part in (*prepended, current_path) if part
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

        process = self.process_factory(
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
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt"
                else 0
            ),
            start_new_session=os.name != "nt",
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
                self.process_terminator(process)
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

    def _active_runtime_health(
        self,
        *,
        force_probe: bool = False,
    ) -> tuple[bool, str]:
        return self._python_health(
            self.python_executable,
            force_probe=force_probe,
        )

    def _python_health(
        self,
        python_executable: Path,
        *,
        force_probe: bool = False,
    ) -> tuple[bool, str]:
        """Cheap, synchronous check that an environment is still intact.

        Deliberately *not* the import probe: that spawns a Python which loads
        torch and the whole decode chain, ~15s warm and far worse cold, and
        `status()` is called from the bridge thread on every poll -- so the UI
        froze while re-proving something `install()` had already proven before
        it wrote the marker. Here we only look at the filesystem, which catches
        the case this is really for: packages removed after a good install.

        ``force_probe`` runs the real import probe instead -- the diagnostic
        path (`finesub doctor`) uses it to catch damage inside packages that
        the directory check cannot see.
        """

        if force_probe:
            return self.runtime_validator(python_executable)
        site_packages = python_executable.parent.parent / "Lib" / "site-packages"
        if not python_executable.is_file() or not site_packages.is_dir():
            return False, "Python 运行环境不完整。"
        missing = [
            name
            for name in REQUIRED_RUNTIME_PACKAGE_DIRS
            if not (site_packages / name).is_dir()
        ]
        if missing:
            return False, f"Python 运行环境缺少必需依赖：{', '.join(missing)}"
        # Stock CTranslate2 satisfies every path check above; only the version
        # separates it from the build fw-refine needs, and dist-info carries it
        # without importing anything.
        labels = [
            item.name
            for item in site_packages.glob("ctranslate2-*.dist-info")
        ]
        if labels and not any(
            REQUIRED_CTRANSLATE2_LOCAL_LABEL in label for label in labels
        ):
            return False, (
                "ctranslate2 是原版而非补丁版，ASR 无法运行；请重装运行环境。"
            )
        return True, ""

    def _validate_python(self, python_executable: Path) -> tuple[bool, str]:
        probe = runtime_probe_source(
            REQUIRED_RUNTIME_IMPORTS,
            REQUIRED_CTRANSLATE2_LOCAL_LABEL,
        )
        command = [str(python_executable), "-I", "-c", probe]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                # Generous on purpose: the probe imports torch (via the
                # separator and silero) plus the whole decode chain, which on a
                # cold cache is seconds, not milliseconds. A timeout here is
                # reported as a broken runtime, so erring short is the costly
                # direction. The result is cached against site-packages mtime,
                # so a healthy environment pays this once.
                timeout=120,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"Python 运行环境检测失败：{error}"
        if result.returncode == 0:
            return True, ""
        details = result.stderr.strip().splitlines()
        reason = details[-1] if details else "required module import failed"
        return False, f"Python 运行环境缺少必需依赖：{reason}"

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
