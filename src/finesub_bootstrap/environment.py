from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
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

from finesub_bootstrap.fsops import remove_tree, write_atomic
from finesub_bootstrap.http_client import apply_network_environment
from finesub_bootstrap.locks import holding_lock
from finesub_bootstrap.model_caches import existing_hf_home
from finesub_bootstrap.models import ResourceStatus
from finesub_bootstrap.paths import AppPaths
from finesub_bootstrap.processes import terminate_process_tree
from finesub_bootstrap.downloader import DownloadPaused
from finesub_bootstrap.token_counter import TOKEN_COUNTER_VARIABLE


def token_counter_overrides(
    resolve: Callable[[], Path | None],
) -> dict[str, str]:
    """Point the LLM layer at a local tokenizer binary, when there is one.

    Without one, token counting falls back to the `countTokens` endpoint: still
    free and still exact, but it needs a key and a network round trip, so a dry
    run stops being something you can do offline. Neither front end can let that
    matter enough to block a task, which is why resolving to None is allowed
    and the answer is then "no opinion" rather than an error.

    `resolve` is a callback, not a path, because resolving is not free: finding
    a system copy means *running* it to check that it counts, and the binary
    loads a vocabulary before it answers. An explicitly configured counter wins
    over anything we could find, so in that case nothing is resolved at all --
    otherwise every command would pay for a probe whose result is discarded.
    """

    if os.environ.get(TOKEN_COUNTER_VARIABLE):
        return {}
    counter = resolve()
    return {TOKEN_COUNTER_VARIABLE: str(counter)} if counter is not None else {}


def shared_environment_overrides(paths: AppPaths) -> dict[str, str]:
    """Point the pipeline at the shared personal-data directory.

    The CLI and the desktop launch the same pipeline against the same
    ``user-data`` tree, so they have to agree on where it is. The knowledge
    base especially: left to resolve itself it walks up from the worker's
    source directory and lands in ``app/versions/<version>/knowledge``, which
    the next app update replaces -- silently taking the knowledge base with it.

    User-facing location variables are only filled when absent. The internal
    agent location tuple always describes this launcher's resolved paths so a
    vendored worker cannot accidentally inherit stale facts from its parent.
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
    from finesub_bootstrap.locks import (
        AGENT_ACTIVITY_ROOT_VARIABLE,
        AGENT_CAPSULE_ROOT_VARIABLE,
        AGENT_IDENTITY_ANCHOR_VARIABLE,
        AGENT_LOCATOR_KIND_MANAGED,
        AGENT_LOCATOR_KIND_VARIABLE,
    )

    # Internal launch facts, not user preferences. In particular, vendored CLI
    # sources cannot infer a custom FINESUB_HOME from their site-packages path.
    overrides.update(
        {
            AGENT_CAPSULE_ROOT_VARIABLE: str(paths.agent_capsules),
            AGENT_ACTIVITY_ROOT_VARIABLE: str(paths.user_data),
            AGENT_IDENTITY_ANCHOR_VARIABLE: str(paths.user_data),
            AGENT_LOCATOR_KIND_VARIABLE: AGENT_LOCATOR_KIND_MANAGED,
        }
    )
    return overrides


# The activation swap is a directory rename, and Windows denies those while
# anything still holds a handle inside the tree -- an antivirus scanning the
# 2.8GB that was just written, a sync client, a shell sitting in the folder.
# Those windows are short and retrying costs nothing once the path is clear.
SWAP_ATTEMPTS = 8
SWAP_BACKOFF_SECONDS = 0.4
SWAP_BACKOFF_CAP_SECONDS = 2.0

#: How much of a failed install's output travels with the exception. Enough to
#: show a person what happened, and to tell a dead mirror from a full disk.
OUTPUT_TAIL_LINES = 50


#: Failures a mirror can plausibly be responsible for. Everything else -- a
#: full disk, a corrupt archive, a digest that did not match -- describes the
#: bytes or this machine, and a second host cannot help with either. Shared
#: with the model downloads, which ask the same question of the same evidence.
def _retryable_install_markers() -> tuple[str, ...]:
    from finesub_bootstrap.model_fetch import NETWORK_FAILURE_MARKERS

    return NETWORK_FAILURE_MARKERS


#: A digest that did not match is the one failure whose meaning depends on who
#: served the bytes. From the canonical lock it means the file is wrong and a
#: second attempt cannot help. From a mirror it means *that mirror* is stale or
#: corrupt -- and the official source is precisely the thing that fixes it.
_MIRROR_ONLY_MARKERS = ("hash mismatch", "hashes do not match", "checksum")


def _install_failure_text(error: BaseException) -> str:
    """What uv said, not what the exception is called.

    A failed subprocess raises `CalledProcessError`, whose message is only
    "returned non-zero exit status 1" -- matching on that alone would mean the
    fallback never fires for the failure it exists for.
    """

    return f"{error} {getattr(error, 'output', '') or ''}".lower()


def _is_retryable_install(error: BaseException) -> bool:
    """Whether a second attempt could plausibly do better."""

    if isinstance(error, (OSError, MemoryError)) and not isinstance(
        error, subprocess.SubprocessError
    ):
        return False
    text = _install_failure_text(error)
    return any(marker in text for marker in _retryable_install_markers())


def _is_retryable_from_mirror(error: BaseException) -> bool:
    """The same question, asked about an install that came from a mirror.

    Wider on purpose: everything the canonical lock would retry, plus a digest
    mismatch, because from a mirror that is not evidence of a bad file -- it is
    evidence of a bad mirror, and the canonical lock verifies the same digests
    when it retries.
    """

    if _is_retryable_install(error):
        return True
    if isinstance(error, (OSError, MemoryError)) and not isinstance(
        error, subprocess.SubprocessError
    ):
        return False
    text = _install_failure_text(error)
    return any(marker in text for marker in _MIRROR_ONLY_MARKERS)


def _apply_download_endpoints(environment: dict[str, str], data_root: Path) -> None:
    """Point model downloads at the configured entry point, if there is one.

    Resolving the region can touch the network, so it must never be what
    stops a worker from starting: with no verified mirror configured -- the
    shipped default -- this is a no-op either way.
    """

    try:
        from finesub_bootstrap import model_fetch
        from finesub_bootstrap.download_routes import resolve_region

        model_fetch.apply_hf_endpoint(
            environment,
            data_root=data_root,
            region=resolve_region(data_root).region,
        )
    except Exception:
        return


def _swap_failure_message(error: OSError) -> str:
    """Say who is likely holding the directory, and what to do about it.

    This is the last step of a multi-minute install and it surfaces verbatim in
    the UI, so the raw ``[WinError 5] Access is denied`` it replaces told the
    user nothing they could act on.
    """

    return (
        "无法启用新的 Python 运行环境：目标目录被占用。"
        "常见原因是杀毒软件或网盘同步正在扫描刚写入的文件，"
        "或有资源管理器/终端停在安装目录里。"
        "请关闭它们后重试；若安装目录位于网盘同步目录或网络盘，"
        f"请改装到本地普通目录。（{error}）"
    )


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
#
# Deliberately the product name and not the whole local label: the question here
# is patched-versus-stock, and stock CTranslate2 never carries a `+finesub...`
# label. Pinning the version too would mean editing this line on every wheel
# rebuild, and getting it wrong reads as "the stock build is installed" against a
# perfectly good runtime.
REQUIRED_CTRANSLATE2_LOCAL_LABEL = "finesub"

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


def _holding_install_lock(
    lock_path: Path,
    *,
    log: LogCallback | None,
    should_pause: PauseCheck | None,
):
    """Serialize runtime installation across FineSub processes.

    The desktop app and the CLI shell can both decide the runtime needs
    (re)building; the staging swap must not run twice concurrently.
    """

    return holding_lock(
        lock_path,
        waiting_message=(
            "Another FineSub process is installing the runtime; "
            "waiting for it to finish"
        ),
        log=log,
        should_pause=should_pause,
        on_pause=lambda: DownloadPaused(
            "Python environment installation paused"
        ),
    )


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
                    reuses_system_python=True,
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
        # A staging tree carrying this exact marker was already built and
        # validated by an attempt that only failed to swap it in. Finishing it
        # is one rename; rebuilding it is minutes and 2.8GB of unpacking.
        if self._staging_is_complete(staging):
            if stage is not None:
                stage("activating", "正在校验并启用 Python 环境")
            self._activate(staging, previous)
            return self.status()
        self._discard(staging)

        self._warn_if_cache_is_on_another_volume(log)
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
            self._install_dependencies(
                uv,
                staging_python,
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
            self._prune_cache(uv, environment, log=log)
        except Exception:
            # A tree that is built and validated has nothing left but the swap,
            # and the next attempt retries exactly that -- keep it. Anything
            # that failed earlier is partial and must not be reused.
            if not self._staging_is_complete(staging):
                self._discard(staging)
            raise
        return self.status()

    def _install_dependencies(
        self,
        uv: Path,
        staging_python: Path,
        environment: dict[str, str],
        *,
        log: LogCallback | None,
        should_pause: PauseCheck | None,
    ) -> None:
        """Install from the regional lock, falling back to the canonical one.

        The whole install is retried rather than individual files: uv resolves
        and installs a lock as a unit, and a half-mirrored environment is not
        something either lock describes.

        Only a network failure earns the retry. A disk error, an unpack error
        or a hash mismatch means the bytes were wrong or could not be stored,
        and asking a second host to serve them again neither diagnoses that nor
        fixes it -- it just spends another few GB before failing the same way.
        """

        def install_from(lock: Path) -> None:
            self._run(
                [
                    str(uv),
                    "pip",
                    "install",
                    "--python",
                    str(staging_python),
                    "--requirement",
                    str(lock),
                ],
                environment,
                log=log,
                should_pause=should_pause,
            )

        regional = self.regional_lock()
        if regional is None:
            install_from(self.runtime_lock)
            return
        try:
            install_from(regional)
        except Exception as error:
            if isinstance(error, DownloadPaused) or not _is_retryable_from_mirror(
                error
            ):
                raise
            if log is not None:
                log(f"镜像安装失败，改用官方源重试：{error}")
            from finesub_bootstrap.download_routes import record_failure

            record_failure(self.paths.data_root, "pypi")
            # uv's own cache is kept: whatever it already verified is reusable
            # regardless of which mirror served it.
            install_from(self.runtime_lock)
            return
        try:
            from finesub_bootstrap.download_routes import record_success

            record_success(self.paths.data_root, "pypi")
        except Exception:
            pass

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
        # Set here rather than in the desktop's prefetch: this is the one path
        # both front ends go through, and the CLI has no prefetch at all --
        # its models are downloaded lazily inside the run.
        _apply_download_endpoints(environment, self.paths.data_root)
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
        # Kept so a failure can say what went wrong. Without it the only
        # evidence is "exit status 1", which is neither a diagnosis for the
        # user nor enough to tell a dead mirror from a full disk.
        tail: deque[str] = deque(maxlen=OUTPUT_TAIL_LINES)

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
            self._drain_logs(lines, log, tail)
            if should_pause is not None and should_pause():
                self.process_terminator(process)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                reader.join(timeout=1)
                self._drain_logs(lines, log, tail)
                raise DownloadPaused("Python environment installation paused")
            time.sleep(0.1)
        reader.join(timeout=1)
        self._drain_logs(lines, log, tail)
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode, command, output="\n".join(tail)
            )

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
        healthy, detail = self._repair_base_interpreter(python_executable)
        if not healthy:
            return False, detail
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

    def _prune_cache(
        self,
        uv: Path,
        environment: dict[str, str],
        *,
        log: LogCallback | None,
    ) -> None:
        """Drop unreachable cache objects after a successful activation.

        Hygiene, not reclamation: because uv hardlinks wheels out of the cache
        into the environment, deleting a cache entry that an environment still
        references frees nothing. Space comes back when the last reference goes
        -- which is why `uv cache clean <package>` is never run automatically:
        it evicts the version currently in use, freeing nothing while making
        the next rebuild download it again.

        Routed through `command_runner` like every other uv invocation. Calling
        `subprocess.run` directly here punched a hole in the seam: tests hand
        this class a stub `uv.exe` that is a couple of bytes of text, so the
        real activation path launched a non-PE file on every such test. On
        Windows that lands in the csrss 16-bit rejection path (one
        `Wow64 Emulation Layer` 1109 event per attempt), which is not something
        a unit test should be exercising a few hundred times per run.
        """

        try:
            self.command_runner(
                [str(uv), "cache", "prune"],
                env=environment,
                capture_output=True,
                timeout=120,
                creationflags=(
                    getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except (OSError, subprocess.SubprocessError) as error:
            if log is not None:
                log(f"Cache prune skipped: {error}")

    def _warn_if_cache_is_on_another_volume(
        self, log: LogCallback | None
    ) -> None:
        """Say so when the environment cannot share storage with the cache.

        uv hardlinks wheels from its cache into the environment when both sit
        on one volume, so the ~5GB they appear to occupy is really one copy.
        Split them across drives and the link becomes a copy: the same install
        silently costs about 5GB more, which is the opposite of what someone
        moving data to another disk is trying to achieve.
        """

        if log is None or os.name != "nt":
            return
        cache_volume = (self.paths.cache / "uv").anchor.lower()
        runtime_volume = self.runtime_root.anchor.lower()
        if not cache_volume or cache_volume == runtime_volume:
            return
        log(
            "警告：下载缓存与运行环境不在同一磁盘"
            f"（{cache_volume} / {runtime_volume}），两者无法共享存储，"
            "预计多占用约 5 GB。若目的是给系统盘腾空间，"
            "建议整个 FineSub 文件夹一起搬走。"
        )

    def _repair_base_interpreter(
        self, python_executable: Path
    ) -> tuple[bool, str]:
        """Keep a moved environment usable by fixing `pyvenv.cfg`'s `home`.

        A virtual environment carries no standard library: `Scripts\\python.exe`
        finds it through the base interpreter recorded in `pyvenv.cfg`. Moving
        the environment is otherwise harmless -- `sys.prefix` follows the
        executable -- so the only thing a hand-moved installation breaks is that
        one absolute path, and only when the base lives inside the tree that
        moved (an interpreter uv installed under `python-builds`, rather than a
        system Python that stayed put).

        Rewriting one line rescues several GB. This runs inside the polling
        health check, so it reads first and writes only when actually broken.
        """

        config = python_executable.parent.parent / "pyvenv.cfg"
        try:
            lines = config.read_text(encoding="utf-8").splitlines()
        except OSError:
            return True, ""  # No pyvenv.cfg: not a venv we manage.
        recorded: str | None = None
        for line in lines:
            key, separator, value = line.partition("=")
            if separator and key.strip() == "home":
                recorded = value.strip()
                break
        if recorded is None or Path(recorded).is_dir():
            return True, ""
        replacement = self._relocated_base(Path(recorded))
        if replacement is None:
            return False, (
                "Python 运行环境的基础解释器已不存在，需要重新安装运行环境。"
            )
        rewritten = [
            f"home = {replacement}"
            if line.partition("=")[0].strip() == "home"
            else line
            for line in lines
        ]
        write_atomic(config, "\n".join(rewritten) + "\n")
        return True, ""

    def _relocated_base(self, recorded: Path) -> Path | None:
        """The same uv-managed interpreter under this installation, if present."""

        builds = self.paths.runtime / "python-builds"
        candidate = builds / recorded.name
        return candidate if candidate.is_dir() else None

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
        tail: deque[str] | None = None,
    ) -> None:
        while True:
            try:
                line = lines.get_nowait()
            except queue.Empty:
                return
            if not line:
                continue
            if log is not None:
                log(line)
            if tail is not None:
                tail.append(line)

    def _activate(self, staging: Path, previous: Path) -> None:
        self._discard(previous)
        # lexists, not Path.exists: the latter follows links, so a stale
        # junction reads as absent while still owning the name, and it reports
        # anything it cannot stat as absent too. Either way the destination
        # would still be taken when the swap runs, and a rename onto an
        # existing directory is exactly the access-denied case below.
        had_active = os.path.lexists(self.runtime_root)
        if had_active:
            self._swap(self.runtime_root, previous)
        try:
            self._swap(staging, self.runtime_root)
        except OSError as error:
            if had_active and not os.path.lexists(self.runtime_root):
                try:
                    self._swap(previous, self.runtime_root)
                except OSError:
                    pass
            raise RuntimeError(_swap_failure_message(error)) from error
        self._discard(previous)

    @staticmethod
    def _swap(source: Path, destination: Path) -> None:
        """Rename a directory, waiting out whoever is still holding it."""

        for attempt in range(1, SWAP_ATTEMPTS + 1):
            try:
                os.replace(source, destination)
                return
            except OSError:
                if attempt == SWAP_ATTEMPTS:
                    raise
                time.sleep(
                    min(
                        SWAP_BACKOFF_SECONDS * attempt,
                        SWAP_BACKOFF_CAP_SECONDS,
                    )
                )

    _discard = staticmethod(remove_tree)

    def _staging_is_complete(self, staging: Path) -> bool:
        """Whether a leftover staging tree is a finished, validated runtime."""

        marker = staging / "finesub-runtime.json"
        try:
            expected = self._marker()
            if json.loads(marker.read_text(encoding="utf-8")) != expected:
                return False
        except (OSError, ValueError):
            return False
        healthy, _ = self._python_health(staging / "Scripts" / "python.exe")
        return healthy

    def _marker(self) -> dict[str, object]:
        if not self.runtime_lock.is_file():
            raise FileNotFoundError(
                "FineSub desktop runtime lock was not found: "
                f"{self.runtime_lock}"
            )
        # Always the canonical lock, never the one this machine happened to
        # install from. The regional lock differs only in which mirror serves
        # each identical, identically-hashed file, so hashing it here would
        # make moving between regions look like "the dependencies changed" and
        # rebuild a 5 GB environment that is already correct.
        return {
            "schemaVersion": self.schema_version,
            "pythonVersion": self.python_version,
            "runtimeLockHash": hashlib.sha256(
                self.runtime_lock.read_bytes()
            ).hexdigest(),
        }

    def regional_lock(self) -> Path | None:
        """The lock for this machine's region, if one shipped and applies.

        Absent -- the shipped default until a mirror passes the release drill
        -- every install uses the canonical lock, which is the official source.
        """

        # The file first, and the region only if there is one. Resolving can
        # cost a network round trip, and asking "which country is this?" before
        # knowing whether any answer would change anything spends it on every
        # install of a release that ships no regional lock at all.
        candidate = self.runtime_lock.with_name(
            self.runtime_lock.name.replace(".toml", ".cn.toml")
        )
        if not candidate.is_file():
            return None
        try:
            from finesub_bootstrap.download_routes import is_degraded, resolve_region

            if is_degraded(self.paths.data_root, "pypi"):
                return None
            if resolve_region(self.paths.data_root).region != "cn":
                return None
        except Exception:
            return None
        return candidate

    def _status(
        self,
        state: str,
        detail: str = "",
        *,
        reuses_system_python: bool = False,
    ) -> ResourceStatus:
        return ResourceStatus(
            id="uv",
            version=f"Python {self.python_version}",
            state=state,
            detail=detail,
            reuses_system_python=reuses_system_python,
        )
