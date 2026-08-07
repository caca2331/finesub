"""The `finesub` command line, shared by every front end that embeds it.

Two of them do: the published CLI wheel, which provisions its own managed root,
and the desktop package, which drives the install it sits in. Both hand the
pipeline the same environment -- knowledge base, `.env`, model caches, limiter
state -- so that logic lives here rather than in either front end. What differs
is only where the pieces come from, which is what the caller supplies when it
builds a `Shell`.

Everything that is not a shell subcommand is forwarded verbatim to the pipeline
inside the managed runtime: the shell must not have an opinion about pipeline
flags.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import shutil
import subprocess
import sys

from finesub_bootstrap.capabilities import capabilities_from_arguments
from finesub_bootstrap.environment import (
    RuntimeEnvironment,
    shared_environment_overrides,
)
from finesub_bootstrap.fsops import move_store, remove_tree
from finesub_bootstrap.locks import try_lock
from finesub_bootstrap import secrets
from finesub_bootstrap.migrations import apply_pending
from finesub_bootstrap.paths import (
    BIG_DATA_NAMES,
    AppPaths,
    ensure_store,
    is_store,
    record_big_data,
)
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap import system_tools

SUBCOMMANDS = ("setup", "doctor", "keys", "uninstall", "relocate", "batch")

_CAPABILITY_REASONS = {
    "git": "the knowledge base is a git repository",
    "yt-dlp": "URL input needs a downloader",
}


def system_tool(resource_id: str):
    """A usable system copy of a managed tool, or None.

    Reusing what the machine already has keeps `finesub setup` from spending
    146MB on a second ffmpeg. yt-dlp is never resolved this way: the pipeline
    imports it from the managed interpreter, which cannot see the user's
    site-packages.
    """

    finder = {
        "ffmpeg": system_tools.find_system_ffmpeg,
        "git": system_tools.find_system_git,
    }.get(resource_id)
    return finder() if finder is not None else None


class Shell:
    """Subcommands over one managed FineSub install."""

    def __init__(
        self,
        *,
        paths: AppPaths,
        resources: ResourceManager,
        runtime: RuntimeEnvironment,
        can_provision: bool = True,
    ) -> None:
        self.paths = paths
        self.resources = resources
        self.runtime = runtime
        # A front end that runs *on* the managed runtime cannot build it: the
        # desktop package's entry point is started by the very interpreter the
        # install would replace, so it reports what is missing and stops.
        self.can_provision = can_provision

    def dispatch(self, arguments: Sequence[str]) -> int:
        command, *rest = arguments
        if command != "uninstall":
            # Cheap (one small JSON read) and needed before anything touches
            # personal data -- which is everything except tearing it down.
            apply_pending(self.paths, log=_print_log)
        if command == "setup":
            return self.setup()
        if command == "doctor":
            return self.doctor()
        if command == "keys":
            return self.keys(rest)
        if command == "uninstall":
            return self.uninstall(rest)
        if command == "relocate":
            return self.relocate(rest)
        if command == "batch":
            return self.run_in_runtime("asr_playground.batch", rest)
        return self.run_in_runtime("asr_playground.pipeline", list(arguments))

    # -- subcommands ----------------------------------------------------

    def setup(self) -> int:
        self.ensure_ready()
        print(f"FineSub runtime is ready under {self.paths.root}")
        return 0

    def doctor(self) -> int:
        # The diagnostic must not trust the instant filesystem check: it exists
        # for exactly the damage inside packages that check cannot see. Takes a
        # few seconds (spawns the runtime Python and imports the whole stack).
        runtime_status = self.runtime.status(force_probe=True)
        try:
            uv_location = str(self.runtime.uv_executable())
        except Exception as error:  # A diagnostic must not crash on its subject.
            uv_location = f"missing ({error})"
        print(f"home         {self.paths.root}")
        print(f"user-data    {self.paths.user_data}")
        print(f"data         {self.paths.big_data}{_relocated_note(self.paths)}")
        for label, directory in (
            ("models", self.paths.models),
            ("cache", self.paths.cache),
            ("tasks", self.paths.tasks),
        ):
            print(f"{label:<12} {directory} {_directory_size(directory)}")
        print(
            f"runtime      {runtime_status.state}"
            + (f" ({runtime_status.detail})" if runtime_status.detail else "")
        )
        print(f"uv           {uv_location}")
        print(f"env-keys     {self._env_keys_report()}")
        for resource_id, note in (
            ("ffmpeg", ""),
            ("git", "installed on demand: knowledge updates"),
            ("yt-dlp", "installed on demand: URL input"),
        ):
            print(f"{resource_id:<12} {self._tool_report(resource_id, note)}")
        # Only ffmpeg and the runtime gate an ordinary run; the on-demand tools
        # are reported for diagnosis, not counted as failures.
        ready = (
            runtime_status.state == "ready"
            and self._tool_state("ffmpeg") == "ready"
        )
        if not ready:
            print(f"\n{self._provisioning_hint()}")
        return 0 if ready else 1

    def _env_keys_report(self) -> str:
        status = secrets.env_status(self.paths.user_data / ".env")
        if not status:
            return "none"
        counts = {state: 0 for state in ("protected", "plaintext", "unreadable")}
        for state in status.values():
            counts[state] = counts.get(state, 0) + 1
        return " / ".join(f"{state} ({count})" for state, count in counts.items())

    def keys(self, arguments: Sequence[str]) -> int:
        """Show configured API keys: masked by default, plaintext on request.

        Masked output survives screenshots and screen shares; `--reveal` goes
        to stdout only (never argv, never a file the user forgets to delete),
        in `NAME=value` form so it can be pasted straight into another `.env`.
        `--out FILE` exists for the machine-transfer flow, with a loud warning.
        """

        reveal = False
        out_file: Path | None = None
        rest = list(arguments)
        while rest:
            argument = rest.pop(0)
            if argument == "--reveal":
                reveal = True
            elif argument == "--out":
                if not rest:
                    print("--out needs a file path", file=sys.stderr)
                    return 2
                out_file = Path(rest.pop(0))
            else:
                print(f"Unknown keys option: {argument}", file=sys.stderr)
                return 2

        env_path = self.paths.user_data / ".env"
        values = secrets.export_env_file(env_path)
        status = secrets.env_status(env_path)
        if not status:
            print("尚未配置任何 API key。")
            return 0
        unreadable = [name for name in status if name not in values]
        if unreadable:
            print(
                f"无法在本机解密：{', '.join(unreadable)}（绑定的是原机器的"
                "Windows 账户；请在原机器上导出）",
                file=sys.stderr,
            )

        if out_file is not None:
            lines = [f"{name}={values[name]}" for name in status if name in values]
            out_file.write_text(
                "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
            )
            print(
                f"明文密钥已写入 {out_file}。它不受任何保护——"
                "用完请立即删除。",
                file=sys.stderr,
            )
            return 0

        for name in status:
            if name not in values:
                continue
            if reveal:
                print(f"{name}={values[name]}")
                continue
            entries = secrets.iter_entries(values[name])
            if not entries:
                print(f"{name:<13} (空)")
                continue
            shown = "  ".join(
                f"{label}={secrets.masked(key)}" if label else secrets.masked(key)
                for label, key in entries
            )
            note = "" if status[name] == "protected" else "  [明文]"
            print(f"{name:<13} {shown}{note}")
        if not reveal:
            print("\n完整明文：finesub keys --reveal（换机/重装前先导出）")
        return 0

    def uninstall(self, arguments: Sequence[str]) -> int:
        """Remove this installation, in three separately-decided pieces.

        Sorted by whether the data can be recreated rather than by where it
        sits: the runtime, models and downloads are rebuildable and go by
        default; finished subtitles and personal data are not, and go only when
        asked. The rebuildable half is *also* kept by default once it has been
        pointed somewhere else, because then another installation is probably
        reading it -- leaving a few GB behind costs disk, deleting someone
        else's copy costs them a download.
        """

        known = {"--purge-user-data", "--purge-tasks", "--keep-big-data", "--purge-big-data"}
        unknown = [argument for argument in arguments if argument not in known]
        if unknown:
            print(f"Unknown uninstall options: {unknown}", file=sys.stderr)
            return 2
        shared_store = self.paths.big_data != self.paths.root
        purge_big_data = (
            "--purge-big-data" in arguments
            if "--purge-big-data" in arguments or "--keep-big-data" in arguments
            else not shared_store
        )
        targets: list[Path] = [self.paths.runtime]
        if purge_big_data:
            targets += [self.paths.models, self.paths.cache]
        if "--purge-tasks" in arguments:
            targets.append(self.paths.tasks)
        if "--purge-user-data" in arguments:
            targets.append(self.paths.user_data)
        failures: list[str] = []
        for target in targets:
            if not os.path.lexists(target):
                continue
            try:
                remove_tree(target)
                print(f"removed {target}")
            except OSError as error:
                failures.append(f"{target}: {error}")
        for directory in (self.paths.big_data, self.paths.root):
            try:
                directory.rmdir()
            except OSError:
                pass  # Not empty, or shared with another installation.
        if failures:
            print(
                "Some paths could not be removed (close running FineSub "
                "processes and retry):",
                file=sys.stderr,
            )
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1
        if not purge_big_data:
            print(
                f"Kept models and downloads at {self.paths.big_data} "
                "(shared with other FineSub installations; "
                "pass --purge-big-data to remove them)."
            )
        if "--purge-tasks" not in arguments and os.path.lexists(self.paths.tasks):
            print(
                f"Kept finished subtitles at {self.paths.tasks} "
                "(pass --purge-tasks to remove them)."
            )
        if "--purge-user-data" not in arguments:
            print(
                f"Kept personal data at {self.paths.user_data} "
                "(pass --purge-user-data to remove it)."
            )
        return 0

    def relocate(self, arguments: Sequence[str]) -> int:
        """Move models, downloads and finished subtitles to another directory.

        Only the big-data root moves. The runtime stays with the installation
        on purpose: it is bound to this version, and uv hardlinks its packages
        out of the download cache, so putting the two on different drives turns
        one 5GB copy into two.
        """

        if "--show" in arguments or not arguments:
            print(f"data     {self.paths.big_data}{_relocated_note(self.paths)}")
            for label, directory in (
                ("models", self.paths.models),
                ("cache", self.paths.cache),
                ("tasks", self.paths.tasks),
            ):
                print(f"{label:<8} {directory} {_directory_size(directory)}")
            print(f"runtime  {self.paths.runtime} (always beside the app)")
            return 0
        destination = (
            self.paths.root
            if "--reset" in arguments
            else Path(arguments[0]).expanduser()
        )
        force = "--force" in arguments
        try:
            destination = self._checked_destination(destination, force=force)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        if destination == self.paths.big_data:
            print(f"Already there: {destination}")
            return 0
        if not self._nothing_is_running():
            print(
                "FineSub 正在运行（有任务或安装在进行），请先等它结束再搬。",
                file=sys.stderr,
            )
            return 1
        if destination.anchor.lower() != self.paths.runtime.anchor.lower():
            print(
                f"注意：{destination} 与运行环境（{self.paths.runtime}）不在同一磁盘，"
                "缓存与运行环境将无法共享存储，预计多占用约 5 GB。"
                "若目的是给系统盘腾空间，建议整个 FineSub 文件夹一起搬。",
                file=sys.stderr,
            )
        relocated = self.paths.with_big_data(destination)
        # Mark the destination a store before anything lands in it, so a crash
        # part-way leaves a directory the next start recognises rather than a
        # pile of unattributed files.
        ensure_store(relocated, log=_print_log)
        moved, leftovers = move_store(
            self.paths.big_data, destination, BIG_DATA_NAMES
        )
        # Record before releasing the sources: interrupted here, the worst case
        # is a duplicate copy left on disk, not data the next start cannot find.
        record_big_data(self.paths.data_root, destination)
        for leftover in leftovers:
            remove_tree(leftover)
        self.paths = relocated
        print(
            f"moved {', '.join(moved)} to {destination}"
            if moved
            else f"registered {destination}"
        )
        return 0

    def _checked_destination(self, destination: Path, *, force: bool) -> Path:
        resolved = destination.expanduser().resolve()
        if not resolved.is_absolute():
            raise ValueError(f"需要绝对路径：{destination}")
        if resolved != self.paths.root and _is_within(resolved, self.paths.root):
            raise ValueError(f"不能放在安装目录里面：{resolved}")
        if _is_within(resolved, self.paths.user_data):
            raise ValueError(f"不能放在个人数据目录里面：{resolved}")
        if resolved != self.paths.root and _is_within(self.paths.runtime, resolved):
            # The install root is exempt: it is the default location, and the
            # runtime lives inside it by definition -- which is exactly what
            # `--reset` asks for.
            raise ValueError(f"不能把运行环境包进去：{resolved}")
        if resolved.exists() and any(resolved.iterdir()) and not is_store(resolved):
            raise ValueError(
                f"目标目录已有内容且不是 FineSub 数据目录：{resolved}"
            )
        return resolved

    def _nothing_is_running(self) -> bool:
        return try_lock(self.paths.tasks / ".active.lock") and try_lock(
            self.paths.runtime / ".install.lock"
        )

    def run_in_runtime(self, module: str, arguments: Sequence[str]) -> int:
        arguments = list(arguments)
        self.ensure_ready()
        self._ensure_capabilities(arguments)
        context = self.runtime.worker_context(
            ffmpeg_bin=self.tool_directory("ffmpeg", "ffmpeg.exe"),
            extra_env=shared_environment_overrides(self.paths),
            extra_path_dirs=self._git_path_dirs(),
            extra_python_path=self._yt_dlp_python_path(),
        )
        environment = os.environ.copy()
        environment.update(context.environment)
        # Unlike the desktop worker, the shell keeps the user's working
        # directory: relative input/output paths belong to the caller.
        return subprocess.call(
            [str(context.python_executable), "-m", module, *arguments],
            env=environment,
        )

    # -- provisioning ---------------------------------------------------

    def ensure_ready(self) -> None:
        if os.name != "nt":
            raise SystemExit(
                "The FineSub managed runtime currently supports Windows x64 only."
            )
        # Before anything is stored, not while resolving: that is what keeps the
        # recorded location describing a directory that exists and that we put
        # data in, and it leaves a user who moved their store room to
        # re-register it before we start downloading a second copy.
        ensure_store(self.paths, log=_print_log)
        missing = self._missing_essentials()
        if not missing:
            return
        if not self.can_provision:
            raise SystemExit(
                f"{', '.join(missing)} is not ready. {self._provisioning_hint()}"
            )
        self._ensure_resource("ffmpeg", "every run decodes media")
        if self.runtime.status().state != "ready":
            print(
                "Setting up the FineSub AI runtime "
                "(the first run downloads several GB).",
                file=sys.stderr,
            )
            self.runtime.install(stage=_print_stage, log=_print_log)

    def _missing_essentials(self) -> list[str]:
        missing = []
        if self._tool_state("ffmpeg") != "ready":
            missing.append("ffmpeg")
        if self.runtime.status().state != "ready":
            missing.append("the Python runtime")
        return missing

    def _provisioning_hint(self) -> str:
        if self.can_provision:
            return "Run `finesub setup` to provision what is missing."
        # The desktop package installs resources from its own UI, which is also
        # where a failed install reports why -- sending the user to a command
        # that cannot provision would be a dead end.
        return (
            "Open FineSub Desktop and finish the resource setup there "
            "(资源 panel), then run this again."
        )

    def _ensure_resource(self, resource_id: str, reason: str) -> None:
        if system_tool(resource_id) is not None:
            return
        if self.resources.status(resource_id).state == "ready":
            return
        if not self.can_provision:
            raise SystemExit(
                f"{resource_id} is missing ({reason}). {self._provisioning_hint()}"
            )
        print(
            f"{resource_id} is missing ({reason}); downloading it now.",
            file=sys.stderr,
        )
        self.resources.install(resource_id, _print_progress, stage=_print_stage)
        print(file=sys.stderr)

    def _ensure_capabilities(self, arguments: Sequence[str]) -> None:
        """Install the tools this particular command turns out to need."""

        for resource_id in capabilities_from_arguments(list(arguments)):
            self._ensure_resource(resource_id, _CAPABILITY_REASONS[resource_id])

    # -- managed tools --------------------------------------------------

    def tool_directory(self, resource_id: str, filename: str):
        """Directory to put on PATH, preferring a system copy."""

        found = system_tool(resource_id)
        if found is not None:
            return found.directory
        active = self.resources.active_file(resource_id, filename)
        return active.parent if active is not None else None

    def _git_path_dirs(self) -> list:
        # A system git is already on PATH; only a managed one needs injecting.
        if system_tool("git") is not None:
            return []
        directory = self.tool_directory("git", "git.exe")
        return [directory] if directory is not None else []

    def _yt_dlp_python_path(self) -> list:
        # Imported, not executed, so it joins PYTHONPATH rather than PATH.
        if self.resources.active_version("yt-dlp") is None:
            return []
        return [self.resources.install_path("yt-dlp")]

    def _tool_state(self, resource_id: str) -> str:
        if system_tool(resource_id) is not None:
            return "ready"
        return self.resources.status(resource_id).state

    def _tool_report(self, resource_id: str, note: str) -> str:
        found = system_tool(resource_id)
        if found is not None:
            return f"ready (system: {found.path})"
        state = self.resources.status(resource_id).state
        return f"{state} ({note})" if note else state


def resource_specs(manifest: dict, *, exclude: Sequence[str] = ()):
    """Resource specs from a runtime manifest, minus the ones a front end owns."""

    from finesub_bootstrap.models import ResourceSpec

    return [
        ResourceSpec.model_validate(resource)
        for resource in manifest["resources"]
        if resource["id"] not in exclude
    ]


def package_shell(root: Path) -> Shell:
    """A shell over the desktop package rooted at ``root``.

    Same install the app drives -- same runtime, models, knowledge base and
    settings -- reached without the window. Provisioning stays with the app:
    this runs *on* the managed interpreter, so it cannot be the thing that
    installs or replaces it.
    """

    import json

    from finesub_bootstrap.paths import load_app_paths

    source = application_source(root)
    paths = load_app_paths(root)
    manifest = json.loads(
        (source / "desktop" / "resources" / "runtime-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    resources = ResourceManager(paths, resource_specs(manifest))

    def managed_uv() -> Path:
        executable = resources.active_file("uv", "uv.exe")
        if executable is None:
            raise FileNotFoundError("uv is not installed")
        return executable

    return Shell(
        paths=paths,
        resources=resources,
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=source,
            runtime_lock=source / "desktop" / "runtime" / "pylock.win-py312.toml",
            uv_executable=managed_uv,
        ),
        can_provision=False,
    )


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.relative_to(parent)
    except ValueError:
        return False
    return True


def _relocated_note(paths: AppPaths) -> str:
    return "" if paths.big_data == paths.root else "  (relocated)"


def _directory_size(directory: Path) -> str:
    if not directory.is_dir():
        return "(not created yet)"
    total = 0
    for current, _directories, names in os.walk(directory):
        for name in names:
            try:
                total += os.stat(os.path.join(current, name)).st_size
            except OSError:
                continue
    return f"{total / 2**30:.1f} GB" if total >= 2**30 else f"{total // 2**20} MB"


def _print_progress(progress) -> None:
    if progress.total <= 0:
        return
    percent = progress.downloaded * 100 // progress.total
    print(
        f"\r  {percent:3d}% ({progress.downloaded // 2**20} / "
        f"{progress.total // 2**20} MiB)",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _print_stage(_key: str, message: str) -> None:
    print(f"\n{message}", file=sys.stderr, flush=True)


def _print_log(line: str) -> None:
    print(f"  {line}", file=sys.stderr, flush=True)


def application_source(root: Path) -> Path:
    """The app snapshot a packaged install is currently running.

    Mirrors the launcher's own resolution so a command line started beside the
    executable runs exactly the version the app would.
    """

    import json

    pointer = root / "app" / "current.json"
    if pointer.is_file():
        try:
            current = json.loads(pointer.read_text(encoding="utf-8")).get("current")
        except (OSError, ValueError, AttributeError):
            current = None
        if isinstance(current, str) and current:
            source = (root / "app" / "versions" / current).resolve()
            if (source / "src" / "asr_playground" / "pipeline.py").is_file():
                return source
    if (root / "src" / "asr_playground" / "pipeline.py").is_file():
        return root
    raise FileNotFoundError(f"No FineSub application source under {root}")
