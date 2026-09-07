"""Entry point of the published `finesub` command.

The subcommands themselves live in `finesub_bootstrap.shell`. What this wheel
adds is where a managed install lives (`FINESUB_HOME`), where the sources come
from (`_vendor`) and where uv comes from (this wheel's own dependency).
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: What this front end says beyond the command list: where a managed install
#: puts things. The commands themselves come from the shared table, so this
#: help cannot drift from what dispatches.
ENVIRONMENT_HELP = """
Environment:
  FINESUB_HOME   Where the managed runtime and downloads live. Defaults to:
                 %LOCALAPPDATA%\\FineSub on Windows,
                 ~/Library/Application Support/FineSub on macOS,
                 ~/.finesub elsewhere. Settings, API keys and the knowledge
                 base live in the shared user-data directory under that root.
"""


def _source_root() -> Path:
    """Use the vendored snapshot when present; otherwise run directly from checkout."""

    if _VENDOR.exists():
        return _VENDOR
    repo_source = _REPO_ROOT / "src"
    if repo_source.exists():
        return repo_source
    return _VENDOR


def _bootstrap_root(source_root: Path) -> Path:
    vendored = source_root / "src" / "finesub_bootstrap"
    return vendored if vendored.is_dir() else source_root / "finesub_bootstrap"


def usage() -> str:
    _ensure_vendor_on_path()
    from finesub_bootstrap.shell import render_usage

    return render_usage() + ENVIRONMENT_HELP


def _ensure_vendor_on_path() -> None:
    source_root = _source_root()
    if source_root == _VENDOR and (source_root / "src").exists():
        vendored_sources = str(source_root / "src")
        if vendored_sources not in sys.path:
            sys.path.insert(0, vendored_sources)
        return
    repo_sources = str(source_root)
    if repo_sources not in sys.path:
        sys.path.insert(0, repo_sources)


def resolve_home() -> Path:
    configured = os.environ.get("FINESUB_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    # sys.platform is the authoritative runtime platform and is also the one
    # tests can safely simulate without changing pathlib's concrete path type.
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "FineSub"
    if os.name == "nt" or sys.platform.startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser().resolve() / "FineSub"
        return Path.home() / "FineSub"
    return Path.home() / ".finesub"


def _uv_executable() -> Path:
    from uv import find_uv_bin

    return Path(find_uv_bin())


def ask_big_data_dir(default_root: Path) -> Path | None:
    """Ask, once, which disk gets the models, cache and finished subtitles.

    Returning None means "use the default", which is also what every
    non-interactive case answers: a CI job, a piped installer or a redirected
    console must never be left waiting on a prompt nobody can see. Checking
    the stream rather than trusting `isatty` alone is deliberate --
    `irm ... | iex` runs with stdin attached to the pipeline.
    """

    if not sys.stdin or not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            f"FineSub 会把模型和缓存放在 {default_root}"
            "（要换位置：finesub relocate <目录>）。",
            file=sys.stderr,
        )
        return None
    print(
        "FineSub 将在这里保存模型、下载缓存和任务产物，建议预留至少 20 GB。\n"
        f"直接回车使用：{default_root}\n"
        "也可以输入其他绝对路径，例如 D:\\FineSub"
    )
    try:
        answer = input("大文件位置：").strip()
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
    return Path(answer) if answer else None


def _shell():
    from finesub_bootstrap.environment import RuntimeEnvironment
    from finesub_bootstrap.paths import load_app_paths
    from finesub_bootstrap.resources import ResourceManager, read_runtime_manifest
    from finesub_bootstrap.shell import Shell, resource_specs

    source_root = _source_root()
    paths = load_app_paths(resolve_home())
    # No path here on purpose: the manifest and the lock ship inside the
    # vendored `finesub_bootstrap`, which is the very package this line
    # imports, so naming them again would be a second copy of where they live.
    manifest = read_runtime_manifest()
    # uv comes from this wheel's own dependency; everything else in the
    # manifest is fetched here -- ffmpeg up front, git and yt-dlp only when a
    # run turns out to need them.
    bootstrap_root = _bootstrap_root(source_root)
    runtime_lock = bootstrap_root / "pylock.win-py312.toml"
    python = None
    if sys.platform == "darwin" and platform.machine().lower() == "arm64":
        runtime_lock = bootstrap_root / "pylock.macos-arm64-py312.toml"
    elif os.name != "nt" and not sys.platform.startswith("win"):
        # No published, accepted Linux runtime exists yet. Preserve the
        # development-shell behavior without pretending the Windows lock is a
        # supported Linux environment.
        python = Path(sys.executable)
    return Shell(
        paths=paths,
        ask_big_data_dir=ask_big_data_dir,
        resources=ResourceManager(
            paths, resource_specs(manifest, exclude=("uv",))
        ),
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=source_root,
            runtime_lock=runtime_lock,
            uv_executable=_uv_executable,
            development_python=python,
        ),
    )


def installed_version() -> str:
    """This wheel's version, from its own installed metadata.

    Only the published CLI *is* the `finesub` distribution -- a checkout runs
    the same sources with no distribution at all -- which is why the update
    check is wired here rather than in the shared `Shell`.
    """

    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("finesub")
    except PackageNotFoundError:
        return ""


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(usage(), end="")
        return 0 if arguments else 2
    _ensure_vendor_on_path()
    shell = _shell()
    check = _update_check(shell.paths, arguments[0])
    if check is not None:
        check.start()
    status = shell.dispatch(arguments)
    if arguments[0] == "uninstall" and status == 0:
        # Only this front end has a shell of its own to remove afterwards.
        print("Now remove the shell itself, e.g. `uv tool uninstall finesub`.")
    if check is not None and (notice := check.notice()):
        # After the command and on stderr: stdout carries pipeline output, and
        # a notice printed up front is a notice nobody reads.
        print(notice, file=sys.stderr)
    return status


def _update_check(paths, command: str):
    """The run's update check, or None when this run must not have one.

    Takes `AppPaths` rather than the shell: the notice is about where this
    install's shared data lives, not about what the shell can do.
    """

    from finesub_bootstrap.update_check import UpdateCheck, enabled

    current = installed_version()
    if not current:
        return None
    if not enabled(
        command=command,
        user_data=paths.user_data,
        isatty=sys.stderr.isatty(),
    ):
        return None
    return UpdateCheck(paths.data_root, current=current)


if __name__ == "__main__":
    raise SystemExit(main())
