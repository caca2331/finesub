"""Entry point of the published `finesub` command.

The subcommands themselves live in `finesub_bootstrap.shell`, shared with the
desktop package's own command line so both hand the pipeline the same
environment. What this wheel adds is where a managed install lives
(`FINESUB_HOME`), where the sources come from (`_vendor`) and where uv comes
from (this wheel's own dependency).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"

#: What this front end says beyond the command list: where a managed install
#: puts things. The commands themselves come from the shared table, so the two
#: front ends cannot drift apart again.
ENVIRONMENT_HELP = """
Environment:
  FINESUB_HOME   Where the managed runtime and downloads live (default:
                 %LOCALAPPDATA%\\FineSub). Settings, API keys and the knowledge
                 base always live in %LOCALAPPDATA%\\FineSub\\user-data, shared
                 with FineSub Desktop.
"""


def usage() -> str:
    _ensure_vendor_on_path()
    from finesub_bootstrap.shell import CLI_FRONT_END, render_usage

    return render_usage(CLI_FRONT_END) + ENVIRONMENT_HELP


def _ensure_vendor_on_path() -> None:
    vendored_sources = str(_VENDOR / "src")
    if vendored_sources not in sys.path:
        sys.path.insert(0, vendored_sources)


def resolve_home() -> Path:
    configured = os.environ.get("FINESUB_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / "FineSub"
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
    from finesub_bootstrap.resources import ResourceManager
    from finesub_bootstrap.shell import Shell, resource_specs

    paths = load_app_paths(resolve_home())
    manifest = json.loads(
        (_VENDOR / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    # uv comes from this wheel's own dependency; everything else in the
    # manifest is fetched here -- ffmpeg up front, git and yt-dlp only when a
    # run turns out to need them.
    return Shell(
        paths=paths,
        ask_big_data_dir=ask_big_data_dir,
        resources=ResourceManager(
            paths, resource_specs(manifest, exclude=("uv",))
        ),
        runtime=RuntimeEnvironment(
            paths=paths,
            app_source=_VENDOR,
            runtime_lock=_VENDOR / "pylock.win-py312.toml",
            uv_executable=_uv_executable,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(usage(), end="")
        return 0 if arguments else 2
    _ensure_vendor_on_path()
    status = _shell().dispatch(arguments)
    if arguments[0] == "uninstall" and status == 0:
        # Only this front end has a shell of its own to remove afterwards.
        print("Now remove the shell itself, e.g. `uv tool uninstall finesub`.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
