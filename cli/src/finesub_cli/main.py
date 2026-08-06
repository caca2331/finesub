"""Entry point of the `finesub` command.

Dispatch is manual rather than argparse-driven because everything that is not
a shell subcommand (setup / doctor / uninstall / batch) is forwarded verbatim
to `asr_playground.pipeline` inside the managed runtime -- the shell must not
have an opinion about pipeline flags.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"

USAGE = """\
FineSub — local long-form audio to subtitles.

Usage:
  finesub <input> [pipeline options...]   Run the pipeline (asr-pipeline flags)
  finesub batch [batch options...]        Run the batch runner
  finesub setup                           Provision the runtime without running
  finesub doctor                          Show runtime status and paths
  finesub uninstall [--purge-user-data]   Remove the managed runtime/models
                                          (personal data only with the flag)

Environment:
  FINESUB_HOME   Managed-data root (default: %LOCALAPPDATA%\\FineSub). Point it
                 at a FineSub Desktop install directory to reuse its runtime
                 and models.
"""


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


def _services(home: Path):
    from finesub_bootstrap.environment import RuntimeEnvironment
    from finesub_bootstrap.models import ResourceSpec
    from finesub_bootstrap.paths import AppPaths
    from finesub_bootstrap.resources import ResourceManager

    paths = AppPaths.for_root(home)
    manifest = json.loads(
        (_VENDOR / "runtime-manifest.json").read_text(encoding="utf-8")
    )
    # uv comes from this wheel's own dependency; everything else in the
    # manifest is fetched here -- ffmpeg up front, git and yt-dlp only when a
    # run turns out to need them.
    specs = [
        ResourceSpec.model_validate(resource)
        for resource in manifest["resources"]
        if resource["id"] != "uv"
    ]
    resources = ResourceManager(paths, specs)
    runtime = RuntimeEnvironment(
        paths=paths,
        app_source=_VENDOR,
        runtime_lock=_VENDOR / "pylock.win-py312.toml",
        uv_executable=_uv_executable,
    )
    return paths, resources, runtime


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


def _system_tool(resource_id: str):
    """A usable system copy of a tool, or None.

    Reusing what the machine already has keeps `finesub setup` from spending
    146MB on a second ffmpeg. yt-dlp is never resolved this way: the pipeline
    imports it from the managed interpreter, which cannot see the user's
    site-packages.
    """

    from finesub_bootstrap import system_tools

    finder = {
        "ffmpeg": system_tools.find_system_ffmpeg,
        "git": system_tools.find_system_git,
    }.get(resource_id)
    return finder() if finder is not None else None


def _ensure_resource(resources, resource_id: str, reason: str) -> None:
    if _system_tool(resource_id) is not None:
        return
    if resources.status(resource_id).state == "ready":
        return
    print(f"{resource_id} is missing ({reason}); downloading it now.", file=sys.stderr)
    resources.install(resource_id, _print_progress, stage=_print_stage)
    print(file=sys.stderr)


def _ensure_capabilities(resources, arguments: list[str]) -> None:
    """Install the on-demand tools this particular command turns out to need."""

    from finesub_bootstrap.capabilities import capabilities_from_arguments

    reasons = {
        "git": "the knowledge base is a git repository",
        "yt-dlp": "URL input needs a downloader",
    }
    for resource_id in capabilities_from_arguments(arguments):
        _ensure_resource(resources, resource_id, reasons[resource_id])


def _tool_directory(resources, resource_id: str, filename: str):
    """Directory to put on PATH, preferring a system copy."""

    found = _system_tool(resource_id)
    if found is not None:
        return found.directory
    active = resources.active_file(resource_id, filename)
    return active.parent if active is not None else None


def _git_path_dirs(resources) -> list:
    # A system git is already on PATH; only a managed one needs injecting.
    if _system_tool("git") is not None:
        return []
    directory = _tool_directory(resources, "git", "git.exe")
    return [directory] if directory is not None else []


def _yt_dlp_python_path(resources) -> list:
    # Imported, not executed, so it joins PYTHONPATH rather than PATH.
    if resources.active_version("yt-dlp") is None:
        return []
    return [resources.install_path("yt-dlp")]


def _tool_state(resources, resource_id: str) -> str:
    if _system_tool(resource_id) is not None:
        return "ready"
    return resources.status(resource_id).state


def _tool_report(resources, resource_id: str, note: str) -> str:
    found = _system_tool(resource_id)
    if found is not None:
        return f"ready (system: {found.path})"
    state = resources.status(resource_id).state
    return f"{state} ({note})" if note else state


def _ensure_ready(paths, resources, runtime) -> None:
    if os.name != "nt":
        raise SystemExit(
            "The FineSub managed runtime currently supports Windows x64 only."
        )
    _ensure_resource(resources, "ffmpeg", "every run decodes media")
    if runtime.status().state != "ready":
        print(
            "Setting up the FineSub AI runtime "
            "(the first run downloads several GB).",
            file=sys.stderr,
        )
        runtime.install(stage=_print_stage, log=_print_log)


def _run_in_runtime(module: str, arguments: list[str]) -> int:
    from finesub_bootstrap.environment import shared_environment_overrides

    home = resolve_home()
    paths, resources, runtime = _services(home)
    _ensure_ready(paths, resources, runtime)
    _ensure_capabilities(resources, arguments)

    context = runtime.worker_context(
        ffmpeg_bin=_tool_directory(resources, "ffmpeg", "ffmpeg.exe"),
        extra_env=shared_environment_overrides(paths),
        extra_path_dirs=_git_path_dirs(resources),
        extra_python_path=_yt_dlp_python_path(resources),
    )
    environment = os.environ.copy()
    environment.update(context.environment)
    # Unlike the desktop worker, the CLI keeps the user's working directory:
    # relative input/output paths belong to the caller.
    return subprocess.call(
        [str(context.python_executable), "-m", module, *arguments],
        env=environment,
    )


def _setup() -> int:
    home = resolve_home()
    paths, resources, runtime = _services(home)
    _ensure_ready(paths, resources, runtime)
    print(f"FineSub runtime is ready under {paths.root}")
    return 0


def _doctor() -> int:
    home = resolve_home()
    paths, resources, runtime = _services(home)
    # The diagnostic must not trust the instant filesystem check: it exists
    # for exactly the damage inside packages that check cannot see. Takes a
    # few seconds (spawns the runtime Python and imports the whole stack).
    runtime_status = runtime.status(force_probe=True)
    try:
        uv_location = str(_uv_executable())
    except Exception as error:  # A diagnostic must not crash on what it checks.
        uv_location = f"missing ({error})"
    print(f"home         {paths.root}")
    print(f"user-data    {paths.user_data}")
    print(f"models       {paths.models}")
    print(
        f"runtime      {runtime_status.state}"
        + (f" ({runtime_status.detail})" if runtime_status.detail else "")
    )
    print(f"uv           {uv_location}")
    for resource_id, note in (
        ("ffmpeg", ""),
        ("git", "installed on demand: knowledge updates"),
        ("yt-dlp", "installed on demand: URL input"),
    ):
        print(f"{resource_id:<12} {_tool_report(resources, resource_id, note)}")
    # Only ffmpeg and the runtime gate an ordinary run; the on-demand tools are
    # reported for diagnosis, not counted as failures.
    ready = (
        runtime_status.state == "ready"
        and _tool_state(resources, "ffmpeg") == "ready"
    )
    if not ready:
        print("\nRun `finesub setup` to provision what is missing.")
    return 0 if ready else 1


def _uninstall(arguments: list[str]) -> int:
    purge_user_data = "--purge-user-data" in arguments
    unknown = [
        argument
        for argument in arguments
        if argument != "--purge-user-data"
    ]
    if unknown:
        print(f"Unknown uninstall options: {unknown}", file=sys.stderr)
        return 2
    home = resolve_home()
    doomed = ["runtime", "models", "cache"] + (
        ["user-data"] if purge_user_data else []
    )
    failures: list[str] = []
    for name in doomed:
        target = home / name
        if not target.exists():
            continue
        try:
            shutil.rmtree(target)
            print(f"removed {target}")
        except OSError as error:
            failures.append(f"{target}: {error}")
    try:
        home.rmdir()
    except OSError:
        pass  # Not empty (user-data kept, or shared with the desktop app).
    if failures:
        print(
            "Some paths could not be removed (close running FineSub "
            "processes and retry):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    if not purge_user_data:
        print(
            f"Kept personal data at {home / 'user-data'} "
            "(pass --purge-user-data to remove it)."
        )
    print("Now remove the shell itself, e.g. `uv tool uninstall finesub`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(USAGE, end="")
        return 0 if arguments else 2
    _ensure_vendor_on_path()
    command, rest = arguments[0], arguments[1:]
    if command == "setup":
        return _setup()
    if command == "doctor":
        return _doctor()
    if command == "uninstall":
        return _uninstall(rest)
    if command == "batch":
        return _run_in_runtime("asr_playground.batch", rest)
    return _run_in_runtime("asr_playground.pipeline", arguments)


if __name__ == "__main__":
    raise SystemExit(main())
