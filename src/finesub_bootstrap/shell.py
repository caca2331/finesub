"""The `finesub` command line, shared by every front end that embeds it.

Two of them do: the published CLI wheel, which provisions its own managed root,
and the desktop package, which drives the install it sits in. Both hand the
pipeline the same environment -- knowledge base, `.env`, model caches, limiter
state -- so that logic lives here rather than in either front end. What differs
is only where the pieces come from, which is what the caller supplies when it
builds a `Shell`.

Everything that is not a shell subcommand is forwarded to the pipeline inside
the managed runtime, with exactly one exception: a run that names no output
gets one, under `tasks`, so that it is filed where both front ends look and a
rerun of the same source can continue where the last one stopped. A run that
*does* name an output is passed through untouched and happens there.

That one exception is the whole budget -- and it only ever adds `-o`, never
rewrites one. The shell has no opinion about any other flag and must not grow
one: knowing which of the pipeline's forty-odd options take a value is a table
that goes stale silently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from finesub_bootstrap import artifacts, task_index, task_output
from finesub_bootstrap.capabilities import (
    _run_properties,
    capabilities_from_arguments,
    preferred_capabilities_from_arguments,
)
from finesub_bootstrap.environment import (
    RuntimeEnvironment,
    shared_environment_overrides,
    token_counter_overrides,
)
from finesub_bootstrap.fsops import move_store, remove_tree
from finesub_bootstrap.locks import (
    AGENT_ACTIVITY_ROOT_VARIABLE,
    AGENT_CAPSULE_ROOT_VARIABLE,
    AGENT_IDENTITY_ANCHOR_VARIABLE,
    AGENT_LOCATOR_KIND_MANAGED,
    AGENT_LOCATOR_KIND_VARIABLE,
    LockUnavailable,
    activity_is_idle,
    active_lock_path,
    holding_activity,
    holding_activity_barrier,
    holding_lock,
    task_lock_path,
    task_workspace_lock_path,
    try_lock,
)
from finesub_bootstrap import secrets
from finesub_bootstrap.migrations import apply_pending
from finesub_bootstrap.paths import (
    BIG_DATA_NAMES,
    AppPaths,
    clear_migration_source,
    ensure_store,
    is_store,
    load_app_paths,
    looks_like_store,
    record_big_data,
    recorded_big_data,
)
from finesub_bootstrap.resources import ResourceManager
from finesub_bootstrap import system_tools

_CAPABILITY_REASONS = {
    "git": "the knowledge base is a git repository",
    "yt-dlp": "URL input needs a downloader",
    "tokcount": "the LLM layer counts tokens locally",
}


#: Front-end ids for `Command.shown_in`.
CLI_FRONT_END = "cli"
PACKAGE_FRONT_END = "package"

#: Column the description starts at in rendered help.
_HELP_COLUMN = 42


@dataclass(frozen=True, slots=True)
class Command:
    """One `finesub` subcommand: how to run it, and how to say so.

    Dispatch and help used to be three separate lists -- an if-chain here and a
    hand-written USAGE string in each front end -- so they drifted: the CLI's
    help never mentioned `agent-task`, and the package's never mentioned
    `keys`. One table, and `test_shell.py` holds each front end's help to it.
    """

    name: str
    #: `Shell` method that runs it. Empty when `runtime_module` handles it.
    method: str = ""
    #: Module `run_in_runtime` executes for the commands that are thin wrappers
    #: around something living in the managed runtime.
    runtime_module: str = ""
    #: Help rows as (left column, description). Multi-row entries carry their
    #: own left-column indentation so continuations line up under the command.
    help: tuple[tuple[str, str], ...] = ()
    #: Which front ends list it. Not a permission: the package shell dispatches
    #: every command, it just does not advertise the two that belong to the app
    #: (installing and removing an installation are the app's job).
    shown_in: frozenset[str] = frozenset({CLI_FRONT_END, PACKAGE_FRONT_END})
    #: `doctor` is the one subcommand with nothing to parse.
    takes_arguments: bool = True


COMMANDS: tuple[Command, ...] = (
    Command(
        name="batch",
        runtime_module="finesub.batch",
        help=(("finesub batch [batch options...]", "Run the batch runner"),),
    ),
    Command(
        name="setup",
        method="setup",
        help=(
            ("finesub setup [--dirs-only]", "Provision the runtime without running"),
            ("              [--data-dir DIR]", "(--dirs-only only settles where the"),
            ("", "big files go, downloading nothing)"),
        ),
        shown_in=frozenset({CLI_FRONT_END}),
    ),
    Command(
        name="doctor",
        method="doctor",
        help=(("finesub doctor", "Show runtime status and paths"),),
        takes_arguments=False,
    ),
    Command(
        name="agent-clean",
        method="agent_clean",
        help=(
            (
                "finesub agent-clean [--all-domains]",
                "Remove retained failed-agent evidence",
            ),
        ),
    ),
    Command(
        name="agent-ping",
        runtime_module="finesub.llm.agent.agent_ping",
        help=(
            (
                "finesub agent-ping [--tier TIER]",
                "Check whether each installed agent CLI",
            ),
            ("", "still answers (spends a little quota)"),
        ),
    ),
    Command(
        name="agent-task",
        runtime_module="finesub.llm.agent.agent_task_control",
        help=(
            (
                "finesub agent-task <subcommand>",
                "Inspect and steer durable agent tasks",
            ),
            ("", "(status, next-task, submit, ...)"),
        ),
    ),
    Command(
        name="keys",
        method="keys",
        help=(
            ("finesub keys [--reveal|--out FILE]", "Show API keys (masked by default);"),
            ("", "export plaintext before switching"),
            ("", "machines or reinstalling Windows"),
        ),
    ),
    Command(
        name="relocate",
        method="relocate",
        help=(
            (
                "finesub relocate [--show|<dir>|--reset]",
                "Move models/downloads/subtitles and",
            ),
            ("", "failed-agent evidence to"),
            ("", "another directory (the runtime stays"),
            ("", "beside the app)"),
        ),
    ),
    Command(
        name="uninstall",
        method="uninstall",
        help=(
            (
                "finesub uninstall [--purge-tasks]",
                "Remove the managed runtime, models",
            ),
            ("               [--purge-user-data]", "and downloads; finished subtitles and"),
            ("", "personal data only with the flags"),
        ),
        shown_in=frozenset({CLI_FRONT_END}),
    ),
)

COMMANDS_BY_NAME = {command.name: command for command in COMMANDS}

#: `agent-clean` runs on this interpreter instead of going through
#: `run_in_runtime` (see `Shell.agent_clean` for why), so its module is not a
#: `runtime_module` on the table above. Named here so the same guard covers it:
#: a moved module leaves the string pointing at nothing and nothing fails until
#: someone runs the command.
AGENT_CLEANUP_MODULE = "finesub.llm.agent.agent_cleanup"

#: The default branch: anything that is not a subcommand is a pipeline run.
#: Listed first in help, and deliberately not a `Command` -- there is no name
#: to dispatch on, and the comparison in `test_shell.py` is about commands.
_PIPELINE_HELP_ROW = (
    "finesub <input> [pipeline options...]",
    "Run the pipeline (asr-pipeline flags)",
)

USAGE_HEADER = "FineSub — local long-form audio to subtitles.\n\nUsage:\n"


def _help_row(left: str, right: str) -> str:
    return f"  {left.ljust(_HELP_COLUMN - 2)}{right}".rstrip()


def render_usage(front_end: str) -> str:
    """The Usage block for one front end, from the table above.

    The front end supplies whatever else its help says -- the CLI wheel
    documents `FINESUB_HOME`, the package says which installation it drives --
    because those are about the front end, not about the commands.
    """

    rows = [_help_row(*_PIPELINE_HELP_ROW)]
    for command in COMMANDS:
        if front_end not in command.shown_in:
            continue
        rows.extend(_help_row(left, right) for left, right in command.help)
    return USAGE_HEADER + "\n".join(rows) + "\n"


@dataclass(frozen=True, slots=True)
class _RunPlan:
    """Where one pipeline run writes, and where its result is wanted."""

    task_id: str
    source: str
    stage: str
    #: The command line as the pipeline will receive it. Identical to what the
    #: user typed whenever they named an output; `-o` is only ever *added*,
    #: never rewritten, so a run they placed happens where they placed it.
    arguments: list[str]
    #: The final SRT this run is configured to produce -- the user's path when
    #: they gave one, otherwise the task directory's. Every other artifact is
    #: derived from it, so it also says which folder fills up during the run.
    output: Path
    #: The name that actually affected this run. The pipeline ignores --name
    #: when -o is explicit, so history must not claim the ignored value ran.
    recorded_name: str = ""
    #: The index entry this run continues, if it continues one. Carried so its
    #: creation time survives when the CLI writes the effective configuration
    #: that a later desktop retry will replay.
    matched: dict | None = None


def _flag_value(arguments: Sequence[str], *names: str) -> str | None:
    """The value of `--flag value` or `--flag=value`, last occurrence winning.

    Reads only the flags it is asked about. The task-history adapter below
    mirrors the TaskRequest-compatible subset of the pipeline parser; its
    schema round-trip test is the guard against that table drifting silently.
    """

    found: str | None = None
    for index, argument in enumerate(arguments):
        if argument in names and index + 1 < len(arguments):
            found = arguments[index + 1]
        else:
            for name in names:
                if argument.startswith(f"{name}="):
                    found = argument.split("=", 1)[1]
    return found


def _boolean_flag(
    arguments: Sequence[str],
    *,
    enabled: tuple[str, ...],
    disabled: tuple[str, ...],
    default: bool,
) -> bool:
    value = default
    for argument in arguments:
        if argument in enabled:
            value = True
        elif argument in disabled:
            value = False
    return value


def _choice_flag(
    arguments: Sequence[str],
    name: str,
    choices: tuple[str, ...],
    default: str,
) -> str:
    value = _flag_value(arguments, name)
    return value if value in choices else default


def _number_flag(
    arguments: Sequence[str],
    name: str,
    *,
    default: float | None,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float | None:
    value = _flag_value(arguments, name)
    if value is None:
        return default
    try:
        number = float(value)
    except ValueError:
        return default
    if not math.isfinite(number):
        return default
    if strictly_positive and number <= 0:
        return default
    if minimum is not None and number < minimum:
        return default
    if maximum is not None and number > maximum:
        return default
    return number


def _integer_choice_flag(
    arguments: Sequence[str],
    name: str,
    choices: tuple[int, ...],
    default: int,
) -> int:
    value = _flag_value(arguments, name)
    try:
        number = int(value) if value is not None else default
    except ValueError:
        return default
    return number if number in choices else default


def system_tool(resource_id: str):
    """A usable system copy of a managed tool, or None.

    Reusing what the machine already has keeps `finesub setup` from spending
    140 MB on a second ffmpeg. yt-dlp is never resolved this way: the pipeline
    imports it from the managed interpreter, which cannot see the user's
    site-packages.
    """

    finder = {
        "ffmpeg": system_tools.find_system_ffmpeg,
        "git": system_tools.find_system_git,
        "tokcount": system_tools.find_system_token_counter,
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
        ask_big_data_dir: Callable[[Path], Path | None] | None = None,
    ) -> None:
        self.paths = paths
        self.resources = resources
        self.runtime = runtime
        # A front end that runs *on* the managed runtime cannot build it: the
        # desktop package's entry point is started by the very interpreter the
        # install would replace, so it reports what is missing and stops.
        self.can_provision = can_provision
        # Only the published CLI supplies this. The desktop app owns where its
        # own data goes, and its packaged command line shares this class -- so
        # a prompt written into `ensure_ready` would appear in a front end that
        # already answers the question elsewhere.
        self.ask_big_data_dir = ask_big_data_dir
        # Probed at most once each, the way `DesktopResourceService` already
        # does. Finding a system tool means *running* it, and the token counter
        # loads a vocabulary before it answers -- the reason its probe is
        # allowed 30 seconds. Four call sites ask about the same tool during
        # one run, and the answer cannot change inside a single command.
        self._system_tools: dict[str, object] = {}

    def dispatch(self, arguments: Sequence[str]) -> int:
        command, *rest = arguments
        if command != "uninstall":
            # Cheap (one small JSON read) and needed before anything touches
            # personal data -- which is everything except tearing it down.
            apply_pending(self.paths, log=_print_log)
        entry = COMMANDS_BY_NAME.get(command)
        if entry is None:
            return self.run_pipeline(list(arguments))
        if entry.runtime_module:
            return self.run_in_runtime(entry.runtime_module, rest)
        handler = getattr(self, entry.method)
        return handler(rest) if entry.takes_arguments else handler()

    # -- subcommands ----------------------------------------------------

    def agent_clean(self, arguments: Sequence[str]) -> int:
        """Run evidence cleanup on this thin CLI's own interpreter.

        This intentionally bypasses ``run_in_runtime``: cleanup must remain
        available when the managed runtime is absent or damaged and must not
        trigger provisioning merely to delete retained text evidence.
        """

        source_paths = [str(self.runtime.app_source), str(self.runtime.app_source / "src")]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            source_paths.append(existing)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(source_paths)
        environment.update(
            {
                AGENT_CAPSULE_ROOT_VARIABLE: str(self.paths.agent_capsules),
                AGENT_ACTIVITY_ROOT_VARIABLE: str(self.paths.user_data),
                AGENT_IDENTITY_ANCHOR_VARIABLE: str(self.paths.user_data),
                AGENT_LOCATOR_KIND_VARIABLE: AGENT_LOCATOR_KIND_MANAGED,
            }
        )
        return subprocess.run(
            [sys.executable, "-m", AGENT_CLEANUP_MODULE, *arguments],
            cwd=self.runtime.app_source,
            env=environment,
            check=False,
        ).returncode

    def setup(self, arguments: Sequence[str] = ()) -> int:
        """Provision this install, or -- with `--dirs-only` -- just place it.

        The two are separate because the installer script calls this. A full
        `setup` downloads and builds several GB, which would turn a one-line
        install command into a multi-minute one; `--dirs-only` records where
        the big files will go and returns, so the choice still happens during
        installation and the download still happens on first use.
        """

        rest = list(arguments)
        dirs_only = False
        requested: Path | None = None
        while rest:
            argument = rest.pop(0)
            if argument == "--dirs-only":
                dirs_only = True
            elif argument == "--data-dir":
                if not rest:
                    print("--data-dir needs a directory", file=sys.stderr)
                    return 2
                requested = Path(rest.pop(0))
            else:
                print(f"Unknown setup option: {argument}", file=sys.stderr)
                return 2

        try:
            self.settle_big_data_location(requested)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        if dirs_only:
            print(f"FineSub will keep models and downloads in {self.paths.big_data}")
            return 0
        self.ensure_ready()
        print(f"FineSub runtime is ready under {self.paths.root}")
        return 0

    # -- first-run location ---------------------------------------------

    def is_new_install(self) -> bool:
        """Whether this machine has never been provisioned here.

        Four conditions, all of them "nothing of ours is here yet". `runtime`
        counts: if it exists, this machine was provisioned before and a missing
        `locations.json` is a damaged record, not a new install -- asking then
        would offer to move data the user never said had moved.
        """

        if recorded_big_data(self.paths.data_root) is not None:
            return False
        default_root = self.paths.root
        if is_store(default_root) or looks_like_store(default_root):
            return False
        if (default_root / "runtime").exists():
            return False
        return not any(
            (default_root / name).exists() for name in BIG_DATA_NAMES
        )

    def settle_big_data_location(self, requested: Path | None = None) -> None:
        """Decide, once, where the big files go -- before anything downloads.

        Order: an explicit `--data-dir`, then the environment, then whatever
        the front end's prompt returns, then the default. A location that is
        already recorded ends this immediately: moving an existing store is
        `relocate`, and doing it silently from a setup command would be a
        several-GB surprise.
        """

        already_placed = not self.is_new_install()
        if requested is not None and already_placed:
            raise ValueError(
                "已经有数据目录了，不能再用 --data-dir 指定；"
                "要搬到别处请用 `finesub relocate <目录>`。"
            )
        if already_placed:
            return
        chosen = requested
        if chosen is None:
            from_environment = os.environ.get("FINESUB_BIG_DATA_DIR", "").strip()
            if from_environment:
                chosen = Path(from_environment)
        if chosen is None and self.ask_big_data_dir is not None:
            chosen = self.ask_big_data_dir(self.paths.root)
        destination = (
            self.paths.big_data
            if chosen is None
            else self._checked_destination(chosen, force=False)
        )
        if destination == self.paths.big_data:
            # Accepting the default is an answer too, and it has to be written
            # down: otherwise the installer asks, records nothing, and the
            # first real run asks the same question again.
            ensure_store(self.paths, log=_print_log)
            return
        candidate = self.paths.with_big_data(destination)
        # Create and record before downloading, then re-read: another first run
        # may have registered a location while this one was still asking, and
        # `ensure_store` adopts theirs rather than leaving a duplicate copy of
        # several GB behind.
        ensure_store(candidate, log=_print_log)
        self.paths = load_app_paths(self.paths.root, data_root=self.paths.data_root)
        if destination.anchor.lower() != self.paths.runtime.anchor.lower():
            print(
                f"注意：{destination} 与运行环境（{self.paths.runtime}）不在同一磁盘，"
                "缓存与运行环境将无法共享存储，预计多占用约 5 GB。"
                "若目的是给系统盘腾空间，建议安装前设置 FINESUB_HOME 把整个 FineSub "
                "放到那块盘上。",
                file=sys.stderr,
            )

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
        print(f"download     {self._download_route_report()}")
        print(f"env-keys     {self._env_keys_report()}")
        for resource_id, note in (
            ("ffmpeg", ""),
            ("git", "installed on demand: knowledge updates"),
            ("yt-dlp", "installed on demand: URL input"),
            ("tokcount", "optional: offline token counting for the LLM layer"),
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

    def _download_route_report(self) -> str:
        """Which entry points a download would use, without testing them.

        A diagnostic reports configuration; verifying a mirror means fetching
        several GB from it, which is not something `doctor` should do behind
        the user's back.
        """

        from finesub_bootstrap import download_routes

        try:
            # Never probes: a diagnostic reports what is configured and what
            # was already decided. Reaching out to a public endpoint is a
            # network call the user did not ask this command to make.
            decision = download_routes.resolve_region(
                self.paths.data_root, probe=lambda: None
            )
        except Exception as error:  # A diagnostic must not crash on its subject.
            return f"unknown ({error})"
        parts = [decision.describe()]
        for resource_class in ("pypi", "huggingface", "github"):
            if download_routes.is_degraded(self.paths.data_root, resource_class):
                parts.append(f"{resource_class}=已停用（连续失败）")
                continue
            mirror = download_routes.active_mirror(
                self.paths.data_root, resource_class, decision.region
            )
            parts.append(f"{resource_class}={_safe_host(mirror) or '官方源'}")
        return "  ".join(parts)

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
            targets += [
                self.paths.models,
                self.paths.cache,
                self.paths.agent_capsules,
            ]
        if "--purge-tasks" in arguments:
            targets.append(self.paths.tasks)
        purge_user_data = "--purge-user-data" in arguments
        failures: list[str] = []
        barrier = ExitStack()
        try:
            barrier.enter_context(
                holding_activity_barrier(
                    self.paths.user_data,
                    legacy_active_lock=active_lock_path(self.paths.tasks),
                    timeout=0,
                )
            )
        except (OSError, LockUnavailable):
            barrier.close()
            print(
                "FineSub local agents are active; wait for them before uninstalling.",
                file=sys.stderr,
            )
            return 1
        with barrier:
            for target in targets:
                if not os.path.lexists(target):
                    continue
                try:
                    remove_tree(target)
                    print(f"removed {target}")
                except OSError as error:
                    failures.append(f"{target}: {error}")
        # The activity gate itself lives in user-data and cannot be removed on
        # Windows while the barrier holds its file handle. Delete this final
        # root only after every agent-sensitive target is gone and the gate is
        # released.
        if purge_user_data and os.path.lexists(self.paths.user_data):
            try:
                remove_tree(self.paths.user_data)
                print(f"removed {self.paths.user_data}")
            except OSError as error:
                failures.append(f"{self.paths.user_data}: {error}")
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
            if os.path.lexists(self.paths.agent_capsules):
                print(
                    f"Kept local-agent failure evidence at "
                    f"{self.paths.agent_capsules}. Run `finesub agent-clean` "
                    "from an installation using this store, or pass "
                    "--purge-big-data, to remove it."
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
        """Move big data, including local-agent evidence, to another directory.

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
                ("agents", self.paths.agent_capsules),
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
        stack = ExitStack()
        try:
            stack.enter_context(
                holding_activity_barrier(
                    self.paths.user_data,
                    legacy_active_lock=active_lock_path(self.paths.tasks),
                    timeout=0,
                )
            )
            stack.enter_context(
                holding_lock(
                    self.paths.runtime / ".install.lock", timeout=0
                )
            )
        except (OSError, LockUnavailable):
            stack.close()
            print(
                "FineSub 正在运行（有任务或安装在进行），请先等它结束再搬。",
                file=sys.stderr,
            )
            return 1
        with stack:
            return self._relocate_when_idle(destination)

    def _relocate_when_idle(self, destination: Path) -> int:
        """Move the big-data tree while the activity barrier is held."""

        if destination.anchor.lower() != self.paths.runtime.anchor.lower():
            print(
                f"注意：{destination} 与运行环境（{self.paths.runtime}）不在同一磁盘，"
                "缓存与运行环境将无法共享存储，预计多占用约 5 GB。"
                "若目的是给系统盘腾空间，建议整个 FineSub 文件夹一起搬。",
                file=sys.stderr,
            )
        relocated = self.paths.with_big_data(destination)
        source_root = self.paths.big_data
        # Mark the destination a store before anything lands in it, so a crash
        # part-way leaves a directory the next start recognises rather than a
        # pile of unattributed files.
        ensure_store(relocated, log=_print_log)
        # Record the destination *before* moving, keeping the old root
        # searchable until the move is confirmed. Recording afterwards could
        # not be crash-safe: on one volume move_store is a rename, so the
        # source is gone the moment it succeeds, and dying before the record
        # left tasks/ somewhere nothing pointed at.
        record_big_data(
            self.paths.data_root, destination, migrating_from=source_root
        )
        moved, leftovers = move_store(source_root, destination, BIG_DATA_NAMES)
        for leftover in leftovers:
            remove_tree(leftover)
        clear_migration_source(self.paths.data_root)
        self.paths = relocated
        print(
            f"moved {', '.join(moved)} to {destination}"
            if moved
            else f"registered {destination}"
        )
        return 0

    def _checked_destination(self, destination: Path, *, force: bool) -> Path:
        expanded = destination.expanduser()
        # Checked before resolving, not after: `resolve()` makes everything
        # absolute by anchoring it to the current directory, so the test that
        # used to sit below it could never fire -- `finesub relocate data`
        # silently moved several GB into whatever directory the user happened
        # to be standing in.
        if not expanded.is_absolute():
            raise ValueError(f"需要绝对路径：{destination}")
        resolved = expanded.resolve()
        if resolved != self.paths.root and _is_within(resolved, self.paths.root):
            raise ValueError(f"不能放在安装目录里面：{resolved}")
        if _is_within(resolved, self.paths.user_data):
            raise ValueError(f"不能放在个人数据目录里面：{resolved}")
        if resolved != self.paths.root and _is_within(self.paths.runtime, resolved):
            # The install root is exempt: it is the default location, and the
            # runtime lives inside it by definition -- which is exactly what
            # `--reset` asks for.
            raise ValueError(f"不能把运行环境包进去：{resolved}")
        if (
            resolved.exists()
            and any(resolved.iterdir())
            and not is_store(resolved)
            and not looks_like_store(resolved)
        ):
            raise ValueError(
                f"目标目录已有内容且不是 FineSub 数据目录：{resolved}"
            )
        return resolved

    def _nothing_is_running(self) -> bool:
        return activity_is_idle(
            self.paths.user_data,
            legacy_active_lock=active_lock_path(self.paths.tasks),
        ) and try_lock(
            self.paths.runtime / ".install.lock"
        )

    def run_pipeline(self, arguments: Sequence[str]) -> int:
        """One transcription, recorded in the shared task index.

        Every run is filed in the index the desktop reads too. Without `-o`,
        its workspace is `tasks/<task-id>`; an explicit `-o` remains the
        workspace, with the records copied back under that task id afterwards.
        A rerun of the same source can therefore find work worth reusing
        whichever front end started it.

        `-o` is honoured literally: the run happens where the user said, and
        the artifacts appear there as they are produced. Watching a folder fill
        up is how a two-hour job shows progress, and a run that dies at the
        translation stage leaves its finished transcript somewhere the user is
        already looking. Afterwards the ASR result and the correction records
        are copied into the task directory, so continuing this work later does
        not depend on a folder we do not own.
        """

        # Before planning, not after. A first run settles which disk the big
        # files go on, and that rebuilds `self.paths` when the answer is not
        # the default -- so planning first would resolve the task directory
        # under the location the user is about to reject, and the run would
        # land outside the store they chose. `ensure_ready` calls this again
        # and finds it already answered.
        self.settle_big_data_location()
        run_arguments = list(arguments)
        with self._announced_as_running():
            with self._claimed_run(run_arguments) as plan:
                return self._planned_run(plan, run_arguments)

    def _planned_run(
        self, plan: "_RunPlan | None", arguments: list[str]
    ) -> int:
        """Run a plan while its task-id lock is held by the caller."""

        if plan is None:
            return self.run_in_runtime("finesub.pipeline", arguments)
        # Filed before it starts, not after it ends. A run killed part way
        # through -- Ctrl-C during the LLM stage, say -- would otherwise leave
        # its directory with nothing in the index pointing at it: the next run
        # would mint a new id and redo separation and recognition, while a
        # perfectly good ASR result sat in a folder nothing could find again.
        self._record_run(plan, state="running")
        status: int | None = None
        try:
            status = self.run_in_runtime("finesub.pipeline", plan.arguments)
        finally:
            # Reached on Ctrl-C too, which is the case worth being careful
            # about: `interrupted` is what the desktop offers to continue, and
            # it is exactly what happened. Leaving `running` behind would make
            # this task unmatchable until something cleared the mark.
            self._archive_records(plan)
            self._record_run(
                plan,
                state=(
                    "interrupted"
                    if status is None
                    else "completed"
                    if status == 0
                    else "failed"
                ),
            )
        return status

    # -- task directories -----------------------------------------------

    @contextmanager
    def _claimed_run(
        self, arguments: list[str]
    ) -> "Iterator[_RunPlan | None]":
        """Choose and exclusively own the task id used by this run.

        A global "something is active" lock cannot identify which task owns a
        ``running`` entry, and it is intentionally best-effort so unrelated
        tasks can still start.  Each planned task therefore has its own
        mandatory lock.  A busy candidate is skipped and gets a new task id;
        a crashed ``running`` entry is reusable only when its durable sidecar
        exists and can actually be acquired.
        """

        start_fresh = False
        while True:
            entries = (
                []
                if start_fresh
                else task_index.read(
                    self._index_path(), self.paths.tasks
                )
            )
            plan = self._plan_run(
                arguments, entries=entries, include_running=True
            )
            if plan is None:
                yield None
                return

            lock_path = task_lock_path(self.paths.tasks, plan.task_id)
            matched_running = (
                plan.matched is not None
                and plan.matched.get("state") == "running"
            )
            # Old versions wrote ``running`` without a task-specific lock.
            # Treating an absent sidecar as a crashed owner would recreate the
            # original ambiguity, so leave that entry alone and start fresh.
            if matched_running and not lock_path.is_file():
                start_fresh = True
                continue

            stack = ExitStack()
            try:
                # Fixed order shared with the desktop worker. The workspace
                # lock matters when a new task id reuses an older task's
                # output directory.
                stack.enter_context(holding_lock(lock_path, timeout=0))
                stack.enter_context(
                    holding_lock(
                        task_workspace_lock_path(
                            self.paths.tasks, plan.output
                        ),
                        timeout=0,
                    )
                )
            except (OSError, LockUnavailable):
                stack.close()
                if plan.matched is not None:
                    start_fresh = True
                    continue
                raise LockUnavailable(
                    f"Could not claim new task {plan.task_id}"
                )
            try:
                if plan.matched is not None:
                    self._announce_continuation(plan)
                yield plan
            finally:
                stack.close()
            return

    @contextmanager
    def _announced_as_running(self) -> "Iterator[None]":
        """Publish this run, plus the legacy best-effort tree signal.

        The unique activity lease is mandatory and can represent any number of
        concurrent tasks. Its registration shares a stable user-data gate with
        relocation, closing the check-then-move window. `.active.lock` remains
        held when available so older consumers still see the historical signal.

        Only entering the legacy lock is guarded. Errors from the run body must
        propagate unchanged.
        """

        with ExitStack() as stack:
            stack.enter_context(holding_activity(self.paths.user_data))
            # A relocation that already owned the gate may have completed
            # while this run waited. Refresh every path-bearing service before
            # planning so the run cannot recreate the just-moved old tree.
            refreshed = load_app_paths(
                self.paths.root, data_root=self.paths.data_root
            )
            self.paths = refreshed
            self.resources.paths = refreshed
            self.runtime.paths = refreshed
            try:
                stack.enter_context(
                    holding_lock(active_lock_path(self.paths.tasks), timeout=5)
                )
            except (OSError, LockUnavailable):
                pass
            yield

    def _index_path(self) -> Path:
        return self.paths.user_data / "tasks.json"

    def _plan_run(
        self,
        arguments: list[str],
        *,
        entries: Sequence[dict] | None = None,
        include_running: bool = False,
    ) -> "_RunPlan | None":
        """Decide which task directory this run belongs to.

        Returns None when the command cannot be read confidently, and the run
        then behaves exactly as it did before task directories existed. The
        only thing asked of the caller is that the source come first, which is
        what `finesub --help` documents -- guessing which token is the input
        would mean knowing which of the pipeline's forty-odd flags take a
        value, a table that would go stale silently and name the wrong file.
        """

        if not arguments or arguments[0].startswith("-"):
            print(
                "Note: the source has to come first for FineSub to file this "
                "run under tasks/; writing to the working directory instead.",
                file=sys.stderr,
            )
            return None
        source = arguments[0]
        requested = _flag_value(arguments, "-o", "--output")
        name = _flag_value(arguments, "--name")
        if name and not requested:
            # The shell later injects -o, which makes the pipeline ignore
            # --name and used to bypass its bare-name validation. Reuse the
            # pipeline's canonical validator before planning or recording.
            from finesub.paths import resolve_name_output_path

            resolve_name_output_path(name)
        stage = _run_properties(arguments)["stage"]
        match = task_index.find_latest(
            (
                task_index.read(self._index_path(), self.paths.tasks)
                if entries is None
                else entries
            ),
            source=source,
            # `_claimed_run` asks to see running candidates only so it can
            # verify and acquire their task-specific sidecar. Merely including
            # one here never makes it safe to reuse.
            include_running=include_running,
        )
        stem = self._task_stem(arguments, source=source, match=match)
        task_id = (
            str(match["task_id"]) if match else task_output.new_task_id(stem)
        )
        if requested:
            # Left exactly as given, arguments included: the user named a
            # destination, and the run belongs there where they can watch it.
            plan = _RunPlan(
                task_id=task_id,
                source=source,
                stage=stage,
                arguments=list(arguments),
                output=Path(requested).expanduser().resolve(),
                recorded_name="",
                matched=match,
            )
        else:
            output = task_output.resolve_task_output(
                self.paths.tasks, task_id, stem=stem
            )
            plan = _RunPlan(
                task_id=task_id,
                source=source,
                stage=stage,
                arguments=[*arguments, "-o", str(output)],
                output=output,
                recorded_name=name or "",
                matched=match,
            )
        return plan

    @staticmethod
    def _announce_continuation(plan: "_RunPlan") -> None:
        """Say what continuing this task will actually save, if anything.

        Filing under an existing task and reusing its work are different
        things: `-o` pointing somewhere new starts from an empty directory, and
        "Continuing task ..." would promise a rerun that skips the expensive
        stages when it is about to redo every one of them.
        """

        stem = plan.output.with_suffix("")
        resumable = any(
            stem.with_name(f"{stem.name}{suffix}").is_file()
            for suffix in artifacts.RECORD_SUFFIXES
        )
        if resumable:
            print(f"Continuing task {plan.task_id}", file=sys.stderr)
        else:
            print(
                f"Filing under task {plan.task_id}; nothing to reuse in "
                f"{plan.output.parent}, so this run starts over.",
                file=sys.stderr,
            )

    @staticmethod
    def _task_stem(
        arguments: Sequence[str], *, source: str, match: dict | None
    ) -> str:
        """What this run's files are called inside the task directory.

        A continued task keeps the name its artifacts already have. Deriving it
        again from the command line would let `--name` rename the run's files
        while it reuses the directory: the pipeline would find no
        `<new-stem>-stable.json`, redo separation and recognition in full, and
        leave two parallel sets of artifacts in one folder -- all while the
        shell claimed to be continuing.
        """

        recorded = (match or {}).get("request")
        previous = recorded.get("output") if isinstance(recorded, dict) else None
        if isinstance(previous, str) and previous:
            return Path(previous).stem
        return task_output.task_stem(
            name=_flag_value(arguments, "--name") or "", source=source
        )

    @staticmethod
    def _recorded_request(plan: "_RunPlan") -> dict[str, object]:
        """The effective CLI settings that the desktop can faithfully replay.

        These defaults deliberately match ``pipeline.parse_args``, not the
        desktop form: notably the CLI defaults knowledge updates off. Keeping
        an older desktop request or letting Pydantic fill absent fields would
        claim the run used settings it did not and could turn a later retry
        from ``--device cpu --knowledge none`` into CUDA plus an update.
        """

        arguments = plan.arguments
        language = _flag_value(arguments, "--language")
        model_name = _flag_value(arguments, "--model") or "large-v3-turbo"
        extra_info = (_flag_value(arguments, "--extra-info") or "").strip()
        extra_info_file = _flag_value(arguments, "--extra-info-file")
        if extra_info_file:
            try:
                file_info = (
                    Path(extra_info_file)
                    .expanduser()
                    .read_text(encoding="utf-8")
                    .strip()
                )
            except (OSError, UnicodeError, ValueError):
                file_info = ""
            extra_info = "\n".join(
                part for part in (extra_info, file_info) if part
            )

        return {
            "input": task_index.canonical_source(plan.source),
            "output": str(plan.output),
            "name": plan.recorded_name,
            "cleanup_intermediate": False,
            "stage": plan.stage,
            "model_name": model_name,
            "device": _choice_flag(
                arguments, "--device", ("cuda", "cpu"), "cuda"
            ),
            "gpu_index": None,
            "gpu_name": "",
            "language": language,
            "gpu_budget_gb": _integer_choice_flag(
                arguments, "--gpu-budget-gb", (4, 8, 12, 16), 4
            ),
            "word": _boolean_flag(
                arguments,
                enabled=("--word", "-w"),
                disabled=("--no-word",),
                default=False,
            ),
            "asr_stabilize_profile": _integer_choice_flag(
                arguments, "--asr-stabilize-profile", (-1, 0, 1, 2), 0
            ),
            "split_length_scale": _number_flag(
                arguments,
                "--split-length-scale",
                default=None,
                minimum=0.6,
                maximum=1.6,
            ),
            "llm_media": _choice_flag(
                arguments, "--llm-media", ("text", "audio", "video"), "video"
            ),
            "llm_retrieval": _choice_flag(
                arguments, "--llm-retrieval", ("none", "local", "native"), "local"
            ),
            "llm_difficulty": _choice_flag(
                arguments,
                "--llm-difficulty",
                ("high", "med", "minimum"),
                "high",
            ),
            "llm_fast": _choice_flag(
                arguments, "--llm-fast", ("auto", "on", "off"), "auto"
            ),
            "llm_output_scale": _number_flag(
                arguments,
                "--llm-output-scale",
                default=1.0,
                strictly_positive=True,
            ),
            "extra_info": extra_info,
            "extra_style": _flag_value(arguments, "--extra-style") or "",
            "knowledge": _choice_flag(
                arguments,
                "--knowledge",
                ("none", "collect", "update"),
                "none",
            ),
            "postprocess_profile": _integer_choice_flag(
                arguments, "--postprocess-profile", (-1, 0, 1, 2, 3, 4), 0
            ),
        }

    def _record_run(self, plan: "_RunPlan", *, state: str) -> None:
        """File this run in the index the desktop reads too.

        Never fatal: a record we could not save must not change what actually
        happened, which is the same rule the desktop's own writer follows.

        The request is the complete TaskRequest-compatible subset of the CLI's
        effective configuration, including CLI defaults. That lets the desktop
        retry what actually ran rather than filling omissions with its own,
        intentionally different defaults.
        """

        now = time.time()
        try:
            request = self._recorded_request(plan)
            task_index.merge_write(
                self._index_path(),
                [
                    {
                        "task_id": plan.task_id,
                        "state": state,
                        "request": request,
                        # The subtitle this stage delivers, once it exists.
                        # Named rather than assumed: `-o` names the *final*
                        # SRT, and every earlier stage delivers a sibling.
                        "outputs": self._delivered_outputs(plan),
                        # A continued task was created when it was created;
                        # resetting it would make a three-week-old run sort
                        # and read as new every time it is picked up again.
                        "created_at": (plan.matched or {}).get("created_at", now),
                        "updated_at": now,
                    }
                ],
                self.paths.tasks,
            )
        except Exception as error:  # A record must not change the outcome.
            print(f"Could not record this task ({error}).", file=sys.stderr)

    @staticmethod
    def _delivered_outputs(plan: "_RunPlan") -> dict[str, str]:
        """Every subtitle this run left behind, for the history entry.

        Filed under the names the desktop's history looks for; anything else is
        a path the other front end can see and still not offer to open.

        All of them, not just the stage that was asked for: a `final-srt` run
        that died in translation still wrote a usable `-raw.srt`, and recording
        only the stage requested would leave that entry with no output at all
        -- the one case where being able to open what did get made matters
        most. Existence decides, which is also why the exit status is not
        consulted here.
        """

        produced: dict[str, str] = {}
        for stage, key in artifacts.DELIVERABLE_KEY_BY_STAGE.items():
            path = artifacts.deliverable(plan.output, stage)
            if path is not None and path.is_file():
                produced[key] = str(path)
        return produced

    def _archive_records(self, plan: "_RunPlan") -> None:
        """Keep a copy of what a later run would need, in the task directory.

        Only for runs that happened somewhere else -- a run in its own task
        directory already has them there. The two files are the ASR result and
        the correction records: small, and the only things a rerun cannot
        reproduce without redoing the expensive half or spending API quota.
        Everything else is left in the user's folder untouched, because it is
        their folder.

        Attempted whatever the run's exit status was: a failure after the ASR
        stage is exactly when having the result recorded matters. Never fatal
        -- a copy we could not make must not change what happened.
        """

        task_directory = self.paths.tasks / plan.task_id
        if plan.output.parent == task_directory:
            return
        if any(path.is_file() for path in artifacts.removable_artifacts(
            task_directory / plan.output.name
        )):
            # That directory is some earlier run's own workspace, not a record
            # store. Dropping this run's `-stable.json` in would leave it
            # holding an `-aligned.json` and a `-stable.json` produced by
            # different settings, and a later rerun there would skip to the
            # newer one while everything around it described the older.
            return
        stem = plan.output.with_suffix("")
        for suffix in artifacts.RECORD_SUFFIXES:
            produced = stem.with_name(f"{stem.name}{suffix}")
            if not produced.is_file():
                continue
            destination = task_directory / produced.name
            # Written aside and renamed into place. A copy interrupted half way
            # -- the process killed, the disk full -- would otherwise leave a
            # truncated `-stable.json` under its final name, and the pipeline
            # skips a stage on the *existence* of its output: the next run
            # would take that file for a finished ASR result, skip recognition,
            # and fail parsing it instead.
            partial = destination.with_name(f".{destination.name}.part")
            try:
                task_directory.mkdir(parents=True, exist_ok=True)
                shutil.copy2(produced, partial)
                os.replace(partial, destination)
            except OSError as error:
                # Best effort in its own right: whatever stopped the copy --
                # a full disk, a permission, a directory that went away -- can
                # stop the cleanup too, and an exception escaping here would
                # skip the run's final record and turn a finished pipeline into
                # a crash. Archiving must never change what happened.
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
                print(
                    f"Could not record {produced.name} ({error}).",
                    file=sys.stderr,
                )

    def run_in_runtime(self, module: str, arguments: Sequence[str]) -> int:
        arguments = list(arguments)
        self.ensure_ready()
        self._ensure_capabilities(arguments)
        self._prefer_capabilities(arguments)
        context = self.runtime.worker_context(
            ffmpeg_bin=self.tool_directory("ffmpeg", "ffmpeg.exe"),
            extra_env={
                **shared_environment_overrides(self.paths),
                **token_counter_overrides(
                    lambda: self.tool_file("tokcount", "tokcount.exe")
                ),
            },
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
        # Asked before anything is stored, so the answer decides where the
        # first download lands rather than where the next one would.
        self.settle_big_data_location()
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
        if self._system_tool(resource_id) is not None:
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

    def _prefer_capabilities(self, arguments: Sequence[str]) -> None:
        """Fetch what this command would benefit from, without depending on it.

        Every failure here is survivable by construction -- the pipeline has a
        fallback for each of these -- so a dead mirror, a full disk or a
        packaged install that cannot provision must all end in the run starting
        anyway, one tier slower.
        """

        for resource_id in preferred_capabilities_from_arguments(list(arguments)):
            if self._system_tool(resource_id) is not None:
                continue
            if self.resources.status(resource_id).usable or not self.can_provision:
                continue
            print(
                f"{resource_id} is missing "
                f"({_CAPABILITY_REASONS[resource_id]}); downloading it now.",
                file=sys.stderr,
            )
            try:
                self.resources.install(
                    resource_id, _print_progress, stage=_print_stage
                )
            except Exception as error:  # Optional by definition; never fatal.
                print(
                    f"{resource_id} could not be installed ({error}); "
                    "continuing without it.",
                    file=sys.stderr,
                )
            print(file=sys.stderr)

    # -- managed tools --------------------------------------------------

    def _system_tool(self, resource_id: str):
        """`system_tool`, asked at most once per resource per command."""

        if resource_id not in self._system_tools:
            self._system_tools[resource_id] = system_tool(resource_id)
        return self._system_tools[resource_id]

    def tool_directory(self, resource_id: str, filename: str):
        """Directory to put on PATH, preferring a system copy."""

        found = self._system_tool(resource_id)
        if found is not None:
            return found.directory
        active = self.resources.active_file(resource_id, filename)
        return active.parent if active is not None else None

    def tool_file(self, resource_id: str, filename: str) -> Path | None:
        """The executable itself, for tools we name rather than put on PATH.

        `filename` describes the managed copy; a system one is taken as found,
        since the machine may well have named it something else.
        """

        found = self._system_tool(resource_id)
        if found is not None:
            return found.path
        return self.resources.active_file(resource_id, filename)

    def _git_path_dirs(self) -> list:
        # A system git is already on PATH; only a managed one needs injecting.
        if self._system_tool("git") is not None:
            return []
        directory = self.tool_directory("git", "git.exe")
        return [directory] if directory is not None else []

    def _yt_dlp_python_path(self) -> list:
        # Imported, not executed, so it joins PYTHONPATH rather than PATH.
        if self.resources.active_version("yt-dlp") is None:
            return []
        return [self.resources.install_path("yt-dlp")]

    def _tool_state(self, resource_id: str) -> str:
        if self._system_tool(resource_id) is not None:
            return "ready"
        return self.resources.status(resource_id).state

    def _tool_report(self, resource_id: str, note: str) -> str:
        found = self._system_tool(resource_id)
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


def _safe_host(url: str) -> str:
    """Scheme and host, for a diagnostic that gets pasted into issues.

    These addresses can be overridden per machine, and an override may carry
    credentials -- `https://user:token@host/simple` is an ordinary way to point
    at a private index. `doctor` output ends up in bug reports and screen
    shares, so it never carries userinfo, path, query or fragment.
    """

    if not url:
        return ""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
    except ValueError:
        return "(unparsable)"
    if not parts.hostname:
        return "(unparsable)"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


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
            if (source / "src" / "finesub" / "pipeline.py").is_file():
                return source
    if (root / "src" / "finesub" / "pipeline.py").is_file():
        return root
    raise FileNotFoundError(f"No FineSub application source under {root}")
