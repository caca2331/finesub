"""Collect one or more finished tasks into a single zip for the developer.

Deterministic half of `agent-tasks/feedback-pack/SKILL.md`. The judgement --
which tasks, which of them a person polished by hand, whether the source media
is worth including -- belongs to the agent reading that file; everything here
is mechanical, and it lives in a script for one reason: a list of artifact
paths written into a prose file drifts from the pipeline the first time a stage
grows a new one.

So this does not carry such a list. It leans on the pipeline's own rule --
every artifact of one run is derived from the delivered SRT and sits beside it
(`README_DEV.md`, the artifact tree) -- and states only the *exclusions*, which
is the half worth being explicit about.

Two modes, from the two things a user might be doing:

* ``debug`` -- something went wrong. Everything about the named tasks, the
  vocal track included, because the audio is often the answer.
* ``corpus`` -- the user is donating material for prompt work. No audio at all,
  and only tasks whose subtitles someone actually corrected by hand.

Never sends anything. It writes a zip and prints its path; who sees it is the
user's decision, taken after reading the manifest this puts inside.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


MODES = ("debug", "corpus")

#: Never packed, in any mode. Secrets first; then the scratch decode, which is
#: a lossless copy of the source (hundreds of MB for an hour of video) carrying
#: nothing the vocal track does not, and which any rerun makes again.
ALWAYS_EXCLUDE = (
    "*.env", ".env*", "*.key", "*.pem", "id_rsa*",
    "*-decoded.flac",
)

#: Bulk that is either regenerable or not the developer's business. Source
#: media is the big one: a downloaded video is hundreds of megabytes, and the
#: vocal track carries the same information for a fraction of it.
SOURCE_MEDIA_SUFFIXES = (
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts",
    ".mp3", ".m4a", ".aac", ".wav", ".opus",
)

#: Audio-shaped artifacts, dropped whole in `corpus` mode.
AUDIO_PATTERNS = ("*-vocal.ogg", "*-vocal.flac", "*.aac", "*.mp4")

#: Regenerable and large; the developer can rebuild it from the vocal track.
CORPUS_ONLY_EXCLUDE = ("*-vad-energy.npz", "*-vad.json")

#: Config keys worth having and safe to send. Everything else in `config.toml`
#: is dropped rather than filtered -- a whitelist fails closed when the file
#: grows a key nobody thought about here.
CONFIG_SECTION_WHITELIST = ("vad", "segmentation", "stabilize", "separator", "llm")

#: `[llm]` holds model composition, which is useful, next to nothing secret --
#: but `proxy` is a URL that may carry credentials.
CONFIG_KEY_BLOCKLIST = ("proxy", "key", "token", "secret", "password")


@dataclass(frozen=True)
class Task:
    """One finished run, named by the subtitle it delivered."""

    final_srt: Path
    refined_srt: Path | None = None

    @property
    def label(self) -> str:
        return self.final_srt.with_suffix("").name

    @property
    def directory(self) -> Path:
        return self.final_srt.parent


def _fingerprint(path: Path | None) -> str:
    """What identifies a file's *content* for the ledger, cheaply.

    Size and mtime rather than a digest: the ledger exists to stop the same
    material being sent twice, not to prove anything, and hashing a gigabyte of
    audio to answer "did this change" would cost more than the packing.
    """

    if path is None or not path.is_file():
        return ""
    status = path.stat()
    return f"{status.st_size}:{status.st_mtime_ns}"


def task_fingerprint(task: Task) -> str:
    """⚠ Not just the path.

    Keying on the path alone means a user who improves their corrected
    subtitles can never contribute them again -- the ledger would say the task
    was already sent. The delivered subtitle *and* the hand-corrected one are
    both in the key, so an edit to either makes this a new contribution.
    """

    return f"{_fingerprint(task.final_srt)}|{_fingerprint(task.refined_srt)}"


def ledger_path(explicit: str | Path | None = None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser().resolve()
    try:
        from finesub.paths import resolve_logs_dir
    except ImportError:
        return None
    logs = resolve_logs_dir()
    # `<user-data>/logs` -- so its parent is the data root this front end uses,
    # which is where a record of what we have sent belongs.
    return None if logs is None else logs.parent / "feedback" / "packed.json"


def read_ledger(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def write_ledger(path: Path | None, record: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:  # pragma: no cover - a read-only data root
        print(f"warning: could not write the ledger ({exc})", file=sys.stderr)


def already_packed(ledger: dict, mode: str, task: Task) -> bool:
    """Whether this exact material already went out under this mode.

    Per mode, deliberately: the same task can go once as a bug report (with
    audio) and again later as corpus material, and neither cancels the other.
    """

    return ledger.get(mode, {}).get(str(task.final_srt)) == task_fingerprint(task)


def _excluded(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns)


def _belongs_to(name: str, label: str) -> bool:
    """Whether a sibling file or directory is this task's.

    ⚠ Not a bare `startswith`: with the `-o` layout, task `a` would then claim
    `abc-raw.srt` from the task next to it. Everything the pipeline derives is
    the stem plus a separator -- `-raw.srt`, `.llm-artifacts`, `.srt` itself.
    """

    return name == label or name.startswith((f"{label}-", f"{label}."))


def task_files(task: Task, mode: str, *, with_source: bool) -> list[Path]:
    """Every file of one task that this mode is willing to send.

    A sweep of the run's own directory rather than a list of suffixes: the
    pipeline puts everything for one input beside the delivered SRT, so walking
    it stays right when a stage grows a new artifact, and the exclusions below
    say what may *not* travel -- which is the half worth being explicit about.

    ⚠ Filtered by the task's own stem, not just by directory. The default
    layout gives each run its own `out/<stem>/`, but a run given an explicit
    `-o` writes its artifacts as *siblings* of that path -- so the directory
    can hold several tasks, and sweeping it whole would pack someone else's
    subtitles under this task's name.
    """

    excluded = [*ALWAYS_EXCLUDE]
    if mode == "corpus":
        excluded += [*AUDIO_PATTERNS, *CORPUS_ONLY_EXCLUDE]
    chosen: list[Path] = []
    for path in sorted(task.directory.rglob("*")):
        if not path.is_file():
            continue
        if not _belongs_to(path.relative_to(task.directory).parts[0], task.label):
            continue
        if _excluded(path, tuple(excluded)):
            continue
        # The vocal track carries the same speech at a fraction of the size, so
        # the source travels only when asked for by name -- and the only reason
        # to ask is a suspected separation problem. The deliveries themselves
        # are `.ogg`/`.flac` and so never land in this branch.
        if path.suffix.lower() in SOURCE_MEDIA_SUFFIXES and not (
            with_source and mode == "debug"
        ):
            continue
        chosen.append(path)
    if task.refined_srt is not None and task.refined_srt.is_file():
        chosen.append(task.refined_srt)
    return chosen


#: `run-<YYYYMMDD>-<HHMMSS>-<stem>.log`, written by `pipeline._run_log`.
_RUN_LOG = re.compile(r"^run-\d{8}-\d{6}-(?P<stem>.+)\.log$")


def _log_stem(name: str) -> str:
    """The name a run log carries, sanitised the way the pipeline sanitises it."""

    return re.sub(r"[^\w.-]+", "-", name)[:60].strip("-") or "run"


def task_log_names(task: Task) -> set[str]:
    """Which log stems belong to this task.

    Two, because the log is named after the *input* while a task is named after
    the subtitle it delivered -- the same for the default layout, different the
    moment someone passes `-o`. The source is read back out of the run metadata
    rather than guessed.
    """

    names = {_log_stem(task.label)}
    metadata = task.directory / f"{task.label}-metadata.json"
    try:
        source = json.loads(metadata.read_text(encoding="utf-8")).get("source")
    except (OSError, ValueError, AttributeError):
        source = None
    if isinstance(source, str) and source:
        names.add(_log_stem(Path(source).stem))
    return names


def run_logs_for(task: Task, limit: int = 3) -> list[Path]:
    """The run logs that belong to this task, newest first.

    Several, because a task can be run more than once and the failed attempt is
    usually the one worth having.

    ⚠ Matched on the *whole* stem, not a substring. `run-*<label>*.log` looked
    fine and quietly collected the neighbours: task `a` matched every log with
    an `a` in it, and task `input` walked off with `my-input`'s. That is the
    same mistake the artifact side made twice (`_belongs_to`), and here it
    would have shipped a stranger's log inside a bundle the user is about to
    send to someone.
    """

    try:
        from finesub.paths import resolve_logs_dir

        logs_dir = resolve_logs_dir()
    except ImportError:
        return []
    if logs_dir is None or not logs_dir.is_dir():
        return []
    wanted = task_log_names(task)
    matches = [
        candidate
        for candidate in logs_dir.glob("run-*.log")
        if candidate.is_file()
        and (match := _RUN_LOG.match(candidate.name)) is not None
        and match.group("stem") in wanted
    ]
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[:limit]


def _toml_scalar(value: object) -> str | None:
    """One config value as TOML, or None for anything that is not a scalar.

    The file is named `.toml`, so it has to parse as one -- `repr` would have
    written `True` and `'text'`, which read fine and load as neither. Nested
    tables and arrays are dropped rather than rendered: this is an excerpt for
    reading, and a half-rendered table is worse than an absent one.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return None


def filtered_config() -> str | None:
    """`config.toml`, reduced to sections that explain a run's behaviour.

    Whitelisted, and dropped entirely if it cannot be parsed: a config file is
    where an API key ends up when someone puts it in the wrong place, and a
    blacklist would send it the first time a key acquires a new name.
    """

    try:
        import tomllib

        from finesub.paths import resolve_config_file
    except ImportError:
        return None
    path = resolve_config_file()
    if path is None or not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    lines: list[str] = []
    for section in CONFIG_SECTION_WHITELIST:
        table = data.get(section)
        if not isinstance(table, dict):
            continue
        kept = {
            key: value
            for key, value in table.items()
            if not any(word in key.lower() for word in CONFIG_KEY_BLOCKLIST)
        }
        if not kept:
            continue
        rendered = [
            f"{key} = {scalar}"
            for key, value in sorted(kept.items())
            if (scalar := _toml_scalar(value)) is not None
        ]
        if not rendered:
            continue
        lines.append(f"[{section}]")
        lines.extend(rendered)
        lines.append("")
    return "\n".join(lines) if lines else None


def destination(mode: str, out_dir: str | Path | None) -> Path:
    """Where the zip lands: the desktop, or the home folder if there is none.

    Not written in stone as `~/Desktop`: on Windows the desktop is routinely
    redirected into OneDrive, and a path that does not exist is worse than a
    slightly less convenient one.
    """

    if out_dir:
        root = Path(out_dir).expanduser()
    else:
        desktop = Path.home() / "Desktop"
        root = desktop if desktop.is_dir() else Path.home()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return root / f"finesub-feedback-{mode}-{stamp}.zip"


def bundle_folders(tasks: list[Task]) -> dict[Path, str]:
    """One folder name per task inside the zip, distinct even when stems collide.

    ⚠ The stem is not unique. `out/a/final.srt` and `out/b/final.srt` are two
    different tasks with the same label, and naming both folders `final` would
    have them overwrite each other -- silently, since a zip happily holds two
    members with one name. The ledger already keys on the absolute path; this
    is the same fact reaching the archive.
    """

    labels = [task.label for task in tasks]
    # A label used by exactly one task is spoken for before any suffix is
    # handed out: two tasks called `final` plus one genuinely called `final-2`
    # would otherwise collide all over again, one step further along.
    taken = {label for label in labels if labels.count(label) == 1}
    counters: dict[str, int] = {}
    folders: dict[Path, str] = {}
    for task in tasks:
        if labels.count(task.label) == 1:
            folders[task.final_srt] = task.label
            continue
        index = counters.get(task.label, 0)
        while True:
            index += 1
            candidate = task.label if index == 1 else f"{task.label}-{index}"
            if candidate not in taken:
                break
        counters[task.label] = index
        taken.add(candidate)
        folders[task.final_srt] = candidate
    return folders


def build_manifest(
    mode: str,
    entries: list[tuple[Task, list[Path]]],
    folders: dict[Path, str],
) -> str:
    """What is in the bundle, in the user's language, for the user to read.

    The point of this file is that nobody has to trust a description of what
    was packed: it lists what actually went in, so the person deciding whether
    to send it is deciding about something they can see.
    """

    lines = [
        f"FineSub 反馈包（{mode}）",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "这个包是给开发者排查问题 / 改进提示词用的。发不发由你决定。",
        "",
        "里面**会有**：字幕全文（原始、纠错、你精修的那份）、这次运行的参数与耗时、",
        "LLM 每次问答的完整记录（含知识库里被用到的条目原文）、运行日志。",
        "日志与产物里带绝对路径，路径中通常含你的用户名。",
        (
            "音频：不包含任何音频。"
            if mode == "corpus"
            else "音频：包含分离出来的人声轨（16 kHz 单声道）。"
        ),
        "",
        "里面**不会有**：API key、.env、config.toml 里任何像密钥的项。",
        "",
        "文件清单：",
    ]
    for task, files in entries:
        # The folder name, not the label: they differ exactly when two tasks
        # share a stem, and that is the case where a reader most needs the
        # heading to match what they see in the zip.
        lines.append(f"  [{folders[task.final_srt]}]")
        for path in files:
            try:
                size = f"{path.stat().st_size / 1024:.0f} KB"
            except OSError:
                size = "?"
            lines.append(f"    {path.name}  ({size})")
    return "\n".join(lines) + "\n"


def pack(
    tasks: list[Task],
    *,
    mode: str,
    out_dir: str | Path | None = None,
    with_source: bool = False,
    ledger_file: str | Path | None = None,
    ignore_ledger: bool = False,
    dry_run: bool = False,
) -> Path | None:
    ledger_at = ledger_path(ledger_file)
    ledger = read_ledger(ledger_at)
    entries: list[tuple[Task, list[Path]]] = []
    for task in tasks:
        if not ignore_ledger and already_packed(ledger, mode, task):
            print(f"skip (already sent): {task.label}")
            continue
        files = task_files(task, mode, with_source=with_source)
        files.extend(run_logs_for(task))
        if not files:
            print(f"skip (nothing to pack): {task.label}")
            continue
        entries.append((task, files))
    if not entries:
        print("nothing to pack")
        return None

    archive = destination(mode, out_dir)
    folders = bundle_folders([task for task, _files in entries])
    manifest = build_manifest(mode, entries, folders)
    if dry_run:
        print(manifest)
        print(f"(dry run) would write {archive}")
        return None

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("MANIFEST.txt", manifest)
        config = filtered_config()
        if config:
            bundle.writestr("config-excerpt.toml", config)
        for task, files in entries:
            for path in files:
                try:
                    inner = path.relative_to(task.directory)
                except ValueError:
                    # Run logs and a hand-corrected subtitle living elsewhere.
                    inner = Path(path.name)
                bundle.write(path, f"{folders[task.final_srt]}/{inner.as_posix()}")

    for task, _files in entries:
        ledger.setdefault(mode, {})[str(task.final_srt)] = task_fingerprint(task)
    write_ledger(ledger_at, ledger)
    print(archive)
    return archive


def _parse_tasks(values: list[str], refined: list[str]) -> list[Task]:
    corrections: dict[str, Path] = {}
    for item in refined:
        task_path, _, refined_path = item.partition("=")
        if not refined_path:
            raise SystemExit(
                f"--refined needs <task-srt>=<refined-srt>, got {item!r}"
            )
        corrections[str(Path(task_path).expanduser().resolve())] = (
            Path(refined_path).expanduser().resolve()
        )
    tasks: list[Task] = []
    for value in values:
        final = Path(value).expanduser().resolve()
        if not final.parent.is_dir():
            raise SystemExit(f"no such task directory: {final.parent}")
        tasks.append(Task(final, corrections.get(str(final))))
    return tasks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=MODES)
    parser.add_argument(
        "tasks",
        nargs="+",
        help="each task's delivered SRT (out/<stem>/<stem>.srt); "
        "every other artifact is derived from it",
    )
    parser.add_argument(
        "--refined",
        action="append",
        default=[],
        metavar="TASK_SRT=REFINED_SRT",
        help="a hand-corrected subtitle for one task. Which tasks have one is "
        "the agent's judgement, not this script's guess.",
    )
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--with-source",
        action="store_true",
        help="also pack the source media (debug mode only). Hundreds of MB; "
        "only worth it when separation itself is suspect.",
    )
    parser.add_argument("--ledger", default=None)
    parser.add_argument(
        "--ignore-ledger",
        action="store_true",
        help="pack tasks that were already sent",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tasks = _parse_tasks(args.tasks, args.refined)
    if args.mode == "corpus":
        without = [task.label for task in tasks if task.refined_srt is None]
        if without:
            raise SystemExit(
                "corpus mode packs only tasks with a hand-corrected subtitle; "
                f"no --refined given for: {', '.join(without)}"
            )
    pack(
        tasks,
        mode=args.mode,
        out_dir=args.out_dir,
        with_source=args.with_source,
        ledger_file=args.ledger,
        ignore_ledger=args.ignore_ledger,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
