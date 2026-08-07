"""Move task outputs out of `user-data` and record them by relative path.

Task outputs are the one big thing a user cannot regenerate, but they are big:
vocals, alignment JSON and the finished subtitle for every task. Keeping them
inside `user-data` made that directory unbounded, which is the opposite of what
the rest of the layout assumes -- so they move to the big-data root, alongside
models and downloads, where they can be pointed at another disk.

The history file is rewritten in the same pass. It used to store absolute
output paths, so moving the folder by hand left every "open folder" in the task
list pointing at nothing; recorded relative to the tasks root, they survive.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from finesub_bootstrap.fsops import move_tree
from finesub_bootstrap.locks import try_lock
from finesub_bootstrap.migrations import Migration
from finesub_bootstrap.paths import AppPaths

MIGRATION_ID = "0003-tasks-out-of-user-data"


def _relative(value: object, previous_root: Path) -> object:
    if not isinstance(value, str) or not value:
        return value
    try:
        return Path(value).relative_to(previous_root).as_posix()
    except ValueError:
        return value


def _rewrite_history(history: Path, previous_root: Path) -> None:
    """Turn recorded outputs into paths relative to the tasks root.

    They were absolute, which is why moving the folder used to break every
    "open" in the task list. Relative entries resolve against wherever the
    tasks root is now, so this conversion is the last time these paths need
    touching.
    """

    try:
        body = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    tasks = body.get("tasks") if isinstance(body, dict) else None
    if not isinstance(tasks, list):
        return
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        request = task.get("request")
        if isinstance(request, dict) and "output" in request:
            converted = _relative(request["output"], previous_root)
            changed = changed or converted != request["output"]
            request["output"] = converted
        outputs = task.get("outputs")
        if isinstance(outputs, dict):
            for key, value in list(outputs.items()):
                converted = _relative(value, previous_root)
                changed = changed or converted != value
                outputs[key] = converted
    if changed:
        history.write_text(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )


def relocate(paths: AppPaths, log: Callable[[str], None]) -> bool:
    stray = paths.user_data / "tasks"
    if stray == paths.tasks or not stray.is_dir():
        return True
    if not try_lock(paths.tasks.parent / ".active.lock") or not try_lock(
        paths.user_data / "tasks.json.lock"
    ):
        # A desktop app is open. Its JobManager holds the history in memory and
        # writes the whole thing back, so anything we rewrite now would be
        # overwritten; and its running task is writing into the directory we
        # want to move. Defer -- the framework retries at the next start.
        log("检测到 FineSub 正在运行，任务产物迁移改到下次启动时进行。")
        return False
    if paths.tasks.is_dir() and any(paths.tasks.iterdir()):
        log(
            f"发现两处任务产物：{stray} 与 {paths.tasks}，未自动合并。"
            "任务目录以任务 id 命名，可直接把子目录并到一起。"
        )
        return False
    if paths.tasks.exists():
        paths.tasks.rmdir()
    move_tree(stray, paths.tasks)
    history = paths.user_data / "tasks.json"
    if history.is_file():
        _rewrite_history(history, stray)
    log(f"任务产物已从 {stray} 迁移到 {paths.tasks}")
    return True


MIGRATION = Migration(id=MIGRATION_ID, run=relocate)
