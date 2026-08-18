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
from contextlib import ExitStack
import json
from pathlib import Path

from finesub_bootstrap.fsops import move_tree, write_atomic
from finesub_bootstrap.locks import (
    LockUnavailable,
    active_lock_path,
    holding_activity_barrier,
    holding_lock,
)
from finesub_bootstrap.migrations import Migration
from finesub_bootstrap.paths import AppPaths

MIGRATION_ID = "0003-tasks-out-of-user-data"


def _previous_roots(paths: AppPaths) -> tuple[Path, ...]:
    """Every tasks root a recorded path may still name.

    Two, because 0002 runs first and moves a portable copy's whole `user-data`
    into the managed location without touching the history inside it. Such a
    history names `<install>/user-data/tasks/...` -- a root that no longer
    exists and that the post-0002 root does not contain, so converting against
    the current one alone silently leaves every entry absolute.
    """

    return (paths.user_data / "tasks", paths.root / "user-data" / "tasks")


def _relative(value: object, previous_roots: tuple[Path, ...]) -> object:
    if not isinstance(value, str) or not value:
        return value
    for root in previous_roots:
        try:
            return Path(value).relative_to(root).as_posix()
        except ValueError:
            continue
    return value


def _load_history(history: Path) -> dict | None:
    try:
        body = json.loads(history.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) else None


def _convert(body: dict, previous_roots: tuple[Path, ...]) -> bool:
    """Rewrite recorded outputs in place; report whether anything moved.

    They were absolute, which is why moving the folder used to break every
    "open" in the task list. Relative entries resolve against wherever the
    tasks root is now, so this conversion is the last time these paths need
    touching. Idempotent: a path that is already relative is under none of
    `previous_roots` either, so a second pass leaves it alone.
    """

    tasks = body.get("tasks")
    if not isinstance(tasks, list):
        return False
    changed = False
    for task in tasks:
        if not isinstance(task, dict):
            continue
        request = task.get("request")
        if isinstance(request, dict) and "output" in request:
            converted = _relative(request["output"], previous_roots)
            changed = changed or converted != request["output"]
            request["output"] = converted
        outputs = task.get("outputs")
        if isinstance(outputs, dict):
            for key, value in list(outputs.items()):
                converted = _relative(value, previous_roots)
                changed = changed or converted != value
                outputs[key] = converted
    return changed


def _history_has_stale_paths(history: Path, previous_roots: tuple[Path, ...]) -> bool:
    """Whether the history still records paths under a pre-move tasks root.

    Checked before any lock is taken, because it decides whether there is work
    at all: an installation whose tree is already moved must not defer forever
    just because a desktop app happens to be open.
    """

    body = _load_history(history)
    return body is not None and _convert(body, previous_roots)


def _rewrite_history(history: Path, previous_roots: tuple[Path, ...]) -> None:
    body = _load_history(history)
    if body is None or not _convert(body, previous_roots):
        return
    write_atomic(history, json.dumps(body, ensure_ascii=False, separators=(",", ":")))


def relocate(paths: AppPaths, log: Callable[[str], None]) -> bool:
    stray = paths.user_data / "tasks"
    if stray == paths.tasks:
        return True
    history = paths.user_data / "tasks.json"
    previous_roots = _previous_roots(paths)
    if not stray.is_dir() and not _history_has_stale_paths(history, previous_roots):
        return True
    stack = ExitStack()
    try:
        stack.enter_context(
            holding_activity_barrier(
                paths.user_data,
                legacy_active_lock=active_lock_path(paths.tasks),
                timeout=0,
            )
        )
        stack.enter_context(
            holding_lock(paths.user_data / "tasks.json.lock", timeout=0)
        )
    except (OSError, LockUnavailable):
        stack.close()
        # A desktop app is open. Its JobManager holds the history in memory and
        # writes the whole thing back, so anything we rewrite now would be
        # overwritten; and its running task is writing into the directory we
        # want to move. Defer -- the framework retries at the next start.
        log("检测到 FineSub 正在运行，任务产物迁移改到下次启动时进行。")
        return False
    with stack:
        return _relocate_when_idle(paths, stray, log)


def _relocate_when_idle(
    paths: AppPaths, stray: Path, log: Callable[[str], None]
) -> bool:
    """Move task outputs while activity starts and history writes are blocked."""

    history = paths.user_data / "tasks.json"
    previous_roots = _previous_roots(paths)
    if not stray.is_dir():
        # The tree is already at the big-data root and only the history lagged
        # behind -- a previous attempt died between the move and the rewrite.
        # Without this the next start sees nothing to move, records the
        # migration as done, and leaves every recorded output pointing at a
        # directory that no longer exists. Nothing self-heals it later either:
        # writes convert paths relative to the *new* root, which these are not
        # under.
        _rewrite_history(history, previous_roots)
        log(f"任务产物此前已迁出 {stray}，本次补写历史中的路径。")
        return True
    if paths.tasks.is_dir() and any(paths.tasks.iterdir()):
        log(
            f"发现两处任务产物：{stray} 与 {paths.tasks}，未自动合并。"
            "任务目录以任务 id 命名，可直接把子目录并到一起。"
        )
        return False
    if paths.tasks.exists():
        paths.tasks.rmdir()
    # History first: it is a pure path-prefix rewrite that does not need the
    # directory to exist, and doing it here means a crash before the move is
    # retried (the source is still there) rather than recorded as finished.
    _rewrite_history(history, previous_roots)
    move_tree(stray, paths.tasks)
    log(f"任务产物已从 {stray} 迁移到 {paths.tasks}")
    return True


# Install-scoped, like 0001 and 0002. The source is the shared user-data, so
# "once any installation has emptied it there is nothing left to find" looks
# right -- but it assumes the source can only shrink, and 0002 *creates* it:
# it moves a portable copy's whole user-data, `tasks/` included, into the
# shared location. Recording this per user let an installation that had no
# tasks mark it done for everyone, and the portable copy that arrived later
# then kept its outputs inside user-data forever -- exactly the unbounded
# growth this migration exists to end. Re-checking costs one `is_dir()`.
MIGRATION = Migration(id=MIGRATION_ID, run=relocate, scope="install")
