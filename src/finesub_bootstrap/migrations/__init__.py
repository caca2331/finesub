"""One-way fixes to a user's data directory, each applied once per install.

The rest of the project carries no backward-compatibility burden: stale
artifacts are simply rerun. Personal data is the exception -- a knowledge base,
API keys and task history cannot be regenerated -- so when their location or
shape changes, something has to move them.

Migrations are identified, never versioned: the desktop app and the CLI share
one ``user-data`` tree, run different versions and skip releases, so "which ids
have already run" is the only question with a reliable answer. Each must be
safe to attempt again, and a migration that cannot finish safely says so
(returning False) instead of recording itself as done.

Nothing here may break startup: a failure is logged and retried next time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import json
import logging
from pathlib import Path

from finesub_bootstrap.locks import LockUnavailable, holding_lock
from finesub_bootstrap.paths import AppPaths

LOGGER = logging.getLogger(__name__)

LEDGER_NAME = ".migrations.json"
# Beside user-data, never inside it: migration 0002 moves that whole tree, and
# an open handle within a directory is exactly what stops Windows renaming it.
LOCK_NAME = ".migrations.lock"
# Long enough to outlast a migration that is actually running (they move data),
# short enough that a start-up never hangs on a stale peer.
LOCK_TIMEOUT_SECONDS = 120

LogCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class Migration:
    id: str
    run: Callable[[AppPaths, LogCallback], bool]


def _default_log(message: str) -> None:
    LOGGER.info("%s", message)


def _ledger_path(paths: AppPaths) -> Path:
    return paths.user_data / LEDGER_NAME


def applied_ids(paths: AppPaths) -> set[str]:
    try:
        body = json.loads(_ledger_path(paths).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    recorded = body.get("applied") if isinstance(body, dict) else None
    return set(recorded) if isinstance(recorded, list) else set()


def _record(paths: AppPaths, migration_id: str) -> None:
    ledger = _ledger_path(paths)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {"applied": sorted(applied_ids(paths) | {migration_id})},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )


def apply_pending(
    paths: AppPaths,
    *,
    log: LogCallback | None = None,
    migrations: Sequence[Migration] | None = None,
) -> list[str]:
    """Run whatever has not run yet; return the ids that completed.

    Serialized across processes: this runs at the start of every front end, so
    a desktop app that is already open and a CLI command started beside it
    would otherwise move the same trees at the same time.
    """

    report = log or _default_log
    pending = MIGRATIONS if migrations is None else migrations
    if all(migration.id in applied_ids(paths) for migration in pending):
        # The common case by far -- do not even open the lock file for it.
        return []
    try:
        with holding_lock(
            # The data root, not the install root: the whole point is to
            # serialize a desktop app against a CLI command started beside it,
            # and their install roots are different directories. The data root
            # is the one path they agree on -- and the one no migration moves.
            paths.data_root / LOCK_NAME,
            waiting_message="另一个 FineSub 进程正在迁移用户数据，等待它完成",
            log=report,
            timeout=LOCK_TIMEOUT_SECONDS,
        ):
            return _apply_locked(paths, pending, report)
    except LockUnavailable as error:
        # Never fatal: whoever holds the lock is doing the same work, and an
        # unfinished migration is retried at the next start.
        LOGGER.warning("Data migrations skipped: %s", error)
        return []


def _apply_locked(
    paths: AppPaths,
    pending: Sequence[Migration],
    report: LogCallback,
) -> list[str]:
    done: list[str] = []
    for migration in pending:
        # Re-read inside the lock: the process we waited for may have just
        # finished this very migration.
        if migration.id in applied_ids(paths):
            continue
        try:
            finished = migration.run(paths, report)
        except Exception as error:  # Startup must survive its own maintenance.
            LOGGER.exception("Data migration %s failed", migration.id)
            report(f"数据迁移 {migration.id} 未完成，稍后会重试：{error}")
            continue
        if not finished:
            continue
        _record(paths, migration.id)
        done.append(migration.id)
    return done


from finesub_bootstrap.migrations import env_protection  # noqa: E402
from finesub_bootstrap.migrations import knowledge_location  # noqa: E402
from finesub_bootstrap.migrations import tasks_location  # noqa: E402
from finesub_bootstrap.migrations import user_data_location  # noqa: E402

# Ordered: personal data reaches its final home before anything inside it is
# moved out again -- and only then is its content encrypted (0004).
MIGRATIONS: tuple[Migration, ...] = (
    user_data_location.MIGRATION,
    knowledge_location.MIGRATION,
    tasks_location.MIGRATION,
    env_protection.MIGRATION,
)
