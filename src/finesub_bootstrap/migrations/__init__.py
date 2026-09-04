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
import os
from pathlib import Path

from finesub_bootstrap.fsops import RECORD_REPLACE, replace_path, write_atomic
from finesub_bootstrap.locks import LockUnavailable, holding_lock
from finesub_bootstrap.paths import AppPaths

LOGGER = logging.getLogger(__name__)

# Beside user-data, never inside it -- the same rule the lock below already
# followed. The ledger used to live *in* user-data, which made migration 0002
# ("bring a portable copy's personal data along") see FineSub's own bookkeeping
# as "the user already has data here" and refuse forever: the first
# installation to complete any migration poisoned it for every other one.
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
    # "user" once per person, "install" once per installation. The ledger lives
    # in the shared user-data, so a migration that fixes something *inside one
    # installation* must be recorded per installation -- otherwise whichever
    # front end starts first marks it done for everyone, and the installation
    # that actually holds the misplaced data skips it forever.
    scope: str = "install"


def _default_log(message: str) -> None:
    LOGGER.info("%s", message)


def _ledger_path(paths: AppPaths) -> Path:
    return paths.data_root / LEDGER_NAME


def _legacy_ledger_path(paths: AppPaths) -> Path:
    """Where builds before 2026-08-07 kept it: inside the tree 0002 moves."""

    return paths.user_data / LEDGER_NAME


def _adopt_legacy_ledger(paths: AppPaths) -> None:
    """Carry an older build's applied-set out of user-data, once.

    Leaving the old copy behind would keep 0002 blocked, which is the whole
    reason the file moved; losing it would re-run every migration. Migrations
    are re-entrant, so a failure here is not fatal -- worst case the set is
    rebuilt on the next start.
    """

    legacy = _legacy_ledger_path(paths)
    if not legacy.is_file():
        return
    try:
        if not _ledger_path(paths).is_file():
            _ledger_path(paths).parent.mkdir(parents=True, exist_ok=True)
            replace_path(legacy, _ledger_path(paths), budget=RECORD_REPLACE)
        else:
            legacy.unlink()
    except OSError:
        LOGGER.debug("could not adopt the legacy migration ledger", exc_info=True)


def _install_key(paths: AppPaths) -> str:
    return os.path.normcase(str(paths.root))


def _read_ledger(paths: AppPaths) -> dict:
    try:
        body = json.loads(_ledger_path(paths).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def applied_ids(paths: AppPaths) -> set[str]:
    """Ids already applied *for this installation*.

    The union of what was done for this user and what was done for this
    installation -- the two buckets exist because one shared ledger now serves
    several installations.
    """

    body = _read_ledger(paths)
    recorded = body.get("applied")
    ids = set(recorded) if isinstance(recorded, list) else set()
    installs = body.get("installs")
    if isinstance(installs, dict):
        per_install = installs.get(_install_key(paths))
        if isinstance(per_install, list):
            ids |= set(per_install)
    return ids


def _record(paths: AppPaths, migration: Migration) -> None:
    ledger = _ledger_path(paths)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    body = _read_ledger(paths)
    if migration.scope == "user":
        recorded = body.get("applied")
        body["applied"] = sorted(
            (set(recorded) if isinstance(recorded, list) else set())
            | {migration.id}
        )
    else:
        installs = body.get("installs")
        installs = installs if isinstance(installs, dict) else {}
        key = _install_key(paths)
        existing = installs.get(key)
        installs[key] = sorted(
            (set(existing) if isinstance(existing, list) else set())
            | {migration.id}
        )
        body["installs"] = installs
    write_atomic(ledger, json.dumps(body, ensure_ascii=False, indent=2))


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
    # Before the applied-set is read for the first time, and before 0002 looks
    # at user-data: an older build's ledger is still sitting in there.
    _adopt_legacy_ledger(paths)
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
        _record(paths, migration)
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
