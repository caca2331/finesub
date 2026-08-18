"""Where a FineSub installation keeps things, and how it finds them again.

Three roots, by how the data behaves rather than by who wrote it:

* **data root** (`%LOCALAPPDATA%\\FineSub`) -- small and irreplaceable: settings,
  API keys, the knowledge base, task history. Identical for every end-user
  form, because one user's knowledge base should not depend on which front end
  opened it. This is also the anchor: everything else can be found from here.
* **install root** -- the application itself plus `runtime/`. Version-bound and
  private to one installation, so it never moves on its own and is never
  shared: two installs at different versions would rebuild each other's
  environment in turn.
* **big-data root** -- `models/`, `cache/`, `tasks/`, `agent-capsules/`.
  Defaults to the install root (a fresh install is self-contained) and may be
  pointed elsewhere, in which case several installations share one copy. Its location is recorded in
  `locations.json` beside the data root.

The record is an accelerator, never the source of truth: every location can be
re-derived from where the code is running, so a user who moves folders in
Explorer is recovered from rather than punished.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

from finesub_bootstrap.fsops import write_atomic
from finesub_bootstrap.locks import holding_lock

# Written next to the executable by the Inno Setup installer (and only by it).
# The updater preserves it across full updates; update payloads never ship it.
INSTALLED_MARKER_NAME = "installed.marker"

# Release packages put the pipeline sources at <root>/app/versions/<version>/,
# a tree that carries pyproject.toml and src/finesub and so is
# indistinguishable from a source checkout by content alone.
_APP_VERSIONS_LAYOUT = ("versions", "app")

LOCATIONS_NAME = "locations.json"
LOCATIONS_LOCK_NAME = "locations.lock"
# Marks a directory as a FineSub big-data store. Load-bearing: its presence is
# how a recorded location is judged still valid, and its absence is how "the
# user deleted or moved that folder" is detected.
STORE_MARKER_NAME = ".finesub-store.json"
BIG_DATA_NAMES = ("models", "cache", "tasks", "agent-capsules")
REGISTER_SCRIPT_NAME = "register-location.cmd"


@dataclass(frozen=True, slots=True)
class AppPaths:
    root: Path
    app: Path
    app_versions: Path
    app_current: Path
    runtime: Path
    data_root: Path
    user_data: Path
    big_data: Path
    models: Path
    cache: Path
    tasks: Path
    agent_capsules: Path
    logs: Path

    @classmethod
    def for_root(
        cls,
        root: Path,
        *,
        data_root: Path | None = None,
        big_data: Path | None = None,
    ) -> "AppPaths":
        """Lay out one installation.

        ``data_root`` defaults to the managed location shared by every end-user
        form; ``big_data`` defaults to the install root, which is what makes a
        fresh install self-contained.
        """

        resolved = root.expanduser().resolve()
        resolved_data = (
            data_root.expanduser().resolve()
            if data_root is not None
            else default_data_root()
        )
        resolved_big = (
            big_data.expanduser().resolve() if big_data is not None else resolved
        )
        user_data = resolved_data / "user-data"
        app = resolved / "app"
        return cls(
            root=resolved,
            app=app,
            app_versions=app / "versions",
            app_current=app / "current.json",
            runtime=resolved / "runtime",
            data_root=resolved_data,
            user_data=user_data,
            big_data=resolved_big,
            models=resolved_big / "models",
            cache=resolved_big / "cache",
            tasks=resolved_big / "tasks",
            agent_capsules=resolved_big / "agent-capsules",
            logs=user_data / "logs",
        )

    def with_big_data(self, big_data: Path) -> "AppPaths":
        resolved = big_data.expanduser().resolve()
        return replace(
            self,
            big_data=resolved,
            models=resolved / "models",
            cache=resolved / "cache",
            tasks=resolved / "tasks",
            agent_capsules=resolved / "agent-capsules",
        )


def default_data_root() -> Path:
    """The one place personal data lives, whichever front end is running."""

    local_app_data = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
    if local_app_data:
        return Path(local_app_data).expanduser().resolve() / "FineSub"
    return Path.home().resolve() / ".finesub"


def locations_path(data_root: Path) -> Path:
    return data_root / LOCATIONS_NAME


def read_locations(data_root: Path) -> dict:
    try:
        body = json.loads(locations_path(data_root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def recorded_big_data(data_root: Path) -> Path | None:
    value = read_locations(data_root).get("bigData")
    return Path(value) if isinstance(value, str) and value else None


def recorded_migration_source(data_root: Path) -> Path | None:
    """The root a relocate is moving *away* from, while it is still running.

    Present only between the moment `relocate` records the destination and the
    moment it finishes. A crash in between leaves it set, which is the point:
    both roots stay searchable until someone confirms the move is complete.
    """

    value = read_locations(data_root).get("migratingFrom")
    return Path(value) if isinstance(value, str) and value else None


def is_store(path: Path) -> bool:
    """Whether `path` is a big-data store we wrote (as opposed to a stray dir)."""

    return (path / STORE_MARKER_NAME).is_file()


def looks_like_store(path: Path) -> bool:
    """Whether `path` holds FineSub big data and nothing else.

    Adoption has to work on directories we never marked: data carried over
    from a release that predates the marker, or a folder a user assembled by
    moving `models` and `cache` out of an old installation. Requiring every
    entry to be one of ours is what keeps this from swallowing, say, an old
    installation directory -- that one also holds an application and a runtime.
    """

    if not path.is_dir():
        return False
    known = {*BIG_DATA_NAMES, STORE_MARKER_NAME, REGISTER_SCRIPT_NAME}
    entries = list(path.iterdir())
    return bool(entries) and all(entry.name in known for entry in entries)


def _anchor_exists(path: Path) -> bool:
    """Whether the volume holding `path` is present at all.

    Separates "the user deleted that folder" from "the drive is unplugged", so
    a temporarily offline portable install falls back for this run instead of
    permanently rewriting the record.
    """

    anchor = Path(path.anchor) if path.anchor else None
    return anchor is None or anchor.exists()


def resolve_big_data_root(
    root: Path,
    data_root: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> Path:
    """Decide the big-data root, judging the record as a whole.

    Deliberately root-level: with a single recorded location, letting one
    missing item (say `models`) redirect the record would drag `cache` and
    `tasks` along with it even though they are still where the record says.
    Individual items are filled in under whichever root wins here.
    """

    recorded = recorded_big_data(data_root)
    if recorded is None:
        return root
    if not _anchor_exists(recorded):
        # Unplugged, not deleted: use the install root for this run and leave
        # the record alone so reconnecting the drive restores it.
        if log is not None:
            log(f"数据目录所在磁盘当前不可用，本次改用 {root}：{recorded}")
        return root
    if not is_store(recorded):
        if log is not None:
            log(
                f"记录的数据目录已不存在：{recorded}。"
                "若你把它移动到了别处，请到新位置双击 register-location.cmd。"
            )
        return root
    return recorded


def load_app_paths(
    root: Path,
    *,
    data_root: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> AppPaths:
    """`AppPaths` for an installation, honouring the recorded big-data root.

    Read-only: resolving never writes the record. The record is updated only
    when something is actually stored (`ensure_store`), so merely looking at a
    broken install -- `finesub doctor` -- leaves the user room to re-register a
    folder they moved.
    """

    resolved_data = (
        data_root.expanduser().resolve()
        if data_root is not None
        else default_data_root()
    )
    big_data = resolve_big_data_root(
        root.expanduser().resolve(), resolved_data, log=log
    )
    paths = AppPaths.for_root(root, data_root=resolved_data, big_data=big_data)
    return _with_interrupted_move_fallback(paths, resolved_data, log=log)


def _with_interrupted_move_fallback(
    paths: AppPaths,
    data_root: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> AppPaths:
    """Fill items still sitting at the old root when a relocate did not finish.

    `relocate` records the destination *before* moving anything, so a crash
    part-way leaves some big-data directories at the new root and the
    rest at the old one. Resolving is deliberately root-level, but that only
    holds once one root owns everything; until the move is confirmed complete,
    each item is taken from wherever it actually is. Without this, an
    interrupted relocate looks like a fresh install -- FineSub re-downloads the
    models and the task history comes up empty while the real one sits at the
    old root with nothing pointing at it.
    """

    source = recorded_migration_source(data_root)
    if source is None or source == paths.big_data:
        return paths
    updates: dict[str, Path] = {}
    for name in BIG_DATA_NAMES:
        if not (paths.big_data / name).is_dir() and (source / name).is_dir():
            updates[name.replace("-", "_")] = source / name
    if not updates:
        return paths
    if log is not None:
        labels = (name.replace("_", "-") for name in sorted(updates))
        log(
            f"上次搬迁未完成：{', '.join(labels)} 仍在 {source}，"
            "本次从原位置读取。再次运行 `finesub relocate` 可以搬完。"
        )
    return replace(paths, **updates)


def ensure_store(paths: AppPaths, *, log: Callable[[str], None] | None = None) -> None:
    """Make the big-data root real, and record it if it is not recorded yet.

    Called before anything is stored, not while resolving: that is what keeps
    the record describing a directory that exists and that we put data in.
    """

    paths.big_data.mkdir(parents=True, exist_ok=True)
    marker = paths.big_data / STORE_MARKER_NAME
    if not marker.is_file():
        write_atomic(
            marker, json.dumps({"schemaVersion": 1, "kind": "finesub-store"})
        )
    _write_register_script(paths.big_data)
    if recorded_big_data(paths.data_root) == paths.big_data:
        return
    record_big_data(paths.data_root, paths.big_data, log=log, adopt=True)


def record_big_data(
    data_root: Path,
    big_data: Path,
    *,
    log: Callable[[str], None] | None = None,
    adopt: bool = False,
    migrating_from: Path | None = None,
) -> None:
    """Record where the big-data root is.

    ``adopt`` is for the "we are about to store something" path: another
    installation may have registered a store while we were deciding to create
    our own, and using theirs beats leaving several GB of duplicate downloads
    behind. An explicit `finesub relocate` passes False -- the user just said
    where they want it.

    ``migrating_from`` marks a move in flight: the destination becomes the
    record *before* any data is placed there, and the old root stays searchable
    until :func:`clear_migration_source` says the move finished. Recording
    afterwards was the other way round -- on the same volume `move_store` is a
    rename, so the source is released the instant it succeeds, and a crash
    before the record left `tasks/` (which the docs call irreplaceable) at a
    location nothing pointed at.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    with holding_lock(data_root / LOCATIONS_LOCK_NAME, timeout=30):
        # Re-read under the lock, so two first-time installs racing do not both
        # download several GB and then have one of the copies orphaned.
        recorded = recorded_big_data(data_root)
        if (
            adopt
            and recorded is not None
            and _anchor_exists(recorded)
            and is_store(recorded)
        ):
            if recorded != big_data and log is not None:
                log(f"已有另一处数据目录被登记，沿用它：{recorded}")
            return
        body = read_locations(data_root)
        body["schemaVersion"] = 1
        body["bigData"] = str(big_data)
        if migrating_from is not None and migrating_from != big_data:
            body["migratingFrom"] = str(migrating_from)
        else:
            body.pop("migratingFrom", None)
        write_atomic(
            locations_path(data_root),
            json.dumps(body, ensure_ascii=False, indent=2),
        )


def clear_migration_source(data_root: Path) -> None:
    """Declare the recorded move complete, so resolving goes root-level again."""

    with holding_lock(data_root / LOCATIONS_LOCK_NAME, timeout=30):
        body = read_locations(data_root)
        if body.pop("migratingFrom", None) is None:
            return
        write_atomic(
            locations_path(data_root),
            json.dumps(body, ensure_ascii=False, indent=2),
        )


# ASCII only, and written with CRLF below: cmd.exe reads a batch file in the
# console code page, so non-ASCII text renders as mojibake on most machines,
# and LF-only line endings make it garble every second line.
_REGISTER_SCRIPT = """\
@echo off
rem Register this folder as FineSub's data directory.
rem After moving it somewhere else in Explorer, double-click this once.
setlocal
set "STORE=%~dp0"
if not exist "%STORE%.finesub-store.json" (
    echo This folder is not a FineSub data directory.>&2
    pause
    exit /b 1
)
set "RECORD=%LOCALAPPDATA%\\FineSub"
if not exist "%RECORD%" mkdir "%RECORD%"
rem Drop the trailing backslash: it would escape the closing quote in the JSON.
set "STORE=%STORE:~0,-1%"
set "STORE=%STORE:\\=\\\\%"
> "%RECORD%\\locations.json" echo {"schemaVersion": 1, "bigData": "%STORE%"}
echo Registered: %~dp0
pause
"""


def _write_register_script(store: Path) -> None:
    script = store / REGISTER_SCRIPT_NAME
    body = _REGISTER_SCRIPT.replace("\n", "\r\n")
    if script.is_file() and script.read_bytes() == body.encode("utf-8"):
        return
    script.write_bytes(body.encode("utf-8"))


def packaged_app_root(source: Path) -> Path | None:
    """The install root shipping `source`, if it is a packaged app snapshot.

    Lets code that only knows where it is imported from tell a release package
    apart from a source checkout -- the pipeline resolves personal-data paths
    that way when it runs outside the launcher, which would otherwise walk up
    into ``app/versions/<version>`` and write there. That directory is replaced
    wholesale by the next update, so anything landing in it is lost silently.
    """

    resolved = source.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        parent = candidate.parent
        if (parent.name, parent.parent.name) == _APP_VERSIONS_LAYOUT:
            return parent.parent.parent
    return None


def packaged_app_paths(source: Path) -> AppPaths | None:
    """`AppPaths` of the install shipping `source`; None outside a package."""

    root = packaged_app_root(source)
    if root is None:
        return None
    return load_app_paths(root)
