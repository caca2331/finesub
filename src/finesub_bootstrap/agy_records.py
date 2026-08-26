"""The agy (Antigravity CLI) project records FineSub leaves behind.

agy registers a project by writing ``~/.gemini/config/projects/<uuid>.json``
itself (``--new-project``); FineSub only keeps the id. The location is agy's
implementation detail -- read off its changelog, not a documented contract --
so everything here treats it as foreign ground: a missing directory, a record
that is not JSON, or a ``folderUri`` that is not a ``file://`` path is simply
skipped, never an error.

Stdlib-only on purpose: the uninstaller runs on the launcher's interpreter
without the main package, and the main package's ``agent-clean`` shares the
same function so the path is defined exactly once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


def agy_project_records_dir() -> Path:
    return Path.home() / ".gemini" / "config" / "projects"


def _folder_paths(record: object) -> list[Path]:
    """Every ``file://`` folder a project record points at."""

    if not isinstance(record, dict):
        return []
    resources = record.get("projectResources")
    if not isinstance(resources, dict):
        return []
    folders: list[Path] = []
    for entry in resources.get("resources") or []:
        uri = entry.get("folderUri") if isinstance(entry, dict) else None
        if not isinstance(uri, str):
            continue
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            continue
        # agy writes `file://C:/x` (drive as host) on Windows and
        # `file:///C:/x` / `file:///home/x` elsewhere; both must resolve.
        raw = unquote((parsed.netloc or "") + parsed.path)
        # Keyed off the string, never off `os.name`: which spelling a record
        # holds depends on the machine that *wrote* it, so a host guard makes
        # a Windows-written record unreadable everywhere else. The trailing
        # test keeps a POSIX `/a:b` -- a legal filename -- from being mistaken
        # for a drive and turned into a relative path.
        if (
            len(raw) > 2
            and raw[0] == "/"
            and raw[1].isalpha()
            and raw[2] == ":"
            and raw[3:4] in ("", "/")
        ):
            raw = raw[1:]
        if raw:
            folders.append(Path(raw))
    return folders


def _is_under(path: Path, root: Path) -> bool:
    try:
        candidate = path.expanduser().resolve(strict=False)
        base = root.expanduser().resolve(strict=False)
    except OSError:
        return False
    try:
        return os.path.commonpath((str(candidate), str(base))) == str(base)
    except ValueError:
        return False


def remove_project_records_under(
    roots: Iterable[Path], *, records_dir: Path | None = None
) -> list[Path]:
    """Delete agy project records whose folders live under any of ``roots``.

    Called after those roots are gone: the records would otherwise point at
    directories that no longer exist, and agy has no unregister command.
    Ownership is decided by directory, not by the ids FineSub remembered, so
    records left by re-registration (a changed hook) are swept too. Returns
    the records removed; a missing records directory is an empty sweep.
    """

    directory = records_dir or agy_project_records_dir()
    bases = [Path(root) for root in roots]
    if not bases or not directory.is_dir():
        return []
    removed: list[Path] = []
    try:
        entries = sorted(directory.glob("*.json"))
    except OSError:
        return []
    for entry in entries:
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        folders = _folder_paths(record)
        if not folders or not all(
            any(_is_under(folder, base) for base in bases) for folder in folders
        ):
            continue
        try:
            entry.unlink()
        except OSError:
            continue
        removed.append(entry)
    return removed
