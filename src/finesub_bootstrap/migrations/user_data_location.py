"""Move a portable copy's `user-data` into the shared managed location.

Portable installs used to keep personal data beside the executable while
installed copies and the CLI kept it under `%LOCALAPPDATA%`. One user could
therefore end up with three knowledge bases and wonder which one the app was
reading. Personal data is small and irreplaceable, so it now lives in exactly
one place regardless of which front end opened it -- and the copy left inside
an old portable folder has to come along.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from finesub_bootstrap.fsops import move_tree
from finesub_bootstrap.migrations import Migration
from finesub_bootstrap.paths import AppPaths

MIGRATION_ID = "0002-user-data-to-managed-location"


def _has_content(directory: Path) -> bool:
    return directory.is_dir() and any(directory.iterdir())


def relocate(paths: AppPaths, log: Callable[[str], None]) -> bool:
    stray = paths.root / "user-data"
    if stray == paths.user_data or not _has_content(stray):
        return True
    if _has_content(paths.user_data):
        # Two sets of personal data, and merging settings, task history and two
        # knowledge git repositories is a content decision no migration can
        # make. Say so on every start until a human picks.
        log(
            f"发现两处个人数据：{stray} 与 {paths.user_data}，未自动合并。"
            "前者在安装目录内，请人工确认后合并或删除。"
        )
        return False
    if paths.user_data.exists():
        paths.user_data.rmdir()  # Empty, or the branch above would have taken it.
    move_tree(stray, paths.user_data)
    log(f"个人数据已从 {stray} 迁移到 {paths.user_data}")
    return True


# Install-scoped, same reason as 0001: the stray user-data is inside one
# installation's own directory.
MIGRATION = Migration(id=MIGRATION_ID, run=relocate, scope="install")
