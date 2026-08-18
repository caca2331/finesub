"""Move a knowledge base out of the app directory and into ``user-data``.

Running the pipeline against a package's own interpreter used to resolve the
knowledge root by walking up from the sources, which in a release package lands
in ``app/versions/<version>/knowledge``. The updater's preserved list is
``user-data``/``models``/``runtime``/``cache`` -- ``app`` is replaced wholesale
-- so the next update would delete that knowledge base without a word.

Resolution no longer goes there (see ``finesub.paths``), but installs
that already wrote one still hold the only copy.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from finesub_bootstrap.fsops import move_tree
from finesub_bootstrap.migrations import Migration
from finesub_bootstrap.paths import AppPaths

MIGRATION_ID = "0001-knowledge-out-of-app-versions"


def _strays(app_versions: Path) -> list[Path]:
    if not app_versions.is_dir():
        return []
    return sorted(
        version / "knowledge"
        for version in app_versions.iterdir()
        if (version / "knowledge").is_dir()
    )


def relocate(paths: AppPaths, log: Callable[[str], None]) -> bool:
    strays = _strays(paths.app_versions)
    if not strays:
        return True
    destination = paths.user_data / "knowledge"
    if destination.is_dir() and any(destination.iterdir()):
        # Two knowledge bases, and merging them is a content decision no
        # migration can make. Leave both and keep saying so until a human
        # picks: the stray is the one an update will delete.
        log(
            "发现两个知识库，未自动合并："
            f"{strays[0]} 与 {destination}。前者位于应用目录，"
            "下次更新会被删除，请人工确认后合并或删除。"
        )
        return False
    newest = max(strays, key=lambda stray: stray.stat().st_mtime)
    if destination.exists():
        destination.rmdir()  # Empty, or the branch above would have taken it.
    # Copy-verify-rename rather than shutil.move: installed copies keep the app
    # on any disk and user-data on C:, and a cross-volume move is copy-then-
    # delete -- interrupt it and the half-copy at the destination sends the
    # next start into the "two knowledge bases" branch above, turning one crash
    # into something a human has to untangle.
    move_tree(newest, destination)
    log(f"知识库已从 {newest} 迁移到 {destination}")
    for remaining in strays:
        if remaining != newest:
            log(f"应用目录下还有一份旧知识库未迁移：{remaining}")
    return True


# Install-scoped: the stray knowledge base sits inside one installation's own
# app/versions, so a CLI that has none of those must not mark this done for the
# desktop package that does.
MIGRATION = Migration(id=MIGRATION_ID, run=relocate, scope="install")
