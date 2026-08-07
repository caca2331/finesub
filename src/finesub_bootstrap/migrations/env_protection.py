"""Bind the API keys in the shared `.env` to the current Windows account.

Runs after the location migrations on purpose: personal data first settles
where it belongs, then its content is protected. The work itself lives in
``finesub_bootstrap.secrets``; this wrapper owns the migration semantics --
never fatal, retried while it returns False -- and the one-way messaging a
silent format change would otherwise lack.
"""

from __future__ import annotations

from collections.abc import Callable

from finesub_bootstrap import secrets
from finesub_bootstrap.migrations import Migration
from finesub_bootstrap.paths import AppPaths

MIGRATION_ID = "0004-protect-env-keys"


def protect(paths: AppPaths, log: Callable[[str], None]) -> bool:
    env_path = paths.user_data / ".env"
    if not secrets.available():
        # Not this machine's feature (no DPAPI): retrying every start would
        # never change the answer, so record the migration as done.
        log("本机不支持密钥保护（无 DPAPI），.env 保持明文。")
        return True
    for sibling in sorted(env_path.parent.glob(".env.*")):
        if sibling.suffix in {".lock", ".tmp"} or sibling == env_path:
            continue
        # Not ours to touch -- but a stale plaintext copy defeats the point.
        log(f"发现 {sibling.name}：它不会被加密，若是旧备份请自行删除。")
    return secrets.protect_env_file(env_path)


MIGRATION = Migration(id=MIGRATION_ID, run=protect)
