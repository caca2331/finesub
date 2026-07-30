"""Central resolution of checkout-relative runtime paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def _walk_up(start: Path) -> Iterable[Path]:
    current = start.expanduser().resolve()
    if current.is_file():
        current = current.parent
    while True:
        yield current
        parent = current.parent
        if parent == current:
            return
        current = parent


def _is_checkout_root(path: Path) -> bool:
    return (
        (path / "pyproject.toml").is_file()
        and (path / "src" / "asr_playground").is_dir()
    )


def resolve_checkout_root(explicit: str | Path | None = None) -> Path | None:
    """Resolve a source checkout without relying on package nesting depth."""

    configured = explicit or os.environ.get("FINESUB_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not _is_checkout_root(candidate):
            raise FileNotFoundError(
                f"FineSub root is missing pyproject.toml/src/asr_playground: {candidate}"
            )
        return candidate

    seen: set[Path] = set()
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in _walk_up(start):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _is_checkout_root(candidate):
                return candidate
    return None


def resolve_env_file(explicit: str | Path | None = None) -> Path | None:
    """Return the configured/source-checkout ``.env`` path when it exists."""

    configured = explicit or os.environ.get("FINESUB_ENV_FILE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    root = resolve_checkout_root()
    if root is None:
        return None
    candidate = root / ".env"
    return candidate if candidate.is_file() else None


def resolve_config_file(explicit: str | Path | None = None) -> Path | None:
    """Return the configured/source-checkout ``config.toml`` when it exists."""

    configured = explicit or os.environ.get("FINESUB_CONFIG_FILE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    root = resolve_checkout_root()
    if root is None:
        return None
    candidate = root / "config.toml"
    return candidate if candidate.is_file() else None


def resolve_state_dir(explicit: str | Path | None = None) -> Path:
    """Return a stable directory for cross-process limiter state."""

    configured = explicit or os.environ.get("FINESUB_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = resolve_checkout_root()
    if root is not None:
        return root / ".state"
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]).expanduser() / "FineSub" / "state"
    if os.environ.get("XDG_STATE_HOME"):
        return Path(os.environ["XDG_STATE_HOME"]).expanduser() / "finesub"
    return Path.home() / ".finesub" / "state"


def token_counter_candidates(
    explicit: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return configured and checkout-local token-counter candidates."""

    configured = explicit or os.environ.get("GEMINI_TOKEN_COUNTER_EXE")
    if configured:
        return (Path(configured).expanduser().resolve(),)
    root = resolve_checkout_root()
    if root is None:
        return ()
    return (
        root / "bin" / "windows-amd64" / "tokcount.exe",
        root / "bin" / "gemini-token-counter",
    )


def resolve_knowledge_root(
    explicit: str | Path | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve knowledge storage without silently binding to an arbitrary CWD."""

    configured = explicit or os.environ.get("FINESUB_KNOWLEDGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = resolve_checkout_root()
    if root is not None:
        return root / "knowledge"
    if required:
        raise RuntimeError(
            "Knowledge root is unavailable outside a source checkout; pass "
            "--knowledge-root or set FINESUB_KNOWLEDGE_ROOT."
        )
    return None
