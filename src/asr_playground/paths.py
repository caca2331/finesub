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


def resolve_state_file(explicit: str | Path | None = None) -> Path:
    """Return the machine-local state file shared by the cross-process limiters.

    A JSON document keyed by subsystem, not a directory -- see
    ``asr_playground.state`` for the locked read-modify-write that guards it.
    """

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


def resolve_separator_model_dir(explicit: str | Path | None = None) -> Path:
    """Return where audio-separator stores model weights.

    ``FINESUB_MODEL_DIR`` (set by the desktop launcher, and later the CLI
    shell) keeps every model family under one uninstallable root; without it
    the historical shared user cache stays in effect so bare checkout runs
    keep reusing already-downloaded weights.
    """

    managed = managed_separator_model_dir(explicit)
    if managed is None:
        return Path.home() / ".cache" / "audio-separator"
    # A 610MB checkpoint the machine already holds is worth finding: the
    # managed directory is where new downloads go, not the only place to look.
    from finesub_bootstrap.model_caches import (
        SEPARATOR_CHECKPOINT,
        existing_separator_dir,
    )

    return existing_separator_dir(managed, SEPARATOR_CHECKPOINT)


def managed_separator_model_dir(
    explicit: str | Path | None = None,
) -> Path | None:
    """Where *this install* keeps separator artefacts, ignoring shared caches.

    Compiled acceleration packages belong here even when the weights are being
    read from a shared cache: they are keyed to one torch build and one GPU, so
    writing them into a directory other tools also use would leave artefacts
    nothing can attribute or clean up.
    """

    configured = explicit or os.environ.get("FINESUB_MODEL_DIR")
    if not configured:
        return None
    return Path(configured).expanduser().resolve() / "audio-separator"


def resolve_name_output_path(name: str) -> Path:
    """Map ``--name <stem>`` to out/<stem>/<stem>.srt.

    The stem names a directory under out/, so anything carrying a separator or
    a parent reference is rejected instead of silently escaping the tree.

    Lives here rather than in ``pipeline`` so the desktop worker can resolve
    output paths without importing the ASR stack.
    """

    stem = name.strip()
    if not stem or "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise ValueError(
            f"--name must be a bare name without path separators, got: {name!r}"
        )
    return Path("out") / stem / f"{stem}.srt"


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
