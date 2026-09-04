"""Central resolution of checkout-relative runtime paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from finesub_bootstrap import token_counter


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
        and (path / "src" / "finesub").is_dir()
        # A release package ships the same two, so content alone cannot tell
        # them apart; only the surrounding layout can.
        and _packaged_root(path) is None
    )


def _packaged_root(source: Path) -> Path | None:
    from finesub_bootstrap.paths import packaged_app_root

    return packaged_app_root(source)


def _managed_user_data() -> Path | None:
    paths = _managed_paths()
    return None if paths is None else paths.user_data


def _managed_paths():
    """Layout of the managed installation on this machine, if there is one.

    The last resort for a bare wheel install: not a checkout, not inside a
    package, but still the same user's machine, so personal data belongs in the
    one place every front end uses rather than in a fifth invented location.
    """

    try:
        from finesub_bootstrap.paths import default_data_root, load_app_paths
    except ImportError:  # pragma: no cover - provisioning layer is optional
        return None
    data_root = default_data_root()
    return load_app_paths(data_root, data_root=data_root)


def _packaged_paths():
    """Directory layout of the install shipping this module, if any.

    Front ends normally hand these paths down through the environment, but the
    pipeline also gets run directly against a package's own interpreter -- when
    the launcher cannot start, that is the way out. Resolving the install
    ourselves keeps that path writing to the same knowledge base, `.env` and
    limiter state the launcher would have pointed at, instead of dropping them
    inside ``app/versions/<version>`` for the next update to delete.
    """

    from finesub_bootstrap.paths import packaged_app_paths

    return packaged_app_paths(Path(__file__))


def resolve_managed_app_paths():
    """Return the packaged/managed layout used by non-checkout processes.

    This deliberately exposes the same resolver used by settings and state so
    lightweight maintenance commands do not grow a second installation
    detector.
    """

    return _packaged_paths() or _managed_paths()


def _packaged_user_data() -> Path | None:
    paths = _packaged_paths()
    return None if paths is None else paths.user_data


def _main_worktree_root(root: Path) -> Path | None:
    """The main checkout behind a linked worktree, if `root` is one.

    A linked worktree's ``.git`` is a file reading
    ``gitdir: <main>/.git/worktrees/<name>``; the main checkout is three
    segments up from there (dropping ``<name>``, ``worktrees`` and ``.git``).
    Worktrees share the main checkout's knowledge base and settings on purpose
    -- otherwise every throwaway worktree would start an empty knowledge base,
    and this repository leans on worktree-isolated agents.
    """

    pointer = root / ".git"
    if not pointer.is_file():
        return None
    try:
        body = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not body.startswith("gitdir:"):
        return None
    recorded = Path(body.removeprefix("gitdir:").strip())
    if not recorded.is_absolute():
        recorded = (root / recorded).resolve()
    if (recorded.parent.name, recorded.parent.parent.name) != ("worktrees", ".git"):
        return None
    return recorded.parents[2]


def is_linked_worktree() -> bool:
    """Whether the checkout we resolved data from is a linked worktree."""

    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in _walk_up(start):
            if _is_checkout_root(candidate):
                return _main_worktree_root(candidate) is not None
    return False


def checkout_data_enabled() -> bool:
    """Whether a checkout keeps its own personal data (the default).

    Almost every run from a checkout is development: the API keys live in the
    repository's own ``.env``, and `docs/knowledge.md`, the run-audit skill and
    `examples/knowledge/` all assume the repository-local layout. Splitting a
    developer's knowledge base from the shared one is recoverable by merging
    later; mixing development noise into it is not.
    """

    configured = os.environ.get("FINESUB_CHECKOUT_DATA")
    if configured is None:
        return True
    return configured.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_checkout_root(explicit: str | Path | None = None) -> Path | None:
    """Resolve a source checkout without relying on package nesting depth."""

    configured = explicit or os.environ.get("FINESUB_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not _is_checkout_root(candidate):
            raise FileNotFoundError(
                "FineSub root must be a source checkout with "
                f"pyproject.toml/src/finesub: {candidate}"
            )
        return candidate

    seen: set[Path] = set()
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in _walk_up(start):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _is_checkout_root(candidate):
                main = _main_worktree_root(candidate)
                return main if main is not None and _is_checkout_root(main) else candidate
    return None


def _checkout_data_root() -> Path | None:
    """The checkout whose data we should use, honouring the opt-out."""

    return resolve_checkout_root() if checkout_data_enabled() else None


def resolve_env_file(explicit: str | Path | None = None) -> Path | None:
    """Return the configured/source-checkout ``.env`` path when it exists."""

    configured = explicit or os.environ.get("FINESUB_ENV_FILE")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    root = _checkout_data_root() or _packaged_user_data() or _managed_user_data()
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
    root = _checkout_data_root() or _packaged_user_data() or _managed_user_data()
    if root is None:
        return None
    candidate = root / "config.toml"
    return candidate if candidate.is_file() else None


def resolve_logs_dir() -> Path | None:
    """Where this front end keeps run logs, or None if it has no data root.

    Resolves exactly like `.env` and `config.toml`: an installed front end
    finds it under `user-data`, a source checkout finds it at the checkout
    root. Deliberately *not* under the run's own output directory -- the log
    has to start before that path is known (argument parsing, device
    resolution, download routing), and a run that dies before producing output
    is the one whose log is worth the most.
    """

    root = _checkout_data_root() or _packaged_user_data() or _managed_user_data()
    if root is None:
        return None
    from finesub_bootstrap.logs import log_directory

    return log_directory(root)


def resolve_model_catalog_override(explicit: str | Path | None = None) -> Path | None:
    """Return the data-root ``model_catalog.psv`` when the user wrote one.

    Model *facts* live in a catalog and their *composition* lives in
    ``config.toml`` (owner decision 2026-08-12). The packaged catalog ships
    with the code as the default; this is the optional override layered on top
    of it, and it resolves exactly like ``.env`` and ``config.toml`` do -- so
    an installed front end finds it under ``user-data`` while a source
    checkout, which has no separate user-data, finds it at the checkout root.
    """

    configured = explicit or os.environ.get("FINESUB_MODEL_CATALOG")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if candidate.is_file() else None
    root = _checkout_data_root() or _packaged_user_data() or _managed_user_data()
    if root is None:
        return None
    candidate = root / "model_catalog.psv"
    return candidate if candidate.is_file() else None


def resolve_state_file(explicit: str | Path | None = None) -> Path:
    """Return the machine-local state file shared by the cross-process limiters.

    A JSON document keyed by subsystem, not a directory -- see
    ``finesub.state`` for the locked read-modify-write that guards it.
    """

    configured = explicit or os.environ.get("FINESUB_STATE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = _checkout_data_root()
    if root is not None:
        return root / ".state"
    packaged = _packaged_paths()
    if packaged is not None:
        # Same file the launcher points at, so limits stay shared between a
        # task started from the app and one started against its interpreter.
        return packaged.cache / "state"
    managed = _managed_paths()
    if managed is not None:
        # Not a checkout and not inside a package: still the same machine, so
        # resolve the managed layout rather than inventing a fifth location.
        return managed.cache / "state"
    if os.environ.get("XDG_STATE_HOME"):
        return Path(os.environ["XDG_STATE_HOME"]).expanduser() / "finesub"
    return Path.home() / ".finesub" / "state"


def token_counter_candidates(
    explicit: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return configured and checkout-local token-counter candidates."""

    configured = explicit or token_counter.configured_path()
    if configured:
        return (Path(configured).expanduser().resolve(),)
    root = resolve_checkout_root()
    if root is None:
        return ()
    # One candidate, not two: the second used to be `bin/gemini-token-counter`,
    # which has never existed in this repository.
    return (root / token_counter.CHECKOUT_RELATIVE_PATH,)


def resolve_separator_model_dir(explicit: str | Path | None = None) -> Path:
    """Return where audio-separator stores model weights.

    ``FINESUB_MODEL_DIR`` (set by the CLI shell) keeps every model family under
    one uninstallable root; without it
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

    Lives here rather than in ``pipeline`` so a front end can resolve output
    paths without importing the ASR stack.
    """

    stem = name.strip()
    if not stem or "/" in stem or "\\" in stem or stem in {".", ".."}:
        raise ValueError(
            f"--name must be a bare name without path separators, got: {name!r}"
        )
    return Path("out") / stem / f"{stem}.srt"


def resolve_reference_data_root() -> Path:
    """Where downloaded source media and the URL->id map live.

    A checkout keeps ``data/reference``: hand-collected material a rerun never
    clears, exactly as the artifact tree documents it. An installed front end
    has no such directory, and resolving one relative to the working directory
    would put durable user data inside the runtime source snapshot that the
    next update replaces -- so it lands in ``user-data`` with everything else
    that has to survive.
    """

    root = _checkout_data_root()
    if root is not None:
        return root / "data" / "reference"
    user_data = _packaged_user_data() or _managed_user_data()
    if user_data is not None:
        return user_data / "reference"
    return Path("data") / "reference"


def resolve_knowledge_root(
    explicit: str | Path | None = None,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve knowledge storage without silently binding to an arbitrary CWD."""

    configured = explicit or os.environ.get("FINESUB_KNOWLEDGE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    root = _checkout_data_root()
    if root is not None:
        return root / "knowledge"
    user_data = _packaged_user_data() or _managed_user_data()
    if user_data is not None:
        return user_data / "knowledge"
    if required:
        raise RuntimeError(
            "Knowledge root is unavailable outside a source checkout; pass "
            "--knowledge-root or set FINESUB_KNOWLEDGE_ROOT."
        )
    return None
