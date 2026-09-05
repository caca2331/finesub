"""Make sure one Hugging Face model is on disk before the stage that needs it.

The desktop has had mirror routing with a per-model fallback since the download
work landed: `desktop_service` runs one prefetch subprocess per model, and
`model_fetch.fetch_with_fallback` gives it a second attempt against the official
source when the mirror is what failed. The CLI had none of it. It downloads
lazily -- whichever library owns the weights fetches them the first time a task
asks -- so a mirror having a bad afternoon produced a failed run rather than a
slower one, and the per-class failure counter never learned anything.

This closes that half without moving the download earlier. "Prepared ahead of
time" and "fetched when the run reaches it" are two moments on one path: call
`ensure_hf_model` at the entry of the stage that loads the weights, where it
costs a `stat` when they are already there. That is where
`separation.place_separator_files()` already sits for the separator's own
files; this is the same shape for the two Hugging Face repositories.

**A subprocess is not fastidiousness.** `huggingface_hub` reads `HF_ENDPOINT`
when it is imported, so a fallback that only rewrites the environment inside
this process would re-run the attempt against the host that just failed. One
process per attempt is what makes the second attempt mean anything -- the same
reason the desktop spawns one.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Mapping

from . import model_fetch
from .hf_verify import (
    MISMATCH_MARKER,
    VerificationMismatch,
    files_present,
    marker_state,
    pinned_snapshot_present,
    verify_and_mark,
)
from .model_caches import (
    _ENSURABLE_HF_CACHE_DIRS,
    default_hf_home,
    existing_hf_home,
    managed_model_dirs,
)
from .model_manifest import entry_for


def _hub_dir(models_root: Path | None) -> Path:
    """The cache root a download will actually write into.

    The environment wins: a worker context resolved cache reuse already, and
    re-deriving it here could name a different directory than the one the
    download used -- which is how a freshly fetched model gets verified in the
    wrong place and reported missing.

    Read live, because the reader is the download *subprocess*, which imports
    `huggingface_hub` after this process hands it an environment. For the other
    question -- where a loader in *this* process will look -- see
    `_loader_hub_dir`.

    Three variables and not one: `huggingface_hub` still honours the legacy
    `HUGGINGFACE_HUB_CACHE`, and expands `~` and `$VAR` in all of them. Reading
    only `HF_HUB_CACHE` literally sent the verification to a directory the
    download had never written to.
    """

    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        configured = os.environ.get(name)
        if configured:
            return _expand(configured)
    home = os.environ.get("HF_HOME")
    if home:
        return _expand(home) / "hub"
    if models_root is not None:
        managed_hf, _ = managed_model_dirs(models_root)
        return existing_hf_home(managed_hf) / "hub"
    return default_hf_home() / "hub"


def _loader_hub_dir() -> Path:
    """The cache root a loader in this process will read.

    Asked of `huggingface_hub` rather than re-derived from the environment.
    The library resolves those variables once, at import, and freezes the
    answer into `constants.HF_HUB_CACHE`; every loader then reads that constant
    and nothing else. So this is not merely a tidier way to spell `_hub_dir` --
    it is the only spelling that stays true when a variable is set after the
    library was imported, and the only one that cannot drift from the library's
    own rules.

    Falls back to the plain rules when the library is absent: a `[harness]`-only
    install has no Hugging Face weights to load, and nothing will ask.
    """

    try:
        from huggingface_hub import constants

        # Normalised the way every download entry point normalises it
        # (`Path(cache_dir).expanduser().resolve()`), because the constant is
        # not the last word: `constants.py` expands `~` *before* `$VAR`, so a
        # variable that itself resolves to `~\...` leaves a literal tilde in
        # there for the download to expand later.
        return Path(constants.HF_HUB_CACHE).expanduser().resolve()
    except Exception:  # noqa: BLE001 - absent, or too old to say
        return _hub_dir(None)


def _expand(value: str) -> Path:
    """`~` and `$VAR` in a cache variable, the way the hub expands them."""

    return Path(os.path.expandvars(value)).expanduser().resolve()


def verify_downloaded(model_id: str, *, models_root: Path | None = None) -> None:
    """Hash a repository this process just fetched, and mark it verified.

    Both front ends call this and neither should own a second copy: the
    desktop's prefetch worker fetches through the libraries, the CLI's stage
    entry fetches through `ensure_hf_model`, and once the bytes are on disk the
    question is the same one. Raising is deliberate -- reporting weights ready
    and letting the first task discover otherwise is the failure this replaces.

    A model the manifest does not describe passes silently; there is nothing to
    compare it against.
    """

    entry = entry_for(model_id)
    cache_dir = _ENSURABLE_HF_CACHE_DIRS.get(model_id)
    if entry is None or cache_dir is None:
        return
    failed = verify_and_mark(_hub_dir(models_root), cache_dir, entry)
    if failed:
        raise VerificationMismatch(
            f"{model_id} 下载后校验失败：{', '.join(failed)}（{MISMATCH_MARKER}）"
        )


def pinned_revision(model_id: str) -> str | None:
    """The revision the manifest pins for `model_id`, if it names one.

    For the loaders: a model that was just ensured and verified at one revision
    must be loaded at that same revision, or Hugging Face may re-resolve `main`
    to something else entirely -- silently undoing the verification.
    """

    entry = entry_for(model_id)
    return (entry.revision or None) if entry is not None else None


def pinned_snapshot_loadable(model_id: str) -> bool:
    """Whether a loader can reach `model_id`'s pinned snapshot without the network.

    The question `local_files_only=True` has to answer before it is safe to
    pass: it is the difference between "skip the hub round trips" and "fail
    where a download would have worked". `ensure_hf_model` having returned is
    *not* that answer -- it writes into the root `_hub_dir` picks, which for a
    managed install with no `HF_*` in the environment is this install's own
    cache while the loader reads the conventional one.

    So the root asked about here is `_loader_hub_dir` -- the one the library
    itself resolved -- and not `_hub_dir`, which answers where a download will
    land.

    Three facts, one more than the fast path above: it also stats the files the
    manifest lists. `_hf_repo_complete` deliberately accepts a snapshot whose
    big file is gone (deleted to reclaim the space, or never linked), and that
    is the one state where an offline load fails in a way no retry can safely
    be keyed on -- CTranslate2 raises a bare `RuntimeError`, indistinguishable
    from the CUDA failures that must not be retried. Cheap, because the check
    is a `stat` each and not the digest the verification path takes.

    And one marker state, but only one. "Stale" and "absent" are reasons to
    re-fetch, never reasons to go to the network for a file list -- that was
    the whole argument for ignoring the marker here. **"Failed" is not in that
    family**: it is positive evidence that the last verification of this very
    snapshot did not pass, and `_discard_failed` is explicitly best effort --
    it says so, and it says the failure marker is what fences off whatever it
    could not delete. A file locked by another process stays behind at the
    manifest's own size, so the three facts above all answer yes and the
    loader is handed bytes we already know are wrong. Whatever `_discard_failed`
    could not remove, this has to refuse.

    ⚠ The marker is read at `_loader_hub_dir()` -- **the root this function
    stats**, not the `_hub_dir(models_root)` that `verify_and_mark` writes to.
    On a managed install with no `HF_*` in the environment those are different
    directories, and asking the download root about the loader root's snapshot
    is a category error either way: the question here is whether *this* root
    can be loaded offline. A download that failed verification somewhere else
    is `prepare`'s problem, and it raises for it.
    """

    entry = entry_for(model_id)
    cache_dir = _ENSURABLE_HF_CACHE_DIRS.get(model_id)
    if entry is None or cache_dir is None:
        return False
    from .model_caches import _hf_repo_complete

    hub = _loader_hub_dir()
    return (
        marker_state(hub, cache_dir, entry) != "failed"
        and _hf_repo_complete(hub, cache_dir)
        and pinned_snapshot_present(hub, cache_dir, entry)
        and files_present(hub, cache_dir, entry)
    )


def _download(model_id: str, environment: Mapping[str, str]) -> None:
    """Run one download attempt in its own interpreter."""

    process = subprocess.run(
        [sys.executable, "-m", __name__, model_id],
        # Both halves of the pipe are pinned to UTF-8. A piped child picks its
        # stdio encoding from the locale, so on a cp936 machine the default
        # would have the child writing cp936 while `text=True` here decodes
        # with the same code page -- fine until a model id or a hub error
        # carries a character it cannot represent, and then the failure is a
        # decode traceback instead of the download error it was reporting.
        env={**os.environ, **environment, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{model_id} download failed: "
            f"{(process.stderr or process.stdout).strip()[-500:]}"
        )


def ensure_hf_model(
    model_id: str,
    *,
    data_root: Path,
    models_root: Path | None = None,
    log: Callable[[str], None] | None = None,
    download: Callable[[str, Mapping[str, str]], None] | None = None,
) -> None:
    """Fetch `model_id` if it is not already present, mirror first.

    A no-op -- one directory check -- when the weights are there, which is the
    common case and the reason this can sit on a hot path. Nothing is raised
    for a model the manifest does not describe: an unlisted model keeps the
    behaviour it has always had, which is that its own library fetches it.
    """

    entry = entry_for(model_id)
    cache_dir = _ENSURABLE_HF_CACHE_DIRS.get(model_id)
    if entry is None or cache_dir is None:
        return
    hub = _hub_dir(models_root)
    from .model_caches import _hf_repo_complete

    # "Present" means the revision the manifest pins, not any revision -- the
    # loaders are handed that pin and would fetch it lazily, outside the mirror
    # routing this exists to provide. A marker saying "stale" or "failed" also
    # voids the fast path; "absent" does not, because a cache that predates the
    # mechanism is not evidence of damage.
    if (
        _hf_repo_complete(hub, cache_dir)
        and pinned_snapshot_present(hub, cache_dir, entry)
        and marker_state(hub, cache_dir, entry) in ("absent", "current")
    ):
        return
    if log is not None:
        log(f"正在获取模型 {model_id}…")
    from .download_routes import resolve_region

    fetch = download or _download

    def fetch_and_verify(environment: Mapping[str, str]) -> None:
        fetch(model_id, environment)
        # Inside the attempt on purpose: a mirror that answers success with
        # wrong bytes is only discovered here, and a verification that runs
        # after the fallback has returned can no longer trigger it. The
        # official attempt passes through this too -- its bytes get no more
        # trust than the mirror's.
        verify_downloaded(model_id, models_root=models_root)

    model_fetch.fetch_with_fallback(
        fetch_and_verify,
        base_environment={},
        data_root=data_root,
        region=resolve_region(data_root).region,
        is_retryable=model_fetch.is_mirror_failure,
    )


def main(argv: list[str] | None = None) -> int:
    """The subprocess half: fetch exactly one repository, then exit.

    Deliberately thin, and deliberately not importing anything from `finesub`:
    the repository and revision come from the manifest, which is the same
    source the verification and the desktop's prefetch read.
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m finesub_bootstrap.model_ensure <model-id>", file=sys.stderr)
        return 2
    entry = entry_for(arguments[0])
    if entry is None or not entry.repo:
        print(f"unknown model id: {arguments[0]}", file=sys.stderr)
        return 2
    from huggingface_hub import snapshot_download

    snapshot_download(entry.repo, revision=entry.revision or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
