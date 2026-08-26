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
    marker_state,
    pinned_snapshot_present,
    verify_and_mark,
)
from .model_caches import (
    _PIPELINE_HF_CACHE_DIRS,
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
    """

    configured = os.environ.get("HF_HUB_CACHE")
    if configured:
        return Path(configured)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    if models_root is not None:
        managed_hf, _ = managed_model_dirs(models_root)
        return existing_hf_home(managed_hf) / "hub"
    return default_hf_home() / "hub"


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
    cache_dir = _PIPELINE_HF_CACHE_DIRS.get(model_id)
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


def _download(model_id: str, environment: Mapping[str, str]) -> None:
    """Run one download attempt in its own interpreter."""

    process = subprocess.run(
        [sys.executable, "-m", __name__, model_id],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
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
    cache_dir = _PIPELINE_HF_CACHE_DIRS.get(model_id)
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
