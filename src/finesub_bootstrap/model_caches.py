"""Prefer model weights the machine already has over downloading them again.

The launchers point the pipeline at a managed ``models`` directory so an
uninstall can take everything with it. The cost is that a machine which has
already downloaded these weights -- through a source checkout, another tool, or
an earlier install -- pays for them a second time, and they are gigabytes.

So: look in the conventional cache first, and only fall back to the managed
directory. Granularity differs by family, and the difference is not cosmetic:

* the separator is one named checkpoint, so the check is exact -- that file, in
  that directory.
* Hugging Face keeps a single content-addressed cache root and offers no way to
  search several. The decision is therefore per-cache, not per-model: if the
  conventional root already holds one of the repositories this pipeline uses,
  it is used for all of them, including anything downloaded later.
"""

from __future__ import annotations

import os
from pathlib import Path


#: The separator checkpoint, named here rather than in separation.py so a
#: path lookup does not have to import torch to learn a filename.
SEPARATOR_CHECKPOINT = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"

QWEN_REFEREE_CACHE_DIR = "models--Qwen--Qwen3-ASR-0.6B-hf"
# Keep the repository id single-sourced with the prefetch worker. The managed
# runtime pins faster-whisper 1.2.1, whose ``large-v3-turbo`` alias resolves to
# this repository; checking the old deepdml cache name meant a successful
# prefetch was reported missing again at the next launch.
WHISPER_REPO_ID = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
WHISPER_CACHE_DIR = f"models--{WHISPER_REPO_ID.replace('/', '--')}"
#: Only a CLI run with an explicit --model reaches this one.
LEGACY_WHISPER_CACHE_DIR = "models--Systran--faster-whisper-large-v3"
#: The Japanese-specialised alternative, a large-v3 finetune converted to
#: CTranslate2. Also only reachable through an explicit ``--model``: it is a
#: listed alternative, not part of the default roster, so no default run
#: downloads it (it is absent from `PIPELINE_MODEL_IDS` on purpose).
WHISPER_JA_REPO_ID = "TransWithAI/whisper-ja-1.5B-ct2"
WHISPER_JA_CACHE_DIR = f"models--{WHISPER_JA_REPO_ID.replace('/', '--')}"

#: Repositories the pipeline pulls from Hugging Face, as cache directory names.
#: Every repository any `--model` can reach, not just the default roster: this
#: tuple answers "does the conventional cache already hold weights of ours",
#: and a machine whose only copy is the Japanese model should reuse that cache
#: exactly as one whose copy is the default.
HF_REPO_DIRS = (
    QWEN_REFEREE_CACHE_DIR,
    WHISPER_CACHE_DIR,
    LEGACY_WHISPER_CACHE_DIR,
    WHISPER_JA_CACHE_DIR,
)

#: Weights a default desktop run ends up downloading, in the order it reaches
#: them. Ids rather than paths: a front end turns them into labels, and the
#: prefetch entry point maps each to the library call that fetches it -- the
#: identity of a model belongs to the code that loads it, not to a second list.
#:
#: The referee is here because `qwen_verify` defaults to "auto", which means
#: "run whenever transformers 5.x imports" -- and the managed runtime always
#: has it. It is not opt-in on the desktop, whatever the CLI flag suggests.
PIPELINE_MODEL_IDS = ("separator", "whisper", "qwen-referee")

#: Manifest id -> cache directory, for every Hugging Face model
#: `model_ensure` can fetch and verify. Wider than `PIPELINE_MODEL_IDS`:
#: `whisper-ja` is never prefetched, but when an explicit `--model` does reach
#: it, it deserves the same mirror routing and pinned-revision verification as
#: the default weights rather than a bare lazy download.
_ENSURABLE_HF_CACHE_DIRS = {
    "whisper": WHISPER_CACHE_DIR,
    "whisper-ja": WHISPER_JA_CACHE_DIR,
    "qwen-referee": QWEN_REFEREE_CACHE_DIR,
}


def default_hf_home() -> Path:
    """Where Hugging Face keeps its cache when nothing overrides it."""

    return Path.home() / ".cache" / "huggingface"


def default_separator_dir() -> Path:
    """Where audio-separator keeps its checkpoints when nothing overrides it."""

    return Path.home() / ".cache" / "audio-separator"


#: Every variable `huggingface_hub` reads to place its cache, including the
#: legacy spelling it still honours. Checking only one of them reads as "the
#: user chose nothing" for a machine that chose quite deliberately.
HF_CACHE_VARS = ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")


def existing_hf_home(managed: Path) -> Path:
    """The HF cache to use: the conventional one if it already has weights."""

    if any(os.environ.get(name) for name in HF_CACHE_VARS):
        # The user pointed it somewhere on purpose; do not second-guess them.
        return managed
    conventional = default_hf_home()
    hub = conventional / "hub"
    if any((hub / name).is_dir() for name in HF_REPO_DIRS):
        return conventional
    return managed


def existing_separator_dir(managed: Path, checkpoint: str) -> Path:
    """The separator directory to use: the conventional one if it has the file."""

    if (managed / checkpoint).is_file():
        return managed
    conventional = default_separator_dir()
    if (conventional / checkpoint).is_file():
        return conventional
    return managed


def managed_model_dirs(models_root: Path) -> tuple[Path, Path]:
    """This install's own HF and separator directories, before cache reuse.

    The two names are decided by the launcher's worker environment
    (``HF_HOME``) and by ``finesub.paths.managed_separator_model_dir``;
    stated once here so a caller that only wants to *look* does not have to
    import either.
    """

    return models_root / "huggingface", models_root / "audio-separator"


def _hf_repo_complete(hub: Path, directory: str) -> bool:
    """Whether `directory` holds a finished snapshot rather than a partial one.

    Three cheap facts, no manifest parsing: the repository directory exists, no
    blob is still marked ``.incomplete``, and at least one snapshot revision
    holds something. Interrupting `snapshot_download` leaves the first of those
    true and the others not -- and treating that as "installed" told the user
    the weights were ready while their first task quietly spent minutes
    fetching the rest.

    The revision has to be non-empty, not merely present: the directory is
    created before the files are linked into it, so an interruption inside that
    window leaves an empty revision and no ``.incomplete`` blob yet.

    One window stays open, and is accepted: files download one after another,
    so an interruption *between* two files -- the previous one linked, the next
    one not yet started -- leaves nothing marked ``.incomplete`` and a
    non-empty revision, which this check calls complete. Closing it would take
    the remote file list, i.e. manifest parsing. The cost when it happens is a
    row that says ready while the first task quietly fetches the remainder --
    the pre-existing behaviour this check narrows, not a new failure.
    """

    repo = hub / directory
    if not repo.is_dir():
        return False
    if any((repo / "blobs").glob("*.incomplete")):
        return False
    return any(
        revision.is_dir() and any(revision.iterdir())
        for revision in (repo / "snapshots").glob("*")
    )


def missing_pipeline_models(models_root: Path) -> tuple[str, ...]:
    """Which of `PIPELINE_MODEL_IDS` a run would still have to download.

    Resolves through the same cache-reuse rules a task uses, so weights already
    sitting in the machine's shared caches count as present -- otherwise the
    desktop would offer to fetch three gigabytes it does not need.
    """

    managed_hf, managed_separator = managed_model_dirs(models_root)
    hub = existing_hf_home(managed_hf) / "hub"
    separator_dir = existing_separator_dir(managed_separator, SEPARATOR_CHECKPOINT)
    missing = []
    for model_id in PIPELINE_MODEL_IDS:
        if model_id == "separator":
            # One named file, downloaded whole: nothing partial to detect.
            present = (separator_dir / SEPARATOR_CHECKPOINT).is_file()
        else:
            cache_dir = _ENSURABLE_HF_CACHE_DIRS[model_id]
            present = _hf_repo_complete(hub, cache_dir) and not _marker_demands_refetch(
                hub, cache_dir, model_id
            )
        if not present:
            missing.append(model_id)
    return tuple(missing)


def _marker_demands_refetch(hub: Path, cache_dir: str, model_id: str) -> bool:
    """Whether a verification marker says these weights must be fetched again.

    Four states, two of which mean "fetch it again". A cache with no marker
    predates the mechanism or came from somewhere else, and is used as it
    always was -- absence is not evidence of damage, and re-downloading
    gigabytes to prove that would be its own bug. A marker naming the current
    manifest is the cheap yes. A marker naming an older one says this cache was
    verified against a revision the manifest has since re-pinned; one recording
    a failure says the last verification did not pass, and reporting that cache
    ready would hand a task weights already known to be wrong.
    """

    try:
        from .hf_verify import marker_state
        from .model_manifest import entry_for
    except ImportError:  # pragma: no cover - manifest is packaged with us
        return False
    entry = entry_for(model_id)
    if entry is None:
        return False
    return marker_state(hub, cache_dir, entry) in ("stale", "failed")
