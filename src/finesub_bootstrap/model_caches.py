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

#: Repositories the pipeline pulls from Hugging Face, as cache directory names.
HF_REPO_DIRS = (
    "models--Qwen--Qwen3-ASR-0.6B-hf",
    "models--deepdml--faster-whisper-large-v3-turbo-ct2",
    "models--Systran--faster-whisper-large-v3",
)


def default_hf_home() -> Path:
    """Where Hugging Face keeps its cache when nothing overrides it."""

    return Path.home() / ".cache" / "huggingface"


def default_separator_dir() -> Path:
    """Where audio-separator keeps its checkpoints when nothing overrides it."""

    return Path.home() / ".cache" / "audio-separator"


def existing_hf_home(managed: Path) -> Path:
    """The HF cache to use: the conventional one if it already has weights."""

    if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE"):
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
