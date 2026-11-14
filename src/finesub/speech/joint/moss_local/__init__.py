"""Local MOSS (`--asr moss-local`): open MOSS-Transcribe-Diarize on this machine's GPU, with run-time rescues for
its known failures, as one more joint-ASR backend.

The source's mix -> ffmpeg 16 kHz float -> `rescue.transcribe` (passes cut to the VRAM budget, rescues, stitching)
-> a segment `model.Transcript`, the shape MOSS Pro's takes, so the stage builds its lines the same way (S5b / S6b).
The model is loaded for the task and unloaded when the step ends, as the Whisper pool is. Everything this step does
to the process -- the allocator cap, the SDPA swap during prefill (`decoder`), the GPU lock -- is undone in
`finally`: a batch goes on to its next item in the same process after a failure.

Why it is built this way and how it compares with MOSS Pro: `docs/joint-asr.md`.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .. import GPU_LOCK, MOSS_LOCAL
from ..model import Transcript, Unit, segment_turns

#: What the metadata records as the model; the revision goes beside it.
MODEL_NAME = "moss-transcribe-diarize"


def transcribe(source: Path, *, language: str | None, duration: float, gpu_tier: str | None,
               device: str | None) -> tuple[Transcript, dict[str, Any]]:
    """(transcript, what the metadata records about the run). Raises `JointAsrError` when this run cannot use
    local MOSS (no suitable card, `--gpu-tier cpu`, `--device cpu`, no usable transformers)."""
    from ...runtime import hf_weights
    from ...runtime.device import moss_local_unusable_reason
    from ...runtime.resources import get_resource_profile
    from ..stage import JointAsrError
    from . import decoder as D

    reason = moss_local_unusable_reason(gpu_tier, device)
    if reason is not None:
        raise JointAsrError(reason)
    try:
        D.check_transformers()
    except RuntimeError as e:
        raise JointAsrError(str(e)) from e
    if not language:
        print("Warning: --asr moss-local without --language: the default (Chinese) prompt, and no check that the "  # product output
              "transcript is in the source's language; an English source may come out in another language. "
              "Give --language.", file=sys.stderr)
    budget_gib = get_resource_profile(gpu_tier, device=device).usable_gib - D.CONTEXT_GIB
    plan = hf_weights.prepare(D.MANIFEST_ID)
    audio = _decode(source)
    with GPU_LOCK, _allocator_capped(budget_gib), _triton_cache_in_checkout(), _quiet_transformers():
        result = _decode_with_model(plan, audio, language, budget_gib)
    units = [Unit.from_dict(s) for s in result.segments if s["end"] > s["start"]]
    transcript = Transcript(MOSS_LOCAL, MODEL_NAME, duration, "segment", segment_turns(units))
    return transcript, {"revision": plan.revision, "kv": result.passes, "events": event_counts(result.events)}


def _decode_with_model(plan, audio: np.ndarray, language: str | None, budget_gib: float):
    import torch

    from . import decoder as D
    from . import rescue

    loaded = None
    try:
        loaded = D.load(plan)
        return rescue.transcribe(*loaded, audio, language, budget_gib)
    except BaseException as e:
        # The traceback's frames -- of this exception and of every one in its chain -- hold the model and a pass's
        # K/V buffers until whoever catches it is done; cleared, the `finally` below frees them before the GPU lock
        # goes to the next user.
        _clear_chain(e)
        raise
    finally:
        loaded = None
        gc.collect()
        torch.cuda.empty_cache()


def _clear_chain(error: BaseException) -> None:
    seen: set[int] = set()
    todo: list[BaseException | None] = [error]
    while todo:
        e = todo.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        traceback.clear_frames(e.__traceback__)
        todo += [e.__cause__, e.__context__]


def event_counts(events: list[dict]) -> dict[str, int]:
    """How often each event kind came up, over every pass, part and gap window (their events nest): the rescues,
    plus the token budget running out (`cap`) and acoustic-event segments (`acoustic`, once per pass that had any).
    The passes' K/V records are left out (they go to `kv`)."""
    counts: Counter[str] = Counter()

    def walk(evs: list[dict]) -> None:
        for e in evs:
            if e["kind"] not in ("kv", "part"):
                counts[e["kind"]] += 1
            walk(e.get("events") or [])

    walk(events)
    return dict(sorted(counts.items()))


def _decode(source: Path) -> np.ndarray:
    """The whole source as 16 kHz mono float32: the input the model was measured on (through int16 it differs)."""
    from ....media.ffmpeg import decode_pcm16k

    return np.frombuffer(decode_pcm16k(source, sample_format="f32le"), dtype=np.float32).copy()


@contextlib.contextmanager
def _allocator_capped(budget_gib: float) -> Iterator[None]:
    """PyTorch's allocator held to the budget on the current card while local MOSS runs, then released: the cap is
    process-wide, and the stages after it (separation, the referee, the aligner) have budgets of their own."""
    import torch

    from ...runtime.device import total_vram_gib

    card = torch.cuda.current_device()
    total = total_vram_gib()
    torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gib / total) if total else 1.0, card)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, card)


@contextlib.contextmanager
def _quiet_transformers() -> Iterator[None]:
    """Without transformers' notice that the remote processor's `feature_extractor_class` is deprecated, and
    without its weight-loading bar: neither is something a user acts on."""
    import logging

    from transformers.utils import logging as hf_logging

    def keep(record: logging.LogRecord) -> bool:
        return "_class = '" not in record.getMessage()

    logger = logging.getLogger("transformers.processing_utils")
    logger.addFilter(keep)
    bar_was_on = hf_logging.is_progress_bar_enabled()
    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        logger.removeFilter(keep)
        if bar_was_on:
            hf_logging.enable_progress_bar()


@contextlib.contextmanager
def _triton_cache_in_checkout() -> Iterator[None]:
    """Triton's compiled kernels beside the other compile caches (a checkout's `cache/`) instead of `~/.triton`,
    unless the user set TRITON_CACHE_DIR; Triton keys the files by source, version and card itself. Only while this
    step runs: Inductor reads the same variable, and the separator's and the referee's kernels belong in their own
    keyed directories. Every Triton compile of local MOSS (the probes, each bucket's graph capture) is inside it."""
    if os.environ.get("TRITON_CACHE_DIR"):
        yield
        return
    from ....paths import resolve_cache_dir

    path = resolve_cache_dir() / "moss-local-triton"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    try:
        yield
    finally:
        os.environ.pop("TRITON_CACHE_DIR", None)

