"""Local MOSS (`--asr moss-local`): open MOSS-Transcribe-Diarize on this machine's GPU, with run-time rescues for
its known failures, as one more joint-ASR backend; and the same model choosing only the speakers of another
backend's lines (`--asr moss-diarization-aligner`, `label_rows`, the decode in `forced`).

The source's mix -> ffmpeg 16 kHz float -> `rescue.transcribe` (parts of at most 10 minutes or what the VRAM budget holds, rescues, stitching)
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TypeVar

import numpy as np

from .. import GPU_LOCK, MOSS_DIAR_ALIGNER, MOSS_LOCAL
from ...runtime.gpu_stage_gate import clear_frames
from ..model import Transcript, Unit, segment_turns
from ..mosspro import SpeechSpans

#: What the metadata records as the model; the revision goes beside it.
MODEL_NAME = "moss-transcribe-diarize"

T = TypeVar("T")


def transcribe(source: Path, *, language: str | None, duration: float, gpu_tier: str | None,
               device: str | None, speech: SpeechSpans | None = None) -> tuple[Transcript, dict[str, Any]]:
    """(transcript, what the metadata records about the run). `speech`: the source's samples and Silero's verdicts,
    shared with the stage's line timing. Raises `JointAsrError` when this run cannot use local MOSS (no suitable
    card, `--gpu-tier cpu`, `--device cpu`, no usable transformers)."""
    from . import rescue

    budget_gib, plan = _prepare(MOSS_LOCAL, gpu_tier, device)
    if not language:
        print("Warning: --asr moss-local without --language: the default (Chinese) prompt, and no check that the "  # product output
              "transcript is in the source's language; an English source may come out in another language. "
              "Give --language.", file=sys.stderr)
    audio = _decode(source)
    speech = speech or SpeechSpans(source)
    result = _run(plan, budget_gib, lambda model, processor: rescue.transcribe(model, processor, audio, language,
                                                                              budget_gib, speech))
    units = [Unit.from_dict(s) for s in result.segments if s["end"] > s["start"]]
    transcript = Transcript(MOSS_LOCAL, MODEL_NAME, duration, "segment", segment_turns(units))
    return transcript, {"revision": plan.revision, "kv": result.passes, "events": event_counts(result.events)}


@dataclass
class Labels:
    """`label_rows`'s answer, per line in the order given: its speaker (`S01`, `S02`, ... in order of first line,
    after the `w<k>:` of the part it was first heard in when the source has several parts; `UNKNOWN_SPEAKER` for a
    line no part held) and margin (None: one candidate, or no part)."""
    speakers: list[str]
    margins: list[float | None]
    #: the parts as decoded (`forced.Part.as_dict`), the voice merges, the lines no part held
    parts: list[dict]
    merges: list[tuple[str, str, float]]
    unlabelled: int
    revision: str


def label_rows(source: Path, lines: Sequence[dict], *, language: str | None, gpu_tier: str | None,
               device: str | None, speech: SpeechSpans) -> Labels:
    """The speakers local MOSS gives `lines` (absolute seconds, in time order; `start`, `end`, `text`) over the
    source: each part's forced decode (`forced.label`), stitched and merged by voice as `moss-local`'s parts are
    (`relink.linked`), each line's label and margin from the part the hand-over keeps it from. Takes GPU_LOCK for
    the decode only. Raises `JointAsrError` as `transcribe` does."""
    from ..model import UNKNOWN_SPEAKER

    budget_gib, plan = _prepare(MOSS_DIAR_ALIGNER, gpu_tier, device)
    audio = _decode(source)
    spans = speech.spans()

    def run(model, processor):
        from ..relink import linked
        from . import forced
        from .rescue import PART_SECONDS

        parts = forced.label(model, processor, audio, lines, spans, language, budget_gib, PART_SECONDS)
        units = [(p.lo, p.hi, [{"row": i, "start": lines[i]["start"], "end": lines[i]["end"], "speaker": str(n),
                                "margin": forced.margin(c)} for i, n, c in p.picks])
                 for p in parts if p.picks]
        got, merges, _ = linked(units, segments=False, audio=speech.pcm, backend=MOSS_DIAR_ALIGNER, orphans="row")
        return parts, got, merges

    parts, got, merges = _run(plan, budget_gib, run)
    by_row = {u["row"]: u for u in got}
    names: dict[str, str] = {}
    speakers, margins = [], []
    for i in range(len(lines)):
        u = by_row.get(i)
        if u is None:
            speakers.append(UNKNOWN_SPEAKER)
            margins.append(None)
            continue
        # the part a label was first heard in stays on it (`stitch`'s `w<k>:`): the judge says which part it came from
        part = u["speaker"][:u["speaker"].index(":") + 1] if u["speaker"].startswith("w") else ""
        speakers.append(names.setdefault(u["speaker"], f"{part}S{len(names) + 1:02d}"))
        margins.append(u["margin"])
    return Labels(speakers, margins, [p.as_dict() for p in parts], merges,
                  sum(s == UNKNOWN_SPEAKER for s in speakers), plan.revision)


def unusable_reason(backend: str, gpu_tier: str | None, device: str | None) -> str | None:
    """Why this run cannot decode with local MOSS -- its card (`moss_local_unusable_reason`) or its transformers --
    as the sentence to stop on; None if it can. Asked when a task is built, before `moss-diarization-aligner`'s
    step 1 (so no API is paid for rows the decode could not label) and before the weights are fetched."""
    from ...runtime.device import moss_local_unusable_reason
    from . import decoder as D

    reason = moss_local_unusable_reason(gpu_tier, device, backend)
    if reason is None:
        try:
            D.check_transformers(backend)
        except RuntimeError as e:
            reason = str(e)
    return reason


def _prepare(backend: str, gpu_tier: str | None, device: str | None):
    """(VRAM budget in GiB, the weights' plan), or `JointAsrError` when this run cannot use local MOSS."""
    from ...runtime import hf_weights
    from ...runtime.resources import get_resource_profile
    from ..stage import JointAsrError
    from . import decoder as D

    reason = unusable_reason(backend, gpu_tier, device)
    if reason is not None:
        raise JointAsrError(reason)
    budget_gib = get_resource_profile(gpu_tier, device=device).usable_gib - D.CONTEXT_GIB
    return budget_gib, hf_weights.prepare(D.MANIFEST_ID)


def _run(plan, budget_gib: float, work: Callable[[Any, Any], T]) -> T:
    """`work(model, processor)` with the model loaded, holding GPU_LOCK and the allocator cap; everything put on the
    process undone after."""
    with GPU_LOCK, _allocator_capped(budget_gib), _triton_cache_in_checkout(), _quiet_transformers():
        return _with_model(plan, work)


def _with_model(plan, work: Callable[[Any, Any], T]) -> T:
    import torch

    from ..relink import VOICEPRINTS_CUDA
    from . import decoder as D

    loaded = None
    try:
        loaded = D.load(plan)
        return work(*loaded)
    except BaseException as e:
        # The traceback's frames -- of this exception and of every one in its chain -- hold the model and a pass's
        # K/V buffers until whoever catches it is done; cleared, the `finally` below frees them before the GPU lock
        # goes to the next user.
        clear_frames(e)
        raise
    finally:
        loaded = None
        VOICEPRINTS_CUDA.release()
        gc.collect()
        torch.cuda.empty_cache()


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

