"""`--asr-aligner`: unit times for MOSS's long segments from Qwen3-ForcedAligner (S6b, `docs/joint-asr.md`).

Heavy: transformers (the `[asr]` extra), a 0.6B model from the Hugging Face hub (1.8 GB, revision pinned and
verified through `model-manifest.json`), about 2.4 GB of RAM on the CPU (float32) or seconds on a GPU (bf16). Only the segments over the splitter's ok limits are sent --
the ones the DP has to cut, about a quarter of the audio.

Each segment's audio, PAD seconds wider on each side but not into a neighbouring segment it does not overlap
(there the aligner put words into the next speaker's time), is aligned to the segment's units. The word list is
built here, not by the processor, whose Japanese split needs a dictionary model. A unit the aligner keeps
nothing of (a lone bracket) takes its neighbour's time.

The boundary logits are decoded by `ordered_classes`, not the processor's `decode_forced_alignment`: the most
likely boundary times under the constraints that no boundary goes back in time and every word lasts at least one
class (80 ms). The official decode takes each boundary's argmax and repairs disorder afterwards, which left
hundreds of zero-length words and needed two fallbacks to Silero (mostly zero-length, or a run faster than a
collapsed line); the ordered decode needs neither, and on every ruler it is at least as close
(`docs/joint-asr.md`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import GPU_LOCK
from .lines import units_of

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
#: Its entry in `finesub_bootstrap/model-manifest.json` (revision pinned, the LFS files' digests).
MANIFEST_ID = "qwen-aligner"
SR = 16000
PAD = 0.5
BATCH = 4


def _kept(unit: str) -> str:
    from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

    return "".join(ch for ch in unit if _is_kept_char(ch))


def _span(segments: Sequence[dict], seg: dict, total: float) -> tuple[float, float]:
    a, b = max(0.0, seg["start"] - PAD), min(total, seg["end"] + PAD)
    before = [x["end"] for x in segments if x is not seg and x["end"] <= seg["start"]]
    after = [x["start"] for x in segments if x is not seg and x["start"] >= seg["end"]]
    return max([a, *before]), min([b, *after])


def ordered_classes(logits: np.ndarray, classes: int) -> np.ndarray | None:
    """The class (time step) of each boundary, start and end per word in turn, maximising the summed log
    probability under: no boundary before the previous one, and a word's end at least one class after its start.
    Only the first `classes` classes (the audio's length) are allowed. None when no such path exists."""
    scores = np.asarray(logits, dtype=np.float64)
    peak = scores.max(axis=1, keepdims=True)
    logp = (scores - peak - np.log(np.exp(scores - peak).sum(axis=1, keepdims=True)))[:, :classes]
    n, count = logp.shape
    if not n or n % 2 or not count:
        return None
    back = np.zeros((n, count), dtype=np.int64)
    previous = logp[0]
    steps = np.arange(count)
    for slot in range(1, n):
        best = np.maximum.accumulate(previous)  # best path ending at or before each class
        improved = np.r_[True, best[1:] > best[:-1]]
        best_at = np.maximum.accumulate(np.where(improved, steps, 0))  # earliest class reaching it
        shift = slot % 2  # an end comes at least one class after its word's start
        current = np.full(count, -np.inf)
        current[shift:] = logp[slot, shift:] + best[:count - shift]
        back[slot, shift:] = best_at[:count - shift]
        previous = current
    last = int(np.argmax(previous))
    if not np.isfinite(previous[last]):
        return None
    path = np.empty(n, dtype=np.int64)
    path[-1] = last
    for slot in range(n - 1, 0, -1):
        path[slot - 1] = back[slot, path[slot]]
    return path


def _fill(units: list[dict]) -> None:
    for k, u in enumerate(units):
        if "start" in u:
            continue
        prev = next((v for v in reversed(units[:k]) if "start" in v), None)
        nxt = next((v for v in units[k + 1:] if "start" in v), None)
        u["start"] = u["end"] = prev["end"] if prev else nxt["start"]


#: Loaded on a task's first use and kept until the stage calls `release` (35t's repairs align spot by spot).
#: `GPU_LOCK` is held for the load, every pass and the release.
_LOADED: tuple | None = None


def _model():
    global _LOADED
    if _LOADED is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        from transformers.utils import logging as hf_logging

        from ...reporting import current_reporter
        from ..runtime import hf_weights
        from ..runtime.device import device_type, resolve_device

        device = resolve_device("cuda", context="the forced aligner (--asr-aligner)")
        if device_type(device) == "xpu":
            # Only ever run on CUDA (bf16) and the CPU; an Intel card would be an unmeasured third dtype and backend.
            current_reporter().warning(
                "cpu-fallback", "the forced aligner (--asr-aligner) has not been run on an Intel GPU; it runs on the CPU.",
                impact="对齐这一步会慢一些")
            device = "cpu"
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        def fetch(p: hf_weights.HfLoad):
            kw = {"revision": p.revision, "local_files_only": p.local_files_only}
            proc = AutoProcessor.from_pretrained(MODEL, **kw)
            return AutoModelForTokenClassification.from_pretrained(MODEL, dtype=dtype, **kw).to(device).eval(), proc

        bar_was_on = hf_logging.is_progress_bar_enabled()
        hf_logging.disable_progress_bar()  # its weight-loading bar is nothing a user acts on (as local MOSS's)
        try:
            _LOADED = hf_weights.offline_first(fetch, hf_weights.prepare(MANIFEST_ID), what="the forced aligner")
        finally:
            if bar_was_on:
                hf_logging.enable_progress_bar()
    return _LOADED


def release() -> None:
    """Drop the model and give its GPU memory back: in a batch the next item's local MOSS caps the allocator at its
    tier's budget, and whatever this process still holds counts against that cap."""
    global _LOADED
    if _LOADED is None:
        return
    import gc

    import torch

    with GPU_LOCK:
        _LOADED = None
        gc.collect()
        torch.cuda.empty_cache()  # a no-op where CUDA was never initialised (a CPU run)


def aligned_units(segments: Sequence[dict], *, all_segments: Sequence[dict],
                  pcm16k: np.ndarray) -> dict[int, list[dict]]:
    """Per index into `segments`, the aligner's units for each segment it aligned (not the fallbacks).
    `all_segments` bound each segment's padded audio; `pcm16k` is the source's 16 kHz int16 samples."""
    wav = pcm16k.astype(np.float32) / 32768
    total = len(wav) / SR
    todo = [k for k, s in enumerate(segments) if any(_kept(u) for u in units_of(s["text"]))]
    if not todo:
        return {}
    from ..runtime.thread_budget import widened_intra_op_threads

    with GPU_LOCK:
        model, proc = _model()
        if model.device.type != "cpu":
            return _aligned(model, proc, segments, todo, all_segments, wav, total)
        with widened_intra_op_threads():
            return _aligned(model, proc, segments, todo, all_segments, wav, total)


def _aligned(model, proc, segments: Sequence[dict], todo: list[int], all_segments: Sequence[dict],
             wav: np.ndarray, total: float) -> dict[int, list[dict]]:
    import torch

    todo.sort(key=lambda k: segments[k]["end"] - segments[k]["start"])  # similar lengths per batch
    out: dict[int, list[dict]] = {}
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        cuts = [_span(all_segments, segments[k], total) for k in batch]
        words = [[_kept(u) for u in units_of(segments[k]["text"]) if _kept(u)] for k in batch]
        convs = [[{"role": "user", "content": [{"type": "audio", "audio": wav[int(a * SR):int(b * SR)]},
                                               *({"type": "text", "text": w} for w in wl)]}]
                 for (a, b), wl in zip(cuts, words)]
        inputs = proc.apply_chat_template(convs, tokenize=True, return_dict=True).to(model.device, model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        step = model.config.timestamp_segment_time / 1000
        stamps = inputs["input_ids"] == model.config.timestamp_token_id
        for r, (k, (a, b), wl) in enumerate(zip(batch, cuts, words)):
            path = ordered_classes(logits[r, stamps[r]].float().cpu().numpy(), int((b - a) / step + 1e-8) + 1)
            if path is None or len(path) != 2 * len(wl):
                continue
            seg = segments[k]
            timed = iter(path.reshape(-1, 2) * step)
            units = []
            for u in units_of(seg["text"]):
                unit = {"text": u, "speaker": seg.get("speaker")}
                if _kept(u):
                    start, end = next(timed)
                    unit.update(start=a + start, end=a + end)
                units.append(unit)
            _fill(units)
            out[k] = [dict(u, start=round(u["start"], 3), end=round(u["end"], 3)) for u in units]
    return out
