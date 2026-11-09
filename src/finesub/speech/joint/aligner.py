"""`--asr-aligner`: unit times for MOSS Pro's long segments from Qwen3-ForcedAligner (S6b, `docs/joint-asr.md`).

Heavy and optional: transformers (the `[asr]` extra), a 0.6B model from the Hugging Face hub, about 2.4 GB of
RAM on the CPU (float32) or seconds on a GPU (bf16). Only the segments over the splitter's ok limits are sent --
the ones the DP has to cut, about a quarter of the audio.

Each segment's audio, PAD seconds wider on each side but not into a neighbouring segment it does not overlap
(there the aligner put words into the next speaker's time), is aligned to the segment's units. The word list is
built here, not by the processor, whose Japanese split needs a dictionary model. A unit the aligner keeps
nothing of (a lone bracket) takes its neighbour's time; a segment whose units come out mostly zero-length, or
with a run faster than a collapsed line, is not aligned and keeps Silero's timing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...subtitles.metrics import weighted_char_count
from . import GPU_LOCK
from .lines import units_of

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
SR = 16000
PAD = 0.5
BATCH = 4
ZERO_SHARE = 0.5
#: A run of at least COLLAPSE_CHARS weighted characters faster than this is a collapsed line.
COLLAPSE_CPS = 25.0
COLLAPSE_CHARS = 5


def _kept(unit: str) -> str:
    from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

    return "".join(ch for ch in unit if _is_kept_char(ch))


def _span(segments: Sequence[dict], seg: dict, total: float) -> tuple[float, float]:
    a, b = max(0.0, seg["start"] - PAD), min(total, seg["end"] + PAD)
    before = [x["end"] for x in segments if x is not seg and x["end"] <= seg["start"]]
    after = [x["start"] for x in segments if x is not seg and x["start"] >= seg["end"]]
    return max([a, *before]), min([b, *after])


def _squeezed(units: list[dict]) -> bool:
    timed = [u for u in units if "start" in u]
    for i in range(len(timed)):
        n = 0.0
        for j in range(i, len(timed)):
            n += weighted_char_count(timed[j]["text"])
            if n >= COLLAPSE_CHARS and n / max(timed[j]["end"] - timed[i]["start"], 1e-3) > COLLAPSE_CPS:
                return True
    return False


def _fill(units: list[dict]) -> None:
    for k, u in enumerate(units):
        if "start" in u:
            continue
        prev = next((v for v in reversed(units[:k]) if "start" in v), None)
        nxt = next((v for v in units[k + 1:] if "start" in v), None)
        u["start"] = u["end"] = prev["end"] if prev else nxt["start"]


#: Loaded once and kept: batch items on the network lane share it. `GPU_LOCK` is held for the load and every pass.
_LOADED: tuple | None = None


def _model():
    global _LOADED
    if _LOADED is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor

        from ..runtime.device import resolve_device

        device = resolve_device("cuda", context="the forced aligner (--asr-aligner)")
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        proc = AutoProcessor.from_pretrained(MODEL)
        model = AutoModelForTokenClassification.from_pretrained(MODEL, dtype=dtype).to(device).eval()
        _LOADED = (model, proc)
    return _LOADED


def aligned_units(segments: Sequence[dict], *, all_segments: Sequence[dict],
                  pcm16k: np.ndarray) -> dict[int, list[dict]]:
    """Per index into `segments`, the aligner's units for each segment it aligned (not the fallbacks).
    `all_segments` bound each segment's padded audio; `pcm16k` is the source's 16 kHz int16 samples."""
    wav = pcm16k.astype(np.float32) / 32768
    total = len(wav) / SR
    todo = [k for k, s in enumerate(segments) if any(_kept(u) for u in units_of(s["text"]))]
    if not todo:
        return {}
    with GPU_LOCK:
        return _aligned(segments, todo, all_segments, wav, total)


def _aligned(segments: Sequence[dict], todo: list[int], all_segments: Sequence[dict], wav: np.ndarray,
             total: float) -> dict[int, list[dict]]:
    import torch

    model, proc = _model()
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
        decoded = proc.decode_forced_alignment(logits=logits, input_ids=inputs["input_ids"], word_lists=words,
                                               timestamp_token_id=model.config.timestamp_token_id)
        for k, (a, _), times in zip(batch, cuts, decoded):
            seg = segments[k]
            timed = iter(times)
            units = []
            for u in units_of(seg["text"]):
                unit = {"text": u, "speaker": seg.get("speaker")}
                if _kept(u):
                    t = next(timed)
                    unit.update(start=a + t["start_time"], end=a + t["end_time"])
                units.append(unit)
            zero = sum(1 for u in units if "start" in u and u["end"] - u["start"] <= 0)
            if zero > ZERO_SHARE * len(units) or _squeezed(units):
                continue
            _fill(units)
            out[k] = [dict(u, start=round(u["start"], 3), end=round(u["end"], 3)) for u in units]
    return out
