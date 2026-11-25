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
class (80 ms). A segment with more words than its audio has classes is decoded again with zero-length words
allowed, and those share the span of the word before them (`_shared`), rather than falling back to Silero. The
decoded classes are then refined below the 80 ms grid (`boundary_times`): a sure boundary inside its class, an
unsure one to its mean over every allowed path, which is where the aligner's large errors sit. The official decode takes each boundary's argmax and repairs disorder afterwards, which left
hundreds of zero-length words and needed two fallbacks to Silero (mostly zero-length, or a run faster than a
collapsed line); the ordered decode needs neither, and on every ruler it is at least as close
(`docs/joint-asr.md`).

On a GPU, Whisper then re-times the boundaries the aligner is unsure of (`whisper_align`, `--asr-aligner`); the
aligner is released before Whisper loads, so the two are never on the card together.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import GPU_LOCK, whisper_align
from .lines import units_of

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
#: Its entry in `finesub_bootstrap/model-manifest.json` (revision pinned, the LFS files' digests).
MANIFEST_ID = "qwen-aligner"
SR = 16000
PAD = 0.5
BATCH = 4
#: A boundary whose most likely class holds at least this probability keeps the decode's class, refined inside it;
#: a less sure one takes its mean time over every allowed path (`boundary_times`).
SURE = 0.8
#: The least a unit lasts once boundaries leave the class grid.
MIN_UNIT = 0.02


def _kept(unit: str) -> str:
    from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

    return "".join(ch for ch in unit if _is_kept_char(ch))


def _span(segments: Sequence[dict], seg: dict, total: float) -> tuple[float, float]:
    a, b = max(0.0, seg["start"] - PAD), min(total, seg["end"] + PAD)
    before = [x["end"] for x in segments if x is not seg and x["end"] <= seg["start"]]
    after = [x["start"] for x in segments if x is not seg and x["start"] >= seg["end"]]
    return max([a, *before]), min([b, *after])


def ordered_classes(logits: np.ndarray, classes: int, *, floor: int = 1) -> np.ndarray | None:
    """The class (time step) of each boundary, start and end per word in turn, maximising the summed log
    probability under: no boundary before the previous one, and a word's end at least `floor` classes after its
    start. Only the first `classes` classes (the audio's length) are allowed. None when no such path exists."""
    logp = _log_softmax(logits)[:, :classes]
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
        shift = floor if slot % 2 else 0  # an end comes at least `floor` classes after its word's start
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


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    scores = np.asarray(logits, dtype=np.float64)
    peak = scores.max(axis=1, keepdims=True)
    return scores - peak - np.log(np.exp(scores - peak).sum(axis=1, keepdims=True))


def boundary_times(logits: np.ndarray, classes: int, path: np.ndarray) -> np.ndarray:
    """Each boundary's time in classes (fractional), from the ordered decode's `path`: where the boundary's own
    distribution is sure (peak >= SURE) the decode's class moved by the parabola through its and its neighbours'
    log probabilities (at most half a class); elsewhere the boundary's mean class over every path the ordered decode
    allows (forward-backward on the same lattice). The caller puts them back in order (`_ordered_seconds`)."""
    logp = _log_softmax(logits)[:, :classes]
    n, count = logp.shape
    refined = path.astype(np.float64)
    for s, c in enumerate(path):
        if 0 < c < count - 1:
            y0, y1, y2 = logp[s, c - 1], logp[s, c], logp[s, c + 1]
            den = y0 - 2 * y1 + y2
            if den < 0:
                refined[s] = c + float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5))
    unsure = np.exp(logp.max(axis=1)) < SURE
    if unsure.any():
        alpha = np.full((n, count), -np.inf)
        alpha[0] = logp[0]
        for s in range(1, n):
            c = np.logaddexp.accumulate(alpha[s - 1])
            if s % 2:
                alpha[s, 1:] = logp[s, 1:] + c[:-1]
            else:
                alpha[s] = logp[s] + c
        beta = np.zeros((n, count))
        for s in range(n - 2, -1, -1):
            after = np.logaddexp.accumulate((logp[s + 1] + beta[s + 1])[::-1])[::-1]  # over classes >= c
            if (s + 1) % 2:  # the next boundary is an end: strictly later
                beta[s, :-1], beta[s, -1] = after[1:], -np.inf
            else:
                beta[s] = after
        post = alpha + beta
        post = np.exp(post - post.max(axis=1, keepdims=True))
        mean = (post @ np.arange(count)) / post.sum(axis=1)
        refined[unsure] = mean[unsure]
    return np.clip(refined, 0, count - 1)


def _ordered_seconds(times: np.ndarray) -> np.ndarray:
    """Boundary times (seconds) with no boundary before the previous one and each unit MIN_UNIT long at least."""
    out = times.copy()
    for s in range(1, len(out)):
        out[s] = max(out[s], out[s - 1] + (MIN_UNIT if s % 2 else 0.0))
    return out


def _shared(units: list[dict]) -> None:
    """Zero-length timed units (only a decode with `floor=0` makes them) share the span of the timed unit before
    them, split evenly; a leading run shares the one after it."""
    timed = [u for u in units if "start" in u]
    runs: list[list[dict]] = []
    for u in timed:
        if runs and u["end"] - u["start"] <= 0 and u["start"] == runs[-1][-1]["end"]:
            runs[-1].append(u)
        elif runs and all(v["end"] - v["start"] <= 0 for v in runs[-1]) and runs[-1][-1]["end"] == u["start"]:
            runs[-1].append(u)  # a run with no timed unit before it joins the one after it
        else:
            runs.append([u])
    for run in runs:
        if len(run) < 2:
            continue
        a, b = run[0]["start"], run[-1]["end"]
        for i, u in enumerate(run):
            u["start"], u["end"] = a + (b - a) * i / len(run), a + (b - a) * (i + 1) / len(run)


def _fill(units: list[dict]) -> None:
    for k, u in enumerate(units):
        if "start" in u:
            continue
        prev = next((v for v in reversed(units[:k]) if "start" in v), None)
        nxt = next((v for v in units[k + 1:] if "start" in v), None)
        u["start"] = u["end"] = prev["end"] if prev else nxt["start"]


#: Loaded on a task's first use and kept until the stage calls `release` (35t's repairs align spot by spot).
#: `GPU_LOCK` is held for the load, every pass and the release. `_LOADED_FOR` is the device it was asked for: the
#: network lane's two workers may run tasks with different `--device` / `--gpu-tier`, so one asking for another
#: device drops it and loads its own.
_LOADED: tuple | None = None
_LOADED_FOR: str | None = None


def _model(requested: str):
    global _LOADED, _LOADED_FOR
    if _LOADED is not None and _LOADED_FOR != requested:
        _drop()
    if _LOADED is None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoProcessor
        from transformers.utils import logging as hf_logging

        from ...reporting import current_reporter
        from ..runtime import hf_weights
        from ..runtime.device import device_type, resolve_device

        device = resolve_device(requested, context="the forced aligner (--asr-aligner)")
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
            _LOADED_FOR = requested
        finally:
            if bar_was_on:
                hf_logging.enable_progress_bar()
    return _LOADED


def release() -> None:
    """Drop the model and give its GPU memory back: in a batch the next item's local MOSS caps the allocator at its
    tier's budget, and whatever this process still holds counts against that cap."""
    with GPU_LOCK:
        _drop()


def _drop() -> None:
    global _LOADED, _LOADED_FOR
    if _LOADED is None:
        return
    import gc

    import torch

    _LOADED = _LOADED_FOR = None
    gc.collect()
    torch.cuda.empty_cache()  # a no-op where CUDA was never initialised (a CPU run)


def aligned_units(segments: Sequence[dict], *, all_segments: Sequence[dict], pcm16k: np.ndarray, device: str = "cuda",
                  retime: whisper_align.Retime | None = None, record: dict | None = None) -> dict[int, list[dict]]:
    """Per index into `segments`, the aligner's units for each segment it aligned (not the fallbacks).
    `all_segments` bound each segment's padded audio; `pcm16k` is the source's 16 kHz int16 samples; `device`
    `cpu` keeps the aligner off the card (`cuda`: there when one is usable, else the CPU with a warning). With `retime`,
    Whisper then re-times the unsure boundaries (`whisper_align`); the aligner is released first, so the two models
    are never on the card together. What Whisper did goes into `record["aligner_whisper"]`; should it fail, the aligner's
    times stay, with a warning and `record["aligner_whisper_error"]`."""
    wav = pcm16k.astype(np.float32) / 32768
    total = len(wav) / SR
    todo = [k for k, s in enumerate(segments) if any(_kept(u) for u in units_of(s["text"]))]
    if not todo:
        return {}
    with GPU_LOCK:
        timed = _aligned(segments, todo, all_segments, wav, total, device)
        if retime is not None and timed:
            _drop()
            _retimed(segments, timed, wav, retime, {} if record is None else record)
    return {k: _units(segments[k], t.a, t.seconds) for k, t in timed.items()}


class _Timed:
    """One aligned segment: its window start `a` and end `b`, its boundary times (seconds from `a`) and, from the
    ordered decode with one class per word at least, the boundaries' peak probabilities (None otherwise)."""

    def __init__(self, a: float, b: float, seconds: np.ndarray, peaks: np.ndarray | None) -> None:
        self.a, self.b, self.seconds, self.peaks = a, b, seconds, peaks


def _retimed(segments: Sequence[dict], timed: dict[int, _Timed], wav: np.ndarray, how: whisper_align.Retime,
             record: dict) -> None:
    keys = [k for k, t in timed.items() if t.peaks is not None]
    jobs = [whisper_align.Job(text=segments[k]["text"], kept_units=[u for u in units_of(segments[k]["text"]) if _kept(u)],
                              a=timed[k].a, b=timed[k].b, times=timed[k].seconds + timed[k].a, peaks=timed[k].peaks)
            for k in keys]
    if not any(job.b - job.a <= whisper_align.WINDOW for job in jobs):  # nothing Whisper could take: no load
        if jobs:
            record["aligner_whisper"] = {"segments": 0, "skipped_long": len(jobs)}
        return
    from ..runtime.gpu_stage_gate import GPU_STAGE_GATE, clear_frames

    # Not on the card with the separator, the local Whisper ASR or the speaker diarization of another batch item:
    # CTranslate2 aborts the process on a CUDA OOM, so this is a family of its own, waiting for theirs to finish.
    # ⚠ Taken while holding `GPU_LOCK`: whoever holds the gate (the separator, VAD-ASR's `wt` lease, diarization)
    # must never take `GPU_LOCK`.
    lease = GPU_STAGE_GATE.acquire(whisper_align.GATE_FAMILY)
    failed = None
    try:
        try:
            changed, record["aligner_whisper"] = whisper_align.retime(jobs, wav, how)
        except BaseException as error:
            # the frames of its traceback and chain hold the model and the encoder output: cleared, the release
            # below frees them, and only then does the gate let another family onto the card
            clear_frames(error)
            if not isinstance(error, Exception):
                raise  # a stop (the batch catches it and goes on)
            failed = str(error)[:300]  # the aligner's own times, as without Whisper
        finally:
            whisper_align.release()
    finally:
        lease.release()
    if failed is not None:
        from ...reporting import current_reporter

        record["aligner_whisper_error"] = failed
        current_reporter().warning("joint-aligner-whisper", f"Whisper 重定时失败（{failed}），长段用对齐器自己的时间",
                                   impact="没把握的切点可能偏一些")
        return
    summary = record["aligner_whisper"]
    for i, times in changed.items():
        t = timed[keys[i]]
        seconds = _ordered_seconds(times - t.a)
        if _inside(seconds, t):
            t.seconds = seconds
        else:  # ordering pushed a unit past the window: the aligner's times, for the whole segment
            summary["segments"] -= 1
            summary["boundaries"] -= int((times != jobs[i].times).sum())
            summary["outside_window"] = summary.get("outside_window", 0) + 1


def _inside(seconds: np.ndarray, t: _Timed) -> bool:
    """Whether every timed unit, as `_units` rounds it, starts before the window's end and ends by it: the window's
    end may be the audio's, and the line DP drops a unit that starts there (an untimed mark after it is folded into
    the word before, `lines.dp_words`). The window ends at the decoded PCM's end, which can run a few ms past the
    container duration the DP cuts at: a unit starting in that sliver is dropped, as it was before Whisper; rare
    enough to leave."""
    edges = np.round(t.a + seconds, 3)
    return bool((edges[0::2] < round(t.b, 3)).all() and (edges[1::2] <= round(t.b, 3)).all())


def _aligned(segments: Sequence[dict], todo: list[int], all_segments: Sequence[dict], wav: np.ndarray,
             total: float, device: str) -> dict[int, _Timed]:
    from ..runtime.thread_budget import widened_intra_op_threads

    model, proc = _model(device)
    if model.device.type != "cpu":
        return _decoded(model, proc, segments, todo, all_segments, wav, total)
    with widened_intra_op_threads():
        return _decoded(model, proc, segments, todo, all_segments, wav, total)


def _decoded(model, proc, segments: Sequence[dict], todo: list[int], all_segments: Sequence[dict],
             wav: np.ndarray, total: float) -> dict[int, _Timed]:
    import torch

    todo.sort(key=lambda k: segments[k]["end"] - segments[k]["start"])  # similar lengths per batch
    out: dict[int, _Timed] = {}
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
            row, classes = logits[r, stamps[r]].float().cpu().numpy(), int((b - a) / step + 1e-8) + 1
            path = ordered_classes(row, classes)
            peaks = None
            if path is not None:
                seconds = _ordered_seconds(boundary_times(row, classes, path) * step)
                # normalised over the allowed classes, as the Whisper step's thresholds were measured;
                # `boundary_times` decides its refinement on the whole row (4 of 19k boundaries differ)
                peaks = np.exp(_log_softmax(row[:, :classes]).max(axis=1))
            else:  # more words than the audio has classes: they share classes rather than leave the aligner
                path = ordered_classes(row, classes, floor=0)
                seconds = None if path is None else path * step
            if seconds is None or len(seconds) != 2 * len(wl):
                continue
            out[k] = _Timed(a, b, seconds, peaks)
    return out


def _units(seg: dict, a: float, seconds: np.ndarray) -> list[dict]:
    timed = iter(seconds.reshape(-1, 2))
    units = []
    for u in units_of(seg["text"]):
        unit = {"text": u, "speaker": seg.get("speaker")}
        if _kept(u):
            start, end = next(timed)
            unit.update(start=a + start, end=a + end)
        units.append(unit)
    _shared(units)
    _fill(units)
    return [dict(u, start=round(u["start"], 3), end=round(u["end"], 3)) for u in units]
