"""Whisper re-times the forced aligner's unsure boundaries in MOSS's long segments (`--asr-aligner` on a GPU;
`docs/joint-asr.md`).

Qwen3-ForcedAligner is close where it is sure of a boundary and makes its large errors where it is not. There
Whisper large-v3-turbo times the boundary instead, with MOSS's text forced as its decode prefix: the patched
CTranslate2's one-pass refine trace, read as the local Whisper pipeline reads it (`fw_refine.align_span_words`,
disfluency starts on: a word whose attention has several peaks starts at the last one, after the false start). The
trace runs a segment's last word on to the closing timestamp, so the segment's end comes from faster-whisper's
teacher-forced alignment (`find_alignment`) on the same encoder output instead. Whisper's times sit a steady few
tens of ms off the aligner's; that offset, the median difference over the aligner's sure boundaries in this call
(one material), is taken off before Whisper's times are used.

Which boundaries keep the aligner's time: an internal one from a peak probability of KEEP_INTERNAL, the segment's
start and end from KEEP_EDGE (selected on two development episodes, read on three held-out ones; the readings
and the arms that lost are in the moss-word-cut report).
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

#: An internal boundary whose peak probability is at least this keeps the aligner's time.
KEEP_INTERNAL = 0.6
#: The same for the segment's start and end, which become line edges.
KEEP_EDGE = 0.8
#: The aligner's boundaries that set Whisper's offset (`aligner.SURE`, kept apart so the two can move apart).
OFFSET_SURE = 0.8
#: Whisper's encoder sees 30 s; a longer window keeps the aligner's times.
WINDOW = 30.0
SR = 16000
#: Its `GPU_STAGE_GATE` family: never on the card together with the separator, the local Whisper ASR or speaker
#: diarization.
GATE_FAMILY = "joint-whisper"


@dataclass(frozen=True)
class Retime:
    """What `aligned_units` re-times with Whisper. `language`: the run's `--language` (None: Whisper detects each
    segment's); `english`: English segments too (`--asr-aligner on`; `auto` leaves them to the aligner, which is
    the closer of the two there)."""

    language: str | None
    english: bool


@dataclass
class Job:
    """One aligned segment: its text and kept units (`aligner._kept`), its window [a, b] in seconds, and the
    aligner's boundary times (seconds, start and end per kept unit) and peak probabilities."""

    text: str
    kept_units: list[str]
    a: float
    b: float
    times: np.ndarray
    peaks: np.ndarray
    wt: np.ndarray | None = field(default=None)
    wh: np.ndarray | None = field(default=None)


#: Loaded by `retime` and dropped by `release`, both under the caller's `GPU_LOCK`.
_LOADED = None


def _model():
    global _LOADED
    if _LOADED is None:
        from ..runtime.device import ct2_cuda_unusable_reason

        if (reason := ct2_cuda_unusable_reason()) is not None:  # before fetching 1.6 GB it could not run
            raise RuntimeError(f"CTranslate2 cannot use CUDA here: {reason}")
        from ..recognition.asr_backend import DEFAULT_MODEL
        from ..recognition.fw_refine_backend import RefinedWhisperModel
        from ..recognition.vad_asr_stage import ensure_asr_weights
        from ..runtime import hf_weights

        _LOADED = hf_weights.offline_first(
            lambda plan: RefinedWhisperModel(DEFAULT_MODEL, device="cuda", compute_type="float16",
                                             revision=plan.revision, local_files_only=plan.local_files_only),
            ensure_asr_weights(DEFAULT_MODEL), what="asr")
    return _LOADED


def release() -> None:
    """Drop the model and give its GPU memory back (the caller holds `GPU_LOCK`). Collects even with none loaded: a
    load that failed half way (the patched CT2's check after the model is built) leaves its object to collect."""
    global _LOADED
    import gc

    _LOADED = None
    gc.collect()


def retime(jobs: Sequence[Job], wav: np.ndarray, how: Retime) -> tuple[dict[int, np.ndarray], dict]:
    """Per index into `jobs`, the boundary times with Whisper's in place of the aligner's unsure ones (only the
    jobs Whisper could time), and what to record. `wav` is the source at 16 kHz, float."""
    from faster_whisper.tokenizer import _LANGUAGE_CODES, Tokenizer

    model = _model()
    tokenizers: dict[str, Tokenizer] = {}
    skipped = {"long": 0, "language": 0, "decode": 0}
    tried, errors = 0, []
    for job in jobs:
        if job.b - job.a > WINDOW:
            skipped["long"] += 1
            continue
        clip = wav[int(job.a * SR):int(job.b * SR)]
        enc, frames = _encode(model, clip)
        lang = (how.language or "").lower().split("-")[0] or _detected(model, enc)
        if lang not in _LANGUAGE_CODES or (lang == "en" and not how.english):
            skipped["language"] += 1
            continue
        if lang not in tokenizers:
            tokenizers[lang] = Tokenizer(model.hf_tokenizer, True, task="transcribe", language=lang)
        tok = tokenizers[lang]
        tokens = tok.encode(job.text)
        tried += 1
        try:
            trace = _trace(model, tok, enc, frames, tokens)
            forced = model.find_alignment(tok, [tokens], enc, frames)[0]
        except (RuntimeError, ValueError, IndexError) as error:  # one segment (a prompt past max_length)
            errors.append(str(error)[:300])
            trace = forced = None
        job.wt = None if trace is None else _unit_times(trace, job.text, job.kept_units)
        job.wh = _unit_times(forced, job.text, job.kept_units) if forced else None
        if job.wt is None or len(job.wt) != len(job.times):
            job.wt = None
            skipped["decode"] += 1
        else:
            job.wt += job.a
        if job.wh is not None:
            job.wh = job.wh + job.a if len(job.wh) == len(job.times) else None
    if errors and len(errors) == tried:  # every decode failed (out of memory, a driver error): the caller warns
        raise RuntimeError(errors[0])
    out, record = combined(jobs)
    if errors:
        record["decode_error"] = errors[0]
    return out, {**record, **{f"skipped_{k}": v for k, v in skipped.items() if v}}


def combined(jobs: Sequence[Job]) -> tuple[dict[int, np.ndarray], dict]:
    """Per index into `jobs`, the aligner's times with Whisper's (offset taken off) where the aligner is unsure,
    for the jobs Whisper timed (`wt`; `wh` for the segment's end); and what to record."""
    off_wt = _offset([(j.wt, j) for j in jobs])
    off_wh = _offset([(j.wh, j) for j in jobs])
    out: dict[int, np.ndarray] = {}
    moved = 0
    for i, job in enumerate(jobs):
        if job.wt is None:
            continue
        keep = job.peaks >= KEEP_INTERNAL
        keep[[0, -1]] = job.peaks[[0, -1]] >= KEEP_EDGE
        times = np.where(keep, job.times, job.wt - off_wt)
        if not keep[-1]:
            times[-1] = job.wh[-1] - off_wh if job.wh is not None else job.times[-1]
        # outside the aligner's window (the offset taken off a last word at the window's end, which may be the
        # source's end): the aligner's time, as a unit starting at or after the audio's end is dropped by the line DP
        late = times > job.b
        late[0::2] |= times[0::2] >= job.b
        times = np.where(~keep & ((times < job.a) | late), job.times, times)
        moved += int((times != job.times).sum())
        out[i] = times
    return out, {"segments": len(out), "boundaries": moved, "offset_ms": round(off_wt * 1000),
                 "end_offset_ms": round(off_wh * 1000)}


def _encode(model, clip: np.ndarray):
    from faster_whisper.audio import pad_or_trim

    feats = model.feature_extractor(clip)
    frames = min(feats.shape[-1], model.feature_extractor.nb_max_frames)
    return model.encode(pad_or_trim(feats[:, :frames])), frames


def _detected(model, enc) -> str:
    return model.model.detect_language(enc)[0][0][0][2:-2]


def _trace(model, tok, enc, frames: int, tokens: list[int]) -> list[dict] | None:
    """Whisper's words (seconds from the window start) from the refine trace with `tokens` forced, disfluency starts
    on; None when the decode does not reproduce them or the trace does not cover them."""
    from ..recognition.fw_refine import TimestampSpan, align_span_words

    prompt = [*tok.sot_sequence, tok.timestamp_begin, *tokens]
    res = model.model.generate(enc, [prompt], beam_size=1, max_length=model.max_length, return_scores=True,
                               return_refine_paths=True, return_refine_weights=True,
                               refine_frames=model.refine_frames,
                               real_audio_frames=max(1, math.ceil(frames / model.input_stride)),
                               suppress_blank=False, suppress_tokens=[])[0]
    if list(res.sequences_ids[0][1:1 + len(tokens)]) != tokens or not res.refine_alignments:
        return None
    item = res.refine_alignments[0]
    span = TimestampSpan(int(item.token_start), int(item.token_end), tuple(int(t) for t in item.alignment_tokens),
                         bool(item.unfinished))
    if span.token_start != 0 or span.token_end < len(tokens):
        return None
    aligned = align_span_words(span=span, path=[(int(a), int(b)) for a, b in item.alignments], tokenizer=tok,
                               language=tok.language_code,
                               chosen_logprobs=list(res.token_logprobs)[span.token_start:span.token_end + 1],
                               alignment_weights=np.asarray(item.alignment_weights, np.float32),
                               alignment_frame_start=int(item.alignment_frame_start), detect_disfluencies=True)
    return [dict(w) for w in aligned.words]


def _unit_times(words: Sequence[dict], text: str, kept_units: Sequence[str]) -> np.ndarray | None:
    """Kept units' (start, end) times, flattened, from Whisper's words over `text`: each word's characters split its
    span evenly and the text between words takes no time; `[*]` (a disfluency marker) is no text. None when the
    words do not spell `text`. English words come without their spaces; the decode may run on past the text."""
    times: list[tuple[float, float] | None] = [None] * len(text)
    pos = 0
    for w in words:
        body = str(w["word"]).strip()
        if w["word"] == "[*]" or not body:
            continue
        if not text[pos:].strip():
            break  # words the decoder went on with after the forced text
        n = len(body)
        i = text.find(body, pos)
        rest = text[pos:].lstrip()
        if (i < 0 or text[pos:i].strip()) and body.startswith(rest):
            i, body = len(text) - len(rest), rest  # the last word runs on into what the decoder added
        elif i < 0 or text[pos:i].strip():
            return None
        start, end = float(w["start"]), float(w["end"])
        for d in range(pos, i):
            times[d] = (start, start)
        for d in range(len(body)):
            times[i + d] = (start + (end - start) * d / n, start + (end - start) * (d + 1) / n)
        pos = i + len(body)
    if not pos or text[pos:].strip():
        return None
    times[pos:] = [times[pos - 1]] * (len(text) - pos)
    out, at = [], 0
    for u in kept_units:
        i = text.find(u, at)
        if i < 0:
            return None
        idx = [i + d for d, ch in enumerate(u) if _kept_char(ch)] or [i]
        out.append((times[idx[0]][0], times[idx[-1]][1]))
        at = i + len(u)
    return np.maximum.accumulate(np.array(out, float).ravel())


def _kept_char(ch: str) -> bool:
    """A character that is not punctuation, a symbol or a space: where a unit's time starts and ends."""
    return unicodedata.category(ch)[0] not in "PSZ"


def _offset(pairs: Sequence[tuple[np.ndarray | None, Job]]) -> float:
    """Median of Whisper's time minus the aligner's over the aligner's sure boundaries; 0 with none."""
    diffs = [(t - j.times)[j.peaks >= OFFSET_SURE] for t, j in pairs if t is not None]
    flat = np.concatenate(diffs) if diffs else np.zeros(0)
    return float(np.median(flat)) if len(flat) else 0.0
