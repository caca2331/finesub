"""Second-model verification evidence via Qwen3-ASR.

Runs at the tail of the ``vad-asr`` stage (the energy track and audio are
only alive there) and produces *evidence*, never decisions: suspect segments
get a ``qwen_verify`` field with what Qwen3-ASR-0.6B heard in their span, and
speech-bearing coverage gaps are recorded in the stage metadata. All
consumers live downstream — the stabilize stage reads the evidence when
deciding drops (docs/asr-stabilize.md), the LLM layer can read the recovered
text as correction candidates.

Validated on 67 adjudicated clips (2026-08-05, docs/wt-refine-handoff.md P1):
phrase suspects 11/11, real-EN vs translation-mode 22/22 separated by the
auto pass's output language, coverage-gap recovery with zero fabrication,
and a drop audit that caught two real shouts our energy legs deleted. Known
weakness: shouts/screams may come back empty or cross-lingually rendered, so
absence of Qwen text must never authorize deleting shout-shaped segments —
only the polysyllabic closing-phrase family uses absence as evidence.

Dependency note: this uses the ``-hf`` checkpoints through native
transformers (``transformers>=5.13,<6``, shipped inside ``[asr]`` so the
same command yields the same stable everywhere; its tokenizers 0.22-0.23 and
huggingface-hub 1.x requirements sit inside faster-whisper's declared
ranges). The alternative — the ``qwen-asr``
wrapper package with the non-hf checkpoints — was implemented first and
rejected: it pins transformers exactly, drags a gradio/flask web stack plus
nagisa/dyNET its inference path never uses, and needs a ``--no-deps``
install that pyproject cannot express. Both paths share the same weights;
parity was verified output-for-output (including identical mishearings) on
the adjudicated smoke clips.

Performance stance: bf16 on GPU (float32 on CPU), one lazy load per run,
sequential per-clip generate — a run verifies a handful of short clips, so
batching, torch.compile and flash-attn would all cost more setup than they
save. Peak VRAM measured ~1.5 GB for 0.6B.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...reporting import current_reporter
from ...text import normalized_compact
from ..preprocessing.audio import (
    get_audio_info_stream,
    load_audio_slice_stream,
    resample_if_needed,
    to_mono,
)
from ..postprocessing import stabilization as asr_stabilize
from ..recognition.segments import coerce_optional_float

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
TARGET_SR = 16000
MAX_NEW_TOKENS = 256

# Segment-level evidence field: {"text": str, "language": str | None}.
VERIFY_KEY = "qwen_verify"
# Stage-metadata list of {"start", "end", "text"} for uncovered spans where
# Qwen heard speech. Evidence only — nothing is inserted into the subtitle
# stream here (cue construction without word timestamps is a separate step).
GAP_RECOVERY_KEY = "qwen_gap_recoveries"

# Uncovered VAD spans shorter than this are not probed (interjection-sized
# gaps are dominated by Qwen's shout blindness; long gaps are where whole
# missed lines live).
GAP_MIN_SEC = 3.0
# Context added around a suspect segment's span. Tight on purpose: wide pads
# bleed neighboring speech into the clip and dilute the evidence.
SEGMENT_PAD_SEC = 0.1
# Latin-run suspect shape (mirrors the stabilize lang-switch gate, minus the
# confidence condition — evidence is cheap, decisions stay downstream).
SUSPECT_MIN_LETTERS = 8
SUSPECT_MIN_LATIN_RATIO = 0.7


def _segment_span(segment: Dict[str, object]) -> Optional[Tuple[float, float]]:
    start = coerce_optional_float(segment.get("start"))
    end = coerce_optional_float(segment.get("end"))
    if (
        start is None
        or end is None
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
    ):
        return None
    return start, end


def _is_closing_phrase_shape(text: str) -> bool:
    compact = normalized_compact(text)
    return any(
        phrase in compact
        and len(compact)
        <= len(phrase) + asr_stabilize.CLOSING_GHOST_MAX_EXTRA_CHARS
        for phrase in asr_stabilize.CLOSING_GHOST_PHRASES
    )


def _is_latin_run(text: str) -> bool:
    letters = [c for c in str(text) if c.isalpha()]
    if len(letters) < SUSPECT_MIN_LETTERS:
        return False
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters) >= SUSPECT_MIN_LATIN_RATIO


def collect_suspect_indices(
    segments: Sequence[Dict[str, object]],
) -> List[int]:
    """Indices of segments worth second-model evidence.

    Three families: whole-segment closing phrases (any rate — the rate-ghost
    subset is already deletable without evidence, the normal-rate rest is
    only deletable WITH it), Latin runs inside a CJK-dominant output
    (real-EN vs translation-mode), and segments the stabilize noise legs
    would currently tag for dropping (so the drop can be vetoed when Qwen
    hears speech there).
    """

    run_cjk_dominant = asr_stabilize._run_is_cjk_dominant(list(segments))
    out: List[int] = []
    for index, segment in enumerate(segments):
        if _segment_span(segment) is None:
            continue
        text = str(segment.get("text") or "")
        if _is_closing_phrase_shape(text):
            out.append(index)
            continue
        if run_cjk_dominant and _is_latin_run(text):
            out.append(index)
            continue
        prospective = asr_stabilize._profile_2_tags(
            segment, run_cjk_dominant=run_cjk_dominant
        )
        if (
            asr_stabilize.TAG_HIGHLY_SUSPECTED_HALLUCINATION in prospective
            or asr_stabilize.TAG_HIGHLY_SUSPECTED_FILLER in prospective
        ):
            out.append(index)
    return out


def collect_gaps(
    vad_intervals: Sequence[Dict[str, object]],
    segments: Sequence[Dict[str, object]],
    *,
    min_sec: float = GAP_MIN_SEC,
) -> List[Tuple[float, float]]:
    """Uncovered VAD spans of at least ``min_sec`` seconds."""

    covered = sorted(
        span for span in (_segment_span(s) for s in segments) if span
    )
    gaps: List[Tuple[float, float]] = []
    for interval in vad_intervals:
        span = _segment_span(interval)
        if span is None:
            continue
        pieces = [span]
        for c_start, c_end in covered:
            next_pieces: List[Tuple[float, float]] = []
            for p_start, p_end in pieces:
                if c_end <= p_start or c_start >= p_end:
                    next_pieces.append((p_start, p_end))
                    continue
                if c_start > p_start:
                    next_pieces.append((p_start, c_start))
                if c_end < p_end:
                    next_pieces.append((c_end, p_end))
            pieces = next_pieces
        gaps.extend(p for p in pieces if p[1] - p[0] >= min_sec)
    return gaps


class QwenReferee:
    """One lazily loaded Qwen3-ASR model per stage run.

    The model (~1.5 GB peak VRAM for 0.6B bf16) is loaded on first use —
    after the stage has closed the Whisper pool — and freed via ``close()``.
    Auto-language transcription via the native transformers path; no
    forced-aligner, no accelerate.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN_MODEL,
        *,
        device: str = "cuda",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    def _ensure_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_name)
            wants_cuda = self._device.startswith("cuda")
            # bf16 on GPU; float32 on CPU, where bf16 kernels are patchy.
            model = AutoModelForMultimodalLM.from_pretrained(
                self._model_name,
                dtype=torch.bfloat16 if wants_cuda else torch.float32,
            )
            if wants_cuda:
                try:
                    model = model.to(self._device)
                except Exception as exc:
                    current_reporter().warning(
                        "cpu-fallback",
                        f"Qwen referee falling back to CPU ({exc})",
                        impact="第二模型校验会明显变慢",
                    )
                    model = model.to(dtype=torch.float32)
            model.eval()
            self._model = model
        return self._model

    def transcribe_batch(
        self, clips: Sequence[np.ndarray]
    ) -> List[Tuple[str, Optional[str]]]:
        """(text, detected language) per 16 kHz mono clip, auto language.

        Sequential generate per clip: a run verifies a handful of short
        clips, so padding-batch bookkeeping would outweigh the gain.
        """

        if not clips:
            return []
        import torch

        model = self._ensure_model()
        processor = self._processor
        out: List[Tuple[str, Optional[str]]] = []
        for clip in clips:
            inputs = processor.apply_transcription_request(
                audio=np.asarray(clip, dtype=np.float32)
            ).to(model.device, model.dtype)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS
                )
            tail = generated[:, inputs["input_ids"].shape[1] :]
            text = str(
                processor.decode(tail, return_format="transcription_only")[0]
                or ""
            ).strip()
            # The raw output carries a "language <name>" prelude before the
            # transcript; best-effort parse, evidence-only.
            raw = processor.decode(tail)[0]
            match = re.search(r"language\s+([A-Za-z_]+)", str(raw))
            language = match.group(1) if match and text else None
            out.append((text, language))
        return out

    def close(self) -> None:
        if self._model is not None:
            self._model = None
            self._processor = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class _SpanReader:
    """Sequential 16 kHz mono reader over the stage's audio file."""

    def __init__(self, audio_path: str) -> None:
        self._path = str(audio_path)
        sr, total = get_audio_info_stream(self._path)
        if sr <= 0:
            raise RuntimeError(f"Invalid sample rate for audio: {self._path}")
        self._sr = int(sr)
        self._total = max(0, int(total))

    def read(self, start: float, end: float) -> np.ndarray:
        first = max(0, int(start * self._sr))
        last = min(self._total, int(end * self._sr))
        if last <= first:
            return np.zeros(0, dtype=np.float32)
        chunk, sr = load_audio_slice_stream(self._path, first, last - first)
        mono = to_mono(chunk)
        resampled, _ = resample_if_needed(mono.unsqueeze(0), int(sr), TARGET_SR)
        return resampled.squeeze(0).cpu().numpy().astype(np.float32)


def apply_verification(
    segments: List[Dict[str, object]],
    *,
    vad_intervals: Sequence[Dict[str, object]],
    audio_path: str,
    referee: QwenReferee,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Attach ``qwen_verify`` evidence and collect gap recoveries.

    Returns (segments-with-evidence, stats). ``stats[GAP_RECOVERY_KEY]``
    carries the recovered-gap evidence for the stage metadata.
    """

    suspect_indices = collect_suspect_indices(segments)
    gaps = collect_gaps(vad_intervals, segments)
    stats: Dict[str, object] = {
        "model": referee._model_name,
        "suspects": len(suspect_indices),
        "gaps_probed": len(gaps),
    }
    if not suspect_indices and not gaps:
        stats[GAP_RECOVERY_KEY] = []
        return segments, stats

    reader = _SpanReader(audio_path)
    clips: List[np.ndarray] = []
    for index in suspect_indices:
        start, end = _segment_span(segments[index])  # type: ignore[misc]
        clips.append(
            reader.read(start - SEGMENT_PAD_SEC, end + SEGMENT_PAD_SEC)
        )
    for start, end in gaps:
        clips.append(reader.read(start, end))

    # Spans clipped away entirely by the audio bounds (or degenerate ones)
    # must not reach the model; they read as "no speech heard".
    min_samples = int(0.05 * TARGET_SR)
    usable = [i for i, clip in enumerate(clips) if len(clip) >= min_samples]
    replies = referee.transcribe_batch([clips[i] for i in usable])
    results: List[Tuple[str, Optional[str]]] = [("", None)] * len(clips)
    for position, reply in zip(usable, replies):
        results[position] = reply

    out = list(segments)
    for position, index in enumerate(suspect_indices):
        text, language = results[position]
        item = dict(out[index])
        item[VERIFY_KEY] = {"text": text, "language": language}
        out[index] = item

    recoveries: List[Dict[str, object]] = []
    for position, (start, end) in enumerate(gaps):
        text, language = results[len(suspect_indices) + position]
        if normalized_compact(text):
            recoveries.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "language": language,
                }
            )
    stats[GAP_RECOVERY_KEY] = recoveries
    stats["gaps_recovered"] = len(recoveries)
    return out, stats
