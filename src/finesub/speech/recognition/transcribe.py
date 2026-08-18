"""ASR alignment pipeline using Whisper timestamped output and VAD post-processing."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import gc
import json
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from ...reporting import current_reporter, reporting_to, terminal_reporter
from ...text import (
    COLLAPSE_STACK_MIN_RUN,
    COLLAPSE_STACK_WORD_SEC,
    COMMON_HALLUCINATION_TEXT,
    GROUP_REPEAT_MIN_COUNT,
    GROUP_REPEAT_MIN_UNITS,
    LONG_WORD_WORDS,
    LONG_WORD_SEC,
    REPEAT_DETECT_MORE_THAN,
    REPEAT_KEEP_RUN,
    cleanup_asr_words_for_fallback,
    coerce_optional_float,
    collapse_stack_run,
    copy_float_fields,
    detect_abnormal_asr_words,
    words_to_text,
)
from . import segments as segment_ops
from . import checkpoint as checkpoint_store
from . import word_starts
from ..preprocessing.audio import (
    TARGET_SR,
    apply_bandpass,
    as_numpy_float32,
    get_audio_info,
    load_audio_slice,
    resample_if_needed,
    to_mono,
)
from ..preprocessing.spectral import weighted_spectral_energy_db
from ..runtime.device import resolve_device
from ..runtime.resource_usage import (
    print_peak_resource_usage,
    reset_peak_gpu_memory_stats_for_run,
    start_stage_memory_sampling,
)


# --------- Tunables (pipeline defaults) ---------
DEFAULT_MODEL = "large-v3-turbo"  # Default Whisper model.
DEFAULT_DEVICE = "cuda"  # Preferred device; falls back to CPU if unavailable.
DEFAULT_GAP_SEC = 0.3  # Synthetic silence inserted right before the next interval.
# Up to this much of the original gap audio is kept after the left interval
# (preserves low-energy tails the VAD cut off); the DEFAULT_GAP_SEC silence
# after it gives the decoder a consistent segmentation cue before the next
# interval. Total inserted duration is always in [0.3, 1.0]s.
GAP_KEEP_REAL_MAX_SEC = 0.7

# Inter-interval synthetic silence: min(BASE + GROWTH * original_gap, MAX).
# Replaces the former fixed DEFAULT_GAP_SEC insertion (gap experiment
# 2026-07-19): tight boundaries get a compact cue, wide pauses a stronger one.
GAP_SILENCE_BASE_SEC = 0.1
GAP_SILENCE_GROWTH = 0.2
GAP_SILENCE_MAX_SEC = 0.8
# The group tail is padded on the same principle as inter-interval gaps:
# up to min(GAP_KEEP_REAL_MAX_SEC, gap to the next interval) of real audio,
# then DEFAULT_GAP_SEC of synthetic silence.
ROUND_DIGITS = 3  # Output JSON float precision.
# no_speech_prob is consumed on a log scale (typical values 1e-4..1e-2);
# 3 digits would collapse everything below 5e-4 to 0.0.
ROUND_DIGITS_BY_KEY = {"no_speech_prob": 6}
# Regroup retries for unstable ASR outputs. Was 3; the third retry
# (scale 2/5) resolved 1/32 groups in the 11-source collapse eval
# (out/collapse-eval) while costing a full subgroup re-decode round, so the
# ladder now hands over to beam/isolation after two retries.
REFINE_SEC = 1.0  # WT refine_whisper_precision equivalent.
# The patched CT2 backend is an explicit opt-in. Cheap path events are part of
# the checkpoint default. Disfluency detection is on: its ``[*]`` blocks and
# leading-start candidates are resolved (and acoustically gated) by
# recognition.word_starts before anything leaves the stage, so no ``[*]``
# reaches the aligned JSON. Decode cost is nil (measured 0.030 vs 0.031
# elapsed/audio on BV1cqLR6hEp3). Uncalibrated boundary entropy/peak rows
# remain research-only and are not emitted for every span.
FW_REFINE_DETECT_DISFLUENCIES = True
FW_REFINE_COLLECT_PATH_SIGNALS = True
FW_REFINE_COLLECT_BOUNDARY_SIGNALS = False
# Last-word extension baseline window length (seconds) before last word end.
LAST_WORD_EXTEND_LOOKAHEAD = 0.2
# Extend while weighted energy stays within this dB margin below baseline.
LAST_WORD_EXTEND_ENERGY_THRESHOLD = 20.0
# Hard cap on per-word extension duration (seconds).
LAST_WORD_EXTEND_MAX_TIME = 1.0
# Forward scan window and hop for extension decision (milliseconds).
LAST_WORD_EXTEND_WINDOW_MS = 25.0
LAST_WORD_EXTEND_HOP_MS = 10.0
# Stop scan when the current low-energy window and this many following
# low-energy windows are all below target.
LAST_WORD_EXTEND_FOLLOWING_LOW_WINDOWS = 2

# --------- Recovery accounting ---------
# The rescue ladder is normal machinery, not news: a run that recovers from
# three abnormal windows produced a correct result, and printing every step of
# every attempt buried the events that do need a person. Each step is counted
# here and shown only under `verbose`; the stage reports the totals once.
_stats_local = threading.local()

#: Progress is reported in twentieths rather than per group. The reporter
#: throttles a terminal further, but a Desktop renderer turns every call into
#: an event, so the bound belongs at the source too.
PROGRESS_STEPS = 20


@contextmanager
def collecting_stats() -> Iterator[Counter]:
    """Accumulate recovery counters for one stage run."""

    previous = getattr(_stats_local, "counter", None)
    counter: Counter = Counter()
    _stats_local.counter = counter
    try:
        yield counter
    finally:
        if previous is None:
            del _stats_local.counter
        else:
            _stats_local.counter = previous


def _report_progress(completed: int, total: int, groups_done: int) -> None:
    """Report interval progress, but only when it crosses a twentieth.

    Groups are the unit a person watches, but how many there will be is only
    known once grouping has run to the end -- so the count is intervals, which
    is known up front, and groups ride along as detail.
    """

    total = max(total, 1)
    step = completed * PROGRESS_STEPS // total
    if getattr(_stats_local, "progress_step", None) == step and completed < total:
        return
    _stats_local.progress_step = step
    current_reporter().progress(
        "aligned",
        completed=completed,
        total=total,
        unit="intervals",
        detail=f"{groups_done} groups" if groups_done else "",
    )


def _count(name: Optional[str]) -> None:
    if name is None:
        return
    counter = getattr(_stats_local, "counter", None)
    if counter is not None:
        counter[name] += 1


def _note(message: str, *, count: Optional[str] = None) -> None:
    """Record one internal recovery step: always counted, shown when verbose."""

    _count(count)
    current_reporter().debug(message)


def _warn(
    code: str,
    message: str,
    *,
    count: Optional[str] = None,
    impact: str = "",
    action: str = "",
) -> None:
    """Report something that changed the result and may need a person."""

    _count(count)
    current_reporter().warning(code, message, impact=impact, action=action)


# --------- Tunables (ASR grouping) ---------
GROUP_TARGET_SEC = 30.0  # Start searching for a split only after this total (sec).
BASE_BREAK_LENGTH = 1.0  # Base gap scale used in adaptive split threshold.
MIN_GROUP_LENGTH = 15.0  # Minimum non-final group duration (sec).
AUTO_LANGUAGE_SHORT_GROUP_SEC = 10.0
AUTO_LANGUAGE_HISTORY_GROUPS = 10
# Minimum summed uncovered complement duration (seconds) to trigger recall ASR.
MIN_DROP_TIME_RECALL = 5.0
# Complements shorter than this never seed recall ASR: isolated slivers
# (e.g. a 0.03s leftover at a coarse segment boundary) reliably hallucinate
# when decoded, and cannot hold real content.
RECALL_COMPLEMENT_MIN_SEC = 0.25
# Previous-block tail window (seconds) used for cross-block complement masking.
PREV_BLOCK_TAIL_SEC = 5.0
# Zero-length segments (mapping monotonicity collapse / zero-length whisper
# words) get this minimal duration so downstream end<=start filters (to_srt,
# LLM chunking) don't silently drop their text; may squeeze the next
# segment's start later.

# --------- Tunables (coverage rescue) ---------
# Greedy decoding can skip the rest of a 30s window after an early EOT and
# leave whole sentences untranscribed while every per-output quality metric
# (no_speech_prob, avg_logprob, abnormal-word checks) looks clean. The only
# reliable signal is coverage: output segments overlapping the batch's
# intervals for far less time than the intervals contain. A batch whose
# output covers less than ASR_COVERAGE_MIN_RATIO of its interval speech
# time, with ASR_COVERAGE_TOLERANCE_SEC slack, runs the rescue ladder:
# beam retry first, then peel-splitting windows (converges to
# interval-by-interval). Batches shorter than tolerance/ratio (~3.3s) are
# exempt.
ASR_COVERAGE_MIN_RATIO = 0.6
ASR_COVERAGE_TOLERANCE_SEC = 2.0
# Beam width for rescue decodes and the last-resort attempt before
# interval-by-interval fallback. The classic WT backend uses naive two-pass
# alignment for beam; patched fw-refine keeps the winning beam's 1-pass trace.
ASR_RESCUE_BEAM_SIZE = 5


def round_floats(obj, digits: int = ROUND_DIGITS):
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, dict):
        return {
            k: round_floats(v, ROUND_DIGITS_BY_KEY.get(k, ROUND_DIGITS))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [round_floats(v, digits) for v in obj]
    return obj


def asr_align_metadata(
    *,
    model: str,
    device: str,
    language: Optional[str],
    gap_sec: float,
) -> Dict[str, object]:
    return {
        "model": model,
        "device": device,
        "language": language or "None",
        "gap_sec": gap_sec,
        "gap_keep_real_max_sec": GAP_KEEP_REAL_MAX_SEC,
        "gap_silence_base_sec": GAP_SILENCE_BASE_SEC,
        "gap_silence_growth": GAP_SILENCE_GROWTH,
        "gap_silence_max_sec": GAP_SILENCE_MAX_SEC,
        "round_digits": ROUND_DIGITS,
        "group_target_sec": GROUP_TARGET_SEC,
        "base_break_length": BASE_BREAK_LENGTH,
        "min_group_length": MIN_GROUP_LENGTH,
        "min_drop_time_recall": MIN_DROP_TIME_RECALL,
        "recall_complement_min_sec": RECALL_COMPLEMENT_MIN_SEC,
        "prev_block_tail_sec": PREV_BLOCK_TAIL_SEC,
        "zero_length_segment_extend_sec": segment_ops.ZERO_LENGTH_SEGMENT_EXTEND_SEC,
        "asr_coverage_min_ratio": ASR_COVERAGE_MIN_RATIO,
        "asr_coverage_tolerance_sec": ASR_COVERAGE_TOLERANCE_SEC,
        "asr_rescue_beam_size": ASR_RESCUE_BEAM_SIZE,
        "refine_sec": REFINE_SEC,
        "last_word_extend_lookahead": LAST_WORD_EXTEND_LOOKAHEAD,
        "last_word_extend_energy_threshold": LAST_WORD_EXTEND_ENERGY_THRESHOLD,
        "last_word_extend_max_time": LAST_WORD_EXTEND_MAX_TIME,
        "last_word_extend_window_ms": LAST_WORD_EXTEND_WINDOW_MS,
        "last_word_extend_hop_ms": LAST_WORD_EXTEND_HOP_MS,
        "last_word_extend_following_low_windows": LAST_WORD_EXTEND_FOLLOWING_LOW_WINDOWS,
        "long_word_sec": LONG_WORD_SEC,
        "long_word_words": LONG_WORD_WORDS,
        "collapse_stack_word_sec": COLLAPSE_STACK_WORD_SEC,
        "collapse_stack_min_run": COLLAPSE_STACK_MIN_RUN,
        "group_repeat_min_count": GROUP_REPEAT_MIN_COUNT,
        "group_repeat_min_units": GROUP_REPEAT_MIN_UNITS,
        "repeat_detect_more_than": REPEAT_DETECT_MORE_THAN,
        "repeat_keep_run": REPEAT_KEEP_RUN,
        "auto_language_short_group_sec": AUTO_LANGUAGE_SHORT_GROUP_SEC,
        "auto_language_history_groups": AUTO_LANGUAGE_HISTORY_GROUPS,
    }


def merge_metadata(base: Optional[Dict[str, object]], align_meta: Dict[str, object]) -> Dict[str, object]:
    merged = dict(base) if isinstance(base, dict) else {}
    merged["asr_align"] = align_meta
    return merged


def normalize_vad_segments(
    raw_segments: List[Dict[str, object]],
    audio_duration: float,
) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    for raw in raw_segments:
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError):
            continue
        start = max(0.0, min(start, audio_duration))
        end = max(0.0, min(end, audio_duration))
        if end <= start:
            continue
        seg = {"start": start, "end": end}
        segments.append(seg)
    segments.sort(key=lambda x: (x["start"], x["end"]))
    return segments


def inserted_gap_parts(
    left: Dict[str, object],
    right: Dict[str, object],
    *,
    silence_sec: float = DEFAULT_GAP_SEC,
) -> Tuple[float, float]:
    """(real_audio_sec, silence_sec) inserted between two batched intervals.

    The first GAP_KEEP_REAL_MAX_SEC of the original gap audio is kept
    verbatim so low-energy tails the VAD cut off stay audible; the synthetic
    silence right before the next interval scales with the original gap
    (min(GAP_SILENCE_BASE_SEC + GAP_SILENCE_GROWTH * gap,
    GAP_SILENCE_MAX_SEC)) so wide pauses keep a proportionally stronger
    segmentation cue while tight boundaries stay compact. ``silence_sec``
    (the CLI ``gap_sec``) no longer drives the inter-interval silence; it
    still sets the tail silence after a group's last interval.
    """

    original_gap = max(0.0, float(right["start"]) - float(left["end"]))
    silence = min(
        GAP_SILENCE_BASE_SEC + GAP_SILENCE_GROWTH * original_gap,
        GAP_SILENCE_MAX_SEC,
    )
    return min(original_gap, GAP_KEEP_REAL_MAX_SEC), silence


def synthetic_gap_seconds(
    left: Dict[str, object],
    right: Dict[str, object],
    *,
    min_gap_sec: float = DEFAULT_GAP_SEC,
) -> float:
    """Total inserted duration between two batched intervals."""

    real_sec, silence_sec = inserted_gap_parts(left, right, silence_sec=min_gap_sec)
    return real_sec + silence_sec


def group_tail_seconds(
    group: List[Dict[str, object]],
    successor: Optional[Dict[str, object]],
    *,
    gap_sec: float,
) -> float:
    """Audio appended after a group's last interval by ``build_combined_audio``.

    Up to ``GAP_KEEP_REAL_MAX_SEC`` of real audio -- bounded by the gap to the
    next interval so the pad never bleeds into speech another group owns --
    followed by ``gap_sec`` of silence.
    """

    if not group:
        return 0.0
    if successor is None:
        tail_real = GAP_KEEP_REAL_MAX_SEC
    else:
        gap = float(successor.get("start", 0.0)) - float(group[-1].get("end", 0.0))
        tail_real = max(0.0, min(gap, GAP_KEEP_REAL_MAX_SEC))
    return tail_real + max(0.0, float(gap_sec))


def combined_group_audio_seconds(
    group: List[Dict[str, object]],
    successor: Optional[Dict[str, object]] = None,
    *,
    gap_sec: float,
) -> float:
    """Length of the audio ``build_combined_audio`` actually produces.

    This is the measure that decides whether a group fits one 30s encoder
    window. It is deliberately separate from ``combined_group_duration``: that
    one answers "how much did the speaker say", which is what the auto-language
    heuristic wants, and padding does not belong in that answer.
    """

    return combined_group_duration(group, gap_sec=gap_sec) + group_tail_seconds(
        group, successor, gap_sec=gap_sec
    )


def combined_group_duration(
    group: List[Dict[str, object]],
    *,
    gap_sec: float,
) -> float:
    duration = sum(
        max(0.0, float(seg["end"]) - float(seg["start"]))
        for seg in group
    )
    duration += sum(
        synthetic_gap_seconds(left, right, min_gap_sec=gap_sec)
        for left, right in zip(group, group[1:])
    )
    return duration


def build_alignment_groups(
    segments: List[Dict[str, object]],
    *,
    gap_sec: float,
    group_target_sec: float = GROUP_TARGET_SEC,
    min_group_length: float = MIN_GROUP_LENGTH,
) -> List[List[Dict[str, object]]]:
    if not segments:
        return []

    groups: List[List[Dict[str, object]]] = []
    n_segments = len(segments)
    target_len = max(0.0, float(group_target_sec))
    min_group_len = max(0.0, float(min_group_length))
    seg_durations = [
        float(seg["end"]) - float(seg["start"])
        for seg in segments
    ]
    duration_prefix = [0.0]
    for seg_len in seg_durations:
        duration_prefix.append(duration_prefix[-1] + seg_len)
    synthetic_gaps = [
        synthetic_gap_seconds(left, right, min_gap_sec=gap_sec)
        for left, right in zip(segments, segments[1:])
    ]
    gap_prefix = [0.0]
    for synthetic_gap in synthetic_gaps:
        gap_prefix.append(gap_prefix[-1] + synthetic_gap)

    def tail_len(end_idx: int) -> float:
        """Pad appended after the interval at ``end_idx - 1``."""

        if end_idx <= 0:
            return 0.0
        successor = segments[end_idx] if end_idx < n_segments else None
        return group_tail_seconds(
            segments[end_idx - 1 : end_idx], successor, gap_sec=gap_sec
        )

    def span_len(start_idx: int, end_idx: int) -> float:
        count = end_idx - start_idx
        if count <= 0:
            return 0.0
        speech = duration_prefix[end_idx] - duration_prefix[start_idx]
        gaps = gap_prefix[end_idx - 1] - gap_prefix[start_idx] if count > 1 else 0.0
        # The tail pad is part of what the encoder sees, so it counts toward
        # both the target and the minimum group length.
        return speech + gaps + tail_len(end_idx)

    group_start = 0

    while group_start < n_segments:
        group_end = group_start
        total_len = 0.0
        emitted = False

        while group_end < n_segments:
            seg = segments[group_end]
            seg_len = float(seg["end"]) - float(seg["start"])
            if group_end > group_start:
                total_len += synthetic_gaps[group_end - 1]
            total_len += seg_len
            group_end += 1

            # `total_len` is content only; the encoder also sees the tail pad,
            # so compare the audio the group would actually produce.
            audio_len = total_len + tail_len(group_end)
            if audio_len <= target_len:
                continue

            min_real_gap = BASE_BREAK_LENGTH * target_len / max(audio_len, 1e-9)
            split_idx = None
            for i in range(group_end - 2, group_start - 1, -1):
                real_gap = float(segments[i + 1]["start"]) - float(segments[i]["end"])
                if real_gap > min_real_gap:
                    candidate_split = i + 1
                    candidate_len = span_len(group_start, candidate_split)
                    if candidate_len >= min_group_len:
                        split_idx = candidate_split
                        break

            if split_idx is None:
                if group_end < n_segments:
                    continue
                groups.append(segments[group_start:n_segments])
                group_start = n_segments
                emitted = True
                break

            groups.append(segments[group_start:split_idx])
            group_start = split_idx
            emitted = True
            break

        if not emitted:
            groups.append(segments[group_start:n_segments])
            break

    return [group for group in groups if group]


class AudioBlockLoader:
    def __init__(
        self,
        path: str,
        *,
        target_sr: int,
        block_seconds: float = 600.0,
        pad_seconds: float = 10.0,
        preprocess: bool = False,
    ) -> None:
        self.path = path
        self.target_sr = target_sr
        self.block_seconds = block_seconds
        self.pad_seconds = pad_seconds
        self.preprocess = preprocess
        self._src_sr, self._src_frames = get_audio_info(path)
        if self._src_sr <= 0 or self._src_frames <= 0:
            raise RuntimeError(f"Unable to read audio info for: {path}")
        self._duration = self._src_frames / float(self._src_sr)
        self._block = None
        self._block_start = 0.0
        self._block_end = 0.0

    @property
    def duration(self) -> float:
        return self._duration

    def close(self) -> None:
        """Drop the cached block.

        One loader per WT shard means N resident blocks (~38MB each at the
        default 600s core / 16kHz), so shards release theirs as they finish
        rather than waiting for the collector."""

        self._block = None
        self._block_start = 0.0
        self._block_end = 0.0

    def _load_block_for_range(self, start_sec: float, end_sec: float) -> None:
        span = max(0.0, end_sec - start_sec)
        if self.block_seconds <= 0 or span > self.block_seconds:
            core_start = max(0.0, start_sec)
            core_end = min(self._duration, end_sec)
        else:
            block_idx = int(start_sec // self.block_seconds)
            core_start = block_idx * self.block_seconds
            # Always cover the full requested range: a slice that straddles a
            # block boundary by more than pad_seconds must not be truncated
            # (that would silently drop audio from a long interval's ASR).
            core_end = min(self._duration, max(core_start + self.block_seconds, end_sec))

        read_start = max(0.0, core_start - self.pad_seconds)
        read_end = min(self._duration, core_end + self.pad_seconds)
        if read_end <= read_start:
            self._block = np.zeros(0, dtype=np.float32)
            self._block_start = read_start
            self._block_end = read_end
            return

        frame_offset = int(round(read_start * self._src_sr))
        num_frames = int(round((read_end - read_start) * self._src_sr))
        waveform, sr = load_audio_slice(self.path, frame_offset, num_frames)
        if sr <= 0:
            raise RuntimeError(f"Invalid sample rate while loading: {self.path}")

        with torch.inference_mode():
            mono = to_mono(waveform)
            if self.preprocess:
                mono = apply_bandpass(mono.unsqueeze(0), sr).squeeze(0)
            resampled, _ = resample_if_needed(mono.unsqueeze(0), sr, self.target_sr)
        block = as_numpy_float32(resampled.squeeze(0))
        self._block = block
        self._block_start = read_start
        self._block_end = read_end

    def get_slice(self, start_sec: float, end_sec: float) -> np.ndarray:
        if end_sec <= start_sec:
            return np.zeros(0, dtype=np.float32)
        if (
            self._block is None
            or start_sec < self._block_start
            or end_sec > self._block_end
        ):
            self._load_block_for_range(start_sec, end_sec)
        if self._block is None or self._block_end <= self._block_start:
            return np.zeros(0, dtype=np.float32)
        rel_start = int(round((start_sec - self._block_start) * self.target_sr))
        rel_end = int(round((end_sec - self._block_start) * self.target_sr))
        rel_start = max(0, min(rel_start, len(self._block)))
        rel_end = max(rel_start, min(rel_end, len(self._block)))
        return self._block[rel_start:rel_end]


def build_combined_audio(
    audio: Optional[np.ndarray],
    sr: int,
    group: List[Dict[str, object]],
    gap_sec: float,
    *,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> Tuple[np.ndarray, List[Tuple[int, float, float, float, float]]]:
    audio_dtype = audio.dtype if audio is not None else np.float32

    def slice_audio(start_sec: float, end_sec: float) -> np.ndarray:
        if end_sec <= start_sec:
            return np.zeros(0, dtype=audio_dtype)
        if audio_loader is not None:
            return audio_loader.get_slice(start_sec, end_sec)
        if audio is None:
            return np.zeros(0, dtype=audio_dtype)
        start_idx = max(0, int(round(start_sec * sr)))
        end_idx = min(len(audio), int(round(end_sec * sr)))
        if end_idx <= start_idx:
            return np.zeros(0, dtype=audio_dtype)
        return audio[start_idx:end_idx]

    offsets: List[Tuple[int, float, float, float, float]] = []
    pieces: List[Tuple[Optional[np.ndarray], int]] = []
    cursor = 0.0
    total_samples = 0

    def append_clip(clip: np.ndarray) -> None:
        nonlocal cursor, total_samples
        n = int(len(clip))
        if n <= 0:
            return
        pieces.append((clip, n))
        total_samples += n
        cursor += n / sr

    def append_silence(seconds: float) -> None:
        nonlocal cursor, total_samples
        silence_samples = int(round(max(0.0, seconds) * sr))
        if silence_samples <= 0:
            return
        pieces.append((None, silence_samples))
        total_samples += silence_samples
        cursor += silence_samples / sr

    for i, seg in enumerate(group):
        start = float(seg["start"])
        end = float(seg["end"])
        clip = slice_audio(start, end)
        if clip.size == 0:
            continue
        offset_start = cursor
        offsets.append((i, start, end, offset_start, offset_start + len(clip) / sr))
        append_clip(clip)
        if i < len(group) - 1:
            real_sec, silence_sec = inserted_gap_parts(
                seg,
                group[i + 1],
                silence_sec=gap_sec,
            )
            if real_sec > 0:
                append_clip(slice_audio(end, end + real_sec))
            append_silence(silence_sec)
        else:
            tail_real = min(
                max(0.0, float(tail_real_limit_sec)),
                GAP_KEEP_REAL_MAX_SEC,
            )
            if tail_real > 0:
                append_clip(slice_audio(end, end + tail_real))
            append_silence(gap_sec)

    if not pieces:
        return np.zeros(0, dtype=audio_dtype), []

    combined = np.empty(total_samples, dtype=audio_dtype)
    write = 0
    for piece, piece_samples in pieces:
        if piece is None:
            combined[write:write + piece_samples] = 0
        else:
            combined[write:write + piece_samples] = piece
        write += piece_samples
    return combined, offsets


def load_interval_audio(
    audio: Optional[np.ndarray],
    sr: int,
    start_sec: float,
    end_sec: float,
    *,
    audio_loader: Optional[AudioBlockLoader] = None,
) -> np.ndarray:
    if end_sec <= start_sec:
        return np.zeros(0, dtype=np.float32)
    if audio_loader is not None:
        clip = audio_loader.get_slice(start_sec, end_sec)
        if clip.dtype != np.float32:
            clip = clip.astype(np.float32, copy=False)
        return clip
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    start_idx = max(0, int(round(start_sec * sr)))
    end_idx = min(len(audio), int(round(end_sec * sr)))
    if end_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    clip = audio[start_idx:end_idx]
    if clip.dtype != np.float32:
        clip = clip.astype(np.float32, copy=False)
    return clip


def extend_last_word_end_with_energy(
    words: List[Dict[str, object]],
    *,
    interval_start: float,
    interval_end: float,
    interval_audio: np.ndarray,
    sr: int,
    next_word_start: Optional[float] = None,
) -> List[Dict[str, object]]:
    if not words:
        return words
    ordered = sorted(words, key=lambda x: (float(x["start"]), float(x["end"])))
    if interval_audio.size <= 0:
        return ordered

    last_word = dict(ordered[-1])
    last_start = float(last_word.get("start", 0.0))
    last_end = float(last_word.get("end", 0.0))
    if last_end <= last_start:
        return ordered

    lookback_start = max(interval_start, last_end - LAST_WORD_EXTEND_LOOKAHEAD)
    lookback_end = min(interval_end, last_end)
    if lookback_end <= lookback_start:
        return ordered

    base_rel_start = max(0, int(round((lookback_start - interval_start) * sr)))
    base_rel_end = min(
        len(interval_audio), int(round((lookback_end - interval_start) * sr))
    )
    if base_rel_end <= base_rel_start:
        return ordered
    baseline_clip = interval_audio[base_rel_start:base_rel_end]
    if baseline_clip.size == 0:
        return ordered

    baseline_tensor = torch.from_numpy(baseline_clip.astype(np.float32, copy=False))
    baseline_frame_len = max(1, int(baseline_tensor.numel()))
    with torch.inference_mode():
        baseline_db = weighted_spectral_energy_db(
            baseline_tensor,
            sample_rate=sr,
            frame_len=baseline_frame_len,
            hop_len=baseline_frame_len,
        )
    if baseline_db.numel() == 0:
        return ordered
    target_db = (
        float(torch.mean(baseline_db).item()) - LAST_WORD_EXTEND_ENERGY_THRESHOLD
    )

    extend_limit = min(interval_end, last_end + LAST_WORD_EXTEND_MAX_TIME)
    if next_word_start is not None:
        extend_limit = min(extend_limit, float(next_word_start))
    if extend_limit <= last_end:
        return ordered

    scan_rel_start = max(0, int(round((last_end - interval_start) * sr)))
    scan_rel_end = min(len(interval_audio), int(round((extend_limit - interval_start) * sr)))
    if scan_rel_end <= scan_rel_start:
        return ordered
    scan_clip = interval_audio[scan_rel_start:scan_rel_end]
    if scan_clip.size == 0:
        return ordered

    frame_len = max(1, int(round(LAST_WORD_EXTEND_WINDOW_MS * sr / 1000.0)))
    hop_len = max(1, int(round(LAST_WORD_EXTEND_HOP_MS * sr / 1000.0)))
    scan_tensor = torch.from_numpy(scan_clip.astype(np.float32, copy=False))
    with torch.inference_mode():
        scan_db = weighted_spectral_energy_db(
            scan_tensor,
            sample_rate=sr,
            frame_len=frame_len,
            hop_len=hop_len,
        )
    if scan_db.numel() == 0:
        return ordered

    hop_sec = hop_len / float(sr)
    window_sec = frame_len / float(sr)
    following_low_windows = max(0, int(LAST_WORD_EXTEND_FOLLOWING_LOW_WINDOWS))
    extended_end = last_end

    for i in range(int(scan_db.numel())):
        window_start = last_end + i * hop_sec
        if window_start >= extend_limit:
            break
        energy_db = float(scan_db[i].item())
        if energy_db >= target_db:
            window_end = min(window_start + window_sec, extend_limit)
            if window_end > extended_end:
                extended_end = window_end
            continue
        if following_low_windows <= 0:
            break
        all_low = True
        for j in range(1, following_low_windows + 1):
            idx = i + j
            if idx >= int(scan_db.numel()):
                all_low = False
                break
            if float(scan_db[idx].item()) >= target_db:
                all_low = False
                break
        if all_low:
            break

    if extended_end > last_end:
        last_word["end"] = min(extend_limit, extended_end)
        ordered[-1] = last_word
    return ordered


def raw_word_text(word: Dict[str, object]) -> str:
    for key in ("text", "word", "token"):
        value = word.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def annotate_words_with_space_before(
    segment_text: object,
    words: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    text = str(segment_text or "")
    cursor = 0
    out: List[Dict[str, object]] = []
    for raw in words:
        item = dict(raw)
        raw_text = raw_word_text(item)
        token = raw_text.strip()
        if not token:
            item["_word_text"] = ""
            item["space_before"] = False
            out.append(item)
            continue

        space_before = False
        if text:
            pos = text.find(token, cursor)
            if pos < 0 and cursor > 0:
                pos = text.find(token)
            if pos >= 0:
                if pos > cursor:
                    space_before = any(ch.isspace() for ch in text[cursor:pos])
                cursor = max(cursor, pos + len(token))
            else:
                space_before = bool(raw_text and raw_text[0].isspace())
        else:
            space_before = bool(raw_text and raw_text[0].isspace())

        if not out:
            space_before = False

        item["_word_text"] = token
        item["space_before"] = bool(space_before)
        out.append(item)
    return out


def _combined_time_to_original(
    t: float,
    offsets: List[Tuple[int, float, float, float, float]],
) -> float:
    """Map a combined-timeline time back to the original timeline.

    Inside an interval the mapping is exact; inside a synthetic gap it is
    proportional between the neighboring interval edges (exact whenever the
    retained gap equals the original one, i.e. original gaps in the 0.3-1.0s
    clamp range); outside the covered range it extends linearly from the
    nearest interval edge.
    """

    if not offsets:
        return t
    prev: Optional[Tuple[int, float, float, float, float]] = None
    for entry in offsets:
        _idx, orig_s, _orig_e, off_s, off_e = entry
        if t < off_s:
            if prev is None:
                return orig_s + (t - off_s)
            _pi, _p_orig_s, p_orig_e, _p_off_s, p_off_e = prev
            span = off_s - p_off_e
            if span <= 0:
                return p_orig_e
            orig_gap = max(0.0, orig_s - p_orig_e)
            # The inserted region starts with kept real gap audio (1:1) and
            # ends with synthetic silence, which maps proportionally onto
            # whatever remains of the original gap.
            real_sec = min(orig_gap, GAP_KEEP_REAL_MAX_SEC)
            real_end = p_off_e + min(real_sec, span)
            if t <= real_end:
                return p_orig_e + (t - p_off_e)
            silence_span = off_s - real_end
            if silence_span <= 0:
                return orig_s
            frac = min(max((t - real_end) / silence_span, 0.0), 1.0)
            return (p_orig_e + real_sec) + frac * (orig_gap - real_sec)
        if t <= off_e:
            return orig_s + (t - off_s)
        prev = entry
    _idx, _orig_s, orig_e, _off_s, off_e = offsets[-1]
    return orig_e + (t - off_e)


def _dominant_interval_index(
    combined_mids: List[float],
    offsets: List[Tuple[int, float, float, float, float]],
    n_intervals: int,
) -> int:
    """Interval holding most of the words (by combined-timeline midpoints)."""

    if not offsets or n_intervals <= 0:
        return 0
    counts = [0] * n_intervals
    for mid in combined_mids:
        idx: Optional[int] = None
        prev: Optional[Tuple[int, float, float, float, float]] = None
        for entry in offsets:
            i, _orig_s, _orig_e, off_s, off_e = entry
            if mid < off_s:
                if prev is None:
                    idx = i
                else:
                    p_i, _pos, _poe, _poffs, p_off_e = prev
                    idx = p_i if (mid - p_off_e) <= (off_s - mid) else i
                break
            if mid <= off_e:
                idx = i
                break
            prev = entry
        if idx is None:
            idx = offsets[-1][0]
        if 0 <= idx < n_intervals:
            counts[idx] += 1
    best = max(range(n_intervals), key=lambda i: counts[i])
    if counts[best] <= 0:
        return max(0, min(offsets[0][0], n_intervals - 1))
    return best


def _map_asr_result_to_intervals(
    result: Dict[str, object],
    group: List[Dict[str, object]],
    offsets: List[Tuple[int, float, float, float, float]],
) -> Tuple[
    List[List[Dict[str, object]]],
    List[List[Dict[str, object]]],
]:
    """Map ASR output back to the original timeline, keeping each
    whisper-timestamped segment whole instead of cutting it at VAD interval
    boundaries.

    Each whisper segment becomes exactly one mapped segment, attached to the
    interval holding most of its words (finalize bookkeeping only — its words
    may extend beyond that interval). Word times map exactly inside intervals
    and 1:1 across the kept real-audio part of gaps, so words spoken in gaps
    keep real coordinates and no gap-word merge heuristics are needed.
    """

    per_interval_asr_segments: List[List[Dict[str, object]]] = [[] for _ in group]
    for seg_item in result.get("segments", []):
        segment_words = seg_item.get("words", []) or []
        asr_segment_words = annotate_words_with_space_before(
            seg_item.get("text"),
            segment_words,
        )
        mapped_words: List[Dict[str, object]] = []
        combined_mids: List[float] = []
        prev_end: Optional[float] = None
        for w in asr_segment_words:
            word_text = str(w.get("_word_text") or raw_word_text(w)).strip()
            if not word_text:
                continue
            w_start = coerce_optional_float(w.get("start"))
            w_end = coerce_optional_float(w.get("end"))
            if w_start is None or w_end is None:
                continue
            w_end = max(w_end, w_start)
            m_start = _combined_time_to_original(w_start, offsets)
            m_end = _combined_time_to_original(w_end, offsets)
            if prev_end is not None and m_start < prev_end:
                m_start = prev_end
            if m_end < m_start:
                m_end = m_start
            prev_end = m_end
            mapped_word = {
                "start": m_start,
                "end": m_end,
                "word": word_text,
                "space_before": bool(w.get("space_before", False)),
            }
            copy_float_fields(w, mapped_word, ("confidence",))
            mapped_words.append(mapped_word)
            combined_mids.append((w_start + w_end) / 2.0)
        if not mapped_words:
            continue
        mapped_segment: Dict[str, object] = {"words": mapped_words}
        copy_float_fields(
            seg_item,
            mapped_segment,
            ("confidence", "no_speech_prob"),
        )
        raw_events = seg_item.get("alignment_events")
        if isinstance(raw_events, list):
            mapped_events: List[Dict[str, object]] = []
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    continue
                event = dict(raw_event)
                for field in (
                    "start",
                    "end",
                    "original_start",
                    "refined_start",
                    "peak_time",
                ):
                    value = coerce_optional_float(event.get(field))
                    if value is not None:
                        event[field] = _combined_time_to_original(value, offsets)
                mapped_events.append(event)
            if mapped_events:
                mapped_segment["alignment_events"] = mapped_events
        dominant = _dominant_interval_index(combined_mids, offsets, len(group))
        per_interval_asr_segments[dominant].append(mapped_segment)

    per_interval_words: List[List[Dict[str, object]]] = []
    for asr_segments in per_interval_asr_segments:
        if not asr_segments:
            per_interval_words.append([])
            continue
        if len(asr_segments) == 1:
            per_interval_words.append(asr_segments[0].get("words") or [])
            continue
        words: List[Dict[str, object]] = []
        for asr_segment in asr_segments:
            words.extend(asr_segment.get("words") or [])
        words.sort(key=lambda x: (float(x["start"]), float(x["end"])))
        per_interval_words.append(words)
    return per_interval_words, per_interval_asr_segments


def _build_transcribe_kwargs(
    *,
    language: Optional[str],
) -> Dict[str, object]:
    # Greedy at a single temperature is what keeps alignment on the one-pass
    # trace; beam search and temperature fallback both leave it.
    kwargs: Dict[str, object] = {
        "beam_size": None,
        "best_of": None,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    return kwargs


def _issues_summary(issues: List[str]) -> str:
    if not issues:
        return "none"
    return "; ".join(issues[:3])


def _first_abnormal_interval_start(
    group: List[Dict[str, object]],
    per_interval_words: List[List[Dict[str, object]]],
) -> Optional[float]:
    for seg, words in zip(group, per_interval_words):
        if not words:
            continue
        issues = detect_abnormal_asr_words([words])
        if issues:
            try:
                return float(seg.get("start", 0.0))
            except (TypeError, ValueError):
                return None
    return None


def _first_abnormal_interval_index(
    per_interval_words: List[List[Dict[str, object]]],
) -> Optional[int]:
    """Index of the first interval whose own words are abnormal, or None when
    no single interval is attributable (e.g. a group-level repeat cycle)."""

    for idx, words in enumerate(per_interval_words):
        if words and detect_abnormal_asr_words([words]):
            return idx
    return None


def _phrase_key(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text)
        if not unicodedata.category(char).startswith("P") and not char.isspace()
    )


# Minimum normalized length before a stacked text counts as the known
# phrase: keeps tiny generic fragments (ありがとう alone) on the rescue path.
_PHRASE_STACK_MIN_CHARS = 6


def _is_known_phrase_stack_only(
    per_interval_words: List[List[Dict[str, object]]],
    issues: List[str],
) -> bool:
    """True when every abnormal signal is a collapse word stack whose stacked
    words spell nothing but the known hallucination phrase (repeats or a
    truncated fragment of it), and each flagged interval's remaining words are
    themselves clean. Such stacks carry no recoverable speech — the stabilize
    phrase cleanup removes the phrase wholesale even mid-segment — so rescue
    decodes are wasted GPU that at best converts the squeeze form into a
    stretched one and re-rolls the healthy remainder (400-window audit: the
    only pure-phrase-anomaly windows were real speech plus a 0-0.02s phrase
    tail, docs/wt-refine-validation.md)."""

    if not issues or not all(
        issue.startswith("collapse_word_stack") for issue in issues
    ):
        return False
    phrase = _phrase_key(COMMON_HALLUCINATION_TEXT)
    saw_stack = False
    for words in per_interval_words:
        if not words:
            continue
        run = collapse_stack_run(words)
        if run is None:
            continue
        saw_stack = True
        start_index, run_len = run
        stacked_text = _phrase_key(
            "".join(
                str(word.get("word") or "")
                for word in words[start_index : start_index + run_len]
            )
        )
        if len(stacked_text) < _PHRASE_STACK_MIN_CHARS:
            return False
        # Repeats and boundary-truncated fragments of the phrase, nothing else.
        repeats = phrase * (len(stacked_text) // len(phrase) + 2)
        if stacked_text not in repeats:
            return False
        remainder = words[:start_index] + words[start_index + run_len :]
        if remainder and detect_abnormal_asr_words([remainder]):
            return False
    return saw_stack


def _transcribe_with_teacher_force_fallback(
    model,
    combined: np.ndarray,
    transcribe_kwargs: Dict[str, object],
    *,
    group_start: float,
) -> Optional[Dict[str, object]]:
    """Transcribe one group, degrading instead of taking the whole run down.

    The fragile step is pairing word groups with the one-pass decoder trace,
    which real hallucinations reaching the decode limit can desynchronise. The
    second attempt is the backend's teacher-force alignment, which derives word
    times without that pairing. If that fails too the group is dropped: losing
    one group's subtitles beats losing the run."""

    try:
        return model.transcribe_wt(combined, **transcribe_kwargs)
    except Exception as exc:
        if transcribe_kwargs.get("force_teacher_force"):
            _warn(
                "asr-group-dropped",
                "teacher-force alignment failed "
                f"(start={group_start:.3f}s, error={exc}); dropping this group",
                count="dropped_groups",
                impact="这一段没有字幕",
            )
            return None
        # A retry that succeeds produced a correct result; it is machinery,
        # not news.
        _note(
            "one-pass alignment failed "
            f"(start={group_start:.3f}s, error={exc}); "
            "retrying with teacher-force alignment",
            count="alignment_retries",
        )
    try:
        return model.transcribe_wt(
            combined, **{**transcribe_kwargs, "force_teacher_force": True}
        )
    except Exception as exc:
        _warn(
            "asr-group-dropped",
            "teacher-force alignment also failed "
            f"(start={group_start:.3f}s, error={exc}); dropping this group",
            count="dropped_groups",
            impact="这一段没有字幕",
        )
        return None


def _transcribe_group_candidate(
    model,
    group: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    gap_sec: float,
    *,
    language: Optional[str],
    auto_language_history: Optional[List[str]] = None,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
    decode_options: Optional[Dict[str, object]] = None,
) -> Tuple[
    List[List[Dict[str, object]]],
    List[List[Dict[str, object]]],
    str,
    List[str],
    bool,
]:
    language_history = auto_language_history if auto_language_history is not None else []
    effective_language, uses_auto_detection = _language_for_group(
        language,
        group,
        gap_sec=gap_sec,
        auto_language_history=language_history,
    )

    combined, offsets = build_combined_audio(
        audio,
        sr,
        group,
        gap_sec,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    if combined.size == 0:
        return (
            [[] for _ in group],
            [[] for _ in group],
            effective_language or "None",
            [],
            uses_auto_detection,
        )

    transcribe_kwargs = _build_transcribe_kwargs(language=effective_language)
    transcribe_kwargs.update(
        {
            "detect_disfluencies": FW_REFINE_DETECT_DISFLUENCIES,
            "collect_refine_signals": FW_REFINE_COLLECT_PATH_SIGNALS,
            "collect_attention_signals": FW_REFINE_COLLECT_BOUNDARY_SIGNALS,
        }
    )
    if decode_options:
        transcribe_kwargs.update(decode_options)
    result = _transcribe_with_teacher_force_fallback(
        model,
        combined,
        transcribe_kwargs,
        group_start=float(group[0].get("start", 0.0)) if group else 0.0,
    )
    if result is None:
        # Same shape as the empty-audio early return: no words, no issues, so
        # the caller treats this group as silence and keeps going.
        return (
            [[] for _ in group],
            [[] for _ in group],
            effective_language or "None",
            [],
            uses_auto_detection,
        )
    lang = effective_language or result.get("language") or "None"
    per_interval_words, per_interval_asr_segments = _map_asr_result_to_intervals(
        result,
        group,
        offsets,
    )
    issues = detect_abnormal_asr_words(per_interval_words)
    return (
        per_interval_words,
        per_interval_asr_segments,
        lang,
        issues,
        uses_auto_detection,
    )


def _finalize_group_candidate(
    group: List[Dict[str, object]],
    per_interval_words: List[List[Dict[str, object]]],
    per_interval_asr_segments: List[List[Dict[str, object]]],
    audio: Optional[np.ndarray],
    sr: int,
    *,
    lang: str,
    audio_loader: Optional[AudioBlockLoader] = None,
) -> List[Dict[str, object]]:
    out_segments: List[Dict[str, object]] = []
    for interval_idx, (seg, words) in enumerate(zip(group, per_interval_words)):
        interval_start = float(seg.get("start", 0.0))
        interval_end = float(seg.get("end", 0.0))
        asr_segments_in_interval = (
            per_interval_asr_segments[interval_idx]
            if interval_idx < len(per_interval_asr_segments)
            else []
        )
        if not asr_segments_in_interval and words:
            asr_segments_in_interval = [
                {
                    "words": sorted(
                        words,
                        key=lambda x: (float(x["start"]), float(x["end"])),
                    )
                }
            ]
        if not asr_segments_in_interval:
            continue
        interval_audio = load_interval_audio(
            audio,
            sr,
            interval_start,
            interval_end,
            audio_loader=audio_loader,
        )
        for asr_segment_idx, asr_segment in enumerate(asr_segments_in_interval):
            asr_segment_words = asr_segment.get("words") or []
            if not asr_segment_words:
                continue
            next_word_start = None
            if (
                asr_segment_idx + 1 < len(asr_segments_in_interval)
                and asr_segments_in_interval[asr_segment_idx + 1]
            ):
                next_asr_segment_words = (
                    asr_segments_in_interval[asr_segment_idx + 1].get("words") or []
                )
                if next_asr_segment_words:
                    next_word_start = float(next_asr_segment_words[0]["start"])
            extended_asr_segment_words = extend_last_word_end_with_energy(
                asr_segment_words,
                interval_start=interval_start,
                interval_end=interval_end,
                interval_audio=interval_audio,
                sr=sr,
                next_word_start=next_word_start,
            )
            cleaned_words = cleanup_asr_words_for_fallback(
                extended_asr_segment_words,
                segment_start=float(extended_asr_segment_words[0]["start"]),
                segment_end=float(extended_asr_segment_words[-1]["end"]),
            )
            if not cleaned_words:
                continue
            item = {
                "start": float(cleaned_words[0]["start"]),
                "end": float(cleaned_words[-1]["end"]),
                "words": cleaned_words,
                "text": words_to_text(cleaned_words),
                "lang": lang,
            }
            if "confidence" in asr_segment:
                item["confidence"] = float(asr_segment["confidence"])
            if "no_speech_prob" in asr_segment:
                item["no_speech_prob"] = float(asr_segment["no_speech_prob"])
            alignment_events = asr_segment.get("alignment_events")
            if isinstance(alignment_events, list) and alignment_events:
                item["alignment_events"] = [
                    dict(event)
                    for event in alignment_events
                    if isinstance(event, dict)
                ]
            out_segments.append(item)
    return out_segments


def _rescue_decode_options() -> Dict[str, object]:
    """Beam decode overrides for rescue attempts (temperature stays 0.0).

    Classic whisper-timestamped switches to naive two-pass alignment; the
    patched fw-refine backend retains one-pass winner lineage.
    """

    return {
        "beam_size": ASR_RESCUE_BEAM_SIZE,
        "best_of": ASR_RESCUE_BEAM_SIZE,
    }


def _intervals_speech_seconds(intervals: List[Dict[str, object]]) -> float:
    total = 0.0
    for item in intervals:
        start = coerce_optional_float(item.get("start"))
        end = coerce_optional_float(item.get("end"))
        if start is None or end is None:
            continue
        total += max(0.0, end - start)
    return total


def _covered_speech_seconds(
    intervals: List[Dict[str, object]],
    segments: List[Dict[str, object]],
) -> float:
    """Seconds of interval time overlapped by output segments.

    Overlap is measured against the intervals, not raw segment durations:
    segments may legitimately span inter-interval gap audio, which must not
    count as covered speech."""

    spans = _extract_merged_segment_spans(segments)
    total = 0.0
    for item in intervals:
        start = coerce_optional_float(item.get("start"))
        end = coerce_optional_float(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        for span_start, span_end in spans:
            if span_start >= end:
                break
            total += max(0.0, min(end, span_end) - max(start, span_start))
    return total


def _coverage_shortfall(
    intervals: List[Dict[str, object]],
    segments: List[Dict[str, object]],
) -> Optional[Tuple[float, float, float]]:
    """Return (speech_sec, covered_sec, required_sec) when the batch output
    covers too little of its interval speech time, else None.

    The threshold max(0, ratio * speech - tolerance) exempts batches shorter
    than tolerance/ratio outright, so single small intervals never trigger."""

    speech = _intervals_speech_seconds(intervals)
    required = max(0.0, ASR_COVERAGE_MIN_RATIO * speech - ASR_COVERAGE_TOLERANCE_SEC)
    if required <= 0.0:
        return None
    covered = _covered_speech_seconds(intervals, segments)
    if covered >= required:
        return None
    return speech, covered, required


def _isolate_abnormal_intervals(
    model,
    intervals: List[Dict[str, object]],
    candidate: Tuple[
        List[List[Dict[str, object]]],
        List[List[Dict[str, object]]],
        str,
        List[str],
        bool,
    ],
    audio: Optional[np.ndarray],
    sr: int,
    gap_sec: float,
    *,
    language: Optional[str],
    language_history: List[str],
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_limit_sec: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Final abnormal-ASR ladder level, modeled on the coverage rescue's
    peeling loop but guided by the abnormal interval position: the clean
    intervals before the first abnormal one are re-decoded together as one
    window (keeping group context), the abnormal interval is isolated into its
    own window, and the remainder is re-decoded and re-examined — repeating
    until clean. Worst case converges to the old interval-by-interval
    fallback, but healthy neighbors are no longer fragmented one per window.

    Every piece is decoded from disjoint audio (front window tails are capped
    at the abnormal interval start), so no speech is transcribed twice.
    Results are kept even when still abnormal (with the cleanup warning), the
    same acceptance the old fallback had."""

    def window_tail(
        batch: List[Dict[str, object]],
        successor: Optional[Dict[str, object]],
    ) -> float:
        if not batch:
            return 0.0
        if successor is None:
            return tail_limit_sec
        gap = float(successor.get("start", 0.0)) - float(batch[-1].get("end", 0.0))
        return max(0.0, min(gap, GAP_KEEP_REAL_MAX_SEC))

    def finalize(
        part: List[Dict[str, object]],
        words: List[List[Dict[str, object]]],
        asr_segments: List[List[Dict[str, object]]],
        part_lang: str,
        part_auto: bool,
    ) -> None:
        finalized = _finalize_group_candidate(
            part,
            words,
            asr_segments,
            audio,
            sr,
            lang=part_lang,
            audio_loader=audio_loader,
        )
        out_segments.extend(finalized)
        if part_auto:
            _record_auto_detected_segment_languages(language_history, finalized)

    def transcribe(
        part: List[Dict[str, object]],
        successor: Optional[Dict[str, object]],
    ):
        return _transcribe_group_candidate(
            model,
            part,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=language_history,
            audio_loader=audio_loader,
            tail_real_limit_sec=window_tail(part, successor),
        )

    out_segments: List[Dict[str, object]] = []
    remaining = list(intervals)
    p_words, p_segments, p_lang, p_issues, p_auto = candidate
    if not p_issues:
        finalize(remaining, p_words, p_segments, p_lang, p_auto)
        return out_segments, []
    k = _first_abnormal_interval_index(p_words)
    if k is None:
        _note(
            "abnormal ASR issues not attributable to one interval; "
            "keeping cleaned window result "
            f"(start={float(remaining[0].get('start', 0.0)):.3f}s, "
            f"issues={_issues_summary(p_issues)})",
            count="unattributed_windows",
        )
        finalize(remaining, p_words, p_segments, p_lang, p_auto)
        return out_segments, []
    iso = remaining[k]
    iso_start = float(iso.get("start", 0.0))
    iso_end = float(iso.get("end", 0.0))
    iso_issues_own = detect_abnormal_asr_words([p_words[k]])
    _note(
        "isolating abnormal interval "
        f"(interval={iso_start:.3f}-{iso_end:.3f}, "
        f"clean_front={k}, rest={len(remaining) - k - 1}, "
        f"issues={_issues_summary(iso_issues_own)})",
        count="isolated_intervals",
    )
    if k > 0:
        front = remaining[:k]
        f_words, f_segments, f_lang, f_issues, f_auto = transcribe(front, iso)
        if f_issues:
            # A degenerate solo re-decode (e.g. a laugh-loop swallowing
            # real speech) must not replace the candidate's front slice.
            # But "interval-clean" can also mean interval-EMPTY: word
            # attachment is whole-whisper-segment by dominant interval, so
            # the candidate may have parked the front's speech on the
            # abnormal interval, leaving the slice hollow. Keep the slice
            # only when it actually covers the front's speech; otherwise
            # decode the front interval-by-interval (the old fallback,
            # which recovers such regions).
            trimmed = [
                [
                    w
                    for w in ws
                    if (float(w["start"]) + float(w["end"])) / 2.0 < iso_start
                ]
                for ws in p_words[:k]
            ]
            slice_finalized = _finalize_group_candidate(
                front,
                trimmed,
                p_segments[:k],
                audio,
                sr,
                lang=p_lang,
                audio_loader=audio_loader,
            )
            # The slice has to be clean *as a whole*, not merely "no single
            # interval got flagged". _first_abnormal_interval_index tests
            # each interval on its own, so a collapse spanning several
            # intervals is diluted below COLLAPSE_STACK_MIN_RUN in every one
            # of them: each looks clean while the slice text *is* the
            # collapse. Re-checking the joined slice catches that.
            slice_issues = detect_abnormal_asr_words(trimmed)
            if (
                not slice_issues
                and _coverage_shortfall(front, slice_finalized) is None
            ):
                _note(
                    "clean-front window re-decode abnormal; keeping the "
                    "original window's clean slice instead "
                    f"(start={float(front[0].get('start', 0.0)):.3f}s, "
                    f"issues={_issues_summary(f_issues)})",
                    count="front_redecode_rejected",
                )
                out_segments.extend(slice_finalized)
                if p_auto:
                    _record_auto_detected_segment_languages(
                        language_history, slice_finalized
                    )
            else:
                _note(
                    "clean-front window re-decode abnormal and the "
                    "candidate slice is unusable too; isolating within the front "
                    f"(start={float(front[0].get('start', 0.0)):.3f}s, "
                    f"redecode_issues={_issues_summary(f_issues)}, "
                    f"slice_issues={_issues_summary(slice_issues) or 'none'}, "
                    f"slice_coverage_low={_coverage_shortfall(front, slice_finalized) is not None})",
                    count="front_redecode_rejected",
                )
                # Recurse rather than shattering the front one interval per
                # window: the front is just a shorter abnormal group, and
                # only the interval that actually fails needs isolating.
                # Terminates because the front is strictly shorter than the
                # window that produced it.
                # The front is a prefix: it has to be fully consumed here
                # (only the tail after `iso` is handed back), so loop until
                # nothing is left. Each round is strictly shorter.
                f_remaining = list(front)
                f_candidate = (f_words, f_segments, f_lang, f_issues, f_auto)
                while f_remaining:
                    f_segments_out, f_rest = _isolate_abnormal_intervals(
                        model,
                        f_remaining,
                        f_candidate,
                        audio,
                        sr,
                        gap_sec,
                        language=language,
                        language_history=language_history,
                        audio_loader=audio_loader,
                        tail_limit_sec=window_tail(f_remaining, iso),
                    )
                    out_segments.extend(f_segments_out)
                    f_remaining = f_rest
                    if f_remaining:
                        f_candidate = transcribe(f_remaining, iso)
        else:
            finalize(front, f_words, f_segments, f_lang, f_auto)
    successor = remaining[k + 1] if k + 1 < len(remaining) else None
    i_words, i_segments, i_lang, i_issues, i_auto = transcribe([iso], successor)
    if i_issues:
        _note(
            "interval-level ASR still abnormal; applying merge-based cleanup fallback "
            f"(interval={iso_start:.3f}-{iso_end:.3f}, "
            f"issues={_issues_summary(i_issues)})",
            count="merge_cleanup_fallbacks",
        )
    finalize([iso], i_words, i_segments, i_lang, i_auto)
    return out_segments, list(remaining[k + 1 :])


def align_group(
    model,
    group: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    gap_sec: float,
    *,
    language: Optional[str],
    auto_language_history: Optional[List[str]] = None,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Returns (segments, unconsumed): isolation hands back everything after
    the interval it rescued so the caller can re-group it with what follows."""

    language_history = auto_language_history if auto_language_history is not None else []

    def batch_tail_limit(
        batch: List[Dict[str, object]],
        successor: Optional[Dict[str, object]],
    ) -> float:
        """Real-audio tail allowance: capped by the gap to the successor
        interval so the tail pad never bleeds into speech that another batch
        transcribes."""

        if not batch:
            return 0.0
        if successor is None:
            return tail_real_limit_sec
        gap = float(successor.get("start", 0.0)) - float(batch[-1].get("end", 0.0))
        return max(0.0, min(gap, tail_real_limit_sec))

    (
        per_interval_words,
        per_interval_asr_segments,
        lang,
        issues,
        uses_auto_detection,
    ) = _transcribe_group_candidate(
        model,
        group,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=language_history,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    if not issues:
        finalized = _finalize_group_candidate(
            group,
            per_interval_words,
            per_interval_asr_segments,
            audio,
            sr,
            lang=lang,
            audio_loader=audio_loader,
        )
        if uses_auto_detection:
            _record_auto_detected_segment_languages(language_history, finalized)
        return finalized, []

    if _is_known_phrase_stack_only(per_interval_words, issues):
        # Phrase-only hallucination stack: nothing recoverable underneath, and
        # the stabilize phrase cleanup removes it wholesale. Skip the rescue
        # ladder so the squeeze form is not converted into a stretched one.
        _note(
            "known hallucination phrase stack; skipping abnormal rescue ladder "
            f"(start={float(group[0].get('start', 0.0)):.3f}s, "
            f"issues={_issues_summary(issues)})",
            count="phrase_stack_groups",
        )
        finalized = _finalize_group_candidate(
            group,
            per_interval_words,
            per_interval_asr_segments,
            audio,
            sr,
            lang=lang,
            audio_loader=audio_loader,
        )
        if uses_auto_detection:
            _record_auto_detected_segment_languages(language_history, finalized)
        return finalized, []

    # Rescue: hand the group straight to abnormal-interval isolation.
    #
    # The regroup-retry ladder and the whole-group beam decode that used to sit
    # in front of this were measured against isolation on 26 rescue groups and
    # lost on every axis: regroup reproduced the failed greedy verbatim in 47%
    # of groups (a pure no-op) and had the highest repeat-loop rate of all
    # paths (42% of lines, worse than the greedy it was rescuing), while beam
    # both collapsed and dropped content (139 output lines vs isolation's 164,
    # one group losing 28s of audio entirely). Isolation is the only path that
    # targets the failing position instead of re-rolling the whole window.
    abnormal_start = _first_abnormal_interval_start(group, per_interval_words)
    _note(
        "abnormal ASR result; isolating abnormal intervals "
        f"(issues={_issues_summary(issues)}"
        + (
            f", abnormal_start={abnormal_start:.3f}s"
            if abnormal_start is not None
            else ""
        )
        + ")",
        count="abnormal_groups",
    )
    return _isolate_abnormal_intervals(
        model,
        group,
        (
            per_interval_words,
            per_interval_asr_segments,
            lang,
            issues,
            uses_auto_detection,
        ),
        audio,
        sr,
        gap_sec,
        language=language,
        language_history=language_history,
        audio_loader=audio_loader,
        tail_limit_sec=tail_real_limit_sec,
    )


def _segment_sort_key(seg: Dict[str, object]) -> Tuple[float, float]:
    return (float(seg.get("start", 0.0)), float(seg.get("end", 0.0)))


def _sort_segments_by_time(
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    return sorted(segments, key=_segment_sort_key)


def _extract_merged_segment_spans(
    segments: List[Dict[str, object]],
) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append((start, end))
    spans.sort(key=lambda x: (x[0], x[1]))
    merged: List[Tuple[float, float]] = []
    for start, end in spans:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
            continue
        merged.append((start, end))
    return merged


def _extract_interval_spans(
    intervals: List[Dict[str, object]],
) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    for interval in intervals:
        try:
            start = float(interval.get("start", 0.0))
            end = float(interval.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        spans.append((start, end))
    spans.sort()
    return spans


def _compute_complement_intervals_from_spans(
    intervals: List[Dict[str, object]],
    segment_spans: List[Tuple[float, float]],
) -> List[Dict[str, object]]:
    interval_spans = _extract_interval_spans(intervals)
    complements: List[Dict[str, object]] = []
    if not interval_spans:
        return complements

    seg_idx = 0
    n_segments = len(segment_spans)
    for interval_start, interval_end in interval_spans:
        while seg_idx < n_segments and segment_spans[seg_idx][1] <= interval_start:
            seg_idx += 1
        cursor = interval_start
        scan_idx = seg_idx
        while scan_idx < n_segments:
            seg_start, seg_end = segment_spans[scan_idx]
            if seg_start >= interval_end:
                break
            overlap_start = max(interval_start, seg_start)
            overlap_end = min(interval_end, seg_end)
            if overlap_end <= overlap_start:
                scan_idx += 1
                continue
            if overlap_start > cursor:
                complements.append(
                    {"start": float(cursor), "end": float(overlap_start)}
                )
            cursor = max(cursor, overlap_end)
            if cursor >= interval_end:
                break
            scan_idx += 1
        if cursor < interval_end:
            complements.append(
                {"start": float(cursor), "end": float(interval_end)}
            )
    return complements


def _compute_complement_intervals(
    intervals: List[Dict[str, object]],
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    segment_spans = _extract_merged_segment_spans(segments)
    return _compute_complement_intervals_from_spans(intervals, segment_spans)


def _has_segment_between(
    segment_spans: List[Tuple[float, float]],
    start: float,
    end: float,
    *,
    start_idx: int = 0,
) -> Tuple[bool, int]:
    if end <= start:
        return False, max(0, int(start_idx))
    idx = max(0, int(start_idx))
    while idx < len(segment_spans) and segment_spans[idx][1] <= start:
        idx += 1
    if idx >= len(segment_spans):
        return False, idx
    seg_start, seg_end = segment_spans[idx]
    has_segment = (seg_start < end) and (seg_end > start)
    return has_segment, idx


def _build_recall_temp_groups(
    intervals: List[Dict[str, object]],
    segments: Optional[List[Dict[str, object]]] = None,
    *,
    segment_spans: Optional[List[Tuple[float, float]]] = None,
    min_drop_time_recall: float = MIN_DROP_TIME_RECALL,
    min_complement_sec: float = RECALL_COMPLEMENT_MIN_SEC,
) -> List[List[Dict[str, object]]]:
    threshold = max(0.0, float(min_drop_time_recall))
    if segment_spans is None:
        segment_spans = _extract_merged_segment_spans(segments or [])
    complements = _compute_complement_intervals_from_spans(intervals, segment_spans)
    if not complements:
        return []

    recall_groups: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    current_total = 0.0
    seg_idx = 0

    for comp in complements:
        comp_start = float(comp["start"])
        comp_end = float(comp["end"])
        if comp_end <= comp_start or comp_end - comp_start < min_complement_sec:
            continue
        comp_item = {"start": comp_start, "end": comp_end}
        comp_dur = comp_end - comp_start
        if not current:
            current = [comp_item]
            current_total = comp_dur
            continue
        prev_end = float(current[-1]["end"])
        blocked, seg_idx = _has_segment_between(
            segment_spans,
            prev_end,
            comp_start,
            start_idx=seg_idx,
        )
        if blocked:
            if current_total >= threshold:
                recall_groups.append(current)
            current = [comp_item]
            current_total = comp_dur
            continue
        current.append(comp_item)
        current_total += comp_dur

    if current and current_total >= threshold:
        recall_groups.append(current)
    return recall_groups


def _recall_tail_limit_sec(
    temp_group: List[Dict[str, object]],
    segment_spans: List[Tuple[float, float]],
    upcoming_interval_starts: List[float],
) -> float:
    """Real-audio tail allowance for a recall batch.

    Same principle as normal group tails (up to GAP_KEEP_REAL_MAX_SEC of gap
    audio preserves low-energy endings), but additionally bounded by the
    next covered segment span: recall complements often end exactly where a
    normal segment starts, and padding into it would re-transcribe covered
    speech."""

    if not temp_group:
        return 0.0
    last_end = float(temp_group[-1].get("end", 0.0))
    limit = GAP_KEEP_REAL_MAX_SEC
    for span_start, span_end in segment_spans:
        if span_end <= last_end:
            continue
        limit = min(limit, max(0.0, span_start - last_end))
        break
    for interval_start in upcoming_interval_starts:
        if interval_start > last_end:
            limit = min(limit, interval_start - last_end)
            break
    return max(0.0, limit)


def _select_tail_segments_for_block(
    block_intervals: List[Dict[str, object]],
    block_segments: List[Dict[str, object]],
    *,
    tail_sec: float = PREV_BLOCK_TAIL_SEC,
) -> List[Dict[str, object]]:
    if not block_intervals or not block_segments:
        return []
    try:
        block_end = max(float(item.get("end", 0.0)) for item in block_intervals)
    except Exception:
        return []
    tail_start = block_end - max(0.0, float(tail_sec))
    tail_segments: List[Dict[str, object]] = []
    for seg in block_segments:
        try:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", 0.0))
        except (TypeError, ValueError):
            continue
        if seg_end <= seg_start:
            continue
        if seg_end <= tail_start:
            continue
        if seg_start >= block_end:
            continue
        tail_segments.append(seg)
    return _sort_segments_by_time(tail_segments)


def _align_group_consume_all(
    model,
    intervals: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    gap_sec: float,
    *,
    language: Optional[str],
    auto_language_history: Optional[List[str]] = None,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> List[Dict[str, object]]:
    """align_group, looping until every interval is consumed.

    Only align_segments' main loop can usefully re-group a handed-back tail
    (it has the intervals that follow); every other caller -- coverage rescue's
    peeling loop, recall batches -- must finish what it was given."""

    out: List[Dict[str, object]] = []
    rest = list(intervals)
    while rest:
        segments, unconsumed = align_group(
            model,
            rest,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=auto_language_history,
            audio_loader=audio_loader,
            tail_real_limit_sec=tail_real_limit_sec,
        )
        out.extend(segments)
        rest = unconsumed
    return out


def _rescue_low_coverage(
    model,
    group: List[Dict[str, object]],
    segments: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    gap_sec: float,
    *,
    language: Optional[str],
    auto_language_history: Optional[List[str]] = None,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> List[Dict[str, object]]:
    """Rescue ladder for window-skipped batches (see coverage tunables).

    Step 1: beam decode of the whole batch; accepted when clean and no
    longer coverage-low. Step 2+: peel the first interval into its own
    greedy window and re-decode the rest as one window, repeating while the
    rear window stays coverage-low (converges to interval-by-interval).
    The rescue result replaces the original only when it covers more."""

    shortfall = _coverage_shortfall(group, segments)
    if shortfall is None:
        return segments
    language_history = auto_language_history if auto_language_history is not None else []
    speech, covered, required = shortfall
    group_start = float(group[0].get("start", 0.0)) if group else 0.0
    _note(
        "low ASR coverage; trying beam rescue "
        f"(start={group_start:.3f}s, covered={covered:.3f}s, "
        f"speech={speech:.3f}s, required={required:.3f}s)",
        count="beam_rescue_attempted",
    )

    (
        beam_words,
        beam_asr_segments,
        beam_lang,
        beam_issues,
        beam_uses_auto_detection,
    ) = _transcribe_group_candidate(
        model,
        group,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=language_history,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
        decode_options=_rescue_decode_options(),
    )
    if not beam_issues:
        beam_finalized = _finalize_group_candidate(
            group,
            beam_words,
            beam_asr_segments,
            audio,
            sr,
            lang=beam_lang,
            audio_loader=audio_loader,
        )
        if _coverage_shortfall(group, beam_finalized) is None:
            _note(
                "beam rescue accepted "
                f"(start={group_start:.3f}s, "
                f"covered={_covered_speech_seconds(group, beam_finalized):.3f}s, "
                f"segments={len(beam_finalized)})",
                count="beam_rescue_accepted",
            )
            if beam_uses_auto_detection:
                _record_auto_detected_segment_languages(
                    language_history, beam_finalized
                )
            return beam_finalized

    if len(group) <= 1:
        return segments

    _note(
        "beam rescue insufficient; splitting group "
        f"(start={group_start:.3f}s, intervals={len(group)})",
        count="beam_rescue_split",
    )
    # Auto-language history written by rejected split windows must not leak
    # (matches the rule that unaccepted candidates never enter the history).
    history_snapshot = list(language_history)

    def window_tail_limit(
        batch: List[Dict[str, object]],
        successor: Optional[Dict[str, object]],
    ) -> float:
        if not batch:
            return 0.0
        if successor is None:
            return tail_real_limit_sec
        gap = float(successor.get("start", 0.0)) - float(batch[-1].get("end", 0.0))
        return max(0.0, min(gap, tail_real_limit_sec))

    rescued: List[Dict[str, object]] = []
    remaining_intervals = list(group)
    while remaining_intervals:
        head = remaining_intervals[:1]
        rest = remaining_intervals[1:]
        head_segments = _align_group_consume_all(
            model,
            head,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=language_history,
            audio_loader=audio_loader,
            tail_real_limit_sec=window_tail_limit(head, rest[0] if rest else None),
        )
        rescued.extend(head_segments)
        if not rest:
            break
        rest_segments = _align_group_consume_all(
            model,
            rest,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=language_history,
            audio_loader=audio_loader,
            tail_real_limit_sec=tail_real_limit_sec,
        )
        if _coverage_shortfall(rest, rest_segments) is None:
            rescued.extend(rest_segments)
            break
        _note(
            "rear window coverage still low; peeling another interval "
            f"(rest_start={float(rest[0].get('start', 0.0)):.3f}s, "
            f"rest_intervals={len(rest)})",
            count="coverage_peels",
        )
        remaining_intervals = rest

    rescued_covered = _covered_speech_seconds(group, rescued)
    original_covered = _covered_speech_seconds(group, segments)
    if rescued_covered <= original_covered:
        language_history[:] = history_snapshot
        _note(
            "coverage rescue kept original result "
            f"(start={group_start:.3f}s, rescued={rescued_covered:.3f}s, "
            f"original={original_covered:.3f}s)",
            count="coverage_rescue_rejected",
        )
        return segments
    _note(
        "coverage rescue accepted split result "
        f"(start={group_start:.3f}s, rescued={rescued_covered:.3f}s, "
        f"original={original_covered:.3f}s)",
        count="coverage_rescue_accepted",
    )
    return rescued


def _align_intervals_group(
    group: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    *,
    model,
    gap_sec: float,
    language: Optional[str],
    auto_language_history: Optional[List[str]] = None,
    audio_loader: Optional[AudioBlockLoader] = None,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    # Group boundary for the auto-language history: everything the decodes,
    # retries and rescues below record collapses into one entry. Only relevant
    # under auto-detection -- an explicit language neither reads nor writes it.
    history_mark = len(auto_language_history) if auto_language_history is not None else 0
    aligned, unconsumed = align_group(
        model,
        group,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=auto_language_history,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    # Coverage rescue only judges the part that was actually consumed; the
    # handed-back tail has not been decoded yet.
    consumed = group[: len(group) - len(unconsumed)] if unconsumed else group
    aligned = _rescue_low_coverage(
        model,
        consumed,
        aligned,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=auto_language_history,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    if auto_language_history is not None and not language:
        _collapse_group_language_entries(auto_language_history, history_mark)
    return _sort_segments_by_time(aligned), unconsumed


def _majority_language(langs: List[object]) -> Optional[str]:
    """Most frequent usable language; on a tie the most recent one wins."""

    values = [
        str(lang).strip()
        for lang in langs
        if str(lang).strip() and str(lang).strip() != "None"
    ]
    if not values:
        return None
    counts: Dict[str, int] = {}
    for lang in values:
        counts[lang] = counts.get(lang, 0) + 1
    max_count = max(counts.values())
    return next(lang for lang in reversed(values) if counts[lang] == max_count)


def _most_frequent_recent_language(
    auto_language_history: List[str],
    *,
    history_groups: int = AUTO_LANGUAGE_HISTORY_GROUPS,
) -> Optional[str]:
    keep = max(0, int(history_groups))
    if keep == 0:
        return None
    return _majority_language(list(auto_language_history[-keep:]))


def _language_for_group(
    configured_language: Optional[str],
    group: List[Dict[str, object]],
    *,
    gap_sec: float,
    auto_language_history: List[str],
) -> Tuple[Optional[str], bool]:
    """Return (effective language, whether this call remains auto-detected)."""

    if configured_language:
        return configured_language, False
    group_duration = combined_group_duration(group, gap_sec=gap_sec)
    if group_duration <= AUTO_LANGUAGE_SHORT_GROUP_SEC:
        recent_language = _most_frequent_recent_language(auto_language_history)
        if recent_language:
            _note(
                "short group reuses recent auto-detected language "
                f"(duration={group_duration:.3f}s, "
                f"language={recent_language}, "
                f"history={auto_language_history})",
                count="short_group_language_reuse",
            )
            return recent_language, False
    return None, True


def _record_auto_detected_segment_languages(
    auto_language_history: List[str],
    segments: List[Dict[str, object]],
) -> None:
    """Stage one decode's segment languages for the group being processed.

    Entries land unaggregated and are folded into a single one by
    ``_collapse_group_language_entries`` at the group boundary, so trimming
    happens there rather than here."""

    for segment in segments:
        lang = str(segment.get("lang") or "").strip()
        if lang and lang != "None":
            auto_language_history.append(lang)


def _collapse_group_language_entries(
    auto_language_history: List[str],
    mark: int,
) -> None:
    """Fold everything recorded since ``mark`` into one entry for this group.

    The history counts detection events per group, not segments. A group can
    emit dozens of segments -- a hallucination stack, or an abnormal window
    isolated into many sub-intervals, each decoded separately -- and letting
    every one of them vote would fill the whole window from a single group,
    flipping the language that later short groups reuse. One group now
    contributes at most one language, so recovering only takes the next
    normally-detected group."""

    dominant = _majority_language(list(auto_language_history[mark:]))
    del auto_language_history[mark:]
    if dominant:
        auto_language_history.append(dominant)
    keep = max(0, int(AUTO_LANGUAGE_HISTORY_GROUPS))
    if keep == 0:
        auto_language_history.clear()
    elif len(auto_language_history) > keep:
        del auto_language_history[:-keep]


def _next_interval_start(
    remaining: List[Dict[str, object]],
    group_size: int,
    successor_start: Optional[float],
) -> Optional[float]:
    """Start of the interval following the current group, or None at the true
    end of the audio.

    ``successor_start`` lets a caller that holds only part of the timeline say
    "the file does not end here": without it the last group would pad and recall
    as if nothing followed. It outlived the sharding that introduced it because
    recall chains still need the distinction."""

    if group_size < len(remaining):
        return float(remaining[group_size].get("start", 0.0))
    return successor_start


def align_segments(
    intervals: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    *,
    model,
    gap_sec: float,
    language: Optional[str],
    audio_loader: Optional[AudioBlockLoader] = None,
    checkpoint_path: Optional[str | Path] = None,
    checkpoint_key: Optional[Dict[str, object]] = None,
    successor_start: Optional[float] = None,
) -> List[Dict[str, object]]:
    if not intervals:
        return []

    out: List[Dict[str, object]] = []
    remaining: List[Dict[str, object]] = list(intervals)
    total_intervals = len(remaining)
    processed_intervals = 0
    group_idx = 0
    _stats_local.progress_step = None
    prev_tail_segments: List[Dict[str, object]] = []
    auto_language_history: List[str] = []

    # Group boundaries are the natural checkpoint: `out`, `remaining`,
    # `prev_tail_segments` and `auto_language_history` are the complete state
    # there, so a crash costs at most one group instead of the whole run.
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    fingerprint: Dict[str, object] = {}
    if checkpoint is not None:
        fingerprint = dict(checkpoint_key or {})
        fingerprint["intervals"] = checkpoint_store.intervals_digest(intervals)
        resumed = checkpoint_store.load(checkpoint, fingerprint)
        if resumed is not None:
            out = list(resumed.get("segments") or [])
            processed_intervals = int(resumed["processed_intervals"])
            group_idx = int(resumed.get("group_idx") or 0)
            prev_tail_segments = list(resumed.get("prev_tail_segments") or [])
            auto_language_history = list(resumed.get("auto_language_history") or [])
            remaining = list(intervals[processed_intervals:])
            _note(
                "resuming ASR from checkpoint "
                f"(intervals={processed_intervals}/{total_intervals}, "
                f"groups_done={group_idx}, segments={len(out)})"
            )
    # Straight away, before the first group takes its time: a resumed run that
    # starts at 60% must not look like it starts at 0, and a fresh one should
    # say how much there is to do rather than nothing at all.
    _report_progress(processed_intervals, total_intervals, group_idx)

    while remaining:
        dynamic_groups = build_alignment_groups(remaining, gap_sec=gap_sec)
        if dynamic_groups:
            group = dynamic_groups[0]
        else:
            group = [remaining[0]]
        if not group:
            break
        group_size = len(group)
        group_idx += 1

        group_start = float(group[0].get("start", 0.0)) if group else 0.0
        processed_after = processed_intervals + group_size
        progress_pct = 100.0 * processed_after / max(total_intervals, 1)
        _note(
            "group ASR "
            f"(start={group_start:.3f}s, "
            f"progress={progress_pct:.1f}%, "
            f"group_iter={group_idx}, "
            f"intervals={processed_after}/{total_intervals})"
        )
        next_start = _next_interval_start(remaining, group_size, successor_start)
        if next_start is not None:
            next_gap = next_start - float(group[-1].get("end", 0.0))
            group_tail_limit = max(0.0, min(next_gap, GAP_KEEP_REAL_MAX_SEC))
        else:
            group_tail_limit = GAP_KEEP_REAL_MAX_SEC
        normal_segments, unconsumed = _align_intervals_group(
            group,
            audio,
            sr,
            model=model,
            gap_sec=gap_sec,
            language=language,
            auto_language_history=auto_language_history,
            audio_loader=audio_loader,
            tail_real_limit_sec=group_tail_limit,
        )
        # Isolation hands back everything after the interval it rescued. Those
        # intervals go back on the queue so the next round re-groups them
        # together with what follows, instead of decoding a short remainder on
        # its own with the least context of any window.
        # Isolation always consumes at least the interval it rescued, so the
        # queue cannot stall; fail loudly rather than spin forever.
        consumed_size = group_size - len(unconsumed)
        if consumed_size <= 0:
            raise RuntimeError(
                f"alignment made no progress at "
                f"{float(group[0].get('start', 0.0)):.3f}s "
                f"(group={group_size}, unconsumed={len(unconsumed)})"
            )
        group = group[:consumed_size]
        group_size = consumed_size
        processed_after = processed_intervals + consumed_size

        segments_for_complement = normal_segments + prev_tail_segments
        segment_spans_for_complement = _extract_merged_segment_spans(
            segments_for_complement
        )
        temp_groups = _build_recall_temp_groups(
            group,
            segment_spans=segment_spans_for_complement,
            min_drop_time_recall=MIN_DROP_TIME_RECALL,
        )

        temp_segments: List[Dict[str, object]] = []
        if temp_groups:
            _note(
                "temporary recall groups "
                f"(count={len(temp_groups)}, threshold={MIN_DROP_TIME_RECALL:.3f}s, "
                f"group_iter={group_idx})"
            )
        upcoming_interval_starts = [
            float(interval.get("start", 0.0)) for interval in group
        ]
        # Same successor rule as the tail limit above: a shard's last group must
        # not treat itself as the end of the file, or its recall windows would
        # run past the real next interval.
        next_start = _next_interval_start(remaining, group_size, successor_start)
        if next_start is not None:
            upcoming_interval_starts.append(next_start)
        for temp_idx, temp_group in enumerate(temp_groups, start=1):
            temp_start = float(temp_group[0].get("start", 0.0)) if temp_group else 0.0
            temp_total = 0.0
            for interval in temp_group:
                temp_total += max(
                    0.0,
                    float(interval.get("end", 0.0)) - float(interval.get("start", 0.0)),
                )
            _note(
                "temporary recall ASR "
                f"(start={temp_start:.3f}s, total_interval_sec={temp_total:.3f}, "
                f"group_iter={group_idx}, temp_group={temp_idx}/{len(temp_groups)})",
                count="temporary_recalls",
            )
            aligned_temp_segments, _temp_unconsumed = _align_intervals_group(
                temp_group,
                audio,
                sr,
                model=model,
                gap_sec=gap_sec,
                language=language,
                auto_language_history=auto_language_history,
                audio_loader=audio_loader,
                    # Tail pad up to the next covered span / interval so a recall
                # chain ending at an interval edge keeps its low-energy tail
                # without re-transcribing covered speech.
                tail_real_limit_sec=_recall_tail_limit_sec(
                    temp_group,
                    segment_spans_for_complement,
                    upcoming_interval_starts,
                ),
            )
            temp_segments.extend(aligned_temp_segments)

        block_segments = _sort_segments_by_time(normal_segments + temp_segments)
        out.extend(block_segments)
        prev_tail_segments = _select_tail_segments_for_block(
            group,
            block_segments,
            tail_sec=PREV_BLOCK_TAIL_SEC,
        )
        remaining = remaining[group_size:]
        processed_intervals = processed_after
        # Reported here, not when the group was picked: the rescue ladder hands
        # back intervals it could not consume, so the count chosen up front is
        # a prediction. Reporting that one made progress reach 100% and then
        # keep going -- the same "do not claim work that did not land" rule the
        # separator follows for cancelled blocks.
        _report_progress(processed_intervals, total_intervals, group_idx)
        if checkpoint is not None:
            checkpoint_store.write(
                checkpoint,
                {
                    "version": checkpoint_store.SCHEMA_VERSION,
                    "fingerprint": fingerprint,
                    "processed_intervals": processed_intervals,
                    "group_idx": group_idx,
                    "segments": out,
                    "prev_tail_segments": prev_tail_segments,
                    "auto_language_history": auto_language_history,
                },
            )
    if checkpoint is not None:
        checkpoint_store.clear(checkpoint)
    return out


def default_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-asr.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ASR alignment from VAD JSON output."
    )
    parser.add_argument("input", help="Path to VAD JSON file.")
    parser.add_argument("--output", "-o", help="Path to output JSON file.")
    parser.add_argument(
        "--audio",
        required=True,
        help="Path to audio file that the VAD JSON was generated from.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda).")
    parser.add_argument("--language", default=None, help="Language override.")
    parser.add_argument(
        "--gap",
        type=float,
        default=DEFAULT_GAP_SEC,
        help=(
            "Synthetic silence inserted before each next interval and at the "
            "group tail (after up to 0.7s of kept real gap audio)."
        ),
    )
    parser.add_argument(
        "--block-seconds",
        type=float,
        default=600.0,
        help="Block size in seconds for streaming ASR (default: 600). Use 0 to disable.",
    )
    parser.add_argument(
        "--pad-seconds",
        type=float,
        default=10.0,
        help="Padding seconds on each side of a block (default: 10).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Normalize "auto" to None (whisper auto-detection).
    if args.language and args.language.strip().lower() == "auto":
        args.language = None
    device_for_usage: Optional[str] = None
    model = None
    # This module is also a standalone dev CLI; without a bound renderer its
    # recovery notes and warnings would reach a reporter that shows nothing.
    scope = ExitStack()
    scope.enter_context(reporting_to(terminal_reporter()))
    scope.enter_context(collecting_stats())
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"Input not found: {input_path}", file=sys.stderr)
            return 1

        device = resolve_device(args.device or DEFAULT_DEVICE, context="ASR alignment")
        device_for_usage = device
        reset_peak_gpu_memory_stats_for_run(device_for_usage)
        align_meta = asr_align_metadata(
            model=args.model,
            device=device,
            language=args.language,
            gap_sec=args.gap,
        )

        output_path = Path(args.output) if args.output else default_output_path(input_path)
        data = json.loads(input_path.read_text(encoding="utf-8"))
        raw_segments = data.get("segments") or []
        if not raw_segments:
            metadata = merge_metadata(data.get("metadata", {}), align_meta)
            payload = {"segments": [], "metadata": metadata}
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {output_path}")
            return 0

        audio_path = Path(args.audio).expanduser().resolve()
        if not audio_path.exists():
            print("Audio file not found. Provide --audio.", file=sys.stderr)
            return 1

        try:
            from .fw_refine_backend import RefinedWhisperModel
        except Exception:
            print(
                "Missing dependency: faster-whisper plus the patched CTranslate2 "
                'runtime. Install with `pip install -e ".[asr]"`.',
                file=sys.stderr,
            )
            return 1

        t0 = time.perf_counter()
        audio_loader = AudioBlockLoader(
            str(audio_path),
            target_sr=TARGET_SR,
            block_seconds=args.block_seconds,
            pad_seconds=args.pad_seconds,
            preprocess=False,
        )
        t_prepare = time.perf_counter() - t0

        audio_duration = audio_loader.duration
        segments = normalize_vad_segments(raw_segments, audio_duration)
        if not segments:
            metadata = merge_metadata(data.get("metadata", {}), align_meta)
            payload = {"segments": [], "metadata": metadata}
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {output_path}")
            return 0

        t0 = time.perf_counter()
        model = RefinedWhisperModel(
            args.model,
            device=device,
            compute_type="float16" if device.startswith("cuda") else "float32",
            refine_sec=REFINE_SEC,
        )
        t_model = time.perf_counter() - t0
        t0 = time.perf_counter()
        aligned_segments = align_segments(
            segments,
            None,
            TARGET_SR,
            model=model,
            gap_sec=args.gap,
            language=args.language,
            audio_loader=audio_loader,
            checkpoint_path=checkpoint_store.path_for_output(output_path),
            checkpoint_key=checkpoint_store.build_key(
                model_name=args.model,
                language=args.language,
                gap_sec=args.gap,
                audio_path=input_path,
                detect_disfluencies=FW_REFINE_DETECT_DISFLUENCIES,
            ),
        )
        t_align = time.perf_counter() - t0

        # No energy track here, so the disfluency blocks all merge back
        # (plain-decode starts, span labels kept); the energy-gated variants
        # run in the combined vad-asr stage.
        aligned_segments, correction_stats = word_starts.apply_disfluency_rules(
            aligned_segments,
            energy_track=None,
        )
        align_meta["word_start_correction"] = correction_stats

        segments = segment_ops.drop_empty_segments(aligned_segments)
        output_segments = [round_floats(seg) for seg in segments]
        metadata = merge_metadata(data.get("metadata", {}), align_meta)
        payload = {
            "segments": output_segments,
            "metadata": metadata,
        }

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        print("Timing:")
        print(f"  prepare_audio_sec: {t_prepare:.3f}")
        print(f"  whisper_load_sec: {t_model:.3f}")
        print(f"  asr_align_sec: {t_align:.3f}")
        return 0
    finally:
        if model is not None:
            try:
                del model
            except Exception:
                pass
        gc.collect()
        if device_for_usage is not None and device_for_usage.strip().lower() == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print_peak_resource_usage(device_for_usage)
        scope.close()


if __name__ == "__main__":
    raise SystemExit(main())
