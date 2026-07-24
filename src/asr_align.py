"""ASR alignment pipeline using Whisper timestamped output and VAD post-processing."""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils import (
    COLLAPSE_STACK_MIN_RUN,
    COLLAPSE_STACK_WORD_SEC,
    COMMON_HALLUCINATION_TEXT,
    GROUP_REPEAT_MIN_COUNT,
    GROUP_REPEAT_MIN_UNITS,
    LONG_WORD_WORDS,
    LONG_WORD_SEC,
    REPEAT_DETECT_MORE_THAN,
    REPEAT_KEEP_RUN,
    TARGET_SR,
    apply_bandpass,
    as_numpy_float32,
    cleanup_asr_words_for_fallback,
    coerce_optional_float,
    copy_float_fields,
    detect_abnormal_asr_words,
    detect_collapse_word_stack,
    min_word_confidence,
    get_audio_info,
    load_audio_slice,
    print_peak_resource_usage,
    reset_peak_gpu_memory_stats_for_run,
    resample_if_needed,
    to_mono,
    weighted_spectral_energy_db,
    words_to_text,
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
ASR_REGROUP_MAX_RETRIES = 2
ASR_TRANSCRIBE_SEED = 0  # Fixed seed for deterministic transcribe calls.
WHISPER_TIMESTAMPED_MODE = "efficient"
WHISPER_TIMESTAMPED_REFINE_SEC = 1.0
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

# --------- Tunables (ASR grouping) ---------
GROUP_TARGET_SEC = 30.0  # Start searching for a split only after this total (sec).
BASE_BREAK_LENGTH = 1.0  # Base gap scale used in adaptive split threshold.
MIN_GROUP_LENGTH = 15.0  # Minimum non-final group duration (sec).
AUTO_LANGUAGE_SHORT_GROUP_SEC = 10.0
AUTO_LANGUAGE_HISTORY_SEGMENTS = 10
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
ZERO_LENGTH_SEGMENT_EXTEND_SEC = 0.01

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
# interval-by-interval fallback. Beam forces whisper-timestamped onto its
# naive two-pass alignment for that call, so rescued word timestamps are
# less precise than the efficient path's.
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
        "zero_length_segment_extend_sec": ZERO_LENGTH_SEGMENT_EXTEND_SEC,
        "asr_coverage_min_ratio": ASR_COVERAGE_MIN_RATIO,
        "asr_coverage_tolerance_sec": ASR_COVERAGE_TOLERANCE_SEC,
        "asr_rescue_beam_size": ASR_RESCUE_BEAM_SIZE,
        "asr_regroup_max_retries": ASR_REGROUP_MAX_RETRIES,
        "asr_transcribe_seed": ASR_TRANSCRIBE_SEED,
        "whisper_timestamped_mode": WHISPER_TIMESTAMPED_MODE,
        "whisper_timestamped_refine_sec": WHISPER_TIMESTAMPED_REFINE_SEC,
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
        "auto_language_history_segments": AUTO_LANGUAGE_HISTORY_SEGMENTS,
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

    def span_len(start_idx: int, end_idx: int) -> float:
        count = end_idx - start_idx
        if count <= 0:
            return 0.0
        speech = duration_prefix[end_idx] - duration_prefix[start_idx]
        gaps = gap_prefix[end_idx - 1] - gap_prefix[start_idx] if count > 1 else 0.0
        return speech + gaps

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

            if total_len <= target_len:
                continue

            min_real_gap = BASE_BREAK_LENGTH * target_len / max(total_len, 1e-9)
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
    seed: int = ASR_TRANSCRIBE_SEED,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "verbose": False,
        "vad": False,
        "compute_word_confidence": True,
        # "detect_disfluencies": True,
        "fp16": True,
        "refine_whisper_precision": WHISPER_TIMESTAMPED_REFINE_SEC,
        # Beam search and temperature fallback force whisper-timestamped's
        # two-pass naive alignment. Greedy single-temperature decoding keeps
        # alignment on the efficient one-pass path.
        "naive_approach": False,
        "beam_size": None,
        "best_of": None,
        "temperature": 0.0,
        "seed": int(seed),
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


def _is_known_phrase_stack_only(
    per_interval_words: List[List[Dict[str, object]]],
    issues: List[str],
) -> bool:
    """True when every abnormal signal is a collapse word stack and every
    stacked interval's text is nothing but the known hallucination phrase
    (possibly repeated). Such stacks carry no recoverable speech — the
    stabilize phrase cleanup removes them wholesale — so rescue decodes are
    wasted GPU that at best converts the squeeze form into a stretched one."""

    if not issues or not all(
        issue.startswith("collapse_word_stack") for issue in issues
    ):
        return False
    phrase = _phrase_key(COMMON_HALLUCINATION_TEXT)
    saw_stack = False
    for words in per_interval_words:
        if not words or detect_collapse_word_stack(words) is None:
            continue
        saw_stack = True
        text = _phrase_key(
            "".join(str(word.get("word") or "") for word in words)
        )
        if not text or text.replace(phrase, ""):
            return False
    return saw_stack


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
    seed: int = ASR_TRANSCRIBE_SEED,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
    decode_options: Optional[Dict[str, object]] = None,
) -> Tuple[
    List[List[Dict[str, object]]],
    List[List[Dict[str, object]]],
    str,
    List[str],
    bool,
]:
    import whisper_timestamped as whisper

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

    transcribe_kwargs = _build_transcribe_kwargs(language=effective_language, seed=seed)
    if decode_options:
        transcribe_kwargs.update(decode_options)
    result = whisper.transcribe(model, combined, **transcribe_kwargs)
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
            out_segments.append(item)
    return out_segments


def _rescue_decode_options() -> Dict[str, object]:
    """Beam decode overrides for rescue attempts (temperature stays 0.0;
    whisper-timestamped switches itself to naive two-pass alignment)."""

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
) -> List[Dict[str, object]]:
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
    pending = candidate
    while remaining:
        if pending is None:
            pending = transcribe(remaining, None)
        p_words, p_segments, p_lang, p_issues, p_auto = pending
        pending = None
        if not p_issues:
            finalize(remaining, p_words, p_segments, p_lang, p_auto)
            break
        k = _first_abnormal_interval_index(p_words)
        if k is None:
            print(
                "Warning: abnormal ASR issues not attributable to one interval; "
                "keeping cleaned window result "
                f"(start={float(remaining[0].get('start', 0.0)):.3f}s, "
                f"issues={_issues_summary(p_issues)})",
                file=sys.stderr,
            )
            finalize(remaining, p_words, p_segments, p_lang, p_auto)
            break
        iso = remaining[k]
        iso_start = float(iso.get("start", 0.0))
        iso_end = float(iso.get("end", 0.0))
        iso_issues_own = detect_abnormal_asr_words([p_words[k]])
        print(
            "Warning: isolating abnormal interval "
            f"(interval={iso_start:.3f}-{iso_end:.3f}, "
            f"clean_front={k}, rest={len(remaining) - k - 1}, "
            f"issues={_issues_summary(iso_issues_own)})",
            file=sys.stderr,
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
                if _coverage_shortfall(front, slice_finalized) is None:
                    print(
                        "Warning: clean-front window re-decode abnormal; keeping the "
                        "original window's clean slice instead "
                        f"(start={float(front[0].get('start', 0.0)):.3f}s, "
                        f"issues={_issues_summary(f_issues)})",
                        file=sys.stderr,
                    )
                    out_segments.extend(slice_finalized)
                    if p_auto:
                        _record_auto_detected_segment_languages(
                            language_history, slice_finalized
                        )
                else:
                    print(
                        "Warning: clean-front window re-decode abnormal and the "
                        "candidate slice is coverage-low; decoding the front "
                        "interval-by-interval "
                        f"(start={float(front[0].get('start', 0.0)):.3f}s, "
                        f"issues={_issues_summary(f_issues)})",
                        file=sys.stderr,
                    )
                    for f_idx, f_interval in enumerate(front):
                        f_successor = (
                            front[f_idx + 1] if f_idx + 1 < len(front) else iso
                        )
                        (
                            fi_words,
                            fi_segments,
                            fi_lang,
                            fi_issues,
                            fi_auto,
                        ) = transcribe([f_interval], f_successor)
                        if fi_issues:
                            print(
                                "Warning: interval-level ASR still abnormal; applying merge-based cleanup fallback "
                                f"(interval={float(f_interval.get('start', 0.0)):.3f}-"
                                f"{float(f_interval.get('end', 0.0)):.3f}, "
                                f"issues={_issues_summary(fi_issues)})",
                                file=sys.stderr,
                            )
                        finalize([f_interval], fi_words, fi_segments, fi_lang, fi_auto)
            else:
                finalize(front, f_words, f_segments, f_lang, f_auto)
        successor = remaining[k + 1] if k + 1 < len(remaining) else None
        i_words, i_segments, i_lang, i_issues, i_auto = transcribe([iso], successor)
        if i_issues:
            print(
                "Warning: interval-level ASR still abnormal; applying merge-based cleanup fallback "
                f"(interval={iso_start:.3f}-{iso_end:.3f}, "
                f"issues={_issues_summary(i_issues)})",
                file=sys.stderr,
            )
        finalize([iso], i_words, i_segments, i_lang, i_auto)
        remaining = remaining[k + 1 :]
    return out_segments


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
    regroup_retries: int = ASR_REGROUP_MAX_RETRIES,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> List[Dict[str, object]]:
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
        return finalized

    if _is_known_phrase_stack_only(per_interval_words, issues):
        # Phrase-only hallucination stack: nothing recoverable underneath, and
        # the stabilize phrase cleanup removes it wholesale. Skip the rescue
        # ladder so the squeeze form is not converted into a stretched one.
        print(
            "Info: known hallucination phrase stack; skipping abnormal rescue ladder "
            f"(start={float(group[0].get('start', 0.0)):.3f}s, "
            f"issues={_issues_summary(issues)})",
            file=sys.stderr,
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
        return finalized

    issue_summary = _issues_summary(issues)
    abnormal_start = _first_abnormal_interval_start(group, per_interval_words)
    max_regroup_retries = max(0, int(regroup_retries))

    if max_regroup_retries > 0:
        for retry in range(1, max_regroup_retries + 1):
            scale = 2.0 / (2.0 + retry)
            temp_group_target_sec = scale * GROUP_TARGET_SEC
            temp_min_group_length = scale * MIN_GROUP_LENGTH
            print(
                "Warning: abnormal ASR result; regroup retry "
                f"(attempt={retry}/{max_regroup_retries}, "
                f"group_target_sec={temp_group_target_sec:.3f}, "
                f"min_group_length={temp_min_group_length:.3f}, "
                f"issues={issue_summary}"
                + (
                    f", abnormal_start={abnormal_start:.3f}s"
                    if abnormal_start is not None
                    else ""
                )
                + ")",
                file=sys.stderr,
            )

            retry_groups = build_alignment_groups(
                group,
                gap_sec=gap_sec,
                group_target_sec=temp_group_target_sec,
                min_group_length=temp_min_group_length,
            )
            if not retry_groups:
                retry_groups = [group]

            retry_candidates = []
            retry_issue_samples: List[str] = []
            has_abnormal = False
            for subgroup_idx, subgroup in enumerate(retry_groups):
                subgroup_successor = (
                    retry_groups[subgroup_idx + 1][0]
                    if subgroup_idx + 1 < len(retry_groups)
                    and retry_groups[subgroup_idx + 1]
                    else None
                )
                (
                    subgroup_words,
                    subgroup_asr_segments,
                    subgroup_lang,
                    subgroup_issues,
                    subgroup_uses_auto_detection,
                ) = _transcribe_group_candidate(
                    model,
                    subgroup,
                    audio,
                    sr,
                    gap_sec,
                    language=language,
                    auto_language_history=language_history,
                    audio_loader=audio_loader,
                    tail_real_limit_sec=batch_tail_limit(
                        subgroup, subgroup_successor
                    ),
                )
                retry_candidates.append(
                    (
                        subgroup,
                        subgroup_words,
                        subgroup_asr_segments,
                        subgroup_lang,
                        subgroup_uses_auto_detection,
                        subgroup_issues,
                    )
                )
                if subgroup_issues:
                    has_abnormal = True
                    subgroup_abnormal_start = _first_abnormal_interval_start(
                        subgroup,
                        subgroup_words,
                    )
                    if len(retry_issue_samples) < 3:
                        issue_label = f"subgroup={subgroup_idx + 1}"
                        if subgroup_abnormal_start is not None:
                            issue_label += f"@{subgroup_abnormal_start:.3f}s"
                        retry_issue_samples.append(
                            f"{issue_label}: {subgroup_issues[0]}"
                        )
                    if abnormal_start is None:
                        abnormal_start = subgroup_abnormal_start

            if not has_abnormal:
                out_segments: List[Dict[str, object]] = []
                for (
                    subgroup,
                    subgroup_words,
                    subgroup_asr_segments,
                    subgroup_lang,
                    subgroup_uses_auto_detection,
                    _subgroup_issues,
                ) in retry_candidates:
                    finalized = _finalize_group_candidate(
                        subgroup,
                        subgroup_words,
                        subgroup_asr_segments,
                        audio,
                        sr,
                        lang=subgroup_lang,
                        audio_loader=audio_loader,
                    )
                    out_segments.extend(finalized)
                    if subgroup_uses_auto_detection:
                        _record_auto_detected_segment_languages(
                            language_history,
                            finalized,
                        )
                return out_segments

            if retry_issue_samples:
                issue_summary = "; ".join(retry_issue_samples)

        print(
            "Warning: max ASR regroup retries reached; falling back to abnormal-interval isolation "
            f"(issues={issue_summary}"
            + (
                f", abnormal_start={abnormal_start:.3f}s"
                if abnormal_start is not None
                else ""
            )
            + ")",
            file=sys.stderr,
        )
    else:
        print(
            "Warning: abnormal ASR result; skipping regroup retries and falling back to abnormal-interval isolation "
            f"(issues={issue_summary}"
            + (
                f", abnormal_start={abnormal_start:.3f}s"
                if abnormal_start is not None
                else ""
            )
            + ")",
            file=sys.stderr,
        )

    # Last resort before fragmenting to intervals: one beam decode of the
    # whole group. Beam explores past the degenerate greedy path while
    # keeping group context; accepted only when clean AND not coverage-low
    # (a near-empty beam result would silently discard content the
    # interval-by-interval fallback can still recover).
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
            print(
                "Info: beam decode accepted before interval-by-interval fallback "
                f"(beam_size={ASR_RESCUE_BEAM_SIZE}, segments={len(beam_finalized)})",
                file=sys.stderr,
            )
            if beam_uses_auto_detection:
                _record_auto_detected_segment_languages(
                    language_history, beam_finalized
                )
            return beam_finalized

    # Final level: abnormal-interval isolation (see _isolate_abnormal_intervals).
    # Clean subgroups from the last regroup retry keep their grouped results;
    # only abnormal subgroups enter the peeling loop.
    if max_regroup_retries > 0 and retry_candidates:
        final_candidates = retry_candidates
    else:
        final_candidates = [
            (
                group,
                per_interval_words,
                per_interval_asr_segments,
                lang,
                uses_auto_detection,
                issues,
            )
        ]

    out_segments: List[Dict[str, object]] = []
    for candidate_idx, (
        sub_intervals,
        sub_words,
        sub_asr_segments,
        sub_lang,
        sub_uses_auto_detection,
        sub_issues,
    ) in enumerate(final_candidates):
        successor = (
            final_candidates[candidate_idx + 1][0][0]
            if candidate_idx + 1 < len(final_candidates)
            and final_candidates[candidate_idx + 1][0]
            else None
        )
        if not sub_issues:
            finalized = _finalize_group_candidate(
                sub_intervals,
                sub_words,
                sub_asr_segments,
                audio,
                sr,
                lang=sub_lang,
                audio_loader=audio_loader,
            )
            out_segments.extend(finalized)
            if sub_uses_auto_detection:
                _record_auto_detected_segment_languages(language_history, finalized)
            continue
        out_segments.extend(
            _isolate_abnormal_intervals(
                model,
                sub_intervals,
                (
                    sub_words,
                    sub_asr_segments,
                    sub_lang,
                    sub_issues,
                    sub_uses_auto_detection,
                ),
                audio,
                sr,
                gap_sec,
                language=language,
                language_history=language_history,
                audio_loader=audio_loader,
                tail_limit_sec=batch_tail_limit(sub_intervals, successor),
            )
        )
    return out_segments


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
    spans.sort(key=lambda x: (x[0], x[1]))
    return spans


def _compute_complement_intervals_from_spans(
    intervals: List[Dict[str, object]],
    segment_spans: List[Tuple[float, float]],
) -> List[Dict[str, float]]:
    interval_spans = _extract_interval_spans(intervals)
    complements: List[Dict[str, float]] = []
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
) -> List[Dict[str, float]]:
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
    print(
        "Warning: low ASR coverage; trying beam rescue "
        f"(start={group_start:.3f}s, covered={covered:.3f}s, "
        f"speech={speech:.3f}s, required={required:.3f}s)",
        file=sys.stderr,
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
            print(
                "Info: beam rescue accepted "
                f"(start={group_start:.3f}s, "
                f"covered={_covered_speech_seconds(group, beam_finalized):.3f}s, "
                f"segments={len(beam_finalized)})",
                file=sys.stderr,
            )
            if beam_uses_auto_detection:
                _record_auto_detected_segment_languages(
                    language_history, beam_finalized
                )
            return beam_finalized

    if len(group) <= 1:
        return segments

    print(
        "Warning: beam rescue insufficient; splitting group "
        f"(start={group_start:.3f}s, intervals={len(group)})",
        file=sys.stderr,
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
        head_segments = align_group(
            model,
            head,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=language_history,
            audio_loader=audio_loader,
            regroup_retries=0,
            tail_real_limit_sec=window_tail_limit(head, rest[0] if rest else None),
        )
        rescued.extend(head_segments)
        if not rest:
            break
        rest_segments = align_group(
            model,
            rest,
            audio,
            sr,
            gap_sec,
            language=language,
            auto_language_history=language_history,
            audio_loader=audio_loader,
            regroup_retries=0,
            tail_real_limit_sec=tail_real_limit_sec,
        )
        if _coverage_shortfall(rest, rest_segments) is None:
            rescued.extend(rest_segments)
            break
        print(
            "Warning: rear window coverage still low; peeling another interval "
            f"(rest_start={float(rest[0].get('start', 0.0)):.3f}s, "
            f"rest_intervals={len(rest)})",
            file=sys.stderr,
        )
        remaining_intervals = rest

    rescued_covered = _covered_speech_seconds(group, rescued)
    original_covered = _covered_speech_seconds(group, segments)
    if rescued_covered <= original_covered:
        language_history[:] = history_snapshot
        print(
            "Info: coverage rescue kept original result "
            f"(start={group_start:.3f}s, rescued={rescued_covered:.3f}s, "
            f"original={original_covered:.3f}s)",
            file=sys.stderr,
        )
        return segments
    print(
        "Info: coverage rescue accepted split result "
        f"(start={group_start:.3f}s, rescued={rescued_covered:.3f}s, "
        f"original={original_covered:.3f}s)",
        file=sys.stderr,
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
    regroup_retries: int = ASR_REGROUP_MAX_RETRIES,
    tail_real_limit_sec: float = GAP_KEEP_REAL_MAX_SEC,
) -> List[Dict[str, object]]:
    aligned = align_group(
        model,
        group,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=auto_language_history,
        audio_loader=audio_loader,
        regroup_retries=regroup_retries,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    aligned = _rescue_low_coverage(
        model,
        group,
        aligned,
        audio,
        sr,
        gap_sec,
        language=language,
        auto_language_history=auto_language_history,
        audio_loader=audio_loader,
        tail_real_limit_sec=tail_real_limit_sec,
    )
    return _sort_segments_by_time(aligned)


def _most_frequent_recent_language(
    auto_language_history: List[str],
    *,
    history_segments: int = AUTO_LANGUAGE_HISTORY_SEGMENTS,
) -> Optional[str]:
    keep = max(0, int(history_segments))
    if keep == 0:
        return None
    recent = [
        str(lang).strip()
        for lang in auto_language_history[-keep:]
        if str(lang).strip() and str(lang).strip() != "None"
    ]
    if not recent:
        return None
    counts: Dict[str, int] = {}
    for lang in recent:
        counts[lang] = counts.get(lang, 0) + 1
    max_count = max(counts.values())
    # On a tie, prefer the most recently auto-detected language.
    return next(lang for lang in reversed(recent) if counts[lang] == max_count)


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
            print(
                "Info: short group reuses recent auto-detected language "
                f"(duration={group_duration:.3f}s, "
                f"language={recent_language}, "
                f"history={auto_language_history})",
                file=sys.stderr,
            )
            return recent_language, False
    return None, True


def _record_auto_detected_segment_languages(
    auto_language_history: List[str],
    segments: List[Dict[str, object]],
) -> None:
    for segment in segments:
        lang = str(segment.get("lang") or "").strip()
        if lang and lang != "None":
            auto_language_history.append(lang)
    keep = max(0, int(AUTO_LANGUAGE_HISTORY_SEGMENTS))
    if keep == 0:
        auto_language_history.clear()
    elif len(auto_language_history) > keep:
        del auto_language_history[:-keep]


def align_segments(
    intervals: List[Dict[str, object]],
    audio: Optional[np.ndarray],
    sr: int,
    *,
    model,
    gap_sec: float,
    language: Optional[str],
    audio_loader: Optional[AudioBlockLoader] = None,
) -> List[Dict[str, object]]:
    if not intervals:
        return []

    out: List[Dict[str, object]] = []
    remaining: List[Dict[str, object]] = list(intervals)
    total_intervals = len(remaining)
    processed_intervals = 0
    group_idx = 0
    prev_tail_segments: List[Dict[str, object]] = []
    auto_language_history: List[str] = []

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
        progress_pentile = min(
            5,
            max(
                1,
                ((processed_after * 5 - 1) // max(total_intervals, 1)) + 1,
            ),
        )
        print(
            "Info: group ASR "
            f"(start={group_start:.3f}s, "
            f"progress_pentile={progress_pentile}/5, "
            f"progress={progress_pct:.1f}%, "
            f"group_iter={group_idx}, "
            f"intervals={processed_after}/{total_intervals})",
            file=sys.stderr,
        )
        if group_size < len(remaining):
            next_gap = float(remaining[group_size].get("start", 0.0)) - float(
                group[-1].get("end", 0.0)
            )
            group_tail_limit = max(0.0, min(next_gap, GAP_KEEP_REAL_MAX_SEC))
        else:
            group_tail_limit = GAP_KEEP_REAL_MAX_SEC
        normal_segments = _align_intervals_group(
            group,
            audio,
            sr,
            model=model,
            gap_sec=gap_sec,
            language=language,
            auto_language_history=auto_language_history,
            audio_loader=audio_loader,
            regroup_retries=ASR_REGROUP_MAX_RETRIES,
            tail_real_limit_sec=group_tail_limit,
        )

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
            print(
                "Info: temporary recall groups "
                f"(count={len(temp_groups)}, threshold={MIN_DROP_TIME_RECALL:.3f}s, "
                f"group_iter={group_idx})",
                file=sys.stderr,
            )
        upcoming_interval_starts = [
            float(interval.get("start", 0.0)) for interval in group
        ]
        if group_size < len(remaining):
            upcoming_interval_starts.append(
                float(remaining[group_size].get("start", 0.0))
            )
        for temp_idx, temp_group in enumerate(temp_groups, start=1):
            temp_start = float(temp_group[0].get("start", 0.0)) if temp_group else 0.0
            temp_total = 0.0
            for interval in temp_group:
                temp_total += max(
                    0.0,
                    float(interval.get("end", 0.0)) - float(interval.get("start", 0.0)),
                )
            print(
                "Info: temporary recall ASR "
                f"(start={temp_start:.3f}s, total_interval_sec={temp_total:.3f}, "
                f"group_iter={group_idx}, temp_group={temp_idx}/{len(temp_groups)})",
                file=sys.stderr,
            )
            aligned_temp_segments = _align_intervals_group(
                temp_group,
                audio,
                sr,
                model=model,
                gap_sec=gap_sec,
                language=language,
                auto_language_history=auto_language_history,
                audio_loader=audio_loader,
                regroup_retries=0,
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
    return out


def _absorb_words_into(
    target: Dict[str, object],
    orphans: List[Dict[str, object]],
    *,
    as_prefix: bool,
) -> Dict[str, object]:
    """Merge orphaned words' text into ``target`` (time span kept as-is).

    Joining honours space_before; the merged word is only as reliable as its
    least-confident source."""

    merged = dict(target)
    orphan_text = words_to_text(orphans)
    target_text = str(merged.get("word") or "")
    if as_prefix:
        space = " " if bool(merged.get("space_before", False)) else ""
        merged["word"] = orphan_text + space + target_text
        merged["space_before"] = bool(orphans[0].get("space_before", False))
    else:
        space = " " if bool(orphans[0].get("space_before", False)) else ""
        merged["word"] = target_text + space + orphan_text
    confidence = min_word_confidence([merged] + orphans)
    if confidence is not None:
        merged["confidence"] = confidence
    return merged


def clamp_segment_overlaps(
    segments: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Pull an overlapping segment end back to the next segment's start.

    Raw word coordinates are globally monotonic, so overlaps only come from
    the speculative energy-based last-word extension (adjacent whisper
    segments in different interval lists cannot bound each other there);
    retracting the earlier segment's end and its extended last word is safe.

    Words the retraction would leave entirely beyond the new end never stay
    as zero-duration leftovers: they merge into the nearest word on the
    owning side — the segment's last surviving word, or, when the whole
    segment collapses, the next segment's first word (as a prefix, with the
    collapsed segment dropped).
    """

    out = [dict(seg) for seg in segments]
    for idx in range(len(out) - 1):
        prev, cur = out[idx], out[idx + 1]
        prev_start = coerce_optional_float(prev.get("start"))
        prev_end = coerce_optional_float(prev.get("end"))
        cur_start = coerce_optional_float(cur.get("start"))
        if prev_start is None or prev_end is None or cur_start is None:
            continue
        if cur_start >= prev_end:
            continue
        new_end = max(cur_start, prev_start)
        prev["end"] = new_end
        surviving: List[Dict[str, object]] = []
        orphans: List[Dict[str, object]] = []
        for word in prev.get("words") or []:
            word_start = coerce_optional_float(word.get("start"))
            word_end = coerce_optional_float(word.get("end"))
            if word_start is not None and word_start >= new_end:
                orphans.append(word)
                continue
            if word_end is not None and word_end > new_end:
                word = dict(word)
                word["end"] = new_end
            surviving.append(word)
        if orphans:
            if surviving:
                surviving[-1] = _absorb_words_into(
                    surviving[-1], orphans, as_prefix=False
                )
            else:
                cur_words = list(cur.get("words") or [])
                if cur_words:
                    cur_words[0] = _absorb_words_into(
                        cur_words[0], orphans, as_prefix=True
                    )
                    cur["words"] = cur_words
                    cur["text"] = words_to_text(cur_words)
                    prev["text"] = ""
                else:
                    # No neighbor word to absorb into: keep the orphans,
                    # clamped to the boundary, instead of losing text.
                    for word in orphans:
                        word = dict(word)
                        word["start"] = new_end
                        word["end"] = new_end
                        surviving.append(word)
        prev["words"] = surviving
    return drop_empty_segments(out)


def extend_zero_length_segments(
    segments: List[Dict[str, object]],
    *,
    min_sec: float = ZERO_LENGTH_SEGMENT_EXTEND_SEC,
) -> List[Dict[str, object]]:
    """Give zero-length segments a minimal duration instead of leaving them
    as dead entries every downstream consumer filters out (to_srt and the
    LLM layer both skip end <= start).

    The extension may squeeze the following segment: its start (and any
    word starts before the new boundary) shifts later. A squeezed segment
    that ends up zero-length itself is handled when the sweep reaches it,
    so chains of coincident zero-length segments resolve sequentially."""

    out = [dict(seg) for seg in segments]
    for idx, seg in enumerate(out):
        start = coerce_optional_float(seg.get("start"))
        end = coerce_optional_float(seg.get("end"))
        if start is None or end is None or end > start:
            continue
        new_end = start + min_sec
        seg["end"] = new_end
        words = [dict(word) for word in seg.get("words") or []]
        if words:
            last = words[-1]
            last_start = coerce_optional_float(last.get("start"))
            last_end = coerce_optional_float(last.get("end"))
            if (
                last_start is not None
                and last_end is not None
                and last_end <= last_start
            ):
                last["end"] = new_end
            seg["words"] = words
        if idx + 1 >= len(out):
            continue
        nxt = out[idx + 1]
        nxt_start = coerce_optional_float(nxt.get("start"))
        if nxt_start is None or nxt_start >= new_end:
            continue
        nxt["start"] = new_end
        nxt_words: List[Dict[str, object]] = []
        for word in nxt.get("words") or []:
            word_start = coerce_optional_float(word.get("start"))
            if word_start is not None and word_start < new_end:
                word = dict(word)
                word["start"] = new_end
                word_end = coerce_optional_float(word.get("end"))
                if word_end is not None and word_end < new_end:
                    word["end"] = new_end
            nxt_words.append(word)
        nxt["words"] = nxt_words
    return out


def drop_empty_segments(segments: List[Dict[str, object]]) -> List[Dict[str, object]]:
    cleaned: List[Dict[str, object]] = []
    for seg in segments:
        words = seg.get("words") or []
        text = seg.get("text")
        has_text = bool(str(text).strip()) if text is not None else False
        if words or has_text:
            cleaned.append(seg)
    return cleaned


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
    device_for_usage: Optional[str] = None
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"Input not found: {input_path}", file=sys.stderr)
            return 1

        device = args.device
        if not device:
            if torch.cuda.is_available():
                device = DEFAULT_DEVICE
            else:
                print(
                    "Warning: CUDA is the default ASR device but is unavailable; falling back to CPU.",
                    file=sys.stderr,
                )
                device = "cpu"
        elif str(device).strip().lower() == "cuda" and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested for ASR alignment but unavailable; falling back to CPU.",
                file=sys.stderr,
            )
            device = "cpu"
        device_for_usage = device
        reset_peak_gpu_memory_stats_for_run(device_for_usage)
        align_meta = asr_align_metadata(
            model=args.model,
            device=device,
            language=args.language,
            gap_sec=args.gap,
        )

        data = json.loads(input_path.read_text(encoding="utf-8"))
        raw_segments = data.get("segments") or []
        if not raw_segments:
            output_path = Path(args.output) if args.output else default_output_path(input_path)
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
            import whisper_timestamped as whisper
        except Exception:
            print(
                "Missing dependency: whisper-timestamped. Install with `pip install whisper-timestamped`.",
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
            output_path = Path(args.output) if args.output else default_output_path(input_path)
            metadata = merge_metadata(data.get("metadata", {}), align_meta)
            payload = {"segments": [], "metadata": metadata}
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {output_path}")
            return 0

        t0 = time.perf_counter()
        model = whisper.load_model(args.model, device=device)
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
        )
        t_align = time.perf_counter() - t0

        segments = drop_empty_segments(aligned_segments)
        output_segments = [round_floats(seg) for seg in segments]
        metadata = merge_metadata(data.get("metadata", {}), align_meta)
        payload = {
            "segments": output_segments,
            "metadata": metadata,
        }

        output_path = Path(args.output) if args.output else default_output_path(input_path)
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
        print_peak_resource_usage(device_for_usage)


if __name__ == "__main__":
    raise SystemExit(main())
