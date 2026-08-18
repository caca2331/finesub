"""Energy-only non-speech detector for vocal-only audio.

This script labels confident non-speech intervals and writes them as SRT segments.
It is intentionally conservative to reduce false "non-speech" labels on speech.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
try:
    import numba as nb
except Exception:
    nb = None

from .audio import (
    get_audio_info_stream,
    load_audio_slice_stream,
    resample_if_needed,
    to_mono,
)
from .spectral import (
    adaptive_weighted_energy,
    compute_band_power_chunks,
    weighted_spectral_energy_db,
)
from ..runtime.device import cuda_usable
from ..runtime.resource_usage import (
    print_peak_resource_usage,
    reset_peak_gpu_memory_stats_for_run,
)

try:
    import torchaudio.functional as AF
except Exception:  # pragma: no cover - optional runtime path
    AF = None


# --------- Tunables (energy non-speech detector) ---------
TARGET_SR = 16000
BLOCK_LENGTH = 300.0
THREAD_NUM = 16
DISABLE_ENERGY_THREADING = True  # ENERGY THREADING is wrong, not consistent and not truly parallelized 
NOISE_STEP2_WORKERS = THREAD_NUM
NOISE_PARALLEL_MIN_WINDOWS = 64

# Tuning guide (for non-speech labels):
# - FP: speech mislabeled as non-speech.
# - FN: non-speech missed (labeled as speech).
# - Recall below means non-speech recall.
#
# Key knobs and expected direction:
# - NON_SPEECH_MARGIN_DB_ENTER / EXIT:
#   higher -> more non-speech positives -> recall up, FN down, FP up.
# - ABS_NON_SPEECH_MAX_DBFS_ENTER / EXIT (less negative values):
#   easier absolute quiet gate -> recall up, FN down, FP up.
# - NOISE_LOCAL_PERCENTILE (CLI: --local-percentile):
#   higher local floor percentile -> higher floor -> recall up, FN down, FP up.
# - NOISE_LOCAL_BLEND (CLI: --local-blend):
#   higher -> tracker follows frame energy more -> usually more aggressive non-speech,
#   so recall up, FN down, FP up.
# - NOISE_TRACK_RISE_ALPHA (CLI: --track-rise-alpha):
#   higher -> floor rises faster on loud regions -> recall up, FN down, FP up.
# - NOISE_TRACK_FOLLOW_ALPHA (CLI: --track-follow-alpha):
#   higher -> floor reacts faster to local changes; often reduces lag, but can
#   increase sensitivity to short fluctuations (tradeoff depends on content).
# - NOISE_LOCAL_WINDOW_SEC / NOISE_LOCAL_HOP_SEC:
#   longer window or larger hop -> smoother/slower floor (typically FP down, FN up).
#   shorter window or smaller hop -> more reactive floor (typically FP up, FN down).
# - MIN_NON_SPEECH_MS:
#   higher -> suppress short non-speech runs (FP down, FN up).
# - MERGE_GAP_MS:
#   higher -> bridge short speech gaps between non-speech runs (FP up, FN down).

# Frame analysis window and stride in milliseconds.
# These are converted to samples using each file's native sample rate.
FRAME_MS = 25.0
HOP_MS = 10.0

# Light normalization (keep this mild; do not over-shape vocal dynamics).
REMOVE_DC = True
HPF_ENABLE = True
HPF_HZ = 70.0
NORM_TARGET_DBFS = -24.0
# RMS window size for local loudness normalization.
NORM_WINDOW_SEC = 120.0
# Gain-update hop; smaller means smoother adaptation but more compute.
NORM_GAIN_HOP_SEC = 1.0
NORM_MAX_GAIN_DB = 6.0
NORM_MIN_GAIN_DB = -4.0
NORM_PEAK_LIMIT = 0.98

# Adaptive spectral-energy weighting:
# frame energy is computed from weighted spectral bands instead of full-band RMS.
# This reduces non-vocal frequency influence while keeping adaptation data-driven.
SPECTRAL_NUM_BANDS = 24
SPECTRAL_CHUNK_FRAMES = 4096
VOCAL_PRIOR_MIN_HZ = 120.0
VOCAL_PRIOR_MAX_HZ = 4000.0
VOCAL_PRIOR_TRANSITION_HZ = 400.0
VOCAL_PRIOR_FLOOR = 0.15
# Smooth static vocal-prior shape (option 2): sigmoid edges in log-frequency.
VOCAL_PRIOR_LOW_HZ = 120.0
VOCAL_PRIOR_HIGH_HZ = 4200.0
VOCAL_PRIOR_LOG_K_LOW = 0.18
VOCAL_PRIOR_LOG_K_HIGH = 0.20
# Adaptive prior blend (option 1): combine static prior with per-recording band occupancy.
VOCAL_PRIOR_ADAPT_LAMBDA = 0.45
VOCAL_PRIOR_OCC_ALPHA = 0.02
VOCAL_PRIOR_OCC_SNR_DB = 2.0
SPECTRAL_WEIGHT_MIN = 0.05
SPECTRAL_SNR_KEEP_DB = 3.0
SPECTRAL_SNR_SOFT_DB = 2.0
SPECTRAL_NOISE_GATE_DB = 6.0
SPECTRAL_NOISE_ALPHA_QUIET = 0.08
SPECTRAL_NOISE_ALPHA_LOUD = 0.005

# Background noise estimation (adaptive floor tracker).
NOISE_INIT_PERCENTILE = 5.0
NOISE_TRACK_GATE_DB = 6.0
NOISE_TRACK_FOLLOW_ALPHA = 0.08
NOISE_TRACK_RISE_ALPHA = 0.005
NOISE_LOCAL_WINDOW_SEC = 120.0
NOISE_LOCAL_HOP_SEC = 1.0
NOISE_LOCAL_BLEND = 0.997

# Non-speech decision rule used internally:
# label frame as non-speech when frame_db <= noise_floor_db + margin.
NON_SPEECH_MARGIN_DB_ENTER = 6

# Extra absolute gate in dBFS so low-energy speech is less likely mislabeled.
ABS_NON_SPEECH_MAX_DBFS_ENTER = -30.0
ABS_NON_SPEECH_MAX_DBFS_EXIT = -28.0

# Interval post-processing:
# drop short intervals, merge very small gaps, then shrink boundaries.
# Keep merge gap small to avoid bridging across short speech bursts.
MIN_NON_SPEECH_MS = 400.0
MERGE_GAP_MS = 100.0
# A speech-like frame's debit is banked and only charged once the speech-like
# run reaches this many consecutive frames; shorter bursts are forgiven.
# Jitter bursts inside no-word regions run 2-5 frames at the median, real
# words 17-24 (FINDINGS Z), so 4 absorbs blips without touching word-driven
# closes. 0/1 restores the pre-2026-08 behavior (every frame charges).
SPEECH_EXIT_MIN_RUN_FRAMES = 4
# Minimum kept interval length after negative padding shrink.
MIN_KEEP_AFTER_SHRINK_MS = 10.0
# If True, use weighted frame counting for MIN_NON_SPEECH_MS and MERGE_GAP_MS.
# This allows confident low-energy non-speech runs to survive at shorter durations,
# and discounts weak/low-energy speech-like gaps when deciding merges.
WEIGHTED_INTERVAL = True

# Negative padding (interval shrink):
# stronger right shrink to avoid grabbing consonants before following speech.
NEGATIVE_PAD_LEFT_MS = 40.0
NEGATIVE_PAD_RIGHT_MS = 140.0

# Absolute floor for a speech interval's peak weighted energy. The relative
# criterion trusts the noise floor, and where the floor has collapsed into
# digital silence a -70 dB rumble sits "24 dB over background" and opens an
# interval. Nothing that quiet is ever speech. Calibrated on 4 clips / 2830
# intervals: word-bearing intervals peak at >= 0 dB (per-clip p0.5 between +1
# and +8; the lone sub-zero case is timestamp bleed from a word spanning
# neighbouring intervals), and below -45 every guard is clean -- zero valid
# words, zero all-words (drift/repeat segments included), zero human-labeled
# real speech, and silero peaks >= 0.3 in 1 of 45 intervals. Word evidence
# starts appearing from -45 upward, so -45 keeps >10 dB of margin
# (tools/vad_tuning/FINDINGS.md appendix X).
MIN_SPEECH_PEAK_DB = -45.0

# SRT text for each non-speech interval.
LABEL_TEXT = "\"\""

# Numerical stability.
DB_EPS = 1e-10
SEGMENT_ENERGY_FIELD = "vad_weighted_energy_db"


@dataclass(frozen=True)
class VadEnergyTrack:
    """Frame-level energy track produced by the file VAD pass.

    Frame times are derived on demand from the uniform grid
    ``start_i = i * hop_sec``, ``end_i = start_i + frame_sec`` in float64;
    storing them as float32 tensors would quantize offsets by ~1 ms beyond
    ~2.3 h of audio and shift boundary-frame selection for short segments.
    """

    energy_db: torch.Tensor
    hop_sec: float
    frame_sec: float
    energy_mode: str
    # Full-band frame dBFS, carried so opt-in post passes (silero assist) can
    # re-run the scorer without another framing pass. Optional: older callers
    # construct the track without it.
    frame_dbfs: Optional[torch.Tensor] = None


def _frame_grid_seconds(sample_rate: int = TARGET_SR) -> Tuple[float, float]:
    """(hop_sec, frame_sec) matching the framing used by the VAD passes."""

    frame_len = max(1, int(round((FRAME_MS / 1000.0) * sample_rate)))
    hop_len = max(1, int(round((HOP_MS / 1000.0) * sample_rate)))
    return hop_len / float(sample_rate), frame_len / float(sample_rate)


def segment_energy_metadata() -> Dict[str, object]:
    """Describe the aligned-segment energy field derived from this VAD."""

    return {
        "field": SEGMENT_ENERGY_FIELD,
        "source": "adaptive_weighted_spectral_energy",
        "aggregation": "overlap_weighted_power_mean_db",
        "frame_ms": float(FRAME_MS),
        "hop_ms": float(HOP_MS),
        "audio": "normalized_vocal",
    }


def aggregate_segment_weighted_energy_db(
    track: VadEnergyTrack,
    start: float,
    end: float,
) -> Optional[float]:
    """Aggregate overlapping weighted-energy frames over ``[start, end]``.

    Frame dB values are converted back to linear power, averaged using each
    frame's overlap duration as its weight, then converted to dB again.
    """

    if track.energy_mode != "weighted":
        return None
    try:
        start_s = float(start)
        end_s = float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
        return None

    n = int(track.energy_db.numel())
    hop_sec = float(track.hop_sec)
    frame_sec = float(track.frame_sec)
    if n <= 0 or hop_sec <= 0.0 or frame_sec <= 0.0:
        return None

    # Frame i covers [i * hop_sec, i * hop_sec + frame_sec); overlap requires
    # i * hop_sec < end_s and i * hop_sec + frame_sec > start_s. Widen the
    # index range by one frame per side and let the overlap mask pick the
    # exact set, so float rounding at the bounds can never drop a frame.
    first = max(0, int(math.floor((start_s - frame_sec) / hop_sec)))
    last = min(n, int(math.ceil(end_s / hop_sec)) + 1)
    if last <= first:
        return None

    energy = track.energy_db[first:last].detach().to(
        device="cpu", dtype=torch.float64
    )
    frame_starts = torch.arange(first, last, dtype=torch.float64) * hop_sec
    overlap = torch.clamp(
        (frame_starts + frame_sec).clamp(max=end_s)
        - frame_starts.clamp(min=start_s),
        min=0.0,
    )
    valid = (overlap > 0.0) & torch.isfinite(energy)
    if not bool(torch.any(valid)):
        return None

    weights = overlap[valid]
    power = torch.pow(10.0, energy[valid] / 10.0)
    mean_power = float(torch.sum(weights * power) / torch.sum(weights))
    if not math.isfinite(mean_power) or mean_power <= 0.0:
        return None
    return 10.0 * math.log10(max(mean_power, DB_EPS))


def _add_timing(timing: Optional[Dict[str, float]], key: str, dt: float) -> None:
    if timing is None:
        return
    timing[key] = float(timing.get(key, 0.0) + float(dt))


def _energy_worker_threads() -> int:
    if DISABLE_ENERGY_THREADING:
        return 1
    return max(1, int(THREAD_NUM))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect non-speech intervals from vocal-only audio using energy."
    )
    parser.add_argument("input", help="Path to input audio file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output SRT (default: <input>-vad_energy.srt).",
    )
    parser.add_argument(
        "-e",
        "--energy-mode",
        choices=("none", "weighted"),
        default="weighted",
        help=(
            "Energy backend for non-speech detection "
            "(default: weighted). Use 'none' to skip vocal-frequency weighting."
        ),
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=None,
        help="Override non-speech SNR margin used by scoring.",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-vad_energy.srt")


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000.0))
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_non_speech_srt(intervals: Sequence[Tuple[float, float]]) -> str:
    lines: List[str] = []
    idx = 1
    for start_s, end_s in intervals:
        if end_s <= start_s:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
        lines.append(LABEL_TEXT)
        lines.append("")
        idx += 1
    if not lines:
        return ""
    return "\n".join(lines).rstrip() + "\n"


def invert_intervals(
    intervals: Sequence[Tuple[float, float]],
    duration_sec: float,
) -> List[Tuple[float, float]]:
    if duration_sec <= 0:
        return []

    out: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in sorted(intervals, key=lambda x: (x[0], x[1])):
        s = max(0.0, min(float(s), duration_sec))
        e = max(0.0, min(float(e), duration_sec))
        if e <= s:
            continue
        if s > cursor:
            out.append((cursor, s))
        if e > cursor:
            cursor = e
    if cursor < duration_sec:
        out.append((cursor, duration_sec))
    return out


def interval_length_bucket_summary(
    intervals: Sequence[Tuple[float, float]],
) -> List[Tuple[str, int, float]]:
    counts = [0, 0, 0, 0, 0, 0, 0]
    for start, end in intervals:
        d = max(0.0, float(end) - float(start))
        if d < 0.2:
            counts[0] += 1
        elif d < 0.3:
            counts[1] += 1
        elif d < 0.4:
            counts[2] += 1
        elif d < 1.0:
            counts[3] += 1
        elif d < 2.0:
            counts[4] += 1
        elif d < 5.0:
            counts[5] += 1
        else:
            counts[6] += 1

    total = sum(counts)
    labels = ["0-0.2", "0.2-0.3", "0.3-0.4", "0.4-1", "1-2", "2-5", "5+"]
    out: List[Tuple[str, int, float]] = []
    for label, count in zip(labels, counts):
        pct = (100.0 * count / total) if total > 0 else 0.0
        out.append((label, count, pct))
    return out


def _db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _linear_to_db(x: torch.Tensor) -> torch.Tensor:
    return 20.0 * torch.log10(torch.clamp(x, min=DB_EPS))


def _pad_for_framing(
    waveform: torch.Tensor,
    frame_len: int,
    hop_len: int,
) -> Tuple[torch.Tensor, int]:
    x = waveform
    if x.numel() < frame_len:
        x = F.pad(x, (0, frame_len - x.numel()))

    n_frames = 1 + int((x.numel() - frame_len) // hop_len)
    tail = x.numel() - (n_frames - 1) * hop_len - frame_len
    if tail < 0:
        tail = 0
    if tail > 0:
        x = F.pad(x, (0, hop_len - tail))
        n_frames = 1 + int((x.numel() - frame_len) // hop_len)
    return x, n_frames


# --- deterministic normalization ---------------------------------------------
# All reductions below (DC mean, RMS window sums) are defined over fixed
# absolute grids with float64 accumulation, so the streamed block path can
# reproduce the whole-file result bit-exactly by running the same ops on the
# same slices. Do not vectorize these into order-unspecified reductions.


def _dc_mean32(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """DC mean as a float32 scalar tensor: float64 sums over BLOCK_LENGTH cells."""

    n = int(waveform.numel())
    cell = max(1, int(round(float(BLOCK_LENGTH) * sample_rate)))
    total = 0.0
    for start in range(0, n, cell):
        total += float(torch.sum(waveform[start : start + cell].double()))
    return torch.tensor(total / float(n), dtype=torch.float32)


def _norm_geometry(sample_rate: int) -> Tuple[int, int, int]:
    """(window_samples, hop_samples, half_before) of the RMS gain grid."""

    window_samples = max(1, int(round(NORM_WINDOW_SEC * sample_rate)))
    hop_samples = max(1, int(round(NORM_GAIN_HOP_SEC * sample_rate)))
    return window_samples, hop_samples, window_samples // 2


def _power_cells(
    x: torch.Tensor,
    x_offset: int,
    n_total: int,
    hop_samples: int,
    first_cell: int,
    cell_count: int,
) -> List[float]:
    """float64 sums of x^2 over absolute 1-anchor-hop cells [k*hop, (k+1)*hop)∩[0,n).

    x holds absolute samples [x_offset, x_offset + len(x)); every requested
    cell must lie inside that range."""

    cells: List[float] = []
    for k in range(first_cell, first_cell + cell_count):
        lo = k * hop_samples
        hi = min(lo + hop_samples, n_total)
        if hi <= lo:
            cells.append(0.0)
            continue
        seg = x[lo - x_offset : hi - x_offset].double()
        cells.append(float(torch.sum(seg * seg)))
    return cells


def _gain_from_power(power_sum: float, length: int) -> float:
    rms = math.sqrt(power_sum / float(max(1, length)) + DB_EPS)
    target_rms = _db_to_linear(NORM_TARGET_DBFS)
    gain = target_rms / max(rms, DB_EPS)
    return min(max(gain, _db_to_linear(NORM_MIN_GAIN_DB)), _db_to_linear(NORM_MAX_GAIN_DB))


def _anchor_gains(
    x: torch.Tensor,
    x_offset: int,
    n_total: int,
    sample_rate: int,
    anchors: Sequence[int],
) -> torch.Tensor:
    """float32 gains at absolute anchor positions (grid anchors via cells,
    the forced final n-1 anchor via a direct float64 sum)."""

    window_samples, hop_samples, half_before = _norm_geometry(sample_rate)
    half_after = window_samples - half_before

    grid = [a for a in anchors if a % hop_samples == 0]
    cell_values: Dict[int, float] = {}
    if grid:
        first_cell = max(0, (min(grid) - half_before)) // hop_samples
        last_cell_excl = (min(max(grid) + half_after, n_total) + hop_samples - 1) // hop_samples
        values = _power_cells(
            x, x_offset, n_total, hop_samples, first_cell, last_cell_excl - first_cell
        )
        cell_values = {first_cell + i: v for i, v in enumerate(values)}

    gains: List[float] = []
    for a in anchors:
        left = max(a - half_before, 0)
        right = min(a + half_after, n_total)
        if a % hop_samples == 0:
            lo_cell = left // hop_samples
            hi_cell = (right + hop_samples - 1) // hop_samples
            power = 0.0
            for k in range(lo_cell, hi_cell):
                power += cell_values[k]
        else:  # the forced final anchor at n-1
            seg = x[left - x_offset : right - x_offset].double()
            power = float(torch.sum(seg * seg))
        gains.append(_gain_from_power(power, right - left))
    return torch.tensor(gains, dtype=torch.float32)


def _apply_gain_segments(
    x: torch.Tensor,
    x_offset: int,
    anchors: Sequence[int],
    gains: torch.Tensor,
    hop_samples: int,
    *,
    tail: bool,
) -> None:
    """In-place linear gain interpolation between anchors (memory-path loop)."""

    base_len = hop_samples
    base_t = None
    if base_len > 1:
        base_t = torch.arange(base_len, dtype=x.dtype, device=x.device) / float(base_len)

    for i in range(len(anchors) - 1):
        start = anchors[i] - x_offset
        end = anchors[i + 1] - x_offset
        seg_len = max(1, end - start)
        if seg_len == 1:
            x[start:end].mul_(gains[i])
            continue
        if base_t is not None and seg_len == base_len:
            t = base_t
        else:
            t = torch.arange(seg_len, dtype=x.dtype, device=x.device) / float(seg_len)
        seg_gains = gains[i] + (gains[i + 1] - gains[i]) * t
        x[start:end].mul_(seg_gains)

    if tail:
        x[anchors[-1] - x_offset :].mul_(gains[-1])


def _full_anchor_list(n: int, hop_samples: int) -> List[int]:
    anchors = list(range(0, n, hop_samples))
    if not anchors or anchors[-1] != n - 1:
        anchors.append(n - 1)
    return anchors


def _apply_local_rms_normalization(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Local RMS normalization with a 120s window and 1s gain anchors.

    Window sums accumulate in float64 over the absolute anchor grid (see the
    section comment): deterministic, and materially more accurate than the old
    whole-file float32 cumsum on hours-long audio."""

    x = waveform
    n = int(x.numel())
    if n == 0:
        return x

    _, hop_samples, _ = _norm_geometry(sample_rate)
    anchors = _full_anchor_list(n, hop_samples)
    gains = _anchor_gains(x, 0, n, sample_rate, anchors)

    if len(anchors) == 1:
        return x * gains[0]

    out = x.clone()
    _apply_gain_segments(out, 0, anchors, gains, hop_samples, tail=True)
    return out


def light_normalize(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    x = waveform.float()

    if REMOVE_DC and x.numel() > 0:
        x = x - _dc_mean32(x, sample_rate)

    if HPF_ENABLE and AF is not None and sample_rate > int(2 * HPF_HZ):
        x = AF.highpass_biquad(x.unsqueeze(0), sample_rate, HPF_HZ).squeeze(0)

    if x.numel() == 0:
        return x

    x = _apply_local_rms_normalization(x, sample_rate)

    # Per-sample, not a global rescale. The RMS step above is what makes the
    # absolute dB gates comparable across files; scaling the whole track by
    # 0.98/peak used to undo part of that calibration by an amount the single
    # loudest sample decided (measured -2.2 dB on kaguya60 off 0.0003% of
    # samples, -4.1 dB on mia off 0.00004%). Clipping the outliers instead
    # leaves every other sample where the normalizer put it. See docs.
    return torch.clamp(x, -NORM_PEAK_LIMIT, NORM_PEAK_LIMIT)


def _load_asr_audio_streamed(
    path: str | Path,
) -> torch.Tensor:
    src_path = str(path)
    src_sr, total_frames = get_audio_info_stream(src_path)
    if src_sr <= 0:
        raise RuntimeError(f"Invalid sample rate for audio: {src_path}")
    if total_frames <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if BLOCK_LENGTH <= 0:
        raise ValueError("BLOCK_LENGTH must be > 0 for streamed loading.")

    block_frames = max(1, int(round(float(BLOCK_LENGTH) * src_sr)))
    chunks: List[torch.Tensor] = []
    for read_start in range(0, total_frames, block_frames):
        read_frames = min(block_frames, total_frames - read_start)
        if read_frames <= 0:
            break
        chunk, read_sr = load_audio_slice_stream(src_path, read_start, read_frames)
        if int(read_sr) <= 0:
            raise RuntimeError(f"Invalid sample rate while loading: {src_path}")
        mono = to_mono(chunk)
        resampled, _ = resample_if_needed(mono.unsqueeze(0), int(read_sr), TARGET_SR)
        chunk_1d = resampled.squeeze(0)
        if chunk_1d.numel() > 0:
            chunks.append(chunk_1d)

    if not chunks:
        return torch.zeros(0, dtype=torch.float32)
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def _frame_dbfs(
    waveform: torch.Tensor,
    frame_len: int,
    hop_len: int,
    sample_rate: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    waveform, _ = _pad_for_framing(waveform, frame_len, hop_len)
    frames = waveform.unfold(0, frame_len, hop_len)
    power = torch.mean(frames * frames, dim=1)
    rms = torch.sqrt(power + DB_EPS)
    db = _linear_to_db(rms)

    starts = torch.arange(db.shape[0], dtype=torch.float32) * (hop_len / float(sample_rate))
    ends = starts + (frame_len / float(sample_rate))
    return db, starts, ends


def _frame_weighted_spectral_db(
    waveform: torch.Tensor,
    frame_len: int,
    hop_len: int,
    sample_rate: int,
) -> torch.Tensor:
    return weighted_spectral_energy_db(
        waveform,
        sample_rate=sample_rate,
        frame_len=frame_len,
        hop_len=hop_len,
        num_bands=SPECTRAL_NUM_BANDS,
        chunk_frames=SPECTRAL_CHUNK_FRAMES,
        vocal_prior_min_hz=VOCAL_PRIOR_MIN_HZ,
        vocal_prior_max_hz=VOCAL_PRIOR_MAX_HZ,
        vocal_prior_floor=VOCAL_PRIOR_FLOOR,
        vocal_prior_low_hz=VOCAL_PRIOR_LOW_HZ,
        vocal_prior_high_hz=VOCAL_PRIOR_HIGH_HZ,
        vocal_prior_log_k_low=VOCAL_PRIOR_LOG_K_LOW,
        vocal_prior_log_k_high=VOCAL_PRIOR_LOG_K_HIGH,
        vocal_prior_adapt_lambda=VOCAL_PRIOR_ADAPT_LAMBDA,
        vocal_prior_occ_alpha=VOCAL_PRIOR_OCC_ALPHA,
        vocal_prior_occ_snr_db=VOCAL_PRIOR_OCC_SNR_DB,
        spectral_weight_min=SPECTRAL_WEIGHT_MIN,
        spectral_snr_keep_db=SPECTRAL_SNR_KEEP_DB,
        spectral_snr_soft_db=SPECTRAL_SNR_SOFT_DB,
        noise_init_percentile=NOISE_INIT_PERCENTILE,
        spectral_noise_gate_db=SPECTRAL_NOISE_GATE_DB,
        spectral_noise_alpha_quiet=SPECTRAL_NOISE_ALPHA_QUIET,
        spectral_noise_alpha_loud=SPECTRAL_NOISE_ALPHA_LOUD,
        db_eps=DB_EPS,
        workers=_energy_worker_threads(),
    )


def _compute_frame_tracks_for_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    energy_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_len = max(1, int(round((FRAME_MS / 1000.0) * sample_rate)))
    hop_len = max(1, int(round((HOP_MS / 1000.0) * sample_rate)))
    frame_dbfs, frame_starts, frame_ends = _frame_dbfs(
        waveform, frame_len, hop_len, sample_rate
    )
    if energy_mode == "weighted":
        energy_db = _frame_weighted_spectral_db(waveform, frame_len, hop_len, sample_rate)
        n = min(frame_dbfs.numel(), energy_db.numel())
    else:
        # "none": do not apply vocal-frequency weighting.
        energy_db = frame_dbfs
        n = frame_dbfs.numel()
    if n <= 0:
        z = torch.zeros(0, dtype=torch.float32)
        return z, z, z, z
    return (
        frame_dbfs[:n],
        energy_db[:n],
        frame_starts[:n],
        frame_ends[:n],
    )


# --- exact streaming (bounded RAM, bit-identical to the in-memory path) --------
# Strategy: band power, frame dBFS and the peak clamp are per-sample or
# per-frame local, so they stream per block; every stateful/global step (DC
# mean, RMS window sums, adaptive spectral tracker, noise floor, scoring) runs
# once globally on deterministic per-block reductions. See docs for the
# derivation.

STREAM_CORE_SEC = 600.0
STREAM_CONTEXT_SEC = 90.0


class WaveformObserver(Protocol):
    """Something that wants the normalized waveform this module already built.

    Handed each core block, in order, exactly once, with nothing overlapping and
    nothing skipped -- so the concatenation of every ``feed`` is the normalized
    waveform. ``reset`` is called once, before the first block; implement it so
    a reused observer starts clean rather than assuming it fires exactly once.

    Blocks are whole seconds of 16 kHz audio, so an observer on a coarser frame
    grid divides evenly as long as its hop divides STREAM_CORE_SEC * TARGET_SR.
    """

    def reset(self) -> None: ...

    def feed(self, block: torch.Tensor) -> None: ...


class _ResampledStream:
    """Sequential 16k mono float32 view of an audio file.

    Reads and resamples the source on the same absolute BLOCK_LENGTH source
    grid as _load_asr_audio_streamed, so the concatenation of everything the
    stream yields is bit-identical to the in-memory load."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        src_sr, src_frames = get_audio_info_stream(self._path)
        if src_sr <= 0:
            raise RuntimeError(f"Invalid sample rate for audio: {self._path}")
        self._src_sr = int(src_sr)
        self._src_frames = max(0, int(src_frames))
        self._block_src = max(1, int(round(float(BLOCK_LENGTH) * self._src_sr)))
        self._next_src = 0
        self._buf = torch.zeros(0, dtype=torch.float32)
        self._buf_start = 0

    @property
    def exhausted(self) -> bool:
        return self._next_src >= self._src_frames

    @property
    def _buf_end(self) -> int:
        return self._buf_start + int(self._buf.numel())

    def _pump(self) -> None:
        read_frames = min(self._block_src, self._src_frames - self._next_src)
        if read_frames <= 0:
            return
        chunk, read_sr = load_audio_slice_stream(self._path, self._next_src, read_frames)
        if int(read_sr) <= 0:
            raise RuntimeError(f"Invalid sample rate while loading: {self._path}")
        self._next_src += read_frames
        mono = to_mono(chunk)
        resampled, _ = resample_if_needed(mono.unsqueeze(0), int(read_sr), TARGET_SR)
        piece = resampled.squeeze(0)
        if piece.numel() > 0:
            self._buf = piece if self._buf.numel() == 0 else torch.cat([self._buf, piece])

    def read(self, start: int, end: int) -> torch.Tensor:
        """Absolute-sample view [start, min(end, total)); forward-only."""

        if start < self._buf_start:
            raise RuntimeError("stream cannot seek backwards")
        while self._buf_end < end and not self.exhausted:
            self._pump()
        end = min(end, self._buf_end)
        return self._buf[start - self._buf_start : max(start, end) - self._buf_start]

    def drop_before(self, start: int) -> None:
        if start > self._buf_start:
            cut = min(start - self._buf_start, int(self._buf.numel()))
            self._buf = self._buf[cut:].clone()
            self._buf_start += cut


def _stream_mean_and_length(path: str | Path) -> Tuple[int, Optional[torch.Tensor]]:
    """One cheap pass: resampled total length + the deterministic DC mean
    (same BLOCK_LENGTH cells as _dc_mean32, so the values match bit-exactly)."""

    stream = _ResampledStream(path)
    cell = max(1, int(round(float(BLOCK_LENGTH) * TARGET_SR)))
    total = 0.0
    n = 0
    while True:
        x = stream.read(n, n + cell)
        if x.numel() == 0:
            break
        total += float(torch.sum(x.double()))
        n += int(x.numel())
        stream.drop_before(n)
    mean = torch.tensor(total / float(n), dtype=torch.float32) if n > 0 else None
    return n, mean


def _streamed_frame_tracks(
    path: str | Path,
    *,
    energy_mode: str,
    core_sec: float = STREAM_CORE_SEC,
    context_sec: float = STREAM_CONTEXT_SEC,
    timing: Optional[Dict[str, float]] = None,
    observer: Optional[WaveformObserver] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Frame tracks (dbfs, energy, starts, ends, duration) computed streamed,
    bit-identical to light_normalize + _compute_frame_tracks_for_waveform on
    the fully loaded audio.

    ``observer`` sees the same normalized blocks (see WaveformObserver); passing
    none leaves every value here untouched."""

    sr = TARGET_SR
    frame_len = max(1, int(round((FRAME_MS / 1000.0) * sr)))
    hop_len = max(1, int(round((HOP_MS / 1000.0) * sr)))
    window_samples, anchor_hop, _ = _norm_geometry(sr)

    core = int(round(core_sec)) * sr
    ctx = int(round(context_sec)) * sr
    if core <= 0 or core % anchor_hop != 0 or ctx % anchor_hop != 0:
        raise ValueError("core_sec/context_sec must be positive whole seconds")
    min_ctx = window_samples // 2 + anchor_hop + frame_len
    if ctx < min_ctx:
        raise ValueError(
            f"context_sec must cover the RMS half-window + 1s + frame ({min_ctx} samples)"
        )

    t_load0 = time.perf_counter()
    n, mean32 = _stream_mean_and_length(path)
    _add_timing(timing, "loading_sec", time.perf_counter() - t_load0)
    duration_sec = n / float(sr)
    if n == 0:
        z = torch.zeros(0, dtype=torch.float32)
        return z, z, z, z, 0.0

    t_energy0 = time.perf_counter()
    if n <= core:
        # Single block == the in-memory code path on the same samples.
        x = _ResampledStream(path).read(0, n)
        x = light_normalize(x, sr)
        if observer is not None:
            observer.reset()
            observer.feed(x)
        frame_dbfs, energy_db, starts, ends = _compute_frame_tracks_for_waveform(
            x, sr, energy_mode=energy_mode
        )
        _add_timing(timing, "energy_sec", time.perf_counter() - t_energy0)
        return frame_dbfs, energy_db, starts, ends, duration_sec

    apply_hpf = HPF_ENABLE and AF is not None and sr > int(2 * HPF_HZ)
    last_grid = ((n - 1) // anchor_hop) * anchor_hop

    # One pass over the file: every step below is either per-block local or a
    # deterministic reduction the global steps consume afterwards.
    stream = _ResampledStream(path)
    dbfs_parts: List[torch.Tensor] = []
    band_parts: List[torch.Tensor] = []
    prior: Optional[torch.Tensor] = None
    c0 = 0
    if observer is not None:
        observer.reset()
    while c0 < n:
        c1 = min(c0 + core, n)
        r0 = max(0, c0 - ctx)
        r1 = min(n, c1 + ctx)
        x = stream.read(r0, r1).clone()
        if REMOVE_DC and mean32 is not None:
            x = x - mean32
        if apply_hpf:
            x = AF.highpass_biquad(x.unsqueeze(0), sr, HPF_HZ).squeeze(0)

        # RMS gain segments must cover [c0, c1 + frame_len) ∩ [0, n): kept
        # frames start before c1 but their windows spill up to frame_len.
        needed_end = min(c1 + frame_len, n)
        if needed_end <= last_grid:
            stop = ((needed_end + anchor_hop - 1) // anchor_hop) * anchor_hop
            anchors = list(range(c0, stop + 1, anchor_hop))
            tail = False
        else:
            anchors = list(range(c0, last_grid + 1, anchor_hop))
            if anchors[-1] != n - 1:
                anchors.append(n - 1)
            tail = True
        gains = _anchor_gains(x, r0, n, sr, anchors)
        _apply_gain_segments(x, r0, anchors, gains, anchor_hop, tail=tail)

        # light_normalize's last step. Per-sample, so it needs no global
        # quantity and the block stays independent.
        x = torch.clamp(x, -NORM_PEAK_LIMIT, NORM_PEAK_LIMIT)

        if observer is not None:
            observer.feed(x[c0 - r0 : c1 - r0])

        tail_x = x[c0 - r0 :]
        dbfs, _starts, _ends = _frame_dbfs(tail_x, frame_len, hop_len, sr)
        kept = (c1 - c0) // hop_len if c1 < n else int(dbfs.numel())
        dbfs_parts.append(dbfs[:kept])
        if energy_mode == "weighted" and kept > 0:
            chunks, _n_bf, block_prior = compute_band_power_chunks(
                tail_x,
                sample_rate=sr,
                frame_len=frame_len,
                hop_len=hop_len,
                num_bands=SPECTRAL_NUM_BANDS,
                chunk_frames=SPECTRAL_CHUNK_FRAMES,
                vocal_prior_min_hz=VOCAL_PRIOR_MIN_HZ,
                vocal_prior_max_hz=VOCAL_PRIOR_MAX_HZ,
                vocal_prior_floor=VOCAL_PRIOR_FLOOR,
                vocal_prior_low_hz=VOCAL_PRIOR_LOW_HZ,
                vocal_prior_high_hz=VOCAL_PRIOR_HIGH_HZ,
                vocal_prior_log_k_low=VOCAL_PRIOR_LOG_K_LOW,
                vocal_prior_log_k_high=VOCAL_PRIOR_LOG_K_HIGH,
                db_eps=DB_EPS,
                workers=_energy_worker_threads(),
            )
            if block_prior is not None:
                prior = block_prior
            band_parts.append(
                torch.cat([c for c in chunks if c.shape[0] > 0], dim=0)[:kept]
            )
        stream.drop_before(max(0, c1 - ctx))
        c0 = c1

    frame_dbfs = (
        torch.cat(dbfs_parts, dim=0) if dbfs_parts else torch.zeros(0, dtype=torch.float32)
    )
    n_frames = int(frame_dbfs.numel())
    if energy_mode == "weighted" and band_parts and prior is not None:
        band_power = torch.cat(band_parts, dim=0)
        energy_db = adaptive_weighted_energy(
            band_power,
            prior=prior,
            init_count=min(int(SPECTRAL_CHUNK_FRAMES), int(band_power.shape[0])),
            vocal_prior_adapt_lambda=VOCAL_PRIOR_ADAPT_LAMBDA,
            vocal_prior_occ_alpha=VOCAL_PRIOR_OCC_ALPHA,
            vocal_prior_occ_snr_db=VOCAL_PRIOR_OCC_SNR_DB,
            vocal_prior_floor=VOCAL_PRIOR_FLOOR,
            spectral_weight_min=SPECTRAL_WEIGHT_MIN,
            spectral_snr_keep_db=SPECTRAL_SNR_KEEP_DB,
            spectral_snr_soft_db=SPECTRAL_SNR_SOFT_DB,
            noise_init_percentile=NOISE_INIT_PERCENTILE,
            spectral_noise_gate_db=SPECTRAL_NOISE_GATE_DB,
            spectral_noise_alpha_quiet=SPECTRAL_NOISE_ALPHA_QUIET,
            spectral_noise_alpha_loud=SPECTRAL_NOISE_ALPHA_LOUD,
            db_eps=DB_EPS,
        )
        m = min(n_frames, int(energy_db.numel()))
        frame_dbfs = frame_dbfs[:m]
        energy_db = energy_db[:m]
        n_frames = m
    else:
        energy_db = frame_dbfs

    starts = torch.arange(n_frames, dtype=torch.float32) * (hop_len / float(sr))
    ends = starts + (frame_len / float(sr))
    _add_timing(timing, "energy_sec", time.perf_counter() - t_energy0)
    return frame_dbfs, energy_db, starts, ends, duration_sec


def _compute_anchor_targets_parallel(
    frame_db: torch.Tensor,
    i0_all: torch.Tensor,
    i1_all: torch.Tensor,
    *,
    q: float,
    global_floor: torch.Tensor,
    workers: int,
) -> torch.Tensor:
    n_frames = int(frame_db.numel())
    anchor_count = int(i0_all.numel())
    if anchor_count <= 0:
        return torch.empty(0, dtype=frame_db.dtype, device=frame_db.device)

    i0_np = i0_all.detach().cpu().numpy().astype(np.int64, copy=False)
    i1_np = i1_all.detach().cpu().numpy().astype(np.int64, copy=False)
    i0_clip = np.clip(i0_np, 0, n_frames)
    i1_clip = np.clip(i1_np, 0, n_frames)
    i1_clip = np.maximum(i1_clip, i0_clip)

    targets_np = np.full(anchor_count, float(global_floor.item()), dtype=np.float64)
    valid_mask = i1_clip > i0_clip
    if not bool(np.any(valid_mask)):
        return torch.from_numpy(targets_np).to(dtype=frame_db.dtype, device=frame_db.device)

    valid_idx = np.nonzero(valid_mask)[0]
    keys = np.stack((i0_clip[valid_mask], i1_clip[valid_mask]), axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_unique = int(unique_keys.shape[0])
    qvals = np.empty(n_unique, dtype=np.float64)

    def _compute_quantile_range(lo: int, hi: int) -> Tuple[int, np.ndarray]:
        local = np.empty(hi - lo, dtype=np.float64)
        for pos, k in enumerate(range(lo, hi)):
            s = int(unique_keys[k, 0])
            e = int(unique_keys[k, 1])
            local[pos] = float(torch.quantile(frame_db[s:e], q).item())
        return lo, local

    worker_count = max(1, min(int(workers), n_unique))
    if worker_count > 1 and n_unique >= NOISE_PARALLEL_MIN_WINDOWS:
        step = int(math.ceil(n_unique / float(worker_count)))
        with cf.ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = [
                ex.submit(
                    _compute_quantile_range,
                    start,
                    min(n_unique, start + step),
                )
                for start in range(0, n_unique, step)
            ]
            for fut in cf.as_completed(futures):
                start, vals = fut.result()
                qvals[start : start + int(vals.shape[0])] = vals
    else:
        _, vals = _compute_quantile_range(0, n_unique)
        qvals[:] = vals

    targets_np[valid_idx] = qvals[inverse]
    return torch.from_numpy(targets_np).to(dtype=frame_db.dtype, device=frame_db.device)


if nb is not None:

    @nb.njit(cache=False)
    def _noise_floor_track_numba(
        frame_db_np: np.ndarray,
        target_floor_np: np.ndarray,
        floor0: float,
        gate: float,
        alpha_follow: float,
        alpha_rise: float,
        blend: float,
    ) -> np.ndarray:
        n = frame_db_np.shape[0]
        out = np.empty(n, dtype=np.float32)
        if n <= 0:
            return out
        out[0] = np.float32(floor0)
        one_minus_blend = np.float32(1.0 - blend)
        for i in range(1, n):
            prev = out[i - 1]
            cur = np.float32(blend * frame_db_np[i] + one_minus_blend * target_floor_np[i])
            if cur <= (prev + gate):
                alpha = alpha_follow
            else:
                alpha = alpha_rise
            out[i] = np.float32(prev + alpha * (cur - prev))
        return out


def estimate_noise_floor_db_local(
    frame_db: torch.Tensor,
    frame_starts: torch.Tensor,
    duration_sec: float,
    *,
    local_window_sec: float,
    local_hop_sec: float,
    local_percentile: float,
    track_gate_db: float,
    follow_alpha: float,
    rise_alpha: float,
    local_blend: float,
) -> torch.Tensor:
    if frame_db.numel() == 0:
        return frame_db

    q = max(0.0, min(1.0, float(local_percentile) / 100.0))
    hop_sec = max(0.1, float(local_hop_sec))
    win_sec = max(hop_sec, float(local_window_sec))
    half = win_sec / 2.0

    anchor_count = max(1, int(math.floor(duration_sec / hop_sec)) + 1)
    anchors = torch.arange(anchor_count, dtype=frame_starts.dtype) * hop_sec
    if duration_sec > 0:
        anchors = torch.clamp(anchors, min=0.0, max=float(duration_sec))

    global_floor = torch.quantile(frame_db, q)

    if duration_sec <= win_sec:
        ws = torch.zeros_like(anchors)
        we = torch.full_like(anchors, float(duration_sec))
    else:
        ws = anchors - half
        we = anchors + half
        left_mask = anchors < half
        right_mask = anchors > (duration_sec - half)
        if bool(torch.any(left_mask)):
            ws[left_mask] = 0.0
            we[left_mask] = win_sec
        if bool(torch.any(right_mask)):
            ws[right_mask] = duration_sec - win_sec
            we[right_mask] = duration_sec

    i0_all = torch.searchsorted(frame_starts, ws)
    i1_all = torch.searchsorted(frame_starts, we, right=True)
    targets = _compute_anchor_targets_parallel(
        frame_db,
        i0_all,
        i1_all,
        q=q,
        global_floor=global_floor,
        workers=int(NOISE_STEP2_WORKERS),
    )

    if frame_db.numel() == 1:
        return torch.minimum(frame_db, targets[:1])

    t_np = frame_starts.detach().cpu().numpy().astype(np.float64, copy=False)
    a_np = anchors.detach().cpu().numpy().astype(np.float64, copy=False)
    y_np = targets.detach().cpu().numpy().astype(np.float64, copy=False)
    interp_np = np.interp(t_np, a_np, y_np)
    target_floor = torch.from_numpy(interp_np).to(dtype=frame_db.dtype, device=frame_db.device)

    gate = float(track_gate_db)
    alpha_follow = max(0.0, min(1.0, float(follow_alpha)))
    alpha_rise = max(0.0, min(1.0, float(rise_alpha)))
    blend = max(0.0, min(1.0, float(local_blend)))
    floor0 = min(float(frame_db[0].item()), float(global_floor.item()))

    if nb is not None and frame_db.device.type == "cpu":
        frame_np = frame_db.detach().cpu().numpy().astype(np.float32, copy=False)
        target_np = target_floor.detach().cpu().numpy().astype(np.float32, copy=False)
        out_np = _noise_floor_track_numba(
            frame_np,
            target_np,
            np.float32(floor0),
            np.float32(gate),
            np.float32(alpha_follow),
            np.float32(alpha_rise),
            np.float32(blend),
        )
        return torch.from_numpy(out_np).to(dtype=frame_db.dtype, device=frame_db.device)

    floor = torch.empty_like(frame_db)
    floor[0] = floor0
    for i in range(1, frame_db.numel()):
        prev = floor[i - 1]
        cur = blend * frame_db[i] + (1.0 - blend) * target_floor[i]
        alpha = alpha_follow if cur <= (prev + gate) else alpha_rise
        floor[i] = prev + alpha * (cur - prev)
    return floor


def _score_to_non_speech_intervals(
    energy_db: torch.Tensor,
    noise_floor_db: torch.Tensor,
    frame_dbfs: torch.Tensor,
    frame_starts: torch.Tensor,
    frame_ends: torch.Tensor,
    duration_sec: float,
    *,
    enter_margin_db: float,
    weighted: bool,
    pause_hints_out: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    n_frames = int(energy_db.numel())
    if n_frames <= 0:
        return intervals

    # Index numpy scalars in the frame loop instead of per-frame tensor.item():
    # this is the only non-numba hot loop in the VAD and .item() dominates its
    # cost. float(np.float32) equals float(tensor.item()) bit-for-bit, so the
    # scored intervals are unchanged.
    energy_np = energy_db.detach().cpu().numpy()
    noise_np = noise_floor_db.detach().cpu().numpy()
    dbfs_np = frame_dbfs.detach().cpu().numpy()
    starts_np = frame_starts.detach().cpu().numpy()
    ends_np = frame_ends.detach().cpu().numpy()

    if n_frames >= 2:
        hop_sec = float(starts_np[1] - starts_np[0])
        if not math.isfinite(hop_sec) or hop_sec <= 0:
            hop_sec = HOP_MS / 1000.0
    else:
        hop_sec = HOP_MS / 1000.0

    margin = max(float(enter_margin_db), 1e-6)
    non_speech_score = max(1.0, (MIN_NON_SPEECH_MS / 1000.0) / hop_sec)
    speech_score_ratio = float(MIN_NON_SPEECH_MS) / max(float(MERGE_GAP_MS), 1e-6)

    score = 0.0
    head = 0.0
    end = 0.0
    is_interval = False
    speech_run = 0
    banked_debit = 0.0
    hint_min_frames = max(1, int(round(PAUSE_HINT_MIN_MS / HOP_MS)))
    cand_quiet_frames = 0
    last_quiet_end = 0.0

    for i in range(n_frames):
        f = float(energy_np[i])
        n = float(noise_np[i])
        dbfs = float(dbfs_np[i])
        start_i = float(starts_np[i])
        end_i = float(ends_np[i])

        quiet_like = (f <= (n + margin)) and (dbfs <= ABS_NON_SPEECH_MAX_DBFS_ENTER)
        speech_like = (f > (n + margin)) or (dbfs >= ABS_NON_SPEECH_MAX_DBFS_EXIT)
        banked_this_frame = False

        if quiet_like:
            speech_run = 0
            banked_debit = 0.0
            cand_quiet_frames += 1
            last_quiet_end = end_i
            # Start candidate span at first positive accumulation frame.
            if score <= 0.0 and not is_interval:
                head = start_i
            if weighted:
                score += 2.0 + min(0.0, (n - f) / margin)
            else:
                score += 1.0
        elif speech_like:
            if weighted:
                speech_term = min(1.0, ((f - n - margin) / 10) ** 2)
                # Keep score monotonic downward on speech decisions.
                speech_term = max(0.1, speech_term)
                debit = speech_score_ratio * speech_term
            else:
                debit = speech_score_ratio
            if SPEECH_EXIT_MIN_RUN_FRAMES <= 1:
                score -= debit
            else:
                # Bank the debit; charge only once the run is sustained, so a
                # jitter blip cannot break an accumulation (FINDINGS Z).
                speech_run += 1
                banked_debit += debit
                if speech_run >= SPEECH_EXIT_MIN_RUN_FRAMES:
                    score -= banked_debit
                    banked_debit = 0.0
                else:
                    banked_this_frame = True

        if score >= non_speech_score and not banked_this_frame:
            # A banked (not yet charged) speech frame keeps the score at the
            # cap but must not extend the span: with an immediate charge the
            # score would already be below the cap here.
            score = non_speech_score
            end = end_i
            is_interval = True

        reached_end = (i == n_frames - 1)
        if score < 0.0 or reached_end:
            if is_interval and end > head:
                intervals.append(
                    (
                        max(0.0, min(head, duration_sec)),
                        max(0.0, min(end, duration_sec)),
                    )
                )
            elif (
                pause_hints_out is not None
                and cand_quiet_frames >= hint_min_frames
            ):
                # A pause candidate that gathered quiet evidence but died
                # before reaching the interval threshold: real pause structure
                # the interval list cannot express. Record it for future
                # splitters (40 ms left of its last quiet frame).
                pause_hints_out.append(
                    round(max(0.0, last_quiet_end - PAUSE_HINT_OFFSET_SEC), 3)
                )
            # Reset state between candidates.
            score = 0.0
            is_interval = False
            speech_run = 0
            banked_debit = 0.0
            cand_quiet_frames = 0
            if not reached_end:
                head = float(starts_np[i + 1])
            else:
                head = min(duration_sec, end_i)
            end = head

    return intervals


# A raw non-speech gap at least this long that dies in the padding shrink is
# real pause evidence the interval list can no longer express; its position is
# recorded as a hint (40 ms left of the gap's right edge, i.e. just before the
# next speech onset) for future splitters.
PAUSE_HINT_MIN_MS = 40.0
PAUSE_HINT_OFFSET_SEC = 0.04


def _apply_negative_padding_hints(
    intervals: Iterable[Tuple[float, float]],
    duration_sec: float,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    left = NEGATIVE_PAD_LEFT_MS / 1000.0
    right = NEGATIVE_PAD_RIGHT_MS / 1000.0
    min_keep = MIN_KEEP_AFTER_SHRINK_MS / 1000.0
    hint_min = PAUSE_HINT_MIN_MS / 1000.0

    out: List[Tuple[float, float]] = []
    hints: List[float] = []
    for s, e in intervals:
        ss = max(0.0, s + left)
        ee = min(duration_sec, e - right)
        if (ee - ss) >= min_keep:
            out.append((ss, ee))
        elif (e - s) >= hint_min:
            hints.append(round(max(0.0, e - PAUSE_HINT_OFFSET_SEC), 3))
    return out, hints


def _apply_negative_padding(
    intervals: Iterable[Tuple[float, float]],
    duration_sec: float,
) -> List[Tuple[float, float]]:
    out, _hints = _apply_negative_padding_hints(intervals, duration_sec)
    return out


# Partial application of the peak floor inside intervals (FINDINGS Z / user
# review): a sub- MIN_SPEECH_PEAK_DB span at an interval's head or tail is
# trimmed (keeping the decoder's lead margins), and one bridging two stretches
# of real content splits the interval. Margins mirror the negative padding.
CARVE_LEAD_IN_SEC = NEGATIVE_PAD_RIGHT_MS / 1000.0
CARVE_LEAD_OUT_SEC = NEGATIVE_PAD_LEFT_MS / 1000.0
CARVE_MIN_TRIM_SEC = 0.20
CARVE_INTERIOR_RUN_SEC = 0.40
CARVE_MIN_SEC = 0.20


def _carve_low_peak_speech(
    non_speech: List[Tuple[float, float]],
    energy_db: torch.Tensor,
    duration_sec: float,
    *,
    min_peak_db: float = None,
) -> List[Tuple[float, float]]:
    """Trim/split speech intervals along sub-threshold spans.

    Operates on (and returns) the non-speech list like the other tail steps.
    """
    peak_floor = MIN_SPEECH_PEAK_DB if min_peak_db is None else float(min_peak_db)
    hop_sec, _frame = _frame_grid_seconds()
    n = len(energy_db)
    e_np = energy_db.detach().cpu().numpy()

    speech: List[Tuple[float, float]] = []
    prev = 0.0
    for a, b in non_speech:
        if a > prev:
            speech.append((prev, a))
        prev = b
    if prev < duration_sec:
        speech.append((prev, duration_sec))

    extra_ns: List[Tuple[float, float]] = []
    for s, e in speech:
        i0 = max(0, int(math.ceil(s / hop_sec - 1e-9)))
        i1 = min(n, int(math.ceil(e / hop_sec - 1e-9)))
        if i1 <= i0:
            continue
        quiet = e_np[i0:i1] < peak_floor
        j = 0
        while j < len(quiet):
            if not quiet[j]:
                j += 1
                continue
            k = j
            while k < len(quiet) and quiet[k]:
                k += 1
            t0, t1 = s + j * hop_sec, s + k * hop_sec
            dur = t1 - t0
            head, tail = j == 0, k == len(quiet)
            if head and tail:
                pass  # whole interval: _absorb_low_peak_speech owns it
            elif head and dur >= CARVE_MIN_TRIM_SEC + CARVE_LEAD_IN_SEC:
                extra_ns.append((s, t1 - CARVE_LEAD_IN_SEC))
            elif tail and dur >= CARVE_MIN_TRIM_SEC + CARVE_LEAD_OUT_SEC:
                extra_ns.append((t0 + CARVE_LEAD_OUT_SEC, e))
            elif not head and not tail and dur >= CARVE_INTERIOR_RUN_SEC:
                c0, c1 = t0 + CARVE_LEAD_OUT_SEC, t1 - CARVE_LEAD_IN_SEC
                if c1 - c0 >= CARVE_MIN_SEC:
                    extra_ns.append((c0, c1))
            j = k
    if not extra_ns:
        return non_speech
    merged = sorted(list(non_speech) + extra_ns)
    out: List[Tuple[float, float]] = []
    for a, b in merged:
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _absorb_low_peak_speech(
    non_speech: List[Tuple[float, float]],
    energy_db: torch.Tensor,
    duration_sec: float,
    *,
    min_peak_db: float = None,
) -> List[Tuple[float, float]]:
    """Fold speech gaps whose peak energy never reaches ``min_peak_db`` back
    into non-speech. Runs on the final (padded) non-speech list so every
    producer -- memory, streamed, CLI -- shares one decision. A gap too short
    to contain a frame is kept: no evidence never removes speech."""
    peak_floor = MIN_SPEECH_PEAK_DB if min_peak_db is None else float(min_peak_db)
    hop_sec, _frame_sec = _frame_grid_seconds()
    n = len(energy_db)

    def quiet(gap_start: float, gap_end: float) -> bool:
        i0 = max(0, int(math.ceil(gap_start / hop_sec - 1e-9)))
        i1 = min(n, int(math.ceil(gap_end / hop_sec - 1e-9)))
        if i1 <= i0:
            return False
        return float(energy_db[i0:i1].max().item()) < peak_floor

    out: List[Tuple[float, float]] = []
    prev_end = 0.0
    for s, e in non_speech:
        if out and quiet(prev_end, s):
            out[-1] = (out[-1][0], e)
        elif not out and s > 0.0 and quiet(0.0, s):
            out.append((0.0, e))
        else:
            out.append((s, e))
        prev_end = e
    if out and prev_end < duration_sec and quiet(prev_end, duration_sec):
        out[-1] = (out[-1][0], duration_sec)
    return out


def _detect_non_speech_intervals_from_tracks(
    frame_dbfs: torch.Tensor,
    energy_db: torch.Tensor,
    frame_starts: torch.Tensor,
    frame_ends: torch.Tensor,
    duration_sec: float,
    *,
    snr_enter_margin_db: float,
    local_window_sec: float,
    local_hop_sec: float,
    local_percentile: float,
    track_gate_db: float,
    track_follow_alpha: float,
    track_rise_alpha: float,
    local_blend: float,
    timing: Optional[Dict[str, float]] = None,
    pause_hints_out: Optional[List[float]] = None,
) -> List[Tuple[float, float]]:
    if frame_dbfs.numel() <= 0:
        return []

    t_noise0 = time.perf_counter()
    noise_floor_db = estimate_noise_floor_db_local(
        energy_db,
        frame_starts,
        duration_sec,
        local_window_sec=local_window_sec,
        local_hop_sec=local_hop_sec,
        local_percentile=local_percentile,
        track_gate_db=track_gate_db,
        follow_alpha=track_follow_alpha,
        rise_alpha=track_rise_alpha,
        local_blend=local_blend,
    )
    _add_timing(timing, "noise_sec", time.perf_counter() - t_noise0)
    intervals = _score_to_non_speech_intervals(
        energy_db,
        noise_floor_db,
        frame_dbfs,
        frame_starts,
        frame_ends,
        duration_sec,
        enter_margin_db=snr_enter_margin_db,
        weighted=bool(WEIGHTED_INTERVAL),
        pause_hints_out=pause_hints_out,
    )
    return intervals


def detect_non_speech_intervals(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    energy_mode: str = "weighted",
    snr_enter_margin_db: float = NON_SPEECH_MARGIN_DB_ENTER,
    local_window_sec: float = NOISE_LOCAL_WINDOW_SEC,
    local_hop_sec: float = NOISE_LOCAL_HOP_SEC,
    local_percentile: float = NOISE_INIT_PERCENTILE,
    track_gate_db: float = NOISE_TRACK_GATE_DB,
    track_follow_alpha: float = NOISE_TRACK_FOLLOW_ALPHA,
    track_rise_alpha: float = NOISE_TRACK_RISE_ALPHA,
    local_blend: float = NOISE_LOCAL_BLEND,
) -> List[Tuple[float, float]]:
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate}")
    energy_mode = str(energy_mode).strip().lower()
    if energy_mode not in {"none", "weighted"}:
        raise ValueError(f"Unsupported energy_mode: {energy_mode}")

    duration_sec = waveform.numel() / float(sample_rate)
    if waveform.numel() == 0:
        return []

    frame_dbfs, energy_db, frame_starts, frame_ends = _compute_frame_tracks_for_waveform(
        waveform,
        sample_rate,
        energy_mode=energy_mode,
    )
    raw = _detect_non_speech_intervals_from_tracks(
        frame_dbfs,
        energy_db,
        frame_starts,
        frame_ends,
        duration_sec,
        snr_enter_margin_db=snr_enter_margin_db,
        local_window_sec=local_window_sec,
        local_hop_sec=local_hop_sec,
        local_percentile=local_percentile,
        track_gate_db=track_gate_db,
        track_follow_alpha=track_follow_alpha,
        track_rise_alpha=track_rise_alpha,
        local_blend=local_blend,
        timing=None,
    )
    padded = _apply_negative_padding(raw, duration_sec)
    padded = _absorb_low_peak_speech(padded, energy_db, duration_sec)
    return _carve_low_peak_speech(padded, energy_db, duration_sec)


def detect_non_speech_intervals_file(
    path: str | Path,
    *,
    energy_mode: str = "weighted",
    core_sec: float = STREAM_CORE_SEC,
    context_sec: float = STREAM_CONTEXT_SEC,
    snr_enter_margin_db: float = NON_SPEECH_MARGIN_DB_ENTER,
    local_window_sec: float = NOISE_LOCAL_WINDOW_SEC,
    local_hop_sec: float = NOISE_LOCAL_HOP_SEC,
    local_percentile: float = NOISE_INIT_PERCENTILE,
    track_gate_db: float = NOISE_TRACK_GATE_DB,
    track_follow_alpha: float = NOISE_TRACK_FOLLOW_ALPHA,
    track_rise_alpha: float = NOISE_TRACK_RISE_ALPHA,
    local_blend: float = NOISE_LOCAL_BLEND,
    timing: Optional[Dict[str, float]] = None,
    observer: Optional[WaveformObserver] = None,
) -> Tuple[
    List[Tuple[float, float]], float, VadEnergyTrack, Dict[str, List[float]]
]:
    """Non-speech intervals, duration, energy track and pause hints by source."""

    energy_mode = str(energy_mode).strip().lower()
    if energy_mode not in {"none", "weighted"}:
        raise ValueError(f"Unsupported energy_mode: {energy_mode}")

    frame_dbfs, energy_db, frame_starts, frame_ends, duration_sec = _streamed_frame_tracks(
        path,
        energy_mode=energy_mode,
        core_sec=float(core_sec),
        context_sec=float(context_sec),
        timing=timing,
        observer=observer,
    )
    scorer_hints: List[float] = []
    raw = _detect_non_speech_intervals_from_tracks(
        frame_dbfs,
        energy_db,
        frame_starts,
        frame_ends,
        duration_sec,
        snr_enter_margin_db=snr_enter_margin_db,
        local_window_sec=local_window_sec,
        local_hop_sec=local_hop_sec,
        local_percentile=local_percentile,
        track_gate_db=track_gate_db,
        track_follow_alpha=track_follow_alpha,
        track_rise_alpha=track_rise_alpha,
        local_blend=local_blend,
        timing=timing,
        pause_hints_out=scorer_hints,
    )
    hop_sec, frame_sec = _frame_grid_seconds()
    track = VadEnergyTrack(
        energy_db=energy_db.detach().to(device="cpu").contiguous(),
        hop_sec=hop_sec,
        frame_sec=frame_sec,
        energy_mode=energy_mode,
        frame_dbfs=frame_dbfs.detach().to(device="cpu").contiguous(),
    )
    padded, pad_hints = _apply_negative_padding_hints(raw, duration_sec)
    padded = _absorb_low_peak_speech(padded, energy_db, duration_sec)
    padded = _carve_low_peak_speech(padded, energy_db, duration_sec)
    # Kept apart by source rather than merged into one sorted list: the two
    # mean different things and only one of them can currently fire. A padding
    # hint needs a raw gap below 190 ms (40 + 140 + min-keep), while the
    # weighted scorer cannot certify a gap below ~200 ms in the first place --
    # so `padding` staying empty is the expected state, and it becoming
    # non-empty is exactly what someone changing NEGATIVE_PAD_RIGHT_MS needs to
    # notice. Counting them separately makes that visible in the artifact.
    pause_hints = {
        "scorer": sorted(set(scorer_hints)),
        "padding": sorted(set(pad_hints)),
    }
    return padded, duration_sec, track, pause_hints


def vad_params(
    *,
    energy_mode: str = "weighted",
    snr_enter_margin_db: Optional[float] = None,
    local_window_sec: float = NOISE_LOCAL_WINDOW_SEC,
    local_hop_sec: float = NOISE_LOCAL_HOP_SEC,
    local_percentile: float = NOISE_INIT_PERCENTILE,
    track_gate_db: float = NOISE_TRACK_GATE_DB,
    track_follow_alpha: float = NOISE_TRACK_FOLLOW_ALPHA,
    track_rise_alpha: float = NOISE_TRACK_RISE_ALPHA,
    local_blend: float = NOISE_LOCAL_BLEND,
) -> Dict[str, object]:
    enter = (
        float(NON_SPEECH_MARGIN_DB_ENTER)
        if snr_enter_margin_db is None
        else float(snr_enter_margin_db)
    )
    return {
        "backend": "vad-energy-score-local120",
        "device": "cpu",
        "target_sr": TARGET_SR,
        "block_length_sec": BLOCK_LENGTH,
        "energy_mode": str(energy_mode),
        "snr_enter_margin_db": enter,
        "local_window_sec": float(local_window_sec),
        "local_hop_sec": float(local_hop_sec),
        "local_percentile": float(local_percentile),
        "track_gate_db": float(track_gate_db),
        "track_follow_alpha": float(track_follow_alpha),
        "track_rise_alpha": float(track_rise_alpha),
        "local_blend": float(local_blend),
    }


def run_vad(
    waveform: torch.Tensor,
    *,
    params: Optional[Dict[str, object]] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    params = dict(params or {})
    params["device"] = "cpu"
    energy_mode = str(params.get("energy_mode") or "weighted")
    snr_enter = float(params.get("snr_enter_margin_db", NON_SPEECH_MARGIN_DB_ENTER))
    local_window_sec = float(params.get("local_window_sec", NOISE_LOCAL_WINDOW_SEC))
    local_hop_sec = float(params.get("local_hop_sec", NOISE_LOCAL_HOP_SEC))
    local_percentile = float(params.get("local_percentile", NOISE_INIT_PERCENTILE))
    track_gate_db = float(params.get("track_gate_db", NOISE_TRACK_GATE_DB))
    track_follow_alpha = float(params.get("track_follow_alpha", NOISE_TRACK_FOLLOW_ALPHA))
    track_rise_alpha = float(params.get("track_rise_alpha", NOISE_TRACK_RISE_ALPHA))
    local_blend = float(params.get("local_blend", NOISE_LOCAL_BLEND))
    x = waveform.to(torch.device("cpu"))

    non_speech = detect_non_speech_intervals(
        x,
        TARGET_SR,
        energy_mode=energy_mode,
        snr_enter_margin_db=snr_enter,
        local_window_sec=local_window_sec,
        local_hop_sec=local_hop_sec,
        local_percentile=local_percentile,
        track_gate_db=track_gate_db,
        track_follow_alpha=track_follow_alpha,
        track_rise_alpha=track_rise_alpha,
        local_blend=local_blend,
    )
    speech = invert_intervals(non_speech, x.numel() / float(TARGET_SR))
    items = [{"start": s, "end": e} for s, e in speech if e > s]
    return items, {"vad": params}


def run_vad_file(
    path: str | Path,
    *,
    params: Optional[Dict[str, object]] = None,
    core_sec: float = STREAM_CORE_SEC,
    context_sec: float = STREAM_CONTEXT_SEC,
    timing: Optional[Dict[str, float]] = None,
    observer: Optional[WaveformObserver] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object], float, VadEnergyTrack]:
    """Streamed run_vad: loading, normalization and framing go block by block
    (RAM bounded by one core block), bit-identical to light_normalize +
    run_vad on the fully loaded audio. Also returns the frame energy track.

    ``observer`` rides along on the normalized blocks (see WaveformObserver), so
    a second signal over the same audio costs no extra decode pass."""

    params = dict(params or {})
    params["device"] = "cpu"
    params["streaming"] = {
        "core_sec": float(core_sec),
        "context_sec": float(context_sec),
    }
    energy_mode = str(params.get("energy_mode") or "weighted")
    snr_enter = float(params.get("snr_enter_margin_db", NON_SPEECH_MARGIN_DB_ENTER))

    non_speech, duration_sec, energy_track, pause_hints = detect_non_speech_intervals_file(
        path,
        energy_mode=energy_mode,
        core_sec=core_sec,
        context_sec=context_sec,
        snr_enter_margin_db=snr_enter,
        local_window_sec=float(
            params.get("local_window_sec", NOISE_LOCAL_WINDOW_SEC)
        ),
        local_hop_sec=float(params.get("local_hop_sec", NOISE_LOCAL_HOP_SEC)),
        local_percentile=float(
            params.get("local_percentile", NOISE_INIT_PERCENTILE)
        ),
        track_gate_db=float(params.get("track_gate_db", NOISE_TRACK_GATE_DB)),
        track_follow_alpha=float(
            params.get("track_follow_alpha", NOISE_TRACK_FOLLOW_ALPHA)
        ),
        track_rise_alpha=float(
            params.get("track_rise_alpha", NOISE_TRACK_RISE_ALPHA)
        ),
        local_blend=float(params.get("local_blend", NOISE_LOCAL_BLEND)),
        timing=timing,
        observer=observer,
    )
    speech = invert_intervals(non_speech, duration_sec)
    items = [{"start": s, "end": e} for s, e in speech if e > s]
    # Observations, not parameters: `params` is what the VAD was asked to do
    # and is compared across the streaming/in-memory paths, so the hints ride
    # alongside it and land in the artifact's `vad_timeline` instead.
    return items, {"vad": params, "pause_hints": pause_hints}, duration_sec, energy_track


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    device_for_usage: Optional[str] = "cuda" if cuda_usable() else None
    reset_peak_gpu_memory_stats_for_run(device_for_usage)
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"Input not found: {input_path}", file=sys.stderr)
            return 1

        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else default_output_path(input_path)
        )

        try:
            snr_enter = (
                float(args.snr)
                if args.snr is not None
                else float(NON_SPEECH_MARGIN_DB_ENTER)
            )
            energy_mode = str(args.energy_mode).strip().lower()
            if energy_mode not in {"none", "weighted"}:
                raise ValueError(f"Unsupported energy_mode: {energy_mode}")

            sr = TARGET_SR
            timing: Dict[str, float] = {
                "loading_sec": 0.0,
                "energy_sec": 0.0,
                "noise_sec": 0.0,
            }
            frame_dbfs, energy_db, frame_starts, frame_ends, duration_sec = _streamed_frame_tracks(
                input_path,
                energy_mode=energy_mode,
                timing=timing,
            )

            raw_non_speech = _detect_non_speech_intervals_from_tracks(
                frame_dbfs,
                energy_db,
                frame_starts,
                frame_ends,
                duration_sec,
                snr_enter_margin_db=snr_enter,
                local_window_sec=NOISE_LOCAL_WINDOW_SEC,
                local_hop_sec=NOISE_LOCAL_HOP_SEC,
                local_percentile=NOISE_INIT_PERCENTILE,
                track_gate_db=NOISE_TRACK_GATE_DB,
                track_follow_alpha=NOISE_TRACK_FOLLOW_ALPHA,
                track_rise_alpha=NOISE_TRACK_RISE_ALPHA,
                local_blend=NOISE_LOCAL_BLEND,
                timing=timing,
            )

            # Step 3: finalize intervals/output from tracked+detected data.
            intervals = _carve_low_peak_speech(
                _absorb_low_peak_speech(
                    _apply_negative_padding(raw_non_speech, duration_sec),
                    energy_db,
                    duration_sec,
                ),
                energy_db,
                duration_sec,
            )
            speech_items = [{"start": s, "end": e} for s, e in invert_intervals(intervals, duration_sec)]
            intervals = [(float(item["start"]), float(item["end"])) for item in speech_items]
        except Exception as exc:
            print(f"Failed to detect non-speech: {exc}", file=sys.stderr)
            return 1

        srt = render_non_speech_srt(intervals)
        output_path.write_text(srt, encoding="utf-8")
        t_total = time.perf_counter() - t0
        t_loading = float(timing.get("loading_sec", 0.0))
        t_energy = float(timing.get("energy_sec", 0.0))
        t_noise = float(timing.get("noise_sec", 0.0))
        t_others = max(0.0, t_total - (t_loading + t_energy + t_noise))

        speed_x = (duration_sec / t_total) if t_total > 0 else float("inf")

        print(
            f"Wrote {output_path} "
            f"(intervals={len(intervals)}, duration={duration_sec:.2f}s, sr={sr}, "
            f"mode={energy_mode}, snr={snr_enter:.2f}, threads={THREAD_NUM})"
        )
        print(
            "Timing: "
            f"loading={t_loading:.3f}s, energy={t_energy:.3f}s, noise={t_noise:.3f}s, "
            f"others={t_others:.3f}s, "
            f"total={t_total:.3f}s, avg_speed={speed_x:.1f}x"
        )
        print("Interval length buckets (sec):")
        for label, count, pct in interval_length_bucket_summary(intervals):
            print(f"  {label}: {count} ({pct:.1f}%)")
        return 0
    finally:
        print_peak_resource_usage(device_for_usage)


if __name__ == "__main__":
    raise SystemExit(main())
