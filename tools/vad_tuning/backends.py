"""VAD backends behind one interface: audio path -> speech intervals [(start, end)].

Silero is run once per clip to get its frame probability track, which is cached; the
threshold and duration post-processing are reimplemented here so a sweep costs
nothing and so the same post-processing can be reused by the hybrid.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

SR = 16000
SILERO_HOP = 512          # samples; silero's native frame step at 16 kHz
SILERO_HOP_SEC = SILERO_HOP / SR

Interval = Tuple[float, float]


# --------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------

def load_mono(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return y.astype(np.float32)


# --------------------------------------------------------------------------
# production energy VAD
# --------------------------------------------------------------------------

def energy_vad(path: Path, overrides: Optional[Dict[str, object]] = None,
               const_overrides: Optional[Dict[str, object]] = None) -> List[Interval]:
    """Run the production detector. `const_overrides` patches module-level tunables
    that `vad_params` does not expose (interval post-processing); it is restored
    afterwards so nothing leaks between runs."""
    from asr_playground.speech.preprocessing import energy as E

    saved = {}
    try:
        for k, v in (const_overrides or {}).items():
            saved[k] = getattr(E, k)
            setattr(E, k, v)
        params = E.vad_params(**(overrides or {}))
        items, _meta, _dur, _track = E.run_vad_file(path, params=params)
    finally:
        for k, v in saved.items():
            setattr(E, k, v)
    return [(float(i["start"]), float(i["end"])) for i in items]


def energy_frame_track(path: Path) -> Tuple[np.ndarray, float, float]:
    """Frame energy in dB plus the noise floor the detector tracked, on its own grid."""
    from asr_playground.speech.preprocessing import energy as E

    _items, _meta, dur, track = E.run_vad_file(path, params=E.vad_params())
    return track.energy_db.numpy(), float(track.hop_sec), float(dur)


# --------------------------------------------------------------------------
# silero
# --------------------------------------------------------------------------

_SILERO = None


def _silero_model():
    global _SILERO
    if _SILERO is None:
        from silero_vad import load_silero_vad

        _SILERO = load_silero_vad()
    return _SILERO


def silero_probs(path: Path, cache: Optional[Path] = None) -> np.ndarray:
    """Per-frame speech probability on a 512-sample grid."""
    if cache is not None and cache.exists():
        return np.load(cache)["probs"]

    import torch

    y = load_mono(path)
    model = _silero_model()
    model.reset_states()
    n = (len(y) // SILERO_HOP) * SILERO_HOP
    frames = torch.from_numpy(y[:n]).reshape(-1, SILERO_HOP)
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), 512):
            chunk = frames[i:i + 512]
            out.extend(float(model(f, SR).item()) for f in chunk)
    probs = np.asarray(out, dtype=np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, probs=probs)
    return probs


@dataclass(frozen=True)
class Hysteresis:
    """Threshold + duration post-processing, shared by silero and the hybrid.

    enter/exit implement hysteresis: a frame joins speech above `enter` and speech
    only ends after dropping below `exit`. `pad` widens each kept region, which is
    what protects word onsets and codas from being clipped.
    """

    enter: float = 0.5
    exit: float = 0.35
    min_speech: float = 0.10
    min_silence: float = 0.30
    pad: float = 0.10

    def apply(self, score: np.ndarray, hop: float, duration: float) -> List[Interval]:
        speech = np.zeros(len(score), dtype=bool)
        on = False
        for i, v in enumerate(score):
            if on:
                on = v >= self.exit
            else:
                on = v >= self.enter
            speech[i] = on

        regions = _runs(speech)
        # close short silences first, then drop short speech blips
        need_sil = int(round(self.min_silence / hop))
        merged: List[List[int]] = []
        for a, b in regions:
            if merged and a - merged[-1][1] < need_sil:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        need_sp = int(round(self.min_speech / hop))
        kept = [(a, b) for a, b in merged if b - a >= need_sp]

        out: List[Interval] = []
        for a, b in kept:
            s = max(0.0, a * hop - self.pad)
            e = min(duration, b * hop + self.pad)
            if out and s <= out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], e))
            else:
                out.append((s, e))
        return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    out, j, n = [], 0, len(mask)
    while j < n:
        if mask[j]:
            k = j
            while k < n and mask[k]:
                k += 1
            out.append((j, k))
            j = k
        else:
            j += 1
    return out


def silero_vad(path: Path, hyst: Hysteresis = Hysteresis(),
               cache: Optional[Path] = None) -> List[Interval]:
    probs = silero_probs(path, cache)
    duration = len(load_mono(path)) / SR
    return hyst.apply(probs, SILERO_HOP_SEC, duration)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def invert(intervals: Sequence[Interval], duration: float) -> List[Interval]:
    out, prev = [], 0.0
    for a, b in sorted(intervals):
        if a > prev:
            out.append((prev, a))
        prev = max(prev, b)
    if prev < duration:
        out.append((prev, duration))
    return out


def union(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    merged: List[Interval] = []
    for s, e in sorted(list(a) + list(b)):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def intersect(a: Sequence[Interval], b: Sequence[Interval]) -> List[Interval]:
    out: List[Interval] = []
    i = j = 0
    a, b = sorted(a), sorted(b)
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(intervals: Sequence[Interval]) -> float:
    return float(sum(e - s for s, e in intervals))
