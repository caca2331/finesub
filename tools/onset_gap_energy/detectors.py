"""Forward-search onset detectors.

All of them share one shape: start at the production timestamp `s`, walk forward at
most `window` seconds, and return the first time the audio looks like it is actually
speaking. They never move a timestamp earlier -- the production error at segment
starts is one-sided (too early), and moving earlier has no acoustic justification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from features import HOP, SR, Tracks, noise_floor_db

FRAME = HOP / SR  # 0.01 s


@dataclass(frozen=True)
class SilenceExit:
    """Advance to the first sustained rise above the local noise floor.

    delta_db  how far above the local floor counts as speech
    sustain   how many consecutive frames must clear it (rejects clicks)
    window    maximum forward move
    backoff   pull the answer back by this much (an onset's energy peaks after
              the articulation actually begins)
    min_move  moves smaller than this are dropped -- keeps the production value
    """

    delta_db: float = 8.0
    sustain: int = 3
    window: float = 1.0
    backoff: float = 0.02
    min_move: float = 0.05
    track: str = "rms_db"

    def __call__(self, tr: Tracks, s: float, limit: Optional[float] = None) -> float:
        i0 = tr.idx(s)
        hi = s + self.window if limit is None else min(s + self.window, limit)
        i1 = min(tr.idx(hi) + 1, len(tr.times))
        if i1 - i0 < self.sustain + 1:
            return s
        sig = getattr(tr, self.track)
        floor = noise_floor_db(tr.rms_db, i0)
        mask = sig[i0:i1] > floor + self.delta_db
        # first index whose next `sustain` frames are all speech
        run = np.convolve(mask.astype(int), np.ones(self.sustain, dtype=int), mode="valid")
        hits = np.flatnonzero(run == self.sustain)
        if hits.size == 0:
            return s
        j = int(hits[0])
        if j == 0:
            return s  # already speaking at the production start
        t = tr.times[i0 + j] - self.backoff
        if t - s < self.min_move:
            return s
        return float(max(s, min(t, hi)))


@dataclass(frozen=True)
class LastGapExit:
    """Advance to the end of the *last* silence run inside the window.

    Different bet from SilenceExit: a filled pause can be voiced, so the first
    energy rise may still be the pause. Taking the last gap before the window's
    end skips over a pause that is itself followed by a short silence.
    """

    delta_db: float = 8.0
    min_gap: float = 0.06
    window: float = 1.0
    backoff: float = 0.02
    min_move: float = 0.05

    def __call__(self, tr: Tracks, s: float, limit: Optional[float] = None) -> float:
        i0 = tr.idx(s)
        hi = s + self.window if limit is None else min(s + self.window, limit)
        i1 = min(tr.idx(hi) + 1, len(tr.times))
        if i1 <= i0 + 2:
            return s
        floor = noise_floor_db(tr.rms_db, i0)
        quiet = tr.rms_db[i0:i1] <= floor + self.delta_db
        need = max(1, int(round(self.min_gap / FRAME)))
        best_end = None
        j = 0
        n = len(quiet)
        while j < n:
            if quiet[j]:
                k = j
                while k < n and quiet[k]:
                    k += 1
                if k - j >= need and k < n:
                    best_end = k
                j = k
            else:
                j += 1
        if best_end is None:
            return s
        t = tr.times[i0 + best_end] - self.backoff
        if t - s < self.min_move:
            return s
        return float(max(s, min(t, hi)))


@dataclass(frozen=True)
class OnsetPeak:
    """Advance to the first strong peak of the spectral-flux onset envelope."""

    z: float = 2.0
    window: float = 1.0
    backoff: float = 0.02
    min_move: float = 0.05

    def __call__(self, tr: Tracks, s: float, limit: Optional[float] = None) -> float:
        i0 = tr.idx(s)
        hi = s + self.window if limit is None else min(s + self.window, limit)
        i1 = min(tr.idx(hi) + 1, len(tr.times))
        if i1 <= i0 + 2:
            return s
        ctx0, ctx1 = max(0, i0 - 300), min(len(tr.onset_env), i1 + 300)
        ctx = tr.onset_env[ctx0:ctx1]
        thr = ctx.mean() + self.z * ctx.std()
        env = tr.onset_env[i0:i1]
        hits = np.flatnonzero(env > thr)
        if hits.size == 0:
            return s
        j = int(hits[0])
        if j == 0:
            return s
        t = tr.times[i0 + j] - self.backoff
        if t - s < self.min_move:
            return s
        return float(max(s, min(t, hi)))


def keep(tr: Tracks, s: float, limit: Optional[float] = None) -> float:
    return s


Detector = Callable[..., float]
