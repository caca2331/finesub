"""Hybrids that keep the energy detector as the recall backbone.

v1-v3 established the asymmetry that shapes these:

  the energy detector loses almost no speech (2.7 s of candidate misses on the gold
  clip, zero words inside), while silero at its default discards 60 s and 19 words.

So silero is not usable as the decision maker, and not needed as a recall net. It is
used only where the energy detector is deliberately imprecise: the fixed 140 ms
right-shrink on non-speech intervals, which makes every speech interval start up to
140 ms before the audio actually resumes. That margin exists to protect consonant
onsets, and it is also what drags filled pauses and breaths into the first word.

`AdaptiveHead` spends that margin only when silero agrees nothing is being said.
`GuardedAggressive` runs the reverse experiment: cut hard on energy, then let silero
veto the cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from backends import SILERO_HOP_SEC, Interval, union

MAX_TRIM_DEFAULT = 0.14   # never give back more than the production right-shrink


def _prob_at(probs: np.ndarray, t: float) -> float:
    i = int(t / SILERO_HOP_SEC)
    if i < 0 or i >= len(probs):
        return 0.0
    return float(probs[i])


def _max_prob(probs: np.ndarray, a: float, b: float) -> float:
    i0 = max(0, int(a / SILERO_HOP_SEC))
    i1 = min(len(probs), int(np.ceil(b / SILERO_HOP_SEC)))
    if i1 <= i0:
        return _prob_at(probs, a)
    return float(probs[i0:i1].max())


@dataclass(frozen=True)
class AdaptiveHead:
    """Trim the head of each speech interval while silero is confident it is silent.

    The trim is bounded by `max_trim` so that in the worst case -- silero wrong on
    every frame -- the result is still no tighter than the energy detector running
    with NEGATIVE_PAD_RIGHT_MS reduced by `max_trim`, an operating point v3 already
    showed keeps every word.
    """

    thr: float = 0.30
    max_trim: float = MAX_TRIM_DEFAULT
    step: float = SILERO_HOP_SEC
    min_keep: float = 0.20     # never shrink an interval below this

    def __call__(self, speech: Sequence[Interval], probs: np.ndarray) -> List[Interval]:
        out: List[Interval] = []
        for s, e in speech:
            limit = min(s + self.max_trim, e - self.min_keep)
            t = s
            while t + self.step <= limit and _max_prob(probs, t, t + self.step) < self.thr:
                t += self.step
            out.append((t, e))
        return out


@dataclass(frozen=True)
class GuardedAggressive:
    """Cut aggressively on energy, then union back anything silero calls speech.

    The reverse bet from AdaptiveHead: let the energy detector be wrong in the
    dangerous direction and rely on silero to undo it.
    """

    thr: float = 0.50
    min_speech: float = 0.10
    pad: float = 0.05

    def __call__(self, aggressive: Sequence[Interval], probs: np.ndarray,
                 duration: float) -> List[Interval]:
        mask = probs >= self.thr
        runs, j, n = [], 0, len(mask)
        while j < n:
            if mask[j]:
                k = j
                while k < n and mask[k]:
                    k += 1
                if (k - j) * SILERO_HOP_SEC >= self.min_speech:
                    runs.append((max(0.0, j * SILERO_HOP_SEC - self.pad),
                                 min(duration, k * SILERO_HOP_SEC + self.pad)))
                j = k
            else:
                j += 1
        return union(aggressive, runs)


@dataclass(frozen=True)
class DropGhostIntervals:
    """Keep every energy interval except those with no speech anywhere in them.

    The synthesis the miyako data points at. Silero cannot be trusted to place
    boundaries -- it cuts real dialogue -- but "this entire interval never once
    looks like speech" is a much weaker claim, and it is exactly the claim that
    identifies the noise-triggered intervals feeding whisper pure background.

    An interval survives if silero's probability crosses `peak_thr` at any single
    frame, so any interval containing real speech is kept whole, boundaries and all.
    """

    peak_thr: float = 0.50
    max_drop_sec: float = 12.0   # never discard a long interval on this evidence alone

    def __call__(self, speech: Sequence[Interval], probs: np.ndarray) -> List[Interval]:
        out: List[Interval] = []
        for s, e in speech:
            i0 = int(s / SILERO_HOP_SEC)
            i1 = max(i0 + 1, int(e / SILERO_HOP_SEC))
            peak = float(probs[i0:i1].max()) if i1 <= len(probs) else 1.0
            if peak < self.peak_thr and (e - s) <= self.max_drop_sec:
                continue
            out.append((s, e))
        return out
