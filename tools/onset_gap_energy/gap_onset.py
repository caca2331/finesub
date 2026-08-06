"""The candidate heuristic, in one place.

Rule: a word start is "too early" when the audio goes quiet almost immediately after
it and then speech resumes. In that case the start belongs at the end of the quiet
stretch. Everything else is left alone.

Gates, and why each one exists:

- `delta_db` above a *local* noise floor decides what "quiet" means. The floor is a
  low percentile of a +-5 s neighbourhood, so it tracks background level rather than
  a global constant.
- `min_gap` rejects inter-syllable dips.
- `max_lag` is the discriminator. If the quiet stretch only starts well after the
  production timestamp, the word really did begin there and we must not touch it.
- `max_move` caps the damage of a wrong fire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from features import Tracks, noise_floor_db

FRAME = 0.01


@dataclass(frozen=True)
class GapOnset:
    delta_db: float = 14.0
    min_gap: float = 0.06
    max_lag: float = 0.16
    max_move: float = 0.80
    window: float = 1.20

    def find(self, tr: Tracks, s: float) -> Optional[Tuple[float, float]]:
        """First qualifying quiet run after `s`, as (gap_start, gap_end)."""
        i0 = tr.idx(s)
        i1 = min(tr.idx(s + self.window) + 1, len(tr.times))
        if i1 - i0 < 3:
            return None
        floor = noise_floor_db(tr.rms_db, i0)
        quiet = tr.rms_db[i0:i1] <= floor + self.delta_db
        need = max(1, int(round(self.min_gap / FRAME)))
        j, n = 0, len(quiet)
        while j < n:
            if quiet[j]:
                k = j
                while k < n and quiet[k]:
                    k += 1
                if k - j >= need and k < n:
                    return float(tr.times[i0 + j]), float(tr.times[i0 + k])
                j = k
            else:
                j += 1
        return None

    def __call__(self, tr: Tracks, s: float, hard_limit: Optional[float] = None) -> float:
        g = self.find(tr, s)
        if g is None:
            return s
        g0, g1 = g
        if g0 - s > self.max_lag:
            return s
        if g1 - s > self.max_move:
            return s
        if hard_limit is not None and g1 > hard_limit:
            return s
        return g1
