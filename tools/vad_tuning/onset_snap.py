"""Move a speech interval's start to where the speech actually starts.

The padding sweep says uniform padding is exhausted: cutting NEGATIVE_PAD_RIGHT_MS
from 140 to 40 recovers 7.6 s of the 71.5 s of dead lead-in and takes the count of
clipped word onsets from 3 to 27. That is the signature of a boundary whose error is
*variance*, not bias -- half the intervals open too early and half already open late,
so a uniform shift trades one error for the other and buys almost nothing.

Fixing variance needs a per-interval decision, which is what the onset study at the
start of this branch found: the gold onset is the end of the last quiet stretch
before the energy rise (within +/-5 ms on 14 of 16 annotated cases), and the
discriminator is the *latency* of that silence, not its depth. That study ran on its
own librosa feature tracks; this reimplements the same rule on the tracks the VAD
already computes, so it costs nothing extra.

Conservative by construction: the start may only move *later* (never earlier, so it
can never uncover more audio), never past `max_move`, and only when a qualifying
quiet stretch is actually found. When in doubt the interval is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Interval = Tuple[float, float]
HOP = 0.01


@dataclass(frozen=True)
class OnsetSnap:
    rise_db: float = 10.0      # dB over the floor that counts as the onset
    quiet_db: float = 4.0      # dB over the floor that still counts as silence
    min_gap: float = 0.12      # silence must last this long to be a real gap;
                               # shorter is stop-consonant closure, not a boundary
    max_move: float = 0.60     # never push the start later than this
    guard: float = 0.03        # keep this much lead-in before the onset

    def __call__(self, speech: Sequence[Interval], energy_db: np.ndarray,
                 floor: np.ndarray) -> Tuple[List[Interval], dict]:
        snr = energy_db - floor
        n = len(snr)
        out: List[Interval] = []
        st = {"moved": 0, "moved_sec": 0.0, "left": 0}
        for s, e in speech:
            a = min(max(int(round(s / HOP)), 0), n - 1)
            lim = min(int(round((s + self.max_move) / HOP)), int(round(e / HOP)), n)
            new = None
            if lim - a > 1:
                loud = snr[a:lim] > self.rise_db
                idx = np.flatnonzero(loud)
                if idx.size:
                    # the first real onset in the window, and the quiet run before it
                    k = a + int(idx[0])
                    q0 = k
                    while q0 > a and snr[q0 - 1] <= self.quiet_db:
                        q0 -= 1
                    if (k - q0) * HOP >= self.min_gap:
                        new = max(s, k * HOP - self.guard)
            if new is not None and new > s and (e - new) > 0.05:
                st["moved"] += 1
                st["moved_sec"] += new - s
                out.append((new, e))
            else:
                st["left"] += 1
                out.append((s, e))
        return out, st
