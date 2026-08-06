"""Re-judge the regions the energy detector kept but silero rejects.

Those regions are not one phenomenon, and treating them uniformly is why the flat
rules in v4/v5 all traded recall for precision. Three scales, three risk profiles:

  whole interval   The energy detector opened an interval on something that never
                   once looks like speech. Dropping it removes a whole block of
                   background from the decoder. This is the *weakest* claim to make
                   ("nothing here"), so it is the safest and the highest value --
                   these are what feed hallucination.

  interior span    A long silero-negative stretch inside an otherwise good interval.
                   Acting means splitting the interval in two. The decoder then gets
                   a segmentation cue instead of a stretch of noise, but grouping
                   changes, so the bar is high.

  boundary span    The interval's head or tail extends past the speech. Head and tail
                   are NOT symmetric: `inserted_gap_parts` keeps up to
                   GAP_KEEP_REAL_MAX_SEC of real audio after the *left* interval, so
                   a tail cut is recoverable, while the audio before an interval is
                   replaced by synthetic silence, so a head cut is permanent. Tails
                   get a loose budget, heads a tight one.

Both signals must agree before anything is removed. Silero says whether it hears
speech; `energy_db - noise_floor` says how much sound is actually there. Real speech
sits far above the local floor (+15..+69 dB in the onset study), weak noise sits just
over it (+6..12 dB). A silero negative on a *loud* region is treated as silero being
wrong -- that is the shouting/singing/distortion case -- and the audio is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backends import SILERO_HOP_SEC, Interval

ENERGY_HOP = 0.01


@dataclass
class Decision:
    kept: List[Interval]
    stats: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptiveRefine:
    # --- silero side -------------------------------------------------------
    sil_neg: float = 0.35        # a frame below this: silero hears no speech
    ghost_peak: float = 0.50     # interval never reaching this: nothing anywhere
    sil_sure: float = 0.20       # silero is emphatic, can act on weaker energy support

    # --- energy side (dB over the tracked local noise floor) ---------------
    snr_noise: float = 12.0      # at or under this the sound is noise-like
    snr_protect: float = 20.0    # over this, assume silero is wrong and keep

    # --- per class ---------------------------------------------------------
    # A ghost is dropped only when BOTH signals agree. Measured on 77 ghost
    # intervals from the clean clips: those containing a real word sit at median
    # 37.6 dB over the floor (p25 29.6), those containing nothing at 10.8 (p75
    # 17.1), miyako's at 11.3. Energy separates them; silero alone does not --
    # silero is emphatically negative on both.
    # Set from the 9 clean-clip ghost intervals that carry text. They are not
    # fillers -- 配信終わっちゃうかもー。/ もう読み上げられないー / 可愛すぎるみ are
    # real utterances, and the particles they contribute (って / の / ます) are
    # sentence pieces, not droppable filler. With the current noise floor those sit
    # at 28-52 dB and only ヤホッ! (14.1 dB, a genuine interjection) falls under
    # 25 dB. Re-derive this if the floor estimator changes again -- the threshold is
    # meaningless without the floor it is measured against; `v13_ghost_snr.py`
    # reprints the table against whatever is in production.
    #   snr_max   clean utterances dropped     empty ghosts (of 54)   miyako ghosts
    #      15     none                                 31                  161
    #      25     ヤホッ! only                          48                  186  <- default
    #      30     + 可愛すぎるみ                        49                  194
    #      35     + なんだこの!                         50                  204
    ghost_snr_max: float = 25.0     # louder than this: keep, whatever silero says
    ghost_snr_long: float = 15.0    # quiet enough to also allow a long drop
    max_ghost_sec: float = 3.0      # normal duration cap
    max_ghost_long_sec: float = 12.0  # cap when the region is very quiet
    max_head_trim: float = 0.16     # unrecoverable, so tight
    max_tail_trim: float = 0.60     # recoverable from the kept gap audio
    min_interior_sec: float = 0.60  # shorter interior gaps are intra-phrase pauses
    min_keep_sec: float = 0.25      # never shrink a kept interval below this

    # --- which classes are active (for ablation) ---------------------------
    do_ghost: bool = True
    # Head trimming is off by default: it duplicates NEGATIVE_PAD_RIGHT_MS, which
    # already trims the systematic head margin, and the ablation showed it buys
    # nothing padR does not while raising clipped word heads from 35 to 182.
    do_head: bool = False
    do_tail: bool = True
    do_interior: bool = True

    def _tracks(self, probs, energy_db, floor, s: float, e: float):
        i0, i1 = int(s / SILERO_HOP_SEC), max(int(s / SILERO_HOP_SEC) + 1,
                                              int(e / SILERO_HOP_SEC))
        p = probs[i0:min(i1, len(probs))]
        j0, j1 = int(s / ENERGY_HOP), max(int(s / ENERGY_HOP) + 1, int(e / ENERGY_HOP))
        j1 = min(j1, len(energy_db))
        snr = energy_db[j0:j1] - floor[j0:j1]
        return (p if len(p) else np.zeros(1, dtype=np.float32),
                snr if len(snr) else np.zeros(1, dtype=np.float32))

    def _snr_of(self, energy_db, floor, a: float, b: float) -> float:
        j0, j1 = int(a / ENERGY_HOP), max(int(a / ENERGY_HOP) + 1, int(b / ENERGY_HOP))
        j1 = min(j1, len(energy_db))
        if j1 <= j0:
            return 99.0
        return float(np.median(energy_db[j0:j1] - floor[j0:j1]))

    def _neg_runs(self, p: np.ndarray) -> List[Tuple[int, int]]:
        m = p < self.sil_neg
        out, j, n = [], 0, len(m)
        while j < n:
            if m[j]:
                k = j
                while k < n and m[k]:
                    k += 1
                out.append((j, k))
                j = k
            else:
                j += 1
        return out

    def __call__(self, speech: Sequence[Interval], probs: np.ndarray,
                 energy_db: np.ndarray, floor: np.ndarray) -> Decision:
        kept: List[Interval] = []
        st = {"ghost_dropped": 0, "ghost_sec": 0.0, "ghost_kept_loud": 0,
              "head_trim": 0, "head_sec": 0.0, "tail_trim": 0, "tail_sec": 0.0,
              "split": 0, "split_sec": 0.0, "protected_loud": 0}

        for s, e in speech:
            dur = e - s
            p, _ = self._tracks(probs, energy_db, floor, s, e)
            peak = float(p.max())

            # ---- whole interval -------------------------------------------
            if self.do_ghost and peak < self.ghost_peak:
                med_snr = self._snr_of(energy_db, floor, s, e)
                budget = (self.max_ghost_long_sec if med_snr <= self.ghost_snr_long
                          else self.max_ghost_sec)
                if med_snr <= self.ghost_snr_max and dur <= budget:
                    st["ghost_dropped"] += 1
                    st["ghost_sec"] += dur
                    continue
                st["ghost_kept_loud"] += 1
                kept.append((s, e))
                continue

            # ---- boundary and interior ------------------------------------
            lo, hi = s, e
            runs = self._neg_runs(p)
            interior: List[Tuple[float, float]] = []
            for a, b in runs:
                ra, rb = s + a * SILERO_HOP_SEC, s + b * SILERO_HOP_SEC
                touches_head = a == 0
                touches_tail = b >= len(p)
                span = rb - ra
                snr = self._snr_of(energy_db, floor, ra, rb)
                emphatic = float(p[a:b].max()) < self.sil_sure

                if touches_head and not touches_tail:
                    if not self.do_head:
                        continue
                    budget = min(self.max_head_trim, span)
                    # heads are permanent: need quiet audio AND an emphatic silero
                    if snr <= self.snr_noise and emphatic and budget > 0.02:
                        lo = min(s + budget, hi - self.min_keep_sec)
                        if lo > s:
                            st["head_trim"] += 1
                            st["head_sec"] += lo - s
                    elif snr > self.snr_protect:
                        st["protected_loud"] += 1
                elif touches_tail and not touches_head:
                    if not self.do_tail:
                        continue
                    budget = min(self.max_tail_trim, span)
                    if snr <= self.snr_noise and budget > 0.02:
                        hi = max(e - budget, lo + self.min_keep_sec)
                        if hi < e:
                            st["tail_trim"] += 1
                            st["tail_sec"] += e - hi
                    elif snr > self.snr_protect:
                        st["protected_loud"] += 1
                elif not touches_head and not touches_tail:
                    if not self.do_interior:
                        continue
                    if (span >= self.min_interior_sec
                            and snr <= self.snr_noise and emphatic):
                        interior.append((ra, rb))

            if hi - lo < self.min_keep_sec:
                kept.append((s, e))
                continue

            if not interior:
                kept.append((lo, hi))
                continue

            # split around the interior gaps
            cur = lo
            pieces: List[Interval] = []
            for ra, rb in interior:
                if ra - cur >= self.min_keep_sec and hi - rb >= self.min_keep_sec:
                    pieces.append((cur, ra))
                    cur = rb
                    st["split"] += 1
                    st["split_sec"] += rb - ra
            pieces.append((cur, hi))
            kept.extend(pieces)

        return Decision(kept=kept, stats=st)
