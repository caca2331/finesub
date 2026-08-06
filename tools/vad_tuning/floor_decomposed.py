"""One variable doing three jobs, split into three.

Production's floor is a fast-falling moving average of the frame energy, and it is
simultaneously the background estimate, the detection distance, and the dwell that
absorbs filled pauses. Measured (floor_defects.py), the entanglement costs:

  creep    the floor sits 6.6-7.7 dB higher above the true local background in dense
           speech than in sparse stretches -- more than the whole 6 dB margin, so in
           dense speech the effective threshold is background + ~13 dB and quiet
           speech falls under it
  jitter   121-238 spurious upward crossings per minute inside no-word regions,
           because a fixed +6 dB is only a couple of sigma when the background
           itself swings

Split:

  level    updated only from frames the detector currently calls quiet, so speech
           density cannot move it. Both directions, unlike the ratcheting MCRA in
           floor_lab which could only fall.
  spread   robust dispersion of those same frames. The threshold becomes
           `level + a + b*spread` -- constant false alarm rate rather than constant
           dB, which is the textbook answer to jitter.
  dwell    an additive offset that rises only while the energy sits in a *moderate*
           band above the threshold, and collapses when it is clearly loud or
           clearly quiet. This is what production got for free by letting the floor
           climb, and it is the one behaviour worth keeping: a filled pause is
           voiced, moderate, and sustained, so the offset catches up to it and the
           whole pause is absorbed -- without the background estimate ever moving.

The decision is fed back into the level update, so this is a closed loop and has to
run frame by frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None


def _kernel(e, anchor, level0, spread0, a, b, alpha_lvl_dn, alpha_lvl_up,
            alpha_spread, alpha_anchor, dwell_up, dwell_down, dwell_band,
            dwell_max, warm):
    n = len(e)
    thr_out = np.empty(n, dtype=np.float64)
    lvl_out = np.empty(n, dtype=np.float64)
    level = level0
    spread = spread0
    d = 0.0
    for i in range(n):
        base = level + a + b * spread
        thr = base + d
        thr_out[i] = thr
        lvl_out[i] = level
        snr = e[i] - level
        quiet = e[i] <= thr
        # Anchor. A decision-fed estimator can only learn from frames it already
        # believes are noise, so if it starts below the background it never sees
        # one and freezes -- which is exactly what the first run did (creep span
        # 66.8 dB, i.e. no tracking at all). The rolling minimum cannot be raised
        # by speech as long as its window contains one quiet moment, so it is a
        # safe thing to be pulled toward when the level has fallen under it.
        if level < anchor[i]:
            level += alpha_anchor * (anchor[i] - level)
        if quiet or i < warm:
            err = e[i] - level
            al = alpha_lvl_dn if err < 0.0 else alpha_lvl_up
            level += al * err
            dev = err if err >= 0.0 else -err
            spread += alpha_spread * (dev - spread)
            if spread < 0.1:
                spread = 0.1
        # dwell
        if not quiet:
            excess = e[i] - base
            if excess <= dwell_band:
                if excess > d:
                    d += dwell_up * (excess - d)
                else:
                    d += dwell_down * (excess - d)
            else:
                d += dwell_down * (0.0 - d)
        else:
            d += dwell_down * (0.0 - d)
        if d < 0.0:
            d = 0.0
        elif d > dwell_max:
            d = dwell_max
    return thr_out, lvl_out


_kernel_impl = _nb.njit(cache=True)(_kernel) if _nb is not None else _kernel


@dataclass(frozen=True)
class Decomposed:
    a: float = 6.0            # fixed part of the detection distance
    b: float = 0.0            # CFAR part: multiples of the background's own spread
    alpha_lvl_dn: float = 0.02
    alpha_lvl_up: float = 0.02
    alpha_spread: float = 0.01
    dwell_up: float = 0.006   # ~1.7 s to absorb a sustained moderate region
    dwell_down: float = 0.08
    dwell_band: float = 20.0  # louder than this over the base is speech, not dwell
    dwell_max: float = 25.0
    warm_sec: float = 2.0
    anchor_win_sec: float = 10.0   # must exceed the longest speech run without a gap
    anchor_bias: float = 3.0       # a rolling minimum is biased low; lift it back
    alpha_anchor: float = 0.05
    # A pause is a local *dip*, not an absolute level. An honest floor can only see
    # pauses that reach the background, so shallow ones -- breath, reverb tail, a
    # speaker who never goes fully quiet -- stay speech and the intervals fuse into
    # long blocks. This is the half of the old creep that was doing useful work:
    # once the floor had been lifted by the preceding speech, the test had quietly
    # become "far below what this person has just been doing". Stating it explicitly
    # gets the segmentation back without the unbounded drift, and it is bounded by
    # construction: the relative line can never rise above recent speech minus drop.
    rel_drop_db: float = 0.0       # 0 disables the relative criterion
    rel_win_sec: float = 4.0
    rel_pct: float = 90.0
    # How far over the background the relative line is ever allowed to lift the
    # threshold. Without it, a loud passage puts the cut at "recent p90 minus drop"
    # regardless of where the background is, and quiet speech right after loud
    # speech is 30 dB over the background yet still under that line -- the same
    # quiet-speech failure as before, wearing a different hat.
    rel_cap_over_level_db: float = 0.0   # 0 = unbounded

    def _relative(self, e: np.ndarray, hop: float) -> np.ndarray:
        """Recent speech level minus the drop -- the "far below what was just said"
        line. Stepped on a coarse grid; it only needs to move at speech pace."""
        grid = 0.25
        n = max(1, int(len(e) * hop / grid) + 1)
        anchors = np.arange(n) * grid
        half = self.rel_win_sec / 2.0
        q = self.rel_pct / 100.0
        vals = np.empty(n, dtype=np.float64)
        for k, a in enumerate(anchors):
            lo = max(0, int((a - half) / hop))
            hi = min(len(e), int((a + half) / hop))
            vals[k] = float(np.quantile(e[lo:hi], q)) if hi > lo else -99.0
        idx = np.clip((np.arange(len(e)) * hop / grid).astype(int), 0, n - 1)
        return vals[idx] - self.rel_drop_db

    def _anchor(self, e: np.ndarray, hop: float) -> np.ndarray:
        from floor_lab import _rolling_min_impl as _rolling_min

        w = max(1, int(self.anchor_win_sec / max(hop, 1e-9)))
        sm = np.empty_like(e)
        prev = e[0]
        al = 0.1
        for i in range(len(e)):          # light smoothing before the minimum
            prev += al * (e[i] - prev)
            sm[i] = prev
        return _rolling_min(np.ascontiguousarray(sm), w) + self.anchor_bias

    def __call__(self, energy: np.ndarray, starts: np.ndarray,
                 duration: float) -> np.ndarray:
        """Returns an effective floor: threshold - 6, so the caller's `+6 margin`
        reproduces this threshold exactly and the rest of the detector is untouched."""
        e = np.ascontiguousarray(energy, dtype=np.float64)
        finite = e[e > -99.0]
        level0 = float(np.quantile(finite if finite.size else e, 0.10))
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        anchor = self._anchor(e, hop)
        rel = self._relative(e, hop) if self.rel_drop_db > 0 else None
        thr, _lvl = _kernel_impl(
            e, anchor, level0, 1.0, float(self.a), float(self.b),
            float(self.alpha_lvl_dn), float(self.alpha_lvl_up),
            float(self.alpha_spread), float(self.alpha_anchor),
            float(self.dwell_up), float(self.dwell_down),
            float(self.dwell_band), float(self.dwell_max),
            int(self.warm_sec / max(hop, 1e-9)))
        if rel is not None:
            thr = np.maximum(thr, rel)
            if self.rel_cap_over_level_db > 0:
                _t, lvl = _kernel_impl(
                    e, anchor, level0, 1.0, float(self.a), float(self.b),
                    float(self.alpha_lvl_dn), float(self.alpha_lvl_up),
                    float(self.alpha_spread), float(self.alpha_anchor),
                    float(self.dwell_up), float(self.dwell_down),
                    float(self.dwell_band), float(self.dwell_max),
                    int(self.warm_sec / max(hop, 1e-9)))
                thr = np.minimum(thr, lvl + self.rel_cap_over_level_db)
        return thr - 6.0

    def diagnostics(self, energy: np.ndarray, starts: np.ndarray,
                    duration: float) -> Tuple[np.ndarray, np.ndarray]:
        """(threshold, level) -- the level is the background estimate proper."""
        e = np.ascontiguousarray(energy, dtype=np.float64)
        finite = e[e > -99.0]
        level0 = float(np.quantile(finite if finite.size else e, 0.10))
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        return _kernel_impl(
            e, self._anchor(e, hop), level0, 1.0, float(self.a), float(self.b),
            float(self.alpha_lvl_dn), float(self.alpha_lvl_up),
            float(self.alpha_spread), float(self.alpha_anchor),
            float(self.dwell_up), float(self.dwell_down),
            float(self.dwell_band), float(self.dwell_max),
            int(self.warm_sec / max(hop, 1e-9)))
