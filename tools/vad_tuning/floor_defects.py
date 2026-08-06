"""Measure the two defects of a frame-driven floor, so a replacement can be judged.

The user's diagnosis, restated as two things a background estimate should satisfy
and this one does not:

  creep   A background estimate must not depend on how much speech is happening.
          Production's rises during speech (alpha_rise is small but not zero), so on
          material with a high background *and* long dense talking the threshold
          `floor + margin` drifts upward and quiet speech falls under it.
          Metric: median floor inside no-word regions, binned by how dense the
          speech around them is. A real background estimate is flat across bins;
          creep shows up as a positive slope.

  jitter  A fixed +6 dB margin has a different false-alarm rate depending on how
          variable the background is. Where the noise itself swings a few dB, the
          floor hugs its bottom (follow 0.08) and every ripple clears the margin.
          Metric: upward crossings of the threshold per minute *inside* no-word
          regions, plus the share of no-word seconds spent above it. Both should be
          near zero for a detector that is not reacting to its own noise.

"No-word regions" come from the union of several ASR runs, the same speech-presence
map the precision metrics use -- a region no configuration ever decoded is not
speech, so anything happening there is the detector arguing with the background.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

HOP = 0.01


def word_mask(spans: Sequence[Tuple[float, float]], n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        a = min(max(int(s / HOP), 0), n)
        b = min(max(int(e / HOP) + 1, a), n)
        m[a:b] = True
    return m


def speech_density(mask: np.ndarray, win_sec: float = 15.0) -> np.ndarray:
    """Fraction of the surrounding window covered by words, per frame."""
    w = max(1, int(win_sec / HOP))
    kernel = np.ones(w, dtype=np.float64)
    pad = w // 2
    padded = np.concatenate([np.zeros(pad), mask.astype(np.float64), np.zeros(w - pad)])
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    out = (csum[w:w + len(mask)] - csum[:len(mask)]) / float(w)
    return out


def local_background(energy: np.ndarray, mask: np.ndarray, win_sec: float = 15.0,
                     pct: float = 20.0) -> np.ndarray:
    """A reference background: low percentile of the *no-word* energy nearby.

    Needed because the raw floor level is not comparable across density bins -- a
    stretch with little speech often genuinely has a lower background (after
    separation, frequently digital silence). Subtracting this isolates how far the
    floor sits above the background it is supposed to be estimating, which is the
    quantity that must not depend on speech density.
    """
    hop = 1.0
    n = max(1, int(len(energy) * HOP / hop) + 1)
    anchors = np.arange(n) * hop
    half = win_sec / 2.0
    t = np.arange(len(energy)) * HOP
    vals = np.full(n, np.nan)
    q = pct / 100.0
    for k, a in enumerate(anchors):
        lo = max(0, int((a - half) / HOP))
        hi = min(len(energy), int((a + half) / HOP))
        seg = energy[lo:hi][~mask[lo:hi]]
        if seg.size > 50:
            vals[k] = float(np.quantile(seg, q))
    ok = ~np.isnan(vals)
    if ok.sum() < 2:
        return np.full(len(energy), float(np.quantile(energy, q)))
    return np.interp(t, anchors[ok], vals[ok])


def creep(floor: np.ndarray, mask: np.ndarray, dens: np.ndarray,
          background: np.ndarray,
          bins: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.01)) -> List[float]:
    """Median (floor - local background) in no-word frames, per density bin."""
    quiet = ~mask
    out: List[float] = []
    for lo, hi in zip(bins, bins[1:]):
        sel = quiet & (dens >= lo) & (dens < hi)
        out.append(float(np.median((floor - background)[sel]))
                   if sel.sum() > 1000 else float("nan"))
    return out


def jitter(energy: np.ndarray, floor: np.ndarray, mask: np.ndarray,
           margin: float = 6.0) -> Tuple[float, float]:
    """(upward crossings per minute, share of seconds above) inside no-word frames."""
    quiet = ~mask
    over = energy > (floor + margin)
    rise = np.zeros(len(over), dtype=bool)
    rise[1:] = over[1:] & ~over[:-1]
    n_quiet = int(quiet.sum())
    if n_quiet == 0:
        return 0.0, 0.0
    minutes = n_quiet * HOP / 60.0
    return float((rise & quiet).sum()) / max(minutes, 1e-9), float(over[quiet].mean())


def report(name: str, energy: np.ndarray, floor: np.ndarray,
           spans: Sequence[Tuple[float, float]], margin: float = 6.0) -> Dict:
    n = len(energy)
    m = word_mask(spans, n)
    d = speech_density(m)
    bg = local_background(energy, m)
    c = creep(floor, m, d, bg)
    cross, share = jitter(energy, floor, m, margin)
    finite = [x for x in c if not np.isnan(x)]
    return {"clip": name, "creep": c,
            "creep_span": (max(finite) - min(finite)) if len(finite) > 1 else 0.0,
            "cross_per_min": cross, "over_share": share}
