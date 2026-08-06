"""Noise-floor estimators, beyond the two that shipped.

Production's estimator is a slow asymmetric EMA of the raw frame energy with the
windowed percentile mixed in at 0.3%. Appendix F measured what that implies: the
window is inert, so the estimator is really "a slow follower of the signal", and the
asymmetric alphas exist to stop speech from dragging it up. That is a workaround for
the fact that it has no idea which frames are speech.

The alternatives here all come from the same observation -- a background estimate
should be built from the *quiet* frames, not from all of them:

  window       the shipped family: percentile over a long window, optionally with
               no-signal frames excluded, optionally clamping the tracker to it
  rollq        a short rolling percentile used directly, with a rise-rate limit so
               a long unbroken utterance cannot inflate it
  minstat      minimum statistics (Martin): rolling minimum of a smoothed track plus
               a bias, the standard estimator for exactly this problem
  mcra         recursive averaging that only updates on frames that already look
               like noise, so speech never enters the average at all

Every estimator returns a per-frame floor in dB on the same grid as energy_db.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from floor_variants import (DEGENERATE_DB, FloorSpec, TrackerSpec, _window_bounds,
                            floor_with_tracker)

try:
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None


def _jit(fn):
    return _nb.njit(cache=True)(fn) if _nb is not None else fn


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

def _rolling_min(x: np.ndarray, width: int) -> np.ndarray:
    """Centered rolling minimum, O(n) via a monotonic deque."""
    n = len(x)
    if width <= 1:
        return x.copy()
    half = width // 2
    out = np.empty(n, dtype=np.float64)
    dq = np.empty(n, dtype=np.int64)
    head = tail = 0          # deque of indices, increasing value
    j = 0
    for i in range(n):
        hi = min(n, i + half + 1)
        while j < hi:
            while tail > head and x[dq[tail - 1]] >= x[j]:
                tail -= 1
            dq[tail] = j
            tail += 1
            j += 1
        lo = max(0, i - half)
        while dq[head] < lo:
            head += 1
        out[i] = x[dq[head]]
    return out


_rolling_min_impl = _jit(_rolling_min)


def _ema_rise_limited(target: np.ndarray, follow: float, rise: float,
                      init: float) -> np.ndarray:
    n = len(target)
    out = np.empty(n, dtype=np.float64)
    prev = init
    for i in range(n):
        t = target[i]
        a = follow if t <= prev else rise
        prev = prev + a * (t - prev)
        out[i] = prev
    return out


_ema_rise_limited_impl = _jit(_ema_rise_limited)


def _mcra(frame: np.ndarray, alpha_up: float, alpha_dn: float, gate_db: float,
          init: float) -> np.ndarray:
    """Recursive averaging that only learns from frames that look like noise.

    A frame more than gate_db over the current estimate is speech as far as the
    estimator is concerned and is skipped entirely -- no asymmetric alpha needed to
    fight it, because it never enters the average.
    """
    n = len(frame)
    out = np.empty(n, dtype=np.float64)
    prev = init
    for i in range(n):
        f = frame[i]
        if f <= prev + gate_db:
            a = alpha_dn if f < prev else alpha_up
            prev = prev + a * (f - prev)
        out[i] = prev
    return out


_mcra_impl = _jit(_mcra)


def _win_percentile(energy: np.ndarray, starts: np.ndarray, duration: float,
                    win_sec: float, hop_sec: float, q: float,
                    exclude_below: Optional[float]) -> np.ndarray:
    """Percentile over a sliding window, evaluated on a hop grid and interpolated."""
    anchors, i0, i1 = _window_bounds(starts, duration, hop_sec, win_sec)
    finite = energy[energy > DEGENERATE_DB]
    fallback = float(np.quantile(finite if finite.size else energy, q))
    vals = np.empty(len(anchors), dtype=np.float64)
    for k, (a, b) in enumerate(zip(i0, i1)):
        seg = energy[a:b]
        if seg.size == 0:
            vals[k] = fallback
            continue
        if exclude_below is not None:
            sel = seg[seg > exclude_below]
            if sel.size:
                seg = sel
        vals[k] = float(np.quantile(seg, q))
    return np.interp(starts, anchors, vals)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Floor:
    name: str
    fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray]

    def __call__(self, energy, starts, duration):
        return self.fn(energy, starts, duration)


def shipped(exclude_db: float = -99.0, gate_db: float = -99.0,
            silent_frac_max: float = 0.15, percentile: float = 5.0,
            window_sec: float = 120.0, clamp_slack: float = 0.0,
            gate_on_exclude: bool = False, name: str = "") -> Floor:
    """The production family, with the two thresholds kept apart.

    `gate_db` decides *whether* the estimate is broken (the percentile has landed on
    "no signal") and how much of the window that no-signal is. `exclude_db` decides
    *what gets removed* once it is. Production sets both to -99; raising only the
    second is the aggressive version -- throw away everything under, say, -50 dB as
    "not background worth measuring" while still only acting where the estimate had
    actually collapsed. Folding the two together makes the sweep non-monotonic,
    because a higher threshold then also switches the rule off via the fraction gate.
    """
    from asr_playground.speech.preprocessing import energy as E

    q = percentile / 100.0
    hop = E.NOISE_LOCAL_HOP_SEC
    ts = TrackerSpec("t", blend=E.NOISE_LOCAL_BLEND, follow=E.NOISE_TRACK_FOLLOW_ALPHA,
                     rise=E.NOISE_TRACK_RISE_ALPHA, gate=E.NOISE_TRACK_GATE_DB,
                     clamp_to_target=True, clamp_slack=clamp_slack)
    label = name or f"excl{exclude_db:.0f}/gate{gate_db:.0f}"

    def fn(e, starts, duration):
        anchors, i0, i1 = _window_bounds(starts, duration, hop, window_sec)
        finite = e[e > gate_db]
        fallback = float(np.quantile(finite if finite.size else e, q))
        vals = np.empty(len(anchors), dtype=np.float64)
        for k, (a, b) in enumerate(zip(i0, i1)):
            seg = e[a:b]
            if seg.size == 0:
                vals[k] = fallback
                continue
            plain = float(np.quantile(seg, q))
            if plain <= gate_db:
                sel = seg[seg > exclude_db]
                mask_db = exclude_db if gate_on_exclude else gate_db
                frac = float((seg <= mask_db).mean())
                if sel.size and frac <= silent_frac_max:
                    plain = float(np.quantile(sel, q))
            vals[k] = plain
        target = np.interp(starts, anchors, vals)
        from floor_variants import track
        return track(e, target, ts, min(float(e[0]), fallback))

    return Floor(label, fn)


def legacy() -> Floor:
    from asr_playground.speech.preprocessing import energy as E

    fs = FloorSpec("l", percentile=E.NOISE_INIT_PERCENTILE,
                   window_sec=E.NOISE_LOCAL_WINDOW_SEC, hop_sec=E.NOISE_LOCAL_HOP_SEC)
    ts = TrackerSpec("l", blend=E.NOISE_LOCAL_BLEND, follow=E.NOISE_TRACK_FOLLOW_ALPHA,
                     rise=E.NOISE_TRACK_RISE_ALPHA, gate=E.NOISE_TRACK_GATE_DB)
    return Floor("legacy", lambda e, s, d: floor_with_tracker(e, s, d, fs, ts))


def rollq(window_sec: float = 15.0, percentile: float = 10.0,
          rise: float = 0.02, follow: float = 1.0,
          exclude_db: Optional[float] = None, name: str = "") -> Floor:
    """Short rolling percentile used directly, with the rise rate limited.

    The long window exists in production because the tracker cannot tell speech from
    background; a percentile that only looks at the last few seconds tracks the
    actual background far better, as long as something stops an unbroken utterance
    from carrying it upward. That is what `rise` is for -- and unlike production's
    alpha_rise it is the *only* job it has, so it can be set on its own merits.
    """
    q = percentile / 100.0
    label = name or f"rollq {window_sec:.0f}s p{percentile:.0f} rise{rise}"

    def fn(e, starts, duration):
        tgt = _win_percentile(e, starts, duration, window_sec, 1.0, q, exclude_db)
        return _ema_rise_limited_impl(np.ascontiguousarray(tgt), float(follow),
                                      float(rise), float(tgt[0]))

    return Floor(label, fn)


def minstat(window_sec: float = 8.0, smooth_alpha: float = 0.12,
            bias_db: float = 6.0, rise: float = 1.0, name: str = "") -> Floor:
    """Minimum statistics: rolling minimum of a smoothed track, plus a bias.

    The minimum of a smoothed power track over a window that contains at least one
    inter-word gap *is* the background, with a known downward bias that the constant
    corrects. No speech/non-speech decision is needed anywhere.
    """
    label = name or f"minstat {window_sec:.0f}s bias{bias_db:.0f}"

    def fn(e, starts, duration):
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        width = max(1, int(round(window_sec / hop)))
        sm = _ema_rise_limited_impl(np.ascontiguousarray(e), float(smooth_alpha),
                                    float(smooth_alpha), float(e[0]))
        mn = _rolling_min_impl(np.ascontiguousarray(sm), width) + bias_db
        if rise >= 1.0:
            return mn
        return _ema_rise_limited_impl(np.ascontiguousarray(mn), 1.0, float(rise),
                                      float(mn[0]))

    return Floor(label, fn)


def mcra(gate_db: float = 6.0, alpha_up: float = 0.002, alpha_dn: float = 0.05,
         name: str = "") -> Floor:
    label = name or f"mcra gate{gate_db:.0f} up{alpha_up} dn{alpha_dn}"

    def fn(e, starts, duration):
        finite = e[e > DEGENERATE_DB]
        init = float(np.quantile(finite if finite.size else e, 0.05))
        return _mcra_impl(np.ascontiguousarray(e), float(alpha_up), float(alpha_dn),
                          float(gate_db), init)

    return Floor(label, fn)


def combine(a: Floor, b: Floor, how: str = "min", name: str = "") -> Floor:
    """min = the more permissive of two estimates (safer for recall)."""
    op = np.minimum if how == "min" else np.maximum
    return Floor(name or f"{how}({a.name}, {b.name})",
                 lambda e, s, d: op(a(e, s, d), b(e, s, d)))
