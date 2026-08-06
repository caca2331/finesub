"""Alternative noise-floor estimators for the energy VAD.

The production floor is a 5th percentile over every frame in a 120 s window. On
separated vocals that is the wrong statistic: the separator emits true digital
silence (energy_db clamped to -100) over a large share of the file -- 10%+ on
miyako -- so the percentile latches onto "no signal at all" instead of "the
background between words". `floor + 6 dB` then sits 15-33 dB below the residual
noise and cannot reject any of it.

The floor should describe background noise. Frames with no signal are not
background noise, they are the separator's output, and including them is what
breaks the estimate. Each variant below changes only which frames the percentile
sees; the tracker that turns per-window targets into a per-frame floor is
production's, unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

DEGENERATE_DB = -99.0     # energy_db is clamped at -100 by DB_EPS


def _window_bounds(frame_starts: np.ndarray, duration: float,
                   hop_sec: float, win_sec: float):
    anchor_count = max(1, int(math.floor(duration / hop_sec)) + 1)
    anchors = np.arange(anchor_count, dtype=np.float64) * hop_sec
    anchors = np.clip(anchors, 0.0, duration)
    half = win_sec / 2.0
    if duration <= win_sec:
        ws = np.zeros_like(anchors)
        we = np.full_like(anchors, duration)
    else:
        ws = anchors - half
        we = anchors + half
        left = anchors < half
        right = anchors > (duration - half)
        ws[left], we[left] = 0.0, win_sec
        ws[right], we[right] = duration - win_sec, duration
    i0 = np.searchsorted(frame_starts, ws)
    i1 = np.searchsorted(frame_starts, we, side="right")
    return anchors, i0, i1


@dataclass(frozen=True)
class FloorSpec:
    name: str
    percentile: float = 5.0
    drop_degenerate: bool = False      # ignore frames with no signal at all
    # "always" drops them unconditionally (shipped rule 1); "if_degenerate" only when
    # the plain percentile has itself landed on the clamp, which makes the rule a
    # repair of the broken case and a no-op everywhere else; "if_sparse" additionally
    # refuses to act on windows that are mostly silence, where the separator really
    # has cleared the background and a low floor is the correct answer.
    drop_when: str = "always"
    max_silent_frac: float = 0.30
    dynamic_range_db: Optional[float] = None  # ignore frames below (win p99 - this)
    # Ceiling on the target, expressed as dB below the window's loud level (p90 of
    # the frames the percentile sees). Dropping digital silence makes the percentile
    # climb, and in a window that is almost all speech it climbs *into* speech --
    # exactly the case where nothing is left to average but voices. This caps how
    # close to the loud level the floor is ever allowed to claim the background sits.
    cap_below_loud_db: Optional[float] = None
    window_sec: float = 120.0
    hop_sec: float = 1.0

    def _should_drop(self, seg: np.ndarray, q: float) -> bool:
        if self.drop_when == "always":
            return True
        if float(np.quantile(seg, q)) > DEGENERATE_DB:
            return False        # the estimate was never broken here
        if self.drop_when == "if_sparse":
            return float((seg <= DEGENERATE_DB).mean()) <= self.max_silent_frac
        return True

    def targets(self, energy_db: np.ndarray, frame_starts: np.ndarray,
                duration: float) -> np.ndarray:
        anchors, i0, i1 = _window_bounds(frame_starts, duration,
                                         self.hop_sec, self.window_sec)
        q = max(0.0, min(1.0, self.percentile / 100.0))
        finite = energy_db[energy_db > DEGENERATE_DB]
        global_floor = float(np.quantile(finite if finite.size else energy_db, q))
        out = np.empty(len(anchors), dtype=np.float64)
        for k, (a, b) in enumerate(zip(i0, i1)):
            seg = energy_db[a:b]
            if seg.size == 0:
                out[k] = global_floor
                continue
            sel = seg
            if self.drop_degenerate and self._should_drop(seg, q):
                sel = sel[sel > DEGENERATE_DB]
            if self.dynamic_range_db is not None and sel.size:
                sel = sel[sel >= np.quantile(seg, 0.99) - self.dynamic_range_db]
            if sel.size == 0:
                out[k] = global_floor
                continue
            val = float(np.quantile(sel, q))
            if self.cap_below_loud_db is not None:
                val = min(val, float(np.quantile(sel, 0.90)) - self.cap_below_loud_db)
            out[k] = val
        return out


def floor_from_targets(energy_db: np.ndarray, frame_starts: np.ndarray,
                       duration: float, spec: FloorSpec,
                       *, gate: float, follow: float, rise: float,
                       blend: float) -> np.ndarray:
    """Run production's tracker over a replacement set of per-window targets."""
    from asr_playground.speech.preprocessing import energy as E

    anchors, _i0, _i1 = _window_bounds(frame_starts, duration,
                                       spec.hop_sec, spec.window_sec)
    targets = spec.targets(energy_db, frame_starts, duration)
    target_floor = np.interp(frame_starts, anchors, targets)

    finite = energy_db[energy_db > DEGENERATE_DB]
    gf = float(np.quantile(finite if finite.size else energy_db,
                           spec.percentile / 100.0))
    floor0 = min(float(energy_db[0]), gf)
    out = E._noise_floor_track_numba(
        energy_db.astype(np.float32),
        target_floor.astype(np.float32),
        np.float32(floor0), np.float32(gate),
        np.float32(follow), np.float32(rise), np.float32(blend),
    )
    return np.asarray(out, dtype=np.float64)


VARIANTS = [
    FloorSpec("production (p5, all frames)"),
    FloorSpec("p5, drop digital silence", drop_degenerate=True),
    FloorSpec("p10, drop digital silence", percentile=10.0, drop_degenerate=True),
    FloorSpec("p20, drop digital silence", percentile=20.0, drop_degenerate=True),
    FloorSpec("p5, 70dB dynamic range", dynamic_range_db=70.0),
    FloorSpec("p5, 60dB dynamic range", dynamic_range_db=60.0),
    FloorSpec("p10 only (keep silence)", percentile=10.0),
    FloorSpec("p5 drop-sil, 30s window", drop_degenerate=True, window_sec=30.0),
]


# ---------------------------------------------------------------------------
# Tracker variants
#
# Production: cur = blend*frame + (1-blend)*target, with blend = 0.997, then
# follow down at 0.08 and rise at 0.005. So the floor is driven almost entirely
# by the raw frame energy and the windowed percentile contributes 0.3%. On
# separated audio the frame energy drops to digital silence often, the tracker
# follows it down fast, and then needs several seconds to climb back -- longer
# than the gaps between utterances. The floor therefore sits far under the real
# background for most of the file.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrackerSpec:
    name: str
    blend: float = 0.997
    follow: float = 0.08
    rise: float = 0.005
    gate: float = 6.0
    clamp_to_target: bool = False   # windowed percentile acts as a lower bound
    hold_on_silence: bool = False   # do not follow into "no signal at all"
    # dB the tracker is still allowed to sit below the windowed percentile. 0 is the
    # shipped hard clamp; a few dB of slack keeps some downward adaptation for a
    # passage that really is quieter than its 120 s neighbourhood.
    clamp_slack: float = 0.0


try:  # optional, only to keep multi-hour sweeps tolerable
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None


def _track_py(energy_db, target_floor, blend, follow, rise, gate,
              clamp, slack, hold, floor0):
    n = len(energy_db)
    out = np.empty(n, dtype=np.float64)
    out[0] = floor0
    omb = 1.0 - blend
    for i in range(1, n):
        prev = out[i - 1]
        f = energy_db[i]
        if hold and f <= DEGENERATE_DB:
            out[i] = prev
            continue
        cur = blend * f + omb * target_floor[i]
        if clamp:
            lo = target_floor[i] - slack
            if cur < lo:
                cur = lo
        alpha = follow if cur <= (prev + gate) else rise
        out[i] = prev + alpha * (cur - prev)
    return out


_track_impl = _nb.njit(cache=True)(_track_py) if _nb is not None else _track_py


def track(energy_db: np.ndarray, target_floor: np.ndarray, spec: TrackerSpec,
          floor0: float) -> np.ndarray:
    return _track_impl(np.ascontiguousarray(energy_db, dtype=np.float64),
                       np.ascontiguousarray(target_floor, dtype=np.float64),
                       float(spec.blend), float(spec.follow), float(spec.rise),
                       float(spec.gate), bool(spec.clamp_to_target),
                       float(spec.clamp_slack), bool(spec.hold_on_silence),
                       float(floor0))


def floor_with_tracker(energy_db, frame_starts, duration, fspec: FloorSpec,
                       tspec: TrackerSpec) -> np.ndarray:
    anchors, _i0, _i1 = _window_bounds(frame_starts, duration,
                                       fspec.hop_sec, fspec.window_sec)
    targets = fspec.targets(energy_db, frame_starts, duration)
    target_floor = np.interp(frame_starts, anchors, targets)
    finite = energy_db[energy_db > DEGENERATE_DB]
    gf = float(np.quantile(finite if finite.size else energy_db, fspec.percentile / 100.0))
    return track(energy_db, target_floor, tspec, min(float(energy_db[0]), gf))


TRACKERS = [
    TrackerSpec("production"),
    TrackerSpec("hold on digital silence", hold_on_silence=True),
    TrackerSpec("percentile as lower bound", clamp_to_target=True),
    TrackerSpec("both", clamp_to_target=True, hold_on_silence=True),
    TrackerSpec("faster rise 0.02", rise=0.02),
    TrackerSpec("faster rise 0.05", rise=0.05),
    TrackerSpec("blend 0.9", blend=0.90),
    TrackerSpec("blend 0.9 + lower bound", blend=0.90, clamp_to_target=True),
]
