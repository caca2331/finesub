"""Turning a per-frame decision into non-speech intervals.

Production accumulates a score: quiet frames add 1-2, speech frames subtract 0.4-4,
latch at 40, close below 0. It is a duration filter and a hysteresis in one, which is
economical but has a failure mode that matters here -- a *quiet* short word inside a
silence subtracts only 0.4/frame, so a 0.12 s one costs 4.8 of the 40 points and the
non-speech interval simply swallows it. That is precisely the class of word the
project cannot afford to lose.

Two alternatives, both with the interval logic made explicit so the duration rules
and the "what may be merged over" rule can be stated separately:

  runs      label frames, then merge and drop runs under explicit rules, with a
            guard that refuses to merge over anything that looks like real speech
  viterbi   a 2-state Viterbi over a per-frame log-likelihood ratio with transition
            penalties, then the same explicit run rules

Every scorer returns raw non-speech intervals; negative padding is applied by the
caller exactly as production does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import numba as _nb
except Exception:  # pragma: no cover
    _nb = None


def _jit(fn):
    return _nb.njit(cache=True)(fn) if _nb is not None else fn


Interval = Tuple[float, float]


def _labels(energy: np.ndarray, floor: np.ndarray, dbfs: np.ndarray,
            margin: float, abs_enter: float, abs_exit: float) -> np.ndarray:
    """0 = quiet, 1 = speech, 2 = neither (production's dead zone)."""
    quiet = (energy <= floor + margin) & (dbfs <= abs_enter)
    speech = (energy > floor + margin) | (dbfs >= abs_exit)
    out = np.full(len(energy), 2, dtype=np.int8)
    out[speech] = 1
    out[quiet] = 0
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if len(mask) == 0:
        return []
    d = np.diff(mask.astype(np.int8))
    idx = np.nonzero(d)[0] + 1
    bounds = np.concatenate(([0], idx, [len(mask)]))
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(bounds) - 1)]


def _viterbi(llr: np.ndarray, enter_cost: float, exit_cost: float) -> np.ndarray:
    """2-state Viterbi. State 1 = non-speech. llr[i] > 0 favours non-speech."""
    n = len(llr)
    out = np.empty(n, dtype=np.int8)
    if n == 0:
        return out
    back = np.empty((n, 2), dtype=np.int8)
    c0 = 0.0            # speech
    c1 = llr[0]         # non-speech
    for i in range(1, n):
        # stay vs switch, for each destination
        a = c0
        b = c1 - exit_cost
        if b > a:
            n0, back[i, 0] = b, 1
        else:
            n0, back[i, 0] = a, 0
        a = c1
        b = c0 - enter_cost
        if b > a:
            n1, back[i, 1] = b, 0
        else:
            n1, back[i, 1] = a, 1
        c0 = n0
        c1 = n1 + llr[i]
    s = 1 if c1 >= c0 else 0
    for i in range(n - 1, 0, -1):
        out[i] = s
        s = back[i, s]
    out[0] = s
    return out


_viterbi_impl = _jit(_viterbi)


def _apply_run_rules(ns: np.ndarray, snr: np.ndarray, hop: float,
                     min_non_speech_ms: float, merge_gap_ms: float,
                     merge_guard_db: float, max_merge_frames: int) -> np.ndarray:
    """Merge short speech gaps between non-speech runs, then drop short runs.

    `merge_guard_db` is the part production does not have: a speech gap is only
    merged over if it is *also* weak. A loud 60 ms syllable between two silences
    stays a syllable.
    """
    out = ns.copy()
    merge_frames = min(int(round(merge_gap_ms / 1000.0 / hop)), max_merge_frames)
    if merge_frames > 0:
        for a, b in _runs(out):
            if out[a] or b - a > merge_frames or a == 0 or b == len(out):
                continue
            if float(snr[a:b].max()) <= merge_guard_db:
                out[a:b] = True
    min_frames = int(round(min_non_speech_ms / 1000.0 / hop))
    for a, b in _runs(out):
        if out[a] and (b - a) < min_frames:
            out[a:b] = False
    return out


def _to_intervals(ns: np.ndarray, starts: np.ndarray, ends: np.ndarray,
                  duration: float) -> List[Interval]:
    out: List[Interval] = []
    for a, b in _runs(ns):
        if not ns[a]:
            continue
        s = max(0.0, min(float(starts[a]), duration))
        e = max(0.0, min(float(ends[b - 1]), duration))
        if e > s:
            out.append((s, e))
    return out


@dataclass(frozen=True)
class Scorer:
    name: str
    fn: Callable

    def __call__(self, energy, floor, dbfs, starts, ends, duration):
        return self.fn(energy, floor, dbfs, starts, ends, duration)


def production(margin: float = 6.0, merge_gap_ms: Optional[float] = None,
               min_non_speech_ms: Optional[float] = None,
               abs_enter: Optional[float] = None) -> Scorer:
    def fn(energy, floor, dbfs, starts, ends, duration):
        import torch

        from asr_playground.speech.preprocessing import energy as E
        saved = (E.MERGE_GAP_MS, E.MIN_NON_SPEECH_MS,
                 E.ABS_NON_SPEECH_MAX_DBFS_ENTER)
        try:
            if merge_gap_ms is not None:
                E.MERGE_GAP_MS = merge_gap_ms
            if min_non_speech_ms is not None:
                E.MIN_NON_SPEECH_MS = min_non_speech_ms
            if abs_enter is not None:
                E.ABS_NON_SPEECH_MAX_DBFS_ENTER = abs_enter
            return E._score_to_non_speech_intervals(
                torch.from_numpy(energy.astype(np.float32)),
                torch.from_numpy(floor.astype(np.float32)),
                torch.from_numpy(dbfs.astype(np.float32)),
                torch.from_numpy(starts.astype(np.float32)),
                torch.from_numpy(ends.astype(np.float32)),
                duration, enter_margin_db=margin, weighted=True)
        finally:
            (E.MERGE_GAP_MS, E.MIN_NON_SPEECH_MS,
             E.ABS_NON_SPEECH_MAX_DBFS_ENTER) = saved

    return Scorer("production score", fn)


def prod_minrun(margin: float = 6.0, min_speech_frames: int = 8,
                min_snr_db: float = 0.0, merge_gap_ms: Optional[float] = None,
                min_non_speech_ms: Optional[float] = None,
                abs_enter: Optional[float] = None,
                name: str = "") -> Scorer:
    """Production's accumulator, plus one hard rule it lacks.

    The accumulator prices a speech frame by how far over the floor it is, so a run
    of *quiet* speech barely dents the score: at the 0.1 floor of the speech term a
    frame costs 0.4 of 40, and a 0.12 s word costs 4.8 -- the surrounding non-speech
    interval closes over it and the word is gone. Duration is evidence the score
    never uses. This adds it: a contiguous run of speech-like frames at least this
    long ends the interval outright, whatever the score says.

    `min_snr_db` optionally requires the run to also clear the floor by that much,
    so pure noise ripple cannot trigger it.
    """
    label = name or f"prod+minrun{min_speech_frames}"


    def fn(energy, floor, dbfs, starts, ends, duration):
        from asr_playground.speech.preprocessing import energy as E

        raw = production(margin, merge_gap_ms, min_non_speech_ms, abs_enter)(
            energy, floor, dbfs, starts, ends, duration)
        if not raw:
            return raw
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        snr = energy - floor
        speech = (snr > margin) | (dbfs >= E.ABS_NON_SPEECH_MAX_DBFS_EXIT)
        if min_snr_db > 0:
            speech &= snr > min_snr_db
        out: List[Interval] = []
        for s, e in raw:
            a = max(0, int(round(s / hop)))
            b = min(len(speech), int(round(e / hop)) + 1)
            cur = s
            run = 0
            for i in range(a, b):
                if speech[i]:
                    run += 1
                    continue
                if run >= min_speech_frames:
                    # split: close the interval before the run, reopen after it
                    lo = float(starts[max(a, i - run)])
                    if lo > cur:
                        out.append((cur, lo))
                    cur = float(starts[i])
                run = 0
            if run >= min_speech_frames:
                lo = float(starts[max(a, b - run)])
                if lo > cur:
                    out.append((cur, lo))
            elif e > cur:
                out.append((cur, e))
        return [(x, y) for x, y in out if y > x]

    return Scorer(label, fn)


def runs(margin: float = 6.0, min_non_speech_ms: float = 400.0,
         merge_gap_ms: float = 100.0, merge_guard_db: float = 12.0,
         name: str = "") -> Scorer:
    label = name or f"runs guard{merge_guard_db:.0f}"

    def fn(energy, floor, dbfs, starts, ends, duration):
        from asr_playground.speech.preprocessing import energy as E

        lab = _labels(energy, floor, dbfs, margin,
                      E.ABS_NON_SPEECH_MAX_DBFS_ENTER, E.ABS_NON_SPEECH_MAX_DBFS_EXIT)
        # the dead zone carries the previous decision forward
        ns = lab == 0
        neutral = np.nonzero(lab == 2)[0]
        if neutral.size:
            prev = np.maximum.accumulate(np.where(lab != 2, np.arange(len(lab)), 0))
            ns[neutral] = ns[prev[neutral]]
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        snr = energy - floor
        ns = _apply_run_rules(ns, snr, hop, min_non_speech_ms, merge_gap_ms,
                              merge_guard_db, 10 ** 9)
        return _to_intervals(ns, starts, ends, duration)

    return Scorer(label, fn)


def viterbi(margin: float = 6.0, scale: float = 6.0, clip: float = 3.0,
            enter_cost: float = 30.0, exit_cost: float = 6.0,
            min_non_speech_ms: float = 400.0, merge_gap_ms: float = 100.0,
            merge_guard_db: float = 12.0, name: str = "") -> Scorer:
    """Globally optimal 2-state path, then the same explicit run rules.

    `enter_cost` >> `exit_cost` is the conservative asymmetry: calling something
    non-speech has to overcome a large penalty, calling it speech is cheap.
    """
    label = name or f"viterbi e{enter_cost:.0f}/x{exit_cost:.0f} s{scale:.0f}"

    def fn(energy, floor, dbfs, starts, ends, duration):
        from asr_playground.speech.preprocessing import energy as E

        snr = energy - floor
        llr = np.clip((margin - snr) / scale, -clip, clip)
        # the absolute gates enter as hard evidence, same thresholds as production
        llr[dbfs >= E.ABS_NON_SPEECH_MAX_DBFS_EXIT] = -clip
        llr[(dbfs > E.ABS_NON_SPEECH_MAX_DBFS_ENTER) & (llr > 0)] = 0.0
        path = _viterbi_impl(np.ascontiguousarray(llr, dtype=np.float64),
                             float(enter_cost), float(exit_cost))
        hop = float(starts[1] - starts[0]) if len(starts) > 1 else 0.01
        ns = _apply_run_rules(path.astype(bool), snr, hop, min_non_speech_ms,
                              merge_gap_ms, merge_guard_db, 10 ** 9)
        return _to_intervals(ns, starts, ends, duration)

    return Scorer(label, fn)
