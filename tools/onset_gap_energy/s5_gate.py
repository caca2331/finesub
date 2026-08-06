"""Step 5: from oracle to detector.

s4 showed the gold onset is almost exactly the end of the last quiet stretch before
the word. A real detector cannot look at the gold, so it must decide *whether* to
move from the shape of the audio right after the production start.

The discriminator tested here: how soon after the production start does the quiet
stretch begin? A start that is too early is followed by silence almost immediately;
a start that is already correct is followed by the word itself, and the next silence
only arrives after the word ends.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from features import compute_tracks, noise_floor_db

FRAME = 0.01


def first_gap(tr, s: float, window: float, delta_db: float, min_gap: float):
    """First quiet run of >= min_gap starting within [s, s+window] that is followed
    by speech. Returns (gap_start, gap_end) or None."""
    i0 = tr.idx(s)
    i1 = min(tr.idx(s + window) + 1, len(tr.times))
    if i1 - i0 < 3:
        return None
    floor = noise_floor_db(tr.rms_db, i0)
    quiet = tr.rms_db[i0:i1] <= floor + delta_db
    need = max(1, int(round(min_gap / FRAME)))
    j = 0
    n = len(quiet)
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


def predict(tr, s, window, delta_db, min_gap, max_lag, max_move):
    g = first_gap(tr, s, window, delta_db, min_gap)
    if g is None:
        return s, None
    g0, g1 = g
    if g0 - s > max_lag:          # silence arrives too late -> the word already started
        return s, ("lag", g0 - s)
    if g1 - s > max_move:         # would move further than we are willing to trust
        return s, ("move", g1 - s)
    return g1, ("fire", g1 - s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--delta", type=float, default=14.0)
    ap.add_argument("--min-gap", type=float, default=0.06)
    ap.add_argument("--window", type=float, default=1.2)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cands = load_candidates(Path(args.gold))
    tr = compute_tracks(Path(args.audio), Path(args.cache) if args.cache else None)
    target = [c for c in cands if c.position in TARGET_SUBSETS]
    safe = [c for c in cands if c.label == "word_onset"]
    safe_sb = [c for c in safe if c.position in TARGET_SUBSETS]

    print("=== how soon does the first qualifying gap start, relative to P? ===")
    print(f"{'#':>3} {'pos':<16} {'label':<13} {'err':>7} {'gapStart-P':>10} {'gapEnd-P':>9} {'gapEnd-G':>9}")
    for c in sorted(cands, key=lambda c: c.plain_start):
        if c.position not in TARGET_SUBSETS and c.label != "word_onset":
            continue
        g = first_gap(tr, c.plain_start, args.window, args.delta, args.min_gap)
        if g is None:
            print(f"{c.index:>3} {c.position:<16} {c.label:<13} {c.error:>+7.3f} {'-':>10} {'-':>9} {'-':>9}")
            continue
        g0, g1 = g
        print(f"{c.index:>3} {c.position:<16} {c.label:<13} {c.error:>+7.3f} "
              f"{g0 - c.plain_start:>10.3f} {g1 - c.plain_start:>9.3f} {g1 - c.onset:>+9.3f}")
    print()

    def run(max_lag, max_move, subset):
        errs, fired = [], 0
        for c in subset:
            p, why = predict(tr, c.plain_start, args.window, args.delta,
                             args.min_gap, max_lag, max_move)
            errs.append(c.onset - p)
            fired += int(p != c.plain_start)
        return np.array(errs), fired

    if args.sweep:
        print("=== gate sweep (delta=%.0f, min_gap=%.2f, window=%.1f) ===" % (
            args.delta, args.min_gap, args.window))
        print(f"{'max_lag':>7} {'max_move':>8} | {'tgt med':>7} {'p90':>6} {'>0.1':>5} {'fire':>5} "
              f"{'W/T/L':>10} | {'sb word_onset dmg':>18} {'all word_onset dmg':>18}")
        for max_lag, max_move in itertools.product((0.05, 0.10, 0.15, 0.25, 0.40),
                                                   (0.4, 0.6, 0.8, 1.2)):
            e, f = run(max_lag, max_move, target)
            b = np.array([c.error for c in target])
            d = np.abs(b) - np.abs(e)
            wtl = f"{int((d>0.02).sum())}/{int((np.abs(d)<=0.02).sum())}/{int((d<-0.02).sum())}"
            e2, f2 = run(max_lag, max_move, safe_sb)
            e3, f3 = run(max_lag, max_move, safe)
            print(f"{max_lag:>7.2f} {max_move:>8.2f} | {np.median(np.abs(e)):>7.3f} "
                  f"{np.quantile(np.abs(e),0.9):>6.3f} {np.mean(np.abs(e)>0.1):>5.0%} {f:>5d} "
                  f"{wtl:>10} | {f'{f2} fired med={np.median(np.abs(e2)):.3f}':>18} "
                  f"{f'{f3} fired med={np.median(np.abs(e3)):.3f}':>18}")
        print()

    max_lag, max_move = 0.15, 0.8
    e, f = run(max_lag, max_move, target)
    print(f"=== chosen gate: max_lag={max_lag} max_move={max_move} ===")
    print(summarize([c.error for c in target], "baseline (target)"))
    print(summarize(e, "gap-exit (target)"))
    e2, f2 = run(max_lag, max_move, safe)
    print(summarize(e2, "gap-exit (word_onset)") + f"   fired {f2}/{len(safe)}")


if __name__ == "__main__":
    main()
