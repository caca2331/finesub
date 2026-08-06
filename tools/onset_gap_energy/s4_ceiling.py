"""Step 4: what is the ceiling of an energy-based fix?

An energy heuristic can only move a start forward when there is a detectable quiet
stretch between the production start and the true onset. This measures how often
that is true, independent of any particular detector.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from features import compute_tracks, noise_floor_db

FRAME = 0.01


def silence_runs(tr, t0: float, t1: float, delta_db: float, floor: float):
    i0, i1 = tr.idx(t0), tr.idx(t1) + 1
    quiet = tr.rms_db[i0:i1] <= floor + delta_db
    runs = []
    j = 0
    while j < len(quiet):
        if quiet[j]:
            k = j
            while k < len(quiet) and quiet[k]:
                k += 1
            runs.append((tr.times[i0 + j], tr.times[i0 + min(k, len(quiet) - 1)], k - j))
            j = k
        else:
            j += 1
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--delta", type=float, default=8.0)
    ap.add_argument("--min-gap", type=float, default=0.06)
    args = ap.parse_args()

    cands = load_candidates(Path(args.gold))
    tr = compute_tracks(Path(args.audio), Path(args.cache) if args.cache else None)
    need = int(round(args.min_gap / FRAME))

    print(f"delta={args.delta} dB above local floor, min gap={args.min_gap}s\n")
    print(f"{'#':>3} {'pos':<16} {'label':<13} {'err':>7} {'dB@P-floor':>10} "
          f"{'gaps in [P,G]':>13} {'longest':>8} {'exit':>7} {'exit-G':>7}")

    ceiling_ok, exits = [], []
    for c in sorted([c for c in cands if c.position in TARGET_SUBSETS], key=lambda c: c.plain_start):
        floor = noise_floor_db(tr.rms_db, tr.idx(c.plain_start))
        at_p = tr.rms_db[tr.idx(c.plain_start)] - floor
        lo, hi = min(c.plain_start, c.onset), max(c.plain_start, c.onset)
        runs = silence_runs(tr, lo, hi + 0.001, args.delta, floor) if hi > lo else []
        good = [r for r in runs if r[2] >= need]
        longest = max((r[2] * FRAME for r in good), default=0.0)
        exit_t = good[-1][1] if good else None
        ceiling_ok.append(bool(good))
        exits.append((c, exit_t))
        print(f"{c.index:>3} {c.position:<16} {c.label:<13} {c.error:>+7.3f} {at_p:>10.1f} "
              f"{len(good):>13d} {longest:>8.3f} "
              f"{(exit_t if exit_t is not None else float('nan')):>7.3f} "
              f"{((exit_t - c.onset) if exit_t is not None else float('nan')):>+7.3f}")

    n = len(ceiling_ok)
    print(f"\ncandidates with a >={args.min_gap}s quiet stretch between P and gold: "
          f"{sum(ceiling_ok)}/{n} ({sum(ceiling_ok)/n:.0%})")

    oracle = [c.onset - (t if t is not None else c.plain_start) for c, t in exits]
    print(summarize([c.error for c, _ in exits], "baseline"))
    print(summarize(oracle, "ORACLE gap-exit"))
    print("  (oracle = told where the gold is, take the last qualifying gap exit before it;")
    print("   no real detector can beat this while only using energy silences.)")


if __name__ == "__main__":
    main()
