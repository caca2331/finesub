"""Step 10: the rows the silence rule cannot reach.

Most remaining segment-boundary errors are *voiced* filled pauses -- there is no quiet
stretch to find, so energy is blind. The only remaining acoustic handle is that a
filled pause is spectrally steady while a real word onset is not. This tests a
second tier: where the silence rule did not fire, move to the strongest spectral
change instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import load_candidates, summarize
from features import compute_tracks
from gap_onset import GapOnset


def spectral_change_peak(tr, s: float, window: float, z: float, min_move: float) -> float:
    i0 = tr.idx(s)
    i1 = min(tr.idx(s + window) + 1, len(tr.times))
    if i1 - i0 < 5:
        return s
    ctx0, ctx1 = max(0, i0 - 400), min(len(tr.flux), i1 + 400)
    ctx = tr.flux[ctx0:ctx1]
    thr = ctx.mean() + z * ctx.std()
    env = tr.flux[i0:i1]
    hits = np.flatnonzero(env > thr)
    hits = hits[hits * 0.01 >= min_move]
    if hits.size == 0:
        return s
    return float(tr.times[i0 + int(hits[0])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    det = GapOnset(min_gap=0.12)
    tr = compute_tracks(Path(args.audio), Path(args.cache))
    cands = load_candidates(Path(args.gold))

    unreached = [c for c in cands if c.position == "segment-boundary" and det(tr, c.plain_start) == c.plain_start]
    safe = [c for c in cands if c.label == "word_onset" and det(tr, c.plain_start) == c.plain_start]

    print(f"segment-boundary rows the silence rule leaves untouched: {len(unreached)}")
    for c in sorted(unreached, key=lambda c: c.index):
        print(f"  #{c.index:>3} {c.label:<13} err {c.error:>+7.3f} "
              f"blk={c.duration:.3f}s '{c.preceding_word}' -> '{c.next_word}'")
    print()

    print("tier-2 = first spectral-flux peak above mean + z*sd, at least min_move after the start")
    print(f"{'z':>4} {'min_move':>8} | {'unreached med':>13} {'p90':>6} {'W/T/L':>9} | "
          f"{'word_onset damage W/T/L':>24}")
    for z in (1.5, 2.0, 2.5, 3.0):
        for min_move in (0.08, 0.15):
            new = [c.onset - spectral_change_peak(tr, c.plain_start, 0.8, z, min_move) for c in unreached]
            base = [c.error for c in unreached]
            d = np.abs(base) - np.abs(new)
            wtl = f"{int((d>0.02).sum())}/{int((np.abs(d)<=0.02).sum())}/{int((d<-0.02).sum())}"
            dn = [c.onset - spectral_change_peak(tr, c.plain_start, 0.8, z, min_move) for c in safe]
            db = [c.error for c in safe]
            dd = np.abs(db) - np.abs(dn)
            wtl2 = f"{int((dd>0.02).sum())}/{int((np.abs(dd)<=0.02).sum())}/{int((dd<-0.02).sum())}"
            a = np.abs(new)
            print(f"{z:>4.1f} {min_move:>8.2f} | {np.median(a):>13.3f} {np.quantile(a,0.9):>6.3f} "
                  f"{wtl:>9} | {wtl2:>24}")
    print()
    print(summarize([c.error for c in unreached], "unreached baseline"))


if __name__ == "__main__":
    main()
