"""Step 11: two rules that judge the candidate position itself.

s5 gated on *when silence arrives after* the start. Two more natural-sounding gates
judge the start point directly:

  low-energy-at-P   the timestamp itself sits in a quiet frame -> snap forward to
                    where speech resumes.
  tight-VAD-head    rebuild a tight speech/silence segmentation (no conservative
                    padding or merging) and ask whether the word start lies *before*
                    the head of the speech region it belongs to. If it does, the
                    start is covering silence and belongs at the head.

Both are tested on their own and combined with GapOnset, on the same gold rows and
the same 25-row damage set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from features import compute_tracks, noise_floor_db
from gap_onset import GapOnset

FRAME = 0.01


def tight_vad(tr, delta_db: float, min_speech: float, min_sil: float,
              floor_halfwidth: int = 500) -> List[Tuple[float, float]]:
    """Speech regions from the frame energy track, with no padding and no merging
    beyond the stated minimum durations. Deliberately tighter than production VAD."""
    floors = np.empty_like(tr.rms_db)
    step = 200
    for i in range(0, len(tr.rms_db), step):
        f = noise_floor_db(tr.rms_db, i + step // 2, floor_halfwidth)
        floors[i:i + step] = f
    mask = tr.rms_db > floors + delta_db

    need_sp = max(1, int(round(min_speech / FRAME)))
    need_si = max(1, int(round(min_sil / FRAME)))

    # close short silences, then drop short speech blips
    regions: List[List[int]] = []
    j, n = 0, len(mask)
    while j < n:
        if mask[j]:
            k = j
            while k < n and mask[k]:
                k += 1
            regions.append([j, k])
            j = k
        else:
            j += 1
    merged: List[List[int]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < need_si:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    kept = [r for r in merged if r[1] - r[0] >= need_sp]
    return [(float(tr.times[a]), float(tr.times[min(b, n - 1)])) for a, b in kept]


def head_snap(vad: List[Tuple[float, float]], s: float, max_move: float) -> float:
    """If s falls before the head of the next speech region, move it to that head."""
    for a, b in vad:
        if b <= s:
            continue
        if a <= s < b:
            return s               # already inside speech
        if a - s <= max_move:      # s is in silence, region starts soon
            return a
        return s
    return s


def low_energy_snap(tr, vad, s: float, delta_db: float, max_move: float) -> float:
    floor = noise_floor_db(tr.rms_db, tr.idx(s))
    if tr.rms_db[tr.idx(s)] > floor + delta_db:
        return s                   # start is loud -> rule declines
    return head_snap(vad, s, max_move)


def paired(base, new, tol=0.02):
    d = np.abs(np.asarray(base)) - np.abs(np.asarray(new))
    return f"{int((d>tol).sum())}/{int((np.abs(d)<=tol).sum())}/{int((d<-tol).sum())}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", required=True)
    args = ap.parse_args()

    tr = compute_tracks(Path(args.audio), Path(args.cache))
    cands = load_candidates(Path(args.gold))
    target = [c for c in cands if c.position in TARGET_SUBSETS]
    safe = [c for c in cands if c.label == "word_onset"]
    gap_det = GapOnset(min_gap=0.12)

    print("=== is the production start itself low-energy? (dB above local floor) ===")
    vals = [(c, tr.rms_db[tr.idx(c.plain_start)] - noise_floor_db(tr.rms_db, tr.idx(c.plain_start)))
            for c in target]
    for thr in (3, 6, 10, 14):
        hit = [c for c, v in vals if v <= thr]
        big = [c for c in hit if abs(c.error) > 0.1]
        miss = [c for c, v in vals if v > thr and abs(c.error) > 0.1]
        print(f"  start quieter than floor+{thr:>2}dB: {len(hit):>2}/21 rows, "
              f"{len(big):>2} of them really need a fix; "
              f"{len(miss):>2} rows needing a fix are MISSED (start is loud)")
    print()

    print("=== tight VAD sweep: head-snap only ===")
    print(f"{'delta':>5} {'minSp':>6} {'minSil':>6} | {'target med':>10} {'p90':>6} {'fire':>5} "
          f"{'W/T/L':>9} | {'damage W/T/L':>12} {'regions':>8}")
    best = None
    for delta in (6, 10, 14):
        for min_sp in (0.05, 0.10):
            for min_sil in (0.06, 0.12):
                vad = tight_vad(tr, delta, min_sp, min_sil)
                new = [c.onset - head_snap(vad, c.plain_start, 0.8) for c in target]
                fire = sum(1 for c in target if head_snap(vad, c.plain_start, 0.8) != c.plain_start)
                dmg = [c.onset - head_snap(vad, c.plain_start, 0.8) for c in safe]
                a = np.abs(new)
                print(f"{delta:>5} {min_sp:>6.2f} {min_sil:>6.2f} | {np.median(a):>10.3f} "
                      f"{np.quantile(a,0.9):>6.3f} {fire:>5d} "
                      f"{paired([c.error for c in target], new):>9} | "
                      f"{paired([c.error for c in safe], dmg):>12} {len(vad):>8d}")
                if best is None or np.median(a) < best[0]:
                    best = (np.median(a), delta, min_sp, min_sil, vad)
    print()

    _, bd, bsp, bsi, bvad = best
    print(f"=== head-on comparison (tight VAD: delta={bd} minSp={bsp} minSil={bsi}) ===")
    rules = {
        "baseline": lambda c: c.plain_start,
        "low-energy-at-P (<=6dB)": lambda c: low_energy_snap(tr, bvad, c.plain_start, 6, 0.8),
        "tight-VAD head-snap": lambda c: head_snap(bvad, c.plain_start, 0.8),
        "GapOnset (recommended)": lambda c: gap_det(tr, c.plain_start),
        "head-snap OR GapOnset": lambda c: (head_snap(bvad, c.plain_start, 0.8)
                                            if head_snap(bvad, c.plain_start, 0.8) != c.plain_start
                                            else gap_det(tr, c.plain_start)),
    }
    for name, fn in rules.items():
        for label, subset in (("target", target), ("damage", safe)):
            new = [c.onset - fn(c) for c in subset]
            fire = sum(1 for c in subset if fn(c) != c.plain_start)
            tag = f"{name} [{label}]"
            line = summarize(new, tag)
            if name != "baseline":
                line += f"  W/T/L={paired([c.error for c in subset], new)} fired={fire}"
            print(line)
        print()

    print("=== which target rows does each rule reach? ===")
    print(f"{'#':>3} {'pos':<16} {'err':>7} | {'headsnap':>9} {'gaponset':>9}")
    for c in sorted(target, key=lambda c: c.index):
        h = head_snap(bvad, c.plain_start, 0.8)
        g = gap_det(tr, c.plain_start)
        print(f"{c.index:>3} {c.position:<16} {c.error:>+7.3f} | "
              f"{(c.onset - h if h != c.plain_start else float('nan')):>+9.3f} "
              f"{(c.onset - g if g != c.plain_start else float('nan')):>+9.3f}")


if __name__ == "__main__":
    main()
