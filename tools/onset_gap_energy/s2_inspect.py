"""Step 2: eyeball the acoustic tracks between the production start and the gold onset.

Prints one ASCII strip per target candidate: 10 ms per column, `P` marks the production
start, `G` the gold onset, `E` the disfluency block end.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates
from features import compute_tracks, noise_floor_db

RAMP = " .:-=+*#%@"


def strip(values: np.ndarray, lo: float, hi: float) -> str:
    if hi <= lo:
        return " " * len(values)
    q = np.clip((values - lo) / (hi - lo), 0, 1)
    return "".join(RAMP[int(v * (len(RAMP) - 1))] for v in q)


def marker_row(times: np.ndarray, marks: list[tuple[float, str]]) -> str:
    row = [" "] * len(times)
    for t, ch in marks:
        j = int(np.argmin(np.abs(times - t)))
        if 0 <= j < len(row):
            row[j] = ch
    return "".join(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--pre", type=float, default=0.30)
    ap.add_argument("--post", type=float, default=0.40)
    args = ap.parse_args()

    cands = [c for c in load_candidates(Path(args.gold)) if c.position in TARGET_SUBSETS]
    tr = compute_tracks(Path(args.audio), Path(args.cache) if args.cache else None)

    for c in sorted(cands, key=lambda c: c.plain_start):
        t0 = min(c.plain_start, c.onset) - args.pre
        t1 = max(c.plain_start, c.onset, c.end) + args.post
        i0, i1 = tr.slice_idx(t0, t1)
        times = tr.times[i0:i1]
        nf = noise_floor_db(tr.rms_db, tr.idx(c.plain_start))

        print(f"--- #{c.index} {c.position}/{c.label} seg{c.segment} "
              f"'{c.preceding_word}' -> '{c.next_word}'  plain={c.plain_start:.3f} "
              f"gold={c.onset:.3f} err={c.error:+.3f} blk=[{c.start:.3f},{c.end:.3f}] "
              f"gap={c.preceding_gap:.3f} floor={nf:.1f}dB")
        print("  mark " + marker_row(times, [(c.plain_start, "P"), (c.onset, "G"),
                                             (c.end, "E"), (c.start, "["), ]))
        print("  rms  " + strip(tr.rms_db[i0:i1], nf, max(nf + 5, tr.rms_db[i0:i1].max())))
        print("  lo   " + strip(tr.lo_db[i0:i1], tr.lo_db[i0:i1].min(), tr.lo_db[i0:i1].max()))
        print("  hi   " + strip(tr.hi_db[i0:i1], tr.hi_db[i0:i1].min(), tr.hi_db[i0:i1].max()))
        print("  h-l  " + strip(tr.hi_ratio[i0:i1], tr.hi_ratio[i0:i1].min(), tr.hi_ratio[i0:i1].max()))
        print("  onst " + strip(tr.onset_env[i0:i1], 0, max(1e-6, tr.onset_env[i0:i1].max())))
        print()


if __name__ == "__main__":
    main()
