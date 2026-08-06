"""Step 6: what would this actually do to a production run?

The gold covers 21 flagged positions. Production has ~160 segment starts per clip and
thousands of after-gap words, none of them annotated. This measures how often the rule
fires there and how far it moves things -- the blast radius, not the accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import compute_tracks
from gap_onset import GapOnset


def load_stable(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="clip ids; artifacts resolved under --root")
    ap.add_argument("--root", required=True, help="out/reference")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gap", type=float, default=0.05, help="after-gap threshold")
    ap.add_argument("--delta", type=float, default=14.0)
    ap.add_argument("--max-lag", type=float, default=0.16)
    args = ap.parse_args()

    det = GapOnset(delta_db=args.delta, max_lag=args.max_lag)
    root = Path(args.root)
    cache_dir = Path(args.cache_dir)

    totals = {"seg": [0, 0, []], "gap": [0, 0, []], "mid": [0, 0, []]}
    print(f"{'clip':<16} {'segstarts':>9} {'fire':>6} {'rate':>6} {'med move':>9} {'p90':>7} | "
          f"{'after-gap':>9} {'fire':>6} {'rate':>6} | {'mid-word':>8} {'fire':>6} {'rate':>6}")
    for clip in args.clips:
        stable = root / clip / f"{clip}-stable.json"
        audio = root / clip / f"{clip}-vocal.flac"
        if not stable.exists() or not audio.exists():
            print(f"{clip:<16} (missing artifacts)")
            continue
        tr = compute_tracks(audio, cache_dir / f"tracks-{clip}.npz")
        segs = load_stable(stable)

        buckets = {"seg": [], "gap": [], "mid": []}
        for seg in segs:
            words = seg.get("words") or []
            prev_end = None
            for wi, w in enumerate(words):
                s = float(w["start"])
                if wi == 0:
                    kind = "seg"
                elif prev_end is not None and s - prev_end >= args.gap:
                    kind = "gap"
                else:
                    kind = "mid"
                limit = float(w["end"])
                p = det(tr, s, hard_limit=limit)
                buckets[kind].append(p - s)
                prev_end = float(w["end"])

        row = [clip]
        for kind in ("seg", "gap", "mid"):
            moves = np.array(buckets[kind])
            fired = moves[moves > 0]
            totals[kind][0] += len(moves)
            totals[kind][1] += len(fired)
            totals[kind][2].extend(fired.tolist())
            row.append((len(moves), len(fired), fired))
        (n1, f1, m1), (n2, f2, m2), (n3, f3, m3) = row[1], row[2], row[3]
        print(f"{clip:<16} {n1:>9d} {f1:>6d} {f1/max(n1,1):>6.0%} "
              f"{(np.median(m1) if len(m1) else 0):>9.3f} "
              f"{(np.quantile(m1,0.9) if len(m1) else 0):>7.3f} | "
              f"{n2:>9d} {f2:>6d} {f2/max(n2,1):>6.0%} | {n3:>8d} {f3:>6d} {f3/max(n3,1):>6.0%}")

    print()
    for kind, label in (("seg", "segment starts"), ("gap", "after-gap words"), ("mid", "mid-phrase words")):
        n, f, moves = totals[kind]
        m = np.array(moves)
        if n == 0:
            continue
        print(f"{label:<18} n={n:>5d} fired={f:>4d} ({f/n:>4.0%})  "
              f"move med={np.median(m) if m.size else 0:.3f} "
              f"p90={np.quantile(m,0.9) if m.size else 0:.3f} "
              f"max={m.max() if m.size else 0:.3f}")


if __name__ == "__main__":
    main()
