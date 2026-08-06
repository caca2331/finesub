"""Step 9: the recommended configuration, scored end to end.

min_gap is raised to 0.12 s because s8 showed 0.06 s picks up stop-consonant closure
(36% firing on stop-initial words vs 21% on sonorant-initial; the gap closes at 0.12).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from features import compute_tracks
from gap_onset import GapOnset


def paired(base, new, tol=0.02):
    d = np.abs(np.asarray(base)) - np.abs(np.asarray(new))
    return (int((d > tol).sum()), int((np.abs(d) <= tol).sum()), int((d < -tol).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--min-gap", type=float, default=0.12)
    ap.add_argument("--max-lag", type=float, default=0.16)
    ap.add_argument("--delta", type=float, default=14.0)
    args = ap.parse_args()

    det = GapOnset(delta_db=args.delta, min_gap=args.min_gap, max_lag=args.max_lag)
    cache_dir = Path(args.cache_dir)
    gold_clip = "BV1cqLR6hEp3"
    tr = compute_tracks(Path(args.root) / gold_clip / f"{gold_clip}-vocal.flac",
                        cache_dir / f"tracks-{gold_clip}.npz")

    cands = load_candidates(Path(args.gold))
    groups = {
        "segment-boundary (n=11)": [c for c in cands if c.position == "segment-boundary"],
        "after-gap (n=10)": [c for c in cands if c.position == "after-gap"],
        "TARGET both (n=21)": [c for c in cands if c.position in TARGET_SUBSETS],
        "word_onset / already right (n=25)": [c for c in cands if c.label == "word_onset"],
        "mid-phrase (out of scope, n=40)": [c for c in cands if c.position == "mid-phrase"],
    }

    print(f"config: delta={args.delta}dB min_gap={args.min_gap} max_lag={args.max_lag} "
          f"max_move={det.max_move} window={det.window}\n")
    for name, subset in groups.items():
        base = [c.error for c in subset]
        new, fired = [], 0
        for c in subset:
            p = det(tr, c.plain_start)
            new.append(c.onset - p)
            fired += int(p != c.plain_start)
        w, t, l = paired(base, new)
        print(f"--- {name}   fired {fired}/{len(subset)}")
        print("    " + summarize(base, "baseline"))
        print("    " + summarize(new, "after fix") + f"  W/T/L={w}/{t}/{l}")
    print()

    print("=== moved rows in the target subset ===")
    for c in sorted(groups["TARGET both (n=21)"], key=lambda c: c.index):
        p = det(tr, c.plain_start)
        if p == c.plain_start:
            continue
        print(f"  #{c.index:>3} {c.position:<16} {c.label:<13} err {c.error:>+7.3f} -> "
              f"{c.onset - p:>+7.3f}   move {p - c.plain_start:.3f}s  "
              f"'{c.preceding_word}' -> '{c.next_word}'")
    print()

    print("=== blast radius across clips ===")
    tot = {"segment start": [0, []], "after-gap word": [0, []]}
    for clip in args.clips:
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        if not stable.exists() or not audio.exists():
            continue
        ctr = compute_tracks(audio, cache_dir / f"tracks-{clip}.npz")
        data = json.loads(stable.read_text(encoding="utf-8"))
        segs = data["segments"] if isinstance(data, dict) else data
        for seg in segs:
            words = seg.get("words") or []
            prev_end = None
            for wi, w in enumerate(words):
                s, e = float(w["start"]), float(w["end"])
                if wi == 0:
                    kind = "segment start"
                elif prev_end is not None and s - prev_end >= 0.05:
                    kind = "after-gap word"
                else:
                    prev_end = e
                    continue
                p = det(ctr, s)
                tot[kind][0] += 1
                if p != s:
                    tot[kind][1].append(p - s)
                prev_end = e
    for kind, (n, moves) in tot.items():
        m = np.array(moves)
        print(f"  {kind:<16} n={n:>5d} fired={m.size:>4d} ({m.size/max(n,1):>4.0%})  "
              f"move med={np.median(m) if m.size else 0:.3f} "
              f"p90={np.quantile(m,0.9) if m.size else 0:.3f} max={m.max() if m.size else 0:.3f}")


if __name__ == "__main__":
    main()
