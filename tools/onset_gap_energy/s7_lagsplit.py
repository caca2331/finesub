"""Step 7: split the rule by how immediate the silence is.

Two very different situations hide behind one `max_lag`:

  immediate  the timestamp itself lands in silence -- moving it to where speech
             starts is correct almost by construction
  delayed    there is real sound first, then a gap. Moving means betting that the
             sound was a filled pause and not the word. A stop consonant's closure
             also looks like this, so a wrong bet here pushes a correct start later.

This reports the gold benefit and the production blast radius for each bucket
separately, so the two can be adopted independently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import TARGET_SUBSETS, load_candidates, summarize
from features import compute_tracks
from gap_onset import GapOnset

BUCKETS = ((0.00, 0.03, "immediate"), (0.03, 0.16, "delayed"), (0.16, 9.9, "late (not fired)"))


def bucket_of(lag: float) -> str:
    for lo, hi, name in BUCKETS:
        if lo <= lag < hi:
            return name
    return "late (not fired)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    det = GapOnset()
    cache_dir = Path(args.cache_dir)
    tr = compute_tracks(Path(args.audio), cache_dir / "tracks-BV1cqLR6hEp3.npz")

    cands = load_candidates(Path(args.gold))
    target = [c for c in cands if c.position in TARGET_SUBSETS]
    safe = [c for c in cands if c.label == "word_onset"]

    print("=== gold: benefit per bucket ===")
    for lo, hi, name in BUCKETS[:2]:
        rows = []
        for c in target:
            g = det.find(tr, c.plain_start)
            lag = (g[0] - c.plain_start) if g else 99.0
            if lo <= lag < hi and (g[1] - c.plain_start) <= det.max_move:
                rows.append((c, g[1]))
        if not rows:
            print(f"{name}: no gold rows")
            continue
        base = [c.error for c, _ in rows]
        new = [c.onset - p for c, p in rows]
        print(f"-- {name}: {len(rows)} of {len(target)} target rows fire")
        print("   " + summarize(base, "baseline"))
        print("   " + summarize(new, "after fix"))
        for c, p in sorted(rows, key=lambda r: r[0].index):
            print(f"     #{c.index:>3} {c.position:<16} {c.label:<13} "
                  f"{c.error:>+7.3f} -> {c.onset - p:>+7.3f}  '{c.preceding_word}'->'{c.next_word}'")
    print()

    print("=== gold word_onset rows (production already correct): which bucket? ===")
    counts = {}
    for c in safe:
        g = det.find(tr, c.plain_start)
        lag = (g[0] - c.plain_start) if g else 99.0
        counts[bucket_of(lag)] = counts.get(bucket_of(lag), 0) + 1
    print("   " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()

    print("=== production blast radius per bucket (segment starts + after-gap words) ===")
    tally = {}
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
                g = det.find(ctr, s)
                lag = (g[0] - s) if g else 99.0
                move = (g[1] - s) if g else 0.0
                b = bucket_of(lag)
                if b != "late (not fired)" and move > det.max_move:
                    b = "late (not fired)"
                key = (kind, b)
                t = tally.setdefault(key, [0, []])
                t[0] += 1
                if b != "late (not fired)":
                    t[1].append(move)
                prev_end = e

    for kind in ("segment start", "after-gap word"):
        total = sum(v[0] for (k, _), v in tally.items() if k == kind)
        print(f"-- {kind}: n={total}")
        for _, _, name in BUCKETS:
            v = tally.get((kind, name))
            if not v:
                continue
            m = np.array(v[1])
            extra = (f" move med={np.median(m):.3f} p90={np.quantile(m,0.9):.3f} max={m.max():.3f}"
                     if m.size else "")
            print(f"     {name:<18} {v[0]:>5d} ({v[0]/max(total,1):>4.0%}){extra}")


if __name__ == "__main__":
    main()
