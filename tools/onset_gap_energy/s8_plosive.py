"""Step 8: is the `immediate` bucket detecting silence, or stop-consonant closure?

A word beginning with /k t p g d b/ opens with 40-100 ms of near-silent closure. The
rule cannot tell that apart from a timestamp landing in a real pause, and moving the
start to the burst would push a *correct* start later.

Test: among kana-initial words (where the first phoneme is readable from the text),
compare the firing rate for stop-initial vs sonorant/vowel-initial words. If closure
is being picked up, stop-initial words fire far more often in the immediate bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import compute_tracks
from gap_onset import GapOnset

STOP = set("かきくけこカキクケコがぎぐげごガギグゲゴたちつてとタチツテトだぢづでどダヂヅデド"
           "ぱぴぷぺぽパピプペポばびぶべぼバビブベボっッ")
SONORANT = set("あいうえおアイウエオなにぬねのナニヌネノまみむめもマミムメモ"
               "やゆよヤユヨらりるれろラリルレロわをんワヲンーはひふへほハヒフヘホ")


def klass(word: str) -> str | None:
    w = word.strip()
    if not w:
        return None
    c = w[0]
    if c in STOP:
        return "stop-initial"
    if c in SONORANT:
        return "sonorant/vowel-initial"
    return None  # kanji or punctuation -- first phoneme unknown


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--min-gaps", nargs="+", type=float, default=[0.06, 0.08, 0.10, 0.12, 0.15])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    positions = []  # (kind, klass, tracks_key, start, end)
    tracks = {}
    for clip in args.clips:
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        if not stable.exists() or not audio.exists():
            continue
        tracks[clip] = compute_tracks(audio, cache_dir / f"tracks-{clip}.npz")
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
                positions.append((kind, klass(w.get("word", "")), clip, s, e))
                prev_end = e

    print(f"positions: {len(positions)}  kana-classifiable: "
          f"{sum(1 for p in positions if p[1])}\n")

    for mg in args.min_gaps:
        det = GapOnset(min_gap=mg)
        stats = {}
        for kind, kl, clip, s, e in positions:
            g = det.find(tracks[clip], s)
            lag = (g[0] - s) if g else 99.0
            move = (g[1] - s) if g else 0.0
            bucket = "immediate" if lag <= 0.03 else ("delayed" if lag <= det.max_lag else None)
            if bucket and move > det.max_move:
                bucket = None
            if kl:
                d = stats.setdefault((kl, "immediate"), [0, 0, []])
                d[0] += 1
                if bucket == "immediate":
                    d[1] += 1
                    d[2].append(move)
                d2 = stats.setdefault((kl, "delayed"), [0, 0, []])
                d2[0] += 1
                if bucket == "delayed":
                    d2[1] += 1
                    d2[2].append(move)
        print(f"--- min_gap={mg:.2f}")
        for bucket in ("immediate", "delayed"):
            parts = []
            for kl in ("stop-initial", "sonorant/vowel-initial"):
                n, f, m = stats.get((kl, bucket), [0, 0, []])
                med = np.median(m) if m else 0.0
                parts.append(f"{kl}: {f}/{n} = {f/max(n,1):.0%} (med move {med:.3f})")
            print(f"    {bucket:<10} " + " | ".join(parts))
    print()
    print("A large stop-vs-sonorant gap in the `immediate` bucket means the rule is")
    print("firing on closure silence, i.e. moving correct starts later.")


if __name__ == "__main__":
    main()
