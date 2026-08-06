"""Re-derive adaptive's ghost_snr_max against whatever floor is in production.

The threshold is a distance above the noise floor, so it means nothing without the
floor it was measured against -- change the estimator and it has to be re-measured.
This lists every ghost interval on the clean clips with the valid words inside it, so
the cut can be placed where it drops noise and not sentences.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC, silero_probs  # noqa: E402
from energy_sweep import compute_tracks, speech_from_tracks  # noqa: E402
from refs import load_valid_words  # noqa: E402

ENERGY_HOP = 0.01


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--noisy")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    rows = []
    for clip in args.clips:
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        if not audio.exists():
            continue
        tr = compute_tracks(audio)
        edb = tr.energy_db.numpy().astype(np.float64)
        floor = tr.noise_floor.numpy().astype(np.float64)
        probs = silero_probs(audio, cache / f"silero-{clip}-vocal.npz")
        words, _ = load_valid_words(Path(args.root) / clip / f"{clip}-stable.json")
        for s, e in speech_from_tracks(tr):
            i0 = int(s / SILERO_HOP_SEC)
            i1 = max(i0 + 1, min(int(e / SILERO_HOP_SEC), len(probs)))
            if float(probs[i0:i1].max()) >= 0.5:
                continue
            j0, j1 = int(s / ENERGY_HOP), max(int(s / ENERGY_HOP) + 1,
                                              min(int(e / ENERGY_HOP), len(edb)))
            snr = float(np.median(edb[j0:j1] - floor[j0:j1]))
            txt = "".join(w.text for w in words
                          if w.start < e and w.end > s).strip()
            rows.append((clip, s, e - s, snr, txt))

    withtext = [r for r in rows if r[4]]
    print(f"{len(rows)} ghost intervals on clean clips, {len(withtext)} carry text")
    print(f"{'clip':<15} {'t':>8} {'dur':>5} {'snr':>6}  text")
    for r in sorted(withtext, key=lambda x: x[3]):
        print(f"{r[0]:<15} {r[1]:>8.2f} {r[2]:>5.2f} {r[3]:>6.1f}  {r[4][:40]}")
    empty = np.array([r[3] for r in rows if not r[4]])
    print(f"\nempty ghosts: median {np.median(empty):.1f} p75 {np.quantile(empty, .75):.1f} "
          f"p90 {np.quantile(empty, .90):.1f}")
    for cut in (15, 20, 25, 30, 35):
        drop_t = sum(1 for r in withtext if r[3] <= cut)
        drop_e = int((empty <= cut).sum())
        print(f"  snr_max {cut:>2}: drops {drop_t} text-bearing, {drop_e}/{len(empty)} empty")

    if args.noisy:
        audio = Path(args.noisy)
        tr = compute_tracks(audio)
        edb = tr.energy_db.numpy().astype(np.float64)
        floor = tr.noise_floor.numpy().astype(np.float64)
        probs = silero_probs(audio, cache / f"silero-{audio.stem}.npz")
        snrs = []
        for s, e in speech_from_tracks(tr):
            i0 = int(s / SILERO_HOP_SEC)
            i1 = max(i0 + 1, min(int(e / SILERO_HOP_SEC), len(probs)))
            if float(probs[i0:i1].max()) >= 0.5:
                continue
            j0, j1 = int(s / ENERGY_HOP), max(int(s / ENERGY_HOP) + 1,
                                              min(int(e / ENERGY_HOP), len(edb)))
            snrs.append(float(np.median(edb[j0:j1] - floor[j0:j1])))
        a = np.array(snrs)
        print(f"\nnoisy: {len(a)} ghosts, median {np.median(a):.1f}")
        for cut in (15, 20, 25, 30, 35):
            print(f"  snr_max {cut:>2}: drops {(a <= cut).sum()}")


if __name__ == "__main__":
    main()
