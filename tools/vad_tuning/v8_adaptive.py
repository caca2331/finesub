"""Does the adaptive refinement stay out of the way on clean audio?

A rule that helps on noisy material but quietly reshapes clean material is not
usable as a default. The test is self-gating: near no-op on the reference clips
(no word lost, filled-pause rejection not degraded), decisive on miyako.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adaptive import AdaptiveRefine
from backends import SILERO_HOP_SEC, silero_probs, total
from energy_sweep import compute_tracks, speech_from_tracks
from refs import load_pause_ref, load_valid_words, load_word_srt
from score import score


def tracks_np(tr):
    return (tr.energy_db.numpy().astype(np.float64),
            tr.noise_floor.numpy().astype(np.float64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--word-srt")
    ap.add_argument("--noisy", help="extra audio path with no word reference")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    det = AdaptiveRefine()
    pause_ref = load_pause_ref(Path(args.gold))

    print("=== clean reference clips: is it a no-op? ===")
    print(f"{'clip':<16} {'kept%':>6} {'dropped':>26} {'trims':>22} {'split':>7}")
    agg = {}
    for clip in args.clips:
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        if not audio.exists():
            continue
        tr = compute_tracks(audio)
        edb, floor = tracks_np(tr)
        probs = silero_probs(audio, cache / f"silero-{clip}-vocal.npz")
        base = speech_from_tracks(tr)
        d = det(base, probs, edb, floor)
        s = d.stats
        dropped = f"{s['ghost_dropped']} ints / {s['ghost_sec']:.1f}s"
        trims = (f"H{s['head_trim']}/{s['head_sec']:.1f}s "
                 f"T{s['tail_trim']}/{s['tail_sec']:.1f}s")
        print(f"{clip:<16} {total(d.kept)/max(total(base),1e-9):>5.1%} "
              f"{dropped:>26} {trims:>22} {s['split']:>7}")
        agg[clip] = (tr, probs, base, d, edb, floor)
    print()

    if args.word_srt:
        clip = "BV1cqLR6hEp3"
        tr, probs, base, d, edb, floor = agg[clip]
        words = load_word_srt(Path(args.word_srt))
        print("=== human-checked timeline (1231 words) ===")
        print(score(base, words, tr.duration, pause_ref).line("production energy"))
        print(score(d.kept, words, tr.duration, pause_ref).line("adaptive refine"))
        print()

    print("=== valid ASR words across clips ===")
    rows_b, rows_a = [], []
    for clip, (tr, probs, base, d, edb, floor) in agg.items():
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        w, _ = load_valid_words(stable)
        rows_b.append(score(base, w, tr.duration))
        rows_a.append(score(d.kept, w, tr.duration))
    for name, rows in (("production energy", rows_b), ("adaptive refine", rows_a)):
        nw = sum(r.words for r in rows)
        rec = 1 - sum(r.word_sec_lost for r in rows) / sum(r.word_sec_total for r in rows)
        print(f"{name:<20} lost={sum(r.words_lost for r in rows):>3d}/{nw} "
              f"recall={rec:>7.3%} clipH={sum(r.clipped_head for r in rows):>3d} "
              f"clipT={sum(r.clipped_tail for r in rows):>3d} "
              f"speech={sum(r.speech_frac*r.duration for r in rows)/sum(r.duration for r in rows):>5.1%}")

    if args.noisy:
        print()
        print("=== noisy material ===")
        audio = Path(args.noisy)
        tr = compute_tracks(audio)
        edb, floor = tracks_np(tr)
        probs = silero_probs(audio, Path(args.cache_dir) / f"silero-{audio.stem}.npz")
        base = speech_from_tracks(tr)
        d = det(base, probs, edb, floor)
        print(f"  intervals {len(base)} -> {len(d.kept)}   "
              f"audio {total(base):.0f}s -> {total(d.kept):.0f}s "
              f"({total(d.kept)/total(base)-1:+.1%})")
        for k, v in d.stats.items():
            print(f"    {k}: {v:.1f}" if isinstance(v, float) else f"    {k}: {v}")


if __name__ == "__main__":
    main()
