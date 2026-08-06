"""Where the two detectors disagree, and which one is wrong.

The word references cannot answer this on their own: every word exists because the
*production* VAD kept its region, so they can show silero dropping speech but never
show production dropping speech. This looks at the disagreement regions directly.

  production says silence, silero says speech  -> candidate production miss. Nothing
      in the transcript can confirm it (a missed region produced no words), so these
      are listed with timestamps for listening.
  production says speech, silero says silence  -> already measured as silero's word
      loss in v1; here it is broken down by what silero threw away.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backends import (SILERO_HOP_SEC, Hysteresis, energy_vad, intersect, invert,
                      load_mono, silero_probs, total)
from refs import covered, load_pause_ref, load_word_srt, load_words_stable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--word-srt")
    ap.add_argument("--stable")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    audio = Path(args.audio)
    duration = len(load_mono(audio)) / 16000
    words = (load_word_srt(Path(args.word_srt)) if args.word_srt
             else load_words_stable(Path(args.stable)))

    prod = energy_vad(audio)
    prod_sil = invert(prod, duration)
    probs = silero_probs(audio, Path(args.cache_dir) / f"silero-{audio.stem}.npz")

    print(f"{audio.name}  duration={duration:.1f}s  production speech={total(prod):.1f}s "
          f"({total(prod)/duration:.1%})\n")

    print("=== production says silence, silero says speech ===")
    print("(candidate production misses; no transcript can confirm them)")
    print(f"{'silero enter':>12} {'regions':>8} {'seconds':>9} {'%dur':>6} "
          f"{'>0.5s':>6} {'max':>7}")
    hits_by_thr = {}
    for enter in (0.9, 0.7, 0.5, 0.3):
        sp = Hysteresis(enter=enter, exit=max(0.05, enter - 0.15),
                        min_speech=0.20, min_silence=0.20, pad=0.0).apply(
                            probs, SILERO_HOP_SEC, duration)
        gap = intersect(prod_sil, sp)
        gap = [(s, e) for s, e in gap if e - s >= 0.20]
        hits_by_thr[enter] = gap
        big = [g for g in gap if g[1] - g[0] > 0.5]
        print(f"{enter:>12.1f} {len(gap):>8d} {total(gap):>9.1f} "
              f"{total(gap)/duration:>6.1%} {len(big):>6d} "
              f"{(max((e-s for s,e in gap), default=0)):>7.2f}")
    print()

    focus = hits_by_thr[0.7]
    focus = sorted(focus, key=lambda g: g[0] - g[1])[:args.top]
    print(f"longest {len(focus)} regions at silero enter=0.7 (listen to these):")
    print(f"{'start':>9} {'end':>9} {'len':>6}  nearest transcript context")
    for s, e in focus:
        near = [w for w in words if w.end > s - 2.0 and w.start < e + 2.0]
        ctx = "".join(w.text for w in near)[:52]
        inside = sum(1 for w in words if w.start >= s and w.end <= e)
        print(f"{s:>9.2f} {e:>9.2f} {e-s:>6.2f}  [{inside} words inside] {ctx}")
    print()

    print("=== production says speech, silero says silence ===")
    print(f"{'silero enter':>12} {'regions':>8} {'seconds':>9} {'words fully inside':>19}")
    for enter in (0.5, 0.3, 0.2):
        sp = Hysteresis(enter=enter, exit=max(0.05, enter - 0.15), pad=0.10).apply(
            probs, SILERO_HOP_SEC, duration)
        sil = invert(sp, duration)
        gap = intersect(prod, sil)
        gap = [(s, e) for s, e in gap if e - s >= 0.20]
        lost = sum(1 for w in words
                   if (w.end > w.start) and covered(gap, w.start, w.end) / (w.end - w.start) >= 0.9)
        print(f"{enter:>12.1f} {len(gap):>8d} {total(gap):>9.1f} {lost:>19d}")


if __name__ == "__main__":
    main()
