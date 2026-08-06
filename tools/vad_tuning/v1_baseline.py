"""Baseline: production energy VAD vs silero, on the one clip with a human timeline.

Establishes the trade the user described -- silero rejects filled pauses better but
drops speech -- as numbers, before anything is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backends import (SILERO_HOP_SEC, Hysteresis, energy_vad, load_mono,
                      silero_probs, total)
from refs import load_pause_ref, load_word_srt, load_words_stable
from score import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--word-srt", help="hand-corrected word SRT (BV1cqLR6hEp3)")
    ap.add_argument("--stable", help="ASR stable.json (circular reference)")
    ap.add_argument("--gold", help="disfluency_gold.json")
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    audio = Path(args.audio)
    cache = Path(args.cache_dir)
    duration = len(load_mono(audio)) / 16000

    words = (load_word_srt(Path(args.word_srt)) if args.word_srt
             else load_words_stable(Path(args.stable)))
    pause_ref = load_pause_ref(Path(args.gold)) if args.gold else None
    print(f"audio {audio.name}  duration={duration:.1f}s  words={len(words)}")
    if pause_ref:
        print(f"pause reference: {len(pause_ref.filled_pause)} filled_pause, "
              f"{len(pause_ref.word_onset)} word_onset blocks")
    print()
    print("lost = word >=90% inside non-speech (subtitle disappears)")
    print("pause_excl / onset_excl = block >=50% inside non-speech (higher pause is good,"
          " higher onset is bad)")
    print()

    prod = energy_vad(audio)
    print(score(prod, words, duration, pause_ref).line("production energy"))

    probs = silero_probs(audio, cache / f"silero-{audio.stem}.npz")
    print()
    for enter, exit_, pad in ((0.5, 0.35, 0.0), (0.5, 0.35, 0.10), (0.5, 0.35, 0.20),
                              (0.3, 0.20, 0.10), (0.2, 0.12, 0.10), (0.1, 0.05, 0.15)):
        h = Hysteresis(enter=enter, exit=exit_, pad=pad)
        sp = h.apply(probs, SILERO_HOP_SEC, duration)
        print(score(sp, words, duration, pause_ref).line(
            f"silero e{enter} x{exit_} p{pad}"))

    print()
    print(f"production speech total {total(prod):.1f}s")


if __name__ == "__main__":
    main()
