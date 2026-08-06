"""Wide baseline: every backend and hybrid, over every clip, on valid ASR words.

Reference is `load_valid_words` -- stable.json minus hallucination / filled-pause /
drift segments, decoder loops, over-long words and bare punctuation.

Known bias, stated once and not forgotten: those words exist because the production
energy VAD kept their regions. The reference therefore cannot show speech the
production detector already dropped, and every `lost` count for a production variant
is a lower bound. It is unbiased for the comparison that matters here -- how much
*additional* speech a candidate discards relative to what the pipeline has already
proven is speech.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from backends import (SILERO_HOP_SEC, Hysteresis, load_mono, silero_probs, total)
from energy_sweep import compute_tracks, speech_from_tracks
from hybrid import AdaptiveHead, GuardedAggressive
from refs import load_pause_ref, load_valid_words
from score import Score, score


def agg(scores, names) -> None:
    for name in names:
        rows = [s for n, s in scores if n == name]
        if not rows:
            continue
        dur = sum(r.duration for r in rows)
        sf = sum(r.speech_frac * r.duration for r in rows) / dur
        lost = sum(r.words_lost for r in rows)
        clipped = sum(r.words_clipped for r in rows)
        ch = sum(r.clipped_head for r in rows)
        ct = sum(r.clipped_tail for r in rows)
        nw = sum(r.words for r in rows)
        rec = 1.0 - sum(r.word_sec_lost for r in rows) / max(
            sum(r.word_sec_total for r in rows), 1e-9)
        print(f"{name:<30} speech={sf:>5.1%} lost={lost:>4d}/{nw} "
              f"({lost/max(nw,1):>6.3%}) recall={rec:>7.3%} "
              f"clipH={ch:>4d} clipT={ct:>4d} n_int={sum(r.n_intervals for r in rows):>5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gold", help="disfluency_gold.json (BV1cqLR6hEp3 only)")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    pause_ref = load_pause_ref(Path(args.gold)) if args.gold else None
    scores = []
    names = []
    annotated = {}

    for clip in args.clips:
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        if not audio.exists() or not stable.exists():
            print(f"{clip}: missing artifacts")
            continue
        words, st = load_valid_words(stable)
        tr = compute_tracks(audio)
        probs = silero_probs(audio, cache / f"silero-{clip}.npz")
        pr = pause_ref if clip == "BV1cqLR6hEp3" else None

        prod = speech_from_tracks(tr)
        aggr = speech_from_tracks(tr, {"NEGATIVE_PAD_RIGHT_MS": 60.0})
        variants = [
            ("production energy", prod),
            ("energy padR=100", speech_from_tracks(tr, {"NEGATIVE_PAD_RIGHT_MS": 100.0})),
            ("energy padR=60", aggr),
            ("silero e0.5 p0.1", Hysteresis(0.5, 0.35, pad=0.10).apply(
                probs, SILERO_HOP_SEC, tr.duration)),
            ("silero e0.2 p0.1", Hysteresis(0.2, 0.12, pad=0.10).apply(
                probs, SILERO_HOP_SEC, tr.duration)),
            ("hybrid head thr0.3", AdaptiveHead(thr=0.30)(prod, probs)),
            ("hybrid head thr0.5", AdaptiveHead(thr=0.50)(prod, probs)),
            ("hybrid head thr0.7", AdaptiveHead(thr=0.70)(prod, probs)),
            ("guard on production", GuardedAggressive(thr=0.5)(prod, probs, tr.duration)),
            ("hybrid guard padR=60", GuardedAggressive(thr=0.5)(aggr, probs, tr.duration)),
            # Both roles at once: silero restores anything it is confident about
            # (recall net), then trims heads only where it is confident of silence.
            ("guard+head 0.5/0.3",
             AdaptiveHead(thr=0.30)(
                 GuardedAggressive(thr=0.5)(aggr, probs, tr.duration), probs)),
            ("guard+head 0.7/0.3",
             AdaptiveHead(thr=0.30)(
                 GuardedAggressive(thr=0.7)(aggr, probs, tr.duration), probs)),
        ]
        print(f"--- {clip}: {st['kept']} valid words "
              f"(dropped {st['dropped_drift']} drift segs, {st['dropped_repeat']} repeats)")
        for name, sp in variants:
            s = score(sp, words, tr.duration, pr)
            scores.append((name, s))
            if name not in names:
                names.append(name)
            if pr is not None:
                annotated[name] = s
            print("   " + s.line(name))
        print()

    print("=== aggregate over all clips (valid ASR words) ===")
    agg(scores, names)

    if annotated:
        print()
        print("=== filled-pause rejection, BV1cqLR6hEp3 only (32 pause / 25 onset blocks) ===")
        print("    higher pause_excl is the goal; onset_excl must stay at 0")
        for name in names:
            s = annotated.get(name)
            if s:
                print(f"{name:<30} pause_excl={s.pause_excluded:>5.1%} "
                      f"onset_excl={s.onset_excluded:>5.1%}")


if __name__ == "__main__":
    main()
