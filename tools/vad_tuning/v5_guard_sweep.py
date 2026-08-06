"""Tune the recall net so it costs as little precision as possible.

`guard on production` fixed almost all remaining word loss but added 2.9pp of audio
and halved filled-pause rejection, because every silero-confident region gets unioned
in -- including short breaths and pauses the energy detector had correctly rejected.

Two knobs decide how selective the rescue is: how sure silero must be (`thr`) and how
long the region must be (`min_speech`). A rescue that matters is a real utterance; a
rescue that costs is a 100 ms breath.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from backends import silero_probs
from energy_sweep import compute_tracks, speech_from_tracks
from hybrid import AdaptiveHead, GuardedAggressive
from refs import load_pause_ref, load_valid_words
from score import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gold", required=True)
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    pause_ref = load_pause_ref(Path(args.gold))

    ctx = []
    for clip in args.clips:
        audio = Path(args.root) / clip / f"{clip}-vocal.flac"
        stable = Path(args.root) / clip / f"{clip}-stable.json"
        if not audio.exists() or not stable.exists():
            continue
        words, _ = load_valid_words(stable)
        tr = compute_tracks(audio)
        probs = silero_probs(audio, cache / f"silero-{clip}.npz")
        ctx.append((clip, tr, probs, words, speech_from_tracks(tr)))

    def evaluate(name, make):
        rows = []
        for clip, tr, probs, words, prod in ctx:
            sp = make(tr, probs, prod)
            rows.append(score(sp, words, tr.duration,
                              pause_ref if clip == "BV1cqLR6hEp3" else None))
        dur = sum(r.duration for r in rows)
        sf = sum(r.speech_frac * r.duration for r in rows) / dur
        lost = sum(r.words_lost for r in rows)
        rec = 1.0 - sum(r.word_sec_lost for r in rows) / sum(r.word_sec_total for r in rows)
        ch = sum(r.clipped_head for r in rows)
        ct = sum(r.clipped_tail for r in rows)
        pe = next(r.pause_excluded for r, (c, *_ ) in zip(rows, ctx) if c == "BV1cqLR6hEp3")
        oe = next(r.onset_excluded for r, (c, *_ ) in zip(rows, ctx) if c == "BV1cqLR6hEp3")
        print(f"{name:<34} speech={sf:>5.1%} lost={lost:>3d} recall={rec:>7.3%} "
              f"clipH={ch:>4d} clipT={ct:>3d} pause_excl={pe:>5.1%} onset_excl={oe:>4.1%}")

    print("baseline")
    evaluate("production energy", lambda tr, p, prod: prod)
    print()

    print("guard on production: thr x min_speech")
    for thr, mins in itertools.product((0.5, 0.7, 0.9), (0.10, 0.25, 0.40, 0.60)):
        evaluate(f"  guard thr={thr} min={mins}",
                 lambda tr, p, prod, thr=thr, mins=mins:
                 GuardedAggressive(thr=thr, min_speech=mins, pad=0.05)(prod, p, tr.duration))
    print()

    print("selective guard + adaptive head trim")
    for thr, mins, htrim in itertools.product((0.7, 0.9), (0.40,), (0.30, 0.50)):
        evaluate(f"  guard {thr}/{mins} + head {htrim}",
                 lambda tr, p, prod, thr=thr, mins=mins, htrim=htrim:
                 AdaptiveHead(thr=htrim)(
                     GuardedAggressive(thr=thr, min_speech=mins, pad=0.05)(prod, p, tr.duration),
                     p))
    print()

    print("same, on the tighter energy base (padR=60)")
    for thr, mins in itertools.product((0.7, 0.9), (0.40,)):
        evaluate(f"  padR60 + guard {thr}/{mins}",
                 lambda tr, p, prod, thr=thr, mins=mins:
                 GuardedAggressive(thr=thr, min_speech=mins, pad=0.05)(
                     speech_from_tracks(tr, {"NEGATIVE_PAD_RIGHT_MS": 60.0}), p, tr.duration))
        evaluate(f"  padR60 + guard {thr}/{mins} + head",
                 lambda tr, p, prod, thr=thr, mins=mins:
                 AdaptiveHead(thr=0.30)(
                     GuardedAggressive(thr=thr, min_speech=mins, pad=0.05)(
                         speech_from_tracks(tr, {"NEGATIVE_PAD_RIGHT_MS": 60.0}), p, tr.duration),
                     p))


if __name__ == "__main__":
    main()
