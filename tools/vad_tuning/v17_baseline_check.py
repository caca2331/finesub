"""Why does the *original* detector lose reference words at all?

The reference words come from production's own runs, so the detector that produced
them ought to score zero lost. It does not, and there are two separate reasons worth
telling apart:

  mislabelled baseline   the "original" row in appendices E/G read
                         NEGATIVE_PAD_RIGHT_MS from the module, which by then was
                         already 100. The references were produced at 140. So that
                         row was never the pre-branch configuration.
  structurally possible  a word's timestamps come from whisper, not from the VAD.
                         `inserted_gap_parts` keeps up to GAP_KEEP_REAL_MAX_SEC of
                         real audio after each interval, the last word may be
                         extended past the interval end, and drift can move a word
                         anywhere. Words legitimately sit outside speech intervals.

This measures both: the true pre-branch config, and where the still-lost words are
relative to the nearest speech interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402
import floor_lab as FL  # noqa: E402
import scorers as SC  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from refs import covered, load_valid_words  # noqa: E402
from v10_quiet import word_levels  # noqa: E402


def speech_from(tr, floor, scorer, pad_right: float):
    from asr_playground.speech.preprocessing import energy as E

    e = tr.energy_db.numpy().astype(np.float64)
    dbfs = tr.frame_dbfs.numpy().astype(np.float64)
    st = tr.frame_starts.numpy().astype(np.float64)
    en = tr.frame_ends.numpy().astype(np.float64)
    fl = floor(e, st, tr.duration)
    raw = scorer(e, fl, dbfs, st, en, tr.duration)
    saved = E.NEGATIVE_PAD_RIGHT_MS
    try:
        E.NEGATIVE_PAD_RIGHT_MS = pad_right
        ns = E._apply_negative_padding(raw, tr.duration)
    finally:
        E.NEGATIVE_PAD_RIGHT_MS = saved
    return [(float(a), float(b)) for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def gap_distance(speech, s: float, e: float):
    """Signed seconds from the word to the nearest speech interval.

    >0 means the word starts this long after the previous interval ended -- the
    region `inserted_gap_parts` may still hand to the decoder as real audio.
    """
    best_after, best_before = 1e9, 1e9
    for a, b in speech:
        if b <= s:
            best_after = min(best_after, s - b)
        if a >= e:
            best_before = min(best_before, a - e)
    return best_after, best_before


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    cache = Path(args.cache_dir)

    prod100 = SC.production(merge_gap_ms=100.0)
    arms = [
        ("TRUE pre-branch (padR140)", FL.legacy(), prod100, 140.0),
        ("legacy floor, padR100", FL.legacy(), prod100, 100.0),
        ("current (padR140)", FL.shipped(), SC.prod_minrun(min_speech_frames=8,
                                                           merge_gap_ms=60.0), 140.0),
    ]
    acc = {a[0]: dict(lost=0, sl=0.0, st=0.0, qlost=0, qsl=0.0, qst=0.0,
                      sp=0.0, dur=0.0, far=[], gap=[]) for a in arms}

    for name, path in clips.items():
        tr = cached_tracks(Path(path), cache)
        e = tr.energy_db.numpy().astype(np.float64)
        words = load_valid_words(Path(stables[name]))[0]
        lv = word_levels(words, e)
        cut = np.quantile(lv, args.quiet_pct / 100.0)
        quiet = [w for w, v in zip(words, lv) if v <= cut]
        for label, floor, scorer, pad in arms:
            sp = speech_from(tr, floor, scorer, pad)
            a = acc[label]
            a["sp"] += sum(y - x for x, y in sp)
            a["dur"] += tr.duration
            for ws, kl, ks, kt in ((words, "lost", "sl", "st"),
                                   (quiet, "qlost", "qsl", "qst")):
                for w in ws:
                    d = w.end - w.start
                    if d <= 0:
                        continue
                    miss = d - covered(sp, w.start, w.end)
                    a[kt] += d
                    a[ks] += miss
                    if miss / d >= 0.9:
                        a[kl] += 1
                        if kl == "lost":
                            after, before = gap_distance(sp, w.start, w.end)
                            a["gap"].append(min(after, before))
        print(f"  done {name}", file=sys.stderr, flush=True)

    hdr = (f"{'arm':<28} {'lost':>5} {'recall':>8} {'Qlost':>5} {'Qrecall':>8} "
           f"{'speech':>7}")
    print(hdr)
    print("-" * len(hdr))
    for label, *_ in arms:
        a = acc[label]
        print(f"{label:<28} {a['lost']:>5d} {1 - a['sl']/a['st']:>8.3%} "
              f"{a['qlost']:>5d} {1 - a['qsl']/a['qst']:>8.3%} "
              f"{a['sp']/a['dur']:>7.1%}")

    print()
    print("where the still-lost words sit, relative to the nearest speech interval")
    print(f"{'arm':<28} {'n':>5} {'<=0.7s (kept gap audio)':>24} {'>0.7s':>8} "
          f"{'median':>8} {'p90':>7}")
    for label, *_ in arms:
        g = np.array(acc[label]["gap"]) if acc[label]["gap"] else np.array([0.0])
        print(f"{label:<28} {len(g):>5d} {float((g <= 0.7).mean()):>24.1%} "
              f"{float((g > 0.7).mean()):>8.1%} {np.median(g):>8.2f} "
              f"{np.quantile(g, 0.9):>7.2f}")


if __name__ == "__main__":
    main()
