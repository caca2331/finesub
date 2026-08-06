"""Step 11: ablate the two floor rules, and price the quiet-speech risk.

v10 measures what shipped. This asks whether each half earns its keep and whether a
softer form of the second rule keeps the precision without the risk:

  rule 1  exclude no-signal frames from the windowed percentile
          "always"        -- as shipped
          "if_degenerate" -- only when the plain percentile has itself landed on the
                             clamp, i.e. only where the estimate was already broken
  rule 2  the windowed percentile is a lower bound on the tracker
          slack 0         -- as shipped
          slack N dB      -- the tracker may still sit N dB under the window

Recall is scored on valid ASR words plus the one human timeline; the quiet decile is
reported separately because that is the only cohort a rising floor can hurt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from energy_sweep import cached_tracks  # noqa: E402
from floor_variants import FloorSpec, TrackerSpec, floor_with_tracker  # noqa: E402
from refs import covered, load_pause_ref, load_valid_words, load_word_srt  # noqa: E402
from score import score  # noqa: E402
from v10_quiet import cohort_recall, speech_with_floor, word_levels  # noqa: E402


def variants() -> List[tuple]:
    def f(name, drop, when="always", cap=None, pct=None, sparse=0.30):
        from asr_playground.speech.preprocessing import energy as E
        return FloorSpec(name, percentile=E.NOISE_INIT_PERCENTILE if pct is None else pct,
                         window_sec=E.NOISE_LOCAL_WINDOW_SEC,
                         hop_sec=E.NOISE_LOCAL_HOP_SEC,
                         drop_degenerate=drop, drop_when=when,
                         cap_below_loud_db=cap, max_silent_frac=sparse)

    def t(name, clamp, slack=0.0, hold=False):
        from asr_playground.speech.preprocessing import energy as E
        return TrackerSpec(name, blend=E.NOISE_LOCAL_BLEND,
                           follow=E.NOISE_TRACK_FOLLOW_ALPHA,
                           rise=E.NOISE_TRACK_RISE_ALPHA,
                           gate=E.NOISE_TRACK_GATE_DB,
                           clamp_to_target=clamp, clamp_slack=slack,
                           hold_on_silence=hold)

    return [
        ("legacy (pre-fix)", f("l", False), t("l", False)),
        ("SHIPPED: rule1+clamp 0", f("a", True), t("c", True, 0.0)),
        ("rule1 if-degen + slack 6", f("d", True, "if_degenerate"), t("c", True, 6.0)),
        # The percentile over non-silent frames lands on quiet *speech* when the gaps
        # are true silence -- speech's own dynamic range is well over 30 dB, which is
        # why capping the target below the loud level did nothing. A lower percentile
        # stays under speech while still finding residual noise where it exists.
        ("p1 voiced + slack 6", f("a", True, pct=1.0), t("c", True, 6.0)),
        # Only act where silence is sporadic: a window that is mostly digital silence
        # has no residual noise to reject, so the low floor there is correct.
        ("if-sparse<30% + clamp 0", f("s", True, "if_sparse"), t("c", True, 0.0)),
        ("if-sparse<30% + slack 6", f("s", True, "if_sparse"), t("c", True, 6.0)),
        ("if-sparse<15% + clamp 0", f("s", True, "if_sparse", sparse=0.15), t("c", True, 0.0)),
        ("if-sparse<50% + clamp 0", f("s", True, "if_sparse", sparse=0.50), t("c", True, 0.0)),
        ("if-sparse<30% p2 + clamp 0", f("s", True, "if_sparse", pct=2.0), t("c", True, 0.0)),
        ("if-sparse<30% p1 + clamp 0", f("s", True, "if_sparse", pct=1.0), t("c", True, 0.0)),
        ("if-sparse<30% p2 + slack 3", f("s", True, "if_sparse", pct=2.0), t("c", True, 3.0)),
        ("if-sparse<50% p2 + clamp 0", f("s", True, "if_sparse", pct=2.0, sparse=0.50),
         t("c", True, 0.0)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--word-srt")
    ap.add_argument("--human-clip")
    ap.add_argument("--gold")
    ap.add_argument("--noisy-clip", default="miyako(noisy)")
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    ap.add_argument("--cache-dir")
    args = ap.parse_args()
    cache = Path(args.cache_dir) if args.cache_dir else None

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    vs = variants()
    acc: Dict[str, dict] = {v[0]: dict(lost=0, sl=0.0, st=0.0, qlost=0, qsl=0.0, qst=0.0,
                                       sp=0.0, dur=0.0, noisy_sp=0.0, hum=None)
                            for v in vs}
    pause = load_pause_ref(Path(args.gold)) if args.gold else None
    hw = load_word_srt(Path(args.word_srt)) if args.word_srt else None

    per_clip: Dict[str, Dict[str, int]] = {}
    for name, path in clips.items():
        tr = cached_tracks(Path(path), cache)
        e = tr.energy_db.numpy().astype(np.float64)
        starts = tr.frame_starts.numpy().astype(np.float64)
        st = stables.get(name)
        words = load_valid_words(Path(st))[0] if st else []
        quiet = []
        if words:
            lv = word_levels(words, e)
            cut = np.quantile(lv, args.quiet_pct / 100.0)
            quiet = [w for w, v in zip(words, lv) if v <= cut]

        for label, fspec, tspec in vs:
            fl = floor_with_tracker(e, starts, tr.duration, fspec, tspec)
            sp = speech_with_floor(tr, fl)
            a = acc[label]
            a["sp"] += sum(b - x for x, b in sp)
            a["dur"] += tr.duration
            if name == args.noisy_clip:
                a["noisy_sp"] = sum(b - x for x, b in sp) / tr.duration
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
                            per_clip.setdefault(label, {})[name] = \
                                per_clip.setdefault(label, {}).get(name, 0) + 1
            if hw is not None and name == args.human_clip:
                a["hum"] = score(sp, hw, tr.duration, pause)
        print(f"  done {name}", file=sys.stderr, flush=True)

    hdr = (f"{'variant':<30} {'lost':>5} {'recall':>8} | {'Qlost':>5} {'Qrecall':>8} | "
           f"{'speech':>7} {'noisy':>7} | {'humLost':>7} {'humRec':>8} {'pause':>6} {'onset':>6}")
    print(hdr)
    print("-" * len(hdr))
    for label, _f, _t in vs:
        a = acc[label]
        h = a["hum"]
        hs = (f"{h.words_lost:>7d} {h.word_recall:>8.3%} {h.pause_excluded:>6.1%} "
              f"{h.onset_excluded:>6.1%}") if h else " " * 30
        print(f"{label:<30} {a['lost']:>5d} {1 - a['sl']/a['st']:>8.3%} | "
              f"{a['qlost']:>5d} {1 - a['qsl']/a['qst']:>8.3%} | "
              f"{a['sp']/a['dur']:>7.1%} {a['noisy_sp']:>7.1%} | {hs}")

    print()
    names = list(clips)
    print(f"{'lost per clip':<30} " + " ".join(f"{n[:9]:>9}" for n in names))
    for label, _f, _t in vs:
        row = per_clip.get(label, {})
        print(f"{label:<30} " + " ".join(f"{row.get(n, 0):>9d}" for n in names))


if __name__ == "__main__":
    main()
