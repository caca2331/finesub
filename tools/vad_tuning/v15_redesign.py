"""Step 15: free exploration of the floor estimator and the interval scorer.

Scored exactly like everything else in this directory, so the numbers are comparable
to FINDINGS appendix E: valid ASR words over 11 clips, the quiet decile called out
separately because that is the cohort a rising floor can hurt, and the one human
timeline as the only non-circular recall reference.

  --round excl      how aggressive the no-signal exclusion may be
  --round floor     alternative estimators (rolling percentile / minimum statistics /
                    noise-only recursive averaging)
  --round score     alternative interval logic on a fixed floor
  --round best      hand-picked cross of the two
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import floor_lab as FL  # noqa: E402
import scorers as SC  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from refs import covered, load_pause_ref, load_valid_words, load_word_srt  # noqa: E402
from score import score  # noqa: E402
from v10_quiet import word_levels  # noqa: E402


def arms(round_name: str) -> List[Tuple[str, object, object]]:
    prod = SC.production()
    cur = FL.shipped(name="floor: shipped (excl -99)")
    if round_name == "excl":
        out = [("legacy", FL.legacy(), prod)]
        out += [(f"excl {t:.0f} (gate -99)", FL.shipped(exclude_db=t), prod)
                for t in (-99.0, -85.0, -70.0, -60.0, -50.0, -40.0)]
        # the same thresholds, but with the fraction gate measured on the *excluded*
        # mask instead of on no-signal -- i.e. "if most of the window is under -50,
        # do not act" rather than "if most of it is digital silence".
        out += [(f"excl {t:.0f} (gate self)",
                 FL.shipped(exclude_db=t, gate_on_exclude=True), prod)
                for t in (-70.0, -60.0, -50.0)]
        return out
    if round_name == "floor":
        return [
            ("legacy", FL.legacy(), prod),
            ("shipped", cur, prod),
            ("rollq 15s p10 rise.02", FL.rollq(15, 10, 0.02), prod),
            ("rollq 30s p10 rise.02", FL.rollq(30, 10, 0.02), prod),
            ("rollq 15s p20 rise.02", FL.rollq(15, 20, 0.02), prod),
            ("rollq 15s p10 rise.005", FL.rollq(15, 10, 0.005), prod),
            ("rollq 60s p5 rise.02", FL.rollq(60, 5, 0.02), prod),
            ("minstat 8s bias6", FL.minstat(8, 0.12, 6.0), prod),
            ("minstat 15s bias6", FL.minstat(15, 0.12, 6.0), prod),
            ("minstat 8s bias10", FL.minstat(8, 0.12, 10.0), prod),
            ("mcra gate6", FL.mcra(6.0), prod),
            ("mcra gate10", FL.mcra(10.0), prod),
            ("min(rollq15, legacy)", FL.combine(FL.rollq(15, 10, 0.02), FL.legacy()), prod),
        ]
    if round_name == "score":
        # Each family is swept along its own operating-point knob, because a scorer
        # that simply keeps more audio always looks better on recall. Only a
        # comparison at matched speech% says anything.
        out = []
        for m in (4.0, 5.0, 6.0, 7.0, 9.0):
            out.append((f"prod score margin{m:.0f}", cur, SC.production(margin=m)))
        for g in (99.0, 20.0, 12.0, 6.0, 0.0):
            out.append((f"runs guard{g:.0f}", cur, SC.runs(merge_guard_db=g)))
        for ec in (8.0, 15.0, 30.0, 60.0, 120.0):
            out.append((f"vit e{ec:.0f}/x6", cur,
                        SC.viterbi(enter_cost=ec, exit_cost=6.0)))
        return out
    if round_name == "score2":
        return [
            ("prod score margin6", cur, SC.production(margin=6.0)),
            ("prod + minrun 6", cur, SC.prod_minrun(min_speech_frames=6)),
            ("prod + minrun 8", cur, SC.prod_minrun(min_speech_frames=8)),
            ("prod + minrun 12", cur, SC.prod_minrun(min_speech_frames=12)),
            ("prod + minrun 8 snr9", cur,
             SC.prod_minrun(min_speech_frames=8, min_snr_db=9.0)),
            ("prod + minrun 12 snr9", cur,
             SC.prod_minrun(min_speech_frames=12, min_snr_db=9.0)),
            ("prod + minrun 8 snr12", cur,
             SC.prod_minrun(min_speech_frames=8, min_snr_db=12.0)),
            ("prod + minrun 10 snr12", cur,
             SC.prod_minrun(min_speech_frames=10, min_snr_db=12.0)),
            ("prod + minrun 8 snr15", cur,
             SC.prod_minrun(min_speech_frames=8, min_snr_db=15.0)),
            ("prod + minrun 6 snr15", cur,
             SC.prod_minrun(min_speech_frames=6, min_snr_db=15.0)),
            ("legacy floor + minrun8 s12", FL.legacy(),
             SC.prod_minrun(min_speech_frames=8, min_snr_db=12.0)),
            ("legacy floor + prod score", FL.legacy(), prod),
        ]
    if round_name == "floor2":
        # Round `floor` was not a fair comparison: every estimator sits at its own
        # absolute offset, and `margin` is calibrated against production's. An
        # estimator whose floor is 15 dB lower just keeps everything and scores a
        # gorgeous recall. Sweep the margin per estimator and read the table at
        # matched speech%.
        fams = [("legacy", FL.legacy()), ("shipped", cur),
                ("rollq15p10", FL.rollq(15, 10, 0.02)),
                ("rollq30p10", FL.rollq(30, 10, 0.02)),
                ("minstat15", FL.minstat(15, 0.12, 6.0)),
                ("mcra6", FL.mcra(6.0))]
        out = []
        for fname, f in fams:
            for m in (6.0, 10.0, 14.0, 18.0, 22.0):
                out.append((f"{fname} m{m:.0f}", f, SC.production(margin=m)))
        return out
    if round_name == "score3":
        out = [("legacy floor + prod", FL.legacy(), prod),
               ("shipped + prod", cur, prod)]
        for n in (5, 6, 7, 8, 10):
            out.append((f"shipped + minrun{n}", cur, SC.prod_minrun(min_speech_frames=n)))
        for n in (6, 8):
            out.append((f"legacy + minrun{n}", FL.legacy(),
                        SC.prod_minrun(min_speech_frames=n)))
        for m in (5.0, 7.0):
            out.append((f"shipped m{m:.0f} + minrun8", cur,
                        SC.prod_minrun(margin=m, min_speech_frames=8)))
        return out
    if round_name == "control":
        # Fair question before adding a rule: can the knob that already exists do
        # this? MERGE_GAP_MS sets how long a speech gap may be merged over -- but it
        # does so through the score ratio, so it only bites on *loud* speech. If it
        # could reproduce minrun, minrun is not worth having.
        out = [("legacy floor + prod", FL.legacy(), prod),
               ("shipped + prod (merge100)", cur, prod)]
        for g in (80.0, 60.0, 40.0, 25.0):
            out.append((f"shipped + merge{g:.0f}", cur, SC.production(merge_gap_ms=g)))
        for mn in (300.0, 250.0):
            out.append((f"shipped + minNS{mn:.0f}", cur,
                        SC.production(min_non_speech_ms=mn)))
        out.append(("shipped + minrun8", cur, SC.prod_minrun(min_speech_frames=8)))
        out.append(("shipped m5 + minrun8", cur,
                    SC.prod_minrun(margin=5.0, min_speech_frames=8)))
        return out
    if round_name == "final":
        out = [("A original (legacy+prod)", FL.legacy(), prod),
               ("B shipped floor + prod", cur, prod)]
        for g in (60.0, 50.0):
            out.append((f"C shipped merge{g:.0f}", cur, SC.production(merge_gap_ms=g)))
        for g in (100.0, 70.0, 60.0, 50.0):
            for n in (6, 8):
                out.append((f"D merge{g:.0f}+minrun{n}", cur,
                            SC.prod_minrun(min_speech_frames=n, merge_gap_ms=g)))
        out.append(("E legacy floor + merge60", FL.legacy(),
                    SC.production(merge_gap_ms=60.0)))
        out.append(("F legacy + merge60 + mr8", FL.legacy(),
                    SC.prod_minrun(min_speech_frames=8, merge_gap_ms=60.0)))
        return out
    raise SystemExit(f"unknown round {round_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--word-srt")
    ap.add_argument("--human-clip")
    ap.add_argument("--gold")
    ap.add_argument("--noisy-clip", default="miyako")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    args = ap.parse_args()

    from asr_playground.speech.preprocessing import energy as E

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    cache = Path(args.cache_dir)
    pause = load_pause_ref(Path(args.gold)) if args.gold else None
    hw = load_word_srt(Path(args.word_srt)) if args.word_srt else None
    plan = arms(args.round)

    acc: Dict[str, dict] = {a[0]: dict(lost=0, sl=0.0, st=0.0, qlost=0, qsl=0.0,
                                       qst=0.0, sp=0.0, dur=0.0, noisy=0.0, hum=None,
                                       n=0)
                            for a in plan}
    per_clip: Dict[str, Dict[str, int]] = {a[0]: {} for a in plan}

    for name, path in clips.items():
        tr = cached_tracks(Path(path), cache)
        e = tr.energy_db.numpy().astype(np.float64)
        dbfs = tr.frame_dbfs.numpy().astype(np.float64)
        starts = tr.frame_starts.numpy().astype(np.float64)
        ends = tr.frame_ends.numpy().astype(np.float64)
        words = load_valid_words(Path(stables[name]))[0] if name in stables else []
        quiet: List = []
        if words:
            lv = word_levels(words, e)
            cut = np.quantile(lv, args.quiet_pct / 100.0)
            quiet = [w for w, v in zip(words, lv) if v <= cut]

        floor_cache: Dict[int, np.ndarray] = {}
        for label, floor, scorer in plan:
            key = id(floor)
            if key not in floor_cache:
                floor_cache[key] = floor(e, starts, tr.duration)
            fl = floor_cache[key]
            raw = scorer(e, fl, dbfs, starts, ends, tr.duration)
            ns = E._apply_negative_padding(raw, tr.duration)
            sp = [(float(a), float(b)) for a, b
                  in E.invert_intervals(ns, tr.duration) if b > a]
            a = acc[label]
            a["sp"] += sum(b - x for x, b in sp)
            a["dur"] += tr.duration
            a["n"] += len(sp)
            if name == args.noisy_clip:
                a["noisy"] = sum(b - x for x, b in sp) / tr.duration
            if hw is not None and name == args.human_clip:
                a["hum"] = score(sp, hw, tr.duration, pause)
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
                            pc = per_clip[label]
                            pc[name] = pc.get(name, 0) + 1
        print(f"  done {name}", file=sys.stderr, flush=True)

    hdr = (f"{'arm':<26} {'lost':>5} {'recall':>8} {'Qlost':>5} {'Qrecall':>8} "
           f"{'speech':>7} {'noisy':>7} {'nInt':>6} {'humLost':>7} {'humRec':>8} "
           f"{'pause':>6}")
    print(hdr)
    print("-" * len(hdr))
    for label, _f, _s in plan:
        a = acc[label]
        h = a["hum"]
        hs = (f"{h.words_lost:>7d} {h.word_recall:>8.3%} {h.pause_excluded:>6.1%}"
              if h else " " * 23)
        print(f"{label:<26} {a['lost']:>5d} {1 - a['sl']/a['st']:>8.3%} "
              f"{a['qlost']:>5d} {1 - a['qsl']/a['qst']:>8.3%} "
              f"{a['sp']/a['dur']:>7.1%} {a['noisy']:>7.1%} {a['n']:>6d} {hs}")

    print()
    names = list(clips)
    print(f"{'lost per clip':<26} " + " ".join(f"{n[:9]:>9}" for n in names))
    for label, _f, _s in plan:
        row = per_clip[label]
        print(f"{label:<26} " + " ".join(f"{row.get(n, 0):>9d}" for n in names))


if __name__ == "__main__":
    main()
