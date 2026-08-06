"""Step 19: re-run the rejected candidates against the metrics that replaced `lost`.

Appendix G rejected the alternative floor estimators and the alternative interval
logic on a `lost`-at-matched-speech% comparison. Appendix I then showed `lost` was
mostly measuring where whisper put a timestamp. So those rejections were made with a
ruler that has since been thrown away, and they do not stand until re-run.

Scored here on what replaced it:

  unvoiced%   speech seconds no word from the ASR union occupies -- delivered noise
  emptyInt    intervals that produced no word at all -- hallucination surface
  waste       dead lead-in before the first real word, on the annotated clip
  guards      human-timeline lost, onset_excl over the 25 annotated real onsets,
              pause_excl over the 32 annotated filled pauses

Each family is swept along its own operating knob, because anything that simply
keeps more audio scores better on noise and worse on everything else.
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

import floor_lab as FL  # noqa: E402
import scorers as SC  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402
from v18_precision import build  # noqa: E402


def arms(kind: str):
    cur, leg = FL.shipped(), FL.legacy()
    prod = SC.production(merge_gap_ms=100.0)
    out = [("A pre-branch", leg, prod), ("B floor gate (current)", cur, prod)]
    if kind == "floor":
        fams = [("rollq15p10", FL.rollq(15, 10, 0.02)),
                ("rollq30p10", FL.rollq(30, 10, 0.02)),
                ("minstat15", FL.minstat(15, 0.12, 6.0)),
                ("mcra6", FL.mcra(6.0))]
        for fname, f in fams:
            for m in (6.0, 12.0, 18.0, 24.0):
                out.append((f"{fname} m{m:.0f}", f,
                            SC.production(margin=m, merge_gap_ms=100.0)))
        for m in (8.0, 10.0):
            out.append((f"B margin {m:.0f}", cur,
                        SC.production(margin=m, merge_gap_ms=100.0)))
    else:
        for g in (99.0, 20.0, 12.0, 6.0):
            out.append((f"runs guard{g:.0f}", cur, SC.runs(merge_guard_db=g)))
        for ec in (8.0, 15.0, 30.0, 60.0):
            out.append((f"vit e{ec:.0f}/x6", cur,
                        SC.viterbi(enter_cost=ec, exit_cost=6.0)))
        for ec in (15.0, 30.0):
            out.append((f"vit e{ec:.0f}/x6 g6", cur,
                        SC.viterbi(enter_cost=ec, exit_cost=6.0, merge_guard_db=6.0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=("floor", "score"))
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--annotated", required=True, help="clip name with the human SRT")
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)
    plan = arms(args.kind)
    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))

    tot = {a[0]: [0, 0.0, 0.0] for a in plan}   # emptyInt, unvoiced_sec, speech_sec
    guard: Dict[str, object] = {}
    ann_tr = None
    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        for label, fl, sc in plan:
            sp = build(tr, fl, sc, 40.0, 140.0)
            v = noise_intervals(sp, wm)
            t = tot[label]
            t[0] += v.empty_intervals
            t[1] += v.unvoiced_sec
            t[2] += v.speech_sec
            if name == args.annotated:
                guard[label] = sp
        if name == args.annotated:
            ann_tr = tr
        print(f"  done {name}", file=sys.stderr, flush=True)

    hdr = (f"{'arm':<22} | {'emptyInt':>8} {'unvoic%':>8} {'speech':>7} | "
           f"{'lost':>4} {'onsetX':>7} {'pauseX':>7} {'waste':>7} {'cutOn':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, *_ in plan:
        ei, us, ss = tot[label]
        sp = guard[label]
        s = score(sp, hw, ann_tr.duration, pause)
        t = tightness(sp, hw)
        print(f"{label:<22} | {ei:>8d} {us/ss:>8.1%} {ss:>6.0f}s | "
              f"{s.words_lost:>4d} {s.onset_excluded:>7.1%} {s.pause_excluded:>7.1%} "
              f"{t.total_waste:>6.1f}s {t.clipped_onsets:>5d}")


if __name__ == "__main__":
    main()
