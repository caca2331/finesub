"""Step 1 candidate: cap the production floor from above.

    floor' = min(production_floor, rolling_min_anchor + cap)

The tracker keeps every behaviour the branch proved load-bearing (frame-driven
locality, filled-pause absorption, shallow-pause splitting) wherever it sits within
`cap` dB of the true background; the cap only binds where creep has lifted the
floor beyond that -- exactly the dense-speech condition step 0 confirmed loses real
subtitles. By construction the capped floor is <= the production floor, so speech
can only be added, never removed: recall cannot get worse.

Scored against the step-0 human labels:

  recovered   the A-group regions (human: real speech / 听不清, never decoded)
              that become speech under the cap
  readmitted  seconds of regions humans labeled noise / filler that the cap
              hands back to the decoder (the cost, delegated downstream per the
              user's doctrine but still counted here)
  clean drift how much the annotated clean clip's intervals move at all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import sys
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from energy_sweep import cached_tracks, speech_from_tracks, Tracks  # noqa: E402
from floor_decomposed import Decomposed  # noqa: E402

Interval = Tuple[float, float]


def capped_speech(tr: Tracks, anchor: np.ndarray, cap: float) -> List[Interval]:
    from asr_playground.speech.preprocessing import energy as E

    floor = np.minimum(tr.noise_floor.numpy().astype(np.float64), anchor + cap)
    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b))
            for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def override_speech(tr: Tracks, anchor: np.ndarray, loud_db: float) -> List[Interval]:
    """Loudness veto: frames further than `loud_db` over the background anchor are
    forced speech-like, by dropping the floor only under those frames. Filled
    pauses sit 10-20 dB over the background and never reach the line; the lost
    segments are 40+ dB over. Everything else about the detector is untouched."""
    from asr_playground.speech.preprocessing import energy as E

    floor = tr.noise_floor.numpy().astype(np.float64).copy()
    e = tr.energy_db.numpy().astype(np.float64)
    loud = e > (anchor + loud_db)
    floor[loud] = np.minimum(floor[loud], e[loud] - 12.0)  # e - floor = 12 > margin
    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b))
            for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def coverage(iv: Interval, speech: List[Interval]) -> float:
    s, e = iv
    tot = 0.0
    for a, b in speech:
        if b <= s:
            continue
        if a >= e:
            break
        tot += min(b, e) - max(a, s)
    return tot / max(e - s, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=VOCAL")
    ap.add_argument("--joined", type=Path, required=True,
                    help="step0-joined.csv (regions + human labels)")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--caps", default="6,8,10,12,15,20")
    args = ap.parse_args()

    caps = [float(c) for c in args.caps.split(",")]
    joined = pd.read_csv(args.joined, encoding="utf-8-sig")

    for name, vocal in (x.split("=", 1) for x in args.clip):
        tr = cached_tracks(Path(vocal), args.cache_dir)
        edb = tr.energy_db.numpy().astype(np.float64)
        starts = tr.frame_starts.numpy().astype(np.float64)
        hop = float(starts[1] - starts[0])
        anchor = Decomposed()._anchor(edb, hop)
        floor = tr.noise_floor.numpy().astype(np.float64)
        over = floor - anchor
        prod = speech_from_tracks(tr)
        prod_sec = sum(e - s for s, e in prod)

        lab = joined[joined["clip"] == name]
        rescue = lab[(lab["kind"] == "added")
                     & (lab["never_decoded_frac"] > 0.5)
                     & (lab["label"].isin(["真语音", "听不清"]))]
        junk = lab[(lab["kind"] == "added")
                   & (lab["label"].isin(["噪声/抖动", "幻觉", "语气词"]))]

        print(f"\n=== {name} ===  floor-anchor: p50 {np.median(over):.1f} "
              f"p90 {np.quantile(over, .9):.1f} p99 {np.quantile(over, .99):.1f} dB; "
              f"prod speech {prod_sec:.0f}s / {len(prod)} 区间; "
              f"待捞回 {len(rescue)} 段, 已标垃圾 {len(junk)} 段")
        def report(tag: str, sp: List[Interval]) -> None:
            sp_sec = sum(e - s for s, e in sp)
            got = sum(1 for _, r in rescue.iterrows()
                      if coverage((r["start"], r["end"]), sp) > 0.5)
            back = sum((r["end"] - r["start"]) * coverage((r["start"], r["end"]), sp)
                       for _, r in junk.iterrows())
            print(f"{tag:>9} {sp_sec - prod_sec:>+9.1f} {len(sp):>5d} "
                  f"{got:>3d}/{len(rescue):<3d} {back:>7.1f}s")

        print(f"{'arm':>9} {'speech+s':>9} {'区间':>5} {'捞回':>6} {'垃圾回流':>8}")
        for cap in caps:
            report(f"cap{cap:.0f}", capped_speech(tr, anchor, cap))
        for loud in (30.0, 35.0, 40.0, 45.0):
            report(f"loud{loud:.0f}", override_speech(tr, anchor, loud))


if __name__ == "__main__":
    main()
