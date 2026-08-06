"""Step 25: does splitting the floor's three jobs fix the two defects?

Scored on the usual precision metrics plus the two defect measures, so a candidate
has to show it removed the creep and the jitter *and* did not give away the
filled-pause rejection that the entangled design bought.
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

import torch  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from floor_decomposed import Decomposed  # noqa: E402
from floor_defects import report  # noqa: E402
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def speech_with(tr, floor: np.ndarray):
    from asr_playground.speech.preprocessing import energy as E

    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b))
            for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def arms(which: str):
    out = [("production", None)]
    if which == "ablate":
        out += [
            ("level only (a6 b0 无驻留)", Decomposed(a=6.0, b=0.0, dwell_max=0.0)),
            ("level + 驻留", Decomposed(a=6.0, b=0.0)),
            ("level + CFAR b1", Decomposed(a=3.0, b=1.0, dwell_max=0.0)),
            ("level + CFAR b1 + 驻留", Decomposed(a=3.0, b=1.0)),
            ("level + CFAR b2 + 驻留", Decomposed(a=2.0, b=2.0)),
            ("level + CFAR b3 + 驻留", Decomposed(a=0.0, b=3.0)),
        ]
    elif which == "tune":
        for du in (0.02, 0.06, 0.2):
            out.append((f"b1 dwell_up={du}", Decomposed(a=3.0, b=1.0, dwell_up=du)))
        for dd in (0.005, 0.02):
            out.append((f"b1 dwell_down={dd}",
                        Decomposed(a=3.0, b=1.0, dwell_up=0.06, dwell_down=dd)))
    elif which == "cap":
        for c in (15.0, 20.0, 25.0, 30.0):
            out.append((f"drop35 cap{c:.0f}",
                        Decomposed(a=8.0, b=0.0, dwell_max=0.0, rel_drop_db=35.0,
                                   rel_cap_over_level_db=c)))
        for d in (30.0, 40.0):
            out.append((f"drop{d:.0f} cap20",
                        Decomposed(a=8.0, b=0.0, dwell_max=0.0, rel_drop_db=d,
                                   rel_cap_over_level_db=20.0)))
        out.append(("drop35 无上界", Decomposed(a=8.0, b=0.0, dwell_max=0.0,
                                              rel_drop_db=35.0)))
        return out
    elif which == "chunk2":
        # drop30 already matches production's chunk sizes and beats its boundary
        # tightness; back off further and try trading `a` down, since the relative
        # criterion has taken over the pause-cutting that `a` was doing.
        for d in (30.0, 35.0, 40.0, 45.0):
            out.append((f"a8 drop{d:.0f}", Decomposed(a=8.0, b=0.0, dwell_max=0.0,
                                                      rel_drop_db=d)))
        for a in (6.0, 7.0):
            out.append((f"a{a:.0f} drop30", Decomposed(a=a, b=0.0, dwell_max=0.0,
                                                       rel_drop_db=30.0)))
        for a in (6.0, 7.0):
            out.append((f"a{a:.0f} drop35", Decomposed(a=a, b=0.0, dwell_max=0.0,
                                                       rel_drop_db=35.0)))
        return out
    elif which == "chunk":
        for d in (0.0, 15.0, 20.0, 25.0, 30.0):
            out.append((f"a8 drop{d:.0f} w4", Decomposed(a=8.0, b=0.0, dwell_max=0.0,
                                                         rel_drop_db=d)))
        for w in (2.0, 8.0):
            out.append((f"a8 drop20 w{w:.0f}", Decomposed(a=8.0, b=0.0, dwell_max=0.0,
                                                          rel_drop_db=20.0,
                                                          rel_win_sec=w)))
        for pc in (75.0, 98.0):
            out.append((f"a8 drop20 p{pc:.0f}", Decomposed(a=8.0, b=0.0, dwell_max=0.0,
                                                           rel_drop_db=20.0,
                                                           rel_pct=pc)))
        return out
    else:
        # Filled-pause rejection is being reassigned to silero / the ASR, so it no
        # longer constrains the detection distance. That frees `a`, which is the
        # knob that actually governs how much background gets in. No dwell: its
        # only job was the pause absorption.
        for a in (6.0, 8.0, 10.0, 12.0, 14.0):
            out.append((f"a={a:.0f} b0", Decomposed(a=a, b=0.0, dwell_max=0.0)))
        for a, b in ((6.0, 1.0), (8.0, 1.0), (6.0, 2.0)):
            out.append((f"a={a:.0f} b{b:.0f}",
                        Decomposed(a=a, b=b, dwell_max=0.0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--round", default="ablate")
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)
    plan = arms(args.round)

    tot = {a[0]: [0, 0.0, 0.0, 0, 0.0] for a in plan}
    defects = {a[0]: [] for a in plan}
    guard: Dict[str, list] = {}
    ann_tr = None
    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        e = tr.energy_db.numpy().astype(np.float64)
        st = tr.frame_starts.numpy().astype(np.float64)
        for label, spec in plan:
            fl = (tr.noise_floor.numpy().astype(np.float64) if spec is None
                  else spec(e, st, tr.duration))
            sp = speech_with(tr, fl)
            v = noise_intervals(sp, wm)
            t = tot[label]
            t[0] += v.empty_intervals
            t[1] += v.unvoiced_sec
            t[2] += v.speech_sec
            t[3] += len(sp)
            t[4] += v.empty_sec
            defects[label].append(report(name, e, fl, wm))
            if name == args.annotated:
                guard[label] = sp
        if name == args.annotated:
            ann_tr = tr
        print(f"  done {name}", file=sys.stderr, flush=True)

    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))
    hdr = (f"{'arm':<20} | {'lost':>4} {'onsetX':>7} {'cutOn':>5} {'humRec':>8} | "
           f"{'空区间':>6} {'空秒':>6} {'虚假/分':>8} | {'区间数':>6} {'speech':>7} "
           f"{'时长p90':>5} {'最长':>6} {'>10s':>4} {'死音频':>6} {'头p90':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, _ in plan:
        ei, us, ss, ni, esec = tot[label]
        sp = guard[label]
        s = score(sp, hw, ann_tr.duration, pause)
        t = tightness(sp, hw)
        d = defects[label]
        cs = float(np.mean([x["creep_span"] for x in d]))
        cp = float(np.mean([x["cross_per_min"] for x in d]))
        ov = float(np.mean([x["over_share"] for x in d]))
        du = np.array([b - a for a, b in sp])
        print(f"{label:<20} | {s.words_lost:>4d} {s.onset_excluded:>7.1%} "
              f"{t.clipped_onsets:>5d} {s.word_recall:>8.3%} | {ei:>6d} "
              f"{esec:>5.0f}s {cp:>8.1f} | {ni:>6d} {ss:>6.0f}s "
              f"{np.quantile(du, .9):>5.2f} {du.max():>6.2f} {int((du > 10).sum()):>4d} "
              f"{t.total_waste:>6.1f}s {np.quantile(t.head_waste, .9):>5.2f}")


if __name__ == "__main__":
    main()
