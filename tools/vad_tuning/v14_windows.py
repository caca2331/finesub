"""Step 14: do the two 120 s windows want to be 5 minutes?

Two independent windows are both set to 120 s and both feed the same decision:

  NORM_WINDOW_SEC        the local RMS gain window (with gain clipped to [-4,+6] dB)
  NOISE_LOCAL_WINDOW_SEC the window the noise-floor percentile is taken over

Widening them is not the same experiment for the two floor estimators. Under the
legacy tracker the windowed percentile contributes 0.3% of the floor, so its window
should barely matter; under the current one it is a hard lower bound, so it should
matter a lot. That asymmetry is the point of running the full 2x2x2.
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
from v10_quiet import speech_with_floor, word_levels  # noqa: E402


def specs(floor_win: float):
    """(legacy, current) floor definitions at a given window width."""
    from asr_playground.speech.preprocessing import energy as E

    common = dict(percentile=E.NOISE_INIT_PERCENTILE, window_sec=floor_win,
                  hop_sec=E.NOISE_LOCAL_HOP_SEC)
    track = dict(blend=E.NOISE_LOCAL_BLEND, follow=E.NOISE_TRACK_FOLLOW_ALPHA,
                 rise=E.NOISE_TRACK_RISE_ALPHA, gate=E.NOISE_TRACK_GATE_DB)
    return {
        "legacy": (FloorSpec("l", **common), TrackerSpec("l", **track)),
        "current": (FloorSpec("c", drop_degenerate=True, drop_when="if_sparse",
                              max_silent_frac=E.NOISE_SILENT_FRAC_MAX, **common),
                    TrackerSpec("c", clamp_to_target=True, clamp_slack=0.0, **track)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--word-srt")
    ap.add_argument("--human-clip")
    ap.add_argument("--gold")
    ap.add_argument("--noisy-clip", default="miyako")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--norm-windows", type=float, nargs="+", default=[120.0, 300.0])
    ap.add_argument("--floor-windows", type=float, nargs="+", default=[120.0, 300.0])
    ap.add_argument("--quiet-pct", type=float, default=10.0)
    args = ap.parse_args()

    from asr_playground.speech.preprocessing import energy as E

    clips = dict(x.split("=", 1) for x in args.clip)
    stables = dict(x.split("=", 1) for x in args.stable)
    cache = Path(args.cache_dir)
    pause = load_pause_ref(Path(args.gold)) if args.gold else None
    hw = load_word_srt(Path(args.word_srt)) if args.word_srt else None

    arms = [(nw, fw, algo)
            for nw in args.norm_windows
            for fw in args.floor_windows
            for algo in ("legacy", "current")]
    acc: Dict[tuple, dict] = {a: dict(lost=0, sl=0.0, st=0.0, qlost=0, qsl=0.0, qst=0.0,
                                      sp=0.0, dur=0.0, noisy=0.0, hum=None) for a in arms}
    per_clip: Dict[tuple, Dict[str, int]] = {a: {} for a in arms}

    saved_norm = E.NORM_WINDOW_SEC
    try:
        for name, path in clips.items():
            words = load_valid_words(Path(stables[name]))[0] if name in stables else []
            for nw in args.norm_windows:
                E.NORM_WINDOW_SEC = nw
                tr = cached_tracks(Path(path), cache / f"norm{int(nw)}")
                e = tr.energy_db.numpy().astype(np.float64)
                starts = tr.frame_starts.numpy().astype(np.float64)
                quiet: List = []
                if words:
                    lv = word_levels(words, e)
                    cut = np.quantile(lv, args.quiet_pct / 100.0)
                    quiet = [w for w, v in zip(words, lv) if v <= cut]
                for fw in args.floor_windows:
                    sp_by_algo = specs(fw)
                    for algo in ("legacy", "current"):
                        fspec, tspec = sp_by_algo[algo]
                        fl = floor_with_tracker(e, starts, tr.duration, fspec, tspec)
                        sp = speech_with_floor(tr, fl)
                        a = acc[(nw, fw, algo)]
                        a["sp"] += sum(b - x for x, b in sp)
                        a["dur"] += tr.duration
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
                                        pc = per_clip[(nw, fw, algo)]
                                        pc[name] = pc.get(name, 0) + 1
            print(f"  done {name}", file=sys.stderr, flush=True)
    finally:
        E.NORM_WINDOW_SEC = saved_norm

    hdr = (f"{'norm':>5} {'floor':>6} {'algo':<8} {'lost':>5} {'recall':>8} "
           f"{'Qlost':>5} {'Qrecall':>8} {'speech':>7} {'noisy':>7} "
           f"{'humLost':>7} {'humRec':>8} {'pause':>6}")
    print(hdr)
    print("-" * len(hdr))
    for nw, fw, algo in arms:
        a = acc[(nw, fw, algo)]
        h = a["hum"]
        hs = (f"{h.words_lost:>7d} {h.word_recall:>8.3%} {h.pause_excluded:>6.1%}"
              if h else " " * 23)
        print(f"{nw:>5.0f} {fw:>6.0f} {algo:<8} {a['lost']:>5d} "
              f"{1 - a['sl']/a['st']:>8.3%} {a['qlost']:>5d} "
              f"{1 - a['qsl']/a['qst']:>8.3%} {a['sp']/a['dur']:>7.1%} "
              f"{a['noisy']:>7.1%} {hs}")

    print()
    names = list(clips)
    print(f"{'lost per clip':<24} " + " ".join(f"{n[:9]:>9}" for n in names))
    for nw, fw, algo in arms:
        row = per_clip[(nw, fw, algo)]
        label = f"n{nw:.0f}/f{fw:.0f}/{algo}"
        print(f"{label:<24} " + " ".join(f"{row.get(n, 0):>9d}" for n in names))


if __name__ == "__main__":
    main()
