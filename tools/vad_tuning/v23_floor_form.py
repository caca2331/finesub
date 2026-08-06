"""Two changes to the shape of the floor, measured on the metrics that replaced `lost`.

  step      the per-frame target takes its anchor window's percentile outright,
            instead of ramping linearly between neighbouring anchors
  cur=target  the tracker stops reading the raw frame at all, so the floor becomes
            an asymmetric EMA of the windowed percentile

The second is the larger change by far. At blend 0.997 the shipped tracker is 99.7%
raw frame and 0.3% window, which is why appendix F found the window width to be a
no-op; setting cur = target inverts that completely and the window becomes the whole
estimator.

Whatever is in `energy.py` right now is the "current" arm -- the cached tracks store
the framing only and the floor is recomputed on load, so edits are picked up without
re-reading audio.
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

from energy_sweep import cached_tracks, speech_from_tracks  # noqa: E402
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--label", default="current")
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)

    ei = us = ss = ni = 0.0
    guard = None
    ann_tr = None
    floor_stats = []
    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        sp = speech_from_tracks(tr)
        v = noise_intervals(sp, wm)
        ei += v.empty_intervals
        us += v.unvoiced_sec
        ss += v.speech_sec
        ni += len(sp)
        e = tr.energy_db.numpy().astype(np.float64)
        fl = tr.noise_floor.numpy().astype(np.float64)
        floor_stats.append((name, float(np.median(fl)), float(np.median(e - fl))))
        if name == args.annotated:
            guard, ann_tr = sp, tr

    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))
    s = score(guard, hw, ann_tr.duration, pause)
    t = tightness(guard, hw)
    print(f"{args.label:<26} | {int(ni):>6d} {int(ei):>8d} {us/ss:>8.1%} {ss:>6.0f}s | "
          f"{s.words_lost:>4d} {s.onset_excluded:>7.1%} {s.pause_excluded:>7.1%} "
          f"{t.total_waste:>6.1f}s {t.clipped_onsets:>5d} | "
          + "  ".join(f"{n[:6]} floor={m:>6.1f} snr={q:>5.1f}" for n, m, q in floor_stats))


if __name__ == "__main__":
    main()
