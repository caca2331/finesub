"""floor = percentile(short window) + offset, swept over its three parameters.

No tracker at all: the floor is the rolling percentile itself. That is the opposite
end of the design space from production, where the floor is 99.7% raw frame -- and
appendix P showed the long-window version of this collapses. A *short* window is the
interesting case, because a 1 s neighbourhood is short enough to sit inside the local
background rather than averaging a whole minute of it.

One thing to know before reading the sweep: `offset` and the detector's enter margin
are the same knob. The test is `e <= floor + margin`, so `p10 + 2` with margin 6 is
identical to `p10 + 0` with margin 8. Only two parameters are really free -- the
window and the percentile -- plus their combined threshold. The offset is swept
anyway because it is how the question was posed, and because reading it as "how far
over the local percentile does the cut sit" is more intuitive than a margin.

Windowing is stepped, not interpolated, matching what production now does.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import torch  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def rolling_percentile(energy: np.ndarray, starts: np.ndarray, duration: float,
                       win_sec: float, pct: float) -> np.ndarray:
    """Centered rolling percentile on an anchor grid, held flat between anchors."""
    hop = max(0.05, min(0.1, win_sec / 10.0))
    n = max(1, int(math.floor(duration / hop)) + 1)
    anchors = np.clip(np.arange(n) * hop, 0.0, duration)
    half = win_sec / 2.0
    lo = np.searchsorted(starts, np.clip(anchors - half, 0.0, None))
    hi = np.searchsorted(starts, anchors + half, side="right")
    q = pct / 100.0
    vals = np.empty(n, dtype=np.float64)
    fallback = float(np.quantile(energy, q))
    for k in range(n):
        a, b = int(lo[k]), int(hi[k])
        vals[k] = float(np.quantile(energy[a:b], q)) if b > a else fallback
    idx = np.clip(np.searchsorted(anchors, starts, side="right") - 1, 0, n - 1)
    return vals[idx]


def speech_with_floor(tr, floor: np.ndarray):
    from asr_playground.speech.preprocessing import energy as E

    raw = E._score_to_non_speech_intervals(
        tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
        tr.frame_starts, tr.frame_ends, tr.duration,
        enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
    ns = E._apply_negative_padding(raw, tr.duration)
    return [(float(a), float(b))
            for a, b in E.invert_intervals(ns, tr.duration) if b > a]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--asr", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--word-srt", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--windows", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--pcts", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.0, 2.0, 6.0])
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)

    combos: List[Tuple[float, float, float]] = [
        (w, p, o) for w in args.windows for p in args.pcts for o in args.offsets]
    tot = {c: [0, 0.0, 0.0, 0] for c in combos}
    guard: Dict[tuple, list] = {}
    ann_tr = None

    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        e = tr.energy_db.numpy().astype(np.float64)
        st = tr.frame_starts.numpy().astype(np.float64)
        cache: Dict[Tuple[float, float], np.ndarray] = {}
        for w, p, o in combos:
            base = cache.get((w, p))
            if base is None:
                base = rolling_percentile(e, st, tr.duration, w, p)
                cache[(w, p)] = base
            sp = speech_with_floor(tr, base + o)
            v = noise_intervals(sp, wm)
            t = tot[(w, p, o)]
            t[0] += v.empty_intervals
            t[1] += v.unvoiced_sec
            t[2] += v.speech_sec
            t[3] += len(sp)
            if name == args.annotated:
                guard[(w, p, o)] = sp
        if name == args.annotated:
            ann_tr = tr
        print(f"  done {name}", file=sys.stderr, flush=True)

    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))
    hdr = (f"{'win/pct/off':<14} | {'区间数':>6} {'emptyInt':>8} {'unvoic%':>8} "
           f"{'speech':>7} | {'lost':>4} {'onsetX':>7} {'pauseX':>7} {'waste':>7} "
           f"{'cutOn':>5}")
    print(hdr)
    print(f"{'生产基线':<14} | {2064:>6d} {271:>8d} {'26.3%':>8} {'4501s':>7} | "
          f"{1:>4d} {'0.0%':>7} {'50.0%':>7} {'71.0s':>7} {3:>5d}")
    print("-" * len(hdr))
    for c in combos:
        ei, us, ss, ni = tot[c]
        sp = guard[c]
        s = score(sp, hw, ann_tr.duration, pause)
        t = tightness(sp, hw)
        label = f"{c[0]:g}s/p{c[1]:g}/+{c[2]:g}"
        print(f"{label:<14} | {ni:>6d} {ei:>8d} {us/ss:>8.1%} {ss:>6.0f}s | "
              f"{s.words_lost:>4d} {s.onset_excluded:>7.1%} {s.pause_excluded:>7.1%} "
              f"{t.total_waste:>6.1f}s {t.clipped_onsets:>5d}")


if __name__ == "__main__":
    main()
