"""What if frame_dbfs were 10*log10(rms) instead of 20*log10(rms)?

`_linear_to_db` is used in exactly one place, `_frame_dbfs`, so the change is
confined to the absolute gate -- `energy_db` is built from band powers and never
passes through it. And the two conventions differ by an exact factor of two,
clamp included (20*log10(1e-10) = -200, half of which is -100), so the alternative
track is `dbfs / 2` and nothing has to be recomputed.

Worth asking because the gate is currently almost inert: measured over four clips it
vetoes 0.0-0.2% of the frames the relative condition accepts, which is why moving
ABS_NON_SPEECH_MAX_DBFS_ENTER from -30 to -22 changed delivered non-speech by less
than 0.1pp. Halving the scale is one way to give it teeth -- the question is whether
the teeth land anywhere useful.
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
from precision import noise_intervals, tightness, word_map_from  # noqa: E402
from refs import load_pause_ref, load_word_srt  # noqa: E402
from score import score  # noqa: E402


def speech_with_dbfs(tr, dbfs: np.ndarray, abs_enter: float, abs_exit: float):
    from asr_playground.speech.preprocessing import energy as E

    saved = (E.ABS_NON_SPEECH_MAX_DBFS_ENTER, E.ABS_NON_SPEECH_MAX_DBFS_EXIT)
    try:
        E.ABS_NON_SPEECH_MAX_DBFS_ENTER = abs_enter
        E.ABS_NON_SPEECH_MAX_DBFS_EXIT = abs_exit
        raw = E._score_to_non_speech_intervals(
            tr.energy_db, tr.noise_floor,
            torch.from_numpy(dbfs.astype(np.float32)),
            tr.frame_starts, tr.frame_ends, tr.duration,
            enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
        ns = E._apply_negative_padding(raw, tr.duration)
    finally:
        E.ABS_NON_SPEECH_MAX_DBFS_ENTER, E.ABS_NON_SPEECH_MAX_DBFS_EXIT = saved
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
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    asr: Dict[str, List[str]] = {}
    for spec in args.asr:
        k, v = spec.split("=", 1)
        asr.setdefault(k, []).append(v)

    # (label, dbfs scale, enter, exit)
    arms = [("20log10 (生产)", 1.0, -30.0, -28.0),
            ("10log10, 门槛不动", 0.5, -30.0, -28.0),
            ("10log10, 门槛也减半", 0.5, -15.0, -14.0),
            ("10log10, enter -20", 0.5, -20.0, -19.0),
            ("10log10, enter -25", 0.5, -25.0, -24.0)]

    tot = {a[0]: [0, 0.0, 0.0, 0] for a in arms}
    guard: Dict[str, list] = {}
    ann_tr = None
    print(f"{'clip':<14} {'条件重叠：相对':>12} {'绝对(20log)':>12} {'绝对(10log)':>12} "
          f"{'被20log否决':>12} {'被10log否决':>12}")
    for name, path in clips.items():
        tr = cached_tracks(Path(path), Path(args.cache_dir))
        wm = word_map_from(asr[name])
        e = tr.energy_db.numpy().astype(np.float64)
        fl = tr.noise_floor.numpy().astype(np.float64)
        d20 = tr.frame_dbfs.numpy().astype(np.float64)
        rel = e <= fl + 6.0
        a20, a10 = d20 <= -30.0, (d20 * 0.5) <= -30.0
        print(f"{name:<14} {rel.mean():>12.1%} {a20.mean():>12.1%} {a10.mean():>12.1%} "
              f"{float((rel & ~a20).mean()):>12.1%} {float((rel & ~a10).mean()):>12.1%}")
        for label, scale, enter, exit_ in arms:
            sp = speech_with_dbfs(tr, d20 * scale, enter, exit_)
            v = noise_intervals(sp, wm)
            t = tot[label]
            t[0] += v.empty_intervals
            t[1] += v.unvoiced_sec
            t[2] += v.speech_sec
            t[3] += len(sp)
            if name == args.annotated:
                guard[label] = sp
        if name == args.annotated:
            ann_tr = tr

    hw = load_word_srt(Path(args.word_srt))
    pause = load_pause_ref(Path(args.gold))
    print()
    hdr = (f"{'arm':<22} | {'区间数':>6} {'emptyInt':>8} {'unvoic%':>8} {'speech':>7} | "
           f"{'lost':>4} {'onsetX':>7} {'pauseX':>7} {'waste':>7} {'cutOn':>5}")
    print(hdr)
    print("-" * len(hdr))
    for label, *_ in arms:
        ei, us, ss, ni = tot[label]
        sp = guard[label]
        s = score(sp, hw, ann_tr.duration, pause)
        t = tightness(sp, hw)
        print(f"{label:<22} | {ni:>6d} {ei:>8d} {us/ss:>8.1%} {ss:>6.0f}s | "
              f"{s.words_lost:>4d} {s.onset_excluded:>7.1%} {s.pause_excluded:>7.1%} "
              f"{t.total_waste:>6.1f}s {t.clipped_onsets:>5d}")


if __name__ == "__main__":
    main()
