"""What the two windows actually do, as opposed to what they score.

`gain`   how much work the local RMS normalizer is doing and how often it runs into
         its own [-4, +6] dB limit. A window so wide that every anchor gets the same
         gain is not "local" at all.
`p90`    the diagnostic the whole floor investigation started from: how far the
         frames silero calls non-speech sit above the tracked floor on noisy audio.
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
from backends import SILERO_HOP_SEC, silero_probs  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from floor_variants import floor_with_tracker  # noqa: E402
from v14_windows import specs  # noqa: E402


def gain_stats(path: Path, norm_win: float):
    from asr_playground.speech.preprocessing import energy as E

    saved = E.NORM_WINDOW_SEC
    try:
        E.NORM_WINDOW_SEC = norm_win
        wav = E._load_asr_audio_streamed(str(path))
        x = wav.float()
        x = x - E._dc_mean32(x, E.TARGET_SR)
        from asr_playground.speech.preprocessing.energy import AF
        if E.HPF_ENABLE and AF is not None:
            x = AF.highpass_biquad(x.unsqueeze(0), E.TARGET_SR, E.HPF_HZ).squeeze(0)
        n = int(x.numel())
        _w, hop, _h = E._norm_geometry(E.TARGET_SR)
        anchors = E._full_anchor_list(n, hop)
        g = E._anchor_gains(x, 0, n, E.TARGET_SR, anchors).numpy()
    finally:
        E.NORM_WINDOW_SEC = saved
    db = 20.0 * np.log10(np.clip(g, 1e-9, None))
    return dict(n=len(db), med=float(np.median(db)),
                p5=float(np.quantile(db, 0.05)), p95=float(np.quantile(db, 0.95)),
                at_max=float((db > E.NORM_MAX_GAIN_DB - 0.01).mean()),
                at_min=float((db < E.NORM_MIN_GAIN_DB + 0.01).mean()),
                spread=float(np.quantile(db, 0.95) - np.quantile(db, 0.05)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--noisy")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--norm-windows", type=float, nargs="+", default=[120.0, 300.0])
    ap.add_argument("--floor-windows", type=float, nargs="+", default=[120.0, 300.0])
    args = ap.parse_args()

    clips = dict(x.split("=", 1) for x in args.clip)
    cache = Path(args.cache_dir)

    print(f"{'clip':<16} {'normWin':>7} {'anchors':>8} {'gain p5':>8} {'med':>6} "
          f"{'p95':>6} {'spread':>7} {'at +6':>6} {'at -4':>6}")
    for name, path in clips.items():
        for nw in args.norm_windows:
            s = gain_stats(Path(path), nw)
            print(f"{name:<16} {nw:>7.0f} {s['n']:>8d} {s['p5']:>8.1f} {s['med']:>6.1f} "
                  f"{s['p95']:>6.1f} {s['spread']:>7.1f} {s['at_max']:>6.1%} "
                  f"{s['at_min']:>6.1%}")

    if not args.noisy:
        return
    from asr_playground.speech.preprocessing import energy as E

    p = Path(args.noisy)
    probs = silero_probs(p, cache / f"silero-{p.stem}.npz")
    print()
    print(f"{'noisy: norm':>11} {'floor':>6} {'algo':<8} {'p75':>6} {'p90':>6} "
          f"{'floor med':>10} {'speech':>7}")
    saved = E.NORM_WINDOW_SEC
    try:
        for nw in args.norm_windows:
            E.NORM_WINDOW_SEC = nw
            tr = cached_tracks(p, cache / f"norm{int(nw)}")
            e = tr.energy_db.numpy().astype(np.float64)
            starts = tr.frame_starts.numpy().astype(np.float64)
            n = min(len(e), int(len(probs) * SILERO_HOP_SEC / 0.01))
            idx = (np.arange(n) * 0.01 / SILERO_HOP_SEC).astype(int).clip(0, len(probs) - 1)
            mask = probs[idx] < 0.2
            for fw in args.floor_windows:
                sp_by_algo = specs(fw)
                for algo in ("legacy", "current"):
                    fspec, tspec = sp_by_algo[algo]
                    fl = floor_with_tracker(e, starts, tr.duration, fspec, tspec)
                    snr = (e - fl)[:n][mask]
                    raw = E._score_to_non_speech_intervals(
                        tr.energy_db, torch.from_numpy(fl.astype(np.float32)),
                        tr.frame_dbfs, tr.frame_starts, tr.frame_ends, tr.duration,
                        enter_margin_db=6.0, weighted=True)
                    ns = E._apply_negative_padding(raw, tr.duration)
                    sp = [(float(a), float(b)) for a, b
                          in E.invert_intervals(ns, tr.duration) if b > a]
                    print(f"{nw:>11.0f} {fw:>6.0f} {algo:<8} "
                          f"{np.quantile(snr, .75):>6.1f} {np.quantile(snr, .90):>6.1f} "
                          f"{np.median(fl):>10.1f} "
                          f"{sum(b - a for a, b in sp)/tr.duration:>7.1%}")
    finally:
        E.NORM_WINDOW_SEC = saved


if __name__ == "__main__":
    main()
