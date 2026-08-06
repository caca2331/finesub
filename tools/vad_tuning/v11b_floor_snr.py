"""Does a candidate floor actually sit where the residual noise can be rejected?

The whole floor investigation started from one number: on separated miyako the frames
silero calls non-speech sat 33 dB over the tracked floor, so `floor + 6` could not
reject any of them. A variant that costs recall has to buy that number down, or it is
paying for nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC, silero_probs  # noqa: E402
from energy_sweep import cached_tracks  # noqa: E402
from floor_variants import floor_with_tracker  # noqa: E402
from v11_floor_ab import variants  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--cache-dir", required=True)
    args = ap.parse_args()

    path = Path(args.audio)
    cache = Path(args.cache_dir)
    tr = cached_tracks(path, cache)
    e = tr.energy_db.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    probs = silero_probs(path, cache / f"silero-{path.stem}.npz")

    n = min(len(e), int(len(probs) * SILERO_HOP_SEC / 0.01))
    idx = (np.arange(n) * 0.01 / SILERO_HOP_SEC).astype(int).clip(0, len(probs) - 1)
    nonspeech = probs[idx] < 0.2

    print(f"{'variant':<30} {'p50':>6} {'p75':>6} {'p90':>6} {'floor p50':>10}")
    for label, fspec, tspec in variants():
        fl = floor_with_tracker(e, starts, tr.duration, fspec, tspec)
        snr = (e - fl)[:n][nonspeech]
        print(f"{label:<30} {np.quantile(snr, 0.5):>6.1f} {np.quantile(snr, 0.75):>6.1f} "
              f"{np.quantile(snr, 0.90):>6.1f} {np.median(fl):>10.1f}")


if __name__ == "__main__":
    main()
