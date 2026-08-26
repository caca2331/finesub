"""Measure, per second, where an arm lost vocals the 44.1k baseline kept.

The catastrophic failure at 16 kHz was not word errors -- it was whole passages
masked to the noise floor. That is visible on the vocal track alone, without
running ASR, so it screens materials cheaply.

Both arms are reduced to a per-second RMS curve in dB. A second is "lost" when
the baseline is clearly voiced and the arm is at or below the floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

VOICED_DB = -45.0
FLOOR_DB = -80.0


def curve(path: Path) -> np.ndarray:
    info = sf.info(str(path))
    rate = info.samplerate
    data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    seconds = int(info.frames // rate)
    rms = np.array(
        [
            np.sqrt((mono[i * rate : (i + 1) * rate] ** 2).mean() + 1e-30)
            for i in range(seconds)
        ]
    )
    return 20 * np.log10(rms + 1e-12)


def compare(baseline: Path, arm: Path):
    a, b = curve(baseline), curve(arm)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lost = (a > VOICED_DB) & (b < FLOOR_DB)
    gained = (b > VOICED_DB) & (a < FLOOR_DB)
    # Longest run of consecutive lost seconds: one 30s hole matters far more
    # than 30 scattered seconds, and only the hole kills a subtitle passage.
    longest = current = 0
    for flag in lost:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return {
        "seconds": n,
        "voiced": int((a > VOICED_DB).sum()),
        "lost": int(lost.sum()),
        "gained": int(gained.sum()),
        "longest_hole": longest,
    }


def main() -> int:
    baseline = Path(sys.argv[1])
    print(f"{'arm':>34} | {'sec':>4} {'voiced':>6} {'lost':>4} {'gained':>6} {'hole':>4}")
    print("-" * 74)
    for candidate in sys.argv[2:]:
        stats = compare(baseline, Path(candidate))
        print(f"{Path(candidate).name:>34} | {stats['seconds']:>4} {stats['voiced']:>6} "
              f"{stats['lost']:>4} {stats['gained']:>6} {stats['longest_hole']:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
