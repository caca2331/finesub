"""How much non-vocal residue does each arm leave where the baseline is quiet?

The 16 kHz arm fails by over-suppressing. The suspicion about 22.05 kHz is the
opposite: it under-suppresses, so instrumental and laughter survive into the
vocal track, VAD calls it speech, and Whisper invents words for it.

Measured on the seconds the 44.1k baseline judged empty.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

QUIET_DB = -60.0


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


def main() -> int:
    baseline = curve(Path(sys.argv[1]))
    quiet = baseline < QUIET_DB
    print(f"baseline quiet seconds (<{QUIET_DB:.0f} dB): {int(quiet.sum())}")
    print(f"{'arm':>34} | {'median':>7} {'p90':>7} {'>-45dB':>7}")
    print("-" * 62)
    for candidate in [sys.argv[1]] + sys.argv[2:]:
        arm = curve(Path(candidate))
        n = min(len(arm), len(baseline))
        window = arm[:n][quiet[:n]]
        loud = int((window > -45.0).sum())
        print(f"{Path(candidate).name:>34} | {np.median(window):>7.1f} "
              f"{np.percentile(window, 90):>7.1f} {loud:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
