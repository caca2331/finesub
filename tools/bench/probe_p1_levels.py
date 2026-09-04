"""Absolute level of what the VAD admitted -- residue vs real sparse speech.

`bench-baselines.md` 17.5 concluded that coverage alone cannot separate
"separator residue got admitted" from "this recording really is mostly quiet".
17.6b then asked the follow-up: is an ABSOLUTE floor any better? The first
answer was wrong because it used the whole-file median, which on these files is
95% silence and therefore measures the noise floor, not the speech.

The right statistic is the level of the frames the VAD actually admitted. This
probe prints both, in true dBFS (`frame_dbfs`) rather than the weighted
spectral scale, plus the file peak -- the peak matters because an absolute
reading is only meaningful if the separated track is NOT level-normalized.

    python -m tools.bench.probe_p1_levels out/DIR [out/DIR ...]

Each DIR is a stage output directory holding `<stem>-vad.json` and the energy
`.npz` its `energy_track.arrays` names.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator

import numpy as np

FLOORS = (-55.0, -50.0, -45.0, -40.0)


def _load(run_dir: str) -> tuple[float, np.ndarray, np.ndarray] | None:
    """(coverage %, all finite dBFS frames, admitted dBFS frames), or None."""

    stem = os.path.basename(os.path.normpath(run_dir))
    vad_path = os.path.join(run_dir, f"{stem}-vad.json")
    if not os.path.exists(vad_path):
        return None
    with open(vad_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    track = payload.get("energy_track") or {}
    npz_path = os.path.join(run_dir, str(track.get("arrays") or ""))
    if not os.path.exists(npz_path):
        return None
    hop = float(track["hop_sec"])
    with np.load(npz_path) as arrays:
        frames = np.asarray(arrays["frame_dbfs"], dtype=np.float64)
    admitted = np.zeros(frames.shape, dtype=bool)
    for segment in payload["segments"]:
        start = int(float(segment["start"]) / hop)
        admitted[start: int(float(segment["end"]) / hop) + 1] = True
    finite = np.isfinite(frames)
    duration = float(payload["audio_duration"])
    speech = sum(s["end"] - s["start"] for s in payload["segments"])
    return 100.0 * speech / duration, frames[finite], frames[admitted & finite]


def _rows(run_dirs: list[str]) -> Iterator[tuple[str, float, np.ndarray, np.ndarray]]:
    for run_dir in run_dirs:
        loaded = _load(run_dir)
        if loaded is None:
            print(f"  (skipped, no vad.json/npz: {run_dir})")
            continue
        yield os.path.basename(os.path.normpath(run_dir)), *loaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    args = parser.parse_args(argv)

    rows = list(_rows(args.run_dirs))
    header = "  ".join(f"{floor:>7.0f}" for floor in FLOORS)
    print(f"{'run':<16} {'cov%':>6} {'peak':>7} | "
          f"{'admitted p50':>12} {'p95':>7} | above {header}")
    for stem, coverage, every, admitted in rows:
        p50 = f"{np.percentile(admitted, 50):.1f}" if admitted.size else "-"
        p95 = f"{np.percentile(admitted, 95):.1f}" if admitted.size else "-"
        cells = "  ".join(
            f"{100 * float((every > floor).mean()):>7.2f}" for floor in FLOORS
        )
        print(f"{stem:<16} {coverage:>6.2f} {every.max():>7.1f} | "
              f"{p50:>12} {p95:>7} |       {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
