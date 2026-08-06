"""The composed "final form" detector under evaluation, in one reproducible place.

    -45 absolute peak floor      (production, default on)
  + exitrun4 accumulator         (candidate default: blips forgiven)
  + gcap10 voicing-gated cap     (candidate silero opt-in: creep un-suppression)
  + ghost-drop                   (shipped silero opt-in)

Resolved after user review: symmetric silero dilation opened the cap's gate
0.3 s before voicing onset, so segment starts drifted early over the noise
prefix (12% of gold intervals >50 ms early, median 0.22 s). Asymmetric
dilation (left 0, right 0.3) removes the drift at zero measured cost, and is
now the default.

  python v32_final_form.py --audio <vocal> --srt-out <path> [--dilate-left 0.3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC, silero_probs  # noqa: E402
from energy_sweep import Tracks, cached_tracks  # noqa: E402
from floor_decomposed import Decomposed  # noqa: E402
from v31_exit_run_and_gated_cap import score_exit_run  # noqa: E402
from asr_playground.speech.preprocessing import energy as E  # noqa: E402
from asr_playground.speech.preprocessing import silero_ghost  # noqa: E402

Interval = Tuple[float, float]

CAP_DB = 10.0
EXIT_RUN = 4
SIL_THR = 0.5


def final_form(tr: Tracks, sil: np.ndarray, *, dilate_left: float = 0.0,
               dilate_right: float = 0.3) -> Tuple[List[Interval], dict]:
    e = tr.energy_db.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    hop = float(starts[1] - starts[0])
    anchor = Decomposed()._anchor(e, hop)
    idx = np.clip((starts / SILERO_HOP_SEC).astype(int), 0, len(sil) - 1)
    voiced = sil[idx] >= SIL_THR
    vd = voiced.copy()
    for s in range(1, max(1, int(dilate_right / hop)) + 1):
        vd[s:] |= voiced[:-s]          # extend后向: after voicing ends
    for s in range(1, max(1, int(dilate_left / hop)) + 1):
        vd[:-s] |= voiced[s:]          # extend前向: before voicing starts
    floor = tr.noise_floor.numpy().astype(np.float64).copy()
    gate = vd & (floor > anchor + CAP_DB)
    floor[gate] = anchor[gate] + CAP_DB
    sp = score_exit_run(tr, EXIT_RUN, floor=floor)

    hop_sec, frame_sec = E._frame_grid_seconds()
    vtrack = E.VadEnergyTrack(energy_db=tr.energy_db, hop_sec=hop_sec,
                              frame_sec=frame_sec, energy_mode="weighted")
    segs = [{"start": s, "end": e_} for s, e_ in sp]
    kept, stats = silero_ghost.drop_ghost_segments(segs, sil, vtrack)
    sp = [(float(x["start"]), float(x["end"])) for x in kept]

    # inside-interval passes: trim/split on noise verdicts, then give merged
    # production seams back to the splitter (v33)
    from energy_sweep import speech_from_tracks
    from v33_partial_apply import carve_intervals, restore_seams
    prod = speech_from_tracks(tr)
    sp, st1 = carve_intervals(sp, e, starts, sil, ceiling_db=-45.0,
                              use_silero_evidence=False)
    sp, st2 = carve_intervals(sp, e, starts, sil, ceiling_db=0.0,
                              use_silero_evidence=True)
    sp, st3 = restore_seams(sp, prod, e, starts, sil)
    stats = {**stats, "carve_certain": st1, "carve_silero": st2, "seams": st3}
    return sp, stats


def main() -> None:
    from asr_playground.subtitles.rendering import format_srt_time as ts

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt-out", required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--dilate-left", type=float, default=0.0)
    ap.add_argument("--dilate-right", type=float, default=0.3)
    args = ap.parse_args()

    vocal = Path(args.audio)
    tr = cached_tracks(vocal, args.cache_dir)
    sil = silero_probs(vocal, args.cache_dir / f"silero-{vocal.stem}.npz")
    sp, stats = final_form(tr, sil, dilate_left=args.dilate_left,
                           dilate_right=args.dilate_right)
    lines = []
    for j, (s, e_) in enumerate(sp, 1):
        lines.append(f"{j}\n{ts(s)} --> {ts(e_)}\n#{j} {e_ - s:.2f}s\n")
    Path(args.srt_out).write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(sp)} intervals, ghost-drop removed {stats['dropped']} "
          f"({stats['dropped_sec']:.1f}s) -> {args.srt_out}")


if __name__ == "__main__":
    main()
