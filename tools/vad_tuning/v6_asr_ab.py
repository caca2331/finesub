"""Run the real ASR stage twice on one file, changing only which VAD supplied the
speech intervals.

Everything downstream of `detect_segments` is untouched: alignment, the DP splitter
and the energy annotation all run exactly as in production. Only `raw_segments` is
swapped, and the energy track is kept in both arms so segment energy stays
comparable.

Intended for audio where the energy detector is suspected to be misled -- weak vocal
noise that separation cannot remove.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from backends import SILERO_HOP_SEC, Hysteresis, silero_probs  # noqa: E402
from hybrid import DropGhostIntervals  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--arm", required=True,
                    choices=("energy", "orig", "silero", "dropghost", "adaptive",
                             "rescue"))
    ap.add_argument("--rescue-npy", default=None,
                    help="rescue arm: .npy of (n,2) intervals unioned onto the "
                         "production VAD output (the loud+silero rescue channel)")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--gpu-budget-gb", type=int, default=8)
    ap.add_argument("--silero-enter", type=float, default=0.5)
    ap.add_argument("--silero-exit", type=float, default=0.35)
    ap.add_argument("--silero-pad", type=float, default=0.10)
    ap.add_argument("--pad-right-ms", type=float, default=None,
                    help="override NEGATIVE_PAD_RIGHT_MS, to isolate it from the "
                         "other differences between the arms")
    ap.add_argument("--suffix", default="", help="extra tag in the output filename")
    ap.add_argument("--group-left-lead-sec", type=float, default=None,
                    help="real audio prepended to each interval at group assembly")
    args = ap.parse_args()

    from asr_playground.speech.preprocessing import vad as vad_detection
    from asr_playground.speech.recognition import stage as recog_stage

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{audio.stem}-{args.arm}{args.suffix}-aligned.json"

    if args.group_left_lead_sec is not None:
        from asr_playground.speech.recognition import transcribe as TR
        TR.GROUP_LEFT_LEAD_SEC = float(args.group_left_lead_sec)
        print(f"[group] GROUP_LEFT_LEAD_SEC = {args.group_left_lead_sec}")

    if args.pad_right_ms is not None:
        from asr_playground.speech.preprocessing import energy as E0
        E0.NEGATIVE_PAD_RIGHT_MS = float(args.pad_right_ms)
        print(f"[pad] NEGATIVE_PAD_RIGHT_MS = {args.pad_right_ms}")

    if args.arm == "orig":
        # The energy VAD as it was before this branch: legacy noise floor (no
        # silence exclusion, no clamp), MERGE_GAP_MS 100, no minimum-speech-run
        # split, NEGATIVE_PAD_RIGHT_MS 140. Patched at the one function both the
        # streamed and in-memory paths route through.
        import numpy as np

        from asr_playground.speech.preprocessing import energy as E
        from floor_lab import legacy as legacy_floor

        if args.pad_right_ms is None:
            E.NEGATIVE_PAD_RIGHT_MS = 140.0
        E.MERGE_GAP_MS = 100.0
        fl = legacy_floor()

        def orig_from_tracks(frame_dbfs, energy_db, frame_starts, frame_ends,
                             duration_sec, **kw):
            import torch
            floor = fl(energy_db.numpy().astype(np.float64),
                       frame_starts.numpy().astype(np.float64), duration_sec)
            return E._score_to_non_speech_intervals(
                energy_db, torch.from_numpy(floor.astype(np.float32)), frame_dbfs,
                frame_starts, frame_ends, duration_sec,
                enter_margin_db=kw.get("snr_enter_margin_db", 6.0),
                weighted=bool(E.WEIGHTED_INTERVAL))

        E._detect_non_speech_intervals_from_tracks = orig_from_tracks
        print("[orig arm] pre-branch VAD: legacy floor, merge100, padR140, no minrun")

    if args.arm == "rescue":
        import numpy as np

        from backends import union as iv_union
        real_detect = vad_detection.detect_segments
        extra = [(float(s), float(e)) for s, e in np.load(args.rescue_npy)]

        def patched_rescue(input_path: Path):
            raw, meta, duration, timing, track = real_detect(input_path)
            base = [(float(x["start"]), float(x["end"])) for x in raw]
            iv = iv_union(base, extra)
            segs = [{"start": s, "end": e} for s, e in iv if e > s]
            meta = dict(meta)
            meta["vad"] = {"backend": "energy+loud-silero-rescue",
                           "rescued": len(extra),
                           "energy_track_kept_for_annotation": True}
            print(f"[rescue arm] {len(base)} energy intervals + {len(extra)} "
                  f"rescued -> {len(segs)}")
            return segs, meta, duration, timing, track

        recog_stage.vad_detection.detect_segments = patched_rescue

    if args.arm in ("silero", "dropghost", "adaptive"):
        real_detect = vad_detection.detect_segments

        def patched(input_path: Path):
            raw, meta, duration, timing, track = real_detect(input_path)
            probs = silero_probs(Path(input_path),
                                 Path(args.cache_dir) / f"silero-{Path(input_path).stem}.npz")
            if args.arm == "silero":
                iv = Hysteresis(args.silero_enter, args.silero_exit,
                                pad=args.silero_pad).apply(probs, SILERO_HOP_SEC, duration)
                backend = {"backend": "silero-vad", "enter": args.silero_enter,
                           "exit": args.silero_exit, "pad": args.silero_pad}
            elif args.arm == "adaptive":
                from adaptive import AdaptiveRefine
                from energy_sweep import compute_tracks
                tr = compute_tracks(Path(input_path))
                base = [(float(x["start"]), float(x["end"])) for x in raw]
                dec = AdaptiveRefine()(base, probs,
                                       tr.energy_db.numpy().astype(float),
                                       tr.noise_floor.numpy().astype(float))
                iv = dec.kept
                backend = {"backend": "energy+silero-adaptive", **dec.stats}
            else:
                base = [(float(x["start"]), float(x["end"])) for x in raw]
                iv = DropGhostIntervals()(base, probs)
                backend = {"backend": "energy-minus-ghost-intervals",
                           "peak_thr": 0.50, "dropped": len(base) - len(iv)}
            segs = [{"start": s, "end": e} for s, e in iv if e > s]
            meta = dict(meta)
            meta["vad"] = {**backend, "energy_track_kept_for_annotation": True}
            print(f"[silero arm] replaced {len(raw)} energy intervals with {len(segs)}")
            return segs, meta, duration, timing, track

        # The stage imports the module, so patching the attribute is enough.
        recog_stage.vad_detection.detect_segments = patched

    t = time.perf_counter()
    recog_stage.run_vad_asr(
        input_path=audio,
        output_path=output,
        model_name=args.model,
        device="cuda",
        language=args.language,
        gpu_budget_gb=args.gpu_budget_gb,
    )
    print(f"[{args.arm}] wrote {output} in {time.perf_counter() - t:.0f}s")


if __name__ == "__main__":
    main()
