"""Churn quantum: how much hand-word missingness moves when exactly ONE
human-confirmed junk interval is removed from the production VAD output.

GROUP_TARGET_SEC jitter turned out to be a no-op (packing is gap-anchored and
the pipeline is fully deterministic), so the churn any interval-set change pays
has to be calibrated by the change's own unit: one dropped interval. Four arms
each drop a different junk interval; each arm's delta against the same-code
base measures the per-interval churn quantum.

  python v30b_single_drop.py --audio <gold vocal> --outdir <dir> \
      --drops 79.5,110.2,156.5,216.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--drops", required=True,
                    help="comma list of times (sec); each arm drops the one "
                         "interval containing that time")
    ap.add_argument("--gpu-budget-gb", type=int, default=8)
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    from asr_playground.speech.preprocessing import vad as vad_detection
    from asr_playground.speech.recognition import stage as recog_stage

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    real_detect = vad_detection.detect_segments

    for t_drop in (float(x) for x in args.drops.split(",")):

        def patched(input_path: Path, _t=t_drop):
            raw, meta, duration, timing, track = real_detect(input_path)
            kept = [x for x in raw
                    if not (float(x["start"]) <= _t <= float(x["end"]))]
            print(f"[single-drop] t={_t}: {len(raw)} -> {len(kept)} intervals")
            assert len(kept) == len(raw) - 1, "expected exactly one interval hit"
            return kept, meta, duration, timing, track

        recog_stage.vad_detection.detect_segments = patched
        tag = f"drop{t_drop:g}".replace(".", "p")
        output = outdir / f"{audio.stem}-{tag}-aligned.json"
        t0 = time.perf_counter()
        recog_stage.run_vad_asr(
            input_path=audio, output_path=output, model_name="large-v3-turbo",
            device="cuda", language=args.language,
            gpu_budget_gb=args.gpu_budget_gb,
        )
        print(f"[single-drop] t={t_drop} -> {output.name} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    recog_stage.vad_detection.detect_segments = real_detect


if __name__ == "__main__":
    main()
