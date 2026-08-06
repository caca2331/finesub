"""Calibrate the end-to-end noise band for packing-only perturbations.

Same audio, same VAD intervals, same decode config -- only GROUP_TARGET_SEC
moves a little, so the interval-into-group packing reshuffles and the decoder
sees different context windows. The spread of hand-confirmed-word missingness
across these arms is the churn floor that any interval-set change (appendix
H/U/W) must clear before its end-to-end delta means anything. Deterministic
reruns measure zero, so this perturbation is the honest replicate.

  python v30_churn_calibration.py --audio <gold vocal> --outdir <dir> \
      --cache-dir <cache> --targets 28.5,29.5,30.5,31.5
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
    ap.add_argument("--targets", default="28.5,29.5,30.5,31.5")
    ap.add_argument("--gpu-budget-gb", type=int, default=8)
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    from asr_playground.speech.recognition import stage as recog_stage
    from asr_playground.speech.recognition import transcribe as TR

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for target in (float(x) for x in args.targets.split(",")):
        TR.GROUP_TARGET_SEC = target
        tag = f"gt{target:g}".replace(".", "p")
        output = outdir / f"{audio.stem}-{tag}-aligned.json"
        t0 = time.perf_counter()
        recog_stage.run_vad_asr(
            input_path=audio,
            output_path=output,
            model_name="large-v3-turbo",
            device="cuda",
            language=args.language,
            gpu_budget_gb=args.gpu_budget_gb,
        )
        print(f"[churn] GROUP_TARGET_SEC={target} -> {output.name} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
