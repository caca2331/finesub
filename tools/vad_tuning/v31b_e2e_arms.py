"""Run the production ASR with a precomputed interval set injected.

Generic successor to v6's arms: any candidate detector variant is computed
offline, saved as an (n,2) .npy, and replayed here through the untouched
production stage -- so the whole re-admitted audio (speech and noise alike)
goes through the real decoder, per the user's requirement that final-output
impact, not interval bookkeeping, is what judges a change.

  python v31b_e2e_arms.py --audio <vocal> --outdir <dir> \
      --arm base=iv-gold-base.npy --arm gcap10=iv-gold-gcap10.npy
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
    ap.add_argument("--arm", action="append", required=True, metavar="TAG=NPY")
    ap.add_argument("--gpu-budget-gb", type=int, default=8)
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    import numpy as np

    from asr_playground.speech.preprocessing import vad as vad_detection
    from asr_playground.speech.recognition import stage as recog_stage

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    real_detect = vad_detection.detect_segments

    for spec in args.arm:
        tag, npy = spec.split("=", 1)
        iv = [(float(s), float(e)) for s, e in np.load(npy)]

        def patched(input_path: Path, _iv=iv, _tag=tag):
            raw, meta, duration, timing, track = real_detect(input_path)
            segs = [{"start": s, "end": e} for s, e in _iv if e > s]
            meta = dict(meta)
            meta["vad"] = {**(meta.get("vad") or {}), "injected_arm": _tag,
                           "injected_n": len(segs)}
            print(f"[{_tag}] injected {len(segs)} intervals "
                  f"(detector produced {len(raw)})")
            return segs, meta, duration, timing, track

        recog_stage.vad_detection.detect_segments = patched
        output = outdir / f"{audio.stem}-{tag}-aligned.json"
        t0 = time.perf_counter()
        recog_stage.run_vad_asr(
            input_path=audio, output_path=output, model_name="large-v3-turbo",
            device="cuda", language=args.language,
            gpu_budget_gb=args.gpu_budget_gb,
        )
        print(f"[{tag}] -> {output.name} ({time.perf_counter() - t0:.0f}s)",
              flush=True)
    recog_stage.vad_detection.detect_segments = real_detect


if __name__ == "__main__":
    main()
