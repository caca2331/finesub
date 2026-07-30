"""Second verifier for `spotcheck.py`, run in the production `asr` env (Whisper, not Qwen).

Keeping the verifier off the Qwen encoder family is the point: it stops the boundary
comparison from resting on a model that shares an audio front-end with the aligner.

    C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.qwen3_explore.verify_whisper \
        --clips out/qwen-explore/spotcheck-<id>/clips.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import whisper

from .common import cer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="ja")
    args = ap.parse_args()

    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    model = whisper.load_model(args.model, device="cuda")

    for clip in clips:
        result = model.transcribe(
            clip["wav"], language=args.language, beam_size=None, temperature=0.0, verbose=False
        )
        text = "".join(seg["text"] for seg in result["segments"]).strip()
        clip["whisper_asr"] = text
        clip["whisper_asr_cer"] = round(cer(clip["text"], text), 3)

    Path(args.clips).write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")

    for key in ("qwen_asr_cer", "whisper_asr_cer"):
        print(f"verifier={key.split('_')[0]}")
        for system in ("qwen", "baseline"):
            vals = [c[key] for c in clips if c["system"] == system and key in c]
            if vals:
                print(
                    f"  {system:9s} n={len(vals)} mean CER={statistics.mean(vals):.3f} "
                    f"median={statistics.median(vals):.3f} "
                    f"exact={sum(1 for v in vals if v == 0)}/{len(vals)}"
                )


if __name__ == "__main__":
    main()
