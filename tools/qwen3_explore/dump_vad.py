"""Dump the production VAD speech intervals so both ASR arms share one segmentation input.

Runs in the production `asr` env (imports `vad_energy`). Both arms must be scored against
the same interval set, otherwise a 断句 comparison is measuring VAD noise.

    C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.qwen3_explore.dump_vad \
        --audio out/reference/<id>/<id>-vocal.flac --out out/qwen-explore/<id>-vad.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from vad_energy import detect_non_speech_intervals_file, invert_intervals  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    non_speech, duration, _track, _hints = detect_non_speech_intervals_file(args.audio)
    speech = invert_intervals(non_speech, duration)
    payload = {
        "audio": str(args.audio),
        "duration": round(float(duration), 3),
        "speech": [[round(float(a), 3), round(float(b), 3)] for a, b in speech],
        "non_speech": [[round(float(a), 3), round(float(b), 3)] for a, b in non_speech],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    total = sum(b - a for a, b in speech)
    print(f"{args.out}: {len(speech)} speech intervals, {total:.1f}s speech / {duration:.1f}s")


if __name__ == "__main__":
    main()
