"""Put the Qwen word stream through the production `segment_split` DP.

Both arms must be cut by the same algorithm with the same VAD intervals, otherwise a 断句
comparison is really comparing two splitters. Runs in the production `asr` env.

    C:/Users/Carl/miniconda3/envs/asr/python.exe -m tools.qwen3_explore.apply_split \
        --raw out/qwen-explore/<id>-Q-rescued-raw.json --vad out/qwen-explore/<id>-vad.json \
        --out out/qwen-explore/<id>-Q-aligned.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from segment_split import DEFAULT_SPLIT_PARAMS, split_params_metadata, split_segments  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--base",
        type=float,
        default=None,
        help="override the per-cut base cost; production is 1.0 and sets the ~8.5-9s cut threshold",
    )
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    intervals = [{"start": float(a), "end": float(b)} for a, b in vad["speech"]]

    params = DEFAULT_SPLIT_PARAMS
    if args.base is not None:
        params = replace(params, base=args.base)

    segments = [s for s in raw["segments"] if s.get("words")]
    split = split_segments(segments, intervals, params=params)

    raw.setdefault("metadata", {}).setdefault("asr_align", {})["segment_split"] = split_params_metadata(params)
    raw["segments"] = split
    Path(args.out).write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.out}: {len(segments)} windows -> {len(split)} segments")


if __name__ == "__main__":
    main()
