"""Convert pipeline JSON segments to TOON, or decode TOON back to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from toon_format import decode, encode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pipeline JSON to TOON (or decode TOON back to JSON)."
    )
    parser.add_argument("input", help="Path to input JSON/TOON file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output file (default: <input>.toon or <input>.json for --decode).",
    )
    parser.add_argument(
        "-d",
        "--decode",
        action="store_true",
        help="Decode TOON back to JSON (default: False).",
    )
    return parser.parse_args()


def extract_segments(data: Any) -> List[Dict[str, object]]:
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list or contain a 'segments' list.")
    out: List[Dict[str, object]] = []
    for seg in data:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text") or ""
        if start is None or end is None:
            continue
        out.append({"start": start, "end": end, "text": text})
    return out


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    if args.decode:
        output_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else input_path.with_suffix(".json")
        )
        toon_text = input_path.read_text(encoding="utf-8")
        obj = decode(toon_text)
        output_path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {output_path}")
        return 0

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_path.with_suffix(".toon")
    )
    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = extract_segments(data)
    toon_text = encode(segments, {"delimiter":"|"})
    output_path.write_text(toon_text, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
