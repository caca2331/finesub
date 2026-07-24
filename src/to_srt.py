"""Convert pipeline JSON segments to SRT subtitle files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pipeline JSON to SRT.")
    parser.add_argument("input", help="Path to pipeline output JSON.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output SRT (default: <input>.srt).",
    )
    parser.add_argument(
        "--word",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write word-level SRT (default: False).",
    )
    return parser.parse_args()


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_segment_srt(segments: List[Dict[str, object]]) -> str:
    lines: list[str] = []
    idx = 1
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text") or "\"\""
        if start is None or end is None:
            continue
        try:
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
        lines.append(str(text).strip())
        lines.append("")
        idx += 1
    return "\n".join(lines).strip() + "\n"


def render_word_srt(segments: List[Dict[str, object]]) -> str:
    lines: list[str] = []
    idx = 1
    found_words = False
    for seg in segments:
        words = seg.get("words") or []
        if words:
            found_words = True
        for w in words:
            start = w.get("start")
            end = w.get("end")
            text = w.get("word") or w.get("text") or ""
            if start is None or end is None:
                continue
            try:
                start_s = float(start)
                end_s = float(end)
            except (TypeError, ValueError):
                continue
            if end_s <= start_s:
                continue
            lines.append(str(idx))
            lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
            lines.append(str(text).strip())
            lines.append("")
            idx += 1

    if not found_words:
        raise ValueError("No segments contain words; cannot write word-level SRT.")
    return "\n".join(lines).strip() + "\n"


def convert_json_to_srt(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    word: bool = False,
) -> Path:
    input_path = Path(input_path).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else input_path.with_suffix(".srt")
    )
    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        raise SystemExit("Input JSON must contain a 'segments' list.")

    if word:
        srt_text = render_word_srt(segments)
    else:
        srt_text = render_segment_srt(segments)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_text, encoding="utf-8")
    print(f"Wrote {output_path}")
    return output_path


def main() -> int:
    args = parse_args()
    convert_json_to_srt(args.input, output_path=args.output, word=args.word)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
