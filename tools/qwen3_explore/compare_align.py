"""Diff Qwen forced-aligner word timings against the baseline's Whisper-DTW timings.

Tokenisations differ (nagisa vs Whisper subwords), so both sides are exploded to one
entry per kept character and compared position by position. Segments whose character
sequences disagree are skipped and counted.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .common import normalize_ja
from .run_align import char_time_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align", required=True, help="run_align.py output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.align).read_text(encoding="utf-8"))

    start_deltas: list[float] = []
    seg_start_deltas: list[float] = []
    seg_end_deltas: list[float] = []
    skipped = 0
    rows: list[tuple[float, int, str, float, float]] = []

    for seg in data["segments"]:
        qwen_chars = char_time_map(seg["words"], "start", "end", "word")
        base_chars = char_time_map(seg["baseline_words"], "start", "end", "word")
        if not qwen_chars or not base_chars:
            skipped += 1
            continue
        if "".join(c for c, _, _ in qwen_chars) != "".join(c for c, _, _ in base_chars):
            skipped += 1
            continue

        deltas = [q[1] - b[1] for q, b in zip(qwen_chars, base_chars)]
        start_deltas.extend(deltas)
        seg_start_deltas.append(qwen_chars[0][1] - base_chars[0][1])
        seg_end_deltas.append(qwen_chars[-1][2] - base_chars[-1][2])
        rows.append(
            (
                max(abs(d) for d in deltas),
                seg["index"],
                seg["text"],
                qwen_chars[0][1] - base_chars[0][1],
                qwen_chars[-1][2] - base_chars[-1][2],
            )
        )

    def stats(values: list[float]) -> str:
        if not values:
            return "n/a"
        absolute = sorted(abs(v) for v in values)
        p = lambda q: absolute[min(len(absolute) - 1, int(q * len(absolute)))]  # noqa: E731
        return (
            f"n={len(values)} mean={statistics.mean(values):+.3f}s "
            f"mean|Δ|={statistics.mean(absolute):.3f}s p50={p(0.5):.3f}s "
            f"p90={p(0.9):.3f}s max={absolute[-1]:.3f}s"
        )

    lines = [
        "# Qwen3-ForcedAligner vs Whisper-DTW word timings\n",
        f"- source: `{args.align}`",
        f"- comparable segments: {len(rows)}, skipped (token mismatch/empty): {skipped}",
        f"- run: RTF {data['metadata']['rtf']}, peak VRAM {data['metadata']['peak_vram_gb']} GB\n",
        "## Per-character start-time delta (Qwen − baseline)\n",
        f"    {stats(start_deltas)}\n",
        "## Segment first-word start delta\n",
        f"    {stats(seg_start_deltas)}\n",
        "## Segment last-word end delta\n",
        f"    {stats(seg_end_deltas)}\n",
        "## Worst 15 segments by max |Δ|\n",
        "| max|Δ| s | seg | first-word Δ | last-word Δ | text |",
        "| --- | --- | --- | --- | --- |",
    ]
    for worst, idx, text, d0, d1 in sorted(rows, reverse=True)[:15]:
        lines.append(f"| {worst:.2f} | {idx} | {d0:+.2f} | {d1:+.2f} | {text[:48]} |")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
