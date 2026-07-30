"""Pair divergent segments with the human-refined subtitle so a human can call the winner.

The refined SRT is the translated, hand-corrected final product, so it is not a
transcription reference — but it is the only human-verified statement of what was
actually said, which is enough to adjudicate most Whisper-vs-Qwen disagreements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_aligned, normalize_ja, read_srt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=path")
    ap.add_argument("--refined-srt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_aligned(args.baseline_aligned)["segments"]
    arms = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        arms[label] = json.loads(Path(path).read_text(encoding="utf-8"))["segments"]
    cues = read_srt(args.refined_srt)

    lines = ["# Adjudication sheet (divergent segments only)\n"]
    lines.append("`REF` is the human-refined translation of whatever overlaps this window.\n")
    for i, seg in enumerate(base):
        variants = {label: segs[i]["text"] for label, segs in arms.items() if i < len(segs)}
        if all(normalize_ja(v) == normalize_ja(seg.get("text", "")) for v in variants.values()):
            continue
        overlap = [
            c.text.replace("\n", " ")
            for c in cues
            if c.end > seg["start"] and c.start < seg["end"]
        ]
        lines.append(f"**[{i:03d}] {seg['start']:.2f}–{seg['end']:.2f}**")
        lines.append(f"- BASE: {seg.get('text', '')}")
        for label, text in variants.items():
            lines.append(f"- {label}: {text}")
        lines.append(f"- REF: {' / '.join(overlap) if overlap else '(no cue)'}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
