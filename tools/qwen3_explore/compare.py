"""Side-by-side report for the ASR arms produced by `run_asr.py`.

There is no clean per-character ground truth for these clips, so this deliberately
reports *divergence* (how far each arm sits from the Whisper baseline) plus a
keyword-hit table for the proper nouns the knowledge base already tracks. Which arm
is right on a divergent line is settled by hand against the human-refined subtitle.

    python -m tools.qwen3_explore.compare --baseline-aligned ... --arm A1=... --arm A2=... \
        --terms 水仙十字 サンドローネ --out report.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .common import cer, load_aligned, normalize_ja


def load_arm(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=path/to/arm.json")
    ap.add_argument("--terms", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_segments = load_aligned(args.baseline_aligned)["segments"]
    base_text = "".join(s.get("text", "") for s in base_segments)

    arms = {}
    for spec in args.arm:
        label, path = spec.split("=", 1)
        arms[label] = load_arm(path)

    lines: list[str] = ["# Qwen3-ASR vs Whisper baseline\n"]

    # ---- run-level table
    lines.append("## Runs\n")
    lines.append("| arm | windows | audio s | wall s | RTF | peak VRAM GB | context |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label, arm in arms.items():
        m = arm["metadata"]
        lines.append(
            f"| {label} | {m['segments']} | {m['audio_seconds']} | {m['wall_seconds']} | "
            f"{m['rtf']} | {m['peak_vram_gb']} | {'yes' if m.get('context') else 'no'} |"
        )

    # ---- divergence from baseline
    lines.append("\n## Divergence from the Whisper baseline (character level)\n")
    lines.append("| arm | whole-transcript CER vs baseline | per-segment CER mean | median | segments identical |")
    lines.append("| --- | --- | --- | --- | --- |")
    for label, arm in arms.items():
        arm_text = "".join(s["text"] for s in arm["segments"])
        whole = cer(base_text, arm_text)
        per = [cer(s["baseline_text"], s["text"]) for s in arm["segments"] if s["baseline_text"]]
        same = sum(1 for s in arm["segments"] if normalize_ja(s["baseline_text"]) == normalize_ja(s["text"]))
        if per:
            lines.append(
                f"| {label} | {whole:.3f} | {statistics.mean(per):.3f} | {statistics.median(per):.3f} | "
                f"{same}/{len(arm['segments'])} |"
            )
        else:
            lines.append(f"| {label} | {whole:.3f} | - | - | - |")

    # ---- proper-noun hits
    if args.terms:
        lines.append("\n## Proper-noun occurrences (knowledge-base terms)\n")
        header = "| term | baseline | " + " | ".join(arms) + " |"
        lines.append(header)
        lines.append("| --- " * (len(arms) + 2) + "|")
        for term in args.terms:
            row = [term, str(base_text.count(term))]
            for arm in arms.values():
                row.append(str("".join(s["text"] for s in arm["segments"]).count(term)))
            lines.append("| " + " | ".join(row) + " |")

    # ---- per-segment side by side (segment-window arms only)
    lines.append("\n## Per-segment side by side\n")
    seg_arms = {k: v for k, v in arms.items() if len(v["segments"]) == len(base_segments)}
    for i, base in enumerate(base_segments):
        variants = {label: arm["segments"][i]["text"] for label, arm in seg_arms.items()}
        if all(normalize_ja(v) == normalize_ja(base.get("text", "")) for v in variants.values()):
            continue
        lines.append(f"**[{i:03d}] {base['start']:.2f}–{base['end']:.2f}**")
        lines.append(f"- BASE: {base.get('text', '')}")
        for label, text in variants.items():
            lines.append(f"- {label}: {text}")
        lines.append("")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
