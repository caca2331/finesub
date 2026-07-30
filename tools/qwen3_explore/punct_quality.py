"""Score each candidate boundary signal against human-refined cue times.

The refined subtitles are translated and re-segmented, so their *text* is not a transcription
reference — but their cue times are hand-adjusted, which makes them the only human-verified
statement of where a subtitle boundary belongs. Precision matters as much as recall here: a
signal that fires everywhere reconstructs every boundary and is still useless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

from .boundary_signals import punctuation_boundaries
from .common import load_aligned, read_srt


def score(name: str, times: list[float], truth: list[float], tol: float) -> str:
    if not times:
        return f"  {name:26s} n=  0"
    hit = sum(1 for t in times if any(abs(t - g) <= tol for g in truth))
    covered = sum(1 for g in truth if any(abs(t - g) <= tol for t in times))
    prec = 100 * hit / len(times)
    rec = 100 * covered / len(truth)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f"  {name:26s} n={len(times):4d}  precision={prec:5.1f}%  recall={rec:5.1f}%  F1={f1:5.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refined-srt", required=True)
    ap.add_argument("--qwen-raw", required=True)
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--tolerance", type=float, default=0.30)
    args = ap.parse_args()

    cues = read_srt(args.refined_srt)
    truth = sorted({round(c.start, 3) for c in cues} | {round(c.end, 3) for c in cues})

    qwen = json.loads(Path(args.qwen_raw).read_text(encoding="utf-8"))["segments"]
    words, punct = [], []
    for seg in qwen:
        if not seg.get("words"):
            continue
        words.extend(seg["words"])
        punct.extend(punctuation_boundaries(seg["text"], seg["words"]))
    words.sort(key=lambda w: w["start"])

    gaps = [(b["start"] - a["end"], a["end"]) for a, b in zip(words, words[1:])]
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))["speech"]
    base = load_aligned(args.baseline_aligned)["segments"]

    print(f"human cue boundaries: {len(truth)}  (tolerance ±{args.tolerance}s)")
    print(score("whisper cue edges", sorted({round(s["start"], 3) for s in base} | {round(s["end"], 3) for s in base}), truth, args.tolerance))
    print(score("VAD interval edges", sorted({round(a, 3) for a, _ in vad} | {round(b, 3) for _, b in vad}), truth, args.tolerance))
    print(score("Qwen punct (sentence)", [t for t, k in punct if k == "sentence"], truth, args.tolerance))
    print(score("Qwen punct (all)", [t for t, _ in punct], truth, args.tolerance))
    for thr in (0.15, 0.25, 0.4, 0.6):
        print(score(f"Qwen word gap >={thr}s", [t for g, t in gaps if g >= thr], truth, args.tolerance))
    combo = sorted(
        {round(t, 3) for t, k in punct if k == "sentence"}
        | {round(t, 3) for g, t in gaps if g >= 0.4}
        | {round(a, 3) for a, _ in vad}
        | {round(b, 3) for _, b in vad}
    )
    print(score("union(sent-punct, gap.4, VAD)", combo, truth, args.tolerance))


if __name__ == "__main__":
    main()
