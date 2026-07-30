"""How much of Whisper's cue segmentation is reconstructible from Qwen-side signals?

Qwen3-ASR emits no timestamps, so segmentation has to be rebuilt from three signals the
pipeline can see: VAD interval gaps, the aligner's inter-word pauses, and the punctuation in
the ASR text. `segment_split` currently uses only the first — it deliberately ignores word-level
gaps because Whisper-DTW word times are smeared (docs/segment_split.md). Qwen's word times come
from a separate non-autoregressive model, so that reason may not carry over; this measures
whether it does.

Whisper's own cue starts are the reference here — not because they are ground truth, but
because they are the segmentation the pipeline is currently tuned around.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers.models.qwen3_asr.processing_qwen3_asr import _is_kept_char

from .common import load_aligned
from .utils_text import punct_kind


def punctuation_boundaries(text: str, words: list[dict]) -> list[tuple[float, str]]:
    """Map each punctuation mark in the raw text onto the end time of the word before it.

    The aligner's word list is punctuation-free (nagisa output is filtered to kept characters),
    so the two are re-joined by counting kept characters.
    """
    ends: list[tuple[int, float]] = []
    seen = 0
    for w in words:
        seen += len(w["word"])
        ends.append((seen, w["end"]))

    out: list[tuple[float, str]] = []
    kept = 0
    for ch in text:
        kind = punct_kind(ch)
        if kind:
            for count, end in ends:
                if count >= kept:
                    out.append((end, kind))
                    break
        elif _is_kept_char(ch):
            kept += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--qwen-raw", required=True, help="rescued_asr.py output")
    ap.add_argument("--vad", required=True)
    ap.add_argument("--tolerance", type=float, default=0.30)
    ap.add_argument("--word-gap", type=float, default=0.20)
    args = ap.parse_args()

    base = load_aligned(args.baseline_aligned)["segments"]
    qwen = json.loads(Path(args.qwen_raw).read_text(encoding="utf-8"))["segments"]
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))["speech"]

    words: list[dict] = []
    punct: list[tuple[float, str]] = []
    for seg in qwen:
        if not seg.get("words"):
            continue
        words.extend(seg["words"])
        punct.extend(punctuation_boundaries(seg["text"], seg["words"]))
    words.sort(key=lambda w: w["start"])

    gaps = [(b["start"] - a["end"], a["end"]) for a, b in zip(words, words[1:])]
    inside = [g for g, _t in gaps if g >= 0]

    vad_edges = [a for a, _ in vad] + [b for _, b in vad]
    big_gap_times = [t for g, t in gaps if g >= args.word_gap]
    punct_times = [t for t, _k in punct]
    sentence_times = [t for t, k in punct if k == "sentence"]

    cuts = [s["start"] for s in base[1:]]

    def near(t, candidates):
        return any(abs(t - c) <= args.tolerance for c in candidates)

    by_vad = [t for t in cuts if near(t, vad_edges)]
    rest = [t for t in cuts if t not in by_vad]
    by_gap = [t for t in rest if near(t, big_gap_times)]
    by_punct = [t for t in rest if near(t, punct_times)]
    by_either = [t for t in rest if near(t, big_gap_times) or near(t, punct_times)]

    n = len(cuts)
    print(f"whisper cue boundaries: {n}  (tolerance ±{args.tolerance}s)")
    print(f"  explained by VAD interval edges     : {len(by_vad):3d} ({100 * len(by_vad) / n:.0f}%)")
    print(f"  of the remaining {len(rest)}:")
    print(f"    Qwen word gap >= {args.word_gap}s          : {len(by_gap):3d} ({100 * len(by_gap) / max(1, len(rest)):.0f}%)")
    print(f"    Qwen punctuation                  : {len(by_punct):3d} ({100 * len(by_punct) / max(1, len(rest)):.0f}%)")
    print(f"    either                            : {len(by_either):3d} ({100 * len(by_either) / max(1, len(rest)):.0f}%)")
    print(
        f"  total reconstructible               : {len(by_vad) + len(by_either):3d} "
        f"({100 * (len(by_vad) + len(by_either)) / n:.0f}%)"
    )
    print()
    print(f"Qwen inter-word gaps: n={len(inside)} median={statistics.median(inside):.3f}s "
          f"mean={statistics.mean(inside):.3f}s  >={args.word_gap}s: {sum(g >= args.word_gap for g in inside)}")
    print(f"punctuation marks: {len(punct)} (sentence-final {len(sentence_times)})")
    print(f"candidate density: VAD edges {len(vad_edges)}, big word gaps {len(big_gap_times)}, "
          f"punct {len(punct_times)} vs {n} cues needed")


if __name__ == "__main__":
    main()
