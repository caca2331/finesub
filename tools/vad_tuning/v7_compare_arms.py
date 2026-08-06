"""Compare the two ASR arms: what the VAD choice did to recognition and splitting.

Runs asr-stabilize (profile 0) on both aligned files, because its drop/tag counts are
the pipeline's own verdict on hallucination and filled-pause segments -- exactly the
damage weak vocal noise is expected to cause.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def load(path: Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return (d["segments"] if isinstance(d, dict) else d), (
        d.get("metadata", {}) if isinstance(d, dict) else {})


def repeat_runs(words, n: int = 3) -> int:
    """Words belonging to a run of >=n identical consecutive tokens (decoder loop)."""
    texts = [str(w.get("word", "")).strip() for w in words]
    hit, i = 0, 0
    while i < len(texts):
        j = i
        while j + 1 < len(texts) and texts[j + 1] == texts[i] and texts[i]:
            j += 1
        if j - i + 1 >= n:
            hit += j - i + 1
        i = j + 1
    return hit


def describe(name: str, segs, meta) -> dict:
    words = [w for s in segs for w in (s.get("words") or [])]
    durs = np.array([float(s["end"]) - float(s["start"]) for s in segs]) if segs else np.array([0.])
    energies = np.array([float(s.get("vad_weighted_energy_db", 0.0)) for s in segs])
    chars = np.array([len(str(s.get("text", ""))) for s in segs])
    tags = Counter(t for s in segs for t in (s.get("tags") or []))
    print(f"--- {name}")
    print(f"    segments={len(segs)}  words={len(words)}  "
          f"chars={int(chars.sum())}  audio_covered={durs.sum():.0f}s")
    print(f"    seg duration  med={np.median(durs):.2f} p90={np.quantile(durs,0.9):.2f} "
          f"max={durs.max():.2f}")
    print(f"    chars/segment med={np.median(chars):.0f} p90={np.quantile(chars,0.9):.0f}")
    print(f"    segment energy_db  med={np.median(energies):.1f} "
          f"p10={np.quantile(energies,0.1):.1f}  below0={int((energies<0).sum())} "
          f"below-20={int((energies<-20).sum())}")
    print(f"    words in repeat runs (>=3): {repeat_runs(words)}")
    print(f"    tags: {dict(tags) or '-'}")
    sp = (meta.get("asr_align") or {}).get("segment_split") or {}
    if sp:
        print(f"    segment_split: synthetic_word_segments="
              f"{sp.get('synthetic_word_segments')}")
    return {"segs": len(segs), "words": len(words), "chars": int(chars.sum()),
            "energies": energies, "durs": durs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", required=True)
    ap.add_argument("--silero", required=True)
    ap.add_argument("--stabilize", action="store_true")
    args = ap.parse_args()

    print("=== aligned (pre-stabilization) ===")
    out = {}
    for name, path in (("energy VAD", args.energy), ("silero VAD", args.silero)):
        segs, meta = load(Path(path))
        out[name] = describe(name, segs, meta)
        print()

    if args.stabilize:
        from asr_playground.speech.postprocessing.stabilization import stabilize_json_file

        print("=== after asr-stabilize profile 0 (pipeline's own hallucination verdict) ===")
        for name, path in (("energy VAD", args.energy), ("silero VAD", args.silero)):
            p = Path(path)
            outp, report = stabilize_json_file(p, output_path=p.with_name(
                p.stem.replace("-aligned", "") + "-stable.json"), profile=0)
            print(f"--- {name}")
            print(f"    segments {report.input_segments} -> {report.output_segments}"
                  f"  (dropped {report.suspicious_segments_dropped} suspicious,"
                  f" {report.phrase_occurrences_removed} phrase occurrences)")
            print(f"    tags: {dict(report.tag_counts) or '-'}")
            segs, meta = load(outp)
            words = [w for s in segs for w in (s.get("words") or [])]
            print(f"    surviving: segments={len(segs)} words={len(words)} "
                  f"chars={sum(len(str(s.get('text',''))) for s in segs)}")
            print()


if __name__ == "__main__":
    main()
