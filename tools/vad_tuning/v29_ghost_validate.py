"""Acceptance run for --vad-silero-suppress.

  miyako (noisy)   base vs suppress through the real production stage: dropped
                   intervals, decode time, hallucination/filler tags after
                   stabilize, whole-transcript char delta
  clean clip       suppress vs the existing production artifact: missing
                   hand-confirmed words (+/-36 packing-noise band applies),
                   plus what was dropped there
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys
REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from refs import load_word_srt  # noqa: E402
from v27b_e2e_compare import (  # noqa: E402
    load_segs, missing_hand_words, stabilize, tags_of,
)


def report(name: str, aligned: Path) -> list:
    d = json.loads(aligned.read_text(encoding="utf-8"))
    meta = d.get("metadata", {})
    vad = meta.get("vad", {}) or {}
    ghost = vad.get("silero_ghost_suppress")
    timing = (meta.get("asr_align") or {}).get("timing", {}) or meta.get("timing", {})
    stable = stabilize(aligned)
    segs = load_segs(stable)
    tg = tags_of(segs)
    chars = sum(len(str(s.get("text", ""))) for s in segs)
    drop_note = (f" dropped={ghost['dropped']} ({ghost['dropped_sec']:.1f}s, "
                 f"silero {ghost.get('silero_sec', 0):.0f}s)") if ghost else ""
    decode = timing.get("asr_sec") or timing.get("total_sec")
    print(f"  {name:>10}: segments={len(segs)} chars={chars} "
          f"幻觉={tg.get('高度疑似幻觉', 0)} 语气={tg.get('高度疑似语气填充词', 0)} "
          f"漂移={tg.get('时间漂移', 0)} timing_total={decode}{drop_note}")
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-dir", type=Path, required=True)
    args = ap.parse_args()
    base = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")

    print("=== miyako (noisy, 70min) base vs suppress ===")
    report("base", args.e2e_dir / "miyako-base-aligned.json")
    report("suppress", args.e2e_dir / "miyako-suppress-aligned.json")

    print("\n=== BV1cqLR6hEp3 (clean guard) ===")
    srt = base / "data/disfluency-gold/BV1cqLR6hEp3/BV1cqLR6hEp3-fixed.srt"
    prod_stable = base / "out/reference/BV1cqLR6hEp3/BV1cqLR6hEp3-stable.json"
    for name, stable_path in (
        ("production", prod_stable),
        ("suppress", stabilize(args.e2e_dir / "BV1cqLR6hEp3-suppress-aligned.json")),
    ):
        segs = load_segs(stable_path)
        miss, n = missing_hand_words(segs, srt)
        tg = tags_of(segs)
        print(f"  {name:>10}: segments={len(segs)} 缺失人工词 {len(miss)}/{n} "
              f"({len(miss)/n*100:.2f}%) 幻觉={tg.get('高度疑似幻觉', 0)} "
              f"漂移={tg.get('时间漂移', 0)}")
    d = json.loads((args.e2e_dir / "BV1cqLR6hEp3-suppress-aligned.json")
                   .read_text(encoding="utf-8"))
    ghost = (d.get("metadata", {}).get("vad") or {}).get("silero_ghost_suppress")
    if ghost:
        print(f"  干净 clip 丢弃: {ghost['dropped']} 段 / {ghost['dropped_sec']:.1f}s")
        for g in ghost["dropped_intervals"][:12]:
            print(f"    {g['start']:7.1f}-{g['end']:7.1f} pk{g['peak_db']} "
                  f"sil{g['silero_peak']}")


if __name__ == "__main__":
    main()
