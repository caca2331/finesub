"""Compare the rescue arm's end-to-end output against the production artifacts.

  clean clip   missing hand-confirmed words (exact text within +/-2 s), both arms
               through the same function so only the delta matters; the +/-36-word
               packaging noise band (FINDINGS M3) applies
  yingtao      the 16 human-confirmed lost regions: what text does each arm place
               inside the region span, did the rescue arm actually recover the line
  both         asr-stabilize tag counts -- did the extra audio buy hallucination
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from refs import load_word_srt  # noqa: E402


def load_segs(path: Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["segments"] if isinstance(d, dict) else d


def stabilize(aligned: Path) -> Path:
    out = aligned.with_name(aligned.stem.replace("-aligned", "") + "-stable.json")
    if not out.exists():
        from asr_playground.speech.postprocessing.stabilization import stabilize_json_file

        stabilize_json_file(aligned, output_path=out)
    return out


def words_of(segs):
    out = []
    for s in segs:
        for w in s.get("words") or []:
            t = str(w.get("word", "")).strip()
            if t:
                out.append((float(w["start"]), t))
    return out


def missing_hand_words(segs, srt: Path, tol: float = 2.0):
    ref = load_word_srt(srt)
    have = words_of(segs)
    by_text = {}
    for t, w in have:
        by_text.setdefault(w, []).append(t)
    miss = []
    for w in ref:
        cand = by_text.get(w.text.strip())
        if not cand or min(abs(c - w.start) for c in cand) > tol:
            miss.append(w)
    return miss, len(ref)


def text_in_span(segs, s: float, e: float, pad: float = 0.5) -> str:
    parts = []
    for seg in segs:
        if float(seg["end"]) > s - pad and float(seg["start"]) < e + pad:
            parts.append(str(seg.get("text", "")).strip())
    return "".join(parts)


def tags_of(segs) -> Counter:
    return Counter(t for s in segs for t in (s.get("tags") or []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-dir", type=Path, required=True)
    ap.add_argument("--joined", type=Path, required=True)
    args = ap.parse_args()

    base = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
    joined = pd.read_csv(args.joined, encoding="utf-8-sig")

    # ---- clean clip: hand-word missing ------------------------------------
    srt = base / "data/disfluency-gold/BV1cqLR6hEp3/BV1cqLR6hEp3-fixed.srt"
    prod_stable = base / "out/reference/BV1cqLR6hEp3/BV1cqLR6hEp3-stable.json"
    resc_stable = stabilize(args.e2e_dir / "BV1cqLR6hEp3-vocal-rescue-aligned.json")
    print("=== BV1cqLR6hEp3 (annotated clean clip) ===")
    for name, p in (("production", prod_stable), ("rescue", resc_stable)):
        segs = load_segs(p)
        miss, n = missing_hand_words(segs, srt)
        tg = tags_of(segs)
        print(f"  {name:>10}: segments={len(segs)} 缺失人工词 {len(miss)}/{n} "
              f"({len(miss)/n*100:.2f}%)  幻觉tag={tg.get('高度疑似幻觉', 0)} "
              f"语气tag={tg.get('高度疑似语气填充词', 0)} "
              f"漂移tag={tg.get('时间漂移', 0)}")

    # ---- yingtao: the 16 lost regions -------------------------------------
    prod_y = base / "out/yingtao/yingtao-stable.json"
    resc_y = stabilize(args.e2e_dir / "yingtao-vocal-rescue-aligned.json")
    pa, ra = load_segs(prod_y), load_segs(resc_y)
    print("\n=== yingtao: 16 个人工确认丢失区域 ===")
    lab = joined[(joined["clip"] == "yingtao") & (joined["kind"] == "added")
                 & (joined["never_decoded_frac"] > 0.5)
                 & (joined["label"].isin(["真语音", "听不清"]))]
    got = 0
    for _, r in lab.iterrows():
        bt = text_in_span(pa, r["start"], r["end"])
        rt = text_in_span(ra, r["start"], r["end"])
        rec = "+" if (rt and not bt) else ("=" if rt else "-")
        got += rec == "+"
        print(f"  {rec} {r['start']:7.1f}s [{r['label']}] 切片ASR「{str(r['asr_text'])[:14]}」"
              f" 生产「{bt[:20]}」→ rescue「{rt[:26]}」")
    print(f"  端到端新增出文本: {got}/{len(lab)}")
    for name, segs in (("production", pa), ("rescue", ra)):
        tg = tags_of(segs)
        chars = sum(len(str(s.get('text', ''))) for s in segs)
        print(f"  {name:>10}: segments={len(segs)} chars={chars} "
              f"幻觉tag={tg.get('高度疑似幻觉', 0)} 语气tag={tg.get('高度疑似语气填充词', 0)} "
              f"漂移tag={tg.get('时间漂移', 0)}")


if __name__ == "__main__":
    main()
