"""Case file: production intervals with real speech but no valid word ("empty-real").

Step-0 human labels found that in a third of sampled empty intervals a human hears
real speech (T4: 14 of 44). The VAD kept the audio, so the loss is downstream.
This walks the artifact chain for each labeled region and files it into one of:

  decode_miss      aligned.json has no words in the region: the full-run decoder
                   produced nothing where snippet ASR (isolated, silence-padded)
                   produced text -- group context or decode-time suppression
  stabilize_drop   aligned has words, stable lost the segment: which profile-0
                   rule, per the removal tags in the stabilize report
  invalid_tagged   stable has the words but they fail the valid-word filter
                   (drift tag, repeat run, over-long, punctuation)
  drifted_out      valid words exist for this audio but their timestamps landed
                   outside the interval -- alignment drift, subtitles exist but
                   are mistimed

Usage:
  python v28_empty_loss.py --joined <tmp/vad-step0/step0-joined.csv> \
      --aligned yingtao=<...-aligned.json> --stable yingtao=<...-stable.json> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from refs import load_valid_words  # noqa: E402


def load_segs(path: Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["segments"] if isinstance(d, dict) else d


def words_in(segs, s: float, e: float, pad: float = 0.0) -> List[Tuple[float, str, list]]:
    out = []
    for seg in segs:
        tags = seg.get("tags") or []
        for w in seg.get("words") or []:
            ws, we = float(w["start"]), float(w["end"])
            if ws < e + pad and we > s - pad:
                out.append((ws, str(w.get("word", "")).strip(), tags))
    return out


def diagnose(region, aligned, stable, valid) -> Tuple[str, str]:
    s, e = region
    a = words_in(aligned, s, e)
    if not a:
        near = words_in(aligned, s, e, pad=2.0)
        return "decode_miss", (f"aligned±2s 内最近词: "
                              f"{''.join(t for _, t, _ in near[:8]) or '(无)'}")
    st = words_in(stable, s, e)
    if not st:
        gone = "".join(t for _, t, _ in a[:10])
        return "stabilize_drop", f"aligned 有「{gone}」→ stable 无"
    va = [w for w in valid if w.start < e and w.end > s]
    if not va:
        tags = {t for _, _, tg in st for t in tg}
        return "invalid_tagged", (f"stable 有「{''.join(t for _, t, _ in st[:10])}」"
                                  f"tags={sorted(tags) or '(无tag,被词级过滤)'}")
    return "drifted_out?", f"valid 词其实在区间内: {''.join(w.text for w in va[:8])}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", type=Path, required=True)
    ap.add_argument("--aligned", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--stable", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--label", default="真语音",
                    help="which human label to walk (default the real-speech case)")
    args = ap.parse_args()

    aligneds = dict(x.split("=", 1) for x in args.aligned)
    stables = dict(x.split("=", 1) for x in args.stable)
    joined = pd.read_csv(args.joined, encoding="utf-8-sig")
    rows = joined[(joined["kind"] == "empty") & (joined["label"] == args.label)]

    counts: dict = {}
    for clip, grp in rows.groupby("clip"):
        if clip not in aligneds:
            print(f"== {clip}: no aligned.json given, skipped ({len(grp)} regions)")
            continue
        aligned = load_segs(Path(aligneds[clip]))
        stable = load_segs(Path(stables[clip]))
        valid, _ = load_valid_words(Path(stables[clip]))
        print(f"\n== {clip} ({len(grp)} 段, label={args.label})")
        for _, r in grp.iterrows():
            kind, detail = diagnose((r["start"], r["end"]), aligned, stable, valid)
            counts[kind] = counts.get(kind, 0) + 1
            print(f"  {r['start']:8.1f}s d{r['dur']:5.2f} pk{r['peak_db']:6.1f} "
                  f"[{kind:>14}] 切片ASR「{str(r['asr_text'])[:14]}」 {detail}")
    print("\n分类合计:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
