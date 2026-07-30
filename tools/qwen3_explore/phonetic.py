"""Split the Whisper-vs-Qwen disagreement into "sounds the same" and "heard something else".

Character CER is the wrong yardstick for this project: a near-homophone (水仙十字 → 推薦獣獅)
is repairable by the LLM correction stage, while a divergence that is also phonetically far
means one of the two actually mis-heard or dropped audio. Readings come from pykakasi, so a
kanji/kana choice difference collapses to zero distance while a real acoustic difference does not.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pykakasi

from .common import cer, edit_distance, load_aligned, normalize_ja

_KKS = pykakasi.kakasi()


def reading(text: str) -> str:
    """Kana reading, with long vowels and small kana folded away (they are ASR noise, not content)."""
    hira = "".join(item["hira"] for item in _KKS.convert(text))
    table = str.maketrans("ぁぃぅぇぉゃゅょっ", "あいうえおやゆよつ")
    return normalize_ja(hira.translate(table)).replace("ー", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-aligned", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--near-threshold", type=float, default=0.34, help="phone-CER at or below this = LLM-repairable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = load_aligned(args.baseline_aligned)["segments"]
    arm = json.loads(Path(args.arm).read_text(encoding="utf-8"))["segments"]

    rows = []
    for i, (b, a) in enumerate(zip(base, arm)):
        bt, at = b.get("text", ""), a["text"]
        if not normalize_ja(bt) or not normalize_ja(at):
            continue
        char = cer(bt, at)
        if char == 0:
            continue
        br, ar = reading(bt), reading(at)
        phone = edit_distance(br, ar) / max(1, len(br))
        rows.append((i, char, phone, bt, at, br, ar))

    if not rows:
        print("no divergent segments")
        return

    near = [r for r in rows if r[2] <= args.near_threshold]
    far = [r for r in rows if r[2] > args.near_threshold]
    same_sound = [r for r in rows if r[2] == 0]

    lines = [
        "# Divergence split by phonetic distance\n",
        f"- arm: `{args.arm}`",
        f"- divergent segments (both sides non-empty): {len(rows)}",
        f"- mean character CER {statistics.mean(r[1] for r in rows):.3f} "
        f"vs mean **phone** CER {statistics.mean(r[2] for r in rows):.3f}",
        f"- identical reading, different characters: **{len(same_sound)} ({100 * len(same_sound) / len(rows):.0f}%)**",
        f"- phone CER <= {args.near_threshold} (near-homophone, LLM-repairable): "
        f"**{len(near)} ({100 * len(near) / len(rows):.0f}%)**",
        f"- phone CER > {args.near_threshold} (genuinely heard differently): "
        f"**{len(far)} ({100 * len(far) / len(rows):.0f}%)**\n",
        "## Phonetically far divergences (the ones the LLM cannot rescue)\n",
        "| seg | phone CER | baseline | arm |",
        "| --- | --- | --- | --- |",
    ]
    for i, _c, p, bt, at, _br, _ar in sorted(far, key=lambda r: -r[2])[:25]:
        lines.append(f"| {i} | {p:.2f} | {bt[:40]} | {at[:40]} |")

    lines += ["\n## Same sound, different characters (free for the LLM to fix)\n", "| seg | baseline | arm | reading |", "| --- | --- | --- | --- |"]
    for i, _c, _p, bt, at, br, _ar in same_sound[:20]:
        lines.append(f"| {i} | {bt[:34]} | {at[:34]} | {br[:34]} |")

    Path(args.out).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))


if __name__ == "__main__":
    main()
