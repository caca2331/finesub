"""Unblind the listening audit and compare it with the third-model metric.

Two questions the edit-distance metric could not answer:

* **Does the direction survive removing orthography?** The judge tags each
  disagreement, so 写法 (both correct, different convention) can be excluded and
  the remainder re-counted. That is the confound the Qwen number carries and
  cannot separate out.
* **Who commits which kind of error?** 漏字 / 多字 / 幻觉 / 听错 attribute the
  failure to an arm, which "who is closer to the judge" never does.

Everything is reported with a Wilson interval on the decisive subset: 200 items
fixes a direction, not a magnitude, and a bare ratio invites reading it as one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TURBO = "turbo"
JA = "ja"
#: Verdicts that name a winner. 平 and 都错 are outcomes, not abstentions, so
#: they are reported rather than dropped -- but a share needs a denominator of
#: decisions, and they are not decisions.
DECISIVE = {TURBO, JA}


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion. Wilson, not normal-approximation: at
    n≈100 and p near 0.5 the difference is small, but it stays honest at the
    per-category counts, which are much smaller."""

    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def report(label: str, counts: Counter) -> None:
    turbo, ja = counts[TURBO], counts[JA]
    decisive = turbo + ja
    lo, hi = wilson(turbo, decisive)
    tie = counts["tie"]
    both = counts["both_wrong"]
    print(
        f"{label:26} turbo {turbo:4d} : ja {ja:4d}   平 {tie:4d}  都错 {both:3d}   "
        f"turbo 占决胜 {100.0 * turbo / max(1, decisive):5.1f}% "
        f"[{100 * lo:.1f}, {100 * hi:.1f}]  n={decisive}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="tmp/bench/ja-ab/audit-key.json")
    parser.add_argument("--session", default="tmp/bench/ja-ab/audit/audit-session.json")
    parser.add_argument("--out", default="tmp/bench/ja-ab/audit-scored.json")
    args = parser.parse_args(argv)

    key_doc = json.loads(Path(args.key).read_text(encoding="utf-8"))
    key = {item["id"]: item for item in key_doc["items"]}
    state = json.loads(Path(args.session).read_text(encoding="utf-8"))

    rows: dict[str, dict] = {}
    for batch, entry in sorted(state["answers"].items()):
        for row in entry["rows"]:
            row = dict(row)
            row["batch"] = batch
            rows[row["id"]] = row

    scored = []
    missing = []
    for item_id, item in key.items():
        row = rows.get(item_id)
        if row is None:
            missing.append(item_id)
            continue
        verdict = row["verdict"].strip()
        if verdict == "甲":
            winner = TURBO if item["turbo_slot"] == "甲" else JA
        elif verdict == "乙":
            winner = TURBO if item["turbo_slot"] == "乙" else JA
        elif verdict in ("平", "平局"):
            winner = "tie"
        else:
            winner = "both_wrong"
        # The category names an arm only through the reason text, which is free
        # form; the category itself is about the *kind* of difference.
        scored.append(
            {
                **item,
                "batch": row["batch"],
                "verdict_raw": verdict,
                "category": row["category"].strip(),
                "reason": row["reason"],
                "winner": winner,
                "agrees_with_qwen": winner == {"a": TURBO, "b": JA, "tie": "tie"}.get(
                    item["qwen_winner"]
                ),
            }
        )

    print(f"scored {len(scored)} of {len(key)}" + (f"  missing {missing}" if missing else ""))
    print("")

    overall = Counter(row["winner"] for row in scored)
    report("全部", overall)

    non_orthographic = Counter(
        row["winner"] for row in scored if row["category"] != "写法"
    )
    report("去掉「写法」类", non_orthographic)

    orthographic = Counter(row["winner"] for row in scored if row["category"] == "写法")
    report("只看「写法」类", orthographic)
    print("")

    print("按类别（谁被判赢）：")
    for category, _count in Counter(row["category"] for row in scored).most_common():
        counts = Counter(row["winner"] for row in scored if row["category"] == category)
        total = sum(counts.values())
        print(
            f"  {category:6} n={total:3d}   turbo {counts[TURBO]:3d}  ja {counts[JA]:3d}  "
            f"平 {counts['tie']:3d}  都错 {counts['both_wrong']:3d}"
        )
    print("")

    # Where the two metrics disagree is where the edit distance was doing
    # something other than judging content.
    both_decisive = [
        row
        for row in scored
        if row["winner"] in DECISIVE and row["qwen_winner"] in ("a", "b")
    ]
    agree = sum(1 for row in both_decisive if row["agrees_with_qwen"])
    print(
        f"与 Qwen 都给出胜负的 {len(both_decisive)} 条里，一致 {agree} "
        f"（{100.0 * agree / max(1, len(both_decisive)):.1f}%）"
    )
    qwen_only = Counter(
        {"a": TURBO, "b": JA}[row["qwen_winner"]]
        for row in scored
        if row["winner"] == "tie" and row["qwen_winner"] in ("a", "b")
    )
    print(
        f"听审判平、Qwen 判了胜负的 {sum(qwen_only.values())} 条：Qwen 给 turbo "
        f"{qwen_only[TURBO]}、给 ja {qwen_only[JA]}"
    )
    print("")

    print("按产物：")
    for product in sorted({row["product"] for row in scored}):
        counts = Counter(row["winner"] for row in scored if row["product"] == product)
        print(
            f"  {product:16} turbo {counts[TURBO]:3d}  ja {counts[JA]:3d}  "
            f"平 {counts['tie']:3d}  都错 {counts['both_wrong']:3d}"
        )

    Path(args.out).write_text(
        json.dumps(scored, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
