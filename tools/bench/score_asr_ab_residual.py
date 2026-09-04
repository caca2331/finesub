"""What survives `--qwen-verify`: the audit minus the errors it cleans up.

The A/B arms were produced with the referee OFF, on purpose -- it had to stay
independent to be a judge. But production runs it, and it exists to delete a
specific family of failures: whole-segment closing phrases, Latin runs inside
CJK output, the absolute-level suspect tier, and segments the stabilize noise
legs would tag as highly-suspected hallucination or filler
(`qwen_referee.collect_suspect_indices`). Mishearing and dropped words are NOT
in that family.

So "who is ahead" has two answers, and the useful one is the second:

* over every disagreement (24.8), which is what the audit reported;
* over the disagreements that would still be there after verification -- the
  ones that reach a subtitle.

Two cuts, because neither alone is honest:

* **by category** -- drop what the listener called 幻觉 / 多字. Interpretable,
  but the listener's label is about the *kind* of difference, not about
  whether this pipeline would catch it.
* **by mechanism** -- drop the items where the losing arm's own segment is one
  `collect_suspect_indices` would hand to the referee. Exact about what the
  production stage looks at. ⚠ It is still "looked at", not "deleted": the
  evidence can also veto a drop, so this cut is an upper bound on what
  verification removes.
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

from finesub.speech.verification.qwen_referee import collect_suspect_indices  # noqa: E402


def wilson(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def report(label: str, rows: list[dict]) -> None:
    counts = Counter(row["winner"] for row in rows)
    turbo, ja = counts["turbo"], counts["ja"]
    decisive = turbo + ja
    lo, hi = wilson(turbo, decisive)
    print(
        f"{label:34} turbo {turbo:3d} : ja {ja:3d}  平 {counts['tie']:3d} "
        f"都错 {counts['both_wrong']:2d}   turbo 占决胜 {100.0 * turbo / max(1, decisive):5.1f}% "
        f"[{100 * lo:.1f}, {100 * hi:.1f}]  n={decisive}"
    )


def suspect_spans(stable_path: str) -> list[tuple[float, float]]:
    segments = json.loads(Path(stable_path).read_text(encoding="utf-8"))["segments"]
    spans = []
    for index in collect_suspect_indices(segments):
        segment = segments[index]
        spans.append((float(segment.get("start", 0.0)), float(segment.get("end", 0.0))))
    return spans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", default="tmp/bench/ja-ab/audit-scored.json")
    parser.add_argument("--report", default="tmp/bench/ja-ab/report.json")
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.scored).read_text(encoding="utf-8"))
    report_doc = json.loads(Path(args.report).read_text(encoding="utf-8"))
    turbo_model, ja_model = report_doc["arms"]
    spans: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for name, entry in report_doc["files"].items():
        for arm, model in (("turbo", turbo_model), ("ja", ja_model)):
            spans[(name, arm)] = suspect_spans(entry["runs"][model]["stable"])

    for row in rows:
        loser = {"turbo": "ja", "ja": "turbo"}.get(row["winner"])
        row["loser_is_suspect"] = False
        if loser is None:
            continue
        start, end = float(row["start"]), float(row["end"])
        for span_start, span_end in spans[(row["product"], loser)]:
            if min(end, span_end) - max(start, span_start) > 0:
                row["loser_is_suspect"] = True
                break

    print(f"{len(rows)} 条听审判决\n")
    report("全部（= 24.8）", rows)
    print("")

    print("按类别剔除：")
    report("  去掉「幻觉」", [r for r in rows if r["category"] != "幻觉"])
    report("  去掉「幻觉」「多字」", [r for r in rows if r["category"] not in ("幻觉", "多字")])
    report(
        "  只留「听错」「漏字」",
        [r for r in rows if r["category"] in ("听错", "漏字")],
    )
    print("")

    print("按机制剔除（输的那一臂的段会被裁判过目）：")
    flagged = [r for r in rows if r["loser_is_suspect"]]
    report("  被剔除的那些", flagged)
    report("  剩下的（会进成品）", [r for r in rows if not r["loser_is_suspect"]])
    print("")

    print("被裁判过目的判决，按类别：")
    for category, _ in Counter(r["category"] for r in flagged).most_common():
        counts = Counter(r["winner"] for r in flagged if r["category"] == category)
        print(
            f"  {category:6} n={sum(counts.values()):3d}  turbo {counts['turbo']:3d} "
            f"ja {counts['ja']:3d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
