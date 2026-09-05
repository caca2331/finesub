"""Baseline for the two §12 thresholds, read off historical correction artifacts.

Offline and free: it only reads `out/**/exchanges/*correction*.md` that past
runs already wrote. Aggregates only -- no reply content is printed.

What it answers:

* how far a model's own `char_count` normally strays from the computed value,
  and **in which direction** (`SYSTEMATIC_CHAR_COUNT_SHARE`);
* how often a correction reply is legitimately pure ASCII (the false-positive
  surface any escape predicate has to clear).
"""

from __future__ import annotations

import collections
import pathlib
import re
import statistics
import sys

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]
MISMATCH = re.compile(
    r"char_count '([^']*)' does not match the computed value ([\d.+]+)"
)
#: The one measured transport fault, for contrast: 15 of 15 rows, 45 against 16.
INCIDENT_SHARE = 1.0
INCIDENT_RATIO = 45 / 16


def _sum_cells(cell: str) -> float:
    return sum(float(part) for part in cell.strip().split("+") if part)


def _block(text: str, tag: str) -> str:
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    return text[start + len(tag) + 2 : end] if 0 <= start < end else ""


def _rows(block: str) -> list[str]:
    return [
        line
        for line in block.splitlines()
        if line.strip()
        and "|" in line
        and not line.startswith("#")
        and not line.startswith("type|")
    ]


def _model(header: str) -> str:
    """The model from the Execution Attempts table, or `?` for older shapes."""

    if "## Execution Attempts" not in header:
        return "?"
    table = header.split("## Execution Attempts", 1)[1].split("##", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("|")]
    for row in rows[2:]:  # past the header and separator rows
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) > 2 and cells[2]:
            return cells[2]
    return "?"


def collect(root: pathlib.Path) -> list[dict]:
    records = []
    for path in sorted(root.glob("out/**/exchanges/*correction*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "## 模型响应" not in text:
            continue
        header, _, reply = text.partition("## 模型响应")
        warnings = (
            header.split("## Validation")[1].split("##")[0]
            if "## Validation" in header
            else ""
        )
        pairs = MISMATCH.findall(warnings)
        ratios = []
        for reported, computed in pairs:
            try:
                lower, upper = _sum_cells(reported), _sum_cells(computed)
            except ValueError:
                continue
            if lower > 0:
                ratios.append(upper / lower)
        records.append(
            {
                "model": _model(header),
                "rows": len(_rows(_block(reply, "translated"))),
                "mismatches": len(pairs),
                "ratios": ratios,
                "reply_ascii": reply.isascii(),
                "ok": "- validation_ok: True" in header,
            }
        )
    return records


def main(argv: list[str] | None = None) -> None:
    # The artifacts live wherever the runs happened, which is not necessarily
    # this checkout: a worktree has no `out/` of its own.
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = pathlib.Path(arguments[0]).expanduser() if arguments else REPOSITORY_ROOT
    print(f"reading artifacts under : {root}")
    records = collect(root)
    usable = [record for record in records if record["rows"]]
    print(f"correction exchanges parsed : {len(records)}")
    print(f"...with parseable CSV rows  : {len(usable)}")
    print()
    print("=== char_count disagreement, per model")
    print(
        f"    incident for contrast: share {INCIDENT_SHARE:.2f}, "
        f"ratio ~{INCIDENT_RATIO:.2f}, direction >1"
    )

    by_model = collections.defaultdict(list)
    for record in usable:
        by_model[record["model"]].append(record)

    worst_share = 0.0
    every_ratio: list[float] = []
    for model, group in sorted(by_model.items(), key=lambda item: -len(item[1])):
        shares = [record["mismatches"] / record["rows"] for record in group]
        ratios = [ratio for record in group for ratio in record["ratios"]]
        every_ratio.extend(ratios)
        worst_share = max(worst_share, max(shares, default=0.0))
        print(f"  {model}  (n={len(group)})")
        print(
            f"    windows with >=1 disagreeing row : "
            f"{sum(1 for share in shares if share > 0)}/{len(shares)}"
            f"   max share {max(shares, default=0.0):.3f}"
        )
        if ratios:
            print(
                f"    computed/reported                : n={len(ratios)}, "
                f"median {statistics.median(ratios):.2f}, "
                f"min {min(ratios):.2f}, max {max(ratios):.2f}, "
                f">1: {sum(1 for ratio in ratios if ratio > 1)}/{len(ratios)}"
            )

    print()
    print(f"  worst baseline share  : {worst_share:.3f}")
    if every_ratio:
        print(
            f"  baseline ratio range  : {min(every_ratio):.2f} .. "
            f"{max(every_ratio):.2f}, above 1: "
            f"{sum(1 for ratio in every_ratio if ratio > 1)}/{len(every_ratio)}"
        )

    print()
    print("=== pure-ASCII replies")
    ascii_replies = [record for record in records if record["reply_ascii"]]
    print(f"  pure-ASCII replies    : {len(ascii_replies)}/{len(records)}")
    for record in ascii_replies:
        print(
            f"    model={record['model']} rows={record['rows']} "
            f"validation_ok={record['ok']}"
        )
    print("  NOTE: every task in this corpus has a CJK target, so this says")
    print("  nothing about English-to-English shapes.")


if __name__ == "__main__":
    main()
