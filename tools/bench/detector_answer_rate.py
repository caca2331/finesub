"""P8 -- how often do the gated detectors actually answer on real input?

`docs/plans/crispasr-followups.md` -> P8: when a feature is gated on "the detector
gave an answer", the number that matters is its **answer rate on real input**,
not its accuracy when it does answer. A detector with 100% precision and 20%
coverage silently closes the gate on the other 80%, and every part still looks
like it is working.

This reads finished `*-aligned.json` artifacts rather than running anything:
the answer rate is a property of the corpus that has already been transcribed,
so a fresh run would only add GPU time and sampling noise. Artifacts written
before a given field existed are reported as "not recorded" rather than as
zero -- absence of a field and a field saying zero are different facts, and
conflating them is the exact mistake this item is about.

    python -m tools.bench.detector_answer_rate ../asr-playground/out
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(path: Path) -> dict | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return None
    align = metadata.get("asr_align")
    if not isinstance(align, dict):
        return None
    return {
        "segments": len(document.get("segments") or []),
        "align": align,
        "vad": metadata.get("vad") or {},
    }


def _row(path: Path, parsed: dict) -> dict[str, object]:
    align = parsed["align"]
    qwen = align.get("qwen_verify")
    redecode = align.get("lang_redecode")
    coverage = align.get("audio_coverage") or {}
    return {
        "name": path.parent.name + "/" + path.name,
        "segments": parsed["segments"],
        "audio_sec": coverage.get("audio_sec"),
        "qwen_recorded": isinstance(qwen, dict),
        "suspects": (qwen or {}).get("suspects") if isinstance(qwen, dict) else None,
        "gaps_probed": (qwen or {}).get("gaps_probed") if isinstance(qwen, dict) else None,
        "gaps_recovered": (
            (qwen or {}).get("gaps_recovered") if isinstance(qwen, dict) else None
        ),
        "redecode_recorded": isinstance(redecode, dict),
        "triggers": (redecode or {}).get("triggers") if isinstance(redecode, dict) else None,
        "adopted": (redecode or {}).get("adopted") if isinstance(redecode, dict) else None,
    }


def tally(rows: list[dict[str, object]], field: str) -> dict[str, object]:
    """Sum one count field over the rows that actually recorded it.

    `sum(r[field] or 0 for r in rows)` is the bug this exists to avoid: it
    turns a *missing* field into a zero and folds it into both the numerator
    and the denominator, which is precisely the "unmeasured is not zero"
    conflation this module is about. A block can carry `qwen_verify` and still
    predate one of its counters.
    """

    present = [r for r in rows if isinstance(r.get(field), (int, float))]
    missing = len(rows) - len(present)
    return {
        "total": sum(int(r[field]) for r in present),
        "runs": len(present),
        "runs_missing": missing,
        "runs_nonzero": sum(1 for r in present if int(r[field]) > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--min-segments", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("*-aligned.json")):
        parsed = _load(path)
        if parsed is None or parsed["segments"] < args.min_segments:
            continue
        rows.append(_row(path, parsed))

    if not rows:
        print("FAIL: no usable artifacts found")
        return 2

    qwen_rows = [r for r in rows if r["qwen_recorded"]]
    redecode_rows = [r for r in rows if r["redecode_recorded"]]

    print(f"artifacts scanned          {len(rows)}")
    print(f"  with qwen_verify         {len(qwen_rows)}")
    print(f"  with lang_redecode       {len(redecode_rows)}")
    print()

    def line(label: str, stat: dict[str, object], *, of_segments: int | None = None) -> None:
        text = f"  {label:<24} {stat['total']}"
        if of_segments:
            text += f"  ({100 * stat['total'] / of_segments:.3f}% of segments)"
        text += f"   [recorded by {stat['runs']} runs"
        if stat["runs_missing"]:
            # Never silently folded into the total as zero.
            text += f", NOT RECORDED by {stat['runs_missing']}"
        text += f"; {stat['runs_nonzero']} nonzero]"
        print(text)

    if qwen_rows:
        suspects = tally(qwen_rows, "suspects")
        probed = tally(qwen_rows, "gaps_probed")
        recovered = tally(qwen_rows, "gaps_recovered")
        segments = sum(
            r["segments"] for r in qwen_rows if isinstance(r.get("suspects"), (int, float))
        )
        print("=== Qwen referee ===")
        print(f"  segments (recorded runs) {segments}")
        line("suspects", suspects, of_segments=segments)
        line("gaps probed", probed)
        line("gaps recovered", recovered)
        # Yield must come from runs that recorded *both* counters. Dividing a
        # `gaps_recovered` total by a `gaps_probed` total drawn from a different
        # set of runs is a ratio of two unrelated populations.
        paired = [
            r
            for r in qwen_rows
            if isinstance(r.get("gaps_probed"), (int, float))
            and isinstance(r.get("gaps_recovered"), (int, float))
        ]
        paired_probed = sum(int(r["gaps_probed"]) for r in paired)
        paired_recovered = sum(int(r["gaps_recovered"]) for r in paired)
        if paired_probed:
            print(
                f"  gap yield                {paired_recovered}/{paired_probed}"
                f" = {100 * paired_recovered / paired_probed:.0f}%"
                f"   [from the {len(paired)} runs recording both counters]"
            )
        elif probed["total"]:
            print("  gap yield                n/a (no run recorded both counters)")
        per_run = [
            100.0 * int(r["suspects"]) / r["segments"]
            for r in qwen_rows
            if isinstance(r.get("suspects"), (int, float)) and r["segments"]
        ]
        if per_run:
            print(
                f"  per-run suspect rate     median {statistics.median(per_run):.3f}%"
                f"  max {max(per_run):.3f}%"
            )
        print()
        ranked = [r for r in qwen_rows if isinstance(r.get("suspects"), (int, float))]
        print("  busiest runs:")
        for r in sorted(ranked, key=lambda r: -int(r["suspects"]))[:8]:
            probed_text = (
                r["gaps_probed"] if isinstance(r["gaps_probed"], (int, float)) else "n/r"
            )
            print(
                f"    {str(r['name'])[:46]:<46} segments {r['segments']:5d}  "
                f"suspects {int(r['suspects']):3d}  gaps_probed {probed_text}"
            )

    print()
    print("=== lang_redecode ===")
    if redecode_rows:
        triggers = tally(redecode_rows, "triggers")
        adopted = tally(redecode_rows, "adopted")
        line("triggers", triggers)
        line("adopted", adopted)
        print(f"  runs that fired          {triggers['runs_nonzero']}/{triggers['runs']}")
    else:
        print("  NOT RECORDED in any scanned artifact -- the field postdates them.")
        print("  This is 'unmeasured', not 'zero': see the module docstring.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
