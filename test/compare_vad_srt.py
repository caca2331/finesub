"""Compare interval coverage between two SRT files.

This module is usable in two ways:
1) Callable API:
   from test.compare_vad_srt import compare_srt_files
2) CLI client:
   python -m test.compare_vad_srt a.srt b.srt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

Interval = Tuple[float, float]

_TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3})$")


@dataclass(frozen=True)
class IntervalComparison:
    label_a: str
    label_b: str
    intervals_a: int
    intervals_b: int
    total_a_sec: float
    total_b_sec: float
    intersection_sec: float
    union_sec: float
    jaccard_similarity: float
    jaccard_distance: float
    a_minus_b_over_union: float
    b_minus_a_over_union: float
    hash_a: str
    hash_b: str
    path_a: str
    path_b: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_srt_intervals(path: str | Path) -> List[Interval]:
    """Read intervals from SRT time lines."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    intervals: List[Interval] = []
    for line in text.splitlines():
        if "-->" not in line:
            continue
        left, right = [p.strip() for p in line.split("-->", 1)]
        start = _parse_srt_timestamp(left)
        end = _parse_srt_timestamp(right)
        if end > start:
            intervals.append((start, end))
    return intervals


def normalize_intervals(intervals: Sequence[Interval]) -> List[Interval]:
    """Sort and merge overlapping/touching intervals."""
    merged: List[List[float]] = []
    for start, end in sorted(intervals):
        s = float(start)
        e = float(end)
        if e <= s:
            continue
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
            continue
        merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


def compare_interval_sets(
    intervals_a: Sequence[Interval],
    intervals_b: Sequence[Interval],
    *,
    label_a: str = "A",
    label_b: str = "B",
    path_a: str = "",
    path_b: str = "",
    hash_a: str = "",
    hash_b: str = "",
) -> IntervalComparison:
    """Compute overlap metrics for two interval sets."""
    a = normalize_intervals(intervals_a)
    b = normalize_intervals(intervals_b)
    total_a = _total_len(a)
    total_b = _total_len(b)
    inter = _intersection_len(a, b)
    union = total_a + total_b - inter

    if union <= 0.0:
        jaccard_similarity = 1.0
        jaccard_distance = 0.0
        a_minus_b = 0.0
        b_minus_a = 0.0
    else:
        jaccard_similarity = inter / union
        jaccard_distance = 1.0 - jaccard_similarity
        a_minus_b = (total_a - inter) / union
        b_minus_a = (total_b - inter) / union

    return IntervalComparison(
        label_a=str(label_a),
        label_b=str(label_b),
        intervals_a=len(a),
        intervals_b=len(b),
        total_a_sec=total_a,
        total_b_sec=total_b,
        intersection_sec=inter,
        union_sec=union,
        jaccard_similarity=jaccard_similarity,
        jaccard_distance=jaccard_distance,
        a_minus_b_over_union=a_minus_b,
        b_minus_a_over_union=b_minus_a,
        hash_a=str(hash_a),
        hash_b=str(hash_b),
        path_a=str(path_a),
        path_b=str(path_b),
    )


def compare_srt_files(
    srt_a: str | Path,
    srt_b: str | Path,
    *,
    label_a: str = "A",
    label_b: str = "B",
) -> IntervalComparison:
    """Read two SRT files and compare interval coverage."""
    path_a = Path(srt_a)
    path_b = Path(srt_b)
    intervals_a = parse_srt_intervals(path_a)
    intervals_b = parse_srt_intervals(path_b)
    return compare_interval_sets(
        intervals_a,
        intervals_b,
        label_a=label_a,
        label_b=label_b,
        path_a=str(path_a),
        path_b=str(path_b),
        hash_a=file_sha256(path_a),
        hash_b=file_sha256(path_b),
    )


def format_report(result: IntervalComparison) -> str:
    return "\n".join(
        [
            f"{result.label_a}_path={result.path_a}",
            f"{result.label_b}_path={result.path_b}",
            f"{result.label_a}_hash_sha256={result.hash_a}",
            f"{result.label_b}_hash_sha256={result.hash_b}",
            f"{result.label_a}_intervals={result.intervals_a}",
            f"{result.label_b}_intervals={result.intervals_b}",
            f"{result.label_a}_total_sec={result.total_a_sec:.6f}",
            f"{result.label_b}_total_sec={result.total_b_sec:.6f}",
            f"intersection_sec={result.intersection_sec:.6f}",
            f"union_sec={result.union_sec:.6f}",
            f"jaccard_similarity={result.jaccard_similarity:.12f}",
            f"jaccard_distance={result.jaccard_distance:.12f}",
            f"({result.label_a}-{result.label_b})/(AUB)={result.a_minus_b_over_union:.12f}",
            f"({result.label_b}-{result.label_a})/(AUB)={result.b_minus_a_over_union:.12f}",
        ]
    )


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _parse_srt_timestamp(value: str) -> float:
    m = _TIME_RE.match(value)
    if m is None:
        raise ValueError(f"Invalid SRT timestamp: {value}")
    hh, mm, ss, ms = (int(x) for x in m.groups())
    return float(hh * 3600 + mm * 60 + ss + (ms / 1000.0))


def _total_len(intervals: Iterable[Interval]) -> float:
    total = 0.0
    for start, end in intervals:
        total += max(0.0, float(end) - float(start))
    return total


def _intersection_len(a: Sequence[Interval], b: Sequence[Interval]) -> float:
    i = 0
    j = 0
    out = 0.0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            out += (e - s)
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two SRT interval sets and print overlap metrics."
    )
    parser.add_argument("srt_a", help="Path to first SRT (A).")
    parser.add_argument("srt_b", help="Path to second SRT (B).")
    parser.add_argument(
        "--label-a",
        default="A",
        help="Label for first set in report (default: A).",
    )
    parser.add_argument(
        "--label-b",
        default="B",
        help="Label for second set in report (default: B).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output instead of text report.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="If set, exit with code 2 when jaccard_distance is greater than this value.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = compare_srt_files(
        args.srt_a,
        args.srt_b,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(result))

    if args.max_distance is not None and result.jaccard_distance > float(args.max_distance):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
