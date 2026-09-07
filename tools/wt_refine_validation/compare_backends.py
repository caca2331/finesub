"""Compare two aligned-json artifacts for the MLX release parity gate."""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _payload(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not an aligned-json object")
    return value


def _words(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        word
        for segment in payload.get("segments", [])
        for word in segment.get("words", [])
        if isinstance(word, dict)
    ]


def _coverage(words: list[dict[str, Any]]) -> float:
    return sum(
        max(0.0, float(word.get("end", 0.0)) - float(word.get("start", 0.0)))
        for word in words
    )


def compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_words = _words(reference)
    candidate_words = _words(candidate)
    paired = list(zip(reference_words, candidate_words))
    boundary_errors_ms = [
        error
        for left, right in paired
        for error in (
            abs(float(left.get("start", 0.0)) - float(right.get("start", 0.0))) * 1000,
            abs(float(left.get("end", 0.0)) - float(right.get("end", 0.0))) * 1000,
        )
    ]
    within_20 = (
        sum(error <= 20.0001 for error in boundary_errors_ms) / len(boundary_errors_ms)
        if boundary_errors_ms
        else 0.0
    )
    reference_coverage = _coverage(reference_words)
    candidate_coverage = _coverage(candidate_words)
    word_delta = (
        abs(len(candidate_words) - len(reference_words)) / len(reference_words)
        if reference_words
        else float(bool(candidate_words))
    )
    coverage_delta = (
        abs(candidate_coverage - reference_coverage) / reference_coverage
        if reference_coverage
        else float(bool(candidate_coverage))
    )
    anomalies = {
        "empty_segments": sum(
            not str(segment.get("text", "")).strip()
            for segment in candidate.get("segments", [])
        ),
        "reversed_words": sum(
            float(word.get("end", 0.0)) < float(word.get("start", 0.0))
            for word in candidate_words
        ),
    }
    reference_text = "\n".join(
        str(segment.get("text", "")) for segment in reference.get("segments", [])
    )
    candidate_text = "\n".join(
        str(segment.get("text", "")) for segment in candidate.get("segments", [])
    )
    return {
        "reference_word_count": len(reference_words),
        "candidate_word_count": len(candidate_words),
        "word_count_delta_ratio": round(word_delta, 6),
        "reference_coverage_sec": round(reference_coverage, 3),
        "candidate_coverage_sec": round(candidate_coverage, 3),
        "coverage_delta_ratio": round(coverage_delta, 6),
        "text_identical": [word.get("word") for word in reference_words]
        == [word.get("word") for word in candidate_words],
        "text_diff": list(
            difflib.unified_diff(
                reference_text.splitlines(),
                candidate_text.splitlines(),
                fromfile="ct2",
                tofile="mlx",
                lineterm="",
            )
        ),
        "boundary": {
            "paired": len(boundary_errors_ms),
            "mean_error_ms": round(mean(boundary_errors_ms), 3)
            if boundary_errors_ms
            else None,
            "max_error_ms": round(max(boundary_errors_ms), 3)
            if boundary_errors_ms
            else None,
            "within_20ms_ratio": round(within_20, 6),
        },
        "anomalies": anomalies,
        "acceptance": {
            "at_least_95pct_within_20ms": within_20 >= 0.95,
            "all_within_60ms": bool(boundary_errors_ms)
            and max(boundary_errors_ms) <= 60.0001,
            "word_count_within_1pct": word_delta <= 0.01,
            "coverage_within_1pct": coverage_delta <= 0.01,
            "no_empty_or_reversed": not any(anomalies.values()),
        },
        "reference_asr_metadata": reference.get("metadata", {}).get("asr_align", {}),
        "candidate_asr_metadata": candidate.get("metadata", {}).get("asr_align", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    report = compare(_payload(args.reference), _payload(args.candidate))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if all(report["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
