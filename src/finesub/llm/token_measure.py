"""Measure subtitle token usage with Gemini countTokens or local counters."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .chunking import (
    load_segments_from_stable_json,
    render_segments_as_csv,
    render_segments_as_srt,
)
from .routing.config import GEMINI_31_FLASH_LITE
from .token_budget import (
    TokenCounter,
    default_token_counter,
)


@dataclass(frozen=True)
class SubtitleTokenFormatComparison:
    path: str
    segments: int
    counter_source: str
    model: str
    srt_chars: int
    csv_chars: int
    char_reduction: int
    char_reduction_pct: float
    srt_tokens: int
    csv_tokens: int
    token_reduction: int
    token_reduction_pct: float
    csv_to_srt_token_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator * 100.0, 1)


def compare_subtitle_token_formats(
    stable_json: str | Path,
    *,
    counter: TokenCounter,
    model: str = "",
) -> SubtitleTokenFormatComparison:
    path = Path(stable_json).expanduser().resolve()
    segments = load_segments_from_stable_json(path)
    srt_text = render_segments_as_srt(segments)
    csv_text = render_segments_as_csv(segments)
    srt_tokens = counter.count_text(srt_text)
    csv_tokens = counter.count_text(csv_text)
    token_reduction = srt_tokens - csv_tokens
    char_reduction = len(srt_text) - len(csv_text)
    ratio = round(csv_tokens / srt_tokens, 3) if srt_tokens > 0 else 0.0
    return SubtitleTokenFormatComparison(
        path=str(path),
        segments=len(segments),
        counter_source=counter.source,
        model=model,
        srt_chars=len(srt_text),
        csv_chars=len(csv_text),
        char_reduction=char_reduction,
        char_reduction_pct=_pct(char_reduction, len(srt_text)),
        srt_tokens=srt_tokens,
        csv_tokens=csv_tokens,
        token_reduction=token_reduction,
        token_reduction_pct=_pct(token_reduction, srt_tokens),
        csv_to_srt_token_ratio=ratio,
    )


def _build_counter(args: argparse.Namespace) -> TokenCounter:
    return default_token_counter(
        model=args.model,
        api_version=args.api_version,
    )


def _print_table(results: Sequence[SubtitleTokenFormatComparison]) -> None:
    headers = [
        "path",
        "segments",
        "srt_tokens",
        "csv_tokens",
        "token_reduction_pct",
        "csv_to_srt_ratio",
        "srt_chars",
        "csv_chars",
        "char_reduction_pct",
        "counter",
    ]
    print("\t".join(headers))  # product output
    for item in results:
        print(  # product output
            "\t".join(
                [
                    item.path,
                    str(item.segments),
                    str(item.srt_tokens),
                    str(item.csv_tokens),
                    f"{item.token_reduction_pct:.1f}",
                    f"{item.csv_to_srt_token_ratio:.3f}",
                    str(item.srt_chars),
                    str(item.csv_chars),
                    f"{item.char_reduction_pct:.1f}",
                    item.counter_source,
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SRT vs CSV subtitle token usage for *-stable.json files."
    )
    parser.add_argument("inputs", nargs="+", help="Path(s) to *-stable.json files.")
    parser.add_argument(
        "--model",
        default=GEMINI_31_FLASH_LITE,
        help="Gemini model used for countTokens.",
    )
    parser.add_argument("--api-version", default="v1beta")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of TSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counter = _build_counter(args)
    results = [
        compare_subtitle_token_formats(path, counter=counter, model=args.model)
        for path in args.inputs
    ]
    if args.json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
