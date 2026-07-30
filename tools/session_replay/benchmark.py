"""Offline merge/drop scoring for a frozen correction replay fixture.

The benchmark is deliberately asymmetric: joining a boundary classified as
``must_not_merge`` starts at the same boundary cost as leaving a
``must_merge`` boundary split, then receives cumulative soft/hard length
surcharges; discarding a ``must_keep`` source costs more than retaining a
``must_drop`` source. This module never calls an LLM or the network.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from llm.chunking import WindowIdMap
from llm.csv_utils import remap_validation_source_ids, validate_translated_csv_text
from llm.exchange_metadata import extract_top_level_tagged_blocks
from asr_playground.subtitles.metrics import weighted_char_count

from .fixture import build_window_from_fixture, load_fixture


@dataclass(frozen=True)
class BenchmarkScore:
    reply: str
    valid: bool
    validation_errors: tuple[str, ...]
    merge_boundaries: int
    discarded_sources: int
    overmerge: tuple[str, ...]
    undermerge: tuple[str, ...]
    false_drop: tuple[str, ...]
    missed_drop: tuple[str, ...]
    soft_limit_overmerge: tuple[str, ...]
    hard_limit_overmerge: tuple[str, ...]
    start_checked_rows: int
    start_mismatches: tuple[str, ...]
    weighted_cost: int | None


def source_fingerprint(source_segments: Sequence[Any]) -> str:
    """Return a stable hash of source ids, timing, and ASR text."""

    body = "\n".join(
        f"{segment.id}|{segment.start:.3f}|{segment.end:.3f}|{segment.text}"
        for segment in source_segments
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_benchmark(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Benchmark must be a JSON object: {path}")
    return dict(data)


def _pair(left: str, right: str) -> str:
    return f"{left}-{right}"


def _as_set(data: Mapping[str, Any], section: str, key: str) -> set[str]:
    section_data = data.get(section)
    if not isinstance(section_data, Mapping):
        raise ValueError(f"Benchmark is missing object {section!r}.")
    values = section_data.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Benchmark {section}.{key} must be a list of strings.")
    return set(values)


def _model_response_from_reply(text: str) -> str:
    """Strip the replay Markdown envelope when ``text`` is a saved reply.

    Validation warnings in the JSON preamble may themselves contain literal
    tags such as ``<void>``. Feeding that envelope to the tagged-block parser
    can therefore corrupt nesting even though the actual model response is
    valid. Raw response files without the heading remain supported.
    """

    marker = "\n## 模型响应\n"
    if marker in text:
        return text.split(marker, 1)[1].lstrip()
    return text


def _start_alignment(
    content: str,
    source_segments: Sequence[Any],
    *,
    clip_start: float = 0.0,
) -> tuple[int, tuple[str, ...]]:
    """Compare reported start cells with the first source's rendered start.

    This is diagnostic only. The production timeline continues to come from
    source ids, and a mismatch must not change structural validity or cost.
    """

    blocks = extract_top_level_tagged_blocks(content, "translated")
    if len(blocks) != 1:
        return 0, ()
    expected = {
        str(index): float(f"{float(segment.start) - clip_start:.1f}")
        for index, segment in enumerate(source_segments, start=1)
    }
    checked = 0
    mismatches: list[str] = []

    for raw_row in blocks[0].splitlines():
        row = raw_row.strip()
        if not row or row.startswith("#") or row.startswith("type|"):
            continue
        lowered_fields = [field.strip().lower() for field in row.split("|")]
        if row.lower().rstrip().endswith("<void>") or "<void>" in lowered_fields:
            continue
        fields = row.split("|", 9)
        if len(fields) < 10 or fields[0].strip().lower() != "sub":
            continue
        first_source = fields[1].split(",", 1)[0].strip()
        if first_source not in expected:
            continue
        checked += 1
        actual = float(fields[2].strip())
        wanted = expected[first_source]
        if abs(actual - wanted) > 0.051:
            mismatches.append(f"{fields[1].strip()}: got {actual:g}, expected {wanted:g}")
    return checked, tuple(mismatches)


def validate_benchmark(data: Mapping[str, Any], source_segments: Sequence[Any]) -> None:
    expected_count = int(data.get("source_count") or 0)
    if expected_count != len(source_segments):
        raise ValueError(
            f"Benchmark expects {expected_count} sources, fixture has {len(source_segments)}."
        )
    expected_fingerprint = str(data.get("source_fingerprint_sha256") or "")
    actual_fingerprint = source_fingerprint(source_segments)
    if expected_fingerprint != actual_fingerprint:
        raise ValueError(
            "Benchmark source fingerprint does not match the fixture: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}."
        )

    boundaries = {
        _pair(left.id, right.id)
        for left, right in zip(source_segments, source_segments[1:])
    }
    merge_data = data.get("merge")
    if not isinstance(merge_data, Mapping) or merge_data.get("default") != "must_not_merge":
        raise ValueError("Benchmark merge.default must be 'must_not_merge'.")
    must_merge = _as_set(data, "merge", "must_merge")
    may_merge = _as_set(data, "merge", "may_merge")
    if must_merge & may_merge:
        raise ValueError("merge.must_merge and merge.may_merge must be disjoint.")
    unknown_boundaries = (must_merge | may_merge) - boundaries
    if unknown_boundaries:
        raise ValueError(f"Benchmark names unknown boundaries: {sorted(unknown_boundaries)}")

    source_ids = {str(segment.id) for segment in source_segments}
    drop_data = data.get("drop")
    if not isinstance(drop_data, Mapping) or drop_data.get("default") != "must_keep":
        raise ValueError("Benchmark drop.default must be 'must_keep'.")
    must_drop = _as_set(data, "drop", "must_drop")
    may_drop = _as_set(data, "drop", "may_drop")
    if must_drop & may_drop:
        raise ValueError("drop.must_drop and drop.may_drop must be disjoint.")
    unknown_ids = (must_drop | may_drop) - source_ids
    if unknown_ids:
        raise ValueError(f"Benchmark names unknown source ids: {sorted(unknown_ids)}")


def score_reply(
    *,
    reply_path: Path,
    benchmark: Mapping[str, Any],
    source_segments: Sequence[Any],
    clip_start: float = 0.0,
    require_start_column: bool = False,
) -> BenchmarkScore:
    """Parse one replay reply and compare its merge/drop decisions to gold."""

    content = _model_response_from_reply(reply_path.read_text(encoding="utf-8"))
    id_map = WindowIdMap.from_segments(source_segments)
    validation = validate_translated_csv_text(
        content,
        id_map.localize_segments(source_segments),
        clip_start=clip_start,
        allow_insert=True,
        require_singles=False,
        require_start_column=require_start_column,
        forbid_start_column=not require_start_column,
    )
    validation = remap_validation_source_ids(validation, id_map)
    errors = tuple(validation.errors)
    if not validation.ok:
        return BenchmarkScore(
            reply=str(reply_path),
            valid=False,
            validation_errors=errors,
            merge_boundaries=0,
            discarded_sources=0,
            overmerge=(),
            undermerge=(),
            false_drop=(),
            missed_drop=(),
            soft_limit_overmerge=(),
            hard_limit_overmerge=(),
            start_checked_rows=0,
            start_mismatches=(),
            weighted_cost=None,
        )

    predicted_merge = {
        _pair(left, right)
        for segment in validation.segments
        for left, right in zip(segment.source_ids, segment.source_ids[1:])
    }
    predicted_drop = set(validation.discarded_ids)
    source_order = {str(segment.id): index for index, segment in enumerate(source_segments)}

    def source_order_key(value: str) -> int:
        first_id = value.replace(",", "-").split("-", 1)[0]
        return source_order.get(first_id, len(source_order))

    must_merge = _as_set(benchmark, "merge", "must_merge")
    may_merge = _as_set(benchmark, "merge", "may_merge")
    must_drop = _as_set(benchmark, "drop", "must_drop")
    may_drop = _as_set(benchmark, "drop", "may_drop")

    overmerge = tuple(sorted(predicted_merge - must_merge - may_merge, key=source_order_key))
    undermerge = tuple(sorted(must_merge - predicted_merge, key=source_order_key))
    false_drop = tuple(sorted(predicted_drop - must_drop - may_drop, key=source_order_key))
    missed_drop = tuple(sorted(must_drop - predicted_drop, key=source_order_key))

    weights = benchmark.get("weights")
    if not isinstance(weights, Mapping):
        raise ValueError("Benchmark is missing object 'weights'.")
    limits = benchmark.get("limits")
    if not isinstance(limits, Mapping):
        raise ValueError("Benchmark is missing object 'limits'.")

    soft_limit_overmerge: list[str] = []
    hard_limit_overmerge: list[str] = []
    for segment in validation.segments:
        segment_boundaries = {
            _pair(left, right)
            for left, right in zip(segment.source_ids, segment.source_ids[1:])
        }
        wrong_boundaries = segment_boundaries - must_merge - may_merge
        if not wrong_boundaries:
            continue
        duration = segment.end - segment.start
        chars = weighted_char_count(segment.translation)
        label = ",".join(segment.source_ids)
        if duration > float(limits["hard_duration_seconds"]) or chars > float(
            limits["hard_weighted_chars"]
        ):
            hard_limit_overmerge.append(label)
        elif duration > float(limits["soft_duration_seconds"]) or chars > float(
            limits["soft_weighted_chars"]
        ):
            soft_limit_overmerge.append(label)

    weighted_cost = (
        int(weights["overmerge_boundary"]) * len(overmerge)
        + int(weights["undermerge"]) * len(undermerge)
        + int(weights["false_drop"]) * len(false_drop)
        + int(weights["missed_drop"]) * len(missed_drop)
        # A hard-limit row has necessarily crossed the soft limit as well:
        # charge the soft surcharge first, then the additional hard surcharge.
        + int(weights["soft_limit_excess"])
        * (len(soft_limit_overmerge) + len(hard_limit_overmerge))
        + int(weights["hard_limit_excess"]) * len(hard_limit_overmerge)
    )
    start_checked_rows, start_mismatches = (
        _start_alignment(content, source_segments, clip_start=clip_start)
        if require_start_column
        else (0, ())
    )
    return BenchmarkScore(
        reply=str(reply_path),
        valid=True,
        validation_errors=(),
        merge_boundaries=len(predicted_merge),
        discarded_sources=len(predicted_drop),
        overmerge=overmerge,
        undermerge=undermerge,
        false_drop=false_drop,
        missed_drop=missed_drop,
        soft_limit_overmerge=tuple(soft_limit_overmerge),
        hard_limit_overmerge=tuple(hard_limit_overmerge),
        start_checked_rows=start_checked_rows,
        start_mismatches=start_mismatches,
        weighted_cost=weighted_cost,
    )


def render_markdown(scores: Sequence[BenchmarkScore]) -> str:
    lines = [
        "| reply | valid | merged boundaries | drops | overmerge | soft-limit | hard-limit | undermerge | false drop | missed drop | start mismatch | weighted cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for score in scores:
        cost = "—" if score.weighted_cost is None else str(score.weighted_cost)
        lines.append(
            f"| {Path(score.reply).name} | {'yes' if score.valid else 'no'} | "
            f"{score.merge_boundaries} | {score.discarded_sources} | "
            f"{len(score.overmerge)} | {len(score.soft_limit_overmerge)} | "
            f"{len(score.hard_limit_overmerge)} | {len(score.undermerge)} | "
            f"{len(score.false_drop)} | {len(score.missed_drop)} | "
            f"{len(score.start_mismatches)}/{score.start_checked_rows} | {cost} |"
        )
    for score in scores:
        lines.extend(["", f"### {score.reply}"])
        if not score.valid:
            lines.append("- invalid: " + "; ".join(score.validation_errors))
            continue
        lines.extend(
            [
                f"- overmerge: {', '.join(score.overmerge) or 'none'}",
                f"- undermerge: {', '.join(score.undermerge) or 'none'}",
                f"- false drop: {', '.join(score.false_drop) or 'none'}",
                f"- missed drop: {', '.join(score.missed_drop) or 'none'}",
                f"- soft-limit overmerge rows: {', '.join(score.soft_limit_overmerge) or 'none'}",
                f"- hard-limit overmerge rows: {', '.join(score.hard_limit_overmerge) or 'none'}",
                f"- start mismatches: {', '.join(score.start_mismatches) or 'none'}",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument(
        "--start-column",
        action="store_true",
        help="Parse a position|start|duration output schema and audit start alignment.",
    )
    parser.add_argument("replies", type=Path, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fixture = load_fixture(args.fixture)
    window = build_window_from_fixture(fixture)
    benchmark = load_benchmark(args.benchmark)
    validate_benchmark(benchmark, window.segments)
    scores = [
        score_reply(
            reply_path=reply,
            benchmark=benchmark,
            source_segments=window.segments,
            clip_start=window.clip_start,
            require_start_column=args.start_column,
        )
        for reply in args.replies
    ]
    if args.json:
        print(json.dumps([asdict(score) for score in scores], ensure_ascii=False, indent=2))
    else:
        print(render_markdown(scores), end="")
    return 0 if all(score.valid for score in scores) else 2


if __name__ == "__main__":
    raise SystemExit(main())
