"""Compare splitter behavior across real-ASR stable JSON variants.

The command performs no ASR.  For each condition it reads one stable JSON and
one VAD cache per dataset directory, calls the production ``segment_split``
implementation, and writes a deterministic Markdown report.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from difflib import SequenceMatcher
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import segment_split as sp  # noqa: E402
from subtitle_metrics import weighted_char_count  # noqa: E402


def _stats(items: list[tuple[float, float]]) -> dict[str, float | int]:
    durations = [duration for duration, _chars in items]
    chars = [chars for _duration, chars in items]
    return {
        "n": len(items),
        "dur_median": statistics.median(durations) if durations else 0.0,
        "dur_gt_4_5": sum(value > 4.5 for value in durations),
        "dur_gt_8": sum(value > 8.0 for value in durations),
        "dur_lt_0_6": sum(value < 0.6 for value in durations),
        "chars_gt_20": sum(value > 20.0 for value in chars),
        "chars_gt_36": sum(value > 36.0 for value in chars),
    }


def _load_intervals(cache_path: Path) -> list[tuple[float, float]]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    raw = data.get("intervals")
    if raw is None:
        raw = [[item["start"], item["end"]] for item in data.get("segments", [])]
    return [(float(start), float(end)) for start, end in raw]


def analyze(stable_path: Path, cache_path: Path, params: sp.SplitParams) -> dict:
    """Apply the production splitter to one stable artifact and collect metrics."""

    intervals = _load_intervals(cache_path)
    zones = sp.build_zones(intervals)
    payload = json.loads(stable_path.read_text(encoding="utf-8"))
    segments = [
        segment
        for segment in payload.get("segments", [])
        if segment.get("text") and segment.get("words")
    ]
    before: list[tuple[float, float]] = []
    after: list[tuple[float, float]] = []
    outliers: list[dict[str, object]] = []
    splits: list[dict[str, object]] = []
    canonical: list[dict[str, object]] = []

    for index, segment in enumerate(segments):
        adjusted = sp.adjust_words(segment["words"], intervals, zones)
        boundaries = sp.score_boundaries(adjusted, intervals, params)
        result = sp.dp_split(adjusted, boundaries, params)
        duration = float(segment["end"]) - float(segment["start"])
        chars = weighted_char_count(str(segment["text"]))
        before.append((duration, chars))

        if len(result.pieces) <= 1:
            pieces = [
                {
                    "start": round(float(segment["start"]), 3),
                    "end": round(float(segment["end"]), 3),
                    "text": str(segment["text"]),
                }
            ]
            after.append((duration, chars))
        else:
            pieces = []
            for start_word, end_word in result.pieces:
                piece_start = min(
                    max(adjusted[start_word].start, float(segment["start"])),
                    float(segment["end"]),
                )
                piece_end = min(
                    max(adjusted[end_word - 1].end, piece_start),
                    float(segment["end"]),
                )
                text = sp.piece_text(adjusted, start_word, end_word)
                after.append((piece_end - piece_start, weighted_char_count(text)))
                pieces.append(
                    {
                        "start": round(piece_start, 3),
                        "end": round(piece_end, 3),
                        "text": text,
                    }
                )

        canonical.append({"segment": index, "pieces": pieces})
        for piece in pieces:
            piece_duration = float(piece["end"]) - float(piece["start"])
            if piece_duration > 8.0:
                outliers.append(
                    {
                        "segment": index,
                        "source_duration": duration,
                        "source_words": len(segment["words"]),
                        "piece_duration": piece_duration,
                        "text": piece["text"],
                        "fallback_like": duration >= 20.0
                        and len(segment["words"]) <= 8,
                    }
                )

        if len(result.pieces) <= 1:
            continue
        cuts = []
        for piece_index in range(len(result.pieces) - 1):
            boundary_index = result.pieces[piece_index][1] - 1
            boundary = boundaries[boundary_index]
            cuts.append(
                {
                    "g": boundary.g,
                    "t": boundary.t,
                    "b": boundary.b,
                    "left": adjusted[boundary_index].text,
                    "right": adjusted[boundary_index + 1].text,
                }
            )
        splits.append(
            {
                "segment": index,
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "gain": result.no_split - result.total,
                "pieces": pieces,
                "cuts": cuts,
            }
        )

    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    word_text = "".join(
        str(word.get("word") or "")
        for segment in segments
        for word in segment.get("words", [])
    )
    return {
        "stable": str(stable_path),
        "segments": len(segments),
        "intervals": len(intervals),
        "split_segments": len(splits),
        "before_items": before,
        "after_items": after,
        "before": _stats(before),
        "after": _stats(after),
        "outliers": outliers,
        "splits": splits,
        "digest": digest,
        "word_text": word_text,
    }


def _condition_summary(results: list[dict]) -> dict:
    before = [value for item in results for value in item["before_items"]]
    after = [value for item in results for value in item["after_items"]]
    outliers = [
        {**outlier, "item": item["item"]}
        for item in results
        for outlier in item["outliers"]
    ]
    return {
        "before": _stats(before),
        "after": _stats(after),
        "split_total": sum(item["split_segments"] for item in results),
        "outliers": outliers,
        "fallback_like": sum(bool(item["fallback_like"]) for item in outliers),
        "digest": hashlib.sha256(
            "".join(item["digest"] for item in results).encode("ascii")
        ).hexdigest(),
        "word_text": "".join(item["word_text"] for item in results),
    }


def _parse_condition(value: str) -> tuple[str, str]:
    try:
        label, suffix = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("condition must be LABEL=SUFFIX") from exc
    if not label or not suffix:
        raise argparse.ArgumentTypeError("condition must be LABEL=SUFFIX")
    return label, suffix


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--condition",
        action="append",
        type=_parse_condition,
        required=True,
        help="repeatable LABEL=STABLE_SUFFIX",
    )
    for field in fields(sp.SplitParams):
        parser.add_argument(
            f"--{field.name.replace('_', '-')}",
            type=float,
            default=getattr(sp.DEFAULT_SPLIT_PARAMS, field.name),
        )
    args = parser.parse_args()
    params = sp.SplitParams(
        **{field.name: getattr(args, field.name) for field in fields(sp.SplitParams)}
    )
    run_root = args.run_root.resolve()
    output = args.output.resolve() if args.output else run_root / "stable.md"
    item_dirs = sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and (path / f"{path.name}-vad.json").exists()
    )
    if not item_dirs:
        parser.error(f"no dataset directories with <name>-vad.json under {run_root}")

    conditions = []
    for label, suffix in args.condition:
        results = []
        for item_dir in item_dirs:
            item = item_dir.name
            stable_path = item_dir / f"{item}{suffix}"
            if not stable_path.exists():
                parser.error(f"missing {stable_path} for condition {label!r}")
            result = analyze(
                stable_path,
                item_dir / f"{item}-vad.json",
                params,
            )
            result["item"] = item
            results.append(result)
        conditions.append(
            {
                "label": label,
                "suffix": suffix,
                "results": results,
                **_condition_summary(results),
            }
        )

    reference_text = conditions[0]["word_text"]
    lines = [
        "# Split Explorer 多条件报告",
        "",
        "## 口径",
        "",
        f"- 数据集：{len(item_dirs)} 项；所有条件均读取真实 ASR 的 stable JSON。",
        "- VAD interval 固定；切分调用生产 `src/segment_split.py`，不复制打分逻辑。",
        f"- 参数：`{json.dumps(sp.split_params_metadata(params), ensure_ascii=False)}`",
        "- 文本相似度仅描述 ASR 输出变化；没有人工转录时不代表识别准确率。",
        "",
        "## 汇总",
        "",
        "| 条件 | stable 段 | 被切段 | 切后段 | <0.6s | >4.5s | >8s | >36c | word 字符 | 相对首组文本相似度 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in conditions:
        before = condition["before"]
        after = condition["after"]
        similarity = SequenceMatcher(
            None,
            reference_text,
            condition["word_text"],
            autojunk=False,
        ).ratio()
        lines.append(
            f"| {_escape(condition['label'])} | {before['n']} | "
            f"{condition['split_total']} | {after['n']} | "
            f"{after['dur_lt_0_6']} | {after['dur_gt_4_5']} | "
            f"{after['dur_gt_8']} | {after['chars_gt_36']} | "
            f"{len(condition['word_text'])} | {similarity:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 每项明细",
            "",
            "| 条件 | 项目 | VAD | stable 段 | 被切段 | 切后段 | >8s 前→后 | >36c 前→后 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for condition in conditions:
        for item in condition["results"]:
            lines.append(
                f"| {_escape(condition['label'])} | {item['item']} | "
                f"{item['intervals']} | {item['segments']} | "
                f"{item['split_segments']} | {item['after']['n']} | "
                f"{item['before']['dur_gt_8']}→{item['after']['dur_gt_8']} | "
                f"{item['before']['chars_gt_36']}→{item['after']['chars_gt_36']} |"
            )

    lines.extend(
        [
            "",
            "## 切后 >8s 残留",
            "",
            "| 条件 | 项目 | 来源段 | 来源时长 | words | 残留时长 | fallback-like | 文本 |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for condition in conditions:
        for outlier in condition["outliers"]:
            lines.append(
                f"| {_escape(condition['label'])} | {outlier['item']} | "
                f"#{outlier['segment']} | {outlier['source_duration']:.2f}s | "
                f"{outlier['source_words']} | {outlier['piece_duration']:.2f}s | "
                f"{'是' if outlier['fallback_like'] else '否'} | "
                f"{_escape(outlier['text'])} |"
            )

    lines.extend(["", "## 实际切分明细", ""])
    for condition in conditions:
        lines.extend([f"### {condition['label']}", ""])
        for item in condition["results"]:
            for split in item["splits"]:
                cuts = "; ".join(
                    f"{cut['left']}|{cut['right']} g={cut['g']:.2f} "
                    f"T={cut['t']:.1f} B={cut['b']:.2f}"
                    for cut in split["cuts"]
                )
                pieces = " / ".join(piece["text"] for piece in split["pieces"])
                lines.append(
                    f"- `{item['item']} #{split['segment']}` "
                    f"{split['start']:.3f}–{split['end']:.3f}s，"
                    f"gain={split['gain']:.2f}；刀：{cuts}；结果：{pieces}"
                )
        lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
