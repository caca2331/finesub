"""Offline survey of fw-refine path signals in finished production artifacts.

Read-only. Scans aligned/stable JSON pairs under the given roots and
cross-tabulates, per output segment, the `alignment_events[]` left by the
adopted decodes against the evidence the pipeline already produces:

- the existing word-level detector (`text.detect_abnormal_asr_words`);
- the known hallucination phrase;
- the stable-stage outcome (dropped / tagged / kept), matched by time overlap.

This measures what the signals say about *post-rescue survivors* only: decodes
rejected during isolation/coverage rescue never reach these artifacts, so
decode-time recall cannot be estimated here (that needs an instrumented rerun
via run.py). See docs/wt-refine-validation.md for the 2026-08-04 findings.

Usage:
    python -m tools.wt_refine_validation.artifact_survey out/acceptance \
        --report tmp/signal-survey.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from asr_playground.text import COMMON_HALLUCINATION_TEXT, detect_abnormal_asr_words

STABLE_TIMELINE_TAGS = {"时间漂移", "mid_segment_start", "split_anchor_uncertain"}


def _load_segments(path: Path) -> Optional[List[dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    segments = data.get("segments") if isinstance(data, dict) else data
    return segments if isinstance(segments, list) else None


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def match_stable(segment: dict, stable_segments: List[dict], *, min_ratio: float = 0.3) -> Optional[dict]:
    """Best-overlap stable segment, or None when the segment was dropped."""

    start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
    best, best_overlap = None, 0.0
    for candidate in stable_segments:
        amount = _overlap(
            start, end, float(candidate.get("start", 0.0)), float(candidate.get("end", 0.0))
        )
        if amount > best_overlap:
            best, best_overlap = candidate, amount
    if best is not None and best_overlap / max(1e-6, end - start) >= min_ratio:
        return best
    return None


def _event_brief(event: dict) -> str:
    kind = str(event.get("type"))
    if kind == "alignment_stack":
        return f"stack(n={event.get('token_count')},tpaf={event.get('tokens_per_active_frame')})"
    if kind == "decoder_repetition":
        return f"rep(n={event.get('token_count')},x{event.get('repeat_count')})"
    if kind == "long_token_span":
        return f"span({event.get('duration')}s)"
    if kind == "unfinished":
        return f"unfin({event.get('token_count')})"
    if kind == "zero_duration_chunk_tail":
        return f"ztail('{event.get('word')}')"
    return kind


def survey_pair(aligned_path: Path, stable_path: Optional[Path]) -> Optional[dict]:
    """Survey one aligned/stable pair; None when it carries no signals at all
    (pre-signal artifact) or cannot be read."""

    segments = _load_segments(aligned_path)
    if not segments or not any(s.get("alignment_events") for s in segments):
        return None
    stable_segments = _load_segments(stable_path) if stable_path else None

    rows: List[dict] = []
    counters: Counter = Counter()
    for index, segment in enumerate(segments):
        words = segment.get("words") or []
        events = segment.get("alignment_events") or []
        issues = detect_abnormal_asr_words([words]) if words else []
        text = str(segment.get("text") or "")
        outcome = None
        tags: List[str] = []
        if stable_segments is not None:
            stable = match_stable(segment, stable_segments)
            if stable is None:
                outcome = "dropped"
            else:
                tags = list(stable.get("tags") or [])
                outcome = "tagged" if tags else "kept"
        counters["segments"] += 1
        if events:
            counters["segments_with_events"] += 1
        if issues:
            counters["segments_with_word_issues"] += 1
        if outcome == "dropped":
            counters["stable_dropped"] += 1
        elif outcome == "tagged":
            counters["stable_tagged"] += 1
        if not (events or issues or COMMON_HALLUCINATION_TEXT in text or outcome == "dropped" or tags):
            continue
        rows.append(
            {
                "index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": text[:120],
                "confidence": segment.get("confidence"),
                "events": events,
                "event_brief": [_event_brief(e) for e in events],
                "word_level_issues": issues,
                "known_phrase": COMMON_HALLUCINATION_TEXT in text,
                "stable_outcome": outcome,
                "stable_tags": tags,
                "timeline_tags_only": bool(tags) and set(tags) <= STABLE_TIMELINE_TAGS,
            }
        )
    return {"stats": dict(counters), "evidence_rows": rows}


def discover_pairs(roots: List[Path]) -> List[tuple[Path, Optional[Path]]]:
    pairs: List[tuple[Path, Optional[Path]]] = []
    for root in roots:
        for aligned in sorted(root.rglob("*aligned*.json")):
            if "wt-aligned" in aligned.name:
                continue
            stable = aligned.with_name(aligned.name.replace("aligned", "stable"))
            pairs.append((aligned, stable if stable.exists() else None))
    return pairs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=Path, help="directories holding aligned/stable JSON artifacts")
    parser.add_argument("--report", type=Path, default=None, help="write the full JSON report here")
    args = parser.parse_args(argv)

    report: Dict[str, dict] = {}
    totals: Counter = Counter()
    event_totals: Counter = Counter()
    for aligned, stable in discover_pairs(list(args.roots)):
        surveyed = survey_pair(aligned, stable)
        if surveyed is None:
            continue
        report[str(aligned)] = surveyed
        totals.update(surveyed["stats"])
        for row in surveyed["evidence_rows"]:
            for event in row["events"]:
                event_totals[str(event.get("type"))] += 1

    print(json.dumps({"totals": dict(totals), "events": dict(event_totals)}, ensure_ascii=False, indent=1))
    for material, surveyed in report.items():
        for row in surveyed["evidence_rows"]:
            if not row["events"]:
                continue
            flags = []
            if row["word_level_issues"]:
                flags.append("word-issue")
            if row["known_phrase"]:
                flags.append("known-phrase")
            outcome = row["stable_outcome"] or "?"
            tags = ",".join(row["stable_tags"])
            print(
                f"{material} #{row['index']} [{float(row['start'] or 0):.0f}s] "
                f"conf={row['confidence']} {outcome}{'(' + tags + ')' if tags else ''} "
                f"{' '.join(flags)}\n"
                f"    {row['text'][:70]}\n"
                f"    {'; '.join(row['event_brief'])}"
            )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
