"""Score anomaly detectors over a window_sweep dump.

Computes, per window, which detectors fire:

- existing production rules, disaggregated: the word-level rules inside
  ``detect_abnormal_asr_words`` (parsed from issue prefixes) plus the coverage
  shortfall trigger;
- signal-based candidates built from fw-refine ``alignment_events``.

With ``--labels`` (a JSON adjudication file mapping "clip:group_index" to
{"label": "anomaly"|"benign", "type": subtype}) it reports per-detector and
combined coverage/precision against the adjudicated ground truth. Without
labels it emits the candidate list to adjudicate (every window any detector
flags) to ``--candidates``.

Offline and read-only; pairs with window_sweep.py.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

EXISTING_RULES = (
    "collapse_word_stack",
    "long_word_duration",
    "long_word_token",
    "repeating_token",
    "repeating_word_run",
    "repeating_group_cycle",
)


def latin_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


def window_id(row: dict) -> str:
    return f"{row['clip']}:{row['group_index']}"


def detector_flags(row: dict) -> dict[str, bool]:
    issues = list(row.get("group_issues") or [])
    flags: dict[str, bool] = {}
    for rule in EXISTING_RULES:
        flags[f"E:{rule}"] = any(issue.startswith(rule) for issue in issues)
    flags["E:coverage_low"] = bool(row.get("coverage_low"))

    events = row.get("events") or []
    by_type: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        by_type[str(event.get("type"))].append(event)

    stacks = by_type.get("alignment_stack", [])
    reps = by_type.get("decoder_repetition", [])
    flags["S:unfinished"] = bool(by_type.get("unfinished"))
    flags["S:long_token_span"] = bool(by_type.get("long_token_span"))
    flags["S:zero_duration_chunk_tail"] = bool(by_type.get("zero_duration_chunk_tail"))
    flags["S:decoder_repetition_any"] = bool(reps)
    flags["S:decoder_repetition_big"] = any(
        int(e.get("token_count") or 0) >= 64 for e in reps
    )
    flags["S:alignment_stack_any"] = bool(stacks)
    flags["S:alignment_stack_big"] = any(
        int(e.get("token_count") or 0) >= 8
        or float(e.get("tokens_per_active_frame") or 0.0) >= 4.0
        for e in stacks
    )
    flags["S:decode_limit_signature"] = flags["S:unfinished"] and (
        flags["S:decoder_repetition_big"] or flags["S:alignment_stack_big"]
    )

    # Language-switch hallucination candidate: a mostly-Latin low-confidence
    # segment inside a clip whose sweep output is dominantly CJK.
    lang_switch = False
    lang_switch_stack = False
    if row.get("_clip_dominant_cjk"):
        for segment in row.get("segments") or []:
            text = str(segment.get("text") or "")
            if len(text) < 8 or latin_ratio(text) < 0.7:
                continue
            confidence = segment.get("confidence")
            if isinstance(confidence, (int, float)) and confidence < 0.6:
                lang_switch = True
                seg_start = float(segment.get("start") or 0.0)
                seg_end = float(segment.get("end") or 0.0)
                for event in stacks:
                    anchor = event.get("start")
                    if (
                        isinstance(anchor, (int, float))
                        and seg_start - 1.0 <= float(anchor) <= seg_end + 1.0
                    ):
                        lang_switch_stack = True
    flags["S:lang_switch_lowconf"] = lang_switch
    flags["S:lang_switch_lowconf_stack"] = lang_switch_stack
    return flags


def annotate_clip_scripts(rows: list[dict]) -> None:
    text_by_clip: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for segment in row.get("segments") or []:
            text_by_clip[row["clip"]].append(str(segment.get("text") or ""))
    dominant: dict[str, bool] = {}
    for clip, texts in text_by_clip.items():
        joined = "".join(texts)
        dominant[clip] = latin_ratio(joined) < 0.5
    for row in rows:
        row["_clip_dominant_cjk"] = dominant.get(row["clip"], False)


def event_brief(event: dict) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", type=Path, help="window_sweep JSONL dump")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--flags-out", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.sweep.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    annotate_clip_scripts(rows)
    flags_by_window = {window_id(row): detector_flags(row) for row in rows}
    detectors = sorted({name for flags in flags_by_window.values() for name in flags})

    if args.flags_out:
        args.flags_out.write_text(
            json.dumps(flags_by_window, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    flagged = [
        row
        for row in rows
        if any(flags_by_window[window_id(row)].values())
    ]
    print(
        f"windows={len(rows)} flagged_by_any={len(flagged)} "
        f"({', '.join(f'{clip}:{n}' for clip, n in sorted(Counter(r['clip'] for r in rows).items()))})"
    )
    fire_counts = Counter(
        name
        for flags in flags_by_window.values()
        for name, value in flags.items()
        if value
    )
    for name in detectors:
        print(f"  {name}: {fire_counts.get(name, 0)}")

    if args.candidates:
        lines = []
        for row in flagged:
            wid = window_id(row)
            flags = flags_by_window[wid]
            active = [name for name, value in flags.items() if value]
            lines.append(
                json.dumps(
                    {
                        "id": wid,
                        "start": row["start"],
                        "end": row["end"],
                        "speech_sec": row["speech_sec"],
                        "flags": active,
                        "issues": row.get("group_issues"),
                        "events": [event_brief(e) for e in row.get("events") or []],
                        "coverage": [
                            row.get("segment_coverage_sec"),
                            row.get("speech_sec"),
                        ],
                        "segments": [
                            {
                                "t": f"{seg['start']:.1f}-{seg['end']:.1f}",
                                "conf": seg.get("confidence"),
                                "text": str(seg.get("text") or "")[:90],
                            }
                            for seg in row.get("segments") or []
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        args.candidates.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"candidates written to {args.candidates}")

    if not args.labels:
        return 0

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    truths = {wid for wid, item in labels.items() if item.get("label") == "anomaly"}
    known = set(labels)
    total_true = len(truths)
    print(f"\nadjudicated: {len(known)} windows, {total_true} true anomalies")

    def coverage_of(predicate) -> tuple[int, int]:
        true_positive = sum(
            1 for wid in truths if predicate(flags_by_window.get(wid, {}))
        )
        false_positive = sum(
            1
            for wid, flags in flags_by_window.items()
            if wid not in truths and predicate(flags)
        )
        return true_positive, false_positive

    print(f"{'detector':40s} {'TP':>3s} {'FP':>3s} {'coverage':>9s}")
    for name in detectors:
        tp, fp = coverage_of(lambda flags, n=name: flags.get(n, False))
        print(f"{name:40s} {tp:3d} {fp:3d} {tp / max(1, total_true):9.1%}")

    unions = {
        "UNION existing word rules": lambda f: any(
            f.get(f"E:{rule}", False) for rule in EXISTING_RULES
        ),
        "UNION existing incl coverage_low": lambda f: any(
            value for name, value in f.items() if name.startswith("E:")
        ),
        "UNION signals raw (any event)": lambda f: any(
            value for name, value in f.items() if name.startswith("S:")
        ),
        "UNION signals tuned": lambda f: f.get("S:decode_limit_signature")
        or f.get("S:decoder_repetition_big")
        or f.get("S:alignment_stack_big")
        or f.get("S:long_token_span")
        or f.get("S:lang_switch_lowconf"),
        "UNION existing + tuned signals": lambda f: any(
            value for name, value in f.items() if name.startswith("E:")
        )
        or f.get("S:decode_limit_signature")
        or f.get("S:decoder_repetition_big")
        or f.get("S:alignment_stack_big")
        or f.get("S:long_token_span")
        or f.get("S:lang_switch_lowconf"),
    }
    print()
    for name, predicate in unions.items():
        tp, fp = coverage_of(predicate)
        print(f"{name:40s} {tp:3d} {fp:3d} {tp / max(1, total_true):9.1%}")

    by_type: dict[str, list[str]] = defaultdict(list)
    for wid in truths:
        by_type[str(labels[wid].get("type") or "unspecified")].append(wid)
    print("\ntrue anomalies by type:")
    for subtype, wids in sorted(by_type.items()):
        print(f"  {subtype}: {len(wids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
