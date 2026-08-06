"""Run the anomaly-heavy WT refine validation corpus.

Heavy imports are intentionally delayed until after the patched CTranslate2
runtime and DLL directories are configured.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ct2-python", type=Path, required=True)
    parser.add_argument("--ct2-bin", type=Path, required=True)
    parser.add_argument("--cuda-bin", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-size", type=int, default=1)
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> list[Any]:
    ct2_python = args.ct2_python.expanduser().resolve()
    sys.path.insert(0, str(ct2_python))
    sys.path.insert(0, str(ROOT / "src"))
    handles = []
    if sys.platform == "win32":
        for directory in (
            ct2_python / "ctranslate2",
            args.ct2_bin.expanduser().resolve(),
            args.cuda_bin.expanduser().resolve(),
        ):
            handles.append(os.add_dll_directory(str(directory)))
    return handles


def normalize_text(text: str) -> str:
    return "".join(
        char.casefold()
        for char in unicodedata.normalize("NFKC", text)
        if char.isalnum()
    )


def edit_similarity(left: str, right: str) -> float | None:
    a = normalize_text(left)
    b = normalize_text(right)
    if not a and not b:
        return None
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for row, left_char in enumerate(a, start=1):
        current = [row]
        for column, right_char in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_char != right_char),
                )
            )
        previous = current
    return round(1.0 - previous[-1] / max(len(a), len(b)), 6)


def select_group(
    groups: list[list[dict[str, object]]],
    selector: dict[str, object],
) -> list[dict[str, object]]:
    mode = str(selector.get("mode") or "")
    if mode == "all":
        return [item for group in groups for item in group]
    if mode != "production_group_containing":
        raise ValueError(f"unsupported selector mode: {mode}")
    target = float(selector["interval_start"])
    matches = [
        group
        for group in groups
        if any(abs(float(item["start"]) - target) <= 0.001 for item in group)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"selector interval_start={target:.3f} matched {len(matches)} groups; "
            "the validation corpus drifted"
        )
    return matches[0]


def event_time(event: dict[str, object]) -> float | None:
    for field in ("start", "original_start", "peak_time", "refined_start", "end"):
        value = event.get(field)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def interval_index_for_time(
    value: float,
    offsets: list[tuple[int, float, float, float, float]],
    mapper,
) -> int | None:
    original = float(mapper(value, offsets))
    for index, start, end, _combined_start, _combined_end in offsets:
        if start <= original <= end:
            return int(index)
    if not offsets:
        return None
    return min(offsets, key=lambda item: min(abs(original - item[1]), abs(original - item[2])))[0]


def route_events(
    events: Iterable[dict[str, object]],
    *,
    interval_for_event,
    word_issue_intervals: Iterable[int] = (),
) -> dict[str, object]:
    """Research-only triage; it records a proposal and never mutates output."""

    hard: dict[int, list[str]] = defaultdict(list)
    deferred: dict[int, list[str]] = defaultdict(list)
    observations: dict[int, list[str]] = defaultdict(list)
    for event in events:
        event_type = str(event.get("type") or "")
        interval_index = interval_for_event(event)
        if interval_index is None:
            continue
        if event_type == "alignment_stack" and int(event.get("token_count") or 0) >= 3:
            hard[interval_index].append(event_type)
        elif event_type == "long_token_span":
            hard[interval_index].append(event_type)
        elif event_type == "zero_duration_chunk_tail":
            deferred[interval_index].append(event_type)
        elif event_type == "decoder_repetition":
            deferred[interval_index].append(event_type)
        elif event_type == "disfluency_candidate":
            observations[interval_index].append(event_type)
    for interval_index in word_issue_intervals:
        hard[int(interval_index)].append("word_level_abnormality")

    if len(hard) == 1:
        index = next(iter(hard))
        return {
            "route": "asr_immediate_isolation",
            "interval_index": index,
            "reason": sorted(set(hard[index])),
        }
    if len(hard) > 1:
        return {
            "route": "defer_finesub_regroup",
            "interval_indexes": sorted(hard),
            "reason": "multiple hard sites require a grouping decision",
        }
    if deferred:
        return {
            "route": "defer_finesub_decision",
            "interval_indexes": sorted(deferred),
            "reason": sorted(
                {item for values in deferred.values() for item in values}
            ),
        }
    if observations:
        return {
            "route": "keep_with_signals",
            "interval_indexes": sorted(observations),
            "reason": "disfluency candidates are observations, not retry triggers",
        }
    return {"route": "keep", "reason": "no actionable refine signal"}


def union_coverage(
    words: Iterable[dict[str, object]],
    start: float,
    end: float,
) -> float:
    spans = []
    for word in words:
        left = max(start, float(word.get("start") or 0.0))
        right = min(end, float(word.get("end") or left))
        if right > left:
            spans.append((left, right))
    spans.sort()
    covered = 0.0
    cursor = start
    for left, right in spans:
        if right <= cursor:
            continue
        left = max(left, cursor)
        covered += right - left
        cursor = right
    return round(covered, 6)


def reference_text(segments: Iterable[Any], start: float, end: float) -> str:
    return " ".join(
        str(segment.text)
        for segment in segments
        if float(segment.end) > start and float(segment.start) < end
    )


def focus_summary(
    per_words: list[list[dict[str, object]]],
    group: list[dict[str, object]],
    interval_index: int,
    reference_segments: list[Any],
) -> dict[str, object]:
    interval = group[interval_index]
    start = float(interval["start"])
    end = float(interval["end"])
    words = [
        word
        for interval_words in per_words
        for word in interval_words
        if start
        <= (float(word.get("start") or 0.0) + float(word.get("end") or 0.0)) / 2.0
        <= end
    ]
    hypothesis = "".join(str(word.get("word") or "") for word in words)
    overlapping_references = [
        segment
        for segment in reference_segments
        if float(segment.end) > start and float(segment.start) < end
    ]
    reference = " ".join(str(segment.text) for segment in overlapping_references)
    left_spill = max(
        [max(0.0, start - float(segment.start)) for segment in overlapping_references],
        default=0.0,
    )
    right_spill = max(
        [max(0.0, float(segment.end) - end) for segment in overlapping_references],
        default=0.0,
    )
    similarity = edit_similarity(hypothesis, reference)
    reference_reliable = (
        bool(overlapping_references)
        and max(left_spill, right_spill) <= 0.5
        and isinstance(similarity, (int, float))
        and similarity >= 0.25
    )
    return {
        "interval_index": interval_index,
        "start": start,
        "end": end,
        "hypothesis": hypothesis,
        "reference": reference,
        "reference_similarity": similarity,
        "reference_reliable": reference_reliable,
        "reference_max_boundary_spill_sec": round(max(left_spill, right_spill), 6),
        "word_coverage_sec": union_coverage(words, start, end),
        "speech_sec": round(end - start, 6),
    }


def candidate_summary(
    result: dict[str, object],
    group: list[dict[str, object]],
    offsets: list[tuple[int, float, float, float, float]],
    *,
    asr_align,
    reference_segments: list[Any],
    focus_interval: int | None,
    elapsed_sec: float,
    audio_sec: float,
) -> tuple[dict[str, object], list[list[dict[str, object]]], list[dict[str, object]]]:
    per_words, _per_segments = asr_align._map_asr_result_to_intervals(result, group, offsets)
    issues = asr_align.detect_abnormal_asr_words(per_words)
    issue_indexes = [
        index
        for index, words in enumerate(per_words)
        if words and asr_align.detect_abnormal_asr_words([words])
    ]
    events = [
        dict(event)
        for segment in result.get("segments", [])
        for event in segment.get("alignment_events", [])
    ]

    def mapped_event_interval(event: dict[str, object]) -> int | None:
        value = event_time(event)
        if value is None:
            return None
        return interval_index_for_time(value, offsets, asr_align._combined_time_to_original)

    for event in events:
        interval_index = mapped_event_interval(event)
        if interval_index is not None:
            event["interval_index"] = interval_index
    route = route_events(
        events,
        interval_for_event=lambda event: event.get("interval_index"),
        word_issue_intervals=issue_indexes,
    )
    all_words = [word for words in per_words for word in words]
    summary: dict[str, object] = {
        "elapsed_sec": round(elapsed_sec, 6),
        "audio_sec": round(audio_sec, 6),
        "compute_sec_per_audio_sec": round(elapsed_sec / max(audio_sec, 1e-9), 6),
        "segment_count": len(result.get("segments", [])),
        "word_count": len(all_words),
        "text": " ".join(str(segment.get("text") or "").strip() for segment in result.get("segments", [])),
        "issues": issues,
        "event_counts": dict(Counter(str(event.get("type")) for event in events)),
        "events": events,
        "route": route,
    }
    if focus_interval is not None:
        summary["focus"] = focus_summary(
            per_words,
            group,
            focus_interval,
            reference_segments,
        )
    return summary, per_words, events


def assess_retry(baseline: dict[str, object], retry: dict[str, object]) -> dict[str, object]:
    base_focus = baseline.get("focus") or {}
    retry_focus = retry.get("focus") or {}
    base_similarity = base_focus.get("reference_similarity")
    retry_similarity = retry_focus.get("reference_similarity")
    raw_similarity_delta = None
    if isinstance(base_similarity, (int, float)) and isinstance(retry_similarity, (int, float)):
        raw_similarity_delta = round(float(retry_similarity) - float(base_similarity), 6)
    reference_reliable = bool(base_focus.get("reference_reliable")) and bool(
        retry_focus.get("reference_reliable")
    )
    similarity_delta = raw_similarity_delta if reference_reliable else None
    coverage_delta = round(
        float(retry_focus.get("word_coverage_sec") or 0.0)
        - float(base_focus.get("word_coverage_sec") or 0.0),
        6,
    )
    baseline_bad = bool(baseline.get("issues"))
    retry_clean = not bool(retry.get("issues"))
    if similarity_delta is not None and similarity_delta >= 0.03:
        decision = "quality_improved"
    elif similarity_delta is not None and similarity_delta <= -0.03:
        decision = "quality_regressed"
    elif baseline_bad and retry_clean and coverage_delta >= -0.1 and reference_reliable:
        decision = "structurally_improved"
    elif baseline_bad and retry_clean and coverage_delta >= -0.1:
        decision = "structurally_improved_reference_ambiguous"
    elif retry_clean and coverage_delta >= -0.1 and reference_reliable:
        decision = "quality_similar_refine_efficiency_candidate"
    else:
        decision = "inconclusive"
    return {
        "decision": decision,
        "reference_similarity_delta": similarity_delta,
        "raw_reference_similarity_delta": raw_similarity_delta,
        "reference_reliable": reference_reliable,
        "word_coverage_delta_sec": coverage_delta,
        "retry_audio_fraction_of_group": round(
            float(retry.get("audio_sec") or 0.0)
            / max(float(baseline.get("audio_sec") or 0.0), 1e-9),
            6,
        ),
    }


def main() -> int:
    args = parse_args()
    runtime_handles = configure_runtime(args)

    import numpy as np

    from asr_playground.speech.preprocessing import vad as vad_detection
    from asr_playground.speech.recognition import transcribe as asr_align
    from asr_playground.speech.recognition.fw_refine_backend import RefinedWhisperModel
    from asr_playground.subtitles.model import parse_srt

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = set(args.case)
    cases = [case for case in manifest["cases"] if not wanted or case["id"] in wanted]
    missing = wanted - {case["id"] for case in cases}
    if missing:
        raise SystemExit(f"unknown cases: {sorted(missing)}")

    corpus_root = args.corpus_root.expanduser().resolve()
    source_cache: dict[str, dict[str, object]] = {}

    def load_source(source_id: str) -> dict[str, object]:
        cached = source_cache.get(source_id)
        if cached is not None:
            return cached
        spec = manifest["sources"][source_id]
        audio_path = corpus_root / spec["audio"]
        raw, _meta, duration, _timing, _track = vad_detection.detect_segments(audio_path)
        intervals = asr_align.normalize_vad_segments(raw, duration)
        groups = asr_align.build_alignment_groups(intervals, gap_sec=0.3)
        references: list[Any] = []
        if spec.get("reference_srt"):
            reference_path = corpus_root / spec["reference_srt"]
            references = parse_srt(reference_path.read_text(encoding="utf-8"))
        cached = {
            "spec": spec,
            "audio_path": audio_path,
            "duration": duration,
            "groups": groups,
            "references": references,
        }
        source_cache[source_id] = cached
        return cached

    model = RefinedWhisperModel(
        str(args.model.expanduser().resolve()),
        device=args.device,
        compute_type="float16" if args.device.startswith("cuda") else "float32",
    )
    rows: list[dict[str, object]] = []
    for case in cases:
        source = load_source(case["source"])
        group = select_group(source["groups"], case["selector"])
        loader = asr_align.AudioBlockLoader(
            str(source["audio_path"]),
            target_sr=16000,
            block_seconds=600.0,
            pad_seconds=10.0,
            preprocess=False,
        )
        try:
            combined, offsets = asr_align.build_combined_audio(
                None,
                16000,
                group,
                0.3,
                audio_loader=loader,
            )
            focus_index = None
            if case["selector"]["mode"] == "production_group_containing":
                target = float(case["selector"]["interval_start"])
                focus_index = next(
                    index
                    for index, interval in enumerate(group)
                    if abs(float(interval["start"]) - target) <= 0.001
                )
            started = time.perf_counter()
            result = model.transcribe_wt(
                np.asarray(combined, dtype=np.float32),
                language=source["spec"].get("language"),
                beam_size=args.beam_size,
                best_of=args.beam_size,
                temperature=0.0,
                collect_refine_signals=True,
                collect_attention_signals=True,
            )
            baseline, baseline_words, _events = candidate_summary(
                result,
                group,
                offsets,
                asr_align=asr_align,
                reference_segments=source["references"],
                focus_interval=focus_index,
                elapsed_sec=time.perf_counter() - started,
                audio_sec=len(combined) / 16000.0,
            )
            row: dict[str, object] = {
                "id": case["id"],
                "bucket": case["bucket"],
                "source": case["source"],
                "evidence": case.get("evidence"),
                "group": {
                    "interval_count": len(group),
                    "start": float(group[0]["start"]),
                    "end": float(group[-1]["end"]),
                    "speech_sec": round(
                        sum(float(item["end"]) - float(item["start"]) for item in group),
                        6,
                    ),
                },
                "baseline": baseline,
            }

            retry_index = None
            route = baseline["route"]
            if route.get("route") == "asr_immediate_isolation":
                retry_index = int(route["interval_index"])
            elif case["bucket"] == "anomaly":
                # Historical probe measures the isolation upper bound even when
                # the new signal policy misses; it does not change the route.
                retry_index = focus_index
            if retry_index is not None:
                retry_group = [group[retry_index]]
                retry_audio, retry_offsets = asr_align.build_combined_audio(
                    None,
                    16000,
                    retry_group,
                    0.3,
                    audio_loader=loader,
                )
                retry_started = time.perf_counter()
                retry_result = model.transcribe_wt(
                    np.asarray(retry_audio, dtype=np.float32),
                    language=source["spec"].get("language"),
                    beam_size=args.beam_size,
                    best_of=args.beam_size,
                    temperature=0.0,
                    collect_refine_signals=True,
                    collect_attention_signals=True,
                )
                retry, _retry_words, _retry_events = candidate_summary(
                    retry_result,
                    retry_group,
                    retry_offsets,
                    asr_align=asr_align,
                    reference_segments=source["references"],
                    focus_interval=0,
                    elapsed_sec=time.perf_counter() - retry_started,
                    audio_sec=len(retry_audio) / 16000.0,
                )
                row["isolation_retry"] = retry
                row["retry_trigger"] = (
                    "policy" if route.get("route") == "asr_immediate_isolation" else "historical_probe"
                )
                assessment_baseline = baseline
                if retry_index != focus_index:
                    assessment_baseline = dict(baseline)
                    assessment_baseline["focus"] = focus_summary(
                        baseline_words,
                        group,
                        retry_index,
                        source["references"],
                    )
                    row["retry_comparison_focus"] = assessment_baseline["focus"]
                row["retry_assessment"] = assess_retry(assessment_baseline, retry)
            rows.append(row)
            print(
                f"{case['id']}: route={baseline['route']['route']} "
                f"events={baseline['event_counts']} issues={len(baseline['issues'])}",
                flush=True,
            )
        finally:
            loader.close()

    policy_retries = [row for row in rows if row.get("retry_trigger") == "policy"]
    retry_fractions = [
        float(row["retry_assessment"]["retry_audio_fraction_of_group"])
        for row in policy_retries
        if row.get("retry_assessment")
    ]
    payload = {
        "manifest_version": manifest["version"],
        "model": str(args.model.expanduser().resolve()),
        "device": args.device,
        "beam_size": args.beam_size,
        "case_count": len(rows),
        "bucket_counts": dict(Counter(str(row["bucket"]) for row in rows)),
        "summary": {
            "routes_by_bucket": {
                bucket: dict(
                    Counter(
                        str(row["baseline"]["route"]["route"])
                        for row in rows
                        if row["bucket"] == bucket
                    )
                )
                for bucket in ("normal", "anomaly")
            },
            "signal_case_counts": dict(
                Counter(
                    event_type
                    for row in rows
                    for event_type in row["baseline"]["event_counts"]
                )
            ),
            "policy_retry_count": len(policy_retries),
            "policy_retry_decisions": dict(
                Counter(
                    str(row["retry_assessment"]["decision"])
                    for row in policy_retries
                    if row.get("retry_assessment")
                )
            ),
            "policy_retry_audio_fraction_median": round(
                statistics.median(retry_fractions), 6
            )
            if retry_fractions
            else None,
        },
        "cases": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output}")
    assert runtime_handles is not None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
