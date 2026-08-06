"""Decode every production alignment group of the given clips, no rescue.

Reproduces the "310 production windows" corpus: saved VAD tracks are grouped
exactly like production (`build_alignment_groups`, gap 0.3s) and each group is
decoded once with the production greedy configuration, path signals on. The
raw first-pass result is dumped per window BEFORE any isolation/coverage
rescue, so downstream scoring can compare what each anomaly detector — the
existing word-level rules as well as signal-based candidates — would have seen
at decode time.

GPU + local model required. One JSONL row per window.

Usage:
    python -m tools.wt_refine_validation.window_sweep \
        --vad-dir out/qwen-explore \
        --corpus-root C:/Users/Carl/Documents/Carl/projects/asr-playground \
        --model C:/Users/Carl/Documents/Carl/models/faster-whisper-large-v3-turbo \
        --output tmp/window-sweep.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tools.wt_refine_validation.run import (  # noqa: E402
    event_time,
    interval_index_for_time,
    union_coverage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vad-dir", type=Path, required=True, help="directory of <clip>-vad.json tracks")
    parser.add_argument("--corpus-root", type=Path, required=True, help="root the VAD tracks' audio paths are relative to")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--clip", action="append", default=[], help="restrict to these clip stems")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-groups", type=int, default=0, help="per clip, for smoke tests")
    parser.add_argument(
        "--detect-disfluencies",
        action="store_true",
        help="enable the disfluency candidates (adds attention collection)",
    )
    return parser.parse_args()


def summarize_mapped_segment(segment: dict, interval_index: int) -> dict | None:
    """Mapped segments only carry words; derive span and text from them."""

    words = segment.get("words") or []
    if not words:
        return None
    text = "".join(
        (" " if word.get("space_before") else "") + str(word.get("word") or "")
        for word in words
    ).strip()
    summary = {
        "interval_index": interval_index,
        "start": float(words[0]["start"]),
        "end": float(words[-1]["end"]),
        "text": text,
        "confidence": segment.get("confidence"),
        "no_speech_prob": segment.get("no_speech_prob"),
    }
    events = segment.get("alignment_events")
    if isinstance(events, list) and events:
        summary["event_types"] = sorted(
            {str(event.get("type")) for event in events if isinstance(event, dict)}
        )
    return summary


def main() -> int:
    args = parse_args()

    import numpy as np

    from asr_playground.speech.recognition import transcribe as asr_align
    from asr_playground.speech.recognition.fw_refine_backend import RefinedWhisperModel

    vad_paths = sorted(args.vad_dir.glob("*-vad.json"))
    if args.clip:
        wanted = set(args.clip)
        vad_paths = [p for p in vad_paths if p.name[: -len("-vad.json")] in wanted]
        missing = wanted - {p.name[: -len("-vad.json")] for p in vad_paths}
        if missing:
            raise SystemExit(f"no VAD track for: {sorted(missing)}")
    if not vad_paths:
        raise SystemExit("no VAD tracks matched")

    model = RefinedWhisperModel(
        str(args.model.expanduser().resolve()),
        device=args.device,
        compute_type="float16" if args.device.startswith("cuda") else "float32",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    corpus_root = args.corpus_root.expanduser().resolve()
    total_windows = 0
    with args.output.open("w", encoding="utf-8") as sink:
        for vad_path in vad_paths:
            clip = vad_path.name[: -len("-vad.json")]
            track = json.loads(vad_path.read_text(encoding="utf-8"))
            audio_path = corpus_root / str(track["audio"])
            if not audio_path.exists():
                print(f"Warning: {clip}: audio missing at {audio_path}; skipped", file=sys.stderr)
                continue
            raw = [{"start": s, "end": e} for s, e in track["speech"]]
            intervals = asr_align.normalize_vad_segments(raw, float(track["duration"]))
            groups = asr_align.build_alignment_groups(intervals, gap_sec=0.3)
            if args.limit_groups:
                groups = groups[: args.limit_groups]
            loader = asr_align.AudioBlockLoader(
                str(audio_path),
                target_sr=16000,
                block_seconds=600.0,
                pad_seconds=10.0,
                preprocess=False,
            )
            try:
                for group_index, group in enumerate(groups):
                    combined, offsets = asr_align.build_combined_audio(
                        None, 16000, group, 0.3, audio_loader=loader
                    )
                    started = time.perf_counter()
                    result = model.transcribe_wt(
                        np.asarray(combined, dtype=np.float32),
                        beam_size=None,
                        best_of=None,
                        temperature=0.0,
                        collect_refine_signals=True,
                        detect_disfluencies=args.detect_disfluencies,
                    )
                    elapsed = time.perf_counter() - started
                    per_words, per_segments = asr_align._map_asr_result_to_intervals(
                        result, group, offsets
                    )
                    issue_by_interval = {
                        index: asr_align.detect_abnormal_asr_words([words])
                        for index, words in enumerate(per_words)
                        if words
                    }
                    events = []
                    for segment in result.get("segments", []):
                        for event in segment.get("alignment_events", []):
                            event = dict(event)
                            value = event_time(event)
                            if value is not None:
                                event["interval_index"] = interval_index_for_time(
                                    value, offsets, asr_align._combined_time_to_original
                                )
                            events.append(event)
                    all_words = [word for words in per_words for word in words]
                    speech_sec = sum(
                        float(item["end"]) - float(item["start"]) for item in group
                    )
                    mapped_segments = [
                        summary
                        for interval_index, segments in enumerate(per_segments)
                        for seg in segments
                        if (summary := summarize_mapped_segment(seg, interval_index))
                    ]
                    shortfall = asr_align._coverage_shortfall(group, mapped_segments)
                    mapped_events = [
                        dict(event)
                        for segments in per_segments
                        for seg in segments
                        for event in (seg.get("alignment_events") or [])
                        if isinstance(event, dict)
                    ]
                    row = {
                        "clip": clip,
                        "group_index": group_index,
                        "mapped_events": mapped_events,
                        "start": float(group[0]["start"]),
                        "end": float(group[-1]["end"]),
                        "interval_count": len(group),
                        "intervals": [
                            [float(item["start"]), float(item["end"])] for item in group
                        ],
                        "speech_sec": round(speech_sec, 6),
                        "audio_sec": round(len(combined) / 16000.0, 6),
                        "elapsed_sec": round(elapsed, 6),
                        "language": result.get("language"),
                        "segments": mapped_segments,
                        "per_interval_words": per_words,
                        "issues_by_interval": {
                            str(k): v for k, v in issue_by_interval.items() if v
                        },
                        "group_issues": asr_align.detect_abnormal_asr_words(per_words),
                        "events": events,
                        "segment_coverage_sec": round(
                            asr_align._covered_speech_seconds(group, mapped_segments), 6
                        ),
                        "coverage_low": shortfall is not None,
                        "coverage_shortfall": list(shortfall) if shortfall else None,
                        "word_coverage_sec": union_coverage(
                            all_words, float(group[0]["start"]), float(group[-1]["end"])
                        ),
                    }
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_windows += 1
            finally:
                loader.close()
            print(f"{clip}: {len(groups)} windows", flush=True)
    print(f"wrote {total_windows} windows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
