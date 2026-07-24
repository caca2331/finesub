"""Simple whisper-timestamped client that writes SRT from an audio file. (for comparison, not used in the pipeline)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_MODEL = "large-v3-turbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run whisper-timestamped on audio and write SRT."
    )
    parser.add_argument("input", help="Path to input audio file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output SRT (default: <input>-wt.srt).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name.")
    parser.add_argument("--device", default=None, help="Device override (cpu/cuda).")
    parser.add_argument("--language", default=None, help="Language override.")
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Optional length penalty passed to whisper decoding.",
    )
    parser.add_argument(
        "--accurate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable higher-quality decoding "
            "(beam_size=5, best_of=5, temperature=[0.0,0.2,0.4,0.6,0.8,1.0])."
        ),
    )
    parser.add_argument(
        "--word",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write word-level SRT (default: False).",
    )
    return parser.parse_args()


def default_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-wt.srt")


def default_word_output_path(input_path: Path) -> Path:
    base = input_path.with_suffix("")
    return base.with_name(f"{base.name}-wt_word.srt")


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(float(seconds) * 1000.0))
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def render_segment_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    idx = 1
    for seg in segments:
        start = seg.get("start")
        end = seg.get("end")
        text = seg.get("text") or '""'
        if start is None or end is None:
            continue
        try:
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
        lines.append(str(text).strip())
        lines.append("")
        idx += 1
    return "\n".join(lines).strip() + "\n"


def render_word_srt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    idx = 1
    found_words = False
    for seg in segments:
        for word in seg.get("words") or []:
            found_words = True
            start = word.get("start")
            end = word.get("end")
            text = word.get("text") or word.get("word") or word.get("token") or ""
            if start is None or end is None:
                continue
            try:
                start_s = float(start)
                end_s = float(end)
            except (TypeError, ValueError):
                continue
            if end_s <= start_s:
                continue
            lines.append(str(idx))
            lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
            lines.append(str(text).strip())
            lines.append("")
            idx += 1
    if not found_words:
        raise ValueError("No word timestamps found in whisper-timestamped output.")
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )

    try:
        import torch
    except Exception as exc:
        print(f"Missing dependency: torch ({exc})", file=sys.stderr)
        return 1

    try:
        import whisper_timestamped as whisper
    except Exception:
        print(
            "Missing dependency: whisper-timestamped. Install with `pip install whisper-timestamped`.",
            file=sys.stderr,
        )
        return 1

    if args.device:
        device = args.device
        if str(device).strip().lower() == "cuda" and not torch.cuda.is_available():
            print(
                "Warning: CUDA requested for whisper-timestamped but unavailable; falling back to CPU.",
                file=sys.stderr,
            )
            device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        print(
            "Warning: CUDA is the default whisper-timestamped device but is unavailable; falling back to CPU.",
            file=sys.stderr,
        )
        device = "cpu"

    try:
        model = whisper.load_model(args.model, device=device)
        audio = whisper.load_audio(str(input_path))
        kwargs: Dict[str, Any] = {
            "verbose": False,
            "vad": False,
            "condition_on_previous_text": False,
            "detect_disfluencies": True,
            "fp16": True,
            "refine_whisper_precision": 1.0,
            # "no_speech_threshold": 1.0,
            # "logprob_threshold": -1.2,
        }
        if args.accurate:
            kwargs["beam_size"] = 5
            kwargs["best_of"] = 5
            kwargs["temperature"] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        if args.length_penalty is not None:
            kwargs["length_penalty"] = float(args.length_penalty)
        if args.language:
            kwargs["language"] = args.language
        result = whisper.transcribe(model, audio, **kwargs)
    except Exception as exc:
        print(f"Failed to run whisper-timestamped: {exc}", file=sys.stderr)
        return 1

    segments = result.get("segments") or []
    if not isinstance(segments, list):
        print("Unexpected whisper-timestamped output: missing segments list.", file=sys.stderr)
        return 1

    try:
        srt_text = render_segment_srt(segments)
    except Exception as exc:
        print(f"Failed to render segment SRT: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(srt_text, encoding="utf-8")
    print(f"Wrote {output_path}")

    if args.word:
        word_output_path = default_word_output_path(input_path)
        try:
            word_srt_text = render_word_srt(segments)
        except Exception as exc:
            print(f"Failed to render word SRT: {exc}", file=sys.stderr)
            return 1
        word_output_path.write_text(word_srt_text, encoding="utf-8")
        print(f"Wrote {word_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
