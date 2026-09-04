"""Convert pipeline JSON segments to SRT subtitle files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..reporting import current_reporter, reporting_to, terminal_reporter
from . import time_order
from .model import warn_on_invalid_srt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert pipeline JSON to SRT.")
    parser.add_argument("input", help="Path to pipeline output JSON.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output SRT (default: <input>.srt).",
    )
    parser.add_argument(
        "--word",
        "-w",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write word-level SRT (default: False).",
    )
    return parser.parse_args()


def format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000.0))
    if total_ms < 0:
        total_ms = 0
    hours, rem = divmod(total_ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _timed(segments: List[Dict[str, object]]) -> List[tuple]:
    """(start, end, text) for renderable cues, ordered by start. Non-positive spans dropped."""
    out = []
    for seg in segments:
        start, end = seg.get("start"), seg.get("end")
        if start is None or end is None:
            continue
        try:
            start_s, end_s = float(start), float(end)
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        out.append((start_s, end_s, seg.get("text") or "\"\""))
    out.sort(key=lambda r: (r[0], r[1]))
    return out


def resolve_overlaps(rows: List[tuple]) -> tuple[List[tuple], int]:
    """Enforce the SRT invariant that cues do not overlap, by truncating the *earlier* cue.

    Overlaps do reach this stage: 43 of them across the 11-clip test bed survive
    `asr_stabilize` profile 0, the largest at 27 s. Two shapes produce them, both upstream
    content defects rather than arithmetic ones — a zero-width interjection nested inside a
    longer cue (already dropped above by the `end <= start` filter), and a hallucinated run
    (`おぉぉぉぉぉ` spanning 47.5-76.6 s) swallowing the real lines spoken inside it.

    Truncating the earlier cue's end is the direction that keeps every line: pushing the later
    cue's start forward would shove a real line past its own end and delete it, which is exactly
    backwards when the offender is the long hallucination on the left. The one case truncation
    cannot serve is two cues starting at the same instant — the earlier would be squeezed to zero
    width — so there the later cue is shifted instead. Either way text is never changed and no cue
    is dropped unless it has nowhere left to go.
    """
    fixed: List[tuple] = []
    for start_s, end_s, text in rows:
        if fixed and start_s < fixed[-1][1]:
            p_start, p_end, p_text = fixed[-1]
            if start_s > p_start:
                fixed[-1] = (p_start, start_s, p_text)
            else:
                start_s = p_end
                if start_s >= end_s:
                    continue
        fixed.append((start_s, end_s, text))
    return fixed, sum(1 for a, b in zip(rows, rows[1:]) if b[0] < a[1])


def render_segment_srt(segments: List[Dict[str, object]]) -> str:
    rows, overlapping = resolve_overlaps(_timed(segments))
    if overlapping:
        # Surfaced, not swallowed: the invariant is restored here, but the cause is upstream.
        current_reporter().warning(
            "overlapping-cues",
            f"{overlapping} overlapping cue(s) in input; truncated the earlier "
            "cue of each pair to keep the SRT ordered",
        )
    lines: list[str] = []
    for idx, (start_s, end_s, text) in enumerate(rows, start=1):
        lines.append(str(idx))
        lines.append(f"{format_srt_time(start_s)} --> {format_srt_time(end_s)}")
        lines.append(str(text).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_word_srt(segments: List[Dict[str, object]]) -> str:
    # This writer builds every cue from word timestamps and never reads the
    # segment span, so the guard reads words too. `resolve_overlaps` covers the
    # span writer; neither says anything about the other.
    time_order.report_backward(segments, using="words", where="word-level SRT")
    lines: list[str] = []
    idx = 1
    found_words = False
    for seg in segments:
        words = seg.get("words") or []
        if words:
            found_words = True
        for w in words:
            start = w.get("start")
            end = w.get("end")
            text = w.get("word") or w.get("text") or ""
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
        raise ValueError("No segments contain words; cannot write word-level SRT.")
    return "\n".join(lines).strip() + "\n"


def convert_json_to_srt(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    word: bool = False,
    validate: bool = True,
) -> Path:
    """Render `input_path`'s segments to SRT.

    `validate=False` is for a caller that writes into a scratch file it will
    keep working on: reporting there names a `.part` path nobody can open, and
    a caller that runs several passes would report the same finding once per
    pass. Such a caller owns validating the artifact it finally delivers.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else input_path.with_suffix(".srt")
    )
    data = json.loads(input_path.read_text(encoding="utf-8"))
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list):
        raise SystemExit("Input JSON must contain a 'segments' list.")

    if word:
        srt_text = render_word_srt(segments)
    else:
        srt_text = render_segment_srt(segments)

    if validate:
        warn_on_invalid_srt(srt_text, where=str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_text, encoding="utf-8")
    # Intermediate artifacts are not announced; the delivered one is, once,
    # by the pipeline's `completed`.
    current_reporter().debug("wrote subtitle", {"path": str(output_path)})
    return output_path


def main() -> int:
    args = parse_args()
    # Bound, or this command produces no output at all: both the overlapping-cue
    # warning and the written path go through the reporter now.
    with reporting_to(terminal_reporter()):
        output = convert_json_to_srt(
            args.input, output_path=args.output, word=args.word
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
