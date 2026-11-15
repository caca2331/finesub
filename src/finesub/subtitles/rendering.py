"""Convert pipeline JSON segments to SRT subtitle files."""

from __future__ import annotations

import argparse
import bisect
import json
import math
from pathlib import Path
from typing import Dict, List, NamedTuple

from .. import settings
from ..reporting import current_reporter, reporting_to, terminal_reporter
from . import line_groups, stable_rows, time_order
from .model import SrtSegment, has_text, render_srt, warn_on_invalid_srt


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
    parser.add_argument(
        "--empty-cues",
        action=argparse.BooleanOptionalAction,
        default=None,
        help='Write a line with no text as a "" cue rather than leave it out (default: left out).',
    )
    return parser.parse_args()


class WrittenSrt(NamedTuple):
    path: Path
    #: Each cue's group key (`line_groups`), in cue order: what the passes after the writer group by. None
    #: when no line has one (a local task without the overlap lane).
    keys: List[str | None] | None


def _cue_key(row: tuple) -> str | None:
    return row[3]


def _cue_start(row: tuple) -> float:
    return row[0]


def _span(start: object, end: object) -> tuple[float, float] | None:
    """`(start, end)` in seconds, or None for a span no cue is written from: missing, not a finite number, or
    not positive."""
    try:
        start_s, end_s = float(start), float(end)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start_s) and math.isfinite(end_s)) or end_s <= start_s:
        return None
    return start_s, end_s


def _timed(segments: List[Dict[str, object]]) -> List[tuple]:
    """(start, end, text, key) for renderable cues, ordered by start. Non-positive spans dropped."""
    out = []
    for seg in segments:
        span = _span(seg.get("start"), seg.get("end"))
        if span is not None:
            out.append((*span, seg.get("text") or "", line_groups.key_of(seg)))
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
    width — so there the later cue is shifted instead, and goes back among the cues still to come at
    its new start: a cue starting inside the gap it jumped is then resolved against the cue before
    it, not against the shifted one. Either way text is never changed and no cue is dropped unless it
    has nowhere left to go.

    Per group key (`line_groups`): two speakers of a cloud task talking at once are two cues on
    screen at once, never truncated against each other.
    """
    found = 0

    def resolve(group: List[tuple]) -> List[tuple]:
        nonlocal found
        fixed: List[tuple] = []
        pending = list(group)
        at = 0
        while at < len(pending):
            start_s, end_s, text, key = pending[at]
            at += 1
            if fixed and start_s < fixed[-1][1]:
                p_start, p_end, p_text, p_key = fixed[-1]
                if start_s > p_start:
                    fixed[-1] = (p_start, start_s, p_text, p_key)
                else:
                    if p_end < end_s:
                        bisect.insort(pending, (p_end, end_s, text, key), lo=at, key=lambda r: (r[0], r[1]))
                    continue
            fixed.append((start_s, end_s, text, key))
        found += sum(1 for a, b in zip(group, group[1:]) if b[0] < a[1])
        return fixed

    fixed = line_groups.grouped(rows, resolve, key=_cue_key, start=_cue_start)
    return fixed, found


def _srt_text(cues: List[tuple]) -> str:
    return render_srt([SrtSegment(idx, start_s, end_s, text) for idx, (start_s, end_s, text, _key) in
                       enumerate(cues, start=1)])


def segment_cues(segments: List[Dict[str, object]]) -> List[tuple]:
    """The span writer's cues, (start, end, text, key)."""
    rows, overlapping = resolve_overlaps(_timed(segments))
    if overlapping:
        # Surfaced, not swallowed: the invariant is restored here, but the cause is upstream.
        current_reporter().warning(
            "overlapping-cues",
            f"{overlapping} overlapping cue(s) in input; truncated the earlier "
            "cue of each pair to keep the SRT ordered",
        )
    return rows


def render_segment_srt(segments: List[Dict[str, object]]) -> str:
    return _srt_text(segment_cues(segments))


def word_cues(segments: List[Dict[str, object]]) -> List[tuple]:
    """The word writer's cues, (start, end, text, key): a word's key is its segment's."""
    # This writer builds every cue from word timestamps and never reads the
    # segment span, so the guard reads words too. `resolve_overlaps` covers the
    # span writer; neither says anything about the other.
    time_order.report_backward(segments, using="words", where="word-level SRT")
    cues: List[tuple] = []
    found_words = False
    for seg in segments:
        words = seg.get("words") or []
        if words:
            found_words = True
        key = line_groups.key_of(seg)
        for w in words:
            span = _span(w.get("start"), w.get("end"))
            if span is not None:
                cues.append((*span, w.get("word") or w.get("text") or "", key))

    if not found_words:
        raise ValueError("No segments contain words; cannot write word-level SRT.")
    # Two speakers' words interleave in time: in order by start (one group: as written).
    return line_groups.grouped(cues, lambda group: group, key=_cue_key, start=_cue_start)


def render_word_srt(segments: List[Dict[str, object]]) -> str:
    return _srt_text(word_cues(segments))


def _with_text(segments: List[Dict[str, object]], *, word: bool) -> List[Dict[str, object]]:
    """`segments` without the lines that have no text, or with `word` without such words."""
    if not word:
        return [seg for seg in segments if has_text(seg.get("text"))]
    return [{**seg, "words": [w for w in seg.get("words") or [] if has_text(w.get("word") or w.get("text"))]}
            for seg in segments]


def convert_json_to_srt(
    input_path: str | Path,
    *,
    output_path: str | Path | None = None,
    word: bool = False,
    validate: bool = True,
    empty_cues: bool | None = None,
) -> WrittenSrt:
    """Render `input_path`'s segments to SRT; the path and each cue's group key.

    `validate=False` is for a caller that writes into a scratch file it will
    keep working on: reporting there names a `.part` path nobody can open, and
    a caller that runs several passes would report the same finding once per
    pass. Such a caller owns validating the artifact it finally delivers.

    `empty_cues` (`settings.EMPTY_CUES`): a line with no text is written as a
    ``""`` cue, so that its span shows and a subtitle editor keeps it; left out
    otherwise. Here and not earlier: the stable JSON keeps every line, and so
    does what the LLM layer reads from it. Left out before the overlaps are
    resolved, so a line nobody sees never cuts short the one before it.
    """

    input_path = Path(input_path).expanduser().resolve()
    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else input_path.with_suffix(".srt")
    )
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # The lines with their speakers, as the task chose them (acoustic row splitting's halves when it did).
    segments = stable_rows.rows(data, stable_rows.read_overlay(input_path, data))
    if not isinstance(segments, list):
        raise SystemExit("Input JSON must contain a 'segments' list.")

    if not settings.EMPTY_CUES.resolve(empty_cues):
        segments = _with_text(segments, word=word)
    cues = word_cues(segments) if word else segment_cues(segments)
    srt_text = _srt_text(cues)
    keys = [_cue_key(cue) for cue in cues]
    keys = keys if any(key is not None for key in keys) else None

    if validate:
        # Not a pipeline stage: this is the standalone json -> srt converter.
        warn_on_invalid_srt(srt_text, where=str(output_path), budget_stage="SRT", keys=keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt_text, encoding="utf-8")
    # Intermediate artifacts are not announced; the delivered one is, once,
    # by the pipeline's `completed`.
    current_reporter().debug("wrote subtitle", {"path": str(output_path)})
    return WrittenSrt(output_path, keys)


def main() -> int:
    args = parse_args()
    # Bound, or this command produces no output at all: both the overlapping-cue
    # warning and the written path go through the reporter now.
    with reporting_to(terminal_reporter()):
        output = convert_json_to_srt(
            args.input, output_path=args.output, word=args.word, empty_cues=args.empty_cues
        )
    print(output.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
