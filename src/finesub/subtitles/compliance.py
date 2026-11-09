"""A whole-file compliance table for subtitles: reading speed, line length, duration, gaps and overlaps.

The rows are the mechanical checks of the Netflix timed-text guides and the IWSLT subtitling track's
compliance script, counted over every cue of a file (or of every file under a directory, pooled). Lengths
are `metrics.weighted_char_count`, which counts half-width characters as half, as Netflix does. The
reading-speed row is pass 3's own rule (``字/秒`` over 9 on a line of 10 characters or more); the other
limits are reference lines, not targets: a human-refined file breaks some of them too, which is why
`docs/segmentation-split.md`「全文件的合规表」 carries the refined sets' own readings to compare against.

Two inputs read as a pair: the table gets a third column, the second minus the first, which is the
reading for an A/B on the same source. Nothing here judges translation quality.

    python -m finesub.subtitles.compliance out/x/x.srt
    python -m finesub.subtitles.compliance arm-a/x.srt arm-b/x.srt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
import sys
from typing import Callable, Sequence

from .metrics import weighted_char_count
from .model import SrtSegment, has_text, parse_srt

#: Pass 3's reading-speed rule (`proofread_system_v1.md`): over 9 characters a second, on a line of at least
#: 10. Netflix's limit for adult Simplified Chinese is the same 9; its Japanese one is 4.
CPS_LIMIT = 9.0
CPS_MIN_CHARS = 10.0
#: Netflix: 16 per line for Simplified Chinese (two lines at most); 36 is the LLM merge's absolute cap at
#: length scale 1.0 (it scales with the task's ratio).
LINE_LENGTHS = (16.0, 20.0, 36.0)
#: Netflix: at least 5/6 s (0.5 s for Japanese), at most 7 s.
SHORT_DURATIONS = (0.5, 5 / 6)
LONG_DURATION = 7.0
#: Netflix: a gap is either 2 frames (at 24 fps) or at least 0.5 s; one in between reads as a flicker.
FLICKER_GAP = (2 / 24, 0.5)


@dataclass(frozen=True)
class Row:
    label: str
    #: The row's value for a set of cues; None when the set has nothing to count it over.
    measure: Callable[[Sequence[SrtSegment]], float | None]
    #: How to print it: a share of cues or gaps, or a plain number.
    share: bool


def _cps(cue: SrtSegment) -> float:
    return weighted_char_count(_text(cue)) / max(cue.end - cue.start, 1e-3)


#: SRT styling a player draws but does not show as text: `<i>`, `<b>`, `<u>`, `<font ...>` and their closing
#: tags, and ASS override blocks such as `{\an8}`.
_STYLING = re.compile(r"</?(?:[ibu]|font)(?:\s[^>]*)?>|\{\\[^}]*\}", re.I)


def _lines(cue: SrtSegment) -> list[str]:
    """The cue's lines as shown on screen: styling markup does not count toward any length."""

    return [_STYLING.sub("", line) for line in cue.text.splitlines()]


def _text(cue: SrtSegment) -> str:
    return "".join(_lines(cue))


def _longest_line(cue: SrtSegment) -> float:
    return max((weighted_char_count(line) for line in _lines(cue)), default=0.0)


def _gaps(cues: Sequence[SrtSegment]) -> list[float]:
    """Start of each cue minus the end of the one before it, by start; negative is an overlap. Pooled files
    contribute their own pairs only."""

    out: list[float] = []
    for group in _by_file(cues):
        ordered = sorted(group, key=lambda cue: (cue.start, cue.end))
        out += [after.start - before.end for before, after in zip(ordered, ordered[1:])]
    return out


def _by_file(cues: Sequence[SrtSegment]) -> list[list[SrtSegment]]:
    groups: dict[str | None, list[SrtSegment]] = {}
    for cue in cues:
        groups.setdefault(cue.key, []).append(cue)
    return list(groups.values())


def _share(predicate: Callable[[SrtSegment], bool]) -> Callable[[Sequence[SrtSegment]], float | None]:
    return lambda cues: sum(map(predicate, cues)) / len(cues) if cues else None


def _gap_share(predicate: Callable[[float], bool]) -> Callable[[Sequence[SrtSegment]], float | None]:
    def measure(cues: Sequence[SrtSegment]) -> float | None:
        gaps = _gaps(cues)
        return sum(map(predicate, gaps)) / len(gaps) if gaps else None

    return measure


def _quantile(value: Callable[[SrtSegment], float], q: float) -> Callable[[Sequence[SrtSegment]], float | None]:
    def measure(cues: Sequence[SrtSegment]) -> float | None:
        if len(cues) < 2:
            return value(cues[0]) if cues else None
        return statistics.quantiles([value(cue) for cue in cues], n=100, method="inclusive")[round(q * 100) - 1]

    return measure


def rows(cps_limit: float = CPS_LIMIT, cps_min_chars: float = CPS_MIN_CHARS) -> list[Row]:
    low, high = FLICKER_GAP
    return [
        Row("cues", len, share=False),
        Row(f"over {cps_limit:g} chars/s (lines of {cps_min_chars:g}+ chars)",
            _share(lambda c: weighted_char_count(_text(c)) >= cps_min_chars and _cps(c) > cps_limit), share=True),
        Row("chars/s p50", _quantile(_cps, 0.5), share=False),
        Row("chars/s p90", _quantile(_cps, 0.9), share=False),
        *[Row(f"longest line over {n:g} chars", _share(lambda c, n=n: _longest_line(c) > n), share=True)
          for n in LINE_LENGTHS],
        Row("longest line p90", _quantile(_longest_line, 0.9), share=False),
        Row("more than 2 lines", _share(lambda c: len(c.text.splitlines()) > 2), share=True),
        *[Row(f"shorter than {d:.3g} s", _share(lambda c, d=d: c.end - c.start < d), share=True)
          for d in SHORT_DURATIONS],
        Row(f"longer than {LONG_DURATION:g} s", _share(lambda c: c.end - c.start > LONG_DURATION), share=True),
        Row("duration p50 (s)", _quantile(lambda c: c.end - c.start, 0.5), share=False),
        Row(f"gaps in ({low:.3g}, {high:g}) s", _gap_share(lambda g: low < g < high), share=True),
        Row("overlapping the cue before", _gap_share(lambda g: g < 0), share=True),
    ]


def read_cues(path: Path) -> list[SrtSegment]:
    """The cues with text in an SRT file, or in every `*.srt` under a directory (each keyed by its file, so
    gaps are taken within a file)."""

    files = sorted(path.rglob("*.srt")) if path.is_dir() else [path]
    cues: list[SrtSegment] = []
    for file in files:
        text = file.read_bytes().decode("utf-8-sig")
        # `""` is how a line with no text is written when it is kept (`model.srt_body_lines`, `--empty-cues`).
        cues += [SrtSegment(c.index, c.start, c.end, c.text, key=str(file)) for c in parse_srt(text)
                 if has_text(c.text) and c.text.strip() != '""']
    return cues


def table(columns: Sequence[tuple[str, Sequence[SrtSegment]]], table_rows: Sequence[Row]) -> str:
    """A Markdown table, one column per input; a pair gets a third, the second minus the first (for a share,
    in percentage points)."""

    def cell(value: float | None, row: Row, *, signed: bool = False) -> str:
        if value is None:
            return "—"
        if row.share:
            return f"{value * 100:+.1f}" if signed else f"{value:.1%}"
        if row.measure is len:
            return f"{value:+.0f}" if signed else f"{value:.0f}"
        return f"{value:+.2f}" if signed else f"{value:.2f}"

    header = ["", *(name for name, _ in columns)] + (["Δ (2nd − 1st; shares in points)"] if len(columns) == 2 else [])
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table_rows:
        values = [row.measure(cues) for _, cues in columns]
        cells = [cell(value, row) for value in values]
        if len(values) == 2:
            first, second = values
            cells.append(cell(None if first is None or second is None else second - first, row, signed=True))
        lines.append("| " + " | ".join([row.label, *cells]) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="SRT files, or directories read as one pool of cues")
    parser.add_argument("--cps-limit", type=float,
                        help=f"reading-speed limit in weighted chars per second (default {CPS_LIMIT:g}; Japanese: 4)")
    parser.add_argument("--cps-min-chars", type=float,
                        help=f"lines shorter than this are not held to the limit (default {CPS_MIN_CHARS:g})")
    args = parser.parse_args(argv)
    columns = [(str(path), read_cues(path)) for path in args.inputs]  # as given: two arms' files share a name
    limits = {"cps_limit": args.cps_limit, "cps_min_chars": args.cps_min_chars}
    print(table(columns, rows(**{name: value for name, value in limits.items() if value is not None})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
