"""Where the speaker judge cuts: one judge window per correction window (owner 2026-10-09).

A correction window's core rows are one judge window, its id the correction window's chunk id: the
correction's planning already sized it. Only as a fallback is a window that does not fit halved, by the
correction's rule (`run.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ....subtitles import line_groups
from ...chunking import SubtitleWindow, is_reasonable_boundary, overlap_count_for_boundary
from ...routing.config import OVERLAP_WINDOW_SECONDS


@dataclass(frozen=True)
class JudgeWindow:
    """Rows `[low, high)` of the stable, judged together. `id` is the correction window's chunk id; a
    half of a window that did not fit is `<window id>-a` / `-b`, and `chunk` stays the window's."""

    id: str
    chunk: str
    low: int
    high: int

    def shown(self, rows: Sequence[Any]) -> range:
        """The rows the prompt shows: the judged ones, and as unjudged context across each edge the rows the
        correction carries over a cut (owner 2026-10-09) -- those starting within `OVERLAP_WINDOW_SECONDS`
        before it (`chunking.overlap_count_for_boundary`, so none past a long pause), and mirrored after the
        window: those starting within it after the window's last row (none before a long pause).

        The correction's prompt has no context after a window: the next window re-does its tail and the
        stitch keeps the newer version. The judge gives each row one verdict and never re-does it, so it
        shows that tail context instead -- either way a row near a cut is decided seeing what the correction's
        final version saw after it. Before a window the judge takes the overlap only, not the correction's
        read-only preceding rows (owner 2026-10-09: as long as the overlap)."""
        before = overlap_count_for_boundary(rows, self.low)
        after = 0
        while self.high + after < len(rows) and \
                rows[self.high + after].start - rows[self.high - 1].start <= OVERLAP_WINDOW_SECONDS:
            after += 1
        return range(self.low - before, self.high + after)

    @property
    def size(self) -> int:
        return self.high - self.low


def split_windows(ids: Sequence[str], correction: Sequence[SubtitleWindow]) -> list[JudgeWindow]:
    """Judge windows tiling every row, cut at each correction window's first core row (its rows after the
    overlap it re-reads from the window before).

    Cutting at the starts (rather than taking each window's own span) tiles the file even where the
    correction windows' cores would leave a gap or overlap."""
    if not ids:
        return []
    position = {key: index for index, key in enumerate(ids)}
    starts: dict[int, str] = {}
    for window in correction:
        core = window.segments[len(window.overlap_segments):]
        if core and core[0].id in position:
            starts.setdefault(position[core[0].id], window.chunk_id)
    cuts = sorted(starts)
    if not cuts or cuts[0] != 0:
        first = correction[0].chunk_id if correction else "0000"
        cuts, starts = [0, *cuts], {0: first, **starts}
    return [JudgeWindow(starts[low], starts[low], low, high) for low, high in zip(cuts, [*cuts[1:], len(ids)])]


def halves(rows: Sequence[Any], low: int, high: int) -> list[tuple[int, int]]:
    """`[low, high)` in two at the reasonable boundary (a sentence end, or a pause) nearest the middle: the
    correction's rule for a window that does not fit (`chunking.split_window_in_half`), without its
    overlap -- the judge's halves tile, and each is shown its neighbours' rows as context anyway."""
    size = high - low
    if size < 2:
        return []
    window = rows[low:high]
    parted = line_groups.parted(rows)[low:high]
    middle = (size - 1) / 2
    candidates = sorted(range(size - 1), key=lambda index: abs(index - middle))
    end = next((index for index in candidates if is_reasonable_boundary(window, index, parted=parted)), int(middle))
    return [(low, low + end + 1), (low + end + 1, high)]
