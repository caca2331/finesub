"""A correction answer whose rows have slid off their sources.

The shape (3.5 Flash Lite above all): the model skips or joins a source mid-window but keeps numbering on,
so from there on the row for source n carries the words of source n+1 (or n+2, ... or, when it skipped a
long stretch, n+100), the window ends a few sources short, and the repair round used to fill those in and
pass. Structure and coverage cannot see it; the text can: such a row reads like another source, not like
its own.

A row is *off* when its corrected text, compared with every other source of the window, is closest to
another one and that one scores at least ``SHIFT_FLOOR`` and beats the row's own sources by
``SHIFT_MARGIN``; its *offset* is how far that source lies from the row's own (past its last source, or
before its first). In a run of consecutive off rows (rows with no text skipped), ``SHIFT_SAME_OFFSET`` rows
with the same offset make a shift: a slide moves rows by one offset, while a recurring word corrected to
how the ASR wrote it elsewhere makes rows read like one source, each at a different offset. An off row whose
own source and the one it reads like are themselves alike (``SHIFT_ALIKE``, or one inside the other: a line
said twice in the window, one copy misheard and corrected to the other) stays in the run but counts toward no
offset. Each run is its own shift and goes back on its own. Readings behind the numbers (2026-11-10, about 2300 recorded answers:
the answers it flags beyond the earlier five-row rule were all real slides of two or three rows; injected
shifts of 5 rows caught 99.3%, longer ones about always): `docs/llm_harness_behavior.md` 「重试与拼接」.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Mapping, Sequence
import unicodedata

SHIFT_MARGIN = 0.15
SHIFT_FLOOR = 0.5
SHIFT_SAME_OFFSET = 2
SHIFT_ALIKE = 0.6

_DROP = re.compile(r"[\W_ー〜]+")
_KATAKANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


def normalize(text: str) -> str:
    """NFKC, lower case, katakana folded to hiragana, punctuation and spaces dropped: so a kana rewrite or a
    full-width digit does not read as different words."""

    return _DROP.sub("", unicodedata.normalize("NFKC", text or "").lower().translate(_KATAKANA))


@dataclass(frozen=True)
class Shift:
    """One run of rows that slid off, in the ids the rows were written in.

    ``start`` is the first source of the run, ``match`` the source whose words that row carries, and
    ``sample`` those words; ``rows`` the rows from the run's first off row to its last, and ``sources`` every
    source of those rows, the stretch a repair has to redo."""

    start: str
    match: str
    sample: str
    rows: int
    sources: tuple[str, ...]


def find_shifts(rows: Sequence[tuple[Sequence[str], str]], sources: Mapping[str, str]) -> tuple[Shift, ...]:
    """The shifts in ``rows`` (each its source ids and corrected text, in order) against ``sources`` (id to
    text, in window order), in row order. Rows naming a source outside ``sources`` or with empty text are
    skipped without breaking a run."""

    order = list(sources)
    index = {source_id: position for position, source_id in enumerate(order)}
    text = {source_id: normalize(value) for source_id, value in sources.items()}
    # One matcher per source, its side of the comparison prepared once for every row compared with it.
    matchers = {source_id: SequenceMatcher(None, "", value, autojunk=False) for source_id, value in text.items()}
    scored: list[tuple[int, str, int | None]] = []  # each row with text: its position, the source it reads like ("" if
    # its own), and how far that source lies from its own
    for position, (ids, corrected) in enumerate(rows):
        own_ids = tuple(ids)
        written = normalize(corrected)
        if not written or not own_ids or any(source_id not in index for source_id in own_ids):
            continue
        need = max(SHIFT_FLOOR, _ratio(written, "".join(text[source_id] for source_id in own_ids)) + SHIFT_MARGIN)
        match = _closest(written, index[own_ids[0]], index[own_ids[-1]], order, matchers, need) if need <= 1 else ""
        offset: int | None = 0
        if match and _alike("".join(text[source_id] for source_id in own_ids), text[match]):
            offset = None  # off, but its own source already says about the same: counts toward no offset
        elif match:
            at = index[match]
            offset = at - index[own_ids[-1]] if at > index[own_ids[-1]] else at - index[own_ids[0]]
        scored.append((position, match, offset))

    shifts: list[Shift] = []
    start = 0
    for end, (_position, match, _offset) in enumerate([*scored, (-1, "", 0)]):
        if match:
            continue
        run = scored[start:end]
        start = end + 1
        counted = Counter(offset for _p, _m, offset in run if offset is not None)
        if not counted or counted.most_common(1)[0][1] < SHIFT_SAME_OFFSET:
            continue
        first, match, _offset = run[0]
        last = run[-1][0]
        shifts.append(Shift(
            start=rows[first][0][0],
            match=match,
            sample=rows[first][1],
            rows=last - first + 1,
            sources=tuple(source_id for row_ids, _text in rows[first:last + 1] for source_id in row_ids),
        ))
    return tuple(shifts)


def _closest(written: str, first: int, last: int, order: Sequence[str], matchers: Mapping[str, SequenceMatcher],
             need: float) -> str:
    """The source outside ``first..last`` that ``written`` reads most like, nearest first on a tie, if it
    scores at least ``need``; else "". A source whose cheap upper bounds stay below ``need`` or the best so
    far is never compared in full (the full ratio is what makes a long window slow)."""

    best, match = 0.0, ""
    for step in range(1, len(order)):
        for position in (last + step, first - step):
            if not 0 <= position < len(order):
                continue
            matcher = matchers[order[position]]
            matcher.set_seq1(written)
            bar = max(need, best)
            if not matcher.b or matcher.real_quick_ratio() < bar or matcher.quick_ratio() < bar:
                continue
            score = matcher.ratio()
            if score > best and score >= need:
                best, match = score, order[position]
    return match


def _alike(a: str, b: str) -> bool:
    """Two sources too alike for a row reading like the other to say it slid: one inside the other, or
    ``SHIFT_ALIKE`` alike (a line said twice in the window, one of the two misheard)."""

    return bool(a and b) and (a in b or b in a or _ratio(a, b) >= SHIFT_ALIKE)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
