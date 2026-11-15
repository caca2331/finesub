"""A correction answer whose rows have slid off their sources.

The shape (3.5 Flash Lite above all): the model skips or joins a source mid-window but keeps numbering on,
so from there on the row for source n carries the words of source n+1 (or n+2, ... or, when it skipped a
long stretch, n+100), the window ends a few sources short, and the repair round used to fill those in and
pass. Structure and coverage cannot see it; the text can: such a row reads like another source, not like
its own.

A row is *off* when its corrected text, compared with every other source of the window, is closest to
another one and that one scores at least ``SHIFT_FLOOR`` and beats the row's own sources by
``SHIFT_MARGIN``. ``SHIFT_RUN`` consecutive off rows that read like ``SHIFT_RUN`` different sources make a
shift: a recurring word corrected to how the ASR wrote it elsewhere makes rows read like one source, not a
shift. Readings behind the numbers (1939 unshifted answers in Japanese, English and Chinese with none
flagged; injected shifts of 5-10 rows caught 92-99%, longer ones always): `docs/llm_harness_behavior.md`
「重试与拼接」.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Mapping, Sequence
import unicodedata

SHIFT_MARGIN = 0.15
SHIFT_FLOOR = 0.5
SHIFT_RUN = 5

_DROP = re.compile(r"[\W_ー〜]+")
_KATAKANA = {code: code - 0x60 for code in range(0x30A1, 0x30F7)}


def normalize(text: str) -> str:
    """NFKC, lower case, katakana folded to hiragana, punctuation and spaces dropped: so a kana rewrite or a
    full-width digit does not read as different words."""

    return _DROP.sub("", unicodedata.normalize("NFKC", text or "").lower().translate(_KATAKANA))


@dataclass(frozen=True)
class Shift:
    """Where the rows slid off, in the ids the rows were written in.

    ``start`` is the first source of the first run, ``match`` the source whose words that row carries, and
    ``sample`` those words; ``sources`` every source of the rows from the first run's start to the last run's
    end, the stretch a repair has to redo."""

    start: str
    match: str
    sample: str
    rows: int
    sources: tuple[str, ...]


def find_shift(rows: Sequence[tuple[Sequence[str], str]], sources: Mapping[str, str]) -> Shift | None:
    """The shift in ``rows`` (each its source ids and corrected text, in order) against ``sources`` (id to
    text, in window order), or None. Rows naming a source outside ``sources`` or with empty text are skipped
    without breaking a run."""

    order = list(sources)
    index = {source_id: position for position, source_id in enumerate(order)}
    text = {source_id: normalize(value) for source_id, value in sources.items()}
    # One matcher per source, its side of the comparison prepared once for every row compared with it.
    matchers = {source_id: SequenceMatcher(None, "", value, autojunk=False) for source_id, value in text.items()}
    scored: list[tuple[tuple[str, ...], str]] = []  # each row's ids and the source it reads like ("" if none)
    for ids, corrected in rows:
        own_ids = tuple(ids)
        written = normalize(corrected)
        if not written or not own_ids or any(source_id not in index for source_id in own_ids):
            continue
        need = max(SHIFT_FLOOR, _ratio(written, "".join(text[source_id] for source_id in own_ids)) + SHIFT_MARGIN)
        match = _closest(written, index[own_ids[0]], index[own_ids[-1]], order, matchers, need) if need <= 1 else ""
        scored.append((own_ids, match))

    runs: list[tuple[int, int]] = []
    start = 0
    for position, (_ids, match) in enumerate([*scored, ((), "")]):
        if match:
            continue
        if len({read for _ids, read in scored[start:position]}) >= SHIFT_RUN:
            runs.append((start, position - 1))
        start = position + 1
    if not runs:
        return None
    first, last = runs[0][0], runs[-1][1]
    ids, match = scored[first]
    return Shift(
        start=ids[0],
        match=match,
        sample=next(corrected for row_ids, corrected in rows if tuple(row_ids) == ids),
        rows=last - first + 1,
        sources=tuple(source_id for row_ids, _match in scored[first:last + 1] for source_id in row_ids),
    )


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


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio() if a and b else 0.0
