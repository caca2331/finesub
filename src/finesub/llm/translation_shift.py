"""A correction answer whose translations slid off their rows while the corrected text stayed in place.

`serial_shift` reads the corrected text, so it misses a model that writes each row's text right and its
translation one or two rows off (seen only in GPT-6 Luna's grouped pass 2, in experimental agent arms). The
translation is in another language, so it is compared with the sources on what survives translation: the
Han characters both share (Jaccard, weight 2), its length against the window's usual translation-to-source
ratio (``-|log|`` of the misfit), whether both carry ``?`` / ``!`` / an ellipsis (0.5 each, plus when they
agree, minus when not), and digits and Latin words (1, plus when they share one, minus when both carry some
and share none).

A row *gains* at offset ``d`` (one of ``DELTAS``) by how much better its translation fits the sources of the
row ``d`` away than both its own and those of the row ``-d`` away (so a pair of rows whose translations were
swapped for word order gains nothing: each fits its partner no better than the other side). A row whose
sources and the neighbour's are too alike to tell apart (one inside the other, or ``SIMILAR`` alike) gains 0.
The window is flagged when, at some offset, a stretch of rows (rows without translation skipped; the
max-sum segment) sums above ``THRESHOLD`` and holds at least ``MIN_GAINING`` rows that each gain more than
``ROW_GAIN``. Windows with fewer than ``MIN_ROWS`` translated rows are not judged: the ratio and the segment
need rows to stand on.

A flag is a warning, never a rejection (owner 2026-11-11: the cases are few and only in an experimental
arm). Every set with a translation column is checked, basicA included (owner 2026-11-11, after 3.5 Flash
Lite's readout). Readings behind the numbers (2026-11-11, about 1700 clean recorded answers with a
translation column, and 204 of Flash Lite's): `docs/llm_harness_behavior.md` 「重试与拼接」.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import statistics
from typing import Mapping, Sequence
import unicodedata


DELTAS = (1, -1, 2, -2)
ROW_GAIN = 0.25
MIN_GAINING = 4
THRESHOLD = 2.5
MIN_ROWS = 15
SIMILAR = 0.6

_HAN = re.compile(r"[㐀-鿿豈-﫿]")
_DIGITS = re.compile(r"\d+")
_LATIN = re.compile(r"[a-z]{2,}")
_PUNCT = {"question": "?？", "exclaim": "!！", "ellipsis": "…‥〜~"}
_CORE = re.compile(r"[\s\W_]")


@dataclass(frozen=True)
class TranslationShift:
    """The flagged stretch, in the ids the rows were written in: from the first source of its first row
    (``start``) to the last source of its last (``end``); ``rows`` translated rows read like the sources
    ``offset`` rows away, and ``score`` is the stretch's summed gain."""

    start: str
    end: str
    rows: int
    offset: int
    score: float


@dataclass(frozen=True)
class _Features:
    han: frozenset[str]
    length: int
    digits: frozenset[str]
    latin: frozenset[str]
    punct: frozenset[str]


def _features(text: str) -> _Features:
    folded = unicodedata.normalize("NFKC", text or "").lower()
    return _Features(
        han=frozenset(_HAN.findall(folded)),
        length=max(len(_plain(folded)), 1),
        digits=frozenset(_DIGITS.findall(folded)),
        latin=frozenset(_LATIN.findall(folded)),
        punct=frozenset(name for name, marks in _PUNCT.items() if any(mark in folded for mark in marks)),
    )


def _fit(translation: _Features, source: _Features, ratio: float) -> float:
    score = 0.0
    union = translation.han | source.han
    if union:
        score += 2.0 * len(translation.han & source.han) / len(union)
    score -= abs(math.log(translation.length / (ratio * source.length)))
    for name in _PUNCT:
        mine, theirs = name in translation.punct, name in source.punct
        if mine or theirs:
            score += 0.5 if mine == theirs else -0.5
    for mine, theirs in ((translation.digits, source.digits), (translation.latin, source.latin)):
        if mine and theirs:
            score += 1.0 if mine & theirs else -1.0
    return score


def _plain(text: str) -> str:
    """NFKC, lower case, punctuation and spaces dropped. Not `serial_shift.normalize`: folding kana as well
    calls more neighbours alike, and the threshold was calibrated without it (one more clean answer flagged
    with it, of 830)."""

    return _CORE.sub("", unicodedata.normalize("NFKC", text or "").lower())


def _alike(a: str, b: str) -> bool:
    return not a or not b or a in b or b in a or SequenceMatcher(None, a, b, autojunk=False).ratio() >= SIMILAR


def find_translation_shift(
    rows: Sequence[tuple[Sequence[str], str]], sources: Mapping[str, str]
) -> TranslationShift | None:
    """The strongest flagged stretch of ``rows`` (each its source ids and translation, in order) against
    ``sources`` (id to text), or ``None``. Rows with no translation, or naming a source outside ``sources``,
    are skipped without breaking a stretch."""

    count = len(rows)
    joined = ["".join(sources.get(source_id, "") for source_id in ids) for ids, _text in rows]
    usable = [bool(text.strip()) and bool(ids) and all(source_id in sources for source_id in ids)
              for ids, text in rows]
    if sum(usable) < MIN_ROWS:
        return None
    said = [_features(text) for _ids, text in rows]
    heard = [_features(text) for text in joined]
    ratio = statistics.median(said[r].length / heard[r].length for r in range(count) if usable[r])
    plain = [_plain(text) for text in joined]
    fits = [{d: _fit(said[r], heard[r + d], ratio) for d in (0, *DELTAS) if 0 <= r + d < count} if usable[r] else None
            for r in range(count)]

    best: TranslationShift | None = None
    for d in DELTAS:
        total, gaining, first = 0.0, 0, None
        for r in range(count):
            fit = fits[r]
            if fit is None:
                continue
            if d not in fit:
                total, gaining, first = 0.0, 0, None
                continue
            gain = 0.0 if _alike(plain[r], plain[r + d]) else fit[d] - max(fit[0], fit.get(-d, -math.inf))
            total += gain
            if total <= 0:
                total, gaining, first = 0.0, 0, None
                continue
            first = r if first is None else first
            gaining += gain > ROW_GAIN
            if gaining >= MIN_GAINING and total > THRESHOLD and (best is None or total > best.score):
                best = TranslationShift(start=rows[first][0][0], end=rows[r][0][-1],
                                        rows=sum(1 for k in range(first, r + 1) if fits[k] is not None),
                                        offset=d, score=total)
    return best
