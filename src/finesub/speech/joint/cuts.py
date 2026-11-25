"""Where the joint-ASR backends cut a source for a model: at the middle of a Silero pause, never at a fixed time
(owner 2026-11-21: 「以后凡是涉及到切分，都应该切在自然断点上面」). A model handed audio that starts mid-utterance
can settle into a broken output style for the whole call: local MOSS parts cut every 10 minutes at fixed times
lost all punctuation in about a quarter of the later parts, and the same audio decoded from another start was
normal. Why, and which cuts: `docs/joint-asr.md`「设计取舍」.

Which pause (`ranked`, owner 2026-11-22: 「最好的切点自然是同时被 silero 和模型自己的断句同意的地方……找了一段
时间还找不到的话，那就放宽要求。就像生产里面 whisper 切断的时候找断点的逻辑一样」): the further the search goes
from the nominal point, the shorter a pause may be, and a pause the model's own output also breaks at
(`segment_breaks`, `word_breaks`) needs a quarter of the length. The cut is the pause that qualifies at the least
search depth, the nearer one among equals; with none qualifying in the range, the one nearest to it (the longest,
counting a pause over L_TRUST s as LONG_COUNTS s: Silero's usual failure is a long stretch of speech missed); with
no pause in the range, the nominal point itself.

Two kinds of edge:

* a limit -- a window's longest audio: the window ends in the `reach` s before it (`end_before`), and the next
  window starts OVERLAP[0] to OVERLAP[1] s before that end (`start_before`). MOSS Pro's key planning (`fill`) cuts
  by the same rule and ranges, over several candidates;
* a pad -- context around a spot asked again: the edge goes out from the spot by at least the pad and at most by
  STRETCH of it more (MIN_STRETCH s at least), so a tuned pad never gives the model less context (`padded`).

A cached answer whose span falls in the ranges the rule allows is taken instead of the span planned now
(`cached`): Silero on another machine places a pause a little apart, a window cut before 2026-11-22 ends up to
CACHED_REACH s before its limit, and a task cut at fixed times before 2026-11-21 finds its answers again (each
fixed cut is the nominal point, inside every range).
"""

from __future__ import annotations

import bisect
import math
from typing import Callable, Iterable, Mapping, Sequence

#: The next window starts this many seconds before the previous one's end: at least, at most.
OVERLAP = (60.0, 90.0)
#: A window's end looks for a pause this far before its limit (owner 2026-11-22: 「那就按30秒吧」).
REACH = 30.0
#: ... and a cached window's end is taken this far before it: the reach windows were cut by from 2026-11-21 to
#: 2026-11-22 (owner: 「也不需要为了这点差别再重转音频了」).
CACHED_REACH = 120.0
#: A pad's edge may go out by this share of the pad beyond it, and by MIN_STRETCH s at least.
STRETCH = 0.5
MIN_STRETCH = 2.0
#: Call names carry times to 0.1 s.
NAMED = 0.1

#: A pause this long qualifies at the nominal point; at search depth d, one of NEED0 * SCALE / (SCALE + d) s.
#: Set by the offline readout (`explore/joint_cut_agree/readout.py`).
NEED0 = 2.0
SCALE = 20.0
#: A pause the model's output breaks at too needs this share of that length.
AGREE_SHARE = 0.25
#: ... where the pause and a break overlap within this many seconds (the models' timestamps are off by about it).
AGREE_TOL = 0.3
#: A pause longer than L_TRUST s counts as LONG_COUNTS s: a long one is often speech Silero missed.
L_TRUST = 2.0
LONG_COUNTS = 1.0
#: Clause and sentence marks a word can end with, the closers that may follow them, for `word_breaks`.
MARKS = set("。．.！!？?、，,…；;：:")
CLOSERS = set("」』）)\"'”’】〉》")

Pauses = Sequence[tuple[float, float]]  # (middle, length), by middle
Breaks = Sequence[tuple[float, float]]  # (from, to), disjoint, by time: where the model's output breaks


def pauses_of(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """(middle, length) of the silences between Silero's speech spans."""
    out, reach = [], None
    for s, e in sorted(spans):
        if reach is not None and s > reach:
            out.append(((reach + s) / 2, s - reach))
        reach = e if reach is None else max(reach, e)
    return out


def segment_breaks(segments: Iterable[Mapping]) -> list[tuple[float, float]]:
    """Where a segment output breaks: from the latest end so far to the next segment's start (two speakers'
    segments may overlap)."""
    out, reach = [], None
    for s in sorted(segments, key=lambda u: u["start"]):
        if reach is not None and s["start"] >= reach:
            out.append((reach, s["start"]))
        reach = s["end"] if reach is None else max(reach, s["end"])
    return _disjoint(out)


def word_breaks(words: Iterable[Mapping]) -> list[tuple[float, float]]:
    """Where a word output breaks: between two words, in output order, with a clause or sentence mark or a change of
    speaker between them, from the end of the word before to the start of the next. A mark's own token is skipped
    (its time can sit on the far side of a pause); a closer's alone (「はい」って) is no break."""
    out, prev, marked = [], None, False
    for w in words:
        text = w["text"].strip()
        if text and all(ch in MARKS or ch in CLOSERS for ch in text):
            marked = marked or _ends_clause(text)
            continue
        if prev is not None and (marked or _ends_clause(prev["text"]) or w.get("speaker") != prev.get("speaker")):
            if w["start"] >= prev["end"] - 0.05:
                out.append((prev["end"], max(prev["end"], w["start"])))
        prev, marked = w, False
    return _disjoint(out)


def _ends_clause(text: str) -> bool:
    t = text.strip().rstrip("".join(CLOSERS))
    return bool(t) and t[-1] in MARKS


def _disjoint(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def agrees(middle: float, length: float, breaks: Breaks) -> bool:
    """Whether the pause overlaps one of the model's breaks, within AGREE_TOL."""
    a, b = middle - length / 2, middle + length / 2
    i = bisect.bisect_left(breaks, a - AGREE_TOL, key=lambda p: p[1])
    return i < len(breaks) and breaks[i][0] <= b + AGREE_TOL


def depth(middle: float, length: float, nominal: float, breaks: Breaks = ()) -> float:
    """How far from `nominal` the search goes before this pause qualifies: its own distance, or further while the
    threshold is still above its length."""
    need = NEED0 * (AGREE_SHARE if agrees(middle, length, breaks) else 1.0)
    counted = length if length <= L_TRUST else LONG_COUNTS
    later = 0.0 if counted >= need else SCALE * (need / max(counted, 1e-6) - 1)
    return max(abs(middle - nominal), later)


def ranked(pauses: Pauses, lo: float, hi: float, nominal: float, breaks: Breaks = ()) -> list[tuple[float, float]]:
    """(middle, depth) of the pauses with their middle in [lo, hi), best first: the least depth, then the nearer
    `nominal`."""
    inside = [(m, depth(m, length, nominal, breaks)) for m, length in pauses if lo <= m < hi]
    return sorted(inside, key=lambda p: (p[1], abs(p[0] - nominal)))


def pause_in(pauses: Pauses, lo: float, hi: float, nominal: float, breaks: Breaks = ()) -> float:
    """The middle of the best pause in [lo, hi) (`ranked`) to 0.1 s (as the calls are named: the audio sent is the
    audio its cached answer is read back against); else `nominal` itself, untouched -- a cut with no pause near it
    stays where it was (a 0.02 s move of a long gap window's start changed its decode)."""
    best = ranked(pauses, lo, hi, nominal, breaks)
    if not best:
        return nominal
    m = best[0][0]
    named = round(m, 1)
    if lo <= named < hi:
        return named
    # rounding took it out of the range: the 0.1 s step on the inside
    inner = math.ceil(m * 10) / 10 if named < lo else math.floor(m * 10) / 10
    return inner if lo <= inner < hi else m


def end_before(limit: float, pauses: Pauses, reach: float = REACH, breaks: Breaks = ()) -> float:
    return pause_in(pauses, limit - reach, limit, limit, breaks)


def start_before(end: float, pauses: Pauses, overlap: tuple[float, float] = OVERLAP, breaks: Breaks = ()) -> float:
    return pause_in(pauses, end - overlap[1], end - overlap[0], end - overlap[0], breaks)


def outward(edge: float, pauses: Pauses, later: bool) -> float:
    """An edge a model's output put next to its words (a Live wide window's, a MOSS Pro gap call's), out to the
    middle of the Silero pause it lies in, to 0.1 s, where that middle and its 0.1 s are further out (later for an
    end: `later`) and the 0.1 s still in the pause; else `edge` itself. Never inwards: the edge is already where the model heard
    the words stop."""
    for m, length in pauses:  # disjoint: one at most holds it
        if m - length / 2 <= edge <= m + length / 2:
            named = round(m, 1)
            out = min(m, named) > edge if later else max(m, named) < edge
            return named if out and m - length / 2 <= named <= m + length / 2 else edge
    return edge


def widened(a: float, b: float, pauses: Pauses, lo: float, hi: float,
            known: Sequence[tuple[float, float]] = ()) -> tuple[float, float]:
    """[a, b], whose edges a model's output put next to its words, each edge `outward`, inside [lo, hi] -- unmoved
    where that would keep its 0.1 s name; or a `known` span (a cached call's) between those and [a, b] (named at
    [a, b] before 2026-11-22). An edge of it named as `a` or `b` is that edge itself, unrounded: a call so named was
    asked from there (before 2026-11-22 too), and its answer's times count from it."""
    out = (_unless_same_name(max(lo, outward(a, pauses, False)), a),
           _unless_same_name(min(hi, outward(b, pauses, True)), b))
    hit = cached(known, (out[0], a), (b, out[1]))
    if hit is None:
        return out
    return _unless_same_name(hit[0], a), _unless_same_name(hit[1], b)


def _unless_same_name(moved: float, edge: float) -> float:
    return edge if f"{moved:.1f}" == f"{edge:.1f}" else moved


def stretch(pad: float) -> float:
    return max(pad * STRETCH, MIN_STRETCH)


def padded(a: float, b: float, pad: float, pauses: Pauses, total: float,
           known: Sequence[tuple[float, float]] = (), longest: float | None = None,
           breaks: Breaks = ()) -> tuple[float, float]:
    """[a, b] with at least `pad` s of context either side, each edge at a pause up to `stretch(pad)` s further out
    (less where that would take the span past `longest`: the pads themselves are always given), inside [0, total];
    a `known` span (a cached call's) with both edges in those ranges instead."""
    out = stretch(pad)
    if longest is not None:
        out = max(0.0, min(out, (longest - (b - a + 2 * pad)) / 2))
    found = cached(known, (max(0.0, a - pad - out), max(0.0, a - pad)), (min(total, b + pad), min(total, b + pad + out)))
    if found is not None:
        return found
    lo = pause_in(pauses, a - pad - out, a - pad, a - pad, breaks) if a - pad > 0 else 0.0
    hi = pause_in(pauses, b + pad, b + pad + out, b + pad, breaks) if b + pad < total else total
    return max(0.0, lo), min(total, hi)


def window_end(start: float, total: float, size: float, pauses: Pauses, reach: float = REACH) -> float:
    """Where a window from `start` ends: the source's end when it is within `size`, else at a pause before
    start + size."""
    return total if total - start <= size else end_before(start + size, pauses, reach)


def window(lo: float, hi: float, size: float, pauses: Pauses, after: float | None = None, *, breaks: Breaks = (),
           reach: float = REACH, cached_reach: float | None = None, overlap: tuple[float, float] = OVERLAP,
           known: Sequence[tuple[float, float]] = ()) -> tuple[tuple[float, float], bool]:
    """The window of [lo, hi] after the one that ended at `after` (None: the first), and whether it is the last: at
    most `size` s, cut by the limit rule (module docstring); its start at a pause the previous window's output
    (`breaks`) agrees with, where one does. A `known` span (a cached call's) whose start and end fall in the ranges
    the rule allows is taken instead, its end looked for as far back as the longer of `reach` (where this rule
    ends a window, so a window it planned is found again on resume) and `cached_reach` (where an older rule did).
    That reach + overlap[1] must be under `size`, or a window could make no headway."""
    if after is None:
        start, starts = lo, (lo, lo)
    else:
        start, starts = start_before(after, pauses, overlap, breaks), (after - overlap[1], after - overlap[0])
    look = reach if cached_reach is None else max(reach, cached_reach)
    if look + overlap[1] >= size and hi - start > size:
        raise ValueError(f"windows of {size:.0f} s cannot look {look:.0f} s back for an end and start up to "
                         f"{overlap[1]:.0f} s before it")
    span = cached(known, starts, lambda s: _end_range(s, hi, size, look))
    if span is not None:  # named to 0.1 s: within that of `hi` it is the last
        return span, span[1] >= hi - NAMED
    span = (start, window_end(start, hi, size, pauses, reach))
    return span, span[1] >= hi  # a planned one ends at `hi` itself


def windows(lo: float, hi: float, size: float, pauses: Pauses, **rule) -> list[tuple[float, float]]:
    """[lo, hi] as windows (`window`), each start planned without the previous window's output."""
    out, after = [], None
    while True:
        span, last = window(lo, hi, size, pauses, after, **rule)
        out.append(span)
        if last:
            return out
        after = span[1]


def _end_range(start: float, total: float, size: float, reach: float) -> tuple[float, float]:
    return (total, total) if total - start <= size else (start + size - reach, start + size)


def spans_named(names: Iterable[str], prefix: str) -> list[tuple[float, float]]:
    """(start, end) of the calls named `{prefix}-{start}-{end}`."""
    out = []
    for name in names:
        head, _, times = name.partition("-")
        a, _, b = times.partition("-")
        if head != prefix:
            continue
        try:
            out.append((float(a), float(b)))
        except ValueError:
            continue
    return out


def cached(known: Sequence[tuple[float, float]], starts: tuple[float, float],
           ends: Callable[[float], tuple[float, float]] | tuple[float, float]) -> tuple[float, float] | None:
    """The known span starting in `starts` and ending in `ends` (a range, or one from the start), each widened by
    NAMED for the names' rounding: the latest end, then the earliest start; None without one."""
    found = []
    for s, e in known:
        if not starts[0] - NAMED <= s <= starts[1] + NAMED:
            continue
        lo, hi = ends(s) if callable(ends) else ends
        if lo - NAMED <= e <= hi + NAMED:
            found.append((s, e))
    return max(found, key=lambda p: (p[1], -p[0])) if found else None
