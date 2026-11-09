"""Give squeezed rows back their time: stretch a row whose span is far too
short for its text into the unclaimed VAD speech after it, then re-split it.

The shape (docs/asr-stabilize.md, "坍缩行" under "补救：坍缩行与空档"): the words are real, but the
segment span was squeezed -- dozens of characters inside a second -- while the
speech they belong to sits, unclaimed, right after the row. About 1.2% of rows
on multi-speaker broadcasts. The start is usually right already (21 of 27
externally placed rows within 0.5 s), so the repair anchors the start and only
grows the end; a row whose true place is inside a neighbour cannot be fixed by
spreading over VAD and is left alone.

The geometry of the stretch, pure: the payload's own VAD timeline, no audio,
no model. Which rows it stretches is not decided here: the recall repair
(`postprocessing/recall_repair`) lets through only the rows the referee hears their speech
after, on stabilize's rows after the drop ladder (a ghost the ladder removes
no longer claims the time after a real row). Stretching every row the plan
names was measured to break about one row an hour and was not accepted.

A stretched row goes through the production splitter alone. Squeezed, it was
too short and gapless for the splitter to cut; stretched, it is often two
speakers' lines folded into one. Alone, not the whole run again: the splitter
is global, and a re-run over everything would also move rows the repair never
touched (12 on 13 windows, with no repair at all); on the repaired rows the
two cut identically (15 of 15).
"""

from __future__ import annotations

from typing import Callable

from ...subtitles.metrics import weighted_char_count
from ...text import coerce_optional_float, normalized_compact
from . import segmentation

#: The criterion: weighted characters per second of span above this, with at
#: least SQUEEZE_MIN_CHARS characters. On 13 windows it hit 46 rows: 36
#: squeezed, 8 hallucinations, 2 wrong words, no good row.
SQUEEZE_MIN_CHARS_PER_SEC = 25.0
SQUEEZE_MIN_CHARS = 5.0
#: How much speech the stretched row asks for: its weighted characters at this
#: rate. The end stops growing once the row's span holds that much VAD speech,
#: or at the next row's start, whichever comes first.
TARGET_CHARS_PER_SEC = 8.0
#: A row that shares a contiguous run of at least ECHO_COVER of its text with
#: another row within ECHO_RADIUS_SEC is left alone: copy fragments and real
#: repeats alike, because stretching a copy makes a visible duplicate.
ECHO_RADIUS_SEC = 10.0
ECHO_COVER = 0.8
#: Words weigh their weighted characters, at least this, when spread.
MIN_WORD_WEIGHT = 0.5
#: The end grows across a silence no longer than this, measured from the last
#: speech the row already holds. Without it a row ran on over 14 s of silence
#: to the next speech (19 rows bridged more than 1 s on 300 artifacts); 0.7
#: and 1.0 gave the same result there, so the value is not sensitive.
MAX_SILENCE_BRIDGE_SEC = 0.7

#: Segment field on every row this pass wrote (a stretched row, or each piece
#: cut out of one): the original row's span, {"start", "end"}. The rows that
#: share it came from one row. It also tells a later stabilize over this output
#: that the row was judged already, as the row it came from: profile 2 skips
#: it (docs/asr-stabilize.md, "坍缩行" under "补救：坍缩行与空档").
SOURCE_FIELD = "squeeze_repair"

_EPS = 1e-6
_SEAM_TAGS = (segmentation.MID_SEGMENT_TAG, segmentation.ANCHOR_UNCERTAIN_TAG)


def row_span(segment: dict[str, object]) -> tuple[float, float] | None:
    """(start, end) of a row, None when either is missing."""
    start = coerce_optional_float(segment.get("start"))
    end = coerce_optional_float(segment.get("end"))
    if start is None or end is None:
        return None
    return start, end


def is_squeezed(segment: dict[str, object]) -> bool:
    span = row_span(segment)
    if span is None:
        return False
    chars = weighted_char_count(str(segment.get("text") or ""))
    duration = max(span[1] - span[0], 1e-3)
    return chars >= SQUEEZE_MIN_CHARS and chars / duration > SQUEEZE_MIN_CHARS_PER_SEC


def _longest_common_run(a: str, b: str) -> int:
    best, prev = 0, [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else 0)
        best = max(best, max(cur))
        prev = cur
    return best


def echo_keys(segments: list[dict[str, object]], spans: list) -> list[tuple[str, float] | None]:
    """(text, start) of each row as the echo check reads it, None without a
    span. A piece this pass cut out of a row reads as that whole row -- its
    text and its original start -- so a later stabilize compares exactly what
    the first one did."""

    texts = [normalized_compact(str(segment.get("text") or "")) for segment in segments]
    whole: dict[tuple, str] = {}
    for segment, text in zip(segments, texts):
        source = segment.get(SOURCE_FIELD)
        if isinstance(source, dict):
            key = (source.get("start"), source.get("end"))
            whole[key] = whole.get(key, "") + text
    keys: list[tuple[str, float] | None] = []
    for segment, text, span in zip(segments, texts, spans):
        source = segment.get(SOURCE_FIELD)
        if isinstance(source, dict) and (start := coerce_optional_float(source.get("start"))) is not None:
            keys.append((whole[(source.get("start"), source.get("end"))], start))
        else:
            keys.append((text, span[0]) if span is not None else None)
    return keys


def echoes(index: int, keys: list[tuple[str, float] | None]) -> bool:
    """Another row within ECHO_RADIUS_SEC holds a contiguous run of at least
    ECHO_COVER of this row's text (longest common substring, so a short line
    is not "covered" by scattered characters of an unrelated long one)."""

    if keys[index] is None or not keys[index][0]:
        return False
    me, start = keys[index]
    for j, key in enumerate(keys):
        if j == index or key is None or abs(key[1] - start) > ECHO_RADIUS_SEC:
            continue
        if len(key[0]) >= 2 and _longest_common_run(me, key[0]) >= ECHO_COVER * len(me):
            return True
    return False


def speech_within(intervals: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    """The parts of `intervals` inside [a, b]."""
    return [(max(a, s), min(b, e)) for s, e in intervals if e > a and s < b]


def _stretched_end(index: int, spans: list, intervals: list[tuple[float, float]], need: float) -> float | None:
    """The new end, or None when the row cannot grow. Grows into VAD speech
    after the end, up to the next row's start, until the row's span holds
    `need` seconds of speech, and stops at a silence longer than
    MAX_SILENCE_BRIDGE_SEC. A row whose own span holds no speech is not
    grown: nothing anchors it to the speech after it. No other row's span
    reaches into the stretch: the guard below refuses a row that some other
    row overlaps."""

    start, end = spans[index]
    others = [s for j, s in enumerate(spans) if s is not None and j != index]
    following = [s[0] for s in others if s[0] >= start]
    limit = min((s for s in following if s >= end - _EPS), default=float("inf"))
    if (
        limit <= end + _EPS
        or any(s < end - _EPS for s in following)
        or any(s[0] < start and s[1] > end + _EPS for s in others)
    ):
        # A row starting at or before this one's end, or an earlier row that
        # already runs past it: stretching would deepen an overlap that is
        # already there (the row's true place is likely inside that one).
        return None
    own = speech_within(intervals, start, end)
    if not own:
        return None
    have = sum(b - a for a, b in own)
    reach = max(b for _a, b in own)
    new_end = end
    for a, b in sorted(speech_within(intervals, end, limit)):
        if have >= need - _EPS or a - reach > MAX_SILENCE_BRIDGE_SEC + _EPS:
            break
        take = min(b - a, need - have)
        have += take
        new_end = a + take
        reach = b
    return new_end if new_end > end + _EPS else None


def spread_words(segment: dict[str, object], start: float, end: float, intervals) -> list[dict[str, object]]:
    """The row's words laid over the VAD speech in [start, end] by character
    weight (over the whole span when it holds none). Adjustment provenance is
    dropped: the splitter's regroup would otherwise restore the squeezed
    coordinates from `raw_start` / `raw_end`."""

    spans = speech_within(intervals, start, end) or [(start, end)]
    total = sum(b - a for a, b in spans)
    words = [
        {k: v for k, v in word.items() if k not in segmentation.ADJUST_PROVENANCE_KEYS}
        for word in segment.get("words") or []
        if isinstance(word, dict)
    ]
    weights = [max(MIN_WORD_WEIGHT, weighted_char_count(str(w.get("word") or ""))) for w in words]
    whole = sum(weights)

    def at(fraction: float, *, opening: bool) -> float:
        # A word that starts exactly where a speech span ends starts where
        # the next one begins; a word that ends there ends there.
        t = fraction * total
        for a, b in spans:
            if t < b - a - _EPS or (not opening and t <= b - a + _EPS):
                return min(b, a + t)
            t -= b - a
        return spans[-1][1]

    acc = 0.0
    for word, weight in zip(words, weights):
        word["start"] = at(acc / whole, opening=True)
        acc += weight
        word["end"] = at(acc / whole, opening=False)
    return words


def resplit(
    row: dict[str, object], original: dict[str, object], intervals_payload, params: segmentation.SplitParams
) -> list[dict[str, object]]:
    """The stretched row through the splitter alone; one row when it is not cut.

    Pieces keep every field of the row the splitter does not write itself
    (energy, evidence, tags), and the first piece keeps the row's own
    seam marker: alone in the splitter, the row's first word always looks like
    the start of an ASR segment, which it need not have been."""

    pieces = segmentation.split_segments([row], intervals_payload, params=params)
    if len(pieces) < 2:
        return [row]
    own = {
        k: v
        for k, v in row.items()
        if k not in ("start", "end", "words", "text", segmentation.ALIGNMENT_EVENTS_FIELD)
    }
    head_tag = bool(((original.get("words") or [{}])[0] or {}).get(segmentation.WHISPER_SEGMENT_WORD_TAG))
    out = []
    for i, piece in enumerate(pieces):
        merged = dict(own)
        merged.update(piece)
        # The row's seam tags describe its start, which only the first piece keeps.
        tags = [str(t) for t in own.get("tags") or [] if i == 0 or str(t) not in _SEAM_TAGS]
        tags += [str(t) for t in piece.get("tags") or [] if str(t) not in tags]
        if tags:
            merged["tags"] = tags
        if i == 0:
            # The start is anchored: the splitter would move the first piece
            # to its first word, which may sit later, where speech begins.
            merged["start"] = min(float(merged["start"]), float(row["start"]))
        if i == 0 and merged.get("words"):
            words = [dict(w) for w in merged["words"]]
            if head_tag:
                words[0][segmentation.WHISPER_SEGMENT_WORD_TAG] = True
            else:
                words[0].pop(segmentation.WHISPER_SEGMENT_WORD_TAG, None)
            merged["words"] = words
        out.append(merged)
    return out


def speech_intervals(intervals_payload: list[dict[str, object]]) -> list[tuple[float, float]]:
    """`vad_timeline.intervals` as sorted (start, end) pairs, malformed ones dropped."""
    return sorted(
        (s, e)
        for item in intervals_payload
        if isinstance(item, dict)
        and (s := coerce_optional_float(item.get("start"))) is not None
        and (e := coerce_optional_float(item.get("end"))) is not None
        and e > s
    )


def stretch_plan(
    segments: list[dict[str, object]],
    intervals_payload: list[dict[str, object]],
    *,
    leave: Callable[[str], bool],
) -> dict[int, float]:
    """The rows `stretch_squeezed` would stretch: index -> new end. Rows
    already carrying SOURCE_FIELD, texts `leave` names, echoes and rows that
    cannot grow are not in it."""

    intervals = speech_intervals(intervals_payload)
    if not intervals:
        return {}
    spans = [row_span(segment) for segment in segments]
    keys = echo_keys(segments, spans)
    plan: dict[int, float] = {}
    for index, segment in enumerate(segments):
        text = str(segment.get("text") or "")
        if (
            spans[index] is None
            or SOURCE_FIELD in segment
            or not is_squeezed(segment)
            or leave(text)
            or echoes(index, keys)
        ):
            continue
        need = weighted_char_count(text) / TARGET_CHARS_PER_SEC
        new_end = _stretched_end(index, spans, intervals, need)
        if new_end is not None:
            plan[index] = new_end
    return plan


def stretch_squeezed(
    segments: list[dict[str, object]],
    intervals_payload: list[dict[str, object]],
    *,
    leave: Callable[[str], bool],
    params: segmentation.SplitParams = segmentation.DEFAULT_SPLIT_PARAMS,
    only: Callable[[int], bool] | None = None,
) -> tuple[list[dict[str, object]], int]:
    """(segments with squeezed rows stretched and re-split, rows stretched).
    `leave(text)` names texts the caller wants untouched (stock-phrase shapes,
    which stabilize removes on their timing); `params` are the split
    parameters the payload was cut with (`segmentation.split_params_of`);
    `only(index)`, when given, picks which of the planned rows are stretched
    (an evaluation arm gates them on a second decoder; production stretches
    them all). Rows already carrying SOURCE_FIELD are left as they are, and
    the echo check reads their pieces as the row they came from, so a second
    pass changes nothing."""

    plan = stretch_plan(segments, intervals_payload, leave=leave)
    intervals = speech_intervals(intervals_payload)
    out: list[dict[str, object]] = []
    stretched = 0
    for index, segment in enumerate(segments):
        if index not in plan or (only is not None and not only(index)):
            out.append(segment)
            continue
        start, end = row_span(segment)
        row = dict(
            segment,
            end=plan[index],
            words=spread_words(segment, start, plan[index], intervals),
            **{SOURCE_FIELD: {"start": start, "end": end}},
        )
        out.extend(resplit(row, segment, intervals_payload, params))
        stretched += 1
    return out, stretched
