"""Which lines may overlap in time: the group key.

A line's key is its `line_speaker`, the person a cloud transcription says speaks it, or an overlap lane (`LANE`
and the moved row's id) for a local line the collapse repair stretched over the next one (`LANE_FIELD`: shown over
it rather than cut short, docs/asr-stabilize.md "坍缩行"; one lane per moved row, so two such rows over each other
are both kept). Lines of one key never overlap; lines of different keys may (two people
talking at once). Every pass that keeps lines apart in time does so within a key only, through `groups`:

* no line has a key (every local task without such a stretch): one group, and the pass sees the list exactly as
  before;
* only lanes (a local task with such a stretch): each lane is a group and every other line one more;
* some lines carry a speaker (a cloud task): a line without a key (written in by a model or the speaker judge) is
  a group of its own -- nobody said who speaks it, so it is not taken for anyone.

A grouped result goes back together by start, same start by key (`in_order`), or, where the caller holds a
list of keys for the items, at the items' own positions (`in_place`). Lines are dicts here by default;
SRT cues (`SrtSegment`) and rendered rows pass their own `key` and `start` readers.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence, TypeVar

#: The field a cloud line carries its speaker in.
LINE_SPEAKER = "line_speaker"
#: The field marking a local line stretched over the next one, its value the moved row's id (`lane_id`), and the
#: prefix of the key such a line has: no speaker a cloud transcription names starts with "~".
LANE_FIELD = "overlap_lane"
LANE = "~overlap"

T = TypeVar("T")


def key_of(item: Mapping[str, Any]) -> str | None:
    value = item.get(LINE_SPEAKER)
    if value is not None:
        return str(value)
    lane = item.get(LANE_FIELD)
    return f"{LANE}:{lane}" if lane else None


def lane_id(source: tuple[float, float]) -> str:
    """`LANE_FIELD`'s value for the pieces of the row moved from `source` (its original (start, end), exact: two
    rows Whisper overlapped may share a start)."""
    return f"{source[0]!r}-{source[1]!r}"


def speakers(keys: Sequence[str | None]) -> bool:
    """Whether some key names a speaker (a cloud task's lines), not only lanes."""
    return any(k is not None and not k.startswith(LANE) for k in keys)


def _start(item: Mapping[str, Any]) -> float:
    return float(item["start"])


def groups(keys: Sequence[str | None]) -> list[list[int]]:
    """Positions per group, each in the order given; one group when no key is set. Lines without a key are one
    group unless some key names a speaker (`speakers`), then each its own."""
    if all(k is None for k in keys):
        return [list(range(len(keys)))]
    alone = speakers(keys)
    out: dict[tuple[bool, object], list[int]] = {}
    for i, k in enumerate(keys):
        out.setdefault((True, k) if k is not None or not alone else (False, i), []).append(i)
    return list(out.values())


def in_order(items: Sequence[T], key: Callable[[T], str | None] = key_of,
             start: Callable[[T], float] = _start) -> list[T]:
    """By start, same start by key (a keyless item first)."""
    return sorted(items, key=lambda s: (start(s), key(s) or ""))


def grouped(items: Sequence[T], run: Callable[[list[T]], list[T]], *, key: Callable[[T], str | None] = key_of,
            start: Callable[[T], float] = _start) -> list[T]:
    """`run` on each group; with one group its result as it is, else the groups' results `in_order`."""
    parts = groups([key(s) for s in items])
    if len(parts) == 1:
        return run(list(items))
    return in_order([s for idx in parts for s in run([items[i] for i in idx])], key, start)


def in_place(items: Sequence[T], run: Callable[[list[T]], list[T]], *,
             key: Callable[[T], str | None] = key_of) -> list[T]:
    """`run` on each group, its results back at the group's positions: nothing moves, so a list of keys that
    matched `items` matches the result (`run` keeps its group's length)."""
    parts = groups([key(s) for s in items])
    if len(parts) == 1:
        return run(list(items))
    out = list(items)
    for idx in parts:
        for i, item in zip(idx, run([items[i] for i in idx]), strict=True):
            out[i] = item
    return out


# --- timed lines: objects with `start`, `end` and `group_key` (the LLM layer's lines, the judge's rows) --------


def _attr_key(item: Any) -> str | None:
    return getattr(item, "group_key", None)


def group_numbers(items: Sequence[Any]) -> list[int]:
    """Each line's group number, in the order given; all 0 when no line has a key."""
    out = [0] * len(items)
    for g, idx in enumerate(groups([_attr_key(item) for item in items])):
        for i in idx:
            out[i] = g
    return out


def covered(items: Sequence[Any]) -> list[bool]:
    """Per line, at its position: whether it starts while a line of another group that started earlier (the
    same start: earlier in the list) has not ended. Swept by start, so a list out of time order (the judge's
    halves follow their line) is read by the lines' times."""
    group = group_numbers(items)
    reach: dict[int, float] = {}  # latest end so far, per group
    out = [False] * len(items)
    for i in sorted(range(len(items)), key=lambda i: (items[i].start, i)):
        out[i] = any(end > items[i].start for other, end in reach.items() if other != group[i])
        reach[group[i]] = max(reach.get(group[i], items[i].end), items[i].end)
    return out


def parted(items: Sequence[Any]) -> list[bool]:
    """Per position i: whether ending a window after line i parts two lines shown at once (`at_once`: of
    different groups, overlapping in time), one up to i and one after it -- read by times, so a list out of time
    order (the judge's halves follow their line) is judged by when its lines are said. Never true without keys."""
    n = len(items)
    group = group_numbers(items)
    out = [False] * n
    if len(set(group)) < 2:
        return out
    marks = [0] * (n + 1)
    active: list[int] = []
    for j in sorted(range(n), key=lambda k: (items[k].start, k)):
        active = [i for i in active if items[i].end > items[j].start]
        for i in active:
            if group[i] != group[j] and items[i].start < items[j].end:
                lo, hi = min(i, j), max(i, j)
                marks[lo] += 1
                marks[hi] -= 1
        active.append(j)
    depth = 0
    for i in range(n):
        depth += marks[i]
        out[i] = depth > 0
    return out


def overlapping(items: Sequence[Any]) -> list[bool]:
    """Per line (in order by start): whether a line of another group overlaps it in time -- it is `covered`, or
    a later line of another group starts before it ends."""
    group = group_numbers(items)
    out = covered(items)
    for i, item in enumerate(items):
        j = i + 1
        while not out[i] and j < len(items) and items[j].start < item.end:
            out[i] = group[j] != group[i]
            j += 1
    return out


def at_once(items: Sequence[Any]) -> Callable[[int, int], bool]:
    """Whether the lines at two positions are of different groups and overlap in time: two people talking at
    once, a change of speaker whatever their speaker labels say. Never true without keys."""
    group = group_numbers(items)
    return lambda i, j: (group[i] != group[j] and items[i].start < items[j].end
                         and items[j].start < items[i].end)


def run_end(items: Sequence[Any]) -> float:
    """Where `items` (a window, or lines merged into one) end: the last line's end, or later where a line of
    another group than the last one ends later."""
    group = group_numbers(items)
    return max([items[-1].end] + [item.end for item, g in zip(items, group) if g != group[-1]])
