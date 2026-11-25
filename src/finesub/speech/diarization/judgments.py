"""The LLM speaker judge's results, read back for the `spk` cell and the split-track export.

The judge (`finesub.llm.stages.speaker_judge`) reads the raw ASR rows with their acoustic labels
(`-stable.json` with `-diar-rows.json`, through `subtitles.stable_rows`) window by window and answers per row: keep, move to another track, open a new track, or abstain. Its answers
live in `<stem>-speaker-judge.json` next to `-stable.json`, and change neither that file nor `-diar-rows.json` --
deleting the sidecar is going back to the acoustic labels. This module is the one reader; the lines everything after
the judge reads are made from what it reads (`product`), so the correction, the `spk` column and the exports all see
the same verdicts.

Reading is checked, because the sidecar outlives what it was judged on (it is a record, never cleaned
up, and only a model call could make it again):

* the rows' ids, times and text must hash as they did when the windows were cut -- a re-transcribed
  stable is a different file, and the whole sidecar is dropped;
* the windows must tile the file in order, each row in exactly one, and a window halved into `parts`
  (its answer or its prompt did not fit) must be tiled by them -- anything else is damage;
* each unit's key -- a window's, or each part's when it was cut: its rows, their acoustic labels and
  marks, the tracks the profiles drew on -- must still match, or that unit falls back to the acoustic
  labels: re-running diarization renumbers the tracks, and an old verdict must never land on a new id;
* each unit's results must read (a list of rows in the shapes the judge writes), or the unit is owed;
* each result's `from` (the row's acoustic main track when it was judged) must still be the row's;
* a carried track's `of` must be the verdict its opener holds now, or that row falls back.

New tracks continue the acoustic numbering: past the largest track id anything in the stable mentions
(a filled line's `mark.track` or a `low` mark's `alternative` can name a track that leads no line, and
ids need not be contiguous), in unit order, then first appearance in the unit; each unit's new tracks
get their own numbers -- the same person in two units is two tracks, merged by a person if at all --
unless a unit judged after it carried the track on (units judged in order hand the next one the tracks
the previous one opened or used, owner 2026-10-09), which the sidecar records as that unit's id and
label (and the opener's verdict). So the numbering follows from the stable and the sidecar alone, whatever order they finished in.

Three passes (`docs/llm_harness_behavior.md`「说话人裁判」) have the judge answer pass 1's shape as well: each unit
then also holds `acts` (row id -> `{"act", "why"}`: `join`, `may_join` or `drop`) and the acts it set aside
(`ignored`). A three-pass task asks for `acts` (`holds(shape=...)`, `covers`, `read_acts`), so a unit judged
without them is judged again, while a single-pass task reuses a unit that has them. A unit marked `trackless`
(acts without results, from a pass 1 that answered the shape alone) moves nobody: the reader skips it, and
the judge judges it again.

A judge that hears the audio may also cut a row where the speaker changes (`splits`: the row, its text
before the cut, `at` -- the second half's first word -- and who speaks after it, written as a result is).
The verdicts stay on the rows as the judge read them: a split row's verdict is its first half's, and the
reader gives the second half's track in `Judgments.halves`; a second half's new label is numbered with
the unit's results, after them. A split that no longer fits its row's words (the words changed, the
text did not) is not read. The cut itself is made in the lines after the judge (`product`), never in the rows
the sidecar was judged on.

It may also add what it heard and the ASR missed: the rest of a row's sentence (`fills`: the row, the text heard
whole, the characters added at its head and tail) and a whole line after a row (`inserts`: the row it follows,
its text, the seconds the judge heard it at on the source timeline, its track named as a result's is). The
reader places them in time over the whole file, since a row's neighbours may sit in another unit: a filled row
grows by its added characters at speech speed into the silence beside it, never over the row before it (as that
one grew) or the row after it; then an insert shows in what is left between its row's end and the next row's
start, as long as it was heard (at least `MIN_INSERT_SECONDS`), cut at the next row. Subtitles do not overlap
yet: an insert with less than `MIN_INSERT_SECONDS` of room is deferred -- kept, with the seconds it was heard at
for when they may, but not shown. No original row moves or shrinks. Every row counts as a neighbour, a dropped
one included.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .labels import SegmentSpeaker, filled_track, speaker_of
from ...subtitles.stable_rows import read_rows

SIDECAR_SUFFIX = "-speaker-judge.json"
#: 2: one window per correction window, cut into `parts` only when its answer does not fit (owner
#: 2026-10-09); a version-1 file (150-row windows, a cast pick each) reads as absent and is judged again.
SIDECAR_VERSION = 2

#: Pass 1's shape answers: a row joins the kept line before it into one subtitle, may be merged with it by
#: the correction, or is left out.
ACT_JOIN = "join"
ACT_MAY = "may_join"
ACT_DROP = "drop"
ACTS = (ACT_JOIN, ACT_MAY, ACT_DROP)
#: Which pass 1 a reader asks for: None is the judge alone (single pass), `SHAPE_TRACKED` the judge with
#: the shape section.
SHAPE_TRACKED = "tracked"
#: Speech speed, for the time of text the judge heard and no word timing covers (characters a second).
CHARS_PER_SECOND = 7.0
#: The least room an insert shows in; with less it is deferred.
MIN_INSERT_SECONDS = 0.5


def sidecar_path(stable_json: str | Path) -> Path:
    """`<stem>-speaker-judge.json` beside `<stem>-stable.json`."""
    stable = Path(stable_json)
    return stable.with_name(f"{stable.name.removesuffix('.json').removesuffix('-stable')}{SIDECAR_SUFFIX}")


def segment_ids(segments: Sequence[Mapping[str, Any]]) -> list[str]:
    """Row ids as `load_segments_from_stable_json` reads them: an explicit `id`, else the 1-based position."""
    return [str(segment.get("id") or index) for index, segment in enumerate(segments, start=1)]


def rows_hash(segments: Sequence[Mapping[str, Any]]) -> str:
    """The rows' ids, times and text -- not their speakers, so re-running diarization keeps the windows."""
    rows = [[key, segment.get("start"), segment.get("end"), str(segment.get("text", ""))]
            for key, segment in zip(segment_ids(segments), segments)]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def acoustic_main(speaker: SegmentSpeaker) -> int | dict | None:
    """A row's `from`: its acoustic track, `{"filled": k}` for a line the marks filled, None when unassigned."""
    if speaker.track is not None:
        return int(speaker.track)
    filled = filled_track(speaker)
    return {"filled": filled} if filled is not None else None


def profile_tracks(speakers: Sequence[SegmentSpeaker]) -> list[int]:
    """The tracks a profile is written for: every track leading at least one line."""
    return sorted({int(speaker.track) for speaker in speakers if speaker.track is not None})


def largest_track(segments: Sequence[Mapping[str, Any]]) -> int:
    """The largest track id anything in these speaker objects names; -1 when none does."""
    found = [-1]
    for segment in segments:
        payload = segment.get("speaker")
        if not isinstance(payload, Mapping):
            continue
        if payload.get("track") is not None:
            found.append(int(payload["track"]))
        found += [int(track) for track in payload.get("tracks") or ()]
        found += [int(track) for track in (payload.get("coverage") or {})]
        mark = payload.get("mark")
        if isinstance(mark, Mapping):
            found += [int(mark[field]) for field in ("track", "alternative") if mark.get(field) is not None]
    return max(found)


def window_key(segments: Sequence[Mapping[str, Any]], ids: Sequence[str], tracks: Sequence[int]) -> str:
    """One window's identity: its rows (id, text, the whole speaker object -- labels and marks) and the
    profile tracks. Not whether the prompt showed the marks: switching that does not void a verdict."""
    rows = [[key, str(segment.get("text", "")), segment.get("speaker")] for key, segment in zip(ids, segments)]
    payload = json.dumps([rows, list(tracks)], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class Filled:
    """A row the judge heard more of: its text whole, and its start and end grown to hold what was added."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Inserted:
    """A line the judge heard missing after the row `after`: `heard` is when (source seconds), `shown` where
    it goes in the subtitles, None while it is deferred (no room without overlapping a row)."""

    after: str
    text: str
    track: int
    heard: tuple[float, float]
    shown: tuple[float, float] | None


@dataclass(frozen=True)
class Merge:
    """Two tracks the judge said are one person, merged file-wide: `track` now reads as `into`. `by` is `link`
    (their people land on one person no unit identified) or `name` (on the same candidate, `name`)."""

    track: int
    into: int
    by: str
    name: str = ""


@dataclass(frozen=True)
class Judgments:
    """What the sidecar changes, after every check: `moved` maps a row id to the track it now leads; `merged` maps a
    track the judge said is another track's person to the track it now reads as (`merges`), after every move."""

    moved: Mapping[str, int]
    new_tracks: tuple[int, ...] = ()
    abstained: frozenset[str] = frozenset()
    #: row id -> (the second half's first word, its track), for each row the judge split
    halves: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    #: row id -> the row as the judge heard it whole
    fills: Mapping[str, Filled] = field(default_factory=dict)
    #: the lines the judge heard missing, in row order
    inserts: tuple[Inserted, ...] = ()
    merged: Mapping[int, int] = field(default_factory=dict)
    #: the merges made, and the pairs the judge called one person that the gate refused (`(a, b, why)`)
    merges: tuple[Merge, ...] = ()
    refused: tuple[tuple[int, int, str], ...] = ()


def _warn(message: str) -> None:
    print(f"Warning: {message}", file=sys.stderr)  # product output


def read_judgments(
    stable_json: str | Path,
    segments: Sequence[Mapping[str, Any]] | None = None,
    *,
    warn: Callable[[str], None] = _warn,
) -> Judgments | None:
    """The verdicts that still hold for this stable, or None when there is no usable sidecar.

    `segments` defaults to the stable's own; a caller that already parsed it passes them. A cloud task's merges
    read its voice comparison (`speech.joint.voices`) when there is one for these rows."""
    loaded = _load(stable_json, segments, warn)
    if loaded is None:
        return None
    segments, _, found = loaded
    from ..joint import voices as joint_voices

    stored = joint_voices.read(stable_json, segments)
    if stored is None and joint_voices.voices_path(stable_json).exists():
        warn(f"{joint_voices.voices_path(stable_json).name} is for other rows or another version; the speaker "
             "judge's merges pass no voice gate (delete it to have it made again)")
    resolved = resolve(segments, found, voices=stored)
    if resolved.stale:
        warn(f"{resolved.stale} speaker-judge window(s) no longer match their rows (diarization re-run?); "
             "those rows keep the acoustic labels")
    return resolved.judgments


def _load(stable_json: str | Path, segments: Sequence[Mapping[str, Any]] | None,
          warn: Callable[[str], None]) -> tuple[Sequence[Mapping[str, Any]], list, list["Unit"]] | None:
    """`(segments, windows, units)` of a usable sidecar, or None (with a warning when there is one but it does not
    read, or describes another transcript)."""
    path = sidecar_path(stable_json)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if segments is None:
            segments = read_rows(stable_json)
    except (OSError, ValueError) as error:
        warn(f"{path.name} unreadable ({error}); speaker judge results ignored")
        return None
    if not isinstance(payload, dict) or payload.get("version") != SIDECAR_VERSION:
        warn(f"{path.name} has an unknown version; speaker judge results ignored")
        return None
    if payload.get("rows_hash") != rows_hash(segments):
        warn(f"{path.name} was judged on another transcript; speaker judge results ignored")
        return None
    windows = list(payload.get("windows") or [])
    found = units(windows, segment_ids(segments))
    if found is None:
        warn(f"{path.name} is damaged (its windows do not tile the file); speaker judge results ignored")
        return None
    return segments, windows, found


@dataclass(frozen=True)
class ShapeActs:
    """Pass 1's shape answers that still hold for this stable."""

    #: row id -> act, from the units that hold
    acts: Mapping[str, str]
    #: row id -> the reason given, for each dropped row (the task report lists them)
    dropped: Mapping[str, str]
    #: the rows of the units that do not hold. The correction maps them onto its own windows (which may
    #: have been planned again since the judge cut its own): a window with any of them goes without pass 1
    #: as a whole, its rows' acts unused
    missing_rows: frozenset[str]
    #: the judge's windows (chunk ids) with a unit that does not hold, for the report
    missing_windows: frozenset[str]


def read_acts(
    stable_json: str | Path,
    segments: Sequence[Mapping[str, Any]] | None = None,
    *,
    shape: str,
    warn: Callable[[str], None] = _warn,
) -> ShapeActs | None:
    """The shape answers of `shape`'s pass 1 (`SHAPE_TRACKED`), or None when there is no usable sidecar. An act
    outside its unit's rows is not read."""
    loaded = _load(stable_json, segments, warn)
    if loaded is None:
        return None
    segments, windows, found = loaded
    ids = segment_ids(segments)
    position = {key: index for index, key in enumerate(ids)}
    tracks = profile_tracks([speaker_of(segment) for segment in segments])
    acts: dict[str, str] = {}
    dropped: dict[str, str] = {}
    missing_rows: set[str] = set()
    missing_windows: set[str] = set()
    for window, (low, high) in zip(windows, tiling(windows, ids) or ()):
        for unit in (unit for unit in found if low <= unit.low < high):
            if not holds(segments, unit, tracks, shape=shape):
                missing_rows.update(ids[unit.low:unit.high])
                missing_windows.add(str(window.get("chunk")))
                continue
            for row, answer in unit.entry["acts"].items():
                if unit.low <= position.get(row, -1) < unit.high:
                    acts[row] = answer["act"]
                    if answer["act"] == ACT_DROP:
                        dropped[row] = answer["why"]
    return ShapeActs(acts, dropped, frozenset(missing_rows), frozenset(missing_windows))


@dataclass(frozen=True)
class Unit:
    """What one judge call answered: a whole window, or one part of a window that was cut. `entry` is
    the sidecar's record of it (status, key, results...); rows `[low, high)`."""

    id: str
    entry: Mapping[str, Any]
    low: int
    high: int


def part_id(window: str, half: int) -> str:
    """A half's id, as the correction names its split windows: `0001-a`, `0001-b`."""
    return f"{window}-{'ab'[half]}"


def units(windows: Sequence[Any], ids: Sequence[str]) -> list[Unit] | None:
    """The units the windows hold, in row order; None when the windows do not tile the file or a cut
    window's parts do not tile it."""
    spans = tiling(windows, ids)
    if spans is None:
        return None
    out = []
    for window, (low, high) in zip(windows, spans):
        parts = window.get("parts")
        if not parts:
            out.append(Unit(str(window.get("id")), window, low, high))
            continue
        inner = tiling(parts, ids[low:high])
        if inner is None:
            return None
        out += [Unit(str(part.get("id")), part, low + start, low + end) for part, (start, end) in zip(parts, inner)]
    return out


@dataclass(frozen=True)
class Resolved:
    judgments: Judgments
    #: `(unit id, label)` -> the track number that label opened
    opened: Mapping[tuple[str, str], int]
    stale: int = 0


def verdict_of(entry: Mapping[str, Any]) -> str:
    """One unit's verdict as it stands -- its key and its results: what a later unit carrying one of its new
    tracks on names, so a re-judged opener (the same label, maybe another person) never inherits it."""
    payload = json.dumps({"key": entry.get("key"), "results": entry.get("results"),
                          **{name: entry[name] for name in ("splits", "inserts", "cuts") if name in entry}},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _well_formed(results: Any) -> bool:
    """Whether a unit's results read: a list of rows, each a move, a new track, a carried track or an
    abstention in the shape the judge writes. A damaged file falls back rather than failing the task."""
    if not isinstance(results, list):
        return False
    for result in results:
        if not isinstance(result, Mapping) or "row" not in result:
            return False
        track, carried = result.get("track"), result.get("carried")
        if track is not None and (isinstance(track, bool) or not isinstance(track, int)):
            return False
        if carried is not None and not (isinstance(carried, Mapping)
                                        and all(isinstance(carried.get(key), str) for key in ("window", "label", "of"))):
            return False
    return True


def _well_formed_acts(acts: Any) -> bool:
    """Whether a unit's shape answers read: row id -> `{"act", "why"}`."""
    return isinstance(acts, Mapping) and all(
        isinstance(row, str) and isinstance(answer, Mapping) and answer.get("act") in ACTS
        and isinstance(answer.get("why"), str) for row, answer in acts.items())


def _well_formed_splits(splits: Any) -> bool:
    """Whether a unit's splits read: results that each carry a word index and name a track."""
    return _well_formed(splits) and all(
        isinstance(split.get("at"), int) and not isinstance(split.get("at"), bool) and split["at"] > 0
        and any(split.get(key) is not None for key in ("track", "label", "carried")) for split in splits)


def _well_formed_cuts(cuts: Any) -> bool:
    """Whether a unit's cuts read: a list of `{"track": int, "row": str}`."""
    return isinstance(cuts, list) and all(
        isinstance(cut, Mapping) and isinstance(cut.get("track"), int) and not isinstance(cut.get("track"), bool)
        and isinstance(cut.get("row"), str) for cut in cuts)


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _seconds(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _well_formed_fills(fills: Any) -> bool:
    """Whether a unit's fills read: a row, its text, and the characters added at its head and tail (some)."""
    return isinstance(fills, list) and all(
        isinstance(fill, Mapping) and isinstance(fill.get("row"), str) and isinstance(fill.get("text"), str)
        and _count(fill.get("head")) and _count(fill.get("tail")) and fill["head"] + fill["tail"] > 0
        for fill in fills)


def _well_formed_inserts(inserts: Any) -> bool:
    """Whether a unit's inserts read: results that each carry a text, heard seconds in order, and a track."""
    return _well_formed(inserts) and all(
        isinstance(insert.get("text"), str) and _seconds(insert.get("start")) and _seconds(insert.get("end"))
        and insert["end"] > insert["start"] and any(insert.get(key) is not None for key in ("track", "label", "carried"))
        for insert in inserts)


def holds(segments: Sequence[Mapping[str, Any]], unit: Unit, tracks: Sequence[int] | None = None, *,
          shape: str | None = None) -> bool:
    """Whether a unit's verdict stands for the pass 1 `shape` asks for: it succeeded, on these very rows, and
    what that pass 1 reads is there and reads -- the results, and the acts (not for the judge alone)."""
    entry = unit.entry
    ids = segment_ids(segments)
    if entry.get("status") != "success":
        return False
    tracks = profile_tracks([speaker_of(segment) for segment in segments]) if tracks is None else tracks
    return _well_formed(entry.get("results")) and \
        _well_formed_splits(entry.get("splits", [])) and _well_formed_fills(entry.get("fills", [])) and \
        _well_formed_inserts(entry.get("inserts", [])) and _well_formed_cuts(entry.get("cuts", [])) and \
        (shape is None or _well_formed_acts(entry.get("acts"))) and \
        entry.get("key") == window_key(segments[unit.low:unit.high], ids[unit.low:unit.high], tracks)


def cut_label(track: int, row: str) -> str:
    from .track_people import cut_label as label

    return label(track, row)


def resolve(segments: Sequence[Mapping[str, Any]], found: Sequence[Unit], *,
            voices: Mapping[str, Any] | None = None, merging: bool = True) -> Resolved:
    """What the units that still hold say, in row order -- the reader's rule, and the numbers the judge
    itself shows for a track it carries forward (it must show the number the reader will give).

    A result is a move to an acoustic track (`track`), a new track (`label`, numbered here), a track an
    earlier unit opened and this one carried on (`carried`: that unit's id, label and verdict -- dropped
    when that unit no longer holds that very verdict), or an abstention. A split's second half and an insert
    name their tracks the same ways, numbered after the unit's results; fills and inserts are placed in time
    once every unit is read (`placed`). `merging=False` leaves the file-wide merges out (`merges`), for a caller
    that only wants the moves and the new tracks' numbers.

    A cut (`cuts`: a track and the row from which it is another person) opens a track after the unit's results
    (`track_people.cut_label`); every row of that track from there on that no unit moved goes to it, the latest cut
    before a row winning, as far as the units the rows are in saw it: the cutting unit's own rows, and a later
    unit's when it was shown that cut as its unit stood (`cuts_seen`, written when the person table is relayed).
    A later unit that never saw it (`parallel`, or the cutting unit judged again after it) keeps what it judged:
    its rows stay where its own answers left them (owner 2026-11-25). A row in no unit that holds follows the cut.
    A row of another track answered the cut track is written as the new track already."""
    from .track_people import opened_key, people_of

    ids = segment_ids(segments)
    speakers = [speaker_of(segment) for segment in segments]
    tracks = profile_tracks(speakers)
    position = {key: index for index, key in enumerate(ids)}
    moved: dict[str, int] = {}
    opened: dict[tuple[str, str], int] = {}
    abstained: set[str] = set()
    halves: dict[str, tuple[int, int]] = {}
    fills: dict[str, Mapping[str, Any]] = {}
    inserts: list[tuple[Mapping[str, Any], int]] = []
    counter = [largest_track(segments) + 1]
    cut_from: dict[int, list[tuple[int, int, str, str, str]]] = {}  # at, its track, cutter, piece, cutter's state
    owner: dict[int, Unit] = {}  # row index -> the unit holding it

    def following_track() -> int:
        counter[0] += 1
        return counter[0] - 1

    stale = 0
    verdicts: dict[str, str] = {}
    for unit in found:
        low, high = unit.low, unit.high
        if unit.entry.get("status") != "success" or unit.entry.get("trackless"):
            continue
        if not holds(segments, unit, tracks):
            stale += 1
            continue
        name = unit.id
        verdicts[name] = verdict_of(unit.entry)
        owner.update((index, unit) for index in range(low, high))

        def track_of(result: Mapping[str, Any]) -> int | None:
            if result.get("label") is not None:
                key = (name, str(result["label"]))
                if key not in opened:
                    opened[key] = following_track()
                return opened[key]
            if isinstance(result.get("carried"), Mapping):
                source = (str(result["carried"].get("window")), str(result["carried"].get("label")))
                if source in opened and verdicts.get(source[0]) == result["carried"].get("of"):
                    return opened[source]
                return None
            return None if result.get("track") is None else int(result["track"])

        def own(result: Mapping[str, Any]) -> str | None:
            row = str(result.get("row", ""))
            index = position.get(row)
            if index is None or not low <= index < high or result.get("from") != acoustic_main(speakers[index]):
                return None
            return row

        for result in unit.entry.get("results") or ():
            row = own(result)
            if row is None:
                continue
            if result.get("abstain"):
                abstained.add(row)
            elif (track := track_of(result)) is not None:
                moved[row] = track
        for split in unit.entry.get("splits") or ():
            row = own(split)
            words = segments[position[row]].get("words") or [] if row is not None else []
            if row is not None and split["at"] < len(words) and (track := track_of(split)) is not None:
                halves[row] = (int(split["at"]), track)

        def inside(edit: Mapping[str, Any]) -> bool:
            return low <= position.get(str(edit.get("row", "")), -1) < high

        fills.update((str(fill["row"]), fill) for fill in unit.entry.get("fills") or () if inside(fill))
        inserts += [(insert, track) for insert in unit.entry.get("inserts") or ()
                    if inside(insert) and (track := track_of(insert)) is not None]
        for cut in unit.entry.get("cuts") or ():
            at = position.get(cut["row"])
            if at is not None and low <= at < high and acoustic_main(speakers[at]) == cut["track"]:
                label = cut_label(cut["track"], cut["row"])
                cut_from.setdefault(cut["track"], []).append(
                    (at, track_of({"label": label}), name, opened_key(name, label), people_of(unit.entry)))
    for index, row in enumerate(ids):
        main = acoustic_main(speakers[index])
        if isinstance(main, int) and main in cut_from and row not in moved:
            holder = owner.get(index)
            seen = (holder.entry.get("cuts_seen") or {}) if holder is not None else {}
            starts = [track for at, track, cutter, piece, made in sorted(cut_from[main]) if at <= index
                      and (holder is None or cutter == holder.id or seen.get(piece) == made)]
            if starts:
                moved[row] = starts[-1]
    grown, shown = placed(segments, fills, inserts)
    judgments = Judgments(moved=moved, new_tracks=tuple(sorted(opened.values())), abstained=frozenset(abstained),
                          halves=halves, fills=grown, inserts=shown)
    if merging:
        spans, rows = talk_spans(segments, judgments)
        holding = [unit for unit in found if unit.id in verdicts]
        done = merges(segments, holding, opened, spans, voices, rows=rows)
        judgments = replace(judgments, merged=done.merged, merges=done.merges, refused=done.refused)
    return Resolved(judgments, opened, stale)


def holding_units(segments: Sequence[Mapping[str, Any]], found: Sequence[Unit]) -> list[Unit]:
    """The units `resolve` reads: succeeded, not trackless, and still holding."""
    tracks = profile_tracks([speaker_of(segment) for segment in segments])
    return [unit for unit in found if unit.entry.get("status") == "success" and not unit.entry.get("trackless")
            and holds(segments, unit, tracks)]


def talk_spans(segments: Sequence[Mapping[str, Any]], judgments: Judgments
               ) -> tuple[dict[int, list[tuple[float, float]]], dict[int, list[int]]]:
    """When each track talks in the product layer of `judgments` (before any merge): moved rows, both halves of a
    split row, filled rows at their grown times and inserted lines on their own tracks; and which whole rows (indices
    into `segments`, not a split row's halves or an inserted line) each track holds there."""
    from .product import WHOLE, product

    made = product(segments, judgments)
    speakers = made.speakers() or {}
    position = {key: index for index, key in enumerate(segment_ids(segments))}
    spans: dict[int, list[tuple[float, float]]] = {}
    rows: dict[int, list[int]] = {}
    for line in made.lines:
        speaker = speakers.get(line.id)
        if speaker is not None and speaker.track is not None:
            spans.setdefault(int(speaker.track), []).append((float(line.row["start"]), float(line.row["end"])))
            if line.part == WHOLE and line.id in position:
                rows.setdefault(int(speaker.track), []).append(position[line.id])
    return spans, rows


#: The first judge prompt whose units merge tracks (`merges`) and write the person table (`track_people.replay`).
#: Units before v27 (`dev`'s v11) never merge: a prompt bump re-judges nothing, so such a unit is reused on resume
#: with its moves, but it casts no vote and the person table skips it. A single pass's windows carry the correction's
#: prompt version (`stages.correction.speaker_book`), never this one's.
MERGING_PROMPT = 27
_TRACK_NAME = re.compile(r"^s(\d+)$")
#: A cut's piece's label (`track_people.cut_label`): the transcription track it was cut from
_CUT_LABEL = re.compile(r"^切s(\d+)自q")


def _prompt_number(entry: Mapping[str, Any]) -> int:
    found = re.fullmatch(r"speaker-judge-v(\d+)", str(entry.get("prompt_version") or ""))
    return int(found.group(1)) if found else 0


def merging_unit(entry: Mapping[str, Any]) -> bool:
    """Whether a unit votes in the merge and is replayed into the person table (`MERGING_PROMPT`)."""
    return _prompt_number(entry) >= MERGING_PROMPT


def unit_track(unit: Unit, key: str, base: int, opened: Mapping[tuple[str, str], int],
               verdicts: Mapping[str, str]) -> int | None:
    """The track a unit's `<tracks>` key or link (`sN`, `@label`) means, as the file numbers it now: `sN` under `base`
    (the first number past the transcription's tracks) is the transcription's; a higher `sN` is read by what it was when
    the unit was judged (its `shown_tracks`: the opener and label behind it) and is None once that opener holds another
    verdict; `@label` is the unit's own new track. None for anything else."""
    found = _TRACK_NAME.match(key)
    if found and int(found.group(1)) < base:
        return int(found.group(1))
    if found:
        shown = (unit.entry.get("shown_tracks") or {}).get(key)
        fresh = isinstance(shown, Mapping) and verdicts.get(str(shown.get("window"))) == shown.get("of")
        return opened.get((str(shown["window"]), str(shown.get("label")))) if fresh else None
    return opened.get((unit.id, key[1:])) if key.startswith("@") else None


def merges(segments: Sequence[Mapping[str, Any]], units: Sequence[Unit], opened: Mapping[tuple[str, str], int],
           spans: Mapping[int, Sequence[tuple[float, float]]], voices: Mapping[str, Any] | None, *,
           rows: Mapping[int, Sequence[int]] | None = None) -> "Merged":
    """The tracks a cloud task's judge said are one person, merged file-wide (`docs/llm_harness_behavior.md`「说话人裁判」):
    `(track -> the track it reads as, the merges, the refused pairs)`. Local tasks merge nothing.

    Votes, from the units that hold and merge (`merging_unit`), read from the person table after every unit
    (`track_people.replay`; `_track_people_votes` says how): tracks whose people land on the same candidate, one for
    each pair; tracks whose people land on one person no unit identified, one (`link`); a unit that wrote two tracks
    to two different candidates, one against. A track whose writes named two different candidates, or that took an
    identification back, is likely two people on one label: its name casts no vote (refused as `names`). Pairs with
    more for than against are merged, the surest first (then by track number), unless the gate refuses them or any
    pair across the two groups: two tracks that talk at once more than `relink.MERGE_OVERLAP` of the shorter one's
    talk (one person never overlaps themselves; read from `spans`, each track's talk in the product layer before the
    merges -- `talk_spans` -- so it holds without voiceprints), two tracks whose voiceprints both exist and compare
    under `voices.FLOOR`, or a pair with more votes against than for. A merged group reads as its track with the most
    talk, the lower number on a tie. Tracks are taken after the moves, so a track the judge opened merges like any
    other. The `unknown` track (lines no speaker was found for) is no one and never merges.

    A track's voiceprint: a transcription track no unit cut is its label's; a track the judge opened (a cut's piece,
    an `@` label) and every track a unit cut (its first piece keeps the number) are their own rows' in the product
    layer before the merges (`rows`, from `talk_spans`; `voices.piece`), none under `relink.MIN_LABEL` clean seconds --
    a pair missing one passes the voice gate."""
    state = _merging(segments, units, opened, spans, voices, rows=rows)
    if state is None:
        return Merged({}, (), ())
    group, joined = state.relay()
    return Merged(state.merged(group), state.made(group, joined), tuple(state.refused))


@dataclass(frozen=True)
class Merged:
    """`merges`' answer: each merged track -> the track it reads as, the merges made and the refused pairs."""

    merged: dict[int, int]
    merges: tuple[Merge, ...]
    refused: tuple[tuple[int, int, str], ...]


@dataclass
class _Merging:
    """What the merge reads (`merges`): the votes, the gate's inputs, and each track's voiceprint reading."""

    labels: dict[int, str]
    known: set[int]
    talk: dict[int, float]
    spans: Mapping[int, Sequence[tuple[float, float]]]
    support: Counter
    against: Counter
    how: dict[tuple[int, int], tuple[str, str]]
    refused: list[tuple[int, int, str]]
    cos: Callable[[int, int], float | None]

    def gate(self, a: int, b: int) -> str | None:
        from ..joint import voices as joint_voices
        from ..joint.relink import MERGE_OVERLAP

        pair = (min(a, b), max(a, b))
        if self.against[pair] > self.support[pair]:
            return "disputed"
        both = joint_voices.overlap_seconds(self.spans.get(a, ()), self.spans.get(b, ()))
        if self.talk[a] and self.talk[b] and both > MERGE_OVERLAP * min(self.talk[a], self.talk[b]):
            return "overlap"
        cos = self.cos(a, b)
        return "voice" if cos is not None and cos < joint_voices.FLOOR else None

    def relay(self) -> tuple[dict[int, set[int]], dict[int, tuple[str, str]]]:
        """The relay's round: pairs with more for than against, the surest first, each through the gate."""
        support, against = self.support, self.against
        group = {track: {track} for track in self.known}
        joined: dict[int, tuple[str, str]] = {}
        for a, b in sorted((pair for pair in support if support[pair] > against[pair]),
                           key=lambda pair: (against[pair] - support[pair], pair)):
            if group[a] is group[b]:
                continue
            why = next((why for x in group[a] for y in group[b] if (why := self.gate(x, y))), None)
            if why:
                self.refused.append((a, b, why))
                continue
            union = group[a] | group[b]
            for track in union:
                group[track] = union
            for track in (a, b):
                joined.setdefault(track, self.how[a, b])
        return group, joined

    def keep(self, members: set[int]) -> int:
        """The track a group reads as: the most talk, the lower number on a tie."""
        return max(members, key=lambda track: (self.talk[track], -track))

    def made(self, group: Mapping[int, set[int]], joined: Mapping[int, tuple[str, str]]) -> tuple[Merge, ...]:
        made: list[Merge] = []
        for members in {id(g): g for g in group.values() if len(g) > 1}.values():
            keep = self.keep(members)
            for track in sorted(members - {keep}):
                made.append(Merge(track, keep, *joined.get(track, joined.get(keep, ("name", "")))))
        return tuple(made)

    def merged(self, group: Mapping[int, set[int]]) -> dict[int, int]:
        merged: dict[int, int] = {}
        for members in {id(g): g for g in group.values() if len(g) > 1}.values():
            keep = self.keep(members)
            for track in sorted(members - {keep}):
                merged[track] = keep
        return merged


def _merging(segments: Sequence[Mapping[str, Any]], units: Sequence[Unit], opened: Mapping[tuple[str, str], int],
             spans: Mapping[int, Sequence[tuple[float, float]]], voices: Mapping[str, Any] | None, *,
             rows: Mapping[int, Sequence[int]] | None = None) -> _Merging | None:
    """The votes and the gate `merges` reads (its rules); None when no track has a label (a local task)."""
    from ..joint import voices as joint_voices
    from ..joint.model import UNKNOWN_SPEAKER

    speakers = [speaker_of(segment) for segment in segments]
    labels: dict[int, str] = {}
    for speaker in speakers:
        if speaker.track is not None and speaker.extras.get("joint_speaker") is not None:
            labels.setdefault(int(speaker.track), str(speaker.extras["joint_speaker"]))
    if not labels:
        return None
    nobody = {track for track, label in labels.items() if label == UNKNOWN_SPEAKER}
    known = {*spans, *labels, *opened.values()} - nobody
    talk = {track: sum(end - start for start, end in spans.get(track, ())) for track in known}

    pair_of = joint_voices.pairs(voices)
    own = set(opened.values()) | {int(found.group(1)) for _, label in opened if (found := _CUT_LABEL.match(label))}
    prints: dict[int, Any] = {}

    def voice_of(track: int) -> Any:
        if track not in prints:
            prints[track] = joint_voices.piece(voices, (rows or {}).get(track, ())) if track in own \
                else joint_voices.label_print(voices, labels[track]) if track in labels else None
        return prints[track]

    def cos(a: int, b: int) -> float | None:
        if a not in own and b not in own:  # two labels: their stored pair
            voice = pair_of(labels[a], labels[b]) if a in labels and b in labels else None
            return voice.get("cos") if voice is not None else None
        return joint_voices.cosine(voice_of(a), voice_of(b))

    support: Counter[tuple[int, int]] = Counter()
    against: Counter[tuple[int, int]] = Counter()
    how: dict[tuple[int, int], tuple[str, str]] = {}
    by_person: dict[str, list[int]] = {}
    shown_as: dict[str, str] = {}
    linked: dict[str, set[int]] = {}
    refused: list[tuple[int, int, str]] = []
    voting = [unit for unit in units if merging_unit(unit.entry)]
    if voting:
        from . import track_people as TP

        _track_people_votes(TP.replay((), [(unit.id, unit.entry) for unit in voting]), voting, opened, known,
                            by_person, shown_as, linked, against, refused)
    for person, tracks in by_person.items():
        for pair in itertools.combinations(sorted(tracks), 2):
            support[pair] += 1
            how.setdefault(pair, ("name", shown_as[person]))
    for tracks in linked.values():
        for pair in itertools.combinations(sorted(tracks), 2):
            support[pair] += 1
            how.setdefault(pair, ("link", ""))
    return _Merging(labels, known, talk, spans, support, against, how, refused, cos)


def _track_people_votes(table: Any, units: Sequence[Unit], opened: Mapping[tuple[str, str], int], known: set[int],
                        by_person: dict[str, list[int]], shown_as: dict[str, str], linked: dict[str, set[int]],
                        against: Counter[tuple[int, int]], refused: list[tuple[int, int, str]]) -> None:
    """The votes, read from the person table after every unit (`track_people`): a track whose person lands on a
    candidate is one name vote for it, unless its writes or its landing named two candidates, or it took an
    identification back (`?`) (refused as `names`: no vote, and no link through that candidate either); tracks whose
    people land on one unidentified person link, that person's own track with them, whatever names their writes went
    through (a refused name never refuses the link); a `@pN` track votes its candidate; a unit that wrote two tracks
    to two candidates is one vote against them."""
    from . import track_people as TP

    def person_track(key: str) -> int | None:
        track = TP.key_track(key)
        if track is None and TP.key_unit(key) is not None:
            track = opened.get((str(TP.key_unit(key)), str(TP.key_label(key))))
        return track if track in known else None

    for key in table.people:
        track = None if TP.candidate(key) else person_track(key)
        if track is None:
            continue
        end = table.land(key)
        names = table.history.get(key, set()) | ({end} if TP.candidate(end) else set())
        if TP.candidate(end) and len(names) > 1:
            refused.append((track, track, "names"))
        elif TP.candidate(end):
            by_person.setdefault(end, []).append(track)
            shown_as.setdefault(end, table.people[end].name)
        elif end != key:
            own = person_track(end)
            linked.setdefault(end, set()).update({track, *(() if own is None else (own,))})
    for unit in units:
        def unit_track_of(side: str) -> int | None:
            track = opened.get((unit.id, side[1:])) if side.startswith("@") else person_track(side)
            return track if track in known else None

        for at, key in (table.persons.get(unit.id) or {}).items():
            if (track := unit_track_of(at)) is not None:
                by_person.setdefault(key, []).append(track)
                shown_as.setdefault(key, table.people[key].name)
        for a, b in table.against.get(unit.id, ()):
            x, y = unit_track_of(a), unit_track_of(b)
            if x is not None and y is not None and x != y:
                against[min(x, y), max(x, y)] += 1


def placed(segments: Sequence[Mapping[str, Any]], fills: Mapping[str, Mapping[str, Any]],
           inserts: Sequence[tuple[Mapping[str, Any], int]]) -> tuple[dict[str, Filled], tuple[Inserted, ...]]:
    """Fills grown into the silence beside their rows, then inserts in the room left (the module's rules)."""
    ids = segment_ids(segments)
    times = [(float(segment["start"]), float(segment["end"])) for segment in segments]
    grown: dict[str, Filled] = {}
    for index, key in enumerate(ids):
        fill = fills.get(key)
        if fill is None:
            continue
        start, end = times[index]
        before = (grown[ids[index - 1]].end if ids[index - 1] in grown else times[index - 1][1]) if index else -math.inf
        after = times[index + 1][0] if index + 1 < len(ids) else math.inf
        head, tail = fill["head"] / CHARS_PER_SECOND, fill["tail"] / CHARS_PER_SECOND
        grown[key] = Filled(str(fill["text"]), round(min(start, max(before, start - head, 0.0)), 3),
                            round(max(end, min(after, end + tail)), 3))
    position = {key: index for index, key in enumerate(ids)}
    shown = []
    for insert, track in sorted(inserts, key=lambda item: position[str(item[0]["row"])]):
        index = position[str(insert["row"])]
        heard = (float(insert["start"]), float(insert["end"]))
        room = (grown[ids[index]].end if ids[index] in grown else times[index][1],
                (grown[ids[index + 1]].start if ids[index + 1] in grown else times[index + 1][0])
                if index + 1 < len(ids) else math.inf)
        start = max(heard[0], room[0])
        end = min(start + max(heard[1] - heard[0], MIN_INSERT_SECONDS), room[1])
        shown.append(Inserted(ids[index], str(insert["text"]), track, heard,
                              (round(start, 3), round(end, 3)) if end - start >= MIN_INSERT_SECONDS else None))
    return grown, tuple(shown)


def tiling(windows: Sequence[Any], ids: Sequence[str]) -> list[tuple[int, int]] | None:
    """Each window's `[low, high)` row span, or None unless they cover every row once, in order."""
    position = {key: index for index, key in enumerate(ids)}
    spans, cursor = [], 0
    for window in windows:
        if not isinstance(window, Mapping):
            return None
        low, high = position.get(str(window.get("first"))), position.get(str(window.get("last")))
        if low != cursor or high is None or high < low:
            return None
        spans.append((low, high + 1))
        cursor = high + 1
    return spans if cursor == len(ids) and ids else None


def decided(track: int, like: SegmentSpeaker) -> SegmentSpeaker:
    """A speaker the judge decided (a moved row, a second half, an inserted line): it leads `track` alone, with no
    `?`, no alternative and no fill. `like` is the acoustic speaker it replaces (its labels and block stay)."""
    return SegmentSpeaker(labels=like.labels, track=track, block=like.block,
                          extras={"tracks": [track], "coverage": {str(track): 1.0}})


def judgeable(segments: Sequence[Mapping[str, Any]]) -> bool:
    """Whether the judge can run on these rows at all: some row has an acoustic track, all in one block."""
    speakers = [speaker_of(segment) for segment in segments]
    return any(s.track is not None for s in speakers) and all(s.block == 0 for s in speakers)


def covers(stable_json: str | Path, *, shape: str | None = None) -> bool:
    """Whether a usable sidecar holds a successful verdict of `shape`'s pass 1 (`holds`) for every window on
    these very rows.

    The prompt version is not asked, here or by the judge (owner 2026-10-09): a verdict a bumped prompt
    would give differently is still the task's own result, reused like any other. A finished task the sidecar does not cover gets the missing
    windows judged; one the judge cannot run on at all (no labels, several blocks) counts as covered,
    so it keeps the finished-task shortcut."""
    path = sidecar_path(stable_json)
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        segments = read_rows(stable_json)
    except (OSError, ValueError):
        return False
    if not judgeable(segments):
        return True
    if not isinstance(payload, dict) or payload.get("version") != SIDECAR_VERSION \
            or payload.get("rows_hash") != rows_hash(segments):
        return False
    found = units(payload.get("windows") or [], segment_ids(segments))
    if found is None:
        return False
    tracks = profile_tracks([speaker_of(segment) for segment in segments])
    return all(holds(segments, unit, tracks, shape=shape) for unit in found)
