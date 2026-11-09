"""Three passes: the lines pass 2 corrects, grouped by pass 1's shape.

Pass 1 (the speaker judge with its shape section) answers, per line, `join` (one subtitle with the kept line
before it), `may_join` (the same person carrying on: pass 2 may merge the two) or `drop` (a filler or a sure
ASR hallucination). This module turns those answers into the grouped lines pass 2 reads and maps pass 2's rows
back to the original lines, so `-annotated.csv` keeps naming original rows (`position`, the `spk` column,
the knowledge harvest and the per-track export all read them). The original lines are the product lines
(`speech.diarization.product`): where a judge that heard the audio split a row or added a line, pass 1's acts,
given on the judge's rows, carry over to them (`line_acts`).

What the script holds back, because the prompt alone did not (the prototype once wrote 19 over-long joins in
a window): a join never crosses two speakers or a window's first core line; a join that would run the group
past the hard threshold's seconds or past the merge source cap stays a separate line marked mergeable instead;
two lines never read mergeable across a change of speaker or when together they would run past the absolute
threshold's seconds. Both thresholds are the task's, at its length ratio (`prompt_constants.Thresholds`): pass 1's
prompt is the same at every ratio, so these two checks are where its joins follow the task's tier. Dropped lines leave the input and are absent from the products, as the lines the
correction itself discards are.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Collection, Iterable, List, Mapping, Sequence

from .chunking import Seam, SubtitleSegment, SubtitleWindow, turn_seams
from .output_protocol import TRANSLATED_MAX_MERGED_SOURCES, TranslatedCsvSegment
from .prompt_constants import Thresholds
from finesub.speech.diarization.judgments import ACT_DROP, ACT_JOIN, ACT_MAY, ShapeActs
from finesub.speech.diarization.labels import SegmentSpeaker, different_speakers
from finesub.speech.diarization.product import FIRST, INSERTED, WHOLE, Line, Product
from finesub.subtitles import line_groups
from finesub.text import is_space_script_letter


@dataclass(frozen=True)
class Grouping:
    #: the kept lines in reading order, each the original ids it joins
    groups: tuple[tuple[str, ...], ...]
    #: the seam before each grouped line, by its first id
    seams: Mapping[str, Seam]
    #: the ids pass 1 dropped
    dropped: tuple[str, ...] = ()
    #: (id, why) for each join the script did not apply: `speaker`, `seconds`, `rows` or `window`
    refused: tuple[tuple[str, str], ...] = ()

    def members(self) -> dict[str, tuple[str, ...]]:
        """Each grouped line's original ids, by its first id."""

        return {group[0]: group for group in self.groups}


def line_acts(shaped: ShapeActs | None, layer: Product) -> tuple[dict[str, str], frozenset[str] | None]:
    """Pass 1's acts and the lines it left unanswered (`judgments.read_acts`, on the judge's rows), carried over
    to the product lines of `layer`: a whole row and a split row's first half keep their row's act; a second half
    and an inserted line have none, so each stands alone (a second half always follows a change of speaker, and
    the act of the line after it is a join to it). A join or `may_join` reaches back to the last line kept (dropped
    lines are skipped, as `group_lines` does), so what lies between decides: across an inserted line only a drop
    stands -- nothing joins it or reads mergeable over it; a line the judge heard missing that has no room yet
    (`deferred`) is not a line but someone spoke there, and only a drop stands when that someone is another person
    than the line kept or the line itself (a join would hide the turn), the act when it is the same person or a
    side has no speaker to tell. A line of an unanswered row is unanswered. `shaped` None (no usable sidecar): no
    acts, and None for the unanswered lines -- every line."""

    if shaped is None:
        return {}, None
    speakers = layer.speakers() or {}
    acts: dict[str, str] = {}
    kept: Line | None = None  # the line a join reaches back to
    heard: list[SegmentSpeaker] = []  # deferred lines since it
    previous: Line | None = None
    for line in layer.lines:
        if previous is not None and previous.source != line.source and previous.source in layer.deferred:
            heard.append(layer.deferred[previous.source])
        act = shaped.acts.get(line.source) if line.part in (WHOLE, FIRST) else None
        if act is not None and act != ACT_DROP and kept is not None and _across(kept, line, heard, speakers):
            act = None
        if act is not None:
            acts[line.id] = act
        if act != ACT_DROP:
            kept, heard = line, []
        previous = line
    return acts, frozenset(line.id for line in layer.lines if line.source in shaped.missing_rows)


def _across(kept: Line, line: Line, heard: Sequence[SegmentSpeaker], speakers: Mapping[str, SegmentSpeaker]) -> bool:
    """Whether joining `line` back to `kept` would cross something: an inserted line, or a deferred one of another
    person than either side."""
    if kept.part == INSERTED:
        return True
    if kept.id not in speakers or line.id not in speakers:
        return False
    return any(different_speakers(speakers[kept.id], one) or different_speakers(one, speakers[line.id])
               for one in heard)


def group_lines(
    segments: Sequence[SubtitleSegment],
    acts: Mapping[str, str],
    *,
    different: Callable[[str, str], bool],
    thresholds: Thresholds,
    breaks: Iterable[str] = (),
) -> Grouping:
    """Group `segments` (original lines in reading order) by pass 1's `acts` (id -> act; a line with none
    stands alone and is not mergeable). `different(a, b)` tells whether two lines' speakers differ (the judged
    ones, or the acoustic labels where no judge names people). `breaks` are the ids that start a window's core
    lines: nothing joins or reads mergeable across one, so every grouped line falls inside one window. A
    dropped break hands its place to the next kept line, which then opens that window's core. ``thresholds``:
    the task's, which the two checks on pass 1's joins hold to."""

    window_starts = frozenset(breaks)
    breaks: set[str] = set()
    by_id = {segment.id: segment for segment in segments}
    groups: List[List[str]] = []
    refused: List[tuple[str, str]] = []
    long_join: set[str] = set()
    dropped: List[str] = []
    pending_break = False
    for segment in segments:
        act = acts.get(segment.id)
        if act == ACT_DROP:
            dropped.append(segment.id)
            pending_break = pending_break or segment.id in window_starts
            continue
        if pending_break or segment.id in window_starts:
            breaks.add(segment.id)
            pending_break = False
        if act == ACT_JOIN and groups:
            why = _refusal(groups[-1], segment, by_id, different, breaks, thresholds)
            if why is None:
                groups[-1].append(segment.id)
                continue
            refused.append((segment.id, why))
            if why in ("seconds", "rows"):
                long_join.add(segment.id)
        groups.append([segment.id])

    seams: dict[str, Seam] = {}
    for index, group in enumerate(groups):
        first = group[0]
        if index == 0:
            seams[first] = Seam.NONE
            continue
        before = groups[index - 1]
        # a window's first core line still follows the overlap carried into it: a change there stays one
        if different(before[-1], first):
            seams[first] = Seam.CHANGE
        elif first not in breaks and (acts.get(first) == ACT_MAY or first in long_join) and \
                by_id[group[-1]].end - by_id[before[0]].start <= thresholds.absolute_seconds:
            seams[first] = Seam.MAY
        else:
            seams[first] = Seam.NONE
    # a change told apart (同时, 换回) on the grouped lines pass 2 reads, the dropped ones gone
    told = turn_seams([replace(by_id[group[0]], end=line_groups.run_end([by_id[m] for m in group]), seam=seams[group[0]])
                       for group in groups])
    seams = {line.id: line.seam for line in told}
    return Grouping(tuple(tuple(group) for group in groups), seams, tuple(dropped), tuple(refused))


def _refusal(
    group: Sequence[str],
    segment: SubtitleSegment,
    by_id: Mapping[str, SubtitleSegment],
    different: Callable[[str, str], bool],
    breaks: set[str],
    thresholds: Thresholds,
) -> str | None:
    """Why `segment` may not join `group`, or None."""

    if segment.id in breaks:
        return "window"
    if different(group[-1], segment.id):
        return "speaker"
    if segment.end - by_id[group[0]].start > thresholds.hard_seconds:
        return "seconds"
    rows = len(group) + 1
    # the merge source cap: two lines, or three when the middle one is a lone filler
    sandwich = rows == 3 and len(by_id[group[1]].text.strip()) <= thresholds.filler_max_chars
    if rows > TRANSLATED_MAX_MERGED_SOURCES and not sandwich:
        return "rows"
    return None


def grouped_windows(windows: Sequence[SubtitleWindow], grouping: Grouping) -> List[SubtitleWindow]:
    """The windows with their lines grouped: a grouped line takes its first line's id, the span of its
    lines and their text joined. A group cut by a window's edge (the overlap carried over from the window
    before, or the read-only lines before it) is left out there; the window's own core lines are whole,
    since `group_lines` broke every group at the core's first line. The read-only lines are grouped too, so
    they read as the window before them was given. A window whose core lines were all dropped is left out."""

    members = grouping.members()
    group_of = {member: group[0] for group in grouping.groups for member in group}
    out = []
    for window in windows:
        segments = _grouped(window.segments, members, group_of, grouping.seams)
        overlap = _grouped(window.overlap_segments, members, group_of, grouping.seams)
        kept = {member for line in segments for member in members[line.id]}
        core = window.segments[len(window.overlap_segments):]
        if segments[:len(overlap)] != overlap or any(line.id in group_of and line.id not in kept for line in core):
            raise ValueError(f"window {window.chunk_id}: a group crosses its first core line")
        if len(segments) == len(overlap):
            continue
        out.append(replace(window, segments=segments, overlap_segments=overlap,
                           preceding_segments=_grouped(window.preceding_segments, members, group_of,
                                                       grouping.seams), grouped=True))
    return out


def pass2_windows(
    windows: Sequence[SubtitleWindow],
    segments: Sequence[SubtitleSegment],
    acts: Mapping[str, str],
    without_pass1: Collection[str] | None,
    *,
    different: Callable[[str, str], bool],
    thresholds: Thresholds,
) -> tuple[List[SubtitleWindow], Grouping]:
    """Pass 2's windows and the grouping that maps its rows back. A window any of whose core lines is in
    `without_pass1` (`judgments.read_acts`' rows no judged unit covers; None when it read no usable sidecar,
    so every line) stays as it is and gets the single pass prompt, and its lines' acts are not used: grouping
    some of its lines and not others would leave it more broken up than single pass, with no error. The
    others are grouped (`grouped_windows`)."""

    missing = None if without_pass1 is None else frozenset(without_pass1)
    cores = {window.chunk_id: window.segments[len(window.overlap_segments):] for window in windows}
    fallback = {
        chunk_id for chunk_id, core in cores.items()
        if missing is None or any(line.id in missing for line in core)
    }
    unused = {line.id for chunk_id in fallback for line in cores[chunk_id]}
    grouping = group_lines(
        segments,
        {line: act for line, act in acts.items() if line not in unused},
        different=different,
        thresholds=thresholds,
        breaks=[core[0].id for core in cores.values() if core],
    )
    grouped = {window.chunk_id: window for window in
               grouped_windows([window for window in windows if window.chunk_id not in fallback], grouping)}
    out = [window if window.chunk_id in fallback else grouped[window.chunk_id]
           for window in windows if window.chunk_id in fallback or window.chunk_id in grouped]
    return out, grouping


def _grouped(
    items: Sequence[SubtitleSegment],
    members: Mapping[str, tuple[str, ...]],
    group_of: Mapping[str, str],
    seams: Mapping[str, Seam],
) -> List[SubtitleSegment]:
    by_id = {item.id: item for item in items}
    out: List[SubtitleSegment] = []
    for item in items:
        first = group_of.get(item.id)
        if first != item.id:  # dropped, or not the group's first line
            continue
        group = members[first]
        if not all(member in by_id for member in group):
            continue
        lines = [by_id[member] for member in group]
        out.append(replace(lines[0], end=line_groups.run_end(lines), text=joined_text(line.text for line in lines),
                           seam=seams[first] if lines[0].seam is not None else None))
    return out


def joined_text(texts: Iterable[str]) -> str:
    """Lines' texts as one: a space only between two letters (or ASCII digits) of a space-delimited script."""

    def spaced(ch: str) -> bool:
        return (ch.isascii() and ch.isdigit()) or is_space_script_letter(ch)

    out = ""
    for text in (text.strip() for text in texts):
        if out and text and spaced(out[-1]) and spaced(text[0]):
            out += " "
        out += text
    return out


def expand_sources(rendered: Sequence[TranslatedCsvSegment], grouping: Grouping) -> List[TranslatedCsvSegment]:
    """Pass 2's rows with each source id (a grouped line) spelled as the original lines it stands for.

    A row's times stay as pass 2 read them, off the grouped lines (whose end is already `run_end` of their
    members). That is `run_end` over the original lines too, except where pass 1 joined two speakers' keys
    (the judge relabelled one), a later line of the first key starts inside the second, and pass 2 merges
    the two anyway (across a change of speaker, which only warns): the row then ends at the later line's end,
    before the second key's. Rare, a slightly short cue at worst; recomputing it would carry the original lines into every
    window's commit."""

    members = grouping.members()
    return [
        replace(row, source_ids=tuple(member for source in row.source_ids for member in members.get(source, (source,))))
        for row in rendered
    ]


def over_cap(rendered: Sequence[TranslatedCsvSegment], grouping: Grouping) -> List[List[str]]:
    """The original lines of each of pass 2's rows that merges more of them than the soft source cap
    (`TRANSLATED_MAX_MERGED_SOURCES`): the validator warns by the grouped lines it was shown."""

    return [list(row.source_ids) for row in expand_sources(rendered, grouping)
            if len(row.source_ids) > TRANSLATED_MAX_MERGED_SOURCES]
