"""The speaker label of one subtitle line (C3, P12).

Labels are written to `*-diar-rows.json`, one entry per stable line (owner 2026-10-15; before that they
rode on `*-stable.json`'s segments, owner 2026-09-05), reach every reader on its lines through
`subtitles/stable_rows.py`, and are passed through to `-annotated.csv`; no tooling of our own for
picking speakers. The LLM speaker judge's verdicts (`judgments.py`) sit in a file of their own, so
the acoustic labels here stay exactly what the acoustic layer said, and deleting that file is going
back to them.

Two rules this module exists to enforce, both easy to get wrong downstream:

* **A missing field means "not labelled", never "the speaker changed."** The two
  have opposite consequences for a subtitle editor, and unlabelled segments will
  be common: anything the acoustic layer is unsure of stays unlabelled by design.
  A third state sits between the two: a `speaker` object with no `track` and a
  `mark` of tier `filled` is a line the acoustic layer left unassigned and the
  confidence marks (`marks.py`) gave a guessed track, in `mark.track` only --
  so every reader of `track` still sees exactly what the acoustic layer said.
* **Track ids are only comparable inside one block.** The same id in two blocks
  may be two different people -- cross-window identity is 44-65% reliable
  (docs/diarization-findings.md 7), so the carrier refuses to imply more than
  that. Identity that survives across blocks travels as a `name`, and only a
  person writes one: the voiceprint layer that was meant to was rejected (plan
  5c.1), and a track-level name from the model would be wrong on as many lines
  as the track is impure (plan 5g.1).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

#: What a label may claim about a speaker. `unknown` is a first-class answer,
#: not a failure: the plan's cost model is built on saying so out loud.
ROLES = ("streamer", "guest", "other", "unknown")

#: Ordered strongest first. `high` and `median` both get published; anything
#: weaker stays out of the recall-first tracks.
CONFIDENCES = ("high", "median", "low")

#: Where a label came from. Combinations join with `+`, e.g. `llm+human`.
SOURCES = ("acoustic", "llm", "human")

#: The `spk` cell's two flags on a line's lead track. They say different things, so a reader of the
#: cell alone must be able to tell them apart (owner 2026-10-14): `?` is a confidence mark of tier
#: `low` (`marks.py`); `~` is "two tracks cover the line almost equally", the flag a run without
#: marks falls back to. The grammar and its reference parser: `examples/spk_cell.py`.
LOW_FLAG = "?"
CLOSE_FLAG = "~"

#: A person-written name stands in the cell where a track id would; these characters are the grammar's
#: own (and `|` the CSV's), so a name shows them as full-width look-alikes -- the way `|` in subtitle text
#: becomes `｜` (`llm/text_cells.encode_text_cell`). The name in `-diar-rows.json` is untouched.
_NAME_RESERVED = str.maketrans({"+": "＋", "?": "？", "~": "～", "(": "（", ")": "）", "*": "＊", "<": "＜",
                                "|": "｜"})
_FORMULA_LEADS = {"=": "＝", "-": "－", "@": "＠"}


def cell_name(name: str) -> str:
    """A name as the `spk` cell shows it: no grammar character, no spreadsheet formula lead, never a track id.

    Composed and decomposed spellings of one name read the same (NFC), as macOS file names do.
    Whitespace and control characters collapse to single spaces: `str.split` breaks on every character
    `str.splitlines` does, so no name can split a CSV row when `respeak_annotated_csv` rereads it, and the
    speaker exports can put the name into a file name."""
    printable = "".join(" " if unicodedata.category(c) == "Cc" else c for c in unicodedata.normalize("NFC", name))
    shown = " ".join(printable.translate(_NAME_RESERVED).split()) or "＿"
    if shown[0] in _FORMULA_LEADS:
        shown = _FORMULA_LEADS[shown[0]] + shown[1:]
    if shown[0] == "s" and shown[1:].isdigit():
        shown = "ｓ" + shown[1:]
    return shown


def judged_cell(final: str, acoustic: str) -> str:
    """A line the speaker judge moved: its cell with the verdicts applied, `<`, the acoustic layer's cell."""
    return f"{final}<{acoustic}"


class SpeakerLabelError(ValueError):
    """A label that cannot be parsed, rather than one that is merely unsure."""


@dataclass(frozen=True)
class SpeakerLabel:
    """One claim about who is speaking. A segment may carry several."""

    role: str
    conf: str
    name: str | None = None
    src: str = "acoustic"

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise SpeakerLabelError(f"role must be one of {ROLES}, got {self.role!r}")
        if self.conf not in CONFIDENCES:
            raise SpeakerLabelError(f"conf must be one of {CONFIDENCES}, got {self.conf!r}")
        for part in self.src.split("+"):
            if part not in SOURCES:
                raise SpeakerLabelError(f"src part must be one of {SOURCES}, got {part!r}")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "conf": self.conf, "src": self.src}
        if self.name is not None:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SpeakerLabel":
        try:
            role, conf = payload["role"], payload["conf"]
        except KeyError as error:
            raise SpeakerLabelError(f"label is missing {error.args[0]!r}: {dict(payload)}") from error
        return cls(role=role, conf=conf, name=payload.get("name"), src=payload.get("src", "acoustic"))


@dataclass(frozen=True)
class SegmentSpeaker:
    """The `speaker` object of one line (an entry of `-diar-rows.json`)."""

    labels: tuple[SpeakerLabel, ...] = ()
    track: int | None = None
    """Which acoustic track this segment fell in. Comparable only within `block`.
    Never a filled-in guess: that lives in `extras["mark"]["track"]` (`marks.py`)."""

    block: int = 0
    """Track ids restart their meaning at each block boundary."""

    review: bool = False
    """Flagged for a human. Not a failure -- the plan budgets for these. With confidence
    marks it means "has a `low` mark" (the cell's `?`); without them, two tracks cover the line almost
    equally (`~`)."""

    extras: Mapping[str, Any] = field(default_factory=dict)
    """`dist`, `margin` and friends: evidence for auditing, opaque here."""

    @property
    def is_labelled(self) -> bool:
        return bool(self.labels)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"labels": [label.to_dict() for label in self.labels], "block": self.block}
        if self.track is not None:
            payload["track"] = self.track
        if self.review:
            payload["review"] = True
        payload.update(self.extras)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SegmentSpeaker":
        known = {"labels", "track", "block", "review"}
        return cls(
            labels=tuple(SpeakerLabel.from_dict(item) for item in payload.get("labels", ())),
            track=payload.get("track"),
            block=payload.get("block", 0),
            review=bool(payload.get("review", False)),
            extras={key: value for key, value in payload.items() if key not in known},
        )


#: What a segment with no `speaker` field means. Not "silence", not "a new
#: speaker" -- just "nobody said".
UNLABELLED = SegmentSpeaker()


def different_speakers(earlier: SegmentSpeaker, later: SegmentSpeaker) -> bool:
    """Whether two lines read in order are two different people by their main tracks: both have a track, in
    the same block, and the tracks differ. What the correction is told not to merge across (a turn); a `?`
    does not soften it, and nothing here looks at times, so lines that overlap in time compare the same way."""
    return (earlier.track is not None and later.track is not None and earlier.block == later.block
            and earlier.track != later.track)


def speaker_of(segment: Mapping[str, Any]) -> SegmentSpeaker:
    """Read a segment's speaker, treating an absent field as unlabelled."""
    payload = segment.get("speaker")
    if payload is None:
        return UNLABELLED
    if not isinstance(payload, Mapping):
        raise SpeakerLabelError(f"speaker must be an object, got {type(payload).__name__}")
    return SegmentSpeaker.from_dict(payload)


def speaker_cell(
    speakers: Sequence[SegmentSpeaker],
    weights: Sequence[float] | None = None,
    review_margin: float | None = None,
) -> str:
    """One `spk` cell for a subtitle line covering these source segments.

    A rendered line may merge several source segments, so its speakers are the
    union of theirs -- joined by `+`, which is also how a single segment with two
    simultaneous speakers reads. Empty when nothing was labelled: the cell says
    "not labelled", never "silence" or "unknown".

    The first id is the line's main speaker -- that is what the user docs promise.
    For one segment the tracks already come ranked by `coverage`; for a merged line
    the order is re-ranked by coverage summed over the sources, each weighted by
    `weights` (the caller passes durations). Without that, a line joining a 2 s
    segment led by s0 and a 10 s one led by s1 read `s0+s1` (audit 5q, K6). A
    source without `coverage` keeps first-seen order, so single segments and
    names are unchanged.

    Names win over track ids when a person has written them, because a track id
    only means something inside its block ([[same_speaker]]) while a name travels.
    Nothing automatic writes one (see the module docstring).

    A first track followed by `~` is a `review` segment of a run without marks: its
    two tracks cover the line almost equally, so the order says little. Measured on AliMeeting (plan
    5l.7): with the flag the first track was not the one whose words these are
    44% of the time, without it 11% -- "check these first" for a proofreader.

    Every track a source lists is shown. A rendering-side rule that hid a
    runner-up below a share of the leader's coverage lived here between
    2026-09-12 and 2026-09-18; it was removed once theta went back to 0.20,
    where it equalled `second_share` and so could never fire (plan closeout 2).

    Confidence marks (`marks.py`), when the run computed them: `review` then
    means "has a `low` mark", the flag reads `?`, and the caller passes `review_margin=None`, so a
    merged line is flagged when any source is -- not re-decided from coverage.
    A `low` source's `alternative` follows the `?` as `(s5)` when every `low`
    source names the same one, every `low` source is led by the line's main track
    (a runner-up competes with its own source's lead, so on a merged line led by
    another source's track it would name a rival of the wrong track), and the line
    does not already list it. A filled
    source (no `track`, `mark.track` instead) reads `s4*`; on a line with any
    acoustic source it is dropped first, because it has no coverage and would
    switch the coverage ranking off for the whole line. Beside sources with no
    track at all it is what the line shows, as the exports file it there.
    """
    weights = list(weights) if weights is not None else None
    filled = [index for index, speaker in enumerate(speakers) if filled_track(speaker) is not None]
    if filled and all(speaker.track is None and not speaker.labels for speaker in speakers):
        held: dict[int, float] = {}
        for index in filled:
            track = filled_track(speakers[index])
            held[track] = held.get(track, 0.0) + (float(weights[index]) if weights is not None else 1.0)
        return "+".join(f"s{track}*" for track in sorted(held, key=lambda track: -held[track]))
    if filled:
        keep = [index for index in range(len(speakers)) if index not in filled]
        speakers = [speakers[index] for index in keep]
        weights = [weights[index] for index in keep] if weights is not None else None
    low_sources = [speaker for speaker in speakers if _tier(speaker) == "low"]
    lows = [speaker.extras["mark"] for speaker in low_sources]
    # `marks.apply` makes `review` mean "has a low mark", so a flagged line with no low source is a run
    # without marks, whose `review` came from coverage.
    cell = _acoustic_cell(speakers, weights, review_margin, LOW_FLAG if lows else CLOSE_FLAG)
    alternatives = {mark.get("alternative") for mark in lows}
    named = any(label.name for speaker in speakers for label in speaker.labels)
    if not cell or not lows or named or len(alternatives) != 1 or None in alternatives:
        return cell
    alternative = f"s{int(alternatives.pop())}"
    first, plus, rest = cell.partition("+")
    if not first.endswith(LOW_FLAG):
        # The grammar puts a runner-up only after `?`. Reachable only from a stable whose segments carry
        # low marks while its metadata says none were computed, where coverage decided the flag instead.
        return cell
    if any(_lead(speaker) != first[:-len(LOW_FLAG)] for speaker in low_sources):
        return cell
    if alternative in [part.rstrip(LOW_FLAG + CLOSE_FLAG) for part in cell.split("+")]:
        return cell
    return f"{first}({alternative}){plus}{rest}"


def _lead(speaker: SegmentSpeaker) -> str | None:
    """The track one source leads with, as the cell writes it."""
    tracks = speaker.extras.get("tracks")
    track = tracks[0] if tracks else speaker.track
    return None if track is None else f"s{int(track)}"


def _tier(speaker: SegmentSpeaker) -> str | None:
    mark = speaker.extras.get("mark")
    return mark.get("tier") if isinstance(mark, Mapping) else None


def filled_track(speaker: SegmentSpeaker) -> int | None:
    """The track a confidence mark filled in for a line the acoustic layer left unassigned, else None.

    The one definition of a filled line (`marks.py`): no `track`, no name, a `filled` mark."""
    if speaker.track is not None or speaker.labels or _tier(speaker) != "filled":
        return None
    return int(speaker.extras["mark"]["track"])


def _acoustic_cell(
    speakers: Sequence[SegmentSpeaker],
    weights: Sequence[float] | None,
    review_margin: float | None,
    flag: str,
) -> str:
    seen: dict[str, bool] = {}
    share: dict[str, float] = {}
    ranked = True
    for index, speaker in enumerate(speakers):
        names = [cell_name(label.name) for label in speaker.labels if label.name]
        tracks = speaker.extras.get("tracks")
        if not tracks and speaker.track is not None:
            tracks = [speaker.track]
        candidates = names or [f"s{int(track)}" for track in (tracks or ())]
        coverage = speaker.extras.get("coverage")
        if names or not isinstance(coverage, Mapping):
            ranked = ranked and not candidates
        weight = float(weights[index]) if weights is not None else 1.0
        for position, candidate in enumerate(candidates):
            flagged = speaker.review and position == 0 and not names
            seen[candidate] = seen.get(candidate, False) or flagged
        if isinstance(coverage, Mapping) and not names:
            # Every track's coverage counts, listed or not: `assign` ranks one segment on all of
            # them, and a track under `second_share` in a long segment still holds that time.
            # Listed-only shares read `s0~+s1` where the whole line is s1's (audit 5q round 3).
            for track, value in coverage.items():
                share[f"s{track}"] = share.get(f"s{track}", 0.0) + weight * float(value)
    order = list(seen)
    merged = len(speakers) > 1
    if ranked and merged:
        order.sort(key=lambda candidate: -share.get(candidate, 0.0))  # stable: ties keep first-seen
    if merged and order:
        # The flag hangs on the first id only -- "who leads this line is a close call" (5o.1 K5).
        # Re-ranking moved ids, so a source's own flag could land on a second id (audit 5q:
        # `s1+s0?`). With coverage and a margin, decide it for the merged line the way
        # `assign` does for one segment: the line as a whole may not be close at all.
        # Otherwise keep "some source was close" as a flag on the lead.
        if ranked and review_margin is not None and len(order) > 1:
            # The runner-up is taken over every track's summed coverage, as `assign` takes
            # ranked[1] over all of a segment's tracks: a track no source listed can still hold
            # more of the line than the listed second (audit 5q round 4). Only the ids shown and
            # their order stay restricted to what the sources listed.
            lead = share.get(order[0], 0.0)
            runner = max((value for track, value in share.items() if track != order[0]), default=0.0)
            close = lead - runner <= review_margin * lead
        else:
            close = any(seen.values())
        seen = {candidate: False for candidate in order}
        seen[order[0]] = close
    return "+".join(f"{candidate}{flag}" if seen[candidate] else candidate for candidate in order)


def same_speaker(first: SegmentSpeaker, second: SegmentSpeaker) -> bool | None:
    """Do two segments have the same speaker? `None` means "cannot tell".

    Unlabelled segments and cross-block track ids both answer `None` rather than
    `False`, because "we do not know" and "it changed" lead a subtitle editor to
    do opposite things.
    """
    named_first = {label.name for label in first.labels if label.name}
    named_second = {label.name for label in second.labels if label.name}
    if named_first and named_second:
        return bool(named_first & named_second)
    if first.track is None or second.track is None:
        return None
    if first.block != second.block:
        return None
    return first.track == second.track
