"""The speaker judge's person table on a cloud task, where every track brings its own person: the candidates, then
one person per transcription track the first time a unit shows it, and one per new track a unit opens with `@label`;
the judge identifies a track by pointing its person at another person.

Keys: a candidate's is `c:<name>`; a transcription track's person `t:<track>` (the track's number in the file, read
like a candidate's from any unit); a new track's `t:<unit id>:<label>`, read by another unit only while the unit that
opened it still holds the very answers it held when the reference was written (`people_of`, recorded in the referring
unit's `people_seen`). A row a unit answered `pN` opens a track that has no person of its own: it is that person.

Each unit's sidecar entry records what it did, already resolved when it was judged (`people_io.read_people`): `stage`
(the transcription tracks it showed, which get their people first), `cuts`, `opened_people` (its `@` labels),
`writes` (`{"who", "to"}`: a person pointed at where the person written landed at that moment, so the pointers never
form a cycle and a person pointed at someone not yet identified follows them), `people` (notes and aliases),
`persons` (its `@pN` tracks -> their candidate), `mentions` and `people_seen`. The table a unit sees is the units
before it that hold, replayed in order (`replay`); a write whose either side no longer reads is dropped, so its person
falls back to its last write that still reads, or to itself. A write recorded while an earlier write of a unit now
dropped stood is kept as recorded, not resolved again.

A unit writes only what its lines changed, each write with its evidence (`why`, and `cut` when it was longer than
`WHY_CHARS`); a write to a track that landed on a candidate when written records that track (`via`: 「经 s3 认作 p1」).
`?` (a write back to itself) takes back an identification: a track that landed on a candidate when it took it back
counts `?` as one more name in its `history`, so it never votes a name again; a track that landed nowhere takes
nothing back. The notes and aliases stay on the tracks (a candidate gets some only from a unit with no track on
them).

A person's state is its last write; a candidate points at no one. Where a person lands (`land`) is the end of its
pointers: a candidate, or a track's person not identified (pointing at itself). The merge (`judgments.merges`) reads
the end state: a track whose person lands on a candidate votes that name, unless its writes, or its landing, named two
different candidates (`history`); tracks landing on one unidentified person link, that person's own track included,
whatever their history.

A unit may cut a transcription track where its speaker changes: `s3 自q91 = p2` makes the track from row 91 on (this
unit's and the later units') another person, a piece keyed like a track the unit opened (`cut_key`: label
`切s3自q91`, which no `@` label can be). The pieces of a track (`Table.pieces`) are its people in order; the latest is
what `s3` means from that unit on (on stage, in the profiles), the earlier ones are `s3-1`, `s3-2`..., oldest first
(`Table.handle`, `older`), so a cut never renames a piece the units before wrote about. A unit's `cuts` replay before
its writes, so everything it wrote about `s3` is the new piece's; they count in `people_of` like its `@` labels.

Only units of `judgments.MERGING_PROMPT` or later write this table; `replay` skips the others."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: What the harness keeps of a write's evidence (the judge is told 40)
WHY_CHARS = 40
#: `?` in a track's `history`: it took back an identification
TAKEN_BACK = "?"
#: What the harness keeps of one alias (the judge is told 20), and of one appended note (the judge is told 60)
ALIAS_CHARS = 30
NOTE_CHARS = 100


def candidate_key(name: str) -> str:
    return f"c:{name}"


def track_key(track: int) -> str:
    return f"t:{track}"


def opened_key(unit_id: str, label: str) -> str:
    return f"t:{unit_id}:{label}"


def cut_label(track: int, row: str) -> str:
    """The label a cut's piece is opened under (`answers` refuses an `@` label starting with 切)."""
    return f"切s{track}自q{row}"


def cut_key(unit_id: str, track: int, row: str) -> str:
    return opened_key(unit_id, cut_label(track, row))


_CUT = re.compile(r"^切s(\d+)自q(.+)$")


def cut_of(key: str) -> tuple[int, str] | None:
    """The transcription track and first row of a cut's piece, None for any other key."""
    found = _CUT.match(key_label(key) or "")
    return (int(found.group(1)), found.group(2)) if found else None


def piece_track(key: str) -> int | None:
    """The transcription track a person is a piece of: its own `t:<track>`, or a cut's."""
    track = key_track(key)
    return track if track is not None else (cut[0] if (cut := cut_of(key)) else None)


def candidate(key: str) -> bool:
    return key.startswith("c:")


def key_unit(key: str) -> str | None:
    """The unit that opened a new track's person, None for a candidate or a transcription track's."""
    parts = key.split(":", 2)
    return parts[1] if key.startswith("t:") and len(parts) == 3 else None


def key_track(key: str) -> int | None:
    """The transcription track a `t:<track>` person stands for."""
    parts = key.split(":")
    return int(parts[1]) if key.startswith("t:") and len(parts) == 2 and parts[1].isdigit() else None


def key_label(key: str) -> str | None:
    return key.split(":", 2)[2] if key_unit(key) is not None else None


def people_of(entry: Mapping[str, Any]) -> str:
    """What a unit wrote, hashed: a reference to a person it opened holds while this does."""
    payload = json.dumps({field_: entry.get(field_) for field_ in ("opened_people", "writes", "people", "persons", "cuts")
                          if field_ != "cuts" or field_ in entry},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class TrackPerson:
    key: str
    name: str
    number: int
    #: the notes: (segment that wrote it, text), oldest first
    notes: list[tuple[int, str]] = field(default_factory=list)
    #: the last segment that named this person (0: none)
    last: int = 0
    #: what the lines say this person is called, oldest first
    aliases: list[str] = field(default_factory=list)
    #: a candidate's other names from the programme's data (production's `Candidate.others`)
    others: tuple[str, ...] = ()
    #: a candidate's description from the preparation task (production's `Candidate.note`), shown whole
    prepared: str = ""


@dataclass
class Table:
    """The table as some units left it. `people` in creation order (their numbers); `point` each track's person ->
    the person its last write points at (itself when none); `since` the segment of that write; `history` each
    track's person -> the candidates its writes landed on when written; `persons` per unit its `@pN` tracks ->
    candidate; `against` per unit the pairs it wrote to two different candidates (person keys, or a `@pN` of the
    unit); `created` the `people_of` of each unit that opened a person; `onstage` the people of the tracks the
    unit about to be judged shows (`stage`); `why` and `via` of each person's last write (absent: none),
    `opened_in` the segment that opened a new track's person, `voiced` the people of off-stage tracks whose voice
    is close to a track on stage (listed so the unit can link them)."""

    people: dict[str, TrackPerson] = field(default_factory=dict)
    point: dict[str, str] = field(default_factory=dict)
    since: dict[str, int] = field(default_factory=dict)
    history: dict[str, set[str]] = field(default_factory=dict)
    persons: dict[str, dict[str, str]] = field(default_factory=dict)
    against: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    created: dict[str, str] = field(default_factory=dict)
    onstage: set[str] = field(default_factory=set)
    why: dict[str, str] = field(default_factory=dict)
    via: dict[str, str] = field(default_factory=dict)
    opened_in: dict[str, int] = field(default_factory=dict)
    voiced: set[str] = field(default_factory=set)
    #: a cut track's pieces, oldest first (its `t:<track>`, then each cut's); a track never cut is not here
    pieces: dict[int, list[str]] = field(default_factory=dict)

    def add(self, key: str, name: str) -> TrackPerson:
        if key not in self.people:
            self.people[key] = TrackPerson(key, name, len(self.people) + 1)
            if not candidate(key):
                self.point[key] = key
        return self.people[key]

    def stage(self, tracks: Iterable[int]) -> list[int]:
        """Give each track a unit shows its person (in track order, as `replay` does), and mark them on stage."""
        shown = sorted(set(tracks))
        for track in shown:
            self.add(track_key(track), f"s{track} 的说话人")
        self.onstage = {self.current(track) for track in shown}
        return shown

    def current(self, track: int) -> str:
        """The person `s<track>` is now: its latest piece, else its own."""
        return (self.pieces.get(track) or [track_key(track)])[-1]

    def cut(self, unit_id: str, track: int, row: str, segment: int) -> str:
        """Open a cut's piece after the track's latest one; the stage follows it."""
        key = cut_key(unit_id, track, row)
        before = self.current(track)
        self.add(key, f"s{track} 第{segment}段 q{row} 起的说话人")
        self.opened_in[key] = segment
        pieces = self.pieces.setdefault(track, [track_key(track)])
        if key not in pieces:
            pieces.append(key)
        if before in self.onstage:
            self.onstage = (self.onstage - {before}) | {key}
        return key

    def older(self, key: str) -> int:
        """An earlier piece's number: 1 for a cut track's first piece (`s3-1`), 2 for the next...; 0 for the
        one `s3` means, and for a track never cut."""
        track = piece_track(key)
        pieces = self.pieces.get(track, []) if track is not None else []
        return pieces.index(key) + 1 if key in pieces[:-1] else 0

    def numbered(self, number: int) -> str | None:
        return next((key for key, person in self.people.items() if person.number == number), None)

    def handle(self, key: str) -> str:
        """What the judge calls a person: `p3` a candidate, `s5` a transcription track, `第3段@甲` a track a
        unit opened."""
        if candidate(key):
            return f"p{self.people[key].number}"
        if (track := piece_track(key)) is not None:
            older = self.older(key)
            return f"s{track}" + (f"-{older}" if older else "")
        return f"第{self.opened_in.get(key, 0)}段@{key_label(key)}"

    def land(self, key: str, point: Mapping[str, str] | None = None) -> str:
        """Where a person's pointers end: a candidate, or a person pointing at itself (cycles cannot be written;
        the walk stops at the first person seen twice anyway)."""
        point = self.point if point is None else point
        seen = {key}
        while not candidate(key) and point.get(key, key) != key and point[key] not in seen:
            key = point[key]
            seen.add(key)
        return key

    def children(self, key: str) -> list[str]:
        return sorted((other for other, to in self.point.items() if to == key and other != key),
                      key=lambda other: self.people[other].number)

    def listed(self) -> list[str]:
        """The people shown: every candidate, every track's person with a description or an alias, pointing at
        someone or `voiced`, and whomever a listed person points at (so an indented line always hangs under a listed
        one). A track on stage with none of these is left to its profile line."""
        earlier = {key for pieces in self.pieces.values() for key in pieces[:-1]}
        shown = {key for key, person in self.people.items()
                 if candidate(key) or person.notes or person.aliases or key in self.voiced
                 or self.point.get(key, key) != key or key in earlier}
        for key in list(shown):
            seen = {key}
            while not candidate(key) and self.point.get(key, key) not in seen:
                key = self.point[key]
                seen.add(key)
                shown.add(key)
        return [key for key in self.people if key in shown]

    def write(self, who: str, to: str, segment: int, *, why: str = "", via: str = "") -> None:
        """Point `who` at `to` (already resolved); a write that would close a cycle leaves `who` on itself. A write
        back to itself while `who` lands on a candidate records `?` in its history."""
        if to != who and self.land(to) == who:
            to = who
        if to == who and candidate(self.land(who)):
            self.history.setdefault(who, set()).add(TAKEN_BACK)
        self.point[who] = to
        self.since[who] = segment
        for kept, value in ((self.why, why), (self.via, via)):
            if value:
                kept[who] = value
            else:
                kept.pop(who, None)
        if candidate(to):
            self.history.setdefault(who, set()).add(to)


def replay(cast: Sequence[str], units: Sequence[tuple[str, Mapping[str, Any]]],
           others: Mapping[str, Sequence[str]] | None = None, prepared: Mapping[str, str] | None = None) -> Table:
    """The table after `units` (`(unit id, sidecar entry)`, the units that hold, in order; units before v27 (`dev`'s
    v11, `judgments.merging_unit`) are skipped). `cast` seeds it with the candidates, numbered in order, `others`
    their other names, `prepared` their descriptions from the preparation task; a candidate a unit
    names that is not in `cast` is added when first named (the merge replays without the cast). Ops were checked
    when written and are applied as they are; a rule that depends on the table (an append to a full description, an
    alias already there) is not checked again."""
    from .judgments import merging_unit

    table = Table()
    for name in cast:
        person = table.add(candidate_key(name), name)
        person.others = tuple((others or {}).get(name, ()))
        person.prepared = (prepared or {}).get(name, "")
    for unit_id, entry in units:
        if not merging_unit(entry):
            continue
        segment = int(entry.get("segment") or 0)
        seen = entry.get("people_seen") or {}

        def ensure(key: Any) -> bool:
            if not isinstance(key, str):
                return False
            if candidate(key):
                table.add(key, key[2:])
                return True
            opener = key_unit(key)
            if opener is not None and opener != unit_id and table.created.get(opener) != seen.get(key):
                return False
            return key in table.people

        table.stage(int(track) for track in entry.get("stage") or ())
        for cut in entry.get("cuts") or ():
            table.cut(unit_id, int(cut["track"]), str(cut["row"]), segment)
        for label in entry.get("opened_people") or ():
            table.add(key := opened_key(unit_id, str(label)), f"第{segment}段新开的轨 @{label} 的说话人")
            table.opened_in[key] = segment
        written: dict[str, str] = {}
        for write in entry.get("writes") or ():
            who, to, via = write.get("who"), write.get("to"), write.get("via")
            if ensure(who) and not candidate(who) and ensure(to):
                table.write(who, to, segment, why=str(write.get("why") or ""),
                            via=via if via and ensure(via) else "")
                written[who] = to
        for op in entry.get("people") or ():
            key = op.get("key")
            if not ensure(key):
                continue
            person = table.people[key]
            if op.get("op") == "add":
                person.notes.append((segment, str(op.get("text", ""))))
            elif op.get("op") == "call" and op.get("text") not in person.aliases:
                person.aliases.append(str(op.get("text", "")))
        persons = {track: key for track, key in (entry.get("persons") or {}).items() if candidate(str(key)) and ensure(key)}
        table.persons[unit_id] = persons
        named = [(who, to) for who, to in written.items() if candidate(to)] + list(persons.items())
        table.against[unit_id] = [(a, b) for (a, x), (b, y) in itertools.combinations(named, 2) if x != y]
        for key in [*table.onstage, *(entry.get("mentions") or ())]:
            if ensure(key):
                table.people[key].last = segment
        if entry.get("opened_people") or entry.get("cuts"):
            table.created[unit_id] = people_of(entry)
    table.onstage = set()
    return table
