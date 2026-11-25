"""The person table as a judge unit on a cloud task reads and writes it (the table itself and its replay are
`speech.diarization.track_people`).

Every track is a person, called by its handle (`Table.handle`: `s5`, `第3段@甲`, and the unit's own `@甲`); `pN` is
only a candidate. What the unit is shown: the listed people (`Table.listed`) as trees, a track indented under the
candidate or the track it was identified as, with the segment and the evidence (`render_table`); on each track's
profile line what the units before identified it as (`profile_names`); the voice pairs with listed tracks off stage
(`prompt.voice_lines`). What it writes: `pN` as a row's answer (a new track for that candidate:
`answers_with_people`); `<tracks>` lines `左 = 右（凭据）` for what its lines changed (`read_people`), where a
candidate or a link without evidence is set aside and `?` takes an identification back; `<people>` lines, an alias
(`被称为`) or a note (`补`) on a track. Everything it writes about people is fail-soft: a line that does not read
is set aside (`people_ignored`), never sent back for repair.

A `<tracks>` line `s3 自q91 = 右` cuts the track where its speaker changes (`Table.cut`): from that line on, wherever it
stands, `s3` in this unit is the new piece. A whole `s3 = pM` is read as a cut at the track's first row here
(`forced`) when the units before moved rows of its piece to a track landing on pM, or answered pM (`guard`, from
`run.cuts_before`). A cut track's earlier pieces are `s3-1`, `s3-2`... for good (`Table.handle`); a handle with a dash
and a number after it is such a piece or refused, never the track. A unit's cuts are read in two steps: every
`sK 自qN` whose left side holds is a cut first, then each line's right side is read as any other line -- so two
cuts in one unit may name each other's pieces, and a cut whose right side is refused still stands, its new piece
unidentified (the line in `people_ignored`) -- as when it says the new piece is the very one it was cut from (an
earlier piece is fine: a track may change back).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Collection, Mapping, Sequence

from ....speech.diarization import track_people as T
from ...prompt_compose import load_prompt_template

TABLE_TEMPLATE = "speaker_judge_people_table_v2.md"
#: With the preparation task's descriptions on the candidates: the table's header also says who wrote 〔准备任务〕
PREPARED_TABLE_TEMPLATE = "speaker_judge_people_table_v3.md"
#: The table's rough ceiling, in characters (owner: 8000 tokens; only noted in the unit's entry)
TABLE_CHARS = 8000
#: How much of one person's notes the table shows, the latest first (the sidecar keeps them all); a candidate's
#: preparation task description is shown whole before them and not counted
NOTES_SHOWN = 300
#: How much of an unidentified track's notes its profile line repeats (the latest first)
PROFILE_NOTES_SHOWN = 120

PHRASES = {
    "candidate": "{indent}- p{number} {name}{others}{aliases}{notes}",
    "track": "{indent}- {handle}{span}（{how}）{aliases}{notes}",
    # a cut track's pieces
    "span_from": "〔第{segment}段 q{row} 起〕",
    "span_until": "〔到第{segment}段 q{row} 之前〕",
    "others": "（别名：{others}）",
    "aliases": "（台词里叫：{aliases}）",
    "track_aliases": "（被称为：{aliases}）",
    "notes": "：{notes}",
    "note": "〔第{segment}段〕{text}",
    # a candidate's description from the preparation task, before the judges' notes
    "prepared": "〔准备任务〕{text}",
    "unknown": "还没认出",
    # an unidentified track's profile line brings what the table recorded on it
    "unknown_noted": "还没认出；人物表里记着它{clues}",
    "clue_aliases": "被称为 {aliases}",
    "clue_notes": "{notes}",
    "clue_join": "，",  # "；" separates a profile line's parts
    # ... and so does one linked to a track no one is identified as yet; a linked track also brings what the table
    # recorded on the last track of its links (the one a candidate is identified on, when it ends at one)
    "linked_noted": "；还没认出，{handle} 自己在人物表里记着{clues}",
    "linked_own": "；{handle} 自己在人物表里记着{clues}",
    "linked_end": "；{handle} 在人物表里记着{clues}",
    "named": "第{segment}段{via}认作{why}",
    "linked": "第{segment}段{via}连上{why}",
    "via": "经 {handle} ",
    "why": "：{why}",
    "no_people": "（无）",
    # the profile line and the voice pairs off stage
    "landed": "前面{via}认作 {name}（第{segment}段{why}）",
    "joined": "前面{via}连到 {handle}（第{segment}段{why}）{further}",
    "further_named": "，{handle} 前面认作 {name}",
    "further_linked": "，{handle} 连到 {other}",
    "profile_why": "：{why}",
    # the moves' destinations (`prompt.moved_phrase`)
    "short_named": "认作 {name}",
    "short_linked": "连到 {handle}",
    "brief": "前面{short}",
}

_AT_PN = re.compile(r"^@p\d+$")
_ANSWER_PN = re.compile(r'("track"\s*:\s*")p(\d+)(")')
_ANSWER_AT_PN = re.compile(r'"track"\s*:\s*"@p(\d+)"')
#: the dashes a piece number may follow (`s3-1`), NFKC'd
_DASHES = "-‐‑‒–—―ー−~〜～"
#: a handle at the start of a side: `?`, `p3`, `s5`, `s3-1`, `@甲`, `第3段@甲` (NFKC: brackets and `？` `＠` are
#: ASCII); `s3前`, a name for an earlier piece the prompt does not use, is caught to be refused
_HANDLE = re.compile(r"^(?:(?P<back>\?)|(?P<letter>[ps])(?P<number>\d+)(?!\d)"
                     r"(?:\s*[" + re.escape(_DASHES) + r"]\s*(?P<piece>\d+)(?!\d)|(?P<old>前\d*))?"
                     r"|(?:第\s*(?P<segment>\d+)\s*段\s*)?@(?P<label>[^\s()=,，、:;；]+))\s*(?P<rest>.*)$")
#: a cut's left side: `s3 自q91`
_CUT_LEFT = re.compile(r"^s(\d+)\s*自\s*q(\S+)$")
#: what a refused right side of a cut adds to its reason (the cut itself stands)
CUT_STANDS = "; the cut stands, its new piece unidentified"
#: why a line on a track the script cut for its voice is dropped
VOICE_GUARDED = "the voice changed and the track was not cut: the script cut it, this unit's lines on it are dropped"
_LINE = re.compile(r"^(.+?)\s*(被称为|被稱為|补|補)\s*:\s*(.*)$")
_VERB = {"被稱為": "被称为", "補": "补"}


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def display(table: T.Table, key: str) -> str:
    """A candidate as `p3 Coco`, a track by its handle."""
    return f"{table.handle(key)} {table.people[key].name}" if T.candidate(key) else table.handle(key)


def _note_texts(person: T.TrackPerson, limit: int = NOTES_SHOWN) -> list[str]:
    """The latest notes that fit `limit` (the latest always), oldest first."""
    shown, size = [], 0
    for segment, text in reversed(person.notes):
        if shown and size + len(text) > limit:
            break
        shown.append(PHRASES["note"].format(segment=segment, text=text))
        size += len(text)
    return shown[::-1]


def _notes(person: T.TrackPerson, limit: int = NOTES_SHOWN, phrase: str = "notes") -> str:
    shown = _note_texts(person, limit)
    return PHRASES[phrase].format(notes=" ".join(shown)) if shown else ""


def _candidate_notes(person: T.TrackPerson) -> str:
    """A candidate's line's notes: the preparation task's description whole, then the judges' notes that fit
    `NOTES_SHOWN`."""
    shown = [*([PHRASES["prepared"].format(text=person.prepared)] if person.prepared else []), *_note_texts(person)]
    return PHRASES["notes"].format(notes=" ".join(shown)) if shown else ""


def _called(table: T.Table, key: str) -> list[str]:
    """A candidate's aliases from the lines: its own, then those of every track landing on it."""
    names = list(table.people[key].aliases)
    for other, person in table.people.items():
        if not T.candidate(other) and table.land(other) == key:
            names += person.aliases
    return list(dict.fromkeys(names))


def _span(table: T.Table, key: str) -> str:
    """Where a cut track's piece starts or ends; empty for a track never cut."""
    track = T.piece_track(key)
    pieces = table.pieces.get(track, []) if track is not None else []
    if key not in pieces:
        return ""
    after = pieces[pieces.index(key) + 1] if pieces.index(key) + 1 < len(pieces) else None
    if after is not None:
        return PHRASES["span_until"].format(segment=table.opened_in.get(after, 0), row=T.cut_of(after)[1])
    cut = T.cut_of(key)
    return PHRASES["span_from"].format(segment=table.opened_in.get(key, 0), row=cut[1]) if cut else ""


def render_table(table: T.Table) -> str:
    """The listed people as the judge reads them: the candidates' trees, then the unidentified tracks', each whose
    latest mention is latest first, never-named ones last by number; under a person, the tracks pointing at them, by
    number and indented, each with how it got there."""
    listed = set(table.listed())

    def tree(key: str) -> list[str]:
        return [key, *(member for child in table.children(key) if child in listed for member in tree(child))]

    def latest(root: str) -> tuple:  # candidates first (the prompt says so), each kind latest mention first
        last = max(table.people[member].last for member in tree(root))
        return (not T.candidate(root), *((-last, table.people[root].number) if last else (1, table.people[root].number)))

    lines: list[str] = []

    def put(key: str, depth: int) -> None:
        person, indent = table.people[key], "  " * depth
        if T.candidate(key):
            called = _called(table, key)
            lines.append(PHRASES["candidate"].format(
                indent=indent, number=person.number, name=person.name,
                others=PHRASES["others"].format(others="、".join(person.others)) if person.others else "",
                aliases=PHRASES["aliases"].format(aliases="、".join(called)) if called else "",
                notes=_candidate_notes(person)))
        else:
            parent = table.point.get(key, key)
            why = PHRASES["why"].format(why=table.why[key]) if table.why.get(key) else ""
            via = PHRASES["via"].format(handle=table.handle(table.via[key])) if table.via.get(key) else ""
            how = (PHRASES["unknown"] if parent == key
                   else PHRASES["named" if T.candidate(parent) else "linked"].format(
                       segment=table.since.get(key, 0), via=via, why=why))
            lines.append(PHRASES["track"].format(
                indent=indent, handle=table.handle(key), span=_span(table, key), how=how,
                aliases=PHRASES["track_aliases"].format(aliases="、".join(person.aliases)) if person.aliases else "",
                notes=_notes(person)))
        for child in table.children(key):
            if child in listed:
                put(child, depth + 1)

    roots = [key for key in table.people if key in listed and table.point.get(key, key) in (key, None)]
    for root in sorted(roots, key=latest):
        put(root, 0)
    return "\n".join(lines) or PHRASES["no_people"]


def table_section(table: T.Table, segment: int, prepared: bool = False) -> str:
    """The table's section of the unit's prompt; `prepared` its header that says who wrote the descriptions."""
    return load_prompt_template(PREPARED_TABLE_TEMPLATE if prepared else TABLE_TEMPLATE, strict=True, segment=segment,
                                table=render_table(table)).rstrip("\n")


def pickable(table: T.Table) -> dict[int, str]:
    """The numbers a row may answer (a new track for that candidate): the candidates -> their key."""
    return {person.number: key for key, person in table.people.items() if T.candidate(key)}


def status(table: T.Table, key: str) -> tuple[str, str]:
    """What the units before identified a track's person as: (short, for a move's destination: `认作 p1 Ema`,
    `连到 s5`, empty while it is no one's; long, for its profile line and its voice pairs: `前面认作 p1 Ema（第2段：
    q65 被叫「エマ」）`, `还没认出`; with what the table recorded on it, `还没认出；人物表里记着它被称为 氷上，〔第2段〕
    自称「私」`; a linked track with what the table recorded on it and on the last track of its links:
    `前面连到 s1（…），s1 前面认作 p2 不死途；s3 自己在人物表里记着被称为 シナズさん；s1 在人物表里记着被称为 兄貴`)."""
    point = table.point.get(key, key)
    clues = _clues(table, key)
    if point == key:
        return "", PHRASES["unknown_noted"].format(clues=clues) if clues else PHRASES["unknown"]
    why = PHRASES["profile_why"].format(why=table.why[key]) if table.why.get(key) else ""
    via = PHRASES["via"].format(handle=table.handle(table.via[key])) if table.via.get(key) else ""
    if T.candidate(point):
        name = display(table, point)
        return (PHRASES["short_named"].format(name=name),
                PHRASES["landed"].format(via=via, name=name, segment=table.since.get(key, 0), why=why))
    end = table.land(key)
    further = ("" if end == point else
               PHRASES["further_named"].format(handle=table.handle(point), name=display(table, end)) if T.candidate(end)
               else PHRASES["further_linked"].format(handle=table.handle(point), other=table.handle(end)))
    last = _last_track(table, key)
    own = "linked_own" if T.candidate(end) else "linked_noted"
    noted = ((PHRASES[own].format(handle=table.handle(key), clues=clues) if clues else "")
             + (PHRASES["linked_end"].format(handle=table.handle(last), clues=far)
                if last != key and (far := _clues(table, last)) else ""))
    return (PHRASES["short_named"].format(name=display(table, end)) if T.candidate(end)
            else PHRASES["short_linked"].format(handle=table.handle(end)),
            PHRASES["joined"].format(via=via, handle=table.handle(point), segment=table.since.get(key, 0), why=why,
                                     further=further) + noted)


def _clues(table: T.Table, key: str) -> str:
    """What the table recorded on a track: its aliases and its latest notes, empty when nothing."""
    person = table.people[key]
    return PHRASES["clue_join"].join(
        [*([PHRASES["clue_aliases"].format(aliases="、".join(person.aliases))] if person.aliases else []),
         *([notes] if (notes := _notes(person, PROFILE_NOTES_SHOWN, "clue_notes")) else [])])


def _last_track(table: T.Table, key: str) -> str:
    """The last track a track's links lead to: where they end, or, when they end at a candidate, the track
    identified as that candidate on the way (the walk of `Table.land`)."""
    seen = {key}
    while table.point.get(key, key) != key and not T.candidate(after := table.point[key]) and after not in seen:
        key = after
        seen.add(key)
    return key


def brief(table: T.Table, key: str) -> str:
    """What the units before identified an off-stage track as, for its voice pairs: `前面认作 p1 Ema`, `前面连到
    s5`, `还没认出`."""
    short = status(table, key)[0]
    return PHRASES["brief"].format(short=short) if short else PHRASES["unknown"]


def profile_names(table: T.Table, keys: Mapping[int, str]) -> dict[int, tuple[str, str]]:
    """Per track (as the file numbers it now) -> its `status`: the profile lines and the moves' destinations
    (`prompt.profiles(people=True)`). A `@pN` track's key is its candidate: it is that person."""
    return {track: (PHRASES["short_named"].format(name=display(table, key)), "") if T.candidate(key)
            else status(table, key) for track, key in keys.items() if key in table.people}


def swap_people(text: str, numbers: set[int]) -> tuple[str, list[int]]:
    """A reply with each row's `"track":"pN"` (N among `numbers`) written as the new track `"@pN"`, and the numbers
    swapped. Any other `pN` is left as it is and refused as any other wrong answer. The agent's validator reads
    replies through it too, so the two never disagree."""
    swapped: list[int] = []

    def swap(found: re.Match) -> str:
        if int(found.group(2)) not in numbers:
            return found.group(0)
        swapped.append(int(found.group(2)))
        return f"{found.group(1)}@p{found.group(2)}{found.group(3)}"

    return _ANSWER_PN.sub(swap, text or ""), swapped


def unswap_people(text: str, numbers: set[int]) -> str:
    """`text` with each row's `"track":"@pN"` that `swap_people` made (N among `numbers`) written back as `pN`."""
    if not numbers:
        return text
    return _ANSWER_AT_PN.sub(lambda found: found.group(0).replace("@p", "p", 1) if int(found.group(1)) in numbers
                             else found.group(0), text)


def answers_with_people(text: str, table: T.Table) -> tuple[str, dict[str, str]]:
    """A reply's `<judge>` rows with each `"track":"pN"` (`pickable`) read as a new track labelled `pN`
    (`swap_people`), and the candidate each `@pN` track is: for every row on `@pN`, so a row the model itself wrote
    `@p3`, or one a repair's base already swapped, is that person's too."""
    picks = pickable(table)
    text, _ = swap_people(text, set(picks))
    return text, {f"@p{number}": picks[number] for number in sorted({int(found) for found in _ANSWER_AT_PN.findall(text)})
                  if number in picks}


def _names(table: T.Table, key: str | None) -> list[str]:
    """A candidate's names, which alone are no evidence; none for anyone else."""
    if key is None or key == "?" or not T.candidate(key):
        return []
    person = table.people[key]
    return [person.name, *person.others, *person.aliases]


def _evidence(rest: str, names: Sequence[str]) -> str:
    """The evidence after a handle: what its brackets hold (nested ones and 「」 kept), else the rest of the line
    without a candidate's name written after its number; a candidate's name or other name alone is none."""
    if "(" in rest:
        inside = rest[rest.index("(") + 1:]
        found = (inside[:inside.rindex(")")] if ")" in inside else inside).strip()
    else:
        found = rest.strip()
        for name in names:
            if name and found.startswith(name):
                found = found[len(name):]
                break
        found = found.strip(" :-—")
    return "" if _nfkc(found).strip("「」『』\"'") in {_nfkc(name) for name in names} else found


def read_people(names: Mapping[str, str], body: str | None, table: T.Table, unit_id: str,
                implied: Mapping[str, str], labels: Sequence[str], segment: int, *,
                rows_on: Mapping[int, Sequence[str]],
                guard: Mapping[int, Collection[str]] | None = None,
                voice_cuts: Mapping[int, str] | None = None) -> dict[str, Any]:
    """What a unit wrote about people, checked against the table it saw (`table`, its tracks on stage): the sidecar
    fields `opened_people`, `writes`, `persons`, `people` (the ops), `mentions`, `people_seen`, `people_ignored`.
    `names` are the `<tracks>` lines (`answers.parse_tracks`), `body` the `<people>` block (None: there is none),
    `implied` the `@pN` tracks `answers_with_people` opened, `labels` the unit's other `@` labels, `segment` the
    unit's own (a `第N段@甲` of its own is its `@甲`).

    A side is a track on stage, a listed track (`Table.listed`), one of the unit's `@` labels, or (on the right) a
    candidate or `?`. A write is resolved as it is read: `= sK` records where sK lands now, the unit's own writes
    before it included, and the track itself as `via` when that is somewhere else; `?`, or the track itself, takes
    it back to itself. A candidate or a link without evidence is set aside.

    `rows_on` (each transcription track on stage -> its ▶ rows here, ids without `q`, in order) is where a unit
    may cut a track (`cuts`, read before every other line); `guard` (track -> candidates) forces one. `voice_cuts`
    (track -> row id, `run.voice_guards`) are the script's own cuts of tracks the reply left whole although their
    voice changed (`voice_guard`); every line this unit wrote on such a track's latest piece (`sK`, either side of a
    `<tracks>` line, the left of a `<people>` one) is dropped: it was written of the track as one person, and which
    piece it meant cannot be told. Its earlier pieces' (`sK-1`) stand."""
    listed = set(table.listed()) | table.onstage
    opened = {label: T.opened_key(unit_id, label) for label in dict.fromkeys(labels) if not _AT_PN.match(f"@{label}")}
    point = dict(table.point)
    writes: list[dict[str, str]] = []
    ops: list[dict[str, Any]] = []
    ignored: list[dict[str, str]] = []
    aliases = {key: {_nfkc(alias) for alias in table.people[key].aliases} for key in table.people}
    pieces = {track: list(keys) for track, keys in table.pieces.items()}
    cuts: list[dict[str, Any]] = []
    cut_lines: dict[str, str | None] = {}

    def side(text: str) -> tuple[str | None, str, str]:
        """(key, why not, the rest of the text) of a handle; `?` for `?`."""
        found = _HANDLE.match(text)
        if found is None:
            return None, "not a handle", ""
        back, letter, number, piece, old, at_segment, label, rest = found.group(
            "back", "letter", "number", "piece", "old", "segment", "label", "rest")
        if back:
            return "?", "", rest
        if letter == "p":
            if piece or old:
                return None, "a candidate has no pieces", rest
            key = table.numbered(int(number))
            return (key, "", rest) if key is not None and T.candidate(key) else (None, "not a candidate", rest)
        if letter == "s":
            if old:
                return None, "a cut track's earlier pieces are s3-1, s3-2... (oldest first)", rest
            keys = pieces.get(int(number)) or [T.track_key(int(number))]
            if piece is not None and not 1 <= int(piece) < len(keys):
                return None, "no such piece (or a dash before the evidence: write it in brackets)", rest
            key = keys[int(piece) - 1] if piece is not None else keys[-1]
            return (key, "", rest) if key in listed else (None, "track neither on stage nor listed", rest)
        if at_segment is None or int(at_segment) == segment:
            key = opened.get(label)
            return (key, "", rest) if key else (None, "@label not opened in <judge>", rest)
        key = next((key for key in listed if table.opened_in.get(key) == int(at_segment)
                    and T.key_label(key) == label), None)
        return (key, "", rest) if key else (None, "no such listed track", rest)

    def cut(track: int, row: str, forced: bool = False, voice: bool = False) -> str:
        key = T.cut_key(unit_id, track, row)
        pieces.setdefault(track, [T.track_key(track)]).append(key)
        listed.add(key)
        cuts.append({"track": track, "row": row, **({"forced": True} if forced else {}),
                     **({"voice_guard": True} if voice else {})})
        return key

    guarded: set[int] = set()

    def on_guarded(text: str) -> bool:
        """Whether a handle is a voice-guarded track's latest piece, written bare (`s3`)."""
        found = _HANDLE.match(_nfkc(text))
        return found is not None and found["letter"] == "s" and not found["piece"] and not found["old"] \
            and int(found["number"]) in guarded

    # the cuts first, wherever they stand, by their left sides alone
    for left, raw in names.items():
        if (found := _CUT_LEFT.match(_nfkc(left))) is None:
            continue
        track, row = int(found.group(1)), found.group(2)
        why = ("track not on stage" if track not in rows_on
               else "not a ▶ row of this track here" if row not in rows_on[track]
               else "cut twice here" if any(done["track"] == track for done in cuts) else "")
        if why:
            ignored.append({"line": f"{left} = {raw}", "why": why})
            cut_lines[left] = None
        else:
            cut_lines[left] = cut(track, row)
    for left, raw in names.items():  # a whole write against the moves before it (`guard`)
        found = _HANDLE.match(_nfkc(left))
        if found is None or found["letter"] != "s" or found["piece"] or found["old"] or "自" in found["rest"]:
            continue
        track = int(found["number"])
        if track not in rows_on or not rows_on[track] or any(done["track"] == track for done in cuts):
            continue
        to, _, rest = side(_nfkc(raw))
        if to is None or to == "?":
            continue
        end = to if T.candidate(to) else table.land(to)
        already = table.land((pieces.get(track) or [T.track_key(track)])[-1]) == end  # moves gathering it
        if T.candidate(end) and not already and end in (guard or {}).get(track, ()) \
                and _evidence(rest, _names(table, to)):
            cut(track, rows_on[track][0], forced=True)
    for track, row in (voice_cuts or {}).items():  # after the reply's own cuts, never a second one
        if track in rows_on and row in rows_on[track] and not any(done["track"] == track for done in cuts):
            cut(track, row, voice=True)
            guarded.add(track)

    for left, raw in names.items():
        text = _nfkc(left)
        if _AT_PN.match(text):  # the `@pN` tracks are their candidate already
            continue
        line = f"{left} = {raw}"
        if guarded and (on_guarded(left) or on_guarded(raw)):
            if cut_lines.get(left, "") is not None:  # a refused cut is set aside already
                ignored.append({"line": line, "why": VOICE_GUARDED + (CUT_STANDS if left in cut_lines else "")})
            continue
        stands = ""
        if left in cut_lines:
            who, why = cut_lines[left], ""
            if who is None:
                continue
            stands = CUT_STANDS
        elif "自" in text:
            ignored.append({"line": line, "why": "a cut is `sK 自qN` with sK a track on stage"})
            continue
        else:
            who, why, _ = side(text)
        if who is None or who == "?" or T.candidate(who):
            ignored.append({"line": line, "why": why or ("a candidate points at no one" if who else "not a track")})
            continue
        to, why, rest = side(_nfkc(raw))
        if to is None:
            ignored.append({"line": line, "why": why + stands})
            continue
        if stands and to == pieces[T.piece_track(who)][-2]:  # A then B then A again is a fair write; not this
            ignored.append({"line": line, "why": "the new piece is said to be the one it was cut from" + stands})
            continue
        evidence = _evidence(rest, _names(table, to))
        if to in ("?", who):
            end, via = who, ""
        elif not evidence:
            ignored.append({"line": line, "why": "no evidence" + stands})
            continue
        else:
            end = to if T.candidate(to) else table.land(to, point)
            if end == who:
                ignored.append({"line": line, "why": "already one person" + stands})
                continue
            via = to if end != to else ""
        point[who] = end
        writes.append({"who": who, "to": end, **({"via": via} if via else {}),
                       **({"why": evidence[:T.WHY_CHARS]} if evidence else {}),
                       **({"cut": len(evidence)} if len(evidence) > T.WHY_CHARS else {})})
    here = [(pieces.get(track) or [key])[-1] if (track := T.piece_track(key)) is not None else key
            for key in (*table.onstage, *opened.values())]
    for line in (body or "").splitlines():
        text = _nfkc(line)
        if not text:
            continue
        found = _LINE.match(text)
        if found is None:
            ignored.append({"line": line, "why": "does not read"})
            continue
        verb, rest = _VERB.get(found.group(2), found.group(2)), found.group(3).strip()
        if guarded and on_guarded(found.group(1)):
            ignored.append({"line": line, "why": VOICE_GUARDED})
            continue
        key, why, _ = side(found.group(1))
        if key is None or key == "?":
            ignored.append({"line": line, "why": why or "not a track"})
            continue
        if T.candidate(key) and any(table.land(other, point) == key for other in here):
            ignored.append({"line": line, "why": "on a track here: written on the track"})
            continue
        if verb == "被称为":
            alias = rest.strip("「」『』\"'")
            if not alias:
                ignored.append({"line": line, "why": "no text"})
            elif _nfkc(alias) in aliases.setdefault(key, set()):
                ignored.append({"line": line, "why": "alias already there"})
            else:
                aliases[key].add(_nfkc(alias))
                ops.append({"op": "call", "key": key, "text": alias[:T.ALIAS_CHARS],
                            **({"cut": len(alias)} if len(alias) > T.ALIAS_CHARS else {})})
        elif not rest:
            ignored.append({"line": line, "why": "no text"})
        else:
            ops.append({"op": "add", "key": key, "text": rest[:T.NOTE_CHARS],
                        **({"cut": len(rest)} if len(rest) > T.NOTE_CHARS else {})})
    mentions = list(dict.fromkeys([*(key for write in writes for key in (write["who"], write["to"], write.get("via"))
                                     if key), *implied.values(), *(op["key"] for op in ops)]))
    seen = {key: table.created.get(opener, "") for key in mentions
            if (opener := T.key_unit(key)) is not None and opener != unit_id}
    return {**({"opened_people": list(opened)} if opened else {}), **({"cuts": cuts} if cuts else {}),
            "writes": writes, "persons": dict(implied),
            "people": ops, "mentions": mentions,
            **({"people_seen": seen} if seen else {}),
            **({"people_ignored": ignored} if ignored else {})}
