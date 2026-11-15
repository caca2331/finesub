"""Reading the judge's replies, refusing anything that is not an answer, and telling a cut-off one.

A refused reply is repaired (agy's answers came back malformed about one time in five in the
experiments): `survey` reads the `<judge>` block line by line and lists every reason it does not answer,
and the rows that do, so the repair round asks only for the rest (`patch.py`); the strict parsers below
refuse with the same reasons. Only what the sidecar needs survives: results, never `keep` -- a word that
only means something inside this prompt (the sidecar: `docs/llm_harness_behavior.md`「说话人裁判」).
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from finesub import jsonl

from ....speech.diarization.judgments import ACT_DROP, ACTS, CHARS_PER_SECOND, acoustic_main
from .prompt import Row
from .windows import JudgeWindow

BLOCK = r"<{0}>(.*?)</{0}>"
_COMPOUND = re.compile(r"s(\d+)(?:\+s\d+)*")


class AnswerError(ValueError):
    """The reply did not answer what was asked; ``errors`` lists every reason found."""

    def __init__(self, *errors: str) -> None:
        super().__init__("; ".join(errors))
        self.errors = tuple(errors)


@dataclass(frozen=True)
class Verdict:
    kind: str  # keep / track / new / abstain
    why: str = ""
    track: int | None = None
    label: str | None = None


def _block(text: str, name: str) -> str:
    match = re.search(BLOCK.format(name), text or "", re.S)
    if match is None:
        raise AnswerError(f"reply has no <{name}> block")
    return match.group(1)


def _verdict(payload: Mapping[str, Any], known: set[int]) -> Verdict | None:
    """A row's `track` answer, or None for one that is not among the answers. A copied cell counts as its main
    track (`s5+s4` is s5, `??` is keep); a track the profiles do not list is refused -- the answer can only
    name those."""
    answer, why = str(payload.get("track", "")).strip(), str(payload.get("why", ""))
    if answer in ("keep", "??"):
        return Verdict("keep", why)
    if answer == "abstain":
        return Verdict("abstain", why)
    if answer.startswith("@") and answer[1:].strip():
        return Verdict("new", why, label=answer[1:].strip())
    if (match := _COMPOUND.fullmatch(answer)) is not None and int(match.group(1)) in known:
        return Verdict("track", why, track=int(match.group(1)))
    return None


_ID_IN_LINE = re.compile(r'"id"\s*:\s*"([^"]+)"')


def line_id(line: str) -> str | None:
    """The row id a line names: its JSON `id`, or for a line that is not JSON, the `"id": "..."` written in it."""
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        match = _ID_IN_LINE.search(line)
        return match.group(1) if match else None
    return str(payload["id"]) if isinstance(payload, Mapping) and "id" in payload else None


#: How many rows one error line names before it says how many more: a reply that went wrong everywhere would
#: otherwise name the whole window, and the reply itself is in front of the model anyway.
LISTED = 10


def listed(items: Sequence[str]) -> str:
    """``items`` joined, the first `LISTED` of them and how many more."""
    shown = ", ".join(items[:LISTED])
    return shown if len(items) <= LISTED else f"{shown} and {len(items) - LISTED} more"


def collapsed(kind: tuple[str, str], items: Sequence[str]) -> str:
    """One error line for every row with the same fault: `"{n} {kind}: {rows}"`, ``kind`` as `(one, several)`."""
    return f"{len(items)} {kind[len(items) != 1]}: {listed(items)}"


#: The faults a `<judge>` row can have, each reported once for all its rows (`collapsed`).
NOT_JSON = ("row is not JSON", "rows are not JSON")
NOT_OBJECT = ("row is not a JSON object", "rows are not JSON objects")
UNEXPECTED = ("row names an id that is not judged here", "rows name ids that are not judged here")
REPEATED = ("id is written more than once", "ids are written more than once")
NOT_AN_ANSWER = ("row's track is not one of the answers", "rows' tracks are not among the answers")
FAULTS = (NOT_JSON, NOT_OBJECT, UNEXPECTED, REPEATED, NOT_AN_ANSWER)  # the order the errors are listed in


@dataclass(frozen=True)
class Survey:
    """A reply's `<judge>` block read line by line. ``rows`` are the rows that answer (by id, in the reply's
    order) and ``lines`` their text, plus the text of each faulty row whose id can be told (an id written twice:
    its first line that is not JSON or not an object, else its first line -- the line the errors speak of);
    ``faulty`` the ids written but not answered (a row that is not JSON or not an object, given twice, or --
    with tracks -- whose `track` is not among the answers), ``missing`` the ids no row names (with tracks: every
    row must be answered), ``untold`` the faulty rows no judged id can be told for (as `line N` of the block),
    ``errors`` every reason the reply does not read, one line per kind of fault naming its rows (by id, or
    `line N`)."""

    found: bool
    rows: dict[str, Mapping[str, Any]]
    lines: dict[str, str]
    faulty: tuple[str, ...]
    missing: tuple[str, ...]
    untold: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def pending(self) -> tuple[str, ...]:
        """The ids still to answer, faulty or missing (in no particular order; callers sort by the window)."""
        return (*self.faulty, *self.missing)


def survey(text: str, ids: Sequence[str], tracks: Sequence[int] | None) -> Survey:
    """Read the `<judge>` block of ``text`` for the judged ``ids``; ``tracks`` the ones the answer may name, or
    None to read the rows alone (`_rows`, on a reply `parse_judge` already read: `track` and rows left out are
    not checked)."""
    try:
        body = _block(text, "judge")
    except AnswerError as error:
        return Survey(False, {}, {}, (), (), (), error.errors)
    wanted, known = set(ids), set(tracks or ())
    rows: dict[str, Mapping[str, Any]] = {}
    lines: dict[str, str] = {}
    faulty: dict[str, None] = {}
    seen: set[str] = set()
    broken: set[str] = set()  # ids with a line that is not JSON or not an object: that line is the one shown
    untold: list[str] = []
    faults: dict[tuple[str, str], list[str]] = {kind: [] for kind in FAULTS}
    for number, line in enumerate((line.strip() for line in jsonl.lines(body) if line.strip()), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload, kind = None, NOT_JSON
        else:
            kind = None if isinstance(payload, Mapping) else NOT_OBJECT
        if kind is not None:
            key = line_id(line)
            if key not in wanted:
                faults[kind].append(f"line {number}")
                untold.append(f"line {number}")
                continue
            faults[kind].append(key)
            if key in seen and key not in faults[REPEATED]:  # a row for this id all the same
                faults[REPEATED].append(key)
            seen.add(key)
            faulty.setdefault(key, None)
            if key not in broken:
                broken.add(key)
                lines[key] = line
            continue
        key = str(payload.get("id", ""))
        if key not in wanted:
            faults[UNEXPECTED].append(repr(key))
            continue
        if key in seen:
            if key not in faults[REPEATED]:
                faults[REPEATED].append(key)
            faulty[key] = None
            continue
        seen.add(key)
        lines[key] = line
        if tracks is not None and _verdict(payload, known) is None:
            faults[NOT_AN_ANSWER].append(f"{key} ({str(payload.get('track', '')).strip()!r})")
            faulty[key] = None
            continue
        rows[key] = payload
    rows = {key: payload for key, payload in rows.items() if key not in faulty}
    missing = tuple(key for key in ids if key not in rows and key not in faulty) if tracks is not None else ()
    errors = [collapsed(kind, items) for kind, items in faults.items() if items]
    if missing:
        errors.append(f"reply misses {len(missing)} of {len(ids)} rows: {listed(missing)}")
    return Survey(True, rows, lines, tuple(faulty), missing, tuple(untold), tuple(errors))


def parse_judge(text: str, ids: Sequence[str], tracks: Sequence[int]) -> dict[str, Verdict]:
    """`{id: verdict}` covering exactly `ids` (`_verdict` reads each); raises `AnswerError` with every reason
    `survey` found."""
    read = survey(text, ids, tracks)
    if read.errors:
        raise AnswerError(*read.errors)
    known = set(tracks)
    return {key: _verdict(payload, known) for key, payload in read.rows.items()}  # type: ignore[misc]


def _rows(text: str, ids: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    """The `<judge>` block's JSON rows by id: each an object, for a judged row, at most once."""
    read = survey(text, ids, None)
    if read.errors:
        raise AnswerError(*read.errors)
    return read.rows


@dataclass(frozen=True)
class Acts:
    """Pass 1's shape answers for one unit, by row id (not the prompt's `q` id): `{row: {"act", "why"}}`,
    and the acts set aside (`{"row", "act", "why"}`): a join or may_join on the unit's first row, which
    has no line before it in the unit, or a row naming more than one act."""

    acts: dict[str, dict[str, str]]
    ignored: list[dict[str, str]]


def parse_acts(text: str, ids: Sequence[str]) -> Acts:
    """The shape fields of a reply whose rows `parse_judge` read: `join` / `may_join` / `drop`, each `true` to
    count. A row may answer none. `ids` are the unit's judged `q` ids in order, the first being the unit's first
    row."""
    acts: dict[str, dict[str, str]] = {}
    ignored: list[dict[str, str]] = []
    for key, payload in _rows(text, ids).items():
        named = [act for act in ACTS if payload.get(act) is True]
        if not named:
            continue
        row, why = key.removeprefix("q"), str(payload.get("why", ""))
        if len(named) > 1:
            ignored.append({"row": row, "act": "+".join(named), "why": "several acts"})
        elif key == ids[0] and named[0] != ACT_DROP:
            ignored.append({"row": row, "act": named[0], "why": "first row"})
        else:
            acts[row] = {"act": named[0], "why": why}
    return Acts(acts, ignored)


@dataclass(frozen=True)
class Split:
    """An audio judge's `split` on one row: the row's text up to where the speaker changes, and who speaks
    after it (a `track` or a `new` verdict; `track` answers who speaks before it)."""

    prefix: str
    second: Verdict


def _speaker(answer: Any, known: set[int]) -> Verdict | None:
    """Who a heard edit names: a track the prompt shows or a new `@label`; None for anything else (`keep` and
    `abstain` included: there is no row of its own to keep)."""
    answer = str(answer or "").strip()
    if answer.startswith("@") and answer[1:].strip():
        return Verdict("new", label=answer[1:].strip())
    if (match := _COMPOUND.fullmatch(answer)) is not None and int(match.group(1)) in known:
        return Verdict("track", track=int(match.group(1)))
    return None


def parse_splits(text: str, ids: Sequence[str], tracks: Sequence[int]) -> tuple[dict[str, Split], list[dict[str, str]]]:
    """The `split` fields, by row id, and the ones set aside: a `track2` that names no track the prompt
    shows and no new one (`keep` and `abstain` included). A bad split does not refuse the reply -- the
    row stays whole."""
    known, out, ignored = set(tracks), {}, []
    for key, payload in _rows(text, ids).items():
        prefix = payload.get("split")
        if not isinstance(prefix, str) or not prefix.strip():
            continue
        row, second = key.removeprefix("q"), _speaker(payload.get("track2"), known)
        if second is None:
            ignored.append({"row": row, "act": "split", "why": f"track2 {payload.get('track2')!r}"})
            continue
        out[row] = Split(prefix.strip(), second)
    return out, ignored


def parse_fills(text: str, ids: Sequence[str]) -> dict[str, str]:
    """The `fill` fields, by row id: the row's text as the judge heard it whole (checked where it is placed)."""
    return {key.removeprefix("q"): payload["fill"].strip() for key, payload in _rows(text, ids).items()
            if isinstance(payload.get("fill"), str) and payload["fill"].strip()}


@dataclass(frozen=True)
class Insert:
    """An audio judge's `insert` after one row: what it heard missing, when (on the audio file's timeline, as
    the prompt shows the rows; `end` None when not given) and who said it."""

    start: float
    end: float | None
    text: str
    who: Verdict


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def parse_inserts(text: str, ids: Sequence[str], tracks: Sequence[int]) -> tuple[dict[str, Insert], list[dict[str, str]]]:
    """The `insert` fields, by the row they follow, and the ones set aside: not an object, no start, no text,
    or a `track` that names no track the prompt shows and no new one. As with `split`, the reply stands."""
    known, out, ignored = set(tracks), {}, []
    for key, payload in _rows(text, ids).items():
        if payload.get("insert") in (None, False, ""):
            continue
        row, insert = key.removeprefix("q"), payload["insert"]
        if not isinstance(insert, Mapping):
            ignored.append({"row": row, "act": "insert", "why": "not an object"})
            continue
        start, words, who = _seconds(insert.get("start")), str(insert.get("text") or "").strip(), \
            _speaker(insert.get("track"), known)
        why = "no start" if start is None else "no text" if not words else \
            f"track {insert.get('track')!r}" if who is None else ""
        if why:
            ignored.append({"row": row, "act": "insert", "why": why})
            continue
        out[row] = Insert(start, _seconds(insert.get("end")), words, who)
    return out, ignored


def _plain(text: str) -> str:
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text)).casefold()


#: An insert's length when the judge gives no end: its text read at speech speed, within these bounds.
INSERT_SECONDS = (0.8, 5.0)


def spoken_seconds(text: str) -> float:
    return min(max(len(_plain(text)) / CHARS_PER_SECOND, INSERT_SECONDS[0]), INSERT_SECONDS[1])


def fill_ends(row_text: str, filled: str) -> tuple[tuple[int, int] | None, str]:
    """`((characters added at the head, at the tail), "")`, or `(None, why)`: the row's own text must stand
    whole inside `filled` and something must be added around it, both compared as `split_at` compares."""
    whole, own = _plain(filled), _plain(row_text)
    at = whole.find(own) if own else -1
    if at < 0:
        return None, "changes the row's own text"
    if len(whole) == len(own):
        return None, "adds nothing"
    return (at, len(whole) - at - len(own)), ""


def split_at(segment: Mapping[str, Any], prefix: str) -> tuple[int | None, str]:
    """`(index of the word the second half starts with, "")`, or `(None, why)`: the word boundary nearest
    to where `prefix` ends in the row's words, both compared without spaces and punctuation and in one
    form (NFKC, case folded); `prefix` must open the row and leave something after it, and every word
    must have its time (the halves take their times from the words)."""
    words = segment.get("words") or []
    if any(word.get("start") is None or word.get("end") is None for word in words):
        return None, "a word has no time"
    whole, target = _plain("".join(str(word.get("word", "")) for word in words)), _plain(prefix)
    if not target or not whole.startswith(target):
        return None, "prefix does not open the row"
    if len(target) >= len(whole):
        return None, "prefix is the whole row"
    bounds, seen = [], 0
    for index, word in enumerate(words[:-1], start=1):
        seen += len(_plain(str(word.get("word", ""))))
        bounds.append((abs(seen - len(target)), index))
    return (min(bounds)[1], "") if bounds else (None, "one word")


def answer_fields(verdict: Verdict, carried: Mapping[int, tuple[str, str, str]]) -> dict[str, Any]:
    """How the sidecar names the track a verdict puts a row (or a half) on: `label` for a new one,
    `carried` for one an earlier unit opened, else `track`."""
    if verdict.kind == "new":
        return {"label": verdict.label}
    if verdict.track in carried:
        window_id, label, of = carried[verdict.track]
        return {"carried": {"window": window_id, "label": label, "of": of}}
    return {"track": verdict.track}


def parse_tracks(text: str) -> dict[str, str]:
    """The `<tracks>` block's `label = name` lines; fail-soft (names only go to the sidecar)."""
    try:
        body = _block(text, "tracks")
    except AnswerError:
        return {}
    pairs = (line.split("=", 1) for line in body.splitlines() if "=" in line)
    return {key.strip(): value.strip() for key, value in pairs if key.strip()}


def cut_off(text: str) -> bool:
    """A reply that opened `<judge>` and never closed it: what the output ceiling leaves when usage
    cannot say so (the correction's open `<translated>`)."""
    opened = (text or "").find("<judge>")
    return opened >= 0 and "</judge>" not in text[opened:]


def results(rows: Sequence[Row], window: JudgeWindow, verdicts: Mapping[str, Verdict],
            carried: Mapping[int, tuple[str, str, str]] | None = None) -> list[dict[str, Any]]:
    """The sidecar's per-row results for one window, in row order: what changed and the abstentions.
    `from` is the row's acoustic main track when judged, so a reader can tell a verdict on another
    labelling apart; naming the track a row already leads (or the one its mark filled) is a keep.
    A carried track (`carried`: its number -> the window, label and verdict that opened it) is written as
    those, never as the number, which depends on what the earlier windows hold."""
    carried = carried or {}
    out = []
    for row in rows[window.low:window.high]:
        verdict = verdicts[f"q{row.id}"]
        origin = acoustic_main(row.speaker)
        base: dict[str, Any] = {"row": row.id, "from": origin, "why": verdict.why}
        current = origin["filled"] if isinstance(origin, dict) else origin
        if verdict.kind == "abstain":
            out.append({**base, "abstain": True})
        elif verdict.kind == "new" or verdict.track in carried or (verdict.kind == "track" and verdict.track != current):
            out.append({**base, **answer_fields(verdict, carried)})
    return out
