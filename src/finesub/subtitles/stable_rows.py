"""The lines everything after diarization reads: `-stable.json` joined with `-diar-rows.json`. The one reader.

Each task file is written once by the step that makes it and never edited afterwards; a rerun replaces it
whole. So `-stable.json` stays exactly as stabilization wrote it, and what diarization says about each of
its lines lives beside it, in `<stem>-diar-rows.json` (`speech/diarization/stage.write_rows`):

    {"version": 1,
     "stable": "<lines_hash of the stable's segments>",
     "speaker": {<run record: model, backend, hints, marks, ...>},
     "split": {"chosen": true, "version": 1, "cuts": 3},
     "rows": [{"speaker": {...}}, {}, {"speaker": {...}, "split": {"at": 3, "speakers": [<left>, <right>]}}, ...]}

`rows` has one entry per stable line, in order. An entry's `speaker` replaces the line's (an entry without
one leaves the line unlabelled); `split` is where acoustic row splitting cut the line -- `at` is the index
of the right half's first word, each speaker has the shape of a line's own `speaker`. `split.chosen` is
the task's choice: true reads the halves, false reads the lines whole. A file whose `stable` does not
match the stable's lines (their times, `lines_hash`) was made for other lines and is not read at all
(the diarization stage writes it again).

⚠ Everything after diarization reads lines through `read_rows` (or `rows` with `read_overlay`) -- the
LLM stage, the speaker judge's checks, the `spk` column, the speaker exports, the raw SRT. A reader that
takes the stable's `segments` instead misses the speakers, sees one line fewer per cut, and every id after
the first cut points at the wrong line without any error (`test_stable_rows.py` guards the readers it knows).
The readers after the LLM speaker judge (the correction, the `spk` column, the speaker exports) take these
lines with what the judge heard applied (`speech.diarization.product`, built on `read_rows`).

No speech or LLM dependency, so the light readers (`llm/chunking.py`) can import it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..text import words_to_text

OVERLAY_SUFFIX = "-diar-rows.json"
OVERLAY_VERSION = 1
SPLIT_KEY = "split"


def overlay_path(stable_json: str | Path) -> Path:
    """`<stem>-diar-rows.json` beside `<stem>-stable.json`."""
    stable = Path(stable_json)
    return stable.with_name(f"{stable.name.removesuffix('.json').removesuffix('-stable')}{OVERLAY_SUFFIX}")


def lines_hash(segments: Sequence[Any]) -> str:
    """What `-diar-rows.json` was made for: each line's times and its words' times, which the labels and the
    cuts are computed from. A stable transcribed again changes them; a hand-fixed word in a line's text does
    not, and keeps the labels (and any name a person wrote into them). The speaker judge's `rows_hash` and
    the correction's fingerprint do count the text, so such a fix still has the judge and the correction
    redo their windows."""
    def times(segment: Any) -> Any:
        if not isinstance(segment, Mapping):
            return segment
        words = segment.get("words") or []
        return [segment.get("start"), segment.get("end"),
                [[word.get("start"), word.get("end")] if isinstance(word, Mapping) else word for word in words]]

    text = json.dumps([times(segment) for segment in segments], ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def overlay_for(segments: Sequence[Any], entries: Sequence[Mapping[str, Any]], *,
                speaker: Mapping[str, Any] | None = None, split: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A `-diar-rows.json` payload for `segments`: one entry per line, the run record, the split record."""
    return {"version": OVERLAY_VERSION, "stable": lines_hash(segments),
            "speaker": None if speaker is None else dict(speaker),
            "split": dict(split) if split is not None else {"chosen": False},
            "rows": [dict(entry) for entry in entries]}


def write_overlay(stable_json: str | Path, overlay: Mapping[str, Any]) -> Path:
    """Write `overlay` beside the stable, whole (through a temporary file, so a reader never sees half)."""
    destination = overlay_path(stable_json)
    temporary = destination.with_name(f".{destination.name}.part")
    temporary.write_text(json.dumps(overlay, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def current(overlay: Any, segments: Any) -> bool:
    """Whether `overlay` was made for exactly these lines."""
    return (isinstance(overlay, Mapping) and overlay.get("version") == OVERLAY_VERSION
            and isinstance(segments, list) and isinstance(overlay.get("rows"), list)
            and len(overlay["rows"]) == len(segments) and overlay.get("stable") == lines_hash(segments))


#: Unreadable rows files already warned about in this process: every reader of a run would say it again.
_WARNED: set[Path] = set()


def unreadable(path: Path) -> str | None:
    """Why the file at `path` cannot be parsed as JSON, or None when it can (or is absent)."""
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        return str(error)
    return None


def stored_overlay(stable_json: str | Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The `-diar-rows.json` on disk made for this stable's lines, or None (absent, unreadable, or for other
    lines). `payload` is the stable, for a caller that already parsed it.

    A file made for other lines is expected (the stable was transcribed again) and passes silently; one
    that does not parse says so -- it may hold names a person wrote, and every reader would otherwise
    quietly go back to lines without speakers."""
    path = overlay_path(stable_json)
    if not path.exists():
        return None
    try:
        overlay = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        if path not in _WARNED:
            _WARNED.add(path)
            print(f"Warning: {path.name} unreadable ({error}); speaker labels and row splits ignored",  # product output
                  file=sys.stderr)
        return None
    try:
        if payload is None:
            payload = json.loads(Path(stable_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    segments = payload.get("segments") if isinstance(payload, Mapping) else None
    return overlay if current(overlay, segments) else None


def legacy_overlay(stable_json: str | Path, payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """⚠ Compatibility, to delete at 1.0: a stable labelled before `-diar-rows.json` existed (0.6.0) carries
    its labels on its own segments and its run record in `metadata.speaker`. Read as the rows file they
    would have been -- labels and marks as they are, no cuts, not chosen -- until the diarization stage
    writes them to disk with the row splitting decided (`stage.seed_legacy`, owner 2026-10-15: carried
    over, never recomputed).
    None for a stable no line of which has a `speaker`, and for one whose labels came from another timeline
    than the `-diar.json` beside it (computed again since: its labels belong to a partition that is gone;
    0.6.0 copied the timeline's `hints` into the record, which is how the two are matched)."""
    segments = payload.get("segments") if isinstance(payload, Mapping) else None
    if not isinstance(segments, list) or not any(isinstance(s, Mapping) and "speaker" in s for s in segments):
        return None
    record = (payload.get("metadata") or {}).get("speaker")
    stable = Path(stable_json)
    timeline = stable.with_name(f"{stable.name.removesuffix('.json').removesuffix('-stable')}-diar.json")
    if timeline.exists():
        try:
            hints = (json.loads(timeline.read_text(encoding="utf-8")).get("metadata") or {}).get("hints")
        except (OSError, ValueError, AttributeError):
            return None
        if not isinstance(record, Mapping) or record.get("hints") != hints:
            return None
    entries = [{"speaker": s["speaker"]} if isinstance(s, Mapping) and "speaker" in s else {} for s in segments]
    return overlay_for(segments, entries, speaker=record if isinstance(record, Mapping) else None,
                       split={"chosen": False, "cuts": 0})


def read_overlay(stable_json: str | Path, payload: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """What the readers apply to this stable's lines: its `-diar-rows.json`, else (compatibility, only when no
    such file exists at all) the labels an old stable carries itself, else None. An unreadable or stale file
    never falls back to the old labels: they may count other lines than the products made since."""
    if payload is None:
        try:
            payload = json.loads(Path(stable_json).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    if overlay_path(stable_json).exists():
        return stored_overlay(stable_json, payload)
    return legacy_overlay(stable_json, payload)


def split_chosen(overlay: Mapping[str, Any] | None) -> bool:
    """The task's choice recorded in `overlay`: True reads the halves of every cut line."""
    split = overlay.get("split") if isinstance(overlay, Mapping) else None
    return isinstance(split, Mapping) and split.get("chosen") is True


def halves(segment: Mapping[str, Any], at: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Line `segment` cut before word `at`: each half keeps the line's fields with its own words, text and
    time. Each half's text is rebuilt from its words, so a hand fix to the cut line's `text` alone does not
    reach the halves. `id` and `alignment_events` stay with the line (events are evidence about the decode, and a
    positional id would repeat); `speaker` is left as the line's, for the caller to replace."""
    words = segment["words"]
    line = {key: value for key, value in segment.items() if key not in ("id", "alignment_events")}
    left, right = copy.deepcopy(line), copy.deepcopy(line)
    left["words"], right["words"] = copy.deepcopy(words[:at]), copy.deepcopy(words[at:])
    left["text"], right["text"] = words_to_text(left["words"]), words_to_text(right["words"])
    left["end"] = float(words[at - 1]["end"])
    right["start"] = float(words[at]["start"])
    return left, right


def _labelled(segment: Any, speaker: Any) -> Any:
    """`segment` with `speaker` as its speaker (None: unlabelled)."""
    line = {key: value for key, value in segment.items() if key != "speaker"}
    if speaker is not None:
        line["speaker"] = copy.deepcopy(speaker)
    return line


def rows(payload: Mapping[str, Any] | Sequence[Any], overlay: Mapping[str, Any] | None) -> list[Any] | Any:
    """The lines to read: the stable's lines with `overlay`'s speakers, and the halves of every cut line
    when the task chose them. Without an overlay made for these lines, the lines with no speakers: labels
    come only from a rows file (`read_overlay`), never from a `speaker` a stable itself still carries. `overlay`
    has no default on purpose: a caller that leaves it out reads no speakers and no halves, silently. A bare list
    of segments is read as it is. A payload whose `segments` is missing or not a list is no stable: that
    value comes back unchanged (usually None), for the caller's own check to reject."""
    if not isinstance(payload, Mapping):
        return list(payload)
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return segments  # not a stable: the caller's own check says so
    if not current(overlay, segments):
        return [_labelled(segment, None) if isinstance(segment, Mapping) else segment for segment in segments]
    entries = overlay["rows"]
    chosen = split_chosen(overlay) and any(isinstance(entry, Mapping) and SPLIT_KEY in entry for entry in entries)
    out: list[Any] = []
    for segment, entry in zip(segments, entries):
        if not isinstance(segment, Mapping):
            out.append(segment)
            continue
        entry = entry if isinstance(entry, Mapping) else {}
        if chosen:
            # Halves have no id of their own, so explicit ids would repeat or skip: all go positional.
            segment = {key: value for key, value in segment.items() if key != "id"}
        split = entry.get(SPLIT_KEY) if chosen else None
        if split is None:
            out.append(_labelled(segment, entry.get("speaker")))
            continue
        out += [_labelled(half, speaker)
                for half, speaker in zip(halves(segment, int(split["at"])), split["speakers"])]
    return out


def read_rows(stable_json: str | Path) -> list[Any] | Any:
    """`rows` of the stable at `stable_json`, with its `-diar-rows.json` when one was made for its lines."""
    payload = json.loads(Path(stable_json).read_text(encoding="utf-8"))
    return rows(payload, read_overlay(stable_json, payload))
