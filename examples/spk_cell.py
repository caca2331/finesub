"""Parse the `spk` cell of finesub's `-annotated.csv`: a reference to copy, with no dependencies.

    python spk_cell.py out/<name>/<name>-annotated.csv     # one JSON object per subtitle line

What the cell means is in docs/manual/diarization.md, section "读 `spk` 列". Its grammar, exactly
(no whitespace but single ASCII spaces inside a name):

    cell    = "" | side | side "<" [ side ]     left of "<": after the speaker judge; right: before it
    side    = filled | heard
    filled  = track "*" { "+" track "*" }        tracks only a confidence mark guessed
    heard   = lead { "+" member }                lead first; the rest by time held, most first
                                                 (first seen instead when a name is on the line)
    lead    = track [ "?" [ "(" track ")" ] | "~" ] | name
    member  = track | name
    track   = "s" digit { digit }
    name    = characters other than  + < ? ~ ( ) * | tab, line break; not starting or ending with a
              space; never of the form "s" digits

`?` is a low confidence mark, with the acoustic layer's runner-up track in brackets when it has one;
`~` is "two tracks cover the line almost equally" (a run without confidence marks). `(` only ever
follows `?`, and `*` never shares a side with the other marks. A name stands where a person wrote one
onto a label, in place of that source's track id and without flags; finesub shows any of the reserved
characters inside it as full-width look-alikes (`+` as `＋`), so a member that is not `s<number>` is a name.

`parse` raises ValueError on anything outside the grammar rather than guessing: a cell this file
does not understand means finesub changed the format, and reading it silently wrong is the failure
this file exists to prevent.
"""

from __future__ import annotations

import json
import re
import sys

TRACK = re.compile(r"s[0-9]+")
_MEMBER = r"[^+<?~()*|\s](?:(?:[^+<?~()*|\s]| )*[^+<?~()*|\s])?"
NAME = re.compile(_MEMBER)
LEAD = re.compile(rf"(?P<member>{_MEMBER})(?:(?P<low>\?)(?:\((?P<alt>s[0-9]+)\))?|(?P<close>~))?")


def parse_side(text: str) -> dict:
    """One side of a cell: `{"tracks": [...], "flag": None | "low" | "close" | "filled", "alternative": str | None}`.

    An empty side is a line nobody was assigned to: no tracks, no flag."""
    if not text:
        return {"tracks": [], "flag": None, "alternative": None}
    parts = text.split("+")
    if all(part.endswith("*") for part in parts):
        tracks = [part[:-1] for part in parts]
        if not all(TRACK.fullmatch(track) for track in tracks):
            raise ValueError(f"filled tracks must be s<number>*: {text!r}")
        return {"tracks": tracks, "flag": "filled", "alternative": None}
    lead = LEAD.fullmatch(parts[0])
    if lead is None or any(not NAME.fullmatch(part) for part in parts[1:]):
        raise ValueError(f"not a spk side: {text!r}")
    if (lead["low"] or lead["close"]) and not TRACK.fullmatch(lead["member"]):
        raise ValueError(f"a name carries no flag: {text!r}")
    flag = "low" if lead["low"] else "close" if lead["close"] else None
    return {"tracks": [lead["member"], *parts[1:]], "flag": flag, "alternative": lead["alt"]}


def parse(cell: str) -> dict:
    """A whole cell.

    `final` is what the line ends up with, `acoustic` what the acoustic layer said; they are the same
    side unless `judged` (the speaker judge moved one of the line's sentences). `main` is the final
    lead -- "who says this line" -- or None for an unlabelled line."""
    if cell.count("<") > 1:
        raise ValueError(f"more than one '<': {cell!r}")
    judged = "<" in cell
    left, _, right = cell.partition("<")
    final = parse_side(left)
    if judged and not final["tracks"]:
        raise ValueError(f"the judge's side is empty: {cell!r}")
    acoustic = parse_side(right) if judged else final
    return {"main": final["tracks"][0] if final["tracks"] else None,
            "judged": judged, "final": final, "acoustic": acoustic}


def rows(path: str):
    """`(line number, position, spk)` per subtitle line of an annotated CSV; line N is the final SRT's Nth.

    Columns are found by the `# type|position|...|spk|note` header rather than by index. `note` is the
    last column and the only one that may hold a raw `|`, so a row is split at most columns - 1 times."""
    with open(path, encoding="utf-8-sig") as handle:
        lines = [line.rstrip("\r\n") for line in handle if line.strip()]
    if not lines or not lines[0].startswith("#"):
        raise ValueError(f"{path}: no '# type|...' header line")
    columns = lines[0].lstrip("#").strip().split("|")
    position, spk = columns.index("position"), columns.index("spk")
    body = [line for line in lines[1:] if not line.startswith("#")]
    for number, line in enumerate(body, start=1):
        cells = line.split("|", len(columns) - 1)
        if len(cells) < len(columns):
            raise ValueError(f"{path}: subtitle line {number} has {len(cells)} of {len(columns)} columns")
        yield number, cells[position], cells[spk]


if __name__ == "__main__":
    for number, position, spk in rows(sys.argv[1]):
        print(json.dumps({"line": number, "position": position, "spk": spk, **parse(spk)}, ensure_ascii=False))
