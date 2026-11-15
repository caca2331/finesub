"""JSON Lines files: one JSON value per line, where a line ends at ``"\\n"``.

Every ledger the pipeline appends to and reads back on resume goes through
here, for two reasons a hand-written loop got wrong:

* **Only ``"\\n"`` ends a line.** ``json.dumps(..., ensure_ascii=False)``
  leaves U+2028, U+2029 and U+0085 unescaped, and ``str.splitlines()`` breaks
  on all three -- so a record holding one (pasted web text, a model reply) was
  cut in two, failed to parse and was skipped on resume: an already-paid call
  paid again, a `--resume-batch` refused. Reading splits on ``"\\n"`` alone;
  `dumps` also escapes U+2028/U+2029 so readers outside this module (a JS
  tool, a user's own `splitlines`) are safe too.
* **A crash leaves half a line.** Appending straight after it fused the next
  record onto it, and that record was then unreadable on every later resume.
  `append` first ends a file that does not end in a newline, so the torn piece
  stays a bad line of its own; `iter_records` skips a torn last line silently
  (a write still in progress looks the same) and reports any other bad line.

Needs nothing beyond the stdlib and `reporting` (itself stdlib-only), so any
layer can use it.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
from typing import Any

from .reporting import current_reporter

#: (path, bad-line count) pairs already reported in this process: one file is
#: often read by several stages of one run, and one warning says it.
_reported: set[tuple[str, int]] = set()


def dumps(record: Any, **options: Any) -> str:
    """One record as one line (no newline), U+2028/U+2029 escaped.

    ``options`` go to `json.dumps` (``sort_keys``, ``default``...);
    ``ensure_ascii`` stays off so the file remains readable as text.
    """

    text = json.dumps(record, ensure_ascii=False, **options)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def lines(text: str) -> list[str]:
    """``text`` cut at ``"\\n"`` only, each line without a trailing ``"\\r"``.

    Counts like `str.splitlines()` (a final newline adds no empty line), so it
    can stand in where a line number or a line cursor is kept.
    """

    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return [part[:-1] if part.endswith("\r") else part for part in parts]


def append(path: str | Path, record: Any, *, fsync: bool = False, **options: Any) -> None:
    """Append one record as one line, creating the file and its directory.

    The line goes out in one write, after a newline if the file does not end
    in one. Callers that share a file across threads keep their own lock.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (dumps(record, **options) + "\n").encode("utf-8")
    with open(target, "a+b", buffering=0) as handle:
        end = handle.seek(0, os.SEEK_END)
        if end:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                data = b"\n" + data
        handle.write(data)
        if fsync:
            os.fsync(handle.fileno())


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """The JSON objects in a JSON Lines file, in order; nothing if it is missing.

    Blank lines are ignored. A line that is not a UTF-8 JSON object is skipped
    and counted, and the count is reported once as a warning; an unterminated
    last line that does not parse is a torn or in-progress write and is
    skipped without one.
    """

    source = Path(path)
    try:
        data = source.read_bytes()
    except FileNotFoundError:
        return
    pieces = data.split(b"\n")
    bad = 0
    for index, piece in enumerate(pieces):
        if not piece.strip():
            continue
        try:
            record = json.loads(piece.decode("utf-8"))
        except ValueError:  # JSONDecodeError and UnicodeDecodeError alike
            record = None
        if isinstance(record, dict):
            yield record
        elif index < len(pieces) - 1:
            bad += 1
    if bad and (str(source), bad) not in _reported:
        _reported.add((str(source), bad))
        current_reporter().warning(
            "jsonl-bad-lines",
            f"skipped {bad} unreadable line(s) in {source}",
        )
