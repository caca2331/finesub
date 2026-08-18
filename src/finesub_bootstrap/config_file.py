"""Surgical writer for the shared, hand-editable ``config.toml``.

``config.toml`` has two kinds of author: a person with an editor, and the
desktop settings panel. That makes byte preservation a hard requirement rather
than a nicety -- the first time a UI click reflows someone's comments and
spacing, the file stops being theirs. So this writes like ``secrets.py`` writes
``.env``: read, replace the one line that owns the setting, write the rest back
untouched.

Deliberately limited to scalar values. Rewriting an array (``[pools]``) means
re-serializing a container, which is where a line-based editor would start
losing the user's line breaks and inline comments -- so it refuses instead of
corrupting, and the day a pool-selection UI needs it, that is the moment to
bring in a round-trip TOML library rather than to extend this.

Stdlib only, on purpose: it lives here so the CLI shell can use it too without
adding a dependency to the managed runtime (see ``secrets.py`` for the same
constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .fsops import write_atomic
from .locks import holding_lock

_LOCK_TIMEOUT_SECONDS = 10.0

_TABLE_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
# One scalar value: a quoted string (which may itself contain '#') or a bare
# token. Used only to find where the old value ends.
_VALUE_RE = re.compile(r"^(?:\"(?:[^\"\\]|\\.)*\"|'[^']*'|[^#\s]+)")
_MULTILINE_MARKERS = ('"""', "'''")
# `[[name]]` opens an array of tables. `_TABLE_RE` cannot match it (it
# excludes brackets), so the preceding table's span would swallow the whole
# block and an edit could rewrite a key inside it. Refuse, like above.
_ARRAY_TABLE_RE = re.compile(r"^\s*\[\[")


class ConfigWriteError(RuntimeError):
    """The edit was refused, or it did not survive a read-back."""


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _format(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        # TOML basic strings and JSON strings agree on the escapes we can emit.
        return json.dumps(value)
    raise ConfigWriteError(
        f"config.toml writes are limited to scalars; got {type(value).__name__}. "
        "Arrays and tables need a round-trip TOML parser to keep the file's "
        "own formatting -- see this module's docstring."
    )


def _key_line_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^(\s*){re.escape(key)}(\s*=\s*)(.*)$")


def _table_spans(lines: list[str]) -> dict[str, tuple[int, int]]:
    """``{table name: (header index, end index exclusive)}`` for simple tables."""

    headers: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = _TABLE_RE.match(line)
        if match:
            headers.append((match.group(1).strip(), index))
    spans: dict[str, tuple[int, int]] = {}
    for position, (name, index) in enumerate(headers):
        end = headers[position + 1][1] if position + 1 < len(headers) else len(lines)
        # A repeated table header is legal TOML only for arrays of tables,
        # which _TABLE_RE already excludes; keep the first either way.
        spans.setdefault(name, (index, end))
    return spans


def _insert_at(lines: list[str], span: tuple[int, int]) -> int:
    """Just after the table's last non-blank line, not after its trailing gap."""

    start, end = span
    index = end
    while index > start + 1 and not lines[index - 1].strip():
        index -= 1
    return index


def update_config_file(
    path: Path,
    updates: Mapping[str, Mapping[str, Any]],
) -> None:
    """Set (``value``) or remove (``None``) settings, one table at a time.

    Everything not named in ``updates`` -- comments, blank lines, key order,
    quoting style, line endings, unrelated tables -- is preserved byte for
    byte. The file is re-parsed after the write and restored if the result
    would not load, so a bad edit cannot cost a user their configuration.

    Callers own the sparse policy: pass ``None`` to drop a setting that is back
    at its default rather than writing the default out.
    """

    updates = {
        table: {key: value for key, value in settings.items()}
        for table, settings in updates.items()
        if settings
    }
    if not updates:
        return
    for settings in updates.values():
        for value in settings.values():
            if value is not None:
                _format(value)  # reject unsupported types before touching disk

    path.parent.mkdir(parents=True, exist_ok=True)
    with holding_lock(_lock_path(path), timeout=_LOCK_TIMEOUT_SECONDS):
        # Read inside the lock, every time: holding a copy from startup and
        # writing it back would silently revert edits made by hand while the
        # app was open.
        original = path.read_bytes().decode("utf-8") if path.is_file() else ""
        if any(marker in original for marker in _MULTILINE_MARKERS):
            raise ConfigWriteError(
                f"{path} contains a multi-line string; this writer cannot edit "
                "it safely. Change the setting by hand."
            )
        if any(_ARRAY_TABLE_RE.match(line) for line in original.splitlines()):
            raise ConfigWriteError(
                f"{path} contains an array of tables ([[...]]); this writer "
                "cannot edit it safely. Change the setting by hand."
            )
        ending = "\r\n" if "\r\n" in original else "\n"
        lines = original.splitlines()
        spans = _table_spans(lines)

        for table, settings in updates.items():
            if table not in spans:
                wanted = {
                    key: value for key, value in settings.items() if value is not None
                }
                if not wanted:
                    continue
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append(f"[{table}]")
                lines.extend(f"{key} = {_format(value)}" for key, value in wanted.items())
                spans = _table_spans(lines)
                continue

            start, end = spans[table]
            for key, value in settings.items():
                pattern = _key_line_re(key)
                for index in range(start + 1, end):
                    match = pattern.match(lines[index])
                    if not match:
                        continue
                    if value is None:
                        del lines[index]
                    else:
                        indent, equals, rest = match.groups()
                        # Everything after the old value comes back verbatim,
                        # its own spacing included: a trailing comment is often
                        # aligned with the ones above it.
                        old = _VALUE_RE.match(rest)
                        trailing = rest[old.end():] if old else ""
                        lines[index] = f"{indent}{key}{equals}{_format(value)}{trailing}"
                    break
                else:
                    if value is None:
                        continue
                    lines.insert(_insert_at(lines, (start, end)), f"{key} = {_format(value)}")
                spans = _table_spans(lines)
                start, end = spans[table]

        text = ending.join(lines)
        if lines and (original.endswith(("\n", "\r\n")) or not original):
            text += ending
        if text == original:
            # Nothing actually changed -- clearing a setting that was never
            # written, most often. Skip the write: it would otherwise create an
            # empty file where there was none, and move an mtime that the
            # readers' cache keys on.
            return
        write_atomic(path, text, newline="")

        try:
            with path.open("rb") as handle:
                parsed = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            write_atomic(path, original, newline="")
            raise ConfigWriteError(f"Refusing a write that broke {path}: {exc}") from exc
        for table, settings in updates.items():
            for key, value in settings.items():
                actual = parsed.get(table, {}).get(key, None)
                if (value is None and actual is not None) or (
                    value is not None and actual != value
                ):
                    write_atomic(path, original, newline="")
                    raise ConfigWriteError(
                        f"Refusing a write that did not take effect: "
                        f"{table}.{key} reads back as {actual!r}"
                    )
