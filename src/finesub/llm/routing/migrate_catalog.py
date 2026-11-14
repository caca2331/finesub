"""Convert a pre-2026-11-09 override catalog (``model_catalog.psv``) to TOML.

    python -m finesub.llm.routing.migrate_catalog [OLD] [-o NEW]

``OLD`` defaults to the data root's ``model_catalog.psv`` (the file the loader
refuses); ``NEW`` defaults to ``model_catalog.toml`` beside it and is never
overwritten. Once ``NEW`` is written, ``OLD`` is renamed to ``OLD.bak`` (or
``OLD.2.bak``, ``OLD.3.bak``, ... when taken): kept as a backup, out of the
loader's way, and still under the repository's ``*.bak`` ignore rule.

The conversion keeps the old meaning, which is not the same as copying cells:

- an old row naming a packaged model *replaced* that row, so every column it
  left blank meant "the default". The new file patches key by key, so such a
  key is written as ``""`` (the loader's default) wherever the packaged model
  states a value -- otherwise it would silently inherit the packaged one;
- where the old and new defaults differ, the old one is written out: ``rpm`` /
  ``tpm`` of local and attached agents (old 100 / 4M, new unlimited). The one
  exception is ``supports_native_search`` (old default false, new true): the
  new default is taken on purpose (owner 2026-11-09) and the affected models
  are listed;
- a row for a model the packaged catalog has since renamed is written under
  the new name.

Before writing, the result is loaded with the new loader and compared with the
old reading field by field (``note`` / ``avoid`` are new, ``supports_native_search``
as above); any difference aborts without writing.

The data root's ``config.toml`` is then scanned for target names this version
renamed or dropped, and each is listed with its replacement (not edited).

To be removed at 1.0.0, together with the loader's refusal of the old file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from finesub.llm.routing.model_catalog import (
    CATALOG_FILENAME,
    CONVERSATIONAL_KIND,
    DEFAULT_PROVIDER_KIND,
    LOCAL_AGENT_KIND,
    OPENAI_COMPAT_KIND,
    PACKAGED_PROVIDER_KINDS,
    ModelCatalogEntry,
    _catalog_path,
    _entry_from_table,
    overlay_catalog_tables,
    read_catalog_tables,
)

#: The old file's columns, in its order; also the order keys are written in.
LEGACY_COLUMNS = (
    "fact_id",
    "provider_tier",
    "provider_kind",
    "base_url",
    "key_env",
    "display_name",
    "api_model_id",
    "max_input_tokens",
    "max_output_tokens",
    "context_window",
    "supports_audio",
    "supports_video",
    "video_high_resolution_only",
    "supports_native_search",
    "thinking",
    "token_scale",
    "rpm",
    "tpm",
    "rpd",
    "tpd",
    "is_free",
    "capability",
    "hint_output_ceiling",
    "fallback_model",
    "quota_pool",
)
_INT_COLUMNS = {"max_input_tokens", "max_output_tokens", "context_window", "rpm", "tpm", "rpd", "tpd"}
_BOOL_COLUMNS = {
    "supports_audio",
    "supports_video",
    "video_high_resolution_only",
    "supports_native_search",
    "is_free",
    "hint_output_ceiling",
}
#: What a blank cell meant in the old loader, for every column whose new
#: default is not derived the same way. Used to read old rows faithfully.
_OLD_DEFAULTS: Dict[str, Any] = {
    "max_output_tokens": 65_536,
    "supports_audio": False,
    "supports_video": False,
    "video_high_resolution_only": False,
    "supports_native_search": False,
    "thinking": True,
    "token_scale": 1.0,
    "rpm": 100,
    "tpm": 4_000_000,
    "rpd": -1,
    "tpd": -1,
    "is_free": False,
    "hint_output_ceiling": False,
}
_UNLIMITED_BY_DEFAULT_KINDS = {LOCAL_AGENT_KIND, CONVERSATIONAL_KIND}
#: Packaged models renamed by this version: old fact id -> new. The Claude Code
#: rows had a version in their name while their model id is the vendor's
#: "latest" alias.
RENAMED_MODELS: Dict[str, str] = {
    "local-claude-opus-5": "local-claude-opus",
    "local-claude-sonnet-5": "local-claude-sonnet",
    "local-claude-haiku-4_5": "local-claude-haiku",
}
#: Packaged target names renamed or dropped by this version: old -> the name
#: to bind instead. A dropped `native-` target's search now rides on the
#: model's own target, on `--llm-retrieval native` calls.
RENAMED_TARGETS: Dict[str, str] = {
    **{
        f"local-codex-{infix}-{model}": f"local-codex-{model}"
        for infix in ("completion", "native")
        for model in ("gpt-6-luna", "gpt-6_1-terra", "gpt-6_1-sol")
    },
    **{
        f"local-claude-{infix}-{old.removeprefix('local-claude-')}": new
        for infix in ("completion", "native")
        for old, new in RENAMED_MODELS.items()
    },
    **{
        f"local-agy-{infix}-gemini-{version}": f"local-agy-gemini-{version}"
        for infix in ("media", "native")
        for version in ("3_8-flash", "4-argon")
    },
    **{
        f"local-dsh-native-{model}": f"local-dsh-{model}"
        for model in ("deepseek-v4_1-flash", "deepseek-v4-pro")
    },
    **{
        f"local-codebuddy-native-{model}": f"local-codebuddy-{model}"
        for model in ("hy3", "glm-5_3-flash", "deepseek-v4_1-flash", "deepseek-v4-pro")
    },
}
_COMPARE_SKIP = {"fact_id", "note", "avoid", "self_reported", "supports_native_search"}


class MigrationError(Exception):
    pass


def read_legacy(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    """``(comment lines, rows)``; each row maps the columns it declares to its cells."""

    comments: List[str] = []
    columns: tuple[str, ...] | None = None
    rows: List[Dict[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            comments.append(line)
            continue
        parts = [part.strip() for part in line.split("|")]
        if columns is None:
            columns = tuple("capability" if name == "quality_score" else name for name in parts)
            unknown = [name for name in columns if name not in LEGACY_COLUMNS]
            if unknown or "fact_id" not in columns:
                raise MigrationError(f"{path}:{number}: not a catalog header (unknown {unknown})")
            continue
        if len(parts) != len(columns):
            raise MigrationError(f"{path}:{number}: expected {len(columns)} fields")
        rows.append(dict(zip(columns, parts)))
    if columns is None:
        raise MigrationError(f"{path} is empty")
    return comments, rows


def _typed_cell(column: str, text: str, where: str) -> Any:
    if column in _INT_COLUMNS:
        try:
            return int(text)
        except ValueError:
            raise MigrationError(f"{where}: {column} must be an integer, got {text!r}") from None
    if column in _BOOL_COLUMNS:
        if text.lower() not in ("true", "false"):
            raise MigrationError(f"{where}: {column} must be true/false, got {text!r}")
        return text.lower() == "true"
    if column == "token_scale":
        number = float(text)
        return int(number) if number.is_integer() else number
    if column == "thinking":
        if text.lower() in ("true", "false"):
            return text.lower() == "true"
        parts = [part.strip() for part in text.split(",")]
        if all(part.isdigit() for part in parts):
            return [int(part) for part in parts]
        return parts
    if column == "capability":
        return int(text) if text.isdigit() else text
    return text


def _kind_of(row: Mapping[str, str]) -> str:
    return (row.get("provider_kind") or "").lower() or PACKAGED_PROVIDER_KINDS.get(
        row.get("provider_tier", ""), DEFAULT_PROVIDER_KIND
    )


def legacy_table(row: Mapping[str, str], where: str) -> Dict[str, Any]:
    """The old reading of a row, every defaulted column spelled out."""

    table: Dict[str, Any] = {}
    for column in LEGACY_COLUMNS[1:]:
        text = row.get(column, "")
        if text:
            table[column] = _typed_cell(column, text, where)
        elif column in _OLD_DEFAULTS:
            table[column] = _OLD_DEFAULTS[column]
    return table


def converted_table(
    row: Mapping[str, str], packaged: Mapping[str, Any] | None, where: str
) -> Dict[str, Any]:
    """The new-format table for one old row (a patch when ``packaged`` is given)."""

    table: Dict[str, Any] = {}
    unlimited_by_default = _kind_of(row) in _UNLIMITED_BY_DEFAULT_KINDS
    for column in LEGACY_COLUMNS[1:]:
        text = row.get(column, "")
        if text:
            table[column] = _typed_cell(column, text, where)
        elif column in ("rpm", "tpm") and unlimited_by_default:
            table[column] = _OLD_DEFAULTS[column]
        elif column == "supports_native_search":
            # Left out, so a patch inherits the packaged value: the free Gemini
            # 3.x rows say false and must stay so; elsewhere the new default
            # (true on tiers with a tool) is taken on purpose.
            continue
        elif packaged is not None and column in packaged:
            table[column] = ""
    return table


def convert(path: Path) -> tuple[str, List[str]]:
    """``(new file text, notices)``; raises MigrationError on any mismatch."""

    comments, rows = read_legacy(path)
    packaged = read_catalog_tables(_catalog_path())
    notices: List[str] = []
    override: Dict[str, Dict[str, Any]] = {}
    expected: Dict[str, ModelCatalogEntry] = {}
    old_ids: Dict[str, str] = {}
    for row in rows:
        old_id = row["fact_id"]
        fact_id = RENAMED_MODELS.get(old_id, old_id)
        if fact_id in old_ids:
            raise MigrationError(
                f"{old_ids[fact_id]} and {old_id} would both be written as {fact_id}; "
                f"rename your own model in {path} and run again"
            )
        old_ids[fact_id] = old_id
        if fact_id != old_id:
            notices.append(f"{old_id}: the packaged model is now called {fact_id}; written under the new name")
        where = f"{path}: {old_id}"
        if _kind_of(row) == OPENAI_COMPAT_KIND and row.get("supports_native_search", "").lower() == "true":
            # Never searched: an old custom endpoint had no tool to hand over.
            # Searching now takes the request fields that switch it on.
            row = {**row, "supports_native_search": ""}
            notices.append(
                f"{old_id}: supports_native_search = true is dropped -- it never searched; "
                "add search_request to make this endpoint search"
            )
        override[fact_id] = converted_table(row, packaged.get(fact_id), where)
        expected[fact_id] = _entry_from_table(
            fact_id, legacy_table(row, where), where=where, self_reported=True
        )
    merged, stated = overlay_catalog_tables(packaged, override)
    for fact_id, old in expected.items():
        new = _entry_from_table(
            fact_id, merged[fact_id], where=f"converted {fact_id}", self_reported=fact_id in stated
        )
        old_fields, new_fields = asdict(old), asdict(new)
        differing = sorted(
            key for key in old_fields if key not in _COMPARE_SKIP and old_fields[key] != new_fields[key]
        )
        if differing:
            details = ", ".join(f"{key}: {old_fields[key]!r} -> {new_fields[key]!r}" for key in differing)
            raise MigrationError(f"{fact_id}: conversion would change {details}")
        if new.supports_native_search and not old.supports_native_search:
            notices.append(
                f"{fact_id}: supports_native_search is now true (the new default); "
                "it searches on --llm-retrieval native. Add supports_native_search = false to keep it from searching"
            )
    return render(comments, override), notices


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(name: str) -> str:
    return name if _BARE_KEY.match(name) else json.dumps(name, ensure_ascii=False)


def _value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
    return json.dumps(value, ensure_ascii=False)


def render(comments: Sequence[str], tables: Mapping[str, Mapping[str, Any]]) -> str:
    lines = list(comments)
    order = {name: index for index, name in enumerate(LEGACY_COLUMNS + ("note", "avoid"))}
    for fact_id, table in tables.items():
        if lines:
            lines.append("")
        lines.append(f"[models.{_key(fact_id)}]")
        for key in sorted(table, key=lambda name: order.get(name, len(order))):
            lines.append(f"{_key(key)} = {_value(table[key])}")
    return "\n".join(lines) + "\n"


def scan_config(config_path: Path) -> List[str]:
    if not RENAMED_TARGETS or not config_path.is_file():
        return []
    found: List[str] = []
    text = config_path.read_text(encoding="utf-8")
    for number, line in enumerate(text.splitlines(), start=1):
        for old, new in RENAMED_TARGETS.items():
            if re.search(rf"(?<![\w-]){re.escape(old)}(?![\w-])", line):
                found.append(f"{config_path}:{number}: {old} -> {new}")
    return found


def _free_backup_name(old: Path) -> Path:
    candidate, n = old.with_name(f"{old.name}.bak"), 1
    while candidate.exists():
        n += 1
        candidate = old.with_name(f"{old.name}.{n}.bak")
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.routing.migrate_catalog",
        description="Convert an old model_catalog.psv override to model_catalog.toml.",
    )
    parser.add_argument("old", nargs="?", help="old file (default: the data root's model_catalog.psv)")
    parser.add_argument("-o", "--output", help=f"new file (default: {CATALOG_FILENAME} beside the old one)")
    args = parser.parse_args(argv)

    from finesub.paths import resolve_config_file, resolve_model_catalog_override

    if args.old:
        old = Path(args.old).expanduser()
    else:
        found = resolve_model_catalog_override()
        if found is None or found.suffix.lower() != ".psv":
            print("No old-format model_catalog.psv found in the data root; nothing to convert.")
            return 0
        old = found
    output = Path(args.output).expanduser() if args.output else old.with_name(CATALOG_FILENAME)
    if output.exists():
        print(f"{output} already exists; not overwriting it.", file=sys.stderr)
        return 1
    try:
        text, notices = convert(old)
    except (MigrationError, ValueError) as exc:
        print(f"Not converted: {exc}", file=sys.stderr)
        return 1
    output.write_text(text, encoding="utf-8")
    backup = _free_backup_name(old)
    old.rename(backup)
    print(f"Wrote {output}; the old file is kept as {backup}.")
    config = resolve_config_file()
    for notice in notices + (scan_config(config) if config is not None else []):
        print(f"  - {notice}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
