"""Local Markdown knowledge base: index + per-entry files, tracked by an embedded git repo."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import threading
import unicodedata
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Sequence

from finesub.paths import resolve_knowledge_root
from finesub.reporting import current_reporter
from finesub.text import t2s_converter

DEFAULT_KNOWLEDGE_ROOT = resolve_knowledge_root(required=False)
TASK_ARTIFACT_FILENAME = "task-artifacts.jsonl"


def knowledge_root_path(knowledge_root: str | Path | None = None) -> Path:
    resolved = resolve_knowledge_root(knowledge_root, required=True)
    assert resolved is not None
    return resolved

KNOWLEDGE_CATEGORIES = ("streamer", "common")
# v14 entry-schema ops: line-oriented increments/edits replace the old
# replace_section/append_history pair (append_history is gone — the 重要经历
# section carries a date per line instead of monthly sub-headings).
KNOWLEDGE_OPS = (
    "append_lines",
    "edit_lines",
    "replace_section",
    "create_entry",
    "delete_entry",
    "rename_entry",
)
EDIT_ACTIONS = ("change", "insert_after", "remove")
COMMON_ENTRY_TYPES = ("游戏", "梗", "事件", "人物")

# Both categories share the fixed 档案 (index source) and 元数据 (harness-owned)
# sections; every other ## section is free-form.
PROFILE_SECTION = "档案"
KNOWLEDGE_METADATA_SECTION = "元数据"
LATEST_UPDATE_LABEL = "最近更新日期"
_LEGACY_UPDATE_LABELS = ("最近更新日期", "最新更新日期")

INVALID_ENTRY_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

KNOWLEDGE_PROPOSALS_RE = re.compile(
    r"<knowledge_proposals\b[^>]*>(?P<body>.*?)</knowledge_proposals>",
    re.IGNORECASE | re.DOTALL,
)
CODE_FENCE_RE = re.compile(r"```(?:jsonl|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LineEdit:
    action: str
    line: int
    content: str = ""


@dataclass(frozen=True)
class KnowledgeProposal:
    category: str
    entry: str
    op: str
    section: str
    content: str
    entry_type: str = ""
    aliases: tuple[str, ...] = ()
    intro: str = ""
    edits: tuple[LineEdit, ...] = ()
    new_key: str = ""
    reason: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "KnowledgeProposal":
        aliases_raw = data.get("aliases") or ()
        if isinstance(aliases_raw, str):
            aliases = tuple(
                part.strip() for part in re.split(r"[、,]", aliases_raw) if part.strip()
            )
        else:
            aliases = tuple(str(item).strip() for item in aliases_raw if str(item).strip())
        edits_raw = data.get("edits") or ()
        edits: list[LineEdit] = []
        if isinstance(edits_raw, Sequence) and not isinstance(edits_raw, str):
            for item in edits_raw:
                if not isinstance(item, Mapping):
                    continue
                try:
                    line = int(item.get("line", 0))
                except (TypeError, ValueError):
                    line = 0
                edits.append(
                    LineEdit(
                        action=str(item.get("action", "")).strip(),
                        line=line,
                        content=str(item.get("content", "")).rstrip(),
                    )
                )
        return cls(
            category=str(data.get("category", "")).strip(),
            entry=str(data.get("entry", "")).strip(),
            op=str(data.get("op", "")).strip(),
            section=str(data.get("section", "")).strip(),
            content=str(data.get("content", "")).strip(),
            entry_type=str(data.get("entry_type", "")).strip(),
            aliases=aliases,
            intro=str(data.get("intro", "")).strip(),
            edits=tuple(edits),
            new_key=str(data.get("new_key", "")).strip(),
            reason=str(data.get("reason", "")).strip(),
            raw=dict(data),
        )

    def validation_error(self) -> str:
        if self.category not in KNOWLEDGE_CATEGORIES:
            return f"unknown category {self.category!r}"
        if not self.entry:
            return "entry is empty"
        if INVALID_ENTRY_CHARS_RE.search(self.entry) or self.entry in {".", ".."}:
            return f"entry {self.entry!r} contains characters invalid in a filename"
        if len(self.entry) > 100:
            return f"entry {self.entry!r} is too long for a filename"
        if self.op not in KNOWLEDGE_OPS:
            return f"unknown op {self.op!r}"
        if self.op in ("append_lines", "replace_section"):
            if not self.section:
                return "section is empty"
            if _normalize_section_name(self.section) == KNOWLEDGE_METADATA_SECTION:
                return "the 元数据 section is harness-owned"
            if not self.content:
                return "content is empty"
        elif self.op == "edit_lines":
            if not self.edits:
                return "edits is empty"
            for edit in self.edits:
                if edit.action not in EDIT_ACTIONS:
                    return f"unknown edit action {edit.action!r}"
                if edit.line < 1:
                    return f"edit line {edit.line!r} must be a positive integer"
                if edit.action != "remove" and not edit.content.strip():
                    return f"edit at line {edit.line} has empty content"
        elif self.op == "create_entry":
            if not self.intro:
                return "create_entry requires intro"
            if self.category == "common" and not self.entry_type:
                return "common create_entry requires entry_type"
            # v17: entry_type errors surface before the reason gate.
            if self.entry_type and self.entry_type not in COMMON_ENTRY_TYPES:
                return (
                    f"unknown entry_type {self.entry_type!r} "
                    f"(must be one of {'/'.join(COMMON_ENTRY_TYPES)})"
                )
            if len((self.reason or "").strip()) < 10:
                return (
                    "create_entry requires a reason explaining why the "
                    "knowledge was not merged into an existing entry"
                )
        elif self.op == "rename_entry":
            if not self.new_key:
                return "rename_entry requires new_key"
            if (
                INVALID_ENTRY_CHARS_RE.search(self.new_key)
                or self.new_key in {".", ".."}
                or len(self.new_key) > 100
            ):
                return f"new_key {self.new_key!r} is not a valid filename"
        if self.entry_type and self.entry_type not in COMMON_ENTRY_TYPES:
            return (
                f"unknown entry_type {self.entry_type!r} "
                f"(must be one of {'/'.join(COMMON_ENTRY_TYPES)})"
            )
        return ""


@dataclass(frozen=True)
class KnowledgeApplyRecord:
    category: str
    entry: str
    op: str
    section: str
    status: str
    reason: str
    target_path: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "entry": self.entry,
            "op": self.op,
            "section": self.section,
            "status": self.status,
            "reason": self.reason,
            "target_path": self.target_path,
        }


@dataclass(frozen=True)
class KnowledgeApplyReport:
    applied: List[KnowledgeApplyRecord]
    skipped: List[KnowledgeApplyRecord]
    committed: bool = False
    commit_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": [record.to_dict() for record in self.applied],
            "skipped": [record.to_dict() for record in self.skipped],
            "committed": self.committed,
            "commit_message": self.commit_message,
        }


@dataclass(frozen=True)
class IndexEntry:
    key: str
    entry_type: str
    aliases: tuple[str, ...]
    intro: str
    # v14: official names in other languages/scripts (the key itself is the
    # source-language name); a separate index field from nicknames/shorthands.
    native_names: tuple[str, ...] = ()

    @property
    def match_terms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.key, *self.native_names, *self.aliases)))

    def to_line(self) -> str:
        key_part = f"{self.key} [{self.entry_type}]" if self.entry_type else self.key
        return (
            f"- {key_part} | {'、'.join(self.native_names)} | "
            f"{'、'.join(self.aliases)} | {self.intro}"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _proposal_payload(text: str) -> str:
    match = KNOWLEDGE_PROPOSALS_RE.search(text or "")
    if match:
        return match.group("body").strip()
    fence = CODE_FENCE_RE.search(text or "")
    if fence:
        return fence.group(1).strip()
    return text or ""


def parse_knowledge_proposals(text: str) -> list[KnowledgeProposal]:
    proposals: list[KnowledgeProposal] = []
    payload = _proposal_payload(text)
    for line_no, raw_line in enumerate(payload.splitlines(), start=1):
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid knowledge proposal JSONL at line {line_no}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Invalid knowledge proposal at line {line_no}: expected object")
        proposals.append(KnowledgeProposal.from_mapping(data))
    return proposals


# ---------------------------------------------------------------------------
# Index handling


def index_path(knowledge_root: str | Path, category: str) -> Path:
    return knowledge_root_path(knowledge_root) / category / "index.md"


def entry_path(knowledge_root: str | Path, category: str, entry: str) -> Path:
    return knowledge_root_path(knowledge_root) / category / f"{entry}.md"


_KEY_TYPE_RE = re.compile(r"^(?P<key>.+?)\s*(?:\[(?P<type>[^\]]+)\])?\s*$")


def _split_names(text: str) -> tuple[str, ...]:
    return tuple(
        part.strip() for part in re.split(r"[、,]", text or "") if part.strip()
    )


def parse_index_text(text: str) -> list[IndexEntry]:
    """Parse index lines; accepts the v14 4-field form and the legacy 3-field
    form (`key | aliases | intro`, no native-name column)."""

    entries: list[IndexEntry] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("- ") or "|" not in line:
            continue
        parts = [part.strip() for part in line[2:].split("|")]
        if len(parts) < 3:
            continue
        key_match = _KEY_TYPE_RE.match(parts[0])
        if not key_match or not key_match.group("key").strip():
            continue
        if len(parts) >= 4:
            native_names = _split_names(parts[1])
            aliases = _split_names(parts[2])
            intro = "|".join(parts[3:]).strip()
        else:
            native_names = ()
            aliases = _split_names(parts[1])
            intro = parts[2]
        entries.append(
            IndexEntry(
                key=key_match.group("key").strip(),
                entry_type=(key_match.group("type") or "").strip(),
                aliases=aliases,
                intro=intro,
                native_names=native_names,
            )
        )
    return entries


def load_index_text(knowledge_root: str | Path, category: str) -> str:
    path = index_path(knowledge_root, category)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_index_entries(knowledge_root: str | Path, category: str) -> list[IndexEntry]:
    return parse_index_text(load_index_text(knowledge_root, category))


def _match_normalize(value: str) -> str:
    """Normalization for key/alias matching: NFKC + casefold + 繁→简归一."""

    normalized = unicodedata.normalize("NFKC", (value or "").strip()).casefold()
    converter = t2s_converter()
    if converter is None:
        return normalized
    return converter.convert(normalized)


def _index_lookup(
    knowledge_root: str | Path, category: str, name: str
) -> IndexEntry | None:
    needle = _match_normalize(name)
    if not needle:
        return None
    for index_entry in load_index_entries(knowledge_root, category):
        if needle in {_match_normalize(term) for term in index_entry.match_terms}:
            return index_entry
    return None


def resolve_entry_key(
    knowledge_root: str | Path,
    name: str,
) -> tuple[str, str] | None:
    """Resolve a primary key or alias to ``(category, key)``; ``None`` if unknown."""

    for category in KNOWLEDGE_CATEGORIES:
        index_entry = _index_lookup(knowledge_root, category, name)
        if index_entry is not None:
            return category, index_entry.key
    return None


def load_entry_texts(
    knowledge_root: str | Path,
    names: Sequence[str],
) -> tuple[dict[str, str], list[str]]:
    """Load entry file contents for primary keys or aliases.

    Returns ``(found: {primary_key: content}, missing: [name, ...])``.
    """

    found: dict[str, str] = {}
    missing: list[str] = []
    for name in names:
        resolved = resolve_entry_key(knowledge_root, name)
        if resolved is None:
            missing.append(name)
            continue
        category, key = resolved
        path = entry_path(knowledge_root, category, key)
        if key in found:
            continue
        if not path.exists():
            missing.append(name)
            continue
        found[key] = path.read_text(encoding="utf-8")
    return found, missing


# Local keyword pre-injection: harness-side (not LLM-robust) matching of index
# keys/aliases against free text (the user note), so knowledge entries reach
# round 1 / the correction windows without depending on the model asking.
KB_KEYWORD_MIN_TERM_CHARS = 2


@dataclass(frozen=True)
class KeywordMatch:
    category: str
    key: str
    hits: int
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "key": self.key,
            "hits": self.hits,
            "matched_terms": list(self.matched_terms),
        }


def match_index_keywords(
    knowledge_root: str | Path,
    text: str,
    *,
    max_entries: int = 8,
    min_term_chars: int = KB_KEYWORD_MIN_TERM_CHARS,
) -> list[KeywordMatch]:
    """Case-insensitive substring matching of index keys/aliases over ``text``.

    Aliases dedupe to their entry (one match per ``(category, key)``); entries
    rank by total occurrence count, index order breaking ties. Terms shorter
    than ``min_term_chars`` are skipped (single characters match everything).
    """

    haystack = (text or "").casefold()
    if not haystack.strip():
        return []
    matches: list[KeywordMatch] = []
    for category in KNOWLEDGE_CATEGORIES:
        for index_entry in load_index_entries(knowledge_root, category):
            hits = 0
            matched_terms: list[str] = []
            for term in index_entry.match_terms:
                cleaned = term.strip()
                if len(cleaned) < min_term_chars:
                    continue
                count = haystack.count(cleaned.casefold())
                if count:
                    hits += count
                    matched_terms.append(cleaned)
            if hits:
                matches.append(
                    KeywordMatch(
                        category=category,
                        key=index_entry.key,
                        hits=hits,
                        matched_terms=tuple(matched_terms),
                    )
                )
    matches.sort(key=lambda match: -match.hits)  # stable: index order breaks ties
    return matches[: max(0, int(max_entries))]


def load_preinjected_entries(
    knowledge_root: str | Path,
    text: str,
    *,
    max_entries: int = 8,
) -> tuple[dict[str, str], list[KeywordMatch]]:
    """Entry bodies for keys/aliases mentioned in ``text``, rank order kept."""

    matches = match_index_keywords(knowledge_root, text, max_entries=max_entries)
    found, _missing = load_entry_texts(knowledge_root, [match.key for match in matches])
    return found, matches


# 档案-section lines that feed the index. Values are `/`-separated names for
# 本名 and 、-separated nicknames for 别名; fully parenthesized parts are
# skeleton placeholders and are skipped.
_PROFILE_LINE_RE = re.compile(r"^-?\s*(?P<label>本名|别名)\s*[:：]\s*(?P<value>.*)$")
_READING_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")


def _parse_entry_for_index(
    text: str,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Extract ``(h1_key, intro, native_names, aliases)`` from entry markdown."""

    lines = (text or "").splitlines()
    key = ""
    intro = ""
    if lines and lines[0].startswith("# "):
        key = lines[0][2:].strip()
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            break
        if not (stripped.startswith("（") and stripped.endswith("）")):
            intro = stripped
        break
    native_names: list[str] = []
    aliases: list[str] = []
    span = _section_span(text, PROFILE_SECTION)
    if span is not None:
        for line in text[span[0] : span[1]].splitlines():
            match = _PROFILE_LINE_RE.match(line.strip())
            if not match:
                continue
            value = match.group("value").strip()
            if match.group("label") == "本名":
                for part in re.split(r"[/／]", value):
                    cleaned = _READING_PAREN_RE.sub("", part).strip()
                    if not cleaned or cleaned == key:
                        continue
                    native_names.append(cleaned)
            else:
                aliases.extend(_split_names(value))
    return key, intro, tuple(dict.fromkeys(native_names)), tuple(dict.fromkeys(aliases))


def _remove_index_line(knowledge_root: str | Path, category: str, key: str) -> None:
    path = index_path(knowledge_root, category)
    if not path.exists():
        return
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not (parsed := parse_index_text(line)) or parsed[0].key != key
    ]
    path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")


def sync_index_from_entry(
    knowledge_root: str | Path,
    category: str,
    key: str,
    *,
    entry_type_hint: str = "",
    extra_aliases: Sequence[str] = (),
) -> None:
    """Rebuild the entry's index line from its file content (v14: the entry
    body — H1, first description line, 档案's 本名/别名 lines — is the single
    source of truth for the index)."""

    target = entry_path(knowledge_root, category, key)
    if not target.exists():
        return
    parsed_key, intro, native_names, aliases = _parse_entry_for_index(
        target.read_text(encoding="utf-8")
    )
    path = index_path(knowledge_root, category)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines()
    existing: IndexEntry | None = None
    line_idx: int | None = None
    for idx, line in enumerate(lines):
        parsed = parse_index_text(line)
        if parsed and parsed[0].key == key:
            existing = parsed[0]
            line_idx = idx
            break
    merged = IndexEntry(
        key=parsed_key or key,
        entry_type=entry_type_hint or (existing.entry_type if existing else ""),
        aliases=tuple(
            dict.fromkeys(
                (*aliases, *(existing.aliases if existing and not aliases else ()),
                 *extra_aliases)
            )
        ),
        intro=intro or (existing.intro if existing else ""),
        native_names=native_names
        or (existing.native_names if existing else ()),
    )
    if line_idx is not None:
        lines[line_idx] = merged.to_line()
    else:
        lines.append(merged.to_line())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry file section operations


SECTION_HEADING_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)


def _normalize_section_name(name: str) -> str:
    """Whitespace-insensitive section-name matching: models drop the spaces in
    names like ``术语 / 系统 / 其他专有名词`` — that must not fork a section."""

    return re.sub(r"\s+", "", (name or "").strip())


def _section_span(text: str, section: str) -> tuple[int, int] | None:
    """Return ``(body_start, body_end)`` of the section body, excluding its heading."""

    wanted = _normalize_section_name(section)
    for match in SECTION_HEADING_RE.finditer(text):
        if _normalize_section_name(match.group("name")) != wanted:
            continue
        body_start = match.end()
        next_match = SECTION_HEADING_RE.search(text, match.end())
        body_end = next_match.start() if next_match else len(text)
        return body_start, body_end
    return None


def load_entry_skeleton(category: str, entry: str, intro: str) -> str:
    """New-entry scaffold from the per-category preset template (v14).

    Presets live in ``prompt_templates/entry_preset_{category}_v1.md`` (single
    source shared with the prompt-side description); empty preset sections are
    kept on purpose — the section names are the checklist of what to collect.
    """

    from ..prompt_compose import load_prompt_template  # lazy: keep base a leaf

    return load_prompt_template(
        f"entry_preset_{category}_v1.md",
        entry_name=entry,
        intro=intro or "（待补：一句话描述）",
        date=current_date(),
    )


_UPDATE_DATE_LINE_RE = re.compile(
    r"^-?\s*(?:%s)\s*[:：]" % "|".join(_LEGACY_UPDATE_LABELS)
)


def ensure_latest_update_date_text(text: str, date: str) -> str:
    """Harness-owned metadata refresh: the 元数据 section stays the last
    section and carries exactly one ``最近更新日期: YYYY-MM-DD`` line (legacy
    ``- 最新更新日期：…`` lines are replaced in place)."""

    if not DATE_RE.match(date):
        raise ValueError("date must be YYYY-MM-DD")
    line = f"{LATEST_UPDATE_LABEL}: {date}"
    span = _section_span(text, KNOWLEDGE_METADATA_SECTION)
    if span is None:
        return f"{text.rstrip()}\n\n## {KNOWLEDGE_METADATA_SECTION}\n\n{line}\n"
    start, end = span
    rows = [
        row
        for row in text[start:end].strip("\n").splitlines()
        if not _UPDATE_DATE_LINE_RE.match(row.strip())
    ]
    new_body = "\n".join([line, *rows]).strip()
    return f"{text[:start].rstrip()}\n\n{new_body}\n\n{text[end:].lstrip()}".rstrip() + "\n"


def replace_section_text(text: str, section: str, new_body: str) -> str:
    body = new_body.strip()
    span = _section_span(text, section)
    if span is None:
        block = f"## {section}\n\n{body}"
        meta_span = _section_span(text, KNOWLEDGE_METADATA_SECTION)
        if meta_span is None:
            return f"{text.rstrip()}\n\n{block}\n"
        heading_start = text.rfind("##", 0, meta_span[0])
        return (
            f"{text[:heading_start].rstrip()}\n\n{block}\n\n{text[heading_start:]}".rstrip()
            + "\n"
        )
    start, end = span
    return f"{text[:start].rstrip()}\n\n{body}\n\n{text[end:].lstrip()}".rstrip() + "\n"


def _line_dedup_token(line: str) -> str:
    """Dedup key for appended rows: the first ``|`` segment (the term-line's
    source-language field) or the whole line, normalized."""

    stripped = line.strip().lstrip("-").strip()
    head = stripped.split("|", 1)[0] if "|" in stripped else stripped
    return _match_normalize(head)


def append_lines_text(text: str, section: str, new_content: str) -> tuple[str, int]:
    """Append rows to a section (v14 line-increment op); returns the new text
    and the number of rows actually appended after dedup.

    A missing section is created right before 元数据 (which stays last); rows
    whose first field already exists in the section are skipped.
    """

    rows = [row.rstrip() for row in new_content.splitlines() if row.strip()]
    span = _section_span(text, section)
    if span is None:
        block = f"## {section}\n\n" + "\n".join(rows)
        meta_span = _section_span(text, KNOWLEDGE_METADATA_SECTION)
        if meta_span is None:
            return f"{text.rstrip()}\n\n{block}\n", len(rows)
        heading_start = text.rfind("##", 0, meta_span[0])
        return (
            f"{text[:heading_start].rstrip()}\n\n{block}\n\n{text[heading_start:]}".rstrip()
            + "\n",
            len(rows),
        )
    start, end = span
    body = text[start:end].strip("\n")
    existing_tokens = {
        _line_dedup_token(row) for row in body.splitlines() if row.strip()
    }
    fresh = [
        row
        for row in rows
        if _line_dedup_token(row) and _line_dedup_token(row) not in existing_tokens
    ]
    if not fresh:
        return text, 0
    new_body = "\n".join([body, *fresh]).strip("\n")
    return (
        f"{text[:start].rstrip()}\n\n{new_body}\n\n{text[end:].lstrip()}".rstrip() + "\n",
        len(fresh),
    )


def apply_line_edits(
    snapshot_text: str, edits: Sequence[LineEdit]
) -> tuple[str, str]:
    """Apply line-numbered edits against a frozen snapshot (v14 ``edit_lines``).

    Line numbers refer to the snapshot as rendered into the prompt; edits are
    applied in descending line order so earlier (lower) numbers stay valid.
    Guarded lines: 1 (the H1 title) and the 元数据 section. Returns
    ``(new_text, error)`` — a non-empty error rejects the whole op.
    """

    lines = snapshot_text.splitlines()
    meta_span = _section_span(snapshot_text, KNOWLEDGE_METADATA_SECTION)
    meta_range: tuple[int, int] | None = None
    if meta_span is not None:
        # Convert char span to 1-indexed line range (heading line included).
        heading_line = snapshot_text[: meta_span[0]].count("\n")  # heading is here
        end_line = snapshot_text[: meta_span[1]].count("\n") + 1
        meta_range = (heading_line, end_line)
    for edit in edits:
        limit = len(lines)
        if edit.action == "insert_after":
            if not (1 <= edit.line <= limit):
                return "", f"insert_after line {edit.line} out of range (1..{limit})"
        else:
            if not (2 <= edit.line <= limit):
                return "", (
                    f"{edit.action} line {edit.line} out of range (2..{limit}; "
                    "line 1 is the H1 title)"
                )
        if meta_range and meta_range[0] <= edit.line <= meta_range[1]:
            return "", f"line {edit.line} is inside the harness-owned 元数据 section"
    ordered = sorted(enumerate(edits), key=lambda item: (-item[1].line, item[0]))
    for _, edit in ordered:
        idx = edit.line - 1
        if edit.action == "change":
            lines[idx : idx + 1] = edit.content.splitlines() or [""]
        elif edit.action == "remove":
            del lines[idx]
        else:  # insert_after
            lines[idx + 1 : idx + 1] = edit.content.splitlines()
    return "\n".join(lines).rstrip() + "\n", ""


# ---------------------------------------------------------------------------
# Embedded git


GIT_MISSING_RETURNCODE = 127
GIT_MISSING_MESSAGE = (
    "git 不可用：知识库是一个内嵌 git 仓库，自动更新需要 git 在 PATH 上。"
)

# Long enough to sit out another process's apply (a few commits), short enough
# that a task never appears to hang on its by-product.
KNOWLEDGE_LOCK_TIMEOUT_SECONDS = 90


def knowledge_lock_path(knowledge_root: str | Path) -> Path:
    """The sidecar that serializes auto-apply across processes.

    A sibling of the knowledge directory rather than a file inside it: the
    directory is a git worktree, and anything inside would be swept into the
    auto-apply commits. Keying on the knowledge root rather than on an install
    root is what makes the lock work at all -- the desktop app, the CLI and a
    checkout each have a different install root but may share one knowledge
    base, and may equally point `--knowledge-root` at different ones.
    """

    root = knowledge_root_path(knowledge_root)
    return root.with_name(f"{root.name}.lock")


@contextmanager
def knowledge_write_lock(knowledge_root: str | Path) -> Iterator[bool]:
    """Hold the knowledge write lock; yields False when someone else has it.

    Failure to acquire is not an error: the subtitle is the product and the
    knowledge base is a by-product, so the caller skips applying and leaves the
    chunk ledger where it is, exactly as it does for a dirty repository.
    """

    try:
        from finesub_bootstrap.locks import LockUnavailable, holding_lock
    except ImportError:  # pragma: no cover - provisioning layer is optional
        yield True
        return
    try:
        with holding_lock(
            knowledge_lock_path(knowledge_root),
            timeout=KNOWLEDGE_LOCK_TIMEOUT_SECONDS,
        ):
            yield True
    except LockUnavailable:
        yield False


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=knowledge-harness",
                "-c",
                "user.email=knowledge@local",
                *args,
            ],
            cwd=root,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, OSError) as error:
        # Nothing installs git as part of this project, so "not on PATH" is an
        # ordinary state -- a desktop machine, a slim container -- not a bug. It
        # has to read back as a failed git call rather than a traceback from
        # deep inside a knowledge update, so every caller's existing non-zero
        # handling applies unchanged.
        return subprocess.CompletedProcess(
            args=("git", *args),
            returncode=GIT_MISSING_RETURNCODE,
            stdout="",
            stderr=f"{GIT_MISSING_MESSAGE}（{error}）",
        )


def git_is_available() -> bool:
    """Whether a `git` executable can be launched at all."""

    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return False
    return True


KNOWLEDGE_AUTO_BRANCH = "unverified"


def ensure_knowledge_git(
    knowledge_root: str | Path,
    *,
    allow_dirty: bool = False,
    snapshot_dirty: bool = False,
    task_id: str = "",
) -> bool:
    """Initialize the embedded git repo if missing and keep the working tree
    on the ``unverified`` branch (v15): every auto-apply commits there; the
    user merges to main manually after review, so main is the explicit anchor
    of verified knowledge. Returns True when the repo is usable."""

    root = knowledge_root_path(knowledge_root)
    root.mkdir(parents=True, exist_ok=True)
    had_git = (root / ".git").exists()
    if not had_git:
        result = _run_git(root, "init", "-q", "-b", KNOWLEDGE_AUTO_BRANCH)
        if result.returncode != 0:
            current_reporter().warning(
                "knowledge-git-init-failed",
                f"failed to init knowledge git repo at {root}: {result.stderr.strip()}",
                impact="本次不会写入知识库",
            )
            return False
    status = _run_git(root, "status", "--porcelain")
    dirty = status.returncode != 0 or bool(status.stdout.strip())
    head = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    branch = head.stdout.strip() if head.returncode == 0 else ""
    if branch and branch != KNOWLEDGE_AUTO_BRANCH:
        if dirty:
            current_reporter().warning(
                "knowledge-repo-dirty",
                f"refusing automatic knowledge update in dirty repo {root} "
                f"on branch {branch}",
                impact="本次不会写入知识库",
                action=f"先切到 {KNOWLEDGE_AUTO_BRANCH}",
            )
            return False
        # `checkout -B` is "create **or reset**": it moved the branch to the
        # current HEAD, so every auto-commit on `unverified` that had not been
        # merged yet became unreachable -- recoverable only from the reflog,
        # and gone for good after gc. Getting there took nothing exotic: look
        # at `main` once, leave the tree clean, run another task. Switch to the
        # branch when it exists; create it only when it does not.
        exists = _run_git(
            root,
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{KNOWLEDGE_AUTO_BRANCH}",
        )
        switched = (
            _run_git(root, "checkout", "-q", KNOWLEDGE_AUTO_BRANCH)
            if exists.returncode == 0
            else _run_git(root, "checkout", "-q", "-b", KNOWLEDGE_AUTO_BRANCH)
        )
        if switched.returncode != 0:
            current_reporter().warning(
                "knowledge-branch-switch-failed",
                f"failed to switch knowledge repo to "
                f"{KNOWLEDGE_AUTO_BRANCH}: {switched.stderr.strip()}",
                impact="本次不会写入知识库",
            )
            return False
    elif not branch:
        symbolic = _run_git(
            root, "symbolic-ref", "HEAD", f"refs/heads/{KNOWLEDGE_AUTO_BRANCH}"
        )
        if symbolic.returncode != 0:
            current_reporter().warning(
                "knowledge-branch-select-failed",
                f"failed to select {KNOWLEDGE_AUTO_BRANCH} in "
                f"knowledge repo: {symbolic.stderr.strip()}",
                impact="本次不会写入知识库",
            )
            return False
    if dirty and not allow_dirty:
        if not snapshot_dirty:
            current_reporter().warning(
                "knowledge-repo-dirty",
                f"refusing automatic knowledge update in dirty repo {root}",
                impact="本次不会写入知识库",
                action="提交或撤销知识库里的改动后重跑",
            )
            return False
        # `user-adjustment` is the signal a human relies on when reviewing
        # `unverified` before merging: it means "a person wrote this, it can be
        # trusted". A dirty tree is just as often the harness's own wreckage --
        # `apply_knowledge_proposals` wrote the files and the process died
        # before `commit_knowledge` -- and labelling that as a human edit
        # launders a half-applied change into a trusted one. When the residue
        # is ours, say so and warn; the ledger has not advanced either, so the
        # chunk will be re-applied on top and may double up.
        residue = _looks_like_harness_residue(root)
        if residue:
            current_reporter().warning(
                "knowledge-harness-residue",
                f"knowledge repo {root} holds uncommitted changes that look "
                "like an interrupted auto-apply, not a human edit; committing "
                "them as harness-residue",
                impact="被中断的那一段会再次应用一遍，可能重复",
                action="合并前先审阅",
            )
        kind = "harness-residue" if residue else "user-adjustment"
        message = (
            f"[{kind}] snapshot before auto-apply\n\n"
            f"change-kind: {kind}\n"
            f"trigger-task: {task_id or 'manual'}"
        )
        if not _commit_all_knowledge_changes(root, message):
            current_reporter().warning(
                "knowledge-snapshot-failed",
                f"failed to snapshot {kind} in knowledge repo {root}",
                impact="本次不会写入知识库",
            )
            return False
    return True


def _commit_all_knowledge_changes(root: Path, message: str) -> bool:
    add_result = _run_git(root, "add", "-A")
    if add_result.returncode != 0:
        current_reporter().warning(
            "knowledge-git-add-failed",
            f"git add failed in knowledge repo: {add_result.stderr.strip()}",
            impact="本次不会写入知识库",
        )
        return False
    diff_result = _run_git(root, "diff", "--cached", "--quiet")
    if diff_result.returncode == 0:
        return False
    commit_result = _run_git(root, "commit", "-q", "-m", message)
    if commit_result.returncode != 0:
        current_reporter().warning(
            "knowledge-git-commit-failed",
            f"git commit failed in knowledge repo: {commit_result.stderr.strip()}",
            impact="本次不会写入知识库",
        )
        return False
    return True


#: Written while an auto-apply is midway through editing files and removed once
#: the commit lands. Its presence in a dirty tree is what separates the
#: harness's own wreckage from a human's uncommitted edit -- a distinction that
#: matters because `user-adjustment` is the trust signal a reviewer reads.
APPLY_MARKER_NAME = ".finesub-applying"


def _apply_marker(root: Path) -> Path:
    # Inside `.git/`, which git never tracks. In the work tree it would be
    # staged by the very `git add -A` it exists to describe, and deleting it
    # afterwards would leave the tree dirty again -- so every subsequent run
    # would see a dirty repo caused by the marker itself.
    return root / ".git" / APPLY_MARKER_NAME


def begin_knowledge_apply(root: Path, task_id: str = "") -> None:
    try:
        _apply_marker(root).write_text(task_id or "manual", encoding="utf-8")
    except OSError:
        pass  # Best effort: losing the hint must not stop the apply.


def end_knowledge_apply(root: Path) -> None:
    try:
        _apply_marker(root).unlink(missing_ok=True)
    except OSError:
        pass


def _looks_like_harness_residue(root: Path) -> bool:
    return _apply_marker(root).is_file()


def commit_knowledge(knowledge_root: str | Path, message: str) -> bool:
    """Stage and commit all knowledge changes; return True if a commit was made."""

    root = knowledge_root_path(knowledge_root)
    if not ensure_knowledge_git(root, allow_dirty=True):
        return False
    committed = _commit_all_knowledge_changes(root, message)
    if committed:
        end_knowledge_apply(root)
    return committed


def knowledge_git_head(knowledge_root: str | Path) -> str:
    """Return the current embedded-repo commit, or an empty string if unborn."""

    root = knowledge_root_path(knowledge_root)
    if not (root / ".git").exists():
        return ""
    result = _run_git(root, "rev-parse", "HEAD")
    return result.stdout.strip() if result.returncode == 0 else ""


def knowledge_git_head_message(knowledge_root: str | Path) -> str:
    """Return the current embedded-repo commit message, or an empty string."""

    root = knowledge_root_path(knowledge_root)
    if not (root / ".git").exists():
        return ""
    result = _run_git(root, "log", "-1", "--format=%B")
    return result.stdout.strip() if result.returncode == 0 else ""


def knowledge_git_is_clean(knowledge_root: str | Path) -> bool:
    """Whether the embedded repository has no staged or unstaged changes."""

    root = knowledge_root_path(knowledge_root)
    if not (root / ".git").exists():
        return False
    result = _run_git(root, "status", "--porcelain")
    return result.returncode == 0 and not result.stdout.strip()


# ---------------------------------------------------------------------------
# Apply


def apply_knowledge_proposals(
    proposal_text: str,
    *,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    task_id: str = "",
    source: str = "",
    commit: bool = True,
    line_editable: Iterable[tuple[str, str]] | None = None,
) -> KnowledgeApplyReport:
    """Apply JSONL proposals to per-entry Markdown files and commit via embedded git.

    v14 phased apply: ``create_entry`` first (so later ops can target the new
    files), then all ``edit_lines`` per entry against one frozen snapshot
    (descending line order), then ``append_lines``/``replace_section`` in
    proposal order, finally metadata refresh + index rebuild per touched entry.

    ``line_editable`` is the set of ``(category, key)`` whose full body was
    rendered (line-numbered, untruncated) into the prompt this round;
    ``edit_lines`` against anything else is skipped. ``None`` disables the
    guard (manual/offline use).
    """

    root = knowledge_root_path(knowledge_root)
    if commit and not ensure_knowledge_git(
        root,
        snapshot_dirty=True,
        task_id=task_id,
    ):
        raise RuntimeError(
            "Knowledge repository is unavailable or could not snapshot "
            "pre-existing user adjustments."
        )
    # From here on the tree may be dirty because *we* made it dirty. The marker
    # is what a later run reads to tell that apart from a human's edit.
    begin_knowledge_apply(root, task_id)
    applied: list[KnowledgeApplyRecord] = []
    skipped: list[KnowledgeApplyRecord] = []
    editable = None if line_editable is None else {tuple(pair) for pair in line_editable}

    def _record(proposal: KnowledgeProposal, status: str, reason: str, target: Path | None = None) -> None:
        record = KnowledgeApplyRecord(
            category=proposal.category,
            entry=proposal.entry,
            op=proposal.op,
            section=proposal.section,
            status=status,
            reason=reason,
            target_path=str(target) if target else "",
        )
        (applied if status == "applied" else skipped).append(record)

    valid: list[tuple[KnowledgeProposal, str]] = []
    for proposal in parse_knowledge_proposals(proposal_text):
        error = proposal.validation_error()
        if error:
            _record(proposal, "skipped", error)
            continue
        note = ""
        target = entry_path(root, proposal.category, proposal.entry)
        if not target.exists():
            # A "new" name may be an existing entry under a script/alias
            # variant (绯月ゆい vs 緋月ゆい) — redirect instead of duplicating.
            index_entry = _index_lookup(root, proposal.category, proposal.entry)
            if index_entry is not None and index_entry.key != proposal.entry:
                note = f" (redirected from {proposal.entry!r})"
                proposal = replace(proposal, entry=index_entry.key)
        valid.append((proposal, note))

    created: set[tuple[str, str]] = set()
    # (category, key) -> (entry_type_hint, extra_aliases) for the index sync.
    touched: dict[tuple[str, str], tuple[str, tuple[str, ...]]] = {}

    def _touch(proposal: KnowledgeProposal) -> None:
        key = (proposal.category, proposal.entry)
        hint, extra = touched.get(key, ("", ()))
        touched[key] = (
            proposal.entry_type or hint,
            tuple(dict.fromkeys((*extra, *proposal.aliases))),
        )

    # Phase A: create_entry.
    for proposal, note in valid:
        if proposal.op != "create_entry":
            continue
        target = entry_path(root, proposal.category, proposal.entry)
        if target.exists():
            _record(proposal, "skipped", f"entry already exists{note}", target)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            load_entry_skeleton(proposal.category, proposal.entry, proposal.intro),
            encoding="utf-8",
        )
        created.add((proposal.category, proposal.entry))
        _touch(proposal)
        _record(proposal, "applied", f"applied{note}", target)

    # Phase B: edit_lines, grouped per entry against one frozen snapshot.
    edit_groups: dict[tuple[str, str], list[tuple[KnowledgeProposal, str]]] = {}
    for proposal, note in valid:
        if proposal.op == "edit_lines":
            edit_groups.setdefault((proposal.category, proposal.entry), []).append(
                (proposal, note)
            )
    for (category, key), group in edit_groups.items():
        target = entry_path(root, category, key)
        error = ""
        if (category, key) in created:
            error = "edit_lines cannot target an entry created in this batch"
        elif not target.exists():
            error = "entry does not exist"
        elif editable is not None and (category, key) not in editable:
            error = "line numbers unavailable (entry not injected or truncated this round)"
        if not error:
            snapshot = target.read_text(encoding="utf-8")
            merged = [edit for proposal, _ in group for edit in proposal.edits]
            new_text, error = apply_line_edits(snapshot, merged)
            if not error:
                target.write_text(new_text, encoding="utf-8")
        for proposal, note in group:
            if error:
                _record(proposal, "skipped", error, target)
            else:
                _touch(proposal)
                _record(
                    proposal,
                    "applied",
                    f"applied ({len(proposal.edits)} edits){note}",
                    target,
                )

    # Phase C: append_lines / replace_section in proposal order.
    for proposal, note in valid:
        if proposal.op not in ("append_lines", "replace_section"):
            continue
        target = entry_path(root, proposal.category, proposal.entry)
        if target.exists():
            text = target.read_text(encoding="utf-8")
        else:
            # Forgiving path for a missing create_entry: scaffold implicitly.
            # v17: implicit creation of a *common* entry must still carry a
            # valid entry_type (otherwise the index line would be typeless);
            # reject and ask for an explicit create_entry instead.
            if proposal.category == "common" and proposal.entry_type not in COMMON_ENTRY_TYPES:
                _record(
                    proposal,
                    "skipped",
                    (
                        "implicit creation of a missing common entry requires "
                        f"a create_entry with a valid entry_type{note}"
                    ),
                    target,
                )
                continue
            text = load_entry_skeleton(proposal.category, proposal.entry, proposal.intro)
            created.add((proposal.category, proposal.entry))
            note = f"{note} (auto-created)" if note else " (auto-created)"
            target.parent.mkdir(parents=True, exist_ok=True)
        if proposal.op == "replace_section":
            text = replace_section_text(text, proposal.section, proposal.content)
            reason = f"applied{note}"
        else:
            text, added = append_lines_text(text, proposal.section, proposal.content)
            reason = (
                f"applied ({added} rows){note}"
                if added
                else f"applied (0 rows; duplicates skipped){note}"
            )
        target.write_text(text, encoding="utf-8")
        _touch(proposal)
        _record(proposal, "applied", reason, target)

    # Phase C2: rename_entry — rewrite H1, move the file, rebuild the index
    # line under the new key (v15; edit_lines keeps line 1 read-only, renames
    # only happen through this explicit op).
    for proposal, note in valid:
        if proposal.op != "rename_entry":
            continue
        target = entry_path(root, proposal.category, proposal.entry)
        if not target.exists():
            _record(proposal, "skipped", f"entry does not exist{note}", target)
            continue
        new_target = entry_path(root, proposal.category, proposal.new_key)
        conflict = _index_lookup(root, proposal.category, proposal.new_key)
        if new_target.exists() or (
            conflict is not None and conflict.key != proposal.entry
        ):
            _record(
                proposal, "skipped", f"new_key already exists ({proposal.new_key!r})"
            )
            continue
        lines = target.read_text(encoding="utf-8").splitlines()
        if lines and lines[0].startswith("# "):
            lines[0] = f"# {proposal.new_key}"
        new_target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        target.unlink()
        _remove_index_line(root, proposal.category, proposal.entry)
        touched.pop((proposal.category, proposal.entry), None)
        hint = conflict.entry_type if conflict else ""
        touched[(proposal.category, proposal.new_key)] = (
            proposal.entry_type or hint,
            proposal.aliases,
        )
        _record(
            proposal,
            "applied",
            f"applied (renamed to {proposal.new_key!r}){note}",
            new_target,
        )

    # Phase D: metadata refresh + index rebuild once per touched entry.
    for (category, key), (entry_type_hint, extra_aliases) in touched.items():
        target = entry_path(root, category, key)
        if not target.exists():
            continue
        text = ensure_latest_update_date_text(
            target.read_text(encoding="utf-8"), current_date()
        )
        target.write_text(text, encoding="utf-8")
        sync_index_from_entry(
            root,
            category,
            key,
            entry_type_hint=entry_type_hint,
            extra_aliases=extra_aliases,
        )

    # Phase E: delete_entry, last — the merge-then-delete guard must see the
    #批内 append/replace results. The deleted entry's key (or an index
    # name of it) has to be findable in another entry of the same category or
    # in another proposal's content; otherwise the delete is refused (v15,
    # same philosophy as the add_mistake anti-fabrication check).
    for proposal, note in valid:
        if proposal.op != "delete_entry":
            continue
        target = entry_path(root, proposal.category, proposal.entry)
        if not target.exists():
            _record(proposal, "skipped", f"entry does not exist{note}", target)
            continue
        index_entry = _index_lookup(root, proposal.category, proposal.entry)
        needles = {
            _match_normalize(term)
            for term in (
                proposal.entry,
                *(index_entry.match_terms if index_entry else ()),
            )
            if term
        }
        haystack_parts = [
            other.content
            for other, _ in valid
            if other is not proposal and other.content
        ]
        for sibling in target.parent.glob("*.md"):
            if sibling.name in (target.name, "index.md"):
                continue
            haystack_parts.append(sibling.read_text(encoding="utf-8"))
        haystack = _match_normalize("\n".join(haystack_parts))
        if not any(needle and needle in haystack for needle in needles):
            _record(
                proposal,
                "skipped",
                "delete refused: entry name not found in any other entry or "
                "proposal content (merge it first)",
                target,
            )
            continue
        target.unlink()
        _remove_index_line(root, proposal.category, proposal.entry)
        touched.pop((proposal.category, proposal.entry), None)
        _record(proposal, "applied", f"applied (deleted){note}", target)

    committed = False
    message = ""
    if applied and commit:
        summary = "; ".join(
            f"{record.category}/{record.entry}:{record.section}" for record in applied[:5]
        )
        if len(applied) > 5:
            summary += f"; +{len(applied) - 5} more"
        message = f"[{task_id or 'manual'}] {summary}"
        if source:
            message += f"\n\nsource: {source}\napplied_at: {_utc_now()}"
        committed = commit_knowledge(root, message)
    if not applied or not commit:
        # Nothing of ours is uncommitted, so the marker would be a false
        # positive on the next run.
        end_knowledge_apply(root)

    return KnowledgeApplyReport(
        applied=applied,
        skipped=skipped,
        committed=committed,
        commit_message=message,
    )


# ---------------------------------------------------------------------------
# Task artifacts (unchanged behavior)


# Parallel window execution appends task artifacts from worker threads; the
# JSONL format is append-only safe, the bare write needs the lock
# (docs/llm_harness_behavior.md).
_APPEND_JSONL_LOCK = threading.Lock()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_JSONL_LOCK:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def append_task_artifact(
    artifact_dir: str | Path,
    *,
    kind: str,
    payload: Mapping[str, Any],
    task_id: str = "",
) -> Path:
    path = Path(artifact_dir).expanduser().resolve() / TASK_ARTIFACT_FILENAME
    _append_jsonl(
        path,
        {
            "task_id": task_id,
            "kind": kind,
            "payload": dict(payload),
            "created_at": _utc_now(),
        },
    )
    return path


def read_task_artifacts(
    paths: Iterable[str | Path],
    *,
    max_tokens: int = 40_000,
    count_tokens: Callable[[str], int] | None = None,
) -> str:
    # Lazy imports keep this module an intra-package leaf (it is imported by
    # nearly every llm module, including indirectly by the token stack).
    from ..token_truncate import cap_tokens

    if count_tokens is None:
        from ..token_budget import default_token_counter

        count_tokens = default_token_counter().count_text
    chunks: list[str] = []
    remaining = max(0, max_tokens)
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            path = path / TASK_ARTIFACT_FILENAME
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        tokens = count_tokens(text)
        if tokens > remaining:
            chunks.append(cap_tokens(text, remaining, count_tokens))
            break
        chunks.append(text)
        remaining -= tokens
        if remaining <= 0:
            break
    return "\n".join(chunks)
