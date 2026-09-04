"""Lossless markdown -> node store importer (plan §7.1).

First pass only: every source line becomes exactly one node, duplicates
included; structured fields (items, reading, names) are *derived indexes*
and never rewrite the verbatim line. Ids are deterministic (UUIDv5 over the
source position) so repeated shadow imports produce the same identity set.
Merge candidates are reported, never applied.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..base import IndexEntry, _match_normalize, _parse_entry_for_index, parse_index_text
from .model import (
    METADATA_SECTION,
    PROFILE_SECTION,
    UPDATED_DATE_LABEL,
    MigrationAux,
    migration_id,
)
from .presets import Preset, SectionSpec, preset_for_category
from .store import KnowledgeStore, Transaction

CATEGORIES: tuple[str, ...] = ("streamer", "common")

#: Where anything needing a judgement call waits. Defined here rather than
#: in ``phase_b`` because both halves of the migration name it and
#: ``phase_b`` already imports from this module.
STAGING_SECTION = "待归类"

_FACT_RE = re.compile(r"^(?P<field>[^:：|]{1,24}?)(?P<sep>[:：][ \t]*)(?P<value>.*)$")
_EVENT_RE = re.compile(r"^(?P<date>\d{4}(?:-\d{2}){0,2})(?P<sep>[:：][ \t]*)(?P<desc>.*)$")
_RELATION_RE = re.compile(r"^(?P<target>[^|]+?)(?P<sep>[ \t]*\|[ \t]*)(?P<desc>.*)$")
_QUOTED_RE = re.compile(r"[「“\"]([^」”\"]+)[」”\"]")
_READING_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")
_MISHEARD_PAREN_RE = re.compile(r"[（(][^（）()]*误听[^（）()]*[）)]")


class ImportFormatError(ValueError):
    pass


@dataclass
class ParsedSection:
    name: str
    blank_after_heading: bool
    #: Blank lines between this section's last line and the next heading. One
    #: of them is the separator every section gets; a longer run is the user's
    #: own spacing and has to be recorded or the archive stops round-tripping
    #: (found 2026-09-01: a four-blank run before `## 元数据` was the last
    #: thing standing between a real third-party knowledge base and the
    #: parity gate).
    trailing_blank: int = 0
    lines: list[tuple[int, str, int]] = field(default_factory=list)  # (lineno, raw, blank_before)


@dataclass
class ParsedEntry:
    path: Path
    h1: str
    intro: str
    sections: list[ParsedSection]
    updated_date: str
    metadata_blank_after: bool
    trailing_newline: bool
    text: str
    has_metadata: bool = True


@dataclass
class MergeCandidate:
    category: str
    subject: str
    surface: str
    kind: str  # identical | complementary | conflict
    node_ids: list[str]
    lines: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ImportReport:
    rev: int
    subjects: int = 0
    nodes: int = 0
    items: int = 0
    memberships: int = 0
    id_map: list[dict[str, Any]] = field(default_factory=list)
    merge_candidates: list[MergeCandidate] = field(default_factory=list)
    source_lock: str = ""
    warnings: list[str] = field(default_factory=list)
    #: Sections the frozen v1 preset does not know. Their lines are imported
    #: VERBATIM as notes under their original name -- the importer is an
    #: archivist, so an unrecognised shape is preserved, never refused and
    #: never guessed at. Reshaping it is Phase B's job (it parks what needs a
    #: judgement call) and judging the content belongs to the LLM pass after
    #: that (owner 2026-09-01: format problems are not aligned by special
    #: cases, they degrade into an ingestion task).
    degraded_sections: list[dict[str, Any]] = field(default_factory=list)
    #: Entry files that share an H1. They import as separate subjects (the
    #: archive still has to round-trip file by file), each loser stamped
    #: with ``duplicate_of`` so Phase B can fold it into the winner.
    duplicate_subjects: list[dict[str, Any]] = field(default_factory=list)


# ---- parsing ----------------------------------------------------------------------


def strip_comments(path: Path, lines: Sequence[str], *, keep: bool = False) -> list[str]:
    """Drop paired multi-line HTML comments before anything else looks at the
    text. The full preview writes its guidance into comments, and the same
    file is edited and harvested back — without this every comment line would
    be classified as a note (review 2026-08-29 P1-2). Comment lines are
    removed ENTIRELY, not blanked, so blank-run bookkeeping stays honest;
    an unclosed comment is a loud error, never a half-swallowed file.

    Handle comments (``<!-- @k12 -->``) trail a content line and are left to
    the caller — only lines that are *nothing but* comment are dropped.

    ``keep`` retains them, and the caller chooses by SOURCE rather than by
    content: the harvester reads a file THIS program wrote, where a whole-line
    comment is our own scaffolding and dropping it is right; the importer
    reads the USER's archive, where such a line is something they wrote, and
    dropping it silently loses it -- and then fails the parity gate with an
    error naming the file but not the cause (found 2026-09-01 on a third
    party's knowledge base whose tooling wrote traceability comments into the
    entries). Validation of unclosed comments is unconditional either way."""

    out: list[str] = []
    depth = 0
    opened_at = 0
    for number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if depth:
            if "-->" in stripped:
                depth = 0
            if keep:
                out.append(raw)
            continue
        if stripped.startswith("<!--") and not stripped.endswith("-->"):
            depth = 1
            opened_at = number
            if keep:
                out.append(raw)
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            if keep:
                out.append(raw)
            continue  # a whole line of comment
        out.append(raw)
    if depth:
        raise ImportFormatError(f"{path}:{opened_at}: unclosed HTML comment")
    return out


def parse_entry_text(path: Path, text: str, *, keep_comments: bool = False) -> ParsedEntry:
    trailing_newline = text.endswith("\n")
    raw_lines = text.split("\n")
    if trailing_newline:
        raw_lines = raw_lines[:-1]
    raw_lines = strip_comments(path, raw_lines, keep=keep_comments)
    if not raw_lines or not raw_lines[0].startswith("# "):
        raise ImportFormatError(f"{path}: first line must be an H1")
    h1 = raw_lines[0][2:].strip()
    intro = ""
    index = 1
    # intro: everything before the first `## ` heading, ignoring blank lines;
    # more than one non-blank line there is a format we do not model.
    intro_lines: list[str] = []
    while index < len(raw_lines) and not raw_lines[index].startswith("## "):
        if raw_lines[index].strip():
            intro_lines.append(raw_lines[index])
        index += 1
    if len(intro_lines) > 1:
        raise ImportFormatError(f"{path}: more than one intro line before the first section")
    intro = intro_lines[0] if intro_lines else ""

    sections: list[ParsedSection] = []
    updated_date = ""
    metadata_blank_after = False
    has_metadata = False
    while index < len(raw_lines):
        heading = raw_lines[index]
        assert heading.startswith("## ")
        name = heading[3:].strip()
        index += 1
        blank_after = index < len(raw_lines) and raw_lines[index] == ""
        body: list[tuple[int, str]] = []
        while index < len(raw_lines) and not raw_lines[index].startswith("## "):
            body.append((index + 1, raw_lines[index]))
            index += 1
        # Counted on the RAW body, before the empty-section normalisation
        # below empties it -- otherwise an empty section with a long blank run
        # loses the run and nobody notices until the parity gate.
        trailing_blank = 0
        for _, raw in reversed(body):
            if raw != "":
                break
            trailing_blank += 1
        # An empty section is `## X` + the separator blank: that blank belongs
        # to the separator, not to the heading layout.
        if all(raw == "" for _, raw in body):
            blank_after = False
            body = []
        # strip the leading blank (heading layout) and trailing blanks (separator)
        if blank_after and body and body[0][1] == "":
            body = body[1:]
        while body and body[-1][1] == "":
            body = body[:-1]
        if name == METADATA_SECTION:
            metadata_blank_after = blank_after
            has_metadata = True
            for lineno, raw in body:
                if not raw.strip():
                    continue
                match = _FACT_RE.match(raw)
                if match and match.group("field").strip() == UPDATED_DATE_LABEL:
                    updated_date = match.group("value").strip()
                else:
                    raise ImportFormatError(f"{path}:{lineno}: unexpected metadata line {raw!r}")
            continue
        section = ParsedSection(
            name=name, blank_after_heading=blank_after, trailing_blank=trailing_blank
        )
        blank_run = 0
        for lineno, raw in body:
            if raw == "":
                blank_run += 1
                continue
            section.lines.append((lineno, raw, blank_run))
            blank_run = 0
        sections.append(section)
    return ParsedEntry(
        path=path,
        h1=h1,
        intro=intro,
        sections=sections,
        updated_date=updated_date,
        metadata_blank_after=metadata_blank_after,
        trailing_newline=trailing_newline,
        text=text,
        has_metadata=has_metadata,
    )


# ---- line classification ----------------------------------------------------------


TERM_FORMAT_ERROR = (
    "term 行必须四列：源语言|中文定名|别名|一句话描述（无别名时第三列留空；描述可含竖线）"
)
LABEL_RE = re.compile(r"^\[(?P<label>[^\[\]]*)\]\s?(?P<body>.*)$")
#: A bare ``[标记]`` with no body — what the full preview renders for an
#: unfilled core label. One definition; parity and the harvest both use it.
EMPTY_SLOT_RE = re.compile(r"^\[[^\[\]]+\]\s*$")
UNKNOWN_LABEL_ERROR = "本节不接受自定义标记（allow_custom_labels = false）"
EMPTY_LABEL_ERROR = "标记名不能为空"
BODY_KIND_ERROR = "本节不接受这种行体"


def split_label(raw: str) -> tuple[str | None, str]:
    """``([标记], 行体)`` — the bracket is the disambiguator, so this needs no
    vocabulary and never has to guess where a field ends (plan §2)."""

    match = LABEL_RE.match(raw)
    if match is None:
        return None, raw
    return match.group("label").strip(), match.group("body")


def classify_body(body: str, body_kinds: Sequence[str]) -> tuple[str, dict[str, Any]]:
    """The body's own shape decides its kind: >=4 pipe segments is a term
    (the LAST column absorbs further pipes, so a desc may contain them),
    exactly 3 is a format error (almost always a missing alias column), and
    anything else is a note.

    Both pipe rules are TERM diagnostics, so they only apply where a term can
    occur. In a section that takes prose and nothing else, a pipe is a
    character: subtitle lines join multiple speakers with ` | `, and refusing
    those made true content unrepresentable to catch a mistake (a term line in
    a prose section) that the section could not have honoured anyway."""

    if "term" not in body_kinds:
        if "note" not in body_kinds:
            return "invalid", {"text": body, "error": BODY_KIND_ERROR}
        return "note", {"text": body}
    parts = body.split("|")
    if len(parts) >= 4:
        return "term", {
            "surface": parts[0],
            "zh": parts[1],
            "desc": "|".join(parts[3:]),
            "alias_text": parts[2],
        }
    if len(parts) == 3:
        return "invalid", {"text": body, "error": TERM_FORMAT_ERROR}
    if "note" not in body_kinds:
        return "invalid", {"text": body, "error": BODY_KIND_ERROR}
    return "note", {"text": body}


def classify_line(raw: str, spec: "SectionSpec", *, preset: "Preset | None" = None) -> tuple[str, dict[str, Any]]:
    """``(kind, payload)`` for one line of the v3 grammar: an optional
    ``[标记]`` prefix followed by a body whose shape decides the kind.

    ``label`` rides the payload as a cross-kind optional key. Registration in
    the preset's ``labels`` table is NOT admission — it only carries
    ``core`` / ``role`` / ``share`` / ``verify``; an unregistered label is
    legal unless the preset sets ``allow_custom_labels = false``."""

    label, body = split_label(raw)
    if label is not None:
        if not label:
            return "invalid", {"text": raw, "error": EMPTY_LABEL_ERROR}
        allowed = preset.allow_custom_labels if preset is not None else True
        if not allowed and spec.label(label) is None:
            return "invalid", {"text": raw, "error": UNKNOWN_LABEL_ERROR}
    kind, payload = classify_body(body, spec.body_kinds)
    if kind != "invalid" and label:
        payload["label"] = label
    return kind, payload


ARCHIVE_TERM_MIN_SEGMENTS = 5


def classify_legacy_line(raw: str, line_form: str) -> tuple[str, dict[str, Any]]:
    """FROZEN version-1 grammar — read only by Phase A's lossless import of
    the markdown archive (plan §8). Terms are five-segment
    ``源|中|别名文本|读音|desc``; ``fact``/``event``/``relation`` split on a
    colon or a pipe and keep the separator verbatim so a re-import stays
    byte-identical. Nothing on the write path may call this."""

    if line_form == "term":
        parts = raw.split("|")
        if len(parts) >= ARCHIVE_TERM_MIN_SEGMENTS:
            surface, zh, alias_text, reading = parts[:4]
            return "term", {
                "surface": surface,
                "zh": zh,
                "alias_text": alias_text,
                "reading": reading,
                "desc": "|".join(parts[4:]),
            }
        return "note", {"text": raw}
    if line_form == "event":
        match = _EVENT_RE.match(raw)
        if match:
            return "event", {
                "occurred_at": match.group("date"),
                "sep": match.group("sep"),
                "description": match.group("desc"),
            }
        return "note", {"text": raw}
    if line_form == "relation":
        match = _RELATION_RE.match(raw)
        if match:
            return "relation", {
                "target": match.group("target"),
                "sep": match.group("sep"),
                "description": match.group("desc"),
            }
        return "note", {"text": raw}
    match = _FACT_RE.match(raw)
    if match:
        return "fact", {
            "field": match.group("field"),
            "sep": match.group("sep"),
            "value": match.group("value"),
        }
    return "note", {"text": raw}


_CLAUSE_END = "。；;，、"
_SENTENCE_ONLY_END = "。；;"


def _is_variant_run(text: str) -> bool:
    """Is this clause nothing but more quoted variants (plus 等/以及 glue)?

    A 误听 clause routinely spills past a 、 into further variants
    (``误听为“ユプカ竜”、“ユブカリュウ”``), so the span has to follow it —
    but it must stop at a clause that introduces a NEW statement."""

    rest = _QUOTED_RE.sub("", text)
    return not re.sub(r"[等以及和与、,，/\s]", "", rest)


def _misheard_span_end(text: str, start: int) -> int:
    """End offset of the 误听 clause that begins at ``start``.

    Stops at the first sentence end, or at a comma/、 whose next clause is not
    a continuation of the variant list. The old code used the SENTENCE as its
    unit, which in Chinese prose routinely swallows the definition that shares
    the sentence — and, on the way in, picked up quoted counter-examples from
    the next clause (review 2026-08-29 §6.1/§6.2)."""

    # `误听: a、b` is the explicit list form — there 、 IS the list separator,
    # so the clause runs to the sentence end.
    list_form = re.match(r"[:：]", text[start:]) is not None
    index = start
    while index < len(text):
        char = text[index]
        if char in _SENTENCE_ONLY_END:
            return index + 1
        if char in "，、" and not list_form:
            following = index + 1
            next_end = len(text)
            for boundary in _CLAUSE_END:
                found = text.find(boundary, following)
                if found != -1:
                    next_end = min(next_end, found)
            if _is_variant_run(text[following:next_end]):
                index = next_end
                continue
            return index + 1
        index += 1
    return len(text)


def extract_misheard(text: str) -> list[str]:
    """Derived index: quoted variants inside the 误听 CLAUSE, or the
    ``误听: a、b`` list form. Original text is never modified."""

    found: list[str] = []
    for match in re.finditer(r"误听", text):
        clause = text[match.end() : _misheard_span_end(text, match.end())].rstrip("。；;，、 ")
        quoted = [_READING_PAREN_RE.sub("", q).strip() for q in _QUOTED_RE.findall(clause)]
        if quoted:
            found.extend(q for q in quoted if q)
            continue
        list_match = re.match(r"[:：]\s*(.+)$", clause.strip())
        if list_match:
            found.extend(p.strip() for p in re.split(r"[、,，/]", list_match.group(1)) if p.strip())
    return list(dict.fromkeys(found))


def storable_term_payload(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Split a parsed term line into what lands in the payload and what lands
    in items (plan §11.2): the alias column is transient, a pasted legacy
    reading becomes one more alias (kana spellings ARE matchable surfaces),
    and 误听 prose moves out of the stored desc (variant extraction happens
    from the raw line separately). Shared by edit and proposal ingestion."""

    out = dict(payload)
    aliases = split_names(str(out.pop("alias_text", "")))
    reading = str(out.pop("reading", "")).strip()
    if reading and reading not in aliases:
        aliases.append(reading)
    out["desc"] = strip_misheard_prose(str(out.get("desc", "")))
    return out, aliases


def strip_misheard_prose(text: str) -> str:
    """Remove the sentences carrying 误听 variants from a description.

    误听 notation in a desc is TRANSPORT syntax (plan §11.2): edit/proposal
    ingestion extracts the variants into misheard items and then calls this
    so the stored desc never carries them — the prompt projection does not
    render misheard, so prose copies would silently drift from the items.
    Sentences without extractable variants are left alone."""

    if "误听" not in text:
        return text
    # parenthesized form first — `主角（误听:「アリズ」）` keeps its sentence
    out = _MISHEARD_PAREN_RE.sub(
        lambda m: "" if extract_misheard(m.group(0)) else m.group(0), text
    )
    result = out
    while True:
        marker = result.find("误听")
        if marker < 0:
            break
        start = max((result.rfind(char, 0, marker) for char in _CLAUSE_END), default=-1) + 1
        end = _misheard_span_end(result, marker + len("误听"))
        if not extract_misheard(result[start:end]):
            break  # a 误听 mention with nothing extractable: leave the prose alone
        result = result[:start] + result[end:]
    result = result.strip("；;，、 ").strip()
    # Hard floor: stripping must never empty a non-empty description. The old
    # sentence-granularity rule silently wiped 4 of 19 descriptions whose
    # definition shared a sentence with the 误听 note (review §6.1).
    return result if result.strip() else out


def split_names(text: str) -> list[str]:
    """Alias-list split. A token with no word character at all (the legacy
    ``—`` placeholder column, stray punctuation) is not a name — dropping it
    here covers import, the candidate scan and the rendered-edit alias diff alike."""

    parts = (part.strip() for part in re.split(r"[、,，]", text or ""))
    return [part for part in parts if part and re.search(r"\w", part)]


def alias_delta(old: Sequence[str], new: Sequence[str]) -> tuple[list[str], list[str]]:
    """``(added, removed)`` of an alias-column rewrite, by normalized identity.

    The apply engine's duplicate check compares ``_match_normalize`` values, so
    the diff must too: rewriting ``Alice`` as ``Ａｌｉｃｅ`` is the *same*
    alias (no add, no remove) — a raw-string diff would emit an add the engine
    rejects as duplicate plus a remove that then deletes the surviving item.
    """

    old_by_norm: dict[str, str] = {}
    for value in old:
        old_by_norm.setdefault(_match_normalize(value), value)
    new_norms = {_match_normalize(value) for value in new}
    added = [value for value in dict.fromkeys(new) if _match_normalize(value) not in old_by_norm]
    removed = [value for norm, value in old_by_norm.items() if norm not in new_norms]
    return added, removed


# ---- import -----------------------------------------------------------------------


def _source_lock(root: Path) -> str:
    git_dir = root / ".git"
    if git_dir.exists():
        try:
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                # cp936 machines decode captured output with the ANSI code
                # page unless pinned (see test_subprocess_text_encoding)
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if head.returncode == 0 and head.stdout.strip():
                return f"git:{head.stdout.strip()}"
        except OSError:
            pass
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.md")):
        if ".git" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"tree:{digest.hexdigest()}"


def import_knowledge_root(
    source_root: str | Path,
    store: KnowledgeStore,
    *,
    categories: tuple[str, ...] = CATEGORIES,
    task_id: str = "migrate",
) -> ImportReport:
    root = Path(source_root)
    lock = _source_lock(root)
    with store.begin("import", task_id=task_id, note=f"source={lock}") as txn:
        report = ImportReport(rev=txn.rev, source_lock=lock)
        for category in categories:
            directory = root / category
            if not directory.is_dir():
                report.warnings.append(f"missing category directory: {directory}")
                continue
            # Phase A parses the markdown archive against the grammar it was
            # written in — the FROZEN version-1 preset, never the current one
            # (kb-line-grammar plan §8).
            preset = preset_for_category(category, legacy=True)
            index_rows = {
                entry.key: entry
                for entry in parse_index_text((directory / "index.md").read_text(encoding="utf-8"))
            } if (directory / "index.md").exists() else {}
            seen_keys: set[str] = set()
            # Parsed up front so same-H1 files can be ranked against each other
            # before any of them is written.
            entries = []
            for path in sorted(directory.glob("*.md")):
                if path.name == "index.md":
                    continue
                # keep_comments: this is the user's archive, so a whole-line
                # comment is their content (see `strip_comments`).
                entries.append((path, parse_entry_text(
                    path, path.read_text(encoding="utf-8"), keep_comments=True
                )))
            losers = _rank_duplicates(entries, category, root, report)
            for path, parsed in entries:
                seen_keys.add(parsed.h1)
                _import_entry(
                    txn, report, category, preset, parsed, index_rows.get(parsed.h1), root,
                    duplicate_of=losers.get(path),
                )
            for key, row in index_rows.items():
                if key in seen_keys:
                    continue
                # Index row without an entry file: keep it as an index-only
                # subject (lossless) rather than silently dropping the name.
                _import_index_only(txn, report, category, row, f"{category}/index.md")
                report.warnings.append(f"{category}/{key}: index row without entry file (imported as index-only subject)")
        txn.set_meta("import_source_lock", lock)
        txn.set_meta("import_source_root", str(root))
    return report


def _rank_duplicates(
    entries: list[tuple[Path, ParsedEntry]],
    category: str,
    root: Path,
    report: ImportReport,
) -> dict[Path, str]:
    """Two files with the same H1 are one entry written twice.

    Imported as-is they become two subjects with the same surface, and name
    resolution then returns whichever it happens to reach first while the
    other's content is unreachable by name -- silently, with no warning and no
    merge candidate (found 2026-09-01 on a real third-party root, where
    `魔法少女の魔女裁判.md` and `魔法少女ノ魔女裁判.md` carried the same H1).

    Ranking (owner 2026-09-01): newest mtime wins; a tie goes to the longer
    file. ⚠ The tie is the NORMAL case, not the exception -- a git clone
    stamps every file with the checkout time, and so does unpacking an
    archive, so length usually decides. A third key (path) keeps it
    deterministic when even that ties.

    Returns ``{loser path: winner subject id}``. The losers are still imported
    in full -- the archive has to round-trip file by file, so nothing is merged
    here; the stamp is what lets Phase B fold them in later.
    """

    by_h1: dict[str, list[tuple[Path, ParsedEntry]]] = {}
    for path, parsed in entries:
        by_h1.setdefault(parsed.h1, []).append((path, parsed))
    losers: dict[Path, str] = {}
    for h1, group in sorted(by_h1.items()):
        if len(group) < 2:
            continue
        ranked = sorted(
            group,
            key=lambda item: (
                -item[0].stat().st_mtime,
                -item[0].stat().st_size,
                str(item[0].relative_to(root)),
            ),
        )
        winner_path = ranked[0][0]
        winner_id = migration_id(
            "subject", category, winner_path.relative_to(root).as_posix()
        )
        for path, _ in ranked[1:]:
            losers[path] = winner_id
        report.duplicate_subjects.append({
            "h1": h1,
            "winner": winner_path.relative_to(root).as_posix(),
            "losers": [p.relative_to(root).as_posix() for p, _ in ranked[1:]],
        })
        report.warnings.append(
            f"{category}: {len(group)} 个文件的标题都是 {h1!r}；"
            f"取 {winner_path.name}（较新/较长），其余的内容将由 Phase B 并入它的"
            f"「{STAGING_SECTION}」"
        )
    return losers


def _import_entry(
    txn: Transaction,
    report: ImportReport,
    category: str,
    preset: Preset,
    parsed: ParsedEntry,
    index_row: IndexEntry | None,
    root: Path,
    duplicate_of: str | None = None,
) -> None:
    rel = parsed.path.relative_to(root).as_posix()
    key, _intro, native_names, aliases = _parse_entry_for_index(parsed.text)
    entry_type = index_row.entry_type if index_row else ""
    if index_row is not None:
        # The index is derived from the entry today, but legacy index rows may
        # still carry names the entry file lacks: keep them (lossless).
        native_names = tuple(dict.fromkeys((*native_names, *index_row.native_names)))
        aliases = tuple(dict.fromkeys((*aliases, *index_row.aliases)))
    reading = ""
    profile_line = next(
        (raw for section in parsed.sections if section.name == PROFILE_SECTION for _, raw, _ in section.lines if raw.startswith("本名")),
        "",
    )
    paren = re.search(r"[（(]([^（）()]*)[）)]", profile_line.split("/")[0]) if profile_line else None
    if paren:
        reading = paren.group(1).strip()
    subject_id = migration_id("subject", category, rel)
    payload = {
        "surface": parsed.h1,
        # the entry's own intro wins; a legacy index row may carry one the file lacks
        "intro": parsed.intro or (index_row.intro if index_row else ""),
        "category": category,
        "entry_type": entry_type,
        "updated_date": parsed.updated_date,
        "section_order": [section.name for section in parsed.sections],
        "native_names": list(native_names),
        "reading": reading,
    }
    txn.create_node(subject_id, "subject", payload)
    txn.set_migration_aux(
        MigrationAux(
            local_id=subject_id,
            legacy_raw=parsed.text,
            source_path=rel,
            source_line=1,
            layout={
                # Only runs longer than the single separator are worth
                # storing; 0/1 is the default the renderer already produces.
                "trailing_blank": {
                    section.name: section.trailing_blank
                    for section in parsed.sections
                    if section.trailing_blank > 1
                },
                "blank_after_heading": {
                    **{section.name: section.blank_after_heading for section in parsed.sections},
                    METADATA_SECTION: parsed.metadata_blank_after,
                },
                "trailing_newline": parsed.trailing_newline,
                "intro_from_index": not parsed.intro and bool(index_row and index_row.intro),
                "has_metadata": parsed.has_metadata,
                **({"duplicate_of": duplicate_of} if duplicate_of else {}),
            },
        )
    )
    report.subjects += 1
    report.id_map.append({"local_id": subject_id, "kind": "subject", "source_path": rel, "source_line": 1})
    for alias in aliases:
        _create_item(txn, report, subject_id, "aliases", alias, fuzzy=False)

    by_surface: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for section_index, section in enumerate(parsed.sections):
        spec = preset.section(section.name)
        if spec is None:
            # Preserved, not refused: every line comes in as a note under its
            # own section name, so the archive still round-trips byte for byte
            # and the parity gate keeps its full strength.
            report.degraded_sections.append(
                {"source_path": rel, "section": section.name, "lines": len(section.lines)}
            )
        for line_index, (lineno, raw, blank_before) in enumerate(section.lines):
            if spec is None:
                kind, node_payload = "note", {"text": raw}
            else:
                kind, node_payload = classify_legacy_line(raw, spec.line_form)
                if kind not in spec.kinds:  # frozen v1 admission
                    kind, node_payload = "note", {"text": raw}
            node_id = migration_id("node", category, rel, str(section_index), str(line_index), raw)
            txn.create_node(node_id, kind, node_payload)
            txn.set_migration_aux(
                MigrationAux(
                    local_id=node_id,
                    legacy_raw=raw,
                    source_path=rel,
                    source_line=lineno,
                    layout={"blank_before": blank_before} if blank_before else {},
                )
            )
            membership_id = migration_id("membership", category, rel, str(section_index), str(line_index))
            txn.create_membership(membership_id, subject_id, node_id, section.name, line_index)
            report.nodes += 1
            report.memberships += 1
            report.id_map.append({"local_id": node_id, "kind": kind, "source_path": rel, "source_line": lineno})
            for variant in extract_misheard(raw):
                _create_item(txn, report, node_id, "misheard", variant, fuzzy=True)
            if kind == "term":
                for alias in split_names(node_payload["alias_text"]):
                    _create_item(txn, report, node_id, "aliases", alias, fuzzy=False)
                by_surface.setdefault(_match_normalize(node_payload["surface"]), []).append((node_id, raw, node_payload))

    for surface_key, group in by_surface.items():
        if len(group) < 2:
            continue
        report.merge_candidates.append(_classify_group(category, parsed.h1, group))


def _import_index_only(
    txn: Transaction, report: ImportReport, category: str, row: IndexEntry, source: str
) -> None:
    subject_id = migration_id("subject", category, f"index-only:{row.key}")
    txn.create_node(
        subject_id,
        "subject",
        {
            "surface": row.key,
            "intro": row.intro,
            "category": category,
            "entry_type": row.entry_type,
            "updated_date": "",
            "section_order": [],
            "native_names": list(row.native_names),
            "reading": "",
        },
    )
    txn.set_migration_aux(MigrationAux(local_id=subject_id, legacy_raw=row.to_line(), source_path=source, source_line=0, layout={"index_only": True}))
    report.subjects += 1
    report.id_map.append({"local_id": subject_id, "kind": "subject", "source_path": source, "source_line": 0})
    for alias in row.aliases:
        _create_item(txn, report, subject_id, "aliases", alias, fuzzy=False)


def _create_item(txn: Transaction, report: ImportReport, owner: str, field_name: str, value: str, *, fuzzy: bool) -> None:
    item_id = migration_id("item", owner, field_name, value)
    try:
        txn.create_item(item_id, owner, field_name, value, fuzzy_enabled=fuzzy)
    except Exception as exc:  # duplicate value on the same owner: keep the first
        if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
            return
        raise
    report.items += 1


def _classify_group(category: str, subject: str, group: list[tuple[str, str, dict[str, Any]]]) -> MergeCandidate:
    raws = [raw for _, raw, _ in group]
    if len(set(raws)) == 1:
        kind = "identical"
    else:
        kind = "complementary"
        for field_name in ("zh", "reading"):
            values = {p[field_name].strip() for _, _, p in group if p[field_name].strip()}
            if len(values) > 1:
                kind = "conflict"
                break
    return MergeCandidate(
        category=category,
        subject=subject,
        surface=group[0][2]["surface"],
        kind=kind,
        node_ids=[node_id for node_id, _, _ in group],
        lines=raws,
    )


def write_report_files(report: ImportReport, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "migration-id-map.jsonl").open("w", encoding="utf-8") as handle:
        for row in report.id_map:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "merge-candidates.jsonl").open("w", encoding="utf-8") as handle:
        for candidate in report.merge_candidates:
            handle.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")
    summary = {
        "rev": report.rev,
        "subjects": report.subjects,
        "nodes": report.nodes,
        "items": report.items,
        "memberships": report.memberships,
        "merge_candidates": {
            kind: sum(1 for c in report.merge_candidates if c.kind == kind)
            for kind in ("identical", "complementary", "conflict")
        },
        "source_lock": report.source_lock,
        "warnings": report.warnings,
    }
    (out / "import-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
