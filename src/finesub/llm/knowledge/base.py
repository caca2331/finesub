"""Knowledge base: shared constants, index model, the read API, the write lock
and task artifacts.

The store itself is ``knowledge.node`` (SQLite, versioned rows, pinned reads
-- ``docs/plans/knowledge-node-plan.md``). This module keeps the function names
the harness has always called (``load_index_text``, ``load_entry_texts``,
``match_index_keywords`` …) and backs them with ``KnowledgeRepo``; the
legacy markdown tree is imported once on first open and then only ever
rendered, never read.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import unicodedata
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Sequence

from finesub.paths import resolve_knowledge_root
from finesub.text import t2s_converter

from .node.model import MATCHABLE_CATEGORIES

DEFAULT_KNOWLEDGE_ROOT = resolve_knowledge_root(required=False)
TASK_ARTIFACT_FILENAME = "task-artifacts.jsonl"
#: The facade's name for the matchable set (defined in `node/model.py`).
#: Aliased rather than re-typed: this constant gates the index, the keyword
#: pre-injection and the snapshot digest, and a second literal here is exactly
#: how `style` would end up visible to one of them and not the others.
KNOWLEDGE_CATEGORIES = MATCHABLE_CATEGORIES
COMMON_ENTRY_TYPES = ("游戏", "动画", "社区", "其他")

# Both categories share the fixed 档案 (index source) and 元数据 (harness-owned)
# sections; every other ## section is free-form.
PROFILE_SECTION = "档案"
KNOWLEDGE_METADATA_SECTION = "元数据"
LATEST_UPDATE_LABEL = "最近更新日期"

INVALID_ENTRY_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KNOWLEDGE_PROPOSALS_RE = re.compile(
    r"<knowledge_proposals\b[^>]*>(?P<body>.*?)</knowledge_proposals>",
    re.IGNORECASE | re.DOTALL,
)
CODE_FENCE_RE = re.compile(r"```(?:jsonl|json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

KNOWLEDGE_LOCK_TIMEOUT_SECONDS = 90.0
_APPEND_JSONL_LOCK = threading.Lock()


def knowledge_root_path(knowledge_root: str | Path | None = None) -> Path:
    resolved = resolve_knowledge_root(knowledge_root, required=True)
    assert resolved is not None
    return resolved


# ---------------------------------------------------------------------------
# Report records


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


# ---------------------------------------------------------------------------
# Index model


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


_KEY_TYPE_RE = re.compile(r"^(?P<key>.+?)\s*(?:\[(?P<type>[^\]]+)\])?\s*$")


def _split_names(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r"[、,]", text or "") if part.strip())


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


def _match_normalize(value: str) -> str:
    """Normalization for key/alias matching: NFKC + casefold + 繁→简归一."""

    normalized = unicodedata.normalize("NFKC", (value or "").strip()).casefold()
    converter = t2s_converter()
    if converter is None:
        return normalized
    return converter.convert(normalized)


def _line_dedup_token(line: str) -> str:
    """Dedup key for appended rows: the first ``|`` segment (the term-line's
    source-language field) or the whole line, normalized."""

    stripped = line.strip().lstrip("-").strip()
    head = stripped.split("|", 1)[0] if "|" in stripped else stripped
    return _match_normalize(head)


# ---------------------------------------------------------------------------
# Legacy entry-file parsing (used by the importer and the index derivation)

SECTION_HEADING_RE = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)
#: v3 identity/alias lines are `[标记] 值`; the v1 `字段: 值` form is still
#: matched so the frozen archive import keeps working (plan §8 Phase A).
_PROFILE_LINE_RE = re.compile(
    r"^-?\s*(?:\[(?P<label>本名|别名)\]\s*|(?P<legacy>本名|别名)\s*[:：]\s*)(?P<value>.*)$"
)
_READING_PAREN_RE = re.compile(r"[（(][^（）()]*[）)]")


def _normalize_section_name(name: str) -> str:
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


def _without_comments(lines: list[str]) -> list[str]:
    """Drop whole-line HTML comments and the blocks they open. The full
    preview writes the preset's guidance into comments, and that guidance
    quotes the label names it documents — parsing it as content would file
    `[别名] 顿号分隔：…` as an alias (found while wiring plan §11)."""

    out: list[str] = []
    inside = False
    for raw in lines:
        stripped = raw.strip()
        if inside:
            inside = "-->" not in stripped
            continue
        if stripped.startswith("<!--") and not stripped.endswith("-->"):
            inside = True
            continue
        if stripped.startswith("<!--"):
            continue
        out.append(raw)
    return out


def _parse_entry_for_index(text: str) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Extract ``(h1_key, intro, native_names, aliases)`` from entry markdown."""

    text = "\n".join(_without_comments((text or "").splitlines()))
    lines = text.splitlines()
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
            if (match.group("label") or match.group("legacy")) == "本名":
                for part in re.split(r"[/／]", value):
                    cleaned = _READING_PAREN_RE.sub("", part).strip()
                    if not cleaned or cleaned == key:
                        continue
                    native_names.append(cleaned)
            else:
                aliases.extend(_split_names(value))
    return key, intro, tuple(dict.fromkeys(native_names)), tuple(dict.fromkeys(aliases))


# ---------------------------------------------------------------------------
# Proposal block syntax (the retry loop only needs "is this JSONL")


def proposal_block_body(text: str) -> str:
    match = KNOWLEDGE_PROPOSALS_RE.search(text or "")
    if match:
        return match.group("body").strip()
    fence = CODE_FENCE_RE.search(text or "")
    if fence:
        return fence.group(1).strip()
    return text or ""


def parse_knowledge_proposals_jsonl(text: str) -> list[dict[str, Any]]:
    """Syntax-only parse of ``<knowledge_proposals>``; ``ValueError`` on bad JSONL."""

    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(proposal_block_body(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid knowledge proposal JSONL at line {line_no}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Invalid knowledge proposal at line {line_no}: expected object")
        rows.append(data)
    return rows


# ---------------------------------------------------------------------------
# Read API (store-backed)


def _repo(knowledge_root: str | Path | None):
    from .node.repo import KnowledgeRepo  # lazy: node.repo imports this module

    return KnowledgeRepo.open(knowledge_root_path(knowledge_root))


# --- generation pin (plan §2.5) --------------------------------------------------
#
# One run reads one revision: research, the correction windows and the query
# rounds all see the store as it was when the run started, however many
# revisions land in the meantime (another process, or another task in this
# one). The pin is *run-scoped* -- a ContextVar keyed by resolved root -- so
# two concurrent runs each see only their own rev and never serialize on each
# other (task-parallelism plan W1; the per-root RLock this replaces made the
# second run wait out the first's whole duration). An explicit ``rev``
# argument always wins, which is how the knowledge update reads its
# ``working_rev`` while a pin is active.
#
# A ContextVar does NOT follow the run into a thread pool by itself: a new
# worker starts with an empty Context. The run's pools rebind it through
# their initializer (`finesub.llm.run_context.bind_llm_worker`), and anything
# crossing a process boundary (the MCP server) gets the identity written down
# instead of inferred from thread state (the spawn env / the task manifest).
_GENERATION_PINS: ContextVar[Mapping[str, int] | None] = ContextVar(
    "finesub_generation_pins", default=None
)


@contextmanager
def pinned_generation_rev(knowledge_root: str | Path | None) -> Iterator[int]:
    root_key = str(knowledge_root_path(knowledge_root))
    rev = _repo(knowledge_root).rev
    pins = dict(_GENERATION_PINS.get() or {})
    pins[root_key] = rev
    token = _GENERATION_PINS.set(pins)
    try:
        yield rev
    finally:
        # Token reset restores the outer mapping, so same-context nesting
        # unwinds stack-wise exactly as the old dict restore did.
        _GENERATION_PINS.reset(token)


def repin_generation_rev(knowledge_root: str | Path | None) -> int | None:
    """Move this run's pin for ``root`` to the store's current rev (plan W3).

    Called at a phase boundary (the parallel barrier, where the entry set is
    fixed): the run's snapshot semantics become "one revision per phase"
    instead of "one per run", so a long run sees what other tasks committed
    meanwhile — still a snapshot (every window of the phase reads the same
    rev). No-op returning ``None`` when the root is not pinned (``--knowledge
    none`` never opened the store, and must not have it opened here). The
    surrounding ``pinned_generation_rev`` still restores the pre-run state on
    exit — its token reset is unaffected by intermediate sets."""

    root_key = str(knowledge_root_path(knowledge_root))
    pins = dict(_GENERATION_PINS.get() or {})
    if root_key not in pins:
        return None
    rev = _repo(knowledge_root).rev
    pins[root_key] = rev
    _GENERATION_PINS.set(pins)
    return rev


def _effective_rev(knowledge_root: str | Path | None, rev: int | None) -> int | None:
    if rev is not None:
        return rev
    return (_GENERATION_PINS.get() or {}).get(str(knowledge_root_path(knowledge_root)))


def active_generation_pins() -> dict[str, int]:
    """This run's pins (resolved root -> rev) -- context-scoped, see above.

    The agent tool session reads it to bind the ``kb_*`` tools to the run's
    pinned revision (plan §4.3): the spawn env gets the root, each task's
    manifest gets the identity. Callers on a fresh thread see an empty
    mapping unless the pool bound them (`run_context`), which is why the
    session host captures the pins on the thread that creates it.
    """

    return dict(_GENERATION_PINS.get() or {})


def bind_generation_pins(pins: Mapping[str, int]) -> None:
    """Rebind this thread to a run's pins (the pool-initializer channel)."""

    _GENERATION_PINS.set(dict(pins))


def kb_index_block_text(knowledge_root: str | Path, rev: int | None = None) -> str:
    """The ``kb_index`` tool's whole text payload.

    One composition, two readers: the MCP server renders the reply from it,
    and the harness computes the required block's ``digest`` from it when it
    creates the task — the fail-closed gate is exactly "the digest of what
    the tool returned equals what the manifest declared" (plan §4.3).
    """

    parts = []
    for category in KNOWLEDGE_CATEGORIES:
        index = load_index_text(knowledge_root, category, rev)
        if index.strip():
            parts.append(f"<{category}_index>\n{index}</{category}_index>")
    return "\n".join(parts)


def knowledge_version(knowledge_root: str | Path | None = None, rev: int | None = None) -> str:
    """Checkpoint identity of the knowledge base (``rev:N``; plan §2.5)."""

    return _repo(knowledge_root).version(_effective_rev(knowledge_root, rev))


def load_index_text(knowledge_root: str | Path, category: str, rev: int | None = None) -> str:
    return _repo(knowledge_root).index_text(category, _effective_rev(knowledge_root, rev))


def load_index_entries(knowledge_root: str | Path, category: str, rev: int | None = None) -> list[IndexEntry]:
    return _repo(knowledge_root).index_entries(category, _effective_rev(knowledge_root, rev))


def resolve_entry_key(
    knowledge_root: str | Path, name: str, rev: int | None = None
) -> tuple[str, str] | None:
    """Resolve a primary key or alias to ``(category, key)``; ``None`` if unknown."""

    resolved = _repo(knowledge_root).resolve(name, _effective_rev(knowledge_root, rev))
    return (resolved.category, resolved.key) if resolved else None


def load_entry_texts(
    knowledge_root: str | Path,
    names: Sequence[str],
    rev: int | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Entry bodies (human projection) for primary keys or aliases.

    Returns ``(found: {primary_key: content}, missing: [name, ...])``.
    """

    return _repo(knowledge_root).load_entry_texts(names, _effective_rev(knowledge_root, rev))


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
    rev: int | None = None,
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
        for index_entry in load_index_entries(knowledge_root, category, rev):
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


#: How many sub-entry hits one injection may carry. Sized from the shadow
#: ledger rather than guessed: 351 matched events over 20 real tasks give a
#: median of 8 hits per task and a maximum of 59, over only 42 distinct nodes.
#: 24 keeps the common case whole and clips the one outlier, at roughly a
#: kilobyte of prompt.
KB_TERM_PREINJECT_MAX_HITS = 24


@dataclass(frozen=True)
class TermMatch:
    """One sub-entry hit, with the address that makes it readable.

    A bare term line is a poor thing to inject -- a character name on its own
    says nothing about which work it belongs to. The parent subject and the
    section it sits in are what turn it into an answerable fact, and both are
    already available: the match carries `subject_id`, the membership carries
    the section.
    """

    subject: str
    section: str
    line: str
    via: tuple[str, ...]
    hits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "section": self.section,
            "line": self.line,
            "via": list(self.via),
            "hits": self.hits,
        }


def match_terms(
    knowledge_root: str | Path,
    text: str,
    *,
    exclude_subjects: Sequence[str] = (),
    max_hits: int = KB_TERM_PREINJECT_MAX_HITS,
    rev: int | None = None,
) -> list[TermMatch]:
    """Sub-entry matches over ``text``, addressed by parent subject and section.

    The engine is the exact index the shadow scanner already runs
    (`node/matching.py`): subject surfaces AND term surfaces AND items, with
    katakana folding so a misheard variant matches however the ASR cast it.
    This is the seam that lets it change injection instead of only booking
    events.

    Why it is worth having, measured on that shadow ledger: of 351 matches over
    20 real tasks, **347 were terms and 4 were subjects**, and **87% of the term
    hits belonged to a subject that was never itself mentioned**. Entry names
    are game titles and streamer names -- rarely said out loud; what gets said
    are the characters and proper nouns inside them.

    ``exclude_subjects`` drops hits whose parent entry is already being injected
    whole, since its body already contains these lines. Measured, that is only
    13% of hits -- the dedupe is correctness, not the main saving.
    """

    matches = _term_matches(knowledge_root, text, rev)
    if not matches:
        return []
    excluded = {str(name).strip() for name in exclude_subjects if str(name).strip()}
    out = [match for match in matches if match.subject not in excluded]
    return out[: max(0, int(max_hits))]


def _term_matches(
    knowledge_root: str | Path, text: str, rev: int | None
) -> list[TermMatch]:
    if not (text or "").strip():
        return []
    from .node.matching import ExactIndex
    from .node.render import format_line, node_aliases

    store = _repo(knowledge_root).store
    at = store.current_rev() if rev is None else rev
    index = ExactIndex.build(store, at, include_tentative=False)

    hits: dict[str, dict[str, Any]] = {}
    for match in index.scan(text or ""):
        key = match.key
        node = store.node(key.node_id, at)
        if node is None or node.kind == "subject":
            continue  # the entry level is `match_index_keywords`' job
        bucket = hits.setdefault(
            key.node_id,
            {"node": node, "subject": key.subject_id, "via": [], "hits": 0},
        )
        bucket["hits"] += 1
        if key.kind != "surface" and key.raw:
            bucket["via"].append(key.raw)

    out: list[TermMatch] = []
    for node_id, bucket in hits.items():
        node = bucket["node"]
        subject = store.node(bucket["subject"], at)
        parents = store.parents(node_id, at)
        out.append(
            TermMatch(
                subject=str((subject.payload if subject else {}).get("surface") or ""),
                section=str(parents[0].section if parents else ""),
                line=format_line(node, node_aliases(store, node, at)),
                via=tuple(dict.fromkeys(bucket["via"])),
                hits=int(bucket["hits"]),
            )
        )
    out.sort(key=lambda item: (-item.hits, item.subject, item.line))
    return out


#: A term line is `源语言|中文定名|别名|一句话描述` (docs/knowledge.md, 行文法
#: v3). The description sits last **and may itself contain pipes**, so it is
#: split off by position with a bounded split rather than by counting fields.
TERM_LINE_COLUMNS = 4


def strip_term_description(line: str) -> str:
    """A term line reduced to what it NAMES: source form, canonical, aliases.

    The names are what an ASR needs -- they are the strings it has to get
    right. The one-line description is what a corrector needs, and it is
    several times the length. Keeping the two separable is what makes an
    injection level that carries every name and no prose possible at all.

    A line that is not a term line (fewer columns) comes back unchanged: this
    trims, it does not filter.
    """

    body = str(line or "")
    fields = body.split("|")
    if len(fields) < TERM_LINE_COLUMNS:
        return body.rstrip()
    return "|".join(fields[: TERM_LINE_COLUMNS - 1]).rstrip()


def term_lines_only(text: str) -> str:
    """Keep the term lines of a rendered entry body, names only.

    Section headings are kept -- a name without the section it lives in is
    exactly the context-free fragment `TermMatch` exists to avoid -- and prose
    lines, guidance comments and empty sections are dropped.
    """

    kept: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            kept.append(line)
            continue
        if line.count("|") >= TERM_LINE_COLUMNS - 1:
            kept.append(strip_term_description(line))
    # A heading with nothing under it is noise once the prose is gone.
    out: list[str] = []
    for index, line in enumerate(kept):
        if line.lstrip().startswith("#"):
            following = kept[index + 1 :]
            if not following or following[0].lstrip().startswith("#"):
                continue
        out.append(line)
    return "\n".join(out)


def render_term_matches(
    matches: Sequence[TermMatch], *, names_only: bool = False
) -> str:
    """The injection block: one addressed line per hit, nothing else.

    Deliberately not the parent entry's body. The point of matching at this
    level is that one spoken name should cost one line rather than a whole
    entry -- `match_index_keywords` already covers the case where the entry
    itself was named.

    ``names_only`` drops each line's one-line description, leaving source form,
    canonical name and aliases. That is the shape a *recogniser* needs: the
    strings it has to get right, without the prose a corrector reads.
    """

    if not matches:
        return ""
    lines = []
    for match in matches:
        address = " / ".join(part for part in (match.subject, match.section) if part)
        body = match.line.lstrip("- ").strip()
        if names_only:
            body = strip_term_description(body)
        lines.append(f"- [{address}] {body}" if address else f"- {body}")
    return "\n".join(lines)


def load_preinjected_entries(
    knowledge_root: str | Path,
    text: str,
    *,
    max_entries: int = 8,
    rev: int | None = None,
) -> tuple[dict[str, str], list[KeywordMatch]]:
    """Entry bodies for keys/aliases mentioned in ``text``, rank order kept."""

    matches = match_index_keywords(knowledge_root, text, max_entries=max_entries, rev=rev)
    found, _missing = load_entry_texts(knowledge_root, [match.key for match in matches], rev)
    return found, matches


# ---------------------------------------------------------------------------
# Cross-process write lock


def knowledge_lock_path(knowledge_root: str | Path) -> Path:
    """The sidecar that serializes auto-apply across processes.

    A sibling of the knowledge directory: the desktop app, the CLI and a
    checkout each have a different install root but may share one knowledge
    base, and may equally point `--knowledge-root` at different ones.
    """

    root = knowledge_root_path(knowledge_root)
    return root.with_name(f"{root.name}.lock")


# The OS byte lock is exclusive per file *handle*, so a second thread of this
# process opening its own handle would read as "someone else has it" and take
# the loser path -- throwing away a generation another task in this very
# process spent minutes producing. In-process writers therefore *share* one
# held OS lock per root (refcounted); serialization between them is the apply
# queue's job (task-parallelism plan W2: 进程内等待，跨进程退让). Keyed by the
# physical resource (the lock path), which is exactly the §1.1 rule.
class _SharedHolder:
    def __init__(self) -> None:
        self.gate = threading.Lock()
        self.holders = 0
        self.release: Callable[[], None] | None = None


_WRITE_LOCK_GUARD = threading.Lock()
_WRITE_LOCK_HOLDERS: dict[str, _SharedHolder] = {}


@contextmanager
def knowledge_write_lock(knowledge_root: str | Path) -> Iterator[bool]:
    """Hold the knowledge write lock; yields False when another PROCESS has it.

    Failure to acquire is not an error: the subtitle is the product and the
    knowledge base is a by-product, so the caller skips applying and leaves the
    chunk ledger where it is. Writers inside this process never fail here --
    they share the held lock and serialize on the apply queue instead.
    """

    try:
        from finesub_bootstrap.locks import LockUnavailable, holding_lock
    except ImportError:  # pragma: no cover - provisioning layer is optional
        yield True
        return
    lock_path = knowledge_lock_path(knowledge_root)
    key = str(lock_path)
    with _WRITE_LOCK_GUARD:
        holder = _WRITE_LOCK_HOLDERS.setdefault(key, _SharedHolder())
    # The gate is held across the OS acquire on purpose: a second in-process
    # writer arriving mid-acquire waits here (bounded by the lock timeout)
    # and then shares the result instead of racing for its own handle.
    acquired = False
    with holder.gate:
        if holder.holders == 0:
            try:
                context = holding_lock(lock_path, timeout=KNOWLEDGE_LOCK_TIMEOUT_SECONDS)
                context.__enter__()
            except LockUnavailable:
                context = None
            if context is not None:
                holder.release = lambda: context.__exit__(None, None, None)
                holder.holders = 1
                acquired = True
        else:
            holder.holders += 1
            acquired = True
    if not acquired:
        yield False
        return
    try:
        yield True
    finally:
        with holder.gate:
            holder.holders -= 1
            if holder.holders == 0 and holder.release is not None:
                release, holder.release = holder.release, None
                release()


# ---------------------------------------------------------------------------
# Task artifacts


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
