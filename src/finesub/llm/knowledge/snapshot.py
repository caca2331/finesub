"""Read-only knowledge access pinned to one store revision (plan §2.5).

This is the agent-side read API (index / search / read / references) that the
``kb_*`` tools wrap. Everything is answered at the captured ``rev``; later
writes by other processes do not leak into a running task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .base import (
    INVALID_ENTRY_CHARS_RE,
    KNOWLEDGE_CATEGORIES,
    IndexEntry,
    _match_normalize,
)
from .node.model import text_digest
from .node.repo import KnowledgeRepo


class KnowledgeSnapshotError(RuntimeError):
    """The requested immutable knowledge view cannot be established/read."""


@dataclass(frozen=True)
class KnowledgeSnapshot:
    root: Path
    rev: int
    index_digest: str

    @classmethod
    def capture(cls, knowledge_root: str | Path) -> "KnowledgeSnapshot":
        root = Path(knowledge_root).expanduser().resolve()
        if not root.is_dir():
            raise KnowledgeSnapshotError(f"Knowledge root does not exist: {root}")
        repo = KnowledgeRepo.open(root)
        rev = repo.rev
        digest = text_digest("\n".join(repo.index_text(category, rev) for category in KNOWLEDGE_CATEGORIES))
        return cls(root=root, rev=rev, index_digest=digest)

    @classmethod
    def at(cls, knowledge_root: str | Path, rev: int) -> "KnowledgeSnapshot":
        root = Path(knowledge_root).expanduser().resolve()
        repo = KnowledgeRepo.open(root)
        if rev > repo.rev:
            raise KnowledgeSnapshotError(f"revision {rev} is beyond the store's {repo.rev}")
        digest = text_digest("\n".join(repo.index_text(category, rev) for category in KNOWLEDGE_CATEGORIES))
        return cls(root=root, rev=rev, index_digest=digest)

    @property
    def identity(self) -> str:
        return f"rev:{self.rev}:index:{self.index_digest[:16]}"

    @property
    def reference(self) -> str:
        return f"knowledge://{self.identity}"

    @property
    def _repo(self) -> KnowledgeRepo:
        return KnowledgeRepo.open(self.root)

    # ---- index ----------------------------------------------------------------------

    def _index(self, category: str) -> list[IndexEntry]:
        if category not in KNOWLEDGE_CATEGORIES:
            raise ValueError(f"Unknown knowledge category: {category!r}")
        return self._repo.index_entries(category, self.rev)

    def _all_entries(self) -> list[tuple[str, IndexEntry]]:
        return [(category, entry) for category in KNOWLEDGE_CATEGORIES for entry in self._index(category)]

    def describe(self) -> dict[str, Any]:
        counts = {category: len(self._index(category)) for category in KNOWLEDGE_CATEGORIES}
        return {
            "snapshot_identity": self.identity,
            "rev": self.rev,
            "categories": list(KNOWLEDGE_CATEGORIES),
            "entry_counts": counts,
            "read_only": True,
        }

    def list(self, *, prefix: str = "", cursor: int = 0, limit: int = 100) -> dict[str, Any]:
        if cursor < 0 or not 1 <= limit <= 500:
            raise ValueError("cursor must be non-negative and limit within [1, 500]")
        needle = _match_normalize(prefix)
        rows = [
            self._entry_row(category, entry)
            for category, entry in self._all_entries()
            if not needle or _match_normalize(entry.key).startswith(needle)
        ]
        page = rows[cursor : cursor + limit]
        next_cursor = cursor + len(page)
        return {
            "snapshot_identity": self.identity,
            "entries": page,
            "next_cursor": next_cursor if next_cursor < len(rows) else None,
        }

    def search(self, query: str, *, limit: int = 20) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be within [1, 100]")
        terms = [_match_normalize(item) for item in query.split() if item.strip()]
        scored: list[tuple[int, str, IndexEntry]] = []
        for category, entry in self._all_entries():
            fields = [entry.key, *entry.native_names, *entry.aliases, entry.intro]
            normalized_fields = [_match_normalize(item) for item in fields]
            score = sum(
                4 if any(term == value for value in normalized_fields) else 1
                for term in terms
                if any(term in value for value in normalized_fields)
            )
            if score or not terms:
                scored.append((score, category, entry))
        scored.sort(key=lambda item: (-item[0], item[1], item[2].key.casefold()))
        return {
            "snapshot_identity": self.identity,
            "entries": [self._entry_row(category, entry) for _score, category, entry in scored[:limit]],
        }

    # ---- entries ----------------------------------------------------------------------

    def _resolve(self, name: str) -> tuple[str, IndexEntry]:
        category_hint = ""
        key_name = name
        if "/" in name:
            category_hint, key_name = name.split("/", 1)
            if category_hint not in KNOWLEDGE_CATEGORIES:
                raise KeyError(name)
        needle = _match_normalize(key_name)
        categories: Sequence[str] = (category_hint,) if category_hint else KNOWLEDGE_CATEGORIES
        for category in categories:
            for entry in self._index(category):
                if needle in {_match_normalize(item) for item in entry.match_terms}:
                    return category, entry
        raise KeyError(name)

    def read(self, key: str) -> dict[str, Any]:
        category, entry = self._resolve(key)
        self._validate_entry_name(entry.key)
        resolved = self._repo.resolve(entry.key, self.rev, category=category)
        if resolved is None:
            raise KeyError(key)
        # agent-facing read: the PROMPT projection (bare lines, round 12)
        content = self._repo.entry_injection_text(resolved.subject_id, self.rev)
        return {
            "snapshot_identity": self.identity,
            "category": category,
            "key": entry.key,
            "content": content,
            "content_digest": text_digest(content),
        }

    def read_many(self, keys: Sequence[str]) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        missing: list[str] = []
        seen: set[tuple[str, str]] = set()
        for key in keys:
            try:
                row = self.read(key)
            except KeyError:
                missing.append(key)
                continue
            identity = (row["category"], row["key"])
            if identity not in seen:
                entries.append(row)
                seen.add(identity)
        return {"snapshot_identity": self.identity, "entries": entries, "missing": missing}

    def references(self, key: str) -> dict[str, Any]:
        target_category, entry = self._resolve(key)
        needle = _match_normalize(entry.key)
        matches: list[dict[str, Any]] = []
        for category, candidate in self._all_entries():
            if category == target_category and candidate.key == entry.key:
                continue
            try:
                content = self.read(f"{category}/{candidate.key}")["content"]
            except KeyError:
                continue
            if needle and needle in _match_normalize(content):
                matches.append(self._entry_row(category, candidate))
        return {
            "snapshot_identity": self.identity,
            "category": target_category,
            "key": entry.key,
            "referenced_by": matches,
        }

    @staticmethod
    def _entry_row(category: str, entry: IndexEntry) -> dict[str, Any]:
        return {
            "category": category,
            "key": entry.key,
            "entry_type": entry.entry_type,
            "native_names": list(entry.native_names),
            "aliases": list(entry.aliases),
            "intro": entry.intro,
        }

    @staticmethod
    def _validate_entry_name(value: str) -> None:
        if not value or INVALID_ENTRY_CHARS_RE.search(value) or value in {".", ".."}:
            raise KnowledgeSnapshotError(f"Invalid knowledge entry name: {value!r}")
