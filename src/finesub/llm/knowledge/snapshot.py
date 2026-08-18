"""Read-only knowledge access pinned to one embedded-git commit/tree."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence

from .base import (
    INVALID_ENTRY_CHARS_RE,
    KNOWLEDGE_CATEGORIES,
    IndexEntry,
    _match_normalize,
    parse_index_text,
)


class KnowledgeSnapshotError(RuntimeError):
    """The requested immutable knowledge view cannot be established/read."""


@dataclass(frozen=True)
class KnowledgeSnapshot:
    root: Path
    commit: str
    tree: str

    @classmethod
    def capture(cls, knowledge_root: str | Path) -> "KnowledgeSnapshot":
        root = Path(knowledge_root).expanduser().resolve()
        if not (root / ".git").exists():
            raise KnowledgeSnapshotError("Knowledge root is not an embedded git repository")
        status = cls._run(root, "status", "--porcelain", "--untracked-files=all")
        if status.stdout.strip():
            raise KnowledgeSnapshotError(
                "Knowledge root has uncommitted changes; commit them before starting a task"
            )
        commit = cls._run(root, "rev-parse", "HEAD").stdout.strip()
        tree = cls._run(root, "rev-parse", "HEAD^{tree}").stdout.strip()
        if not cls._is_object_id(commit) or not cls._is_object_id(tree):
            raise KnowledgeSnapshotError("Knowledge repository has no readable HEAD/tree")
        return cls(root=root, commit=commit, tree=tree)

    @property
    def identity(self) -> str:
        return f"git:{self.commit}:tree:{self.tree}"

    @property
    def reference(self) -> str:
        return f"knowledge://{self.identity}"

    @staticmethod
    def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise KnowledgeSnapshotError(
                f"git {' '.join(args)} failed for knowledge snapshot: {detail}"
            )
        return result

    @staticmethod
    def _is_object_id(value: str) -> bool:
        return len(value) in {40, 64} and all(
            character in "0123456789abcdef" for character in value
        )

    def _read_path(self, relative: str) -> str:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise KnowledgeSnapshotError(f"Invalid knowledge path: {relative!r}")
        result = subprocess.run(
            ["git", "-C", str(self.root), "show", f"{self.commit}:{path.as_posix()}"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise KeyError(relative)
        return result.stdout

    def _index(self, category: str) -> list[IndexEntry]:
        if category not in KNOWLEDGE_CATEGORIES:
            raise ValueError(f"Unknown knowledge category: {category!r}")
        try:
            content = self._read_path(f"{category}/index.md")
        except KeyError:
            content = ""
        return parse_index_text(content)

    def _all_entries(self) -> list[tuple[str, IndexEntry]]:
        return [
            (category, entry)
            for category in KNOWLEDGE_CATEGORIES
            for entry in self._index(category)
        ]

    def describe(self) -> dict[str, Any]:
        counts = {category: len(self._index(category)) for category in KNOWLEDGE_CATEGORIES}
        return {
            "snapshot_identity": self.identity,
            "categories": list(KNOWLEDGE_CATEGORIES),
            "entry_counts": counts,
            "read_only": True,
        }

    def list(
        self,
        *,
        prefix: str = "",
        cursor: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
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
            "entries": [
                self._entry_row(category, entry)
                for _score, category, entry in scored[:limit]
            ],
        }

    def _resolve(self, name: str) -> tuple[str, IndexEntry]:
        category_hint = ""
        key_name = name
        if "/" in name:
            category_hint, key_name = name.split("/", 1)
            if category_hint not in KNOWLEDGE_CATEGORIES:
                raise KeyError(name)
        needle = _match_normalize(key_name)
        categories: Sequence[str] = (
            (category_hint,) if category_hint else KNOWLEDGE_CATEGORIES
        )
        for category in categories:
            for entry in self._index(category):
                if needle in {_match_normalize(item) for item in entry.match_terms}:
                    return category, entry
        raise KeyError(name)

    def read(self, key: str) -> dict[str, Any]:
        category, entry = self._resolve(key)
        self._validate_entry_name(entry.key)
        content = self._read_path(f"{category}/{entry.key}.md")
        return {
            "snapshot_identity": self.identity,
            "category": category,
            "key": entry.key,
            "content": content,
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
        return {
            "snapshot_identity": self.identity,
            "entries": entries,
            "missing": missing,
        }

    def references(self, key: str) -> dict[str, Any]:
        target_category, entry = self._resolve(key)
        needle = _match_normalize(entry.key)
        matches: list[dict[str, Any]] = []
        for category, candidate in self._all_entries():
            if category == target_category and candidate.key == entry.key:
                continue
            try:
                text = self._read_path(f"{category}/{candidate.key}.md")
            except KeyError:
                continue
            normalized = _match_normalize(re.sub(r"[`*_#>\[\]()]", " ", text))
            if needle and needle in normalized:
                matches.append(self._entry_row(category, candidate))
        return {
            "snapshot_identity": self.identity,
            "key": entry.key,
            "references": matches,
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
        if (
            not value
            or value in {".", ".."}
            or len(value) > 100
            or INVALID_ENTRY_CHARS_RE.search(value)
        ):
            raise KnowledgeSnapshotError(f"Invalid knowledge entry key: {value!r}")
