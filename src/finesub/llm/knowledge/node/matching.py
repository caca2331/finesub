"""Exact-match retrieval engine (plan §4.1–§4.2, step 4a).

One engine, two consumers: the agent's ``kb_search`` issues queries against
it, and the harness runs it over window text **in shadow** — matches are
recorded as flat ``matched`` events and change nothing about injection or
required blocks (v8). Fuzzy matching is deliberately absent (plan §8 step 7).

Matchable corpus = ``subject``/``term`` surfaces plus ``items`` rows with
``exact_enabled``. Text and keys are normalized the same way: the existing
key normalization (NFKC + casefold + t2s) plus katakana→hiragana folding, so
a katakana misheard variant matches however the ASR happened to cast it.
Spans are offsets into the *normalized* text — good enough for the shadow
ledger, which cares about which item fired, not byte positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..base import _match_normalize
from .model import digest
from .store import KnowledgeStore

ALGO_VERSION = "exact-1"

_KATAKANA_DELTA = ord("か") - ord("カ")


def scan_normalize(text: str) -> str:
    """Key/text normalization for exact matching: kana-folded ``_match_normalize``.

    NFKC runs first (inside ``_match_normalize``) so halfwidth katakana becomes
    fullwidth before the fold — folding first would leave ｶﾞ-style ASR output
    unfolded.
    """

    normalized = _match_normalize(text or "")
    return "".join(
        chr(ord(ch) + _KATAKANA_DELTA) if "ァ" <= ch <= "ヶ" else ch for ch in normalized
    )


@dataclass(frozen=True)
class MatchKey:
    text: str  # normalized form actually matched
    kind: str  # surface | alias | misheard
    node_id: str
    subject_id: str
    item_id: str | None = None
    requires_subject_context: bool = False
    raw: str = ""  # the stored surface/item value before normalization


@dataclass(frozen=True)
class Match:
    key: MatchKey
    start: int  # offsets into the normalized text
    end: int


class ExactIndex:
    """Inverted exact index over one pinned rev. Cheap to rebuild — the whole
    corpus is a few hundred keys — so there is no incremental maintenance."""

    def __init__(self, rev: int, keys: dict[str, list[MatchKey]]) -> None:
        self.rev = rev
        self._keys = keys

    @classmethod
    def build(
        cls, store: KnowledgeStore, rev: int | None = None, *, include_tentative: bool = False
    ) -> "ExactIndex":
        """``include_tentative=False`` is the model-facing corpus (kb_search:
        tentative entities are shadow-only, plan §11.5); the shadow scanners
        pass True — matching tentative content is exactly how it earns its
        corroboration."""

        at = store.current_rev() if rev is None else rev
        subject_of: dict[str, str] = {}
        for subject in store.subjects(at):
            subject_of[subject.local_id] = subject.local_id
            stack = [subject.local_id]
            while stack:
                parent = stack.pop()
                for membership in store.children(parent, at):
                    if membership.child_id not in subject_of:
                        subject_of[membership.child_id] = subject.local_id
                        stack.append(membership.child_id)

        keys: dict[str, list[MatchKey]] = {}

        def add(key: MatchKey) -> None:
            if len(key.text) >= 2:
                keys.setdefault(key.text, []).append(key)

        for subject in store.subjects(at):
            if subject.maturity == "tentative" and not include_tentative:
                continue
            surface = subject.payload.get("surface", "")
            add(MatchKey(scan_normalize(surface), "surface", subject.local_id, subject.local_id, raw=surface))
        for term in store.nodes_of_kind("term", at):
            subject_id = subject_of.get(term.local_id)
            if subject_id is None:
                continue  # orphaned (e.g. under a retired subject): not reachable, not matchable
            if term.maturity == "tentative" and not include_tentative:
                continue
            surface = term.payload.get("surface", "")
            add(MatchKey(scan_normalize(surface), "surface", term.local_id, subject_id, raw=surface))
        for relation in store.nodes_of_kind("relation", at):
            # Relation targets are spoken names too (plan §11.2): パパベルト
            # gets misheard like any term, so the target joins the corpus and
            # relation nodes may carry alias/misheard items (handled by the
            # generic items loop below).
            subject_id = subject_of.get(relation.local_id)
            if subject_id is None:
                continue
            if relation.maturity == "tentative" and not include_tentative:
                continue
            target = relation.payload.get("target", "")
            add(MatchKey(scan_normalize(target), "surface", relation.local_id, subject_id, raw=target))
        tentative_nodes = {
            node.local_id
            for kind in ("subject", "term", "relation", "fact", "event", "note")
            for node in store.nodes_of_kind(kind, at)
            if node.maturity == "tentative"
        } if not include_tentative else set()
        for item in store.all_items(at):
            if not item.exact_enabled:
                continue
            if not include_tentative and (
                item.maturity == "tentative" or item.local_id in tentative_nodes
            ):
                continue
            subject_id = subject_of.get(item.local_id)
            if subject_id is None:
                continue
            add(
                MatchKey(
                    scan_normalize(item.value),
                    "misheard" if item.field == "misheard" else "alias",
                    item.local_id,
                    subject_id,
                    item_id=item.item_id,
                    requires_subject_context=item.requires_subject_context,
                    raw=item.value,
                )
            )
        return cls(at, keys)

    def scan(self, text: str) -> list[Match]:
        """Every occurrence of every key in ``text`` (normalized substring)."""

        haystack = scan_normalize(text)
        matches: list[Match] = []
        for needle, owners in self._keys.items():
            start = haystack.find(needle)
            while start != -1:
                matches.extend(Match(owner, start, start + len(needle)) for owner in owners)
                start = haystack.find(needle, start + 1)
        matches.sort(key=lambda m: (m.start, m.end, m.key.kind, m.key.node_id))
        return matches

    def search(self, query: str) -> list[MatchKey]:
        """``kb_search`` backend: keys equal to or contained in the query."""

        needle = scan_normalize(query)
        if not needle:
            return []
        out: list[MatchKey] = []
        for text, owners in self._keys.items():
            if text in needle or needle in text:
                out.extend(owners)
        return out


def shadow_scan(
    store: KnowledgeStore,
    windows: Iterable[tuple[str, str]],
    *,
    task_id: str,
    rev: int | None = None,
) -> int:
    """One run's shadow pass (plan §4.2 step 4a): build the index once at the
    pinned rev, scan every ``(window_id, text)``, book the flat events.
    Returns how many event rows were new."""

    at = store.current_rev() if rev is None else rev
    index = ExactIndex.build(store, at, include_tentative=True)
    inserted = 0
    for window_id, text in windows:
        inserted += log_matched(store, index.scan(text), task_id=task_id, window_id=window_id, rev=at)
    return inserted


def log_matched(
    store: KnowledgeStore,
    matches: Iterable[Match],
    *,
    task_id: str,
    window_id: str = "",
    rev: int | None = None,
) -> int:
    """Record shadow ``matched`` events (flat, idempotent). Returns how many
    rows were new. No genealogy: v1 keeps trace/parent empty (plan §4.2).

    ``rev`` is part of the event identity: the same window rescanned against a
    later revision (new misheard items, changed corpus) is a *new* set of
    facts, while a resume/replay of the same run at the same rev deduplicates.
    """

    at = store.current_rev() if rev is None else rev
    inserted = 0
    for match in matches:
        payload = {
            "kind": "matched",
            "opportunity": "",
            "task_id": task_id,
            "window_id": window_id or None,
            "subject_id": match.key.subject_id,
            "node_id": match.key.node_id,
            "item_id": match.key.item_id,
            "matcher": ALGO_VERSION,
            "rev": at,
            "span": f"{match.start}-{match.end}",
            "algo_version": ALGO_VERSION,
        }
        cursor = store.conn.execute(
            "INSERT OR IGNORE INTO events(dedupe_key, trace_id, parent_event_id, kind, opportunity,"
            " task_id, window_id, subject_id, node_id, item_id, matcher, rev, span, algo_version)"
            " VALUES (?, '', NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                digest(payload),
                payload["kind"],
                payload["opportunity"],
                payload["task_id"],
                payload["window_id"],
                payload["subject_id"],
                payload["node_id"],
                payload["item_id"],
                payload["matcher"],
                at,
                payload["span"],
                payload["algo_version"],
            ),
        )
        inserted += cursor.rowcount if cursor.rowcount > 0 else 0
    # the connection is autocommit (isolation_level=None): each insert lands
    # immediately unless the caller opened an explicit revision transaction
    return inserted
