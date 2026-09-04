"""Exact-match engine (plan §4.1–4.2, step 4a): normalization, scan, shadow events."""

from __future__ import annotations

from finesub.llm.knowledge.node.matching import ExactIndex, log_matched, scan_normalize
from finesub.llm.knowledge.node.store import KnowledgeStore


def _seed(store: KnowledgeStore) -> None:
    with store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]},
        )
        txn.create_node(
            "T", "term", {"surface": "スカラマシュ", "zh": "散兵", "alias_text": "", "reading": "", "desc": ""}
        )
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I1", "S", "misheard", "ユプカ竜", fuzzy_enabled=True)
        txn.create_item("I2", "T", "aliases", "散兵", fuzzy_enabled=False)
        txn.create_item("I3", "T", "misheard", "残表", fuzzy_enabled=True)
        txn.create_item("I4", "S", "misheard", "げ", fuzzy_enabled=True)  # 1 char: below min length
        txn.create_item("I5", "S", "aliases", "Genshin", fuzzy_enabled=False)


def test_scan_normalize_folds_katakana_and_case() -> None:
    assert scan_normalize("ユプカ竜") == scan_normalize("ゆぷか竜")
    assert scan_normalize("GENSHIN") == scan_normalize("genshin")
    assert scan_normalize("ー") == "ー"  # the long-vowel mark is not a kana letter
    # halfwidth katakana goes through NFKC first, then the fold
    assert scan_normalize("ﾕﾌﾟｶ竜") == scan_normalize("ゆぷか竜")


def test_scan_hits_misheard_across_kana_and_maps_to_subject(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        index = ExactIndex.build(store)
        matches = index.scan("昨日ゆぷか竜をやった、残表が出た")
        by_kind = {(m.key.kind, m.key.text) for m in matches}
        assert ("misheard", scan_normalize("ユプカ竜")) in by_kind
        assert ("misheard", "残表") in by_kind
        assert all(m.key.subject_id == "S" for m in matches)
        misheard = next(m for m in matches if m.key.text == "残表")
        assert misheard.key.node_id == "T" and misheard.key.item_id == "I3"
        # 1-char keys never enter the index
        assert not any(m.key.item_id == "I4" for m in index.scan("げ"))


def test_scan_matches_surfaces_and_respects_exact_disabled(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        with store.begin("user") as txn:
            txn.update_item("I5", exact_enabled=False, expected_from_rev=1)
        index = ExactIndex.build(store)
        assert any(m.key.kind == "surface" and m.key.node_id == "T" for m in index.scan("スカラマシュ强"))
        assert not index.scan("玩 Genshin 去")  # demoted item no longer matches


def test_orphaned_nodes_under_retired_subject_are_not_matchable(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        with store.begin("user") as txn:
            txn.tombstone_membership("M")
            txn.tombstone_node("S")
        index = ExactIndex.build(store)
        assert index.scan("残表とスカラマシュとユプカ竜") == []


def test_log_matched_is_idempotent_per_rev(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        index = ExactIndex.build(store)
        matches = index.scan("残表が出た")
        assert log_matched(store, matches, task_id="t1", window_id="w1") == len(matches) > 0
        assert log_matched(store, matches, task_id="t1", window_id="w1") == 0
        assert log_matched(store, matches, task_id="t1", window_id="w2") == len(matches)
        # rev is part of the event identity: rescanning against a later
        # revision records new facts instead of vanishing into the dedupe
        assert log_matched(store, matches, task_id="t1", window_id="w1", rev=2) == len(matches)
        rows = store.conn.execute("SELECT kind, subject_id, item_id, matcher FROM events").fetchall()
        assert all(r["kind"] == "matched" and r["subject_id"] == "S" for r in rows)


def test_shadow_scan_windows_helper_books_events_and_never_raises(tmp_path) -> None:
    """The correction-run hook (plan §4.2 step 4a): books matched events at the
    pinned rev, and swallows anything — telemetry must not sink a run."""

    from types import SimpleNamespace

    from finesub.llm.knowledge.node.repo import KnowledgeRepo
    from finesub.llm.stages.correction.run import _shadow_scan_windows

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common", "section_order": []},
        )
        txn.create_item("I", "S", "misheard", "残表", fuzzy_enabled=True)
    window = SimpleNamespace(chunk_id="w1", segments=[SimpleNamespace(text="残表が出た")])
    _shadow_scan_windows(tmp_path, "t1", "rev:1", [window])
    rows = repo.store.conn.execute("SELECT window_id, rev FROM events WHERE kind='matched'").fetchall()
    assert [(r["window_id"], r["rev"]) for r in rows] == [("w1", 1)]
    # rerun deduplicates; a broken window object is swallowed
    _shadow_scan_windows(tmp_path, "t1", "rev:1", [window])
    assert repo.store.conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"] == 1
    _shadow_scan_windows(tmp_path, "t1", "rev:1", [object()])


def test_search_backend_substring_semantics(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        index = ExactIndex.build(store)
        hits = index.search("この残表って何")
        assert any(k.kind == "misheard" and k.node_id == "T" for k in hits)
        assert index.search("") == []
