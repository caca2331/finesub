"""Field-level events, claim evidence and the signals report (plan §5, step 5)."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from finesub.llm.knowledge.node.model import digest
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import (
    log_exposed_entries,
    log_landed_windows,
    record_evidence,
    refined_alignment_evidence,
    subject_pack_node_ids,
)
from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.knowledge.report import build_report


def _seed(store: KnowledgeStore) -> None:
    with store.begin("import") as txn:
        txn.create_node(
            "S",
            "subject",
            {"surface": "原神", "intro": "游戏", "category": "common",
             "native_names": ["原神"], "section_order": ["角色"]},
        )
        txn.create_node(
            "T", "term",
            {"surface": "スカラマシュ", "zh": "散兵", "alias_text": "", "reading": "", "desc": "角色"},
        )
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I1", "T", "misheard", "残表", fuzzy_enabled=True)
        txn.create_item("I2", "S", "misheard", "ユプカ竜", fuzzy_enabled=True)


# ---------------------------------------------------------------------------
# evidence


def test_record_evidence_is_idempotent_and_dated(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        kwargs = dict(
            node_id="T", field_path="items/I1", value_hash=digest("残表"),
            verdict="refuted", evidence_kind="refined_srt", task_id="t1", span="w1",
        )
        assert record_evidence(store, **kwargs) is True
        assert record_evidence(store, **kwargs) is False  # replay books nothing
        row = store.conn.execute("SELECT * FROM evidence").fetchone()
        assert row["verdict"] == "refuted" and row["created_at"]
        with pytest.raises(ValueError):
            record_evidence(store, **{**kwargs, "verdict": "maybe"})


def test_schema_v1_store_is_migrated_in_place(tmp_path) -> None:
    """A store created before evidence.created_at opens cleanly at v2."""

    path = tmp_path / "kb.sqlite"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', '1')")
    conn.execute(
        "CREATE TABLE evidence (evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " dedupe_key TEXT NOT NULL UNIQUE, node_id TEXT NOT NULL, field_path TEXT NOT NULL,"
        " value_hash TEXT NOT NULL, verdict TEXT NOT NULL, evidence_kind TEXT NOT NULL,"
        " source_ref TEXT, task_id TEXT NOT NULL, span TEXT, algo_version TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    with KnowledgeStore(path) as store:
        assert store.meta("schema_version") == "5"
        assert record_evidence(
            store, node_id="N", field_path="payload.zh", value_hash="h",
            verdict="confirmed", evidence_kind="manual", task_id="t",
        )


# ---------------------------------------------------------------------------
# exposed


def test_exposed_entries_split_correction_from_context(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    # 残表 (T's misheard) fires in the window text: T is a correction
    # opportunity, the rest of the pack is context. Reruns deduplicate.
    inserted = log_exposed_entries(
        repo, ["原神"], task_id="t1", window_id="w1", window_text="残表が出た", rev=1
    )
    assert inserted == 2
    assert log_exposed_entries(
        repo, ["原神"], task_id="t1", window_id="w1", window_text="残表が出た", rev=1
    ) == 0
    rows = repo.store.conn.execute(
        "SELECT node_id, opportunity FROM events WHERE kind='exposed' ORDER BY node_id"
    ).fetchall()
    assert [(r["node_id"], r["opportunity"]) for r in rows] == [
        ("S", "context"), ("T", "correction"),
    ]


def test_subject_pack_honors_the_sections_filter(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    assert set(subject_pack_node_ids(repo.store, "S", 1)) == {"S", "T"}
    assert subject_pack_node_ids(repo.store, "S", 1, sections=["档案"]) == ["S"]


# ---------------------------------------------------------------------------
# landed


def test_landed_needs_the_name_new_in_corrected(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        windows = [
            ("w1", "残表が出た", "スカラマシュが出た"),   # surface landed
            ("w2", "彼の話", "散兵の話"),                # zh landed
            ("w3", "スカラマシュ強い", "スカラマシュ強い"),  # already in raw: not landed
        ]
        assert log_landed_windows(store, windows, task_id="t1", rev=1) == 2
        assert log_landed_windows(store, windows, task_id="t1", rev=1) == 0
        rows = store.conn.execute(
            "SELECT window_id, node_id FROM events WHERE kind='landed' ORDER BY window_id"
        ).fetchall()
        assert [(r["window_id"], r["node_id"]) for r in rows] == [("w1", "T"), ("w2", "T")]


def test_landed_run_helper_is_fail_soft(tmp_path) -> None:
    from finesub.llm.stages.correction.run import _log_landed_windows

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    window = SimpleNamespace(
        chunk_id="w1", source_ids=("s1", "s2"),
        segments=[SimpleNamespace(text="残表が出た")],
    )
    rendered = [
        SimpleNamespace(source_ids=("s1",), corrected_text="スカラマシュが出た"),
        SimpleNamespace(source_ids=("s9",), corrected_text="散兵"),  # foreign window: ignored
    ]
    _log_landed_windows(tmp_path, "t1", "rev:1", [window], rendered)
    rows = repo.store.conn.execute("SELECT node_id FROM events WHERE kind='landed'").fetchall()
    assert [r["node_id"] for r in rows] == ["T"]
    _log_landed_windows(tmp_path, "t1", "rev:1", [object()], rendered)  # swallowed


def test_window_exposure_helper_is_fail_soft(tmp_path) -> None:
    from finesub.llm.stages.correction.attempts import _log_window_exposures

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    run = SimpleNamespace(
        knowledge_enabled=True, knowledge_root=tmp_path, knowledge_version="rev:1",
        task_id="t1",
    )
    window = SimpleNamespace(chunk_id="w1", segments=[SimpleNamespace(text="残表が出た")])
    _log_window_exposures(run, window, ["原神"])
    assert repo.store.conn.execute(
        "SELECT COUNT(*) AS n FROM events WHERE kind='exposed'"
    ).fetchone()["n"] == 2
    _log_window_exposures(run, object(), ["原神"])  # swallowed
    disabled = SimpleNamespace(knowledge_enabled=False)
    _log_window_exposures(disabled, window, ["原神"])  # no-op without knowledge


# ---------------------------------------------------------------------------
# refined-SRT alignment evidence


def test_refined_alignment_confirms_and_refutes_the_item_claim(tmp_path) -> None:
    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        windows = [
            # misheard fired, correction reached the output, refined kept it
            ("w1", "残表が出た", "散兵が出た", "散兵登场了"),
            # refined overturned the correction: strongest negative evidence
            ("w2", "残表が強い", "散兵が強い", "他很强"),
            # no correction consistent with the claim: nothing to judge
            ("w3", "残表かな", "そのままだ", "就这样"),
        ]
        assert refined_alignment_evidence(store, windows, task_id="t1", rev=1) == (1, 1)
        assert refined_alignment_evidence(store, windows, task_id="t1", rev=1) == (0, 0)
        rows = store.conn.execute(
            "SELECT field_path, verdict, span FROM evidence ORDER BY span"
        ).fetchall()
        assert [(r["field_path"], r["verdict"], r["span"]) for r in rows] == [
            ("items/I1", "confirmed", "w1"),
            ("items/I1", "refuted", "w2"),
        ]


def test_refined_evidence_runs_inside_the_update_when_applied(tmp_path, monkeypatch) -> None:
    """The update entry point books the deterministic evidence in refined mode
    (plan §5.4's write-back gap) before any LLM chunk."""

    from finesub.llm.knowledge import update as update_module

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    monkeypatch.setattr(update_module, "is_linked_worktree", lambda: False)
    monkeypatch.setattr(
        update_module,
        "build_knowledge_materials",
        lambda **kwargs: SimpleNamespace(
            mode=update_module.MODE_REFINED_ALIGNED,
            warnings=[],
            chunks=[
                SimpleNamespace(
                    windows=[
                        SimpleNamespace(
                            chunk_id="w1", raw_csv="残表が出た",
                            final_csv="散兵が出た", refined_csv="散兵登场了",
                        )
                    ]
                )
            ],
        ),
    )
    # Stop right after the write-back: fingerprinting pulls routing/config.
    monkeypatch.setattr(
        update_module, "_task_fingerprint",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("stop here")),
    )
    final = tmp_path / "x-final.srt"
    final.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")
    refined = tmp_path / "refined.srt"
    refined.write_text("1\n00:00:00,000 --> 00:00:01,000\n散兵登场了\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stop here"):
        update_module.run_knowledge_update(
            final_srt=final, refined_srt=refined, knowledge_root=tmp_path,
            execute=True, apply=True, client=object(), task_id="t1",
        )
    rows = repo.store.conn.execute("SELECT verdict FROM evidence").fetchall()
    assert [r["verdict"] for r in rows] == ["confirmed"]


# ---------------------------------------------------------------------------
# report


def test_report_lists_counts_admissions_and_conversion(tmp_path) -> None:
    from finesub.llm.knowledge.node.matching import ExactIndex, log_matched

    repo = KnowledgeRepo.open(tmp_path)
    store = repo.store
    _seed(store)
    index = ExactIndex.build(store, 1)
    # three windows: matched + exposed(correction), never landed -> admitted
    for window in ("w1", "w2", "w3"):
        log_matched(store, index.scan("残表が出た"), task_id="t", window_id=window, rev=1)
        log_exposed_entries(
            repo, ["原神"],
            task_id="t", window_id=window, window_text="残表が出た", rev=1,
        )
    record_evidence(
        store, node_id="T", field_path="items/I1", value_hash=digest("残表"),
        verdict="refuted", evidence_kind="refined_srt", task_id="t", span="w1",
    )
    lines = "\n".join(build_report(store, rev=1))
    assert "## 原神" in lines
    assert "matched 3" in lines and "correction 3" in lines
    assert "refuted 1" in lines and "latest" in lines
    assert "残表 → スカラマシュ" in lines  # high false-trigger admission
    assert "never matched: ユプカ竜" in lines
    assert "exact-1: matched 3 → exposed 3 → landed 0" in lines
    # one landed window breaks the admission
    log_landed_windows(store, [("w2", "残表が出た", "散兵が出た")], task_id="t", rev=1)
    lines = "\n".join(build_report(store, rev=1))
    assert "残表 → スカラマシュ" not in lines
    assert build_report(store, rev=1, subject="不存在")[0].startswith("no subject")


def test_report_cli_prints_and_filters(tmp_path, capsys) -> None:
    from finesub.llm.knowledge.report import main

    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    assert main(["--root", str(tmp_path), "--subject", "原神"]) == 0
    out = capsys.readouterr().out
    assert "knowledge signals report — rev 1" in out


def test_report_joins_agent_exposures_into_the_correction_denominator(tmp_path) -> None:
    """Agent-side exposures arrive as opportunity=context (the tool server has
    no window text); the report derives the correction split from the join
    with the shadow scan — same node, same task, same window (review
    2026-08-27). Without this the admission only ever saw the REST path."""

    from finesub.llm.knowledge.node.matching import ExactIndex, log_matched
    from finesub.llm.knowledge.node.signals import log_exposed_nodes

    repo = KnowledgeRepo.open(tmp_path)
    store = repo.store
    _seed(store)
    index = ExactIndex.build(store, 1)
    for window in ("w1", "w2", "w3"):
        log_matched(store, index.scan("残表が出た"), task_id="run", window_id=window, rev=1)
        # what the MCP server books for a kb_read serving this window
        log_exposed_nodes(
            store, [("S", "T")], task_id="run", window_id=window, rev=1
        )
    lines = "\n".join(build_report(store, rev=1))
    assert "残表 → スカラマシュ" in lines  # admitted purely from agent exposures
    assert "correction 3" in lines


def test_report_separates_stale_value_evidence(tmp_path) -> None:
    """Evidence sticks to the value it judged (§5.3): after the item value is
    rewritten, the old claim's rows must not decorate the new value."""

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        record_evidence(
            store, node_id="T", field_path="items/I1", value_hash=digest("残表"),
            verdict="confirmed", evidence_kind="refined_srt", task_id="t", span="w1",
        )
        with store.begin("user") as txn:
            txn.update_item("I1", value="新誤聴")
        lines = "\n".join(build_report(store))
        assert "items/I1 (新誤聴) [stale value]: confirmed 1" in lines
        record_evidence(
            store, node_id="T", field_path="items/I1", value_hash=digest("新誤聴"),
            verdict="refuted", evidence_kind="refined_srt", task_id="t", span="w2",
        )
        lines = "\n".join(build_report(store))
        assert "items/I1 (新誤聴): confirmed 0 · refuted 1" in lines  # current-value row
        assert "[stale value]: confirmed 1" in lines                  # history stays visible


def test_report_join_is_rev_scoped(tmp_path) -> None:
    """Correction task ids are stable filenames: a rerun of the same material
    at a newer knowledge revision must not join rev-1 matches with rev-2
    exposures into a correction opportunity (review 2026-08-27 round 6)."""

    from finesub.llm.knowledge.node.matching import ExactIndex, log_matched
    from finesub.llm.knowledge.node.signals import log_exposed_nodes

    repo = KnowledgeRepo.open(tmp_path)
    store = repo.store
    _seed(store)
    with store.begin("user") as txn:  # advance to rev 2
        txn.update_node("S", visibility="shareable")
    index = ExactIndex.build(store, 1)
    log_matched(store, index.scan("残表が出た"), task_id="run", window_id="w1", rev=1)
    log_exposed_nodes(store, [("S", "T")], task_id="run", window_id="w1", rev=2)
    lines = "\n".join(build_report(store, rev=2))
    assert "exposed 1 (correction 0)" in lines  # rev-1 match does not qualify rev-2 exposure
    assert "残表 → スカラマシュ" not in lines


def test_book_revision_evidence_user_and_transcript(tmp_path) -> None:
    """Plan §11.4 / O8-O9: a user apply endorses the values it set; a harness
    apply records revision-level transcript provenance. Booked per semantic
    payload field and per item, deduped on replay."""

    from finesub.llm.knowledge.node.signals import book_revision_evidence

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        with store.begin("user") as txn:
            txn.update_node(
                "T",
                payload={"surface": "スカラマシュ", "zh": "散兵", "desc": "新描述"},
            )
            txn.create_item("I3", "T", "aliases", "国崩")
            rev = txn.rev
        booked = book_revision_evidence(store, rev, evidence_kind="user", task_id="edit-1")
        assert booked == 4  # surface + zh + desc + the new item
        rows = {
            (r["field_path"], r["evidence_kind"], r["verdict"])
            for r in store.conn.execute("SELECT field_path, evidence_kind, verdict FROM evidence")
        }
        assert ("payload.desc", "user", "confirmed") in rows
        assert ("items/I3", "user", "confirmed") in rows
        # replay books nothing new
        assert book_revision_evidence(store, rev, evidence_kind="user", task_id="edit-1") == 0


def test_unverifiable_verdict_and_claim_state(tmp_path) -> None:
    from finesub.llm.knowledge.node.signals import record_evidence

    with KnowledgeStore(tmp_path / "kb.sqlite") as store:
        _seed(store)
        record_evidence(
            store, node_id="T", field_path="payload.zh", value_hash=digest("散兵"),
            verdict="unverifiable", evidence_kind="external", task_id="verify-1",
        )
        record_evidence(
            store, node_id="T", field_path="items/I1", value_hash=digest("残表"),
            verdict="refuted", evidence_kind="refined_srt", task_id="t",
        )
        lines = "\n".join(build_report(store))
    assert "state=unverifiable" in lines
    assert "state=suspect" in lines
