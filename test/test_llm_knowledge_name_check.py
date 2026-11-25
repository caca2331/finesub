"""The name check (docs/knowledge.md「中文定名的核查」): which lines are owed a check, landing a session's
corrections, the repair round, the not-found mark, the check note a person reads, `knowledge verify`'s term object,
and a task's check step (clues, the lines sent, batches, the ledger).

No model is called: the session's answer is given.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from finesub.llm.knowledge import name_check, name_check_step
from finesub.llm.knowledge.base import KnowledgeRun, bind_knowledge_run, load_entry_texts, match_terms, pinned_generation_rev
from finesub.llm.knowledge.name_check import (
    book_name_checks,
    build_claims,
    due_lines,
    parse_name_check_answer,
    parse_name_check_results,
    run_name_check_session,
)
from finesub.llm.knowledge.feedback import NameCheckRecord
from finesub.llm.knowledge.node.model import digest
from finesub.llm.knowledge.node.name_trust import (
    CHECKED,
    HUMAN,
    RECHECK_GAP,
    UNCHECKED,
    UNCHECKED_MARK,
    UNCHECKED_NOT_FOUND_MARK,
    not_found_wait,
    zh_state,
)
from finesub.llm.knowledge.node.render import render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import book_revision_evidence, record_evidence


@pytest.fixture(autouse=True)
def _no_run_left_bound():
    yield
    bind_knowledge_run(KnowledgeRun())


def _seed(root) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    """原神: a wrong placeholder name and a line whose name contains it, a right name, a short name, a line
    without a name, a tentative one; 主播A: a streamer's nickname."""

    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common",
                                         "entry_type": "游戏", "native_names": ["原神"], "section_order": ["至冬"]})
        txn.create_node("U", "term", {"surface": "クルースニク", "zh": "克鲁斯尼克", "desc": "至冬的能量，音译，未定名。"})
        txn.create_membership("MU", "S", "U", "至冬", 0)
        txn.create_node("D", "term", {"surface": "クルースニク晶鉱", "zh": "克鲁斯尼克晶矿", "desc": "矿石。"})
        txn.create_membership("MD", "S", "D", "至冬", 1)
        txn.create_node("C", "term", {"surface": "アナスタシア", "zh": "阿纳斯塔西娅", "desc": "至冬女皇的本名。"})
        txn.create_membership("MC", "S", "C", "至冬", 2)
        txn.create_node("Z", "term", {"surface": "フィリン", "zh": "", "desc": "地名。"})
        txn.create_membership("MZ", "S", "Z", "至冬", 3)
        txn.create_node("T", "term", {"surface": "グレイ", "zh": "格赖埃", "desc": "暂定。"}, maturity="tentative")
        txn.create_membership("MT", "S", "T", "至冬", 4)
    return repo


def _book(repo: KnowledgeRepo, node_id: str, verdict: str, *, days_ago: float = 0, span: str | None = None,
          kind: str = "external", value: str | None = None, task: str | None = None) -> None:
    node = repo.store.node(node_id)
    assert node is not None
    assert record_evidence(repo.store, node_id=node_id, field_path="payload.zh",
                           value_hash=digest(node.payload["zh"] if value is None else value), verdict=verdict,
                           evidence_kind=kind, task_id=task or f"t-{verdict}-{days_ago}-{span}",
                           source_ref="https://example.org/p" if verdict != "unverifiable" else None, span=span)
    when = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    repo.store.conn.execute("UPDATE evidence SET created_at=? WHERE evidence_id=last_insert_rowid()", (when,))


def _state(repo: KnowledgeRepo, node_id: str) -> str | None:
    node = repo.store.node(node_id)
    assert node is not None
    return zh_state(repo.store, node)


def _answer(*rows: dict) -> str:
    return "<name_check_results>\n" + json.dumps(list(rows), ensure_ascii=False) + "\n</name_check_results>"


def _land(repo: KnowledgeRepo, node_ids, answer: str, *, task_id: str = "task-1"):  # type: ignore[no-untyped-def]
    claims = build_claims(repo.store, list(node_ids))
    return claims, book_name_checks(repo.store, parse_name_check_results(answer, claims), claims, task_id=task_id)


def _zh_evidence(repo: KnowledgeRepo, node_id: str) -> list[tuple[str, str, str]]:
    return [(row["value_hash"], row["verdict"], row["evidence_kind"]) for row in repo.store.conn.execute(
        "SELECT value_hash, verdict, evidence_kind FROM evidence WHERE node_id=? AND field_path='payload.zh'"
        " AND evidence_kind='external' ORDER BY evidence_id", (node_id,))]


# ---- the session's lines ----------------------------------------------------------------------------------------


def test_a_checked_name_brings_the_lines_whose_name_contains_it(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U", "C"])
    assert [(c.node_id, c.derived_from) for c in claims] == [("U", None), ("C", None), ("D", "n1")]
    first = build_claims(repo.store, ["D", "U"])
    assert [(c.node_id, c.derived_from) for c in first] == [("D", "n2"), ("U", None)], "asked for first, still marked"
    deep = _seed(tmp_path / "deep")
    with deep.store.begin("import") as txn:
        txn.create_node("V", "term", {"surface": "クルースニク晶鉱脈", "zh": "克鲁斯尼克晶矿脉", "desc": "矿脉。"})
        txn.create_membership("MV", "S", "V", "至冬", 5)
    for order in (["V", "D", "U"], ["U", "D", "V"]):
        picked = build_claims(deep.store, order)
        parent = {c.node_id: c.derived_from for c in picked}
        by_claim = {c.claim_id: c.node_id for c in picked}
        assert by_claim[parent["V"]] == "D" and by_claim[parent["D"]] == "U" and parent["U"] is None, order
    tie = _seed(tmp_path / "tie")
    with tie.store.begin("import") as txn:
        for i, (nid, zh) in enumerate([("A", "天空城"), ("B", "空城堡"), ("C2", "天空城堡")]):
            txn.create_node(nid, "term", {"surface": nid, "zh": zh, "desc": "x"})
            txn.create_membership("M" + nid, "S", nid, "至冬", 10 + i)
    for order in (["A", "B", "C2"], ["B", "A", "C2"], ["C2", "B", "A"]):
        picked = build_claims(tie.store, order)
        by_claim = {c.claim_id: c.node_id for c in picked}
        assert by_claim[next(c.derived_from for c in picked if c.node_id == "C2")] == "A", order
    with tie.store.begin("import") as txn:
        txn.create_node("A2", "term", {"surface": "A2", "zh": "天空城", "desc": "同名的另一行"})
        txn.create_membership("MA2", "S", "A2", "至冬", 20)
    for order in (["A", "A2", "C2"], ["A2", "A", "C2"]):
        picked = build_claims(tie.store, order)
        assert next(c.derived_from for c in picked if c.node_id == "C2") is None, "two parents of one name: on its own"
    assert claims[0].subject == "原神" and claims[0].desc == "至冬的能量，音译，未定名。"
    short = _seed(tmp_path / "short")
    with short.store.begin("harness") as txn:
        txn.update_node("U", payload={**short.store.node("U").payload, "zh": "克鲁"})
        txn.update_node("D", payload={**short.store.node("D").payload, "zh": "克鲁晶矿"})
    assert [c.node_id for c in build_claims(short.store, ["U"])] == ["U"], "a two-character name brings none"
    _book(repo, "D", "confirmed")
    _book(repo, "D", "confirmed", days_ago=30)
    assert _state(repo, "D") == CHECKED
    assert [c.node_id for c in build_claims(repo.store, ["U"])] == ["U", "D"], "a derived line comes, owed a check or not"
    with repo.store.begin("user") as txn:
        txn.update_node("D", payload={**repo.store.node("D").payload, "zh": "克鲁斯尼克结晶石"})
    book_revision_evidence(repo.store, repo.rev, evidence_kind="user")
    assert _state(repo, "D") == HUMAN
    assert [c.node_id for c in build_claims(repo.store, ["U"])] == ["U"], "a person's line does not come"
    acked = _seed(tmp_path / "acked")
    acked.store.conn.execute("INSERT INTO line_acks(node_id, rev, value, created_at) VALUES ('D', ?, '', ?)",
                             (acked.rev, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    assert _state(acked, "D") == HUMAN
    assert [c.node_id for c in build_claims(acked.store, ["U"])] == ["U"], "nor one a person acked"


def _record(found: str = "克什尼克", *, source: str = "クルースニク", kb_zh: str = "克鲁斯尼克", url: str = "https://w/k",
            origin: str = "window w1", **kwargs) -> NameCheckRecord:
    return NameCheckRecord(source=source, kb_zh=kb_zh, found=found, url=url if found else "", origin=origin, **kwargs)


def test_records_find_their_line_by_source_word_and_library_name_and_the_rest_are_counted(tmp_path) -> None:
    repo = _seed(tmp_path)
    repo.store.conn.execute("INSERT INTO line_acks(node_id, rev, value, created_at) VALUES ('C', ?, '', ?)",
                            (repo.rev, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    assert _state(repo, "C") == HUMAN
    followed: list[str] = []
    clues = name_check_step.collect_clues(repo.store, [
        _record(kb_zh="克魯斯尼克", url="https://vertexaisearch.cloud.google.com/r/1", quote="至冬的克什尼克……"),
        _record("克鲁斯尼克", origin="prepare"),
        _record("", query="クルースニク 中文"),
        _record("克什尼克", url="https://vertexaisearch.cloud.google.com/r/1", origin="window w2"),  # the same again
        _record("库鲁斯尼克", url="https://w/other"),
        _record(source="ない", kb_zh="无"),
        _record(source="アナスタシア", kb_zh="阿纳斯塔西娅"),
        _record(source="グレイ", kb_zh="格赖埃"),
    ])
    assert list(clues.by_line) == ["U"] and clues.counts == {"U": 5} and clues.disputed == {"U"}
    assert clues.dropped == {"no_line": 1, "human": 1, "not_sendable": 1}, "unmatched, a person's line, a tentative one"
    assert [clue.get("found", "") for clue in clues.by_line["U"]] == ["克什尼克", "库鲁斯尼克", "克鲁斯尼克"], \
        "at most three, disagreeing spellings first, one per spelling and page"
    assert clues.by_line["U"][0] == {"found": "克什尼克", "url": "https://vertexaisearch.cloud.google.com/r/1",
                                     "quote": "至冬的克什尼克……", "origin": "window w1"}
    claims = name_check_step.follow_clue_links(
        build_claims(repo.store, ["U"], clues=clues.by_line),
        follow=lambda url: followed.append(url) or url.replace("https://vertexaisearch.cloud.google.com/r/1",
                                                                "https://page/1"))
    assert followed == ["https://vertexaisearch.cloud.google.com/r/1", "https://w/other", "https://w/k"], \
        "once per link"
    assert claims[0].clues[0]["url"] == "https://page/1"
    prompt = name_check.render_name_check_prompt(claims)
    assert "线索（窗口 w1）：搜到写法「克什尼克」，页面 https://page/1，摘录「至冬的克什尼克……」" in prompt
    assert "线索（准备任务）：搜到写法「克鲁斯尼克」" in prompt and "先看线索给的页面" in prompt
    searched = build_claims(repo.store, ["U"], clues={"U": [{"query": "クルースニク 中文", "origin": "prepare"}]})
    assert "线索（准备任务）：搜了「クルースニク 中文」，没搜到中文写法" in name_check.render_name_check_prompt(searched)


def test_without_clues_the_prompt_says_nothing_of_them(tmp_path) -> None:
    """`knowledge verify` has no clues: the paragraph on them is not there, the rest reads as before it existed."""

    repo = _seed(tmp_path)
    prompt = name_check.render_name_check_prompt(build_claims(repo.store, ["U"]))
    assert "线索" not in prompt
    assert "原文对着第一列那种语言的官方或权威来源核。\n- 每一行给一种结论：" in prompt


# ---- landing ----------------------------------------------------------------------------------------------------


def test_a_correction_lands_as_a_harness_revision_setting_only_what_it_names(tmp_path) -> None:
    repo = _seed(tmp_path)
    before = {node_id: dict(repo.store.node(node_id).payload) for node_id in ("U", "D")}
    answer = _answer(
        {"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克", "desc": "至冬的能量。", "label": "x"},
         # a model's extra keys reach nothing: only the source word, the name and the description are ever set
         "url": "https://wiki.example/k", "surface": "改掉原文"},
        {"id": "n2", "verdict": "corrected", "set": {"zh": "克什尼克晶矿"}, "url": "https://wiki.example/k"},
    )
    _claims, report = _land(repo, ["U"], answer)
    assert [row["claim_id"] for row in report.corrected] == ["n1", "n2"] and report.rev is not None
    for node_id, name in (("U", "克什尼克"), ("D", "克什尼克晶矿")):
        node = repo.store.node(node_id)
        changed = {key for key in node.payload if node.payload[key] != before[node_id].get(key)}
        assert changed <= {"zh", "desc"} and node.payload["zh"] == name
        # the old name refuted; the new one gets no confirmation from the session that wrote it
        assert _zh_evidence(repo, node_id) == [(digest(before[node_id]["zh"]), "refuted", "external")]
        assert _state(repo, node_id) == UNCHECKED and node_id in due_lines(repo.store, limit=10)
    assert repo.store.node("U").payload["desc"] == "至冬的能量。", "the 未定名 wording is gone"
    revision = repo.store.conn.execute("SELECT kind, note, task_id FROM revisions ORDER BY rev DESC LIMIT 1").fetchone()
    assert (revision["kind"], revision["note"], revision["task_id"]) == ("harness", "kb-name-check", "task-1")
    lines = name_check.landing_lines(report.to_dict())
    assert "修正：「原神」クルースニク|克鲁斯尼克：定名 克鲁斯尼克 → 克什尼克；描述改为「至冬的能量。」（https://wiki.example/k）" in lines


def test_a_line_left_behind_by_its_renamed_parent_is_marked_for_the_person(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U"])
    assert [(c.node_id, c.along) for c in claims] == [("U", False), ("D", True)]
    answer = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://wiki.example/k"})
    report = book_name_checks(repo.store, parse_name_check_results(answer, claims), claims, task_id="t")
    lines = name_check.landing_lines(report.to_dict())
    assert "没跟着改：「原神」クルースニク晶鉱|克鲁斯尼克晶矿：它所随的 クルースニク|克鲁斯尼克 改了名，这一行这次没改，要改请自己改库" in lines
    assert not any(line.startswith("没有可用结论") for line in lines)
    failed = _seed(tmp_path / "failed")
    claims = build_claims(failed.store, ["U"])
    searched = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://wiki.example/k"},
                       {"id": "n2", "verdict": "not_found", "kind": "search_failed", "note": "搜索工具报错"})
    report = book_name_checks(failed.store, parse_name_check_results(searched, claims), claims, task_id="t")
    assert [row["line"] for row in report.left_behind] == ["クルースニク晶鉱|克鲁斯尼克晶矿"], \
        "a search that failed is no answer: the line still keeps the old name"
    repo2 = _seed(tmp_path / "held")
    claims = build_claims(repo2.store, ["U"])
    held = book_name_checks(repo2.store, parse_name_check_results(answer, claims), claims, task_id="t", write=False)
    assert not any(line.startswith("没跟着改") for line in name_check.landing_lines(held.to_dict())), \
        "the rename did not land (the lock was elsewhere): nothing to tell the person to fix"


def test_a_batch_names_no_line_of_another_batch(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U", "D"], derive_for=set())
    assert [(c.node_id, c.along, c.derived_from) for c in claims] == [("U", False, None), ("D", False, "n1")]
    out = name_check_step.still_sendable(repo.store, claims[1:])
    assert [(c.claim_id, c.derived_from) for c in out] == [("n2", None)], "D goes for itself, its hint U elsewhere"


def test_a_correction_of_the_source_word_or_description_confirms_the_name(tmp_path) -> None:
    repo = _seed(tmp_path)
    _claims, report = _land(repo, ["C"], _answer(
        {"id": "n1", "verdict": "corrected", "set": {"source": "アナスタシア・ロマノヴァ"}, "url": "https://w/a",
         "note": "日文官网写全名"}))
    node = repo.store.node("C")
    assert node.payload["surface"] == "アナスタシア・ロマノヴァ" and node.payload["zh"] == "阿纳斯塔西娅"
    assert _zh_evidence(repo, "C") == [(digest("阿纳斯塔西娅"), "confirmed", "external")]
    assert "原文 アナスタシア → アナスタシア・ロマノヴァ" in name_check.landing_lines(report.to_dict())[0]


def test_a_corrected_that_changes_nothing_is_a_confirmation_of_the_whole_line(tmp_path) -> None:
    """A spelling the same name has in traditional or full-width form is no change."""

    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "corrected", "set": {"zh": "阿納斯塔西婭"},
                                             "url": "https://w/a"}), claims)
    assert [(row["verdict"], row.get("set")) for row in rows] == [("confirmed", None)]
    report = book_name_checks(repo.store, rows, claims, task_id="t")
    assert [row["claim_id"] for row in report.confirmed] == ["n1"]
    core = repo.store.conn.execute("SELECT verdict FROM evidence WHERE node_id='C' AND field_path='payload:core'").fetchall()
    assert [row["verdict"] for row in core] == ["confirmed"]


def test_a_confirmed_changes_nothing_and_a_value_that_breaks_the_line_voids_the_row(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U", "C"])
    rows = parse_name_check_results(_answer(
        {"id": "n1", "verdict": "confirmed", "set": {"zh": "克什尼克"}, "url": "https://w/k"},
        {"id": "n2", "verdict": "corrected", "set": {"zh": "安娜|丝塔夏", "desc": "女皇。"}, "url": "https://w/a"},
        {"id": "n3", "verdict": "corrected", "set": {"source": "クルースニク\n晶鉱"}, "url": "https://w/b"}), claims)
    assert [row["claim_id"] for row in parse_name_check_results(_answer(
        {"id": "n2", "verdict": "corrected", "set": {"zh": ""}, "url": "https://w/a"},
        {"id": "n1", "verdict": "corrected", "set": {"zh": None, "desc": "能量。"}, "url": "https://w/k"}), claims)] == [], \
        "a blank or non-string name is no conclusion either"
    assert parse_name_check_results(_answer({"id": "n1", "verdict": "corrected", "set": [{"zh": "克什尼克"}],
                                             "url": "https://w/k"}), claims) == [], "nor a set that is not an object"
    assert parse_name_check_results(_answer(
        {"id": "n1", "verdict": "corrected", "set": None, "url": "https://w/k"},
        {"id": "n2", "verdict": "corrected", "set": {"alias": "x"}, "url": "https://w/a"},
        {"id": "n3", "verdict": "corrected", "url": "https://w/b"}), claims) == [], \
        "nor a corrected whose set is null, missing, or names nothing it may set"
    assert [(row["claim_id"], row.get("set")) for row in rows] == [("n1", None)], \
        "a confirmed sets nothing; a broken value is no conclusion, not a confirmation of the old name"
    rows = parse_name_check_results(_answer(
        {"id": "n2", "verdict": "corrected", "set": {"desc": "女皇的本名|真名。"}, "url": "https://w/a"}), claims)
    assert rows[0]["set"] == {"desc": "女皇的本名|真名。"}, "a description may hold a |"


def test_a_not_found_may_still_correct_the_description_but_not_the_name(tmp_path) -> None:
    repo = _seed(tmp_path)
    _claims, report = _land(repo, ["C"], _answer({"id": "n1", "verdict": "not_found", "kind": "temporary",
                                                  "set": {"zh": "别的名字", "desc": "至冬女皇。"}}))
    node = repo.store.node("C")
    assert node.payload["zh"] == "阿纳斯塔西娅" and node.payload["desc"] == "至冬女皇。"
    assert [row["claim_id"] for row in report.not_found] == ["n1"] == [row["claim_id"] for row in report.corrected]
    assert [verdict for _hash, verdict, _kind in _zh_evidence(repo, "C")] == ["unverifiable"]


def test_a_correction_not_written_this_time_leaves_the_line_owed_a_check(tmp_path, monkeypatch) -> None:
    """The lock held elsewhere, or a race the store rolls back: no refutation (it would be final), the line comes
    round again. A value the validator refuses is final: refuted, listed."""

    from finesub.llm.knowledge.node import apply

    answer = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"})
    repo = _seed(tmp_path / "locked")
    claims = build_claims(repo.store, ["U"])
    report = book_name_checks(repo.store, parse_name_check_results(answer, claims), claims, task_id="t", write=False)
    assert [row.get("retry") for row in report.blocked] == [True] and _zh_evidence(repo, "U") == []
    assert "U" in due_lines(repo.store, limit=10)
    assert any("下次核查会再来" in line for line in name_check.landing_lines(report.to_dict()))

    for name, result, final in (
        ("race", apply.ApplyResult(rev=None, rolled_back=True, rollback_reason="stale write during commit: x"), False),
        ("refused", apply.ApplyResult(rev=None, rejected_ops=[(0, "zh 不合格")]), True),
    ):
        repo = _seed(tmp_path / name)
        monkeypatch.setattr(apply, "apply_envelope", lambda *args, _r=result, **kwargs: _r)
        claims = build_claims(repo.store, ["U"])
        report = book_name_checks(repo.store, parse_name_check_results(answer, claims), claims, task_id="t")
        assert repo.store.node("U").payload["zh"] == "克鲁斯尼克" and len(report.blocked) == 1
        assert ("U" in due_lines(repo.store, limit=10)) is not final, name
        assert ((digest("克鲁斯尼克"), "refuted", "external") in _zh_evidence(repo, "U")) is final, name


def test_a_write_landing_first_drops_the_correction_and_leaves_the_line_owed_a_check(tmp_path, monkeypatch) -> None:
    """The common race: another writer in this process commits a new version of the line between the landing's read
    and its apply, so the store drops the update as stale -- no rollback, no rejection, nothing written. That is
    not a correction: no refutation, the line comes round again."""

    from finesub.llm.knowledge.node import apply

    repo = _seed(tmp_path)
    real = apply.apply_envelope

    def interleaved(store, envelope, **kwargs):  # type: ignore[no-untyped-def]
        with store.begin("harness") as txn:
            txn.update_node("U", payload={**store.node("U").payload, "desc": "别人刚改过。"})
        return real(store, envelope, **kwargs)

    monkeypatch.setattr(apply, "apply_envelope", interleaved)
    _claims, report = _land(repo, ["U"], _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"},
                                                  "url": "https://w/k"}))
    assert repo.store.node("U").payload["zh"] == "克鲁斯尼克" and report.corrected == []
    assert [row.get("retry") for row in report.blocked] == [True] and _zh_evidence(repo, "U") == []
    assert "U" in due_lines(repo.store, limit=10)


def test_lines_given_no_usable_conclusion_are_reported(tmp_path) -> None:
    """A line sent along (its name contains a sent one's) is owed an answer only when that one is renamed."""

    repo = _seed(tmp_path)
    claims, report = _land(repo, ["U", "C"], _answer({"id": "n1", "verdict": "confirmed", "url": "not a url"}))
    assert [c.along for c in claims] == [False, False, True]
    assert len(report.unanswered) == 2
    assert any(line.startswith("没有可用结论（没答或答得不合格）2 行") for line in name_check.landing_lines(report.to_dict()))
    renamed = _seed(tmp_path / "renamed")
    _claims, report = _land(renamed, ["U"], _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"},
                                                     "url": "https://w/k"}))
    assert report.unanswered == [], "the line sent along it is left behind, not left without a conclusion"
    assert [row["line"] for row in report.left_behind] == ["クルースニク晶鉱|克鲁斯尼克晶矿"]


def test_a_line_the_batch_size_left_out_is_marked_once_its_parent_is_renamed(tmp_path) -> None:
    """A batch past its size leaves the last lines sent along out; a 已核 one is not due again, so once the parent's
    new name lands it is marked for the person like a line sent along and left unanswered."""

    repo = _seed(tmp_path)
    with repo.store.begin("import") as txn:
        for i in range(8):
            txn.create_node(f"D{i}", "term", {"surface": f"クルースニク{i}号", "zh": f"克鲁斯尼克{i}号", "desc": "矿石。"})
            txn.create_membership(f"MD{i}", "S", f"D{i}", "至冬", 10 + i)
    claims = build_claims(repo.store, ["U"])
    (batch,) = name_check_step.plan_batches(claims, 8)
    left_out = {claim.node_id for claim in claims} - {claim.node_id for claim in batch}
    assert len(batch) == 8 and len(left_out) == 2
    answer = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://wiki.example/k"})
    report = book_name_checks(repo.store, parse_name_check_results(answer, batch), batch, task_id="t")
    marked = {row["line"] for row in report.left_behind}
    assert {f"{repo.store.node(i).payload['surface']}|{repo.store.node(i).payload['zh']}" for i in left_out} <= marked
    assert all(row["parent"] == "クルースニク|克鲁斯尼克" for row in report.left_behind)


def test_a_humans_line_is_listed_not_changed(tmp_path) -> None:
    repo = _seed(tmp_path)
    with repo.store.begin("user") as txn:
        txn.update_node("C", payload={**repo.store.node("C").payload, "zh": "安娜斯塔西娅"})
    book_revision_evidence(repo.store, repo.rev, evidence_kind="user")
    assert _state(repo, "C") == HUMAN
    _claims, report = _land(repo, ["C"], _answer({"id": "n1", "verdict": "corrected", "url": "https://w/a",
                                                  "set": {"zh": "阿纳斯塔西娅", "desc": "女皇。"}}))
    node = repo.store.node("C")
    assert node.payload["zh"] == "安娜斯塔西娅" and node.payload["desc"] == "至冬女皇的本名。" and report.corrected == []
    assert [row["reason"] for row in report.blocked] == [HUMAN]
    assert (digest("安娜斯塔西娅"), "refuted", "external") in _zh_evidence(repo, "C")
    assert _state(repo, "C") == HUMAN
    assert any("没改（人定）" in line for line in name_check.landing_lines(report.to_dict()))


def test_a_checked_name_can_be_overturned(tmp_path) -> None:
    repo = _seed(tmp_path)
    _book(repo, "C", "confirmed", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "C", "confirmed")
    assert _state(repo, "C") == CHECKED
    _land(repo, ["C"], _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "阿娜斯塔西娅"}, "url": "https://w/a"}))
    assert repo.store.node("C").payload["zh"] == "阿娜斯塔西娅" and _state(repo, "C") == UNCHECKED


def test_a_not_found_books_its_kind_and_marks_the_line(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    _land(repo, ["C"], _answer({"id": "n1", "verdict": "not_found", "kind": "permanent", "note": "只在直播里"}))
    row = repo.store.conn.execute("SELECT verdict, span, source_ref FROM evidence WHERE node_id='C'").fetchone()
    assert (row["verdict"], row["span"], row["source_ref"]) == ("unverifiable", "permanent", None)
    assert _state(repo, "C") == UNCHECKED


def test_a_not_found_put_down_to_failed_searching_books_nothing_and_stays_owed(tmp_path) -> None:
    """A Codex session once wrote that it could not use its search tool and judged 暂时: one failed search held
    the line back 30 days. Put down to the search, the answer books nothing and the line goes again."""

    repo = _seed(tmp_path)
    answer = _answer({"id": "n1", "verdict": "not_found", "kind": "search_failed", "note": "搜索工具报错",
                      "set": {"desc": "不该落库的描述"}})
    claims, report = _land(repo, ["C"], answer)
    assert repo.store.conn.execute("SELECT COUNT(*) FROM evidence WHERE node_id='C'").fetchone()[0] == 0
    assert [row["line"] for row in report.search_failed] and not report.not_found and not report.corrected
    assert not report.unanswered, "it answered; the answer is that it could not search"
    assert "C" in due_lines(repo.store, limit=10)
    assert any(line.startswith("搜索失败、没记：") and "搜索工具报错" in line
               for line in name_check.landing_lines(report.to_dict()))
    # only a session that searches itself is told it may answer so; on the local route the harness searched
    assert "search_failed" in name_check.render_name_check_prompt(claims)
    assert "search_failed" not in name_check.render_name_check_prompt(claims, pages="")


# ---- the scan ---------------------------------------------------------------------------------------------------


def test_the_scan_skips_lines_without_a_name_tentative_ones_and_checked_ones(tmp_path) -> None:
    repo = _seed(tmp_path)
    assert set(due_lines(repo.store, limit=10)) == {"U", "D", "C"}
    _book(repo, "C", "confirmed", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "C", "confirmed")
    assert "C" not in due_lines(repo.store, limit=10), "已核 is not owed a check"


def test_one_task_counts_once_however_far_apart_its_rows(tmp_path) -> None:
    repo = _seed(tmp_path)
    _book(repo, "C", "confirmed", kind="refined_srt", task="refine-1", span="w1", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "C", "confirmed", kind="refined_srt", task="refine-1", span="w2")  # the same refined subtitle, resumed
    assert _state(repo, "C") == UNCHECKED
    _book(repo, "C", "confirmed", task="check-2")
    assert _state(repo, "C") == CHECKED


def test_a_refutation_and_a_permanent_not_found_are_terminal(tmp_path) -> None:
    repo = _seed(tmp_path)
    _book(repo, "U", "refuted", days_ago=0)  # a refutation counts only if booked after the version was written
    _book(repo, "C", "unverifiable", span="permanent", days_ago=400)
    assert set(due_lines(repo.store, limit=10)) == {"D"}


def test_one_confirmation_waits_out_the_gap_before_the_second_check(tmp_path) -> None:
    repo = _seed(tmp_path)
    _book(repo, "C", "confirmed", days_ago=RECHECK_GAP.days - 1)
    assert "C" not in due_lines(repo.store, limit=10)
    repo2 = _seed(tmp_path / "later")
    _book(repo2, "C", "confirmed", days_ago=RECHECK_GAP.days)
    assert "C" in due_lines(repo2.store, limit=10)


@pytest.mark.parametrize("times, waited_days, due", [
    (1, 29, False), (1, 30, True),
    (2, 59, False), (2, 60, True),
    (3, 119, False), (3, 120, True),
    (4, 179, False), (4, 180, True),
    (6, 179, False), (6, 180, True),  # capped at 180
])
def test_a_temporary_not_found_backs_off(tmp_path, times, waited_days, due) -> None:
    repo = _seed(tmp_path)
    for n in range(times):
        _book(repo, "C", "unverifiable", span="temporary", days_ago=waited_days + 400 * (times - 1 - n))
    assert ("C" in due_lines(repo.store, limit=10)) is due
    assert not_found_wait(times) == timedelta(days=min(30 * 2 ** (times - 1), 180))


def test_the_scan_takes_the_longest_unchecked_first(tmp_path) -> None:
    repo = _seed(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S2", "subject", {"surface": "星穹铁道", "intro": "游戏", "category": "common",
                                          "entry_type": "游戏", "native_names": ["星穹铁道"], "section_order": ["人物"]})
        txn.create_node("H", "term", {"surface": "三月なのか", "zh": "三月七", "desc": "角色。"})
        txn.create_membership("MH", "S2", "H", "人物", 0)
    _book(repo, "U", "unverifiable", span="temporary", days_ago=500, task="old")  # due again, checked long ago
    _book(repo, "D", "unverifiable", span="temporary", days_ago=40, task="recent")
    assert due_lines(repo.store, limit=2) == ["C", "H"]
    assert due_lines(repo.store, limit=10) == ["C", "H", "U", "D"]


# ---- the mark ---------------------------------------------------------------------------------------------------


def test_the_not_found_mark_rides_the_current_name_on_every_read_path(tmp_path, monkeypatch) -> None:
    from test.test_llm_knowledge_name_trust import _body, _server, _tool

    repo = _seed(tmp_path)
    _book(repo, "C", "unverifiable", span="temporary")
    _book(repo, "U", "unverifiable", span="permanent", value="旧名字")  # on another value: not this name's
    marked_c = f"アナスタシア|阿纳斯塔西娅||{UNCHECKED_NOT_FOUND_MARK}至冬女皇的本名。"
    marked_u = f"クルースニク|克鲁斯尼克||{UNCHECKED_MARK}至冬的能量，音译，未定名。"
    with pinned_generation_rev(tmp_path, correct_only=True):
        injected = load_entry_texts(tmp_path, ["原神"])[0]["原神"].splitlines()
        (hit,) = [h for h in match_terms(tmp_path, "阿纳斯塔西娅来了") if h.line.startswith("アナスタシア")]
    server = _server(monkeypatch, tmp_path, correct_only=True)
    read = [_body(line) for line in _tool(server, "kb_read", entry="原神")["text"].splitlines()]
    handle = next(line.rsplit("<!-- ", 1)[1].split()[0] for line in _tool(server, "kb_read", entry="原神")["text"]
                  .splitlines() if line.startswith("アナスタシア"))
    node = _tool(server, "kb_read_node", handle=handle)["text"]
    for lines in (injected, read):
        assert marked_c in lines and marked_u in lines
    assert node == marked_c == hit.line
    assert UNCHECKED_NOT_FOUND_MARK not in render_subject(repo.store, "S", mode="prompt")
    _book(repo, "C", "confirmed", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "C", "confirmed")
    assert marked_c.replace(UNCHECKED_NOT_FOUND_MARK, "") in render_subject(
        repo.store, "S", mode="prompt", correct_only=True), "a 已核 line carries no mark"


def test_a_person_reads_each_lines_check_note_and_the_harvest_drops_it(tmp_path) -> None:
    from finesub.llm.knowledge.node.edit import edit_subject

    repo = _seed(tmp_path)
    _book(repo, "U", "confirmed", days_ago=2)
    _book(repo, "D", "unverifiable", span="temporary", days_ago=0)
    _book(repo, "C", "confirmed", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "C", "confirmed")
    text = render_subject(repo.store, "S", mode="human", preview="full")  # what rendered/ holds
    due = (datetime.now(timezone.utc) + timedelta(days=30)).date().isoformat()
    assert "- クルースニク|克鲁斯尼克||至冬的能量，音译，未定名。  <!-- 未核，印证 1 次 -->" in text
    assert f"- クルースニク晶鉱|克鲁斯尼克晶矿||矿石。  <!-- 查不到，{due} 重核 -->" in text
    assert "- アナスタシア|阿纳斯塔西娅||至冬女皇的本名。  <!-- 已核 -->" in text
    assert "<!--" not in next(line for line in text.splitlines() if "フィリン" in line), "no name, no note"
    assert "<!-- 未核" not in render_subject(repo.store, "S", mode="prompt"), "a model never reads it"
    report = edit_subject(repo, repo.store.node("S"), text)
    assert (report.created, report.updated) == (0, 0), "the note is not content"
    assert repo.store.node("U").payload["desc"] == "至冬的能量，音译，未定名。"
    edited = text.replace("矿石。  <!--", "至冬的矿石。  <!--")
    edit_subject(repo, repo.store.node("S"), edited)
    assert repo.store.node("D").payload["desc"] == "至冬的矿石。"
    own = render_subject(repo.store, "S", mode="human", preview="full").replace(
        "至冬女皇的本名。  <!-- 已核 -->", "至冬女皇的本名。 <!-- 未核实的旧称 -->")
    edit_subject(repo, repo.store.node("S"), own)
    assert repo.store.node("C").payload["desc"] == "至冬女皇的本名。 <!-- 未核实的旧称 -->", "only the generated notes go"
    from finesub.llm.knowledge.node.edit import _bare_line

    assert _bare_line("- 至冬的首都在北边。 <!-- 未核 -->") == "至冬的首都在北边。 <!-- 未核 -->", "a free sentence keeps it"
    assert _bare_line("- 甲说欢迎 | 乙说再见。 <!-- 未核 -->") == "甲说欢迎 | 乙说再见。 <!-- 未核 -->", "a | is not a term"


# ---- knowledge verify -------------------------------------------------------------------------------------------


def test_verify_lands_the_name_check_as_the_harness_not_the_user(tmp_path, monkeypatch, capsys) -> None:
    from finesub.llm.knowledge import maintain, verify

    repo = _seed(tmp_path)
    monkeypatch.setattr(maintain, "_role_client", lambda _models: _Client())
    monkeypatch.setattr(verify, "run_verify_session", lambda claims, client: [])
    monkeypatch.setattr(name_check, "run_name_check_session", lambda claims, client, route: _session(
        claims, _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"})))
    assert maintain.main(["--root", str(tmp_path), "verify", "--execute", "--apply", "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "修正：「原神」クルースニク|克鲁斯尼克：定名 克鲁斯尼克 → 克什尼克" in out
    repo = KnowledgeRepo.open(tmp_path)
    assert repo.store.node("U").payload["zh"] == "克什尼克"
    kinds = {row["evidence_kind"] for row in repo.store.conn.execute(
        "SELECT evidence_kind FROM evidence WHERE node_id='U' AND field_path='payload.zh'")}
    assert "user" not in kinds and _state(repo, "U") == UNCHECKED
    assert repo.store.conn.execute("SELECT kind FROM revisions ORDER BY rev DESC LIMIT 1").fetchone()["kind"] == "harness"


class _ProviderError(Exception):
    """A provider's own error type (the Gemini route's derives from Exception, not RuntimeError)."""


@pytest.mark.parametrize("failing", ["facts", "names"])
def test_either_verify_session_failing_with_a_provider_error_spares_the_other(tmp_path, monkeypatch, capsys,
                                                                           failing) -> None:
    from finesub.llm.knowledge import maintain, verify

    _seed(tmp_path)
    monkeypatch.setattr(maintain, "_role_client", lambda _models: _Client())
    fact = {"claim_id": "c1", "kind": "note", "line": "x", "node_id": "C"}
    monkeypatch.setattr(verify, "unverified_claims", lambda store, limit: [fact])
    booked: list = []
    monkeypatch.setattr(verify, "book_verify_results", lambda store, rows, task_id: booked.append(rows) or len(rows))

    def facts(claims, client):  # type: ignore[no-untyped-def]
        if failing == "facts":
            raise _ProviderError("quota exhausted on every route")
        return [{**fact, "verdict": "unverifiable", "url": "", "note": ""}]

    def names(claims, client, route):  # type: ignore[no-untyped-def]
        if failing == "names":
            raise _ProviderError("quota exhausted on every route")
        return _session(claims, _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/k"}))

    monkeypatch.setattr(verify, "run_verify_session", facts)
    monkeypatch.setattr(name_check, "run_name_check_session", names)
    assert maintain.main(["--root", str(tmp_path), "verify", "--execute", "--apply", "--limit", "1"]) == 1
    err = capsys.readouterr().err
    assert "quota exhausted" in err
    if failing == "facts":
        assert not booked and "印证：「原神」クルースニク|克鲁斯尼克" in err
    else:
        assert len(booked) == 1 and "booked 1 evidence row(s)" in err and "印证：" not in err


def test_verify_takes_in_a_persons_pending_edit_before_it_lands(tmp_path, monkeypatch) -> None:
    from finesub.llm.knowledge import maintain, verify

    repo = _seed(tmp_path)
    repo.refresh_rendered()
    path = next((tmp_path / "rendered").rglob("原神*.md"))
    path.write_text(path.read_text(encoding="utf-8").replace("至冬女皇的本名。", "至冬女皇的真名。"), encoding="utf-8")
    monkeypatch.setattr(maintain, "_role_client", lambda _models: _Client())
    monkeypatch.setattr(verify, "run_verify_session", lambda claims, client: [])
    monkeypatch.setattr(name_check, "run_name_check_session", lambda claims, client, route: _session(
        claims, _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"})))
    assert maintain.main(["--root", str(tmp_path), "verify", "--execute", "--apply", "--limit", "1"]) == 0
    repo = KnowledgeRepo.open(tmp_path)
    assert repo.store.node("C").payload["desc"] == "至冬女皇的真名。", "the person's edit went in first"
    assert repo.store.node("U").payload["zh"] == "克什尼克" and not repo.rendered_dirty_files()


def test_corrections_wait_while_rendered_holds_edits_not_taken_in(tmp_path) -> None:
    repo = _seed(tmp_path)
    repo.refresh_rendered()
    path = next((tmp_path / "rendered").rglob("原神*.md"))
    path.write_text(path.read_text(encoding="utf-8").replace("至冬女皇的本名。", "至冬女皇的真名。"), encoding="utf-8")
    claims = build_claims(repo.store, ["U"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"},
                                             "url": "https://w/k"}), claims)
    report = book_name_checks(repo.store, rows, claims, task_id="t", hold=name_check.pending_edits_hold(repo))
    assert repo.store.node("U").payload["zh"] == "克鲁斯尼克" and report.blocked[0]["retry"]
    assert "rendered/" in report.blocked[0]["reason"] and _zh_evidence(repo, "U") == [], "nothing final on the old name"


def test_a_held_correction_books_nothing_so_its_line_comes_back(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "not_found", "kind": "permanent",
                                             "set": {"desc": "女皇。"}}), claims)
    report = book_name_checks(repo.store, rows, claims, task_id="t", hold="rendered/ 里有没收进库的编辑")
    assert report.blocked[0]["retry"] and _zh_evidence(repo, "C") == []
    assert "C" in due_lines(repo.store, limit=10), "not ended by a 根本 not-found booked ahead of its correction"


def test_a_replay_books_the_evidence_a_landed_correction_did_not_reach(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "not_found", "kind": "permanent",
                                             "set": {"desc": "女皇。"}}), claims)
    real = name_check.record_evidence
    monkeypatch.setattr(name_check, "record_evidence", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        book_name_checks(repo.store, rows, claims, task_id="t")
    assert repo.store.node("C").payload["desc"] == "女皇。" and _zh_evidence(repo, "C") == []
    monkeypatch.setattr(name_check, "record_evidence", real)
    report = book_name_checks(repo.store, rows, claims, task_id="t")  # the resumed task lands its kept answer again
    assert report.corrected and _zh_evidence(repo, "C") == [(digest("阿纳斯塔西娅"), "unverifiable", "external")]


def test_a_line_retired_before_the_landing_is_stale_not_a_replay(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "corrected", "set": {"desc": ""},
                                             "url": "https://w/a"}), claims)
    with repo.store.begin("user") as txn:
        txn.tombstone_membership("MC")
        txn.tombstone_node("C")
    report = book_name_checks(repo.store, rows, claims, task_id="t")
    assert report.stale and not report.corrected and _zh_evidence(repo, "C") == []


def test_a_correction_that_races_books_nothing_either(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "not_found", "kind": "permanent",
                                             "set": {"desc": "女皇。"}}), claims)
    monkeypatch.setattr(name_check, "_land_line", lambda *a, **k: ("retry", "并发写入"))
    report = book_name_checks(repo.store, rows, claims, task_id="t")
    assert report.blocked[0]["retry"] and _zh_evidence(repo, "C") == [] and "C" in due_lines(repo.store, limit=10)


def test_a_line_retired_after_the_landing_read_it_is_a_retry_not_a_refusal(tmp_path, monkeypatch) -> None:
    from finesub.llm.knowledge.node import apply as apply_mod

    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U"])
    rows = parse_name_check_results(_answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"},
                                             "url": "https://w/k"}), claims)
    real = apply_mod.apply_envelope

    def retire_first(store, envelope, **kwargs):  # type: ignore[no-untyped-def]
        with store.begin("user") as txn:
            txn.tombstone_membership("MU")
            txn.tombstone_node("U")
        return real(store, envelope, **kwargs)

    monkeypatch.setattr(apply_mod, "apply_envelope", retire_first)
    report = book_name_checks(repo.store, rows, claims, task_id="t")
    assert report.blocked and report.blocked[0].get("retry"), report.blocked
    assert (digest("克鲁斯尼克"), "refuted", "external") not in _zh_evidence(repo, "U")


def test_verify_dry_run_lists_the_names_owed_a_check(tmp_path, capsys) -> None:
    from finesub.llm.knowledge import maintain

    _seed(tmp_path)
    assert maintain.main(["--root", str(tmp_path), "verify"]) == 0
    out = capsys.readouterr().out
    assert "term line(s) owed a check" in out and "克鲁斯尼克晶矿" in out and "name-check prompt rendered" in out
    assert "格赖埃" not in out, "a tentative line is not sent"


# ---- the repair round ----------------------------------------------------------------------------------------------


class _Reply:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requested_output_tokens = 0


class _Answering:
    """A client whose replies are given in order; it keeps the conversations it was sent."""

    def __init__(self, *replies: str, retrieval: str = "native") -> None:
        self.replies = list(replies)
        self.retrieval = retrieval
        self.sent: list[list[dict]] = []

    def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["retrieval"] == self.retrieval and kwargs["task_group"] == "research"
        self.sent.append([dict(m) for m in messages])
        return _Reply(self.replies.pop(0))


def test_lines_without_a_usable_conclusion_are_asked_again_alone_in_the_same_conversation(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U", "C"])  # n1 U, n2 C, n3 D along U
    first = _answer({"id": "n1", "verdict": "corrected", "set": "克什尼克", "url": "https://w/k"},
                    {"id": "n2", "verdict": "confirmed", "url": "https://w/c"})
    rows, problems = parse_name_check_answer(first, claims)
    assert [row["claim_id"] for row in rows] == ["n2"] and problems == {"n1": "set"}, "n3 owes nothing yet"
    repaired = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"},
                       {"id": "n2", "verdict": "not_found"})
    final = _answer({"id": "n3", "verdict": "corrected", "set": {"zh": "克什尼克晶矿"}, "url": "https://w/k"})
    client = _Answering(first, repaired, final)
    result = run_name_check_session(claims, client=client, retries=3)
    assert [(row["claim_id"], row["verdict"]) for row in result.rows] == [("n2", "confirmed"), ("n1", "corrected"),
                                                                         ("n3", "corrected")]
    assert result.problems == {} and len(client.sent) == 3, "n1's rename makes n3 owed: one more round"
    second = client.sent[1]
    assert [m["role"] for m in second] == ["user", "assistant", "user"] and second[1]["content"] == first
    assert "- n1：set 不合格" in second[2]["content"] and "n2" not in second[2]["content"]
    assert "- n3：没有给结论" in client.sent[2][2]["content"]
    assert client.sent[2][1]["content"] == repaired, "each round sends the first exchange and the last reply"
    stuck = run_name_check_session(claims, client=_Answering(first, first), retries=1)
    assert stuck.problems == {"n1": "set"} and [row["claim_id"] for row in stuck.rows] == ["n2"], "the rest kept"
    none = run_name_check_session(claims, client=_Answering(first), retries=0)
    assert none.problems == {"n1": "set"} and len(none.answers) == 1


def test_an_answer_that_does_not_read_is_asked_again_whole(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    good = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"})
    client = _Answering("没有块", good)
    result = run_name_check_session(claims, client=client, retries=2)
    assert [row["claim_id"] for row in result.rows] == ["n1"] and result.problems == {}
    assert "上面的回答读不了" in client.sent[1][2]["content"]
    never = run_name_check_session(claims, client=_Answering("没有块", "还是没有"), retries=1)
    assert never.rows == [] and never.problems == {"n1": "unreadable"}


# ---- the local path -----------------------------------------------------------------------------------------------


class _Search:
    """A search client: pages and queries answer from the dicts given, anything else fails; it keeps what it was
    asked."""

    def __init__(self, pages=None, results=None, *, keys: bool = True) -> None:  # type: ignore[no-untyped-def]
        self.pages, self.results, self.keys = dict(pages or {}), dict(results or {}), keys
        self.fetched: list[str] = []
        self.searched: list[str] = []

    def has_provider(self) -> bool:
        return self.keys

    def extract_many(self, requests):  # type: ignore[no-untyped-def]
        from finesub.llm.web_search import QueryExtractResult

        self.fetched += [request.url for request in requests]
        return [QueryExtractResult(url=r.url, provider="exa", content=self.pages[r.url]) if r.url in self.pages
                else QueryExtractResult(url=r.url, error="HTTP 404") for r in requests]

    def search_many(self, queries):  # type: ignore[no-untyped-def]
        from finesub.llm.web_search import QuerySearchResult, SearchResultItem

        self.searched += list(queries)
        return [QuerySearchResult(query=q, provider="tavily", items=(SearchResultItem("t", self.results[q], "摘要"),))
                if q in self.results else QuerySearchResult(query=q, error="HTTP 500") for q in queries]


def _local(search: _Search) -> name_check.CheckRoute:
    return name_check.CheckRoute("local", search, count_tokens=len)


def _queries(*lines: str) -> str:
    return "<search_queries>\n" + "\n".join(lines) + "\n</search_queries>"


def test_the_local_path_reads_the_clue_pages_then_the_queries_it_listed(tmp_path) -> None:
    repo = _seed(tmp_path)
    clues = {"U": [{"found": "克什尼克", "url": "https://w/k", "origin": "window w1"},
                   {"found": "克什尼克", "url": "https://w/k2", "origin": "prepare"},
                   {"found": "库鲁斯尼克", "url": "https://w/k3", "origin": "window w2"}]}
    claims = build_claims(repo.store, ["U", "C"], clues=clues, derive_for=())  # n1 U, n2 C
    search = _Search(pages={"https://w/k": "克什尼克是至冬的能量"}, results={"アナスタシア 原神": "https://w/c"})
    client = _Answering(_queries("n2|アナスタシア 原神", "n2|阿纳斯塔西娅", "n2|third over the cap", "n9|no such line",
                                 "n1"),
                        _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"},
                                {"id": "n2", "verdict": "confirmed", "url": "https://w/c"}),
                        retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search))
    assert search.fetched == ["https://w/k", "https://w/k2"], "two clue pages a line"
    assert search.searched == ["アナスタシア 原神", "阿纳斯塔西娅"]
    opening = client.sent[0][0]["content"]
    assert "n1 的线索页面：" in opening and "克什尼克是至冬的能量" in opening and "<search_queries>" in opening
    assert "列 query 去搜" in opening and "联网搜。" not in opening
    second = client.sent[1]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "n2 的 query：" in second[2]["content"] and "https://w/c" in second[2]["content"]
    assert "（搜索失败：HTTP 500）" in second[2]["content"] and "<name_check_results>" in second[2]["content"]
    assert [row["claim_id"] for row in result.rows] == ["n1", "n2"] and result.problems == {}
    assert result.search == {"pages": 2, "pages_failed": 1, "queries": 2, "queries_failed": 1, "queries_dropped": 3,
                             "no_material": []}


def test_queries_that_differ_only_in_case_are_one_search_serving_every_line_that_listed_it(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["U", "C"], derive_for=())  # n1 U, n2 C
    search = _Search(results={"Genshin 原神": "https://w/g"})
    client = _Answering(_queries("n1|Genshin 原神", "n2|genshin 原神"),
                        _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/g"},
                                {"id": "n2", "verdict": "confirmed", "url": "https://w/g"}), retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search))
    assert search.searched == ["Genshin 原神"]
    assert result.search["no_material"] == [] and [row["claim_id"] for row in result.rows] == ["n1", "n2"]


def test_a_line_the_local_path_read_nothing_for_is_left_unanswered_not_not_found(tmp_path) -> None:
    repo = _seed(tmp_path)
    clues = {"U": [{"found": "克什尼克", "url": "https://w/gone", "origin": "window w1"}]}
    claims = build_claims(repo.store, ["U", "C", "Z"], clues=clues)  # n1 U, n2 C, n3 D along U
    # U: its clue page failed and it listed no query; C: every query failed
    search = _Search(results={})
    client = _Answering(_queries("n2|アナスタシア"), retrieval="none")
    nothing = run_name_check_session(claims, client=client, route=_local(search))
    assert len(client.sent) == 1, "nothing read for any line: no answer asked for"
    assert nothing.rows == [] and nothing.problems == {"n1": "no_material", "n2": "no_material", "n3": "no_material"}
    # with something read for C, U's answer is dropped and U asked about no further
    search = _Search(results={"アナスタシア": "https://w/c"})
    client = _Answering(_queries("n2|アナスタシア"),
                        _answer({"id": "n1", "verdict": "not_found", "kind": "temporary"},
                                {"id": "n2", "verdict": "confirmed", "url": "https://w/c"}),
                        retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search), retries=2)
    assert len(client.sent) == 2 and "这次不用给结论：n1、n3" in client.sent[1][2]["content"]
    assert [row["claim_id"] for row in result.rows] == ["n2"] and result.problems == {"n1": "no_material", "n3": "no_material"}
    report = book_name_checks(repo.store, result.rows, claims, task_id="t")
    assert [row["line"] for row in report.unanswered] == ["クルースニク|克鲁斯尼克"] and not report.not_found
    assert _zh_evidence(repo, "U") == [], "no not-found to back off on"


def test_a_local_repair_round_continues_the_conversation_after_the_results(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    search = _Search(results={"q": "https://w/c"})
    client = _Answering(_queries("n1|q"), _answer({"id": "n1", "verdict": "confirmed"}),
                        _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"}), retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search), retries=1)
    assert [row["claim_id"] for row in result.rows] == ["n1"]
    repair = client.sent[2]
    assert [m["role"] for m in repair] == ["user", "assistant", "user", "assistant", "user"]
    assert repair[:3] == client.sent[1] and "url 里填" in repair[4]["content"]


def test_the_local_material_stays_within_its_budget(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C"])
    search = _Search(results={f"q{i}": "https://w/" + "x" * 40_000 for i in range(2)})
    client = _Answering(_queries("n1|q0", "n1|q1"), _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"}),
                        retrieval="none")
    run_name_check_session(claims, client=client, route=_local(search))
    body = client.sent[1][2]["content"]
    results = body[body.index("<search_results>"):body.index("</search_results>")]
    assert len(results) <= name_check.LOCAL_RESULT_TOKENS + 100


def test_a_line_whose_material_the_budget_dropped_was_not_read(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C", "U"], derive_for=set())
    monkeypatch.setattr(name_check, "LOCAL_RESULT_TOKENS", 400)
    search = _Search(results={f"q{i}": "https://w/" + "x" * 20_000 for i in range(2)})
    client = _Answering(_queries("n1|q0", "n2|q1"),
                        _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"},
                                {"id": "n2", "verdict": "not_found", "kind": "temporary"}), retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search))
    assert result.search["no_material"] == ["n2"], "q1's results were cut out of the prompt: n2 read nothing"
    assert [row["claim_id"] for row in result.rows] == ["n1"] and result.problems == {"n2": "no_material"}


def test_a_line_sent_along_a_line_with_nothing_read_is_not_judged_either(tmp_path) -> None:
    repo = _seed(tmp_path)
    claims = build_claims(repo.store, ["C", "U"], derive_for={"U"})
    assert [(c.node_id, c.along) for c in claims] == [("C", False), ("U", False), ("D", True)]
    search = _Search(results={"q0": "https://w/c"})
    client = _Answering(_queries("n1|q0"),
                        _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"},
                                {"id": "n3", "verdict": "not_found", "kind": "temporary"}), retrieval="none")
    result = run_name_check_session(claims, client=client, route=_local(search))
    assert result.search["no_material"] == ["n2", "n3"], "D reads U's material, and U read none"
    assert [row["claim_id"] for row in result.rows] == ["n1"]


def test_the_route_is_native_search_then_the_local_path_then_none(tmp_path) -> None:
    class Routes:
        def __init__(self, *eligible: bool) -> None:
            self.eligible = list(eligible)

        def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
            if not self.eligible.pop(0):
                raise RuntimeError(f"No eligible target (native_search={kwargs['native_search']})")

    assert name_check.name_check_route(Routes(True), search=_Search()) == (name_check.NATIVE_ROUTE, "")
    route, why = name_check.name_check_route(Routes(False, True), search=_Search())
    assert route is not None and route.kind == "local" and why == ""
    assert name_check.name_check_route(Routes(False, True), search=_Search(keys=False))[1].startswith("没有能自己联网")
    assert name_check.name_check_route(Routes(False, False), search=_Search()) == (
        None, "No eligible target (native_search=False)")


def test_verify_checks_each_batch_against_the_base_as_it_goes(tmp_path, monkeypatch) -> None:
    from finesub.llm.knowledge import maintain, verify

    repo = _seed(tmp_path)

    class Local(_Client):
        def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["native_search"]:
                raise RuntimeError("no native search")

    monkeypatch.setattr(maintain, "_role_client", lambda _models: Local())
    monkeypatch.setattr("finesub.llm.web_search.WebSearchClient", lambda **kwargs: _Search())
    monkeypatch.setattr(name_check_step, "LOCAL_BATCH_LINES", 1)
    monkeypatch.setattr(verify, "run_verify_session", lambda claims, client: [])
    batches: list = []

    def session(claims, client, route):  # type: ignore[no-untyped-def]
        batches.append([claim.node_id for claim in claims])
        if len(batches) == 1:  # meanwhile another process moves D where the check may not send it
            with repo.store.begin("harness") as txn:
                txn.move_membership("MD", section="待归类")
        return _session(claims, _answer(*({"id": claim.claim_id, "verdict": "confirmed", "url": "https://w/p"}
                                          for claim in claims)))

    monkeypatch.setattr(name_check, "run_name_check_session", session)
    maintain.main(["--root", str(tmp_path), "verify", "--execute", "--apply", "--limit", "3"])
    assert batches == [["U"], ["C"]], "D was planned, then moved into 待归类 before its batch went"


def test_verify_takes_the_local_path_batch_by_batch_and_keeps_the_batches_before_a_failure(tmp_path, monkeypatch,
                                                                                        capsys) -> None:
    from finesub.llm.knowledge import maintain, verify

    _seed(tmp_path)

    class Local(_Client):
        def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["native_search"]:
                raise RuntimeError("no native search")

    monkeypatch.setattr(maintain, "_role_client", lambda _models: Local())
    monkeypatch.setattr("finesub.llm.web_search.WebSearchClient", lambda **kwargs: _Search())
    monkeypatch.setattr(name_check_step, "LOCAL_BATCH_LINES", 1)
    monkeypatch.setattr(verify, "run_verify_session", lambda claims, client: [])
    batches: list = []

    def session(claims, client, route):  # type: ignore[no-untyped-def]
        assert route.kind == "local"
        batches.append([claim.claim_id for claim in claims])
        if len(batches) == 3:
            raise RuntimeError("search quota gone")
        return _session(claims, _answer(*({"id": claim.claim_id, "verdict": "confirmed", "url": "https://w/p"}
                                          for claim in claims)))

    monkeypatch.setattr(name_check, "run_name_check_session", session)
    assert maintain.main(["--root", str(tmp_path), "verify", "--execute", "--apply", "--limit", "3"]) == 1
    err = capsys.readouterr().err
    assert batches == [["n1"], ["n2"], ["n3"]] and "search quota gone" in err, \
        "verify sends each line for itself: a line whose name holds another's is no line sent along"
    assert err.count("印证：") == 2 and "没有可用结论（没答或答得不合格）1 行" in err


def test_the_step_takes_the_local_path_in_batches_of_its_own_size(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    monkeypatch.setattr(name_check_step, "LOCAL_BATCH_LINES", 1)
    monkeypatch.setattr("finesub.llm.web_search.follow_search_redirect", lambda url: url)

    class Local(_Client):
        def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs["native_search"]:
                raise RuntimeError("no native search")

    sessions: list = []

    def session(claims, client, retries=None, route=None):  # type: ignore[no-untyped-def]
        assert route.kind == "local"
        sessions.append([claim.node_id for claim in claims])
        result = _session(claims, _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"}))
        result.search = {"pages": 1, "pages_failed": 1, "queries": 2, "queries_failed": 0, "queries_dropped": 0,
                         "no_material": []}
        return result

    monkeypatch.setattr(name_check_step, "run_name_check_session", session)
    records = [_record("阿纳斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅"),
               _record("克鲁斯尼克")]
    summary = name_check_step.run_name_check_step(knowledge_root=repo.root, artifact_dir=tmp_path / "art",
                                                  task_id="t3", client=Local(), records=records, search=_Search())
    assert summary["route"] == "local" and len(sessions) == 2 and all(len(batch) == 1 for batch in sessions)
    from finesub.llm.task_report import write_task_report

    text = write_task_report(tmp_path / "art", task_id="t3").read_text(encoding="utf-8")
    assert "中文定名核查（本地搜索）：送去 2 行，2 批" in text
    assert "本地搜索：取线索页 1 页（没取到 1），搜 2 条 query（失败 0）" in text


# ---- a task's check step ------------------------------------------------------------------------------------------


class _Client:
    def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs == {"native_search": True, "task_group": "research"}


def _session(claims, text: str) -> name_check.SessionResult:  # type: ignore[no-untyped-def]
    """A session that answered ``text`` at once."""

    rows, problems = parse_name_check_answer(text, claims)
    return name_check.SessionResult(rows=rows, answers=[text], problems=problems)


def _step(tmp_path, repo, monkeypatch, *, answer, calls: list, records=None, **kwargs):  # type: ignore[no-untyped-def]
    """The step with the session given: ``answer`` a reply text, or a callable of the claims returning one."""

    def session(claims, client, retries=None, route=None):  # type: ignore[no-untyped-def]
        calls.append([c.node_id for c in claims])
        return _session(claims, answer(claims) if callable(answer) else answer)

    monkeypatch.setattr(name_check_step, "run_name_check_session", session)
    monkeypatch.setattr("finesub.llm.web_search.follow_search_redirect", lambda url: url)
    return name_check_step.run_name_check_step(
        knowledge_root=repo.root, artifact_dir=tmp_path / "art", task_id="task-zh", client=_Client(),
        records=[_record()] if records is None else records, **kwargs)


RENAME = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "克什尼克"}, "url": "https://w/k"},
                 {"id": "n2", "verdict": "corrected", "set": {"zh": "克什尼克晶矿"}, "url": "https://w/k"})


def _ledger(tmp_path) -> dict:  # type: ignore[type-arg]
    return json.loads((tmp_path / "art" / name_check_step.LEDGER_FILENAME).read_text(encoding="utf-8"))


def test_a_resumed_step_neither_asks_again_nor_lands_twice(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    calls: list = []
    report = _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls)
    assert calls == [["U", "D"]] and len(report["batches"][0]["report"]["corrected"]) == 2
    revs, evidence = repo.rev, repo.store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    again = _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls)
    assert calls == [["U", "D"]] and again["replayed"]
    # cut off after the answer was kept and before it was booked: landed once, from the stored answer
    saved = _ledger(tmp_path)
    saved["batches"][0]["status"] = "answered"
    (tmp_path / "art" / name_check_step.LEDGER_FILENAME).write_text(
        json.dumps({"task_id": "task-zh", "batches": saved["batches"]}, ensure_ascii=False), encoding="utf-8")
    _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls)
    assert calls == [["U", "D"]], "the stored batch answered U and D: nothing asked again"
    assert repo.rev == revs and repo.store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == evidence
    assert _ledger(tmp_path)["done"]


def test_a_resume_lands_the_unlanded_batch_and_asks_only_the_lines_left_unanswered(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    monkeypatch.setattr(name_check_step, "NATIVE_BATCH_LINES", 2)
    records = [_record(), _record("阿娜斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅")]
    calls: list = []
    # batch 1 (U, D) answers U only; batch 2 (C) is answered, then the run stops before it lands
    first = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/k"})
    second = _answer({"id": "n1", "verdict": "corrected", "set": {"zh": "阿娜斯塔西娅"}, "url": "https://w/c"})
    real_land = name_check_step._land
    landed: list = []

    def land_first_only(repo_, root, batch):  # type: ignore[no-untyped-def]
        if landed:
            raise KeyboardInterrupt
        landed.append(batch["run_id"])
        real_land(repo_, root, batch)

    monkeypatch.setattr(name_check_step, "_land", land_first_only)
    with pytest.raises(KeyboardInterrupt):
        _step(tmp_path, repo, monkeypatch, answer=lambda claims: first if claims[0].node_id == "U" else second,
              calls=calls, records=records)
    assert calls == [["U", "D"], ["C"]]
    assert [b["status"] for b in _ledger(tmp_path)["batches"]] == ["booked", "answered"]
    monkeypatch.setattr(name_check_step, "_land", real_land)
    calls.clear()
    confirm_c = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/c"})
    _step(tmp_path, repo, monkeypatch, answer=confirm_c, calls=calls, records=records)
    assert repo.store.node("C").payload["zh"] == "阿娜斯塔西娅", "batch 2 landed from what it stored"
    assert calls == [], "U and C were answered and D, sent along with a name kept, owes nothing"
    run_ids = {row["task_id"] for row in repo.store.conn.execute(
        "SELECT task_id FROM evidence WHERE node_id='C' AND field_path='payload.zh'")}
    assert run_ids == {_ledger(tmp_path)["batches"][1]["run_id"]}


def test_a_step_that_cannot_land_keeps_the_answer_and_spares_the_task(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    calls: list = []
    monkeypatch.setattr(name_check_step, "book_name_checks",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("database is locked")))
    _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls)
    assert [b["status"] for b in _ledger(tmp_path)["batches"]] == ["answered"] and not _ledger(tmp_path).get("done")
    monkeypatch.undo()
    import contextlib

    @contextlib.contextmanager
    def held(_root):  # type: ignore[no-untyped-def]
        yield False

    # a correction left to retry (the lock held elsewhere) keeps the batch open too, and the step not done
    monkeypatch.setattr("finesub.llm.knowledge.base.knowledge_write_lock", held)
    _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls)
    assert [b["status"] for b in _ledger(tmp_path)["batches"]] == ["answered"] and calls == [["U", "D"]]
    assert not _ledger(tmp_path).get("done")


class _NoRoute(_Client):
    def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("No eligible target for research/quality (requires native_search)")


def test_a_resume_without_a_route_leaves_the_rest_for_a_resume_with_one(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    monkeypatch.setattr(name_check_step, "NATIVE_BATCH_LINES", 2)
    records = [_record(), _record("阿娜斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅")]
    confirm = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/k"})
    calls: list = []

    def fail_on_c(claims):  # type: ignore[no-untyped-def]
        if claims[0].node_id == "C":
            raise RuntimeError("search quota gone")
        return confirm

    assert _step(tmp_path, repo, monkeypatch, answer=fail_on_c, calls=calls, records=records)["failed"]
    assert calls == [["U", "D"], ["C"]]
    skipped = name_check_step.run_name_check_step(knowledge_root=repo.root, artifact_dir=tmp_path / "art",
                                                  task_id="task-zh", client=_NoRoute(), records=records)
    assert skipped["skipped"] == "no_route" and not _ledger(tmp_path).get("done"), "C is still owed"
    calls.clear()
    _step(tmp_path, repo, monkeypatch, answer=confirm, calls=calls, records=records)
    assert calls == [["C"]] and _ledger(tmp_path)["done"]


def test_a_finished_step_replays_only_its_own_records_and_not_under_no_resume(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    one = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/k"})
    calls: list = []
    _step(tmp_path, repo, monkeypatch, answer=one, calls=calls)
    assert calls == [["U", "D"]] and _ledger(tmp_path)["done"]
    assert _step(tmp_path, repo, monkeypatch, answer=one, calls=calls)["replayed"]
    assert not _step(tmp_path, repo, monkeypatch, answer=one, calls=calls, resume=False).get("replayed")
    assert calls == [["U", "D"]], "no resume reads the step afresh, and U, answered already, is not asked again"
    # the windows redone: new records, the line they name asked, U still not
    more = [_record(), _record("阿娜斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅")]
    _step(tmp_path, repo, monkeypatch, answer=one, calls=calls, records=more)
    assert calls == [["U", "D"], ["C"]]


def test_clue_links_are_followed_only_on_lines_a_route_takes(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    _book(repo, "C", "confirmed", days_ago=1)  # agreeing and confirmed a day ago: not sent
    followed: list = []
    monkeypatch.setattr("finesub.llm.web_search.follow_search_redirect", lambda url: followed.append(url) or url)
    records = [_record(url="https://r/u"), _record("阿纳斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅",
                                                   url="https://r/c")]
    name_check_step.run_name_check_step(knowledge_root=repo.root, artifact_dir=tmp_path / "a", task_id="t",
                                        client=_NoRoute(), records=records)
    assert followed == [], "no route: nothing followed"
    monkeypatch.setattr(name_check_step, "run_name_check_session",
                        lambda claims, **kwargs: _session(claims, _answer({"id": "n1", "verdict": "confirmed",
                                                                           "url": "https://w/k"})))
    name_check_step.run_name_check_step(knowledge_root=repo.root, artifact_dir=tmp_path / "b", task_id="t",
                                        client=_Client(), records=records)
    assert "https://r/u" in followed and "https://r/c" not in followed, "C does not go: its link is not followed"


def test_each_session_books_under_its_own_run_id(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    one = _answer({"id": "n1", "verdict": "confirmed", "url": "https://w/k"})
    _step(tmp_path / "a", repo, monkeypatch, answer=one, calls=[])
    _step(tmp_path / "b", repo, monkeypatch, answer=one, calls=[])
    tasks = {row["task_id"] for row in repo.store.conn.execute(
        "SELECT task_id FROM evidence WHERE node_id='U' AND field_path='payload.zh'")}
    assert len(tasks) == 2 and all(task.startswith("task-zh-zh-") for task in tasks), tasks


def test_a_ledger_that_cannot_be_written_lands_nothing(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    monkeypatch.setattr(name_check_step, "_write_ledger", lambda *a, **k: (_ for _ in ()).throw(OSError("locked")))
    rev, evidence = repo.rev, repo.store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    report = _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=[])
    assert report["failed"].startswith("ledger")
    assert repo.rev == rev and repo.store.conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0] == evidence


def test_without_records_the_step_takes_no_lock_and_writes_no_ledger(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    monkeypatch.setattr("finesub.llm.knowledge.base.knowledge_write_lock", lambda _root: pytest.fail("no lock"))
    assert _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=[], records=[]) is None
    assert not (tmp_path / "art").exists()
    matched_none = _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=[], records=[_record(source="ない")])
    assert matched_none["dropped"] == {"no_line": 1} and matched_none["batches"] == []
    assert not (tmp_path / "art" / name_check_step.LEDGER_FILENAME).exists()


def test_no_apply_asks_and_keeps_nothing(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    calls: list = []
    rev = repo.rev
    report = _step(tmp_path, repo, monkeypatch, answer=RENAME, calls=calls, apply=False)
    assert calls == [["U", "D"]] and len(report["rows"]) == 2 and repo.rev == rev
    assert not (tmp_path / "art" / name_check_step.LEDGER_FILENAME).exists()
    assert _zh_evidence(repo, "U") == []


def test_the_step_says_why_it_cannot_run(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path / "kb")
    notes: list = []

    class NoRoute(_Client):
        def ensure_eligible_target(self, role, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("No eligible target for research/quality (requires native_search)")

    from finesub.reporting import reporting_to

    class Reporter:
        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            return lambda *args, **kwargs: notes.append((name, args))

    monkeypatch.setattr("finesub.llm.web_search.follow_search_redirect", lambda url: url)
    with reporting_to(Reporter()):
        result = name_check_step.run_name_check_step(knowledge_root=repo.root, artifact_dir=tmp_path / "art",
                                                     task_id="t2", records=[_record()], client=NoRoute())
    assert result["skipped"] == "no_route" and [name for name, _ in notes if name != "debug"] == ["note"]
    from finesub.llm.task_report import write_task_report

    text = write_task_report(tmp_path / "art", task_id="t2").read_text(encoding="utf-8")
    assert "中文定名核查：送去 2 行" in text and "克什尼克（https://w/k）" in text and "下一个带 --knowledge update" in text


# ---- which lines go, in which batch -------------------------------------------------------------------------------


def test_a_disputed_line_goes_unless_its_refutation_counts_and_another_only_when_owed(tmp_path) -> None:
    repo = _seed(tmp_path)
    _book(repo, "U", "confirmed", days_ago=1)  # confirmed a day ago: not owed a check
    _book(repo, "C", "confirmed", days_ago=1)
    clues = name_check_step.collect_clues(repo.store, [
        _record(), _record("阿纳斯塔西娅", source="アナスタシア", kb_zh="阿纳斯塔西娅")])
    assert clues.disputed == {"U"}
    assert name_check_step.lines_to_send(repo.store, clues) == ["U"], "U disputed goes; C agreeing waits the gap"
    claims = build_claims(repo.store, ["U", "C"], clues=clues.by_line, derive_for=clues.disputed)
    assert [(c.node_id, c.along) for c in claims] == [("U", False), ("C", False), ("D", True)]
    assert [c.node_id for c in build_claims(repo.store, ["C"], derive_for=set())] == ["C"]
    _book(repo, "U", "refuted", days_ago=0)  # a refusal on the current version: terminal
    assert name_check_step.lines_to_send(repo.store, clues) == []
    with repo.store.begin("harness") as txn:  # written again after the refutation: it no longer counts
        txn.update_node("U", payload={**repo.store.node("U").payload, "desc": "改过描述。"})
    repo.store.conn.execute("UPDATE revisions SET created_at=? WHERE rev=?",
                            ((datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(timespec="seconds"),
                             repo.rev))
    assert name_check_step.lines_to_send(repo.store, clues) == ["U"]
    assert name_check_step.lines_to_send(repo.store, clues, skip=["U"]) == []


def test_a_checked_line_a_record_spells_otherwise_goes_to_the_check(tmp_path) -> None:
    """已核 is no shelter: a window's record of another sourced spelling sends the line, which the check may correct
    (the zh acceptance clip's deliberately wrong 已核 クルースニク)."""

    repo = _seed(tmp_path)
    _book(repo, "U", "confirmed", days_ago=RECHECK_GAP.days + 1)
    _book(repo, "U", "confirmed")
    assert _state(repo, "U") == CHECKED
    clues = name_check_step.collect_clues(repo.store, [_record()])
    assert clues.disputed == {"U"} and name_check_step.lines_to_send(repo.store, clues) == ["U"]


def test_lines_go_in_batches_with_what_follows_them_and_their_entry_together(tmp_path) -> None:
    repo = _seed(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S2", "subject", {"surface": "星穹铁道", "intro": "游戏", "category": "common",
                                          "entry_type": "游戏", "native_names": ["星穹铁道"], "section_order": ["人物"]})
        for n in range(31):
            txn.create_node(f"H{n}", "term", {"surface": f"h{n}", "zh": f"名字{n:02d}号", "desc": "角色。"})
            txn.create_membership(f"MH{n}", "S2", f"H{n}", "人物", n)
    ids = [f"H{n}" for n in range(31)]
    claims = build_claims(repo.store, [*ids[:15], "U", "C", *ids[15:]], derive_for={"U"})
    batches = name_check_step.plan_batches(claims, 30)
    assert [len(batch) for batch in batches] == [30, 4], "31 + 3 lines: two batches"
    assert [c.node_id for c in batches[0]] == ids[:30], "an entry's lines together, though U came between them"
    assert [c.node_id for c in batches[1]] == ["H30", "U", "D", "C"], "D follows U into its batch"
    assert [len(b) for b in name_check_step.plan_batches(claims[:9], 8)] == [8, 1]
    unit = build_claims(repo.store, ["C", "U"], derive_for={"U"})
    assert [[c.node_id for c in b] for b in name_check_step.plan_batches(unit, 2)] == [["C"], ["U", "D"]], \
        "a line and the one it brings stay together"
    assert [[c.node_id for c in b] for b in name_check_step.plan_batches(unit, 1)] == [["C"], ["U"]], \
        "a group past the batch size keeps its first lines: the batch size holds, D is not brought this time"
    out = name_check_step._current(repo.store, batches[1])
    assert [c.claim_id for c in out] == ["n1", "n2", "n3", "n4"]
    assert next(c for c in out if c.node_id == "D").derived_from == next(c for c in out if c.node_id == "U").claim_id
    with repo.store.begin("harness") as txn:
        txn.update_node("U", payload={**repo.store.node("U").payload, "zh": "克什尼克"})
    assert [c.node_id for c in name_check_step._current(repo.store, batches[1])] == ["H30", "C"], \
        "a line changed since the plan drops, and the line it brought with it"
    with repo.store.begin("harness") as txn:
        txn.move_membership("MC", section="待归类")
    assert [c.node_id for c in name_check_step._current(repo.store, batches[1])] == ["H30"], \
        "a line moved since the plan into a section the check may not send out drops too"


def _zh_task(tmp_path: Path, knowledge: str):  # type: ignore[no-untyped-def]
    from finesub.llm.correction_request import CorrectionRequest
    from finesub.llm.knowledge.update import research_context_in_artifact_dir

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "克鲁斯尼克的力量。", "lang": "zh"}]}, ensure_ascii=False),
        encoding="utf-8")
    out = tmp_path / "clip.srt"
    (tmp_path / "clip-translated.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\n克什尼克的力量。\n", encoding="utf-8")
    art = tmp_path / "clip.llm-artifacts"
    art.mkdir()
    research_context_in_artifact_dir(art, "clip").write_text(json.dumps({
        "context_pack": {"general_context": {}, "window_contexts": []},
        "injected_entries": ["原神"], "keep_entries": []}, ensure_ascii=False), encoding="utf-8")
    from finesub.llm.routing.profiles import resolve_profile

    return CorrectionRequest(stable_json=stable, output_path=out, postprocess_profile=None, knowledge=knowledge,
                             profile=resolve_profile("text", "none", "quality"), knowledge_root=tmp_path / "kb",
                             task_artifact_dir=art), art


def test_a_finished_correct_only_task_run_with_update_runs_the_name_check_alone(tmp_path, monkeypatch) -> None:
    """A finished correct-only task run again with `update`: no window, and its knowledge update stage is the name
    check alone -- no proposals."""

    from finesub.llm import correction_translation as orchestration
    from finesub.llm.knowledge import update as update_mod

    _seed(tmp_path / "kb")
    request, art = _zh_task(tmp_path, "update")
    checks: list = []
    monkeypatch.setattr(orchestration, "_correct_only_warning", lambda _message: None)
    monkeypatch.setattr(update_mod, "RoleClient", lambda **kw: object())
    monkeypatch.setattr(update_mod, "run_name_check_step", lambda **kw: checks.append(kw) or {"lines": 1})
    monkeypatch.setattr(update_mod, "build_knowledge_materials",
                        lambda **kw: pytest.fail("a correct-only task proposes nothing"))
    outcome = orchestration.run_correction_task(request)
    assert outcome.result_reused and outcome.knowledge_update["name_check"] == {"lines": 1}
    assert outcome.knowledge_update["skipped"] == "correct_only" and outcome.knowledge_update["chunks"] == []
    assert [(kw["artifact_dir"], kw["apply"]) for kw in checks] == [(art.resolve(), True)]


def test_a_reading_task_checks_nothing(tmp_path, monkeypatch) -> None:
    from finesub.llm import correction_translation as orchestration

    repo = _seed(tmp_path / "kb")
    request, _art = _zh_task(tmp_path, "read")
    monkeypatch.setattr(orchestration, "_correct_only_warning", lambda _message: None)
    monkeypatch.setattr(orchestration, "run_post_correction_knowledge_update",
                        lambda **kw: pytest.fail("read writes nothing"))
    rev = repo.rev
    outcome = orchestration.run_correction_task(request)
    assert outcome.result_reused and outcome.knowledge_update is None
    assert KnowledgeRepo.open(tmp_path / "kb").rev == rev


def test_the_knowledge_update_checks_names_before_it_proposes(tmp_path, monkeypatch) -> None:
    from finesub.llm.knowledge import update as update_mod

    order: list[str] = []
    monkeypatch.setattr(update_mod, "run_name_check_step", lambda **kw: order.append("check") or None)

    def materials(**kw):  # type: ignore[no-untyped-def]
        order.append("materials")
        raise ValueError("stop here")

    monkeypatch.setattr(update_mod, "build_knowledge_materials", materials)
    monkeypatch.setattr(update_mod, "RoleClient", lambda **kw: object())
    final = tmp_path / "clip.srt"
    final.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="stop here"):
        update_mod.run_knowledge_update(final_srt=final, knowledge_root=tmp_path / "kb", apply=False,
                                        artifact_dir=tmp_path / "art")
    assert order == ["check", "materials"]
    order.clear()
    with pytest.raises(ValueError):
        update_mod.run_knowledge_update(final_srt=final, knowledge_root=tmp_path / "kb", execute=False,
                                        artifact_dir=tmp_path / "art")
    assert order == ["materials"], "a dry run checks nothing"


def test_a_correct_only_task_under_update_asks_for_the_name_checks_alone(tmp_path) -> None:
    """Read and update search the same; update adds the feedback block, which holds only `name_checks`."""

    from dataclasses import replace

    from finesub.llm import correction_translation as orchestration
    from finesub.llm.routing.profiles import resolve_profile

    request, _art = _zh_task(tmp_path, "update")
    moded, _ = orchestration._with_task_mode(request, tmp_path / "art")
    assert moded.knowledge == "update" and orchestration.collects_feedback(moded) is True
    _seed(tmp_path / "kb")
    seen: dict[str, dict] = {}
    for knowledge in ("read", "update"):
        plan: dict = {}
        orchestration.run_correction_task(replace(
            request, mode="plan", knowledge=knowledge, test_profile=True, on_plan=plan.update,
            output_path=tmp_path / knowledge / "clip.srt", task_artifact_dir=None,
            profile=resolve_profile("text", "native", "quality")))
        seen[knowledge] = plan
    for key in ("research_messages", "correction_messages"):
        read, update = (json.dumps(seen[k][key], ensure_ascii=False) for k in ("read", "update"))
        assert "name_checks" in update and "name_checks" not in read, key
        assert "knowledge_hints" not in update, key
    assert "先搜一次这个名字" in seen["read"]["research_messages"][0][0]["content"]


def test_a_correct_only_task_under_update_plans_a_knowledge_update_that_proposes_nothing() -> None:
    from finesub.llm.stages import task_graph as tg

    grant = tg.CapabilityGrant(search="native", knowledge="update", style=False)
    graph = tg.plan_task_graph(fast_enabled=False, grant=grant, correct_only=True)
    assert graph.kinds == ("prepare", "translate", "knowledge_update") and graph.tasks[-1].tools == ()
    assert tg.plan_task_graph(fast_enabled=False, grant=grant).tasks[-1].tools == (tg.TOOL_KB_PROPOSE,)
    assert "name_check" not in tg.TASK_KINDS


# ---- the preparation task -----------------------------------------------------------------------------------------


def test_a_preparation_that_reads_and_searches_searches_unchecked_names_and_records_them() -> None:
    """Translating or correct-only, a preparation task that reads the library and can search searches an 未核 name
    before adopting it; under ``update`` it records what it found in `name_checks` (a correct-only task's feedback
    holds nothing else), and without search it does neither."""

    from finesub.llm.prepare_prompts import build_prepare_task_messages

    def text(**kwargs):  # type: ignore[no-untyped-def]
        return "\n".join(m["content"] for m in build_prepare_task_messages(
            transcript="--- window 1 ---\n1|大家好", knowledge_indexes={"common": "- 原神 | 游戏"}, **kwargs))

    for correct_only in (True, False):
        asked = text(correct_only=correct_only, native_search=True, emits_queries=False, collect_task_feedback=True)
        assert "先搜一次这个名字" in asked and "name_checks" in asked and UNCHECKED_NOT_FOUND_MARK in asked
        assert ("knowledge_hints" in asked) is not correct_only, "a correct-only task records the name checks only"
        assert "定下来的定名也可能错" in asked, "a settled line's sourced other spelling is recorded too"
        read = text(correct_only=correct_only, native_search=True, emits_queries=False)
        assert "先搜一次这个名字" in read and "name_checks" not in read, "read searches too, but records nothing"
        blind = text(correct_only=correct_only, native_search=False, emits_queries=False, collect_task_feedback=True)
        assert "先搜一次这个名字" not in blind and "name_checks" not in blind
    assert "<task_update_feedback>" not in text(correct_only=True, native_search=False, emits_queries=False,
                                                collect_task_feedback=True), "nothing to record without search"
    assert "kb_name_conflicts" not in text(correct_only=True, native_search=True, collect_task_feedback=True)


# ---- sharing ------------------------------------------------------------------------------------------------------


def test_a_library_with_name_check_evidence_exports_is_approved_and_pulls(tmp_path) -> None:
    from finesub.llm.knowledge.node.store import KnowledgeStore
    from finesub.llm.knowledge.share.exchange import build_push_bundle, strip_local_maps, validate_bundle
    from finesub.llm.knowledge.share.server import ShareService
    from finesub.llm.knowledge.share.sync import apply_snapshot

    with KnowledgeStore(tmp_path / "client.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common",
                                             "native_names": ["原神"], "section_order": ["角色"]},
                            visibility="shareable")
            txn.create_node("T", "term", {"surface": "スカラマシュ", "zh": "散兵", "desc": "角色"}, visibility="shareable")
            txn.create_membership("M", "S", "T", "角色", 0)
        for verdict, span, url in (("confirmed", None, "https://w/a"), ("refuted", None, "https://w/b"),
                                   ("unverifiable", "temporary", None)):
            record_evidence(store, node_id="T", field_path="payload.zh", value_hash=digest("散兵"), verdict=verdict,
                            evidence_kind="external", task_id="kb-name-check", source_ref=url, span=span)
        bundle = strip_local_maps(build_push_bundle(store, ["S"], idempotency_key="nc-1"))
    validate_bundle(bundle)
    zh = [c for c in bundle["claim_summaries"] if c["field_path"] == "payload.zh"]
    assert zh and zh[0]["confirmed_count"] == 1 and zh[0]["refuted_count"] == 1
    service = ShareService(tmp_path / "srv", maintainer_token="mt")
    token = service.register()["token"]
    queue_id = service.push(token, bundle)["queue_id"]
    item = service.lease("mt")["items"][0]
    service.verdict("mt", queue_id=queue_id, lease_token=item["lease_token"],
                    expected_version=item["verdict_version"], verdict="approve", override="name-check fixture")
    with KnowledgeStore(tmp_path / "puller.sqlite") as puller:
        report = apply_snapshot(puller, service.snapshot(), remote="r1")
        assert report.created_nodes == 2


@pytest.mark.parametrize("answer, said", [
    ("我核完了，都对。", "这份回答里有 0 个"),
    ("<name_check_results>\n[{\"id\": \"n1\",]\n</name_check_results>", "解析不了（第 1 行"),
    ("<name_check_results>\n{\"id\": \"n1\"}\n</name_check_results>", "要是一个 JSON 数组"),
])
def test_an_answer_that_does_not_read_is_told_why_in_words(answer, said):
    claim = name_check.Claim(claim_id="n1", node_id="x", surface="クルースニク", zh="克鲁斯尼克", desc="", subject="原神")
    with pytest.raises(ValueError) as caught:
        name_check.parse_name_check_answer(answer, [claim])
    assert said in str(caught.value) and "must" not in str(caught.value)
