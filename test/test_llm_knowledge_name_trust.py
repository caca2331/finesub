"""The trust state of a term line's Chinese name and the mark a correct-only task reads (docs/knowledge.md
「中文定名的可信状态」).

The state is worked out at render time from the evidence on the current value of `payload.zh` alone; the mark
`（未核）` opens the description column of an unchecked line in the correct-only projection, whichever way the task
reads the line, and nowhere else.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer
from finesub.llm.knowledge.base import (
    KnowledgeRun,
    bind_knowledge_run,
    load_entry_texts,
    match_terms,
    pinned_generation_rev,
)
from finesub.llm.knowledge.node.human_marks import ack_lines
from finesub.llm.knowledge.node.model import digest
from finesub.llm.knowledge.node.name_trust import (
    CHECKED,
    HUMAN,
    RECHECK_GAP,
    UNCHECKED,
    UNCHECKED_MARK,
    UNCHECKED_NOT_FOUND_MARK,
    check_due,
    trust_marks,
    zh_state,
    zh_statuses,
)
from finesub.llm.knowledge.node.render import render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import EVIDENCE_KIND_REFINED, book_revision_evidence, record_evidence

UNCHECKED_LINE = "クルースニク|克鲁斯尼克||（未核）至冬的能量。"
CHECKED_LINE = "アナスタシア|阿纳斯塔西娅||至冬女皇的本名。"
EMPTY_DESC_LINE = "ストレルカ|斯特雷尔卡||（未核）"
NO_NAME_LINE = "フィリン|||至冬的地名，未定名。"


@pytest.fixture(autouse=True)
def _no_run_left_bound():
    yield
    bind_knowledge_run(KnowledgeRun())


def _seed(root) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    """One game entry: an unchecked term, one whose name external sources checked twice a gap apart, one with an
    empty description, one without a Chinese name, and a free sentence."""

    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common",
                                         "entry_type": "游戏", "native_names": ["原神"],
                                         "section_order": ["档案", "至冬"]})
        txn.create_node("U", "term", {"surface": "クルースニク", "zh": "克鲁斯尼克", "desc": "至冬的能量。"})
        txn.create_membership("MU", "S", "U", "至冬", 0)
        txn.create_node("C", "term", {"surface": "アナスタシア", "zh": "阿纳斯塔西娅", "desc": "至冬女皇的本名。"})
        txn.create_membership("MC", "S", "C", "至冬", 1)
        txn.create_node("E", "term", {"surface": "ストレルカ", "zh": "斯特雷尔卡", "desc": ""})
        txn.create_membership("ME", "S", "E", "至冬", 2)
        txn.create_node("Z", "term", {"surface": "フィリン", "zh": "", "desc": "至冬的地名，未定名。"})
        txn.create_membership("MZ", "S", "Z", "至冬", 3)
        txn.create_node("N", "note", {"text": "至冬篇偏严肃。"})
        txn.create_membership("MN", "S", "N", "至冬", 4)
    _check(repo, "C", source_ref="https://wiki.biligame.com/ys/阿纳斯塔西娅")
    return repo


def _confirm(repo: KnowledgeRepo, node_id: str, kind: str, *, field_path: str = "payload.zh",
             value: str | None = None, source_ref: str | None = None, days_ago: float = 0,
             created_at: str | None = None) -> None:
    """One confirmation, booked ``days_ago`` (each day its own task, so two are two pieces of evidence)."""

    node = repo.store.node(node_id)
    assert node is not None
    assert record_evidence(repo.store, node_id=node_id, field_path=field_path,
                           value_hash=digest(node.payload["zh"] if value is None else value), verdict="confirmed",
                           evidence_kind=kind, task_id=f"test-{days_ago}", source_ref=source_ref)
    when = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    repo.store.conn.execute("UPDATE evidence SET created_at=? WHERE evidence_id=last_insert_rowid()",
                            (when if created_at is None else created_at,))


def _check(repo: KnowledgeRepo, node_id: str, kind: str = "external", *, source_ref: str | None = None) -> None:
    """Two confirmations of the current name the recheck gap apart: what makes it 已核."""

    _confirm(repo, node_id, kind, source_ref=source_ref, days_ago=RECHECK_GAP.days + 1)
    _confirm(repo, node_id, kind, source_ref=source_ref)


def _state(repo: KnowledgeRepo, node_id: str) -> str | None:
    node = repo.store.node(node_id)
    assert node is not None
    return zh_state(repo.store, node)


def _change(repo: KnowledgeRepo, node_id: str, kind: str = "harness", **fields: str) -> int:
    node = repo.store.node(node_id)
    assert node is not None
    with repo.store.begin(kind) as txn:
        txn.update_node(node_id, payload={**node.payload, **fields})
    return repo.rev


# ---- the state -------------------------------------------------------------------------------------------------


def test_a_name_checked_twice_a_gap_apart_is_checked_and_a_rename_voids_it(tmp_path) -> None:
    repo = _seed(tmp_path)
    assert _state(repo, "U") == UNCHECKED
    _confirm(repo, "U", "external", source_ref="https://example.org/a")
    assert _state(repo, "U") == UNCHECKED, "one check rests on pages the model picked"
    _confirm(repo, "U", "external", source_ref="https://example.org/a", days_ago=RECHECK_GAP.days - 1)
    assert _state(repo, "U") == UNCHECKED, "two checks within the gap are not a recheck"
    # a hair past the gap: the timestamps are cut to the second, and the two are taken a moment apart
    _confirm(repo, "U", "external", source_ref="https://example.org/a", days_ago=RECHECK_GAP.days + 0.001)
    assert _state(repo, "U") == CHECKED
    _change(repo, "U", desc="至冬的能量体系。")
    assert _state(repo, "U") == CHECKED, "a description edit leaves the name's state alone"
    _change(repo, "U", zh="克什尼克")
    assert _state(repo, "U") == UNCHECKED, "the evidence was on the old name"


def test_a_refined_subtitle_that_kept_the_name_checks_it_too(tmp_path) -> None:
    repo = _seed(tmp_path)
    _confirm(repo, "U", "external", source_ref="https://example.org/a", days_ago=RECHECK_GAP.days)
    _confirm(repo, "U", EVIDENCE_KIND_REFINED)
    assert _state(repo, "U") == CHECKED, "either kind counts as one of the two checks"


def test_a_confirmation_without_a_timestamp_cannot_space_a_recheck(tmp_path) -> None:
    repo = _seed(tmp_path)
    _confirm(repo, "U", "external", source_ref="https://example.org/a", created_at="")
    _confirm(repo, "U", "external", source_ref="https://example.org/b")
    assert _state(repo, "U") == UNCHECKED


@pytest.mark.parametrize("field_path, kind", [
    ("payload:core", "external"),       # `knowledge verify`'s whole-line claim: often a foreign page
    ("payload.desc", "external"),
    ("payload.zh", "transcript"),       # where a task's write came from, not a check
])
def test_evidence_elsewhere_or_of_another_kind_does_not_count(tmp_path, field_path, kind) -> None:
    repo = _seed(tmp_path)
    node = repo.store.node("U")
    assert node is not None
    value = {"payload:core": "the whole group", "payload.desc": node.payload["desc"]}.get(field_path)
    _confirm(repo, "U", kind, field_path=field_path, value=value)
    assert _state(repo, "U") == UNCHECKED


def test_a_refuted_or_unverifiable_name_stays_unchecked(tmp_path) -> None:
    repo = _seed(tmp_path)
    for verdict in ("refuted", "unverifiable"):
        record_evidence(repo.store, node_id="U", field_path="payload.zh", value_hash=digest("克鲁斯尼克"),
                        verdict=verdict, evidence_kind="external", task_id="test", source_ref="https://x/y")
    assert _state(repo, "U") == UNCHECKED


def _refute(repo: KnowledgeRepo, node_id: str) -> None:
    node = repo.store.node(node_id)
    assert node is not None
    record_evidence(repo.store, node_id=node_id, field_path="payload.zh", value_hash=digest(node.payload["zh"]),
                    verdict="refuted", evidence_kind="external", task_id="test-refute", source_ref="https://x/z")


def test_a_refutation_outweighs_a_confirmation_but_not_a_human(tmp_path) -> None:
    repo = _seed(tmp_path)
    assert _state(repo, "C") == CHECKED
    _refute(repo, "C")
    assert _state(repo, "C") == UNCHECKED, "a check and a refutation of the same name disagree: look again"
    rev = _change(repo, "E", kind="user", zh="斯特雷尔卡号")
    book_revision_evidence(repo.store, rev, evidence_kind="user")
    _refute(repo, "E")
    assert _state(repo, "E") == HUMAN, "a person's name stands whatever a check said of it"


def _refute_at(repo: KnowledgeRepo, node_id: str, *, days_ago: float = 0, created_at: str | None = None,
               value: str | None = None) -> None:
    node = repo.store.node(node_id)
    assert node is not None
    record_evidence(repo.store, node_id=node_id, field_path="payload.zh",
                    value_hash=digest(node.payload["zh"] if value is None else value), verdict="refuted",
                    evidence_kind="external", task_id=f"test-refute-{days_ago}", source_ref="https://x/z")
    when = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    repo.store.conn.execute("UPDATE evidence SET created_at=? WHERE evidence_id=last_insert_rowid()",
                            (when if created_at is None else created_at,))


def _age(repo: KnowledgeRepo, days: float) -> None:
    """Move every revision and every evidence row ``days`` into the past, so what is booked next comes after it."""

    for table in ("revisions", "evidence"):
        for row in repo.store.conn.execute(f"SELECT rowid AS r, created_at FROM {table}").fetchall():
            if row["created_at"]:
                when = datetime.fromisoformat(row["created_at"]) - timedelta(days=days)
                repo.store.conn.execute(f"UPDATE {table} SET created_at=? WHERE rowid=?",
                                        (when.isoformat(timespec="seconds"), row["r"]))


def _status(repo: KnowledgeRepo, node_id: str):  # type: ignore[no-untyped-def]
    node = repo.store.node(node_id)
    assert node is not None
    return zh_statuses(repo.store, [node])[node_id]


def test_a_refutation_counts_only_against_the_version_it_was_booked_on(tmp_path) -> None:
    repo = _seed(tmp_path)
    _refute_at(repo, "U", days_ago=2)
    _change(repo, "U", zh="克什尼克")
    _change(repo, "U", zh="克鲁斯尼克")  # a knowledge update writes back the name a check corrected away
    status = _status(repo, "U")
    assert not status.refuted and status.state == UNCHECKED and check_due(status)


def test_a_refutation_with_no_later_write_is_terminal(tmp_path) -> None:
    repo = _seed(tmp_path)
    _age(repo, 2)
    _refute_at(repo, "U", days_ago=1)  # the correction was refused: nothing written since
    status = _status(repo, "U")
    assert status.refuted and not check_due(status)
    _change(repo, "U", desc="至冬的能量体系。")
    status = _status(repo, "U")
    assert not status.refuted and check_due(status), "any later write voids it; a new refusal books a new one"


def test_an_undated_refutation_predates_every_version(tmp_path) -> None:
    repo = _seed(tmp_path)
    _refute_at(repo, "U", created_at="")
    assert not _status(repo, "U").refuted


def test_confirmations_before_the_latest_refutation_do_not_count(tmp_path) -> None:
    repo = _seed(tmp_path)
    # a checked name corrected away and written back is unchecked, not checked
    _refute_at(repo, "C", days_ago=1)
    _change(repo, "C", zh="阿纳斯塔西亚")
    _change(repo, "C", zh="阿纳斯塔西娅")
    assert _state(repo, "C") == UNCHECKED
    # a checked name whose correction was refused, then only its description edited: unchecked, owed a check
    repo2 = _seed(tmp_path / "refused")
    _age(repo2, 2)
    _refute_at(repo2, "C", days_ago=1)
    _change(repo2, "C", desc="至冬女皇。")
    status = _status(repo2, "C")
    assert status.state == UNCHECKED and check_due(status)
    # one confirmation, refuted, written back: the next confirmation is the first again
    repo3 = _seed(tmp_path / "once")
    _confirm(repo3, "U", "external", source_ref="https://example.org/a", days_ago=RECHECK_GAP.days + 3)
    _refute_at(repo3, "U", days_ago=RECHECK_GAP.days + 2)
    _change(repo3, "U", zh="克什尼克")
    _change(repo3, "U", zh="克鲁斯尼克")
    _confirm(repo3, "U", "external", source_ref="https://example.org/b")
    status = _status(repo3, "U")
    assert status.state == UNCHECKED and len(status.confirmed) == 1


def test_a_human_revision_makes_the_name_the_humans(tmp_path) -> None:
    repo = _seed(tmp_path)
    rev = _change(repo, "C", kind="user", zh="阿纳斯塔西亚")
    book_revision_evidence(repo.store, rev, evidence_kind="user")
    assert _state(repo, "C") == HUMAN
    _change(repo, "C", desc="至冬女皇。")
    assert _state(repo, "C") == HUMAN, "the model editing the description keeps the human's name"
    _change(repo, "C", zh="安娜斯塔西娅")
    assert _state(repo, "C") == UNCHECKED


def test_an_ack_counts_only_while_the_line_carries_the_name_it_endorsed(tmp_path) -> None:
    repo = _seed(tmp_path)
    # an ack needs a mark: a human wrote the line (a bare user revision books no evidence), the model moved it away
    _change(repo, "U", kind="user", desc="人写的描述。")
    _change(repo, "U", desc="模型写的描述。")
    assert ack_lines(repo.store, "S", "U")
    assert _state(repo, "U") == HUMAN
    _change(repo, "U", desc="模型又改了描述。")
    assert _state(repo, "U") == HUMAN, "the acked version's name is still the name"
    _change(repo, "U", zh="克什尼克")
    assert _state(repo, "U") == UNCHECKED, "the acked version named it differently"


def test_a_line_without_a_chinese_name_has_no_state(tmp_path) -> None:
    repo = _seed(tmp_path)
    assert _state(repo, "Z") is None and _state(repo, "N") is None


# ---- the mark ----------------------------------------------------------------------------------------------------


def _server(monkeypatch, root, *, correct_only: bool) -> HarnessToolServer:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(root))
    server = HarnessToolServer(type("Ledger", (), {"record_kb_read": lambda self, **_: None})(),
                               assignment_id="a", worker_id="w", session_id="s", instance_id="i",
                               tools=list(KB_TOOL_NAMES))
    server._task = {"task_id": "t", "lease_generation": 1}
    server._manifest = {"metadata": {"knowledge_identity": f"rev:{KnowledgeRepo.open(root).rev}", "kb_tools": "read",
                                     **({"kb_correct_only": True} if correct_only else {})}}
    return server


_RPC_IDS = iter(range(1, 10_000))


def _tool(server: HarnessToolServer, name: str, **arguments) -> dict:
    reply = server._call(name, arguments, rpc_id=next(_RPC_IDS))
    assert reply["isError"] is False, reply
    return json.loads(reply["content"][0]["text"])


def _body(line: str) -> str:
    return line.split("  <!-- ", 1)[0]


def test_every_way_a_correct_only_task_reads_a_line_shows_the_same_mark(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    with pinned_generation_rev(tmp_path, correct_only=True):
        injected = load_entry_texts(tmp_path, ["原神"])[0]["原神"].splitlines()
        (hit,) = match_terms(tmp_path, "克鲁斯尼克的力量")
    server = _server(monkeypatch, tmp_path, correct_only=True)
    read = [_body(line) for line in _tool(server, "kb_read", entry="原神")["text"].splitlines()]
    by_section = [_body(line) for line in _tool(server, "kb_read", entry="原神", sections=["至冬"])["text"].splitlines()]
    handle = next(line.rsplit("<!-- ", 1)[1].split()[0] for line in _tool(server, "kb_read", entry="原神")["text"]
                  .splitlines() if line.startswith("クルースニク"))
    node = _tool(server, "kb_read_node", handle=handle)["text"]
    for lines in (injected, read, by_section):
        assert UNCHECKED_LINE in lines and CHECKED_LINE in lines and EMPTY_DESC_LINE in lines
        assert NO_NAME_LINE in lines, "a line without a name has no state and no mark"
    assert node == UNCHECKED_LINE == hit.line
    assert injected == read, "the pre-injected entry and kb_read are the same projection"


def test_no_other_face_carries_the_mark(tmp_path, monkeypatch) -> None:
    """A run that does not ask for the marks (a knowledge session's reads) reads as it always did, and so does the
    human projection, which is what `rendered/` and `show` print."""

    repo = _seed(tmp_path)
    with pinned_generation_rev(tmp_path):
        injected = load_entry_texts(tmp_path, ["原神"])[0]["原神"]
        assert [hit.line for hit in match_terms(tmp_path, "クルースニク")] == [UNCHECKED_LINE.replace(UNCHECKED_MARK, "")]
    plain = _server(monkeypatch, tmp_path, correct_only=False)
    faces = [injected, render_subject(repo.store, "S", mode="prompt"), repo.entry_text("S"),
             _tool(plain, "kb_read", entry="原神")["text"]]
    assert all(UNCHECKED_MARK not in face for face in faces)
    assert injected == render_subject(repo.store, "S", mode="prompt")


def test_the_mark_never_reaches_the_library_through_rendered(tmp_path) -> None:
    from finesub.llm.knowledge.node.edit import harvest_rendered_edits

    repo = _seed(tmp_path)
    repo.refresh_rendered()
    files = list((tmp_path / "rendered").rglob("*.md"))
    assert files and all(UNCHECKED_MARK not in path.read_text(encoding="utf-8") for path in files)
    rev = repo.rev
    assert harvest_rendered_edits(repo) == [] and repo.rev == rev
    assert UNCHECKED_MARK not in json.dumps([node.payload for node in repo.store.nodes_of_kind("term")],
                                            ensure_ascii=False)


def test_the_mark_follows_the_state_without_a_revision(tmp_path) -> None:
    """Evidence is not a revision: a line checked since the last write reads unmarked at the same revision."""

    repo = _seed(tmp_path)
    before = render_subject(repo.store, "S", mode="prompt", correct_only=True)
    _check(repo, "U", source_ref="https://example.org/a")
    after = render_subject(repo.store, "S", mode="prompt", correct_only=True)
    assert UNCHECKED_LINE in before and UNCHECKED_LINE.replace(UNCHECKED_MARK, "") in after


# ---- the task ---------------------------------------------------------------------------------------------------


ZH = [("zh", "克鲁斯尼克的力量。"), ("zh", "开始吧。")]
JA = [("ja", "クルースニクの力。"), ("ja", "はじめます。")]


def _task(tmp_path, root, langs, monkeypatch=None, **kwargs) -> dict:  # type: ignore[no-untyped-def]
    """The dry run's prompts (`on_plan`), or with ``monkeypatch`` what the task's own reads return: the plan
    previews no pre-injected entry, so the reads are taken where the task stands when it would run."""

    from finesub.llm import correction_translation as orchestration
    from finesub.llm.correction_request import CorrectionRequest
    from finesub.llm.routing.profiles import resolve_profile

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": [
        {"id": str(i), "start": i * 2.0, "end": i * 2.0 + 1.0, "text": text, "lang": lang}
        for i, (lang, text) in enumerate(langs, start=1)]}, ensure_ascii=False), encoding="utf-8")
    seen: dict = {}
    if monkeypatch is not None:
        def run_task(request, artifact_dir, language):  # type: ignore[no-untyped-def]
            seen["entries"] = load_entry_texts(root, ["原神"])[0]
            seen["terms"] = [hit.line for hit in match_terms(root, "クルースニク 克鲁斯尼克")]
            return orchestration.CorrectionOutcome()

        monkeypatch.setattr(orchestration, "_run_task", run_task)
    orchestration.run_correction_task(CorrectionRequest(
        stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan", knowledge_root=root,
        profile=resolve_profile("text", "local", "quality"), test_profile=True, on_plan=seen.update,
        extra_info="原神 至冬", **kwargs))
    if monkeypatch is None:
        return {"prompts": json.dumps([seen["research_messages"], seen["correction_messages"]], ensure_ascii=False)}
    return seen


def test_a_correct_only_task_reads_the_same_under_read_and_update(tmp_path, monkeypatch) -> None:
    from finesub.llm import correction_translation as orchestration

    monkeypatch.setattr(orchestration, "_correct_only_warning", lambda _message: None)
    root = tmp_path / "kb"
    _seed(root)
    with monkeypatch.context() as patched:
        reads = _task(tmp_path, root, ZH, patched, knowledge="read")
        assert reads == _task(tmp_path, root, ZH, patched, knowledge="update")
    assert UNCHECKED_LINE in reads["entries"]["原神"] and reads["terms"] == [UNCHECKED_LINE]


def test_a_translating_task_reads_the_marks_on_its_full_projection(tmp_path, monkeypatch) -> None:
    """A translating task sees the 未核 marks too (owner 2026-11-20), on the projection it always read: free
    sentences and all, nothing else changed."""

    root = tmp_path / "kb"
    repo = _seed(root)
    with monkeypatch.context() as patched:
        reads = _task(tmp_path, root, JA, patched)
    marked = render_subject(repo.store, "S", mode="prompt", trust_marks=True)
    assert reads["entries"]["原神"] == marked and reads["terms"] == [UNCHECKED_LINE]
    assert "至冬篇偏严肃。" in marked, "a translating task keeps the free sentences"
    assert marked.replace(UNCHECKED_MARK, "") == render_subject(repo.store, "S", mode="prompt")
    server = _server(monkeypatch, root, correct_only=False)
    server._manifest["metadata"]["kb_trust_marks"] = True
    assert UNCHECKED_LINE in _tool(server, "kb_read", entry="原神")["text"]
    _check(repo, "U", source_ref="https://example.org/a")
    with monkeypatch.context() as patched:
        assert _task(tmp_path, root, JA, patched)["terms"] == [UNCHECKED_LINE.replace(UNCHECKED_MARK, "")]


def test_the_not_found_mark_shows_only_while_it_holds_the_line_back(tmp_path) -> None:
    repo = _seed(tmp_path)
    node = repo.store.node("U")
    assert node is not None
    record_evidence(repo.store, node_id="U", field_path="payload.zh", value_hash=digest("克鲁斯尼克"),
                    verdict="unverifiable", evidence_kind="external", task_id="t1", source_ref=None, span="temporary")
    now = datetime.now(timezone.utc)
    assert trust_marks(repo.store, [node], now=now)["U"] == UNCHECKED_NOT_FOUND_MARK
    assert trust_marks(repo.store, [node], now=now + timedelta(days=31))["U"] == UNCHECKED_MARK
    record_evidence(repo.store, node_id="U", field_path="payload.zh", value_hash=digest("克鲁斯尼克"),
                    verdict="unverifiable", evidence_kind="external", task_id="t2", source_ref=None, span="permanent")
    assert trust_marks(repo.store, [node], now=now + timedelta(days=400))["U"] == UNCHECKED_NOT_FOUND_MARK
