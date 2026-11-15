"""Refined-alignment marks on a style's conventions (batch-2 item 3): the
`<style_marks>` block, the harness check, `judged` evidence, the share export
that leaves it out, and the report that lists it per line."""

from __future__ import annotations

import re

from finesub.llm.client import LLMCallResult
from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.signals import EVIDENCE_KIND_JUDGED, book_event, record_evidence
from finesub.llm.knowledge.report import build_report
from finesub.llm.knowledge.share.exchange import build_push_bundle
from finesub.llm.knowledge.style_marks import (
    book_style_marks,
    convention_section,
    parse_style_marks,
    refined_start_times,
)
from finesub.llm.prompts import build_knowledge_update_messages
from finesub.llm.routing.config import LLMRole

from .test_llm_knowledge_update import FakeTokenCounter, _write_task_outputs

RULE = "说话者用自己名字自指时译文改用「我」"
EXAMPLE = "Nono酱小时候 → 我小时候"


def _repo(root) -> KnowledgeRepo:
    KnowledgeRepo.forget(root)
    return KnowledgeRepo.open(root)


def _style(root) -> tuple[KnowledgeRepo, str]:
    assert maintain_main(["--root", str(root), "new", "style", "某字幕组", "--intro", "一套风格"]) == 0
    assert maintain_main(
        ["--root", str(root), "new", "common", "游戏X", "--intro", "一个游戏", "--type", "游戏"]
    ) == 0
    repo = _repo(root)
    sid = repo.subjects("style")[0].local_id
    gid = repo.subjects("common")[0].local_id
    with repo.store.begin("import") as txn:
        txn.create_node("C1", "note", {"text": RULE, "label": "自称转换"})
        txn.create_membership("MC1", sid, "C1", "约定", 0)
        txn.create_node("E1", "note", {"text": EXAMPLE, "label": "自称转换"})
        txn.create_membership("ME1", sid, "E1", "正例", 0)
        txn.create_node("T1", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("MT1", gid, "T1", "角色", 0)
    return repo, sid


def _handles(repo: KnowledgeRepo) -> HandleMap:
    handles = HandleMap()
    for subject in repo.subjects():
        render_subject(repo.store, subject.local_id, mode="prompt", handles=handles)
    return handles


def _handle(handles: HandleMap, node_id: str) -> str:
    return next(h for h, (ident, _rev) in handles.nodes.items() if ident == node_id)


def _book(repo, sid, handles, body: str, starts=frozenset({"12.3", "15.0"})):  # type: ignore[no-untyped-def]
    return book_style_marks(
        repo.store, f"<knowledge_proposals>\n</knowledge_proposals>\n<style_marks>\n{body}\n</style_marks>",
        handles=handles, style_subject_id=sid, refined_starts=set(starts),
        read_rev=repo.rev, task_id="task-1",
    )


def _judged(repo) -> list:  # type: ignore[no-untyped-def]
    return repo.store.conn.execute(
        "SELECT node_id, field_path, verdict, span, evidence_kind FROM evidence"
        " WHERE evidence_kind=? ORDER BY span", (EVIDENCE_KIND_JUDGED,)
    ).fetchall()


def test_the_convention_section_comes_from_the_preset() -> None:
    assert convention_section("style") == "约定"


def test_a_valid_mark_is_booked_as_judged_evidence(tmp_path) -> None:
    repo, sid = _style(tmp_path)
    handles = _handles(repo)
    c1 = _handle(handles, "C1")
    report = _book(repo, sid, handles, f"{c1}|confirmed|12.3|精修把自称译成了我")
    assert report == {"booked": 1, "duplicates": 0, "dropped": []}
    (row,) = _judged(repo)
    assert (row["node_id"], row["field_path"], row["verdict"], row["span"]) == (
        "C1", "payload.text", "confirmed", "12.3"
    )


def test_the_same_mark_on_the_same_refined_row_counts_once(tmp_path) -> None:
    repo, sid = _style(tmp_path)
    handles = _handles(repo)
    c1 = _handle(handles, "C1")
    report = _book(
        repo, sid, handles,
        f"{c1}|confirmed|12.3|一\n{c1}｜confirmed｜12.3｜重复（全角竖线也认）\n{c1}|refuted|15.0|另一行推翻了",
    )
    assert report["booked"] == 2 and report["duplicates"] == 1
    assert [(r["verdict"], r["span"]) for r in _judged(repo)] == [("confirmed", "12.3"), ("refuted", "15.0")]


def test_marks_that_do_not_check_out_are_dropped_and_counted(tmp_path) -> None:
    repo, sid = _style(tmp_path)
    handles = _handles(repo)
    c1, e1, t1 = (_handle(handles, n) for n in ("C1", "E1", "T1"))
    report = _book(
        repo, sid, handles,
        "\n".join(
            [
                f"{e1}|confirmed|12.3|正例行不标",
                f"{t1}|confirmed|12.3|别的条目的行",
                "@k99|confirmed|12.3|本次没渲染的句柄",
                f"{c1}|confirmed|99.9|不在本组精修行里",
                f"{c1}|maybe|12.3|不认识的判定",
                f"{c1}|confirmed",
            ]
        ),
    )
    assert report["booked"] == 0 and len(report["dropped"]) == 6
    assert [d["reason"] for d in report["dropped"]] == [
        "not a line of the pinned style's convention section",
        "not a line of the pinned style's convention section",
        "handle not rendered in this call",
        "start time is not one of this call's refined rows",
        "verdict must be one of ('confirmed', 'refuted')",
        "malformed: needs 句柄|verdict|开始时间|理由",
    ]
    assert _judged(repo) == []


def test_no_block_no_marks() -> None:
    assert parse_style_marks("<knowledge_proposals>\n</knowledge_proposals>") == []


def test_start_times_are_read_as_this_call_rendered_them() -> None:
    csv = "# start|end|text\n12.3|13.0|你好\n15.0|16.2|再见\n"
    assert refined_start_times([csv, ""]) == {"12.3", "15.0"}


def _messages(**overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(refined=True, task_summary="t", window_packs="w", kb_entries="k")
    kwargs.update(overrides)
    return "\n".join(m["content"] for m in build_knowledge_update_messages(**kwargs))


def test_the_block_is_asked_for_only_with_refined_subtitles_and_a_pinned_style() -> None:
    assert "<style_marks>" in _messages(style_marks_section="约定")
    assert "「约定」节" in _messages(style_marks_section="约定")
    assert "<style_marks>" not in _messages()
    assert "<style_marks>" not in _messages(refined=False, style_marks_section="约定")
    # without a style the refined prompt is exactly what it was
    assert _messages() == _messages(style_marks_section=None)
    assert (
        "最后提醒（读完以上全部输入后）：输出有且仅有一个"
        " `<knowledge_proposals>` 块（可为空块）；除上述块外不要输出任何其他文字。"
    ) in _messages()
    assert "（可为空块），其后再输出一个 `<style_marks>` 块（可为空块）；" in _messages(
        style_marks_section="约定"
    )


def test_the_share_export_leaves_judged_evidence_home(tmp_path) -> None:
    repo, sid = _style(tmp_path)
    record_evidence(repo.store, node_id="C1", field_path="payload.text", value_hash="h",
                    verdict="confirmed", evidence_kind=EVIDENCE_KIND_JUDGED, task_id="t", span="12.3")
    record_evidence(repo.store, node_id="C1", field_path="payload.text", value_hash="h",
                    verdict="confirmed", evidence_kind="user", task_id="t")
    with repo.store.begin("user") as txn:
        for node_id in (sid, "C1"):
            txn.update_node(node_id, visibility="shareable")
    bundle = build_push_bundle(repo.store, [sid], idempotency_key="k")
    kinds = [kind for claim in bundle["claim_summaries"] for kind in claim["evidence_kinds"]]
    assert "user" in kinds and EVIDENCE_KIND_JUDGED not in kinds


def test_style_usage_is_not_a_node_exposure_in_the_share_bundle(tmp_path) -> None:
    """`exposed` with opportunity style_injected is the report's usage count;
    the bundle's per-node `exposed_count` leaves it out, as the report does."""

    repo, sid = _style(tmp_path)
    with repo.store.begin("user") as txn:
        txn.update_node(sid, visibility="shareable")
    for task, opportunity in (("t1", "style_injected"), ("t2", "style_injected"), ("t3", "context")):
        book_event(repo.store, {"kind": "exposed", "opportunity": opportunity, "task_id": task,
                                "subject_id": sid, "node_id": sid, "rev": repo.rev,
                                "algo_version": "x"})
    bundle = build_push_bundle(repo.store, [sid], idempotency_key="k")
    counts = {claim["exposed_count"] for claim in bundle["claim_summaries"]}
    assert counts == {1}


def test_the_report_lists_judged_counts_per_convention_line(tmp_path) -> None:
    repo, sid = _style(tmp_path)
    handles = _handles(repo)
    c1 = _handle(handles, "C1")
    _book(repo, sid, handles, f"{c1}|confirmed|12.3|a\n{c1}|confirmed|15.0|b")
    lines = build_report(repo.store)
    section = lines[lines.index("## style 约定的精修对照（judged：模型判断，与上面的确定式证据分开计数）"):]
    assert "### 某字幕组（使用率：未记）" in section
    assert any(line.startswith("- [自称转换]") and "遵守 2 · 推翻 0" in line for line in section)
    # judged rows are not claim evidence: the claim lines never count them
    assert not any(line.strip().startswith("evidence") and "confirmed 2" in line for line in lines)
    # once the correction side books usage, the report counts distinct tasks
    for task in ("t1", "t1", "t2"):
        book_event(repo.store, {"kind": "exposed", "opportunity": "style_injected", "task_id": task,
                                "subject_id": sid, "node_id": sid, "rev": repo.rev,
                                "algo_version": "style-injected-1"})
    lines = build_report(repo.store)
    assert "### 某字幕组（使用率：2 个任务）" in lines
    # usage is the style section's number only, never a node's exposed count
    assert all("exposed 0 " in line for line in lines if line.startswith("- 某字幕组:"))


# ---- the knowledge update end to end ------------------------------------------------


class _MarkingClient:
    """Answers with an empty proposal block and one mark on the convention
    line, whose handle it reads from the prompt it was given."""

    def __init__(self, start: str) -> None:
        self.start = start
        self.calls: list = []

    def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        user = messages[1]["content"]
        handle = re.search(re.escape(RULE) + r"\s*<!-- (@k\d+) -->", user).group(1)
        content = (
            "<knowledge_proposals>\n</knowledge_proposals>\n"
            f"<style_marks>\n{handle}|confirmed|{self.start}|精修照做了\n</style_marks>"
        )
        return LLMCallResult(content=content, role=LLMRole.GENERAL_CAPABLE, model="fake",
                             fallback_used=False, raw_response={})


def _run(tmp_path, start: str):  # type: ignore[no-untyped-def]
    from finesub.llm.knowledge.update import run_knowledge_update

    final_srt, _paths, refined = _write_task_outputs(tmp_path, with_refined=True)
    knowledge_root = tmp_path / "knowledge"
    knowledge_root.mkdir()
    _style(knowledge_root)
    client = _MarkingClient(start)
    report = run_knowledge_update(
        final_srt=final_srt, refined_srt=refined, task_id="task-1", task_summary="t",
        knowledge_root=knowledge_root, token_counter=FakeTokenCounter(), client=client,
        style_names=["某字幕组"],
    )
    return report, client, _repo(knowledge_root)


def test_a_refined_update_books_the_marks_its_call_can_back(tmp_path) -> None:
    report, client, repo = _run(tmp_path, "0.0")  # the refined row the call rendered
    assert "<style_marks>" in client.calls[0][0]["content"] + client.calls[0][1]["content"]
    assert report["chunks"][0]["style_marks"]["booked"] == 1
    assert [(r["node_id"], r["span"]) for r in _judged(repo)] == [("C1", "0.0")]


def test_a_refined_update_drops_a_time_from_outside_its_rows(tmp_path) -> None:
    report, _client, repo = _run(tmp_path, "2.0")  # a final-SRT time, not a refined row
    marks = report["chunks"][0]["style_marks"]
    assert marks["booked"] == 0 and len(marks["dropped"]) == 1
    assert _judged(repo) == []
