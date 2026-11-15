"""A knowledge-update batch that does not validate as a whole loses only the ops that caused it.

What only the folded batch shows (`docs/knowledge.md`, 整批校验不过时的拒收) is traced back to the ops that wrote
it, which `apply_envelope(invalid_ops="reject")` rejects with the reason ``整批校验：…`` while the rest commits (what
the translation layer can see is refused there: `test_llm_knowledge_base.py`, `test_llm_knowledge_split.py`). A problem that traces to nothing the batch wrote still rolls the whole batch back, and every caller
but the knowledge update keeps the whole-batch rollback.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from finesub.llm.knowledge.node import apply as apply_mod
from finesub.llm.knowledge.node.apply import WHOLE_BATCH_PREFIX, apply_envelope
from finesub.llm.knowledge.node.envelope import Envelope
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.knowledge.update import _is_rejection

SRC = Path(__file__).resolve().parents[1] / "src" / "finesub"


# --- fixtures -----------------------------------------------------------------------------------------------------


def _seed(store: KnowledgeStore) -> None:
    """A common entry with a labelled 档案 and a 角色 section, and a style entry with one convention and its example."""

    with store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "游戏A", "intro": "一个游戏", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})
        txn.create_node("SN", "note", {"label": "本名", "text": "游戏A"})
        txn.create_membership("MSN", "S", "SN", "档案", 0)
        txn.create_node("SR", "note", {"label": "收录范围", "text": "只收角色名"})
        txn.create_membership("MSR", "S", "SR", "档案", 1)
        txn.create_node("ST", "term", {"surface": "アリス", "zh": "爱丽丝", "desc": "主角"})
        txn.create_membership("MST", "S", "ST", "角色", 0)
        txn.create_node("Y", "subject", {"surface": "某组", "intro": "某字幕组的风格", "category": "style",
                                         "section_order": ["档案", "约定", "正例", "反例", "待归类"]})
        txn.create_node("YN", "note", {"label": "本名", "text": "某组"})
        txn.create_membership("MYN", "Y", "YN", "档案", 0)
        txn.create_node("YC", "note", {"label": "引用标注", "text": "引用他人的话用「」"})
        txn.create_membership("MYC", "Y", "YC", "约定", 0)
        txn.create_node("YE", "note", {"label": "引用标注", "text": "甲|乙 → 甲说「不去」"})
        txn.create_membership("MYE", "Y", "YE", "正例", 0)
        txn.create_node("YS", "note", {"label": "自称", "text": "甲|乙 → 我说的"})
        txn.create_membership("MYS", "Y", "YS", "待归类", 0)


def _store(tmp_path: Path) -> KnowledgeStore:
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    _seed(store)
    return store


def _envelope(store: KnowledgeStore, ops: list[dict]) -> Envelope:
    """Raw ids, no handles: the engine without the translation layer, so a fixture reaches `validate_overlay`."""

    return Envelope(
        task_id="t", assignment_id="a", context_epoch=0, knowledge_read_rev=store.current_rev(), ops=ops,
        handle_bindings=[],
        draft_bindings=[op["handle"] for op in ops if op.get("op") == "create" and op.get("handle")],
    )


def _note(parent: str, section: str, label: str, text: str, handle: str | None = None) -> dict:
    op = {"op": "create", "kind": "note", "parent": parent, "section": section,
          "payload": {"label": label, "text": text} if label else {"text": text}}
    if handle:
        op["handle"] = handle
    return op


def _repo(tmp_path: Path) -> KnowledgeRepo:
    KnowledgeRepo.forget(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    _seed(repo.store)
    return repo


def _apply(repo: KnowledgeRepo, *proposals: dict, invalid_ops: str = "reject", allow=("streamer", "common", "style")):
    handles = HandleMap()
    for subject in repo.subjects():
        repo.entry_prompt_text(subject.local_id, handles)
    text = "<knowledge_proposals>\n" + "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals) \
        + "\n</knowledge_proposals>"
    report = apply_model_proposals(text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
                                   allow_categories=allow, invalid_ops=invalid_ops)
    return report, handles, text


def _handle(handles: HandleMap, node_id: str) -> str:
    return next(b["handle"] for b in handles.bindings() if b["kind"] == "node" and b["id"] == node_id)


def _labels(store: KnowledgeStore, subject: str, section: str) -> list[str]:
    return sorted(
        str(store.node(m.child_id).payload.get("label") or "")
        for m in store.children(subject) if m.section == section
    )


# --- duplicate labels: who is blamed --------------------------------------------------------------------------------


@pytest.mark.parametrize("run", range(5))
def test_two_new_lines_with_one_label_lose_the_later_one_every_time(tmp_path, run) -> None:
    """New against new: the earlier write wins, whatever ids the rebuild draws."""

    with _store(tmp_path) as store:
        ops = [_note("S", "档案", "当前版本", "2.7"), _note("S", "档案", "当前版本", "2.8"),
               {"op": "update", "id": "ST", "set": {"payload.desc": "女主角"}}]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        assert not result.rolled_back and result.rev is not None
        assert [index for index, _ in result.rejected_ops] == [1]
        assert result.rejected_ops[0][1].startswith(WHOLE_BATCH_PREFIX + "标记 [当前版本]")
        assert result.applied_ops == [0, 2]
        texts = [store.node(m.child_id).payload["text"] for m in store.children("S") if m.section == "档案"]
        assert "2.7" in texts and "2.8" not in texts


def test_a_new_line_loses_to_the_stored_one_and_the_rollback_names_it(tmp_path) -> None:
    """New against stored: the new one is rejected. The rollback the other callers keep now names the line the
    batch wrote too, not whichever node id sorted later."""

    with _store(tmp_path) as store:
        ops = [_note("S", "档案", "收录范围", "什么都收", handle="@n"),
               {"op": "update", "id": "ST", "set": {"payload.desc": "女主角"}}]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")
        assert [index for index, _ in result.rejected_ops] == [0] and result.applied_ops == [1]
        assert store.node("SR").payload["text"] == "只收角色名"

    with KnowledgeStore(tmp_path / "rollback.sqlite") as store:
        _seed(store)
        result = apply_envelope(store, _envelope(store, ops))
        assert result.rolled_back and f"node {result.created['@n']}: 标记 [收录范围]" in result.rollback_reason
        assert "node SR" not in result.rollback_reason


@pytest.mark.parametrize("mode", ("reject", "rollback"))
def test_rewriting_the_stored_line_s_body_does_not_make_it_the_new_one(tmp_path, mode) -> None:
    """Peer review 2026-11-09: a body rewrite of the stored [收录范围] used to
    count as the batch writing that line, so it outranked the real duplicate
    and was rejected first -- both edits lost, and the repair round was told
    an update should "use update"."""

    with _store(tmp_path) as store:
        ops = [
            _note("S", "档案", "收录范围", "什么都收", handle="@n"),
            {"op": "update", "id": "SR", "set": {"payload.text": "只收角色名与地名"}},
            {"op": "update", "id": "ST", "set": {"payload.desc": "女主角"}},
        ]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops=mode)
        if mode == "reject":
            assert [index for index, _ in result.rejected_ops] == [0] and result.applied_ops == [1, 2]
            assert store.node("SR").payload["text"] == "只收角色名与地名"
        else:
            assert result.rolled_back
            assert "node SR" not in result.rollback_reason


def test_an_update_onto_a_stored_label_is_rejected_and_the_record_says_so(tmp_path) -> None:
    """The translation layer does not check an `update`'s label, so this reaches `validate_overlay`: the update is
    rejected, the other proposal commits, and each report record still matches its op."""

    repo = _repo(tmp_path)
    handles = HandleMap()
    for subject in repo.subjects():
        repo.entry_prompt_text(subject.local_id, handles)
    relabel = {"op": "update", "id": _handle(handles, "SR"), "line": "[本名] 游戏A 的另一个名字", "reason": "r"}
    add = {"op": "append_lines", "category": "common", "entry": "游戏A", "section": "角色",
           "content": "ボブ|鲍勃||配角", "reason": "r"}
    report, _handles, _text = _apply(repo, relabel, add)

    assert not report.rolled_back and report.rev is not None
    assert [(r.op, r.reason.startswith(WHOLE_BATCH_PREFIX + "标记 [本名]")) for r in report.skipped] == [("update", True)]
    assert [r.op for r in report.applied] == ["append_lines"]
    assert repo.store.node("SR").payload["label"] == "收录范围"
    assert _is_rejection(report.to_dict()["skipped"][0])
    # And it reaches the repair round as ITS proposal: the update names its line by handle, so matching by entry
    # alone quoted the append_lines that had landed (peer review 2026-11-09).
    from finesub.llm.knowledge.update import _dropped_rows_text, rejected_entries

    pairs = rejected_entries(report.to_dict())
    assert pairs == [("common", "游戏A")]
    shown = _dropped_rows_text(report.to_dict(), pairs, keep=_is_rejection, proposal_text=_text)
    assert '"op": "update"' in shown and "append_lines" not in shown


# --- other problems -------------------------------------------------------------------------------------------------


def test_a_section_over_its_cap_rejects_the_lines_this_batch_put_there(tmp_path) -> None:
    """Every edge the batch added to the full section goes, with the reason; the rest of the batch commits and the
    report records match the ops. The translation layer does not count lines, so this is the engine's."""

    repo = _repo(tmp_path)
    with repo.store.begin("import") as txn:  # 约定 at its cap of 20
        for n in range(19):
            txn.create_node(f"YC{n}", "note", {"label": f"约定{n}", "text": f"第 {n} 条"})
            txn.create_membership(f"MYC{n}", "Y", f"YC{n}", "约定", n + 1)
    conventions = {"op": "append_lines", "category": "style", "entry": "某组", "section": "约定",
                   "content": "[语气词] 语气词不译\n[数字] 数字用阿拉伯数字", "reason": "r"}
    example = {"op": "append_lines", "category": "style", "entry": "某组", "section": "反例",
               "content": "甲|乙 → 甲说“不去”", "reason": "r"}
    report, _handles, _text = _apply(repo, conventions, example)

    assert not report.rolled_back and report.rev is not None
    assert [(r.op, r.section) for r in report.applied] == [("append_lines", "反例")]
    assert [r.section for r in report.skipped] == ["约定", "约定"]
    assert all(r.reason.startswith(WHOLE_BATCH_PREFIX + "小节「约定」最多 20 行") for r in report.skipped)
    assert len(_labels(repo.store, "Y", "约定")) == 20


def test_engine_level_rejection_keeps_conflicts_rejections_and_created_apart(tmp_path) -> None:
    with _store(tmp_path) as store:
        ops = [_note("S", "档案", "收录范围", "什么都收", handle="@bad"),
               _note("S", "档案", "当前版本", "2.7", handle="@good")]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        assert result.conflicts == []
        assert {index for index, _ in result.rejected_ops} == {0} and result.applied_ops == [1]
        assert store.node(result.created["@good"]).payload["text"] == "2.7"
        assert set(result.created) == {"@good"}  # the rejected create was never folded


def test_relabelling_a_convention_rejects_only_the_relabel(tmp_path) -> None:
    """`labels_from`: the stored example is not to blame for the convention the batch took away."""

    with _store(tmp_path) as store:
        ops = [{"op": "update", "id": "YC", "set": {"payload.label": "引号"}},
               _note("Y", "反例", "", "甲|乙 → 甲说“不去”")]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        assert [index for index, _ in result.rejected_ops] == [0] and result.applied_ops == [1]
        assert "在「约定」里没有对应的那一条" in result.rejected_ops[0][1]
        assert store.node("YC").payload["label"] == "引用标注"


def test_moving_an_orphan_example_in_rejects_only_the_move(tmp_path) -> None:
    with _store(tmp_path) as store:
        ops = [{"op": "move_membership", "membership": "MYS", "section": "正例"},
               _note("Y", "反例", "", "甲|乙 → 甲说“不去”")]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        assert [index for index, _ in result.rejected_ops] == [0] and result.applied_ops == [1]
        assert [m.section for m in store.parents("YS")] == ["待归类"]


def test_what_hangs_off_a_rejected_line_goes_with_it_under_the_root_reason(tmp_path) -> None:
    with _store(tmp_path) as store:
        ops = [_note("S", "档案", "收录范围", "什么都收", handle="@n"),
               {"op": "add_item", "id": "@n", "field": "misheard", "value": "什么都手"},
               _note("S", "档案", "当前版本", "2.7")]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        reasons = dict(result.rejected_ops)
        assert set(reasons) == {0, 1} and result.applied_ops == [2]
        assert reasons[1] == apply_mod.VOIDED_WITH_PREFIX + reasons[0]
        assert "@n" not in reasons[1]


def test_a_problem_traced_to_nothing_the_batch_wrote_rolls_back_and_says_which(tmp_path) -> None:
    """A duplicate already in the store, in a section the batch touches: no op of the batch explains it."""

    with _store(tmp_path) as store:
        with store.begin("import") as txn:  # legacy: two stored lines with one label
            txn.create_node("SX", "note", {"label": "收录范围", "text": "也收地名"})
            txn.create_membership("MSX", "S", "SX", "档案", 2)
        ops = [_note("S", "档案", "当前版本", "2.7")]
        result = apply_envelope(store, _envelope(store, ops), invalid_ops="reject")

        assert result.rolled_back and result.rev is None
        assert "标记 [收录范围]" in result.rollback_reason and "内重复" in result.rollback_reason


# --- the report and the repair round --------------------------------------------------------------------------------


def test_with_every_real_change_rejected_the_date_stamp_writes_no_revision(tmp_path) -> None:
    repo = _repo(tmp_path)
    handles = HandleMap()
    for subject in repo.subjects():
        repo.entry_prompt_text(subject.local_id, handles)
    rev = repo.rev
    relabel = {"op": "update", "id": _handle(handles, "SR"), "line": "[本名] 游戏A 的另一个名字", "reason": "r"}
    report, _handles, _text = _apply(repo, relabel)

    assert report.rev is None and not report.rolled_back and repo.rev == rev
    assert [r.op for r in report.skipped] == ["update"] and report.applied == []


def test_the_identity_line_is_not_bookkeeping(tmp_path, monkeypatch) -> None:
    """Only the `updated_date` stamps are passed as bookkeeping: the added `[本名]` line is real data."""

    from finesub.llm.knowledge.node import proposals as proposals_mod

    seen: dict = {}
    real = proposals_mod.apply_envelope

    def spy(store, envelope, **kwargs):
        seen["ops"], seen["bookkeeping"] = envelope.ops, list(kwargs["bookkeeping_ops"])
        return real(store, envelope, **kwargs)

    monkeypatch.setattr(proposals_mod, "apply_envelope", spy)
    repo = _repo(tmp_path)
    _apply(repo, {"op": "create_entry", "category": "common", "entry": "游戏B", "entry_type": "游戏",
                  "intro": "另一个游戏", "reason": "库中没有"},
           {"op": "append_lines", "category": "common", "entry": "游戏A", "section": "角色",
            "content": "ボブ|鲍勃||配角", "reason": "r"})

    identity = [i for i, op in enumerate(seen["ops"]) if op.get("payload", {}).get("label") == "本名"]
    stamps = [i for i, op in enumerate(seen["ops"]) if op.get("set") == {"payload.updated_date": repo.today()}]
    assert identity and stamps and seen["bookkeeping"] == stamps and not set(identity) & set(stamps)


def test_every_whole_batch_problem_text_reads_as_a_rejection_the_model_can_fix() -> None:
    """Each text `validate_overlay` can produce, behind the prefix, is a rejection for the repair round -- not a
    conflict, not "already there". Read off the checks' source, so a new text is covered when it is written."""

    tree = ast.parse((SRC / "llm" / "knowledge" / "node" / "apply.py").read_text(encoding="utf-8"))
    checks = [node for node in tree.body if isinstance(node, ast.FunctionDef)
              and (node.name == "_overlay_problems" or node.name.startswith("_check_"))]
    texts: list[str] = []
    for function in checks:
        for call in ast.walk(function):
            if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_Problem" and len(call.args) >= 3:
                pieces = [n.value for n in ast.walk(call.args[2]) if isinstance(n, ast.Constant)
                          and isinstance(n.value, str)]
                texts.append("".join(pieces) or "<dynamic>")
    assert len(texts) >= 15, texts
    from finesub.llm.knowledge.node.model import validate_payload

    for kind, payload in (("note", {}), ("note", {"text": " "}), ("term", {"desc": "x"}), ("term", {"surface": "",
                          "desc": "x"}), ("subject", {"surface": "", "intro": "i", "category": "common"})):
        with pytest.raises(ValueError) as caught:
            validate_payload(kind, payload)
        texts.append(str(caught.value))
    for text in texts:
        assert _is_rejection({"reason": WHOLE_BATCH_PREFIX + text}), text
        assert _is_rejection({"reason": apply_mod.VOIDED_WITH_PREFIX + WHOLE_BATCH_PREFIX + text}), text


def test_only_the_knowledge_update_rejects_every_other_caller_rolls_back() -> None:
    """`knowledge edit`, the maintenance session, the base-style migration, `knowledge verify` and the shared-
    conflict repair keep the whole-batch rollback: only `update.py` passes ``invalid_ops="reject"`` -- in the main
    round and in the repair round."""

    passing: dict[str, list[str]] = {}
    for path in SRC.rglob("*.py"):
        for call in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for keyword in getattr(call, "keywords", ()) if isinstance(call, ast.Call) else ():
                if keyword.arg == "invalid_ops":
                    passing.setdefault(path.relative_to(SRC).as_posix(), []).append(ast.unparse(keyword.value))
    assert passing.pop("llm/knowledge/update.py") == ["'reject'", "'reject'"]
    assert passing == {"llm/knowledge/node/proposals.py": ["invalid_ops", "invalid_ops"]}  # passed through


def test_the_default_rolls_back_the_same_fixture(tmp_path) -> None:
    with _store(tmp_path) as store:
        ops = [_note("S", "档案", "当前版本", "2.7"), _note("S", "档案", "当前版本", "2.8")]
        rev = store.current_rev()
        result = apply_envelope(store, _envelope(store, ops))
        assert result.rolled_back and store.current_rev() == rev
