"""Compression notices for entries past their soft size (plan kb-entry-size,
step 2): only on surfaces that can write, asking for at least X% with no
target length or token count; an entry asked to compress is shown whole."""

from __future__ import annotations

import json
import tomllib

import pytest

from finesub.llm.knowledge.entries import EntrySelection, render_kb_entry_excerpt
from finesub.llm.knowledge.entry_size import compression_percent, size_notice
from finesub.llm.knowledge.node.presets import (
    _preset_folder,
    PresetError,
    parse_preset,
    preset_for_category,
)
from finesub.llm.knowledge.node.render import entry_prompt_tokens
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _grow(root, surface: str, lines: int, *, width: int = 60, category: str = "common") -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    KnowledgeRepo.forget(root)
    repo = KnowledgeRepo.open(root)
    sid = f"S-{surface}"
    with repo.store.begin("import") as txn:
        txn.create_node(sid, "subject", {"surface": surface, "intro": "一个游戏", "category": category,
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})
        for i in range(lines):
            txn.create_node(f"{sid}-T{i}", "note", {"text": f"第{i}行" + "很长的描述" * (width // 5)})
            txn.create_membership(f"{sid}-M{i}", sid, f"{sid}-T{i}", "角色", i)
    return repo


def test_every_category_declares_the_soft_size_and_whether_it_splits() -> None:
    assert {c: preset_for_category(c).soft_entry_tokens for c in ("common", "streamer", "style")} == {
        "common": 5000, "streamer": 5000, "style": 5000
    }
    assert [preset_for_category(c).splittable for c in ("common", "streamer", "style")] == [True, False, False]


def test_a_soft_size_at_or_over_the_cap_is_refused() -> None:
    data = tomllib.loads((_preset_folder() / "common.toml").read_text(encoding="utf-8"))
    data["soft_entry_tokens"] = data["max_entry_tokens"]
    with pytest.raises(PresetError, match="soft_entry_tokens"):
        parse_preset(data)


@pytest.mark.parametrize(("tokens", "percent"), [(5000, 20), (6000, 35), (8000, 50), (4000, 0)])
def test_the_cut_is_worked_out_toward_four_fifths_of_the_soft_size(tokens, percent) -> None:
    assert compression_percent(tokens, 5000) == percent


def test_the_notice_names_a_percentage_and_nothing_about_length(tmp_path) -> None:
    repo = _grow(tmp_path, "甲", 10)
    assert size_notice(repo.store, "S-甲") == ""
    repo = _grow(tmp_path / "big", "乙", 160)
    tokens = entry_prompt_tokens(repo.store, "S-乙")
    assert tokens > 6000
    notice = size_notice(repo.store, "S-乙")
    compress = notice.splitlines()[0]  # the split paragraph that may follow quotes per-section sizes
    assert f"至少删减 {compression_percent(tokens, 5000)}%" in compress and "截掉" in compress
    assert "4000" not in compress and str(tokens) not in compress and "token" not in compress


def test_an_entry_asked_to_compress_is_prefetched_whole_past_the_cap(tmp_path) -> None:
    repo = _grow(tmp_path, "乙", 160)
    assert entry_prompt_tokens(repo.store, "S-乙") > 6000
    block, handles = render_kb_entry_excerpt(
        [EntrySelection(category="common", key="乙", score=1.0, exists=True)],
        tmp_path, count_tokens=len,
    )
    assert block.included == ("common/乙",) and "至少删减" in block.text
    assert len([h for h in handles.nodes]) == 161  # every line kept its handle


def test_one_pushed_back_into_the_shared_rest_is_cut_and_asked_nothing(tmp_path) -> None:
    _grow(tmp_path, "乙", 160)
    block, _ = render_kb_entry_excerpt(
        [EntrySelection(category="common", key="乙", score=1.0, exists=True)],
        tmp_path, count_tokens=len, block_limit=3000,
    )
    assert block.truncated == ("common/乙",) and "至少删减" not in block.text


def test_a_writer_reading_the_whole_entry_gets_the_notice_a_reader_does_not(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer

    _grow(tmp_path, "乙", 160)

    class _Ledger:
        def record_kb_read(self, **kwargs):  # type: ignore[no-untyped-def]
            pass

    def reply(entitlement: str) -> str:
        monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(tmp_path))
        server = HarnessToolServer(
            _Ledger(), assignment_id="a", worker_id="w", session_id="s", instance_id="i",
            tools=list(KB_TOOL_NAMES),
        )
        server._task = {"task_id": "t", "lease_generation": 1}
        server._manifest = {"metadata": {"knowledge_identity": "rev:1", "kb_tools": entitlement}}
        out = server._call("kb_read", {"entry": "乙"}, rpc_id=1)
        assert out["isError"] is False, out["content"][0]["text"]
        return json.loads(out["content"][0]["text"])["text"]

    assert reply("propose").startswith("（⚠ 本条目偏长")
    assert "至少删减" not in reply("read")


def test_the_narrow_repairs_are_not_asked_to_compress(tmp_path) -> None:
    """Owner 2026-10-03 (cold read Q4): candidate repair, material absorption
    and the conflict repair may only change what their input names, so the
    notice -- and the split offer that rides on it -- stays off them. The
    conflict repair still sees the oversized entry whole."""

    from finesub.llm.knowledge.node.repair import render_repair_prompt

    repo = _grow(tmp_path, "乙", 160)
    material = render_repair_prompt(repo, "S-乙", material="一段素材")
    candidates = render_repair_prompt(repo, "S-乙", candidates=[{"kind": "term", "surface": "甲"}])
    for prompt in (material, candidates):
        assert "至少删减" not in prompt and "split_entry" not in prompt
    assert "并对每条候选给出裁定" in candidates and "并对每条候选给出裁定" not in material
    block, handles = render_kb_entry_excerpt(
        [EntrySelection(category="common", key="乙", score=1.0, exists=True)],
        tmp_path, count_tokens=len, ask_to_compress=False,
    )
    assert block.included == ("common/乙",) and "至少删减" not in block.text
    assert len([h for h in handles.nodes]) == 161


def test_the_report_lists_entries_past_the_cap(tmp_path) -> None:
    from finesub.llm.knowledge.report import build_report

    repo = _grow(tmp_path, "乙", 160)
    lines = build_report(repo.store)
    start = lines.index(next(line for line in lines if line.startswith("## 超过注入上限的条目")))
    assert lines[start + 1].startswith("- 乙（common）：约 ") and "上限 6000" in lines[start + 1]
    assert "纠错注入时少了：角色" in lines[start + 1]


def test_of_two_entries_asked_to_compress_the_first_goes_whole_with_the_notice(tmp_path) -> None:
    from finesub.llm.knowledge.node.repo import KnowledgeRepo as Repo

    _grow(tmp_path, "甲", 160)
    store_root = tmp_path
    repo = Repo.open(store_root)
    with repo.store.begin("import") as txn:
        sid = "S-乙"
        txn.create_node(sid, "subject", {"surface": "乙", "intro": "一个游戏", "category": "common",
                                         "entry_type": "游戏", "section_order": ["档案", "角色"]})
        for i in range(160):
            txn.create_node(f"{sid}-T{i}", "note", {"text": f"第{i}行" + "很长的描述" * 12})
            txn.create_membership(f"{sid}-M{i}", sid, f"{sid}-T{i}", "角色", i)
    one = entry_prompt_tokens(repo.store, "S-甲")
    block, _ = render_kb_entry_excerpt(
        [EntrySelection(category="common", key="甲", score=2.0, exists=True),
         EntrySelection(category="common", key="乙", score=1.0, exists=True)],
        tmp_path, count_tokens=len, block_limit=int(one * 2.5),
    )
    assert block.included == ("common/甲",) and block.truncated == ("common/乙",)
    assert block.text.count("至少删减") == 1


def test_the_update_style_keeps_its_whole_share_next_to_an_entry_to_compress(tmp_path) -> None:
    _grow(tmp_path, "乙", 160)
    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("ST", "subject", {"surface": "某组", "intro": "一套风格", "category": "style",
                                          "section_order": ["档案", "约定"]})
        for i in range(20):
            txn.create_node(f"ST-{i}", "note", {"text": "约定" * 40, "label": f"规则{i}"})
            txn.create_membership(f"ST-M{i}", "ST", f"ST-{i}", "约定", i)
    block, _ = render_kb_entry_excerpt(
        [EntrySelection(category="style", key="某组", score=float("inf"), exists=True, pinned=True),
         EntrySelection(category="common", key="乙", score=1.0, exists=True)],
        tmp_path, count_tokens=len, block_limit=12_000,
    )
    assert "style/某组" in block.included


def test_the_conflict_repair_round_is_not_asked_to_compress(tmp_path) -> None:
    """End to end (acceptance review S1): the conflict repair's messages carry
    the oversized entry whole with neither the notice nor the split contract,
    and its agent extras switch the notice off for `kb_read` as well."""

    from finesub.llm.knowledge.update import repair_knowledge_proposals
    from finesub.llm.token_budget import HeuristicTokenCounter

    _grow(tmp_path, "乙", 160)
    seen = {}

    class _Client:
        def complete(self, _role, messages, **kwargs):  # type: ignore[no-untyped-def]
            seen["messages"], seen["extras"] = messages, kwargs.get("agent_task_extras") or {}
            raise RuntimeError("stop after the call is built")

    report = {"conflicts": [{}], "skipped": [
        {"category": "common", "entry": "乙", "op": "append_lines", "reason": "conflict: rev moved"}
    ]}
    result = repair_knowledge_proposals(
        report, knowledge_root=tmp_path, llm_client=_Client(), counter=HeuristicTokenCounter(),
        task_summary="t", read_rev=0, task_id="t", apply_task_id="a", source="s",
        difficulty="quality", sampling={}, artifact_path=None,
    )
    assert result and result.get("attempted")
    text = "\n".join(str(m["content"]) for m in seen["messages"])
    assert "第159行" in text  # served whole
    # The shared rule 8 names `split_entry` conditionally ("when the notice
    # gives the split criterion"); what must be absent is the notice and the
    # op's own contract row.
    for needle in ("至少删减", '{"op":"split_entry"'):
        at = text.find(needle)
        assert at < 0, text[max(0, at - 300):at + 100]
    assert seen["extras"]["kb_tools"] == "propose" and seen["extras"]["kb_size_notice"] is False


def test_a_writer_told_not_to_compress_reads_without_the_notice(tmp_path, monkeypatch) -> None:
    from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer
    from finesub.llm.client import _agent_task_inputs
    from finesub.llm.routing.config import CapabilityTier

    _grow(tmp_path, "乙", 160)

    class _Ledger:
        def record_kb_read(self, **kwargs):  # type: ignore[no-untyped-def]
            pass

    def reply(extras):  # type: ignore[no-untyped-def]
        metadata = _agent_task_inputs(
            [{"role": "user", "content": "u"}], validator_spec=None, variant="",
            capability_tier=CapabilityTier.CAPABLE, max_repair_attempts=0, retrieval="none",
            call_kwargs={"kb_tools": "propose", "knowledge_root": str(tmp_path),
                         "knowledge_identity": "rev:1", **extras},
        )["metadata"]
        monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(tmp_path))
        server = HarnessToolServer(
            _Ledger(), assignment_id="a", worker_id="w", session_id="s", instance_id="i",
            tools=list(KB_TOOL_NAMES),
        )
        server._task = {"task_id": "t", "lease_generation": 1}
        server._manifest = {"metadata": metadata}
        out = server._call("kb_read", {"entry": "乙"}, rpc_id=1)
        assert out["isError"] is False, out["content"][0]["text"]
        return json.loads(out["content"][0]["text"])["text"]

    assert "至少删减" in reply({})  # the knowledge update still asks
    assert "至少删减" not in reply({"kb_size_notice": False})


def test_every_proposal_repair_call_is_recorded_for_the_token_books(tmp_path) -> None:
    """Each call is booked, the one whose answer did not read included; a raise
    writes the call error with what it spent."""

    from finesub import jsonl
    from finesub.llm.client import LLMCallResult
    from finesub.llm.knowledge.base import TASK_ARTIFACT_FILENAME
    from finesub.llm.knowledge.update import repair_knowledge_proposals
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.token_budget import HeuristicTokenCounter

    _grow(tmp_path, "乙", 4)
    error = RuntimeError("down")
    error._harness_execution_attempts = [{"backend": "local_agent", "usage": {"input_tokens": 9}}]

    class _Client:
        calls = 0

        def complete(self, _role, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls > 1:
                raise error
            return LLMCallResult(content="not jsonl", role=LLMRole.GENERAL_CAPABLE, model="m",
                                 fallback_used=False, raw_response={}, execution_attempts=[{"backend": "x"}])

    art = tmp_path / "art"
    art.mkdir()
    report = {"conflicts": [{}], "skipped": [
        {"category": "common", "entry": "乙", "op": "append_lines", "reason": "conflict: rev moved"}
    ]}
    repair_knowledge_proposals(
        report, knowledge_root=tmp_path, llm_client=_Client(), counter=HeuristicTokenCounter(),
        task_summary="t", read_rev=0, task_id="t", apply_task_id="a", source="s",
        difficulty="quality", sampling={}, artifact_path=art,
    )
    kinds = {r["kind"]: r["payload"] for r in jsonl.iter_records(art / TASK_ARTIFACT_FILENAME)}
    assert kinds["knowledge_proposal_repair_response"]["execution_attempts"] == [{"backend": "x"}]
    assert kinds["knowledge_proposal_repair_call_error"]["execution_attempts"] == error._harness_execution_attempts
    assert kinds["knowledge_proposal_repair_call_error"]["call"].endswith("-attempt1")
