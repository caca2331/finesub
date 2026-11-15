"""The preparation task as one logical conversation (plan §2.1 / §8, stage B.1).

Pinned here: the request channel follows the grant; a delivery in the first
reply ends the task with one call; a request batch is filled exactly once and
handed back as the next turn of the SAME conversation (assistant turn replayed
verbatim); the fill turn must deliver; the payload is the research-context
shape; resume replays both turns without a call.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.client import LLMCallResult
from finesub.llm.prepare_prompts import (
    build_prepare_results_message,
    build_prepare_task_messages,
)
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.session_contract import SESSION_CONTRACTS
from finesub.llm.stages import task_graph
from finesub.llm.stages.prepare_task import parse_prepare_reply, run_prepare_task
from finesub.llm.web_search import QueryExtractResult, QuerySearchResult, SearchResultItem

TRANSCRIPT = "--- window 0001 ---\n1|こんにちは\n2|ゲームBを始めます\n"
PACK = {
    "general_context": {"global_summary": "游戏B 实况", "sources": []},
    "window_contexts": [{"window_ids": ["0001"], "context": "开场"}],
}
DELIVERY = (
    "<reasoning>看完了</reasoning>\n<context_pack>\n"
    + json.dumps(PACK, ensure_ascii=False)
    + "\n</context_pack>\n<keep_entries>\n主播A\n</keep_entries>"
)
DELIVERY_WITH_FEEDBACK = (
    DELIVERY + "\n<task_update_feedback>\n主播A 的本名要更新\n</task_update_feedback>"
)
REQUEST = (
    "<reasoning>先查</reasoning>\n<analysis_notes>游戏B 待定</analysis_notes>\n"
    "<requested_entries>\n主播A\n</requested_entries>\n<keep_entries></keep_entries>\n"
    "<search_queries>\n游戏B 剧情\n</search_queries>"
)


class RecordingClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []
        self.raw_response = {"usageMetadata": {"candidatesTokenCount": 10}}

    def complete(self, role, messages, **kwargs):
        self.calls.append((role, [dict(m) for m in messages], kwargs))
        return LLMCallResult(
            content=self.responses.pop(0) if self.responses else "",
            role=role,
            model="fake",
            fallback_used=False,
            raw_response=self.raw_response,
        )


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_many(self, requests, *, max_queries=None):
        queries = [getattr(item, "query", item) for item in requests]
        self.calls.append((tuple(queries), max_queries))
        return [
            QuerySearchResult(
                query=query,
                provider="fake",
                items=(SearchResultItem(title="wiki", url="https://example.test", snippet="恐怖游戏"),),
            )
            for query in queries
        ]

    def extract_many(self, requests, *, max_urls=None):
        return []


def _knowledge_root(tmp_path: Path, *, with_index: bool) -> Path:
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    if with_index:
        (root / "streamer" / "index.md").write_text("- 主播A | エーちゃん | 测试主播\n", encoding="utf-8")
        (root / "streamer" / "主播A.md").write_text(
            "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
            encoding="utf-8",
        )
        (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


# --- prompt shape follows the grant ----------------------------------------


def test_request_channel_follows_the_grant() -> None:
    both = build_prepare_task_messages(
        transcript=TRANSCRIPT, emits_queries=True, emits_entries=True,
        knowledge_indexes={"streamer": "- 主播A | | x", "common": ""},
    )
    system = both[0]["content"]
    assert "获取缺失输入" in system
    assert "<search_queries>" in system and "<requested_entries>" in system
    assert "<context_pack>" in system

    kb_only = build_prepare_task_messages(
        transcript=TRANSCRIPT, emits_queries=False, emits_entries=True,
        knowledge_indexes={"streamer": "- 主播A | | x", "common": ""},
    )[0]["content"]
    assert "<requested_entries>" in kb_only and "<search_queries>" not in kb_only

    native = build_prepare_task_messages(
        transcript=TRANSCRIPT, emits_queries=False, emits_entries=False, native_search=True
    )[0]["content"]
    assert "联网检索（你自己执行）" in native
    assert "获取缺失输入" not in native

    nothing = build_prepare_task_messages(
        transcript=TRANSCRIPT, emits_queries=False, emits_entries=False
    )
    system, user = nothing[0]["content"], nothing[1]["content"]
    assert "获取缺失输入" not in system
    assert "`sources` 留空" in system
    assert "<transcript>" in user and "1|こんにちは" in user
    assert "知识库索引" not in user


def test_the_url_extracts_block_and_its_name_come_only_with_extracts() -> None:
    """Cold read P056: no extracts, no `<note_url_extracts>` block and no
    mention of it in the input list; with extracts, both."""

    with_extracts = build_prepare_task_messages(transcript=TRANSCRIPT, note_url_extracts="页面正文")
    without = build_prepare_task_messages(transcript=TRANSCRIPT)
    assert "<note_url_extracts>" in with_extracts[1]["content"] and "页面正文" in with_extracts[1]["content"]
    assert "用户备注中 URL 的页面提取内容" in with_extracts[0]["content"]
    assert "<note_url_extracts>" not in without[1]["content"]
    assert "页面提取内容" not in without[0]["content"]


def test_results_message_names_empty_channels_explicitly() -> None:
    message = build_prepare_results_message(
        requested_entries=True, requested_search=True, entry_details_text="", search_results="", knowledge=True
    )
    assert message["role"] == "user"
    assert "均不存在" in message["content"]
    assert "没有返回任何结果" in message["content"]
    assert "本轮不能再提出任何请求" in message["content"]


# --- reply parsing ----------------------------------------------------------


def test_parse_delivery_wins_and_requests_need_an_open_channel() -> None:
    delivered = parse_prepare_reply(DELIVERY, allow_requests=True, expect_queries=True, expect_entries=True)
    assert delivered.delivered and delivered.keep_entries_raw == ("主播A",)
    requested = parse_prepare_reply(REQUEST, allow_requests=True, expect_queries=True, expect_entries=True)
    assert not requested.delivered
    assert requested.requests.search_queries == ("游戏B 剧情",)
    with pytest.raises(ValueError):
        parse_prepare_reply(REQUEST, allow_requests=False, expect_queries=True, expect_entries=True)
    with pytest.raises(ValueError):
        parse_prepare_reply("<reasoning>x</reasoning>", allow_requests=True, expect_queries=False, expect_entries=False)


def test_session_contracts_cover_both_reply_shapes() -> None:
    assert not SESSION_CONTRACTS["prepare_task_delivery"].validate(DELIVERY)
    assert not SESSION_CONTRACTS["prepare_task_request"].validate(REQUEST)
    assert SESSION_CONTRACTS["prepare_task_delivery"].validate(REQUEST)


# --- the conversation -------------------------------------------------------


def test_direct_delivery_is_one_call(tmp_path) -> None:
    client = RecordingClient([DELIVERY])
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )
    assert len(client.calls) == 1
    assert payload["context_pack"]["general_context"]["global_summary"] == "游戏B 实况"
    assert payload["rounds"]["task_flow"] == "batched"
    assert payload["rounds"]["tool_round"] is False
    assert payload["round1"]["search_queries"] == []


def test_delivery_collects_task_feedback_only_when_asked(tmp_path) -> None:
    """`--knowledge update` (or a caller asking for feedback): the delivery is asked for a
    <task_update_feedback> block, the harness keeps its raw body in the payload
    and books it as `research_task_feedback`; without the switch the prompt
    never mentions the block."""
    feedback = json.dumps(
        {"knowledge_hints": [], "asr_corrections": [], "uncertainties": ["游戏B 待定"]},
        ensure_ascii=False,
    )
    client = RecordingClient(
        [DELIVERY + "\n<task_update_feedback>\n" + feedback + "\n</task_update_feedback>"]
    )
    art = tmp_path / "art"
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=art,
        task_id="t",
        max_parse_retries=0,
        collect_task_feedback=True,
    )
    system = client.calls[0][1][0]["content"]
    assert "<task_update_feedback>" in system
    assert "游戏B 待定" in payload["task_update_feedback"]
    kinds = [
        json.loads(line)["kind"]
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "research_task_feedback" in kinds
    quiet = build_prepare_task_messages(transcript=TRANSCRIPT)[0]["content"]
    assert "task_update_feedback" not in quiet


def test_request_batch_is_filled_once_in_the_same_conversation(tmp_path) -> None:
    client = RecordingClient([REQUEST, DELIVERY])
    search = FakeSearchClient()
    art = tmp_path / "art"
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=search,
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=art,
        task_id="t",
        max_parse_retries=0,
    )
    assert len(client.calls) == 2
    first, second = client.calls[0][1], client.calls[1][1]
    # The fill turn continues the first turn: same opening, the model's own
    # request replayed as its assistant message, then the results as user.
    assert second[:2] == first[:2]
    assert second[2] == {"role": "assistant", "content": REQUEST}
    # t1 announces its fill turn (Anthropic writes its end for t2 to read); t2 is the last
    assert client.calls[0][2]["followup_expected"] is True
    assert client.calls[1][2]["followup_expected"] is False
    assert second[3]["role"] == "user"
    assert "<knowledge_entries>" in second[3]["content"] and "主播资料" in second[3]["content"]
    assert "<search_results>" in second[3]["content"] and "恐怖游戏" in second[3]["content"]
    assert search.calls == [(("游戏B 剧情",), 8)] or search.calls[0][0] == ("游戏B 剧情",)
    assert payload["rounds"]["tool_round"] is True
    assert payload["round1"]["requested_entries"] == ["主播A"]
    assert payload["injected_entries"] == ["主播A"]
    assert payload["keep_entries"] == ["主播A"]
    assert payload["context_pack"]["window_contexts"][0]["window_id"] == "0001"
    kinds = [json.loads(line)["kind"] for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kinds.count("prepare_task_request") == 1
    assert kinds.count("prepare_task_response") == 1
    assert "research_search_results" in kinds


def test_the_fill_turn_must_deliver(tmp_path) -> None:
    """A second request is a parse failure under the retry budget, never a second fill."""

    client = RecordingClient([REQUEST, REQUEST, REQUEST])
    with pytest.raises(RuntimeError):
        run_prepare_task(
            transcript=TRANSCRIPT,
            knowledge_root=_knowledge_root(tmp_path, with_index=True),
            client=client,
            search_client=FakeSearchClient(),
            profile=resolve_profile("text", "local", "quality"),
            task_artifact_dir=tmp_path / "art",
            task_id="t",
            max_parse_retries=1,
        )
    assert len(client.calls) == 3  # one request turn + two delivery attempts


def test_no_channel_means_no_requests_and_material_only_delivery(tmp_path) -> None:
    client = RecordingClient([DELIVERY])
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        client=client,
        profile=resolve_profile("text", "none", "quality"),
        knowledge_enabled=False,
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )
    system = client.calls[0][1][0]["content"]
    assert "获取缺失输入" not in system
    assert payload["rounds"]["entries_available"] is False
    assert payload["keep_entries"] == []


def test_resume_replays_both_turns_without_a_call(tmp_path) -> None:
    root = _knowledge_root(tmp_path, with_index=True)
    art = tmp_path / "art"
    kwargs = dict(
        transcript=TRANSCRIPT,
        knowledge_root=root,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=art,
        task_id="t",
        max_parse_retries=0,
    )
    first = RecordingClient([REQUEST, DELIVERY])
    run_prepare_task(client=first, **kwargs)
    again = RecordingClient([])
    payload = run_prepare_task(client=again, **kwargs)
    assert again.calls == []
    assert payload["context_pack"]["general_context"]["global_summary"] == "游戏B 实况"


# --- agentic strategy (stage C): one reply, the executor's own tools ----------------


def test_agentic_prepare_names_tools_and_must_deliver(tmp_path, monkeypatch) -> None:
    from finesub.llm.stages import prepare_task as module

    monkeypatch.setattr(
        module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    client = RecordingClient([DELIVERY])
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )
    assert len(client.calls) == 1
    system = client.calls[0][1][0]["content"]
    assert "由你自行取得" in system and "`web_search`" in system and "`kb_search`" in system
    assert "<search_queries>" not in system and "<requested_entries>" not in system
    # Cold read Q2/P011: the index is the required `kb_index` block, not inlined
    # as well; the preinjected entries stay in the payload.
    user = client.calls[0][1][1]["content"]
    assert "知识库索引：" not in user and "<preinjected_entries>" in user
    assert "必读块 `kb_index`" in system and "`kb_index` 读" not in system
    assert payload["rounds"]["task_flow"] == "agentic"
    assert payload["rounds"]["tool_round"] is False
    # The keep chain survives: the executor read `主播A` through its own
    # `kb_read`, so "what it could have seen" is the base, not a fill that
    # never ran. Filtering against the (empty) fill dropped every agentic keep
    # and the windows started with no transfer seed at all.
    assert payload["keep_entries"] == ["主播A"]

    # A request batch is not an answer here: the channel is closed.
    again = RecordingClient([REQUEST])
    with pytest.raises(RuntimeError):
        run_prepare_task(
            transcript=TRANSCRIPT,
            knowledge_root=_knowledge_root(tmp_path / "b", with_index=True),
            client=again,
            profile=resolve_profile("text", "local", "quality"),
            task_artifact_dir=tmp_path / "art-b",
            task_id="t",
            max_parse_retries=0,
        )


def test_agentic_keeps_the_chain_alive_on_every_retrieval_vector(tmp_path, monkeypatch) -> None:
    """`<keep_entries>` is how a task says what the windows still need.

    Batched asks for it wherever a window can ask an entry back (`local`, or a
    readable base on any vector); without that the harness just transfers the
    whole injected selection instead. Agentic has nothing injected to transfer
    -- the executor read the base itself -- so unless it names what it read,
    `none`/`native` windows start with an empty transfer chain and no way to
    fill it. The prompt also has to point at the right source: there is no
    `<knowledge_entries>` block to pick from.
    """

    from finesub.llm.stages import prepare_task as module

    monkeypatch.setattr(
        module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    for retrieval in ("none", "native", "local"):
        client = RecordingClient([DELIVERY])
        payload = run_prepare_task(
            transcript=TRANSCRIPT,
            knowledge_root=_knowledge_root(tmp_path / retrieval, with_index=True),
            client=client,
            search_client=FakeSearchClient(),
            profile=resolve_profile("text", retrieval, "quality"),
            task_artifact_dir=tmp_path / f"art-{retrieval}",
            task_id="t",
            max_parse_retries=0,
        )
        system = client.calls[0][1][0]["content"]
        assert "<keep_entries>" in system, retrieval
        assert "`kb_read` 读过的" in system, retrieval
        assert "`<knowledge_entries>` 中实际出现" not in system, retrieval
        assert payload["keep_entries"] == ["主播A"], retrieval


def test_knowledge_off_never_reads_the_base_even_if_asked(tmp_path, monkeypatch) -> None:
    """Plan §8: "KB 关闭不读普通索引／正文". The prompt describes no entry blocks
    without a readable index, so a reply that carries them anyway is
    unsolicited -- and serving it would read entry bodies out of a base this
    run was told not to touch. The agentic keep fix widened that door: it drops
    the visibility filter, so an unsolicited `<keep_entries>` went straight to
    the store and on into the windows."""

    from finesub.llm.stages import prepare_task as module

    monkeypatch.setattr(
        module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    root = _knowledge_root(tmp_path, with_index=True)  # a real base, just not granted
    reads: list = []
    real = module.load_entry_texts
    monkeypatch.setattr(
        module, "load_entry_texts",
        lambda *a, **k: (reads.append(a) or real(*a, **k)),
    )

    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=root,
        knowledge_enabled=False,
        client=RecordingClient([DELIVERY]),  # DELIVERY keeps `主播A`
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )

    assert payload["keep_entries"] == []
    assert reads == []


def test_the_preparation_call_lands_on_the_cell_its_strategy_came_from(
    tmp_path, monkeypatch
) -> None:
    """`chain_serves_tool_session` derives the strategy from
    `role_config_for("research", profile.difficulty)`. The call has to reach
    the same cell, or a preset binding an agent at intermediate and an API at
    quality gets the "fetch it yourself" prompt on a tool-less API with the
    harness already stood down. difficulty is also what picks that cell's
    prompt variant and thinking level."""

    seen: list = []

    class Client(RecordingClient):
        def complete(self, role, messages, **kwargs):
            seen.append(kwargs.get("difficulty"))
            return super().complete(role, messages, **kwargs)

    run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        knowledge_enabled=False,
        client=Client([DELIVERY]),
        profile=resolve_profile("text", "none", "intermediate"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )

    assert seen == ["intermediate"]


def test_the_delivering_calls_urls_reach_the_knowledge_update(tmp_path, monkeypatch) -> None:
    """`research_task_feedback.retrieval_urls` is harness-side call evidence:
    what the answering model actually retrieved. The window path never stopped
    filling it; the preparation task hardcoded `[]`, so every agent search URL
    behind a piece of feedback was dropped before the update could read it."""

    # The shape `retrieval_urls_from_response` reads: the agent transport's
    # own normalized event rows, which the model cannot edit.
    raw = {
        "candidates": [{"finishReason": "STOP"}],
        "agent": {"events": [{"tool": "web_search", "urls": ["https://example.test/a"]}]},
    }
    client = RecordingClient([DELIVERY_WITH_FEEDBACK])
    client.raw_response = raw
    run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        knowledge_enabled=False,
        client=client,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
        collect_task_feedback=True,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "art" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    feedback = next(r["payload"] for r in records if r["kind"] == "research_task_feedback")
    assert feedback["retrieval_urls"] == ["https://example.test/a"]


# --- the opt-in ---------------------------------------------------------------


def test_strategy_pin_env(monkeypatch) -> None:
    from finesub.llm.routing.profiles import resolve_profile

    monkeypatch.delenv(task_graph.TASK_STRATEGY_ENV, raising=False)
    assert task_graph.task_flow_strategy_override() is None
    monkeypatch.setenv(task_graph.TASK_STRATEGY_ENV, "batched")
    assert task_graph.task_flow_strategy_override() == "batched"
    assert task_graph.resolve_execution_strategy(
        resolve_profile("text", "local", "quality")
    ) == ("batched", "env_pin")
    monkeypatch.setenv(task_graph.TASK_STRATEGY_ENV, "serial")
    with pytest.raises(ValueError):
        task_graph.task_flow_strategy_override()


def test_research_stage_runs_the_preparation_task(tmp_path, monkeypatch) -> None:
    from finesub.llm import research as research_module

    seen: dict = {}

    def fake_prepare(**kwargs):
        seen.update(kwargs)
        return {"context_pack": PACK, "keep_entries": [], "rounds": {"task_flow": "batched"}}

    monkeypatch.setattr("finesub.llm.stages.prepare_task.run_prepare_task", fake_prepare)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "こんにちは"}]}),
        encoding="utf-8",
    )
    context_path = tmp_path / "art" / "clip-research-context.json"
    pack = research_module.run_research_stage(
        stable_json=stable,
        context_path=context_path,
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        knowledge_enabled=False,
        test_profile=True,
        task_artifact_dir=tmp_path / "art",
        profile=resolve_profile("text", "none", "quality"),
    )
    assert seen["transcript"].startswith("--- window")
    assert pack.general_context["global_summary"] == "游戏B 实况"
    saved = json.loads(context_path.read_text(encoding="utf-8"))
    assert saved["rounds"]["task_flow"] == "batched"


# --- what the opening message carries --------------------------------------


class _ExtractClient:
    """Deep-fetches whatever URLs the harness asks for; never searches."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def search_many(self, requests, *, max_queries=None):
        return []

    def extract_many(self, requests, *, max_urls=None):
        self.calls.append([req.url for req in requests])
        return [
            QueryExtractResult(
                url=req.url,
                provider="exa",
                title="page",
                content=("TOXIC_PAYLOAD" if "bad" in req.url else "official synopsis text"),
            )
            for req in requests
        ]


def test_note_urls_are_prefetched_into_the_opening_message(tmp_path) -> None:
    client = RecordingClient([DELIVERY])
    extracts = _ExtractClient()
    run_prepare_task(
        transcript=TRANSCRIPT,
        extra_info="素材来源: https://www.youtube.com/watch?v=abc 的切片",
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        client=client,
        search_client=extracts,
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )
    user = client.calls[0][1][1]["content"]
    assert "official synopsis text" in user
    assert "<note_url_extracts>" in user
    assert extracts.calls == [["https://www.youtube.com/watch?v=abc"]]


def test_content_filter_ladder_drops_a_toxic_note_extract(tmp_path, monkeypatch) -> None:
    """Leave-one-out over the opening's URL extracts identifies and blacklists the toxic one."""

    blocked_raw = {
        "choices": [{"finish_reason": "content_filter", "message": {}}],
        "usageMetadata": {"candidatesTokenCount": 0},
    }
    ok_raw = {"usageMetadata": {"candidatesTokenCount": 10}}
    replies = [("", blocked_raw), ("", blocked_raw), (DELIVERY, ok_raw)]
    seen: list[str] = []

    class Client:
        def complete(self, role, messages, **kwargs):
            seen.append(messages[1]["content"])
            content, raw = replies.pop(0)
            return LLMCallResult(content=content or None, role=role, model="fake", fallback_used=False, raw_response=raw)

    monkeypatch.setattr("finesub.llm.content_filter.time.sleep", lambda _s: None)
    art = tmp_path / "art"
    run_prepare_task(
        transcript=TRANSCRIPT,
        extra_info="链接: https://example.test/good 与 https://example.test/bad-page",
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        client=Client(),
        search_client=_ExtractClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=art,
        task_id="t",
        max_parse_retries=0,
    )
    assert any("TOXIC_PAYLOAD" in text for text in seen)
    assert any("official synopsis text" in text and "TOXIC_PAYLOAD" not in text for text in seen)
    from finesub.llm.content_filter import BLACKLIST_ARTIFACT_KIND, LADDER_ARTIFACT_KIND

    kinds = {json.loads(line)["kind"] for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    assert LADDER_ARTIFACT_KIND in kinds and BLACKLIST_ARTIFACT_KIND in kinds


def test_note_matched_entries_are_preinjected_and_keepable(tmp_path) -> None:
    """An entry the user's note names is shown up front; a delivery may keep it
    even when the task never requested anything (no fill ran)."""

    client = RecordingClient([DELIVERY])
    art = tmp_path / "art"
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        extra_info="今天是エーちゃん的杂谈回",
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=art,
        task_id="t",
        max_parse_retries=0,
    )
    user = client.calls[0][1][1]["content"]
    assert "<preinjected_entries>" in user and "# 主播A" in user and "主播资料" in user
    records = [json.loads(line) for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    report = next(r for r in records if r["kind"] == "knowledge_preinjection")["payload"]
    assert report["source"] == "prepare_task"
    assert report["matches"][0]["key"] == "主播A"
    assert report["included"] == ["主播A"]
    assert len(client.calls) == 1
    assert payload["keep_entries"] == ["主播A"]


@pytest.mark.parametrize("retrieval", ["none", "native"])
def test_batched_none_native_prepare_prunes_like_local(tmp_path, monkeypatch, retrieval) -> None:
    """window-kb-request-channel plan, acceptance 6: with a readable base the
    windows can ask entries back, so the preparation task requests at the
    per-window cap (8, not 12), is asked for `<keep_entries>`, and what it does
    not keep is dropped instead of being passed through whole."""

    from finesub.llm.stages import prepare_task as module

    monkeypatch.setattr(module, "resolve_execution_strategy", lambda profile, **kw: ("batched", "t"))
    monkeypatch.setattr(task_graph, "resolve_execution_strategy", lambda profile, **kw: ("batched", "t"))
    request = (
        "<reasoning>先查</reasoning>\n<analysis_notes></analysis_notes>\n"
        "<requested_entries>\n主播A\n</requested_entries>\n<keep_entries></keep_entries>"
    )
    delivery_keeping_nothing = (
        "<reasoning>看完了</reasoning>\n<context_pack>\n"
        + json.dumps(PACK, ensure_ascii=False)
        + "\n</context_pack>\n<keep_entries>\n</keep_entries>"
    )
    client = RecordingClient([request, delivery_keeping_nothing])
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        profile=resolve_profile("text", retrieval, "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
    )
    system = client.calls[0][1][0]["content"]
    assert "单独上限 8 条" in system and "<keep_entries>" in system
    assert payload["injected_entries"] == ["主播A"]
    assert payload["keep_entries"] == []


def test_batched_keep_names_the_preinjected_entries_too() -> None:
    """Cold read P007: the delivery check accepts the fill's entries plus the
    preinjected ones (`visible_keys`); the prompt says the same."""

    from finesub.llm.prepare_prompts import build_prepare_task_messages

    system = build_prepare_task_messages(
        transcript="[window 0001]\n1|一", knowledge_indexes={"streamer": "- 主播A | | 测试主播"}
    )[0]["content"]
    assert "`<preinjected_entries>` 或 `<knowledge_entries>` 中出现的词条 key" in system
    assert "这两处之外的 key 会被忽略" in system


def test_the_worker_shell_says_how_to_fetch_a_tool_block() -> None:
    """The manifest marks the `kb_index` block `read: tool`; the shell names
    that mode next to `read_context` and `read: seen` (cold read P011)."""

    from finesub.llm.prompts import agent_tool_worker_bootstrap
    import inspect

    params = inspect.signature(agent_tool_worker_bootstrap).parameters
    kwargs = {name: "x" for name, p in params.items() if p.default is inspect.Parameter.empty}
    assert "`read: tool`" in agent_tool_worker_bootstrap(**kwargs)
