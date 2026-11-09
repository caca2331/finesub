"""Correction window as one task with an in-conversation request turn (stage B.2/B.3).

Pinned: the prompt shape follows the grant; a request batch is filled once
and the delivery continues the SAME conversation (assistant turn replayed,
results as the next user turn, media on that last user turn); a direct
delivery is validated as attempt 0 without a second call; fast + local under
the task flow is one whole-input window task with no fused round and no
separate preparation.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

from finesub.llm import correction_translation as orchestration
from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.exchange_metadata import is_session_response_record
from finesub.llm.prompts import ContextPack, build_correction_csv_messages
from finesub.llm.request_prompts import build_window_task_messages
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.web_search import QuerySearchResult, SearchResultItem
from test.conftest import setattr_correction

STABLE = {
    "segments": [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
        {"id": "2", "start": 1.5, "end": 2.5, "text": "二。"},
    ]
}
DELIVERY = (
    "<translated>\ntype|position|corrected_text|translation|conf|note\n"
    "sub|1|一。|一|8|\n"
    "sub|2|二。|二|7|术语note\n"
    "</translated>\n<next_advice></next_advice>"
)
#: A delivery that also keeps an entry. Nothing asked for the block under
#: `--knowledge none`; the model volunteering it is the case that matters.
DELIVERY_KEEPING = DELIVERY + "\n<keep_entries>\n主播A\n</keep_entries>"
REQUEST = (
    "<reasoning>先查</reasoning>\n<window_notes>主播A 待定</window_notes>\n"
    "<requested_entries>\n主播A\n</requested_entries>\n<keep_entries></keep_entries>\n"
    "<search_queries>\n游戏B 剧情\n</search_queries>"
)


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def search_many(self, requests, *, max_queries=None):
        queries = [getattr(item, "query", item) for item in requests]
        self.calls.append(tuple(queries))
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


def _knowledge_root(tmp_path: Path) -> Path:
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text("- 主播A | エーちゃん | 测试主播\n", encoding="utf-8")
    (root / "streamer" / "主播A.md").write_text(
        "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


def _window():
    return dataclasses.replace(plan_correction_windows(
        [SubtitleSegment("1", 0.0, 1.0, "一。"), SubtitleSegment("2", 1.5, 2.5, "二。")],
        counter=FakeTokenCounter(),
    )[0], length_scale=1.0)


def _kwargs(**overrides):
    base = dict(
        window=_window(),
        context_pack=ContextPack(),
        previous_advice="",
        window_notes="",
        search_results="",
        entry_details="",
        profile=resolve_profile("text", "local", "quality"),
        knowledge_enabled=True,
    )
    base.update(overrides)
    return base


# --- prompt shapes ------------------------------------------------------------


def test_without_the_channel_the_prompt_is_byte_identical() -> None:
    assert build_window_task_messages(**_kwargs()) == build_correction_csv_messages(**_kwargs())


def test_request_channel_opens_the_task_without_results_yet() -> None:
    messages = build_window_task_messages(
        request_channel=True,
        knowledge_indexes={"streamer": "- 主播A | | 测试主播", "common": ""},
        carried_entries="# 主播B\n资料",
        **_kwargs(search_results="should not leak", entry_details="union"),
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "获取缺失输入" in system
    assert "<search_queries>" in system and "<requested_entries>" in system
    assert "知识库索引" in system and "主播A" in system
    assert "<search_results>" not in user
    assert "should not leak" not in user
    assert "# 主播B" in user and "union" not in user


def test_no_index_means_no_entry_requests() -> None:
    system = build_window_task_messages(request_channel=True, **_kwargs(knowledge_enabled=False))[0]["content"]
    assert "<search_queries>" in system and "<requested_entries>" not in system


def test_fill_turn_continues_the_same_conversation() -> None:
    opening = build_window_task_messages(request_channel=True, **_kwargs())
    messages = build_window_task_messages(
        request_channel=True,
        request_reply=REQUEST,
        requested_entries=True,
        requested_search=True,
        **_kwargs(search_results="--- query: 游戏B 剧情 ---\n恐怖游戏", entry_details="# 主播A\n主播资料。"),
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[:2] == opening
    assert messages[2]["content"] == REQUEST
    fill = messages[3]["content"]
    assert "<search_results>" in fill and "恐怖游戏" in fill
    assert "<entry_details>" in fill and "主播资料" in fill
    assert "本轮不能再提出任何请求" in fill
    assert "<next_advice>" in fill  # serial trailing blocks are still owed


@pytest.mark.parametrize("variant", ["capableC", "basicA"])
def test_fill_turn_orders_blocks_by_the_variant(variant) -> None:
    """The fill turn asks for `<translated>` after `<reasoning>`, and no set is told to write
    `<singles>` (cold read P001; basicA dropped its own, owner 2026-10-31)."""

    fill = build_window_task_messages(
        request_channel=True, request_reply=REQUEST, requested_search=True,
        **_kwargs(search_results="恐怖游戏", variant=variant),
    )[3]["content"]
    assert "<singles>" not in fill
    assert "<translated>" in fill


def test_the_keep_block_names_the_block_its_entries_are_in() -> None:
    """Cold read P009: a window sees its carried entries in `<entry_details>`;
    the preparation task's request batch keeps naming `<preinjected_entries>`."""

    from finesub.llm.prepare_prompts import build_prepare_task_messages

    system = build_window_task_messages(
        request_channel=True, knowledge_indexes={"streamer": "- 主播A | | 测试主播"}, **_kwargs()
    )[0]["content"]
    keep = system.split("<keep_entries>\n", 1)[1].split("\n", 1)[0]
    assert "`<entry_details>`" in keep and "preinjected" not in keep
    prepare = build_prepare_task_messages(
        transcript="[window 0001]\n1|一", knowledge_indexes={"streamer": "- 主播A | | 测试主播"}
    )[0]["content"]
    assert "每行一个 `<preinjected_entries>` 中实际可见、要留到交付时的词条" in prepare


def test_the_request_section_names_no_block_outside_the_contract() -> None:
    system = build_window_task_messages(request_channel=True, **_kwargs(variant="capableC"))[0]["content"]
    assert "<singles>" not in system


def test_fill_turn_keeps_the_style_block_next_to_requested_entries() -> None:
    """`--style auto` on the fast window: the picked rules ride in the fill
    turn -- also when the same batch requested entries (the two inputs used
    to overwrite each other)."""
    from finesub.llm.request_prompts import build_correction_fill_message

    fill = build_correction_fill_message(
        entry_details="# 主播A\n主播资料。",
        requested_entries=True,
        requested_search=True,
        search_results="恐怖游戏",
        style_block="<style name=\"S\">风格约定</style>",
    )["content"]
    assert "风格约定" in fill and "主播资料" in fill and "恐怖游戏" in fill
    assert fill.index("风格约定") < fill.index("<entry_details>")


# --- the conversation, end to end through the window loop --------------------


class _Client:
    responses: list[str] = []
    calls: list[tuple] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def complete(self, role, messages, **kwargs):
        if callable(messages):
            messages = messages("capableC")
        type(self).calls.append((role, [dict(m) for m in messages], kwargs))
        return LLMCallResult(
            content=type(self).responses.pop(0),
            role=role,
            model="fake",
            fallback_used=False,
            raw_response={"candidates": [{"finishReason": "STOP"}]},
        )


def _run(
    tmp_path,
    monkeypatch,
    responses,
    *,
    continuity="serial",
    knowledge_enabled=True,
    retrieval="local",
    with_search_client=True,
    stable=None,
    max_window_subtitle_tokens=None,
    initial_transfer_keys=(),
):
    stable_path = tmp_path / "clip-stable.json"
    stable_path.write_text(json.dumps(stable or STABLE), encoding="utf-8")

    class Client(_Client):
        pass

    Client.responses = list(responses)
    Client.calls = []
    setattr_correction(monkeypatch, "RoleClient", Client)
    search = FakeSearchClient()
    art = tmp_path / "art"
    out = execute_correction_windows(
        stable_json=stable_path,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=dataclass_replace(resolve_profile("text", retrieval, "quality"), continuity=continuity),
        knowledge_root=_knowledge_root(tmp_path),
        search_client=search if with_search_client else None,
        task_artifact_dir=art,
        knowledge_enabled=knowledge_enabled,
        max_retries_per_window=1,
        max_window_subtitle_tokens=max_window_subtitle_tokens,
        initial_transfer_keys=initial_transfer_keys,
        resume=False,
        length_scale=1.0,
    )
    kinds = [json.loads(line)["kind"] for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    records = [json.loads(line) for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    return out, Client.calls, search, kinds, records


def test_request_batch_then_delivery_in_one_conversation(tmp_path, monkeypatch) -> None:
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [REQUEST, DELIVERY])
    assert out.exists()
    assert len(calls) == 2
    first, second = calls[0][1], calls[1][1]
    assert [m["role"] for m in first] == ["system", "user"]
    assert "获取缺失输入" in first[0]["content"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[:2] == first[:2]
    assert second[2]["content"] == REQUEST
    # the request turn announces its fill turn; the fill turn is the last
    assert calls[0][2]["followup_expected"] is True
    assert not calls[1][2].get("followup_expected", False)
    assert "恐怖游戏" in second[3]["content"] and "主播资料" in second[3]["content"]
    assert search.calls == [("游戏B 剧情",)]
    request = next(r for r in records if r["kind"] == "correction_query_response")
    assert request["payload"]["task_flow"] == "batched" and request["payload"]["delivered"] is False
    assert request["payload"]["resolved_entry_keys"] == ["主播A"]
    assert "correction_search_results" in kinds
    window = next(r for r in records if r["kind"] == "correction_window_response")
    assert window["payload"]["validation_ok"] is True
    assert "主播A" in window["payload"]["injected_entries"]


def test_direct_delivery_is_validated_as_attempt_zero(tmp_path, monkeypatch) -> None:
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [DELIVERY])
    assert out.exists()
    assert len(calls) == 1
    assert search.calls == []
    request = next(r for r in records if r["kind"] == "correction_query_response")
    assert request["payload"]["delivered"] is True
    window = next(r for r in records if r["kind"] == "correction_window_response")
    assert window["payload"]["attempt"] == 0 and window["payload"]["validation_ok"] is True
    assert "correction_search_results" not in kinds

    # One request, counted once. The record above stays (it is the audit face
    # for "the channel was open and went unused"), but the same `raw_response`
    # must not become a second session, a second token row or a second
    # exchange -- that doubled every direct delivery in the cost report and in
    # the A/B call counts.
    assert is_session_response_record("correction_query_response", request["payload"]) is False
    rows = next(
        r for r in records if r["kind"] == "token_distribution_report"
    )["payload"]["rows"]
    assert [row["call"] for row in rows] == ["correction_window"]
    assert len(list((tmp_path / "art" / "exchanges").glob("*"))) == 1


def test_a_format_retry_is_a_call_that_gets_recorded(tmp_path, monkeypatch) -> None:
    """A reply that is neither a delivery nor a request batch is redone -- but
    it was still a call that was made and paid for. Only the last reply used to
    reach `_record_turn`, so a retried request turn under-reported both its
    request count and its tokens, and its failed reply had no exchange at all."""

    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, ["没有输出标签块", DELIVERY])
    assert out.exists()
    assert len(calls) == 2

    requests = [r["payload"] for r in records if r["kind"] == "correction_query_response"]
    assert [(r["attempt"], r["delivered"], bool(r["parse_error"])) for r in requests] == [
        (0, False, True),
        (1, True, False),
    ]
    assert "correction_query_format_retry" in kinds
    # The failed call gets a token row of its own; the delivering one is
    # counted once, as the window (see the direct-delivery test above).
    rows = next(
        r for r in records if r["kind"] == "token_distribution_report"
    )["payload"]["rows"]
    assert [row["call"] for row in rows] == ["correction_query", "correction_window"]
    assert len(list((tmp_path / "art" / "exchanges").glob("*"))) == 2


def test_repairing_a_direct_delivery_sees_no_fill_turn(tmp_path, monkeypatch) -> None:
    """A direct delivery that fails validation is repaired on the opening
    messages: no replayed "request turn" (it never asked) and no fill turn
    claiming requests were served."""
    incomplete = DELIVERY.replace("sub|2|二。|二|7|术语note\n", "")
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [incomplete, DELIVERY])
    assert out.exists()
    assert len(calls) == 2
    assert [m["role"] for m in calls[1][1]] == ["system", "user"]
    assert calls[1][2]["previous_output"] == incomplete
    assert search.calls == []
    windows = [r for r in records if r["kind"] == "correction_window_response"]
    assert [w["payload"]["validation_ok"] for w in windows] == [False, True]


def test_parallel_driver_takes_the_same_request_turn(tmp_path, monkeypatch) -> None:
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [REQUEST, DELIVERY.replace("<next_advice></next_advice>", "")], continuity="parallel")
    assert out.exists()
    assert len(calls) == 2
    assert [m["role"] for m in calls[1][1]] == ["system", "user", "assistant", "user"]


def test_an_agentic_window_transfers_what_it_read_itself(tmp_path, monkeypatch) -> None:
    """An agentic window has no request turn, so `injected_keys` is whatever
    the preparation task seeded. Filtering its `<keep_entries>` against that
    set stops the transfer chain there, even though the executor read the base
    itself with `kb_read`."""

    from finesub.llm.stages import task_graph as task_graph_module

    monkeypatch.setattr(
        task_graph_module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [DELIVERY_KEEPING])

    assert out.exists()
    window = next(r for r in records if r["kind"] == "correction_window_response")
    assert window["payload"]["keep_entries"] == ["主播A"]


def test_knowledge_off_never_transfers_an_unsolicited_keep(tmp_path, monkeypatch) -> None:
    """The same door the preparation task just closed, one module over.

    With the base off nothing is injected, and `injected_keys` being empty used
    to be what kept this branch shut. `or run.agent_fetches` removed that
    implication: an unsolicited `<keep_entries>` then read the store and the
    body reached the next window -- on a run told not to touch the base
    (plan §8)."""

    from finesub.llm.stages import task_graph as task_graph_module

    monkeypatch.setattr(
        task_graph_module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [DELIVERY_KEEPING], knowledge_enabled=False
    )

    assert out.exists()
    window = next(r for r in records if r["kind"] == "correction_window_response")
    assert window["payload"]["keep_entries"] == []
    assert "主播资料" not in json.dumps(records, ensure_ascii=False)


# --- agentic strategy: the executor fetches for itself (stage C) ----------------


def test_agentic_window_has_no_request_turn_and_names_its_tools(tmp_path, monkeypatch) -> None:
    from finesub.llm.stages import task_graph as task_graph_module

    monkeypatch.setattr(
        task_graph_module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [DELIVERY])
    assert out.exists()
    assert len(calls) == 1
    messages = calls[0][1]
    assert [m["role"] for m in messages] == ["system", "user"]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "由你自行取得" in system and "`web_search`" in system and "`kb_search`" in system
    assert "获取缺失输入（同一任务内" not in system
    # it searches with its own tools: nothing in the system says a harness
    # fills a block or that it cannot go online (N3 for the agentic window)
    assert "联网检索结果使用规则" in system and "与知识库条目冲突" in system
    assert "harness 代为检索" not in system and "你自己不能联网" not in system
    assert "<search_results>" not in user
    assert search.calls == []
    assert "correction_query_response" not in kinds
    graph = next(r for r in records if r["kind"] == "correction_window_response")
    assert graph["payload"]["validation_ok"] is True


def test_batched_window_needs_the_batched_strategy(tmp_path, monkeypatch) -> None:
    """A mixed chain resolves to batched: the request turn runs, as before."""

    from finesub.llm.stages import task_graph as task_graph_module

    monkeypatch.setattr(
        task_graph_module, "resolve_execution_strategy", lambda profile, **kw: ("batched", "test")
    )
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [REQUEST, DELIVERY])
    assert len(calls) == 2 and "correction_query_response" in kinds


# --- fast + local: one whole-input task ----------------------------------------


def test_fast_local_under_the_task_flow_is_one_window_task(tmp_path, monkeypatch) -> None:
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(STABLE), encoding="utf-8")
    window = _window()
    monkeypatch.setattr(
        orchestration,
        "decide_fast_mode",
        lambda **kwargs: orchestration.FastDecision(mode="auto", enabled=True, reason="test", window=window),
    )
    monkeypatch.setattr(orchestration, "run_research_stage", lambda **kwargs: pytest.fail("no separate preparation under fast"))
    executed: dict = {}

    def fake_execute(**kwargs):
        executed.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    from finesub.llm.correction_request import CorrectionRequest

    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable,
            output_path=tmp_path / "out" / "clip.srt",
            postprocess_profile=None,
            knowledge="none",
            knowledge_root=_knowledge_root(tmp_path),
            profile=resolve_profile("text", "local", "quality"),
            test_profile=True,
        )
    )
    assert executed["plan"].windows == [window] and executed["plan"].fast
    assert "seed_query_results" not in executed and "entry_details" not in executed
    graph = next(
        json.loads(line)
        for line in (tmp_path / "out" / "clip.llm-artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "task_graph"
    )
    assert [t["kind"] for t in graph["payload"]["tasks"]] == ["translate"]


# --- the request turn's checkpoint (L1) ---------------------------------------


def _run_resumable(tmp_path, monkeypatch, responses, *, resumable=True):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(STABLE), encoding="utf-8")

    class Client(_Client):
        def complete(self, role, messages, **kwargs):
            result = super().complete(role, messages, **kwargs)
            return dataclass_replace(result, resumable=resumable)

    Client.responses = list(responses)
    Client.calls = []
    setattr_correction(monkeypatch, "RoleClient", Client)
    search = FakeSearchClient()
    execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "local", "quality"),
        knowledge_root=(
            tmp_path / "knowledge" if (tmp_path / "knowledge").exists() else _knowledge_root(tmp_path)
        ),
        search_client=search,
        task_artifact_dir=tmp_path / "art",
        max_retries_per_window=1,
        resume=True,
        length_scale=1.0,
    )
    return Client.calls, search


def test_request_turn_checkpoint_replays_the_reply_but_reexecutes_search(tmp_path, monkeypatch) -> None:
    """The model's request is an L1 checkpoint; the harness-side search is
    deterministic state and is redone (docs/llm_local_agent.md §7)."""

    bad_delivery = DELIVERY.replace("sub|2|二。|二|7|术语note\n", "")
    with pytest.raises(Exception):
        _run_resumable(tmp_path, monkeypatch, [REQUEST, bad_delivery, bad_delivery])
    calls, search = _run_resumable(tmp_path, monkeypatch, [DELIVERY])
    # One call: the delivery. The request turn came back from the checkpoint.
    assert len(calls) == 1
    assert [m["role"] for m in calls[0][1]] == ["system", "user", "assistant", "user"]
    assert calls[0][1][2]["content"] == REQUEST
    assert search.calls == [("游戏B 剧情",)]
    replays = [
        json.loads(line)["payload"]
        for line in (tmp_path / "art" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["kind"] == "session_checkpoint_replay"
    ]
    assert any(r["session"] == "request" for r in replays)


def test_a_non_resumable_request_turn_never_seeds_the_checkpoint(tmp_path, monkeypatch) -> None:
    """Gate D answer C (docs/llm_local_agent.md §7): a reply that leaned on
    implicit provider history must not enter L1 -- the rerun re-sends it."""

    bad_delivery = DELIVERY.replace("sub|2|二。|二|7|术语note\n", "")
    with pytest.raises(Exception):
        _run_resumable(tmp_path, monkeypatch, [REQUEST, bad_delivery, bad_delivery], resumable=False)
    calls, _search = _run_resumable(tmp_path, monkeypatch, [REQUEST, DELIVERY], resumable=False)
    assert len(calls) == 2, "the request turn must be re-sent, not replayed"
    checkpoints = tmp_path / "art" / "session-checkpoints.jsonl"
    assert not checkpoints.exists() or '"request"' not in checkpoints.read_text(encoding="utf-8")


# --- retrieval=none|native: the entries-only request channel --------------------
# (window-kb-request-channel plan: the API path asks for entries per window,
# in the same request turn as `local`, just without search.)

ENTRY_REQUEST = (
    "<reasoning>查库</reasoning>\n<window_notes>主播A 待定</window_notes>\n"
    "<requested_entries>\n主播A\n</requested_entries>\n<keep_entries></keep_entries>"
)


@pytest.mark.parametrize("retrieval", ["none", "native"])
@pytest.mark.parametrize("continuity", ["serial", "parallel"])
def test_entries_only_channel_fills_entries_without_searching(
    tmp_path, monkeypatch, retrieval, continuity
) -> None:
    """A search client is even handed in: without the search grant nothing is
    searched, and its absence would not skip the turn either."""

    delivery = DELIVERY if continuity == "serial" else DELIVERY.replace("<next_advice></next_advice>", "")
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [ENTRY_REQUEST, delivery], continuity=continuity, retrieval=retrieval
    )
    assert out.exists() and len(calls) == 2
    system = calls[0][1][0]["content"]
    assert "获取缺失输入" in system and "<requested_entries>" in system
    assert "搜索 query 输出规则" not in system
    assert "你自己没有联网搜索能力" not in system
    assert ("联网搜索用你自带的工具" in system) is (retrieval == "native")
    fill = calls[1][1][3]["content"]
    assert "主播资料" in fill and "<search_results>" not in fill
    assert search.calls == []
    assert "correction_query_format_retry" not in kinds
    request = next(r for r in records if r["kind"] == "correction_query_response")
    assert request["payload"]["resolved_entry_keys"] == ["主播A"]


def test_a_keep_only_request_passes_the_shape_gate(tmp_path, monkeypatch) -> None:
    keep_only = "<reasoning>r</reasoning>\n<window_notes></window_notes>\n<keep_entries></keep_entries>"
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [keep_only, DELIVERY], retrieval="none"
    )
    assert out.exists() and len(calls) == 2
    assert "correction_query_format_retry" not in kinds


def test_none_without_a_readable_base_opens_no_channel(tmp_path, monkeypatch) -> None:
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [DELIVERY], retrieval="none", knowledge_enabled=False
    )
    assert out.exists() and len(calls) == 1
    assert "获取缺失输入" not in calls[0][1][0]["content"]
    assert "correction_query_response" not in kinds


def test_agentic_none_window_names_only_the_knowledge_tools(tmp_path, monkeypatch) -> None:
    """Its tool session carries `kb_*` but no web tools (`client.py`), so the
    note must not promise `web_search`; and it can read the base itself, so
    it is asked to keep."""

    from finesub.llm.stages import task_graph as task_graph_module

    monkeypatch.setattr(
        task_graph_module, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "test")
    )
    out, calls, search, kinds, records = _run(tmp_path, monkeypatch, [DELIVERY], retrieval="none")
    assert out.exists() and len(calls) == 1
    system = calls[0][1][0]["content"]
    assert "由你自行取得" in system and "`kb_search`" in system
    assert "`web_search`" not in system
    assert "<keep_entries>" in system
    assert "correction_query_response" not in kinds


@pytest.mark.parametrize("retrieval", ["none", "native", "local"])
@pytest.mark.parametrize("knowledge_enabled", [True, False])
def test_system_keep_slot_and_user_keep_reminder_agree(retrieval, knowledge_enabled) -> None:
    """The two ends of one demand: asked for in the system contract exactly
    when the user-side reminder lists it."""

    system, user = build_correction_csv_messages(
        **_kwargs(
            profile=resolve_profile("text", retrieval, "quality"),
            knowledge_enabled=knowledge_enabled,
        )
    )
    assert ("<keep_entries>" in system["content"]) is ("`<keep_entries>`" in user["content"])


@pytest.mark.parametrize("continuity", ["serial", "parallel"])
def test_no_search_client_still_runs_the_entries_only_turn(tmp_path, monkeypatch, continuity) -> None:
    """Production never builds a search client outside `local`; its absence
    used to skip the request turn outright."""

    delivery = DELIVERY if continuity == "serial" else DELIVERY.replace("<next_advice></next_advice>", "")
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [ENTRY_REQUEST, delivery],
        continuity=continuity, retrieval="none", with_search_client=False,
    )
    assert out.exists() and len(calls) == 2
    assert "主播资料" in calls[1][1][3]["content"]
    assert "correction_query_response" in kinds


def test_a_none_window_drops_what_it_did_not_keep(tmp_path, monkeypatch) -> None:
    """window-kb-request-channel plan, acceptance 6: a window can ask an entry
    back, so the chain follows `<keep_entries>` (cap 8) instead of passing the
    whole set through -- the second window starts without the entry the
    preparation task seeded and the first window did not keep. (Passed
    through whole, as without a channel, it would still be there.)"""

    stable = {
        "segments": [
            {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
            {"id": "2", "start": 100.0, "end": 101.0, "text": "二。"},
        ]
    }
    first_delivery = (
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|\n</translated>\n<next_advice></next_advice>\n<keep_entries></keep_entries>"
    )
    second_delivery = (
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|二。|二|8|\n</translated>\n<next_advice></next_advice>"
    )
    out, calls, search, kinds, records = _run(
        tmp_path, monkeypatch, [first_delivery, second_delivery],
        retrieval="none", stable=stable, max_window_subtitle_tokens=8,
        initial_transfer_keys=("主播A",),
    )
    windows = [r for r in records if r["kind"] == "correction_window_response"]
    assert len(windows) == 2, [r["kind"] for r in records]
    assert "主播资料" in calls[0][1][1]["content"]
    assert windows[0]["payload"]["injected_entries"] == ["主播A"]
    assert windows[0]["payload"]["keep_entries"] == []
    assert "主播资料" not in calls[1][1][1]["content"]
    assert windows[1]["payload"]["injected_entries"] == []


@pytest.mark.parametrize(
    ("retrieval", "indexes", "opening"),
    [
        # `local`: the wording that shipped before none/native had a channel.
        ("local", {"streamer": "- 主播A | | x"}, "你自己没有联网搜索能力，也看不到知识库条目全文；两者都由 harness 代为取得。"),
        ("local", {}, "你自己没有联网搜索能力；搜索由 harness 代为执行。"),
        ("none", {"streamer": "- 主播A | | x"}, "`<entry_details>` 之外的知识库条目你看不到全文，需要的由 harness 代为取得。"),
        (
            "native",
            {"streamer": "- 主播A | | x"},
            "`<entry_details>` 之外的知识库条目你看不到全文，需要的由 harness 代为取得。联网搜索用你自带的工具，不在这里请求。",
        ),
    ],
)
def test_the_channel_opens_with_the_sentence_of_its_grant(retrieval, indexes, opening) -> None:
    """Pinned verbatim: a wording change here reaches every window of the
    default path, so it has to show up red rather than ship unnoticed."""

    from finesub.llm.request_prompts import render_window_request_channel

    profile = resolve_profile("text", retrieval, "quality")
    section = render_window_request_channel(
        knowledge_indexes=indexes,
        search=profile.external_injection,
        native_search=profile.native_search,
    )
    first_item = section.splitlines()[1]
    assert first_item.startswith("1. " + opening + "读完本窗口输入后"), first_item



def test_the_fill_asks_basic_a_for_its_decisions_first() -> None:
    """P437 (cold read round 4): the fill restates the delivery as the window's prompt set has it."""

    def fill(variant: str) -> str:
        return build_window_task_messages(
            request_channel=True,
            request_reply="<window_notes></window_notes>",
            knowledge_indexes={"streamer": "- 主播A | | 测试主播"},
            requested_entries=True,
            **_kwargs(variant=variant),
        )[-1]["content"]

    assert "（先 `<decisions>`、再 `<translated>` 完整终稿" in fill("basicA")
    assert "（`<translated>` 完整终稿" in fill("capableC") and "<decisions>" not in fill("capableC")
