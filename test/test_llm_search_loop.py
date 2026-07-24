from __future__ import annotations

import json

import pytest

from llm.client import LLMCallResult
from llm.config import SEARCH_LOOP_THINKING_BUDGET, LLMRole
from llm.search_loop import (
    DEGRADED_PACK_NOTICE,
    EVIDENCE_PACK_HEADER,
    parse_contract_json,
    run_search_loop,
)
from llm.web_search import QueryExtractResult, QuerySearchResult, SearchResultItem

pytestmark = pytest.mark.slow


def _contract_body(priorities: dict[str, int] | None = None) -> str:
    priorities = priorities or {"F1": 5, "F2": 3}
    return json.dumps(
        {
            "goal": "查证游戏B的剧情与BOSS名",
            "facts": [
                {
                    "id": fact_id,
                    "fact": f"事实 {fact_id}",
                    "priority": priority,
                    "done_when": "找到官方来源",
                }
                for fact_id, priority in priorities.items()
            ],
            "out_of_scope": ["常识"],
        },
        ensure_ascii=False,
    )


def _search_result(query: str) -> QuerySearchResult:
    return QuerySearchResult(
        query=query,
        provider="tavily",
        items=(
            SearchResultItem(
                title="wiki", url="https://example.test", snippet=f"{query} 的资料"
            ),
        ),
    )


def _extract_result(url: str, guided: str) -> QueryExtractResult:
    return QueryExtractResult(
        url=url,
        provider="exa",
        content=f"{url} 整页内容" + (f"（重点：{guided}）" if guided else ""),
    )


class FakeSearchClient:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], int | None]] = []
        self.extract_calls: list[tuple[tuple[tuple[str, str], ...], int | None]] = []

    def search_many(self, queries, *, max_queries=None):
        normalized = tuple(getattr(item, "query", item) for item in queries)
        guided = tuple(getattr(item, "guided_query", "") for item in queries)
        self.calls.append((normalized, max_queries))
        return [_search_result(query) for query in normalized]

    def extract_many(self, requests, *, max_urls=None):
        normalized = tuple((req.url, req.guided_query) for req in requests)
        self.extract_calls.append((normalized, max_urls))
        return [_extract_result(url, guided) for url, guided in normalized]


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[LLMRole, list, dict]] = []

    def complete(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return LLMCallResult(
            content=self.responses.pop(0),
            role=role,
            model="fake-lite",
            fallback_used=False,
            raw_response={},
        )


def test_parse_contract_json_is_tolerant() -> None:
    assert parse_contract_json(_contract_body())["goal"].startswith("查证")
    assert parse_contract_json("") == {}
    assert parse_contract_json("不是 JSON") == {}


def test_premature_pack_warnings_use_latest_status_across_ledger() -> None:
    from llm.search_loop import _premature_pack_warnings

    fact_index = {
        "F1": {"priority": 5},
        "F2": {"priority": 3},
        "F3": {"priority": 1},  # below the gate
    }
    # Accumulated ledger: F1 not_found in round 0 then confirmed in round 1
    # (last-wins → resolved, not flagged); F2 stays partial; F3 partial but
    # priority 1 < 2.
    ledger = (
        "## 搜索轮 0\nF1: not_found 查不到\nF2: partial 部分\nF3: partial 弱\n\n"
        "## 搜索轮 1\nF1: confirmed 找到了 (https://x)\n"
    )
    warnings = _premature_pack_warnings(ledger, fact_index)
    flagged = {w["fact_id"] for w in warnings}
    assert flagged == {"F2"}
    assert warnings[0]["status"] == "partial" and warnings[0]["priority"] == 3


def test_premature_pack_warnings_catch_earlier_unresolved_facts() -> None:
    """Regression: a fact unresolved in round 0 and not restated later must
    still be flagged — the scan is over the accumulated ledger, not the
    current round's delta."""
    from llm.search_loop import _premature_pack_warnings

    fact_index = {"F1": {"priority": 4}}
    ledger = (
        "## 搜索轮 0\nF1: not_found 暂无\n\n"
        "## 搜索轮 1\n新发现：某支线更新\n"  # round-1 delta never mentions F1
    )
    warnings = _premature_pack_warnings(ledger, fact_index)
    assert [w["fact_id"] for w in warnings] == ["F1"]


def test_search_loop_round_notice_selected_by_round_without_duplication() -> None:
    from llm.prompts import build_search_loop_messages

    non_final = build_search_loop_messages(
        round_index=1, max_rounds=3, is_final_round=False
    )
    user = next(m["content"] for m in non_final if m["role"] == "user")
    assert "默认继续检索、不要过早收尾" in user
    assert "剩余 1 轮" in user  # remaining = max_rounds - round_index - 1
    # The full continue notice is injected once (no verbatim double-injection).
    assert user.count("不要过早收尾") == 1

    final = build_search_loop_messages(
        round_index=2, max_rounds=3, is_final_round=True
    )
    final_user = next(m["content"] for m in final if m["role"] == "user")
    assert "本轮必须输出 `<evidence_pack>` 收尾" in final_user
    assert "不要过早收尾" not in final_user


def test_loop_stops_when_evidence_pack_is_emitted() -> None:
    pack = (
        "<progress_update>\nF1: confirmed 官方名「王」 (https://example.test)\n</progress_update>\n"
        "<evidence_pack>\n## 结论\nF1 confirmed\n## 关键证据摘录\n（略）\n## 未解决\n（无）\n</evidence_pack>"
    )
    client = FakeClient([pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
    )

    assert not result.degraded
    assert result.search_rounds_executed == 1
    assert result.evidence_pack.startswith(EVIDENCE_PACK_HEADER)
    assert "F1 confirmed" in result.evidence_pack
    assert "## 搜索轮 0" in result.progress_log
    assert len(client.calls) == 1
    # Loop calls use the lightweight role with medium per-call thinking.
    role, _, kwargs = client.calls[0]
    assert role == LLMRole.LIGHTWEIGHT
    assert kwargs["thinking_budget"] == SEARCH_LOOP_THINKING_BUDGET == 26_214
    assert kwargs["thinking_level"] == "medium"


def test_search_judge_resumes_after_search_is_reexecuted(tmp_path) -> None:
    pack = (
        "<reasoning>证据充分。</reasoning>\n"
        "<progress_update>F1: confirmed 官方名 (https://example.test)</progress_update>\n"
        "<evidence_pack>## 结论\nF1 confirmed\n## 关键证据摘录\n"
        "官方资料\n## 未解决\n（无）</evidence_pack>"
    )
    artifact_dir = tmp_path / "artifacts"
    first_search = FakeSearchClient()
    first = run_search_loop(
        contract_body=_contract_body({"F1": 5}),
        round0_queries=["游戏B 官方名"],
        client=FakeClient([pack]),
        search_client=first_search,
        max_rounds=1,
        task_artifact_dir=artifact_dir,
    )

    class BlockingClient:
        def complete(self, role, messages, **kwargs):
            raise AssertionError("validated search judge should resume")

    resumed_search = FakeSearchClient()
    resumed = run_search_loop(
        contract_body=_contract_body({"F1": 5}),
        round0_queries=["游戏B 官方名"],
        client=BlockingClient(),
        search_client=resumed_search,
        max_rounds=1,
        task_artifact_dir=artifact_dir,
    )

    assert first.evidence_pack == resumed.evidence_pack
    assert first_search.calls == resumed_search.calls
    artifacts = [
        json.loads(line)
        for line in (artifact_dir / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    replay = [item for item in artifacts if item["kind"] == "session_checkpoint_replay"]
    assert replay[-1]["payload"]["session"] == "search-judge"
    assert replay[-1]["payload"]["key"] == "search-loop:round0"


def test_loop_round_cap_priority_decrement_and_dedup() -> None:
    followup1 = (
        "<progress_update>\nF1: partial 候选名\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名\n游戏B 剧情\n</search_queries>"
    )
    followup2 = (
        "<progress_update>\nF2: partial 线索\n</progress_update>\n"
        "<search_queries>\nF2|游戏B 结局\nF1|游戏B BOSS 官方名\n</search_queries>"
    )
    final_pack = (
        "<progress_update>\nF1: confirmed\n</progress_update>\n"
        "<evidence_pack>\n## 结论\n完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([followup1, followup2, final_pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情", "游戏B 剧情", "游戏B 发售日"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        round0_query_cap=8,
        followup_query_cap=4,
    )

    assert result.search_rounds_executed == 3
    assert len(client.calls) == 3
    # Round 0 dedups repeated queries; follow-up rounds strip fact tags and
    # drop queries already executed in earlier rounds.
    assert search_client.calls[0][0] == ("游戏B 剧情", "游戏B 发售日")
    assert search_client.calls[1][0] == ("游戏B BOSS 官方名",)
    assert search_client.calls[2][0] == ("游戏B 结局",)
    # Priorities decremented once per searched round for tagged facts only.
    facts = {fact["id"]: fact for fact in result.contract["facts"]}
    assert facts["F1"]["priority"] == 4  # queried in round 1 (round-2 dupe dropped)
    assert facts["F2"]["priority"] == 2  # queried in round 2
    # The final-round prompt announces the hard boundary.
    final_user = client.calls[2][1][1]["content"]
    assert "本轮必须输出 `<evidence_pack>` 收尾" in final_user
    assert not result.degraded


def test_loop_runs_extracts_with_guided_and_combined_cap() -> None:
    # Follow-up asks for 2 queries + 4 extract URLs. With followup_query_cap=2
    # the budget is 4 half-units: 2 queries cost 4, leaving no room for extracts.
    followup = (
        "<progress_update>\nF1: partial\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名 >> 关注 BOSS 别名\n游戏B 结局\n</search_queries>\n"
        "<extract_urls>\n"
        "https://example.test/a >> 提取 BOSS 名\n"
        "https://example.test/b\n"
        "https://example.test/c\n"
        "https://example.test/d\n"
        "</extract_urls>"
    )
    final_pack = (
        "<progress_update>\nF1: confirmed\n</progress_update>\n"
        "<evidence_pack>\n## 结论\n完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([followup, final_pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        round0_query_cap=8,
        followup_query_cap=2,
    )

    # Two queries consume the whole budget -> no extract this round.
    assert search_client.calls[1][0] == ("游戏B BOSS 官方名", "游戏B 结局")
    assert search_client.extract_calls == []
    # Guided query rides along on the search request, not the query text.
    assert result.rounds[1]["queries"] == ["游戏B BOSS 官方名", "游戏B 结局"]
    assert not result.degraded


def test_loop_extract_urls_consume_half_a_query_unit_each() -> None:
    # 1 query (cost 2) + budget 4 leaves 2 half-units == 2 extract URLs.
    followup = (
        "<progress_update>\nF1: partial\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名\n</search_queries>\n"
        "<extract_urls>\n"
        "https://example.test/a >> 提取阵营\n"
        "https://example.test/b\n"
        "https://example.test/c\n"
        "</extract_urls>"
    )
    final_pack = (
        "<evidence_pack>\n## 结论\n完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([followup, final_pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        round0_query_cap=8,
        followup_query_cap=2,
    )

    urls = [url for url, _ in search_client.extract_calls[0][0]]
    assert urls == ["https://example.test/a", "https://example.test/b"]
    # Guided query is carried on the first extract request only.
    assert search_client.extract_calls[0][0][0] == ("https://example.test/a", "提取阵营")
    assert result.executed_extract_urls == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    # Extracted page content is injected into the next loop round's results.
    assert result.rounds[1]["extract_urls"] == [
        "https://example.test/a",
        "https://example.test/b",
    ]
    round1_user = client.calls[1][1][1]["content"]
    assert "<current_research_contract>" in round1_user
    assert "<previous_search_request>" in round1_user
    assert "F1|游戏B BOSS 官方名" in round1_user
    assert "https://example.test/a >> 提取阵营" in round1_user
    assert "https://example.test/c" not in round1_user  # over cap, not executed
    assert round1_user.index("<previous_search_request>") < round1_user.index(
        "<search_results>"
    )
    assert "--- 深度提取 url: https://example.test/a ---" in round1_user
    assert not result.degraded


def test_final_round_without_pack_degrades_to_raw_results() -> None:
    followup = (
        "<progress_update>\nF1: partial\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名\n</search_queries>"
    )
    # Final round disobeys and emits queries again -> degraded fallback.
    disobedient = (
        "<progress_update>\nF1: partial 还想继续\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 别名\n</search_queries>"
    )
    client = FakeClient([followup, disobedient])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=2,
    )

    assert result.degraded
    assert DEGRADED_PACK_NOTICE in result.evidence_pack
    # Progress ledger and raw results are preserved in the fallback pack.
    assert "F1: partial" in result.evidence_pack
    assert "游戏B 剧情 的资料" in result.evidence_pack
    assert result.search_rounds_executed == 2


def test_unparseable_loop_output_retries_then_degrades() -> None:
    client = FakeClient(["完全没有标签", "还是没有标签"])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        max_parse_retries=1,
    )

    assert result.degraded
    assert len(client.calls) == 2
    assert "游戏B 剧情 的资料" in result.evidence_pack


def test_loop_artifacts_and_token_rows_are_recorded(tmp_path) -> None:
    pack = (
        "<progress_update>\nF1: confirmed\n</progress_update>\n"
        "<evidence_pack>\n## 结论\nF1 confirmed：完成\nF2 confirmed：完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([pack])
    token_rows: list[dict] = []

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=FakeSearchClient(),
        max_rounds=3,
        task_artifact_dir=tmp_path,
        task_id="task-1",
        token_rows=token_rows,
    )

    assert not result.degraded
    assert [row["call"] for row in token_rows] == ["search_loop"]
    artifacts = [
        json.loads(line)
        for line in (tmp_path / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [artifact["kind"] for artifact in artifacts]
    assert kinds == ["search_loop_round", "search_loop_round"]
    # First record: executed search round; second: the loop model call.
    assert artifacts[0]["payload"]["queries"] == ["游戏B 剧情"]
    assert artifacts[1]["payload"]["has_evidence_pack"] is True
    meta = result.to_metadata()
    assert meta["search_rounds_executed"] == 1
    assert meta["executed_queries"] == ["游戏B 剧情"]


class InflatedTokenCounter:
    """Deterministic counter that inflates sizes so block budgets bind."""

    source = "test-inflated"

    def __init__(self, per_char: int = 20) -> None:
        self.per_char = per_char

    def count_text(self, text: str) -> int:
        return len(text or "") * self.per_char

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def test_truncated_or_dropped_query_results_keep_fact_priority() -> None:
    # F1's follow-up section fits the budget; F2's is far over the 4k
    # per-section cap and gets truncated. Only F1 may be decremented.
    long_query = "B" * 300
    followup1 = (
        "<progress_update>\nF1: partial\n</progress_update>\n"
        f"<search_queries>\nF1|游戏B BOSS 官方名\nF2|{long_query}\n</search_queries>"
    )
    final_pack = (
        "<progress_update>\nF1: confirmed\n</progress_update>\n"
        "<evidence_pack>\n## 结论\n完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([followup1, final_pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=2,
        round0_query_cap=8,
        followup_query_cap=4,
        token_counter=InflatedTokenCounter(),
    )

    facts = {fact["id"]: fact for fact in result.contract["facts"]}
    assert facts["F1"]["priority"] == 4  # fully rendered -> decremented
    assert facts["F2"]["priority"] == 3  # truncated by budget -> untouched
    followup_round = result.rounds[1]
    assert followup_round["decremented_facts"] == ["F1"]
    assert "F1" in followup_round["touched_facts"] and "F2" in followup_round["touched_facts"]
    report = followup_round["render_report"]
    assert long_query in report["truncated"] + report["dropped"]
    assert "游戏B BOSS 官方名" in report["included"]


def test_loop_requests_knowledge_entries_injected_next_round_with_dedupe(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text("", encoding="utf-8")
    (knowledge_root / "common" / "index.md").write_text(
        "- 崩坏星穹铁道 [游戏] | 崩铁、星铁 | 回合制 RPG\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "崩坏星穹铁道.md").write_text(
        "# 崩坏星穹铁道\n\n## 简介\n\n- 开拓者。\n", encoding="utf-8"
    )

    followup1 = (
        "<progress_update>\nF1: partial\n</progress_update>\n"
        "<requested_entries>\n崩铁\n没有的条目\n</requested_entries>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名\n</search_queries>"
    )
    followup2 = (
        "<progress_update>\nF1: partial 2\n</progress_update>\n"
        "<requested_entries>\n星铁\n</requested_entries>\n"
        "<search_queries>\nF1|游戏B 结局\n</search_queries>"
    )
    final_pack = (
        "<progress_update>\nF1: confirmed\n</progress_update>\n"
        "<evidence_pack>\n## 结论\n完成\n## 关键证据摘录\n## 未解决\n</evidence_pack>"
    )
    client = FakeClient([followup1, followup2, final_pack])

    result = run_search_loop(
        contract_body=_contract_body({"F1": 5}),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=FakeSearchClient(),
        max_rounds=3,
        knowledge_root=knowledge_root,
        task_artifact_dir=tmp_path / "artifacts",
    )

    assert not result.degraded
    # Round 0's judge call sees the indices but no entries yet.
    round0_user = client.calls[0][1][1]["content"]
    assert "崩坏星穹铁道 [游戏] | 崩铁、星铁" in round0_user
    assert "<knowledge_entries>\n（无）" in round0_user
    # The round-1 judge call gets the entry requested in round 0's output.
    round1_user = client.calls[1][1][1]["content"]
    assert "开拓者。" in round1_user
    assert "<previous_requested_entries>\n崩铁\n没有的条目" in round1_user
    assert round1_user.index("<previous_requested_entries>") < round1_user.index(
        "<knowledge_entries>"
    )
    # Round 2 is final: indices are blanked and the duplicate alias request
    # (星铁 -> same entry) is deduped, so no entry text is injected again.
    round2_user = client.calls[2][1][1]["content"]
    assert "开拓者。" not in round2_user
    assert "<streamer_index>\n（空）" in round2_user
    assert "<common_index>\n（空）" in round2_user

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    entry_records = [
        a["payload"] for a in artifacts if "requested_entries" in a.get("payload", {})
    ]
    assert entry_records[0]["requested_entries"] == ["崩铁", "没有的条目"]
    assert entry_records[0]["injected_entries"] == ["崩坏星穹铁道"]
    assert entry_records[0]["missing_entries"] == ["没有的条目"]
    # Second request resolves to the already-injected entry -> nothing new.
    assert entry_records[1]["requested_entries"] == ["星铁"]
    assert entry_records[1]["injected_entries"] == []


# ---------------------------------------------------------------------------
# v2 search-loop flow (loop_version="v2"): every round emits a full Evidence
# Pack; <search_queries> is the continuation signal, its absence terminates.
# ---------------------------------------------------------------------------


def _v2_pack(conclusion: str, *, queries: str = "") -> str:
    pack = (
        "<evidence_pack>\n"
        "## 结论\n"
        f"{conclusion}\n"
        "## 关键证据摘录\n略\n"
        "## 未解决\n（无）\n"
        "</evidence_pack>"
    )
    if queries:
        pack += f"\n<search_queries>\n{queries}\n</search_queries>"
    return pack


def test_check_fact_coverage_flags_only_facts_absent_from_conclusion() -> None:
    from llm.search_loop import _check_fact_coverage

    fact_index = {"F1": {"priority": 5}, "F2": {"priority": 3}, "F3": {"priority": 1}}
    # F1 mentioned normally; F3 mentioned with the [unresolved] prefix; F2 only
    # appears in the 关键证据摘录 section (outside 结论) so it stays uncovered.
    pack = (
        "## 结论\n"
        "F1 confirmed 官方名（https://x）\n"
        "[unresolved] F3 仍无可靠来源\n"
        "## 关键证据摘录\n"
        "F2 出现在证据里但没有写进结论\n"
        "## 未解决\n（无）\n"
    )
    assert _check_fact_coverage(pack, fact_index) == ["F2"]

    full = "## 结论\nF1 ok\nF2 ok\nF3 ok\n## 未解决\n"
    assert _check_fact_coverage(full, fact_index) == []


def test_build_search_loop_v2_messages_threads_previous_pack() -> None:
    from llm.prompts import build_search_loop_v2_messages

    non_final = build_search_loop_v2_messages(
        round_index=1,
        max_rounds=3,
        is_final_round=False,
        previous_evidence_pack="## 结论\nF1 partial 候选名",
    )
    assert [m["role"] for m in non_final] == ["system", "user"]
    user = next(m["content"] for m in non_final if m["role"] == "user")
    assert "F1 partial 候选名" in user  # prior round's pack is fed back in
    assert "不要过早收尾" in user  # non-final continue notice reused from v1
    # v2 drops the progress_update block entirely.
    assert "progress_update" not in user

    final = build_search_loop_v2_messages(
        round_index=2, max_rounds=3, is_final_round=True
    )
    final_user = next(m["content"] for m in final if m["role"] == "user")
    assert "本轮必须输出 `<evidence_pack>` 收尾" in final_user
    assert "不要过早收尾" not in final_user


def test_v2_pack_without_queries_terminates_immediately() -> None:
    client = FakeClient([_v2_pack("F1 confirmed\nF2 confirmed")])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        loop_version="v2",
    )

    assert not result.degraded
    assert result.search_rounds_executed == 1
    assert len(client.calls) == 1
    assert "F1 confirmed" in result.evidence_pack


def test_v2_pack_with_queries_continues_then_terminates() -> None:
    r1 = _v2_pack("F1 partial 候选名", queries="F2|游戏B 结局")
    r2 = _v2_pack("F1 confirmed\nF2 confirmed")
    client = FakeClient([r1, r2])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        loop_version="v2",
    )

    assert not result.degraded
    assert result.search_rounds_executed == 2
    assert len(client.calls) == 2
    assert "F2 confirmed" in result.evidence_pack
    # The follow-up query was actually executed for the next round.
    assert search_client.calls[1][0] == ("游戏B 结局",)
    # Round 1's prompt carries round 0's evidence pack in <previous_evidence_pack>.
    second_user = client.calls[1][1][1]["content"]
    assert "F1 partial 候选名" in second_user


def test_v2_missing_pack_retries_within_round() -> None:
    # v2 requires an evidence pack every round; a queries-only reply is a parse
    # miss that triggers the in-round retry rather than terminating.
    no_pack = "<search_queries>\nF1|游戏B BOSS\n</search_queries>"
    client = FakeClient([no_pack, _v2_pack("F1 confirmed\nF2 confirmed")])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=3,
        loop_version="v2",
    )

    assert not result.degraded
    assert result.search_rounds_executed == 1  # no extra search round from the retry
    assert len(client.calls) == 2  # retried because the pack was missing
    assert "F1 confirmed" in result.evidence_pack


def test_v2_degrades_when_final_round_yields_no_pack() -> None:
    no_pack = "<search_queries>\nF1|游戏B BOSS\n</search_queries>"
    client = FakeClient([no_pack, no_pack])
    search_client = FakeSearchClient()

    result = run_search_loop(
        contract_body=_contract_body(),
        round0_queries=["游戏B 剧情"],
        client=client,
        search_client=search_client,
        max_rounds=1,  # round 0 is the final round
        loop_version="v2",
    )

    assert result.degraded
    assert result.evidence_pack.endswith(DEGRADED_PACK_NOTICE) or (
        DEGRADED_PACK_NOTICE in result.evidence_pack
    )
