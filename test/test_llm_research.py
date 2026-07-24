from __future__ import annotations

from dataclasses import replace
import json

import pytest

from llm.chunking import SubtitleSegment, plan_correction_windows
from llm.client import LLMCallResult
from llm.config import SEARCH_LOOP_THINKING_BUDGET, LLMRole
from llm.prompts import ContextPack
from llm.research import (
    check_research_input_limit,
    load_research_context,
    parse_round1_output,
    parse_round2_output,
    render_research_transcript,
    resolve_round1_entries,
    run_research,
)
from llm.web_search import QuerySearchResult, SearchResultItem, QueryExtractResult


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))



def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "第一句。"),
        SubtitleSegment("2", 1.2, 2.0, "第二句还没完"),
        SubtitleSegment("3", 2.0, 3.0, "现在结束。"),
        SubtitleSegment("4", 3.2, 4.0, "第四句。"),
        SubtitleSegment("5", 4.2, 5.0, "第五句。"),
    ]


class FakeSearchClient:
    def __init__(self, results: list[QuerySearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[tuple[str, ...], int | None]] = []

    def search_many(self, queries, *, max_queries=None):
        requested = [getattr(item, "query", item) for item in queries]
        self.calls.append((tuple(requested), max_queries))
        if not self.results:
            return []
        # Like the real client, echo each request's query into its result
        # (priority decrement matches rendered labels against request queries).
        return [
            replace(self.results[index % len(self.results)], query=query)
            for index, query in enumerate(requested)
        ]

    def extract_many(self, requests, *, max_urls=None):
        return []


def test_render_research_transcript_marks_windows_without_duplicates() -> None:
    segments = _segments()
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
    )
    transcript = render_research_transcript(segments, windows)

    for window in windows:
        assert f"--- window {window.chunk_id} ---" in transcript
    for segment in segments:
        assert transcript.count(f"{segment.id}|") == 1


def test_parse_round1_output_reads_tag_blocks() -> None:
    text = (
        "好的，以下是结果。\n"
        "<requested_entries>\n主播A\n- 游戏B\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n1. 游戏B 剧情\n游戏B 角色名\n</search_queries>\n"
    )

    result = parse_round1_output(text)

    assert result.requested_entries == ("主播A", "游戏B")
    assert result.keep_entries == ()
    assert result.search_queries == ("游戏B 剧情", "游戏B 角色名")
    assert result.analysis_notes == ""
    assert result.research_contract == ""


def test_resolve_round1_entries_caps_channels_and_merges_keep_first(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "index.md").write_text(
        "- 游戏B [游戏] | Game B | 测试游戏\n", encoding="utf-8"
    )
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n\n主播资料。\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "游戏B.md").write_text(
        "# 游戏B\n\n游戏资料。\n", encoding="utf-8"
    )

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["Game B", "不存在"],
        keep_names=["エーちゃん", "Game B"],
        visible_keep_keys=["主播A"],
        max_requested_entries=1,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]  # keep channel wins shared cap
    assert missing == ["不存在"]
    assert ignored == ["Game B"]  # indexed, but not visible as a preinjection
    assert dropped == ["游戏B"]


def test_resolve_round1_entries_enforces_eight_each_and_twelve_total(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    kept_keys = [f"保留{i}" for i in range(10)]
    requested_keys = [f"请求{i}" for i in range(10)]
    (streamer / "index.md").write_text(
        "".join(f"- {key} | keep-{i} | 测试\n" for i, key in enumerate(kept_keys)),
        encoding="utf-8",
    )
    (common / "index.md").write_text(
        "".join(
            f"- {key} [游戏] | request-{i} | 测试\n"
            for i, key in enumerate(requested_keys)
        ),
        encoding="utf-8",
    )
    for key in [*kept_keys, *requested_keys]:
        target = streamer if key.startswith("保留") else common
        (target / f"{key}.md").write_text(f"# {key}\n\n内容。\n", encoding="utf-8")

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=requested_keys,
        keep_names=kept_keys,
        visible_keep_keys=kept_keys,
    )

    assert list(selected) == [*kept_keys[:8], *requested_keys[:4]]
    assert missing == []
    assert ignored == []
    assert dropped == [*kept_keys[8:], *requested_keys[4:]]


def test_resolve_round1_entries_does_not_report_cross_channel_duplicate_as_dropped(
    tmp_path,
) -> None:
    knowledge_root = tmp_path / "knowledge"
    streamer = knowledge_root / "streamer"
    common = knowledge_root / "common"
    streamer.mkdir(parents=True)
    common.mkdir(parents=True)
    (streamer / "index.md").write_text("- 主播A | エーちゃん | 测试\n", encoding="utf-8")
    (common / "index.md").write_text("", encoding="utf-8")
    (streamer / "主播A.md").write_text("# 主播A\n\n内容。\n", encoding="utf-8")

    selected, missing, ignored, dropped = resolve_round1_entries(
        knowledge_root,
        requested_names=["エーちゃん"],
        keep_names=["主播A"],
        visible_keep_keys=["主播A"],
        max_requested_entries=0,
        max_keep_entries=1,
        max_total_entries=1,
    )

    assert list(selected) == ["主播A"]
    assert missing == []
    assert ignored == []
    assert dropped == []


def test_parse_round1_output_reads_optional_analysis_notes() -> None:
    text = (
        "<analysis_notes>\n主播在玩游戏B（待定）。\n</analysis_notes>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )

    result = parse_round1_output(text)

    assert result.analysis_notes == "主播在玩游戏B（待定）。"


def test_parse_round1_output_contract_required_only_in_multi_round() -> None:
    contract = '{"goal": "查证游戏B", "facts": [], "out_of_scope": []}'
    text = (
        "<analysis_notes></analysis_notes>\n"
        f"<research_contract>\n{contract}\n</research_contract>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 剧情\n</search_queries>"
    )

    result = parse_round1_output(text, expect_contract=True)
    assert result.research_contract == contract

    without_contract = (
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )
    assert parse_round1_output(without_contract).research_contract == ""
    with pytest.raises(ValueError, match="research_contract"):
        parse_round1_output(without_contract, expect_contract=True)


def test_parse_round1_output_requires_both_blocks() -> None:
    with pytest.raises(ValueError, match="search_queries"):
        parse_round1_output(
            "<requested_entries>\n主播A\n</requested_entries>\n"
            "<keep_entries></keep_entries>"
        )


def test_parse_round1_output_accepts_empty_blocks() -> None:
    result = parse_round1_output(
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>"
    )

    assert result.requested_entries == ()
    assert result.search_queries == ()


def test_parse_round2_output_reads_context_pack_tag() -> None:
    payload = {
        "general_context": {"global_summary": "摘要"},
        "window_contexts": [{"window_id": "0001", "context": "背景一"}],
    }
    text = f"<context_pack>\n{json.dumps(payload, ensure_ascii=False)}\n</context_pack>"

    pack = parse_round2_output(text)

    assert pack.general_context["global_summary"] == "摘要"
    assert pack.window_context_for("0001-a") == "背景一"


def test_parse_round2_output_falls_back_to_bare_json() -> None:
    payload = {"general_context": {"global_summary": "摘要"}, "window_contexts": []}

    pack = parse_round2_output(json.dumps(payload, ensure_ascii=False))

    assert pack.general_context["global_summary"] == "摘要"


def test_check_research_input_limit_raises_on_overflow() -> None:
    messages = [{"role": "user", "content": "字" * 10_000}]

    with pytest.raises(ValueError, match="Split the audio"):
        check_research_input_limit(
            messages,
            round_name="round 1",
            limits=type("L", (), {"prompt_input_limit": 10})(),
            counter=FakeTokenCounter(),
        )


def test_run_research_two_rounds_with_local_search(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n\n## 档案\n\n关西腔。\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "index.md").write_text("", encoding="utf-8")

    round1 = (
        "<requested_entries>\nエーちゃん\n未知条目\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 剧情\n</search_queries>"
    )
    round2_payload = {
        "general_context": {"global_summary": "主播A 在玩游戏B。"},
        "window_contexts": [{"window_id": "0001", "context": "开场杂谈"}],
    }
    round2 = (
        f"<context_pack>\n{json.dumps(round2_payload, ensure_ascii=False)}\n</context_pack>"
    )
    responses = [round1, round2]
    seen = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            seen.append((role, messages, kwargs))
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={
                    "usageMetadata": {"candidatesTokenCount": 200, "thoughtsTokenCount": 50}
                },
            )

    search_client = FakeSearchClient(
        [
            QuerySearchResult(
                query="游戏B 剧情",
                provider="tavily",
                items=(
                    SearchResultItem(
                        title="wiki", url="https://example.test", snippet="恐怖游戏"
                    ),
                ),
            )
        ]
    )

    payload = run_research(
        transcript="--- window 0001 ---\n1|こんにちは\n",
        extra_info="来源 URL",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=search_client,
        task_artifact_dir=tmp_path / "artifacts",
        task_id="task-1",
        search_rounds=1,
        token_counter=FakeTokenCounter(),
    )

    assert [role for role, _, _ in seen] == [LLMRole.GENERAL_CAPABLE] * 2
    assert search_client.calls == [(("游戏B 剧情",), 8)]
    round2_user = seen[1][1][1]["content"]
    assert "关西腔" in round2_user
    assert "恐怖游戏" in round2_user
    assert "https://example.test" in round2_user
    assert payload["injected_entries"] == ["主播A"]
    assert payload["missing_entries"] == ["未知条目"]
    assert payload["round1"]["search_queries"] == ["游戏B 剧情"]
    assert payload["search_results"][0]["provider"] == "tavily"

    context_path = tmp_path / "research-context.json"
    context_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    pack = load_research_context(context_path)
    assert isinstance(pack, ContextPack)
    assert pack.general_context["global_summary"] == "主播A 在玩游戏B。"
    assert pack.window_context_for("0001") == "开场杂谈"

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    kinds = [artifact["kind"] for artifact in artifacts]
    assert kinds == [
        "research_round1_response",
        "research_search_results",
        "research_round2_response",
        "token_distribution_report",
    ]
    report = payload["token_report"]
    assert report["phase"] == "research"
    assert [row["call"] for row in report["rows"]] == [
        "research_round1",
        "research_round2",
    ]
    assert report["rows"][0]["tokens"]["output_tokens"] == 200
    assert report["totals"]["thinking_tokens"] == 100
    assert report["totals"]["call_count"] == 2

    exchanges = sorted((tmp_path / "artifacts" / "exchanges").glob("*.md"))
    assert [path.name for path in exchanges] == [
        "001-research-round1-attempt0.md",
        "002-research-round2-attempt0.md",
    ]
    round1_exchange = exchanges[0].read_text(encoding="utf-8")
    assert "## 请求（user）" in round1_exchange
    assert "エーちゃん" in round1_exchange
    assert "恐怖游戏" in exchanges[1].read_text(encoding="utf-8")


def test_run_research_skips_search_when_disabled(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 剧情\n</search_queries>",
        '<context_pack>{"general_context": {"global_summary": "摘要"},'
        ' "window_contexts": []}</context_pack>',
    ]

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    search_client = FakeSearchClient()
    payload = run_research(
        transcript="1|你好\n",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=search_client,
        enable_web_search=False,
        token_counter=FakeTokenCounter(),
    )

    assert search_client.calls == []
    assert payload["search_results"] == []
    assert payload["context_pack"]["general_context"]["global_summary"] == "摘要"


def test_run_research_resumes_both_validated_sessions(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    responses = [
        "<reasoning>分析字幕。</reasoning>\n"
        "<analysis_notes>没有待查事实。</analysis_notes>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        "<reasoning>整理上下文。</reasoning>\n"
        '<context_pack>{"general_context":{"global_summary":"摘要"},'
        '"window_contexts":[]}</context_pack>\n'
        "<keep_entries></keep_entries>",
    ]

    class FirstClient:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    class BlockingClient:
        def complete(self, role, messages, **kwargs):
            raise AssertionError("validated research session should resume")

    kwargs = dict(
        transcript="--- window 0001 ---\n1|你好\n",
        knowledge_root=knowledge_root,
        search_client=FakeSearchClient(),
        enable_web_search=False,
        task_artifact_dir=artifact_dir,
        token_counter=FakeTokenCounter(),
    )
    first = run_research(client=FirstClient(), **kwargs)
    resumed = run_research(client=BlockingClient(), **kwargs)

    assert not responses
    assert resumed["context_pack"] == first["context_pack"]
    assert (artifact_dir / "research-round1-input.json").exists()
    assert (artifact_dir / "research-round2-input.json").exists()
    records = [
        json.loads(line)
        for line in (artifact_dir / "session-checkpoints.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["session"] for record in records] == [
        "research-r1",
        "research-r2",
    ]
    artifacts = [
        json.loads(line)
        for line in (artifact_dir / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        item["payload"]["session"]
        for item in artifacts
        if item["kind"] == "session_checkpoint_replay"
    ] == ["research-r1", "research-r2"]


def test_run_research_retries_on_parse_error(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    responses = [
        "这不是任何标签块",
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {"global_summary": "摘要"},'
        ' "window_contexts": []}</context_pack>',
    ]

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    payload = run_research(
        transcript="1|你好\n",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=FakeSearchClient(),
        search_rounds=1,
        token_counter=FakeTokenCounter(),
    )

    assert payload["context_pack"]["general_context"]["global_summary"] == "摘要"
    assert not responses


def test_run_research_multi_round_uses_search_loop_and_evidence_pack(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    contract = {
        "goal": "查证游戏B的剧情",
        "facts": [
            {"id": "F1", "fact": "游戏B BOSS名", "priority": 5, "done_when": "找到官方名"}
        ],
        "out_of_scope": [],
    }
    round1 = (
        "<analysis_notes>\n主播在玩游戏B（待定）。\n</analysis_notes>\n"
        f"<research_contract>\n{json.dumps(contract, ensure_ascii=False)}\n</research_contract>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 剧情\n</search_queries>"
    )
    loop1 = (
        "<progress_update>\nF1: partial 找到候选名 (https://example.test)\n</progress_update>\n"
        "<search_queries>\nF1|游戏B BOSS 官方名\n</search_queries>"
    )
    loop2 = (
        "<progress_update>\nF1: confirmed 官方名为「王」 (https://example.test/wiki)\n</progress_update>\n"
        "<evidence_pack>\n## 结论\nF1 confirmed：BOSS 官方名「王」\n## 关键证据摘录\n（略）\n## 未解决\n（无）\n</evidence_pack>"
    )
    round2 = (
        '<context_pack>{"general_context": {"global_summary": "摘要"},'
        ' "window_contexts": []}</context_pack>'
    )
    responses = [round1, loop1, loop2, round2]
    seen = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            seen.append((role, messages, kwargs))
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    search_client = FakeSearchClient(
        [
            QuerySearchResult(
                query="游戏B 剧情",
                provider="tavily",
                items=(
                    SearchResultItem(
                        title="wiki", url="https://example.test", snippet="恐怖游戏"
                    ),
                ),
            )
        ]
    )

    payload = run_research(
        transcript="--- window 0001 ---\n1|こんにちは\n",
        extra_info="来源 URL",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=search_client,
        task_artifact_dir=tmp_path / "artifacts",
        task_id="task-1",
        search_rounds=3,
        token_counter=FakeTokenCounter(),
    )

    # round1 (general) + 2 loop calls (lightweight) + round2 (general)
    roles = [role for role, _, _ in seen]
    assert roles == [
        LLMRole.GENERAL_CAPABLE,
        LLMRole.LIGHTWEIGHT,
        LLMRole.LIGHTWEIGHT,
        LLMRole.GENERAL_CAPABLE,
    ]
    assert seen[1][2]["thinking_budget"] == SEARCH_LOOP_THINKING_BUDGET == 26_214
    assert seen[1][2]["thinking_level"] == "medium"
    # Round 1 prompt asks for a Research Contract.
    assert "<research_contract>" in seen[0][1][0]["content"]
    # Loop prompts carry background (extra_info + analysis notes).
    loop_user = seen[1][1][1]["content"]
    assert "来源 URL" in loop_user
    assert "主播在玩游戏B（待定）。" in loop_user
    # Two search rounds executed: round 0 plus the loop's follow-up.
    assert [call[0] for call in search_client.calls] == [
        ("游戏B 剧情",),
        ("游戏B BOSS 官方名",),
    ]
    # Round 2 receives the evidence pack, not raw results; system prompt uses
    # the evidence-pack fragment, and round-1 notes are injected.
    round2_system = seen[3][1][0]["content"]
    round2_user = seen[3][1][1]["content"]
    assert "Evidence Pack" in round2_system
    assert "BOSS 官方名「王」" in round2_user
    assert "<round1_notes>" in round2_user
    assert "主播在玩游戏B（待定）。" in round2_user

    assert payload["round1"]["analysis_notes"] == "主播在玩游戏B（待定）。"
    assert json.loads(payload["round1"]["research_contract"])["goal"] == "查证游戏B的剧情"
    loop_meta = payload["search_loop"]
    assert loop_meta["degraded"] is False
    assert loop_meta["search_rounds_executed"] == 2
    assert loop_meta["contract"]["facts"][0]["priority"] == 4
    assert payload["search_results"] == []

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    kinds = [artifact["kind"] for artifact in artifacts]
    assert kinds.count("search_loop_round") >= 2
    summary = next(a for a in artifacts if a["kind"] == "research_search_results")
    assert summary["payload"]["multi_round"] is True
    assert summary["payload"]["search_loop"]["search_rounds_executed"] == 2


@pytest.mark.slow
def test_run_research_preextracts_urls_from_extra_info(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text("", encoding="utf-8")
    (knowledge_root / "common" / "index.md").write_text("", encoding="utf-8")

    round1 = (
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )
    round2 = (
        '<context_pack>\n{"general_context": {}, "window_contexts": []}\n</context_pack>'
    )
    responses = [round1, round2]
    seen = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            seen.append(messages)
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"usageMetadata": {"candidatesTokenCount": 10}},
            )

    class ExtractClient:
        calls: list[list[str]] = []

        def search_many(self, queries, *, max_queries=None):
            return []

        def extract_many(self, requests, *, max_urls=None):
            ExtractClient.calls.append([req.url for req in requests])
            return [
                QueryExtractResult(
                    url=requests[0].url,
                    provider="exa",
                    title="Video",
                    content="official synopsis text",
                )
            ]

    run_research(
        transcript="1|hello",
        extra_info="素材来源: https://www.youtube.com/watch?v=abc 的切片",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=ExtractClient(),
        enable_web_search=True,
        search_rounds=1,
        task_artifact_dir=tmp_path / "artifacts",
        task_id="t1",
    )

    round1_user = seen[0][1]["content"]
    assert "official synopsis text" in round1_user
    assert "<note_url_extracts>" in round1_user
    assert ExtractClient.calls == [["https://www.youtube.com/watch?v=abc"]]


def test_run_research_content_filter_ladder_drops_toxic_note_extract(
    tmp_path, monkeypatch
) -> None:
    """Round 1 leave-one-out identifies a toxic URL extract and blacklists it."""

    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text("", encoding="utf-8")
    (knowledge_root / "common" / "index.md").write_text("", encoding="utf-8")

    round1 = (
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )
    round2 = (
        '<context_pack>\n{"general_context": {}, "window_contexts": []}\n</context_pack>'
    )
    # First complete is blocked (empty + content_filter); after the toxic
    # extract is dropped the same round-1 body must succeed.
    blocked_raw = {
        "choices": [{"finish_reason": "content_filter", "message": {}}],
        "usageMetadata": {"candidatesTokenCount": 0},
    }
    ok_raw = {"usageMetadata": {"candidatesTokenCount": 10}}
    responses: list[tuple[str, dict]] = [
        ("", blocked_raw),  # full set
        ("", blocked_raw),  # leave-one-out dropped the safe URL
        (round1, ok_raw),  # leave-one-out dropped the toxic URL
        (round2, ok_raw),
    ]
    seen_extracts: list[str] = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            user = messages[1]["content"]
            seen_extracts.append(user)
            content, raw = responses.pop(0)
            return LLMCallResult(
                content=content or None,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response=raw,
            )

    class ExtractClient:
        def search_many(self, queries, *, max_queries=None):
            return []

        def extract_many(self, requests, *, max_urls=None):
            return [
                QueryExtractResult(
                    url=req.url,
                    provider="exa",
                    title="page",
                    content=("TOXIC_PAYLOAD" if "bad" in req.url else "safe synopsis"),
                )
                for req in requests
            ]

    monkeypatch.setattr("llm.content_filter.time.sleep", lambda _s: None)
    artifacts = tmp_path / "artifacts"
    run_research(
        transcript="1|hello",
        extra_info=(
            "链接: https://example.test/good 与 https://example.test/bad-page"
        ),
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=ExtractClient(),
        enable_web_search=True,
        search_rounds=1,
        task_artifact_dir=artifacts,
        task_id="t1",
    )

    # First attempt had both extracts; a later attempt dropped the toxic one.
    assert any("TOXIC_PAYLOAD" in text for text in seen_extracts)
    assert any(
        "safe synopsis" in text and "TOXIC_PAYLOAD" not in text
        for text in seen_extracts
    )
    from llm.content_filter import BLACKLIST_ARTIFACT_KIND, LADDER_ARTIFACT_KIND

    kinds = {
        json.loads(line)["kind"]
        for line in (artifacts / "task-artifacts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }
    assert LADDER_ARTIFACT_KIND in kinds
    assert BLACKLIST_ARTIFACT_KIND in kinds


def test_run_research_preinjects_note_matched_entries_into_round1(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    (knowledge_root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n\n## 档案\n\n关西腔。\n", encoding="utf-8"
    )
    (knowledge_root / "common" / "index.md").write_text("", encoding="utf-8")

    round1 = (
        "<requested_entries></requested_entries>\n"
        "<keep_entries>エーちゃん</keep_entries>\n"
        "<search_queries></search_queries>"
    )
    round2 = '<context_pack>{"general_context": {}, "window_contexts": []}</context_pack>'
    responses = [round1, round2]
    seen = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            seen.append((role, messages, kwargs))
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    payload = run_research(
        transcript="--- window 0001 ---\n1|こんにちは\n",
        extra_info="今天是エーちゃん的杂谈回",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        enable_web_search=False,
        task_artifact_dir=tmp_path / "artifacts",
        task_id="task-1",
        search_rounds=1,
        token_counter=FakeTokenCounter(),
    )

    round1_user = seen[0][1][1]["content"]
    assert "<preinjected_entries>" in round1_user
    assert "# 主播A" in round1_user
    assert "## 主播A" not in round1_user
    assert "关西腔。" in round1_user
    round2_user = seen[1][1][1]["content"]
    assert "<knowledge_entries>" in round2_user
    assert "关西腔。" in round2_user

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    preinjection = [a for a in artifacts if a["kind"] == "knowledge_preinjection"]
    assert preinjection
    report = preinjection[0]["payload"]
    assert report["source"] == "research_round1"
    assert report["matches"][0]["key"] == "主播A"
    assert report["matches"][0]["matched_terms"] == ["エーちゃん"]
    assert report["included"] == ["主播A"]
    assert payload["round1"]["keep_entries"] == ["エーちゃん"]
    assert payload["injected_entries"] == ["主播A"]
    assert payload["ignored_keep_entries"] == []
    assert payload["context_pack"] == {"general_context": {}, "window_contexts": {}}


def test_run_research_stage_writes_planning_metadata(tmp_path, monkeypatch) -> None:
    from llm.profiles import resolve_profile
    from llm.prompts import PROMPT_VERSION
    from llm.config import WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    from llm.research import run_research_stage

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)

    responses = [
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>",
        '<context_pack>{"general_context": {}, "window_contexts": []}</context_pack>',
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    monkeypatch.setattr("llm.research.LiteLLMRoleClient", FakeClient)

    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        enable_web_search=False,
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("mm", "med"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    planning = payload["planning"]
    assert planning["prompt_version"] == PROMPT_VERSION
    assert planning["context_reserve_tokens"] == WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    assert planning["profile_id"] == "mm-med"
    assert planning["output_scale"] == 1.0
    assert planning["stable_json_hash"].startswith("sha256:")
    assert planning["extra_info_hash"].startswith("sha256:")
    assert planning["knowledge_inputs_hash"].startswith("sha256:")
    assert planning["enable_web_search"] is False
    assert planning["search_rounds"] == 0
    assert planning["collect_task_feedback"] is False
