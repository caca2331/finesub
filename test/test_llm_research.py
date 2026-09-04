from __future__ import annotations

from dataclasses import replace
import json

import pytest

from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.routing.config import LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.prompts import ContextPack
from finesub.llm.research import (
    check_research_input_limit,
    load_research_context,
    merge_research_general_contexts,
    parse_round1_output,
    parse_round2_output,
    plan_research_chunks,
    render_research_transcript,
    resolve_research_transcript_cap,
    resolve_round1_entries,
    run_research,
)
from finesub.llm.web_search import QuerySearchResult, SearchResultItem, QueryExtractResult


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
        "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )
    (knowledge_root / "common" / "游戏B.md").write_text(
        "# 游戏B\n游戏资料。\n\n## 档案\n本名: 游戏B\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
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
        (target / f"{key}.md").write_text(
            f"# {key}\n内容。\n\n## 档案\n本名: {key}\n\n## 元数据\n最近更新日期: 2026-08-01\n",
            encoding="utf-8",
        )

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
    (streamer / "主播A.md").write_text(
        "# 主播A\n内容。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
    )

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
    assert pack.unbound_window_contexts == {"0001": "背景一"}


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

    # `run_research` returns the model's shape: notes named by window id only.
    # Attaching source ranges is `run_research_stage`'s job (it owns the window
    # list), so this payload is not yet a loadable artifact -- and a file in that
    # shape is refused rather than injected without its window notes.
    context_path = tmp_path / "research-context.json"
    context_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert ContextPack.from_dict(payload["context_pack"]).has_unbound_window_contexts
    with pytest.raises(ValueError, match="no source-id ranges"):
        load_research_context(context_path)

    # The pre-interval on-disk shape is refused the same way -- notably *not*
    # read as "this context simply had no window notes".
    legacy = dict(payload)
    legacy["context_pack"] = {
        "general_context": {"global_summary": "旧"},
        "window_contexts": {"0001": "开场杂谈"},
    }
    context_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="predates source-interval addressing"):
        load_research_context(context_path)

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


def _knowledge_root(tmp_path, *, with_index: bool):
    root = tmp_path / "knowledge"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    if with_index:
        (root / "streamer" / "index.md").write_text(
            "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
        )
        (root / "streamer" / "主播A.md").write_text(
            "# 主播A\n主播资料。\n\n## 档案\n本名: 主播A\n\n## 元数据\n最近更新日期: 2026-08-01\n",
        encoding="utf-8",
        )
        (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


class _RecordingClient:
    """Replies in order and records every (role, messages) it was called with."""

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []

    def complete(self, role, messages, **kwargs):
        self.calls.append((role, messages, kwargs))
        return LLMCallResult(
            content=self.responses.pop(0) if self.responses else "",
            role=role,
            model="fake",
            fallback_used=False,
            raw_response={},
        )


_R1_ENTRIES_ONLY = (
    "<analysis_notes>要点</analysis_notes>\n"
    "<requested_entries>\n主播A\n</requested_entries>\n"
    "<keep_entries></keep_entries>"
)
_R2_PACK = (
    '<context_pack>{"general_context": {"global_summary": "摘要"},'
    ' "window_contexts": []}</context_pack>'
)


def test_retrieval_none_without_an_index_runs_no_research_round(tmp_path) -> None:
    """Neither half applies: no search agent to serve queries, no index to read."""

    client = _RecordingClient([])
    search_client = FakeSearchClient()
    payload = run_research(
        transcript="1|你好\n",
        knowledge_root=_knowledge_root(tmp_path, with_index=False),
        client=client,
        search_client=search_client,
        profile=resolve_profile("text", "none", "quality"),
        token_counter=FakeTokenCounter(),
    )

    assert client.calls == []
    assert search_client.calls == []
    assert payload["rounds"] == {
        "round1": False,
        "round2": False,
        "local_search": False,
        "native_search": False,
        "entries_available": False,
    }
    assert payload["context_pack"]["general_context"] == {}


def test_retrieval_none_with_an_index_runs_r1_only_to_pick_entries(tmp_path) -> None:
    client = _RecordingClient([_R1_ENTRIES_ONLY])
    payload = run_research(
        transcript="1|你好\n",
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "none", "quality"),
        token_counter=FakeTokenCounter(),
    )

    assert len(client.calls) == 1, "R2 has no external input to digest"
    system = client.calls[0][1][0]["content"]
    assert "<requested_entries>" in system
    assert "<search_queries>" not in system, "no local agent would execute them"
    # R2 is skipped, so nothing downstream would ever read analysis_notes.
    assert "<analysis_notes>" not in system
    assert payload["rounds"]["round1"] is True
    assert payload["rounds"]["round2"] is False
    # The picked entries still reach the first correction window.
    assert payload["keep_entries"] == ["主播A"]


def test_retrieval_native_runs_r2_with_the_models_own_search(tmp_path) -> None:
    client = _RecordingClient([_R1_ENTRIES_ONLY, _R2_PACK])
    search_client = FakeSearchClient()
    payload = run_research(
        transcript="1|你好\n",
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        client=client,
        search_client=search_client,
        profile=resolve_profile("text", "native", "quality"),
        token_counter=FakeTokenCounter(),
    )

    assert len(client.calls) == 2
    assert search_client.calls == [], "native search never uses the local agent"
    assert client.calls[1][2]["retrieval"] == "native"
    r2_system = client.calls[1][1][0]["content"]
    assert "你具备联网搜索能力" in r2_system
    assert "<search_results>" not in client.calls[1][1][1]["content"]
    # No query round downstream, so R2 is not asked to prune.
    assert "<keep_entries>" not in r2_system
    assert payload["keep_entries"] == ["主播A"]
    assert payload["context_pack"]["general_context"]["global_summary"] == "摘要"


def test_r1_gets_the_whole_entry_budget_when_it_is_the_only_pick(tmp_path) -> None:
    """No per-window query round means the increment-sized reserve is pure loss."""

    knowledge_root = _knowledge_root(tmp_path, with_index=True)
    local = _RecordingClient(
        [_R1_ENTRIES_ONLY + "\n<search_queries></search_queries>", _R2_PACK]
    )
    run_research(
        transcript="1|你好\n",
        knowledge_root=knowledge_root,
        client=local,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        search_rounds=1,
        token_counter=FakeTokenCounter(),
    )
    none = _RecordingClient([_R1_ENTRIES_ONLY])
    run_research(
        transcript="1|你好\n",
        knowledge_root=knowledge_root,
        client=none,
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "none", "quality"),
        token_counter=FakeTokenCounter(),
    )

    assert "单独上限 8 条" in local.calls[0][1][0]["content"]
    assert "单独上限 12 条" in none.calls[0][1][0]["content"]


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
        search_rounds=1,
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
    # Thinking comes from the preset knob via the search_judge cell config
    # now, not per-call kwargs.
    assert "thinking_level" not in seen[1][2]
    from finesub.llm.routing.config import role_config_for

    assert role_config_for("search_judge", "quality").thinking_level == "medium"
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

    monkeypatch.setattr("finesub.llm.content_filter.time.sleep", lambda _s: None)
    artifacts = tmp_path / "artifacts"
    run_research(
        transcript="1|hello",
        extra_info=(
            "链接: https://example.test/good 与 https://example.test/bad-page"
        ),
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=ExtractClient(),
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
    from finesub.llm.content_filter import BLACKLIST_ARTIFACT_KIND, LADDER_ARTIFACT_KIND

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
    assert payload["context_pack"] == {"general_context": {}, "window_contexts": []}


def test_run_research_stage_writes_planning_metadata(tmp_path, monkeypatch) -> None:
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.prompts import PROMPT_VERSION
    from finesub.llm.routing.config import WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    from finesub.llm.research import run_research_stage

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
        '<context_pack>{"general_context": {}, "window_contexts": '
        '[{"window_id":"0001","context":"背景"}]}</context_pack>',
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

    monkeypatch.setattr("finesub.llm.research.RoleClient", FakeClient)

    context_path = tmp_path / "clip-research-context.json"
    run_research_stage(
        stable_json=stable_json,
        context_path=context_path,
        knowledge_root=knowledge_root,
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "local", "quality"),
    )

    payload = json.loads(context_path.read_text(encoding="utf-8"))
    planning = payload["planning"]
    assert planning["prompt_version"] == PROMPT_VERSION
    assert planning["context_reserve_tokens"] == WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS
    assert planning["profile_id"] == (
        "correction_media=audio,planning_media=audio,retrieval=local,"
        "difficulty=quality,continuity=serial"
    )
    assert planning["output_scale"] == 1.0
    assert planning["stable_json_hash"].startswith("sha256:")
    assert planning["extra_info_hash"].startswith("sha256:")
    assert planning["knowledge_inputs_hash"].startswith("sha256:")
    assert planning["knowledge_enabled"] is True
    assert planning["search_rounds"] == 1
    assert planning["collect_task_feedback"] is False
    assert payload["context_pack"]["window_contexts"] == [
        {
            "window_id": "0001",
            "first_source_id": "1",
            "last_source_id": "1",
            "context": "背景",
        }
    ]


def test_knowledge_tri_state_remains_in_research_audit_metadata(tmp_path) -> None:
    """Generation settings remain observable even though they do not gate reuse."""

    from finesub.llm.research import planning_metadata

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    knowledge_root = _knowledge_root(tmp_path, with_index=True)
    common = dict(
        stable_json=stable,
        knowledge_root=knowledge_root,
        extra_info="",
        search_rounds=1,
        max_window_subtitle_tokens=None,
    )
    off = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=False,
        collect_task_feedback=False,
        **common,
    )
    on = planning_metadata(
        resolve_profile("text", "none", "quality"),
        knowledge_enabled=True,
        collect_task_feedback=True,
        **common,
    )

    assert off != on
    assert off["knowledge_enabled"] is False and on["knowledge_enabled"] is True
    assert off["knowledge_inputs_hash"] == ""
    assert on["knowledge_inputs_hash"].startswith("sha256:")


def test_difficulty_is_normalized_in_research_audit_metadata(tmp_path) -> None:
    """Difficulty is not a research input; other generation axes stay auditable."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        search_rounds=1,
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    high = planning_metadata(resolve_profile("audio", "local", "quality"), **common)
    med = planning_metadata(resolve_profile("audio", "local", "intermediate"), **common)
    other_axis = planning_metadata(
        resolve_profile("text", "local", "quality"), **common
    )

    # The full vector stays recorded, so a mixed-difficulty run is auditable...
    assert high["profile_id"] != med["profile_id"]
    # ...but difficulty is not a reuse key: it picks the correction prompt
    # variant and the thinking knob, neither of which the research rounds read.
    assert research_reuse_key(high) == research_reuse_key(med)
    # Correction media changes geometry, not what research means.
    assert research_reuse_key(other_axis) == research_reuse_key(high)


def test_every_planning_field_is_classified_as_gate_or_audit(tmp_path) -> None:
    """L3's answer to `_task_fingerprint`'s include-list assertion.

    `research_reuse_key` picks named fields, so a field added to
    `planning_metadata` without a decision would be silently ignored -- an old
    context reused past something that should have invalidated it. Classifying
    is the whole point of the whitelist; forgetting must fail here, loudly,
    instead of in someone's subtitles.
    """

    from finesub.llm.research import (
        RESEARCH_REUSE_AUDIT_ONLY,
        RESEARCH_REUSE_GATES,
        planning_metadata,
        research_reuse_key,
    )

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    planning = planning_metadata(
        resolve_profile("audio", "local", "quality"),
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        search_rounds=2,
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )

    classified = set(RESEARCH_REUSE_GATES) | set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(planning) == classified
    assert not set(RESEARCH_REUSE_GATES) & set(RESEARCH_REUSE_AUDIT_ONLY)
    assert set(research_reuse_key(planning)) == set(RESEARCH_REUSE_GATES)


def test_retrieval_mode_gates_research_reuse_even_when_rounds_match(
    tmp_path, monkeypatch
) -> None:
    """`none` and `native` both leave search_rounds at 0, so the narrow
    `research_semantics_id` must distinguish them after geometry is removed."""

    from finesub.llm.research import planning_metadata, research_reuse_key

    stable = tmp_path / "stable.json"
    stable.write_text('{"segments": []}', encoding="utf-8")
    common = dict(
        stable_json=stable,
        knowledge_root=_knowledge_root(tmp_path, with_index=True),
        extra_info="",
        search_rounds=2,
        max_window_subtitle_tokens=None,
        knowledge_enabled=True,
        collect_task_feedback=False,
    )
    none = planning_metadata(resolve_profile("audio", "none", "quality"), **common)
    native = planning_metadata(resolve_profile("audio", "native", "quality"), **common)
    local = planning_metadata(resolve_profile("audio", "local", "quality"), **common)

    # The cheap discriminator is genuinely blind here...
    assert none["search_rounds"] == native["search_rounds"] == 0
    assert local["search_rounds"] == 2
    # ...so the reuse key has to carry the mode itself.
    assert research_reuse_key(none) != research_reuse_key(native)
    assert research_reuse_key(none) != research_reuse_key(local)


def test_planning_media_never_reaches_the_research_rounds(tmp_path) -> None:
    """`planning_media` is the *query* round's clip, not the research round's.

    It does not belong in the L3 key: `build_research_round1_messages` takes no
    profile at all, research R1/R2 pass no `file_ref` (fast round 1 is the only
    caller that does, and it carries the *correction* clip), `planning_task_group`
    is referenced only from `stages/correction`, and `research_stage_runs`
    reads `retrieval` alone. L3 therefore uses `research_semantics_id=retrieval`.
    """

    import inspect

    from finesub.llm import research as research_module
    from finesub.llm.prompts import build_research_round1_messages

    assert "profile" not in inspect.signature(build_research_round1_messages).parameters
    source = inspect.getsource(research_module)
    # The one place research.py forwards a media handle is the shared round
    # runner's own signature/pass-through, never a research call site.
    assert source.count("file_ref=file_ref") == 1

    knowledge_root = _knowledge_root(tmp_path, with_index=True)
    profile = resolve_profile("audio", "local", "quality")
    assert research_module.research_stage_runs(
        profile, knowledge_root=knowledge_root
    ) is research_module.research_stage_runs(
        replace(profile, planning_media="text"), knowledge_root=knowledge_root
    )


def test_native_sources_are_flagged_when_the_backend_reports_no_urls() -> None:
    """A `sources` list that nothing can check must not read like a citation.

    The native prompt asks the model to cite what it verified. On a backend
    that reports no result URLs the reply comes back with site-level "sources"
    the harness cannot corroborate -- and in the 2026-08-15 A/B the one
    checkable claim carried that way was a wrong official title.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "global_summary": "x",
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}],
        }
    )
    agy = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "search_web", "query": "q", "urls": []}],
        }
    ]

    marked = _mark_unverified_sources(pack, agy)

    assert marked.general_context["sources"][0]["verified"] is False
    assert "无法核对" in marked.general_context["sources_note"]


def test_native_sources_are_left_alone_when_urls_were_recorded() -> None:
    """Codex/Claude keep a per-call URL ledger, so there is something to check."""

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "local_agent",
            "search_events": [{"tool": "web_search", "urls": ["https://example.org/a"]}],
        }
    ]

    marked = _mark_unverified_sources(pack, grounded)

    assert marked is pack
    assert "verified" not in marked.general_context["sources"][0]
    assert "sources_note" not in marked.general_context


def test_marking_skips_rounds_that_are_not_context_packs() -> None:
    from finesub.llm.research import _mark_unverified_sources

    round1 = parse_round1_output(
        "<requested_entries></requested_entries>"
        "<keep_entries></keep_entries>"
        "<search_queries></search_queries>"
    )
    assert _mark_unverified_sources(round1, []) is round1


def test_gemini_native_sources_are_not_stamped_unverified() -> None:
    """The shipped default's native path is Gemini `google_search`, not an agent.

    Since the client parses `groundingMetadata`, this path exits at the URL
    check with something real to show. The backend list still has to hold: an
    older Gemini reply with no grounding row must not be stamped either --
    saying "the backend reported nothing" about a provider that does report is
    untrue. `test_llm_switch_matrix` asserts that route exists.
    """

    from finesub.llm.research import _mark_unverified_sources

    pack = ContextPack(
        general_context={
            "sources": [{"title": "Some Wiki", "url": "https://example.org"}]
        }
    )
    grounded = [
        {
            "backend": "gemini_rest",
            "search_events": [
                {
                    "tool": "google_search",
                    "queries": ["q"],
                    "urls": ["https://example.org/a"],
                }
            ],
        }
    ]
    assert _mark_unverified_sources(pack, grounded) is pack

    bare = [{"backend": "gemini_rest", "return_code": 200}]
    marked = _mark_unverified_sources(pack, bare)

    assert marked is pack
    assert "sources_note" not in marked.general_context


# --- chunked research for ultra-long material (plan W7 / P8) -----------------


def _many_windows():
    segments = [
        SubtitleSegment(str(n), float(n), float(n) + 0.8, f"第{n}句的内容文本。")
        for n in range(1, 13)
    ]
    windows = plan_correction_windows(
        segments,
        counter=FakeTokenCounter(),
        max_window_subtitle_tokens=30,
    )
    assert len(windows) >= 3, "fixture must yield several research windows"
    return segments, windows


def test_plan_research_chunks_packs_whole_windows() -> None:
    segments, windows = _many_windows()
    counter = FakeTokenCounter()
    chunks = plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=60
    )
    assert len(chunks) >= 2
    # Every window lands in exactly one chunk, order preserved, never split.
    flattened = [window.chunk_id for chunk in chunks for window in chunk]
    assert flattened == [window.chunk_id for window in windows]
    # 0 disables chunking outright.
    assert plan_research_chunks(
        windows, count_tokens=counter.count_text, max_tokens=0
    ) == [list(windows)]
    # A cap smaller than any single window still never splits one.
    tiny = plan_research_chunks(windows, count_tokens=counter.count_text, max_tokens=1)
    assert [len(chunk) for chunk in tiny] == [1] * len(windows)


def test_merged_general_context_is_deterministic_and_capped() -> None:
    counter = FakeTokenCounter()
    merged = merge_research_general_contexts(
        {"c02": {"b": "2"}, "c01": {"a": "1"}}, count_tokens=counter.count_text
    )
    assert list(merged) == ["合并说明", "c01", "c02"]  # instruction first, stable order
    assert "证据更充分" in merged["合并说明"]  # the conflict rule rides the pack
    assert merge_research_general_contexts({}, count_tokens=counter.count_text) == {}

    # The cap drops whole trailing chunk blocks, never truncates mid-JSON.
    big = {"填充": "x" * 400}
    capped = merge_research_general_contexts(
        {"c01": big, "c02": big, "c03": big},
        count_tokens=counter.count_text,
        max_tokens=counter.count_text(json.dumps({"合并说明": "x" * 200, "c01": big}, ensure_ascii=False)),
    )
    kept = [key for key in capped if key != "合并说明"]
    assert kept and kept == sorted(kept) and len(kept) < 3
    assert kept[0] == "c01"  # earlier material survives


def test_research_transcript_cap_reads_config(monkeypatch) -> None:
    from finesub.llm.routing import api_keys

    monkeypatch.setattr(api_keys, "read_config", lambda: {})
    assert resolve_research_transcript_cap() == 100_000  # default ON (plan W7)
    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda: {"chunking": {"research_transcript_max_tokens": 0}},
    )
    assert resolve_research_transcript_cap() == 0
    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda: {"chunking": {"research_transcript_max_tokens": 50_000}},
    )
    assert resolve_research_transcript_cap() == 50_000
    monkeypatch.setattr(
        api_keys,
        "read_config",
        lambda: {"chunking": {"research_transcript_max_tokens": -1}},
    )
    with pytest.raises(ValueError, match="research_transcript_max_tokens"):
        resolve_research_transcript_cap()


def test_chunked_research_merges_packs_and_replays_finished_chunks(
    tmp_path, monkeypatch
) -> None:
    """Plan W7: every chunk runs the full rounds over its own slice, notes
    bind to disjoint source intervals, keep entries union in chunk order, and
    a rerun replays finished chunks off their per-chunk files."""

    import re

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import DEFAULT_PROFILE

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {
                "general_context": {"专名": f"来自{chunk_label}"},
                "window_contexts": [
                    {"window_id": window_id, "context": f"note-{chunk_label}"}
                ],
            },
            "keep_entries": ["共有词条", f"词条{chunk_label}"],
            "rounds": {"round1": True, "round2": True},
            "injected_entries": [f"词条{chunk_label}"],
            "token_report": {"totals": {"total_tokens": 10}},
        }

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    kwargs = dict(
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=DEFAULT_PROFILE,
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
    )
    payload = research_mod._run_chunked_research(chunk_plan, **kwargs)

    assert calls == ["c01", "c02", "c03"]
    pack = ContextPack.from_dict(payload["context_pack"])
    assert len(pack.window_contexts) == 3
    assert all(note.first_source_id and note.last_source_id for note in pack.window_contexts)
    assert list(pack.general_context) == ["合并说明", "c01", "c02", "c03"]
    # Union in chunk order, deduplicated.
    assert payload["keep_entries"][:2] == ["共有词条", "词条c01"]
    assert len(payload["keep_entries"]) == len(set(payload["keep_entries"]))
    for record in payload["research_chunks"]:
        assert (tmp_path / record["file"]).exists()
        assert record["replayed"] is False
    assert payload["token_report"]["totals"]["total_tokens"] == 30

    # A rerun must replay every finished chunk without a single LLM call.
    def exploding_run_research(**_kwargs):
        raise AssertionError("a finished chunk was re-run")

    monkeypatch.setattr(research_mod, "run_research", exploding_run_research)
    replayed = research_mod._run_chunked_research(chunk_plan, **kwargs)
    assert [record["replayed"] for record in replayed["research_chunks"]] == [True] * 3
    assert replayed["keep_entries"] == payload["keep_entries"]
    assert replayed["context_pack"] == payload["context_pack"]


def test_chunked_research_fans_out_under_parallel_continuity(tmp_path, monkeypatch) -> None:
    """W7 follow-through: chunks are independent, so `continuity=parallel`
    dispatches them concurrently -- one chunk rides one worker, the shared
    writers (exchange logger / checkpoint store / search client) are single
    instances, and the merge stays index-ordered whatever finishes first."""

    import re
    import threading

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    seen: dict[str, dict] = {}
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=10)
    overlapped = threading.Event()

    def fake_run_research(*, transcript, chunk_label, exchange_logger,
                          checkpoint_store, **kwargs):
        if not overlapped.is_set():
            try:
                barrier.wait()  # proves at least two chunks run concurrently
                overlapped.set()
            except threading.BrokenBarrierError:
                pass
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        with lock:
            seen[chunk_label] = {
                "exchange_logger": exchange_logger,
                "checkpoint_store": checkpoint_store,
            }
        return {
            "context_pack": {
                "general_context": {"专名": f"来自{chunk_label}"},
                "window_contexts": [
                    {"window_id": window_id, "context": f"note-{chunk_label}"}
                ],
            },
            "keep_entries": [f"词条{chunk_label}"],
            "rounds": {"round1": True},
            "injected_entries": [],
            "token_report": {"totals": {"total_tokens": 1}},
        }

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    payload = research_mod._run_chunked_research(
        chunk_plan,
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=resolve_profile(continuity="parallel"),
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
        parallel_windows=2,
    )
    assert sorted(seen) == ["c01", "c02", "c03"]
    assert overlapped.is_set()  # the barrier really saw two chunks in flight
    # One shared instance each, whichever worker served the chunk.
    loggers = {id(entry["exchange_logger"]) for entry in seen.values()}
    stores = {id(entry["checkpoint_store"]) for entry in seen.values()}
    assert len(loggers) == 1 and len(stores) == 1
    # Assembly is index-ordered regardless of completion order.
    assert [r["chunk_id"] for r in payload["research_chunks"]] == ["c01", "c02", "c03"]
    assert payload["keep_entries"] == ["词条c01", "词条c02", "词条c03"]


def test_chunked_research_stays_serial_on_a_conversational_backend(tmp_path, monkeypatch) -> None:
    """Plan W6, research face: a person's own agent has ONE queue, so fanning
    chunks out just stacks N assignments behind it. The correction stage forces
    serial for this reason but only for its own role and only later, which left
    this phase dispatching against the same single agent."""

    import threading

    import finesub.llm.research as research_mod
    from finesub.reporting import NullReporter, reporting_to
    from finesub.llm.routing.profiles import resolve_profile

    class ConversationalClient:
        asked: list = []

        def routes_to_conversational(self, role, task_group="", difficulty=""):
            ConversationalClient.asked.append((role, task_group))
            return True

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:3]]
    active = 0
    peak = 0
    lock = threading.Lock()
    warnings: list[str] = []

    def fake_run_research(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            active -= 1
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    class _Reporter(NullReporter):
        def warning(self, code, message, *, impact="", action="") -> None:
            warnings.append(code)

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    with reporting_to(_Reporter()):
        research_mod._run_chunked_research(
            chunk_plan,
            segments=segments,
            context_file=tmp_path / "research-context.json",
            extra_info="",
            knowledge_root=tmp_path,
            knowledge_enabled=False,
            profile=resolve_profile(continuity="parallel"),
            test_profile=True,
            task_artifact_dir=None,
            task_id="t",
            search_rounds=1,
            token_counter=FakeTokenCounter(),
            collect_task_feedback=False,
            resume=True,
            client=ConversationalClient(),
            transcript_cap=1000,
            parallel_windows=4,
        )
    assert peak == 1
    assert warnings == ["conversational-forced-serial"]
    # asked with the research rounds' own cell, not the correction one
    assert ConversationalClient.asked and ConversationalClient.asked[0][1] == "research"


def test_chunked_research_stays_serial_without_parallel_continuity(tmp_path, monkeypatch) -> None:
    """Default runs (`continuity=serial`) keep chunk execution serial: the
    fan-out follows the task's existing intra-task opt-in, not a new knob."""

    import threading

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[window] for window in windows[:2]]
    active = 0
    peak = 0
    lock = threading.Lock()

    def fake_run_research(**kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        with lock:
            active -= 1
        return {
            "context_pack": {"general_context": {}, "window_contexts": []},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    research_mod._run_chunked_research(
        chunk_plan,
        segments=segments,
        context_file=tmp_path / "research-context.json",
        extra_info="",
        knowledge_root=tmp_path,
        knowledge_enabled=False,
        profile=resolve_profile(continuity="serial"),
        test_profile=True,
        task_artifact_dir=None,
        task_id="t",
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        collect_task_feedback=False,
        resume=True,
        client=object(),
        transcript_cap=1000,
        parallel_windows=4,
    )
    assert peak == 1


def test_chunk_resume_honours_the_stage_level_reuse_gates(tmp_path, monkeypatch) -> None:
    """Reviewer 2026-08-30 P1-2: the chunk files are an L3-adjacent cache, so
    they obey the same gates -- prompt_version or test_profile changing must
    re-run the chunk; difficulty changing must NOT (it is deliberately exempt:
    switching to intermediate exists to stop burning quota, not to redo it)."""

    import re
    from dataclasses import replace as dc_replace

    import finesub.llm.research as research_mod
    from finesub.llm.routing.profiles import resolve_profile

    segments, windows = _many_windows()
    chunk_plan = [[windows[0]]]
    calls: list[str] = []

    def fake_run_research(*, transcript, chunk_label, **kwargs):
        calls.append(chunk_label)
        window_id = re.search(r"--- window (\S+) ---", transcript).group(1)
        return {
            "context_pack": {"general_context": {},
                             "window_contexts": [{"window_id": window_id, "context": "n"}]},
            "keep_entries": [],
            "rounds": {},
            "injected_entries": [],
            "token_report": {"totals": {}},
        }

    monkeypatch.setattr(research_mod, "run_research", fake_run_research)
    profile = resolve_profile()

    def run(**overrides):
        kwargs = dict(
            segments=segments,
            context_file=tmp_path / "research-context.json",
            extra_info="",
            knowledge_root=tmp_path,
            knowledge_enabled=False,
            profile=profile,
            test_profile=True,
            task_artifact_dir=None,
            task_id="t",
            search_rounds=1,
            token_counter=FakeTokenCounter(),
            collect_task_feedback=False,
            resume=True,
            client=object(),
            transcript_cap=1000,
        )
        kwargs.update(overrides)
        return research_mod._run_chunked_research(chunk_plan, **kwargs)

    run()
    assert calls == ["c01"]
    run()  # unchanged inputs: replayed off the chunk file
    assert calls == ["c01"]
    run(test_profile=False)  # a smoke chunk must never serve a real run
    assert calls == ["c01", "c01"]
    monkeypatch.setattr(research_mod, "PROMPT_VERSION", "test-bumped-version")
    run(test_profile=False)  # a prompt upgrade re-runs the chunk
    assert calls == ["c01", "c01", "c01"]
    # Difficulty is exempt from the gates: the finished chunk replays.
    run(test_profile=False, profile=dc_replace(profile, difficulty="intermediate"))
    assert calls == ["c01", "c01", "c01"]
