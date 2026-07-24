from __future__ import annotations

import json

import pytest

from llm.chunking import SubtitleSegment
from llm.client import LLMCallResult
from llm.config import CapabilityTier, LLMRole
from llm.correction_translation import _fast_execute_kwargs
from llm.stages.correction_loop import execute_correction_windows
from llm.profiles import resolve_profile, window_output_budget
from llm.srt_utils import parse_srt
from llm.stages.fast_session import (
    FastSessionResult,
    acquire_fast_context,
    load_fast_context,
    parse_fast_round1_output,
    run_fast_session,
)
from llm.stages.correction_loop import QueryRoundProduct
from llm.stages.plan import FAST_WINDOW_CHUNK_ID, FastDecision, decide_fast_mode, plan_fast_window


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


class HugePromptCounter(FakeTokenCounter):
    """Prompt-level counts blow the input budget; per-text counts stay tiny."""

    def count_texts(self, texts) -> int:
        list(texts)
        return 300_000


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "一。"),
        SubtitleSegment("2", 100.0, 102.5, "二。"),
    ]


def _stable_json(tmp_path):
    path = tmp_path / "clip-stable.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 100.0, "end": 102.5, "text": "二。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plan_fast_window_covers_everything_with_edge_pads() -> None:
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("mm", "med")
    )

    assert window.chunk_id == FAST_WINDOW_CHUNK_ID
    assert [segment.id for segment in window.segments] == ["1", "2"]
    assert window.overlap_segments == []
    assert window.boundary_reason == "fast_single_window"
    # Both edges get the 60s global pad; start clamps to 0.
    assert window.clip_start == 0.0
    assert window.clip_end == 162.5

    clamped = plan_fast_window(
        _segments(),
        audio_duration=120.0,
        counter=FakeTokenCounter(),
        profile=resolve_profile("mm", "med"),
    )
    assert clamped.clip_end == 120.0


def test_fast_round1_resumes_validated_session(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "low")
    )
    response = (
        "<reasoning>分析完整窗口。</reasoning>\n"
        "<analysis_notes>简短直播片段。</analysis_notes>\n"
        "<requested_entries></requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries></search_queries>"
    )

    class FirstClient:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=response,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    class BlockingClient:
        def complete(self, role, messages, **kwargs):
            raise AssertionError("validated fast round 1 should resume")

    kwargs = dict(
        window=window,
        segment_count=len(_segments()),
        knowledge_root=knowledge_root,
        enable_web_search=False,
        task_artifact_dir=artifact_dir,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "low"),
    )
    first, _ = run_fast_session(client=FirstClient(), **kwargs)
    resumed, _ = run_fast_session(client=BlockingClient(), **kwargs)

    assert resumed.payload["fast"] == first.payload["fast"]
    assert resumed.payload["context_pack"] == first.payload["context_pack"]
    assert resumed.payload["token_report"]["totals"]["call_count"] == 0
    assert (artifact_dir / "fast-round-input.json").exists()
    records = [
        json.loads(line)
        for line in (artifact_dir / "session-checkpoints.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["session"] for record in records] == ["fast-round1"]


def test_decide_fast_mode_auto_enables_small_input(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("mm", "med"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
    )

    assert decision.enabled
    assert decision.reason == "fits fast budgets"
    assert decision.window is not None
    assert decision.output_budget == window_output_budget(fast=True)
    assert 0 < decision.expected_output_tokens <= decision.output_budget
    assert 0 < decision.round1_input_tokens <= decision.input_budget
    assert decision.to_metadata()["enabled"] is True


def test_decide_fast_mode_off_and_bad_mode(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="off",
        token_counter=FakeTokenCounter(),
    )
    assert not decision.enabled
    assert decision.window is None

    with pytest.raises(ValueError, match="auto/on/off"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="fastest",
            token_counter=FakeTokenCounter(),
        )


def test_decide_fast_mode_auto_falls_back_when_output_over_budget(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("mm", "med", output_scale=10_000),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
    )

    assert not decision.enabled
    assert "expected output" in decision.reason


def test_decide_fast_mode_on_raises_when_over_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="--fast on"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="on",
            profile=resolve_profile("mm", "med", output_scale=10_000),
            knowledge_root=tmp_path / "kb",
            token_counter=FakeTokenCounter(),
        )


def test_decide_fast_mode_checks_round1_input_reserve(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("mm", "med"),
        knowledge_root=tmp_path / "kb",
        token_counter=HugePromptCounter(),
    )

    assert not decision.enabled
    assert "round-1 input" in decision.reason


def test_parse_fast_round1_output_uses_wider_notes_cap() -> None:
    notes = "笔" * 1_800  # over the research round-1 cap (1500), under the fast cap
    text = (
        f"<analysis_notes>\n{notes}\n</analysis_notes>\n"
        "<research_contract>\n目标: 修正字幕\nF1|P1|主播常用语\n</research_contract>\n"
        "<requested_entries>\n条目甲\n</requested_entries>\n"
        "<keep_entries>\n条目乙\n</keep_entries>\n"
        "<search_queries>\n游戏A 剧情\n</search_queries>"
    )

    result = parse_fast_round1_output(text, expect_contract=True)

    assert result.analysis_notes == notes
    assert result.requested_entries == ("条目甲",)
    assert result.keep_entries == ("条目乙",)
    assert result.search_queries == ("游戏A 剧情",)
    assert "F1|P1" in result.research_contract

    no_contract = parse_fast_round1_output(text, expect_contract=False)
    assert no_contract.research_contract == ""


def test_fast_session_result_seeds_single_window_and_fingerprints() -> None:
    result = FastSessionResult(
        analysis_notes="笔记", search_results_text="结果", entry_details_text="条目"
    )

    assert result.seed_query_results() == {
        FAST_WINDOW_CHUNK_ID: QueryRoundProduct(
            search_results="结果", window_notes="笔记"
        )
    }
    assert result.fingerprint().startswith("fast:")
    changed = FastSessionResult(
        analysis_notes="笔记", search_results_text="别的", entry_details_text="条目"
    )
    assert changed.fingerprint() != result.fingerprint()


def test_load_fast_context_rejects_normal_research_context(tmp_path) -> None:
    path = tmp_path / "clip-research-context.json"
    path.write_text(
        json.dumps({"context_pack": {"general_context": {}, "window_contexts": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not a fast-mode context"):
        load_fast_context(path)


def test_fast_execute_kwargs_by_route() -> None:
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("mm", "low")
    )
    decision = FastDecision(mode="auto", enabled=True, reason="", window=window)
    ctx = FastSessionResult(
        analysis_notes="笔记",
        entry_details_text="条目",
        search_results_text="结果",
        evidence_pack_mode=True,
    )
    file_ref = object()

    mm_kwargs = _fast_execute_kwargs(decision, ctx, file_ref, resolve_profile("mm", "med"))
    assert mm_kwargs["windows_override"] == [window]
    assert mm_kwargs["seed_query_results"] == ctx.seed_query_results()
    assert mm_kwargs["entry_details"] == "条目"
    assert mm_kwargs["evidence_pack_mode"] is True
    assert mm_kwargs["extra_fingerprint"] == ctx.fingerprint()
    assert mm_kwargs["file_ref_seed"] == {window.chunk_id: file_ref}

    # The text route seeds nothing beyond the single window (no injections).
    text_kwargs = _fast_execute_kwargs(decision, None, None, resolve_profile("text", "med"))
    assert text_kwargs == {"windows_override": [window]}


def test_fast_seeds_replace_query_round_in_correction_loop(tmp_path, monkeypatch) -> None:
    profile = resolve_profile("mm", "low")
    stable_json = _stable_json(tmp_path)
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=profile
    )
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
            calls.append((role, messages))
            return LLMCallResult(
                content=(
                    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|one|一|8|译1字；宜保持独立\n"
                    "sub|2|1.0|two|二|8|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|one|一|8|\nsub|2|1.0|two|二|8|\n</translated>"
                    "\n<next_advice></next_advice>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class ExplodingSearchClient:
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            raise AssertionError("fast seeds must not construct a search client")

    monkeypatch.setattr("llm.stages.correction_loop.LiteLLMRoleClient", FakeClient)
    monkeypatch.setattr("llm.stages.correction_loop.WebSearchClient", ExplodingSearchClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        enable_web_search=True,
        profile=profile,
        windows_override=[window],
        seed_query_results={
            FAST_WINDOW_CHUNK_ID: QueryRoundProduct(
                search_results="假搜索结果内容", window_notes="快速分析笔记"
            )
        },
        entry_details="## 条目甲\n\n条目甲的内容",
        evidence_pack_mode=True,
    )

    # Exactly one correction call: no query round, seeds injected verbatim.
    assert len(calls) == 1
    role, messages = calls[0]
    assert role is LLMRole.AUDIO_MULTIMODAL
    prompt_text = "\n".join(str(message.get("content", "")) for message in messages)
    assert "假搜索结果内容" in prompt_text
    assert "快速分析笔记" in prompt_text
    assert "条目甲的内容" in prompt_text
    assert [segment.text for segment in parse_srt(output.read_text(encoding="utf-8"))] == [
        "一",
        "二",
    ]


def test_acquire_fast_context_persists_and_reuses(tmp_path) -> None:
    profile = resolve_profile("mm", "low")
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=profile
    )
    round1_calls = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
            round1_calls.append(role)
            return LLMCallResult(
                content=(
                    "<analysis_notes>\n主播在玩游戏A。\n</analysis_notes>\n"
                    "<requested_entries>\n</requested_entries>\n"
                    "<keep_entries>\n</keep_entries>\n"
                    "<search_queries>\n游戏A 剧情\n</search_queries>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            return []

    context_path = tmp_path / "clip-research-context.json"
    session_kwargs = dict(
        window=window,
        segment_count=2,
        knowledge_root=tmp_path / "kb",
        client=FakeClient(),
        search_client=FakeSearchClient(),
        enable_web_search=True,
        search_rounds=1,
        token_counter=FakeTokenCounter(),
        profile=profile,
    )

    result, file_ref, reused = acquire_fast_context(
        context_path=context_path, **session_kwargs
    )

    # mm-low has no audio: round 1 ran on general_capable, nothing uploaded.
    assert round1_calls == [LLMRole.GENERAL_CAPABLE]
    assert file_ref is None
    assert not reused
    assert result.analysis_notes == "主播在玩游戏A。"
    saved = json.loads(context_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "fast"
    assert saved["fast"]["search_queries"] == ["游戏A 剧情"]

    reloaded, ref2, reused2 = acquire_fast_context(
        context_path=context_path, **session_kwargs
    )
    assert reused2 and ref2 is None
    assert reloaded.analysis_notes == result.analysis_notes
    assert reloaded.fingerprint() == result.fingerprint()
    # Reuse never re-calls the model.
    assert round1_calls == [LLMRole.GENERAL_CAPABLE]

    changed_kwargs = {**session_kwargs, "extra_info": "新的任务备注"}
    _changed, ref3, reused3 = acquire_fast_context(
        context_path=context_path, **changed_kwargs
    )
    assert not reused3 and ref3 is None
    assert round1_calls == [LLMRole.GENERAL_CAPABLE, LLMRole.GENERAL_CAPABLE]


def test_fast_r1_keep_entry_is_injected_into_search_loop_and_correction(tmp_path) -> None:
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
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("mm", "low")
    )
    contract = json.dumps(
        {
            "goal": "查证游戏A",
            "facts": [],
            "out_of_scope": [],
        },
        ensure_ascii=False,
    )
    responses = [
        (
            "<analysis_notes>主播A在玩游戏A（待定）。</analysis_notes>\n"
            f"<research_contract>{contract}</research_contract>\n"
            "<requested_entries></requested_entries>\n"
            "<keep_entries>エーちゃん</keep_entries>\n"
            "<search_queries>游戏A 剧情</search_queries>"
        ),
        (
            "<progress_update>F1: partial 搜索摘要</progress_update>\n"
            "<evidence_pack>## 结论\n部分确认\n## 关键证据摘录\n"
            "- 摘要\n## 未解决\n- [unresolved] 无</evidence_pack>"
        ),
    ]
    seen_messages = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
            seen_messages.append(messages)
            return LLMCallResult(
                content=responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            return []

        def extract_many(self, requests, *, max_urls=None):
            return []

    result, _file_ref = run_fast_session(
        window=window,
        segment_count=2,
        extra_info="今天是エーちゃん的直播",
        knowledge_root=knowledge_root,
        client=FakeClient(),
        search_client=FakeSearchClient(),
        enable_web_search=True,
        search_rounds=2,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("mm", "low"),
    )

    loop_user = seen_messages[1][1]["content"]
    assert "<previous_kept_entries>\nエーちゃん" in loop_user
    assert "关西腔。" in loop_user
    assert "关西腔。" in result.entry_details_text
    assert result.payload["fast"]["keep_entries"] == ["エーちゃん"]
    assert result.payload["injected_entries"] == ["主播A"]
