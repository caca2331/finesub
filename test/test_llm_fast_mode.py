from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.chunking import SubtitleSegment
from finesub.llm.client import LLMCallResult, RoleClient
from finesub.llm.routing.config import CapabilityTier, LLMRole
from finesub.llm.correction_translation import _fast_execute_kwargs
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.routing.profiles import resolve_profile, window_output_budget
from finesub.subtitles.model import parse_srt
from finesub.llm.stages.fast_session import (
    FastSessionResult,
    acquire_fast_context,
    load_fast_context,
    parse_fast_round1_output,
    run_fast_session,
)
from finesub.llm.stages.correction import QueryRoundProduct
from finesub.llm.stages.plan import FAST_WINDOW_CHUNK_ID, FastDecision, decide_fast_mode, plan_fast_window
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.routing.model_router import ModelRouter
from finesub.llm.rate_limit import ModelRateLimiter


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
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("audio", "local", "quality")
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
        profile=resolve_profile("audio", "local", "quality"),
    )
    assert clamped.clip_end == 120.0


def test_agent_only_fast_media_uses_local_ref_without_gemini_upload(
    tmp_path, monkeypatch
) -> None:
    profile = resolve_profile("audio", "none", "quality")
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=profile
    )
    # The policy alone no longer reaches an agent -- it only subtracts
    # backends. `agy` is the packaged preset whose groups name one, and it goes
    # into the *router*: how the clip is carried is decided from the same
    # catalog that decides who answers, not from the global one.
    from finesub.llm.routing import model_routes

    routes = model_routes.load_model_routes(user_config={"preset": "agy"})
    settings = ExecutionSettings(policy_id="agent-only")
    client = RoleClient(
        router=ModelRouter(routes, policy_id="agent-only"),
        execution_settings=settings,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    uploads = []
    monkeypatch.setattr(
        "finesub.llm.client.upload_gemini_file",
        lambda path: uploads.append(path),
    )
    monkeypatch.setattr(client, "ensure_eligible_target", lambda *a, **k: None)

    def fake_extract(_source, _start, _end, output):
        Path(output).write_bytes(b"audio")
        return Path(output)

    monkeypatch.setattr(
        "finesub.media.clips.extract_window_clip", fake_extract
    )
    seen = {}

    def fake_complete(role, messages, **kwargs):
        seen["file_ref"] = kwargs.get("file_ref")
        return LLMCallResult(
            content=(
                "<reasoning>done</reasoning>\n"
                "<analysis_notes>notes</analysis_notes>\n"
                "<requested_entries></requested_entries>\n"
                "<keep_entries></keep_entries>\n"
                "<search_queries></search_queries>"
            ),
            role=role,
            model="fake-agy",
            fallback_used=False,
            raw_response={},
        )

    monkeypatch.setattr(client, "complete", fake_complete)
    _result, file_ref = run_fast_session(
        window=window,
        segment_count=len(_segments()),
        audio_path=tmp_path / "audio.wav",
        task_artifact_dir=tmp_path / "artifacts",
        client=client,
        token_counter=FakeTokenCounter(),
        profile=profile,
        search_rounds=1,
    )

    assert uploads == []
    assert file_ref is seen["file_ref"]
    assert file_ref is not None and file_ref.file_id == ""
    assert Path(file_ref.local_path).is_file()


def test_fast_round1_resumes_validated_session(tmp_path) -> None:
    knowledge_root = tmp_path / "knowledge"
    (knowledge_root / "streamer").mkdir(parents=True)
    (knowledge_root / "common").mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "none", "efficiency")
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
        task_artifact_dir=artifact_dir,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "efficiency"),
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


def test_fast_round1_prompt_and_parse_follow_the_knowledge_switch(tmp_path) -> None:
    """Empty knowledge inputs drop every knowledge-owned piece of both prompts.

    Index injection and the entry blocks share one predicate (docs/llm_prompts.md): a
    ``--knowledge none`` (or empty-base) fast round must not be told to pick
    entries off an index it does not have, and the parser must not demand the
    blocks back.
    """

    from finesub.llm.prompts import build_fast_round1_messages

    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("audio", "local", "quality")
    )
    with_kb = build_fast_round1_messages(
        window=window,
        streamer_index="- 主播A | 别名\n",
        common_index="",
        profile=resolve_profile("audio", "local", "quality"),
    )
    joined = with_kb[0]["content"] + with_kb[1]["content"]
    assert "<requested_entries>" in joined and "<streamer_index>" in joined

    without_kb = build_fast_round1_messages(
        window=window, profile=resolve_profile("audio", "local", "quality")
    )
    joined = without_kb[0]["content"] + without_kb[1]["content"]
    for marker in (
        "知识库",
        "<requested_entries>",
        "<keep_entries>",
        "<streamer_index>",
        "<preinjected_entries>",
    ):
        assert marker not in joined, marker
    # The numbered lists close back up rather than leaving holes.
    assert "2. 提出联网搜索 query" in without_kb[0]["content"]

    # A correct knowledge-off reply omits the blocks; only expect_entries=True
    # (the knowledge-on shape) demands them.
    reply = (
        "<reasoning>ok</reasoning>\n<analysis_notes>要点</analysis_notes>\n"
        "<search_queries></search_queries>"
    )
    parsed = parse_fast_round1_output(reply, expect_entries=False)
    assert parsed.requested_entries == () and parsed.keep_entries == ()
    with pytest.raises(ValueError):
        parse_fast_round1_output(reply, expect_entries=True)


def test_fast_session_with_knowledge_off_never_touches_the_base(tmp_path, monkeypatch) -> None:
    """`--knowledge none` must not read indices or entry bodies anywhere.

    Even a reply that happens to name a real entry key must not open the base
    back up through ``resolve_round1_entries``.
    """

    import finesub.llm.stages.fast_session as fast_session

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("knowledge base was read with knowledge off")

    monkeypatch.setattr(fast_session, "load_index_text", _forbidden)
    monkeypatch.setattr(fast_session, "render_preinjected_entries", _forbidden)
    monkeypatch.setattr(fast_session, "resolve_round1_entries", _forbidden)

    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "none", "efficiency")
    )

    class Client:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=(
                    "<reasoning>分析。</reasoning>\n"
                    "<analysis_notes>要点。</analysis_notes>\n"
                    "<search_queries></search_queries>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    result, _ = run_fast_session(
        window=window,
        segment_count=len(_segments()),
        knowledge_root=tmp_path / "kb",
        knowledge_enabled=False,
        extra_info="提到了主播A",
        client=Client(),
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "efficiency"),
        resume=False,
    )
    assert result.entry_details_text == ""


def test_decide_fast_mode_with_knowledge_off_never_reads_the_base(tmp_path, monkeypatch) -> None:
    """The fast gate must not grow with a knowledge base the run will not see."""

    import finesub.llm.stages.plan as plan_module

    monkeypatch.setattr(
        plan_module,
        "load_index_text",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("knowledge base was read with knowledge off")
        ),
    )
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        knowledge_enabled=False,
        token_counter=FakeTokenCounter(),
    )
    assert decision.enabled


def test_decide_fast_mode_auto_enables_small_input(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
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
        profile=resolve_profile("audio", "local", "quality", output_scale=10_000),
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
            profile=resolve_profile("audio", "local", "quality", output_scale=10_000),
            knowledge_root=tmp_path / "kb",
            token_counter=FakeTokenCounter(),
        )


def test_decide_fast_mode_auto_falls_back_when_over_subtitle_cap(tmp_path) -> None:
    """The fast window is the whole input, so it is the most exposed to the
    quality guardrail the multi-window planner enforces."""
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
        max_window_subtitle_tokens=1,
    )

    assert not decision.enabled
    assert "max_window_subtitle_tokens" in decision.reason
    meta = decision.to_metadata()
    assert meta["max_window_subtitle_tokens"] == 1
    assert meta["subtitle_input_tokens"] > 1


def test_decide_fast_mode_subtitle_cap_zero_disables_the_gate(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
        knowledge_root=tmp_path / "kb",
        token_counter=FakeTokenCounter(),
        max_window_subtitle_tokens=0,
    )

    assert decision.enabled
    assert decision.to_metadata()["max_window_subtitle_tokens"] == 0


def test_decide_fast_mode_on_raises_when_over_subtitle_cap(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_window_subtitle_tokens"):
        decide_fast_mode(
            stable_json=_stable_json(tmp_path),
            fast="on",
            profile=resolve_profile("audio", "local", "quality"),
            knowledge_root=tmp_path / "kb",
            token_counter=FakeTokenCounter(),
            max_window_subtitle_tokens=1,
        )


def test_decide_fast_mode_checks_round1_input_reserve(tmp_path) -> None:
    decision = decide_fast_mode(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("audio", "local", "quality"),
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
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "local", "quality")
    )
    decision = FastDecision(mode="auto", enabled=True, reason="", window=window)
    ctx = FastSessionResult(
        analysis_notes="笔记",
        entry_details_text="条目",
        search_results_text="结果",
        evidence_pack_mode=True,
    )
    file_ref = object()

    mm_kwargs = _fast_execute_kwargs(decision, ctx, file_ref, resolve_profile("audio", "local", "quality"))
    assert mm_kwargs["windows_override"] == [window]
    assert mm_kwargs["seed_query_results"] == ctx.seed_query_results()
    assert mm_kwargs["entry_details"] == "条目"
    assert mm_kwargs["evidence_pack_mode"] is True
    assert mm_kwargs["extra_fingerprint"] == ctx.fingerprint()
    assert mm_kwargs["file_ref_seed"] == {window.chunk_id: file_ref}

    # The text route seeds nothing beyond the single window (no injections).
    text_kwargs = _fast_execute_kwargs(decision, None, None, resolve_profile("text", "none", "quality"))
    assert text_kwargs == {"windows_override": [window]}


def test_fast_seeds_replace_query_round_in_the_correction_loop(tmp_path, monkeypatch) -> None:
    profile = resolve_profile("text", "local", "quality")
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
                messages = messages("capableC")
            calls.append((role, messages))
            return LLMCallResult(
                content=(
                    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|0.0|one|一|8|1|译1字；宜保持独立\n"
                    "sub|2|1.0|0.0|two|二|8|1|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|one|一|8|1|\nsub|2|1.0|0.0|two|二|8|1|\n</translated>"
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

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)
    monkeypatch.setattr("finesub.llm.stages.correction.run.WebSearchClient", ExplodingSearchClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
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
    profile = resolve_profile("text", "local", "quality")
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=profile
    )
    round1_calls = []

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
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

    # Exempt (L3 whitelist): the run's execution identity says nothing about
    # whether this committed context is still the right one.
    class _OtherGroup(FakeClient):
        execution_identity = {"routing_identity_digest": "another-model-group"}

    _same, ref3, reused3 = acquire_fast_context(
        context_path=context_path, **{**session_kwargs, "client": _OtherGroup()}
    )
    assert reused3 and ref3 is None
    assert round1_calls == [LLMRole.GENERAL_CAPABLE]

    # Gated: the notes are a knob the user turns *at this artifact*, so reusing
    # the old context would silently ignore the request.
    _changed, ref4, reused4 = acquire_fast_context(
        context_path=context_path,
        **{**session_kwargs, "extra_info": "新的任务备注"},
    )
    assert not reused4 and ref4 is None
    assert round1_calls == [LLMRole.GENERAL_CAPABLE, LLMRole.GENERAL_CAPABLE]

    changed_window = plan_fast_window(
        [*_segments()[:-1], SubtitleSegment("2", 62.0, 102.5, "源字幕变了。")],
        counter=FakeTokenCounter(),
        profile=profile,
    )
    _changed, ref5, reused5 = acquire_fast_context(
        context_path=context_path,
        **{**session_kwargs, "window": changed_window},
    )
    assert not reused5 and ref5 is None
    assert len(round1_calls) == 3


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
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("text", "local", "quality")
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
                messages = messages("capableC")
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
        search_rounds=2,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "local", "quality"),
    )

    loop_user = seen_messages[1][1]["content"]
    assert "<previous_kept_entries>\nエーちゃん" in loop_user
    assert "关西腔。" in loop_user
    assert "关西腔。" in result.entry_details_text
    assert result.payload["fast"]["keep_entries"] == ["エーちゃん"]
    assert result.payload["injected_entries"] == ["主播A"]


def test_fast_gate_names_a_too_small_envelope_instead_of_a_negative_budget(
    tmp_path,
) -> None:
    """The round-2 reserve is an absolute token count calibrated against
    Gemini's 194k prompt limit. A bound group with a smaller envelope made
    ``prompt_limit - reserve`` negative, so every input "exceeded" a negative
    budget -- true, and unreadable. The gate now says the actual thing.

    ``--fast on`` still errors rather than downgrading silently, same as the
    budget gates it sits next to.
    """

    from dataclasses import replace as dc_replace

    import pytest

    from finesub.llm.routing.config import DEFAULT_LIMITS
    from finesub.llm.routing.profiles import FAST_ROUND2_INPUT_RESERVE_TOKENS
    from finesub.llm.stages.plan import decide_fast_mode

    small = dc_replace(DEFAULT_LIMITS, prompt_input_limit=23_000)
    # No stable JSON is read: the envelope decides before any input does.
    decision = decide_fast_mode(
        stable_json=tmp_path / "absent.json", fast="auto", limits=small
    )

    assert decision.enabled is False
    assert "23000" in decision.reason
    assert str(FAST_ROUND2_INPUT_RESERVE_TOKENS) in decision.reason
    assert "-" not in decision.reason.split("envelope")[1].split("<=")[0]

    with pytest.raises(ValueError, match="does not fit fast mode"):
        decide_fast_mode(
            stable_json=tmp_path / "absent.json", fast="on", limits=small
        )

    # off still short-circuits first.
    assert (
        decide_fast_mode(
            stable_json=tmp_path / "absent.json", fast="off", limits=small
        ).reason
        == "fast mode disabled"
    )
