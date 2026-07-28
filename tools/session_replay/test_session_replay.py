"""Tests for tools/session_replay (fixture extract + dry-run assemble, no API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_replay.exchange_parse import split_exchange_sections
from session_replay.benchmark import score_reply
from session_replay.fixture import (
    CorrectionFixture,
    _parse_payload_json,
    build_window_from_fixture,
    extract_fixture_from_exchange,
    save_fixture,
)
from session_replay.sessions.correction import (
    CorrectionSessionAdapter,
    SampleResult,
    call_result_meta,
    replay_temperature,
)
from session_replay.run import resolve_sampling_plan
from session_replay.sessions.base import pin_client_role_to_free_model
from session_replay.sessions.research import _load_research_fixture
from llm.chunking import SubtitleSegment


def _write_minimal_stable(path: Path) -> None:
    segs = [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "hello"},
        {"id": "2", "start": 1.2, "end": 2.5, "text": "world"},
    ]
    path.write_text(
        json.dumps({"segments": segs}, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_minimal_exchange(path: Path, *, search_body: str) -> None:
    user = f"""请严格根据 system 指令。

<entry_details>
词条A 详情
</entry_details>

<previous_advice>
（无）
</previous_advice>

<pre_round_notes>
窗口笔记：专名可疑
</pre_round_notes>

本窗口搜索结果：
<search_results>
{search_body}
</search_results>

<preceding_context>
</preceding_context>

<asr_result>
1|0.0|1.0|0.2|hello
2|1.2|1.3|0.0|world
</asr_result>

最后提醒：输出字幕。
"""
    text = "\n".join(
        [
            "# correction-0001-attempt0",
            "",
            "## 请求（system）",
            "",
            "system prompt 含 <search_results> 示例不应被抽取",
            "<search_results>",
            "EXAMPLE_ONLY_SHOULD_NOT_BE_USED",
            "</search_results>",
            "",
            "## 请求（user）",
            "",
            user,
            "",
            "## 模型响应",
            "",
            "<translated>",
            "sub|1|1.0|hello|你好|8|",
            "</translated>",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def test_research_fixture_fallback_uses_run_root_context(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    research_context = tmp_path / "clip-research-context.json"
    research_context.write_text(
        json.dumps({"context_pack": {"general_context": {"summary": "ok"}}}),
        encoding="utf-8",
    )

    fixture = _load_research_fixture(
        artifact_dir,
        "round1",
        research_context=research_context,
    )

    assert fixture["_from_context"] is True
    assert fixture["context_pack"]["general_context"]["summary"] == "ok"


def test_split_exchange_sections_prefers_named_roles() -> None:
    sections = split_exchange_sections(
        "## 请求（system）\n\nSYS\n\n## 请求（user）\n\nUSER\n\n## 模型响应\n\nRESP\n"
    )
    assert sections["system"] == "SYS"
    assert sections["user"] == "USER"
    assert sections["response"] == "RESP"


def test_legacy_dynamic_json_payload_remains_readable() -> None:
    user = (
        "动态窗口 payload：\n\n"
        '{"current_asr_csv":"<asr_result>\\n1|0.0|1.0|0.0|hello\\n</asr_result>"}'
        "\n\n最后提醒：输出字幕。"
    )
    payload = _parse_payload_json(user)
    assert payload["current_asr_csv"].splitlines()[1].startswith("1|")


def test_extract_fixture_freezes_search_not_system_example(tmp_path: Path) -> None:
    run_dir = tmp_path / "BV_test"
    artifacts = run_dir / "llm-artifacts" / "exchanges"
    artifacts.mkdir(parents=True)
    stable = run_dir / "BV_test-stable.json"
    _write_minimal_stable(stable)
    # Minimal audio stub: skip probe by using clip from compute — need a
    # real-ish file or soft-fail. Use empty wav-less: probe will fail.
    # Provide clip via correction-windows cache and monkey by creating a
    # tiny aac is hard; instead create a dummy file and skip duration probe
    # by making audio missing — compute_clip_range still works without duration.
    search_body = (
        "--- query: foo ---\n"
        "provider: exa\n"
        "- title\n"
        "  Summary: search hit\n"
        "\n"
        "## 深度提取结果（整页内容）\n"
        "### url: https://example.com/page\n"
        "extract body line\n"
    )
    exchange = artifacts / "005-correction-0001-attempt0.md"
    _write_minimal_exchange(exchange, search_body=search_body)

    research = run_dir / "BV_test-research-context.json"
    research.write_text(
        json.dumps(
            {
                "planning": {"profile_id": "mm-med"},
                "context_pack": {
                    "general_context": {"global_summary": "g"},
                    "window_contexts": {"0001": "w1"},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    fixture = extract_fixture_from_exchange(run=run_dir, chunk_id="0001")
    assert "EXAMPLE_ONLY_SHOULD_NOT_BE_USED" not in fixture.search_results
    assert "search hit" in fixture.search_results
    assert "深度提取结果" in fixture.search_results
    assert "extract body line" in fixture.search_results
    assert fixture.window_notes.startswith("窗口笔记")
    assert "词条A" in fixture.entry_details
    assert fixture.previous_advice == ""
    assert fixture.window["source_ids"] == ["1", "2"]
    assert fixture.context_pack["general_context"]["global_summary"] == "g"


def test_rebuild_messages_embeds_frozen_search(tmp_path: Path) -> None:
    run_dir = tmp_path / "BV_test2"
    (run_dir / "llm-artifacts" / "exchanges").mkdir(parents=True)
    stable = run_dir / "BV_test2-stable.json"
    _write_minimal_stable(stable)
    search_body = "FROZEN_SEARCH_AND_EXTRACT_MARKER\nline2"
    _write_minimal_exchange(
        run_dir / "llm-artifacts" / "exchanges" / "001-correction-0001-attempt0.md",
        search_body=search_body,
    )
    (run_dir / "BV_test2-research-context.json").write_text(
        json.dumps(
            {
                "planning": {"profile_id": "mm-med"},
                "context_pack": {"general_context": {}, "window_contexts": {}},
            }
        ),
        encoding="utf-8",
    )
    fixture = extract_fixture_from_exchange(run=run_dir, chunk_id="0001")
    # mm-med needs audio path only as label for dry message build; missing file ok.
    adapter = CorrectionSessionAdapter()
    messages = adapter.build_messages(fixture)
    user = next(m["content"] for m in messages if m["role"] == "user")
    assert "FROZEN_SEARCH_AND_EXTRACT_MARKER" in user
    window = build_window_from_fixture(fixture)
    assert [s.id for s in window.segments] == ["1", "2"]


def test_call_result_meta_extracts_usage_from_raw_response() -> None:
    class _Call:
        content = "x"
        model = "gemini/gemini-3.5-flash"
        api_key_label = "sp"
        thinking_level = "medium"
        thinking_budget = 26214
        fallback_used = False
        raw_response = {
            "choices": [{"finish_reason": "stop", "message": {"content": "x"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
                "completion_tokens_details": {"reasoning_tokens": 120},
                "prompt_tokens_details": {"audio_tokens": 300},
            },
        }

    meta = call_result_meta(_Call())
    assert meta["model"] == "gemini/gemini-3.5-flash"
    assert meta["thinking_level"] == "medium"
    assert meta["thinking_budget"] == 26214
    assert meta["finish_reason"] == "stop"
    assert meta["usage"]["thinking_tokens"] == 120
    assert meta["usage"]["output_tokens"] == 380
    assert meta["usage"]["prompt_audio_tokens"] == 300
    assert meta["usage"]["total_input_tokens"] == 1000


def test_replay_temperature_decreases_after_every_call() -> None:
    assert replay_temperature(1.0, 1) == 1.0
    assert replay_temperature(1.0, 2) == 0.99
    assert replay_temperature(0.2, 3) == 0.18
    assert replay_temperature(0.0, 2) == 0.0


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("3.6-flash", (2, 5)),
        ("gemini/gemini-3.5-flash", (2, 5)),
        ("3.5-flash-lite", (3, 10)),
        (None, (3, 9)),
    ],
)
def test_resolve_sampling_plan_model_defaults(
    model: str | None, expected: tuple[int, int]
) -> None:
    assert resolve_sampling_plan(model) == expected


def test_resolve_sampling_plan_explicit_values_override_defaults() -> None:
    assert resolve_sampling_plan("3.6-flash", n=4, max_attempts=12) == (4, 12)
    with pytest.raises(ValueError, match="smaller"):
        resolve_sampling_plan("3.5-flash-lite", n=4, max_attempts=3)


def test_pin_model_prefers_exact_flash_over_flash_lite() -> None:
    from llm.client import LiteLLMRoleClient
    from llm.config import LLMRole

    client = LiteLLMRoleClient()
    selected = pin_client_role_to_free_model(
        client, LLMRole.GENERAL_CAPABLE, "3.5-flash"
    )
    assert selected == "gemini/gemini-3.5-flash"
    endpoints = client.role_configs[LLMRole.GENERAL_CAPABLE].endpoint_chain
    assert [endpoint.litellm_model for endpoint in endpoints] == [selected]


def test_pin_model_rejects_ambiguous_fuzzy_name() -> None:
    from llm.client import LiteLLMRoleClient
    from llm.config import LLMRole

    client = LiteLLMRoleClient()
    with pytest.raises(RuntimeError, match="ambiguous"):
        pin_client_role_to_free_model(client, LLMRole.GENERAL_CAPABLE, "3.5")


def test_write_reply_persists_usage_meta(tmp_path: Path) -> None:
    adapter = CorrectionSessionAdapter()
    sample = SampleResult(
        ok=True,
        index=1,
        attempt=2,
        content="<translated>\nsub|1|1.0|a|b|9|\n</translated>",
        model="gemini/gemini-3.1-flash-lite",
        call_meta={
            "model": "gemini/gemini-3.1-flash-lite",
            "api_key_label": "sp",
            "thinking_level": "medium",
            "thinking_budget": 26214,
            "fallback_used": False,
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 10,
                "prompt_text_tokens": 10,
                "prompt_audio_tokens": 0,
                "uncached_input_tokens": 10,
                "cached_input_tokens": 0,
                "total_input_tokens": 10,
                "thinking_tokens": 0,
                "output_tokens": 20,
                "total_output_tokens": 20,
                "total_tokens": 30,
            },
        },
    )
    path = tmp_path / "reply-01.md"
    adapter._write_reply(path, sample)
    text = path.read_text(encoding="utf-8")
    meta = json.loads(text.split("```json", 1)[1].split("```", 1)[0])
    assert meta["usage"]["thinking_tokens"] == 0
    assert meta["usage"]["output_tokens"] == 20
    assert meta["thinking_level"] == "medium"
    assert meta["finish_reason"] == "stop"
    assert meta["temperature"] == 1.0


def test_summary_includes_token_totals(tmp_path: Path) -> None:
    adapter = CorrectionSessionAdapter()
    fixture = CorrectionFixture(
        session="correction",
        version=1,
        profile_id="mm-med",
        chunk_id="0001",
        evidence_pack_mode=False,
        task_update_feedback=False,
        context_pack={"general_context": {}, "window_contexts": {}},
        previous_advice="",
        entry_details="",
        query={"window_notes": "n", "search_results": "sr", "requested_entry_keys": []},
        window={
            "chunk_id": "0001",
            "source_ids": ["1"],
            "overlap_source_ids": [],
            "preceding_source_ids": [],
            "boundary_reason": "t",
            "clip_start": 0.0,
            "clip_end": 2.0,
            "budget": {
                "input_tokens": 1,
                "subtitle_input_tokens": 1,
                "estimated_output_tokens": 1,
                "total_with_margin": 2,
                "token_counter_source": "t",
            },
        },
        media={"run_dir": str(tmp_path), "audio_path": "", "video_path": ""},
        stable_json=str(tmp_path / "x-stable.json"),
    )
    usage = {
        "prompt_tokens": 100,
        "prompt_text_tokens": 80,
        "prompt_audio_tokens": 20,
        "uncached_input_tokens": 90,
        "cached_input_tokens": 10,
        "total_input_tokens": 100,
        "thinking_tokens": 50,
        "output_tokens": 200,
        "total_output_tokens": 250,
        "total_tokens": 350,
    }
    sample = SampleResult(
        ok=True,
        index=1,
        attempt=1,
        content="<translated>\nok|1|1.0|a|译|9|\n</translated>",
        model="gemini/gemini-3.5-flash",
        path=tmp_path / "reply-01.md",
        translated_path=tmp_path / "reply-01.translated.csv",
        call_meta={
            "model": "gemini/gemini-3.5-flash",
            "thinking_level": "medium",
            "usage": usage,
        },
    )
    summary = adapter._write_summary(
        out_dir=tmp_path,
        fixture=fixture,
        fixture_src=tmp_path / "fixture.json",
        label="t",
        note="meta test",
        successes=[sample],
        failures=[],
        dry_run=False,
    )
    text = summary.read_text(encoding="utf-8")
    assert "## Token 汇总" in text
    assert "thinking=50" in text
    assert "think=50" in text
    assert "base_temperature: 1.00" in text
    assert "temperature: 1.00" in text


# ---------------------------------------------------------------------------
# Variant guard: non-correction rounds have no variant set — a --variant /
# --force-tier request must raise, never silently serve the baseline.
# ---------------------------------------------------------------------------


def test_reject_unsupported_variant_noop_when_unset() -> None:
    from session_replay.sessions.base import reject_unsupported_variant

    # None / empty means "production default" — must not raise.
    reject_unsupported_variant("query", variant=None, force_tier=None)
    reject_unsupported_variant("research-r1", variant="", force_tier="")


def test_reject_unsupported_variant_raises_on_variant() -> None:
    from session_replay.sessions.base import reject_unsupported_variant

    with pytest.raises(NotImplementedError, match="no registered prompt variants"):
        reject_unsupported_variant("query", variant="queryB")


def test_reject_unsupported_variant_raises_on_force_tier() -> None:
    from session_replay.sessions.base import reject_unsupported_variant

    with pytest.raises(NotImplementedError, match="correction"):
        reject_unsupported_variant("fast-round1", force_tier="basic")


def test_non_correction_adapters_reject_variant_in_build_messages() -> None:
    from session_replay.sessions.fast_round import FastRound1SessionAdapter
    from session_replay.sessions.query import QuerySessionAdapter
    from session_replay.sessions.research import (
        ResearchR1SessionAdapter,
        ResearchR2SessionAdapter,
    )
    from session_replay.sessions.search_judge import SearchJudgeSessionAdapter

    # The guard fires before any fixture is touched, so a bare dict/None is
    # enough to reach it.
    for adapter in (
        QuerySessionAdapter(),
        ResearchR1SessionAdapter(),
        ResearchR2SessionAdapter(),
        SearchJudgeSessionAdapter(),
        FastRound1SessionAdapter(),
    ):
        with pytest.raises(NotImplementedError):
            adapter.build_messages({}, variant="capableB")


def test_save_load_fixture_roundtrip(tmp_path: Path) -> None:
    fixture = CorrectionFixture(
        session="correction",
        version=1,
        profile_id="mm-med",
        chunk_id="0001",
        evidence_pack_mode=False,
        task_update_feedback=False,
        context_pack={"general_context": {}, "window_contexts": {}},
        previous_advice="",
        entry_details="",
        query={"window_notes": "n", "search_results": "sr", "requested_entry_keys": []},
        window={
            "chunk_id": "0001",
            "source_ids": ["1"],
            "overlap_source_ids": [],
            "preceding_source_ids": [],
            "boundary_reason": "t",
            "clip_start": 0.0,
            "clip_end": 2.0,
            "budget": {
                "input_tokens": 1,
                "subtitle_input_tokens": 1,
                "estimated_output_tokens": 1,
                "total_with_margin": 2,
                "token_counter_source": "t",
            },
        },
        media={"run_dir": str(tmp_path), "audio_path": "", "video_path": ""},
        stable_json=str(tmp_path / "x-stable.json"),
    )
    path = tmp_path / "correction-0001.json"
    save_fixture(path, fixture)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["query"]["search_results"] == "sr"


def test_merge_drop_benchmark_scores_neutral_and_soft_limit_separately(
    tmp_path: Path,
) -> None:
    sources = [
        SubtitleSegment(id="1", start=0.0, end=1.0, text="a"),
        SubtitleSegment(id="2", start=1.1, end=5.2, text="b"),
    ]
    benchmark = {
        "weights": {
            "overmerge_boundary": 1,
            "undermerge": 1,
            "false_drop": 5,
            "missed_drop": 2,
            "soft_limit_excess": 1,
            "hard_limit_excess": 2,
        },
        "limits": {
            "soft_duration_seconds": 4,
            "soft_weighted_chars": 20,
            "hard_duration_seconds": 7,
            "hard_weighted_chars": 36,
        },
        "merge": {"must_merge": [], "may_merge": []},
        "drop": {"must_drop": [], "may_drop": []},
    }
    reply = tmp_path / "reply.md"
    reply.write_text(
        "<translated>\n"
        "sub|1,2|5.2|0|a b|短译|high|2|test\n"
        "</translated>\n",
        encoding="utf-8",
    )

    score = score_reply(
        reply_path=reply,
        benchmark=benchmark,
        source_segments=sources,
    )
    assert score.valid
    assert score.overmerge == ("1-2",)
    assert score.soft_limit_overmerge == ("1,2",)
    assert score.hard_limit_overmerge == ()
    assert score.weighted_cost == 2  # wrong boundary 1 + soft excess 1

    benchmark["merge"]["may_merge"] = ["1-2"]
    neutral = score_reply(
        reply_path=reply,
        benchmark=benchmark,
        source_segments=sources,
    )
    assert neutral.overmerge == ()
    assert neutral.soft_limit_overmerge == ()
    assert neutral.weighted_cost == 0

    hard_sources = [
        sources[0],
        SubtitleSegment(id="2", start=1.1, end=8.2, text="b"),
    ]
    benchmark["merge"]["may_merge"] = []
    hard = score_reply(
        reply_path=reply,
        benchmark=benchmark,
        source_segments=hard_sources,
    )
    assert hard.soft_limit_overmerge == ()
    assert hard.hard_limit_overmerge == ("1,2",)
    assert hard.weighted_cost == 4  # wrong boundary 1 + soft 1 + hard 2

    discard_reply = tmp_path / "discard.md"
    discard_reply.write_text(
        "<translated>\n"
        "discard|1|uncertain\n"
        "sub|2|4.1|0|b|乙|high|1|test\n"
        "</translated>\n",
        encoding="utf-8",
    )
    false_drop = score_reply(
        reply_path=discard_reply,
        benchmark=benchmark,
        source_segments=sources,
    )
    assert false_drop.false_drop == ("1",)
    assert false_drop.weighted_cost == 5

    keep_reply = tmp_path / "keep.md"
    keep_reply.write_text(
        "<translated>\n"
        "sub|1|1.0|0.1|a|甲|high|1|test\n"
        "sub|2|4.1|0|b|乙|high|1|test\n"
        "</translated>\n",
        encoding="utf-8",
    )
    benchmark["drop"]["must_drop"] = ["1"]
    missed_drop = score_reply(
        reply_path=keep_reply,
        benchmark=benchmark,
        source_segments=sources,
    )
    assert missed_drop.missed_drop == ("1",)
    assert missed_drop.weighted_cost == 2


def test_merge_drop_benchmark_ignores_reply_markdown_tag_warnings(
    tmp_path: Path,
) -> None:
    sources = [SubtitleSegment(id="1", start=0.0, end=1.0, text="a")]
    benchmark = {
        "weights": {
            "overmerge_boundary": 1,
            "undermerge": 1,
            "false_drop": 5,
            "missed_drop": 2,
            "soft_limit_excess": 1,
            "hard_limit_excess": 2,
        },
        "limits": {
            "soft_duration_seconds": 4,
            "soft_weighted_chars": 20,
            "hard_duration_seconds": 7,
            "hard_weighted_chars": 36,
        },
        "merge": {"must_merge": [], "may_merge": []},
        "drop": {"must_drop": [], "may_drop": []},
    }
    reply = tmp_path / "reply.md"
    reply.write_text(
        '# replay\n\n```json\n{"validation_warnings":["Row retracted (<void>)."]}\n```\n\n'
        "## 模型响应\n\n"
        "<translated>\nsub|1|1.0|0|a|甲|high|1|test\n</translated>\n",
        encoding="utf-8",
    )

    score = score_reply(
        reply_path=reply,
        benchmark=benchmark,
        source_segments=sources,
    )
    assert score.valid
    assert score.weighted_cost == 0


def test_merge_drop_benchmark_reports_start_mismatches_without_invalidating(
    tmp_path: Path,
) -> None:
    sources = [
        SubtitleSegment(id="188", start=12.34, end=13.0, text="a"),
        SubtitleSegment(id="189", start=13.1, end=14.0, text="b"),
    ]
    benchmark = {
        "weights": {
            "overmerge_boundary": 1,
            "undermerge": 1,
            "false_drop": 5,
            "missed_drop": 2,
            "soft_limit_excess": 1,
            "hard_limit_excess": 2,
        },
        "limits": {
            "soft_duration_seconds": 4,
            "soft_weighted_chars": 20,
            "hard_duration_seconds": 7,
            "hard_weighted_chars": 36,
        },
        "merge": {"must_merge": [], "may_merge": ["188-189"]},
        "drop": {"must_drop": [], "may_drop": []},
    }
    reply = tmp_path / "reply-start.md"
    reply.write_text(
        "<translated>\n"
        "type|position|start|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|12.3|0.7|0.1|a|甲|high|1|ok\n"
        "sub|2|3.1|0.9|0|b|乙|high|1|lost hundreds digit\n"
        "</translated>\n",
        encoding="utf-8",
    )

    score = score_reply(
        reply_path=reply,
        benchmark=benchmark,
        source_segments=sources,
        require_start_column=True,
    )
    assert score.valid
    assert score.start_checked_rows == 2
    assert score.start_mismatches == ("2: got 3.1, expected 13.1",)
    assert score.weighted_cost == 0
