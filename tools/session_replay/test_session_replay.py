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
    load_fixture,
    save_fixture,
)
from session_replay.sessions.correction import (
    CorrectionSessionAdapter,
    SampleResult,
    _thinking_override_kwargs,
    call_result_meta,
    replay_seed,
    replay_temperature,
)
from session_replay.run import resolve_sampling_plan
from session_replay.sessions.base import pin_client_role_to_free_model
from finesub.llm.chunking import SubtitleSegment
from finesub.llm.prompts import ContextPack


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
            "sub|1|1.0|0.0|hello|你好|8|2|",
            "</translated>",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


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
                "planning": {"profile_id": "media=audio,retrieval=local,difficulty=quality"},
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
    assert fixture.context_pack["window_contexts"] == [
        {
            "window_id": "0001",
            "first_source_id": "1",
            "last_source_id": "2",
            "context": "w1",
        }
    ]
    messages = CorrectionSessionAdapter().build_messages(fixture)
    user = next(message["content"] for message in messages if message["role"] == "user")
    assert "w1" in user


def test_extract_fixture_reads_the_task_flow_delivery_exchange(tmp_path: Path) -> None:
    """A task-flow delivery exchange renders the whole conversation: the
    window's inputs are in the *opening* user message (retrieval slots hold the
    placeholder), what the batch produced is in the fill turn. The fixture must
    freeze the opening's CSV/advice and the fill's search results -- not the
    last user section wholesale, and never the placeholder."""
    not_retrieved_yet = "（本轮尚未检索；需要由 harness 代为取得的输入时按 system 指令先提请求）"

    run_dir = tmp_path / "BV_flow"
    exchanges = run_dir / "llm-artifacts" / "exchanges"
    exchanges.mkdir(parents=True)
    _write_minimal_stable(run_dir / "BV_flow-stable.json")
    opening = (
        "请严格根据 system 指令。\n\n<entry_details>\n词条A 详情\n</entry_details>\n\n"
        "<previous_advice>\n上窗建议\n</previous_advice>\n\n<pre_round_notes>\n（无）\n</pre_round_notes>\n\n"
        f"本窗口搜索结果：\n<search_results>\n{not_retrieved_yet}\n</search_results>\n\n"
        "<preceding_context>\n</preceding_context>\n\n<asr_result>\n1|0.0|1.0|0.2|hello\n2|1.2|1.3|0.0|world\n</asr_result>\n\n最后提醒：输出字幕。"
    )
    request = "<reasoning>先查</reasoning>\n<window_notes>专名可疑</window_notes>\n<search_queries>\nfoo\n</search_queries>"
    fill = (
        "以下是你上一条回复所请求的输入。\n\n<entry_details>\n词条A 详情\n词条B 详情\n</entry_details>\n\n"
        "<search_results>\nFILLED_SEARCH\n</search_results>\n\n请交付。"
    )
    (exchanges / "003-01-correction-0001-attempt0.md").write_text(
        "\n".join(
            [
                "# correction-0001-attempt0", "",
                "## 请求（system）", "", "system", "",
                "## 请求（user）", "", opening, "",
                "## 请求（assistant）", "", request, "",
                "## 请求（user）", "", fill, "",
                "## 模型响应", "", "<translated>\nsub|1|1.0|0.0|hello|你好|8|2|\n</translated>", "",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "BV_flow-research-context.json").write_text(
        json.dumps({"planning": {"profile_id": "media=text,retrieval=local,difficulty=quality"},
                    "context_pack": {"general_context": {}, "window_contexts": {}}}),
        encoding="utf-8",
    )
    fixture = extract_fixture_from_exchange(run=run_dir, chunk_id="0001")
    assert fixture.search_results == "FILLED_SEARCH"
    assert "词条B 详情" in fixture.entry_details
    assert fixture.previous_advice == "上窗建议"
    assert fixture.window_notes == "专名可疑"
    assert fixture.window["source_ids"] == ["1", "2"]

    # A direct delivery (two sections, placeholder left in place) freezes nothing.
    (exchanges / "003-01-correction-0001-attempt0.md").write_text(
        "\n".join(
            ["# correction-0001-attempt0", "", "## 请求（system）", "", "system", "",
             "## 请求（user）", "", opening, "", "## 模型响应", "", "<translated>\n</translated>", ""]
        ),
        encoding="utf-8",
    )
    direct = extract_fixture_from_exchange(run=run_dir, chunk_id="0001")
    assert direct.search_results == ""
    assert direct.entry_details == "词条A 详情"


def test_load_legacy_fixture_binds_only_its_frozen_chunk(tmp_path: Path) -> None:
    fixture = CorrectionFixture(
        session="correction",
        version=1,
        profile_id="media=text,retrieval=local,difficulty=intermediate",
        chunk_id="0001",
        task_update_feedback=False,
        context_pack={
            "general_context": {"global_summary": "g"},
            "window_contexts": {"0001": "target", "0002": "other"},
        },
        previous_advice="",
        entry_details="",
        query={"window_notes": "", "search_results": ""},
        window={"source_ids": ["10", "11", "12"]},
        media={},
    )
    path = save_fixture(tmp_path / "legacy.json", fixture)

    loaded = load_fixture(path)
    pack = ContextPack.from_dict(loaded.context_pack).with_source_order(
        ["10", "11", "12"]
    )

    assert pack.window_context_for_source_ids(["11"], chunk_id="0001") == "target"
    assert pack.unbound_window_contexts == {"0002": "other"}


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
                "planning": {"profile_id": "media=audio,retrieval=local,difficulty=quality"},
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
        model = "gemini/gemini-3.6-flash"
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
    assert meta["model"] == "gemini/gemini-3.6-flash"
    assert meta["thinking_level"] == "medium"
    assert meta["thinking_budget"] == 26214
    assert meta["finish_reason"] == "stop"
    assert meta["usage"]["thinking_tokens"] == 120
    assert meta["usage"]["output_tokens"] == 380
    assert meta["usage"]["prompt_audio_tokens"] == 300
    assert meta["usage"]["total_input_tokens"] == 1000


def test_replay_holds_temperature_and_re_rolls_through_the_seed() -> None:
    """n replies have to be draws from one distribution.

    Walking the temperature down per attempt made every draw come from a
    different one, so the spread across a reply set measured the ramp as much
    as the prompt. The seed carries the re-roll instead.
    """

    assert replay_temperature(1.0) == 1.0
    assert replay_temperature(0.2) == 0.2
    seeds = [replay_seed(attempt) for attempt in range(1, 6)]
    assert len(set(seeds)) == len(seeds)
    assert replay_temperature(0.0) == 0.0


def test_replay_thinking_override_is_cli_only() -> None:
    profile_without_legacy_field = object()
    assert _thinking_override_kwargs(profile_without_legacy_field) == {}
    assert _thinking_override_kwargs(
        profile_without_legacy_field, thinking_level="medium"
    ) == {"thinking_level": "medium"}


def test_live_correction_replay_factory_receives_variant_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import session_replay.sessions.correction as correction_session
    from finesub.llm.routing.config import CapabilityTier

    stable = tmp_path / "sample-stable.json"
    _write_minimal_stable(stable)
    fixture = CorrectionFixture(
        session="correction",
        version=1,
        profile_id="media=text,retrieval=local,difficulty=efficiency",
        chunk_id="0001",
        task_update_feedback=False,
        context_pack={"general_context": {}, "window_contexts": []},
        previous_advice="",
        entry_details="",
        query={"window_notes": "", "search_results": ""},
        window={
            "chunk_id": "0001",
            "source_ids": ["1"],
            "clip_start": 0.0,
            "clip_end": 1.0,
            "budget": {
                "input_tokens": 1,
                "subtitle_input_tokens": 1,
                "estimated_output_tokens": 1,
                "token_counter_source": "test",
            },
        },
        media={"run_dir": str(tmp_path)},
        stable_json=str(stable),
    )
    fixture_path = save_fixture(tmp_path / "fixture.json", fixture)

    class FakeCall:
        content = (
            "<decisions>\n- ok\n</decisions>\n"
            "<translated>\n"
            "type|position|start|corrected_text|translation|conf|note\n"
            "sub|1|0.0|hello|你好|high|\n"
            "</translated>\n<next_advice></next_advice>"
        )
        model = "fake"
        api_key_label = "test"
        thinking_level = "medium"
        thinking_budget = 0
        fallback_used = False
        capability_tier = CapabilityTier.BASIC
        variant = "basicA"
        raw_response = {"usage": {}}

    class FakeClient:
        def __init__(self, **kwargs):
            configured = next(iter(kwargs["role_configs"].values()))
            assert configured.variant == "basicA"

        def complete(self, _role, messages, **_kwargs):
            built = messages("basicA")
            assert any(item.get("role") == "user" for item in built)
            return FakeCall()

    monkeypatch.setattr(correction_session, "RoleClient", FakeClient)
    result = CorrectionSessionAdapter().run(
        run=tmp_path,
        chunk_id="0001",
        out_dir=tmp_path / "out",
        n=1,
        max_attempts=1,
        fixture_override=fixture_path,
    )

    assert len(result.successes) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("3.6-flash", (2, 5)),
        ("gemini/gemini-3.8-flash", (2, 5)),
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


def test_pin_model_prefers_exact_flash_over_flash_lite(monkeypatch) -> None:
    import dataclasses

    from finesub.llm.client import RoleClient
    from finesub.llm.routing import model_catalog
    from finesub.llm.routing.config import LLMRole

    # No packaged generation ships both a Flash and a Flash Lite any more, so
    # plant the longer id the exact match must not fall into.
    rows = model_catalog.default_model_catalog()
    lite = next(e for e in rows if e.fact_id == "gemini-free-3_5-flash-lite")
    planted = dataclasses.replace(
        lite, fact_id="gemini-free-3_6-flash-lite", api_model_id="gemini/gemini-3.6-flash-lite"
    )
    monkeypatch.setattr(model_catalog, "default_model_catalog", lambda: (*rows, planted))

    client = RoleClient()
    selected = pin_client_role_to_free_model(
        client, LLMRole.GENERAL_CAPABLE, "3.6-flash"
    )
    assert selected == "gemini/gemini-3.6-flash"
    pinned = client.role_configs[LLMRole.GENERAL_CAPABLE]
    endpoints = pinned.endpoint_chain
    assert [endpoint.api_model_id for endpoint in endpoints] == [selected]
    assert pinned.model_group_id == ""


def test_pin_model_rejects_ambiguous_fuzzy_name() -> None:
    from finesub.llm.client import RoleClient
    from finesub.llm.routing.config import LLMRole

    client = RoleClient()
    with pytest.raises(RuntimeError, match="ambiguous"):
        pin_client_role_to_free_model(client, LLMRole.GENERAL_CAPABLE, "flash-lite")


def test_write_reply_persists_usage_meta(tmp_path: Path) -> None:
    adapter = CorrectionSessionAdapter()
    sample = SampleResult(
        ok=True,
        index=1,
        attempt=2,
        content="<translated>\nsub|1|1.0|0.0|a|b|9|0.5|\n</translated>",
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
        profile_id="media=audio,retrieval=local,difficulty=quality",
        chunk_id="0001",
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
        model="gemini/gemini-3.6-flash",
        path=tmp_path / "reply-01.md",
        translated_path=tmp_path / "reply-01.translated.csv",
        call_meta={
            "model": "gemini/gemini-3.6-flash",
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
    from session_replay.sessions.task_flow import PrepareTaskSessionAdapter

    # The guard fires before any fixture is touched, so a bare dict/None is
    # enough to reach it.
    for adapter in (PrepareTaskSessionAdapter(),):
        with pytest.raises(NotImplementedError):
            adapter.build_messages({}, variant="capableC")


def test_indexes_from_fixture_reads_both_shapes() -> None:
    """Flat per-category keys (old fixtures) and the nested mapping the
    round-input dumps write both resolve to one ``knowledge_indexes`` mapping
    (review 2026-09-07, third round: the nested shape once replayed empty)."""

    from session_replay.fixture import indexes_from_fixture

    flat = {
        "streamer_index": "- 星野灯 | | 阿灯 | 个人势",
        "common_index": "- 原神 [游戏] | Genshin | 原 | RPG",
    }
    assert set(indexes_from_fixture(flat)) == {"streamer", "common"}
    nested = {"knowledge_indexes": {"streamer": "- 星野灯 | | 阿灯 | 个人势"}}
    assert indexes_from_fixture(nested) == {"streamer": "- 星野灯 | | 阿灯 | 个人势"}


def test_save_load_fixture_roundtrip(tmp_path: Path) -> None:
    fixture = CorrectionFixture(
        session="correction",
        version=1,
        profile_id="media=audio,retrieval=local,difficulty=quality",
        chunk_id="0001",
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
        "type|position|corrected_text|translation|conf|note\n"
        "sub|1,2|a b|短译|high|test\n"
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
        "type|position|corrected_text|translation|conf|note\n"
        "discard|1|uncertain\n"
        "sub|2|b|乙|high|test\n"
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
        "type|position|corrected_text|translation|conf|note\n"
        "sub|1|a|甲|high|test\n"
        "sub|2|b|乙|high|test\n"
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
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|a|甲|high|test\n</translated>\n",
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
        "type|position|start|corrected_text|translation|conf|note\n"
        "sub|1|12.3|a|甲|high|ok\n"
        "sub|2|3.1|b|乙|high|lost hundreds digit\n"
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


def test_replay_native_search_is_read_from_the_effective_profile() -> None:
    """A `retrieval=native` prompt has no evidence pack.

    Dispatching it without the model's search tool would produce an arm with
    neither injected evidence nor the means to find any -- and that reads as
    "native is worse" for a reason unrelated to retrieval.
    """

    from tools.session_replay.sessions.base import replay_wants_native_search

    local_fixture = {"profile_id": "media=audio,retrieval=local,difficulty=quality"}
    native_fixture = {"profile_id": "media=text,retrieval=native,difficulty=quality"}

    assert replay_wants_native_search(local_fixture) is False
    assert replay_wants_native_search(native_fixture) is True
    # The override wins over whatever the fixture was captured under.
    assert replay_wants_native_search(
        local_fixture, "media=text,retrieval=native,difficulty=quality"
    ) is True
    assert replay_wants_native_search(
        native_fixture, "media=text,retrieval=local,difficulty=quality"
    ) is False
    # Nested shape and absent profile both degrade to "not native".
    assert replay_wants_native_search(
        {"profile": {"profile_id": "retrieval=native"}}
    ) is True
    assert replay_wants_native_search({}) is False
    assert replay_wants_native_search(None, None) is False


def _gold_for_alignment() -> dict:
    """A three-row gold: rows 1|2|3, with the 2-3 boundary a must_merge."""

    return {
        "source_count": 3,
        "source_fingerprint_sha256": "stale",
        "policy": {"merge_default": "must_not_merge", "drop_default": "must_keep"},
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
        "merge": {"default": "must_not_merge", "must_merge": ["2-3"], "may_merge": []},
        "drop": {"default": "must_keep", "must_drop": [], "may_drop": ["1"]},
        "sources": [
            {"id": "1", "start": 0.0, "duration": 1.0, "text": "a"},
            {"id": "2", "start": 2.0, "duration": 1.0, "text": "b"},
            {"id": "3", "start": 4.0, "duration": 1.0, "text": "c"},
        ],
    }


def test_a_gold_survives_the_window_being_re_segmented() -> None:
    """Judgments follow the audio, not the index.

    The gold's ids stop meaning anything the moment the window is cut
    differently -- which is what the 2026-08-05 VAD change did to
    BV1ojjc6MEAs-0001 (286 sources -> 303), leaving the scorer refusing to run
    at all. What still means something is *when* a boundary is: if the new cut
    still breaks at that moment, the judgment applies to whichever rows now
    meet there.
    """

    from session_replay.alignment import align_benchmark

    # Same audio, re-cut: old row 1 is now two rows, so every later id shifts.
    resegmented = [
        SubtitleSegment(id="1", start=0.0, end=0.4, text="a1"),
        SubtitleSegment(id="2", start=0.5, end=1.0, text="a2"),
        SubtitleSegment(id="3", start=2.0, end=3.0, text="b"),
        SubtitleSegment(id="4", start=4.0, end=5.0, text="c"),
    ]
    aligned = align_benchmark(_gold_for_alignment(), resegmented)

    # "2-3" was the boundary at t=3.0; that is now "3-4".
    assert aligned.benchmark["merge"]["must_merge"] == ["3-4"]
    assert aligned.lost_must_merge == 0
    # The split inside old row 1 is a boundary nobody audited: it must be
    # neutral, not an overmerge -- the gold's must_not_merge default was
    # earned by auditing every boundary in *its* window, not this one.
    assert "1-2" in aligned.unaudited_boundaries
    assert "1-2" in aligned.benchmark["merge"]["may_merge"]
    # A droppable old row stays droppable in every row it became.
    assert set(aligned.benchmark["drop"]["may_drop"]) >= {"1", "2"}


def test_alignment_reports_the_judgments_it_could_not_carry() -> None:
    """A partial score must never read as a full one."""

    from session_replay.alignment import align_benchmark

    # The 2-3 boundary (t=3.0) is gone: the new cut joined b and c outright.
    joined = [
        SubtitleSegment(id="1", start=0.0, end=1.0, text="a"),
        SubtitleSegment(id="2", start=2.0, end=5.0, text="bc"),
    ]
    aligned = align_benchmark(_gold_for_alignment(), joined)

    assert aligned.benchmark["merge"]["must_merge"] == []
    assert aligned.lost_must_merge == 1
    note = aligned.coverage_note(boundary_total=1, source_total=2)
    assert "1 must_merge" in note and "boundaries" in note


def test_a_gold_without_its_sources_still_refuses_a_mismatched_window() -> None:
    """Alignment needs something to align by; guessing is not an option."""

    import pytest as _pytest

    from session_replay.alignment import align_benchmark

    gold = _gold_for_alignment()
    gold.pop("sources")
    with _pytest.raises(ValueError, match="does not carry the source rows"):
        align_benchmark(gold, [SubtitleSegment(id="1", start=0.0, end=1.0, text="a")])


def test_a_gold_whose_text_moved_is_not_aligned_away() -> None:
    """Alignment answers a re-cut, and nothing else.

    The fingerprint covers the ASR text as well as the cut, so a decode change
    trips it while every id still means exactly what the gold meant. Aligning
    there is a no-op remap that quietly drops the fingerprint -- that is, it
    deletes the only signal saying the gold has gone stale.
    """

    import pytest as _pytest

    from session_replay.benchmark import BenchmarkWindowMismatchError, prepare_benchmark

    gold = _gold_for_alignment()
    same_timing_new_text = [
        SubtitleSegment(id="1", start=0.0, end=1.0, text="A"),
        SubtitleSegment(id="2", start=2.0, end=3.0, text="B"),
        SubtitleSegment(id="3", start=4.0, end=5.0, text="C"),
    ]
    with _pytest.raises(BenchmarkWindowMismatchError, match="segmentation is unchanged"):
        prepare_benchmark(gold, same_timing_new_text)


def test_a_broken_gold_is_never_aligned_into_working() -> None:
    """A wrong default is wrong on every window; alignment must not hide it."""

    import pytest as _pytest

    from session_replay.benchmark import BenchmarkMalformedError, prepare_benchmark

    gold = _gold_for_alignment()
    gold["drop"]["default"] = "may_keep"
    resegmented = [
        SubtitleSegment(id="1", start=0.0, end=0.4, text="a1"),
        SubtitleSegment(id="2", start=0.5, end=1.0, text="a2"),
        SubtitleSegment(id="3", start=2.0, end=3.0, text="b"),
        SubtitleSegment(id="4", start=4.0, end=5.0, text="c"),
    ]
    with _pytest.raises(BenchmarkMalformedError):
        prepare_benchmark(gold, resegmented)


def test_a_row_that_joined_a_droppable_source_to_a_kept_one_is_neutral() -> None:
    """Neither verdict is right for it, so it gets neither.

    Inheriting `must_drop` would have the scorer reward deleting content the
    gold says must be kept -- the one thing a drop gold exists to forbid.
    """

    from session_replay.alignment import align_benchmark

    gold = _gold_for_alignment()
    gold["drop"] = {"default": "must_keep", "must_drop": ["1"], "may_drop": []}
    # The re-cut swallowed old row 1 (must_drop) and old row 2 (must_keep)
    # into a single new row.
    joined = [
        SubtitleSegment(id="1", start=0.0, end=3.0, text="ab"),
        SubtitleSegment(id="2", start=4.0, end=5.0, text="c"),
    ]
    aligned = align_benchmark(gold, joined)

    assert aligned.benchmark["drop"]["must_drop"] == []
    assert "1" in aligned.benchmark["drop"]["may_drop"]
    assert aligned.mixed_sources == frozenset({"1"})
    # Covered by a verdict, but the verdict is the permissive one, so it is
    # not one of the rows the score can move on.
    note = aligned.coverage_note(1, 2)
    assert "2/2 sources are covered" in note
    assert "merged a droppable row into one that must be kept" in note
    assert aligned.as_json(1, 2)["sources_discriminating"] == 1


def test_a_millisecond_of_drift_does_not_hand_over_a_verdict() -> None:
    """Rows that merely touch are neighbours, not the same row."""

    from session_replay.alignment import align_benchmark

    gold = _gold_for_alignment()
    gold["drop"] = {"default": "must_keep", "must_drop": ["1"], "may_drop": []}
    # New row 2 starts 1ms before old row 1 ended: an overlap, but not a claim.
    drifted = [
        SubtitleSegment(id="1", start=0.0, end=0.999, text="a"),
        SubtitleSegment(id="2", start=0.999, end=3.0, text="b"),
        SubtitleSegment(id="3", start=4.0, end=5.0, text="c"),
    ]
    aligned = align_benchmark(gold, drifted)

    assert aligned.benchmark["drop"]["must_drop"] == ["1"]
    assert "2" not in aligned.mixed_sources


def test_a_lost_must_drop_is_reported_as_a_must_drop() -> None:
    """Coverage that mislabels what it lost is coverage nobody can act on."""

    from session_replay.alignment import align_benchmark

    gold = _gold_for_alignment()
    gold["drop"] = {"default": "must_keep", "must_drop": ["2"], "may_drop": ["1"]}
    # Old rows 1 and 2 are both gone from the audio the new cut covers.
    remainder = [SubtitleSegment(id="1", start=4.0, end=5.0, text="c")]
    aligned = align_benchmark(gold, remainder)

    assert aligned.lost_must_drop == 1 and aligned.lost_may_drop == 1
    assert "1 must_drop" in aligned.coverage_note(0, 1)


def test_json_output_says_the_score_is_a_floor(tmp_path, capsys) -> None:
    """`--json` used to carry the numbers with none of the caveat.

    An automated comparison is exactly the reader that cannot see the prose
    coverage line, and exactly the one that would read a partial gold's score
    as a full one.
    """

    import json as _json

    from session_replay import benchmark as benchmark_module

    gold = _gold_for_alignment()
    resegmented = [
        SubtitleSegment(id="1", start=0.0, end=0.4, text="a1"),
        SubtitleSegment(id="2", start=0.5, end=1.0, text="a2"),
        SubtitleSegment(id="3", start=2.0, end=3.0, text="b"),
        SubtitleSegment(id="4", start=4.0, end=5.0, text="c"),
    ]
    prepared, aligned = benchmark_module.prepare_benchmark(gold, resegmented)
    assert aligned is not None
    payload = _json.loads(_json.dumps(aligned.as_json(3, 4)))

    assert payload["aligned"] is True
    assert payload["boundaries_total"] == 3 and payload["boundaries_covered"] < 3
    # Never above covered, and only equal when no verdict is permissive.
    assert payload["boundaries_discriminating"] <= payload["boundaries_covered"]
    assert payload["sources_discriminating"] <= payload["sources_covered"]
    assert "1-2" in payload["boundaries_unaudited"]
    assert "aligned onto a re-segmented window" in payload["note"]
    assert prepared["merge"]["must_merge"] == ["3-4"]


def test_a_permissive_gold_verdict_is_covered_but_not_discriminating() -> None:
    """The distinction the two counts exist for, on a hand-written may_drop.

    Nothing is re-cut here: the gold itself says "either answer is fine" about
    row 1. The scorer subtracts `may_drop` from both `false_drop` and
    `missed_drop`, so that row cannot move the score whatever the model does
    -- and one count called "scored" used to claim otherwise.
    """

    from session_replay.alignment import align_benchmark

    gold = _gold_for_alignment()
    gold["drop"] = {"default": "must_keep", "must_drop": [], "may_drop": ["1"]}
    unchanged = [
        SubtitleSegment(id="1", start=0.0, end=1.0, text="a"),
        SubtitleSegment(id="2", start=2.0, end=3.0, text="b"),
    ]
    aligned = align_benchmark(gold, unchanged)

    payload = aligned.as_json(1, 2)
    assert payload["sources_covered"] == 2
    assert payload["sources_discriminating"] == 1
    # Not a re-cut artefact: the person wrote this one.
    assert payload["sources_mixed"] == []


# --- task-flow adapters (docs/llm_harness_research.md, 「准备任务」) ----


def test_prepare_task_adapter_replays_the_dumped_builder_input(tmp_path) -> None:
    from tools.session_replay.registry import get_session

    adapter = get_session("prepare-task")
    fixture = {
        "transcript": "--- window 0001 ---\n1|テスト\n",
        "extra_info": "",
        "knowledge_indexes": {"streamer": "- 主播A | | 测试", "common": ""},
        "emits_queries": True,
        "emits_entries": True,
        "native_search": False,
        "max_requested_entries": 12,
        "agentic": False,
        "style_catalog": "",
    }
    messages = adapter.build_messages(fixture)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "获取缺失输入" in messages[0]["content"]
    assert adapter.validate_reply("<reasoning>x</reasoning><context_pack>{}</context_pack>") == []
    assert adapter.validate_reply(
        "<reasoning>x</reasoning><analysis_notes>n</analysis_notes><requested_entries></requested_entries>"
        "<keep_entries></keep_entries><search_queries></search_queries>"
    ) == []
    assert adapter.validate_reply("<reasoning>x</reasoning>")  # neither shape


def test_window_request_adapter_builds_the_opening_only(tmp_path) -> None:
    import json as _json

    from tools.session_replay.registry import get_session
    from tools.session_replay.fixture import CorrectionFixture

    stable = tmp_path / "x-stable.json"
    stable.write_text(
        _json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    adapter = get_session("correction-request")
    fixture = CorrectionFixture.from_dict(
        {
            "chunk_id": "0001",
            "profile_id": "correction_media=text,retrieval=local,difficulty=quality,continuity=serial",
            "context_pack": {"general_context": {}, "window_contexts": []},
            "window": {"chunk_id": "0001", "source_ids": ["1"], "clip_start": 0.0, "clip_end": 1.0},
            "stable_json": str(stable),
        }
    )
    messages = adapter.build_messages(fixture)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "获取缺失输入" in messages[0]["content"]
    agentic = adapter.build_messages(fixture, agentic=True)
    assert "由你自行取得" in agentic[0]["content"] and "获取缺失输入（同一任务内" not in agentic[0]["content"]
    assert adapter.validate_reply(
        "<reasoning>x</reasoning><window_notes></window_notes><search_queries></search_queries>", fixture
    ) == []


def test_replay_follows_the_source_runs_preset_and_falls_back_when_it_is_gone(tmp_path, capsys, monkeypatch) -> None:
    import json as _json

    from finesub import config as app_config
    from finesub.llm.routing.model_routes import default_model_routes, install_runtime_preset
    from tools.session_replay import run as replay_run

    # This suite runs outside test/conftest.py: pin an empty config so a
    # developer's own [llm] preset cannot make it red or falsely green.
    empty = tmp_path / "config.toml"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("FINESUB_CONFIG_FILE", str(empty))
    app_config.clear_config_cache()
    default_model_routes.cache_clear()

    artifacts = tmp_path / "run" / "llm-artifacts"
    artifacts.mkdir(parents=True)
    record = artifacts / "resolved-switches.json"
    record.write_text(_json.dumps({"active_preset": "fast"}), encoding="utf-8")
    install_runtime_preset(None)
    assert replay_run._source_run_preset(tmp_path / "run") == "fast"
    record.write_text(_json.dumps({"active_preset": "gone"}), encoding="utf-8")
    assert replay_run._source_run_preset(tmp_path / "run") is None
    assert "no longer exists" in capsys.readouterr().err
    app_config.clear_config_cache()
    default_model_routes.cache_clear()



@pytest.mark.parametrize(("native_search", "retrieval"), [(False, "none"), (True, "native")])
def test_text_replay_calls_complete_with_its_real_signature(tmp_path, monkeypatch, native_search, retrieval) -> None:
    """`complete()` takes the three-way `retrieval` switch; the generic loop
    passed the retired `native_search` boolean and every correction-request /
    prepare-task replay failed before reaching the API (da7329dd..930a6379)."""

    import inspect

    from finesub.llm.client import RoleClient
    from tools.session_replay.sessions import base

    seen = {}

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def complete(self, role, messages, **kwargs):
            inspect.signature(RoleClient.complete).bind(None, role, messages, **kwargs)
            seen.update(kwargs)

            class Call:
                content = "ok"
            return Call()

    monkeypatch.setattr(base, "RoleClient", FakeClient)
    base.run_text_replay(
        session_name="t", messages=[{"role": "user", "content": "x"}],
        validate_reply=lambda _content: [], out_dir=tmp_path, n=1, max_attempts=1,
        native_search=native_search,
    )
    assert seen["retrieval"] == retrieval and "native_search" not in seen


# --- three passes: pass 1 (speaker judge), pass 2 regrouped, pass 3 (proofread) -----------------------------


class _JudgeClient:
    """Answers every judge call: rows keep their track, the second row joins the first."""

    def __init__(self, **_kwargs):
        self.calls = 0

    def complete(self, role, messages, **kwargs):
        import re

        from finesub.llm.client import LLMCallResult

        self.calls += 1
        ids = re.findall(r"▶ (q\d+) ", messages[1]["content"])
        lines = [json.dumps({"id": key, "track": "keep", "why": "w", **({"join": True} if n == 1 else {})})
                 for n, key in enumerate(ids)]
        return LLMCallResult(content="<judge>\n" + "\n".join(lines) + "\n</judge>\n<tracks>\ns0 = 甲\n</tracks>",
                             role=role, model="fake", fallback_used=False,
                             raw_response={"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}})


def _three_pass_run(tmp_path: Path) -> Path:
    """A run dir whose judge ran pass 1 (tracked shape) over two correction windows, through production."""
    from finesub.llm.chunking import SubtitleWindow
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.speaker_judge import run as judge_run
    from finesub.llm.stages.speaker_judge.prompt import Candidate
    from finesub.speech.diarization import judgments as sidecar

    run = tmp_path / "clip"
    run.mkdir()
    stable = run / "clip-stable.json"
    speaker = {"track": 0, "tracks": [0], "block": 0, "coverage": {"0": 0.9}, "labels": []}
    stable.write_text(json.dumps({"segments": [
        {"id": str(i), "start": 2.0 * i, "end": 2.0 * i + 1.5, "text": f"line {i}", "speaker": speaker}
        for i in range(1, 5)]}), encoding="utf-8")
    windows = [SubtitleWindow(f"{n:04d}", [SubtitleSegment(str(i), 2.0 * i, 2.0 * i + 1.5, f"line {i}")
                                           for i in ids], [], "even", None)
               for n, ids in ((1, (1, 2)), (2, (3, 4)))]
    judge_run.run_speaker_judge(
        stable_json=stable, correction_windows=windows, profile=resolve_profile("text", "none", "quality"),
        client=_JudgeClient(), inputs=judge_run.JudgeInputs("clip", (Candidate("甲"),), summary=lambda ids: "概要"), media="text",
        audio=None, task_artifact_dir=run / "clip.llm-artifacts", task_id="t", shape=sidecar.SHAPE_TRACKED)
    return run


def test_pass_1_replays_one_judge_unit_from_the_runs_dump(tmp_path, monkeypatch) -> None:
    from tools.session_replay.registry import get_session
    from tools.session_replay.sessions import three_pass

    run = _three_pass_run(tmp_path)
    adapter = get_session("speaker-judge")
    dry = adapter.run(run=run, chunk_id="0002", out_dir=tmp_path / "dry", dry_run=True)
    assert "定字幕的形状" in dry.prompt_system_path.read_text(encoding="utf-8")
    user = dry.prompt_user_path.read_text(encoding="utf-8")
    assert "▶ q3 " in user and "▶ q1 " not in user and "概要" in dry.prompt_system_path.read_text(encoding="utf-8")
    monkeypatch.setattr(three_pass, "RoleClient", _JudgeClient)
    live = adapter.run(run=run, chunk_id="0002", out_dir=tmp_path / "live", n=1, max_attempts=2)
    entry = json.loads((tmp_path / "live" / "reply-01.json").read_text(encoding="utf-8"))
    assert len(live.successes) == 1 and entry["status"] == "success" and entry["acts"] == {"4": {"act": "join", "why": "w"}}
    assert (tmp_path / "live" / "exchanges").is_dir(), "production's session logs the exchange"
    with pytest.raises(ValueError, match="0009"):
        adapter.run(run=run, chunk_id="0009", out_dir=tmp_path / "none", dry_run=True)
    with pytest.raises(NotImplementedError):
        adapter.run(run=run, chunk_id="0002", out_dir=tmp_path / "v", dry_run=True, variant="capableC")


def test_a_three_pass_windows_fixture_is_regrouped_by_pass_1(tmp_path) -> None:
    from finesub.speech.diarization import judgments as sidecar

    run = _three_pass_run(tmp_path)
    fixture = CorrectionFixture.from_dict({
        "chunk_id": "0001", "stable_json": str(run / "clip-stable.json"),
        "window": {"chunk_id": "0001", "source_ids": ["1", "2"]}, "three_pass": {"shape": sidecar.SHAPE_TRACKED}})
    window = build_window_from_fixture(fixture)
    assert window.grouped and [(s.id, s.text) for s in window.segments] == [("1", "line 1 line 2")]
    one_pass = build_window_from_fixture(CorrectionFixture.from_dict({**fixture.to_dict(), "three_pass": {}}))
    assert not one_pass.grouped and [s.id for s in one_pass.segments] == ["1", "2"]


def test_a_three_pass_fixture_takes_the_planned_rows_not_the_grouped_ones(tmp_path) -> None:
    from session_replay.fixture import _planned_rows, _three_pass_record

    art = tmp_path / "art"
    art.mkdir()
    (art / "task-artifacts.jsonl").write_text(
        json.dumps({"kind": "correction_mode", "payload": {"mode": "single"}}) + "\n"
        + json.dumps({"kind": "correction_mode", "payload": {"mode": "three-pass", "shape": "tracked"}}) + "\n",
        encoding="utf-8")
    assert _three_pass_record(art) == {"shape": "tracked"}
    assert _planned_rows(art, "0001", ["1"]) == (["1"], []), "a fast run writes no plan"
    (art / "correction-window-plan.json").write_text(json.dumps({"plan": {"windows": [
        {"chunk_id": "0001", "segment_ids": ["1", "2", "3"], "overlap_ids": ["1"]}]}}), encoding="utf-8")
    assert _planned_rows(art, "0001", ["1", "3"]) == (["1", "2", "3"], ["1"])
    with pytest.raises(ValueError, match="0001-a"):
        _planned_rows(art, "0001-a", [])


def test_pass_3_replays_one_window_from_the_runs_dump(tmp_path, monkeypatch) -> None:
    from finesub.llm.client import LLMCallResult
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.llm.stages.proofread import run as proofread_run
    from finesub.llm.stages.proofread.prompt import Line, Window
    from tools.session_replay.registry import get_session
    from tools.session_replay.sessions import three_pass

    run = tmp_path / "clip"
    art = run / "clip.llm-artifacts"
    art.mkdir(parents=True)
    lines = [Line(2.0, "あ", "あ", "啊"), Line(3.0, "いい", "いい", "好")]
    inputs = proofread_run.ProofreadInputs("clip", None, "（风格）", {"0001": "甲 → 甲"})
    proofread_run._dump_inputs(art, lines, [Window("0001", (0, 1))], inputs,
                               profile=resolve_profile("text", "none", "quality"), task_id="t")

    class Client:
        def __init__(self, **_kwargs):
            pass

        def complete(self, role, messages, **kwargs):
            assert kwargs["validator_spec"]["params"] == {"count": 2}, "production's agent validator rides along"
            return LLMCallResult(content="<proofread>\n</proofread>", role=role, model="fake", fallback_used=False,
                                 raw_response={})

    adapter = get_session("proofread")
    dry = adapter.run(run=run, chunk_id="0001", out_dir=tmp_path / "dry", dry_run=True)
    user = dry.prompt_user_path.read_text(encoding="utf-8")
    assert "（风格）" in user and "甲 → 甲" in user and "|啊|" in user
    monkeypatch.setattr(three_pass, "RoleClient", Client)
    live = adapter.run(run=run, chunk_id="0001", out_dir=tmp_path / "live", n=1, max_attempts=1)
    assert len(live.successes) == 1
    assert json.loads((tmp_path / "live" / "reply-01.json").read_text(encoding="utf-8"))["changes"] == []
