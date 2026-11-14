"""Provider search on user-declared API endpoints (why: the catalog decision record in docs/llm_design_notes.md).

Anthropic endpoints get the server-side `web_search` tool, capped and never
resumed; an OpenAI-compatible endpoint searches only through the fields its
`search_request` names. The tool goes out only on a native-retrieval call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finesub.llm import provider_transports
from finesub.llm.client import (
    RoleClient,
    is_likely_ip_risk_error,
    is_quota_or_rate_limit_error,
    is_retryable_provider_error,
    provider_status_code,
)
from finesub.llm.http import StreamProgress
from finesub.llm.rate_limit import ComboCooldownPhase, ModelRateLimiter
from finesub.llm.routing.config import LLMRole, role_config_for
from finesub.llm.routing.model_catalog import (
    OPENAI_COMPAT_REQUEST_KEYS,
    load_effective_catalog,
)
from finesub.llm.routing.model_router import FailureKind, ModelRouter, classify_failure
from finesub.llm.routing.model_routes import load_model_routes, target_search_tool
from finesub.llm.streaming import ProviderStreamError

STREAMS = Path(__file__).parent / "fixtures" / "llm_streams"

CLAUDE = """
[models.claude]
provider_tier = "anthro"
provider_kind = "anthropic"
base_url = "https://api.anthropic.com"
api_model_id = "claude-sonnet-5"
max_input_tokens = 200000
"""
QWEN = """
[models.qwen]
provider_tier = "dashscope"
provider_kind = "openai_compat"
base_url = "https://dashscope.example/v1"
api_model_id = "qwen-max"
max_input_tokens = 128000
search_request = { enable_search = true }
"""
DEEPSEEK = """
[models.ds]
provider_tier = "deepseek"
provider_kind = "openai_compat"
base_url = "https://api.deepseek.com"
api_model_id = "deepseek-v4-flash"
max_input_tokens = 128000
"""


def _catalog(tmp_path: Path, text: str):
    path = tmp_path / "model_catalog.toml"
    path.write_text(text, encoding="utf-8")
    return {entry.fact_id: entry for entry in load_effective_catalog(path)}


# --- catalog -----------------------------------------------------------------


def test_unstated_search_follows_the_dialect(tmp_path) -> None:
    facts = _catalog(tmp_path, CLAUDE + QWEN + DEEPSEEK)

    assert facts["claude"].supports_native_search is True
    assert target_search_tool(facts["claude"]) == "web_search"
    assert facts["qwen"].supports_native_search is True
    assert target_search_tool(facts["qwen"]) == "search_request"
    assert facts["qwen"].search_request_body == {"enable_search": True}
    # No way to ask this provider to search: it cannot be said to.
    assert facts["ds"].supports_native_search is False
    assert target_search_tool(facts["ds"]) == ""
    # The attached conversational agent has no tool to hand over.
    assert facts["local-conversational-agent"].supports_native_search is False


def test_a_stated_false_turns_an_anthropic_endpoint_off(tmp_path) -> None:
    facts = _catalog(tmp_path, CLAUDE + "supports_native_search = false\n")

    assert facts["claude"].supports_native_search is False
    assert target_search_tool(facts["claude"]) == ""


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (DEEPSEEK + "supports_native_search = true\n", "needs search_request"),
        (CLAUDE + "search_request = { enable_search = true }\n", "only for provider_kind=openai_compat"),
        (DEEPSEEK + "search_request = { model = \"other\" }\n", "may not set \\['model'\\]"),
        (DEEPSEEK + "search_request = { since = 2026-11-09 }\n", "only JSON values"),
        (DEEPSEEK + "search_request = {}\n", "must be a table of request fields"),
        (DEEPSEEK + "search_request = true\n", "must be a table of request fields"),
    ],
)
def test_search_request_mistakes_fail_at_load(tmp_path, text, message) -> None:
    with pytest.raises(ValueError, match=message):
        _catalog(tmp_path, text)


def test_search_request_is_part_of_the_routing_identity(tmp_path) -> None:
    # Only models the active preset routes to enter the digest.
    config = {
        "preset": "mine",
        "model_groups": {"mine-corr": {"targets": ["qwen"]}},
        "presets": {"mine": {"name": "mine", "bindings": {"correction-text/quality": "mine-corr"}}},
    }
    base = load_model_routes(
        user_config=config, catalog=tuple(_catalog(tmp_path, QWEN).values())
    )
    changed = load_model_routes(
        user_config=config,
        catalog=tuple(
            _catalog(
                tmp_path, QWEN.replace("{ enable_search = true }", "{ enable_search = true, search_options = { forced_search = true } }")
            ).values()
        ),
    )

    assert changed.routing_identity_digest != base.routing_identity_digest


# --- transports --------------------------------------------------------------


def _feeding(lines, seen=None):
    def fake_stream(url, *, headers, payload, assembler, timeout):
        if seen is not None:
            seen.update(payload=dict(payload))
        for line in lines:
            assembler.feed_line(line)
        return assembler.finish()

    return fake_stream


def _sse(*events: str) -> list[str]:
    lines: list[str] = []
    for event in events:
        lines += [f"data: {event}", ""]
    return lines


def _anthropic(monkeypatch, lines, *, web_search=True, seen=None):
    monkeypatch.setattr(provider_transports, "_stream_json", _feeding(lines, seen))
    return provider_transports.anthropic_generate(
        base_url="https://api.anthropic.com",
        api_key="sk-ant",
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "芙莉莲的声优是谁？"}],
        max_tokens=4096,
        thinking_level="",
        timeout=30.0,
        progress=StreamProgress(),
        web_search=web_search,
    )


def test_anthropic_search_call_reads_answer_ledger_and_count(monkeypatch) -> None:
    seen: dict = {}
    lines = (STREAMS / "anthropic-web-search.sse").read_text(encoding="utf-8").split("\n")

    response = _anthropic(monkeypatch, lines, seen=seen)

    assert seen["payload"]["tools"] == [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
    ]
    # The line that opens the search is not answer text; citations split the
    # answer into deltas that still join up.
    assert response["choices"][0]["message"]["content"] == "芙莉莲由种崎敦美配音。"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["search_events"] == [
        {
            "tool": "web_search",
            "query": "葬送的芙莉莲 声优",
            "queries": ["葬送的芙莉莲 声优"],
            "urls": ["https://example.org/a", "https://example.org/b"],
        }
    ]
    assert response["usage"]["web_search_requests"] == 1


def test_only_the_text_right_before_a_search_is_dropped() -> None:
    blocks = [
        {"type": "text", "text": "先查一下。"},
        {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {"query": "a"}},
        {"type": "web_search_tool_result", "tool_use_id": "s1", "content": []},
        {"type": "text", "text": "第一段答案。"},
        {"type": "text", "text": "再核对一下。"},
        {"type": "server_tool_use", "id": "s2", "name": "web_search", "input": {"query": "b"}},
        {"type": "web_search_tool_result", "tool_use_id": "s2", "content": []},
        {"type": "text", "text": "第二段答案。"},
    ]

    assert provider_transports._anthropic_answer_text(blocks) == "第一段答案。第二段答案。"


def test_a_call_without_search_sends_no_tool(monkeypatch) -> None:
    seen: dict = {}
    lines = (STREAMS / "anthropic-omitted-thinking.sse").read_text(encoding="utf-8").split("\n")

    _anthropic(monkeypatch, lines, web_search=False, seen=seen)

    assert "tools" not in seen["payload"]


def _stopped(reason: str, *blocks: str) -> list[str]:
    return _sse(
        '{"type":"message_start","message":{"id":"m","model":"claude-sonnet-5","usage":{"input_tokens":10}}}',
        *blocks,
        '{"type":"message_delta","delta":{"stop_reason":"%s"},"usage":{"output_tokens":3}}' % reason,
        '{"type":"message_stop"}',
    )


def test_a_paused_search_turn_moves_on_without_retry(monkeypatch) -> None:
    with pytest.raises(provider_transports.SearchTurnPaused) as raised:
        _anthropic(monkeypatch, _stopped("pause_turn"))

    exc = raised.value
    # Nothing in it reads as retryable, a quota, a network problem or a
    # status: the chain hands the call on, and no retry re-bills the searches.
    assert provider_status_code(exc) is None
    assert not is_retryable_provider_error(exc)
    assert not is_quota_or_rate_limit_error(exc)
    assert not is_likely_ip_risk_error(exc)
    assert classify_failure(exc) is FailureKind.TRANSIENT


@pytest.mark.parametrize(
    "lines",
    [
        _stopped("tool_use"),
        _stopped(
            "end_turn",
            '{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"t1","name":"web_search","input":{}}}',
            '{"type":"content_block_stop","index":0}',
        ),
    ],
    ids=["stop_reason", "client_block"],
)
def test_an_endpoint_that_hands_the_search_back_fails_the_call(monkeypatch, lines) -> None:
    with pytest.raises(ProviderStreamError) as raised:
        _anthropic(monkeypatch, lines)

    exc = raised.value
    assert exc.status_code == 400
    assert "supports_native_search = false" in str(exc)
    assert not is_likely_ip_risk_error(exc)
    assert classify_failure(exc) is FailureKind.PERMANENT


def test_openai_compat_merges_search_request_only_when_given(monkeypatch) -> None:
    payloads: list[dict] = []
    lines = (STREAMS / "gemini-openai-compat.sse").read_text(encoding="utf-8").split("\n")

    def fake_stream(url, *, headers, payload, assembler, timeout):
        payloads.append(dict(payload))
        for line in lines:
            assembler.feed_line(line)
        return assembler.finish()

    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)
    for search_request in (None, {"enable_search": True}):
        provider_transports.openai_compat_generate(
            base_url="https://dashscope.example/v1",
            api_key="sk",
            model="qwen-max",
            messages=[{"role": "system", "content": "规则"}, {"role": "user", "content": "hi"}],
            max_tokens=1024,
            thinking_level="high",
            timeout=30.0,
            progress=StreamProgress(),
            search_request=search_request,
        )

    plain, searching = payloads
    # Every key the transport writes is one a `search_request` may not set.
    assert set(plain) == OPENAI_COMPAT_REQUEST_KEYS
    assert searching == {**plain, "enable_search": True}


# --- client ------------------------------------------------------------------


def _client(tmp_path, catalog_text, members, *, max_retries=0):
    routes = load_model_routes(
        user_config={
            "preset": "mine",
            "model_groups": {"mine-corr": {"targets": members}},
            "presets": {
                "mine": {"name": "mine", "bindings": {"correction-text/quality": "mine-corr"}}
            },
        },
        catalog=tuple(_catalog(tmp_path, catalog_text).values()),
    )
    return RoleClient(
        router=ModelRouter(routes),
        max_retries=max_retries,
        rate_limiter=ModelRateLimiter(enabled=False),
        role_configs={
            LLMRole.AUDIO_MULTIMODAL: role_config_for("correction-text", "quality", routes=routes)
        },
    )


def _complete(client, retrieval):
    return client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        retrieval=retrieval,
        task_group="correction-text",
        difficulty="quality",
    )


def _reply(**extra):
    return {
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5000, "completion_tokens": 2, **extra.pop("usage", {})},
        **extra,
    }


def test_client_records_anthropic_searches_and_skips_calibration(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []
    event = {"tool": "web_search", "query": "q", "queries": ["q"], "urls": ["https://example.org/a"]}

    def fake(**kwargs):
        calls.append(kwargs)
        searched = kwargs["web_search"]
        return _reply(
            usage={"web_search_requests": 1} if searched else {},
            **({"search_events": [event]} if searched else {}),
        )

    monkeypatch.setattr("finesub.llm.provider_transports.anthropic_generate", fake)
    monkeypatch.setenv("FINESUB_KEY_ANTHRO", "sk-ant")
    client = _client(tmp_path, CLAUDE, ["claude"])

    searched = _complete(client, "native")
    plain = _complete(client, "none")

    assert [call["web_search"] for call in calls] == [True, False]
    answering = searched.execution_attempts[-1]
    assert answering["search_events"] == [event]
    assert answering["web_search_requests"] == 1
    # The searched call's input holds the results it read: not a sample.
    accepted = [row for row in searched.route_decision["candidates"] if row.get("decision") == "accepted"]
    assert "estimate_calibration" not in accepted[0]
    accepted = [row for row in plain.route_decision["candidates"] if row.get("decision") == "accepted"]
    assert "estimate_calibration" in accepted[0]
    assert "search_events" not in plain.execution_attempts[-1]


def test_client_sends_search_request_only_on_a_native_call(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return _reply()

    monkeypatch.setattr("finesub.llm.provider_transports.openai_compat_generate", fake)
    monkeypatch.setenv("FINESUB_KEY_DASHSCOPE", "sk")
    client = _client(tmp_path, QWEN, ["qwen"])

    searched = _complete(client, "native")
    _complete(client, "none")

    assert [call["search_request"] for call in calls] == [{"enable_search": True}, None]
    # The ledger can say the search was on, not what it found.
    assert searched.execution_attempts[-1]["search_events"] == [
        {"tool": "search_request", "query": "", "queries": [], "urls": []}
    ]


@pytest.mark.parametrize("phase", [ComboCooldownPhase.NONE, ComboCooldownPhase.PROBE])
def test_a_paused_turn_goes_to_the_next_member_once(monkeypatch, tmp_path, phase) -> None:
    """Also when it was the probe of a cooling model: the server answered, so
    the cool-down ends instead of restarting for the calls that do not search."""

    claude_calls: list[dict] = []
    exhausted: list[object] = []
    cleared: list[object] = []

    def paused(**kwargs):
        claude_calls.append(kwargs)
        raise provider_transports.SearchTurnPaused("paused")

    monkeypatch.setattr("finesub.llm.provider_transports.anthropic_generate", paused)
    monkeypatch.setattr(
        "finesub.llm.provider_transports.openai_compat_generate", lambda **kwargs: _reply()
    )
    monkeypatch.setattr(
        ModelRateLimiter, "note_combo_exhausted", lambda self, *a, **k: exhausted.append(a)
    )
    monkeypatch.setattr(
        ModelRateLimiter, "clear_combo_cooldown", lambda self, endpoint, **k: cleared.append(endpoint)
    )
    monkeypatch.setattr(ModelRateLimiter, "combo_cooldown_phase", lambda self, *a, **k: phase)
    monkeypatch.setattr(ModelRateLimiter, "claim_combo_probe", lambda self, *a, **k: True)
    monkeypatch.setenv("FINESUB_KEY_ANTHRO", "sk-ant")
    monkeypatch.setenv("FINESUB_KEY_DASHSCOPE", "sk")
    client = _client(tmp_path, CLAUDE + QWEN, ["claude", "qwen"], max_retries=3)

    result = _complete(client, "native")

    assert result.target_id == "qwen"
    assert len(claude_calls) == 1
    assert exhausted == []
    assert [endpoint.provider_tier for endpoint in cleared][:1] == ["anthro"]


def test_a_handed_back_search_fails_the_call_without_fallback(monkeypatch, tmp_path) -> None:
    qwen_calls: list[dict] = []

    def handed_back(**kwargs):
        raise ProviderStreamError(400, "returned the web_search tool; set supports_native_search = false")

    monkeypatch.setattr("finesub.llm.provider_transports.anthropic_generate", handed_back)
    monkeypatch.setattr(
        "finesub.llm.provider_transports.openai_compat_generate",
        lambda **kwargs: qwen_calls.append(kwargs) or _reply(),
    )
    monkeypatch.setenv("FINESUB_KEY_ANTHRO", "sk-ant")
    monkeypatch.setenv("FINESUB_KEY_DASHSCOPE", "sk")
    client = _client(tmp_path, CLAUDE + QWEN, ["claude", "qwen"], max_retries=3)

    with pytest.raises(ProviderStreamError, match="supports_native_search = false"):
        _complete(client, "native")
    assert qwen_calls == []
