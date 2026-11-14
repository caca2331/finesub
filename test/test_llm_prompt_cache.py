from __future__ import annotations

import threading
import time

import httpx
import pytest

from finesub.llm import prompt_cache, provider_transports
from finesub.llm.client import validation_retry_sampling_kwargs
from finesub.llm.http import StreamProgress

MESSAGES = [
    {"role": "system", "content": "规则"},
    {"role": "user", "content": "窗口 1"},
]
OK_BODY = {
    "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
}


@pytest.fixture(autouse=True)
def _fresh_cache_state(monkeypatch) -> None:
    monkeypatch.setattr(prompt_cache, "_rejecting_endpoints", set())
    monkeypatch.setattr(prompt_cache, "_warm_until", {})
    monkeypatch.setattr(prompt_cache, "_in_flight", {})


def _openai(base_url: str = "https://openrouter.ai/api/v1", messages=MESSAGES):
    return provider_transports.openai_compat_generate(
        base_url=base_url,
        api_key="k",
        model="m",
        messages=messages,
        max_tokens=None,
        thinking_level=None,
        timeout=30.0,
        progress=StreamProgress(),
    )


def _rejection(field: str, status: int = 400) -> provider_transports.ProviderHTTPError:
    response = httpx.Response(status, text=f'{{"error": "Extra inputs are not permitted: {field}"}}')
    return provider_transports.ProviderHTTPError(f"HTTP {status} {response.text}", response)


def test_prefix_key_names_the_system_only() -> None:
    """Windows of one run differ after the system; they must share the key."""

    other_window = [MESSAGES[0], {"role": "user", "content": "窗口 2"}]
    assert prompt_cache.prefix_key(MESSAGES) == prompt_cache.prefix_key(other_window)
    assert prompt_cache.prefix_key(MESSAGES).startswith("finesub-")
    assert prompt_cache.prefix_key([{"role": "system", "content": "别的规则"}]) != prompt_cache.prefix_key(MESSAGES)
    assert prompt_cache.prefix_key([{"role": "user", "content": "只有 user"}]) == ""


def test_openai_compat_sends_the_prefix_key(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(provider_transports, "_stream_json", lambda url, **kw: sent.append(dict(kw["payload"])) or OK_BODY)

    _openai()

    assert sent[0]["prompt_cache_key"] == prompt_cache.prefix_key(MESSAGES)


def test_an_endpoint_refusing_the_key_is_asked_again_without_it_and_remembered(monkeypatch) -> None:
    sent = []

    def fake_stream(url, **kw):
        payload = dict(kw["payload"])
        sent.append(payload)
        if "prompt_cache_key" in payload and url.startswith("http://localhost"):
            raise _rejection("prompt_cache_key", status=422)  # FastAPI's answer to an extra field
        return OK_BODY

    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)

    assert _openai("http://localhost:8000/v1")["choices"][0]["message"]["content"] == "答案"
    _openai("http://localhost:8000/v1")
    _openai("https://openrouter.ai/api/v1")

    # refused, re-sent without; the next call goes without at once; another
    # endpoint still gets it
    assert ["prompt_cache_key" in payload for payload in sent] == [True, False, False, True]


def test_any_other_400_surfaces_as_it_is(monkeypatch) -> None:
    def fake_stream(url, **kw):
        raise _rejection("max_tokens")

    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)

    with pytest.raises(provider_transports.ProviderHTTPError):
        _openai()
    assert not prompt_cache.cache_key_rejected("https://openrouter.ai/api/v1")


def _anthropic_payload(monkeypatch, messages, *, followup_expected: bool) -> dict:
    seen = {}

    def fake_stream(url, **kw):
        seen.update(kw["payload"])
        return {"content": [{"type": "text", "text": "答"}], "stop_reason": "end_turn", "usage": {}}

    monkeypatch.setattr(provider_transports, "_stream_json", fake_stream)
    provider_transports.anthropic_generate(
        base_url="https://api.anthropic.com",
        api_key="k",
        model="claude-haiku-5-5",
        messages=messages,
        max_tokens=100,
        thinking_level=None,
        timeout=30.0,
        progress=StreamProgress(),
        followup_expected=followup_expected,
    )
    return seen


def test_anthropic_writes_the_last_turn_only_when_a_next_turn_will_resend_it(monkeypatch) -> None:
    breakpoint_after = {"type": "ephemeral"}

    plain = _anthropic_payload(monkeypatch, MESSAGES, followup_expected=False)
    opening = _anthropic_payload(monkeypatch, MESSAGES, followup_expected=True)

    assert plain["system"][0]["cache_control"] == breakpoint_after
    assert opening["system"][0]["cache_control"] == breakpoint_after
    assert plain["messages"][-1] == {"role": "user", "content": "窗口 1"}
    assert opening["messages"][-1] == {
        "role": "user",
        "content": [{"type": "text", "text": "窗口 1", "cache_control": breakpoint_after}],
    }


def test_a_later_turn_marks_the_end_of_the_opening_where_it_was_written(monkeypatch) -> None:
    """A breakpoint only finds entries at or before itself: the fill (or a
    repair) must mark the opening's end exactly as the opening did."""

    later = _anthropic_payload(
        monkeypatch,
        [*MESSAGES, {"role": "assistant", "content": "请求"}, {"role": "user", "content": "补充"}],
        followup_expected=False,
    )
    opening = _anthropic_payload(monkeypatch, MESSAGES, followup_expected=True)

    assert later["messages"][0] == opening["messages"][0]
    assert later["messages"][1:] == [
        {"role": "assistant", "content": "请求"},
        {"role": "user", "content": "补充"},
    ]


@pytest.mark.parametrize("followup_expected", [False, True])
@pytest.mark.parametrize("with_system", [False, True])
@pytest.mark.parametrize("later_turn", [False, True])
def test_anthropic_never_asks_for_more_than_four_breakpoints(
    monkeypatch, followup_expected, with_system, later_turn
) -> None:
    messages = [*MESSAGES] if with_system else [MESSAGES[1]]
    if later_turn:
        messages += [{"role": "assistant", "content": "请求"}, {"role": "user", "content": "补充"}]
    payload = _anthropic_payload(monkeypatch, messages, followup_expected=followup_expected)

    blocks = [*payload.get("system", [])] + [
        block for turn in payload["messages"] if isinstance(turn["content"], list) for block in turn["content"]
    ]
    assert sum("cache_control" in block for block in blocks) <= 4  # the Messages API's limit


def test_concurrent_requests_on_a_cold_prefix_wait_for_the_first() -> None:
    scope = ("anthropic", "https://api.anthropic.com", "m", "finesub-x")
    order: list[str] = []
    leader_inside = threading.Event()
    release_leader = threading.Event()

    def leader() -> None:
        with prompt_cache.warm_prefix(scope):
            order.append("leader-start")
            leader_inside.set()
            release_leader.wait(5)
            order.append("leader-end")

    def follower() -> None:
        with prompt_cache.warm_prefix(scope):
            order.append("follower")

    lead = threading.Thread(target=leader)
    lead.start()
    assert leader_inside.wait(5)
    follow = threading.Thread(target=follower)
    follow.start()
    time.sleep(0.05)
    assert order == ["leader-start"]  # the follower is held back
    release_leader.set()
    lead.join(5)
    follow.join(5)

    assert order == ["leader-start", "leader-end", "follower"]
    # warm now: a later burst does not queue behind anyone
    with prompt_cache.warm_prefix(scope):
        assert scope not in prompt_cache._in_flight


def test_a_failed_leader_leaves_the_prefix_cold() -> None:
    scope = ("openai_compat", "https://api.deepseek.com", "m", "finesub-y")

    with pytest.raises(RuntimeError):
        with prompt_cache.warm_prefix(scope):
            raise RuntimeError("HTTP 500")

    assert scope not in prompt_cache._warm_until
    assert scope not in prompt_cache._in_flight


def test_a_warm_prefix_goes_cold_after_the_shortest_cache_lifetime(monkeypatch) -> None:
    scope = ("anthropic", "u", "m", "finesub-z")
    with prompt_cache.warm_prefix(scope):
        pass
    now = time.monotonic()
    monkeypatch.setattr(prompt_cache.time, "monotonic", lambda: now + prompt_cache.WARM_SECONDS + 1)

    with prompt_cache.warm_prefix(scope):
        assert scope in prompt_cache._in_flight  # led again


def test_seed_goes_only_on_a_blind_re_ask() -> None:
    assert "seed" not in validation_retry_sampling_kwargs(0)
    assert "seed" not in validation_retry_sampling_kwargs(2, repair=True)
    first, second = validation_retry_sampling_kwargs(1), validation_retry_sampling_kwargs(2)
    assert first["seed"] != second["seed"]
    # the temperature ladder is unchanged by either
    assert validation_retry_sampling_kwargs(2, repair=True)["temperature"] == second["temperature"]


# --- Wiring: what the runtime hands the gate and the transports -------------


def _hedged(monkeypatch, custom_kind: str, *, key_id: str = "key-a", followup_expected: bool = False):
    from contextlib import contextmanager

    from finesub.llm import llm_runtime

    scopes, sent = [], {}

    @contextmanager
    def recording_gate(scope):
        scopes.append(scope)
        yield

    def fake_transport(**kwargs):
        sent.update(kwargs)
        return OK_BODY

    monkeypatch.setattr(prompt_cache, "warm_prefix", recording_gate)
    monkeypatch.setattr(provider_transports, "openai_compat_generate", fake_transport)
    monkeypatch.setattr(provider_transports, "anthropic_generate", fake_transport)
    monkeypatch.setattr(llm_runtime, "_gemini_generate_content", fake_transport)
    llm_runtime._generate_hedged(
        custom_kind=custom_kind,
        custom_base_url="https://example.test" if custom_kind else "",
        api_key="k",
        key_id=key_id,
        model_name="m",
        messages=MESSAGES,
        max_tokens=100,
        temperature=1.0,
        thinking_level=None,
        thinking_cfg={},
        native_search_tool=None,
        env_map={},
        timeouts=llm_runtime.ApiTimeouts(soft=30.0, hard=60.0),
        note_backup=lambda: None,
        record=lambda request: None,
        followup_expected=followup_expected,
    )
    return scopes[0], sent


@pytest.mark.parametrize("custom_kind", ["openai_compat", "anthropic"])
def test_paid_dialects_pass_the_gate_per_key_and_system(monkeypatch, custom_kind) -> None:
    scope, _ = _hedged(monkeypatch, custom_kind)
    other_key, _ = _hedged(monkeypatch, custom_kind, key_id="key-b")

    assert scope[0] == custom_kind
    assert scope[-1] == prompt_cache.prefix_key(MESSAGES)
    assert scope != other_key  # caches live per account


def test_gemini_bypasses_the_gate(monkeypatch) -> None:
    scope, _ = _hedged(monkeypatch, "")
    assert scope[-1] == ""  # `warm_prefix` lets an empty prefix through at once


def test_only_anthropic_is_told_a_next_turn_follows(monkeypatch) -> None:
    _, anthropic = _hedged(monkeypatch, "anthropic", followup_expected=True)
    _, openai = _hedged(monkeypatch, "openai_compat", followup_expected=True)

    assert anthropic["followup_expected"] is True
    assert "followup_expected" not in openai  # automatic caches need no hint


def test_a_seeded_re_ask_writes_no_tail_breakpoint(monkeypatch, tmp_path) -> None:
    """The next turn re-sends the opening without the seed, so a breakpoint
    behind a seeded opening would write an entry nobody reads."""

    from finesub.llm.client import RoleClient
    from finesub.llm.rate_limit import ModelRateLimiter
    from finesub.llm.routing.config import LLMRole, role_config_for
    from finesub.llm.routing.model_catalog import load_effective_catalog
    from finesub.llm.routing.model_router import ModelRouter
    from finesub.llm.routing.model_routes import load_model_routes

    catalog_path = tmp_path / "model_catalog.toml"
    catalog_path.write_text(
        """
[models.claude]
provider_tier = "anthro"
provider_kind = "anthropic"
base_url = "https://api.anthropic.com"
api_model_id = "claude-haiku-5-5"
max_input_tokens = 200000
max_output_tokens = 32000
thinking = false
capability = "low"
""",
        encoding="utf-8",
    )
    routes = load_model_routes(
        user_config={
            "preset": "mine",
            "model_groups": {"g": {"targets": ["claude"]}},
            "presets": {"mine": {"name": "mine", "bindings": {"correction-text/quality": "g"}}},
        },
        catalog=load_effective_catalog(catalog_path),
    )
    sent = []

    def fake_anthropic(**kwargs):
        sent.append(kwargs)
        return OK_BODY

    monkeypatch.setattr(provider_transports, "anthropic_generate", fake_anthropic)
    monkeypatch.setenv("FINESUB_KEY_ANTHRO", "sk-test")
    client = RoleClient(
        router=ModelRouter(routes),
        max_retries=0,
        rate_limiter=ModelRateLimiter(enabled=False),
        role_configs={LLMRole.AUDIO_MULTIMODAL: role_config_for("correction-text", "quality", routes=routes)},
    )

    client.complete(LLMRole.AUDIO_MULTIMODAL, MESSAGES, followup_expected=True)
    client.complete(LLMRole.AUDIO_MULTIMODAL, MESSAGES, followup_expected=True, seed=3)

    assert [call["followup_expected"] for call in sent] == [True, False]
    assert sent[1]["messages"][-1]["content"].endswith("(seed=3)")


def test_openai_compat_cache_writes_are_split_out_of_the_input() -> None:
    from finesub.llm.client import extract_token_distribution

    dist = extract_token_distribution(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 600, "cache_write_tokens": 300},
            }
        }
    )

    assert dist["total_input_tokens"] == 1000
    assert dist["cached_input_tokens"] == 600
    assert dist["cache_write_input_tokens"] == 300
    assert dist["uncached_input_tokens"] == 100
