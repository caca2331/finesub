from __future__ import annotations

import json

from finesub.llm.chunking import SubtitleSegment
from finesub.llm.client import LLMCallResult, RoleClient
from finesub.llm.routing.config import (
    CapabilityTier,
    GEMINI_25_FLASH,
    LLMRole,
    thinking_budget_for_level,
)
from finesub.llm.rate_limit import ModelRateLimiter
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.output_protocol import validate_translated_csv_text
from finesub.llm.routing.profiles import resolve_profile
from finesub.subtitles.model import parse_srt
from finesub.llm.stages.correction import correction_role_for_profile


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _stable_json(tmp_path):
    path = tmp_path / "clip-stable.json"
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.5, "end": 2.5, "text": "二。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_correction_role_selection_per_profile() -> None:
    assert correction_role_for_profile(resolve_profile("audio", "local", "quality")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("video", "local", "quality")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "local", "quality")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "none", "efficiency")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "none", "quality")) is LLMRole.AUDIO_MULTIMODAL
    # text-high too: native search is a per-call capability, not a role.
    assert correction_role_for_profile(resolve_profile("text", "native", "quality")) is LLMRole.AUDIO_MULTIMODAL


def test_validator_rejects_insert_rows_on_every_route() -> None:
    """v63 retired inserts: rejection no longer depends on having audio.

    This used to assert that inserts were allowed with audio and rejected
    without it, via an `allow_insert` switch. Both production call sites had
    already pinned it to False, so the permissive half was unreachable and the
    switch is gone.
    """
    segments = [SubtitleSegment("1", 0.0, 1.0, "一。")]
    text = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|one|一|8|1|\ninsert|0.5,0.4|0.4|0.0|two|二|5|1|\n</translated>"

    result = validate_translated_csv_text(text, segments, require_singles=False)

    assert not result.ok
    assert any("insert" in error for error in result.errors)


def test_text_route_runs_without_audio_search_or_query_round(tmp_path, monkeypatch) -> None:
    stable_json = _stable_json(tmp_path)
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append((role, kwargs))
            return LLMCallResult(
                # The efficiency cell serves basicB, whose CSV carries the start
                # column; the real client reports the served variant on the
                # result, so the fake does too (validation reads it back).
                content=(
                    "<translated>\ntype|position|start|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|0.0|1.0|0.5|one|一|high|1|\n"
                    "sub|2|1.5|1.0|0.0|two|二|high|1|\n</translated>"
                    "\n<next_advice></next_advice>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                variant="basicB",
            )

    class ExplodingSearchClient:
        def search_many(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("text route must not run the local search agent")

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        search_client=ExplodingSearchClient(),
        profile=resolve_profile("text", "none", "efficiency"),
    )

    # Single correction call on audio_multimodal (3.7-first) with the low
    # thinking override; no query round happened.
    assert len(calls) == 1
    role, kwargs = calls[0]
    assert role is LLMRole.AUDIO_MULTIMODAL
    # Thinking is the preset knob now: the packaged default's correction
    # efficiency knob carries "low"; no per-call override kwargs.
    assert "thinking_level" not in kwargs
    from finesub.llm.routing.config import role_config_for

    assert role_config_for("correction-text", "efficiency").thinking_level == "low"
    assert kwargs["file_ref"] is None
    assert [segment.text for segment in parse_srt(output.read_text(encoding="utf-8"))] == [
        "一",
        "二",
    ]


def test_text_route_retries_when_model_emits_insert(tmp_path, monkeypatch) -> None:
    stable_json = _stable_json(tmp_path)
    responses = [
        # First attempt sneaks in an insert row -> structural error -> retry.
        (
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
            "sub|1|1.0|0.0|one|一|8|1|译1字；宜保持独立\n"
            "sub|2|1.0|0.0|two|二|8|1|译1字；宜保持独立\n"
            "</singles>\n"
            "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|one|一|8|1|\ninsert|0.5,0.4|0.4|0.0|x|插|5|1|\nsub|2|1.0|0.0|two|二|8|1|\n</translated>"
        ),
        (
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
            "sub|1|1.0|0.0|one|一|8|1|译1字；宜保持独立\n"
            "sub|2|1.0|0.0|two|二|8|1|译1字；宜保持独立\n"
            "</singles>\n"
            "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|one|一|8|1|\nsub|2|1.0|0.0|two|二|8|1|\n</translated>"
        ),
    ]
    attempts = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            attempts.append(role)
            return LLMCallResult(
                content=responses[len(attempts) - 1],
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
    )

    assert len(attempts) == 2
    assert "插" not in output.read_text(encoding="utf-8")


def test_native_search_capability_filters_within_the_bound_group(monkeypatch) -> None:
    """``complete(native_search=True)`` is a per-call filter (plan v2 D4).

    The role stays ``audio_multimodal`` -- it names the job. Asking for the
    capability filters the bound group down to the members that can ground
    (paid 3.7), instead of switching to some other chain; without the
    capability the same role keeps its 3.7-first
    group; a test profile never enables the tool.
    """

    captured = {}

    def fake_chat_complete(messages, *, model, native_search_tool=None, **kwargs):
        captured["model"] = model
        captured["native_search_tool"] = native_search_tool
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr("finesub.llm.llm_runtime.chat_complete", fake_chat_complete)
    client = RoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        native_search=True,
    )
    assert captured["native_search_tool"] == "google_search"
    assert result.target_id == "gemini-paid-3_7-flash"

    # Without the capability the same role keeps its own 3.7-first group.
    client.complete(LLMRole.AUDIO_MULTIMODAL, [{"role": "user", "content": "hi"}])
    assert captured["native_search_tool"] is None
    assert captured["model"] != GEMINI_25_FLASH

    # Test profile never enables the tool.
    captured.clear()
    test_client = RoleClient(
        test_profile=True,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    test_client.complete(
        LLMRole.AUDIO_MULTIMODAL,
        [{"role": "user", "content": "hi"}],
        native_search=True,
    )
    assert captured["native_search_tool"] is None


def _kb_with_two_entries(tmp_path):
    root = tmp_path / "kb"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n- 主播B | ビーちゃん | 另一个\n",
        encoding="utf-8",
    )
    (root / "common" / "index.md").write_text("", encoding="utf-8")
    for key in ("主播A", "主播B"):
        (root / "streamer" / f"{key}.md").write_text(
            f"# {key}\n\n资料。\n", encoding="utf-8"
        )
    return root


_TWO_WINDOW_CSV = (
    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
    "sub|1|1.0|0.0|one|一|8|1|译1字；宜保持独立\n"
    "sub|2|1.0|0.0|two|二|8|1|译1字；宜保持独立\n"
    "</singles>\n"
    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
    "sub|1|1.0|0.0|one|一|8|1|\nsub|2|1.0|0.0|two|二|8|1|\n</translated>\n"
    "<next_advice></next_advice>\n"
    "<keep_entries></keep_entries>"
)


def test_window_without_a_query_round_keeps_every_entry(tmp_path, monkeypatch) -> None:
    """An empty <keep_entries> must not drop the chain when nothing can re-request.

    With no per-window query round, a dropped entry is gone for the whole run,
    so the harness transfers the current set instead of honouring the model's
    pruning (docs/llm_harness_behavior.md).
    """

    stable_json = _stable_json(tmp_path)
    seen_entry_blocks = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            # The correction call passes a per-tier composer, not a message list.
            built = messages("capableC") if callable(messages) else messages
            seen_entry_blocks.append(built[-1]["content"])
            return LLMCallResult(
                content=_TWO_WINDOW_CSV,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        knowledge_root=_kb_with_two_entries(tmp_path),
        initial_transfer_keys=["主播A", "主播B"],
        profile=resolve_profile("text", "none", "quality"),
    )

    assert "主播A" in seen_entry_blocks[0]
    assert "主播B" in seen_entry_blocks[0]


def test_window_without_a_query_round_is_not_asked_to_prune(tmp_path) -> None:
    """The prompt must not request a block the harness ignores."""

    from finesub.llm.prompt_compose import compose_correction_system

    injected = compose_correction_system(resolve_profile("audio", "local", "quality"))
    assert "<keep_entries>" in injected

    for switches in (("text", "none", "quality"), ("text", "native", "quality")):
        system = compose_correction_system(resolve_profile(*switches))
        assert "<keep_entries>" not in system, switches

    # No knowledge base at all: nothing to keep either way.
    assert "<keep_entries>" not in compose_correction_system(
        resolve_profile("audio", "local", "quality"), knowledge_enabled=False
    )


def _kb_with_index(tmp_path):
    root = tmp_path / "kb"
    (root / "streamer").mkdir(parents=True)
    (root / "common").mkdir(parents=True)
    (root / "streamer" / "index.md").write_text(
        "- 主播A | エーちゃん | 测试主播\n", encoding="utf-8"
    )
    (root / "streamer" / "主播A.md").write_text("# 主播A\n\n资料。\n", encoding="utf-8")
    (root / "common" / "index.md").write_text("", encoding="utf-8")
    return root


def test_knowledge_none_stops_the_correction_side_reading_too(tmp_path, monkeypatch) -> None:
    """`--knowledge none` must not inject indices or entry rules per window.

    The research half honoured the tri-state while the correction half derived
    its own flag from whether index files existed on disk, so a populated
    knowledge base was still read and injected into every query round.
    """

    stable_json = _stable_json(tmp_path)
    knowledge_root = _kb_with_index(tmp_path)
    seen = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            built = messages("capableC") if callable(messages) else messages
            seen.append(built[0]["content"] + built[-1]["content"])
            return LLMCallResult(
                content=(
                    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|0.0|one|一|8|1|译1字；宜保持独立\n"
                    "sub|2|1.0|0.0|two|二|8|1|译1字；宜保持独立\n</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|0.0|one|一|8|1|\nsub|2|1.0|0.0|two|二|8|1|\n</translated>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        knowledge_root=knowledge_root,
        knowledge_enabled=False,
        profile=resolve_profile("audio", "local", "quality"),
    )

    prompt = "\n".join(seen)
    assert "主播A" not in prompt, "the index leaked into the correction prompt"
    assert "<keep_entries>" not in prompt, "pruning was requested with knowledge off"

    # With the switch on, the same base does reach the prompt.
    seen.clear()
    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out2.srt",
        token_counter=FakeTokenCounter(),
        knowledge_root=knowledge_root,
        knowledge_enabled=True,
        profile=resolve_profile("audio", "local", "quality"),
    )
    assert "<keep_entries>" in "\n".join(seen)


def test_a_direct_call_still_names_where_its_agent_evidence_goes(
    tmp_path, monkeypatch
) -> None:
    """`task_artifact_dir` is optional, and the fallback used to be "nowhere".

    A conversational session's kept exchanges live inside an assignment tree
    that a clean finish clears, so the scope has to be told where to file them
    before it closes. A run names the directory; a stage entered directly does
    not, and that is precisely the case with no other record of what was asked
    and answered -- so the derived artifact directory has to count, the same
    one the window clips already fall back to.
    """

    stable_json = _stable_json(tmp_path)
    named: list = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=(
                    "<translated>\ntype|position|start|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|0.0|1.0|0.5|one|一|high|1|\n"
                    "sub|2|1.5|1.0|0.0|two|二|high|1|\n</translated>"
                    "\n<next_advice></next_advice>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                variant="basicB",
            )

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)
    monkeypatch.setattr(
        "finesub.llm.stages.correction.run.set_run_evidence_destination",
        named.append,
    )

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "efficiency"),
    )

    from finesub_bootstrap.artifacts import ARTIFACT_DIR_SUFFIX

    assert named == [(tmp_path / "out.srt").with_suffix(ARTIFACT_DIR_SUFFIX)]


def test_a_named_artifact_directory_still_wins(tmp_path, monkeypatch) -> None:
    """The derivation is a fallback, not an override."""

    stable_json = _stable_json(tmp_path)
    named: list = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=(
                    "<translated>\ntype|position|start|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|0.0|1.0|0.5|one|一|high|1|\n"
                    "sub|2|1.5|1.0|0.0|two|二|high|1|\n</translated>"
                    "\n<next_advice></next_advice>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                variant="basicB",
            )

    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)
    monkeypatch.setattr(
        "finesub.llm.stages.correction.run.set_run_evidence_destination",
        named.append,
    )

    artifacts = tmp_path / "chosen"
    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        task_artifact_dir=artifacts,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "efficiency"),
    )

    assert named == [artifacts]
