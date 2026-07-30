from __future__ import annotations

import json

from llm.chunking import SubtitleSegment
from llm.client import LLMCallResult, LiteLLMRoleClient
from llm.config import LLMRole, thinking_budget_for_level
from llm.rate_limit import ModelRateLimiter
from llm.stages.correction_loop import execute_correction_windows
from llm.csv_utils import validate_translated_csv_text
from llm.profiles import resolve_profile
from asr_playground.subtitles.model import parse_srt
from llm.stages.correction_loop import correction_role_for_profile


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
    assert correction_role_for_profile(resolve_profile("mm", "med")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("mm", "high")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("mm", "low")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "low")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "med")) is LLMRole.AUDIO_MULTIMODAL
    assert correction_role_for_profile(resolve_profile("text", "high")) is LLMRole.INTERNET_CAPABLE


def test_validator_rejects_insert_rows_when_audio_less() -> None:
    segments = [SubtitleSegment("1", 0.0, 1.0, "一。")]
    text = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|one|一|8|\ninsert|0.5,0.4|0.4|two|二|5|\n</translated>"

    with_audio = validate_translated_csv_text(text, segments, allow_insert=True, require_singles=False)
    without_audio = validate_translated_csv_text(text, segments, allow_insert=False, require_singles=False)

    assert with_audio.ok
    assert not without_audio.ok
    assert any("insert" in error for error in without_audio.errors)


def test_text_route_runs_without_audio_search_or_query_round(tmp_path, monkeypatch) -> None:
    stable_json = _stable_json(tmp_path)
    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append((role, kwargs))
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
        def search_many(self, *args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("text route must not run the local search agent")

    monkeypatch.setattr("llm.stages.correction_loop.LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        enable_web_search=True,
        search_client=ExplodingSearchClient(),
        profile=resolve_profile("text", "low"),
    )

    # Single correction call on audio_multimodal (3.6-first) with the low
    # thinking override; no query round happened.
    assert len(calls) == 1
    role, kwargs = calls[0]
    assert role is LLMRole.AUDIO_MULTIMODAL
    assert kwargs["thinking_level"] == "low"
    assert kwargs["thinking_budget"] == thinking_budget_for_level("low")
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
            "sub|1|1.0|one|一|8|译1字；宜保持独立\n"
            "sub|2|1.0|two|二|8|译1字；宜保持独立\n"
            "</singles>\n"
            "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|one|一|8|\ninsert|0.5,0.4|0.4|x|插|5|\nsub|2|1.0|two|二|8|\n</translated>"
        ),
        (
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
            "sub|1|1.0|one|一|8|译1字；宜保持独立\n"
            "sub|2|1.0|two|二|8|译1字；宜保持独立\n"
            "</singles>\n"
            "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|one|一|8|\nsub|2|1.0|two|二|8|\n</translated>"
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

    monkeypatch.setattr("llm.stages.correction_loop.LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        enable_web_search=False,
        profile=resolve_profile("text", "med"),
    )

    assert len(attempts) == 2
    assert "插" not in output.read_text(encoding="utf-8")


def test_internet_capable_role_enables_native_search_tool(monkeypatch) -> None:
    captured = {}

    def fake_chat_complete(messages, *, model, native_search_tool=None, **kwargs):
        captured["model"] = model
        captured["native_search_tool"] = native_search_tool
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        }

    monkeypatch.setattr("llm.llm_runtime.chat_complete", fake_chat_complete)
    client = LiteLLMRoleClient(rate_limiter=ModelRateLimiter(enabled=False))

    result = client.complete(
        LLMRole.INTERNET_CAPABLE, [{"role": "user", "content": "hi"}]
    )
    assert result.content == "ok"
    assert captured["native_search_tool"] == "google_search"

    # Test profile never enables the tool.
    captured.clear()
    test_client = LiteLLMRoleClient(
        test_profile=True,
        rate_limiter=ModelRateLimiter(enabled=False),
    )
    test_client.complete(LLMRole.INTERNET_CAPABLE, [{"role": "user", "content": "hi"}])
    assert captured["native_search_tool"] is None
