from __future__ import annotations

from dataclasses import replace as dataclass_replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import finesub.llm.correction_translation as correction_orchestration
import finesub.llm.research as research
from finesub.llm.stages.correction import attempts as correction_attempts
from finesub.llm.stages.correction import commit as correction_commit
from finesub.llm.stages.correction import context as correction_context
from finesub.llm.stages.correction import run as correction_run

from .conftest import setattr_correction

from finesub.llm.client import LLMCallResult, UploadedFileRef
from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.routing.config import CapabilityTier, LLMRole
from finesub.llm.stages.correction import execute_correction_windows, run_window_query_round
from finesub.llm.stages.correction.attempts import (
    _extract_next_advice,
    _extract_task_update_feedback,
)
from finesub.llm.stages.correction.metadata import _is_output_limited, _output_limit_check
from finesub.llm.stages.correction.query_round import _extract_window_notes
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.prompts import ContextPack, render_advice_ledger
from finesub.subtitles.model import parse_srt


def _setattr_both(monkeypatch, name, value):
    """Patch a name across the window loop and the orchestration module."""

    setattr_correction(monkeypatch, name, value)



class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _write_research_context(path: Path, planning: dict, summary: str = "old") -> None:
    path.write_text(
        json.dumps(
            {
                "context_pack": {
                    "general_context": {"summary": summary},
                    "window_contexts": [{
                        "window_id": "0001",
                        "first_source_id": "1",
                        "last_source_id": "2",
                        "context": "窗口背景",
                    }],
                },
                "planning": planning,
            }
        ),
        encoding="utf-8",
    )


def test_research_context_survives_model_and_knowledge_changes(tmp_path) -> None:
    """L3 whitelist: how it was produced is audit, not an invalidation key."""

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    context = tmp_path / "clip-research-context.json"
    saved = {
        "prompt_version": "v1",
        "geometry_profile_id": "correction_media=audio,...",
        "stable_json_hash": "sha256:source",
        "extra_info_hash": "sha256:notes",
        "search_rounds": 2,
        # Everything below is exempt.
        "execution_identity": {"routing_identity_digest": "old-model-group"},
        "knowledge_inputs_hash": "sha256:kb-before-someone-elses-update",
        "knowledge_enabled": True,
        "collect_task_feedback": False,
        "profile_id": "…,difficulty=quality,continuity=serial",
    }
    _write_research_context(context, saved, summary="旧模型已完成")

    current = dict(
        saved,
        execution_identity={"routing_identity_digest": "new-model-group"},
        knowledge_inputs_hash="sha256:kb-after",
        collect_task_feedback=True,
        profile_id="…,difficulty=intermediate,continuity=parallel",
    )
    loaded = correction_orchestration._load_reusable_research_context(
        context, expected_planning=current
    )

    assert loaded is not None
    assert loaded.general_context["summary"] == "旧模型已完成"
    assert loaded.window_contexts[0].context == "窗口背景"


@pytest.mark.parametrize(
    "field, value",
    [
        ("stable_json_hash", "sha256:other-source"),
        ("prompt_version", "v2"),
        ("extra_info_hash", "sha256:new-notes"),
        ("search_rounds", 3),
        ("research_semantics_id", "native"),
    ],
)
def test_research_context_reuse_rejects_gated_changes(tmp_path, field, value) -> None:
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    context = tmp_path / "clip-research-context.json"
    saved = {
        "prompt_version": "v1",
        "geometry_profile_id": "correction_media=audio,...",
        "stable_json_hash": "sha256:source",
        "extra_info_hash": "sha256:notes",
        "search_rounds": 2,
    }
    _write_research_context(context, saved)

    assert (
        correction_orchestration._load_reusable_research_context(
            context, expected_planning={**saved, field: value}
        )
        is None
    )


def test_research_context_reuse_backs_up_corrupt_file(tmp_path) -> None:
    context = tmp_path / "clip-research-context.json"
    context.write_text("{broken", encoding="utf-8")
    assert (
        correction_orchestration._load_reusable_research_context(
            context, expected_planning={"stable_json_hash": "sha256:x"}
        )
        is None
    )
    assert (tmp_path / "clip-research-context.json.invalid").read_text(
        encoding="utf-8"
    ) == "{broken"



def test_output_limited_detection_ignores_finish_reason_and_content_shape() -> None:
    response = {
        "candidates": [{"finishReason": "MAX_TOKENS"}],
        "usageMetadata": {
            "candidatesTokenCount": 400,
            "thoughtsTokenCount": 800,
        },
    }

    assert not _is_output_limited(response, 65_536)


def test_output_limited_detection_uses_usage_metadata_near_cap() -> None:
    near_cap = {
        "usageMetadata": {"candidatesTokenCount": 60_000, "thoughtsTokenCount": 5_500}
    }
    far_from_cap = {
        "usageMetadata": {"candidatesTokenCount": 400, "thoughtsTokenCount": 800}
    }

    assert _is_output_limited(near_cap, 65_536)
    assert not _is_output_limited(far_from_cap, 65_536)
    check = _output_limit_check(near_cap, 65_536)
    assert check == {
        "basis": "output_tokens_plus_thinking_tokens",
        "visible_output_tokens": 60_000,
        "thinking_tokens": 5_500,
        "observed_output_tokens": 65_500,
        "max_output_tokens": 65_536,
        "margin_tokens": 100,
        "threshold_tokens": 65_436,
        "limited": True,
    }


def test_agent_only_correction_media_stays_local_without_gemini_upload(
    tmp_path, monkeypatch
) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "finesub.llm.routing.execution_policy.load_execution_settings",
        lambda: ExecutionSettings(policy_id="agent-only"),
    )
    # The policy alone no longer reaches an agent -- it only subtracts
    # backends. `agy` is the packaged preset whose groups name one, and it
    # reaches the clip decision through the *client's own* router rather than
    # the process default, so the two tables cannot disagree.
    from finesub.llm.routing import model_routes

    agy_routes = model_routes.load_model_routes(user_config={"preset": "agy"})
    _setattr_both(monkeypatch, "probe_audio_duration", lambda _: 1.0)
    uploads = []
    monkeypatch.setattr(
        "finesub.llm.client.upload_gemini_file",
        lambda path: uploads.append(path),
    )

    def fake_extract(_source, _start, _end, out_path, **_kwargs):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"test-aac")
        return out

    _setattr_both(monkeypatch, "extract_window_clip", fake_extract)
    seen_refs = []
    content = (
        "<singles>\n"
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一。|8|2|译2字；宜保持独立\n"
        "</singles>\n"
        "<translated>\n"
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一。|high|2|\n"
        "</translated>\n"
        "<next_advice></next_advice>"
    )

    class FakeClient:
        execution_settings = ExecutionSettings(policy_id="agent-only")
        router = SimpleNamespace(routes=agy_routes)

        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            seen_refs.append(kwargs["file_ref"])
            return LLMCallResult(
                content=content,
                role=role,
                model="local-agy-media-gemini-3_7-flash",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                target_id="local-agy-media-gemini-3_7-flash",
                backend="local_agent",
            )

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "none", "quality"),
    )

    assert len(seen_refs) == 1
    assert seen_refs[0].local_path
    assert seen_refs[0].file_id == ""
    assert uploads == []


def test_extract_task_update_feedback_reads_first_feedback_block() -> None:
    content = (
        "<singles>\nsub|1|1.0|0.0|x|1|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\nsub|1|1.0|0.0|good|好|high|1|\n</translated>\n"
        '<task_update_feedback>{"reusable_terms":["A 固定译为甲"]}</task_update_feedback>'
    )

    assert _extract_task_update_feedback(content) == '{"reusable_terms":["A 固定译为甲"]}'


def test_extract_next_advice_reads_block_and_caps_length() -> None:
    content = (
        "<singles>\nsub|1|1.0|0.0|x|1|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\nsub|1|1.0|0.0|good|好|high|1|\n</translated>\n"
        "<next_advice>术语X固定译为Y。</next_advice>"
    )

    assert _extract_next_advice(content) == "术语X固定译为Y。"
    assert _extract_next_advice("<translated></translated>") == ""
    # Cap is token-based now: with a 1-token-per-char counter the 2000-char
    # body must land inside the truncation window just below 800.
    long_content = f"<next_advice>{'字' * 2000}</next_advice>"
    capped = _extract_next_advice(long_content, count_tokens=len)
    assert 700 <= len(capped) <= 800


def test_extract_window_notes_is_best_effort() -> None:
    assert (
        _extract_window_notes("<window_notes>疑似BOSS名（待定）</window_notes>")
        == "疑似BOSS名（待定）"
    )
    assert _extract_window_notes("没有标签块") == ""
    duplicated = "<window_notes>甲</window_notes><window_notes>乙</window_notes>"
    assert _extract_window_notes(duplicated) == ""
    long_content = f"<window_notes>{'字' * 2000}</window_notes>"
    capped = _extract_window_notes(long_content, count_tokens=len)
    assert 700 <= len(capped) <= 800


def test_render_advice_ledger_labels_windows_and_skips_empty() -> None:
    rendered = render_advice_ledger(
        [
            ("0001", "术语X固定译为Y。"),
            ("0002", "   "),
            ("0003-a", "说话人开始疲惫。"),
        ]
    )

    assert rendered == "[window 0001]\n术语X固定译为Y。\n\n[window 0003-a]\n说话人开始疲惫。"
    assert render_advice_ledger([]) == ""


def test_task_update_feedback_is_requested_and_retained(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "A。"}]}),
        encoding="utf-8",
    )
    content = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|A|甲|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|A|甲|high|1|\n"
        "</translated>\n"
        "<next_advice></next_advice>\n"
        "<task_update_feedback>"
        '{"reusable_terms":["A 固定译为甲"],"asr_corrections":[],"context_clues":[],"uncertainties":[]}'
        "</task_update_feedback>"
    )
    seen_systems = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            seen_systems.append(messages[0]["content"])
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={
                    "candidates": [{"finishReason": "STOP"}],
                    "usageMetadata": {
                        "promptTokenCount": 123,
                        "candidatesTokenCount": 45,
                        "totalTokenCount": 168,
                    },
                    "modelVersion": "fake-version",
                },
            )

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
        task_id="task-1",
        task_update_feedback=True,
    )

    assert "task_update_feedback" in seen_systems[0]
    assert "<task_update_feedback>" not in output.read_text(encoding="utf-8")
    assert parse_srt(output.read_text(encoding="utf-8"))[0].text == "甲"
    assert parse_srt((tmp_path / "out-translated.srt").read_text(encoding="utf-8"))[0].text == "甲"
    assert parse_srt((tmp_path / "out-corrected.srt").read_text(encoding="utf-8"))[0].text == "A"
    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    feedback_artifacts = [
        artifact for artifact in artifacts if artifact["kind"] == "correction_window_task_feedback"
    ]

    assert feedback_artifacts[0]["payload"]["feedback"].startswith('{"reusable_terms"')
    assert artifacts[0]["payload"]["task_update_feedback"].startswith('{"reusable_terms"')
    response_payload = artifacts[0]["payload"]
    assert response_payload["window"]["source_id_range"] == ["1", "1"]
    assert response_payload["request"]["message_text_chars"] > 0
    assert response_payload["request"]["message_fingerprints"][0]["text_sha256"]
    assert response_payload["provider"]["usageMetadata"]["totalTokenCount"] == 168
    assert response_payload["provider"]["modelVersion"] == "fake-version"
    assert response_payload["response"]["content_chars"] > 0


def test_validation_error_retries_same_window_without_splitting(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
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
    bad = "<translated>\nsub|9|1.0|0.0|nine|九|high|1|\n</translated>"
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二。|二|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|high|1|\n"
        "sub|2|1.0|0.0|二。|二|high|1|\n"
        "</translated>"
    )
    responses = [bad, good]
    sampling_kwargs: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            self.calls += 1
            sampling_kwargs.append(
                {
                    "temperature": kwargs.get("temperature"),
                    "seed": kwargs.get("seed"),
                }
            )
            return LLMCallResult(
                content=responses.pop(0),
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=1,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
    )
    segments = parse_srt(output.read_text(encoding="utf-8"))

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retries = [artifact for artifact in artifacts if artifact["kind"] == "correction_window_retry"]

    assert retries[0]["payload"]["reason"] == "validation_same_window"
    assert retries[0]["payload"]["retry_chunk_id"] == "0001"
    assert retries[0]["payload"]["tail_chunk_ids"] == []
    assert retries[0]["payload"]["failed_window"]["source_id_range"] == ["1", "2"]
    assert retries[0]["payload"]["retry_window"]["source_id_range"] == ["1", "2"]
    assert [item["temperature"] for item in sampling_kwargs] == [1.0, 0.99]
    assert sampling_kwargs[0]["seed"] != sampling_kwargs[1]["seed"]
    assert [segment.text for segment in segments] == ["一", "二"]
    assert (tmp_path / "out-raw.srt").exists()
    assert [
        segment.text
        for segment in parse_srt(
            (tmp_path / "out-corrected.srt").read_text(encoding="utf-8")
        )
    ] == [
        "一。",
        "二。",
    ]
    assert (tmp_path / "out-translated.srt").exists()


def _two_segment_stable_json(path):
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


def test_a_validation_retry_hands_back_the_output_and_the_reasons(
    tmp_path, monkeypatch
) -> None:
    """The retry after a validation failure is a repair round, not a re-throw.

    Every retry used to re-send the identical prompt at a new temperature, so a
    model that had written the right content and slipped on one mechanical rule
    got no way to learn which rule -- 605-row window 0001 of BV1ojjc6MEAs cost
    six calls and a day's free quota that way.
    """

    stable_json = _two_segment_stable_json(tmp_path / "clip-stable.json")
    bad = "<translated>\nsub|9|1.0|0.0|nine|九|high|1|\n</translated>"
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二。|二|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|high|1|\n"
        "sub|2|1.0|0.0|二。|二|high|1|\n"
        "</translated>"
    )
    responses = [bad, good]
    repair_inputs: list[tuple[str, list[str]]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            repair_inputs.append(
                (
                    kwargs.get("previous_output", ""),
                    list(kwargs.get("validation_errors", ())),
                )
            )
            return LLMCallResult(
                content=responses.pop(0),
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=1,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
    )

    assert repair_inputs[0] == ("", [])
    previous, errors = repair_inputs[1]
    assert previous == bad
    assert errors, "the second attempt must be told why the first was rejected"

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retries = [a for a in artifacts if a["kind"] == "correction_window_retry"]
    responses_logged = [
        a for a in artifacts if a["kind"] == "correction_window_response"
    ]
    assert retries[0]["payload"]["repair_context"] is True
    assert [row["payload"]["repair_round"] for row in responses_logged] == [
        False,
        True,
    ]


def test_a_split_window_does_not_inherit_the_previous_output(
    tmp_path, monkeypatch
) -> None:
    """Repair context is about *this* window's rows.

    A window that got split is a different task: last round's answer covers
    rows this attempt is not being asked about, so handing it over is worse
    than saying nothing.
    """

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.2, "end": 2.0, "text": "二"},
                    {"id": "3", "start": 2.2, "end": 3.0, "text": "三。"},
                    {"id": "4", "start": 3.2, "end": 4.0, "text": "四"},
                    {"id": "5", "start": 4.2, "end": 5.0, "text": "五。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    truncated = (
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|high|1|\n"
    )
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二|二|8|1|译1字；宜保持独立\n"
        "sub|3|1.0|0.0|三。|三|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|high|1|\nsub|2|1.0|0.0|二|二|high|1|\n"
        "sub|3|1.0|0.0|三。|三|high|1|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|四|四|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|五。|五|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|四|四|high|1|\nsub|2|1.0|0.0|五。|五|high|1|\n</translated>"
    )
    responses = [
        ("MAX_TOKENS", truncated),
        ("STOP", first_half),
        ("STOP", second_half),
    ]
    repair_inputs: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            repair_inputs.append(kwargs.get("previous_output", ""))
            finish, content = responses.pop(0)
            usage = (
                {"candidatesTokenCount": 60_000, "thoughtsTokenCount": 5_500}
                if finish == "MAX_TOKENS"
                else {"candidatesTokenCount": 100, "thoughtsTokenCount": 50}
            )
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={
                    "candidates": [{"finishReason": finish}],
                    "usageMetadata": usage,
                },
            )

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
    )

    # The truncated attempt failed on size, not on a rule it could be told
    # about, and the two halves that follow are different windows.
    assert repair_inputs == ["", "", ""]


def test_output_limited_splits_window_in_half_and_forwards_advice(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.2, "end": 2.0, "text": "二"},
                    {"id": "3", "start": 2.2, "end": 3.0, "text": "三。"},
                    {"id": "4", "start": 3.2, "end": 4.0, "text": "四"},
                    {"id": "5", "start": 4.2, "end": 5.0, "text": "五。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二|二|8|1|译1字；宜保持独立\n"
        "sub|3|1.0|0.0|三。|三|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\nsub|2|1.0|0.0|二|二|high|1|\nsub|3|1.0|0.0|三。|三|high|1|\n</translated>\n"
        "<next_advice>术语X固定译为Y。</next_advice>"
    )
    # Dense 1s spacing with the default overlap floor (10) exceeds the first
    # half's length, so the -b half falls back to no overlap: it starts at "4".
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|四|四|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|五。|五|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|四|四|high|1|\nsub|2|1.0|0.0|五。|五|high|1|\n</translated>\n"
        "<next_advice></next_advice>"
    )
    responses = [
        ("MAX_TOKENS", truncated),
        ("STOP", first_half),
        ("STOP", second_half),
    ]
    seen_messages = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            seen_messages.append(messages)
            finish, content = responses.pop(0)
            usage = (
                {"candidatesTokenCount": 60_000, "thoughtsTokenCount": 5_500}
                if finish == "MAX_TOKENS"
                else {"candidatesTokenCount": 100, "thoughtsTokenCount": 50}
            )
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={
                    "candidates": [{"finishReason": finish}],
                    "usageMetadata": usage,
                },
            )

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
    )
    segments = parse_srt(output.read_text(encoding="utf-8"))

    assert [segment.text for segment in segments] == ["一", "二", "三", "四", "五"]
    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    retries = [artifact for artifact in artifacts if artifact["kind"] == "correction_window_retry"]
    responses_artifacts = [
        artifact
        for artifact in artifacts
        if artifact["kind"] == "correction_window_response"
    ]
    assert retries[0]["payload"]["reason"] == "output_limited_split_in_half"
    assert retries[0]["payload"]["finish_reason"] == "MAX_TOKENS"
    assert retries[0]["payload"]["output_limit_check"]["limited"] is True
    assert retries[0]["payload"]["retry_chunk_id"] == "0001-a"
    assert retries[0]["payload"]["tail_chunk_ids"] == ["0001-b"]
    assert responses_artifacts[0]["payload"]["finish_reason"] == "MAX_TOKENS"
    assert responses_artifacts[0]["payload"]["output_limit_check"][
        "observed_output_tokens"
    ] == 65_500

    first_exchange = sorted((tmp_path / "artifacts" / "exchanges").glob("*.md"))[0]
    exchange_text = first_exchange.read_text(encoding="utf-8")
    assert "- finish_reason: MAX_TOKENS" in exchange_text
    # One folded line, not four near-identical ones.
    assert (
        "- output_limit: observed 65500 / threshold 65436 / max 65536 "
        "(basis: output_tokens_plus_thinking_tokens)"
    ) in exchange_text

    # The second half receives the first half's advice via the cumulative
    # ledger (labelled with the emitting window); earlier calls see none.
    assert "术语X固定译为Y。" not in seen_messages[1][1]["content"]
    assert "[window 0001-a]\n术语X固定译为Y。" in seen_messages[2][1]["content"]


def test_each_executed_window_gets_its_own_clip_upload(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.2, "end": 2.0, "text": "二"},
                    {"id": "3", "start": 2.2, "end": 3.0, "text": "三。"},
                    {"id": "4", "start": 3.2, "end": 4.0, "text": "四"},
                    {"id": "5", "start": 4.2, "end": 5.0, "text": "五。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二|二|8|1|译1字；宜保持独立\n"
        "sub|3|1.0|0.0|三。|三|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\nsub|2|1.0|0.0|二|二|high|1|\nsub|3|1.0|0.0|三。|三|high|1|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|四|四|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|五。|五|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|四|四|high|1|\nsub|2|1.0|0.0|五。|五|high|1|\n</translated>"
    )
    responses = [
        ("MAX_TOKENS", truncated),
        ("STOP", first_half),
        ("STOP", second_half),
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            finish, content = responses.pop(0)
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={
                    "candidates": [{"finishReason": finish}],
                    "usageMetadata": (
                        {
                            "candidatesTokenCount": 60_000,
                            "thoughtsTokenCount": 5_500,
                        }
                        if finish == "MAX_TOKENS"
                        else {
                            "candidatesTokenCount": 100,
                            "thoughtsTokenCount": 50,
                        }
                    ),
                },
            )

    extracted: list[str] = []
    uploaded: list[str] = []

    def fake_extract(audio_path, clip_start, clip_end, out_path, **kwargs):
        assert clip_end <= 100.0
        extracted.append(str(out_path))
        return out_path

    def fake_upload(path):
        uploaded.append(str(path))
        return UploadedFileRef(
            file_id=f"files/{len(uploaded)}", filename=str(path), mime_type="audio/aac"
        )

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)
    _setattr_both(
        monkeypatch,
        "probe_audio_duration", lambda _: 100.0)
    _setattr_both(
        monkeypatch,
        "extract_window_clip", fake_extract)
    monkeypatch.setattr("finesub.llm.client.upload_gemini_file", fake_upload)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("audio", "none", "quality"),
    )

    # One clip + upload per executed window, -a/-b halves included.
    assert [Path(p).name for p in extracted] == [
        "0001.aac",
        "0001-a.aac",
        "0001-b.aac",
    ]
    assert uploaded == extracted


def test_same_window_validation_retry_reuses_clip_upload(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    responses = [
        ("STOP", "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|9|1.0|0.0|x|9|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|9|1.0|0.0|坏|坏|high|1|\n</translated>"),
        ("STOP", "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|x|1|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\n</translated>"),
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            finish, content = responses.pop(0)
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": finish}]},
            )

    uploads = []
    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)
    _setattr_both(
        monkeypatch,
        "probe_audio_duration", lambda _: 100.0)
    _setattr_both(
        monkeypatch,
        "extract_window_clip",
        lambda audio_path, clip_start, clip_end, out_path, **kwargs: out_path,
    )
    monkeypatch.setattr(
        "finesub.llm.client.upload_gemini_file",
        lambda path: uploads.append(str(path))
        or UploadedFileRef(file_id="files/1", filename=str(path), mime_type="audio/aac"),
    )

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("audio", "none", "quality"),
    )

    assert len(uploads) == 1


def test_split_second_half_gets_raw_preceding_context(tmp_path, monkeypatch) -> None:
    # v13: continuity is input-only. After a truncation split, the -b half's
    # prompt must carry the read-only raw preceding block (negative times) and
    # no trace of the -a half's corrected output.
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": str(i + 1),
                        "start": i * 40.0,
                        "end": i * 40.0 + 1.0,
                        "text": f"第{i + 1}句。",
                    }
                    for i in range(6)
                ]
            }
        ),
        encoding="utf-8",
    )
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|第一句。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|第二句。|二|8|1|译1字；宜保持独立\n"
        "sub|3|1.0|0.0|第三句。|三|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|第一句。|一|high|1|\nsub|2|1.0|0.0|第二句。|二|high|1|\nsub|3|1.0|0.0|第三句。|三|high|1|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|第四句。|四|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|第五句。|五|8|1|译1字；宜保持独立\n"
        "sub|3|1.0|0.0|第六句。|六|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|第四句。|四|high|1|\nsub|2|1.0|0.0|第五句。|五|high|1|\nsub|3|1.0|0.0|第六句。|六|high|1|\n</translated>"
    )
    responses = [
        ("MAX_TOKENS", truncated),
        ("STOP", first_half),
        ("STOP", second_half),
    ]
    seen_messages = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            seen_messages.append(messages)
            finish, content = responses.pop(0)
            return LLMCallResult(
                content=content,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={
                    "candidates": [{"finishReason": finish}],
                    "usageMetadata": (
                        {
                            "candidatesTokenCount": 60_000,
                            "thoughtsTokenCount": 5_500,
                        }
                        if finish == "MAX_TOKENS"
                        else {
                            "candidatesTokenCount": 100,
                            "thoughtsTokenCount": 50,
                        }
                    ),
                },
            )

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("text", "none", "quality"),
    )

    # Sparse 40s spacing: zero content-driven overlap at the split, so the -b
    # half starts at canonical id 4 (clip_start 115.0). Model-facing target ids
    # restart at 1, while the three raw references are numbered -2,-1,0.
    second_user = seen_messages[2][1]["content"]
    assert "<preceding_context>" in second_user
    assert "0|-35.0|1.0|0.0|第3句。" in second_user
    assert "1|5.0|1.0|" in second_user
    # Decoupled from the -a half's output: its corrected/translated rows never
    # reach the -b prompt (raw text is 第3句。, corrected was 第三句。).
    assert "第三句。" not in second_user
    assert "previous_output_context" not in second_user


def test_provider_error_does_not_split_window(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            raise RuntimeError("HTTP 503 high demand")

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    with pytest.raises(RuntimeError):
        execute_correction_windows(
            stable_json=stable_json,
            output_path=tmp_path / "out.srt",
            token_counter=FakeTokenCounter(),
            max_retries_per_window=1,
            profile=resolve_profile("text", "none", "quality"),
            task_artifact_dir=tmp_path / "artifacts",
        )

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert [artifact["kind"] for artifact in artifacts] == ["correction_window_call_error"]
    payload = artifacts[0]["payload"]
    assert payload["window"]["source_id_range"] == ["1", "1"]
    assert payload["request"]["requested_output_tokens"] == 65_536
    assert payload["request"]["message_text_chars"] > 0


def test_query_round_searches_once_and_injects_results(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
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
    from finesub.llm.web_search import QuerySearchResult, SearchResultItem

    query_output = (
        "<window_notes>\n本窗口疑似在打BOSS，BOSS名待定。\n</window_notes>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 角色名\n</search_queries>"
    )
    bad = "<translated>\nsub|9|1.0|0.0|nine|九|high|1|\n</translated>"
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "sub|2|1.0|0.0|二。|二|8|1|译1字；宜与前一句合并\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|\n"
        "sub|2|1.0|0.0|二。|二|7|1|术语note\n"
        "</translated>"
    )
    correction_responses = [bad, good]
    query_calls = []
    correction_messages_seen = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            if role == LLMRole.LIGHTWEIGHT_MULTIMODAL:
                query_calls.append(messages)
                return LLMCallResult(
                    content=query_output,
                    role=role,
                    model="fake-lite",
                    fallback_used=False,
                    raw_response={
                        "candidates": [{"finishReason": "STOP"}],
                        "usageMetadata": {
                            "candidatesTokenCount": 30,
                            "thoughtsTokenCount": 100,
                        }
                    },
                )
            correction_messages_seen.append(messages)
            return LLMCallResult(
                content=correction_responses.pop(0),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class FakeSearchClient:
        def __init__(self) -> None:
            self.calls = []

        def search_many(self, queries, *, max_queries=None):
            self.calls.append(
                (tuple(getattr(item, "query", item) for item in queries), max_queries)
            )
            return [
                QuerySearchResult(
                    query="游戏B 角色名",
                    provider="tavily",
                    items=(
                        SearchResultItem(
                            title="角色wiki",
                            url="https://example.test/wiki",
                            snippet="角色小明的资料",
                        ),
                    ),
                )
            ]

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)
    search_client = FakeSearchClient()

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=1,
        search_client=search_client,
        task_artifact_dir=tmp_path / "artifacts",
    )

    # One query round + one search despite the correction validation retry.
    assert len(query_calls) == 1
    assert search_client.calls == [(("游戏B 角色名",), 8)]
    assert len(correction_messages_seen) == 2
    for messages in correction_messages_seen:
        assert "角色小明的资料" in messages[1]["content"]
        assert "https://example.test/wiki" in messages[1]["content"]
        # Query-round window notes are injected into the correction prompt.
        assert "<pre_round_notes>" in messages[1]["content"]
        assert "本窗口疑似在打BOSS，BOSS名待定。" in messages[1]["content"]
    # Insert/插轴 deprecated (v63+): final SRT is source-backed rows only.
    assert [segment.text for segment in parse_srt(output.read_text(encoding="utf-8"))] == [
        "一",
        "二",
    ]
    # Annotated CSV retains type/conf/note.
    annotated = (tmp_path / "out-annotated.csv").read_text(encoding="utf-8")
    assert "# type|position|duration|gap|corrected|translation|conf|char_count|note" in annotated
    assert "|二。|二|high|1|术语note" in annotated
    assert "insert|" not in annotated

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    kinds = [artifact["kind"] for artifact in artifacts]
    assert kinds.count("correction_query_response") == 1
    assert kinds.count("correction_search_results") == 1
    query_response = next(
        artifact for artifact in artifacts if artifact["kind"] == "correction_query_response"
    )
    assert query_response["payload"]["queries"] == ["游戏B 角色名"]
    assert query_response["payload"]["window_notes"] == "本窗口疑似在打BOSS，BOSS名待定。"
    assert query_response["payload"]["finish_reason"] == "STOP"
    assert query_response["payload"]["output_limit_check"]["limited"] is False
    assert query_response["payload"]["usage"]["thinking_tokens"] == 100

    report = next(
        artifact for artifact in artifacts if artifact["kind"] == "token_distribution_report"
    )
    rows = report["payload"]["rows"]
    assert [row["call"] for row in rows] == [
        "correction_query",
        "correction_window",
        "correction_window",
    ]
    assert rows[0]["tokens"]["thinking_tokens"] == 100
    assert report["payload"]["totals"]["call_count"] == 3

    exchanges = sorted((tmp_path / "artifacts" / "exchanges").glob("*.md"))
    # Block numbering: the window claimed block 001 at scheduling time; its
    # calls (query round, then correction attempts) sub-number in call order.
    assert [path.name for path in exchanges] == [
        "001-01-correction-0001-query-attempt0.md",
        "001-02-correction-0001-attempt0.md",
        "001-03-correction-0001-attempt1.md",
    ]
    query_exchange = exchanges[0].read_text(encoding="utf-8")
    assert "## 请求（system）" in query_exchange
    assert "## 模型响应" in query_exchange
    assert "游戏B 角色名" in query_exchange
    assert "- finish_reason: STOP" in query_exchange
    assert "- output_limit: observed 130 / " in query_exchange
    assert "角色小明的资料" in exchanges[1].read_text(encoding="utf-8")


def test_query_round_resumes_model_output_but_reexecutes_search(tmp_path) -> None:
    from finesub.llm.web_search import QuerySearchResult, SearchResultItem

    window = plan_correction_windows(
        [SubtitleSegment("1", 0.0, 1.0, "你好。")],
        counter=FakeTokenCounter(),
    )[0]
    artifact_dir = tmp_path / "artifacts"
    response = (
        "<reasoning>需要核对名称。</reasoning>\n"
        "<window_notes>疑似提到游戏B。</window_notes>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>游戏B 官方名</search_queries>"
    )

    class FirstClient:
        def complete(self, role, messages, **kwargs):
            return LLMCallResult(
                content=response,
                role=role,
                model="fake-lite",
                fallback_used=False,
                raw_response={},
            )

    class BlockingClient:
        def complete(self, role, messages, **kwargs):
            raise AssertionError("validated query round should resume")

    class SearchClient:
        def __init__(self):
            self.calls = []

        def search_many(self, queries, *, max_queries=None):
            normalized = tuple(item.query for item in queries)
            self.calls.append((normalized, max_queries))
            return [
                QuerySearchResult(
                    query=query,
                    provider="fake",
                    items=(
                        SearchResultItem(
                            title="官方资料",
                            url="https://example.test/game-b",
                            snippet="游戏B 的官方名称。",
                        ),
                    ),
                )
                for query in normalized
            ]

    first_search = SearchClient()
    kwargs = dict(
        window=window,
        context_pack=ContextPack(),
        audio_label="",
        previous_advice="",
        file_ref=None,
        knowledge_root=tmp_path / "knowledge",
        streamer_index="",
        common_index="",
        task_artifact_dir=artifact_dir,
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "local", "quality"),
        checkpoint_extra_identity={"task_fingerprint": "same-task"},
    )
    first = run_window_query_round(
        client=FirstClient(), search_client=first_search, **kwargs
    )
    resumed_search = SearchClient()
    resumed = run_window_query_round(
        client=BlockingClient(), search_client=resumed_search, **kwargs
    )

    assert resumed == first
    assert first_search.calls == resumed_search.calls
    artifacts = [
        json.loads(line)
        for line in (artifact_dir / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    replay = [item for item in artifacts if item["kind"] == "session_checkpoint_replay"]
    assert replay[-1]["payload"]["session"] == "query"
    assert replay[-1]["payload"]["key"] == window.chunk_id


def test_query_round_failure_is_best_effort(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            if role == LLMRole.LIGHTWEIGHT_MULTIMODAL:
                return LLMCallResult(
                    content="没有输出标签块",
                    role=role,
                    model="fake-lite",
                    fallback_used=False,
                    raw_response={},
                )
            return LLMCallResult(
                content=(
                    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|high|1|\n</translated>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            raise AssertionError("search should not run without queries")

    _setattr_both(
        monkeypatch,
        "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        search_client=FakeSearchClient(),
        task_artifact_dir=tmp_path / "artifacts",
    )

    assert parse_srt(output.read_text(encoding="utf-8"))[0].text == "一"
    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    query_response = next(
        artifact for artifact in artifacts if artifact["kind"] == "correction_query_response"
    )
    assert query_response["payload"]["parse_error"]


# --- Correction-window mid-loop resume -------------------------------------

_RESUME_STABLE = {
    "segments": [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
        {"id": "2", "start": 1.5, "end": 2.5, "text": "二。"},
    ]
}
_RESUME_GOOD = (
    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
    "sub|1|1.0|0.0|一。|一|8|1|\n"
    "sub|2|1.0|0.0|二。|二|7|1|术语note\n"
    "</translated>\n<next_advice></next_advice>"
)


class _CountingClient:
    """Returns a fixed good correction; counts and optionally forbids calls."""

    forbid = False
    content = _RESUME_GOOD

    def __init__(self, *args, **kwargs) -> None:
        pass

    def complete(self, role, messages, **kwargs):
        if callable(messages):  # tiered factory (correction round)
            messages = messages("capableC")
        if type(self).forbid:
            raise AssertionError(f"resume should not call the LLM (role={role})")
        type(self).calls += 1
        return LLMCallResult(
            content=type(self).content,
            role=role,
            model="fake",
            fallback_used=False,
            raw_response={"candidates": [{"finishReason": "STOP"}]},
        )


def _run_windows(
    tmp_path,
    monkeypatch,
    *,
    artifact_dir,
    resume=True,
    extra_style="",
    forbid=False,
    difficulty="quality",
    continuity="serial",
    execution_identity=None,
):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")

    class Client(_CountingClient):
        calls = 0

    Client.forbid = forbid
    if execution_identity is not None:
        Client.execution_identity = execution_identity
    _setattr_both(
        monkeypatch,
        "RoleClient", Client)
    out = execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=dataclass_replace(
            resolve_profile("text", "none", difficulty), continuity=continuity
        ),
        extra_style=extra_style,
        task_artifact_dir=artifact_dir,
        max_retries_per_window=1,
        resume=resume,
    )
    return out, Client.calls


def test_existing_translated_output_still_runs_requested_knowledge_update(
    tmp_path, monkeypatch
) -> None:
    out = tmp_path / "clip.srt"
    translated = tmp_path / "clip-translated.srt"
    translated.write_text("existing", encoding="utf-8")
    updates: list[dict] = []
    monkeypatch.setattr(
        correction_orchestration,
        "run_post_correction_knowledge_update",
        lambda **kwargs: updates.append(kwargs),
        raising=True,
    )

    result = correction_orchestration.run_full_correction(
        stable_json=tmp_path / "clip-stable.json",
        output_path=out,
        audio_path=None,
        postprocess_profile=None,
        knowledge="update",
    )

    assert result == translated.resolve()
    assert updates and Path(updates[0]["result_srt_path"]) == translated.resolve()


def test_fast_mode_on_a_non_local_vector_still_runs_the_research_stage(
    tmp_path, monkeypatch
) -> None:
    """fast + retrieval∈{none,native}: r1 is the session's only entry pick.

    The fused fast round only exists under retrieval=local, so skipping the
    research stage on these vectors silently disabled the knowledge base for
    every fast run (and the pre-refactor keyword seeding was already removed
    in its favour). The stage must run and seed the fused window's transfer
    chain.
    """

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    window = plan_correction_windows(
        [
            SubtitleSegment("1", 0.0, 1.0, "一。"),
            SubtitleSegment("2", 1.5, 2.5, "二。"),
        ],
        counter=FakeTokenCounter(),
    )[0]
    monkeypatch.setattr(
        correction_orchestration,
        "decide_fast_mode",
        lambda **kwargs: correction_orchestration.FastDecision(
            mode="auto", enabled=True, reason="test", window=window
        ),
    )

    research_calls: list[dict] = []

    def fake_research(**kwargs):
        research_calls.append(kwargs)
        Path(kwargs["context_path"]).write_text(
            json.dumps({"keep_entries": ["主播A"]}), encoding="utf-8"
        )
        return ContextPack()

    monkeypatch.setattr(correction_orchestration, "run_research_stage", fake_research)

    executed: dict = {}

    def fake_execute(**kwargs):
        executed.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(
        correction_orchestration, "execute_correction_windows", fake_execute
    )

    kb = tmp_path / "kb"
    (kb / "streamer").mkdir(parents=True)
    (kb / "streamer" / "index.md").write_text("- 主播A | 别名\n", encoding="utf-8")
    (kb / "common").mkdir()
    (kb / "common" / "index.md").write_text("", encoding="utf-8")

    correction_orchestration.run_full_correction(
        stable_json=stable,
        output_path=tmp_path / "out" / "clip.srt",
        audio_path=None,
        postprocess_profile=None,
        knowledge="collect",
        knowledge_root=kb,
        profile=resolve_profile("text", "none", "quality"),
        test_profile=True,
    )

    assert research_calls, "the research stage must run for a fast none-vector"
    assert executed.get("windows_override") == [window]
    assert executed.get("initial_transfer_keys") == ["主播A"]

    # Without retrieval or a readable index there is nothing to research: the
    # fast run stays a single fused call.
    research_calls.clear()
    executed.clear()
    correction_orchestration.run_full_correction(
        stable_json=stable,
        output_path=tmp_path / "out2" / "clip.srt",
        audio_path=None,
        postprocess_profile=None,
        knowledge="none",
        knowledge_root=kb,
        profile=resolve_profile("text", "none", "quality"),
        test_profile=True,
    )
    assert not research_calls
    assert executed.get("windows_override") == [window]


def test_correction_resume_replays_cached_windows_without_recalling_llm(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    out1, calls1 = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls1 >= 1
    cache_file = art / "correction-windows.jsonl"
    assert cache_file.exists()
    record = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["chunk_id"] and record["content"] == _RESUME_GOOD
    assert record["source_ids"] == ["1", "2"]

    text1 = out1.read_text(encoding="utf-8")
    # Second run: any LLM call is a failure; output must come from the cache.
    out2, calls2 = _run_windows(tmp_path, monkeypatch, artifact_dir=art, forbid=True)
    assert calls2 == 0
    assert out2.read_text(encoding="utf-8") == text1
    assert "|二。|二|high|1|术语note" in (tmp_path / "out-annotated.csv").read_text(encoding="utf-8")
    cached_kinds = [
        json.loads(line)["kind"]
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "correction_window_cached" in cached_kinds


def test_correction_resume_survives_a_difficulty_switch(tmp_path, monkeypatch) -> None:
    """Difficulty is out of the window fingerprint, so an explicit
    switch resumes completed windows instead of starting over; the cache
    records which variant each window really used."""

    art = tmp_path / "artifacts"
    out1, calls1 = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls1 >= 1
    cache_file = art / "correction-windows.jsonl"
    record = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["variant"]
    assert record["difficulty"] == "quality"
    text1 = out1.read_text(encoding="utf-8")

    out2, calls2 = _run_windows(
        tmp_path, monkeypatch, artifact_dir=art, difficulty="intermediate", forbid=True
    )
    assert calls2 == 0
    assert out2.read_text(encoding="utf-8") == text1


def test_persisted_window_plan_pins_chunk_ids_against_replanning_drift(
    tmp_path, monkeypatch
) -> None:
    """Under the same fingerprint the stored window plan wins, so
    chunk ids -- and with them the resume cache keys -- survive a counter
    that would now plan different boundaries."""

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    plan_file = art / "correction-window-plan.json"
    assert plan_file.exists()
    stored = json.loads(plan_file.read_text(encoding="utf-8"))
    assert stored["source_fingerprint"] and stored["plan"]["windows"]
    assert stored["geometry"]["geometry_profile_id"]

    real_plan = correction_run.plan_correction_windows

    def drifted(segments, **kwargs):
        return [
            dataclass_replace(window, chunk_id=f"9{window.chunk_id}")
            for window in real_plan(segments, **kwargs)
        ]

    monkeypatch.setattr(correction_run, "plan_correction_windows", drifted)
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, forbid=True)
    assert calls == 0


def test_reused_plan_refits_pending_leaf_before_dispatch(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    (art / "correction-windows.jsonl").unlink()

    class StopAfterPreflight:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, *args, **kwargs):
            raise RuntimeError("stop-after-refit")

    _setattr_both(monkeypatch, "RoleClient", StopAfterPreflight)
    with pytest.raises(RuntimeError, match="stop-after-refit"):
        execute_correction_windows(
            stable_json=tmp_path / "clip-stable.json",
            output_path=tmp_path / "refit.srt",
            token_counter=FakeTokenCounter(),
            profile=resolve_profile("text", "none", "quality"),
            task_artifact_dir=art,
            max_window_subtitle_tokens=10,
            max_retries_per_window=0,
        )

    records = [
        json.loads(line)
        for line in (art / "correction-windows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["chunk_id"] == "0001"
    assert records[0]["split_into"] == ["0001-a", "0001-b"]
    assert records[0]["reason"] == "resume_refit"
    artifacts = [
        json.loads(line)
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = next(item for item in artifacts if item["kind"] == "window_refit_report")
    assert report["payload"]["splits"][0]["failures"] == ["quality_cap"]


def test_unchanged_resume_refits_nothing(tmp_path, monkeypatch) -> None:
    """The refit predicate must not be stricter than the planner's own.

    Every pending leaf of a reused plan is measured against the envelope before
    dispatch. If that predicate ever drifts away from
    ``validate_correction_budget`` + the subtitle cap, a plain resume starts
    splitting windows nobody asked to split -- and splits never merge back.
    """

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    before = (art / "correction-windows.jsonl").read_text(encoding="utf-8")

    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, forbid=True)

    assert calls == 0
    assert (art / "correction-windows.jsonl").read_text(encoding="utf-8") == before
    artifacts = [
        json.loads(line)
        for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert not [item for item in artifacts if item["kind"] == "window_refit_report"]


def test_refit_names_the_envelope_when_a_window_cannot_be_split(
    tmp_path, monkeypatch
) -> None:
    """A single segment over the cap stops the run *before* the first call.

    Dispatching it would burn a call that every group member skips on
    ``input_limit`` and end as `No eligible target`, so the error carries what
    the user needs to act: the segment ids, the estimate, the limit it broke and
    which group member set that limit.
    """

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    (art / "correction-windows.jsonl").unlink()
    monkeypatch.setattr(
        correction_context, "split_window_in_half", lambda *a, **k: None
    )

    with pytest.raises(ValueError, match="unsplittable") as excinfo:
        execute_correction_windows(
            stable_json=tmp_path / "clip-stable.json",
            output_path=tmp_path / "refit.srt",
            token_counter=FakeTokenCounter(),
            profile=resolve_profile("text", "none", "quality"),
            task_artifact_dir=art,
            max_window_subtitle_tokens=10,
            max_retries_per_window=0,
        )

    message = str(excinfo.value)
    assert "chunk=0001" in message
    assert "culprit=quality_cap" in message
    assert "quality_csv_limit=10" in message
    assert "envelope_source=" in message


_FINGERPRINT_ARGS = dict(
    prompt_version="v",
    extra_style="",
    test_profile=False,
    source_fingerprint="s",
    media_identity={},
    extra="",
)


def test_window_invalidation_inputs_is_the_whole_fingerprint_payload(
    monkeypatch,
) -> None:
    """The include list is the contract, and it is enforced rather than
    documented: a payload field missing from it raises instead of being
    silently left out of the fingerprint."""

    assert correction_commit.WINDOW_INVALIDATION_INPUTS == (
        "prompt_version",
        "extra_style",
        "test_profile",
        "source_fingerprint",
        "media_identity",
        "extra",
    )
    assert correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)

    # Stand in for "someone adds a payload field and forgets to classify it".
    monkeypatch.setattr(
        correction_commit,
        "WINDOW_INVALIDATION_INPUTS",
        ("prompt_version", "extra_style", "test_profile", "source_fingerprint"),
        raising=True,
    )
    with pytest.raises(AssertionError) as excinfo:
        correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)
    assert "media_identity" in str(excinfo.value)
    assert "extra" in str(excinfo.value)


def test_every_whitelisted_input_actually_moves_the_fingerprint() -> None:
    """The other direction: a name on the list that no longer reaches the
    payload would be a dead promise."""

    base = correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)
    changed = {
        "prompt_version": "v2",
        "extra_style": "俏皮些",
        "test_profile": True,
        "source_fingerprint": "s2",
        "media_identity": {"audio": {"size": 1}},
        "extra": "seed",
    }
    assert set(changed) == set(correction_commit.WINDOW_INVALIDATION_INPUTS)
    for field, value in changed.items():
        assert (
            correction_commit._task_fingerprint(**{**_FINGERPRINT_ARGS, field: value})
            != base
        ), field


def test_correction_resume_keeps_committed_windows_across_model_group_switch(
    tmp_path, monkeypatch
) -> None:
    """The headline case: routing identity is not a window invalidation key
    (docs/llm_local_agent.md SS11)."""

    art = tmp_path / "artifacts"
    _run_windows(
        tmp_path,
        monkeypatch,
        artifact_dir=art,
        execution_identity={
            "policy_id": "api-only",
            "routing_identity_digest": "group-a",
            "local_agent_reasoning_effort": "high",
        },
    )
    _out, calls = _run_windows(
        tmp_path,
        monkeypatch,
        artifact_dir=art,
        execution_identity={
            "policy_id": "agent-only",
            "routing_identity_digest": "group-b",
            "local_agent_reasoning_effort": "low",
        },
        difficulty="intermediate",
        continuity="parallel",
        forbid=True,
    )
    assert calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        # The user's own instruction: a difference they can see in the output.
        {"extra_style": "翻得更俏皮"},
    ],
)
def test_correction_resume_reruns_on_gated_changes(tmp_path, monkeypatch, kwargs) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, **kwargs)
    assert calls > 0


def test_correction_resume_reruns_on_prompt_version_bump(tmp_path, monkeypatch) -> None:
    """PROMPT_VERSION is the output contract; CLAUDE.md makes the bump
    invalidate resume caches on purpose."""

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    # Both the window ledger's fingerprint and the query round's checkpoint
    # hash read PROMPT_VERSION, and they read it from different modules now --
    # bumping only one of them would leave half the caches valid, which is not
    # what a contract bump means.
    setattr_correction(
        monkeypatch, "PROMPT_VERSION", "zh-subtitle-correction-csv-v-next"
    )
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls > 0


def test_media_identity_ignores_mtime_but_tracks_size(tmp_path) -> None:
    """A re-download of the same audio is not a new source."""

    media = tmp_path / "source.aac"
    media.write_bytes(b"first")
    first = correction_commit._media_identity(media)
    os.utime(media, (0, 0))
    assert correction_commit._media_identity(media) == first

    media.write_bytes(b"a longer second version")
    assert correction_commit._media_identity(media) != first
    assert correction_commit._media_identity(None) == {}


def test_correction_resume_disabled_ignores_existing_cache(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, resume=False)
    assert calls >= 1


def test_correction_resume_ignores_cache_on_input_hash_mismatch(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    # A structurally stale window cannot pass the source-only core hash.
    cache_file = art / "correction-windows.jsonl"
    record = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    record["input_hash_core"] = "sha256:deadbeef0000"
    cache_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls >= 1


def test_advice_ledger_front_truncates_to_token_budget() -> None:
    from finesub.llm.routing.config import ADVICE_LEDGER_MAX_TOKENS
    from finesub.llm.token_truncate import truncate_text_only

    ledger = [(f"{index:04d}", f"建议{index}。" + "字" * 300) for index in range(40)]
    rendered = render_advice_ledger(ledger)
    # Same composition as the correction loop's injection site: keep the tail
    # (newest windows), drop the oldest advice beyond the 8k-token budget.
    capped = truncate_text_only(
        rendered,
        ADVICE_LEDGER_MAX_TOKENS,
        len,
        keep="tail",
        heuristic_count=len,
        prefer_natural_boundary=True,
    )

    assert len(capped) <= ADVICE_LEDGER_MAX_TOKENS  # 1 token per char counter
    assert "[window 0039]" in capped
    assert "[window 0000]" not in capped


def test_query_round_requests_knowledge_entries_for_correction(tmp_path, monkeypatch) -> None:
    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
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

    query_output = (
        "<window_notes>\n杂谈回。\n</window_notes>\n"
        "<requested_entries>\nエーちゃん\n未知条目\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>"
    )
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|一。|一|8|1|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|8|1|\n</translated>"
    )
    query_messages_seen = []
    correction_messages_seen = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            if role == LLMRole.LIGHTWEIGHT_MULTIMODAL:
                query_messages_seen.append(messages)
                return LLMCallResult(
                    content=query_output,
                    role=role,
                    model="fake-lite",
                    fallback_used=False,
                    raw_response={},
                )
            correction_messages_seen.append(messages)
            return LLMCallResult(
                content=good,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class NoSearchClient:
        def search_many(self, queries, *, max_queries=None):  # pragma: no cover
            raise AssertionError("no queries were emitted")

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        search_client=NoSearchClient(),
        knowledge_root=knowledge_root,
        task_artifact_dir=tmp_path / "artifacts",
    )

    # The query round sees both indices.
    query_user = query_messages_seen[0][1]["content"]
    assert "<streamer_index>" in query_user
    assert "主播A | エーちゃん" in query_user
    # The requested entry body reaches the correction round's entry_details.
    correction_user = correction_messages_seen[0][1]["content"]
    assert "<entry_details>" in correction_user
    assert "关西腔。" in correction_user

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    query_response = next(
        artifact for artifact in artifacts if artifact["kind"] == "correction_query_response"
    )
    assert query_response["payload"]["requested_entries"] == ["エーちゃん", "未知条目"]
    assert query_response["payload"]["resolved_entry_keys"] == ["主播A"]
    assert query_response["payload"]["missing_entries"] == ["未知条目"]
    # The correction round records the unified injection set (v17: transfers +
    # this window's resolved requests rendered once).
    window_response = next(
        artifact
        for artifact in artifacts
        if artifact["kind"] == "correction_window_response"
    )
    assert window_response["payload"]["injected_entries"] == ["主播A"]


def test_keep_entries_transfer_chain_across_windows(tmp_path, monkeypatch) -> None:
    """v17 pass-through: window 1's <keep_entries> reaches window 2's query
    round as carried context and its correction round as entry_details; the
    resume cache records the chain."""

    from dataclasses import replace

    from finesub.llm.chunking import load_segments_from_stable_json, plan_correction_windows

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 2.0, "end": 3.0, "text": "二。"},
                ]
            }
        ),
        encoding="utf-8",
    )
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

    segments = load_segments_from_stable_json(stable_json)
    counter = FakeTokenCounter()
    window_one = plan_correction_windows(segments[:1], counter=counter)[0]
    window_two = replace(
        plan_correction_windows(segments[1:], counter=counter)[0], chunk_id="0002"
    )
    windows = [window_one, window_two]

    query_output = (
        "<window_notes>\n杂谈回。\n</window_notes>\n"
        "<requested_entries>\nエーちゃん\n</requested_entries>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>"
    )
    outputs = [
        (
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|x|1|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|一。|一|8|1|\n</translated>\n"
            "<next_advice></next_advice>\n"
            "<keep_entries>\n主播A\n</keep_entries>"
        ),
            (
                "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|x|2|8|0.5|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|0.0|二。|二|8|1|\n</translated>\n"
                "<next_advice></next_advice>\n"
                "<keep_entries></keep_entries>"
            ),
    ]
    query_users: list[str] = []
    correction_users: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages("capableC")
            if role == LLMRole.LIGHTWEIGHT_MULTIMODAL:
                query_users.append(messages[1]["content"])
                return LLMCallResult(
                    content=query_output,
                    role=role,
                    model="fake-lite",
                    fallback_used=False,
                    raw_response={},
                )
            content = outputs[len(correction_users) % len(outputs)]
            correction_users.append(messages[1]["content"])
            return LLMCallResult(
                content=content,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class NoSearchClient:
        def search_many(self, queries, *, max_queries=None):  # pragma: no cover
            raise AssertionError("no queries were emitted")

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=counter,
        search_client=NoSearchClient(),
        knowledge_root=knowledge_root,
        task_artifact_dir=tmp_path / "artifacts",
        windows_override=windows,
    )

    # Window 1's query round has no carried entries; window 2's carries the
    # entry kept by window 1 (full text, marked as auto-injected).
    assert "（无）" in query_users[0].split("<carried_entries>")[1]
    assert "关西腔。" in query_users[1].split("<carried_entries>")[1]
    # Both correction rounds see the entry: window 1 via its own request,
    # window 2 via the transfer.
    assert "关西腔。" in correction_users[0]
    assert "关西腔。" in correction_users[1]
    # The resume cache records the chain per window.
    cache_lines = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "correction-windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    by_chunk = {record["chunk_id"]: record for record in cache_lines}
    assert by_chunk["0001"]["keep_entries"] == ["主播A"]
    assert by_chunk["0001"]["injected_entries"] == ["主播A"]
    assert by_chunk["0002"]["keep_entries"] == []
    assert by_chunk["0002"]["injected_entries"] == ["主播A"]

    # Knowledge text is an execution input, not a reason to throw away windows
    # that were already parsed and committed.
    (knowledge_root / "streamer" / "主播A.md").write_text(
        "# 主播A\n\n## 档案\n\n东京腔。\n", encoding="utf-8"
    )
    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=counter,
        search_client=NoSearchClient(),
        knowledge_root=knowledge_root,
        task_artifact_dir=tmp_path / "artifacts",
        windows_override=windows,
    )
    assert len(correction_users) == 2


def test_an_unclosed_translated_block_splits_instead_of_killing_the_run() -> None:
    """The content heuristic 46206b1 promised to hand to validation.

    Usage counts stay the primary signal -- flash reports finish_reason=length
    on complete answers, which is why that commit demoted it. But when usage is
    absent or under-reports, a genuinely truncated reply looked like an
    ordinary validation failure: the same oversized window went out five more
    times and the run died on "Window NNNN failed validation".
    """
    from finesub.llm.output_protocol import looks_truncated_translated

    cut_off = (
        "<translated>\n"
        "type|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|0.0|one|一|high|1|\n"
    )
    complete = (
        "<translated>\n"
        "sub|1|1.0|0.0|one|一|high|1|\n"
        "</translated>"
    )

    assert looks_truncated_translated(cut_off)
    assert not looks_truncated_translated(complete)


def test_research_reuse_key_excludes_window_geometry(tmp_path) -> None:
    """Interval-addressed research notes survive correction re-planning."""

    from finesub.llm.research import plan_geometry_metadata, research_reuse_key

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    profile = resolve_profile("audio", "local", "quality")
    geometry = plan_geometry_metadata(
        profile, audio_duration=120.0, max_window_subtitle_tokens=None
    )
    planning = research.planning_metadata(
        profile,
        stable_json=stable,
        audio_duration=120.0,
        max_window_subtitle_tokens=None,
        knowledge_enabled=False,
    )

    reuse_key = research_reuse_key(planning)
    assert "prompt_version" in reuse_key
    assert not (set(geometry) - {"prompt_version"}) & set(reuse_key)
    assert reuse_key["research_semantics_id"] == "local"


def test_geometry_gate_reacts_to_the_group_envelope(tmp_path) -> None:
    """A user-declared smaller model moves the boundaries, so it must replan;
    switching between equally-sized groups must not."""

    from finesub.llm.routing.config import ModelLimits
    from finesub.llm.research import plan_geometry_metadata

    profile = resolve_profile("audio", "local", "quality")
    base = plan_geometry_metadata(profile, limits=ModelLimits())
    same_envelope = plan_geometry_metadata(profile, limits=ModelLimits())
    smaller = plan_geometry_metadata(
        profile, limits=ModelLimits(prompt_input_limit=128_000)
    )

    assert base == same_envelope
    assert base != smaller


def test_geometry_ignores_switches_the_planner_never_reads() -> None:
    profile = resolve_profile("audio", "local", "quality")

    assert (
        dataclass_replace(profile, continuity="parallel").geometry_id
        == profile.geometry_id
    )
    assert (
        dataclass_replace(profile, difficulty="efficiency").geometry_id
        == profile.geometry_id
    )
    assert (
        dataclass_replace(profile, planning_media="text").geometry_id
        == profile.geometry_id
    )
    # ...but the ones it does read still move it.
    assert (
        dataclass_replace(profile, output_scale=2.0).geometry_id
        != profile.geometry_id
    )


def test_difficulty_is_geometry_neutral_only_up_to_the_switches_it_pins() -> None:
    """`intermediate` is the documented "downgrade and keep going" path;
    `efficiency` is not, because resolving it pins correction_media=text and
    retrieval=none, which really does move the window boundaries."""

    from finesub.llm.research import plan_geometry_metadata

    quality = plan_geometry_metadata(resolve_profile("audio", "local", "quality"))
    intermediate = plan_geometry_metadata(
        resolve_profile("audio", "local", "intermediate")
    )
    efficiency = plan_geometry_metadata(resolve_profile("text", "none", "efficiency"))

    assert quality == intermediate
    assert quality != efficiency


def test_group_envelope_reaches_the_geometry_through_difficulty() -> None:
    """A preset may bind a smaller group at a lower difficulty, and then the
    difficulty *does* change the envelope. Packaged presets do not, which is
    why switching difficulty keeps windows today -- pin the mechanism, not the
    coincidence."""

    from finesub.llm.routing.capabilities import correction_planning_limits

    envelopes = {
        difficulty: correction_planning_limits(
            resolve_profile("audio", "local", difficulty)
        )
        for difficulty in ("quality", "intermediate")
    }
    assert len({(lim.prompt_input_limit, lim.output_limit) for lim in envelopes.values()}) == 1


def test_serial_replay_of_parallel_windows_warns_about_the_advice_ledger(
    tmp_path, monkeypatch, capsys
) -> None:
    """G3: allowed, but the ledger later live windows read starts short."""

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity="parallel")
    capsys.readouterr()
    _out, calls = _run_windows(
        tmp_path, monkeypatch, artifact_dir=art, continuity="serial", forbid=True
    )

    assert calls == 0
    assert "produced in parallel mode" in capsys.readouterr().err
