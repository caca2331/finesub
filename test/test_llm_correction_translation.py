from __future__ import annotations

from finesub.reporting import reporting_to

from dataclasses import replace as dataclass_replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import finesub.llm.correction_translation as correction_orchestration
from finesub.llm.correction_request import CorrectionRequest
import finesub.llm.research as research
from finesub.llm.stages.correction import commit as correction_commit
from finesub.llm.stages.correction import context as correction_context
from finesub.llm.stages.correction import run as correction_run

from .conftest import setattr_correction

from finesub.llm.client import LLMCallResult
from finesub.llm.media_upload import UploadedFileRef
from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.routing.config import LLMRole
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.stages.correction.attempts import (
    _extract_next_advice,
    _extract_task_update_feedback,
)
from finesub.llm.stages.correction.metadata import _is_output_limited, _output_limit_check
from finesub.llm.stages.correction.request_turn import _extract_window_notes
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.routing.execution_policy import ExecutionSettings
from finesub.llm.prompts import render_advice_ledger
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
        # the correction's prompt version: audit-only, a bump keeps the preparation task's product
        prompt_version="v2",
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
        ("extra_info_hash", "sha256:new-notes"),
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
        "finesub.llm.media_upload.upload_gemini_file",
        lambda path, **_: uploads.append(path),
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
        "type|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一。|8|译2字；宜保持独立\n"
        "</singles>\n"
        "<translated>\n"
        "type|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一。|high|\n"
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
                model="local-agy-gemini-3_8-flash",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
                target_id="local-agy-gemini-3_8-flash",
                backend="local_agent",
            )

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("audio", "none", "quality"),
        length_scale=1.0,
    )

    assert len(seen_refs) == 1
    assert seen_refs[0].local_path
    assert seen_refs[0].file_id == ""
    assert uploads == []


def test_extract_task_update_feedback_reads_first_feedback_block() -> None:
    content = (
        "<singles>\nsub|1|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\nsub|1|good|好|high|\n</translated>\n"
        '<task_update_feedback>{"reusable_terms":["A 固定译为甲"]}</task_update_feedback>'
    )

    assert _extract_task_update_feedback(content) == '{"reusable_terms":["A 固定译为甲"]}'


def test_extract_next_advice_reads_block_and_caps_length() -> None:
    content = (
        "<singles>\nsub|1|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\nsub|1|good|好|high|\n</translated>\n"
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
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|A|甲|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|A|甲|high|\n"
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
        length_scale=1.0,
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
    bad = "<translated>\nsub|9|nine|九|high|\n</translated>"
    good = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二。|二|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|high|\n"
        "sub|2|二。|二|high|\n"
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
        length_scale=1.0,
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
    # The retry is a repair round, which differs by its own turn: no seed on
    # either call, so the opening user message it re-sends is the one sent.
    assert [item["seed"] for item in sampling_kwargs] == [None, None]
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
    bad = "<translated>\nsub|9|nine|九|high|\n</translated>"
    good = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二。|二|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|high|\n"
        "sub|2|二。|二|high|\n"
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
        length_scale=1.0,
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


_PATCH_HEADER = "type|position|corrected_text|translation|conf|note"


def _four_segment_stable_json(path):
    path.write_text(json.dumps({"segments": [
        {"id": str(i), "start": i * 1.5, "end": i * 1.5 + 1.0, "text": f"{i}。"} for i in range(1, 5)
    ]}), encoding="utf-8")
    return path


def _rows(*ids: int) -> str:
    return "\n".join(f"sub|{i}|{i}。|{i}|high|" for i in ids)


def _run_repairs(tmp_path, monkeypatch, responses: list[str], *, retries: int = 1):
    """Run one four-row window against canned replies; give back each call's repair kwargs and the artifacts."""

    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append({key: kwargs.get(key) for key in ("previous_output", "validation_errors", "repair_instruction")})
            return LLMCallResult(content=responses.pop(0), role=LLMRole.AUDIO_MULTIMODAL, model="fake",
                                 fallback_used=False, raw_response={"candidates": [{"finishReason": "STOP"}]},
                                 variant="capableC")

    _setattr_both(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(
        stable_json=_four_segment_stable_json(tmp_path / "clip-stable.json"), output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(), max_retries_per_window=retries,
        profile=resolve_profile("text", "none", "quality"), task_artifact_dir=tmp_path / "artifacts",
        length_scale=1.0,
    )
    artifacts = [json.loads(line) for line in
                 (tmp_path / "artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    return calls, artifacts


def test_a_mostly_right_answer_is_repaired_with_a_patch(tmp_path, monkeypatch) -> None:
    """Three of four rows passed: the repair asks for a `<patch>` naming source 4, the harness lays it over the
    three rows and validates the whole; the record keeps the whole answer and says how it was repaired."""

    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3)}\n</translated>"
    patch = f"<patch>\n{_PATCH_HEADER}\n{_rows(4)}\n</patch>"
    calls, artifacts = _run_repairs(tmp_path, monkeypatch, [rejected, patch])

    assert calls[0]["repair_instruction"] == ""
    assert calls[1]["previous_output"] == rejected
    assert "<patch>" in calls[1]["repair_instruction"] and "要改的源序号：4" in calls[1]["repair_instruction"]
    responses = [a["payload"] for a in artifacts if a["kind"] == "correction_window_response"]
    assert responses[1]["validation_ok"] is True
    assert responses[1]["repair_mode"] == "patch" and responses[1]["patch_units"] == 1
    assert _rows(1, 2, 3, 4) in responses[1]["response_content"]
    retry = next(a["payload"] for a in artifacts if a["kind"] == "correction_window_retry")
    assert retry["repair_mode"] == "patch"
    assert len(parse_srt((tmp_path / "out.srt").read_text(encoding="utf-8"))) == 4


def test_an_unusable_patch_sends_the_same_base_back(tmp_path, monkeypatch) -> None:
    """A reply with no `<patch>` changes nothing: the next round repairs the same base, the patch's own error
    first, then the validator's."""

    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3)}\n</translated>"
    patch = f"<patch>\n{_PATCH_HEADER}\n{_rows(4)}\n</patch>"
    calls, artifacts = _run_repairs(tmp_path, monkeypatch, [rejected, "no block at all", patch], retries=2)

    assert calls[2]["previous_output"] == rejected
    errors = list(calls[2]["validation_errors"])
    assert "<patch>" in errors[0] and errors[1:] == list(calls[1]["validation_errors"])
    responses = [a["payload"] for a in artifacts if a["kind"] == "correction_window_response"]
    assert [r.get("patch_reply") for r in responses] == [None, "invalid", "patch"]
    assert responses[2]["validation_ok"] is True


def test_a_patch_that_merges_but_still_fails_is_the_next_base(tmp_path, monkeypatch) -> None:
    """What the model is shown as its last answer is the whole its patch made, and the next patch goes over it."""

    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2)}\n</translated>"
    patch_3 = f"<patch>\n{_PATCH_HEADER}\n{_rows(3)}\n</patch>"
    patch_4 = f"<patch>\n{_PATCH_HEADER}\n{_rows(4)}\n</patch>"
    calls, artifacts = _run_repairs(tmp_path, monkeypatch, [rejected, patch_3, patch_4], retries=2)

    assert _rows(1, 2, 3) in calls[2]["previous_output"] and "<patch>" not in calls[2]["previous_output"]
    assert "要改的源序号：4" in calls[2]["repair_instruction"]
    responses = [a["payload"] for a in artifacts if a["kind"] == "correction_window_response"]
    assert [r.get("patch_reply") for r in responses] == [None, "patch", "patch"]
    assert responses[2]["validation_ok"] is True


def test_a_patch_answer_is_not_asked_again_for_not_thinking(tmp_path, monkeypatch) -> None:
    """The zero-thinking retry is for whole answers: a patch is a few rows, and not thinking over it is normal
    (owner 2026-10-24). On the default Gemini API most patch replies report no thinking at all."""

    replies = [(f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3)}\n</translated>", 800),
               (f"<patch>\n{_PATCH_HEADER}\n{_rows(4)}\n</patch>", None)]
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            calls.append(kwargs)
            content, thoughts = replies.pop(0)
            usage = {"promptTokenCount": 1000, "candidatesTokenCount": 200,
                     **({"thoughtsTokenCount": thoughts} if thoughts is not None else {})}
            return LLMCallResult(content=content, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                                 raw_response={"usageMetadata": usage, "candidates": [{"finishReason": "STOP"}]},
                                 variant="capableC", thinking_level="medium")

    _setattr_both(monkeypatch, "RoleClient", FakeClient)
    execute_correction_windows(
        stable_json=_four_segment_stable_json(tmp_path / "clip-stable.json"), output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(), max_retries_per_window=1,
        profile=resolve_profile("text", "none", "quality"), task_artifact_dir=tmp_path / "artifacts",
        length_scale=1.0,
    )
    assert len(calls) == 2 and not any(call.get("fresh_session") for call in calls)
    assert len(parse_srt((tmp_path / "out.srt").read_text(encoding="utf-8"))) == 4


def test_a_leaked_thought_before_a_patch_is_cut_at_the_patch(tmp_path, monkeypatch) -> None:
    """Reasoning shipped as the reply, drafting a `<patch>` before the real one: the last complete `<patch>` is
    the answer, as `<translated>` is on a whole answer."""

    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3)}\n</translated>"
    leaked = (f"thought\nSource 4 is missing, so I write\n<patch>\n{_PATCH_HEADER}\nsub|4|draft\n</patch>\nthen fix it.\n"
              f"<patch>\n{_PATCH_HEADER}\n{_rows(4)}\n</patch>")
    _, artifacts = _run_repairs(tmp_path, monkeypatch, [rejected, leaked])

    response = [a["payload"] for a in artifacts if a["kind"] == "correction_window_response"][1]
    assert response["validation_ok"] is True and response["patch_reply"] == "patch"
    assert response["thought_stripped"] > 0 and "draft" not in response["response_content"]


def test_a_mostly_wrong_answer_is_still_rewritten_whole(tmp_path, monkeypatch) -> None:
    """Under half the window passed (PATCH_MIN_COVERAGE): the repair asks for the whole answer, as before."""

    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1)}\n</translated>"
    calls, artifacts = _run_repairs(
        tmp_path, monkeypatch, [rejected, f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3, 4)}\n</translated>"])

    assert calls[1]["previous_output"] == rejected and calls[1]["repair_instruction"] == ""
    responses = [a["payload"] for a in artifacts if a["kind"] == "correction_window_response"]
    assert responses[1]["repair_mode"] == "full" and responses[1]["validation_ok"] is True


def test_a_cell_with_variant_overrides_repairs_whole(tmp_path, monkeypatch) -> None:
    """A model of the cell with its own variant may answer the next attempt in another row format."""

    from finesub.llm.routing import config as routing_config

    real = routing_config.role_config_for
    monkeypatch.setattr(routing_config, "role_config_for", lambda *args, **kwargs: dataclass_replace(
        real(*args, **kwargs), variant_overrides={"some-target": "basicA"}))
    rejected = f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3)}\n</translated>"
    calls, _ = _run_repairs(
        tmp_path, monkeypatch, [rejected, f"<translated>\n{_PATCH_HEADER}\n{_rows(1, 2, 3, 4)}\n</translated>"])

    assert calls[1]["previous_output"] == rejected and calls[1]["repair_instruction"] == ""


def test_a_spent_repair_chain_is_replaced_by_a_fresh_blind_session(
    tmp_path, monkeypatch
) -> None:
    """Tier 2 of the retry budget (docs/llm_harness_behavior.md "重试与拼接").

    A session that keeps failing its own repairs is presumed degenerate, so
    once the in-chain budget is spent the window goes to a *fresh* session:
    the replacement call carries no repair context (an agent would open a new
    conversation, a stateless endpoint throws blind), and the chain after it
    repairs normally again. Total calls are the product of the (n+1) forms.
    """

    stable_json = _two_segment_stable_json(tmp_path / "clip-stable.json")
    bad = "<translated>\nsub|9|nine|九|high|\n</translated>"
    good = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二。|二|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|high|\n"
        "sub|2|二。|二|high|\n"
        "</translated>"
    )
    responses = [bad, bad, bad, good]
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
        max_replacements_per_window=1,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
        length_scale=1.0,
    )

    assert not responses, "all four budgeted calls must have been spent"
    # Chain 1: blind first call, then a repair. Chain 2 (the replacement):
    # blind again -- the degenerate chain's output is deliberately dropped --
    # then a repair carrying the *new* chain's failure.
    assert repair_inputs[0] == ("", [])
    assert repair_inputs[1][0] == bad and repair_inputs[1][1]
    assert repair_inputs[2] == ("", [])
    assert repair_inputs[3][0] == bad and repair_inputs[3][1]

    artifacts = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "task-artifacts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    retries = [a for a in artifacts if a["kind"] == "correction_window_retry"]
    assert [
        (r["payload"]["repair_context"], r["payload"]["replacement"])
        for r in retries
    ] == [(True, False), (False, True), (True, False)]


def test_the_per_window_budget_is_the_product_of_both_tiers(
    tmp_path, monkeypatch
) -> None:
    """retries=0, replacements=1 means exactly two blind calls, then failure."""

    stable_json = _two_segment_stable_json(tmp_path / "clip-stable.json")
    bad = "<translated>\nsub|9|nine|九|high|\n</translated>"
    calls: list[tuple[str, list[str]]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(
                (
                    kwargs.get("previous_output", ""),
                    list(kwargs.get("validation_errors", ())),
                )
            )
            return LLMCallResult(
                content=bad,
                role=LLMRole.AUDIO_MULTIMODAL,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    with pytest.raises(RuntimeError, match="failed validation"):
        execute_correction_windows(
            stable_json=stable_json,
            output_path=tmp_path / "out.srt",
            token_counter=FakeTokenCounter(),
            max_retries_per_window=0,
            max_replacements_per_window=1,
            profile=resolve_profile("text", "none", "quality"),
            task_artifact_dir=tmp_path / "artifacts",
            length_scale=1.0,
        )

    # Two chains of one call each; with no in-chain repairs every call is
    # blind, which is exactly the pre-2026-08-15 behaviour tier 2 re-creates
    # on a stateless endpoint.
    assert calls == [("", []), ("", [])]


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
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|high|\n"
    )
    first_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二|二|8|译1字；宜保持独立\n"
        "sub|3|三。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|high|\nsub|2|二|二|high|\n"
        "sub|3|三。|三|high|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|四|四|8|译1字；宜保持独立\n"
        "sub|2|五。|五|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|四|四|high|\nsub|2|五。|五|high|\n</translated>"
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
        length_scale=1.0,
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
    truncated = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\n"
    first_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二|二|8|译1字；宜保持独立\n"
        "sub|3|三。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\nsub|2|二|二|high|\nsub|3|三。|三|high|\n</translated>\n"
        "<next_advice>术语X固定译为Y。</next_advice>"
    )
    # Dense 1s spacing with the default overlap floor (10) exceeds the first
    # half's length, so the -b half falls back to no overlap: it starts at "4".
    second_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|四|四|8|译1字；宜保持独立\n"
        "sub|2|五。|五|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|四|四|high|\nsub|2|五。|五|high|\n</translated>\n"
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
        length_scale=1.0,
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
    truncated = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\n"
    first_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|译1字；宜保持独立\n"
        "sub|2|二|二|8|译1字；宜保持独立\n"
        "sub|3|三。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\nsub|2|二|二|high|\nsub|3|三。|三|high|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|四|四|8|译1字；宜保持独立\n"
        "sub|2|五。|五|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|四|四|high|\nsub|2|五。|五|high|\n</translated>"
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

    def fake_upload(path, **_):
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
    monkeypatch.setattr("finesub.llm.media_upload.upload_gemini_file", fake_upload)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        profile=resolve_profile("audio", "none", "quality"),
        length_scale=1.0,
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
        ("STOP", "<singles>\ntype|position|corrected_text|translation|conf|note\nsub|9|x|9|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|corrected_text|translation|conf|note\nsub|9|坏|坏|high|\n</translated>"),
        ("STOP", "<singles>\ntype|position|corrected_text|translation|conf|note\nsub|1|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\n</translated>"),
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
        "finesub.llm.media_upload.upload_gemini_file",
        lambda path, **_: uploads.append(str(path))
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
        length_scale=1.0,
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
    truncated = "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\n"
    first_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|第一句。|一|8|译1字；宜保持独立\n"
        "sub|2|第二句。|二|8|译1字；宜保持独立\n"
        "sub|3|第三句。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|第一句。|一|high|\nsub|2|第二句。|二|high|\nsub|3|第三句。|三|high|\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|第四句。|四|8|译1字；宜保持独立\n"
        "sub|2|第五句。|五|8|译1字；宜保持独立\n"
        "sub|3|第六句。|六|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|第四句。|四|high|\nsub|2|第五句。|五|high|\nsub|3|第六句。|六|high|\n</translated>"
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
        length_scale=1.0,
    )

    # Sparse 40s spacing: zero content-driven overlap at the split, so the -b
    # half starts at canonical id 4 (clip_start 115.0). Model-facing target ids
    # restart at 1, while the three raw references are numbered -2,-1,0.
    second_user = seen_messages[2][1]["content"]
    assert "<preceding_context>" in second_user
    assert "0|-35.0|1.0|39.0|第3句。" in second_user
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
            length_scale=1.0,
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


def test_request_turn_failure_is_best_effort(tmp_path, monkeypatch) -> None:
    """A request turn that never yields a usable reply degrades to "no
    retrieval": the window is still corrected (the legacy query round had the
    same contract)."""

    stable_json = tmp_path / "clip-stable.json"
    stable_json.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    calls: list[int] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(len(messages))
            if len(messages) == 2:  # the request turn: garbage, every attempt
                return LLMCallResult(content="没有输出标签块", role=role, model="fake", fallback_used=False, raw_response={})
            return LLMCallResult(
                content=(
                    "<singles>\ntype|position|corrected_text|translation|conf|note\n"
                    "sub|1|一。|一|8|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|high|\n</translated>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            raise AssertionError("search should not run without queries")

    _setattr_both(monkeypatch, "RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        search_client=FakeSearchClient(),
        profile=resolve_profile("text", "local", "quality"),
        task_artifact_dir=tmp_path / "artifacts",
        length_scale=1.0,
    )
    assert output.exists()
    # format retries on the request turn, then one delivery in the same conversation
    assert calls[-1] == 4 and calls[:-1] and all(n == 2 for n in calls[:-1])


_RESUME_STABLE = {
    "segments": [
        {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
        {"id": "2", "start": 1.5, "end": 2.5, "text": "二。"},
    ]
}
_RESUME_GOOD = (
    "<translated>\ntype|position|corrected_text|translation|conf|note\n"
    "sub|1|一。|一|8|\n"
    "sub|2|二。|二|7|术语note\n"
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
        length_scale=1.0,
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

    outcome = correction_orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=tmp_path / "clip-stable.json",
            output_path=out,
            audio_path=tmp_path / "clip.wav",
            postprocess_profile=None,
            knowledge="update",
        )
    )

    assert outcome.result == translated.resolve() and outcome.result_reused
    assert updates and Path(updates[0]["result_srt_path"]) == translated.resolve()


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
    assert "|二。|二|high|1||术语note" in (tmp_path / "out-annotated.csv").read_text(encoding="utf-8")
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


def test_correction_plan_is_resolved_once_and_survives_a_media_switch(tmp_path) -> None:
    """`resolve_correction_plan` writes the plan it computes, so a second call
    reads it back as reused -- why the run resolves once and hands the result to
    both the speaker judge and the correction (a fresh plan must stay fresh for
    the ledger). The fingerprint has no modality in it: a resume under another
    `--llm-media` keeps the boundaries the judge split inside."""

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    art = tmp_path / "artifacts"
    kwargs = dict(
        stable_json=stable, task_artifact_dir=art, test_profile=True, audio_path=None,
        video_path=None, token_counter=FakeTokenCounter(), max_window_subtitle_tokens=10,
    )
    first = correction_run.resolve_correction_plan(profile=resolve_profile("text", "none", "quality"), **kwargs)
    assert not first.reused and (art / "correction-window-plan.json").exists()
    again = correction_run.resolve_correction_plan(profile=resolve_profile("audio", "none", "quality"), **kwargs)
    assert again.reused
    assert [w.chunk_id for w in again.windows] == [w.chunk_id for w in first.windows]
    assert again.task_fingerprint == first.task_fingerprint
    fast = correction_run.resolve_correction_plan(
        profile=resolve_profile("text", "none", "quality"), windows_override=first.windows[:1], **kwargs
    )
    assert fast.fast and not fast.reused and fast.windows == first.windows[:1]


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
            length_scale=1.0,
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


#: Four lines, so a half still has two and can in principle be halved again --
#: which makes `MAX_SPLITS`, not "unsplittable", the thing that stops a refit.
_REFIT_STABLE = {
    "segments": [
        {"id": str(i + 1), "start": i * 2.0, "end": i * 2.0 + 1.5, "text": f"这是第{i + 1}句台词。"}
        for i in range(4)
    ]
}
_REFIT_GOOD = (
    "<translated>\n"
    "type|position|corrected_text|translation|conf|note\n"
    + "".join(
        f"sub|{i + 1}|这是第{i + 1}句台词。|第{i + 1}句|high|\n" for i in range(4)
    )
    + "</translated>\n<next_advice></next_advice>"
)


def _refit_run(tmp_path, monkeypatch, *, artifact_dir, **kwargs):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_REFIT_STABLE), encoding="utf-8")

    class Client(_CountingClient):
        calls = 0
        content = _REFIT_GOOD

    _setattr_both(monkeypatch, "RoleClient", Client)
    return execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=artifact_dir,
        max_retries_per_window=0,
        **kwargs,
        length_scale=1.0,
    )


def test_refit_stops_at_the_split_cap_instead_of_halving_forever(
    tmp_path, monkeypatch
) -> None:
    """Resume's refit spends the same split budget the retry loop does.

    It used to recurse without asking, so a reused plan whose leaves still did
    not fit kept halving -- once more on every resume, past the cap, growing
    the very call count the cap exists to bound.

    The cap of 15 is picked against `FakeTokenCounter`: one line costs 11 CSV
    tokens (so planning still succeeds and the error cannot come from there),
    two cost 23, four cost 46. The reused four-line window fails, both halves
    fail, and each half still holds two lines -- so nothing but `MAX_SPLITS`
    can stop the recursion.
    """

    art = tmp_path / "artifacts"
    _refit_run(tmp_path, monkeypatch, artifact_dir=art)
    (art / "correction-windows.jsonl").unlink()

    class ForbidCalls:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, *args, **kwargs):
            raise AssertionError("refit must fail before any call is made")

    _setattr_both(monkeypatch, "RoleClient", ForbidCalls)
    with pytest.raises(ValueError, match=r"already 1 split\(s\) deep"):
        _refit_run(
            tmp_path, monkeypatch, artifact_dir=art, max_window_subtitle_tokens=15
        )

    committed = [
        json.loads(line)["chunk_id"]
        for line in (art / "correction-windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert committed == ["0001"]
    assert not [chunk_id for chunk_id in committed if chunk_id.count("-") > 1]


def test_a_split_tree_committed_under_the_old_cap_still_replays() -> None:
    """Lowering the cap bounds what may be *created*, not what already exists.

    A finished artifact is not re-judged by today's parameters (README_DEV
    「复用的依据是任务身份」), so a `0001-a-a` committed when the cap was 2
    still rebuilds and replays. What the cap refuses is one more split.
    """

    from finesub.llm.chunking import (
        SubtitleWindow,
        estimate_window_budget,
    )
    from finesub.llm.routing.config import DEFAULT_LIMITS
    from finesub.llm.stages.correction.context import ResumeLedger, WindowGeometry

    counter = FakeTokenCounter()
    segments = [
        SubtitleSegment(str(i + 1), i * 20.5, i * 20.5 + 20.0, "台词")
        for i in range(8)
    ]
    geometry = WindowGeometry(
        profile=resolve_profile("text", "none", "quality"),
        counter=counter,
        limits=DEFAULT_LIMITS,
        global_first_id="1",
        global_last_id="8",
        audio_duration=None,
    )
    window = SubtitleWindow(
        chunk_id="0001",
        segments=segments,
        overlap_segments=[],
        boundary_reason="test",
        budget=estimate_window_budget(segments, audio_seconds=164.0, counter=counter),
        length_scale=1.0,
    )

    first, second = geometry.split(window)
    deeper = geometry.split(first)
    assert deeper is not None and deeper[0].chunk_id == "0001-a-a"

    ledger = ResumeLedger(
        enabled=True,
        path=None,
        task_fingerprint="t",
        records={
            "0001": {"split_into": [first.chunk_id, second.chunk_id]},
            first.chunk_id: {"split_into": [half.chunk_id for half in deeper]},
        },
    )

    leaves = ledger.expand_cached_splits(window, geometry)
    assert [leaf.chunk_id for leaf in leaves] == [
        "0001-a-a",
        "0001-a-b",
        second.chunk_id,
    ]
    # Rebuilt in full, yet no further split may be created from them.
    assert not any(geometry.may_split(leaf) for leaf in leaves[:2])
    assert geometry.may_split(window)


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
            length_scale=1.0,
        )

    message = str(excinfo.value)
    assert "chunk=0001" in message
    assert "culprit=quality_cap" in message
    assert "quality_csv_limit=10" in message
    assert "envelope_source=" in message


_FINGERPRINT_ARGS = dict(
    test_profile=False,
    source_fingerprint="s",
    media_identity={},
)


def test_window_invalidation_inputs_is_the_whole_fingerprint_payload(
    monkeypatch,
) -> None:
    """The include list is the contract, and it is enforced rather than
    documented: a payload field missing from it raises instead of being
    silently left out of the fingerprint."""

    assert correction_commit.WINDOW_INVALIDATION_INPUTS == (
        "test_profile",
        "source_fingerprint",
        "media_identity",
    )
    assert correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)

    # Stand in for "someone adds a payload field and forgets to classify it".
    monkeypatch.setattr(
        correction_commit,
        "WINDOW_INVALIDATION_INPUTS",
        ("extra_style", "test_profile", "source_fingerprint"),
        raising=True,
    )
    with pytest.raises(AssertionError) as excinfo:
        correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)
    assert "media_identity" in str(excinfo.value)


def test_every_whitelisted_input_actually_moves_the_fingerprint() -> None:
    """The other direction: a name on the list that no longer reaches the
    payload would be a dead promise."""

    base = correction_commit._task_fingerprint(**_FINGERPRINT_ARGS)
    changed = {
        "test_profile": True,
        "source_fingerprint": "s2",
        "media_identity": {"audio": {"size": 1}},
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


def test_correction_resume_survives_a_style_change_mid_run(tmp_path, monkeypatch) -> None:
    """`extra_style` used to invalidate every committed window.

    It is the user's own instruction, so changing it mid-run does leave the
    file half in one voice and half in another — owner 2026-09-02 judged that
    acceptable rather than pay for re-running everything already finished
    (`docs/llm_harness_behavior.md`, 「纠错窗口中途 resume」, which records what this overrides).
    The named `--style` entries are out for the same reason.
    """

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    _out, calls = _run_windows(
        tmp_path, monkeypatch, artifact_dir=art, extra_style="翻得更俏皮", forbid=True
    )
    assert calls == 0


_OLDER_PROMPT = "zh-subtitle-correction-csv-v-older"


def _ledger_records(art) -> list:
    return [json.loads(line) for line in (art / "correction-windows.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("continuity", ["serial", "parallel"])
def test_correction_resume_keeps_committed_windows_across_a_prompt_version_bump(
    tmp_path, monkeypatch, reported, continuity
) -> None:
    """A bump that only rewords the instructions keeps every committed window (owner 2026-11-08): the
    record says which version answered it, and the run warns once, counting each version apart."""

    art = tmp_path / "artifacts"
    setattr_correction(monkeypatch, "PROMPT_VERSION", _OLDER_PROMPT)
    _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity=continuity)
    assert [record["prompt_version"] for record in _ledger_records(art) if "chunk_id" in record] == [_OLDER_PROMPT]
    plan = (art / "correction-window-plan.json").read_text(encoding="utf-8")
    monkeypatch.undo()
    reported.warnings.clear()

    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity=continuity, forbid=True)

    assert calls == 0
    assert (art / "correction-window-plan.json").read_text(encoding="utf-8") == plan
    older = [item for item in reported.warnings if item.code == "correction-older-prompt-windows"]
    assert len(older) == 1 and f"{_OLDER_PROMPT} 1 窗" in older[0].message
    assert "correction-windows.jsonl" in older[0].action and "当前版本答过的窗也会重做" in older[0].action


def test_a_window_that_no_longer_validates_is_redone_and_not_counted(tmp_path, monkeypatch, reported) -> None:
    """The warning counts windows really replayed, not records read: an older window whose answer the
    current validator no longer passes is asked again, silently."""

    art = tmp_path / "artifacts"
    setattr_correction(monkeypatch, "PROMPT_VERSION", _OLDER_PROMPT)
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    monkeypatch.undo()
    records = _ledger_records(art)
    records[0]["content"] = "<translated>\ntype|position|corrected_text|translation|conf|note\n</translated>"
    (art / "correction-windows.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    reported.warnings.clear()

    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art)

    assert calls >= 1
    assert "correction-older-prompt-windows" not in reported.codes()


@pytest.mark.parametrize("continuity", ["serial", "parallel"])
@pytest.mark.parametrize("field, value", [("variant", "capableB"), ("capability_tier", "no-such-tier")])
def test_a_record_naming_a_variant_or_tier_that_no_longer_exists_is_redone(
    tmp_path, monkeypatch, continuity, field, value
) -> None:
    """`capableB` was deleted (2026-11): with the prompt version out of the fingerprint such a record can
    reach replay, and it reads as one the current validator no longer passes -- redone, never a crash."""

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity=continuity)
    records = _ledger_records(art)
    records[0][field] = value
    (art / "correction-windows.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")

    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity=continuity)

    assert calls >= 1


def test_after_a_bump_the_window_plan_and_the_parallel_entry_set_are_reused(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    options = dict(stable_json=stable, task_artifact_dir=art, test_profile=False, audio_path=None,
                   video_path=None, token_counter=FakeTokenCounter(), profile=resolve_profile("text", "none"))
    first = correction_run.resolve_correction_plan(**options)
    correction_commit._append_window_cache(
        art / "correction-windows.jsonl",
        {"parallel_entry_set": ["甲"], "task_fingerprint": first.task_fingerprint})
    setattr_correction(monkeypatch, "PROMPT_VERSION", "zh-subtitle-correction-csv-v-next")
    setattr_correction(monkeypatch, "GROUPED_PROMPT_VERSION", "zh-subtitle-correction-3pass-v-next")

    second = correction_run.resolve_correction_plan(**options)

    assert second.reused and second.task_fingerprint == first.task_fingerprint
    assert correction_commit._load_parallel_entry_set(
        art / "correction-windows.jsonl", second.task_fingerprint) == ["甲"]


def test_a_grouped_window_and_the_fallback_window_never_replay_each_other(tmp_path) -> None:
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")
    plain = correction_run.resolve_correction_plan(
        stable_json=stable, task_artifact_dir=None, test_profile=False, audio_path=None, video_path=None,
        token_counter=FakeTokenCounter(), profile=resolve_profile("text", "none"),
    ).windows[0]
    plain = dataclass_replace(plain, length_scale=1.0)
    grouped = dataclass_replace(plain, grouped=True)

    def ledger(answered):
        record = {"chunk_id": plain.chunk_id, "input_hash_core": correction_commit._window_input_hash(answered),
                  "content": _RESUME_GOOD, "variant": "capableC", "capability_tier": "capable"}
        return correction_context.ResumeLedger(
            enabled=True, path=None, task_fingerprint="f", records={plain.chunk_id: record})

    assert ledger(plain).replay(plain) is not None
    assert ledger(plain).replay(grouped) is None
    assert ledger(grouped).replay(plain) is None


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


def test_keep_entries_transfer_chain_across_windows(tmp_path, monkeypatch) -> None:
    """v17 pass-through, task-flow shape: window 1's <keep_entries> reaches
    window 2's opening message as carried entries and its delivery as
    entry_details; the resume cache records the chain."""

    from dataclasses import replace

    from finesub.llm.chunking import load_segments_from_stable_json

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

    # Window 1 requests the entry; window 2 keeps what it was handed.
    requests = [
        "<reasoning>x</reasoning>\n<window_notes>杂谈回。</window_notes>\n"
        "<requested_entries>\nエーちゃん\n</requested_entries>\n<keep_entries></keep_entries>\n"
        "<search_queries>\n</search_queries>",
        "<reasoning>x</reasoning>\n<window_notes></window_notes>\n"
        "<requested_entries></requested_entries>\n<keep_entries>\n主播A\n</keep_entries>\n"
        "<search_queries>\n</search_queries>",
    ]
    outputs = [
        (
            "<singles>\ntype|position|corrected_text|translation|conf|note\nsub|1|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|一。|一|8|\n</translated>\n"
            "<next_advice></next_advice>\n"
            "<keep_entries>\n主播A\n</keep_entries>"
        ),
        (
            "<singles>\ntype|position|corrected_text|translation|conf|note\nsub|1|x|2|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|二。|二|8|\n</translated>\n"
            "<next_advice></next_advice>\n"
            "<keep_entries></keep_entries>"
        ),
    ]
    openings: list[str] = []
    deliveries: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            if len(messages) == 2:  # the request turn
                openings.append(messages[1]["content"])
                content = requests[len(openings) - 1]
            else:  # the delivery turn, same conversation
                deliveries.append("\n".join(str(m["content"]) for m in messages if m["role"] == "user"))
                content = outputs[len(deliveries) - 1]
            return LLMCallResult(
                content=content, role=role, model="fake", fallback_used=False,
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
        length_scale=1.0,
    )

    # Window 1 opens with no carried entries; window 2 opens with the entry
    # window 1 kept (full text).
    assert "（无）" in openings[0].split("<entry_details>")[1]
    assert "关西腔。" in openings[1].split("<entry_details>")[1]
    # Both deliveries see the entry: window 1 via its own request (fill),
    # window 2 via the transfer (opening + fill).
    assert "关西腔。" in deliveries[0]
    assert "关西腔。" in deliveries[1]
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
    # Window 2's request turn kept the carried entry, so the chain continues
    # even though its delivery's own <keep_entries> is empty (union, v17).
    assert by_chunk["0002"]["keep_entries"] == ["主播A"]
    assert by_chunk["0002"]["injected_entries"] == ["主播A"]

    # Knowledge text is an execution input, not a reason to throw away windows
    # that were already parsed and committed: a rerun after the entry changed
    # replays both windows without a single call.
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
        length_scale=1.0,
    )
    assert len(openings) == 2 and len(deliveries) == 2


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
        "type|position|corrected_text|translation|conf|note\n"
        "sub|1|one|一|high|\n"
    )
    complete = (
        "<translated>\n"
        "sub|1|one|一|high|\n"
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
    # the correction's prompt version rides in the geometry metadata and is audit-only too
    assert "prompt_version" in geometry and not set(geometry) & set(reuse_key)
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
    tmp_path, monkeypatch, reported
) -> None:
    """G3: allowed, but the ledger later live windows read starts short."""

    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art, continuity="parallel")
    reported.warnings.clear()
    _out, calls = _run_windows(
        tmp_path, monkeypatch, artifact_dir=art, continuity="serial", forbid=True
    )

    assert calls == 0
    assert reported.codes() == ["correction-advice-missing"]
    assert "produced in parallel mode" in reported.joined()


def test_a_correction_run_reports_one_progress_event_per_window(
    tmp_path, monkeypatch, reported
) -> None:
    """The stage's own line, and the reason the whole migration happened.

    Counted through `commit_window`, so a window replayed from the ledger is
    just as visible as one that was corrected here -- a resumed run that shows
    nothing until it reaches fresh work reads as a hang.
    """

    _run_windows(tmp_path, monkeypatch, artifact_dir=tmp_path / "artifacts")

    windows = [
        call for call in reported.progress_calls if call["stage"] == "translated-srt"
    ]
    assert windows, "the correction stage reported no progress at all"
    assert [call["completed"] for call in windows] == list(
        range(1, len(windows) + 1)
    )
    assert all(call["unit"] == "windows" for call in windows)
    assert all(call["detail"] for call in windows)


def test_a_split_window_grows_the_denominator_the_stage_reports() -> None:
    """A window that overran its envelope splits, and the whole gets bigger.

    Reported through the counter rather than the driver's own bookkeeping
    because the two drivers grow different things: the serial one its window
    list, the parallel one a queue inside a single worker whose future count
    never changes.
    """

    from finesub.llm.stages.correction.progress import WindowProgress

    class _Recorder:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def progress(self, stage, *, completed, total=None, unit="", detail="") -> None:
            self.calls.append((completed, total))

        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    recorder = _Recorder()
    with reporting_to(recorder):
        progress = WindowProgress(3)

    progress.unit_done("0001")
    progress.add_units(1)  # 0002 split into 0002-a / 0002-b
    progress.unit_done("0002-a")
    progress.unit_done("0002-b")

    assert recorder.calls == [(1, 3), (2, 4), (3, 4)]
    assert progress.splits == 1


def test_a_task_rewrites_its_report_even_when_this_run_spent_nothing(tmp_path, monkeypatch) -> None:
    """A stale book must not survive in the Markdown either.

    The report used to be written *during* the run, with whatever
    `agent-session-usage.json` was in the artifact directory at the time. A
    later run that uses no agent session deletes that JSON -- and the report
    went on quoting tokens whose source was gone. The task now books the
    sessions first and writes the report after.
    """

    from finesub.llm.exchange_metadata import AGENT_SESSION_USAGE_FILENAME

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / AGENT_SESSION_USAGE_FILENAME).write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "provider_tier": "LOCAL_AGY",
                        "model": "gemini-3.7-flash",
                        "usage": {"total_input_tokens": 99999, "output_tokens": 5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    correction_orchestration.write_task_report(artifacts, task_id="t", outputs={})
    assert "99999" in (artifacts / "task-report.md").read_text(encoding="utf-8")

    # A run that executes but never opens an agent session (the windows are
    # stubbed; the text-only fast shape needs no preparation).
    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )

    def windows(**kwargs):  # type: ignore[no-untyped-def]
        translated = Path(kwargs["output_path"]).with_name("clip-translated.srt")
        translated.write_text("", encoding="utf-8")
        return translated

    monkeypatch.setattr(correction_orchestration, "execute_correction_windows", windows)
    monkeypatch.setattr(
        correction_orchestration, "run_research_stage", lambda **k: pytest.fail("no preparation")
    )
    correction_orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable,
            output_path=tmp_path / "clip.srt",
            profile=resolve_profile("text", "none", "quality"),
            fast="on",
            postprocess_profile=None,
            task_id="t",
            task_artifact_dir=artifacts,
            knowledge="none",
            test_profile=True,
        )
    )
    assert not (artifacts / AGENT_SESSION_USAGE_FILENAME).exists()
    assert "99999" not in (artifacts / "task-report.md").read_text(encoding="utf-8")


def test_a_window_may_be_split_at_most_once() -> None:
    """Each split doubles what that window costs, and a half-sized window that
    still does not fit is not a size problem any more.

    Owner set the cap to 2 on 2026-09-02 and to 1 on 2026-09-03. The depth is
    read off the id because `split_window_in_half` suffixes `-a`/`-b`: the id
    IS the lineage, so it cannot drift from a counter."""

    from finesub.llm.chunking import SubtitleWindow, estimate_window_budget
    from finesub.llm.stages.correction.context import WindowGeometry

    def window(chunk_id: str) -> SubtitleWindow:
        segments = [SubtitleSegment("1", 0.0, 1.0, "テスト")]
        return SubtitleWindow(
            chunk_id=chunk_id,
            segments=segments,
            overlap_segments=[],
            boundary_reason="test",
            budget=estimate_window_budget(segments, audio_seconds=1.0),
            length_scale=1.0,
        )

    assert WindowGeometry.MAX_SPLITS == 1
    assert WindowGeometry.split_depth(window("0007")) == 0
    assert WindowGeometry.split_depth(window("0007-a")) == 1
    # The reading rule is independent of the cap: it still answers 2 for an id
    # the cap now keeps production from ever creating.
    assert WindowGeometry.split_depth(window("0007-a-b")) == 2
