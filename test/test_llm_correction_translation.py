from __future__ import annotations

import json
from pathlib import Path

import pytest

import llm.correction_translation as correction_orchestration
import llm.stages.correction_loop as correction_loop

from llm.client import LLMCallResult, UploadedFileRef
from llm.chunking import SubtitleSegment, plan_correction_windows
from llm.config import CapabilityTier, LLMRole
from llm.stages.correction_loop import (
    _extract_next_advice,
    _extract_task_update_feedback,
    _extract_window_notes,
    _is_output_limited,
    _output_limit_check,
    execute_correction_windows,
    run_window_query_round,
)
from llm.knowledge.base import append_task_artifact
from llm.profiles import resolve_profile
from llm.prompts import ContextPack, render_advice_ledger
from llm.srt_utils import parse_srt


def _setattr_both(monkeypatch, name, value):
    """Patch a name on both the loop module and the orchestration module.

    execute_correction_windows binds these names in llm.stages.correction_loop;
    run_full_correction binds a few of them in llm.correction_translation.
    """

    patched = False
    for module in (correction_loop, correction_orchestration):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value, raising=True)
            patched = True
    assert patched, f"test patch target {name!r} no longer exists"



class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))



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


def test_extract_task_update_feedback_reads_first_feedback_block() -> None:
    content = (
        "<singles>\nsub|1|1.0|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\n1|good|好\n</translated>\n"
        '<task_update_feedback>{"reusable_terms":["A 固定译为甲"]}</task_update_feedback>'
    )

    assert _extract_task_update_feedback(content) == '{"reusable_terms":["A 固定译为甲"]}'


def test_extract_next_advice_reads_block_and_caps_length() -> None:
    content = (
        "<singles>\nsub|1|1.0|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\n1|good|好\n</translated>\n"
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
        "sub|1|1.0|A|甲|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "1|A|甲\n"
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
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        enable_web_search=False,
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
    bad = "<translated>\n9|nine|九\n</translated>"
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
        "sub|2|1.0|二。|二|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "1|一。|一\n"
        "2|二。|二\n"
        "</translated>"
    )
    responses = [bad, good]
    sampling_kwargs: list[dict] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self.calls = 0

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=1,
        enable_web_search=False,
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
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
        "sub|2|1.0|二|二|8|译1字；宜保持独立\n"
        "sub|3|1.0|三。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n2|二|二\n3|三。|三\n</translated>\n"
        "<next_advice>术语X固定译为Y。</next_advice>"
    )
    # Dense 1s spacing with the default overlap floor (10) exceeds the first
    # half's length, so the -b half falls back to no overlap: it starts at "4".
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|4|1.0|四|四|8|译1字；宜保持独立\n"
        "sub|5|1.0|五。|五|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n4|四|四\n5|五。|五\n</translated>\n"
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
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        enable_web_search=False,
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
    assert "- output_limit_basis: output_tokens_plus_thinking_tokens" in exchange_text
    assert "- output_limit_observed_tokens: 65500" in exchange_text
    assert "- output_limit_threshold_tokens: 65436" in exchange_text

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
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
        "sub|2|1.0|二|二|8|译1字；宜保持独立\n"
        "sub|3|1.0|三。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n2|二|二\n3|三。|三\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|4|1.0|四|四|8|译1字；宜保持独立\n"
        "sub|5|1.0|五。|五|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n4|四|四\n5|五。|五\n</translated>"
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
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)
    _setattr_both(
        monkeypatch,
        "probe_audio_duration", lambda _: 100.0)
    _setattr_both(
        monkeypatch,
        "extract_window_clip", fake_extract)
    _setattr_both(
        monkeypatch,
        "upload_gemini_file", fake_upload)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=tmp_path / "audio.wav",
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        enable_web_search=False,
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
        ("STOP", "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|9|1.0|x|9|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n9|坏|坏\n</translated>"),
        ("STOP", "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n</translated>"),
    ]

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)
    _setattr_both(
        monkeypatch,
        "probe_audio_duration", lambda _: 100.0)
    _setattr_both(
        monkeypatch,
        "extract_window_clip",
        lambda audio_path, clip_start, clip_end, out_path, **kwargs: out_path,
    )
    _setattr_both(
        monkeypatch,
        "upload_gemini_file",
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
        enable_web_search=False,
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
    truncated = "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n"
    first_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|第一句。|一|8|译1字；宜保持独立\n"
        "sub|2|1.0|第二句。|二|8|译1字；宜保持独立\n"
        "sub|3|1.0|第三句。|三|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|第一句。|一\n2|第二句。|二\n3|第三句。|三\n</translated>"
    )
    second_half = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|4|1.0|第四句。|四|8|译1字；宜保持独立\n"
        "sub|5|1.0|第五句。|五|8|译1字；宜保持独立\n"
        "sub|6|1.0|第六句。|六|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n4|第四句。|四\n5|第五句。|五\n6|第六句。|六\n</translated>"
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
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)

    execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        max_retries_per_window=2,
        enable_web_search=False,
    )

    # Sparse 40s spacing: zero content-driven overlap at the split, so the -b
    # half starts at id 4 (clip_start 115.0) and its preceding block carries
    # the raw ids 1-3 at negative clip-relative times.
    second_user = seen_messages[2][1]["content"]
    assert "<preceding_context>" in second_user
    assert "3|-35.0|1.0|0.0|第3句。" in second_user
    assert "4|5.0|1.0|" in second_user
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
                messages = messages(CapabilityTier.CAPABLE)
            raise RuntimeError("HTTP 503 high demand")

    _setattr_both(
        monkeypatch,
        "LiteLLMRoleClient", FakeClient)

    with pytest.raises(RuntimeError):
        execute_correction_windows(
            stable_json=stable_json,
            output_path=tmp_path / "out.srt",
            token_counter=FakeTokenCounter(),
            max_retries_per_window=1,
            enable_web_search=False,
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
    from llm.web_search import QuerySearchResult, SearchResultItem

    query_output = (
        "<window_notes>\n本窗口疑似在打BOSS，BOSS名待定。\n</window_notes>\n"
        "<keep_entries></keep_entries>\n"
        "<search_queries>\n游戏B 角色名\n</search_queries>"
    )
    bad = "<translated>\n9|nine|九\n</translated>"
    good = (
        "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
        "sub|2|1.0|二。|二|8|译1字；宜与前一句合并\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
        "sub|1|1.0|一。|一|8|\n"
        "sub|2|1.0|二。|二|7|术语note\n"
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
                messages = messages(CapabilityTier.CAPABLE)
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
        "LiteLLMRoleClient", FakeClient)
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
    assert [path.name for path in exchanges] == [
        "001-correction-0001-query-attempt0.md",
        "002-correction-0001-attempt0.md",
        "003-correction-0001-attempt1.md",
    ]
    query_exchange = exchanges[0].read_text(encoding="utf-8")
    assert "## 请求（system）" in query_exchange
    assert "## 模型响应" in query_exchange
    assert "游戏B 角色名" in query_exchange
    assert "- finish_reason: STOP" in query_exchange
    assert "- output_limit_observed_tokens: 130" in query_exchange
    assert "角色小明的资料" in exchanges[1].read_text(encoding="utf-8")


def test_query_round_resumes_model_output_but_reexecutes_search(tmp_path) -> None:
    from llm.web_search import QuerySearchResult, SearchResultItem

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
        profile=resolve_profile("mm", "low"),
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
                messages = messages(CapabilityTier.CAPABLE)
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
                    "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n1|一。|一\n</translated>"
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
        "LiteLLMRoleClient", FakeClient)

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
    "sub|1|1.0|一。|一|8|\n"
    "sub|2|1.0|二。|二|7|术语note\n"
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
            messages = messages(CapabilityTier.CAPABLE)
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


def _run_windows(tmp_path, monkeypatch, *, artifact_dir, resume=True, extra_style="", forbid=False):
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps(_RESUME_STABLE), encoding="utf-8")

    class Client(_CountingClient):
        calls = 0

    Client.forbid = forbid
    _setattr_both(
        monkeypatch,
        "LiteLLMRoleClient", Client)
    out = execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        enable_web_search=False,
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


def test_correction_resume_ignores_cache_on_fingerprint_mismatch(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    # Changing a fingerprinted input (extra_style) must recompute, not reuse.
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, extra_style="翻得更俏皮")
    assert calls >= 1


def test_correction_resume_invalidates_on_prompt_version_bump(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    monkeypatch.setattr(
        correction_loop,
        "PROMPT_VERSION",
        "zh-subtitle-correction-csv-v-next",
        raising=True,
    )
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls >= 1


def test_correction_resume_fingerprint_includes_source_media_identity(tmp_path) -> None:
    media = tmp_path / "source.aac"
    media.write_bytes(b"first")
    common = dict(
        extra_style="",
        common_mistakes_block="",
        context_pack=None,
        test_profile=False,
        task_update_feedback=False,
    )
    first = correction_loop._task_fingerprint(
        **common,
        media_identity={"audio": correction_loop._media_identity(media)},
    )
    media.write_bytes(b"second-version")
    second = correction_loop._task_fingerprint(
        **common,
        media_identity={"audio": correction_loop._media_identity(media)},
    )

    assert first != second


def test_correction_resume_disabled_ignores_existing_cache(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art, resume=False)
    assert calls >= 1


def test_correction_resume_ignores_cache_on_input_hash_mismatch(tmp_path, monkeypatch) -> None:
    art = tmp_path / "artifacts"
    _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    # A stale window input (e.g. regenerated stable.json) no longer hashes to the
    # cached input_hash, so that window must recompute.
    cache_file = art / "correction-windows.jsonl"
    record = json.loads(cache_file.read_text(encoding="utf-8").splitlines()[0])
    record["input_hash"] = "sha256:deadbeef0000"
    cache_file.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _out, calls = _run_windows(tmp_path, monkeypatch, artifact_dir=art)
    assert calls >= 1


def test_advice_ledger_front_truncates_to_token_budget() -> None:
    from llm.config import ADVICE_LEDGER_MAX_TOKENS
    from llm.token_truncate import truncate_text_only

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
        "sub|1|1.0|一。|一|8|译1字；宜保持独立\n"
        "</singles>\n"
        "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|一。|一|8|\n</translated>"
    )
    query_messages_seen = []
    correction_messages_seen = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
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

    _setattr_both(monkeypatch, "LiteLLMRoleClient", FakeClient)

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

    from llm.chunking import load_segments_from_stable_json, plan_correction_windows

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
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|x|1|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|一。|一|8|\n</translated>\n"
            "<next_advice></next_advice>\n"
            "<keep_entries>\n主播A\n</keep_entries>"
        ),
        (
            "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|2|1.0|x|2|8|译1字；宜保持独立\n</singles>\n<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|2|1.0|二。|二|8|\n</translated>\n"
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
                messages = messages(CapabilityTier.CAPABLE)
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

    _setattr_both(monkeypatch, "LiteLLMRoleClient", FakeClient)

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

    # A query-round-requested entry is part of the exact model input. Changing
    # its body invalidates both cached windows instead of replaying stale text.
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
    assert len(correction_users) == 4
    assert "东京腔。" in correction_users[2]
