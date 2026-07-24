from __future__ import annotations

import json
import sys
from pathlib import Path

from llm import correction_translation
from llm.chunking import SubtitleSegment
from llm.client import LLMCallResult, UploadedFileRef, attach_file_to_messages
from llm.clip_prefetch import WindowClipPrefetcher
from llm.config import CapabilityTier, LLMRole
from llm.stages.correction_loop import execute_correction_windows
from llm.profiles import VIDEO_SAMPLE_FPS, resolve_profile
from llm.srt_utils import parse_srt
from llm.stages.fast_session import run_fast_session
from llm.stages.plan import plan_fast_window


class FakeTokenCounter:
    source = "test-fake"

    def count_text(self, text: str) -> int:
        return max(1, len(text or "") // 2)

    def count_texts(self, texts) -> int:
        return sum(self.count_text(text) for text in texts)

    def count_audio_seconds(self, seconds: float) -> int:
        return max(0, int(seconds * 32))


def _segments() -> list[SubtitleSegment]:
    return [
        SubtitleSegment("1", 0.0, 1.0, "一。"),
        SubtitleSegment("2", 1.5, 2.5, "二。"),
    ]


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


def _fake_upload(path: Path) -> UploadedFileRef:
    path = Path(path)
    mime = "video/mp4" if path.suffix == ".mp4" else "audio/aac"
    return UploadedFileRef(file_id=f"files/{path.name}", filename=path.name, mime_type=mime)


def test_attach_file_marks_video_clips_low_res_low_fps() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]

    video = attach_file_to_messages(
        messages, UploadedFileRef("files/1", "0001.mp4", "video/mp4")
    )
    video_block = video[1]["content"][-1]["file"]
    assert video_block["detail"] == "low"
    assert video_block["video_metadata"] == {"fps": VIDEO_SAMPLE_FPS}

    audio = attach_file_to_messages(
        messages, UploadedFileRef("files/2", "0001.aac", "audio/aac")
    )
    audio_block = audio[1]["content"][-1]["file"]
    assert "detail" not in audio_block
    assert "video_metadata" not in audio_block


def test_prefetcher_respects_clip_suffix(tmp_path) -> None:
    extracted: list[tuple[Path, Path]] = []

    def fake_extract(src, start, end, out):
        extracted.append((Path(src), Path(out)))
        return Path(out)

    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("mm", "high")
    )
    prefetcher = WindowClipPrefetcher(
        tmp_path / "v.mp4",
        tmp_path / "clips",
        extract_fn=fake_extract,
        upload_fn=_fake_upload,
        clip_suffix=".mp4",
    )
    try:
        ref = prefetcher.get_ref(window)
    finally:
        prefetcher.shutdown()

    assert extracted[0][0].name == "v.mp4"
    assert extracted[0][1].name == "0001.mp4"
    assert ref is not None and ref.mime_type == "video/mp4"


def test_mm_high_sends_mp4_to_correction_and_aac_to_query_round(
    tmp_path, monkeypatch
) -> None:
    profile = resolve_profile("mm", "high")
    stable_json = _stable_json(tmp_path)
    audio_path = tmp_path / "a.flac"
    video_path = tmp_path / "v.mp4"
    audio_extracts: list[Path] = []
    video_extracts: list[Path] = []
    calls: list[tuple[LLMRole, UploadedFileRef | None, str]] = []

    def fake_audio_extract(src, start, end, out):
        assert Path(src) == audio_path
        audio_extracts.append(Path(out))
        return Path(out)

    def fake_video_extract(src, start, end, out):
        assert Path(src) == video_path
        video_extracts.append(Path(out))
        return Path(out)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
            prompt_text = "\n".join(str(m.get("content", "")) for m in messages)
            calls.append((role, kwargs.get("file_ref"), prompt_text))
            if role is LLMRole.LIGHTWEIGHT_MULTIMODAL:
                content = (
                    "<window_notes>注</window_notes>\n"
                    "<keep_entries></keep_entries>\n"
                    "<search_queries>\n游戏A\n</search_queries>"
                )
            else:
                content = (
                    "<singles>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\n"
                    "sub|1|1.0|one|一|8|译1字；宜保持独立\n"
                    "sub|2|1.0|two|二|8|译1字；宜保持独立\n"
                    "</singles>\n"
                    "<translated>\ntype|position|duration|gap|corrected_text|translation|conf|char_count|note\nsub|1|1.0|one|一|8|\nsub|2|1.0|two|二|8|\n</translated>"
                    "\n<next_advice></next_advice>"
                )
            return LLMCallResult(
                content=content,
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            return []

    monkeypatch.setattr(
        "llm.stages.correction_loop.probe_audio_duration", lambda path: 200.0
    )
    monkeypatch.setattr(
        "llm.stages.correction_loop.extract_window_clip", fake_audio_extract
    )
    monkeypatch.setattr(
        "llm.stages.correction_loop.extract_window_video_clip", fake_video_extract
    )
    monkeypatch.setattr("llm.stages.correction_loop.upload_gemini_file", _fake_upload)
    monkeypatch.setattr("llm.stages.correction_loop.LiteLLMRoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=audio_path,
        video_path=video_path,
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        enable_web_search=True,
        search_client=FakeSearchClient(),
        profile=profile,
    )

    # Query round: lite model with the .aac; correction round: audio_multimodal
    # with the .mp4 (single window "0001").
    assert [role for role, _, _ in calls] == [
        LLMRole.LIGHTWEIGHT_MULTIMODAL,
        LLMRole.AUDIO_MULTIMODAL,
    ]
    query_ref = calls[0][1]
    correction_ref = calls[1][1]
    assert query_ref is not None and query_ref.filename == "0001.aac"
    assert correction_ref is not None and correction_ref.filename == "0001.mp4"
    # Media identity is carried by file_ref; v40 no longer duplicates the
    # local filename inside the textual prompt.
    assert "0001.aac（" not in calls[0][2]
    assert "0001.mp4（" not in calls[1][2]
    assert audio_extracts and audio_extracts[0].name == "0001.aac"
    assert video_extracts and video_extracts[0].name == "0001.mp4"
    assert [segment.text for segment in parse_srt(output.read_text(encoding="utf-8"))] == [
        "一",
        "二",
    ]


def test_fast_session_uploads_the_video_clip_on_mm_high(tmp_path, monkeypatch) -> None:
    profile = resolve_profile("mm", "high")
    window = plan_fast_window(
        _segments(), counter=FakeTokenCounter(), profile=profile
    )
    video_extracts: list[Path] = []
    seen_prompts: list[str] = []

    def fake_video_extract(src, start, end, out):
        video_extracts.append(Path(out))
        return Path(out)

    class FakeClient:
        def complete(self, role, messages, **kwargs):
            if callable(messages):  # tiered factory (correction round)
                messages = messages(CapabilityTier.CAPABLE)
            seen_prompts.append(
                "\n".join(str(m.get("content", "")) for m in messages)
            )
            return LLMCallResult(
                content=(
                    "<analysis_notes>\n笔记。\n</analysis_notes>\n"
                    "<requested_entries>\n</requested_entries>\n"
                    "<keep_entries>\n</keep_entries>\n"
                    "<search_queries>\n</search_queries>"
                ),
                role=role,
                model="fake",
                fallback_used=False,
                raw_response={},
            )

    monkeypatch.setattr("llm.audio_clips.extract_window_video_clip", fake_video_extract)
    monkeypatch.setattr("llm.stages.fast_session.upload_gemini_file", _fake_upload)

    result, file_ref = run_fast_session(
        window=window,
        segment_count=2,
        audio_path=tmp_path / "a.flac",
        video_path=tmp_path / "v.mp4",
        clip_dir=tmp_path / "clips",
        knowledge_root=tmp_path / "kb",
        client=FakeClient(),
        enable_web_search=False,
        token_counter=FakeTokenCounter(),
        profile=profile,
    )

    assert file_ref is not None and file_ref.filename == "0001.mp4"
    assert video_extracts and video_extracts[0].name == "0001.mp4"
    assert "0001.mp4（" not in seen_prompts[0]
    assert result.analysis_notes == "笔记。"


def test_cli_rejects_bad_video_flag_combinations(tmp_path, monkeypatch, capsys) -> None:
    stable = _stable_json(tmp_path)

    # --video outside mm-high.
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", str(stable), "--route", "mm", "--level", "med", "--video", "v.mp4"],
    )
    assert correction_translation.main() == 2
    assert "--video only applies" in capsys.readouterr().err

    # mm-high --execute without --video.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(stable),
            "--route",
            "mm",
            "--level",
            "high",
            "--execute",
            "--audio",
            "a.flac",
        ],
    )
    assert correction_translation.main() == 2
    assert "--video is required" in capsys.readouterr().err
