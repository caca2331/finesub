from __future__ import annotations

import json
import sys
from pathlib import Path

from finesub.llm import correction_translation
from finesub.llm.chunking import SubtitleSegment
from finesub.llm.client import LLMCallResult, attach_file_to_messages
from finesub.llm.media_upload import UploadedFileRef
from finesub.llm.clip_prefetch import WindowClipPrefetcher
from finesub.llm.routing.config import LLMRole
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.routing.profiles import VIDEO_SAMPLE_FPS, resolve_profile
from finesub.llm.stages.plan import plan_fast_window


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


def _fake_upload(path: Path, *_args, **_kwargs) -> UploadedFileRef:
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
        _segments(), counter=FakeTokenCounter(), profile=resolve_profile("video", "local", "quality")
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


def test_video_run_carries_one_clip(tmp_path, monkeypatch) -> None:
    """--media video: the window task carries one mp4; the request turn and
    the delivery share it."""

    profile = resolve_profile("video", "local", "quality")
    stable_json = _stable_json(tmp_path)
    audio_path = tmp_path / "a.flac"
    video_path = tmp_path / "v.mp4"
    audio_extracts: list[Path] = []
    video_extracts: list[Path] = []
    calls: list[tuple[LLMRole, UploadedFileRef | None, int]] = []

    def fake_audio_extract(src, start, end, out):
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
            if callable(messages):
                messages = messages("capableC")
            calls.append((role, kwargs.get("file_ref"), len(messages)))
            content = (
                "<singles>\ntype|position|corrected_text|translation|conf|note\n"
                "sub|1|one|一|8|译1字；宜保持独立\n"
                "sub|2|two|二|8|译1字；宜保持独立\n"
                "</singles>\n"
                "<translated>\ntype|position|corrected_text|translation|conf|note\nsub|1|one|一|8|\nsub|2|two|二|8|\n</translated>"
                "\n<next_advice></next_advice>"
            )
            return LLMCallResult(
                content=content, role=role, model="fake", fallback_used=False,
                raw_response={"candidates": [{"finishReason": "STOP"}]},
            )

    class FakeSearchClient:
        def search_many(self, queries, *, max_queries=None):
            return []

    monkeypatch.setattr(
        "finesub.llm.stages.correction.run.probe_audio_duration", lambda path: 200.0
    )
    monkeypatch.setattr(
        "finesub.llm.stages.correction.run.extract_window_clip", fake_audio_extract
    )
    monkeypatch.setattr(
        "finesub.llm.stages.correction.run.extract_window_video_clip", fake_video_extract
    )
    monkeypatch.setattr("finesub.llm.media_upload.upload_gemini_file", _fake_upload)
    monkeypatch.setattr("finesub.llm.stages.correction.run.RoleClient", FakeClient)

    output = execute_correction_windows(
        stable_json=stable_json,
        output_path=tmp_path / "out.srt",
        audio_path=audio_path,
        video_path=video_path,
        clip_dir=tmp_path / "clips",
        token_counter=FakeTokenCounter(),
        search_client=FakeSearchClient(),
        profile=profile,
        length_scale=1.0,
    )

    assert output.exists()
    # The request turn delivered at once: one call, on the correction role,
    # carrying the correction clip; no second clip was ever cut.
    assert [role for role, _, _ in calls] == [LLMRole.AUDIO_MULTIMODAL]
    assert calls[0][1] is not None and calls[0][1].filename == "0001.mp4"
    assert [p.name for p in video_extracts] == ["0001.mp4"]
    assert audio_extracts == []


def test_cli_rejects_bad_video_flag_combinations(tmp_path, monkeypatch, capsys) -> None:
    stable = _stable_json(tmp_path)

    # --video outside media=video.
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", str(stable), "--media", "audio", "--video", "v.mp4"],
    )
    assert correction_translation.main() == 2
    assert "--video only applies" in capsys.readouterr().err

    # media=video --execute without --video.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(stable),
            "--media",
            "video",
            "--execute",
            "--audio",
            "a.flac",
        ],
    )
    assert correction_translation.main() == 2
    assert "--video is required" in capsys.readouterr().err


def test_cli_refuses_the_speaker_judge_on_efficiency(tmp_path, monkeypatch, capsys) -> None:
    """The judge needs the three-pass split, which efficiency does not run (owner 2026-10-30)."""

    stable = _stable_json(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            str(stable),
            "--media",
            "text",
            "--retrieval",
            "none",
            "--difficulty",
            "efficiency",
            "--speaker-judge",
            "text",
        ],
    )
    assert correction_translation.main() == 2
    assert "three-pass split" in capsys.readouterr().err


def test_clip_sampling_rate_matches_what_the_rest_call_asks_for() -> None:
    """One clip serves both transports only while the two rates agree.

    agy cannot request server-side sampling, so the clip itself is cut at the
    rate the REST path also sends as `videoMetadata.fps`. `media` may not
    import `llm`, so the two constants are pinned here rather than shared --
    and the filter string is built from the constant so it cannot drift on its
    own either.
    """

    from finesub.media.ffmpeg import VIDEO_FILTER, VIDEO_SAMPLE_FPS
    from finesub.llm.routing.profiles import VIDEO_SAMPLE_FPS as REQUESTED_FPS

    assert VIDEO_SAMPLE_FPS == REQUESTED_FPS
    assert f"fps=fps={VIDEO_SAMPLE_FPS:g}" in VIDEO_FILTER
    assert f"setpts=N/({VIDEO_SAMPLE_FPS:g}*TB)" in VIDEO_FILTER
