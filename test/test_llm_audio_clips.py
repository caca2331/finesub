from __future__ import annotations

import shutil
from dataclasses import dataclass

import pytest

from llm.audio_clips import (
    CLIP_EDGE_PAD_SECONDS,
    CLIP_PAD_SECONDS,
    CLIP_AUDIO_SUFFIX,
    CLIP_VIDEO_SUFFIX,
    compute_clip_range,
    default_clip_path,
    default_video_clip_path,
    extract_window_clip,
)
from llm.ffmpeg_clips import (
    AUDIO_CODEC_ARGS,
    VIDEO_ENCODER_ARGS,
    build_audio_clip_command,
    build_video_clip_command,
    extract_audio_clip,
    probe_media_duration,
)


@dataclass(frozen=True)
class Seg:
    id: str
    start: float
    end: float


def test_compute_clip_range_pads_interior_windows_by_5s() -> None:
    segments = [Seg("10", 100.0, 101.0), Seg("11", 130.0, 131.0)]

    clip_start, clip_end = compute_clip_range(
        segments, global_first_id="1", global_last_id="99", audio_duration=1000.0
    )

    assert clip_start == pytest.approx(100.0 - CLIP_PAD_SECONDS)
    assert clip_end == pytest.approx(131.0 + CLIP_PAD_SECONDS)


def test_compute_clip_range_uses_60s_edge_pads_with_clamping() -> None:
    segments = [Seg("1", 10.0, 11.0), Seg("2", 50.0, 51.0)]

    clip_start, clip_end = compute_clip_range(
        segments, global_first_id="1", global_last_id="99", audio_duration=1000.0
    )
    assert clip_start == 0.0
    assert clip_end == pytest.approx(51.0 + CLIP_PAD_SECONDS)

    clip_start, clip_end = compute_clip_range(
        segments, global_first_id="0", global_last_id="2", audio_duration=80.0
    )
    assert clip_start == pytest.approx(10.0 - CLIP_PAD_SECONDS)
    assert clip_end == 80.0

    clip_start, clip_end = compute_clip_range(
        segments, global_first_id="0", global_last_id="2", audio_duration=1000.0
    )
    assert clip_end == pytest.approx(51.0 + CLIP_EDGE_PAD_SECONDS)


def test_compute_clip_range_never_cuts_below_last_segment_end() -> None:
    segments = [Seg("5", 10.0, 20.5)]

    _clip_start, clip_end = compute_clip_range(
        segments, global_first_id="0", global_last_id="9", audio_duration=20.0
    )

    assert clip_end == 20.5


def test_compute_clip_range_rejects_empty_window() -> None:
    with pytest.raises(ValueError):
        compute_clip_range([], global_first_id="1", global_last_id="2")


def test_default_clip_path_uses_chunk_id(tmp_path) -> None:
    assert default_clip_path(tmp_path, "0001-a").name == f"0001-a{CLIP_AUDIO_SUFFIX}"


def test_default_video_clip_path_uses_mp4(tmp_path) -> None:
    assert default_video_clip_path(tmp_path, "0001").name == f"0001{CLIP_VIDEO_SUFFIX}"


def test_build_audio_clip_command_uses_aac_mono_16k_32k(tmp_path) -> None:
    src = tmp_path / "input.wav"
    out = tmp_path / "0001.aac"
    cmd = build_audio_clip_command("ffmpeg", src, 10.0, 20.0, out)

    assert cmd[:4] == ["ffmpeg", "-y", "-nostdin", "-ss"]
    assert "-i" in cmd and str(src) in cmd
    assert "-t" in cmd and "10.000" in cmd
    assert "-vn" in cmd
    assert cmd[-len(AUDIO_CODEC_ARGS) - 1 : -1] == AUDIO_CODEC_ARGS
    assert cmd[-1] == str(out)


def test_build_video_clip_command_includes_hwaccel_filters_and_faststart(tmp_path) -> None:
    src = tmp_path / "input.mp4"
    out = tmp_path / "0001.mp4"
    cmd = build_video_clip_command(
        "ffmpeg", src, 1500.0, 2700.0, out, hwaccel="auto"
    )

    assert "-hwaccel" in cmd and "auto" in cmd
    assert "-map" in cmd and "0:v:0" in cmd and "0:a:0?" in cmd
    joined = " ".join(cmd)
    assert "-vf" in cmd and "fps=fps=1" in joined
    assert "-af" in cmd and "aresample=16000" in joined
    assert "-movflags" in cmd and "+faststart" in cmd
    assert VIDEO_ENCODER_ARGS[0] in cmd


def test_extract_window_clip_rejects_empty_range(tmp_path) -> None:
    src = tmp_path / "src.wav"
    src.write_bytes(b"not used")

    with pytest.raises(ValueError):
        extract_window_clip(src, 1.0, 1.0, tmp_path / "out.aac")


def test_extract_audio_clip_invokes_ffmpeg(monkeypatch, tmp_path) -> None:
    src = tmp_path / "src.wav"
    src.write_bytes(b"x")
    out = tmp_path / "clips" / "0001.aac"
    captured: list[list[str]] = []

    def fake_run(args, **kwargs):
        captured.append(args)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("llm.ffmpeg_clips.subprocess.run", fake_run)
    result = extract_audio_clip(src, 0.5, 1.5, out, ffmpeg="ffmpeg")

    assert result == out
    assert captured
    assert captured[0][-len(AUDIO_CODEC_ARGS) - 1 : -1] == AUDIO_CODEC_ARGS


def test_probe_media_duration_uses_ffprobe(monkeypatch, tmp_path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"x")

    def fake_run(args, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "123.456\n", "stderr": ""})()

    monkeypatch.setattr("llm.ffmpeg_clips.subprocess.run", fake_run)
    assert probe_media_duration(src, ffprobe="ffprobe") == pytest.approx(123.456)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_extract_window_clip_writes_mono_16k_aac(tmp_path) -> None:
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    src = tmp_path / "src.wav"
    sample_rate = 8_000
    seconds = 2.0
    t = np.linspace(0.0, seconds, int(sample_rate * seconds), endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 220 * t)], axis=1)
    soundfile.write(src, stereo.astype("float32"), sample_rate)

    out = tmp_path / "clips" / "0001.aac"
    result = extract_window_clip(src, 0.5, 1.5, out)

    probe = pytest.importorskip("subprocess").run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels",
            "-of",
            "default=noprint_wrappers=1",
            str(result),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    text = probe.stdout
    assert "codec_name=aac" in text
    assert "sample_rate=16000" in text
    assert "channels=1" in text


def test_extract_video_clip_falls_back_to_cpu_decode_on_hwaccel_failure(
    monkeypatch, tmp_path
) -> None:
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    out = tmp_path / "0001.mp4"
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "-hwaccel" in args:
            return type("R", (), {"returncode": 1, "stdout": "", "stderr": "hwaccel fail"})()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("llm.ffmpeg_clips.subprocess.run", fake_run)

    from llm.ffmpeg_clips import extract_video_clip

    result = extract_video_clip(src, 0.0, 1.0, out, ffmpeg="ffmpeg")

    assert result == out
    assert len(calls) == 2
    assert "-hwaccel" in calls[0] and "auto" in calls[0]
    assert "-hwaccel" not in calls[1]
