from __future__ import annotations

import shutil
from dataclasses import dataclass

import pytest

from finesub.media.clips import (
    CLIP_EDGE_PAD_SECONDS,
    CLIP_PAD_SECONDS,
    CLIP_AUDIO_SUFFIX,
    CLIP_VIDEO_SUFFIX,
    compute_clip_range,
    default_clip_path,
    default_video_clip_path,
    extract_window_clip,
)
from finesub.media.ffmpeg import (
    AUDIO_CODEC_ARGS,
    VIDEO_ENCODER_ARGS,
    build_audio_clip_command,
    build_video_clip_command,
    containerize_audio_for_agy,
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
    assert "-vf" in cmd and "fps=fps=0.25" in joined
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

    monkeypatch.setattr("finesub.media.ffmpeg.subprocess.run", fake_run)
    result = extract_audio_clip(src, 0.5, 1.5, out, ffmpeg="ffmpeg")

    assert result == out
    assert captured
    assert captured[0][-len(AUDIO_CODEC_ARGS) - 1 : -1] == AUDIO_CODEC_ARGS


def test_probe_media_duration_uses_ffprobe(monkeypatch, tmp_path) -> None:
    src = tmp_path / "audio.wav"
    src.write_bytes(b"x")

    def fake_run(args, **kwargs):
        return type("R", (), {"returncode": 0, "stdout": "123.456\n", "stderr": ""})()

    monkeypatch.setattr("finesub.media.ffmpeg.subprocess.run", fake_run)
    assert probe_media_duration(src, ffprobe="ffprobe") == pytest.approx(123.456)


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)
def test_agy_audio_container_keeps_full_audio_with_one_video_frame(tmp_path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    assert ffmpeg and ffprobe
    src = tmp_path / "two-seconds.wav"
    subprocess = pytest.importorskip("subprocess")
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            str(src),
        ],
        check=True,
    )
    out = containerize_audio_for_agy(src, tmp_path / "agy.mp4", ffmpeg=ffmpeg)
    duration = float(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    frames = int(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nw=1:nk=1",
                str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert duration == pytest.approx(2.0, abs=0.1)
    assert frames == 1


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

    monkeypatch.setattr("finesub.media.ffmpeg.subprocess.run", fake_run)

    from finesub.media.ffmpeg import extract_video_clip

    result = extract_video_clip(src, 0.0, 1.0, out, ffmpeg="ffmpeg")

    assert result == out
    assert len(calls) == 2
    assert "-hwaccel" in calls[0] and "auto" in calls[0]
    assert "-hwaccel" not in calls[1]


def test_a_local_clip_is_uploaded_once_per_client_not_once_per_call(
    tmp_path, monkeypatch
) -> None:
    """Under an agent policy the clip is local until an API candidate answers.

    That upload used to be cached per `complete()` call, so one window paid for
    it again on its query round, its correction round and every retry -- while
    the api-only path uploaded once per window.
    """

    import finesub.llm.client as client_module

    clip = tmp_path / "window.aac"
    clip.write_bytes(b"clip")
    uploads: list[str] = []

    def fake_upload(path, *, api_key=None):
        uploads.append(str(path))
        return client_module.UploadedFileRef(
            file_id=f"files/{len(uploads)}",
            filename="window.aac",
            mime_type="audio/aac",
            local_path=str(path),
        )

    monkeypatch.setattr(client_module, "upload_gemini_file", fake_upload)
    monkeypatch.setattr(client_module, "_first_gemini_api_key", lambda tier=None: "k")

    local_ref = client_module.local_media_file_ref(clip)
    assert local_ref.file_id == ""

    instance = client_module.RoleClient.__new__(client_module.RoleClient)
    instance._remote_media_refs = {}

    first = instance._uploaded_media_ref(local_ref, provider_tier="GEMINI_FREE")
    second = instance._uploaded_media_ref(local_ref, provider_tier="GEMINI_FREE")

    assert first == second
    assert len(uploads) == 1

    # A different key pool is a different Files namespace, so it uploads again.
    instance._uploaded_media_ref(local_ref, provider_tier="GEMINI_PAID")
    assert len(uploads) == 2

    # A rejected reuse (the object expires after 48h) must not be cached for
    # the rest of the run.
    instance._forget_uploaded_media_ref(local_ref, provider_tier="GEMINI_FREE")
    instance._uploaded_media_ref(local_ref, provider_tier="GEMINI_FREE")
    assert len(uploads) == 3


def test_window_media_ref_is_the_only_place_the_policy_is_read(
    tmp_path, monkeypatch
) -> None:
    """Correction and fast mode had this decision copied out verbatim.

    The failure mode of the copy drifting is a silent Gemini upload under an
    agent policy, which is exactly what the local reference exists to avoid.
    """

    import finesub.llm.client as client_module
    from finesub.llm.stages import fast_session
    from finesub.llm.stages.correction import run as correction_run

    assert correction_run.window_media_ref is client_module.window_media_ref
    assert fast_session.window_media_ref is client_module.window_media_ref

    clip = tmp_path / "window.mp4"
    clip.write_bytes(b"clip")
    monkeypatch.setattr(
        client_module,
        "upload_gemini_file",
        lambda path, **_: client_module.UploadedFileRef(
            file_id="files/1",
            filename="window.mp4",
            mime_type="video/mp4",
            local_path=str(path),
        ),
    )

    class _Settings:
        def __init__(self, policy_id):
            self.policy_id = policy_id

    # An agent-capable policy is necessary but not sufficient: the bound groups
    # have to name an agent. The packaged default preset names none, so every
    # policy uploads eagerly there -- deferring under a plan that will
    # certainly answer from the API only moves the upload later.
    for policy in ("agent-only", "agent-text-preferred", "api-only"):
        eager = client_module.window_media_ref(
            clip, execution_settings=_Settings(policy)
        )
        assert eager.file_id == "files/1", policy

    from finesub.llm.routing import model_routes

    agy = model_routes.load_model_routes(user_config={"preset": "agy"})
    assert agy.binds_local_agent() is True

    # Answered from the *caller's* catalog, which is the one its router plans
    # from. Reading the global one instead would let the table that decides how
    # the clip is carried disagree with the table that decides who answers.
    for policy in ("agent-only", "agent-text-preferred"):
        ref = client_module.window_media_ref(
            clip, execution_settings=_Settings(policy), routes=agy
        )
        assert ref.file_id == "", policy
        assert ref.agy_prepared is True, policy

    uploaded = client_module.window_media_ref(
        clip, execution_settings=_Settings("api-only"), routes=agy
    )
    assert uploaded.file_id == "files/1"

    # And the global catalog does not get a vote when one was passed: an agy
    # router still defers even while the process default binds no agent.
    monkeypatch.setattr(
        model_routes, "default_model_routes", lambda: model_routes.load_model_routes()
    )
    assert (
        client_module.window_media_ref(
            clip, execution_settings=_Settings("agent-only"), routes=agy
        ).file_id
        == ""
    )


