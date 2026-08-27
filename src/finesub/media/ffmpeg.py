"""ffmpeg helpers for LLM window media clips (audio default, video opt-in)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# Gemini bills audio at 32 tok/s on 16 kHz mono regardless of container/codec.
AUDIO_CODEC_ARGS = ["-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k"]

# The local agy backend cannot request Gemini's server-side sampling, so the
# reusable window clip itself is cut at the rate both paths use.  The REST call
# asks for the same rate and therefore sees the same frames, which is what lets
# one clip serve both without an agy-only second transcode.  The value has to
# match `finesub.llm.routing.profiles.VIDEO_SAMPLE_FPS`; media may not import llm, so a test
# pins the two together rather than a shared import.
VIDEO_SAMPLE_FPS = 0.25
VIDEO_FILTER = (
    f"fps=fps={VIDEO_SAMPLE_FPS:g}:start_time=0:round=near,"
    f"scale=-2:720:flags=lanczos,setpts=N/({VIDEO_SAMPLE_FPS:g}*TB)"
)
AUDIO_FILTER = "aresample=16000:async=1:first_pts=0,asetpts=N/SR/TB"
VIDEO_ENCODER_ARGS = [
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "18",
    "-pix_fmt",
    "yuv420p",
]


def resolve_ffmpeg() -> str:
    path = shutil.which(FFMPEG_BIN)
    if not path:
        raise RuntimeError("ffmpeg not found on PATH; required for LLM media clips")
    return path


def describe_ffmpeg() -> dict[str, str]:
    """Which ffmpeg this run will use, for the run metadata.

    The launchers may hand the pipeline a managed copy or reuse whatever the
    machine already had, so the binary is no longer a constant of the install.
    Recording it means a transcoding difference between two machines can be
    traced instead of guessed at.
    """

    try:
        path = resolve_ffmpeg()
    except RuntimeError:
        return {}
    try:
        result = subprocess.run(
            [path, "-version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        banner = result.stdout.splitlines()[0].strip() if result.stdout else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        banner = ""
    return {"path": path, "version": banner}


def resolve_ffprobe() -> str:
    path = shutil.which(FFPROBE_BIN)
    if not path:
        raise RuntimeError("ffprobe not found on PATH; required for LLM media clips")
    return path


def _format_seconds(seconds: float) -> str:
    return f"{max(0.0, seconds):.3f}"


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {detail}")


def probe_media_duration(
    media_path: str | Path,
    *,
    ffprobe: str | None = None,
) -> float:
    """Return media duration in seconds via ffprobe."""
    ffprobe_bin = ffprobe or resolve_ffprobe()
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {detail}")
    text = (result.stdout or "").strip()
    if not text:
        raise RuntimeError(f"ffprobe returned no duration for {media_path}")
    return float(text)


def media_has_video_stream(
    media_path: str | Path,
    *,
    ffprobe: str | None = None,
) -> bool:
    """Whether ffprobe sees a first video stream in the container."""

    ffprobe_bin = ffprobe or resolve_ffprobe()
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {detail}")
    return any(line.strip() == "video" for line in result.stdout.splitlines())


def _clip_duration(clip_start: float, clip_end: float) -> float:
    duration = clip_end - clip_start
    if duration <= 0:
        raise ValueError(
            f"Empty clip range [{clip_start:.2f}, {clip_end:.2f}]"
        )
    return duration


def build_audio_clip_command(
    ffmpeg: str,
    input_path: Path,
    clip_start: float,
    clip_end: float,
    out_path: Path,
) -> list[str]:
    duration = _clip_duration(clip_start, clip_end)
    return [
        ffmpeg,
        "-y",
        "-nostdin",
        "-ss",
        _format_seconds(clip_start),
        "-i",
        str(input_path),
        "-t",
        _format_seconds(duration),
        "-vn",
        *AUDIO_CODEC_ARGS,
        str(out_path),
    ]


def build_video_clip_command(
    ffmpeg: str,
    input_path: Path,
    clip_start: float,
    clip_end: float,
    out_path: Path,
    *,
    hwaccel: str | None = None,
) -> list[str]:
    duration = _clip_duration(clip_start, clip_end)
    cmd = [
        ffmpeg,
        "-y",
        "-nostdin",
        "-ss",
        _format_seconds(clip_start),
    ]
    if hwaccel:
        cmd.extend(["-hwaccel", hwaccel])
    cmd.extend(
        [
            "-i",
            str(input_path),
            "-t",
            _format_seconds(duration),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            VIDEO_FILTER,
            "-af",
            AUDIO_FILTER,
            *VIDEO_ENCODER_ARGS,
            *AUDIO_CODEC_ARGS,
            "-movflags",
            "+faststart",
            str(out_path),
        ]
    )
    return cmd


def transcode_to_lossless_audio(
    input_path: str | Path,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Decode any container's audio track into FLAC, unchanged otherwise.

    Deliberately passes no ``-ac``/``-ar``: this exists to make a file readable,
    not to resample it. Callers that want a smaller or narrower artifact should
    say so themselves.
    """

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-nostdin",
            "-i",
            str(input_path),
            "-vn",
            "-c:a",
            "flac",
            str(out),
        ]
    )
    return out


def extract_audio_clip(
    input_path: str | Path,
    clip_start: float,
    clip_end: float,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Extract an audio-only raw AAC mono-16k clip for LLM upload."""
    # Before locating ffmpeg, not after: an impossible range is wrong on its own
    # terms, and answering it with "ffmpeg not found" makes the caller chase the
    # environment instead of the argument. It also let a pure validation test
    # depend on a binary being installed, which is how a CI job ended up
    # apt-installing ffmpeg for one assertion.
    _clip_duration(clip_start, clip_end)
    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        build_audio_clip_command(
            ffmpeg_bin, Path(input_path), clip_start, clip_end, out
        )
    )
    return out


def extract_video_clip(
    input_path: str | Path,
    clip_start: float,
    clip_end: float,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Extract a 0.25 fps / 720p H.264 + AAC multimodal clip.

    Tries ``-hwaccel auto`` for decode first; on failure retries with CPU decode.
    Video encoding stays on libx264 (lightweight relative to decode/filter).
    """
    _clip_duration(clip_start, clip_end)  # see extract_audio_clip
    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    input_file = Path(input_path)

    try:
        _run_ffmpeg(
            build_video_clip_command(
                ffmpeg_bin,
                input_file,
                clip_start,
                clip_end,
                out,
                hwaccel="auto",
            )
        )
        return out
    except RuntimeError:
        pass

    _run_ffmpeg(
        build_video_clip_command(
            ffmpeg_bin,
            input_file,
            clip_start,
            clip_end,
            out,
            hwaccel=None,
        )
    )
    return out


def containerize_audio_for_agy(
    input_path: str | Path,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Put an audio clip in an MP4 with exactly one tiny black video frame.

    agy's ``view_file`` rejects bare audio MIME types before the payload reaches
    Gemini.  A single-frame video track preserves the complete audio duration
    without paying a duration-proportional visual-token cost.
    """

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    source = Path(input_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = probe_media_duration(source)
    if duration <= 0:
        raise RuntimeError(f"audio has no positive duration: {source}")
    # Give the single frame a duration equal to the audio. A normal 1 fps
    # color source plus ``-frames:v 1`` truncates the MP4 to one second even
    # though the audio input is longer.
    frame_rate = 1.0 / duration
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-nostdin",
            "-i",
            str(source),
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s=16x16:r={frame_rate:.12f}",
            "-map",
            "1:v:0",
            "-map",
            "0:a:0",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-tune",
            "stillimage",
            "-pix_fmt",
            "yuv420p",
            *AUDIO_CODEC_ARGS,
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    return out


def transcode_video_for_agy(
    input_path: str | Path,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Normalize an arbitrary video attachment to agy's 0.25 fps envelope."""

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    source = Path(input_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(
        [
            ffmpeg_bin,
            "-y",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            VIDEO_FILTER,
            "-af",
            AUDIO_FILTER,
            *VIDEO_ENCODER_ARGS,
            *AUDIO_CODEC_ARGS,
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    return out
