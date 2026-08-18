"""Per-window audio clip ranges and extraction for LLM correction calls."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, Tuple

from .ffmpeg import extract_audio_clip, extract_video_clip, probe_media_duration


CLIP_PAD_SECONDS = 5.0
CLIP_EDGE_PAD_SECONDS = 60.0
CLIP_SAMPLE_RATE = 16_000
CLIP_AUDIO_SUFFIX = ".aac"
CLIP_VIDEO_SUFFIX = ".mp4"


class _TimedSegment(Protocol):
    id: str
    start: float
    end: float


def compute_clip_range(
    window_segments: Sequence[_TimedSegment],
    *,
    global_first_id: str,
    global_last_id: str,
    audio_duration: float | None = None,
) -> Tuple[float, float]:
    """Clip range for a window: segment span plus padding.

    Regular windows pad 5s on both sides; the window containing the globally
    first/last segment pads 60s at that edge instead (keyed by segment id so
    -a/-b split halves inherit the rule). The start clamps to 0; the end clamps
    to ``audio_duration`` when known, but never below the last segment's end.
    """
    if not window_segments:
        raise ValueError("window_segments must not be empty")
    first = window_segments[0]
    last = window_segments[-1]
    lead = CLIP_EDGE_PAD_SECONDS if first.id == global_first_id else CLIP_PAD_SECONDS
    trail = CLIP_EDGE_PAD_SECONDS if last.id == global_last_id else CLIP_PAD_SECONDS
    clip_start = max(0.0, first.start - lead)
    clip_end = last.end + trail
    if audio_duration is not None:
        clip_end = max(min(clip_end, audio_duration), last.end)
    return clip_start, clip_end


def default_clip_dir(task_artifact_dir: str | Path) -> Path:
    """Where one task's window clips live: inside its own artifact directory.

    They are private to the task and disposable, so they belong where the rest
    of its scratch state already is -- removed with it, and never relative to
    whatever directory the process happens to be started from.
    """

    return Path(task_artifact_dir).expanduser() / "clips"


def default_clip_path(clip_dir: str | Path, chunk_id: str) -> Path:
    return Path(clip_dir) / f"{chunk_id}{CLIP_AUDIO_SUFFIX}"


def default_video_clip_path(clip_dir: str | Path, chunk_id: str) -> Path:
    """Output path for opt-in visual multimodal clips (not used by default)."""
    return Path(clip_dir) / f"{chunk_id}{CLIP_VIDEO_SUFFIX}"


def probe_audio_duration(audio_path: str | Path) -> float:
    return probe_media_duration(audio_path)


def extract_window_clip(
    audio_path: str | Path,
    clip_start: float,
    clip_end: float,
    out_path: str | Path,
    *,
    target_sample_rate: int = CLIP_SAMPLE_RATE,
) -> Path:
    """Extract [clip_start, clip_end] as raw mono AAC at 16 kHz / 32 kbps.

    Gemini processes audio as 16 kHz mono at a fixed 32 tok/s, so downmixing
    and re-encoding shrinks the upload without changing tokens or quality. The
    output is always overwritten to avoid reusing a stale clip.

    ``target_sample_rate`` is kept for API compatibility; ffmpeg always emits
    16 kHz mono AAC.
    """
    del target_sample_rate  # fixed by AUDIO_CODEC_ARGS in ffmpeg_clips
    return extract_audio_clip(audio_path, clip_start, clip_end, out_path)


def extract_window_video_clip(
    media_path: str | Path,
    clip_start: float,
    clip_end: float,
    out_path: str | Path,
) -> Path:
    """Extract a visual+audio clip for future multimodal tasks (opt-in only)."""
    return extract_video_clip(media_path, clip_start, clip_end, out_path)
