"""ffmpeg helpers: LLM window media clips (audio default, video opt-in) and the cloud ASR upload."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

from finesub_bootstrap.processes import no_window

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
            creationflags=no_window(),
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


def run_capture(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a media helper and capture its output as text.

    `text=True` on its own decodes with the machine's ANSI code page, which on
    a zh-CN Windows install is cp936. ffmpeg and ffprobe emit UTF-8 metadata --
    a track title, a comment tag -- and the decode then raises
    `UnicodeDecodeError` inside `subprocess`'s reader thread, before the caller
    ever sees an exit status. The failure belongs to the input file's tags, not
    to anything the run did, and every capture in this package goes through
    here so the decoding cannot drift apart one call site at a time.

    `errors="replace"` rather than `strict`: this output is a diagnostic or a
    number to parse, and a byte that is not valid UTF-8 must not outrank
    whatever ffprobe was asked.
    """

    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=no_window(),
    )


def _run_ffmpeg(args: list[str]) -> None:
    result = run_capture(args)
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
    result = run_capture(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
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

    return _has_stream(media_path, "video", ffprobe=ffprobe)


def media_has_audio_stream(
    media_path: str | Path,
    *,
    ffprobe: str | None = None,
) -> bool:
    """Whether ffprobe sees a first audio stream in the container."""

    return _has_stream(media_path, "audio", ffprobe=ffprobe)


def _has_stream(media_path: str | Path, kind: str, *, ffprobe: str | None) -> bool:
    ffprobe_bin = ffprobe or resolve_ffprobe()
    result = run_capture(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            f"{kind[0]}:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"ffprobe failed (exit {result.returncode}): {detail}")
    return any(line.strip() == kind for line in result.stdout.splitlines())


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


def extract_upload_mp3(
    input_path: str | Path,
    clip_start: float,
    clip_end: float,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """A span of the source as the cloud ASR backends upload it: 16 kHz mono 64 kbps MP3."""
    duration = _clip_duration(clip_start, clip_end)
    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + ".part" + out.suffix)
    _run_ffmpeg([
        ffmpeg_bin, "-y", "-nostdin", "-loglevel", "error",
        "-ss", _format_seconds(clip_start), "-t", _format_seconds(duration),
        "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(temporary),
    ])
    temporary.replace(out)
    return out


def decode_pcm16k(input_path: str | Path, *, ffmpeg: str | None = None, sample_format: str = "s16le") -> bytes:
    """The whole source as 16 kHz mono PCM: signed 16-bit little-endian, or `f32le` (float32)."""
    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    result = subprocess.run(
        [ffmpeg_bin, "-nostdin", "-loglevel", "error", "-i", str(input_path),
         "-vn", "-ac", "1", "-ar", "16000", "-f", sample_format, "-"],
        capture_output=True,
        creationflags=no_window(),
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed (exit {result.returncode}): {detail}")
    return result.stdout


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
    without paying a duration-proportional visual-token cost, but past five
    minutes the model hears only up to the last multiple of five minutes below
    the clip's length: the driver sends long audio through
    ``split_audio_for_agy`` instead.
    """

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    source = Path(input_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # The decoded length, not ffprobe's: on raw ADTS (the window clips) ffprobe
    # guesses from the opening frames' bitrate, and a frame that long either
    # cuts the audio's end off or outlasts it.
    duration = len(decode_pcm16k(source, ffmpeg=ffmpeg_bin)) / 2 / _PCM_RATE
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


#: agy hears a video attachment whole only up to five minutes, so the driver
#: sends audio and video as pieces at most this long. If agy ever takes a long
#: attachment whole, both go back to one attachment together (owner 2026-11-02).
AGY_AUDIO_PIECE_SECONDS = 295.0
#: how far a cut may move from its equal-length target to land between rows
AGY_AUDIO_CUT_SEARCH_SECONDS = 30.0
_PCM_RATE = 16000
_FRAMES_PER_SECOND = 100  # 10 ms energy frames
_QUIET_FRAMES = 20  # the 0.2 s a cut's loudness is averaged over


def _decode_pcm(ffmpeg_bin: str, source: Path):
    """Mono 16 kHz samples of `source` as floats: its true length, which
    ffprobe misreads on raw ADTS."""
    import numpy as np

    return np.frombuffer(decode_pcm16k(source, ffmpeg=ffmpeg_bin), dtype=np.int16).astype(np.float32)


def agy_audio_cuts(
    samples,
    *,
    speech: Sequence[tuple[float, float]] = (),
    piece: float = AGY_AUDIO_PIECE_SECONDS,
    search: float = AGY_AUDIO_CUT_SEARCH_SECONDS,
) -> list[float]:
    """Where to cut 16 kHz `samples` into pieces at most `piece` seconds long.

    One cut per equal-length target, moved within `search` of it to the
    quietest 0.2 s that lies outside every `speech` span (start, end seconds:
    the rows the prompt shows, so a cut falls between two rows); where every
    frame there is inside a row, to the quietest 0.2 s at all.
    """
    import math

    import numpy as np

    total = len(samples) / _PCM_RATE
    count = math.ceil(total / piece)
    if count <= 1:
        return []
    hop = _PCM_RATE // _FRAMES_PER_SECOND
    energy = np.square(samples[: len(samples) // hop * hop]).reshape(-1, hop).mean(axis=1)
    loudness = np.convolve(energy, np.ones(_QUIET_FRAMES) / _QUIET_FRAMES, mode="same")
    centres = (np.arange(len(loudness)) + 0.5) / _FRAMES_PER_SECOND
    between = np.ones(len(loudness), dtype=bool)
    for start, end in speech:
        between &= ~((centres > start) & (centres < end))
    cuts, previous = [], 0.0
    for k in range(1, count):
        target = total * k / count
        low = max(target - search, total - (count - k) * piece, previous + 1.0)
        high = min(target + search, previous + piece)
        span = slice(int(low * _FRAMES_PER_SECOND), int(high * _FRAMES_PER_SECOND) + 1)
        frames, free = loudness[span], between[span]
        if not len(frames):
            cut = min(target, high)
        else:
            pick = np.where(free, frames, np.inf) if free.any() else frames
            cut = low + float(np.argmin(pick)) / _FRAMES_PER_SECOND
        cuts.append(cut)
        previous = cut
    return cuts


def agy_media_cuts(
    input_path: str | Path,
    *,
    speech: Sequence[tuple[float, float]] = (),
    ffmpeg: str | None = None,
) -> list[float]:
    """Where agy's driver cuts this media (`agy_audio_cuts` on its decoded
    audio); none when it fits one attachment or has no audio (a silent video
    goes whole, and past five minutes agy may still take only its start)."""

    if not media_has_audio_stream(input_path):
        return []
    return agy_audio_cuts(_decode_pcm(ffmpeg or resolve_ffmpeg(), Path(input_path)), speech=speech,
                          piece=AGY_AUDIO_PIECE_SECONDS, search=AGY_AUDIO_CUT_SEARCH_SECONDS)


def split_video_for_agy(
    input_path: str | Path,
    out_dir: str | Path,
    stem: str,
    cuts: Sequence[float],
    *,
    ffmpeg: str | None = None,
) -> list[tuple[Path, float, float]]:
    """The video as pieces between `cuts` (from `agy_media_cuts`), each
    transcoded to agy's 0.25 fps envelope as `transcode_video_for_agy` does.
    Returns (piece, start, end) in seconds of the whole, measured by decoding
    each piece's audio. The last piece runs to the end of the video: a picture
    that outlasts the sound is not capped (production's window clips are cut
    to one length for both)."""

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    source = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pieces, start = [], 0.0
    for k, (low, high) in enumerate(zip([0.0, *cuts], [*cuts, None]), 1):
        target = out / f"{stem}-p{k}.mp4"
        span = ["-ss", f"{low:.3f}", *(["-t", f"{high - low:.3f}"] if high is not None else [])]
        _run_ffmpeg([ffmpeg_bin, "-y", "-nostdin", *span, "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                     "-vf", VIDEO_FILTER, "-af", AUDIO_FILTER, *VIDEO_ENCODER_ARGS, *AUDIO_CODEC_ARGS,
                     "-movflags", "+faststart", str(target)])
        length = len(_decode_pcm(ffmpeg_bin, target)) / _PCM_RATE
        pieces.append((target, start, start + length))
        start += length
    return pieces


def split_audio_for_agy(
    input_path: str | Path,
    out_dir: str | Path,
    stem: str,
    *,
    speech: Sequence[tuple[float, float]] = (),
    ffmpeg: str | None = None,
) -> list[tuple[Path, float, float]]:
    """The audio as MP4 pieces at most AGY_AUDIO_PIECE_SECONDS long, each with a
    video track of one black frame lasting one second (billed as audio; the
    video track is what keeps the model's clock right). Cuts as
    `agy_audio_cuts`, on AAC frame boundaries (the stream is copied, not
    re-encoded). Returns (piece, start, end) in seconds of the whole audio,
    measured by decoding each piece."""

    ffmpeg_bin = ffmpeg or resolve_ffmpeg()
    source = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cuts = agy_media_cuts(source, speech=speech, ffmpeg=ffmpeg_bin)
    adts = out / f"{stem}-whole.aac"
    if source.suffix.lower() == ".aac":  # window clips are ADTS already
        shutil.copyfile(source, adts)
    else:
        _run_ffmpeg([ffmpeg_bin, "-y", "-nostdin", "-i", str(source), "-vn", *AUDIO_CODEC_ARGS, "-f", "adts",
                     str(adts)])
    if cuts:
        _run_ffmpeg([ffmpeg_bin, "-y", "-nostdin", "-i", str(adts), "-c", "copy", "-f", "segment",
                     "-segment_times", ",".join(f"{c:.3f}" for c in cuts),
                     "-reset_timestamps", "1", str(out / f"{stem}-p%02d.aac")])
        raws = [out / f"{stem}-p{k:02d}.aac" for k in range(len(cuts) + 1)]
        adts.unlink()
    else:
        raws = [adts]
    pieces, start = [], 0.0
    for k, raw in enumerate(raws, 1):
        length = len(_decode_pcm(ffmpeg_bin, raw)) / _PCM_RATE
        target = out / f"{stem}-p{k}.mp4"
        _run_ffmpeg([ffmpeg_bin, "-y", "-nostdin", "-f", "lavfi", "-i", "color=c=black:s=16x16:r=1:d=1",
                     "-i", str(raw), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                     "-c:a", "copy", "-movflags", "+faststart", str(target)])
        raw.unlink()
        pieces.append((target, start, start + length))
        start += length
    return pieces


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


def build_spliced_audio_command(
    ffmpeg: str,
    input_path: Path,
    pieces: Sequence[tuple[float, float]],
    gap: float,
    out_path: Path,
) -> list[str]:
    """One audio file of `pieces` of one source, in order, `gap` seconds of silence between them."""
    if not pieces:
        raise ValueError("pieces must not be empty")
    sources = "".join(f"[s{index}]" for index in range(len(pieces)))
    parts, labels = [f"[0:a]asplit={len(pieces)}{sources}" if len(pieces) > 1 else "[0:a]anull[s0]"], []
    for index, (start, end) in enumerate(pieces):
        _clip_duration(start, end)
        parts.append(
            f"[s{index}]atrim=start={_format_seconds(start)}:end={_format_seconds(end)},asetpts=PTS-STARTPTS,"
            f"aformat=sample_rates=16000:channel_layouts=mono[p{index}]"
        )
        labels.append(f"[p{index}]")
        if index < len(pieces) - 1:
            parts.append(f"anullsrc=r=16000:cl=mono,atrim=duration={_format_seconds(gap)}[g{index}]")
            labels.append(f"[g{index}]")
    graph = ";".join(parts) + f";{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"
    return [ffmpeg, "-y", "-nostdin", "-i", str(input_path), "-filter_complex", graph,
            "-map", "[out]", "-vn", *AUDIO_CODEC_ARGS, str(out_path)]


def extract_spliced_audio(
    input_path: str | Path,
    pieces: Sequence[tuple[float, float]],
    gap: float,
    out_path: str | Path,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """`build_spliced_audio_command`, run: raw AAC mono 16k, like the window clips."""
    command = build_spliced_audio_command(ffmpeg or resolve_ffmpeg(), Path(input_path), pieces, gap, Path(out_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg(command)
    return Path(out_path)
