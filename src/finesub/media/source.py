"""URL/media source helpers shared by reference ingest and the main pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import re
import threading

from ..paths import resolve_reference_data_root
from ..reporting import current_reporter

URL_MAP_FILENAME = "url-map.json"
#: Scraped metadata about a URL, kept beside the id map rather than inside it.
#: A separate file because the two have different lifetimes: the id map is
#: load-bearing (it is what keeps reruns offline and artifact paths stable),
#: this is a convenience cache that can be deleted at any time.
URL_INFO_FILENAME = "url-info.json"

#: A scraped title goes into an LLM prompt, so it is untrusted text from the
#: open web. It gets flattened to one line (a newline could otherwise forge the
#: `媒体文件:` / `视频来源 URL:` fields around it) and capped.
TITLE_MAX_CHARS = 200
YTDLP_RETRY_OPTIONS = {
    "retries": 10,
    "fragment_retries": 10,
    "extractor_retries": 5,
    "socket_timeout": 30,
    "continuedl": True,
}
# Whole-download attempts on top of yt-dlp's internal fragment retries: 1
# initial try + 4 automatic retries (v15), exponential backoff between them.
DOWNLOAD_MAX_ATTEMPTS = 5
DOWNLOAD_BACKOFF_SECONDS = 5.0


def _retry_pause(what: str, attempt: int, error: Exception | str) -> None:
    import time

    delay = DOWNLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1))
    current_reporter().warning(
        "download-retry",
        f"{what} download attempt {attempt}/{DOWNLOAD_MAX_ATTEMPTS} "
        f"failed ({error}); retrying in {delay:.0f}s",
    )
    time.sleep(delay)

_UNSAFE_ID_CHARS_RE = re.compile(r'[\\/:*?"<>|\s\x00-\x1f]+')
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def sanitize_video_id(video_id: str) -> str:
    return _UNSAFE_ID_CHARS_RE.sub("_", video_id.strip()) or "video"


def is_url(value: str) -> bool:
    return bool(_URL_RE.match((value or "").strip()))


# Guards the url-map read-modify-write against concurrent download workers
# (in-process only; the batch runner is single-process by design).
_URL_MAP_LOCK = threading.Lock()


def url_map_path(data_dir: Path) -> Path:
    return data_dir / URL_MAP_FILENAME


def load_url_map(data_dir: Path) -> dict[str, str]:
    path = url_map_path(data_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_url_map(data_dir: Path, mapping: dict[str, str]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    url_map_path(data_dir).write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def record_video_id(url: str, video_id: str, data_dir: Path) -> None:
    """Merge one url->id entry into the map without losing concurrent writes."""

    with _URL_MAP_LOCK:
        mapping = load_url_map(data_dir)
        mapping[url] = video_id
        save_url_map(data_dir, mapping)


def url_info_path(data_dir: Path) -> Path:
    return data_dir / URL_INFO_FILENAME


def load_url_info(data_dir: Path) -> dict[str, dict]:
    path = url_info_path(data_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def clean_scraped_title(raw: object) -> str:
    """One line, bounded, or empty. Never raises.

    Everything about this is because the value ends up in an LLM prompt and
    came from someone else's web page: whitespace (newlines included) collapses
    so it cannot fabricate the surrounding `key: value` lines, and the length is
    capped so a pathological title cannot crowd out the rest of the context.
    """

    text = " ".join(str(raw or "").split())
    if len(text) > TITLE_MAX_CHARS:
        text = text[: TITLE_MAX_CHARS - 1] + "…"
    return text


def record_video_info(url: str, title: str, data_dir: Path) -> None:
    """Merge one url->metadata entry without losing concurrent writes.

    An empty title is recorded rather than skipped: "we asked and there was
    nothing" has to be distinguishable from "we never asked", or every rerun
    re-probes a URL that has no title or that rate-limited us.
    """

    title = clean_scraped_title(title)
    with _URL_MAP_LOCK:
        info = load_url_info(data_dir)
        info[url] = {**info.get(url, {}), "title": title}
        data_dir.mkdir(parents=True, exist_ok=True)
        url_info_path(data_dir).write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )


class _SilentLogger:
    """Swallow yt-dlp's own output for probes whose failure is not news."""

    def debug(self, message: str) -> None:
        return

    warning = error = debug


def resolve_video_title(url: str, data_dir: Path) -> str:
    """Best-effort scraped title for a URL; ``""`` when it cannot be had.

    Cache first, then one metadata-only `extract_info`. Best-effort in the
    strong sense -- **every** failure returns an empty string, because this is
    a nicety attached to the extra-info block and nothing downstream is allowed
    to depend on it. In particular it must not turn an offline rerun, a dead
    URL, or a missing yt-dlp into a failed transcription.

    Separate from `resolve_video_id` on purpose: that function's contract is
    that a rerun stays offline, and folding a title probe into it would quietly
    break exactly that for every URL resolved before titles existed.
    """

    # One `try` around everything, including the cache read: an unreadable
    # cache file is as much "no title available" as a dead URL is, and the
    # caller has no branch for either.
    try:
        cached = load_url_info(data_dir)
        if url in cached:
            # Present-but-empty is a RESULT, not a miss. Without this a URL
            # that answered 412 (bilibili does, under any rate limiting) or
            # simply carries no title gets probed again on every single rerun,
            # which is the offline-rerun contract broken by the back door.
            # Refreshing is deliberately an explicit act: delete url-info.json.
            return clean_scraped_title((cached.get(url) or {}).get("title"))

        import yt_dlp

        # A silent logger, not just `quiet`: yt-dlp writes extractor failures
        # straight to stderr regardless, and a failure here is a non-event --
        # the caller simply gets no title. Bilibili answers 412 under any kind
        # of rate limiting, which would otherwise print a red ERROR line in the
        # middle of a run that is going perfectly well.
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "logger": _SilentLogger(),
        }
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
        title = clean_scraped_title((info or {}).get("title"))
        record_video_info(url, title, data_dir)
        return title
    except Exception as exc:  # network, extractor, missing dependency, disk
        current_reporter().debug("no scraped title", {"url": url, "error": str(exc)})
        # Record the failure too, for the same reason: one probe per URL, ever.
        try:
            record_video_info(url, "", data_dir)
        except Exception:  # a cache we cannot write is not worth a failed run
            pass
        return ""


def resolve_video_id(url: str, data_dir: Path) -> str:
    """URL -> stable video id, cached in data_dir so reruns stay offline."""

    with _URL_MAP_LOCK:
        mapping = load_url_map(data_dir)
        if url in mapping:
            return mapping[url]
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("URL 输入需要 yt-dlp：pip install yt-dlp") from exc

    with yt_dlp.YoutubeDL({"quiet": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    video_id = sanitize_video_id(str(info.get("id") or ""))
    record_video_id(url, video_id, data_dir)
    # Free: this call already fetched the metadata the title lives in, so
    # taking it here is what keeps `resolve_video_title` off the network for
    # every URL first seen after this landed.
    record_video_info(url, info.get("title"), data_dir)
    return video_id


def _media_target_dir(data_dir: Path, video_id: str, target_dir: str | Path | None) -> Path:
    return Path(target_dir) if target_dir is not None else Path(data_dir) / video_id


def _stem_video_path(target_dir: Path, stem: str) -> Path:
    return target_dir / f"{stem}.mp4"


def _audio_source_glob(stem: str) -> str:
    return f"{stem}-audio-source.*"


def _is_audio_source(path: Path, stem: str) -> bool:
    return path.name.startswith(f"{stem}-audio-source.")


def _is_ytdlp_format_part(path: Path, stem: str) -> bool:
    return bool(re.match(rf"^{re.escape(stem)}\.f\d+\.", path.name))


def download_audio(
    url: str,
    data_dir: Path | None = None,
    *,
    video_id: str | None = None,
    target_dir: str | Path | None = None,
) -> tuple[str, Path]:
    """Download the best audio track as target_dir/<stem><ext> (skip if present).

    The container is whatever yt-dlp chose; nothing is re-encoded here, so the
    extension varies. Stages that need a narrower or seekable form derive it.
    """

    data_dir = Path(data_dir) if data_dir is not None else resolve_reference_data_root()
    video_id = video_id or resolve_video_id(url, data_dir)
    target = _media_target_dir(data_dir, video_id, target_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing = _select_audio_files(target, video_id)
    if existing:
        audio = existing[0]
        current_reporter().debug(
            "skipping download", {"existing": str(audio)}
        )
        return video_id, audio
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("URL 输入需要 yt-dlp：pip install yt-dlp") from exc

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(target / f"{video_id}-audio-source.%(ext)s"),
        "noplaylist": True,
        **YTDLP_RETRY_OPTIONS,
    }
    last_error: Exception | str = ""
    sources: list[Path] = []
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
        except Exception as exc:  # yt-dlp DownloadError and friends
            last_error = exc
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                _retry_pause("audio", attempt, exc)
            continue
        sources = _complete_files(target.glob(_audio_source_glob(video_id)))
        if sources:
            break
        last_error = f"yt-dlp finished but no audio file found under {target}"
        if attempt < DOWNLOAD_MAX_ATTEMPTS:
            _retry_pause("audio", attempt, last_error)
    if not sources:
        raise RuntimeError(
            f"Audio download for {url} failed after {DOWNLOAD_MAX_ATTEMPTS} "
            f"attempts: {last_error}"
        )
    # Keep what yt-dlp gave us. Re-encoding here used to cost a lossy generation
    # and a downmix before separation ever saw the audio; the stages that need a
    # narrower form now derive it themselves.
    audio = sources[0].replace(target / f"{video_id}{sources[0].suffix}")
    _remove_sources(sources, keep=audio)
    return video_id, audio


def download_video(
    url: str,
    data_dir: Path | None = None,
    *,
    video_id: str | None = None,
    target_dir: str | Path | None = None,
) -> tuple[str, Path]:
    """Download a capped-resolution video as target_dir/<stem>.mp4."""

    data_dir = Path(data_dir) if data_dir is not None else resolve_reference_data_root()
    video_id = video_id or resolve_video_id(url, data_dir)
    target = _media_target_dir(data_dir, video_id, target_dir)
    target.mkdir(parents=True, exist_ok=True)
    existing = _select_video_files(target, video_id)
    if existing:
        try:
            validate_video_audio_coverage(existing[0])
        except RuntimeError as exc:
            # A corrupt file must not be reused forever just because it exists.
            current_reporter().warning(
                "video-revalidated",
                f"existing video failed validation, re-downloading: {exc}",
            )
            existing[0].unlink(missing_ok=True)
        else:
            current_reporter().debug(
                "skipping video download", {"existing": str(existing[0])}
            )
            return video_id, existing[0]
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("URL 输入需要 yt-dlp：pip install yt-dlp") from exc

    options = {
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "format_sort": ["res:720", "+fps"],
        "outtmpl": str(target / f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "merge_output_format": "mp4",
        **YTDLP_RETRY_OPTIONS,
    }
    # Resumed stream downloads corrupt regularly (Bilibili range requests):
    # a bad merge shows up as diverging track durations. Download errors and
    # validation failures share the same whole-download attempt budget (v15:
    # 5 attempts with backoff).
    last_error: Exception | str = ""
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
        except Exception as exc:  # yt-dlp DownloadError and friends
            last_error = exc
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                _retry_pause("video", attempt, exc)
            continue
        existing = _select_video_files(target, video_id)
        if not existing:
            last_error = f"yt-dlp finished but no video file found under {target}"
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                _retry_pause("video", attempt, last_error)
            continue
        try:
            validate_video_audio_coverage(existing[0])
        except RuntimeError as exc:
            last_error = exc
            existing[0].unlink(missing_ok=True)
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                _retry_pause("video", attempt, exc)
            continue
        return video_id, existing[0]
    raise RuntimeError(
        f"Video download for {url} failed after {DOWNLOAD_MAX_ATTEMPTS} "
        f"attempts: {last_error}"
    )


# A resumed/corrupt stream download can merge into an mp4 whose audio track
# only covers a prefix of the video (seen: 50s of audio in a 2014s video);
# ffmpeg/yt-dlp exit 0 throughout, so without this check every later stage
# silently processes just that prefix — and existence-skip makes it permanent.
AUDIO_COVERAGE_TOLERANCE_SECONDS = 10.0


def _probe_stream_durations(path: Path) -> dict[str, float]:
    """Max duration per codec_type ({'video': s, 'audio': s}); {} when unprobeable."""

    try:
        from .ffmpeg import resolve_ffprobe, run_capture

        cmd = [
            resolve_ffprobe(),
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration",
            "-of",
            "json",
            str(path),
        ]
        result = run_capture(cmd)
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except Exception:
        return {}
    durations: dict[str, float] = {}
    for stream in streams:
        try:
            duration = float(stream.get("duration"))
        except (TypeError, ValueError):
            continue
        codec_type = str(stream.get("codec_type") or "")
        durations[codec_type] = max(durations.get(codec_type, 0.0), duration)
    return durations


def validate_video_audio_coverage(video_path: str | Path) -> None:
    """Raise when the container's video/audio track durations diverge.

    Either direction means a corrupt resumed download (seen both ways: 50s of
    audio in a 2014s video, and an 870s video track under 2014s of audio).
    """

    durations = _probe_stream_durations(Path(video_path))
    video_duration = durations.get("video")
    audio_duration = durations.get("audio")
    if not video_duration or not audio_duration:
        return
    longer = max(video_duration, audio_duration)
    tolerance = max(AUDIO_COVERAGE_TOLERANCE_SECONDS, 0.02 * longer)
    if abs(video_duration - audio_duration) > tolerance:
        raise RuntimeError(
            f"{video_path}: video stream covers {video_duration:.0f}s but audio "
            f"covers {audio_duration:.0f}s — the download is likely corrupt. "
            "Delete the file (and any derived <stem>.ogg / downstream artifacts) "
            "and re-run to download it again."
        )


def _complete_files(paths) -> list[Path]:
    return sorted(path for path in paths if not path.name.endswith(".part"))


def _select_audio_files(target_dir: Path, stem: str) -> list[Path]:
    files = [
        path
        for path in _complete_files(target_dir.glob(f"{stem}.*"))
        if path.suffix.lower()
        in {".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".webm"}
        and not _is_audio_source(path, stem)
    ]
    return sorted(files)


def _select_video_files(target_dir: Path, stem: str) -> list[Path]:
    preferred = _stem_video_path(target_dir, stem)
    if preferred.exists() and not preferred.name.endswith(".part"):
        return [preferred]
    files = [
        path
        for path in _complete_files(target_dir.glob(f"{stem}.*"))
        if path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
        and not _is_ytdlp_format_part(path, stem)
        and not _is_audio_source(path, stem)
    ]
    return sorted(files)


def _remove_sources(paths: list[Path], *, keep: Path) -> None:
    keep_resolved = keep.resolve()
    for path in paths:
        try:
            if path.resolve() != keep_resolved:
                path.unlink(missing_ok=True)
        except Exception:
            pass
