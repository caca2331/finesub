"""Background window clip extraction and Gemini upload prefetch."""

from __future__ import annotations

import concurrent.futures as cf
import threading
from pathlib import Path
from typing import Callable, Dict, Sequence

from finesub.media.clips import CLIP_AUDIO_SUFFIX, extract_window_clip
from finesub.reporting import bind_reporter, current_reporter
from .chunking import SubtitleWindow
from .client import UploadedFileRef, with_media_duration


class WindowClipPrefetcher:
    """Extract + upload window clips on a background thread.

    After window planning, schedule the first window immediately. At the
    start of window *i*, schedule window *i+1*. ``get_ref`` blocks only until
    the requested window's clip (and upload, when configured) finishes.
    """

    def __init__(
        self,
        audio_path: str | Path,
        clip_base_dir: str | Path,
        *,
        extract_fn: Callable[..., Path] = extract_window_clip,
        upload_fn: Callable[[Path, threading.Event], UploadedFileRef] | None = None,
        clip_suffix: str = CLIP_AUDIO_SUFFIX,
        # Serial execution prefetches one window ahead; parallel dispatch cuts
        # the next batch, so ffmpeg + upload get their own small concurrency
        # cap (they spend bandwidth and disk, not LLM quota -- plan A.6).
        max_workers: int = 1,
    ) -> None:
        self._audio_path = Path(audio_path)
        self._clip_base_dir = Path(clip_base_dir)
        self._extract_fn = extract_fn
        self._upload_fn = upload_fn
        # Set at shutdown. An upload mid-retry sees it between attempts and
        # stops; a request already on the wire still runs to its own timeout.
        self._cancel = threading.Event()
        self._clip_suffix = clip_suffix
        self._executor = cf.ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="llm-clip",
            # Bound at construction: the prefetcher outlives no scope of its
            # own, and its threads would otherwise report into the void.
            initializer=bind_reporter,
            initargs=(current_reporter(),),
        )
        self._futures: Dict[str, cf.Future[UploadedFileRef | None]] = {}
        self._results: Dict[str, UploadedFileRef | None] = {}
        self._lock = threading.Lock()

    def schedule(self, window: SubtitleWindow) -> None:
        chunk_id = window.chunk_id
        with self._lock:
            if chunk_id in self._results or chunk_id in self._futures:
                return
            self._futures[chunk_id] = self._executor.submit(
                self._extract_and_upload, window
            )

    def prefetch_next(
        self, windows: Sequence[SubtitleWindow], current_index: int
    ) -> None:
        next_index = current_index + 1
        if 0 <= next_index < len(windows):
            self.schedule(windows[next_index])

    def get_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        chunk_id = window.chunk_id
        with self._lock:
            if chunk_id in self._results:
                return self._results[chunk_id]
            future = self._futures.get(chunk_id)
            if future is None:
                future = self._executor.submit(self._extract_and_upload, window)
                self._futures[chunk_id] = future

        try:
            ref = future.result()
        except BaseException:
            with self._lock:
                if self._futures.get(chunk_id) is future:
                    self._futures.pop(chunk_id, None)
            raise

        with self._lock:
            if chunk_id in self._results:
                return self._results[chunk_id]
            self._results[chunk_id] = ref
            if self._futures.get(chunk_id) is future:
                self._futures.pop(chunk_id, None)
            return ref

    def shutdown(self) -> None:
        # Anything still queued is for windows nobody will ask about again --
        # at teardown (normal end or drain-after-failure) every extraction
        # still pending would burn an ffmpeg run and a Gemini Files upload
        # for nothing, so cancel instead of running the queue dry.
        self._cancel.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _extract_and_upload(self, window: SubtitleWindow) -> UploadedFileRef | None:
        clip_path = self._clip_base_dir / f"{window.chunk_id}{self._clip_suffix}"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        self._extract_fn(
            self._audio_path,
            window.clip_start,
            window.clip_end,
            clip_path,
        )
        if self._upload_fn is None:
            return None
        # The window says exactly how long the clip is; the clip file itself
        # cannot be probed reliably (see UploadedFileRef.duration_seconds).
        return with_media_duration(
            self._upload_fn(clip_path, self._cancel),
            float(window.clip_end) - float(window.clip_start),
        )
