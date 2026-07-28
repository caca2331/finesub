"""Background window clip extraction and Gemini upload prefetch."""

from __future__ import annotations

import concurrent.futures as cf
import threading
from pathlib import Path
from typing import Callable, Dict, Sequence

from .audio_clips import CLIP_AUDIO_SUFFIX, extract_window_clip
from .chunking import SubtitleWindow
from .client import UploadedFileRef


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
        upload_fn: Callable[[Path], UploadedFileRef] | None = None,
        clip_suffix: str = CLIP_AUDIO_SUFFIX,
    ) -> None:
        self._audio_path = Path(audio_path)
        self._clip_base_dir = Path(clip_base_dir)
        self._extract_fn = extract_fn
        self._upload_fn = upload_fn
        self._clip_suffix = clip_suffix
        self._executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-clip")
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
        self._executor.shutdown(wait=True)

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
        return self._upload_fn(clip_path)
