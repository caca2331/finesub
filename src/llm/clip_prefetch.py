"""Background window clip extraction and Gemini upload prefetch."""

from __future__ import annotations

import concurrent.futures as cf
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

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
        # key: chunk_id -> Future
        self._futures: Dict[str, cf.Future[UploadedFileRef | None]] = {}
        # key: chunk_id -> UploadedFileRef
        self._refs: Dict[str, UploadedFileRef] = {}
        # lock protecting _futures and _refs
        self._lock = threading.Lock()

    def schedule(self, window: SubtitleWindow) -> None:
        chunk_id = window.chunk_id
        with self._lock:
            # already have uploaded ref -> nothing to do
            if chunk_id in self._refs:
                return
            existing = self._futures.get(chunk_id)
            # if an in-progress future exists, don't submit another
            if existing is not None and not existing.done():
                return
            # else submit a new background task and store the future
            fut = self._executor.submit(self._extract_and_upload, window)
            self._futures[chunk_id] = fut

    def prefetch_next(
        self, windows: Sequence[SubtitleWindow], current_index: int
    ) -> None:
        next_index = current_index + 1
        if 0 <= next_index < len(windows):
            self.schedule(windows[next_index])

    def get_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        chunk_id = window.chunk_id
        # fast path: check cached ref under lock
        with self._lock:
            cached = self._refs.get(chunk_id)
            future = self._futures.get(chunk_id)
        if cached is not None:
            return cached

        if future is None:
            # no background task -> do extraction/upload synchronously
            ref = self._extract_and_upload(window)
            if ref is not None:
                with self._lock:
                    self._refs[chunk_id] = ref
            return ref

        # wait for the background future to finish (do not hold lock while blocking)
        ref = future.result()
        if ref is not None:
            with self._lock:
                self._refs[chunk_id] = ref
                # clean up the future entry now that result is stored
                self._futures.pop(chunk_id, None)
        return ref

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def _extract_and_upload(self, window: SubtitleWindow) -> UploadedFileRef | None:
        chunk_id = window.chunk_id
        clip_path = self._clip_base_dir / f"{chunk_id}{self._clip_suffix}"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        # extraction (may be IO/ffmpeg heavy)
        self._extract_fn(
            self._audio_path,
            window.clip_start,
            window.clip_end,
            clip_path,
        )
        if self._upload_fn is None:
            # ensure we remove any future entry if called from worker
            with self._lock:
                self._futures.pop(chunk_id, None)
            return None
        ref = self._upload_fn(clip_path)
        # store result under lock and remove the future mapping
        with self._lock:
            self._refs[chunk_id] = ref
            self._futures.pop(chunk_id, None)
        return ref