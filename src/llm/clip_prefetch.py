"""Background window clip extraction and Gemini upload prefetch."""

from __future__ import annotations

import concurrent.futures as cf
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
        self._futures: Dict[str, cf.Future[UploadedFileRef | None]] = {}
        self._refs: Dict[str, UploadedFileRef] = {}

    def schedule(self, window: SubtitleWindow) -> None:
        chunk_id = window.chunk_id
        if chunk_id in self._refs or chunk_id in self._futures:
            return
        self._futures[chunk_id] = self._executor.submit(self._extract_and_upload, window)

    def prefetch_next(
        self, windows: Sequence[SubtitleWindow], current_index: int
    ) -> None:
        next_index = current_index + 1
        if 0 <= next_index < len(windows):
            self.schedule(windows[next_index])

    def get_ref(self, window: SubtitleWindow) -> UploadedFileRef | None:
        chunk_id = window.chunk_id
        cached = self._refs.get(chunk_id)
        if cached is not None:
            return cached

        future = self._futures.get(chunk_id)
        if future is None:
            ref = self._extract_and_upload(window)
            if ref is not None:
                self._refs[chunk_id] = ref
            return ref

        ref = future.result()
        if ref is not None:
            self._refs[chunk_id] = ref
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
