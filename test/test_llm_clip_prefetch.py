from __future__ import annotations

import concurrent.futures as cf
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from llm.client import UploadedFileRef
from llm.clip_prefetch import WindowClipPrefetcher


@dataclass(frozen=True)
class _Win:
    chunk_id: str
    clip_start: float = 0.0
    clip_end: float = 1.0


def test_prefetcher_schedules_first_window_then_next(tmp_path) -> None:
    order: list[str] = []

    def fake_extract(audio_path, clip_start, clip_end, out_path, **kwargs):
        order.append(f"extract:{Path(out_path).name}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return Path(out_path)

    def fake_upload(path: Path) -> UploadedFileRef:
        order.append(f"upload:{path.name}")
        return UploadedFileRef(
            file_id=f"files/{path.name}",
            filename=str(path),
            mime_type="audio/aac",
        )

    windows = [_Win("0001"), _Win("0002"), _Win("0003")]
    prefetcher = WindowClipPrefetcher(
        tmp_path / "audio.wav",
        tmp_path / "clips",
        extract_fn=fake_extract,
        upload_fn=fake_upload,
    )
    prefetcher.schedule(windows[0])
    try:
        prefetcher.prefetch_next(windows, 0)
        ref = prefetcher.get_ref(windows[0])
        assert ref is not None
        assert ref.file_id == "files/0001.aac"
        assert order[:2] == ["extract:0001.aac", "upload:0001.aac"]

        prefetcher.prefetch_next(windows, 1)
        ref2 = prefetcher.get_ref(windows[1])
        assert ref2 is not None
        assert ref2.file_id == "files/0002.aac"
        assert order.count("extract:0002.aac") == 1
        assert order.count("upload:0002.aac") == 1
    finally:
        prefetcher.shutdown()


def test_prefetch_runs_off_main_thread(tmp_path) -> None:
    main_id = threading.get_ident()
    worker_ids: list[int] = []

    def fake_extract(audio_path, clip_start, clip_end, out_path, **kwargs):
        worker_ids.append(threading.get_ident())
        time.sleep(0.05)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return Path(out_path)

    prefetcher = WindowClipPrefetcher(
        tmp_path / "audio.wav",
        tmp_path / "clips",
        extract_fn=fake_extract,
        upload_fn=None,
    )
    win = _Win("0001")
    prefetcher.schedule(win)
    try:
        prefetcher.get_ref(win)
    finally:
        prefetcher.shutdown()

    assert worker_ids
    assert worker_ids[0] != main_id


def test_concurrent_get_ref_extracts_and_uploads_once(tmp_path) -> None:
    callers = 8
    caller_barrier = threading.Barrier(callers)
    count_lock = threading.Lock()
    extract_count = 0
    upload_count = 0

    def fake_extract(audio_path, clip_start, clip_end, out_path, **kwargs):
        nonlocal extract_count
        with count_lock:
            extract_count += 1
        time.sleep(0.05)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return Path(out_path)

    def fake_upload(path: Path) -> UploadedFileRef:
        nonlocal upload_count
        with count_lock:
            upload_count += 1
        return UploadedFileRef(
            file_id=f"files/{path.name}",
            filename=str(path),
            mime_type="audio/aac",
        )

    prefetcher = WindowClipPrefetcher(
        tmp_path / "audio.wav",
        tmp_path / "clips",
        extract_fn=fake_extract,
        upload_fn=fake_upload,
    )
    win = _Win("0001")

    def fetch() -> UploadedFileRef | None:
        caller_barrier.wait()
        return prefetcher.get_ref(win)

    try:
        with cf.ThreadPoolExecutor(max_workers=callers) as executor:
            refs = list(executor.map(lambda _: fetch(), range(callers)))
    finally:
        prefetcher.shutdown()

    assert all(ref is not None for ref in refs)
    assert {ref.file_id for ref in refs if ref is not None} == {"files/0001.aac"}
    assert extract_count == 1
    assert upload_count == 1


def test_completed_prefetch_without_upload_is_not_extracted_again(tmp_path) -> None:
    extract_count = 0

    def fake_extract(audio_path, clip_start, clip_end, out_path, **kwargs):
        nonlocal extract_count
        extract_count += 1
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x")
        return Path(out_path)

    prefetcher = WindowClipPrefetcher(
        tmp_path / "audio.wav",
        tmp_path / "clips",
        extract_fn=fake_extract,
        upload_fn=None,
    )
    win = _Win("0001")
    prefetcher.schedule(win)
    try:
        assert prefetcher.get_ref(win) is None
        assert prefetcher.get_ref(win) is None
    finally:
        prefetcher.shutdown()

    assert extract_count == 1
