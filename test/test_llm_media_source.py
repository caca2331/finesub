from __future__ import annotations

import pytest

from llm import media_source
from llm.media_source import validate_video_audio_coverage


def test_coverage_ok_within_tolerance(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        media_source,
        "_probe_stream_durations",
        lambda path: {"video": 2000.0, "audio": 1995.0},
    )
    validate_video_audio_coverage(tmp_path / "video.mp4")  # no raise


def test_coverage_raises_on_truncated_audio_track(monkeypatch, tmp_path) -> None:
    # The BV1ojjc6MEAs incident #1: a resumed/corrupt m4a merged into an mp4
    # with 2014s of video but only 50s of audio; every stage ran "successfully".
    monkeypatch.setattr(
        media_source,
        "_probe_stream_durations",
        lambda path: {"video": 2014.7, "audio": 50.2},
    )
    with pytest.raises(RuntimeError, match="likely corrupt"):
        validate_video_audio_coverage(tmp_path / "video.mp4")


def test_coverage_raises_on_truncated_video_track(monkeypatch, tmp_path) -> None:
    # Incident #2 (same video, next download): this time the VIDEO track was
    # the truncated one (870s of video under 2014s of audio) — window clips
    # past 870s came out audio-only and Gemini file processing FAILED.
    monkeypatch.setattr(
        media_source,
        "_probe_stream_durations",
        lambda path: {"video": 870.0, "audio": 2014.8},
    )
    with pytest.raises(RuntimeError, match="likely corrupt"):
        validate_video_audio_coverage(tmp_path / "video.mp4")


def test_coverage_skips_when_unprobeable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(media_source, "_probe_stream_durations", lambda path: {})
    validate_video_audio_coverage(tmp_path / "video.mp4")  # no raise


def test_record_video_id_concurrent_writers_lose_no_entries(tmp_path) -> None:
    # Two batch download workers resolve different URLs at once; the url-map
    # read-modify-write must merge instead of clobbering.
    import threading

    urls = [f"https://example.com/v{i}" for i in range(32)]
    barrier = threading.Barrier(4)

    def record(chunk: list[str]) -> None:
        barrier.wait()
        for url in chunk:
            media_source.record_video_id(url, url.rsplit("/", 1)[1], tmp_path)

    threads = [
        threading.Thread(target=record, args=(urls[i::4],)) for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    mapping = media_source.load_url_map(tmp_path)
    assert len(mapping) == len(urls)
    assert mapping["https://example.com/v7"] == "v7"
