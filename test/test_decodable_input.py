from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asr_playground.speech.preprocessing import audio as preprocessing_audio


def _write_wav(path: Path, seconds: float = 0.1) -> Path:
    sf.write(str(path), np.zeros(int(16000 * seconds), dtype="float32"), 16000)
    return path


def test_a_readable_input_is_used_as_is(tmp_path) -> None:
    source = _write_wav(tmp_path / "clip.wav")

    path, temporary = preprocessing_audio.ensure_decodable_input(source, tmp_path)

    assert path == source
    assert temporary is None


def test_an_unreadable_container_is_decoded_once(tmp_path, monkeypatch) -> None:
    source = tmp_path / "clip.webm"
    source.write_bytes(b"not audio")
    calls: list[Path] = []

    def transcode(input_path, out_path, **kwargs):
        calls.append(Path(out_path))
        return _write_wav(Path(out_path))

    monkeypatch.setattr(
        "asr_playground.media.ffmpeg.transcode_to_lossless_audio", transcode
    )

    path, temporary = preprocessing_audio.ensure_decodable_input(source, tmp_path)
    assert path == temporary == tmp_path / "clip-decoded.flac"
    assert path.exists()

    # The caller deletes it only on success, so a second stage must reuse it.
    again, _ = preprocessing_audio.ensure_decodable_input(source, tmp_path)
    assert again == path
    assert len(calls) == 1


def test_a_decode_killed_partway_leaves_nothing_reusable(tmp_path, monkeypatch) -> None:
    source = tmp_path / "clip.webm"
    source.write_bytes(b"not audio")

    def transcode(input_path, out_path, **kwargs):
        # ffmpeg writes as it decodes, so an interrupted run leaves a valid but
        # truncated file behind. It must not be able to claim the final name --
        # reusing one would transcribe part of the input as if it were all of it.
        _write_wav(Path(out_path), seconds=0.01)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(
        "asr_playground.media.ffmpeg.transcode_to_lossless_audio", transcode
    )

    with pytest.raises(RuntimeError):
        preprocessing_audio.ensure_decodable_input(source, tmp_path)

    assert not (tmp_path / "clip-decoded.flac").exists()
    assert list(tmp_path.glob("*.part.flac")) == []
