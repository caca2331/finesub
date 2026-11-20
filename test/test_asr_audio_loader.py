"""The aligner's block loader: the streamed 16k audio Whisper is fed, slice by slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


def _reconstruct_via_block_loader(path: Path, *, block_seconds: float, pad_seconds: float, step: float) -> np.ndarray:
    """Rebuild the whole 16k timeline through AudioBlockLoader, as run_vad_asr does."""
    from finesub.speech.recognition import transcribe as asr_align
    from finesub.speech.preprocessing import energy as vad_energy
    from finesub.speech.preprocessing.audio import get_audio_info

    sr, frames = get_audio_info(str(path))
    dur = frames / float(sr)
    loader = asr_align.AudioBlockLoader(
        str(path),
        target_sr=vad_energy.TARGET_SR,
        block_seconds=block_seconds,
        pad_seconds=pad_seconds,
        preprocess=False,
    )
    parts = []
    s = 0.0
    while s < dur:
        e = min(dur, s + step)
        parts.append(loader.get_slice(s, e).copy())
        s = e
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def test_streaming_loader_matches_inmemory_audio_no_resample(tmp_path, monkeypatch) -> None:
    """The alignment streaming path must feed Whisper the same 16k samples the old
    in-memory path did. At 16k (no resample) the two must be bit-identical."""
    import soundfile as sf

    from finesub.speech.preprocessing import energy as vad_energy

    monkeypatch.setattr(vad_energy, "BLOCK_LENGTH", 2.0)  # force multiple block boundaries
    sr = vad_energy.TARGET_SR
    n = int(7.3 * sr)
    t = np.arange(n) / sr
    mono = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    src = tmp_path / "clip16k.wav"
    sf.write(str(src), mono, sr, subtype="PCM_16")

    with torch.inference_mode():
        old = vad_energy._load_asr_audio_streamed(str(src)).detach().cpu().numpy().astype(np.float32)
    new = _reconstruct_via_block_loader(src, block_seconds=4.0, pad_seconds=1.0, step=3.0)

    assert old.shape == new.shape
    assert np.array_equal(old, new)


def test_streaming_loader_matches_inmemory_audio_with_resample(tmp_path, monkeypatch) -> None:
    """At 44.1k the two paths resample in different block layouts; differences must
    stay negligible and confined to block boundaries (Whisper output is unaffected)."""
    import soundfile as sf

    from finesub.speech.preprocessing import energy as vad_energy

    monkeypatch.setattr(vad_energy, "BLOCK_LENGTH", 2.0)
    src_sr = 44100
    n = int(7.3 * src_sr)
    t = np.arange(n) / src_sr
    mono = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    stereo = np.stack([mono, mono], axis=1).astype(np.float32)
    src = tmp_path / "clip44k.wav"
    sf.write(str(src), stereo, src_sr, subtype="PCM_16")

    with torch.inference_mode():
        old = vad_energy._load_asr_audio_streamed(str(src)).detach().cpu().numpy().astype(np.float32)
    new = _reconstruct_via_block_loader(src, block_seconds=4.0, pad_seconds=1.0, step=3.0)

    assert abs(old.shape[0] - new.shape[0]) <= 1
    m = min(old.shape[0], new.shape[0])
    diff = np.abs(old[:m] - new[:m])
    # Negligible and rare: only a few samples of resampler ringing at boundaries.
    assert diff.max() < 5e-2
    assert np.count_nonzero(diff > 1e-4) < 0.001 * m


def test_block_loader_slice_crossing_boundary_is_not_truncated(tmp_path) -> None:
    """A slice that straddles a block boundary by more than pad_seconds must
    still return the whole requested range (regression guard for the loader)."""
    from finesub.speech.recognition import transcribe as asr_align
    import soundfile as sf

    sr = 16000
    n = int(9.0 * sr)
    ramp = (np.arange(n, dtype=np.float32) / n)  # strictly increasing -> position-identifiable
    src = tmp_path / "ramp16k.wav"
    sf.write(str(src), ramp, sr, subtype="PCM_16")

    loader = asr_align.AudioBlockLoader(
        str(src), target_sr=sr, block_seconds=4.0, pad_seconds=1.0, preprocess=False
    )
    # [3.0, 7.0] crosses the block-1 boundary (4.0) by 3s, far more than pad=1s.
    clip = loader.get_slice(3.0, 7.0)
    expected = int(round((7.0 - 3.0) * sr))
    assert abs(clip.shape[0] - expected) <= 1
    # PCM_16 round-trip tolerance; values must track the source ramp, not stop early.
    assert clip[0] == pytest.approx(3.0 / 9.0, abs=1e-3)
    assert clip[-1] == pytest.approx(7.0 / 9.0, abs=1e-3)


def test_block_loader_does_not_reload_for_a_slice_past_the_end(tmp_path, monkeypatch) -> None:
    """The last group's tail slice runs past the file's end, and `_block_end`
    stops at the end, so every such slice used to decode the whole block again
    (audit S-5). It is now served from the block already held when a reload
    would read that same window -- and returns bit-identical samples, resampled
    44.1k stereo included. A reload that would read a different window still
    happens."""
    from finesub.speech.recognition import transcribe as asr_align
    import soundfile as sf

    src_sr = 44100
    t = np.arange(int(9.0 * src_sr)) / src_sr
    mono = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 3000 * t)
    src = tmp_path / "clip44k.wav"
    sf.write(str(src), np.stack([mono, mono], axis=1).astype(np.float32), src_sr, subtype="PCM_16")

    loads = []
    real_load = asr_align.AudioBlockLoader._load_block_for_range

    def counting_load(self, start_sec, end_sec):
        loads.append((start_sec, end_sec))
        return real_load(self, start_sec, end_sec)

    monkeypatch.setattr(asr_align.AudioBlockLoader, "_load_block_for_range", counting_load)

    def loader(block_seconds: float):
        return asr_align.AudioBlockLoader(
            str(src), target_sr=16000, block_seconds=block_seconds, pad_seconds=1.0
        )

    # One block holds the whole file: the tail slice reads exactly its window.
    held = loader(60.0)
    held.get_slice(2.0, 3.0)
    tails = [held.get_slice(8.8, 9.5) for _ in range(5)]
    assert len(loads) == 1
    fresh = loader(60.0).get_slice(8.8, 9.5)  # what each of those calls used to reload
    assert len(fresh) > 0
    for tail in tails:
        assert tail.dtype == fresh.dtype and np.array_equal(tail, fresh)

    # The file ends inside block 1's pad, so block 1 is held up to the end while
    # a slice starting in block 2 reads [7, 9]: a different window, reloaded as before.
    loads.clear()
    edge = loader(4.0)
    edge.get_slice(5.0, 6.0)
    edge.get_slice(8.5, 9.5)
    assert len(loads) == 2
