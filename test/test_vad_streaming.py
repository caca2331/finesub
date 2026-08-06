"""Streamed VAD must be bit-identical to the in-memory path (plan B).

The reference chain is _load_asr_audio_streamed -> light_normalize ->
_compute_frame_tracks_for_waveform / run_vad on the fully loaded audio; the
streamed chain is _streamed_frame_tracks / run_vad_file with a small core so
short synthetic files still exercise multiple blocks. Assertions use exact
tensor equality — near-misses mean a determinism regression, not tolerance
tuning."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_playground.speech.preprocessing import energy as vad_energy


def _synth_speechlike(duration_sec: float, sr: int, *, seed: int = 7) -> np.ndarray:
    """Tone bursts + noise + silences so VAD has real structure to segment."""

    rng = np.random.default_rng(seed)
    n = int(duration_sec * sr)
    t = np.arange(n) / sr
    x = 0.004 * rng.standard_normal(n)  # noise floor
    burst = int(2.0 * sr)
    period = int(5.0 * sr)
    for start in range(0, n - burst, period):
        seg = np.arange(burst) / sr
        f0 = 180 + (start % (7 * period)) / period * 40
        tone = 0.25 * np.sin(2 * np.pi * f0 * seg) * (1 + 0.4 * np.sin(2 * np.pi * 3 * seg))
        tone += 0.08 * np.sin(2 * np.pi * 5 * f0 * seg)
        x[start : start + burst] += tone
    # slow loudness drift so the RMS gain track is non-trivial
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * t / 97.0)
    return x.astype(np.float32)


def _write_wav(path, data: np.ndarray, sr: int, *, stereo: bool = False) -> None:
    import soundfile as sf

    if stereo:
        data = np.stack([data, 0.9 * data], axis=1)
    sf.write(str(path), data, sr, subtype="PCM_16")


def _memory_tracks(path, energy_mode: str = "weighted"):
    with torch.inference_mode():
        audio = vad_energy._load_asr_audio_streamed(str(path))
        normalized = vad_energy.light_normalize(audio, vad_energy.TARGET_SR)
        tracks = vad_energy._compute_frame_tracks_for_waveform(
            normalized, vad_energy.TARGET_SR, energy_mode=energy_mode
        )
    return tracks, audio.numel()


def _assert_tracks_equal(path, *, core_sec: float, context_sec: float = 90.0) -> None:
    (m_dbfs, m_energy, m_starts, m_ends), n_samples = _memory_tracks(path)
    with torch.inference_mode():
        s_dbfs, s_energy, s_starts, s_ends, duration = vad_energy._streamed_frame_tracks(
            path, energy_mode="weighted", core_sec=core_sec, context_sec=context_sec
        )

    assert duration == n_samples / float(vad_energy.TARGET_SR)
    assert s_dbfs.shape == m_dbfs.shape
    assert torch.equal(s_dbfs, m_dbfs), "frame dBFS diverged from the in-memory path"
    assert torch.equal(s_energy, m_energy), "adaptive energy diverged from the in-memory path"
    assert torch.equal(s_starts, m_starts)
    assert torch.equal(s_ends, m_ends)


def test_streamed_tracks_bit_identical_multiblock_16k(tmp_path) -> None:
    # 190.37s at 16k (no resample), core 60s -> 4 blocks; the odd tail length
    # exercises the forced final n-1 gain anchor and EOF frame padding.
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "clip16k.wav"
    _write_wav(wav, _synth_speechlike(190.37, sr), sr)
    _assert_tracks_equal(wav, core_sec=60.0)


def test_streamed_tracks_bit_identical_with_resample_grid(tmp_path, monkeypatch) -> None:
    # 44.1k stereo source: both paths must resample on the same absolute
    # BLOCK_LENGTH source grid. Shrink BLOCK_LENGTH so a short file still
    # crosses several resample-block seams.
    monkeypatch.setattr(vad_energy, "BLOCK_LENGTH", 30.0)
    src_sr = 44100
    wav = tmp_path / "clip44k.wav"
    _write_wav(wav, _synth_speechlike(160.2, src_sr, seed=11), src_sr, stereo=True)
    _assert_tracks_equal(wav, core_sec=70.0)


def _clipping_input(sr: int, seconds: float = 150.0):
    """Quiet bed + sparse loud clicks: the RMS gain boosts toward -24 dBFS and
    the clicks then push |x| past the limit, so the clamp really engages."""
    x = _synth_speechlike(seconds, sr, seed=23) * 0.05
    for pos in range(sr * 3, len(x) - sr, sr * 13):
        x[pos : pos + 40] += np.float32(0.95)
    return x


def test_streamed_tracks_bit_identical_when_the_clamp_engages(tmp_path) -> None:
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "clip-peak.wav"
    _write_wav(wav, _clipping_input(sr), sr)

    # Sanity: without the clamp this input exceeds the limit.
    with torch.inference_mode(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(vad_energy, "NORM_PEAK_LIMIT", 1e9)
        audio = vad_energy._load_asr_audio_streamed(str(wav))
        unlimited = vad_energy.light_normalize(audio, sr)
    assert float(torch.max(torch.abs(unlimited))) > vad_energy.NORM_PEAK_LIMIT

    _assert_tracks_equal(wav, core_sec=60.0)


def test_clamp_is_per_sample_not_a_global_rescale(tmp_path) -> None:
    """The limit is enforced by clipping the outliers, so every sample the
    normalizer left below it must come through untouched. A global rescale --
    what this used to do -- would move all of them."""
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "clip-peak.wav"
    _write_wav(wav, _clipping_input(sr, seconds=60.0), sr)

    with torch.inference_mode():
        audio = vad_energy._load_asr_audio_streamed(str(wav))
        limited = vad_energy.light_normalize(audio, sr)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(vad_energy, "NORM_PEAK_LIMIT", 1e9)
            unlimited = vad_energy.light_normalize(audio, sr)

    # The bound is whatever 0.98 rounds to in float32, which is a hair above it.
    limit32 = float(torch.tensor(vad_energy.NORM_PEAK_LIMIT, dtype=torch.float32))
    assert float(torch.max(torch.abs(limited))) <= limit32
    inside = torch.abs(unlimited) <= limit32
    assert torch.equal(limited[inside], unlimited[inside])
    touched = int((~inside).sum())
    assert 0 < touched < unlimited.numel() // 1000  # outliers only, not the track


def test_streamed_single_block_short_file(tmp_path) -> None:
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "short.wav"
    _write_wav(wav, _synth_speechlike(45.5, sr, seed=3), sr)
    _assert_tracks_equal(wav, core_sec=600.0)  # file < core -> single block


def test_run_vad_file_matches_run_vad_intervals(tmp_path) -> None:
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "clip-intervals.wav"
    _write_wav(wav, _synth_speechlike(190.37, sr, seed=5), sr)

    with torch.inference_mode():
        audio = vad_energy._load_asr_audio_streamed(str(wav))
        normalized = vad_energy.light_normalize(audio, sr)
        mem_items, mem_meta = vad_energy.run_vad(normalized, params=vad_energy.vad_params())

    with torch.inference_mode():
        st_items, st_meta, duration, energy_track = vad_energy.run_vad_file(
            str(wav), params=vad_energy.vad_params(), core_sec=60.0
        )

    assert st_items == mem_items  # exact float equality, not tolerance
    assert duration == audio.numel() / float(sr)
    assert energy_track.energy_mode == "weighted"
    assert energy_track.hop_sec == pytest.approx(vad_energy.HOP_MS / 1000.0)
    assert energy_track.frame_sec == pytest.approx(vad_energy.FRAME_MS / 1000.0)
    assert int(energy_track.energy_db.numel()) > 0
    assert st_meta["vad"]["streaming"]["core_sec"] == 60.0
    assert {k: v for k, v in st_meta["vad"].items()
                if k not in ("streaming", "pause_hints")} == mem_meta["vad"]


def test_streamed_rejects_undersized_context(tmp_path) -> None:
    sr = vad_energy.TARGET_SR
    wav = tmp_path / "ctx.wav"
    _write_wav(wav, _synth_speechlike(130.0, sr, seed=9), sr)
    with pytest.raises(ValueError, match="context_sec"):
        vad_energy._streamed_frame_tracks(
            wav, energy_mode="weighted", core_sec=60.0, context_sec=30.0
        )
