"""The vectorized silero probability pass must reproduce silero's own per-frame
recurrence -- including across feed() boundaries.

Marked heavy-resource: it loads the silero model. The decision logic on top of
the probabilities is covered model-free in test_vad_silero_ghost.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from asr_playground.speech.preprocessing import energy as vad_energy
from asr_playground.speech.preprocessing import silero_ghost
from asr_playground.speech.preprocessing.energy import TARGET_SR

pytestmark = pytest.mark.heavy_resource

HOP = silero_ghost.SILERO_HOP


def reference_probs(wav: torch.Tensor) -> np.ndarray:
    """Silero's shipped stateful model, one 512-sample frame at a time.

    This is what the production path did before it was vectorized; it stays here
    as the thing the fast path is checked against.
    """
    model = silero_ghost._model()
    model.reset_states()
    flat = wav.reshape(-1)
    n = (flat.numel() // HOP) * HOP
    frames = flat[:n].reshape(-1, HOP)
    out = np.empty(len(frames), dtype=np.float32)
    with torch.no_grad():
        for i in range(len(frames)):
            out[i] = float(model(frames[i], TARGET_SR).item())
    return out


def synthetic_speechlike(seconds: float, seed: int = 0) -> torch.Tensor:
    """Alternating voiced-ish bursts and near-silence, so the probabilities
    actually swing across the thresholds the assist keys on."""
    g = torch.Generator().manual_seed(seed)
    n = int(seconds * TARGET_SR)
    t = torch.arange(n, dtype=torch.float32) / TARGET_SR
    voiced = sum(torch.sin(2 * np.pi * f * t) / (i + 1)
                 for i, f in enumerate((140.0, 280.0, 420.0, 700.0, 1400.0)))
    burst = ((t * 2.0).floor() % 2 == 0).float()
    x = 0.25 * voiced * burst + 0.01 * torch.randn(n, generator=g)
    return x.float()


def test_vectorized_matches_per_frame_loop():
    wav = synthetic_speechlike(20.0)
    ref = reference_probs(wav)
    got = silero_ghost.frame_probs(wav)

    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-4
    # The assist thresholds the probabilities; no frame may cross differently.
    for thr in (silero_ghost.GHOST_SILERO_PEAK_MAX, silero_ghost.SIL_EVID,
                silero_ghost.CAP_SIL_THR):
        assert np.array_equal(got >= thr, ref >= thr), f"threshold flip at {thr}"


@pytest.mark.parametrize("chunk", [1, 7, 64, 8192])
def test_chunk_size_does_not_change_the_result(chunk):
    # Chunking only sets the extractor batch; the recurrence is unbroken. Tiny
    # chunks reassociate the LSTM's float math, hence a tolerance and not
    # exact equality.
    wav = synthetic_speechlike(12.0, seed=1)
    ref = silero_ghost.SileroProbStream(chunk_frames=silero_ghost.SILERO_CHUNK_FRAMES).feed(wav)
    got = silero_ghost.SileroProbStream(chunk_frames=chunk).feed(wav)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize(
    "cuts",
    [
        pytest.param([0, 4 * TARGET_SR, 8 * TARGET_SR], id="hop-aligned"),
        pytest.param([0, 100 * HOP + 37, 4 * TARGET_SR + 1, 9 * TARGET_SR - 5],
                     id="ragged"),
    ],
)
def test_feed_in_pieces_carries_state_and_context(cuts):
    """Piecewise feeding is what the streaming fusion relies on: the LSTM state,
    the 64-sample context and any sub-hop remainder must all cross the seam."""
    wav = synthetic_speechlike(12.0, seed=2)
    bounds = list(cuts) + [wav.numel()]
    ref = silero_ghost.frame_probs(wav)

    stream = silero_ghost.SileroProbStream()
    got = np.concatenate([stream.feed(wav[a:b]) for a, b in zip(bounds, bounds[1:])])

    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-6


def test_independent_pieces_would_diverge():
    """Guards the reason feed() is stateful: a fresh state per piece is not the
    same signal, so nobody may 'simplify' the seam handling away."""
    wav = synthetic_speechlike(12.0, seed=2)
    ref = silero_ghost.frame_probs(wav)
    half = (wav.numel() // 2 // HOP) * HOP
    naive = np.concatenate([
        silero_ghost.frame_probs(wav[:half]),
        silero_ghost.frame_probs(wav[half:]),
    ])
    assert np.abs(naive - ref).max() > 1e-3


def test_reset_starts_a_new_stream():
    wav = synthetic_speechlike(6.0, seed=3)
    stream = silero_ghost.SileroProbStream()
    first = stream.feed(wav)
    stream.reset()
    again = stream.feed(wav)
    assert np.array_equal(first, again)


def test_short_remainder_is_dropped_like_the_loop():
    wav = synthetic_speechlike(6.0, seed=4)[: 100 * HOP + 37]
    assert silero_ghost.frame_probs(wav).shape == (100,)
    assert silero_ghost.frame_probs(wav[: HOP - 1]).shape == (0,)


# --- riding along on the VAD's streaming blocks ------------------------------


def _write(tmp_path, seconds, sr=TARGET_SR, seed=8, gain=1.0):
    import soundfile as sf

    wav = (gain * synthetic_speechlike(seconds, seed=seed)).numpy()
    path = tmp_path / "a.wav"
    sf.write(str(path), wav, sr, subtype="PCM_16")
    return path


def _collected(path, *, core_sec, context_sec=90.0, device="cpu"):
    collector = silero_ghost.SileroProbCollector(device)
    with torch.inference_mode():
        vad_energy._streamed_frame_tracks(
            path, energy_mode="weighted", core_sec=core_sec,
            context_sec=context_sec, observer=collector,
        )
    return collector.probs()


def _in_memory(path, device="cpu"):
    with torch.inference_mode():
        wav = vad_energy._load_asr_audio_streamed(str(path))
        wav = vad_energy.light_normalize(wav, TARGET_SR)
    return silero_ghost.frame_probs(wav, device=device)


def test_collector_matches_the_in_memory_pass_multiblock(tmp_path):
    """core_sec=100 over 260 s means three blocks: the LSTM state and the
    64-sample context have to survive every seam."""
    path = _write(tmp_path, 260.0)
    got, ref = _collected(path, core_sec=100.0), _in_memory(path)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-6


def test_collector_matches_on_the_single_block_path(tmp_path):
    path = _write(tmp_path, 30.0, seed=9)
    got, ref = _collected(path, core_sec=600.0), _in_memory(path)  # < core
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-6


def test_collector_matches_on_clipping_input(tmp_path):
    """Quiet bed + sparse loud clicks, so the peak clamp really engages: the
    blocks the collector sees must be the clamped ones."""
    import soundfile as sf

    wav = 0.05 * synthetic_speechlike(260.0, seed=10).numpy()
    for pos in range(TARGET_SR * 3, len(wav) - TARGET_SR, TARGET_SR * 13):
        wav[pos : pos + 40] += np.float32(0.95)
    path = tmp_path / "clip-peak.wav"
    sf.write(str(path), wav, TARGET_SR, subtype="PCM_16")

    with torch.inference_mode(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(vad_energy, "NORM_PEAK_LIMIT", 1e9)
        unlimited = vad_energy.light_normalize(
            vad_energy._load_asr_audio_streamed(str(path)), TARGET_SR
        )
    assert float(torch.max(torch.abs(unlimited))) > vad_energy.NORM_PEAK_LIMIT

    got, ref = _collected(path, core_sec=100.0), _in_memory(path)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-6


def test_no_observer_leaves_the_tracks_untouched(tmp_path):
    path = _write(tmp_path, 260.0, seed=11)
    with torch.inference_mode():
        plain = vad_energy._streamed_frame_tracks(
            path, energy_mode="weighted", core_sec=100.0
        )
        hooked = vad_energy._streamed_frame_tracks(
            path, energy_mode="weighted", core_sec=100.0,
            observer=silero_ghost.SileroProbCollector(),
        )
    for a, b in zip(plain[:4], hooked[:4]):
        assert torch.equal(a, b)
    assert plain[4] == hooked[4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_agrees_with_cpu_on_every_threshold():
    wav = synthetic_speechlike(30.0, seed=5)
    ref = silero_ghost.frame_probs(wav, device="cpu")
    got = silero_ghost.frame_probs(wav, device="cuda")
    assert np.abs(got - ref).max() < 1e-3
    for thr in (silero_ghost.GHOST_SILERO_PEAK_MAX, silero_ghost.SIL_EVID,
                silero_ghost.CAP_SIL_THR):
        assert np.array_equal(got >= thr, ref >= thr), f"threshold flip at {thr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tf32_setting_is_restored():
    """The separator wants TF32 on; the assist must not leave it off."""
    prev = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    try:
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = True
        silero_ghost.frame_probs(synthetic_speechlike(2.0), device="cuda")
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def test_cuda_request_without_cuda_falls_back(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert silero_ghost.resolve_silero_device("cuda") == "cpu"
    assert "Warning:" in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_building_a_cuda_stream_leaves_the_cached_model_on_cpu():
    """SileroProbStream deep-copies before .to(device); .to() on the JIT
    submodules is in-place, so without the copy a CUDA stream would strand the
    process-global model on the GPU and break every later CPU caller."""
    wav = synthetic_speechlike(2.0, seed=6)
    ref = silero_ghost.frame_probs(wav, device="cpu")
    silero_ghost.SileroProbStream(device="cuda")
    assert next(silero_ghost._model()._model.parameters()).device.type == "cpu"
    assert np.array_equal(silero_ghost.frame_probs(wav, device="cpu"), ref)


def test_loading_the_model_restores_torch_thread_count():
    before = torch.get_num_threads()
    silero_ghost._MODEL = None
    try:
        silero_ghost._model()
    finally:
        pass
    assert torch.get_num_threads() == before
