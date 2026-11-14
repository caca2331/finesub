"""The Nemotron-3 backend without its weights: the front end, the chunk loop, vbx's geometry, fetching.

The numbers the backend was chosen on came from NeMo eager; the check against it on real audio is
`tools/diarization_explore/nemotron3_port.py` (GPU, not in this suite). What is pinned here is
what that check cannot see on its own: that the front end is the checkpoint's, that the loop never
lets a chunk see past its look-ahead, and that vbx's windows land where the assembly puts them back.
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from finesub.speech.diarization import chunking  # noqa: E402
from finesub.speech.diarization.backend import ARTIFACT_ENV  # noqa: E402
from finesub.speech.diarization import nemotron3_backend as backend  # noqa: E402
from finesub.speech.diarization import nemotron3_model as model  # noqa: E402
from finesub.speech.diarization import stage as diarization  # noqa: E402


# --- the front end -------------------------------------------------------------------------------

def test_the_filterbank_is_librosas_to_the_bit():
    """NeMo and transformers both take their filters from librosa; ours must be the same numbers."""
    librosa = pytest.importorskip("librosa")
    reference = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0.0, fmax=8000, norm="slaney")
    assert np.array_equal(model.mel_filters(128), reference)


def test_window_and_filters_are_rounded_as_the_checkpoint_stores_them():
    """⚠ The `.nemo` checkpoint is bf16, buffers included, and NeMo loads those buffers. Measured:
    the full-precision front end moved 15% of frames on a real file (`_as_checkpoint`)."""
    window = torch.hann_window(model.WIN_LENGTH, periodic=False)
    rounded = model._as_checkpoint(window)
    assert rounded.dtype == torch.float32
    assert torch.equal(rounded, window.to(torch.bfloat16).float())
    assert not torch.equal(rounded, window), "the rounding is a real change, not a no-op"


def test_features_have_the_centred_frame_count_and_a_zeroed_last_frame():
    samples = 16000 * 3 + 77
    features, valid = model.log_mel(torch.randn(1, samples) * 0.1, num_mels=128, device="cpu")
    assert valid == samples // model.HOP_LENGTH
    assert features.shape == (valid + 1, 128)
    assert torch.all(features[valid] == 0)
    assert torch.isfinite(features).all()


def test_cutting_the_features_into_pieces_changes_nothing(monkeypatch):
    """Ten-minute pieces keep memory flat; they must not move a single kept frame."""
    audio = torch.randn(1, 16000 * 4) * 0.1
    whole, _ = model.log_mel(audio, num_mels=128, device="cpu")
    monkeypatch.setattr(model, "feature_segments",
                        partial(chunking.feature_segments, segment_frames=37, context_frames=10))
    pieces, _ = model.log_mel(audio, num_mels=128, device="cpu")
    assert torch.allclose(whole, pieces, rtol=0, atol=1e-5)


# --- a tiny network through the real loop ---------------------------------------------------------

TINY = {
    "audio_config": {"hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 2, "intermediate_size": 32, "num_mel_bins": 128,
                     "subsampling_factor": 8, "hidden_act": "gelu",
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}},
    "head_config": {"hidden_size": 8, "num_speakers": 4},
    "streaming_config": {"speaker_cache_length": 16, "speaker_cache_silence_frames_per_speaker": 1,
                         "prediction_score_threshold": 0.25, "latest_frames_score_boost": 0.05,
                         "strong_boost_rate": 0.75, "weak_boost_rate": 1.5,
                         "min_positive_scores_rate": 0.5},
    "chunk_length": 5, "chunk_right_context": 2, "fifo_length": 3, "speaker_cache_update_period": 4,
}


def _random_network(tmp_path):
    torch.manual_seed(0)
    (tmp_path / model.CONFIG_NAME).write_text(json.dumps(TINY), encoding="utf-8")
    network = model.Nemotron3Network(model.Nemotron3Config.from_file(tmp_path / model.CONFIG_NAME))
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0.0, 0.3)
    return network.eval()


@pytest.fixture
def tiny(tmp_path):
    """A random network in the checkpoint's layout, read from a `config.json` of the same shape."""
    return _random_network(tmp_path)


def test_the_weights_load_strictly_under_the_repositorys_names(tmp_path):
    """`strict=True`: a renamed or missing tensor is an error, not a silently random layer."""
    safetensors = pytest.importorskip("safetensors.torch")
    network = _random_network(tmp_path)
    safetensors.save_file({name: value.contiguous() for name, value in network.state_dict().items()},
                          str(tmp_path / model.WEIGHTS_NAME))
    loaded = model.load(tmp_path, "cpu")
    audio = torch.randn(1, 16000 * 2) * 0.1
    assert np.array_equal(model.speaker_probabilities(network, audio, device="cpu"),
                          model.speaker_probabilities(loaded, audio, device="cpu"))


def test_one_row_per_spectrogram_frame_for_every_track(tiny):
    samples = 16000 * 10 + 123
    probabilities = model.speaker_probabilities(tiny, torch.randn(1, samples) * 0.1, device="cpu")
    assert probabilities.shape == (samples // model.HOP_LENGTH + 1, 4)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_the_speaker_cache_fills_and_is_compressed_to_its_length(tiny, monkeypatch):
    """Ten seconds is 25 chunks of this tiny geometry: far past the 16-frame cache."""
    seen = []
    original = model.SpeakerCache._compress

    def compress(self, embeds, probs, silence):
        kept = original(self, embeds, probs, silence)
        seen.append(kept[0].shape[1])
        return kept

    monkeypatch.setattr(model.SpeakerCache, "_compress", compress)
    model.speaker_probabilities(tiny, torch.randn(1, 16000 * 10) * 0.1, device="cpu")
    assert seen and set(seen) == {TINY["streaming_config"]["speaker_cache_length"]}


def test_a_chunk_never_sees_past_its_look_ahead(tiny):
    """The first chunk's answer depends on its own frames and its look-ahead, and nothing later."""
    frames = (TINY["chunk_length"] + TINY["chunk_right_context"]) * 8
    reach = frames * model.HOP_LENGTH + model.N_FFT
    audio = torch.randn(1, 16000 * 3) * 0.1
    changed = audio.clone()
    changed[:, reach:] = torch.randn(1, audio.shape[1] - reach) * 0.1
    first = TINY["chunk_length"] * 8
    a = model.speaker_probabilities(tiny, audio, device="cpu")[:first]
    b = model.speaker_probabilities(tiny, changed, device="cpu")[:first]
    assert np.allclose(a, b, atol=1e-6)


# --- vbx's geometry ---------------------------------------------------------------------------------

def test_the_windows_are_put_back_exactly_where_they_were_cut():
    """⚠ Windows and assembly must agree on where chunk c starts, or every boundary shifts.

    One speaker: after cutting into 16 s / 1.6 s windows and assembling with every window assigned
    to one cluster, the 20 ms map must come back unchanged.
    """
    from finesub.speech.diarization import diarizen_assemble

    grid = np.zeros((2500, 1), dtype=np.float32)  # 50 s
    grid[300:900, 0] = 1.0
    grid[1700:2450, 0] = 1.0
    local = backend.windows(grid)
    rebuilt = diarizen_assemble.assemble(local, np.zeros(local.shape[::2], dtype=np.int8))
    assert len(rebuilt) >= len(grid)
    assert np.array_equal(rebuilt[:len(grid), 0], grid[:, 0])


def test_the_rasterised_activity_spans_the_audio_on_the_20ms_grid():
    probabilities = np.zeros((1001, 2), dtype=np.float32)   # 10 s at 10 ms
    probabilities[100:400, 1] = 0.9
    grid = backend.rasterised(probabilities, samples=160000, source_step=0.01)
    assert grid.shape == (500, 2)
    assert grid[:, 0].sum() == 0
    assert grid[50:200, 1].all() and grid[200:, 1].sum() == 0


def test_activity_hands_over_the_grid_it_declares(monkeypatch):
    """The declared 20 ms and the returned array must agree, or every boundary is mis-timed."""
    from finesub.speech.diarization import diarizen_cluster, diarizen_embed
    from finesub.speech.diarization.diarizen_cluster import ClusterResult

    probabilities = np.zeros((1601, 8), dtype=np.float32)
    probabilities[:800, 0] = 0.9
    monkeypatch.setattr(backend, "frame_activity", lambda *a, **k: probabilities)
    asked = {}

    def embed(path, waveform, local, **kwargs):
        asked.update(kwargs)
        return np.zeros((local.shape[0], local.shape[2], 4))

    monkeypatch.setattr(diarizen_embed, "embed", embed)
    monkeypatch.setattr(diarizen_cluster, "cluster", lambda embeddings, local, *a, **k: ClusterResult(
        hard=np.zeros(local.shape[::2], dtype=np.int8), soft=np.ones((local.shape[0], local.shape[2], 1)),
        centroids=np.zeros((1, 4)), centroid_of_track=np.zeros(1, dtype=np.int16),
        single_speaker_log_bayes=0.0, log_bayes_embeddings=0, centroid_weights=np.ones(1)))

    handle = backend.Handle(model_dir=None, embedding=None, plda_dir=None)
    output = backend.activity(torch.zeros(1, 16000 * 16), handle=handle, device="cpu")
    seconds = output.activity.shape[0] * backend.BACKEND.frame_seconds
    assert 15.0 < seconds < 17.5
    assert asked.get("only_active") is True, "voiceprints only for pairs that speak, as measured"
    assert output.evidence is not None and output.evidence.native_step == backend.NATIVE_STEP


def test_only_active_embeds_the_speaking_pairs_in_the_same_batches(monkeypatch):
    """⚠ Batch composition changes the answer on this chain, so the skipping has to be the
    measured one: batches are formed from the active pairs, in chunk-then-track order."""
    from contextlib import contextmanager

    from finesub.speech.diarization import diarizen_embed

    batches = []

    def fake(fbank, weights):
        batches.append(int(weights.shape[0]))
        return torch.arange(weights.shape[0], dtype=torch.float32)[:, None].repeat(1, 256)

    @contextmanager
    def loaded(path, device):
        yield fake

    monkeypatch.setattr(diarizen_embed, "loaded_model", loaded)
    local = np.zeros((3, 800, 4), dtype=np.float32)
    local[0, :100, 0] = 1.0
    local[1, :100, 2] = 1.0
    local[2, :2, 3] = 1.0        # two frames: not more than the minimum, so skipped
    waveform = torch.zeros(1, 16000 * 20)

    out = diarizen_embed.embed("unused", waveform, local, device="cpu", only_active=True)
    assert out.shape == (3, 4, 256)
    assert batches == [2], "the two speaking pairs share one batch"
    assert np.isfinite(out[0, 0]).all() and np.isfinite(out[1, 2]).all()
    assert np.isnan(out[2, 3]).all() and np.isnan(out[0, 1]).all()

    batches.clear()
    diarizen_embed.embed("unused", waveform, local, device="cpu")
    assert batches == [2] * 6, "without it every (chunk, track) pair is embedded, as DiariZen does"


# --- the registry and fetching ---------------------------------------------------------------------

def test_nemotron3_is_the_default():
    assert diarization.DEFAULT_BACKEND == "nemotron3"
    assert diarization.resolve_backend(None) is backend.BACKEND


@pytest.fixture(autouse=True)
def _loader_present(request, monkeypatch):
    """`ensure` checks for `safetensors` first; the CI environment has none, and only the test
    below is about that check."""
    if "missing_loader" not in request.node.name:
        monkeypatch.setattr(backend, "_require_loader", lambda: None)


def test_a_missing_loader_costs_the_labels_not_the_run(monkeypatch):
    """`safetensors` arrives only through `transformers` in `[asr]`. Without it `load` would raise a
    bare ImportError after the ASR, which the stage does not catch."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("safetensors"):
            raise ImportError("no safetensors")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(diarization.DiarizationUnavailable, match=r"\[asr\]"):
        backend.ensure(None, "cpu")


def _complete(root, device):
    for name in backend.required_files(device):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return root


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_a_local_directory_wins_and_picks_the_voiceprint_build_for_the_device(tmp_path, monkeypatch, device):
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    handle = backend.ensure(_complete(tmp_path, device), device)
    assert handle.model_dir == tmp_path
    assert handle.embedding == tmp_path / backend.embedding_file(device)
    assert handle.plda_dir == tmp_path / "plda"


@pytest.mark.parametrize("missing", backend.required_files("cuda"))
def test_a_local_directory_missing_any_file_is_not_used(tmp_path, monkeypatch, missing):
    """⚠ Every file the run opens is checked first: a gap found later is a bare torch or numpy
    exception, which the stage does not catch -- the run would end after the ASR."""
    from finesub.speech.runtime import hf_weights

    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    monkeypatch.setattr(backend, "_already_fetched", lambda: True)
    monkeypatch.setattr(hf_weights, "prepare_directory", lambda entry: None)
    _complete(tmp_path, "cuda")
    (tmp_path / missing).unlink()
    with pytest.raises(diarization.DiarizationUnavailable):
        backend.ensure(tmp_path, "cuda")


def _hosted(monkeypatch, tmp_path, *, fetched):
    from finesub.speech.runtime import hf_weights

    asked = []
    monkeypatch.delenv(ARTIFACT_ENV, raising=False)
    monkeypatch.setattr(backend, "_already_fetched", lambda: fetched)

    def prepare_directory(entry):
        asked.append(entry)
        return tmp_path / entry

    monkeypatch.setattr(hf_weights, "prepare_directory", prepare_directory)
    return asked


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_the_hosted_pieces_come_from_their_two_repositories(tmp_path, monkeypatch, device):
    asked = _hosted(monkeypatch, tmp_path, fetched=True)
    handle = backend.ensure(None, device)
    assert asked == [backend.HOSTED_MODEL, backend.HOSTED_EMBEDDING]
    assert handle.model_dir == tmp_path / backend.HOSTED_MODEL
    assert handle.embedding == tmp_path / backend.HOSTED_EMBEDDING / backend.embedding_file(device)


def test_the_download_is_announced_only_when_one_is_needed(tmp_path, monkeypatch):
    from finesub import reporting

    said = []

    class Recorder(reporting.NullReporter):
        def warning(self, code, message, *, impact="", action=""):
            said.append(code)

    monkeypatch.setattr(reporting, "current_reporter", lambda: Recorder())
    _hosted(monkeypatch, tmp_path, fetched=False)
    backend.ensure(None, "cuda")
    assert said == ["nemotron3-download"]
    said.clear()
    _hosted(monkeypatch, tmp_path, fetched=True)
    backend.ensure(None, "cuda")
    assert said == []


def test_a_corrupt_or_failed_download_costs_the_labels_not_the_run(tmp_path, monkeypatch):
    from finesub.speech.runtime import hf_weights
    from finesub_bootstrap.hf_verify import VerificationMismatch

    _hosted(monkeypatch, tmp_path, fetched=True)

    def corrupt(entry):
        raise VerificationMismatch("sha256 differs")

    monkeypatch.setattr(hf_weights, "prepare_directory", corrupt)
    with pytest.raises(diarization.DiarizationUnavailable, match="failed verification"):
        backend.ensure(None, "cuda")

    monkeypatch.setattr(hf_weights, "prepare_directory", lambda entry: None)
    with pytest.raises(diarization.DiarizationUnavailable) as error:
        backend.ensure(None, "cuda")
    assert ARTIFACT_ENV in str(error.value)


def test_the_manifest_fetches_only_the_two_files_the_port_opens():
    """NVIDIA's repository also carries a .nemo, a GGUF and demo media -- about 300 MB never read."""
    from finesub_bootstrap.model_manifest import entry_for

    entry = entry_for(backend.HOSTED_MODEL)
    assert entry.repo == backend.MODEL_NAME and entry.listed_only
    assert {file.name for file in entry.files} == set(backend.MODEL_FILES)
