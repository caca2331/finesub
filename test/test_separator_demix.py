"""The in-memory demix runner's contract with audio-separator's own maths.

The runner replaced a file-to-file call into audio-separator, so the parts of
``prepare_mix``/``demix`` that had been doing work invisibly are now ours to get
right. These are the ones that are not obvious from the happy path.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from finesub.speech.preprocessing.separator import demix


class _EchoModel(torch.nn.Module):
    """Stands in for the Roformer: returns its input, asserting stereo like it does."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.seen_channels: list[int] = []

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.dim() == 3, audio.shape
        self.seen_channels.append(audio.shape[1])
        if audio.shape[1] != 2:
            raise AssertionError(
                "stereo needs to be set to True if passing in audio signal that "
                "is stereo (channel dimension of 2)"
            )
        return audio


def _model_instance(model: torch.nn.Module, *, rate: int = 44100, chunk_hop: int = 441):
    return SimpleNamespace(
        model_run=model,
        sample_rate=rate,
        overlap=8,
        pitch_shift=0,
        segment_size=256,
        override_model_segment_size=False,
        normalization_threshold=0.9,
        amplification_threshold=0.0,
        model_data_cfgdict=SimpleNamespace(
            audio=SimpleNamespace(sample_rate=rate, hop_length=chunk_hop),
            model=SimpleNamespace(stft_hop_length=chunk_hop),
            inference=SimpleNamespace(dim_t=801),
            training=SimpleNamespace(
                target_instrument="Vocals",
                instruments=["Vocals", "Instrumental"],
            ),
        ),
    )


def test_mono_source_is_duplicated_to_the_stereo_the_model_demands() -> None:
    """``load_audio_slice`` always returns 2-D, so ``ndim == 1`` never fires.

    audio-separator got its mono-to-stereo conversion for free: it read the
    block back with ``librosa.load``, which hands a mono file over as a 1-D
    array. Reading the waveform directly does not, and a mono source then
    reaches a checkpoint trained with ``stereo: true``.
    """

    model = _EchoModel()
    frames = 44100 * 9
    waveform = torch.zeros(1, frames, dtype=torch.float32)
    waveform[0, ::100] = 0.4

    stem, rate = demix.separate_waveform(
        _model_instance(model), waveform, 44100, use_autocast=False
    )

    assert rate == 44100
    assert model.seen_channels and set(model.seen_channels) == {2}
    assert stem.shape == (2, frames)
    np.testing.assert_allclose(stem[0], stem[1])


def test_output_keeps_the_source_length_and_channel_count() -> None:
    model = _EchoModel()
    frames = 44100 * 9
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(2, frames, generator=generator) * 0.05

    stem, _ = demix.separate_waveform(
        _model_instance(model), waveform, 44100, use_autocast=False
    )

    assert stem.shape == (2, frames)
    # An echo model plus the window/counter pair is the identity up to float
    # rounding, which is what proves the overlap-add is reassembling in place.
    np.testing.assert_allclose(stem, waveform.numpy(), atol=1e-6)


def test_the_callers_waveform_is_not_normalized_in_place() -> None:
    """``spec_utils.normalize`` scales in place; the block tensor is not ours."""

    model = _EchoModel()
    waveform = torch.full((2, 44100 * 9), 0.99, dtype=torch.float32)
    before = waveform.clone()

    demix.separate_waveform(
        _model_instance(model), waveform, 44100, use_autocast=False
    )

    assert torch.equal(waveform, before)


@pytest.mark.parametrize("requested", [True, False])
def test_autocast_reaches_the_model_because_nothing_else_applies_it(
    requested: bool,
) -> None:
    """The wrapper this runner replaced is where AMP used to be turned on.

    ``Separator.separate`` wraps the whole call in ``autocast``; calling
    ``model_run`` directly does not inherit that. Losing it ran the model in
    FP32 against packages compiled under autocast, which measured 12 dB of
    per-second SNR against the AMP path on loud audio -- a wrong answer that
    still sounds like music, so nothing downstream would have flagged it.
    """

    seen: list[bool] = []

    class _Recorder(_EchoModel):
        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            seen.append(torch.is_autocast_enabled("cpu"))
            return super().forward(audio)

    demix.separate_waveform(
        _model_instance(_Recorder()),
        torch.zeros(2, 44100 * 9, dtype=torch.float32),
        44100,
        use_autocast=requested,
    )

    assert seen and set(seen) == {requested}


def test_a_block_shorter_than_one_chunk_is_padded_not_written_backwards() -> None:
    """audio-separator's last-chunk branch goes negative here.

    Below one chunk it computes ``start = frames - chunk``, and a negative start
    slices from the far end of the buffer, so the block lands in the wrong
    place with no error. Cannot happen on a planned block -- the worker ladder
    floors a core at 150s -- but a short file goes through whole.
    """

    model = _EchoModel()
    frames = 44100  # one second, against a 2.55s chunk for short audio
    generator = torch.Generator().manual_seed(0)
    waveform = torch.randn(2, frames, generator=generator) * 0.05

    stem, _ = demix.separate_waveform(
        _model_instance(model), waveform, 44100, use_autocast=False
    )

    assert stem.shape == (2, frames)
    np.testing.assert_allclose(stem, waveform.numpy(), atol=1e-6)


def test_a_multi_target_checkpoint_is_refused_rather_than_mis_run() -> None:
    instance = _model_instance(_EchoModel())
    instance.model_data_cfgdict.training.target_instrument = None

    with pytest.raises(RuntimeError, match="single-target"):
        demix.separate_waveform(
            instance, torch.zeros(2, 44100), 44100, use_autocast=False
        )
