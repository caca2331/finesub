"""Run one block through the Roformer without going near a file.

audio-separator's own entry point is file-to-file: it reads the block with
``librosa.load``, demixes it, and writes the stem with pydub, which shells out
to ffmpeg. finesub then reads that file straight back to append it to the merged
track. Around a block that costs a WAV write, a decode, an int16 conversion, a
FLAC encode and a FLAC decode -- and the demix itself sizes its CPU buffers by
``len(training.instruments)`` rather than by the one stem this checkpoint
actually produces, so every overlap-add and the final division run twice over
several hundred megabytes.

None of that is inference. This module keeps the maths and drops the rest: the
waveform finesub already decoded goes in, the separated stem comes out, and the
caller appends it. E13 in ``docs/separator-optimization.md`` has the measurements
and E14 has what it is worth end to end.

**The chunking maths here is audio-separator's, deliberately.** The window,
the counter and the final division are reproduced exactly rather than
simplified: with ``overlap=8`` the step equals the chunk size, so on every chunk
but the last the window cancels itself -- but ``(x*w)/w`` is not bit-identical
to ``x`` in floating point, and the last chunk really is a crossfade against the
one before it. Simplifying either would move the audio for no measurable gain
(600s of overlap-add and division together cost 0.24s).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _normalize(wave: np.ndarray, max_peak: float, min_peak: float | None) -> np.ndarray:
    """``spec_utils.normalize``, in place, on our own array."""

    peak = np.abs(wave).max()
    if peak > max_peak:
        wave *= max_peak / peak
    elif min_peak is not None and peak != 0 and peak < min_peak:
        wave *= min_peak / peak
    return wave


def _chunk_size(model_instance: Any, duration_sec: float) -> int:
    config = model_instance.model_data_cfgdict
    segment = config.inference.dim_t
    if model_instance.override_model_segment_size or duration_sec < 10.0:
        # Its own short-audio branch: below ten seconds it switches to the
        # configured segment size. Reproduced so a tiny input keeps behaving the
        # way it does today rather than the way long inputs do.
        segment = model_instance.segment_size
    hop = getattr(config.model, "stft_hop_length", None) or config.audio.hop_length
    return int(hop) * (int(segment) - 1)


def _to_model_rate(
    mix: np.ndarray,
    source_rate: int,
    model_rate: int,
) -> np.ndarray:
    """What ``librosa.load(..., sr=model_rate)`` would have done to the block.

    ``res_type`` is spelled out because the default is what makes this match:
    audio-separator never asks for a resampler either, so both sides get
    librosa's ``soxr_hq``.
    """

    if source_rate == model_rate:
        return mix
    import librosa

    return librosa.resample(
        mix,
        orig_sr=source_rate,
        target_sr=model_rate,
        res_type="soxr_hq",
    )


def separate_waveform(
    model_instance: Any,
    waveform: torch.Tensor,
    source_rate: int,
    *,
    use_autocast: bool,
) -> tuple[np.ndarray, int]:
    """Separate one block. Returns the stem as ``[channels, frames]`` and its rate.

    ``waveform`` is ``[channels, frames]`` float32 at ``source_rate``, exactly
    what ``load_audio_slice`` hands back.

    ``use_autocast`` is not optional and is not a tuning knob: audio-separator
    applies it in ``Separator.separate``, an outer wrapper this runner replaces,
    so leaving it out silently ran the whole model in FP32. That is not merely
    slower -- the compiled packages are built under autocast, and the mismatch
    showed up as 12 dB of per-second SNR against the AMP path on loud audio.
    Callers pass ``separator.use_autocast``.
    """

    config = model_instance.model_data_cfgdict
    target = config.training.target_instrument
    if not target:
        raise RuntimeError(
            "The demix runner is written for a single-target Roformer; this "
            f"checkpoint declares instruments {list(config.training.instruments)} "
            "and no target_instrument."
        )
    if model_instance.pitch_shift:
        raise RuntimeError("The demix runner does not implement pitch shifting.")

    model_rate = int(model_instance.sample_rate)
    # Copied once, up front: normalization scales in place and the caller's
    # tensor is not ours to touch. Everything downstream then owns its array.
    mix = np.array(waveform.detach().cpu().numpy(), dtype=np.float32)
    if mix.ndim == 1:
        mix = mix[None, :]
    mix = _to_model_rate(mix, source_rate, model_rate)
    if mix.shape[0] == 1:
        # ``prepare_mix``'s mono branch. It is not optional: the model asserts
        # that a stereo-trained checkpoint gets two channels, and
        # ``load_audio_slice`` always returns 2-D, so a mono source arrives as
        # [1, N] rather than as the 1-D array librosa.load would have produced.
        # Duplicated after the resample, which is where librosa.load did it too
        # and which halves the work.
        mix = np.repeat(mix, 2, axis=0)
    mix = np.ascontiguousarray(mix)
    _normalize(
        mix,
        model_instance.normalization_threshold,
        model_instance.amplification_threshold,
    )

    frames = mix.shape[1]
    chunk = _chunk_size(model_instance, frames / model_rate)
    desired_step = int(model_instance.overlap * config.audio.sample_rate)
    step = chunk if desired_step <= 0 else min(desired_step, chunk)

    if frames < chunk:
        # audio-separator does not handle this: its last-chunk branch computes
        # `start = frames - chunk`, and a negative start slices from the far end
        # of the buffer, quietly writing the block into the wrong place. It
        # cannot happen on a planned block (the worker ladder floors a core at
        # 150s) but a short file goes through here whole. Padding also keeps the
        # fixed-shape AOTI runners fed, which a short chunk would not.
        padded = np.zeros((mix.shape[0], chunk), dtype=np.float32)
        padded[:, :frames] = mix
        mix = padded

    model = model_instance.model_run
    device = next(model.parameters()).device
    mix_tensor = torch.from_numpy(mix)

    # `np.hamming` rather than `scipy.signal.windows.hamming`, which is what
    # audio-separator calls: the two are the same formula and differ by ~5e-16
    # in float64, which the float32 cast below erases -- verified identical at
    # this chunk size. Worth the check because it keeps scipy out of a module
    # the light test environment imports.
    window = torch.tensor(np.hamming(chunk), dtype=torch.float32)
    # One stem, not len(instruments): the model returns the target only, and the
    # extra plane in audio-separator's buffers just receives a broadcast copy of
    # it. At a 600s block each plane is 212 MiB.
    channels, run_frames = mix.shape
    result = torch.zeros((channels, run_frames), dtype=torch.float32)
    counter = torch.zeros((channels, run_frames), dtype=torch.float32)

    autocast_enabled = bool(use_autocast) and torch.amp.autocast_mode.is_autocast_available(
        device.type
    )
    with torch.no_grad(), torch.autocast(device.type, enabled=autocast_enabled):
        for offset in range(0, run_frames, step):
            part = mix_tensor[:, offset : offset + chunk]
            length = part.shape[-1]
            if offset + chunk > run_frames:
                part = mix_tensor[:, -chunk:]
                length = chunk
                start = run_frames - chunk
            else:
                start = offset
            separated = model(part.to(device).unsqueeze(0))[0].cpu()
            span = min(length, separated.shape[-1], chunk)
            if span <= 0:
                continue
            result[..., start : start + span] += separated[..., :span] * window[:span]
            counter[..., start : start + span] += window[:span]

    # Sliced back to the caller's length, which only matters when the block was
    # padded above; the merge trims by frame count and must not inherit padding.
    stem = (result / counter.clamp(min=1e-10)).numpy()[:, :frames]
    del result, counter
    return _normalize(
        stem,
        model_instance.normalization_threshold,
        model_instance.amplification_threshold,
    ), model_rate
