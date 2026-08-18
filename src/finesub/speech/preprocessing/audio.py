"""Getting audio off disk in a form the readers can use.

Decoding, slicing, resampling and channel folding -- everything up to a
waveform tensor. What is then *done* with that waveform to get a loudness
curve lives in `spectral.py`; the two halves shared a file and nothing else.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
import wave

import numpy as np
import torch

try:
    import numba as nb
except Exception:
    nb = None


try:
    # Resampling and filtering only. Decoding is soundfile's job -- see
    # ensure_decodable_input for why torchaudio is kept out of that path.
    import torchaudio.functional as AF
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(f"torchaudio is required for audio utilities: {exc}") from exc

# Hard dependency: decoding is soundfile-only by design, so a missing one has
# to fail at import with its own message. Degrading to None only moved the
# failure into ensure_decodable_input, where it read as "this container needs
# ffmpeg" and then failed again, less legibly, in the reader.
import soundfile as sf


# --------- Shared constants ---------
TARGET_SR = 16000
BANDPASS_LOW_HZ = 70.0
BANDPASS_HIGH_HZ = 8000.0


def apply_bandpass(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply a bandpass filter (70 Hz – 8 kHz) to improve VAD/ASR input quality."""
    if sample_rate <= 2 * BANDPASS_LOW_HZ:
        return waveform
    out = AF.highpass_biquad(waveform, sample_rate, BANDPASS_LOW_HZ)
    nyq = sample_rate / 2.0
    if BANDPASS_HIGH_HZ < nyq - 1.0:
        out = AF.lowpass_biquad(out, sample_rate, BANDPASS_HIGH_HZ)
    return out


def is_wav_path(path: str | Path) -> bool:
    return str(path).strip().lower().endswith((".wav", ".wave"))


def _read_wav_info(path: str | Path) -> Tuple[int, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        frames = int(wf.getnframes())
    return sr, frames


def _load_wav_slice(path: str | Path, frame_offset: int, num_frames: int) -> Tuple[torch.Tensor, int]:
    frame_offset = max(0, int(frame_offset))
    num_frames = max(0, int(num_frames))
    if num_frames <= 0:
        return torch.zeros(1, 0, dtype=torch.float32), 0

    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        total = int(wf.getnframes())
        if frame_offset >= total:
            return torch.zeros(channels, 0, dtype=torch.float32), sr
        read_frames = min(num_frames, total - frame_offset)
        wf.setpos(frame_offset)
        raw = wf.readframes(read_frames)

    if not raw:
        return torch.zeros(channels, 0, dtype=torch.float32), sr

    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        arr = (arr - 128.0) / 128.0
    elif sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        u8 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (
            u8[:, 0].astype(np.int32)
            | (u8[:, 1].astype(np.int32) << 8)
            | (u8[:, 2].astype(np.int32) << 16)
        )
        sign = 1 << 23
        i32 = (i32 ^ sign) - sign
        arr = i32.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / float(1 << 31)
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if channels > 1:
        arr = arr.reshape(-1, channels).T
    else:
        arr = arr.reshape(1, -1)
    return torch.from_numpy(arr.copy()), sr


def ensure_decodable_input(
    input_path: Path,
    workdir: Path,
) -> Tuple[Path, Optional[Path]]:
    """Give the readers below a file soundfile can open.

    They are soundfile-only by design: routing decoding through torchaudio meant
    torchcodec, a shared-FFmpeg install it could actually find, and a torchcodec
    build matching the torch version -- three things to get right on a user's
    machine for a path that one ffmpeg call avoids entirely.

    Video containers are never readable whatever the codec inside, and
    libsndfile's format list varies by build, so probe rather than guess from the
    suffix. The conversion keeps the rate and channel count: it exists to change
    the container, not the signal.

    Returns the path to read plus the temporary file the caller must delete once
    it succeeds -- ``None`` when the input was already usable.
    """

    try:
        sf.info(str(input_path))
        return input_path, None
    except Exception:
        pass

    from ...media.ffmpeg import transcode_to_lossless_audio

    converted = workdir / f"{input_path.stem}-decoded.flac"
    # Left behind by an earlier run: reuse it instead of paying the decode
    # twice. Safe only because the name is claimed by a rename after ffmpeg
    # exits -- a run killed mid-decode leaves the partial file under the
    # temporary name, so a truncated decode can never be picked up here and
    # silently transcribed as the whole input.
    if not converted.exists():
        # Keeps the .flac extension last: ffmpeg picks the muxer from it, so a
        # name ending in .part would fail before decoding anything.
        partial = converted.with_name(f"{converted.stem}.{os.getpid()}.part.flac")
        try:
            transcode_to_lossless_audio(input_path, partial)
            os.replace(partial, converted)
        finally:
            partial.unlink(missing_ok=True)
    return converted, converted


def get_audio_info_stream(path: str) -> Tuple[int, int]:
    errs: List[str] = []
    try:
        info = sf.info(path)
        sr = int(info.samplerate)
        num_frames = int(info.frames)
        if sr > 0 and num_frames >= 0:
            return sr, num_frames
    except Exception as exc:
        errs.append(f"soundfile.info: {type(exc).__name__}: {exc}")

    if is_wav_path(path):
        try:
            return _read_wav_info(path)
        except Exception as exc:
            errs.append(f"wave: {type(exc).__name__}: {exc}")

    raise RuntimeError("Unable to read audio info. " + " | ".join(errs))


def load_audio_slice_stream(path: str, frame_offset: int, num_frames: int) -> Tuple[torch.Tensor, int]:
    if num_frames <= 0:
        return torch.zeros(1, 0, dtype=torch.float32), 0
    frame_offset = max(0, int(frame_offset))
    num_frames = max(0, int(num_frames))
    errs: List[str] = []

    # Direct random-access path for WAV.
    if is_wav_path(path):
        try:
            return _load_wav_slice(path, frame_offset, num_frames)
        except Exception as exc:
            errs.append(f"wave: {type(exc).__name__}: {exc}")

    try:
        data, sr = sf.read(
            path,
            start=frame_offset,
            frames=num_frames,
            dtype="float32",
            always_2d=True,
        )
        waveform = torch.from_numpy(np.ascontiguousarray(data.T))
        return waveform, int(sr)
    except Exception as exc:
        errs.append(f"soundfile.read(slice): {type(exc).__name__}: {exc}")

    raise RuntimeError("Unable to stream audio slice. " + " | ".join(errs))


def get_audio_info(path: str) -> Tuple[int, int]:
    return get_audio_info_stream(path)


def get_audio_duration_sec(path: str) -> float:
    sr, num_frames = get_audio_info(path)
    if sr <= 0:
        raise RuntimeError(f"Invalid sample rate for audio: {path}")
    return num_frames / float(sr)


def load_audio_slice(path: str, frame_offset: int, num_frames: int) -> Tuple[torch.Tensor, int]:
    return load_audio_slice_stream(path, frame_offset, num_frames)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 1:
        return waveform
    if waveform.size(0) == 1:
        return waveform.squeeze(0)
    return waveform.mean(dim=0)


def resample_if_needed(
    waveform: torch.Tensor, sample_rate: int, target_sr: int
) -> Tuple[torch.Tensor, int]:
    if sample_rate == target_sr:
        return waveform, sample_rate
    return AF.resample(waveform, sample_rate, target_sr), target_sr


def as_numpy_float32(waveform: torch.Tensor) -> np.ndarray:
    return waveform.detach().cpu().numpy().astype(np.float32, copy=False)
