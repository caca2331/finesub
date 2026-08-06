"""Frame-level acoustic tracks over the separated vocal audio.

Everything is on a fixed 10 ms hop so tracks can be indexed by time directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

SR = 16000
HOP = 160          # 10 ms
WIN = 400          # 25 ms
N_FFT = 512


@dataclass
class Tracks:
    times: np.ndarray        # frame centre times
    rms_db: np.ndarray       # broadband energy, dBFS
    lo_db: np.ndarray        # 80-1000 Hz (voiced energy)
    hi_db: np.ndarray        # 2000-7000 Hz (fricative / burst energy)
    flux: np.ndarray         # half-wave rectified spectral flux (log-mel)
    onset_env: np.ndarray    # librosa onset strength
    hi_ratio: np.ndarray     # hi_db - lo_db

    def idx(self, t: float) -> int:
        return int(np.clip(round(t * SR / HOP), 0, len(self.times) - 1))

    def slice_idx(self, t0: float, t1: float) -> Tuple[int, int]:
        return self.idx(t0), max(self.idx(t0) + 1, self.idx(t1) + 1)


def compute_tracks(audio_path: Path, cache: Path | None = None) -> Tracks:
    if cache is not None and cache.exists():
        z = np.load(cache)
        return Tracks(**{k: z[k] for k in z.files})

    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN, center=True))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    power = S ** 2

    def band_db(lo: float, hi: float) -> np.ndarray:
        m = (freqs >= lo) & (freqs < hi)
        return 10.0 * np.log10(power[m].sum(axis=0) + 1e-12)

    rms_db = 10.0 * np.log10(power.sum(axis=0) + 1e-12)
    lo_db = band_db(80, 1000)
    hi_db = band_db(2000, 7000)

    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_mels=64, fmax=8000)
    logmel = librosa.power_to_db(mel)
    d = np.diff(logmel, axis=1, prepend=logmel[:, :1])
    flux = np.maximum(d, 0.0).sum(axis=0)

    onset_env = librosa.onset.onset_strength(S=logmel, sr=SR, hop_length=HOP)

    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=SR, hop_length=HOP)
    tracks = Tracks(
        times=times,
        rms_db=rms_db,
        lo_db=lo_db,
        hi_db=hi_db,
        flux=flux,
        onset_env=onset_env,
        hi_ratio=hi_db - lo_db,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **tracks.__dict__)
    return tracks


def noise_floor_db(rms_db: np.ndarray, t_idx: int, half_width: int = 500) -> float:
    """Local 10th-percentile energy, as a stand-in for the noise floor."""
    lo = max(0, t_idx - half_width)
    hi = min(len(rms_db), t_idx + half_width)
    return float(np.quantile(rms_db[lo:hi], 0.10))
