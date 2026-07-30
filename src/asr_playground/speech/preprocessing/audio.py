"""Audio utility helpers for the ASR pipeline."""

from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
import wave

import numpy as np
import torch
import torch.nn.functional as F

try:
    import numba as nb
except Exception:
    nb = None

try:
    import torchaudio
    import torchaudio.functional as AF
except Exception as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError(f"torchaudio is required for audio utilities: {exc}") from exc

try:
    import soundfile as sf
except Exception:
    sf = None


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


def _load_with_soundfile(path: str) -> Tuple[torch.Tensor, int]:
    if sf is None:
        raise RuntimeError("soundfile is unavailable")
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.ascontiguousarray(data.T))
    return waveform, int(sr)


def get_audio_info_stream(path: str) -> Tuple[int, int]:
    errs: List[str] = []
    if sf is not None:
        try:
            info = sf.info(path)
            sr = int(info.samplerate)
            num_frames = int(info.frames)
            if sr > 0 and num_frames >= 0:
                return sr, num_frames
        except Exception as exc:
            errs.append(f"soundfile.info: {type(exc).__name__}: {exc}")

    if hasattr(torchaudio, "info"):
        try:
            info = torchaudio.info(path)
            sr = int(info.sample_rate)
            num_frames = int(info.num_frames or 0)
            if sr > 0 and num_frames > 0:
                return sr, num_frames
        except Exception as exc:
            errs.append(f"torchaudio.info: {type(exc).__name__}: {exc}")

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

    if sf is not None:
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

    try:
        waveform, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
        return waveform, int(sr)
    except Exception as exc:
        errs.append(f"torchaudio.load(slice): {type(exc).__name__}: {exc}")

    raise RuntimeError("Unable to stream audio slice. " + " | ".join(errs))


def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    errs: List[str] = []
    if sf is not None:
        try:
            return _load_with_soundfile(path)
        except Exception as exc:
            errs.append(f"soundfile.read: {type(exc).__name__}: {exc}")

    try:
        waveform, sr = torchaudio.load(path)
        return waveform, int(sr)
    except Exception as exc:
        errs.append(f"torchaudio.load: {type(exc).__name__}: {exc}")

    if hasattr(torchaudio, "load_with_torchcodec"):
        try:
            waveform, sr = torchaudio.load_with_torchcodec(path)
            return waveform, int(sr)
        except Exception as exc:
            errs.append(f"torchaudio.load_with_torchcodec: {type(exc).__name__}: {exc}")

    raise RuntimeError("Unable to load audio. " + " | ".join(errs))


def get_audio_info(path: str) -> Tuple[int, int]:
    return get_audio_info_stream(path)


def get_audio_duration_sec(path: str) -> float:
    sr, num_frames = get_audio_info(path)
    if sr <= 0:
        raise RuntimeError(f"Invalid sample rate for audio: {path}")
    return num_frames / float(sr)


def load_audio_slice(path: str, frame_offset: int, num_frames: int) -> Tuple[torch.Tensor, int]:
    return load_audio_slice_stream(path, frame_offset, num_frames)


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), waveform, sample_rate)


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


def _pad_for_framing(
    waveform: torch.Tensor,
    frame_len: int,
    hop_len: int,
) -> Tuple[torch.Tensor, int]:
    x = waveform
    if x.numel() < frame_len:
        x = F.pad(x, (0, frame_len - x.numel()))

    n_frames = 1 + int((x.numel() - frame_len) // hop_len)
    tail = x.numel() - (n_frames - 1) * hop_len - frame_len
    if tail < 0:
        tail = 0
    if tail > 0:
        x = F.pad(x, (0, hop_len - tail))
        n_frames = 1 + int((x.numel() - frame_len) // hop_len)
    return x, n_frames


def _build_vocal_filterbank(
    sample_rate: int,
    frame_len: int,
    *,
    num_bands: int,
    vocal_prior_min_hz: float,
    vocal_prior_max_hz: float,
    vocal_prior_floor: float,
    vocal_prior_low_hz: float,
    vocal_prior_high_hz: float,
    vocal_prior_log_k_low: float,
    vocal_prior_log_k_high: float,
    db_eps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_freq = (frame_len // 2) + 1
    nyq = float(sample_rate) / 2.0
    freqs = torch.linspace(0.0, nyq, steps=n_freq, device=device, dtype=dtype)

    fmin = max(20.0, min(float(vocal_prior_min_hz), nyq - 1.0))
    fmax = max(fmin + 1.0, min(float(vocal_prior_max_hz), nyq - 1.0))
    band_count = max(4, int(num_bands))
    edges = torch.logspace(
        torch.log10(torch.tensor(fmin, device=device, dtype=dtype)),
        torch.log10(torch.tensor(fmax, device=device, dtype=dtype)),
        steps=band_count + 2,
        device=device,
        dtype=dtype,
    )

    fb = torch.zeros(n_freq, band_count, device=device, dtype=dtype)
    for i in range(band_count):
        left = edges[i]
        center = edges[i + 1]
        right = edges[i + 2]
        up = (freqs - left) / torch.clamp(center - left, min=db_eps)
        down = (right - freqs) / torch.clamp(right - center, min=db_eps)
        tri = torch.minimum(up, down)
        fb[:, i] = torch.clamp(tri, min=0.0)

    fb_sum = torch.clamp(fb.sum(dim=0, keepdim=True), min=db_eps)
    fb = fb / fb_sum

    centers = edges[1:-1]
    low_hz = max(20.0, min(float(vocal_prior_low_hz), nyq - 1.0))
    high_hz = max(low_hz + 1.0, min(float(vocal_prior_high_hz), nyq - 1.0))
    log_centers = torch.log(torch.clamp(centers, min=20.0))
    log_low = torch.log(torch.tensor(low_hz, device=device, dtype=dtype))
    log_high = torch.log(torch.tensor(high_hz, device=device, dtype=dtype))
    k_low = max(1e-3, float(vocal_prior_log_k_low))
    k_high = max(1e-3, float(vocal_prior_log_k_high))

    left = torch.sigmoid((log_centers - log_low) / k_low)
    right = torch.sigmoid((log_high - log_centers) / k_high)
    prior = left * right
    prior = float(vocal_prior_floor) + (1.0 - float(vocal_prior_floor)) * prior
    prior = torch.clamp(prior, min=float(vocal_prior_floor), max=1.0)
    return fb, prior


def _band_power_from_frame_range(
    waveform: torch.Tensor,
    *,
    frame_start: int,
    frame_end: int,
    frame_len: int,
    hop_len: int,
    window: torch.Tensor,
    fbank: torch.Tensor,
    db_eps: float,
) -> torch.Tensor:
    sample_start = int(frame_start) * hop_len
    sample_end = (int(frame_end) - 1) * hop_len + frame_len
    chunk = waveform[sample_start:sample_end]
    frames = chunk.unfold(0, frame_len, hop_len)
    spec = torch.fft.rfft(frames * window, n=frame_len, dim=1)
    power = spec.real.square() + spec.imag.square()
    band_power = torch.matmul(power, fbank)
    return torch.clamp(band_power, min=db_eps)


def _adaptive_weighted_energy_torch(
    band_power_2d: torch.Tensor,
    *,
    n_frames: int,
    prior: torch.Tensor,
    init_count: int,
    prior_lambda: float,
    occ_alpha: float,
    occ_snr_db: float,
    spectral_snr_keep_db: float,
    spectral_snr_soft_db: float,
    vocal_prior_floor: float,
    spectral_weight_min: float,
    spectral_noise_gate_db: float,
    spectral_noise_alpha_quiet: float,
    spectral_noise_alpha_loud: float,
    noise_q: float,
    db_eps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty(n_frames, dtype=dtype, device=device)
    if n_frames <= 0 or band_power_2d.numel() == 0:
        return out

    init_n = max(1, min(int(init_count), int(band_power_2d.shape[0])))
    noise = torch.quantile(band_power_2d[:init_n], noise_q, dim=0)
    noise = torch.clamp(noise, min=db_eps)
    occ = torch.zeros_like(prior)
    occ_norm = torch.ones_like(prior)
    soft_den = max(float(spectral_snr_soft_db), 1e-6)
    prior_floor = float(vocal_prior_floor)
    w_min = float(spectral_weight_min)

    idx = 0
    for j in range(int(band_power_2d.shape[0])):
        p = band_power_2d[j]
        snr_db = 10.0 * torch.log10(torch.clamp(p / noise, min=db_eps))
        wa = torch.sigmoid((snr_db - float(spectral_snr_keep_db)) / soft_den)

        occ = (1.0 - occ_alpha) * occ + occ_alpha * (snr_db > occ_snr_db).to(dtype=occ.dtype)
        occ_min = torch.min(occ)
        occ_max = torch.max(occ)
        if float((occ_max - occ_min).item()) > 1e-6:
            occ_norm = (occ - occ_min) / (occ_max - occ_min)
        else:
            occ_norm.fill_(1.0)

        prior_adapt = torch.clamp(
            (1.0 - prior_lambda) * prior + prior_lambda * occ_norm,
            min=prior_floor,
            max=1.0,
        )
        w = torch.clamp(prior_adapt * wa, min=w_min, max=1.0)
        e = torch.sum(w * p) / torch.clamp(torch.sum(w), min=db_eps)
        out[idx] = 10.0 * torch.log10(torch.clamp(e, min=db_eps))

        alpha = torch.where(
            snr_db <= float(spectral_noise_gate_db),
            float(spectral_noise_alpha_quiet),
            float(spectral_noise_alpha_loud),
        )
        noise = noise + alpha * (p - noise)
        idx += 1
    return out


if nb is not None:

    @nb.njit(cache=True, fastmath=True)
    def _adaptive_weighted_energy_numba(
        band_power: np.ndarray,
        prior: np.ndarray,
        init_count: int,
        prior_lambda: float,
        occ_alpha: float,
        occ_snr_db: float,
        spectral_snr_keep_db: float,
        spectral_snr_soft_db: float,
        vocal_prior_floor: float,
        spectral_weight_min: float,
        spectral_noise_gate_db: float,
        spectral_noise_alpha_quiet: float,
        spectral_noise_alpha_loud: float,
        noise_q: float,
        db_eps: float,
    ) -> np.ndarray:
        n_frames, n_bands = band_power.shape
        out = np.empty(n_frames, dtype=np.float32)
        if n_frames == 0:
            return out

        init_n = init_count
        if init_n <= 0:
            init_n = n_frames
        if init_n > n_frames:
            init_n = n_frames

        noise = np.empty(n_bands, dtype=np.float32)
        q = noise_q
        if q < 0.0:
            q = 0.0
        if q > 1.0:
            q = 1.0

        for b in range(n_bands):
            col = np.sort(band_power[:init_n, b].copy())
            pos = q * float(init_n - 1)
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            if lo < 0:
                lo = 0
            if hi >= init_n:
                hi = init_n - 1
            w = pos - float(lo)
            v = col[lo] * (1.0 - w) + col[hi] * w
            if v < db_eps:
                v = db_eps
            noise[b] = np.float32(v)

        occ = np.zeros(n_bands, dtype=np.float32)
        occ_norm = np.ones(n_bands, dtype=np.float32)
        snr_db = np.empty(n_bands, dtype=np.float32)
        wa = np.empty(n_bands, dtype=np.float32)
        prior_adapt = np.empty(n_bands, dtype=np.float32)
        soft_den = spectral_snr_soft_db if spectral_snr_soft_db > 1e-6 else 1e-6

        for i in range(n_frames):
            for b in range(n_bands):
                ratio = band_power[i, b] / noise[b]
                if ratio < db_eps:
                    ratio = db_eps
                s = 10.0 * np.log10(ratio)
                snr_db[b] = np.float32(s)
                wa[b] = np.float32(1.0 / (1.0 + np.exp(-(s - spectral_snr_keep_db) / soft_den)))
                occ[b] = np.float32(
                    (1.0 - occ_alpha) * occ[b]
                    + occ_alpha * (1.0 if s > occ_snr_db else 0.0)
                )

            occ_min = occ[0]
            occ_max = occ[0]
            for b in range(1, n_bands):
                if occ[b] < occ_min:
                    occ_min = occ[b]
                if occ[b] > occ_max:
                    occ_max = occ[b]

            if (occ_max - occ_min) > 1e-6:
                den = occ_max - occ_min
                for b in range(n_bands):
                    occ_norm[b] = (occ[b] - occ_min) / den
            else:
                for b in range(n_bands):
                    occ_norm[b] = 1.0

            num = 0.0
            den_w = 0.0
            for b in range(n_bands):
                pa = (1.0 - prior_lambda) * prior[b] + prior_lambda * occ_norm[b]
                if pa < vocal_prior_floor:
                    pa = vocal_prior_floor
                if pa > 1.0:
                    pa = 1.0
                prior_adapt[b] = np.float32(pa)
                w_band = pa * wa[b]
                if w_band < spectral_weight_min:
                    w_band = spectral_weight_min
                if w_band > 1.0:
                    w_band = 1.0
                num += w_band * band_power[i, b]
                den_w += w_band

            if den_w < db_eps:
                den_w = db_eps
            e = num / den_w
            if e < db_eps:
                e = db_eps
            out[i] = np.float32(10.0 * np.log10(e))

            for b in range(n_bands):
                alpha = spectral_noise_alpha_quiet
                if snr_db[b] > spectral_noise_gate_db:
                    alpha = spectral_noise_alpha_loud
                noise[b] = np.float32(noise[b] + alpha * (band_power[i, b] - noise[b]))

        return out


def compute_band_power_chunks(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    frame_len: int,
    hop_len: int,
    num_bands: int = 24,
    chunk_frames: int = 4096,
    vocal_prior_min_hz: float = 120.0,
    vocal_prior_max_hz: float = 4000.0,
    vocal_prior_floor: float = 0.15,
    vocal_prior_low_hz: float = 120.0,
    vocal_prior_high_hz: float = 4200.0,
    vocal_prior_log_k_low: float = 0.18,
    vocal_prior_log_k_high: float = 0.20,
    db_eps: float = 1e-10,
    workers: int = 1,
) -> Tuple[List[torch.Tensor], int, Optional[torch.Tensor]]:
    """Stateless front half of weighted_spectral_energy_db: framed band power.

    Returns (band_chunks, n_frames, prior). Every value is per-frame local
    (windowed rfft x filterbank, no cross-frame state), so how the frames are
    split into chunks/blocks cannot change them — the streamed VAD relies on
    this to reproduce whole-file results bit-exactly."""

    waveform, n_frames = _pad_for_framing(waveform, frame_len, hop_len)
    if n_frames <= 0:
        return [], 0, None

    window = torch.hann_window(
        frame_len,
        periodic=True,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    fbank, prior = _build_vocal_filterbank(
        sample_rate,
        frame_len,
        num_bands=num_bands,
        vocal_prior_min_hz=vocal_prior_min_hz,
        vocal_prior_max_hz=vocal_prior_max_hz,
        vocal_prior_floor=vocal_prior_floor,
        vocal_prior_low_hz=vocal_prior_low_hz,
        vocal_prior_high_hz=vocal_prior_high_hz,
        vocal_prior_log_k_low=vocal_prior_log_k_low,
        vocal_prior_log_k_high=vocal_prior_log_k_high,
        db_eps=db_eps,
        device=waveform.device,
        dtype=waveform.dtype,
    )

    chunk_n = max(1, int(chunk_frames))
    ranges = [
        (frame_start, min(n_frames, frame_start + chunk_n))
        for frame_start in range(0, n_frames, chunk_n)
    ]

    band_chunks: List[torch.Tensor] = [
        torch.empty(0, dtype=waveform.dtype, device=waveform.device)
        for _ in ranges
    ]

    def _compute_one(idx: int, frame_start: int, frame_end: int) -> Tuple[int, torch.Tensor]:
        chunk_power = _band_power_from_frame_range(
            waveform,
            frame_start=frame_start,
            frame_end=frame_end,
            frame_len=frame_len,
            hop_len=hop_len,
            window=window,
            fbank=fbank,
            db_eps=db_eps,
        )
        return idx, chunk_power

    max_workers = max(1, min(int(workers), len(ranges)))
    if max_workers > 1 and len(ranges) > 1:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_compute_one, idx, frame_start, frame_end)
                for idx, (frame_start, frame_end) in enumerate(ranges)
            ]
            for fut in cf.as_completed(futures):
                idx, chunk_power = fut.result()
                band_chunks[idx] = chunk_power
    else:
        for idx, (frame_start, frame_end) in enumerate(ranges):
            _, chunk_power = _compute_one(idx, frame_start, frame_end)
            band_chunks[idx] = chunk_power

    return band_chunks, n_frames, prior


def adaptive_weighted_energy(
    band_power_2d: torch.Tensor,
    *,
    prior: torch.Tensor,
    init_count: int,
    vocal_prior_adapt_lambda: float = 0.45,
    vocal_prior_occ_alpha: float = 0.02,
    vocal_prior_occ_snr_db: float = 2.0,
    vocal_prior_floor: float = 0.15,
    spectral_weight_min: float = 0.05,
    spectral_snr_keep_db: float = 3.0,
    spectral_snr_soft_db: float = 2.0,
    noise_init_percentile: float = 5.0,
    spectral_noise_gate_db: float = 6.0,
    spectral_noise_alpha_quiet: float = 0.08,
    spectral_noise_alpha_loud: float = 0.005,
    db_eps: float = 1e-10,
) -> torch.Tensor:
    """Stateful back half of weighted_spectral_energy_db: the causal adaptive
    spectral tracker over a full band-power track ([n_frames x n_bands]).

    `init_count` is how many leading frames seed the noise percentile — the
    whole-file path uses its first chunk (min(chunk_frames, n_frames)); a
    streamed caller must pass the same value to stay bit-identical."""

    n_frames = int(band_power_2d.shape[0])
    prior_lambda = max(0.0, min(1.0, float(vocal_prior_adapt_lambda)))
    occ_alpha = max(0.0, min(1.0, float(vocal_prior_occ_alpha)))
    occ_snr_db = float(vocal_prior_occ_snr_db)
    noise_q = max(0.0, min(1.0, float(noise_init_percentile) / 100.0))

    use_numba = (
        nb is not None
        and band_power_2d.device.type == "cpu"
        and band_power_2d.dtype in {torch.float32, torch.float64}
    )
    if use_numba:
        bp_np = band_power_2d.detach().to(dtype=torch.float32, device="cpu").numpy()
        bp_np = np.ascontiguousarray(bp_np)
        prior_np = prior.detach().to(dtype=torch.float32, device="cpu").numpy()
        out_np = _adaptive_weighted_energy_numba(
            bp_np,
            prior_np,
            int(init_count),
            float(prior_lambda),
            float(occ_alpha),
            float(occ_snr_db),
            float(spectral_snr_keep_db),
            float(spectral_snr_soft_db),
            float(vocal_prior_floor),
            float(spectral_weight_min),
            float(spectral_noise_gate_db),
            float(spectral_noise_alpha_quiet),
            float(spectral_noise_alpha_loud),
            float(noise_q),
            float(db_eps),
        )
        out = torch.from_numpy(out_np)
        return out.to(dtype=band_power_2d.dtype, device=band_power_2d.device)

    return _adaptive_weighted_energy_torch(
        band_power_2d,
        n_frames=n_frames,
        prior=prior,
        init_count=int(init_count),
        prior_lambda=prior_lambda,
        occ_alpha=occ_alpha,
        occ_snr_db=occ_snr_db,
        spectral_snr_keep_db=float(spectral_snr_keep_db),
        spectral_snr_soft_db=float(spectral_snr_soft_db),
        vocal_prior_floor=float(vocal_prior_floor),
        spectral_weight_min=float(spectral_weight_min),
        spectral_noise_gate_db=float(spectral_noise_gate_db),
        spectral_noise_alpha_quiet=float(spectral_noise_alpha_quiet),
        spectral_noise_alpha_loud=float(spectral_noise_alpha_loud),
        noise_q=noise_q,
        db_eps=float(db_eps),
        dtype=band_power_2d.dtype,
        device=band_power_2d.device,
    )


def weighted_spectral_energy_db(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    frame_len: int,
    hop_len: int,
    num_bands: int = 24,
    chunk_frames: int = 4096,
    vocal_prior_min_hz: float = 120.0,
    vocal_prior_max_hz: float = 4000.0,
    vocal_prior_floor: float = 0.15,
    vocal_prior_low_hz: float = 120.0,
    vocal_prior_high_hz: float = 4200.0,
    vocal_prior_log_k_low: float = 0.18,
    vocal_prior_log_k_high: float = 0.20,
    vocal_prior_adapt_lambda: float = 0.45,
    vocal_prior_occ_alpha: float = 0.02,
    vocal_prior_occ_snr_db: float = 2.0,
    spectral_weight_min: float = 0.05,
    spectral_snr_keep_db: float = 3.0,
    spectral_snr_soft_db: float = 2.0,
    noise_init_percentile: float = 5.0,
    spectral_noise_gate_db: float = 6.0,
    spectral_noise_alpha_quiet: float = 0.08,
    spectral_noise_alpha_loud: float = 0.005,
    db_eps: float = 1e-10,
    workers: int = 1,
) -> torch.Tensor:
    band_chunks, n_frames, prior = compute_band_power_chunks(
        waveform,
        sample_rate=sample_rate,
        frame_len=frame_len,
        hop_len=hop_len,
        num_bands=num_bands,
        chunk_frames=chunk_frames,
        vocal_prior_min_hz=vocal_prior_min_hz,
        vocal_prior_max_hz=vocal_prior_max_hz,
        vocal_prior_floor=vocal_prior_floor,
        vocal_prior_low_hz=vocal_prior_low_hz,
        vocal_prior_high_hz=vocal_prior_high_hz,
        vocal_prior_log_k_low=vocal_prior_log_k_low,
        vocal_prior_log_k_high=vocal_prior_log_k_high,
        db_eps=db_eps,
        workers=workers,
    )
    if n_frames <= 0 or prior is None:
        return torch.zeros(0, dtype=waveform.dtype, device=waveform.device)

    band_power = torch.cat([c for c in band_chunks if c.shape[0] > 0], dim=0)
    init_count = int(band_chunks[0].shape[0]) if band_chunks else n_frames
    return adaptive_weighted_energy(
        band_power,
        prior=prior,
        init_count=init_count,
        vocal_prior_adapt_lambda=vocal_prior_adapt_lambda,
        vocal_prior_occ_alpha=vocal_prior_occ_alpha,
        vocal_prior_occ_snr_db=vocal_prior_occ_snr_db,
        vocal_prior_floor=vocal_prior_floor,
        spectral_weight_min=spectral_weight_min,
        spectral_snr_keep_db=spectral_snr_keep_db,
        spectral_snr_soft_db=spectral_snr_soft_db,
        noise_init_percentile=noise_init_percentile,
        spectral_noise_gate_db=spectral_noise_gate_db,
        spectral_noise_alpha_quiet=spectral_noise_alpha_quiet,
        spectral_noise_alpha_loud=spectral_noise_alpha_loud,
        db_eps=db_eps,
    )
