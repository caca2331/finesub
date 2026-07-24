"""RMS normalization helpers and CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from utils import load_audio, save_audio


# Target RMS amplitude after normalization (rough loudness proxy in linear scale).
RMS_TARGET = 0.1
# Absolute peak ceiling after gain is applied, to avoid clipping.
RMS_PEAK_LIMIT = 0.99
# Guard threshold to avoid dividing by ~zero on silence/nearly silent audio.
RMS_EPS = 1e-8


def rms_normalize_waveform(
    waveform: torch.Tensor,
    *,
    target_rms: float = RMS_TARGET,
    peak_limit: float = RMS_PEAK_LIMIT,
    eps: float = RMS_EPS,
) -> torch.Tensor:
    if waveform.numel() == 0:
        return waveform
    x = waveform.float()
    rms = torch.sqrt(torch.mean(x * x))
    if not torch.isfinite(rms):
        return waveform
    rms_value = float(rms.detach().cpu().item())
    if rms_value <= eps:
        return waveform
    gain = float(target_rms) / rms_value
    peak = float(torch.max(torch.abs(x)).detach().cpu().item())
    if peak > 0 and peak_limit > 0:
        gain = min(gain, float(peak_limit) / peak)
    if abs(gain - 1.0) <= 1e-6:
        return waveform
    return waveform * gain


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_name(f"{input_path.stem}-rms{input_path.suffix}")
    return input_path.with_name(f"{input_path.name}-rms.wav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply RMS normalization to an audio file and write the result."
    )
    parser.add_argument("input", help="Path to input audio file.")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output audio file (default: <input>-rms<ext>).",
    )
    parser.add_argument(
        "--target-rms",
        type=float,
        default=RMS_TARGET,
        help=f"Target RMS amplitude (default: {RMS_TARGET}).",
    )
    parser.add_argument(
        "--peak-limit",
        type=float,
        default=RMS_PEAK_LIMIT,
        help=f"Peak ceiling to avoid clipping (default: {RMS_PEAK_LIMIT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else default_output_path(input_path)
    )
    waveform, sample_rate = load_audio(str(input_path))
    normalized = rms_normalize_waveform(
        waveform,
        target_rms=args.target_rms,
        peak_limit=args.peak_limit,
    )
    save_audio(str(output_path), normalized, sample_rate)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
