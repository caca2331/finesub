"""Streaming VAD detection for the recognition stage."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from . import energy


def detect_segments(
    input_path: Path,
    *,
    observer: energy.WaveformObserver | None = None,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    float,
    dict[str, float],
    energy.VadEnergyTrack,
]:
    """Detect speech intervals and return their energy track and timing.

    ``observer`` sees the normalized blocks as they go by -- how the opt-in
    silero assist gets its probabilities without a second pass over the audio.
    """

    timing: dict[str, float] = {}
    vad_params = energy.vad_params()

    # Loading, normalization and framing run block by block. Recognition later
    # streams the audio back from disk, so neither step retains the full file.
    started = time.perf_counter()
    with torch.inference_mode():
        raw_segments, vad_meta, audio_duration, energy_track = energy.run_vad_file(
            str(input_path),
            params=vad_params,
            timing=timing,
            observer=observer,
        )
    timing["vad_sec"] = time.perf_counter() - started

    if energy_track.energy_mode == "weighted":
        vad_meta.setdefault("vad", {})["segment_energy"] = (
            energy.segment_energy_metadata()
        )

    return raw_segments, vad_meta, audio_duration, timing, energy_track
