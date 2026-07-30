"""Heavy-resource budget test for the two GPU stages of the pipeline.

Runs the *real* production stage functions — `vocal_separation.run_vocal_separation`
(BS-Roformer on GPU) and `vad_asr.run_vad_asr` (Whisper on GPU) — on a short
synthetic clip and asserts that the measured peak GPU/RAM never exceeds the
selected `ResourceProfile` caps. This is the empirical counterpart to
`test_resource_profiles.py`, which only checks the static budget arithmetic.

Design notes (why this stays reusable and honest):
- Skipped unless `--run-heavy-resource` is passed, and skipped when CUDA is
  unavailable (the GPU cap is the whole point — a CPU run would assert nothing).
- Drives each stage through the same reset/measure plumbing the production code
  uses (`reset_peak_gpu_memory_stats_for_run` + `_peak_gpu_memory_bytes`), so the
  numbers matched here are exactly what the pipeline prints and warns on.
- Peak GPU memory is read per stage: each production stage resets peak stats at
  its start, so reading right after it returns attributes the peak to that stage.
  Peak process RAM (`PeakWorkingSetSize` on Windows) is process-lifetime, which is
  the correct quantity for the whole-process RAM budget.
- The synthetic clip is continuous and loud so the energy VAD emits speech
  intervals (otherwise `run_vad_asr` short-circuits and never loads Whisper,
  making the GPU assertion vacuous). ~90s forces multiple 30s ASR groups.
- Parametrized over `gpu_budget_gb`; a 12/16GB machine can assert its own tier.
  Clip length is overridable via `RESOURCE_TEST_SECONDS` to also stress the RAM
  path with a long clip.

Run:  python -m pytest -q test/test_resource_budget_pipeline.py --run-heavy-resource
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from asr_playground.speech.recognition import stage as vad_asr
from asr_playground.speech.preprocessing import separation as vocal_separation
from asr_playground.speech.runtime.resources import (
    get_resource_profile,
    resource_limit_violations,
)
from asr_playground.speech.runtime.resource_usage import (
    _peak_gpu_memory_bytes,
    _peak_process_memory_bytes,
)

pytestmark = pytest.mark.heavy_resource  # `pipeline` marker added by conftest

_SR = 44_100


def _synth_speechlike_clip(path: Path, seconds: float, sr: int = _SR) -> None:
    """Write a continuous, loud, speech-like stereo WAV.

    Not real speech — a few formant-ish tones under a slow syllabic envelope plus
    light noise, scaled to ~-20 dBFS RMS. Loud + continuous is what matters: the
    energy VAD must see it as speech so Whisper actually loads and runs.
    """
    import soundfile as sf

    n = int(round(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    # Syllabic amplitude envelope (~4 Hz) kept well above zero so no frame reads
    # as silence.
    env = 0.6 + 0.4 * (0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t)))
    tones = (
        np.sin(2.0 * np.pi * 180.0 * t)
        + 0.6 * np.sin(2.0 * np.pi * 700.0 * t)
        + 0.4 * np.sin(2.0 * np.pi * 2500.0 * t)
    )
    rng = np.random.default_rng(0)
    noise = 0.05 * rng.standard_normal(n)
    mono = env * tones + noise
    rms = float(np.sqrt(np.mean(mono**2))) or 1.0
    target_rms = 10.0 ** (-20.0 / 20.0)  # -20 dBFS
    mono = np.clip(mono * (target_rms / rms), -0.99, 0.99).astype(np.float32)
    stereo = np.stack([mono, mono], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), stereo, sr, subtype="PCM_16")


def _measure(device: str) -> Tuple[Optional[int], Optional[int]]:
    return _peak_gpu_memory_bytes(device), _peak_process_memory_bytes()


def _assert_within_budget(
    label: str,
    gpu_budget_gb: int,
    peak_gmem: Optional[int],
    peak_mem: Optional[int],
) -> None:
    profile = get_resource_profile(gpu_budget_gb)
    violations = resource_limit_violations(
        peak_gpu_bytes=peak_gmem,
        peak_ram_bytes=peak_mem,
        profile=profile,
    )
    gib = 1024**3
    detail = (
        f"peak_gmem={0.0 if peak_gmem is None else peak_gmem / gib:.2f}GiB "
        f"(limit {profile.gpu_limit_bytes / gib:.2f}), "
        f"peak_mem={0.0 if peak_mem is None else peak_mem / gib:.2f}GiB "
        f"(limit {profile.ram_limit_bytes / gib:.2f})"
    )
    assert not violations, f"{label} exceeded {gpu_budget_gb}GB profile: {violations}; {detail}"


@pytest.mark.parametrize("gpu_budget_gb", [4])
def test_gpu_stages_stay_within_budget(
    request: pytest.FixtureRequest, tmp_path: Path, gpu_budget_gb: int
) -> None:
    # Defense-in-depth: never run the heavy body during a normal/full test run,
    # even if the conftest heavy_resource skip hook is changed. Must be requested
    # explicitly with --run-heavy-resource.
    if not request.config.getoption("--run-heavy-resource"):
        pytest.skip("requires --run-heavy-resource (heavy: loads BS-Roformer + Whisper on GPU)")

    import torch

    if not torch.cuda.is_available():
        pytest.skip("resource-budget test requires CUDA (the GPU cap is the point)")

    seconds = float(os.environ.get("RESOURCE_TEST_SECONDS", "90"))
    src = tmp_path / "clip.wav"
    _synth_speechlike_clip(src, seconds)

    # --- Stage 1: vocal separation (resets GPU peak stats at its own start). ---
    vocal = tmp_path / "clip-vocal.flac"
    vocal_separation.run_vocal_separation(
        src, output_path=vocal, gpu_budget_gb=gpu_budget_gb
    )
    assert vocal.exists()
    sep_gmem, sep_mem = _measure("cuda")
    _assert_within_budget("vocal separation", gpu_budget_gb, sep_gmem, sep_mem)

    # --- Stage 2: VAD-ASR (resets GPU peak stats at its own start; loads Whisper
    # only because the synthetic clip yields speech intervals). ---
    aligned = tmp_path / "clip-aligned.json"
    vad_asr.run_vad_asr(vocal, output_path=aligned, gpu_budget_gb=gpu_budget_gb)
    assert aligned.exists()
    asr_gmem, asr_mem = _measure("cuda")
    # Guard against a vacuous pass: if Whisper never ran, GPU peak would be ~0.
    assert asr_gmem is not None and asr_gmem > 256 * 1024**2, (
        "VAD-ASR GPU peak too low — Whisper likely never loaded, so the budget "
        f"assertion would be vacuous (peak_gmem={asr_gmem})"
    )
    _assert_within_budget("VAD-ASR", gpu_budget_gb, asr_gmem, asr_mem)
