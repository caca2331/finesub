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
- Parametrized over every tier, so one run measures the whole ladder: the
  separator's instance count is the only thing a tier changes, and that is
  exactly what the GPU peak is expected to track.
  Clip length is overridable via `RESOURCE_TEST_SECONDS` to also stress the RAM
  path with a long clip -- note the 300s worker ladder, which caps a 90s clip at
  one separator worker whatever the tier asks for. Measuring the wider tiers
  therefore needs `RESOURCE_TEST_SECONDS` past 300 (2 workers) and 600 (3).

Run:  python -m pytest -q test/test_resource_budget_pipeline.py --run-heavy-resource
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from finesub.speech.recognition import vad_asr_stage as vad_asr
from finesub.speech.preprocessing.separator import separation as vocal_separation
from finesub.speech.runtime.resources import (
    get_resource_profile,
    resource_limit_violations,
)
from finesub.speech.runtime.resource_usage import (
    _peak_gpu_memory_bytes,
    _peak_process_memory_bytes,
)

pytestmark = pytest.mark.heavy_resource  # `pipeline` marker added by conftest

#: Real speech for the ASR half. `assets/` is gitignored, so this is a local
#: convenience rather than a fixture the suite can rely on -- the test skips
#: with a message when it is absent rather than asserting on silence.
_SPEECH_ASSET_RELATIVE = "assets/harvard.flac"
_REPO_SPEECH_ASSET = Path(__file__).resolve().parents[1] / _SPEECH_ASSET_RELATIVE

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
    gpu_tier: str,
    peak_gmem: Optional[int],
    peak_mem: Optional[int],
) -> None:
    profile = get_resource_profile(gpu_tier)
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
    # Printed whether or not it fails: the point of a heavy run is the number,
    # and a passing assertion that swallows it makes the next calibration start
    # from scratch.
    print(f"[budget] {gpu_tier:9} {label:18} {detail}")
    assert not violations, f"{label} exceeded the {gpu_tier} tier: {violations}; {detail}"


@pytest.mark.timeout(1800)
def test_the_separator_gives_its_weights_back(request, tmp_path: Path) -> None:
    """A finished separation must not keep the card.

    Measured 2026-09-01, before the fix: 0.62 GiB of fp32 Roformer weights were
    still resident after `run_vocal_separation` returned -- with the pool's
    master dropped, zero leases, `gc.collect()`, `empty_cache()` and even
    `torch.compiler.reset()` all done. Per *call*, so a batch (which runs its
    files as threads in one process) leaked it once per file: ten files, six
    GiB, on a card the `entry` tier says needs three.

    Two runs, because one would not have caught it: the give-away was the
    second run starting from the first one's residue.
    """

    if not request.config.getoption("--run-heavy-resource"):
        pytest.skip("requires --run-heavy-resource (loads BS-Roformer on GPU)")

    import gc

    import torch

    if not torch.cuda.is_available():
        pytest.skip("the point is GPU residency")

    src = tmp_path / "clip.wav"
    _synth_speechlike_clip(src, 60.0)

    def live_cpu_tensor_bytes() -> int:
        """Host-side torch tensors still alive. Watching only CUDA is how the
        first attempt at this fix passed review-free: it moved the weights to
        the host with `.to("cpu")` and the leak simply changed address."""

        total = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor) and not obj.is_cuda:
                    total += obj.numel() * obj.element_size()
            except Exception:  # noqa: BLE001 - some objects dislike isinstance
                continue
        return total

    def settled() -> tuple[int, int]:
        gc.collect()
        torch.cuda.empty_cache()
        return int(torch.cuda.memory_allocated()), live_cpu_tensor_bytes()

    base_cuda, base_cpu = settled()
    cuda_readings: list[int] = []
    cpu_readings: list[int] = []
    # Three runs, not two: one shows a leak's size, two show it accumulates,
    # and the third is what separates "bounded one-off" from "grows forever".
    for index in range(3):
        vocal_separation.run_vocal_separation(
            src, output_path=tmp_path / f"vocal-{index}.flac", gpu_tier="entry"
        )
        cuda_now, cpu_now = settled()
        cuda_readings.append(cuda_now - base_cuda)
        cpu_readings.append(cpu_now - base_cpu)

    gib = 1024**3
    print(
        "[leak] cuda="
        + ", ".join(f"{value/gib:.3f}G" for value in cuda_readings)
        + " | cpu tensors="
        + ", ".join(f"{value/gib:.3f}G" for value in cpu_readings)
    )
    readings = cuda_readings

    # The weights are ~0.6GiB; anything approaching that is the leak back. The
    # allowance is deliberately well under one copy of them, and it does NOT
    # scale with the run count -- residue that grows per call is the bug.
    cap = int(0.15 * gib)
    for index, value in enumerate(cuda_readings, start=1):
        assert value < cap, (
            f"{index} separation(s) left {value/gib:.3f}GiB on the card"
        )
    for index, value in enumerate(cpu_readings, start=1):
        assert value < cap, (
            f"{index} separation(s) left {value/gib:.3f}GiB of host tensors -- "
            "the weights were relocated rather than released"
        )


# The suite-wide `--timeout=120` bounds a hang, not a slow test; this one is
# slow by construction (two real models over minutes of audio, three times).
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("gpu_tier", ["entry", "standard", "high"])
def test_gpu_stages_stay_within_budget(
    request: pytest.FixtureRequest, tmp_path: Path, gpu_tier: str
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
        src, output_path=vocal, gpu_tier=gpu_tier
    )
    assert vocal.exists()
    sep_gmem, sep_mem = _measure("cuda")
    _assert_within_budget("vocal separation", gpu_tier, sep_gmem, sep_mem)

    # --- Stage 2: VAD-ASR. Needs REAL speech: the synthetic clip is tones under
    # a syllabic envelope, which the energy VAD accepts but the decoder
    # transcribes to nothing -- measured 2026-09-01, zero segments at every
    # tier. This half of the test was passing on a guard (`peak_gmem > 256MB`)
    # that the separator's leaked weights satisfied for it; with the leak fixed
    # there is nothing to hide behind, so it either runs on real audio or says
    # it did not run. ---
    speech = _REPO_SPEECH_ASSET
    if speech is None or not speech.is_file():
        pytest.skip(
            "VAD-ASR budget needs real speech; put one at "
            f"{_SPEECH_ASSET_RELATIVE} (assets/ is gitignored). "
            "The separator budget above did run."
        )
    aligned = tmp_path / "clip-aligned.json"
    vad_asr.run_vad_asr(speech, output_path=aligned, gpu_tier=gpu_tier)
    assert aligned.exists()
    asr_gmem, asr_mem = _measure("cuda")
    # Guard against a vacuous pass, read off the ARTIFACT rather than the GPU
    # counter. The counter cannot answer this: CT2 allocates outside torch, so
    # `max_memory_reserved` never saw the decoder at all
    # (`lang_redecode.WHISPER_RESIDENT_GIB_BY_MODEL` says as much). The old
    # `peak_gmem > 256MB` check only ever passed because the separator was
    # leaking 0.6GiB of weights into this stage's measurement -- fixing that
    # leak is what exposed it. Segments exist only if the decoder ran.
    payload = json.loads(aligned.read_text(encoding="utf-8"))
    segments = payload.get("segments") or []
    assert segments, (
        "VAD-ASR produced no segments — the decoder never ran, so the budget "
        "assertion below would be vacuous"
    )
    _assert_within_budget("VAD-ASR", gpu_tier, asr_gmem, asr_mem)
