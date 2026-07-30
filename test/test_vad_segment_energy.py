from __future__ import annotations

import math

import pytest
import torch

from asr_playground.speech.recognition import stage as vad_asr
from asr_playground.speech.preprocessing import energy as vad_energy


def _track(
    energy_db: list[float],
    hop_sec: float,
    frame_sec: float,
    *,
    mode: str = "weighted",
) -> vad_energy.VadEnergyTrack:
    return vad_energy.VadEnergyTrack(
        energy_db=torch.tensor(energy_db, dtype=torch.float32),
        hop_sec=hop_sec,
        frame_sec=frame_sec,
        energy_mode=mode,
    )


def test_constant_weighted_energy_is_unchanged() -> None:
    track = _track([-23.5, -23.5], 0.01, 0.025)

    value = vad_energy.aggregate_segment_weighted_energy_db(track, 0.005, 0.03)

    assert value == pytest.approx(-23.5)


def test_weighted_energy_uses_overlap_weighted_linear_power_mean() -> None:
    track = _track([-20.0, -10.0], 0.02, 0.02)
    expected_power = (0.01 * 10 ** (-20.0 / 10.0) + 0.02 * 10 ** (-10.0 / 10.0)) / 0.03

    value = vad_energy.aggregate_segment_weighted_energy_db(track, 0.01, 0.04)

    assert value == pytest.approx(10.0 * math.log10(expected_power))


def test_very_short_segment_weights_all_overlapping_frames() -> None:
    track = _track([-20.0, -10.0], 0.01, 0.025)
    expected_power = (10 ** (-20.0 / 10.0) + 10 ** (-10.0 / 10.0)) / 2.0

    value = vad_energy.aggregate_segment_weighted_energy_db(track, 0.019, 0.021)

    assert value == pytest.approx(10.0 * math.log10(expected_power))


def test_multi_hour_offsets_keep_exact_frame_selection() -> None:
    # 10 h of frames at 10 ms hop; a float32 frame-time grid carries ~4 ms
    # error here and used to shift boundary-frame selection for short segments.
    hop_sec, frame_sec = 0.01, 0.025
    loud_index = int(round(36000.0 / hop_sec))
    n_frames = loud_index + 10
    energy = torch.full((n_frames,), -60.0, dtype=torch.float32)
    energy[loud_index] = -10.0
    track = vad_energy.VadEnergyTrack(
        energy_db=energy,
        hop_sec=hop_sec,
        frame_sec=frame_sec,
        energy_mode="weighted",
    )

    # Frames overlapping (36000.002, 36000.006): exactly loud_index-2..loud_index
    # with overlaps 3 ms / 4 ms / 4 ms.
    value = vad_energy.aggregate_segment_weighted_energy_db(
        track, 36000.002, 36000.006
    )

    expected_power = (
        0.003 * 10 ** (-60.0 / 10.0)
        + 0.004 * 10 ** (-60.0 / 10.0)
        + 0.004 * 10 ** (-10.0 / 10.0)
    ) / 0.011
    assert value == pytest.approx(10.0 * math.log10(expected_power))


@pytest.mark.parametrize(
    ("track", "start", "end"),
    [
        (_track([-20.0], 0.01, 0.025, mode="none"), 0.0, 0.02),
        (_track([-20.0], 0.01, 0.025), 1.0, 2.0),
        (_track([], 0.01, 0.025), 0.0, 1.0),
        (_track([-20.0], 0.01, 0.025), 1.0, 1.0),
        (_track([-20.0], 0.0, 0.025), 0.0, 0.02),
    ],
)
def test_weighted_energy_is_optional_when_unavailable(
    track: vad_energy.VadEnergyTrack,
    start: float,
    end: float,
) -> None:
    assert vad_energy.aggregate_segment_weighted_energy_db(track, start, end) is None


def test_final_aligned_segment_boundaries_drive_energy_annotation() -> None:
    track = _track([-30.0, -10.0], 0.02, 0.02)
    segments = [
        {"start": 0.0, "end": 0.02, "text": "quiet"},
        {"start": 0.02, "end": 0.04, "text": "loud"},
    ]

    annotated = vad_asr.annotate_segments_with_vad_energy(segments, track)

    assert annotated[0][vad_energy.SEGMENT_ENERGY_FIELD] == pytest.approx(-30.0)
    assert annotated[1][vad_energy.SEGMENT_ENERGY_FIELD] == pytest.approx(-10.0)
    assert vad_energy.SEGMENT_ENERGY_FIELD not in segments[0]


def test_segment_energy_metadata_describes_aligned_field() -> None:
    metadata = vad_energy.segment_energy_metadata()

    assert metadata == {
        "field": "vad_weighted_energy_db",
        "source": "adaptive_weighted_spectral_energy",
        "aggregation": "overlap_weighted_power_mean_db",
        "frame_ms": 25.0,
        "hop_ms": 10.0,
        "audio": "normalized_vocal",
    }
