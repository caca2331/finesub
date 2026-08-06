"""Decision logic of the opt-in silero ghost suppression (no model, no audio)."""

from __future__ import annotations

import numpy as np
import torch

from asr_playground.speech.preprocessing import silero_ghost
from asr_playground.speech.preprocessing.energy import TARGET_SR, VadEnergyTrack

HOP_SIL = silero_ghost.SILERO_HOP / TARGET_SR  # 32 ms


def make_track(n_frames: int, db: float, loud: list[tuple[int, int, float]] = ()):
    e = torch.full((n_frames,), float(db))
    for i0, i1, v in loud:
        e[i0:i1] = float(v)
    return VadEnergyTrack(energy_db=e, hop_sec=0.01, frame_sec=0.025,
                          energy_mode="weighted")


def probs_with(n: int, spans: list[tuple[float, float, float]]):
    p = np.zeros(n, dtype=np.float32)
    for s, e, v in spans:
        p[int(s / HOP_SIL):int(e / HOP_SIL)] = v
    return p


def run(segments, probs, track, **kw):
    kept, stats = silero_ghost.drop_ghost_segments(segments, probs, track, **kw)
    return [s["start"] for s in kept], stats


def test_ghost_dropped_voiced_kept():
    track = make_track(1000, -20.0)
    probs = probs_with(320, [(4.0, 5.0, 0.95)])  # voiced only in the second seg
    segs = [{"start": 1.0, "end": 2.0}, {"start": 4.0, "end": 5.0}]
    kept, stats = run(segs, probs, track)
    assert kept == [4.0]
    assert stats["dropped"] == 1
    assert stats["dropped_intervals"][0]["start"] == 1.0


def test_loud_interval_survives_silero():
    # 200 frames of +5 dB inside the segment: the energy guard vetoes the drop.
    track = make_track(1000, -20.0, loud=[(100, 300, 5.0)])
    probs = probs_with(320, [])
    segs = [{"start": 1.0, "end": 3.0}]
    kept, _ = run(segs, probs, track)
    assert kept == [1.0]


def test_long_interval_never_dropped():
    track = make_track(3000, -20.0)
    probs = probs_with(1000, [])
    segs = [{"start": 1.0, "end": 1.0 + silero_ghost.GHOST_MAX_DROP_SEC + 1.0}]
    kept, _ = run(segs, probs, track)
    assert kept == [1.0]


def test_segment_outside_probs_kept():
    # Audio tail shorter than a silero frame: no evidence, no drop.
    track = make_track(1000, -20.0)
    probs = np.zeros(0, dtype=np.float32)
    segs = [{"start": 1.0, "end": 2.0}]
    kept, _ = run(segs, probs, track)
    assert kept == [1.0]


def test_threshold_is_strict():
    track = make_track(1000, -20.0)
    probs = probs_with(320, [(1.0, 2.0, silero_ghost.GHOST_SILERO_PEAK_MAX)])
    segs = [{"start": 1.0, "end": 2.0}]
    kept, _ = run(segs, probs, track)  # peak == threshold -> not `< max` -> kept
    assert kept == [1.0]
