"""Default-on tail steps: pause hints, low-peak carving, seam restoration."""

from __future__ import annotations

import numpy as np
import torch

from asr_playground.speech.preprocessing import energy as E
from asr_playground.speech.preprocessing import silero_ghost as SG

HOP, _ = E._frame_grid_seconds()


def test_pause_hint_recorded_for_cancelled_gap():
    # raw gap of 100 ms dies in the padding shrink (needs >= 190 ms to survive)
    padded, hints = E._apply_negative_padding_hints([(1.0, 1.1), (5.0, 6.0)], 10.0)
    assert (5.04, 5.86) in [(round(a, 2), round(b, 2)) for a, b in padded]
    assert hints == [round(1.1 - E.PAUSE_HINT_OFFSET_SEC, 3)]


def _scorer_track(labels):
    """Scorer inputs from per-frame 'loud'/'quiet' labels at a 10 ms hop
    (mirrors test_intervals; test modules are not importable from each other)."""
    arr = np.array(labels)
    energy_arr = np.where(arr == "quiet", -60.0, -10.0).astype(np.float32)
    starts = (np.arange(len(labels)) * 0.01).astype(np.float32)
    ends = (starts + 0.025).astype(np.float32)
    return {
        "energy_db": torch.from_numpy(energy_arr),
        "noise_floor_db": torch.from_numpy(np.full(len(labels), -40.0, dtype=np.float32)),
        "frame_dbfs": torch.from_numpy(energy_arr.copy()),
        "frame_starts": torch.from_numpy(starts),
        "frame_ends": torch.from_numpy(ends),
        "duration_sec": float(ends[-1]) if len(labels) else 0.0,
    }


def test_scorer_hint_for_dying_pause_candidate():
    # 8 quiet frames (80 ms) between loud stretches: gathers score, never
    # reaches the interval threshold, dies -> hint at its last quiet frame -40ms
    track = _scorer_track(["loud"] * 50 + ["quiet"] * 8 + ["loud"] * 50)
    hints: list = []
    out = E._score_to_non_speech_intervals(
        **track, enter_margin_db=6.0, weighted=True, pause_hints_out=hints)
    assert out == []
    assert len(hints) == 1
    assert abs(hints[0] - (0.585 - E.PAUSE_HINT_OFFSET_SEC)) < 0.02


def test_scorer_no_hint_when_interval_emitted():
    track = _scorer_track(["loud"] * 50 + ["quiet"] * 50 + ["loud"] * 50)
    hints: list = []
    out = E._score_to_non_speech_intervals(
        **track, enter_margin_db=6.0, weighted=True, pause_hints_out=hints)
    assert len(out) == 1
    assert hints == []


def test_pause_hint_needs_minimum_gap():
    _, hints = E._apply_negative_padding_hints([(1.0, 1.03)], 10.0)
    assert hints == []  # 30 ms < PAUSE_HINT_MIN_MS


def make_energy(n, db, spans=()):
    e = torch.full((n,), float(db))
    for s, end, v in spans:
        e[int(s / HOP):int(end / HOP)] = float(v)
    return e


def test_carve_splits_interior_low_peak_bridge():
    # speech [1, 5] with a sub -45 bridge at [2.5, 3.5]
    e = make_energy(600, 5.0, spans=[(2.5, 3.5, -60.0)])
    ns = [(0.0, 1.0), (5.0, 6.0)]
    out = E._carve_low_peak_speech(ns, e, 6.0)
    assert len(out) == 3
    c = out[1]
    assert abs(c[0] - (2.5 + E.CARVE_LEAD_OUT_SEC)) < 0.02
    assert abs(c[1] - (3.5 - E.CARVE_LEAD_IN_SEC)) < 0.02


def test_carve_trims_low_peak_head():
    # speech [1, 3] whose first 0.6 s never reaches -45
    e = make_energy(600, 5.0, spans=[(1.0, 1.6, -60.0)])
    ns = [(0.0, 1.0), (3.0, 4.0)]
    out = E._carve_low_peak_speech(ns, e, 6.0)
    assert abs(out[0][1] - (1.6 - E.CARVE_LEAD_IN_SEC)) < 0.02


def test_carve_leaves_short_dips_alone():
    e = make_energy(600, 5.0, spans=[(2.0, 2.2, -60.0)])
    ns = [(0.0, 1.0), (5.0, 6.0)]
    assert E._carve_low_peak_speech(ns, e, 6.0) == ns


def test_seam_restored_at_exact_bounds_unless_loud():
    e_np = np.full(1000, -20.0)
    e_np[int(5.0 / HOP):int(5.2 / HOP)] = 3.0  # loud content in the second gap
    base = [(0.0, 2.0), (2.3, 4.8), (5.3, 7.0)]
    merged = [(0.0, 7.0)]
    out, restored = SG._restore_seams(merged, base, e_np, HOP)
    assert restored == 1
    assert (2.0, 2.3) in [(round(a, 2), round(b, 2))
                          for a, b in zip([iv[1] for iv in out[:-1]],
                                          [iv[0] for iv in out[1:]])]
