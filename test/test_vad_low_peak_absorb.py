"""The absolute peak floor on speech intervals (_absorb_low_peak_speech)."""

from __future__ import annotations

import torch

from asr_playground.speech.preprocessing import energy as E

HOP, _FRAME = E._frame_grid_seconds()


def make_energy(n: int, db: float, spans: list[tuple[float, float, float]] = ()):
    e = torch.full((n,), float(db))
    for s, end, v in spans:
        e[int(s / HOP):int(end / HOP)] = float(v)
    return e


def test_quiet_gap_absorbed_between_non_speech():
    # non-speech [0,1] and [2,3]; the speech gap [1,2] never exceeds -50, while
    # the trailing speech [3,4] is loud and must survive
    e = make_energy(400, -70.0, spans=[(3.2, 3.8, -10.0)])
    out = E._absorb_low_peak_speech([(0.0, 1.0), (2.0, 3.0)], e, 4.0)
    assert out == [(0.0, 3.0)]


def test_loud_gap_kept():
    e = make_energy(400, -70.0, spans=[(1.2, 1.5, -20.0), (3.2, 3.8, -20.0)])
    out = E._absorb_low_peak_speech([(0.0, 1.0), (2.0, 3.0)], e, 4.0)
    assert out == [(0.0, 1.0), (2.0, 3.0)]


def test_threshold_boundary_keeps_at_exact_floor():
    e = make_energy(400, -70.0,
                    spans=[(1.2, 1.5, E.MIN_SPEECH_PEAK_DB), (3.2, 3.8, -20.0)])
    out = E._absorb_low_peak_speech([(0.0, 1.0), (2.0, 3.0)], e, 4.0)
    assert len(out) == 2  # peak == floor is not `< floor`


def test_leading_and_trailing_quiet_speech_absorbed():
    e = make_energy(500, -70.0, spans=[(1.0, 2.0, -10.0)])
    # speech before the first non-speech and after the last, both quiet
    out = E._absorb_low_peak_speech([(0.5, 0.8), (2.5, 3.0)], e, 5.0)
    assert out[0][0] == 0.0
    assert out[-1][1] == 5.0


def test_gap_with_no_frame_start_inside_kept():
    # (1.002, 1.008) contains no frame start on the 10 ms grid: no evidence,
    # so the gap stays speech even though everything around is quiet
    e = make_energy(400, -70.0, spans=[(3.2, 3.8, -20.0)])
    out = E._absorb_low_peak_speech([(0.0, 1.002), (1.008, 3.0)], e, 4.0)
    assert len(out) == 2
