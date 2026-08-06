"""Cheap sweeps over the production detector's post-processing.

The expensive half of the detector (load, normalize, framing, spectral weighting,
noise-floor tracking) does not depend on the interval knobs, so it is computed once
per clip and cached. Only the scoring loop and the negative padding are re-run.

`verify()` checks this shortcut against the real streamed entrypoint before any
sweep result is believed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

Interval = Tuple[float, float]

# The knobs this module can sweep. Read from the module rather than restated:
# a copy here silently reverts production changes inside the shortcut, which is
# exactly what happened when NEGATIVE_PAD_RIGHT_MS moved 140 -> 100 and every
# sweep kept scoring the old value.
_SWEEPABLE = (
    "MIN_NON_SPEECH_MS",
    "MERGE_GAP_MS",
    "NEGATIVE_PAD_LEFT_MS",
    "NEGATIVE_PAD_RIGHT_MS",
    "ABS_NON_SPEECH_MAX_DBFS_ENTER",
    "ABS_NON_SPEECH_MAX_DBFS_EXIT",
    "WEIGHTED_INTERVAL",
)


def _defaults() -> Dict[str, object]:
    from asr_playground.speech.preprocessing import energy as E

    return {k: getattr(E, k) for k in _SWEEPABLE}


class _DefaultsProxy(dict):
    """Always reflects the current module values."""

    def __iter__(self):
        return iter(_defaults())

    def items(self):
        return _defaults().items()

    def keys(self):
        return _defaults().keys()

    def __getitem__(self, k):
        return _defaults()[k]


DEFAULTS = _DefaultsProxy()


@dataclass
class Tracks:
    frame_dbfs: object
    energy_db: object
    frame_starts: object
    frame_ends: object
    noise_floor: object
    duration: float


def compute_tracks(path: Path, snr_enter: float = 6.0) -> Tracks:
    import torch

    from asr_playground.speech.preprocessing import energy as E

    # Use the project's own loader, not librosa. test/test_vad_streaming.py pins
    # the reference chain as _load_asr_audio_streamed -> light_normalize -> ...,
    # and loading through librosa instead makes this shortcut drift from the
    # streamed production path (40 ms on some clips once the noise floor became
    # more sensitive to its input).
    wav = E._load_asr_audio_streamed(str(path))
    wav = E.light_normalize(wav, E.TARGET_SR)
    dbfs, edb, starts, ends = E._compute_frame_tracks_for_waveform(
        wav, E.TARGET_SR, energy_mode="weighted")
    duration = wav.numel() / float(E.TARGET_SR)
    floor = E.estimate_noise_floor_db_local(
        edb, starts, duration,
        local_window_sec=E.NOISE_LOCAL_WINDOW_SEC,
        local_hop_sec=E.NOISE_LOCAL_HOP_SEC,
        local_percentile=E.NOISE_INIT_PERCENTILE,
        track_gate_db=E.NOISE_TRACK_GATE_DB,
        follow_alpha=E.NOISE_TRACK_FOLLOW_ALPHA,
        rise_alpha=E.NOISE_TRACK_RISE_ALPHA,
        local_blend=E.NOISE_LOCAL_BLEND,
    )
    return Tracks(dbfs, edb, starts, ends, floor, duration)


def cached_tracks(path: Path, cache_dir: Optional[Path]) -> Tracks:
    """compute_tracks with an npz cache -- the framing pass over an hour of audio is
    the expensive part of every sweep and does not depend on any knob being swept."""
    import torch

    if cache_dir is None:
        return compute_tracks(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    from asr_playground.speech.preprocessing import energy as E

    f = cache_dir / f"tracks2-{path.stem}.npz"
    if f.exists():
        z = np.load(f)
        dbfs, edb, starts, ends = (torch.from_numpy(z[k])
                                   for k in ("dbfs", "edb", "starts", "ends"))
        duration = float(z["duration"])
        # The floor is deliberately NOT cached: it is the thing under study, and a
        # stale copy would silently score an old estimator.
        floor = E.estimate_noise_floor_db_local(
            edb, starts, duration,
            local_window_sec=E.NOISE_LOCAL_WINDOW_SEC,
            local_hop_sec=E.NOISE_LOCAL_HOP_SEC,
            local_percentile=E.NOISE_INIT_PERCENTILE,
            track_gate_db=E.NOISE_TRACK_GATE_DB,
            follow_alpha=E.NOISE_TRACK_FOLLOW_ALPHA,
            rise_alpha=E.NOISE_TRACK_RISE_ALPHA,
            local_blend=E.NOISE_LOCAL_BLEND,
        )
        return Tracks(dbfs, edb, starts, ends, floor, duration)
    tr = compute_tracks(path)
    np.savez(f, dbfs=tr.frame_dbfs.numpy(), edb=tr.energy_db.numpy(),
             starts=tr.frame_starts.numpy(), ends=tr.frame_ends.numpy(),
             duration=np.float64(tr.duration))
    return tr


def _load_16k(path: Path) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(path), sr=16000, mono=True)
    return y.astype(np.float32)


def non_speech_from_tracks(tr: Tracks, consts: Optional[Dict[str, float]] = None,
                           snr_enter: float = 6.0) -> List[Interval]:
    from asr_playground.speech.preprocessing import energy as E

    consts = {**_defaults(), **(consts or {})}
    saved = {}
    try:
        for k, v in consts.items():
            saved[k] = getattr(E, k)
            setattr(E, k, v)
        raw = E._score_to_non_speech_intervals(
            tr.energy_db, tr.noise_floor, tr.frame_dbfs,
            tr.frame_starts, tr.frame_ends, tr.duration,
            enter_margin_db=snr_enter,
            weighted=bool(consts["WEIGHTED_INTERVAL"]),
        )
        padded = E._apply_negative_padding(raw, tr.duration)
        padded = E._absorb_low_peak_speech(padded, tr.energy_db, tr.duration)
        return E._carve_low_peak_speech(padded, tr.energy_db, tr.duration)
    finally:
        for k, v in saved.items():
            setattr(E, k, v)


def speech_from_tracks(tr: Tracks, consts: Optional[Dict[str, float]] = None,
                       snr_enter: float = 6.0) -> List[Interval]:
    from asr_playground.speech.preprocessing import energy as E

    ns = non_speech_from_tracks(tr, consts, snr_enter)
    return [(float(s), float(e)) for s, e in E.invert_intervals(ns, tr.duration) if e > s]


def verify(path: Path, tol: float = 0.011) -> bool:
    """Check the shortcut against the streamed production entrypoint.

    Exact as of 2026-08-04 (0 of 197 boundaries differ) once the shortcut loads
    through `_load_asr_audio_streamed` and reads the interval constants from the
    module instead of restating them. Any nonzero deviation means the shortcut has
    drifted and sweep results built on it should not be trusted -- run the sweep's
    conclusions back through `run_vad_file` before believing them.
    """
    from asr_playground.speech.preprocessing import energy as E

    items, _m, _d, _t = E.run_vad_file(path, params=E.vad_params())
    ref = [(float(i["start"]), float(i["end"])) for i in items]
    got = speech_from_tracks(compute_tracks(path))
    if len(ref) != len(got):
        print(f"MISMATCH: streamed={len(ref)} intervals, shortcut={len(got)}")
        return False
    diffs = [max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(ref, got)]
    worst = max(diffs) if diffs else 0.0
    n_off = sum(1 for d in diffs if d > 1e-9)
    print(f"shortcut vs streamed: {len(ref)} intervals, {n_off} differ, "
          f"max deviation {worst*1000:.1f} ms")
    return worst <= tol
