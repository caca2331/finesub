"""Times for the units inside a MOSS Pro segment, which only has a start and an end (S5b, `docs/joint-asr.md`).

S5b, the default: Silero's speech probability per 32 ms frame of the mix; inside a segment the speech seconds
are shared out by weighted character count, so a pause Silero hears becomes a gap between the two units around
it, and the punctuation mark nearest a pause is moved onto it. Segment edges are trimmed to Silero's speech, a
start by at most TRIM_MAX_START and an end by at most TRIM_MAX_END, and a last run shorter than EDGE_BLIP after a
pause is not speech: Silero misses quiet onsets, tails and lines under music, and an unbounded trim cut them
off. Silero only decides times; every unit comes from MOSS Pro, whatever Silero hears. A segment Silero hears
less than MIN_VOICED of falls back to plain interpolation.

S6b (`--asr-aligner`): the same, but a segment over the splitter's ok limits takes its unit times from the
forced aligner (`aligner.py`) where the aligner gives them.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from ..postprocessing import segmentation as seg
from ...subtitles.metrics import weighted_char_count
from .lines import interpolated, unit_weight, units_of

SR = 16000
HOP = 512  # Silero's frame at 16 kHz: 32 ms
FRAME = HOP / SR

#: A pause inside a segment counts when it is at least this long (shorter dips are breathing, or jitter).
PAUSE_MIN = 0.15
#: Silero must hear at least this share of a segment for its spans to be trusted; else plain interpolation.
MIN_VOICED = 0.3
#: A punctuation mark within this many weighted characters (or this share of the segment) of a pause's
#: interpolated place is moved onto the pause.
SNAP_CHARS = 4.0
SNAP_SHARE = 0.15
#: A trimmed segment edge goes to Silero's speech edge plus these pads (its off edge already lags the speech).
TRIM_PAD_START = 0.1
TRIM_PAD_END = 0.0
#: How far every Silero off edge is moved back (its lag behind the speech end).
OFF_LAG = 0.1
#: Bounds on the trim, asymmetric because the edges fail differently: MOSS Pro's starts are already on time,
#: its ends run late (the readout and the reasons: `docs/joint-asr.md`).
TRIM_MAX_START = 0.1
TRIM_MAX_END = 1.0
#: A last speech run shorter than this, after a pause, is a click or a breath.
EDGE_BLIP = 0.1


def frame_probabilities(pcm16k: np.ndarray) -> np.ndarray:
    """Silero's speech probability per HOP-sample frame of 16 kHz mono int16 samples (production's model)."""
    import torch

    from ..preprocessing.silero_ghost import silero_exclusive

    wav = torch.from_numpy(pcm16k.astype(np.float32) / 32768)
    out = np.empty(len(wav) // HOP, np.float32)
    with silero_exclusive() as model, torch.inference_mode():
        model.reset_states()
        for k in range(len(out)):
            out[k] = model(wav[k * HOP:(k + 1) * HOP], SR).item()
        model.reset_states()
    return out


def speech_spans(p: np.ndarray, thr: float = 0.5, min_gap: float = 0.1,
                 min_speech: float = 0.06) -> list[tuple[float, float]]:
    """Speech by hysteresis (on at `thr`, off under `thr - 0.15`, Silero's own rule); gaps under `min_gap`
    closed, speech under `min_speech` dropped."""
    on, off = thr, thr - 0.15
    spans, start, state = [], 0, False
    for k, v in enumerate(p):
        if not state and v >= on:
            state, start = True, k
        elif state and v < off:
            state = False
            spans.append([start * FRAME, k * FRAME])
    if state:
        spans.append([start * FRAME, len(p) * FRAME])
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < min_gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_speech]


def _clip(spans: Sequence[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    return [(max(x, a), min(y, b)) for x, y in spans if y > a and x < b and min(y, b) - max(x, a) > 1e-3]


def _punct_rank(unit: str) -> int | None:
    """0 sentence end, 1 clause, None no punctuation at the unit's end."""
    cls = seg.punct_class(unit.rstrip()[-1]) if unit.strip() else "none"
    return 0 if cls == "sentence" else 1 if cls in ("clause", "closing") else None


def warp(segment: dict, spans: Sequence[tuple[float, float]]) -> list[dict]:
    """S5b: a segment's units timed by the speech Silero hears inside it (module docstring)."""
    us = units_of(segment["text"])
    a, b = segment["start"], segment["end"]
    inside = _clip(spans, a, b)
    voiced = sum(y - x for x, y in inside)
    if not us:
        return []
    if voiced < MIN_VOICED * max(b - a, 1e-3) or not inside:
        return interpolated(segment)
    runs: list[list[float]] = []
    for x, y in inside:
        if runs and x - runs[-1][1] < PAUSE_MIN:
            runs[-1][1] = y
        else:
            runs.append([x, y])
    if len(runs) > 1 and runs[-1][1] - runs[-1][0] < EDGE_BLIP:
        inside = [(x, y) for x, y in inside if y <= runs[-2][1]]
        runs.pop()
    na = min(max(a, inside[0][0] - TRIM_PAD_START), a + TRIM_MAX_START)
    nb = max(min(b, inside[-1][1] + TRIM_PAD_END - OFF_LAG), b - TRIM_MAX_END)
    a, b = na, nb  # a bound only widens [na, nb], so Silero's first run still starts in it
    for r in runs[:-1]:  # an end inside the segment is Silero's off edge, which lags the speech
        r[1] = max(r[0] + 0.01, r[1] - OFF_LAG)
    runs[0][0], runs[-1][1] = a, b
    weights = [unit_weight(u) for u in us]
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    total_w = cum[-1]
    anchors: list[tuple[int, int]] = []  # (units before it, the pause after run r)
    if len(runs) > 1:
        speech = [y - x for x, y in runs]
        before = np.concatenate([[0.0], np.cumsum(speech)])
        tol = max(SNAP_CHARS, SNAP_SHARE * total_w)
        for r in sorted(range(len(runs) - 1), key=lambda r: -(runs[r + 1][0] - runs[r][1])):
            x = before[r + 1] / before[-1] * total_w
            lo_j = max((j for j, rr in anchors if rr < r), default=0)
            hi_j = min((j for j, rr in anchors if rr > r), default=len(us))
            cands = [j for j in range(max(lo_j + 1, 1), min(hi_j, len(us)))
                     if _punct_rank(us[j - 1]) is not None and abs(cum[j] - x) <= tol]
            if cands:
                anchors.append((min(cands, key=lambda j: (_punct_rank(us[j - 1]), abs(cum[j] - x))), r))
        anchors.sort()
    bounds = [(0, 0)] + [(j, r + 1) for j, r in anchors] + [(len(us), len(runs))]
    out: list[dict] = []
    for (j0, r0), (j1, r1) in zip(bounds, bounds[1:]):
        piece = runs[r0:r1] if r1 > r0 else [runs[min(r0, len(runs) - 1)]]
        out.extend(_spread(us[j0:j1], weights[j0:j1], piece, segment.get("speaker")))
    return out


def _spread(units: list[str], weights: list[float], runs: list[list[float]], speaker) -> list[dict]:
    """Units over speech runs by weight; a unit that would straddle a pause keeps its larger part."""
    if not units:
        return []
    speech = [y - x for x, y in runs]
    total_s, total_w = sum(speech), sum(weights)
    edges = np.concatenate([[0.0], np.cumsum(speech)])

    def at(s: float) -> tuple[float, int]:
        k = max(min(int(np.searchsorted(edges, s, side="right")) - 1, len(runs) - 1), 0)
        return runs[k][0] + (s - edges[k]), k

    out, acc = [], 0.0
    for u, w in zip(units, weights):
        s0 = acc / total_w * total_s
        acc += w
        s1 = acc / total_w * total_s
        t0, k0 = at(s0 + 1e-9)
        t1, k1 = at(s1 - 1e-9)
        if k1 > k0:
            if runs[k0][1] - t0 >= t1 - runs[k1][0]:
                t1 = runs[k0][1]
            else:
                t0 = runs[k1][0]
        out.append({"start": round(t0, 3), "end": round(max(t1, t0 + 0.01), 3), "text": u, "speaker": speaker})
    return out


def too_long(segment: dict, params: seg.SplitParams) -> bool:
    return (weighted_char_count(segment["text"]) > params.chars_ok_hi
            or segment["end"] - segment["start"] > params.dur_ok_hi)


def timed(segments: Sequence[dict], probabilities: np.ndarray, *, duration: float, params: seg.SplitParams,
          align: Callable[[list[dict], list[dict]], dict[int, list[dict]]] | None = None) -> tuple[list[dict], dict]:
    """The segments (by start, then end) with their timed `units`, and counts. `align(long_segments, segments)` (S6b)
    returns, per index into the first list, the aligner's units for each segment it aligned."""
    segs = [s for s in sorted(segments, key=lambda s: (s["start"], s["end"]))
            if s["text"].strip() and s["start"] < duration]
    spans = speech_spans(probabilities)
    aligned: dict[int, list[dict]] = {}
    if align is not None:
        long_ones = [i for i, s in enumerate(segs) if too_long(s, params)]
        got = align([segs[i] for i in long_ones], segs)
        aligned = {long_ones[k]: us for k, us in got.items()}
    out = []
    for i, s in enumerate(segs):
        us = aligned.get(i) or warp(s, spans)
        if us:
            out.append(dict(s, units=us))
    return out, {"segments": len(segs), "aligned": len(aligned)}
