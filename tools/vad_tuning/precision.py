"""Two metrics that ask what the VAD is actually for, now that recall is settled.

`lost` was the wrong question. Appendix I showed 93.5% of "lost" words sit inside the
0.7 s of real gap audio `inserted_gap_parts` hands to the decoder anyway -- the ASR
never lost them, only their coordinates fell outside a speech interval. So the metric
was mostly measuring where whisper decided to put a timestamp.

What is left worth optimising is precision, in two separable senses:

tightness   how much dead audio sits between a speech interval's start and the first
            real word in it (and symmetrically at the tail). Only measurable where
            real word boundaries are known, i.e. the one hand-corrected clip.

noise       speech intervals that contain no speech at all -- the detector opened on
            background. Measurable with no annotation whatever: run the ASR, and any
            interval that produced no word is a candidate. Using the *union* of words
            from several different VAD configurations as the speech-presence map
            keeps this from being circular in the one direction that matters, since
            a region every configuration failed to decode is unlikely to be speech.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from refs import Word, covered

Interval = Tuple[float, float]


@dataclass
class Tightness:
    n_intervals: int
    head_waste: np.ndarray      # seconds of lead-in before the first word
    tail_waste: np.ndarray      # seconds of run-out after the last word
    clipped_onsets: int         # word starts that fell outside any interval
    clipped_tails: int
    total_waste: float

    def line(self, name: str) -> str:
        h, t = self.head_waste, self.tail_waste
        return (f"{name:<26} n={self.n_intervals:>4d} "
                f"head med={np.median(h):>5.2f} p90={np.quantile(h, .9):>5.2f} "
                f"tail med={np.median(t):>5.2f} p90={np.quantile(t, .9):>5.2f} "
                f"waste={self.total_waste:>7.1f}s "
                f"cutOnset={self.clipped_onsets:>3d} cutTail={self.clipped_tails:>3d}")


def tightness(speech: Sequence[Interval], words: Sequence[Word]) -> Tightness:
    """Dead audio at each interval edge, measured against real word boundaries.

    Only intervals that contain at least one word are measured -- an interval with no
    word in it is a noise interval, which is the other metric's business, not this
    one's. Counting it here would let a detector look "tight" by opening on noise.
    """
    speech = sorted(speech)
    ws = sorted(words, key=lambda w: w.start)
    starts = np.array([w.start for w in ws])
    ends = np.array([w.end for w in ws])

    head, tail = [], []
    cut_on = cut_tail = 0
    n_used = 0
    for s, e in speech:
        lo = int(np.searchsorted(starts, s))
        hi = int(np.searchsorted(starts, e))
        inside = list(range(lo, hi))
        if not inside:
            continue
        n_used += 1
        head.append(max(0.0, float(starts[inside[0]]) - s))
        tail.append(max(0.0, e - float(ends[inside[-1]])))
    for w in ws:
        probe = min(0.05, (w.end - w.start) / 3)
        if covered(speech, w.start, w.start + probe) < probe * 0.5:
            cut_on += 1
        if covered(speech, w.end - probe, w.end) < probe * 0.5:
            cut_tail += 1
    h = np.array(head) if head else np.array([0.0])
    t = np.array(tail) if tail else np.array([0.0])
    return Tightness(n_used, h, t, cut_on, cut_tail, float(h.sum() + t.sum()))


@dataclass
class NoiseVerdict:
    n_intervals: int
    empty_intervals: int
    empty_sec: float
    speech_sec: float
    empty_durs: np.ndarray
    unvoiced_sec: float = 0.0

    @property
    def empty_frac(self) -> float:
        return self.empty_sec / max(self.speech_sec, 1e-9)

    @property
    def unvoiced_frac(self) -> float:
        """Speech seconds no word occupies, wherever they sit.

        `empty_frac` counts whole intervals, so it is gamed by anything that merges
        a noise-only stretch into a neighbouring interval that does contain a word --
        raising MIN_NON_SPEECH_MS does exactly that, and the noise still reaches the
        decoder. This one counts the seconds themselves and cannot be merged away.
        """
        return self.unvoiced_sec / max(self.speech_sec, 1e-9)

    def line(self, name: str) -> str:
        d = self.empty_durs if self.empty_durs.size else np.array([0.0])
        return (f"{name:<26} n={self.n_intervals:>4d} "
                f"empty={self.empty_intervals:>4d} "
                f"({self.empty_intervals / max(self.n_intervals, 1):>5.1%}) "
                f"{self.empty_sec:>7.1f}s = {self.empty_frac:>5.1%} of speech "
                f"| dur med={np.median(d):>4.2f} max={d.max():>5.2f}")


def noise_intervals(speech: Sequence[Interval], word_map: Sequence[Tuple[float, float]],
                    *, min_overlap: float = 0.02) -> NoiseVerdict:
    """Intervals containing no word from the speech-presence map.

    `min_overlap` guards against a word that merely grazes the edge counting as
    occupancy -- 20 ms is under the shortest real word in any of the references.
    """
    speech = sorted(speech)
    wm = sorted(word_map)
    ws = np.array([w[0] for w in wm]) if wm else np.zeros(0)
    we = np.array([w[1] for w in wm]) if wm else np.zeros(0)

    merged = merge_spans(wm)
    ms = np.array([m[0] for m in merged]) if merged else np.zeros(0)
    me = np.array([m[1] for m in merged]) if merged else np.zeros(0)

    empty, empty_sec, durs, voiced_sec = 0, 0.0, [], 0.0
    for s, e in speech:
        if ws.size:
            lo = int(np.searchsorted(we, s, side="right"))
            hi = int(np.searchsorted(ws, e, side="left"))
            occupied = 0.0
            for k in range(lo, hi):
                occupied += max(0.0, min(float(we[k]), e) - max(float(ws[k]), s))
        else:
            occupied = 0.0
        if occupied < min_overlap:
            empty += 1
            empty_sec += e - s
            durs.append(e - s)
        if ms.size:
            lo = int(np.searchsorted(me, s, side="right"))
            hi = int(np.searchsorted(ms, e, side="left"))
            for k in range(lo, hi):
                voiced_sec += max(0.0, min(float(me[k]), e) - max(float(ms[k]), s))
    total = sum(e - s for s, e in speech)
    return NoiseVerdict(len(speech), empty, empty_sec, total,
                        np.array(durs) if durs else np.array([0.0]),
                        unvoiced_sec=max(0.0, total - voiced_sec))


def word_map_from(paths: Sequence, drop_tagged: bool = True) -> List[Tuple[float, float]]:
    """Union of word spans across several ASR runs, as a speech-presence map.

    Tagged segments (hallucination / filled pause / drift) are dropped by default:
    a hallucination is text the decoder invented from noise, so counting it as
    evidence of speech would make a noise interval look occupied.
    """
    import json
    from pathlib import Path

    from refs import BAD_TAGS, DRIFT_TAG

    spans: List[Tuple[float, float]] = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        segs = d["segments"] if isinstance(d, dict) else d
        for seg in segs:
            tags = seg.get("tags") or []
            if drop_tagged and (any(t in tags for t in BAD_TAGS) or DRIFT_TAG in tags):
                continue
            for w in (seg.get("words") or []):
                s, e = float(w["start"]), float(w["end"])
                if e > s and str(w.get("word", "")).strip():
                    spans.append((s, e))
    return sorted(spans)


def merge_spans(spans: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s, e in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out
