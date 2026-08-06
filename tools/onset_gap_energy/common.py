"""Shared loading helpers for the onset gap/energy exploration.

The gold annotations are tracked (`tools/wt_refine_validation/disfluency_gold.json`);
the audio, VAD track and source runs are local-only, so every path is an argument.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

TARGET_SUBSETS = ("segment-boundary", "after-gap")


@dataclass
class Candidate:
    index: int
    start: float
    end: float
    duration: float
    label: str
    position: str
    onset: float
    onset_fraction: float
    preceding_gap: float
    preceding_word: str
    next_word: str
    reported_start: float          # disfluency run (= onset for filled_pause)
    plain_start: float             # production run (disfluency off) -> the value to fix
    segment: int
    note: Optional[str]

    @property
    def error(self) -> float:
        """Production start error, signed. Positive = production is too early."""
        return self.onset - self.plain_start


def load_candidates(gold_path: Path) -> List[Candidate]:
    data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    out: List[Candidate] = []
    for i, c in enumerate(data["disfluency_candidates"]):
        out.append(
            Candidate(
                index=i,
                start=c["start"],
                end=c["end"],
                duration=c["duration"],
                label=c["label"],
                position=c["position"],
                onset=c["onset"],
                onset_fraction=c["onset_fraction"],
                preceding_gap=c["preceding_gap"],
                preceding_word=c["preceding_word"],
                next_word=c["next_word"],
                reported_start=c["next_word_reported_start"],
                plain_start=c["next_word_start_without_disfluencies"],
                segment=c["segment"],
                note=c.get("annotator_note"),
            )
        )
    return out


def load_vad(vad_path: Path) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]:
    data = json.loads(Path(vad_path).read_text(encoding="utf-8"))
    speech = [(float(a), float(b)) for a, b in data["speech"]]
    non_speech = [(float(a), float(b)) for a, b in data["non_speech"]]
    return speech, non_speech, float(data["duration"])


def summarize(errors: Sequence[float], label: str = "") -> str:
    a = np.abs(np.asarray(list(errors), dtype=float))
    if a.size == 0:
        return f"{label}: (empty)"
    return (
        f"{label:<28} n={a.size:3d} med={np.median(a):.3f} mean={a.mean():.3f} "
        f"p90={np.quantile(a, 0.9):.3f} p95={np.quantile(a, 0.95):.3f} "
        f"max={a.max():.3f} >0.1s={np.mean(a > 0.1):.0%} >0.3s={np.mean(a > 0.3):.0%}"
    )
