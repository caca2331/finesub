"""What a cloud backend hands on: timed units grouped into speaker turns.

A *unit* is a word where the provider times words (35t, MAI2) and a segment
where it only times segments (MOSS Pro). A *turn* is one speaker's unbroken run
in the provider's output order; turns of different speakers may overlap in
time, which is the point of keeping them apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Unit:
    start: float
    end: float
    text: str
    speaker: str | None

    def as_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text, "speaker": self.speaker}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Unit":
        return cls(float(data["start"]), float(data["end"]), str(data.get("text", "")), data.get("speaker"))


@dataclass(frozen=True)
class Turn:
    speaker: str
    units: tuple[Unit, ...]

    @property
    def start(self) -> float:
        return min(u.start for u in self.units)

    @property
    def end(self) -> float:
        return max(u.end for u in self.units)


#: The key a unit without a speaker label gets (every cloud row has a key; `docs/joint-asr.md`).
UNKNOWN_SPEAKER = "unknown"


@dataclass
class Transcript:
    provider: str
    model: str
    duration: float
    #: "word" (35t, MAI2) or "segment" (MOSS Pro and local MOSS: a turn is one segment).
    unit: str
    turns: list[Turn]
    #: What the backend did beyond one plain call (windows, continuations, re-requests, repairs).
    log: list[dict[str, Any]] = field(default_factory=list)

    def units(self) -> list[Unit]:
        """Every unit, in output order (turn by turn)."""
        return [u for turn in self.turns for u in turn.units]


def turns_from_units(units: Iterable[Unit]) -> list[Turn]:
    """Consecutive units of one speaker (in the order given) as one turn; a unit
    without a speaker gets `UNKNOWN_SPEAKER`."""

    turns: list[Turn] = []
    run: list[Unit] = []
    for unit in units:
        speaker = unit.speaker if unit.speaker is not None else UNKNOWN_SPEAKER
        if run and speaker != (run[0].speaker if run[0].speaker is not None else UNKNOWN_SPEAKER):
            turns.append(Turn(_speaker(run), tuple(run)))
            run = []
        run.append(unit)
    if run:
        turns.append(Turn(_speaker(run), tuple(run)))
    return turns


def segment_turns(segments: Sequence[Unit]) -> list[Turn]:
    """One turn per segment (MOSS Pro's segments are already turns)."""
    return [Turn(_speaker([s]), (s,)) for s in segments]


def _speaker(run: Sequence[Unit]) -> str:
    return run[0].speaker if run[0].speaker is not None else UNKNOWN_SPEAKER
