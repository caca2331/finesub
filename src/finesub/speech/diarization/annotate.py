"""Speaker labels as `-diar-rows.json` writes them: one entry per stable line, plus the run record.

Owner decision (2026-10-15): the labels live beside `-stable.json`, in `-diar-rows.json`, not on its
segments -- every task file is written once by its own step (`subtitles/stable_rows.py` has the format).
These are pure transforms; `stage.write_rows` puts them in the file.

Two rules, both consequences of decisions recorded elsewhere:

* A line with nothing to say gets an entry with **no `speaker` key at all**, rather than an empty one.
  Absence is the documented way to say "not labelled" (labels.py), and writing `{}` as the speaker would
  make every reader decide for itself what that means.
* The run-level record says which model produced the labels. Its conservative `reproducible: False`
  means that regeneration across numeric paths is not guaranteed, not that identical-input repeats are
  random (`docs/diarization-findings.md` 5.1).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .labels import SegmentSpeaker


def entry(speaker: SegmentSpeaker) -> dict[str, Any]:
    """One line's entry: `{"speaker": ...}`, or `{}` for a line with nothing to say."""
    # A filled line has no `track` but a `mark` naming one (`marks.py`): it is written too.
    if speaker.is_labelled or speaker.track is not None or "mark" in speaker.extras:
        return {"speaker": speaker.to_dict()}
    return {}


def entries(speakers: Sequence[SegmentSpeaker]) -> list[dict[str, Any]]:
    """One entry per line, in order."""
    return [entry(speaker) for speaker in speakers]


def speaker_metadata(
    *,
    model: str,
    artifacts_revision: str,
    track_count: int,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """The run-level record that travels with the labels.

    `reproducible: False` is not boilerplate. Measured on 61.6 minutes: scaling
    the input by 1+1e-7 -- inaudible, far under any recording's noise floor --
    can change per-subtitle assignments because the speaker cache keeps frames
    by a discrete top-k. Same-input, same-path repeats were bit-exact in later
    tests; a changed input/path can change quality, not just arbitrary IDs.
    This field does not promise equivalence across such changes. Task artifacts
    are reused rather than silently regenerated with today's numerical path.
    """
    return {
        "model": model,
        "artifacts_revision": artifacts_revision,
        "track_count": track_count,
        "parameters": dict(parameters),
        "reproducible": False,
    }
