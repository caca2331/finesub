"""Builders for the correction's speaker turns: labelled stables, one planned window, a reply in csv.

The turns' own tests are `test_llm_speaker_turns.py`; grouping, three passes, overlap and the
proofread build their inputs with these too.
"""

from __future__ import annotations

import dataclasses
import json

from finesub.llm.chunking import plan_correction_windows
from finesub.subtitles.stable_rows import overlay_for, write_overlay

from .fakes import FakeTokenCounter

HEADER = "type|position|corrected_text|translation|conf|note"


def speaker(track, block=0, **extra):
    return {"track": track, "tracks": [track], "block": block, "coverage": {str(track): 0.9}, "labels": [], **extra}


def labelled_stable(tmp_path, speakers, texts=None):
    """A stable whose lines carry `speakers` (None: no track) through a rows file, as the diarization stage writes."""
    lines = [{"id": str(index), "start": index * 2.0, "end": index * 2.0 + 1.0,
              "text": (texts[index - 1] if texts else f"line {index}")}
             for index in range(1, len(speakers) + 1)]
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": lines}), encoding="utf-8")
    write_overlay(stable, overlay_for(lines, [{"speaker": s} if s is not None else {} for s in speakers],
                                      speaker={"marks": None}))
    return stable


def window_of(segments):
    return dataclasses.replace(plan_correction_windows(segments, counter=FakeTokenCounter())[0], length_scale=1.0)


def reply(*rows):
    return "<translated>\n" + HEADER + "\n" + "\n".join(rows) + "\n</translated>"


def grouped_reply(*rows):
    """An answer in the six columns three passes' pass 2 writes (`capableC-grouped`), as capableC does."""
    return reply(*rows)
