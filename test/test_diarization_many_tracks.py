"""The artifact side with more than four tracks -- the precondition for a second backend.

Everything downstream of the timeline was written against Sortformer, which emits exactly four
tracks on every file. A clustering backend emits as many as it finds: DiariZen produced 5 to 23 on
the same material, and 13 on one four-person episode. Before any of it can be wired in, the
artifacts have to mean something at those counts.

These are contract tests, not quality ones. They ask whether an export stays readable and whether
a prompt still shows the model every track -- not whether the tracks are right.
"""

from __future__ import annotations

from finesub.speech.diarization.export import (
    check_completeness,
    group_by_speaker,
    render_ass,
    render_speaker_srts,
)
from finesub.speech.diarization.labels import SegmentSpeaker
from finesub.speech.diarization.segments import TrackSegment
from finesub.speech.diarization.stage import timeline_hints

MANY = 13


def line(start, end, text="x"):
    return {"start": start, "end": end, "text": text}


def on_track(track):
    return SegmentSpeaker(track=track, extras={"tracks": [track]})


def many_track_scene(tracks=MANY):
    """One line per track, back to back."""
    subtitles = [line(i, i + 1, f"line {i}") for i in range(tracks)]
    return subtitles, [on_track(i) for i in range(tracks)]


# --- the exports ------------------------------------------------------------------------

def test_every_line_still_reaches_a_track_at_thirteen():
    subtitles, speakers = many_track_scene()
    groups = group_by_speaker(subtitles, speakers)
    assert check_completeness(subtitles, groups) == []
    assert len(render_speaker_srts(subtitles, speakers, empty_cues=False)) == MANY


def test_the_ass_gives_every_track_its_own_colour():
    """The one reason this file exists is to tell speakers apart on screen.

    The palette was five colours picked by hand for a four-track model, indexed modulo its length,
    so the sixth track repeated the first. Two speakers sharing a colour is not a cosmetic problem here:
    it silently undoes what the export is for.
    """
    subtitles, speakers = many_track_scene(MANY)
    styles = [row for row in render_ass(subtitles, speakers).splitlines() if row.startswith("Style: ")]
    assert len(styles) == MANY
    colours = [row.split(",")[3] for row in styles]
    assert len(set(colours)) == MANY, f"{MANY - len(set(colours))} tracks share a colour with another"


def test_the_first_colours_are_unchanged_for_four_tracks():
    """Widening the palette must not restyle what Sortformer already produces."""
    subtitles, speakers = many_track_scene(4)
    styles = [row for row in render_ass(subtitles, speakers).splitlines() if row.startswith("Style: ")]
    colours = [row.split(",")[3] for row in styles]
    assert colours == ["&H00FFFFFF", "&H0080FFFF", "&H00FF8080", "&H0080FF80"]


# --- the run record ---------------------------------------------------------------------

def test_the_hints_survive_more_than_four_tracks():
    """`fourth_track_share` is defined off the four busiest tracks; more tracks must not break it."""
    tracks = [TrackSegment(i, float(i), float(i) + 1.0) for i in range(MANY)]
    hints = timeline_hints(tracks)
    assert 0.0 <= hints["fourth_track_share"] <= 1.0
    # Every unordered pair is reported, not just the pairs among the first four.
    assert len(hints["co_activity"]) == MANY * (MANY - 1) // 2


def test_the_hints_do_not_pretend_a_fourth_track_exists():
    tracks = [TrackSegment(i, float(i), float(i) + 1.0) for i in range(2)]
    assert timeline_hints(tracks)["fourth_track_share"] == 0.0
