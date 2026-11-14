"""Speaker labels as `-diar-rows.json` entries.

The one that matters: an unlabelled line's entry carries no `speaker` key at all (a reader must not have
to guess what an empty one means).
"""

from __future__ import annotations

from finesub.speech.diarization.annotate import entries, entry, speaker_metadata
from finesub.speech.diarization.labels import SegmentSpeaker, SpeakerLabel, speaker_of

LABELLED = SegmentSpeaker(labels=(SpeakerLabel("streamer", "high"),), track=0, block=1)
PLACED = SegmentSpeaker(track=2, block=1)
UNPLACED = SegmentSpeaker()


def test_a_labelled_line_gets_a_speaker_object():
    speaker = entry(LABELLED)["speaker"]
    assert speaker["labels"][0]["role"] == "streamer"
    assert speaker["track"] == 0


def test_an_unlabelled_line_gets_no_key_at_all():
    assert entry(UNPLACED) == {}, "absence is how the schema says 'not labelled'"


def test_a_track_without_a_role_is_still_worth_writing():
    # The acoustic layer places a line long before anyone names the person.
    speaker = entry(PLACED)["speaker"]
    assert speaker["track"] == 2
    assert speaker["labels"] == []


def test_a_filled_line_is_written():
    filled = SegmentSpeaker(extras={"mark": {"tier": "filled", "track": 3, "by": "slot"}})
    speaker = entry(filled)["speaker"]
    assert speaker["mark"]["track"] == 3 and "track" not in speaker
    assert speaker_of({"speaker": speaker}) == filled


def test_one_entry_per_line_in_order():
    assert [bool(e) for e in entries([LABELLED, UNPLACED, PLACED])] == [True, False, True]


def test_metadata_records_that_the_timeline_is_not_reproducible():
    metadata = speaker_metadata(
        model="sortformer-4spk-v2.1", artifacts_revision="abc123", track_count=4, parameters={"enter": 0.5}
    )
    assert metadata["reproducible"] is False, "same audio, different hardware, ~7% of labels move"
    assert metadata["model"] == "sortformer-4spk-v2.1"
