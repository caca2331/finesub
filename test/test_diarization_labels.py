"""The speaker label carried on a stable.json segment.

Most of these guard one distinction: "we did not label this" is not "the speaker
changed". A subtitle editor that confuses the two invents speaker turns out of
segments the acoustic layer deliberately declined to judge -- and declining is
common by design.
"""

from __future__ import annotations

import pytest

from finesub.speech.diarization.labels import (
    UNLABELLED,
    SegmentSpeaker,
    SpeakerLabel,
    SpeakerLabelError,
    same_speaker,
    speaker_of,
)


def test_a_segment_without_a_speaker_field_is_unlabelled():
    assert speaker_of({"start": 0.0, "end": 1.0, "text": "hi"}) is UNLABELLED
    assert UNLABELLED.is_labelled is False


def test_unlabelled_against_anything_is_unknown_not_different():
    labelled = SegmentSpeaker(labels=(SpeakerLabel("streamer", "high"),), track=0)
    assert same_speaker(UNLABELLED, labelled) is None, "must not read as a speaker change"


def test_the_same_track_in_the_same_block_is_the_same_speaker():
    first = SegmentSpeaker(track=1, block=3)
    second = SegmentSpeaker(track=1, block=3)
    assert same_speaker(first, second) is True


def test_a_different_track_in_the_same_block_is_a_different_speaker():
    assert same_speaker(SegmentSpeaker(track=0, block=3), SegmentSpeaker(track=1, block=3)) is False


def test_the_same_track_across_blocks_is_undecidable():
    first = SegmentSpeaker(track=1, block=3)
    second = SegmentSpeaker(track=1, block=4)
    assert same_speaker(first, second) is None, "ids only mean something inside a block"


def test_a_name_beats_the_track_id_across_blocks():
    first = SegmentSpeaker(labels=(SpeakerLabel("guest", "high", name="miyako"),), track=1, block=3)
    second = SegmentSpeaker(labels=(SpeakerLabel("guest", "median", name="miyako"),), track=2, block=9)
    assert same_speaker(first, second) is True, "identity travels as a name, not an id"


def test_different_names_are_different_people_even_in_one_block():
    first = SegmentSpeaker(labels=(SpeakerLabel("guest", "high", name="a"),), track=1, block=1)
    second = SegmentSpeaker(labels=(SpeakerLabel("guest", "high", name="b"),), track=1, block=1)
    assert same_speaker(first, second) is False


def test_a_segment_may_carry_several_labels():
    speaker = SegmentSpeaker(
        labels=(
            SpeakerLabel("streamer", "high"),
            SpeakerLabel("other", "median", name="anime line"),
        )
    )
    assert len(speaker.to_dict()["labels"]) == 2


def test_round_trip_keeps_everything_including_unknown_evidence_fields():
    speaker = SegmentSpeaker(
        labels=(SpeakerLabel("guest", "median", name="miyako", src="llm+human"),),
        track=2,
        block=1,
        review=True,
        extras={"dist": 0.51, "margin": 0.19},
    )
    again = SegmentSpeaker.from_dict(speaker.to_dict())
    assert again == speaker


def test_unknown_is_a_valid_role():
    assert SpeakerLabel("unknown", "low").role == "unknown"


@pytest.mark.parametrize("role, conf, src", [("host", "high", "acoustic"), ("streamer", "very-high", "acoustic"),
                                             ("streamer", "high", "llm+vibes")], ids=["role", "conf", "src"])
def test_a_bad_field_is_rejected(role, conf, src):
    with pytest.raises(SpeakerLabelError):
        SpeakerLabel(role, conf, src=src)


def test_combined_sources_are_allowed():
    assert SpeakerLabel("streamer", "high", src="acoustic+llm+human").src == "acoustic+llm+human"


def test_a_label_missing_its_role_says_so():
    with pytest.raises(SpeakerLabelError, match="role"):
        SpeakerLabel.from_dict({"conf": "high"})


def test_a_speaker_field_of_the_wrong_shape_is_an_error_not_unlabelled():
    with pytest.raises(SpeakerLabelError):
        speaker_of({"speaker": ["streamer"]})
