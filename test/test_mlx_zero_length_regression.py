from copy import deepcopy

from finesub.speech.recognition.segments import normalize_segment_timeline


def test_cross_segment_zero_length_word_is_contained():
    original = [
        {"start": 1., "end": 2., "words": [{"word": "a", "start": 1.8, "end": 2.}]},
        {"start": 2., "end": 4., "alignment_events": [{"type": "evidence"}],
         "words": [{"word": "b", "start": 1.5, "end": 1.5, "confidence": .8},
                   {"word": "c", "start": 2.5, "end": 3.}]},
    ]
    snapshot = deepcopy(original)
    fixed, repairs = normalize_segment_timeline(original)
    assert fixed[1]["words"][0]["start"] == fixed[1]["words"][0]["end"] == 2.
    assert fixed[1]["words"][0]["confidence"] == .8
    assert fixed[1]["alignment_events"] == original[1]["alignment_events"]
    assert original == snapshot
    assert len(repairs) == 1
    assert normalize_segment_timeline(fixed)[0] == fixed
