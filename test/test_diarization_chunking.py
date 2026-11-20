"""Chunk geometry for the Sortformer loop.

Two of these guard a defect that already happened: the loop bounded a step's
predictions by the tensor width rather than the valid length, which let the
embeddings outrun their predictions on the final short chunk. Nothing raised --
the speaker cache was updated with misaligned pairs -- until a step contained a
silence frame, which is why a five-minute file passed and a sixty-minute one
crashed.
"""

from __future__ import annotations

from finesub.speech.diarization.chunking import (
    ChunkWindow,
    SortformerGeometry,
    chunk_prediction_span,
    chunk_windows,
    pre_encoded_length,
)

GEOMETRY = SortformerGeometry()

#: ⚠ **The checkpoint's own geometry, pinned by value.** `SortformerGeometry`'s defaults are ours
#: to move -- `chunk_right_context` went to 80 on 2026-09-20 -- and the parity test below asks a
#: different question: does our windowing reproduce NeMo's loader *for the configuration NeMo was
#: run with*. Letting it inherit the module default silently repointed it at a geometry NeMo never
#: saw, and its widths went from [1512, 1520, 1448] to [2144, 2152, 1448] while still calling
#: itself a parity test. Parity is about the algorithm; what we choose to feed it is a separate
#: decision, and the two must not share a constant.
NEMO_GEOMETRY = SortformerGeometry(chunk_right_context=1)


def windows(total, valid=None, geometry=None):
    geometry = geometry or GEOMETRY
    return list(chunk_windows(total, valid if valid is not None else total, geometry))


def test_first_window_has_no_left_context_and_the_rest_do():
    got = windows(GEOMETRY.chunk_frames * 3)
    assert got[0].left == 0, "nothing precedes the first chunk"
    assert all(window.left == GEOMETRY.subsampling_factor for window in got[1:])


def test_last_window_has_no_right_context():
    got = windows(GEOMETRY.chunk_frames * 3)
    assert got[-1].right == 0
    # Read off the geometry, not written as `subsampling_factor`: those were the same number only
    # while `chunk_right_context` was 1, so the literal was a coincidence rather than a contract.
    lookahead = GEOMETRY.chunk_right_context * GEOMETRY.subsampling_factor
    assert all(window.right == lookahead for window in got[:-1])


def test_windows_tile_the_sequence_exactly():
    total = GEOMETRY.chunk_frames * 4 + 137
    got = windows(total)
    assert got[0].start == 0
    assert got[-1].end == total
    for earlier, later in zip(got, got[1:]):
        assert earlier.end == later.start, "chunks abut; only the context overlaps"


def test_a_300_second_file_matches_nemos_own_loader():
    # 300s -> 30001 frames, padded to 30016 by the front end's pad_to=16.
    got = windows(30016, valid=30000, geometry=NEMO_GEOMETRY)
    assert len(got) == 20
    assert [got[0].width, got[1].width, got[-1].width] == [1512, 1520, 1448]
    assert got[-1].valid_frames == 1432, "the tail of the window is feature padding"


def test_valid_frames_stops_at_the_audio_not_the_padding():
    got = windows(30016, valid=30000)
    assert all(window.valid_frames == window.width for window in got[:-1])
    assert got[-1].valid_frames < got[-1].width


def test_pre_encoded_length_matches_three_stride_two_convolutions():
    assert pre_encoded_length(1512, 8) == 189
    assert pre_encoded_length(1520, 8) == 190
    assert pre_encoded_length(1448, 8) == 181


def test_prediction_span_is_bounded_by_the_valid_length_not_the_width():
    # The final chunk of a 300s file: 181 frames wide, 179 of them audio.
    window = ChunkWindow(index=19, start=28576, end=30016, left=8, right=0, valid_frames=1432)
    left, right, length = chunk_prediction_span(window, 181, 179, GEOMETRY)
    assert (left, right) == (1, 0)
    assert length == 178, "width would give 180, and the two extra frames have no predictions"


def test_prediction_span_of_a_full_chunk_is_the_whole_chunk():
    window = ChunkWindow(index=1, start=1504, end=3008, left=8, right=8, valid_frames=1520)
    left, right, length = chunk_prediction_span(window, 190, 190, GEOMETRY)
    assert (left, right, length) == (1, 1, 188)
    assert length == GEOMETRY.chunk_len


def test_one_short_file_is_a_single_window():
    got = windows(640, valid=600)
    assert len(got) == 1
    assert (got[0].start, got[0].end, got[0].left, got[0].right) == (0, 640, 0, 0)
    assert got[0].valid_frames == 600


# -- long-file featurizing ----------------------------------------------------------------------
#
# The front end's intermediates cost 2.50 GiB per hour of audio, so an eleven-hour file cannot be
# featurized in one call. These pin the geometry that splits it. The failure they guard is silent:
# a segment whose frame grid sits a fraction of a hop off the whole file's still produces plausible
# features, and every one of them is wrong.

from finesub.speech.diarization.chunking import (  # noqa: E402
    FEATURE_HOP_SAMPLES,
    FEATURE_PAD_TO,
    feature_frame_counts,
    feature_segments,
)

#: (samples, total, valid) read off the exported front end itself, not derived.
MEASURED_FRAME_COUNTS = [
    (1, 16, 0), (159, 16, 0), (160, 16, 1), (161, 16, 1), (320, 16, 2),
    (16000, 112, 100), (16159, 112, 100), (16160, 112, 101), (16161, 112, 101),
    # 2560 samples are exactly 16 hops: 16 valid frames, but the centred STFT yields a 17th, so the
    # padding goes to 32, not 16. Deriving total from `valid` alone gets this wrong.
    (2560, 32, 16), (2561, 32, 16), (5065295, 31664, 31658), (480000, 3008, 3000), (480001, 3008, 3000),
]


def test_frame_counts_match_the_exported_front_end():
    for samples, total, valid in MEASURED_FRAME_COUNTS:
        assert feature_frame_counts(samples) == (total, valid), samples


def _long(hours=11.0):
    return int(hours * 3600 * 16000)


def test_segments_tile_the_valid_frames_with_no_gap_or_overlap():
    samples = _long()
    _, valid = feature_frame_counts(samples)
    frames = 0
    for segment in feature_segments(samples):
        assert segment.frame_start == frames, "every segment starts where the previous one stopped"
        frames += segment.frame_count
    assert frames == valid


def test_every_kept_frame_is_inside_its_own_calls_valid_frames():
    # A call zero-pads the frames past its own valid length; keeping one of those would write padding
    # into the middle of the file.
    samples = _long()
    for segment in feature_segments(samples):
        _, call_valid = feature_frame_counts(segment.sample_end - segment.sample_start)
        assert segment.keep_from + segment.frame_count <= call_valid, segment


def test_segment_frame_grid_sits_on_the_whole_files_grid():
    # Context that is not a multiple of the hop shifts every frame by a fraction: nothing matches.
    samples = _long() + 12345
    for segment in feature_segments(samples):
        assert segment.sample_start % FEATURE_HOP_SAMPLES == 0, segment
        assert segment.sample_start // FEATURE_HOP_SAMPLES + segment.keep_from == segment.frame_start, segment


def test_context_is_clamped_at_the_ends_of_the_file():
    samples = _long(1.0)
    segments = feature_segments(samples, segment_frames=60_000, context_frames=1_000)
    assert segments[0].sample_start == 0 and segments[0].keep_from == 0, "nothing to borrow before the start"
    assert segments[-1].sample_end == samples, "nothing to borrow after the end"
    assert all(s.keep_from == 1_000 for s in segments[1:]), "interior segments carry full left context"


def test_a_file_shorter_than_one_segment_is_one_call():
    segments = feature_segments(16000 * 60, segment_frames=60_000)
    assert len(segments) == 1 and segments[0].sample_start == 0


def test_segment_geometry_rejects_nonsense():
    import pytest

    with pytest.raises(ValueError):
        feature_segments(160_000, segment_frames=0)
    with pytest.raises(ValueError):
        feature_segments(160_000, context_frames=-1)


def test_padding_constant_is_what_the_front_end_pads_to():
    assert all(total % FEATURE_PAD_TO == 0 for _, total, _ in MEASURED_FRAME_COUNTS)
