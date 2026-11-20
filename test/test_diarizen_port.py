"""The DiariZen port's pure-numpy halves, pinned where they fail silently.

Everything neural in the port is checked by `tools/diarizen_export/check_*.py`, which need a GPU
and the frozen research caches and so cannot run here. What CAN run here is the rest of it -- the
mask arithmetic, the PLDA transforms, the assignment, the overlap-add -- and that is also where
the module docstrings say the traps are. Each one below is one of those sentences, asserted.

⚠ These are not equivalence tests. The port's agreement with the research path is a measurement
(`docs/diarization-findings.md` 5.3), not a property a unit test can hold. What these hold is the
narrower thing: the arithmetic still does what the comments say it does.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from finesub.speech.diarization import diarizen_assemble as assembler
from finesub.speech.diarization import diarizen_cluster as clusterer
from finesub.speech.diarization import diarizen_embed as embedder
from finesub.speech.diarization import diarizen_runner as runner

#: scipy is an `[asr]` dependency; the CI lanes install `[harness,dev]` only.
#: The clustering and the median filter import it inside the function, so
#: only the tests that reach them need it -- the rest of the file still runs.
needs_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None, reason="needs scipy (the [asr] extra)"
)


# --- powerset decoding -------------------------------------------------------------------------

def test_powerset_mapping_is_size_major_and_starts_with_silence():
    """The class order is the model's, not ours: silence, then singletons, then pairs."""
    mapping = runner.powerset_mapping().numpy()

    # Every subset of four speakers: 1 + 4 + 6 + 4 + 1.
    assert mapping.shape == (16, runner.MAX_SPEAKERS_PER_FRAME)
    assert not mapping[0].any(), "class 0 is silence"
    assert [int(row.sum()) for row in mapping] == [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4]
    assert [tuple(np.flatnonzero(row)) for row in mapping[5:11]] == [
        (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
    ], "within a size, itertools.combinations order"


def test_to_multilabel_takes_a_hard_argmax_not_a_threshold():
    """⚠ The whole chain's sensitivity lives here: one winning class, then its speaker set.

    A near-tie does not produce a blend -- it produces whichever class won by the last bit, which
    is why a 1e-3 difference upstream becomes a different speaker downstream.
    """
    import torch

    mapping = runner.powerset_mapping()
    logits = torch.zeros((1, 2, 11))
    logits[0, 0, 1] = 1.0            # frame 0: speaker 0 alone
    logits[0, 1, 5] = 0.5            # frame 1: speakers 0 and 1, and only just
    logits[0, 1, 1] = 0.4999

    out = runner.to_multilabel(logits, mapping).numpy()
    assert list(out[0, 0]) == [1, 0, 0, 0]
    assert list(out[0, 1]) == [1, 1, 0, 0], "the winning class is taken whole, not thresholded"


@needs_scipy
def test_median_filter_removes_a_single_frame_flicker():
    """It sits AFTER the cached segmentation and BEFORE everything downstream."""
    data = np.zeros((1, 21, 2), dtype=np.float32)
    data[0, :, 0] = 1.0
    data[0, 10, 0] = 0.0             # one frame dropped out of a steady speaker
    data[0, 10, 1] = 1.0             # and one frame of a speaker who is never there otherwise

    filtered = runner.median_filtered(data)
    assert filtered[0, 10, 0] == 1.0, "a one-frame hole is filled"
    assert filtered[0, 10, 1] == 0.0, "a one-frame spike is removed"


# --- the embedding masks -----------------------------------------------------------------------

def test_overlap_is_excluded_but_not_when_that_leaves_nothing():
    """⚠ Without the fallback, anyone who only ever speaks over somebody embeds near-silence."""
    frames = 800
    data = np.zeros((2, frames, 4), dtype=np.float32)
    # Chunk 0: speaker 0 talks alone for most of it, overlapping speaker 1 at the start.
    data[0, :, 0] = 1.0
    data[0, :100, 1] = 1.0
    # Chunk 1: speaker 2 is ONLY ever overlapped, so cleaning would leave it empty.
    data[1, :, 2] = 1.0
    data[1, :, 3] = 1.0

    masks = embedder.chunk_masks(data)
    assert masks[0, 50, 0] == 0.0, "overlapped frames are dropped for a speaker who has others"
    assert masks[0, 500, 0] == 1.0
    assert masks[1, :, 2].sum() == frames, "a wholly-overlapped speaker keeps its full mask"


def test_crop_pads_the_last_chunk_instead_of_returning_a_short_one():
    """The embedder must see the duration the mask describes, even past the end of the file."""
    import torch

    waveform = torch.ones((1, 1000))
    piece = embedder.crop_chunk(waveform, 2, window_size=500, step_size=400)
    assert piece.shape == (1, 500)
    assert float(piece[0, :200].sum()) == 200.0, "the real samples are kept"
    assert float(piece[0, 200:].sum()) == 0.0, "and the rest is zero, not wrapped or repeated"


# --- clustering --------------------------------------------------------------------------------

@needs_scipy
def test_plda_dimensions_come_back_largest_variance_first(tmp_path):
    """⚠ The eigen solve reorders, and the reversal puts it back. Skipping it keeps every shape."""
    rng = np.random.default_rng(0)
    dimension = 6
    transform = rng.normal(size=(dimension, dimension))
    np.savez(tmp_path / "xvec_transform.npz",
             mean1=np.zeros(dimension), mean2=np.zeros(dimension), lda=np.eye(dimension))
    np.savez(tmp_path / "plda.npz", mu=np.zeros(dimension), tr=transform,
             psi=np.linspace(1.0, 4.0, dimension))

    _, _, psi = clusterer.load_plda(tmp_path)
    assert np.all(np.diff(psi) <= 0), "psi must be descending, or every later cut takes the wrong axes"


@needs_scipy
def test_constrained_argmax_never_gives_one_chunk_two_speakers_the_same_identity():
    """Plain argmax would; two local speakers in one chunk are by construction different people."""
    soft = np.array([[[0.9, 0.8, 0.1],
                      [0.85, 0.2, 0.1]]])       # both prefer cluster 0
    hard = clusterer.constrained_argmax(soft)
    assert len(set(hard[0])) == 2, "the assignment separates them"
    assert 0 in hard[0], "and still gives the strongest claim its first choice"


def test_embedding_filter_drops_the_frame_requirement_rather_than_returning_nothing():
    """Short or wholly-overlapped material must still cluster, not fall off the end."""
    chunks, frames, speakers = 2, 100, 4
    segmentations = np.zeros((chunks, frames, speakers), dtype=np.float32)
    # Nobody has 10% of a chunk to themselves: every active frame is shared.
    segmentations[:, :5, 0] = 1.0
    segmentations[:, :5, 1] = 1.0
    embeddings = np.zeros((chunks, speakers, 8), dtype=np.float32)
    embeddings[:, 0] = 1.0
    embeddings[:, 1] = 2.0
    embeddings[:, 2:] = np.nan

    kept, chunk_idx, _ = clusterer.filter_embeddings(embeddings, segmentations)
    assert len(kept) == 4, "the two active speakers in both chunks survive the fallback"
    assert not np.isnan(kept).any(), "and NaN rows never do"
    assert sorted(chunk_idx) == [0, 0, 1, 1]


def test_there_is_no_way_to_ask_vbx_for_the_path_that_was_not_ported():
    """`loopProb > 0` takes a forward-backward branch nobody checked.

    ⚠ The guard is the SIGNATURE, not a check inside: the parameter does not exist, so a caller
    reaching for the HMM branch gets a TypeError at the call rather than the GMM answer dressed up
    as an HMM one. Asserted because a later "helpfully" added `**kwargs` would remove it silently.
    """
    import inspect

    parameters = inspect.signature(clusterer.vbx).parameters
    assert "loop_prob" not in parameters and "loopProb" not in parameters
    kinds = [p.kind for p in parameters.values()]
    # **kwargs here would swallow loopProb and silently return the GMM answer.
    assert inspect.Parameter.VAR_KEYWORD not in kinds


@needs_scipy
def test_unassignable_local_speakers_are_renumbered_the_way_the_original_does():
    """⚠ When there are fewer clusters than local speakers, `-2` becomes cluster 0.

    `constrained_argmax` leaves `-2` for a speaker it could not place, and the `np.unique`
    renumbering that follows maps it to 0 and shifts the real clusters up. That is surprising, and
    it is **faithful**: the upstream does exactly this (`pyannote/audio/pipelines/clustering.py`,
    "re-number clusters from 0 to num_large_clusters"). Pinned so nobody "fixes" it into a
    divergence, and because every equivalence file measured so far had 5-23 clusters, meaning this
    branch -- the one few-speaker material takes -- was never exercised by any of them.
    """
    soft = np.zeros((2, 4, 2))
    soft[:, 0] = [0.9, 0.1]
    soft[:, 1] = [0.1, 0.9]
    soft[:, 2] = [0.5, 0.5]
    soft[:, 3] = [0.4, 0.4]

    hard = clusterer.constrained_argmax(soft)
    assert (hard == -2).any(), "two clusters cannot absorb four local speakers"

    _, renumbered = np.unique(hard, return_inverse=True)
    renumbered = renumbered.reshape(hard.shape)
    assert renumbered.min() == 0 and not (renumbered == -2).any()


def test_inactive_local_speakers_are_masked_before_grouping():
    """The step `SpeakerDiarization.apply` does just before `reconstruct`, which we lacked.

    ⚠ It is inert on real material (verified bit-identical on three files) but it is what makes
    the `-2` branch in `assemble` reachable at all: the sentinel `constrained_argmax` leaves has
    already been renumbered away by then, so without this masking nothing is ever skipped.
    """
    chunks, frames = 2, 800
    segmentations = np.zeros((chunks, frames, 3), dtype=np.float32)
    segmentations[0, :, 0] = 1.0      # only speaker 0 is ever active
    hard = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int8)

    timeline = assembler.assemble(segmentations, hard)
    # ⚠ The column DISAPPEARS rather than being kept and empty: the cluster count is taken after
    # the masking, which is where the original takes it too (`reconstruct` runs after
    # `hard_clusters[inactive_speakers] = -2`). A cluster that lives only on inactive speakers is
    # not a speaker. On real material nothing is lost this way -- verified bit-identical on three
    # files, all of which kept their full column count.
    assert timeline.shape[1] == 1, "only the cluster with a live speaker survives"
    assert timeline[:, 0].sum() > 0


@needs_scipy
def test_clustering_survives_material_with_nothing_to_cluster():
    """Silence must not hand back a NaN centroid with a RuntimeWarning beside it."""
    import warnings

    segmentations = np.zeros((2, 100, 4), dtype=np.float32)
    embeddings = np.full((2, 4, 8), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = clusterer.cluster(embeddings, segmentations, "unused")
    assert result.hard.shape == (2, 4) and not result.hard.any()
    assert result.centroids.shape == (0, 8), "shaped, not a NaN pretending to be an answer"
    assert len(result.centroid_of_track) >= 1, "and the mapping is still well formed"
    assert result.single_speaker_log_bayes == 0.0, "nothing to compare against here"


def test_silence_assembles_into_an_empty_timeline_rather_than_raising():
    """⚠ 'Nobody spoke' is an answer, and it used to be a `ValueError` that killed the task.

    With every slot masked, `max(hard_clusters)` is the `-2` sentinel and the cluster count came
    out `-1`, which `np.full` rejects. `_run_diarization_stage` catches only
    `DiarizationUnavailable`, so an instrumental -- or any vocal track the separator emptied --
    took the whole run down AFTER the ASR had finished, which is the one thing this stage
    promises not to do.
    """
    segmentations = np.zeros((1, 799, 4), dtype=np.float32)
    hard = np.zeros((1, 4), dtype=np.int8)

    timeline = assembler.assemble(segmentations, hard)
    assert timeline.shape[1] == 0, "no speakers, and that is the answer"
    assert timeline.shape[0] > 0, "the frame axis still says how long the file was"
    # ⚠ The whole downstream has to accept it too, which is the part a unit test on `assemble`
    # alone would miss: the segmenter and the sample picker each index that axis.
    from finesub.speech.diarization.samples import pick_samples
    from finesub.speech.diarization.segments import segments_from_activity

    assert list(segments_from_activity(timeline.tolist(), frame_seconds=0.02)) == []
    assert pick_samples(timeline, step=0.02) == []


def test_an_empty_waveform_assembles_without_inventing_a_frame_axis():
    """Zero chunks: the aggregation's frame total goes negative before anything else can fail."""
    assert assembler.assemble(np.zeros((0, 799, 4), dtype=np.float32),
                              np.zeros((0, 4), dtype=np.int8)).shape == (0, 0)


# --- assembly ----------------------------------------------------------------------------------

def test_aggregate_treats_nan_as_absent_rather_than_as_zero():
    """⚠ 'This chunk said nothing here' is not 'this chunk said silence here'.

    ⚠ Asserted on the AVERAGING path, because that is the only place the two differ. With
    `skip_average=True` a NaN read as 0 adds 0 and changes nothing, so a test there would pass
    whether the masking existed or not. Dividing is what exposes it: a NaN counted as a real
    observation inflates the denominator, and the speaker count -- which is exactly that call --
    comes out halved.
    """
    per_chunk = 800
    scores = np.full((2, per_chunk, 1), np.nan, dtype=np.float32)
    scores[0, :, 0] = 1.0            # chunk 0 speaks throughout
    # chunk 1 stays NaN: it has no opinion, and must not become a vote for silence.

    step = assembler.SEGMENTATION_DURATION * assembler.SEGMENTATION_STEP_RATIO
    out = assembler.aggregate(scores, chunk_step=step, skip_average=False, missing=0.0)

    overlap = assembler.closest_frame(step + 0.5 * assembler.FRAME_DURATION)
    assert out[overlap, 0] == 1.0, (
        "where both chunks cover the frame but only one has an opinion, the average is over one "
        "observation -- reading NaN as 0.0 would give 0.5 here"
    )


def test_to_diarization_keeps_exactly_the_counted_speakers_even_on_a_near_tie():
    """A frame that counts one speaker gets one, even where two are nearly tied."""
    frames = 4
    clustered = np.zeros((1, frames, 3), dtype=np.float32)
    clustered[0, :, 0] = 0.9
    clustered[0, :, 1] = 0.89        # very close, and still must lose
    clustered[0, :, 2] = 0.1
    count = np.ones((assembler.closest_frame(
        assembler.SEGMENTATION_DURATION + 0.5 * assembler.FRAME_DURATION) + 1, 1), dtype=np.uint8)

    binary = assembler.to_diarization(clustered, count)
    speaking = binary.sum(axis=1)
    assert set(np.unique(speaking)) <= {0, 1}, "never more than the count asks for"
    start = assembler.closest_frame(0.5 * assembler.FRAME_DURATION)
    assert list(binary[start]) == [1, 0, 0]


def test_closest_frame_lands_on_the_nearest_frame_centre():
    """⚠ The half-frame in the caller is what makes this true; it is not a fudge factor."""
    for index in (0, 1, 17, 1000):
        centre = assembler.FRAME_START + index * assembler.FRAME_STEP \
            + 0.5 * assembler.FRAME_DURATION
        assert assembler.closest_frame(centre) == index


@needs_scipy
def test_merge_suggestions_point_the_small_track_at_the_big_one():
    """"Smaller" is the whole semantics, and track ids carry no information about size.

    ⚠ Ids come from the `np.unique` renumbering of cluster ids, so ordering by them -- which this
    function did until 2026-09-20 -- picks arbitrary merge targets and lets a track that barely
    speaks serve as an established one. The big track is deliberately given the LAST id here, so
    any ordering that is really the id order fails.
    """
    import numpy as np

    from finesub.speech.diarization.diarizen_cluster import merge_suggestions

    # ⚠ The big track gets the LAST id, so an ordering that is really the id order fails.
    centroids = np.array([[1.0, 0.02], [1.0, 0.01], [1.0, 0.0]])
    sizes = np.array([1, 2, 1000])

    found = merge_suggestions(centroids, np.array([0, 1, 2], dtype=np.int16), sizes)
    pointed = {hint["track"]: hint["like"] for hint in found}

    assert 2 not in pointed, "the largest track has nothing bigger to fold into"
    assert pointed, "the two small tracks are candidates"
    for small, big in pointed.items():
        assert sizes[big] > sizes[small], (
            f"track {small} ({sizes[small]} frames) was told to fold into {big} "
            f"({sizes[big]} frames) -- a suggestion must point at a LARGER track")


@needs_scipy
def test_merge_suggestions_keep_holds_the_largest_tracks_not_the_lowest_ids():
    """`keep=1` means the one track that speaks most, whatever number it was given."""
    import numpy as np

    from finesub.speech.diarization.diarizen_cluster import merge_suggestions

    centroids = np.array([[1.0, 0.02], [1.0, 0.01], [1.0, 0.0]])
    found = merge_suggestions(centroids, np.array([0, 1, 2], dtype=np.int16),
                              np.array([5, 3, 900]), keep=1)

    assert {hint["track"] for hint in found} == {0, 1}
    assert {hint["like"] for hint in found} == {2}
