"""Chunk-level decisions -> one frame-level timeline: overlap-add, speaker count, top-k.

The last stage of the port. Clustering has said which global speaker each (chunk, local speaker)
belongs to; this turns 3052 overlapping 16 s chunks into one `(frames, speakers)` map, which is
what `segments_from_activity` downstream already understands.

The subtleties, all of them silent when wrong:

* **The frame grid is the model's receptive field, not the chunk window.** Frames are 20 ms apart
  with a 25 ms duration and start at -0.0075 s; chunks are 16 s apart by 1.6 s. Mixing the two up
  produces a timeline of plausible length that is offset.
* **A chunk lands at `closest_frame(chunk.start + 0.5 * frame.duration)`.** The half-frame is not
  a fudge -- it is what makes the rounding land on the frame whose CENTRE is nearest.
* **`skip_average` is on for the diarization path**: the aggregation sums, and the speaker count
  does the averaging. Dividing here as well halves everything.
* **The speaker count is computed from the BINARISED segmentation, then rounded**, and the final
  step keeps the `count` loudest speakers per frame -- not everyone above a threshold. A frame with
  count 1 gets exactly one speaker even where two are nearly tied.

⚠ NaN means "this chunk said nothing here", not zero. It is masked out of the sum and restored as
`missing` afterwards; turning it into 0 early makes silence outvote speech.
"""

from __future__ import annotations

from typing import Any

from .diarizen_runner import SEGMENTATION_DURATION, SEGMENTATION_STEP_RATIO

#: The model's receptive field, read off the research path's own cache rather than derived:
#: `(start, duration, step)` in seconds.
FRAME_START = -0.00753125
FRAME_DURATION = 0.025
FRAME_STEP = 0.02

EPSILON = 1e-12


def closest_frame(time: float, *, start: float = FRAME_START, step: float = FRAME_STEP) -> int:
    """`SlidingWindow.closest_frame` -- the frame whose centre is nearest to `time`."""
    return int(round((time - start - 0.5 * FRAME_DURATION) / step))


def aggregate(scores, *, chunk_step: float, skip_average: bool = True, missing: float = 0.0):
    """Overlap-add `(chunks, frames_per_chunk, classes)` onto one frame axis.

    Mirrors `Inference.aggregate` with `hamming=False` and no warm-up, which is what the
    diarization path uses. Returns `(frames, classes)`.
    """
    import numpy as np

    chunks, per_chunk, classes = scores.shape
    mask = 1 - np.isnan(scores)
    filled = np.nan_to_num(scores, copy=True, nan=0.0)

    last_start = (chunks - 1) * chunk_step
    total = closest_frame(last_start + SEGMENTATION_DURATION + 0.5 * FRAME_DURATION) + 1
    summed = np.zeros((total, classes), dtype=np.float32)
    counts = np.zeros((total, classes), dtype=np.float32)
    seen = np.zeros((total, classes), dtype=np.float32)

    for index in range(chunks):
        start = closest_frame(index * chunk_step + 0.5 * FRAME_DURATION)
        stop = start + per_chunk
        summed[start:stop] += filled[index] * mask[index]
        counts[start:stop] += mask[index]
        seen[start:stop] = np.maximum(seen[start:stop], mask[index])

    average = summed if skip_average else summed / np.maximum(counts, EPSILON)
    average[seen == 0.0] = missing
    return average


def speaker_count(binarised, *, chunk_step: float):
    """`(frames, 1)` instantaneous number of speakers, aggregated then rounded."""
    import numpy as np

    per_frame = np.sum(binarised, axis=-1, keepdims=True)
    aggregated = aggregate(per_frame, chunk_step=chunk_step, skip_average=False, missing=0.0)
    return np.rint(aggregated).astype(np.uint8)


def to_diarization(clustered, count):
    """Keep the `count` loudest speakers in each frame; everyone else is off."""
    import numpy as np

    activations = aggregate(clustered, chunk_step=SEGMENTATION_DURATION * SEGMENTATION_STEP_RATIO,
                            skip_average=True, missing=0.0)
    frames = min(len(activations), len(count))
    activations, counted = activations[:frames], count[:frames]

    sorted_speakers = np.argsort(-activations, axis=-1)
    binary = np.zeros_like(activations, dtype=np.int8)
    # `count` is at most the number of clusters; a frame asking for more gets all of them.
    for rank in range(activations.shape[-1]):
        binary[np.arange(frames), sorted_speakers[:, rank]] = rank < counted[:, 0]
    return binary


def assemble(segmentations, hard_clusters) -> Any:
    """`(frames, clusters)` 0/1 timeline from per-chunk activity plus its cluster assignment.

    ⚠ A local speaker with no activity in its chunk is marked `-2` and skipped, exactly as
    `SpeakerDiarization.apply` does just before `reconstruct`. Without it the `-2` branch below is
    unreachable, because the `-2` that `constrained_argmax` leaves for an unassignable speaker has
    already been renumbered to a real cluster by `np.unique` inside `cluster()` -- which is what
    the original does too (`clustering.py`, "re-number clusters from 0"), so that renumbering is
    faithful and this masking is the half that was missing.

    ⚠ **"Nobody spoke" is a legal answer here, not an exception.** When every slot is masked --
    silence, an instrumental, a vocal track the separator emptied -- `max(hard_clusters)` is `-2`
    and the cluster count would be `-1`, which `np.full` rejects with `ValueError: negative
    dimensions are not allowed`. `_run_diarization_stage` catches only `DiarizationUnavailable`,
    so that used to take the whole task down AFTER the ASR had finished -- the one thing the
    stage promises not to do. A `(frames, 0)` map carries the same meaning and every consumer
    downstream already handles it: no tracks, no labels, an empty timeline on disk.
    """
    import numpy as np

    hard_clusters = np.array(hard_clusters, copy=True)
    chunks, per_chunk, _ = segmentations.shape
    step = SEGMENTATION_DURATION * SEGMENTATION_STEP_RATIO
    if not chunks:
        # No audio at all: there is no frame axis to put anything on either.
        return np.zeros((0, 0), dtype=np.int8)
    hard_clusters[np.sum(segmentations, axis=1) == 0] = -2

    counted = speaker_count(segmentations, chunk_step=step)
    clusters = int(np.max(hard_clusters)) + 1
    if clusters < 1:
        # The frame axis is the one the normal path would have produced, so an empty timeline
        # still says how long the file was.
        return np.zeros((len(counted), 0), dtype=np.int8)
    grouped = np.full((chunks, per_chunk, clusters), np.nan, dtype=np.float32)
    for index in range(chunks):
        assignment = hard_clusters[index]
        for target in np.unique(assignment):
            if target == -2:
                continue
            grouped[index, :, target] = np.max(segmentations[index][:, assignment == target],
                                               axis=1)
    return to_diarization(grouped, counted)
