"""The lab's frame reduction: DiariZen's native 20 ms map down to a coarser grid.

Production stopped reducing (it reads the native grid); the beds and checkers here still score on
the 80 ms grid every lab reading uses, so the reduction lives beside them.
"""

from __future__ import annotations

from finesub.speech.diarization.diarizen_assemble import FRAME_STEP


def to_frame_grid(native, *, source_step: float = FRAME_STEP, target_step: float):
    """Reduce a `(frames, speakers)` map from one frame rate to a coarser one, by max.

    ⚠ `max`, not mean: the map is 0/1 activity, and a speaker who is on for part of a coarse
    frame is on in that frame. Averaging would turn "spoke for 20 of these 80 ms" into 0.25 and
    then, wherever a threshold follows, into silence.

    ⚠ A trailing partial frame is DROPPED rather than padded, because a padded one would claim
    the speaker stopped at a boundary the audio never reached. At 80 ms the loss is under a frame.
    """
    import numpy as np

    ratio = int(round(target_step / source_step))
    if ratio < 1:
        raise ValueError(f"target step {target_step} is finer than the source step {source_step}")
    if ratio == 1:
        return native
    usable = (len(native) // ratio) * ratio
    # ⚠ The first dimension is spelled out rather than inferred: a timeline with no speakers at
    # all is `(frames, 0)`, and numpy cannot solve `-1` for an array of size zero.
    return (np.asarray(native)[:usable]
            .reshape(usable // ratio, ratio, native.shape[1]).max(axis=1))
