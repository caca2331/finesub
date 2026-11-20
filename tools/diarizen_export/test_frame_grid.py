"""The lab's frame reduction (`frame_grid.py`).

⚠ **Not collected by the default suite**: `testpaths` does not include `tools/`, so these run only
when someone runs them here. They moved with the function (2026-11-19), once production read the
native grid and only the beds still reduced it.
"""

from __future__ import annotations

def test_reduction_to_the_coarse_grid_keeps_a_speaker_who_spoke_at_all():
    """⚠ `max`, not mean: 20 ms of speech inside an 80 ms frame is speech in that frame."""
    import numpy as np

    from frame_grid import to_frame_grid

    native = np.zeros((8, 2), dtype=np.float32)
    native[1, 0] = 1.0          # one 20 ms frame of speaker 0, inside the first coarse frame
    native[4:8, 1] = 1.0        # all of the second coarse frame for speaker 1

    coarse = to_frame_grid(native, source_step=0.02, target_step=0.08)
    assert coarse.shape == (2, 2)
    assert coarse[0, 0] == 1.0, "a speaker active for part of the frame is active in it"
    assert coarse[0, 1] == 0.0
    assert list(coarse[1]) == [0.0, 1.0]


def test_reduction_drops_a_trailing_partial_frame_rather_than_padding_it():
    """Padding would claim the speaker stopped at a boundary the audio never reached."""
    import numpy as np

    from frame_grid import to_frame_grid

    native = np.ones((9, 1), dtype=np.float32)
    assert len(to_frame_grid(native, source_step=0.02, target_step=0.08)) == 2


def test_reduction_to_the_same_rate_is_the_identity():
    """A backend already at the target rate must not be rewritten on the way past."""
    import numpy as np

    from frame_grid import to_frame_grid

    native = np.ones((4, 3), dtype=np.float32)
    assert to_frame_grid(native, source_step=0.08, target_step=0.08) is native
