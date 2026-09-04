"""Exact reuse of an encoder output for byte-identical features.

The Whisper encoder is a pure function of its feature array, so handing it the
same numbers twice must produce the same result. This keeps a few recent
results so the second call can be skipped -- *exact* reuse, with no tolerance
to tune and no way to change what a run produces.

**Why it exists.** faster-whisper's `transcribe()` detects the language by
encoding a window, throws that encoder output away, and then encodes again in
the seek loop to decode the same window (`detect_language` does not return it,
and `encoder_output` is still `None` when the loop starts). Our coverage beam
rescue re-decodes the same group from the same audio for the same reason
(`docs/plans/crispasr-followups.md` -> A2).

**What it deliberately does not do.** Abnormal-interval isolation and coverage
recall re-decode *different* audio, so they miss -- correctly. The encoder is
full bidirectional self-attention over all 1500 positions: change one sample
and the whole output changes. For isolation, that change is the entire point.

⚠ **Measured ceiling.** On a real 23.5 min asset this takes 18 of ~65 redundant
encodes. The rest are *not* byte-identical: faster-whisper's detection slice
keeps one trailing frame that the seek loop's slice drops, so the two arrays
coincide only when both truncate at the full 3000 frames -- i.e. only for
windows that fill all 30 s. Recovering the remainder would mean changing what
gets encoded, which is a behaviour change needing quality validation, not a
free lunch. See `docs/bench-baselines.md`.

Lives in its own module rather than inside `fw_refine_backend` so its guards
run without the `[asr]` extra, which CI cannot install.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: How many recent encoder outputs to keep. One entry already covers the
#: dominant case -- a detection immediately followed by the decode of the same
#: window -- so this is small on purpose: each entry pins about 3.8 MB of GPU
#: memory (1500 x 1280 fp16), and the 4 GB profile has to live with the total.
#: Zero disables the cache, which is the A/B arm that proves it changes nothing.
DEFAULT_ENTRIES = 4


class EncoderCache:
    """A tiny exact-match memo from a feature array to an encoder output."""

    __slots__ = ("_entries", "_limit")

    def __init__(self, limit: int = DEFAULT_ENTRIES) -> None:
        self._entries: list[tuple[np.ndarray, Any]] = []
        self._limit = int(limit)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, features: np.ndarray) -> Any | None:
        """The stored output for byte-identical features, or None.

        Shape is checked first because `array_equal` on mismatched shapes is
        the common case and the cheap rejection keeps the scan negligible next
        to an encode.
        """

        for key, output in self._entries:
            if key.shape == features.shape and np.array_equal(key, features):
                return output
        return None

    def put(self, features: np.ndarray, output: Any) -> None:
        if self._limit <= 0:
            # Written out rather than left to the slice below: `del lst[:-0]`
            # is `del lst[:0]`, which deletes nothing -- so the disabled arm
            # would silently become an unbounded one.
            return
        # A copy, because faster-whisper reuses its feature buffers between
        # windows. Keeping a reference would let a later window rewrite this
        # key in place, turning an exact cache into a wrong one.
        self._entries.append((np.array(features, copy=True), output))
        del self._entries[: -self._limit]

    def clear(self) -> None:
        self._entries.clear()
