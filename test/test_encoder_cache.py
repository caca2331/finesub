"""The exact-reuse memo behind A2 and the language-detection re-encode.

Its whole claim is that it cannot change a run's output, so the guards here are
about *exactness*: a hit must require byte-identical features, a stored key must
not be able to change underneath the cache, and the disabled arm must really be
disabled -- that arm is what proves the optimisation is invariant.

Lives apart from `test_fw_refine.py` deliberately: that module skips without the
`[asr]` extra, which CI cannot install, so a guard living there would never run
where it matters.
"""

from __future__ import annotations

import numpy as np

from finesub.speech.recognition.encoder_cache import DEFAULT_ENTRIES, EncoderCache


def _features(seed: int, shape: tuple[int, int] = (4, 6)) -> np.ndarray:
    return np.full(shape, float(seed), dtype=np.float32)


def test_identical_features_reuse_the_stored_output() -> None:
    cache = EncoderCache()
    sentinel = object()
    cache.put(_features(1), sentinel)

    assert cache.get(_features(1)) is sentinel


def test_different_features_miss() -> None:
    cache = EncoderCache()
    cache.put(_features(1), object())

    assert cache.get(_features(2)) is None


def test_an_empty_cache_misses_rather_than_raising() -> None:
    assert EncoderCache().get(_features(1)) is None


def test_one_differing_sample_is_a_miss() -> None:
    """Exact means exact: the encoder is not a local function of its input.

    Whisper's encoder is bidirectional self-attention over all positions, so a
    single changed sample changes the whole output. A cache that tolerated
    "close enough" would silently corrupt every window it hit.
    """

    cache = EncoderCache()
    cache.put(_features(1), object())

    nearly = _features(1)
    nearly[0, 0] = np.float32(1.0000001)

    assert cache.get(nearly) is None


def test_a_different_shape_is_a_miss() -> None:
    cache = EncoderCache()
    cache.put(_features(1, (4, 6)), object())

    assert cache.get(_features(1, (4, 7))) is None


def test_the_stored_key_is_a_copy() -> None:
    """faster-whisper reuses its feature buffers between windows.

    Holding a reference would let the next window rewrite the key in place, and
    the cache would then return window A's encoder output for window B -- an
    exact cache turned into a wrong one, with no error anywhere.
    """

    cache = EncoderCache()
    features = _features(1)
    sentinel = object()
    cache.put(features, sentinel)

    features[:] = 99.0  # the caller reuses its buffer for the next window

    assert cache.get(features) is None, "the mutated buffer must not hit"
    assert cache.get(_features(1)) is sentinel, "the original content must still hit"


def test_the_cache_evicts_oldest_first() -> None:
    cache = EncoderCache(limit=2)
    first, second, third = object(), object(), object()
    cache.put(_features(1), first)
    cache.put(_features(2), second)
    cache.put(_features(3), third)

    assert len(cache) == 2
    assert cache.get(_features(1)) is None
    assert cache.get(_features(2)) is second
    assert cache.get(_features(3)) is third


def test_a_zero_limit_really_disables_it() -> None:
    """`del lst[:-0]` deletes nothing -- the off arm must not become unbounded.

    This matters beyond tidiness: the zero-limit arm is what an A/B runs to
    show that enabling the cache leaves segments bit-identical. An arm that
    quietly cached everything would prove nothing.
    """

    cache = EncoderCache(limit=0)
    for seed in range(5):
        cache.put(_features(seed), object())

    assert len(cache) == 0
    assert cache.get(_features(0)) is None


def test_a_negative_limit_is_treated_as_disabled() -> None:
    cache = EncoderCache(limit=-1)
    cache.put(_features(1), object())

    assert len(cache) == 0


def test_clear_releases_the_entries() -> None:
    """An idle pooled model must not keep pinning GPU memory it is not using."""

    cache = EncoderCache()
    cache.put(_features(1), object())
    cache.clear()

    assert len(cache) == 0
    assert cache.get(_features(1)) is None


def test_the_default_limit_is_small() -> None:
    """Each entry pins ~3.8 MB of GPU memory; the 4 GB profile pays for it."""

    assert 1 <= DEFAULT_ENTRIES <= 8
    assert len(EncoderCache()) == 0
