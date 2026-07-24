import pytest

from tools.split_explorer import asr_gap


def test_adaptive_silence_uses_original_gap_and_cap() -> None:
    assert asr_gap.adaptive_silence_seconds(
        0.0, base_sec=0.1, factor=0.2, cap_sec=0.8
    ) == pytest.approx(0.1)
    assert asr_gap.adaptive_silence_seconds(
        1.0, base_sec=0.1, factor=0.2, cap_sec=0.8
    ) == pytest.approx(0.3)
    assert asr_gap.adaptive_silence_seconds(
        10.0, base_sec=0.1, factor=0.2, cap_sec=0.8
    ) == pytest.approx(0.8)


def test_temporary_gap_policy_is_scoped() -> None:
    original_cap = asr_gap.asr_align.GAP_KEEP_REAL_MAX_SEC
    original_function = asr_gap.asr_align.inserted_gap_parts
    left = {"start": 0.0, "end": 1.0}
    right = {"start": 2.0, "end": 3.0}

    with asr_gap.temporary_gap_policy(
        real_gap_max_sec=0.7,
        adaptive_gap=(0.1, 0.2, 0.8),
    ):
        assert asr_gap.asr_align.GAP_KEEP_REAL_MAX_SEC == pytest.approx(0.7)
        assert asr_gap.asr_align.inserted_gap_parts(
            left, right, silence_sec=99.0
        ) == pytest.approx((0.7, 0.3))

    assert asr_gap.asr_align.GAP_KEEP_REAL_MAX_SEC == original_cap
    assert asr_gap.asr_align.inserted_gap_parts is original_function


@pytest.mark.parametrize(
    "real_gap_max,adaptive",
    [(-0.1, None), (0.7, (-0.1, 0.2, 0.8)), (0.7, (0.9, 0.2, 0.8))],
)
def test_invalid_gap_policy_is_rejected(real_gap_max, adaptive) -> None:
    with pytest.raises(ValueError):
        asr_gap._validate_policy(real_gap_max, adaptive)
