"""The two absolute-dBFS tiers: one drops, one only tags.

Both live on true `frame_dbfs`, NOT on the adaptive weighted `energy_db` that
`MIN_SPEECH_PEAK_DB` uses. Measured over 11838 intervals the two peaks differ
by a median 30.8 dB, so treating them as one number is how a floor ends up
~31 dB looser than intended (`docs/bench-baselines.md` 17.14).

What the tests pin, and why each one is a way this can go wrong in silence:

* both conditions have to bind -- the peak alone is what eats 54.5% of a
  whispered reading, and the pair is what makes an absolute floor safe on
  unvoiced speech;
* the drop tier removes the interval before the decoder, the suspect tier does
  NOT (it only tags), and confusing the two is a silent transcript loss;
* the tag reaching the referee is a separate wire from the tag existing, so a
  run with `--qwen-verify off` carries the field and buys no inference.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.preprocessing import energy as vad_energy
from finesub.speech.recognition import vad_asr_stage
from finesub.speech.verification import qwen_referee


def _track(level_db: float, seconds: float = 4.0) -> vad_energy.VadEnergyTrack:
    hop, frame = vad_energy._frame_grid_seconds()
    frames = int(seconds / hop)
    return vad_energy.VadEnergyTrack(
        energy_db=torch.zeros(frames),
        hop_sec=hop,
        frame_sec=frame,
        energy_mode="weighted",
        frame_dbfs=torch.full((frames,), float(level_db)),
    )


class TestTierCriterion:
    def test_both_conditions_have_to_bind(self) -> None:
        """A flat span at -65 dBFS has peak AND mean at -65: the peak clears
        the drop threshold, the mean does not, so it must NOT be dropped."""

        assert vad_energy.level_tier(-65.0, -65.0) == vad_energy.LEVEL_TIER_SUSPECT
        assert vad_energy.level_tier(-61.0, -71.0) == vad_energy.LEVEL_TIER_DROP
        # ...and the mirror: a mean deep enough but a peak that is not.
        assert vad_energy.level_tier(-30.0, -80.0) is None

    @pytest.mark.parametrize(
        "peak,pmean,expected",
        [
            (-10.0, -20.0, None),
            (-36.0, -46.0, vad_energy.LEVEL_TIER_SUSPECT),
            (-61.0, -71.0, vad_energy.LEVEL_TIER_DROP),
            # Exactly on a threshold is NOT below it.
            (
                vad_energy.LOW_LEVEL_DROP_PEAK_DBFS,
                vad_energy.LOW_LEVEL_DROP_PMEAN_DBFS,
                vad_energy.LEVEL_TIER_SUSPECT,
            ),
        ],
    )
    def test_the_tier_table(self, peak, pmean, expected) -> None:
        assert vad_energy.level_tier(peak, pmean) == expected

    def test_the_two_floors_are_not_the_same_number(self) -> None:
        """`MIN_SPEECH_PEAK_DB` is a weighted-scale peak and these are dBFS.
        The median offset between them is 30.8 dB, so a change that quietly
        makes them share a constant is a ~31 dB change in strictness."""

        assert vad_energy.MIN_SPEECH_PEAK_DB != vad_energy.LOW_LEVEL_DROP_PEAK_DBFS


class TestSpanLevel:
    def test_the_mean_is_a_power_mean_not_a_db_mean(self) -> None:
        """They are different quantities: on a span that is half -20 and half
        -60, the power mean sits just under -23 while a dB mean would say -40.
        Reading the wrong one shifts every threshold by more than 15 dB."""

        hop, _ = vad_energy._frame_grid_seconds()
        half = int(2.0 / hop)
        frames = torch.cat([torch.full((half,), -20.0), torch.full((half,), -60.0)])

        peak, pmean = vad_energy.span_level_dbfs(frames, 0.0, 4.0)

        assert peak == pytest.approx(-20.0)
        assert pmean == pytest.approx(-23.01, abs=0.05)

    def test_a_span_with_no_frames_reads_as_no_evidence(self) -> None:
        assert vad_energy.span_level_dbfs(torch.zeros(0), 0.0, 1.0) is None
        assert (
            vad_energy.span_level_dbfs(torch.full((100,), float("nan")), 0.0, 1.0)
            is None
        )


class TestDropTier:
    def test_a_drop_tier_gap_is_folded_back_into_non_speech(self) -> None:
        """The whole point: the decoder never sees it."""

        track = _track(-75.0, seconds=10.0)
        non_speech = [(0.0, 2.0), (6.0, 10.0)]  # speech gap 2.0-6.0

        merged = vad_energy._absorb_low_level_speech(
            non_speech, track.frame_dbfs, 10.0
        )

        assert merged == [(0.0, 10.0)]

    def test_an_ordinary_gap_is_left_alone(self) -> None:
        track = _track(-20.0, seconds=10.0)
        non_speech = [(0.0, 2.0), (6.0, 10.0)]

        assert (
            vad_energy._absorb_low_level_speech(non_speech, track.frame_dbfs, 10.0)
            == non_speech
        )

    def test_a_track_with_no_non_speech_at_all_is_still_absorbed(self) -> None:
        """The degenerate case this tier most obviously exists for: a recording
        that is quiet end to end has ONE speech interval and no non-speech, and
        every branch of the merge loop needs an existing entry -- so without a
        pre-loop answer it is the one shape that slips through."""

        track = _track(-75.0, seconds=10.0)

        assert vad_energy._absorb_low_level_speech([], track.frame_dbfs, 10.0) == [
            (0.0, 10.0)
        ]

    def test_a_loud_track_with_no_non_speech_is_left_alone(self) -> None:
        track = _track(-20.0, seconds=10.0)

        assert vad_energy._absorb_low_level_speech([], track.frame_dbfs, 10.0) == []

    def test_a_track_without_dbfs_is_inert_rather_than_a_crash(self) -> None:
        """`frame_dbfs` is optional on the track and a stored prefix may have
        none. No evidence must never remove speech."""

        non_speech = [(0.0, 2.0), (6.0, 10.0)]
        assert vad_energy._absorb_low_level_speech(non_speech, None, 10.0) == non_speech


class TestSuspectTierOnlyTags:
    def _annotate(self, level_db: float) -> dict:
        annotated = vad_asr_stage.annotate_segments_with_vad_energy(
            [{"start": 0.5, "end": 3.5, "text": "hello"}], _track(level_db)
        )
        return annotated[0]

    def test_a_suspect_segment_is_tagged(self) -> None:
        item = self._annotate(-46.0)

        assert (
            item[vad_energy.SEGMENT_LEVEL_TIER_FIELD] == vad_energy.LEVEL_TIER_SUSPECT
        )
        assert item["text"] == "hello", "tagging must not touch the transcript"

    def test_an_ordinary_segment_carries_no_field_at_all(self) -> None:
        """Absent rather than null: a field that is always present carries no
        signal in an artifact diff."""

        assert vad_energy.SEGMENT_LEVEL_TIER_FIELD not in self._annotate(-20.0)

    def test_the_tag_is_what_reaches_the_referee(self) -> None:
        run = {"start": 0.0, "end": 5.0, "text": "A long stretch of English"}
        quiet = {
            "start": 10.0,
            "end": 12.0,
            "text": "ordinary words here",
            vad_energy.SEGMENT_LEVEL_TIER_FIELD: vad_energy.LEVEL_TIER_SUSPECT,
        }

        assert qwen_referee.collect_suspect_indices([run, quiet]) == [1]

    def test_a_drop_tag_is_not_a_suspect(self) -> None:
        """Nothing in the drop tier should reach the aligned segments at all;
        if one ever does, it must not silently become referee spend."""

        run = {"start": 0.0, "end": 5.0, "text": "A long stretch of English"}
        dropped = {
            "start": 10.0,
            "end": 12.0,
            "text": "ordinary words here",
            vad_energy.SEGMENT_LEVEL_TIER_FIELD: vad_energy.LEVEL_TIER_DROP,
        }

        assert qwen_referee.collect_suspect_indices([run, dropped]) == []


class TestTheDropTierIsActuallyReached:
    """A check that is wired, tested, and never reached is the exact shape this
    project keeps getting bitten by. `energy.py` has three producers of the
    final interval list -- memory, streamed and CLI -- and they each assemble
    the tail by hand, so a new leg added to one of them is silently absent from
    the other two."""

    def test_every_producer_applies_it(self) -> None:
        import inspect

        source = inspect.getsource(vad_energy)
        applied = source.count("_absorb_low_level_speech(")
        defined = source.count("def _absorb_low_level_speech(")

        assert defined == 1
        assert applied - defined == 3, (
            "all three interval producers must apply the drop tier; "
            f"found {applied - defined} call sites"
        )

    def test_it_sits_with_the_floor_it_is_a_sibling_of(self) -> None:
        """Order is load-bearing: both absolute floors judge the intervals the
        decoder actually gets, so both run on the FINAL padded list. Pinning
        them as neighbours is the cheapest way to keep a future edit from
        moving one of them upstream of the padding on its own."""

        import inspect

        source = inspect.getsource(vad_energy)
        sites = [
            m.start() for m in re.finditer(r"_absorb_low_peak_speech\(", source)
            if not source[max(0, m.start() - 4): m.start()].endswith("def ")
        ]
        assert len(sites) == 3, f"expected 3 call sites, found {len(sites)}"
        for position in sites:
            window = source[max(0, position - 400): position + 400]
            assert "_absorb_low_level_speech(" in window, (
                "the two absolute floors must stay adjacent at every call site"
            )
