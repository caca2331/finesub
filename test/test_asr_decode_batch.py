"""The A1 decode-batch knob: resolver, provenance, and that it reaches the
decoder.

The batch assembly is `transcribe.DecodePrefetch` (wired 2026-09-02). This
file pins the resolver, the tier table as the default, and that the stage
hands the resolved value to `align_segments` rather than only recording it --
"wired, tested, never reached" is the failure this project keeps finding in
its own guards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.recognition import vad_asr_stage
from finesub.speech.runtime import resources


class TestResolver:
    def test_every_profile_has_a_table_entry(self) -> None:
        for budget in resources.gpu_tier_names():
            assert resources.resolve_asr_decode_batch(gpu_tier=budget) == (
                resources.ASR_DECODE_BATCH_BY_TIER[budget]
            )
            assert resources.ASR_DECODE_BATCH_BY_TIER[budget] >= 1

    @pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "  auto  "])
    def test_auto_and_absent_both_mean_ask_the_profile(self, value) -> None:
        assert resources.resolve_asr_decode_batch(value, gpu_tier="high") == (
            resources.ASR_DECODE_BATCH_BY_TIER["high"]
        )

    def test_a_given_profile_is_not_resolved_a_second_time(self, monkeypatch) -> None:
        """The ASR stage already resolved one, and resolving again asked the
        driver a second time -- without knowing the user had asked for the CPU,
        which is how `--device cpu` was still told its card sat idle."""

        def fail(*args, **kwargs):
            raise AssertionError("the tier was resolved a second time")

        monkeypatch.setattr(resources, "get_resource_profile", fail)
        profile = resources.RESOURCE_PROFILES["high"]

        assert (
            resources.resolve_asr_decode_batch(None, profile=profile)
            == profile.asr_decode_batch
        )

    def test_an_explicit_value_beats_the_profile(self) -> None:
        assert resources.resolve_asr_decode_batch(4, gpu_tier="entry") == 4
        assert resources.resolve_asr_decode_batch("8", gpu_tier="entry") == 8

    @pytest.mark.parametrize("value", ["nonsense", "1.5.2", object()])
    def test_a_bad_value_degrades_rather_than_failing_the_run(self, value) -> None:
        """A batch size is a performance knob; taking a transcription down
        over one is the wrong trade."""
        assert resources.resolve_asr_decode_batch(value) == 1

    def test_zero_and_negative_clamp_to_one(self) -> None:
        assert resources.resolve_asr_decode_batch(0) == 1
        assert resources.resolve_asr_decode_batch(-3) == 1


class TestItReachesTheDecoder:
    def test_the_stage_hands_the_value_to_align_segments(self) -> None:
        import inspect

        source = inspect.getsource(vad_asr_stage.run_vad_asr)
        assert "decode_batch=decode_batch" in source
        assert "asr-decode-batch-inert" not in source


class TestProvenanceNotCompatibility:
    def test_the_batch_size_is_recorded_as_provenance(self) -> None:
        import inspect

        source = inspect.getsource(vad_asr_stage.run_vad_asr)
        assert 'align_meta["asr_decode_batch"] = decode_batch' in source

    def test_it_is_not_part_of_the_checkpoint_key(self) -> None:
        """Changing the batch size must not expire a partial: resuming with a
        different one is legitimate and loses no data (bench-baselines 10.4).
        A compatibility key that grows this field turns a warning into a
        silent full recompute."""
        from finesub.speech.recognition import checkpoint as checkpoint_store

        key = checkpoint_store.build_key(
            model_name="large-v3-turbo",
            language=None,
            gap_sec=0.3,
            audio_path=Path(__file__),
            lang_redecode=False,
        )
        assert "asr_decode_batch" not in key
