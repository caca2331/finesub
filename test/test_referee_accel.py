"""Qwen referee efficiency: batched decode, the compiled step's gate, and the
load that hides under the Whisper decode (docs/bench-baselines.md 二十一).

No model is loaded here. The referee's model and processor are replaced by
fakes that record what generate was asked for, so the contracts under test are
the ones the measurements depend on: results come back in input order across
length-sorted batches, the compiled path is taken only when it pays, and it
runs its own (smaller) batches through the fixed-shape decoder.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.reporting import NullReporter, current_reporter, reporting_to
from finesub.speech.recognition import vad_asr_stage
from finesub.speech.runtime import phase_timing
from finesub.speech.verification import qwen_referee


class _Inputs(dict):
    def to(self, *_args, **_kwargs):
        return self


class _Ids:
    def __init__(self, batch: int, prompt_len: int) -> None:
        self.shape = (batch, prompt_len)


class FakeProcessor:
    """Prompt length grows with the longest clip; rows are labelled by clip length."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def apply_transcription_request(self, audio):
        lengths = [len(a) for a in audio]
        self.batches.append(lengths)
        prompt_len = 20 + max(lengths) // 1000
        return _Inputs(input_ids=_Ids(len(lengths), prompt_len), lengths=lengths)

    def decode(self, tail, return_format="raw"):
        rows = [int(row[0]) for row in np.asarray(tail)]
        if return_format == "transcription_only":
            return ["" if length == 0 else f"t{length}" for length in rows]
        return [f"language Japanese<asr_text>t{length}" for length in rows]


class FakeModel:
    def __init__(self, device="cuda:0", fail_static=False) -> None:
        self.device = device
        self.dtype = None
        self.generation_config = type("GC", (), {"eos_token_id": [1, 2], "pad_token_id": 2})()
        self.calls: list[dict] = []
        self.fail_static = fail_static

    def generate(self, *, input_ids, lengths, max_new_tokens, **options):
        self.calls.append({"batch": list(lengths), "options": dict(options)})
        prompt_len = input_ids.shape[1]
        out = np.zeros((len(lengths), prompt_len + 1), dtype=np.int64)
        out[:, -1] = lengths
        return out


class FakeDecoder:
    """Stands in for FixedShapeDecoder: labels rows like the fake model does."""

    def __init__(self, fail=False, max_cache_len=1024) -> None:
        self.calls: list[list[int]] = []
        self.compiled_shapes: set[tuple[int, int]] = set()
        self.fail = fail
        self.max = max_cache_len
        self.closed = False

    def fits(self, prompt_len, max_new_tokens):
        return prompt_len + max_new_tokens <= self.max

    def generate(self, inputs, *, max_new_tokens, eos_token_ids, pad_token_id):
        if self.fail:
            raise RuntimeError("no graph today")
        lengths = inputs["lengths"]
        prompt_len = inputs["input_ids"].shape[1]
        self.calls.append(list(lengths))
        self.compiled_shapes.add(
            (len(lengths), qwen_referee.FixedShapeDecoder.cache_len_for(prompt_len, max_new_tokens, self.max))
        )
        out = np.zeros((len(lengths), prompt_len + 1), dtype=np.int64)
        out[:, -1] = lengths
        return out

    def close(self):
        self.closed = True


class Recorder(NullReporter):
    def __init__(self) -> None:
        self.warnings: list[tuple[str, str]] = []

    def warning(self, code, message, *, impact="", action=""):
        self.warnings.append((code, message))


def make_referee(
    monkeypatch, *, accel="auto", device="cuda:0", fail_static=False, vram=8.0,
    warm_shapes=frozenset(),
):
    referee = qwen_referee.QwenReferee(accel=accel, vram_budget_gib=vram)
    referee._model = FakeModel(device=device, fail_static=fail_static)
    referee._processor = FakeProcessor()
    referee._decoder = FakeDecoder(fail=fail_static)
    monkeypatch.setattr(referee, "_prepare_accel", lambda: None)
    monkeypatch.delenv(qwen_referee.ACCEL_ENV, raising=False)
    # The on-disk shape record stands in for "this machine has compiled it".
    recorded = set(warm_shapes)
    monkeypatch.setattr(qwen_referee, "_compiled_shapes", lambda _m: set(recorded))
    monkeypatch.setattr(
        qwen_referee, "_record_compiled_shape", lambda _m, shape: recorded.add(shape)
    )
    referee.recorded_shapes = recorded
    return referee


SHORT = qwen_referee.FixedShapeDecoder.SHORT_CACHE_LEN
FULL = qwen_referee.MAX_CACHE_LEN


def wanted(referee, pacing_sec, shapes=((1, SHORT),)):
    return referee._compile_wanted(pacing_sec, shapes=set(shapes))


def clip(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * qwen_referee.TARGET_SR), dtype=np.float32)


class TestBatchedDecode:
    def test_results_keep_input_order_across_length_sorted_batches(self, monkeypatch):
        referee = make_referee(monkeypatch)
        seconds = [3.0, 1.0, 2.5, 0.5, 4.0, 1.5, 2.0, 3.5, 0.7, 1.2, 2.2]
        clips = [clip(s) for s in seconds]

        replies = referee.transcribe_batch(clips)

        assert replies == [(f"t{len(c)}", "Japanese") for c in clips]
        batches = referee._model.calls
        assert [len(b["batch"]) for b in batches] == [11]
        seen = [length for b in batches for length in b["batch"]]
        assert seen == sorted(seen), "batches are length-sorted to keep padding small"
        assert all(b["options"] == {} for b in batches), "small call stays eager"

    def test_empty_transcript_carries_no_language(self, monkeypatch):
        referee = make_referee(monkeypatch)
        # A zero-length clip decodes to "" in the fake; the language prelude
        # is still there, and must not be trusted on its own.
        assert referee.transcribe_batch([np.zeros(0, dtype=np.float32)]) == [("", None)]

    def test_empty_call_costs_nothing(self, monkeypatch):
        referee = qwen_referee.QwenReferee()
        monkeypatch.setattr(
            referee, "_ensure_model", lambda: pytest.fail("loaded for nothing")
        )
        assert referee.transcribe_batch([]) == []


class TestCompileGate:
    def test_auto_waits_for_enough_pacing_audio_when_the_shape_is_warm(self, monkeypatch):
        referee = make_referee(monkeypatch, warm_shapes={(8, SHORT)})

        assert not wanted(referee, qwen_referee.COMPILE_MIN_AUDIO_SEC - 1, shapes=((8, SHORT),))
        assert wanted(referee, qwen_referee.COMPILE_MIN_AUDIO_SEC, shapes=((8, SHORT),))

    def test_a_cold_shape_raises_the_bar(self, monkeypatch):
        referee = make_referee(monkeypatch, warm_shapes={(8, SHORT)})
        cold = qwen_referee.COMPILE_MIN_AUDIO_SEC_COLD

        # One of the two shapes is unknown to this machine: cold floor.
        assert not wanted(referee, cold - 1, shapes=((8, SHORT), (3, SHORT)))
        assert wanted(referee, cold, shapes=((8, SHORT), (3, SHORT)))

    def test_the_cache_length_is_part_of_the_shape(self, monkeypatch):
        # The same batch size over the other cache length is a different
        # graph: neither the on-disk record nor this process's graphs make
        # it warm, so it keeps the cold floor and its own `qwen.compile`.
        referee = make_referee(monkeypatch, warm_shapes={(1, SHORT)})
        cold = qwen_referee.COMPILE_MIN_AUDIO_SEC_COLD
        assert wanted(referee, qwen_referee.COMPILE_MIN_AUDIO_SEC, shapes=((1, SHORT),))
        assert not wanted(referee, cold - 1, shapes=((1, FULL),))

        referee._decoder.compiled_shapes.add((1, SHORT))
        assert not wanted(referee, cold - 1, shapes=((1, FULL),))
        assert wanted(referee, cold, shapes=((1, FULL),))

    def test_both_cache_lengths_compile_and_record_separately(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        stats: dict = {}
        with phase_timing.collect(into=stats):
            referee.transcribe_batch([clip(1.0)])  # fake prompt 36 + 256 -> 512
            referee.transcribe_batch([clip(30.0)])  # fake prompt 500 + 256 -> 1024
            referee.transcribe_batch([clip(2.0)])  # 512 again, already warm
        assert stats["qwen.compile"].calls == 2
        assert referee._decoder.compiled_shapes == {(1, SHORT), (1, FULL)}
        assert referee.recorded_shapes == {(1, SHORT), (1, FULL)}

    def test_nothing_fits_means_nothing_to_compile(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        assert not wanted(referee, 1e9, shapes=())

    def test_once_compiled_the_process_stays_on_the_fast_path(self, monkeypatch):
        referee = make_referee(monkeypatch)
        referee._decoder.compiled_shapes.add((qwen_referee.BATCH_CLIPS, SHORT))

        assert wanted(referee, 0.0, shapes=((qwen_referee.BATCH_CLIPS, SHORT),))

    def test_a_compiled_shape_is_recorded_for_the_next_process(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        referee.transcribe_batch([clip(1.0), clip(2.0)])
        assert referee.recorded_shapes == {(2, SHORT)}

    def test_on_and_off_ignore_the_threshold(self, monkeypatch):
        assert wanted(make_referee(monkeypatch, accel="on"), 0.0)
        assert not wanted(make_referee(monkeypatch, accel="off"), 1e9)

    def test_env_kill_switch_beats_on(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        monkeypatch.setenv(qwen_referee.ACCEL_ENV, "0")
        assert not wanted(referee, 1e9)

    def test_auto_needs_a_known_and_sufficient_vram_budget(self, monkeypatch):
        plenty = qwen_referee.COMPILE_MIN_AUDIO_SEC_COLD * 10

        assert not wanted(make_referee(monkeypatch, vram=None), plenty)
        assert not wanted(
            make_referee(monkeypatch, vram=qwen_referee.COMPILE_MIN_VRAM_GIB - 0.1),
            plenty,
        )
        assert wanted(
            make_referee(monkeypatch, vram=qwen_referee.COMPILE_MIN_VRAM_GIB), plenty
        )
        # `on` is the bench probes' switch and vouches for the card itself.
        assert wanted(make_referee(monkeypatch, accel="on", vram=None), 0.0)

    def test_the_tail_pass_can_raise_the_budget(self, monkeypatch):
        referee = make_referee(monkeypatch, vram=1.0)
        plenty = qwen_referee.COMPILE_MIN_AUDIO_SEC_COLD * 10
        assert not wanted(referee, plenty)
        referee.set_vram_budget(qwen_referee.COMPILE_MIN_VRAM_GIB)
        assert wanted(referee, plenty)

    def test_cpu_never_compiles(self, monkeypatch):
        assert not wanted(make_referee(monkeypatch, accel="on", device="cpu"), 1e9)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError):
            qwen_referee.QwenReferee(accel="maybe")

    def test_the_default_is_auto(self, monkeypatch):
        """The stage builds referees with defaults, so `auto` -- gated on
        pacing audio and VRAM -- is what production gets."""
        assert qwen_referee.QwenReferee()._accel == "auto"
        referee = make_referee(monkeypatch, vram=None)
        assert not wanted(referee, 1e9), "no budget vouched for: no compile"


class TestBatchPlan:
    def sec(self, *seconds):
        return [int(s * qwen_referee.TARGET_SR) for s in seconds]

    def test_sorted_and_capped_by_count(self):
        batches = qwen_referee.plan_batches(self.sec(*([1.0] * 19)))
        assert [len(b) for b in batches] == [16, 3]
        batches = qwen_referee.plan_batches(self.sec(*([1.0] * 19)), max_clips=8)
        assert [len(b) for b in batches] == [8, 8, 3]
        lengths = self.sec(3, 1, 2, 0.5, 4, 1.5, 2.5, 3.5, 0.7, 1.2, 2.2)
        batches = qwen_referee.plan_batches(lengths)
        assert [len(b) for b in batches] == [11]
        flat = [lengths[i] for b in batches for i in b]
        assert flat == sorted(lengths)
        assert sorted(i for b in batches for i in b) == list(range(len(lengths)))

    def test_capped_by_padded_audio(self):
        # Eight 30 s clips would be 240 s padded; the cap is 120 s -> 4 + 4.
        batches = qwen_referee.plan_batches(self.sec(*([30.0] * 8)))
        assert [len(b) for b in batches] == [4, 4]
        # Mixed: the short ones share a batch until a long one breaks the cap.
        batches = qwen_referee.plan_batches(self.sec(1, 1, 1, 50, 50, 50))
        assert [len(b) for b in batches] == [3, 2, 1]
        for b in batches:
            longest = max(self.sec(1, 1, 1, 50, 50, 50)[i] for i in b)
            assert len(b) * longest <= qwen_referee.BATCH_MAX_PADDED_SEC * qwen_referee.TARGET_SR

    def test_an_oversized_clip_still_runs_alone(self):
        batches = qwen_referee.plan_batches(self.sec(300.0, 2.0))
        assert [len(b) for b in batches] == [1, 1]

    def test_empty(self):
        assert qwen_referee.plan_batches([]) == []


class TestCompiledPath:
    def test_compiled_calls_go_through_the_fixed_shape_decoder(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        clips = [clip(1.0), clip(2.0), clip(3.0)]

        replies = referee.transcribe_batch(clips)

        assert replies == [(f"t{len(c)}", "Japanese") for c in clips]
        assert referee._model.calls == [], "generate() is the eager path only"
        assert referee._decoder.calls == [[len(c) for c in clips]]
        assert referee._decoder.compiled_shapes == {(3, SHORT)}

    def test_compiled_batches_follow_the_compiled_cap(self, monkeypatch):
        monkeypatch.setattr(qwen_referee, "COMPILED_BATCH_CLIPS", 8)
        referee = make_referee(monkeypatch, accel="on")
        clips = [clip(1.0)] * 19

        referee.transcribe_batch(clips)

        assert [len(c) for c in referee._decoder.calls] == [8, 8, 3]

    def test_the_batch_cap_is_read_at_call_time(self, monkeypatch):
        monkeypatch.setattr(qwen_referee, "BATCH_CLIPS", 4)
        assert [len(b) for b in qwen_referee.plan_batches([16000] * 9)] == [4, 4, 1]

    def test_first_compiled_call_is_timed_separately(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        stats: dict = {}
        with phase_timing.collect(into=stats):
            referee.transcribe_batch([clip(1.0)])
            referee.transcribe_batch([clip(1.0)])
        assert stats["qwen.compile"].calls == 1
        assert stats["qwen.infer"].calls == 2

    def test_prompt_beyond_the_cache_takes_eager(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        # 20 + 1200 prompt tokens + 256 new > MAX_CACHE_LEN in the fake.
        referee.transcribe_batch([clip(1200.0)])
        assert len(referee._model.calls) == 1 and referee._decoder.calls == []

    def test_compile_failure_falls_back_and_stays_eager(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on", fail_static=True)
        recorder = Recorder()
        clips = [clip(1.0), clip(2.0)]
        with reporting_to(recorder):
            first = referee.transcribe_batch(clips)
            second = referee.transcribe_batch(clips)

        assert first == second == [(f"t{len(c)}", "Japanese") for c in clips]
        assert [code for code, _ in recorder.warnings] == ["referee-accel-disabled"]
        assert len(referee._model.calls) == 2, "fallback, then eager for good"
        assert referee._accel_failed

    def test_close_closes_the_decoder(self, monkeypatch):
        referee = make_referee(monkeypatch, accel="on")
        decoder = referee._decoder
        referee.transcribe_batch([clip(1.0)])
        referee.close()
        assert decoder.closed and referee._decoder is None and referee._model is None


class TestAccelCacheDir:
    def test_no_cuda_means_no_directory(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert qwen_referee._accel_cache_dir("m") is None

    def test_layout_is_keyed_and_ends_in_inductor(self, monkeypatch):
        import torch

        if not torch.cuda.is_available():
            pytest.skip("key needs a CUDA torch")
        path = qwen_referee._accel_cache_dir(qwen_referee.DEFAULT_QWEN_MODEL)
        assert path is not None
        assert path.name == "inductor" and path.parents[1].name == "qwen-referee-accel"
        assert len(path.parent.name) == 12
        assert path != qwen_referee._accel_cache_dir("another/model")


class TestPhaseMerge:
    def test_merge_adds_up_like_into(self):
        into: dict = {}
        with phase_timing.collect(into=into):
            with phase_timing.phase("qwen.load"):
                pass
        other: dict = {}
        with phase_timing.collect(into=other):
            with phase_timing.phase("qwen.load"):
                pass
            with phase_timing.phase("qwen.infer"):
                pass
        phase_timing.merge(into, other)
        assert into["qwen.load"].calls == 2 and into["qwen.infer"].calls == 1


class TestWarmUnderDecode:
    def profile(self, tier):
        from finesub.speech.runtime.resources import get_resource_profile

        return get_resource_profile(tier)

    def test_off_cpu_and_entry_tier_do_not_warm(self):
        args = dict(device="cuda", model_name="large-v3-turbo")
        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="off", resource_profile=self.profile("high"), **args
            )
            is None
        )
        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="auto",
                device="cpu",
                resource_profile=self.profile("high"),
                model_name="large-v3-turbo",
                # The request: a *resolved* "cpu" alone no longer means "leave
                # the card alone" (it is what a CT2-only fallback reads too).
                requested_device="cpu",
            )
            is None
        )
        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="auto", resource_profile=self.profile("entry"), **args
            )
            is None
        ), "4 GB card: the referee waits for the pool to be released"

    def test_standard_tier_warms_beside_turbo(self, monkeypatch):
        # The placement arithmetic, not this host's hardware: referee placement
        # asks torch since the ASR stage got its own oracle, so without this the
        # test would pass here and fail on every GPU-less CI runner. The card's
        # *live* free VRAM is the other half of the same sentence -- question 5
        # reads the driver, so an unstubbed run answers about whatever else the
        # machine has open (this went red the day another job held 14.9 of the
        # 16.3 GiB). The entry-tier case above needs neither: it is vetoed one
        # question earlier, by the tier's own spare budget.
        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: 24.0
        )
        assert (
            vad_asr_stage.referee_warm_device(
                qwen_verify="auto",
                device="cuda",
                resource_profile=self.profile("standard"),
                model_name="large-v3-turbo",
            )
            == "cuda"
        )

    def test_warm_thread_reports_where_the_stage_does_and_hands_back_its_phases(self):
        seen = {}

        class Referee:
            def warm(self):
                seen["reporter"] = current_reporter()
                with phase_timing.phase("qwen.load"):
                    pass

        reporter = NullReporter()
        stats: dict = {}
        with reporting_to(reporter):
            warm = vad_asr_stage.RefereeWarm(Referee())
            warm.join(stats)

        assert seen["reporter"] is reporter
        assert stats["qwen.load"].calls == 1
        assert warm.error is None and warm.elapsed_sec >= 0.0

    def test_failed_warm_is_kept_not_raised(self):
        class Referee:
            def warm(self):
                raise RuntimeError("boom")

        warm = vad_asr_stage.RefereeWarm(Referee())
        warm.join({})
        assert warm.error == "RuntimeError: boom"

    def test_failed_warm_does_not_pin_the_loader_locals(self):
        # A kept exception would keep its traceback, and the traceback the
        # loader's frame with the half-built model in it. The summary must
        # let that local go before the tail pass loads its own copy.
        import gc
        import weakref

        class HalfBuiltModel:
            pass

        holder = {}

        class Referee:
            def warm(self):
                model = HalfBuiltModel()
                holder["ref"] = weakref.ref(model)
                raise RuntimeError("cuda out of memory")

        warm = vad_asr_stage.RefereeWarm(Referee())
        warm.join({})
        gc.collect()
        assert warm.error == "RuntimeError: cuda out of memory"
        assert holder["ref"]() is None


class TestFixedShapeDecoder:
    """The greedy loop mirrors generate(): EOS stops a row, pad follows, the
    loop ends when every row is done, and shapes never change after prefill."""

    EOS, PAD, V = 9, 8, 12

    def make(self, script, monkeypatch):
        import torch

        from finesub.speech.verification.qwen_decode import FixedShapeDecoder

        calls = {"steps": 0, "shapes": set()}

        class Out:
            def __init__(self, logits):
                self.logits = logits

        def model(**kw):
            if kw.get("inputs_embeds") is not None:
                batch = kw["inputs_embeds"].shape[0]
                index = 0  # prefill produces each row's first token
            else:
                batch = kw["input_ids"].shape[0]
                calls["steps"] += 1
                calls["shapes"].add(
                    (
                        tuple(kw["input_ids"].shape),
                        tuple(kw["attention_mask"]["full_attention"].shape),
                        tuple(kw["position_ids"].shape),
                    )
                )
                index = calls["steps"]
            calls.setdefault("positions", []).append(kw["position_ids"].tolist())
            logits = torch.full((batch, 1, self.V), -1.0)
            for row in range(batch):
                token = script[row][index] if index < len(script[row]) else self.EOS
                logits[row, 0, token] = 1.0
            return Out(logits)

        decoder = FixedShapeDecoder(model, max_cache_len=32, compile_step=False)
        monkeypatch.setattr(decoder, "_cache_for", lambda batch, cache_len: object())
        monkeypatch.setattr(
            decoder,
            "_prefill_embeddings",
            lambda inputs: torch.zeros(*inputs["input_ids"].shape, 4),
        )
        return decoder, calls

    def test_cache_length_follows_the_need(self, monkeypatch):
        from finesub.speech.verification.qwen_decode import FixedShapeDecoder

        decoder = FixedShapeDecoder(object(), max_cache_len=1024, compile_step=False)
        assert decoder._cache_len_for(200, 256) == 512
        assert decoder._cache_len_for(400, 256) == 1024
        small = FixedShapeDecoder(object(), max_cache_len=32, compile_step=False)
        assert small._cache_len_for(3, 5) == 32, "never longer than max"

    def test_layout_matches_generate(self, monkeypatch):
        import torch

        script = [[5, 6, self.EOS], [7, self.EOS]]
        decoder, calls = self.make(script, monkeypatch)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3], [0, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1], [0, 1, 1]]),
            "input_features": torch.zeros(2, 4, 4),
            "input_features_mask": torch.ones(2, 4),
        }
        out = decoder.generate(inputs, max_new_tokens=5, eos_token_ids=[self.EOS], pad_token_id=self.PAD)
        # generate()'s layout: prompt, then the new tokens up to the longest
        # row, pad after a row's EOS.
        assert out.tolist() == [
            [1, 2, 3, 5, 6, self.EOS],
            [0, 2, 3, 7, self.EOS, self.PAD],
        ]
        # prefill + 2 steps: the loop stops once both rows have their EOS
        assert calls["steps"] == 2
        assert calls["shapes"] == {((2, 1), (2, 1, 1, 32), (2, 1))}

    def test_positions_follow_each_rows_padding_like_generate(self, monkeypatch):
        # generate() derives positions from the padding mask (cumsum - 1,
        # pads at 0): the left-padded row counts from its first real token,
        # not from the padded prompt start, and keeps its own count in the
        # steps. Cache slots stay shared -- that is what keeps the shapes fixed.
        import torch

        script = [[5, 6, self.EOS], [7, 6, self.EOS]]
        decoder, calls = self.make(script, monkeypatch)
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3, 4], [0, 0, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]]),
        }
        decoder.generate(inputs, max_new_tokens=5, eos_token_ids=[self.EOS], pad_token_id=self.PAD)
        prefill, *steps = calls["positions"]
        assert prefill == [[0, 1, 2, 3], [0, 0, 0, 1]]
        assert steps == [[[4], [2]], [[5], [3]]]

    def test_budget_bounds_the_loop(self, monkeypatch):
        import torch

        decoder, calls = self.make([[5] * 50], monkeypatch)
        inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.tensor([[1, 1]])}
        out = decoder.generate(inputs, max_new_tokens=4, eos_token_ids=[self.EOS], pad_token_id=self.PAD)
        assert out.tolist() == [[1, 2, 5, 5, 5, 5]] and calls["steps"] == 3

    def test_prompt_that_cannot_fit_is_refused(self, monkeypatch):
        import torch

        decoder, _ = self.make([[5]], monkeypatch)
        assert not decoder.fits(30, 4)
        with pytest.raises(ValueError):
            decoder.generate(
                {"input_ids": torch.zeros(1, 30, dtype=torch.long), "attention_mask": torch.ones(1, 30)},
                max_new_tokens=4,
                eos_token_ids=[self.EOS],
                pad_token_id=self.PAD,
            )
