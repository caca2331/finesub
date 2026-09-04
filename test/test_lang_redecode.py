"""Inline language-collapse redecode: trigger, adjudication, ledger rollback."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.speech.runtime import device as device_module
from finesub.speech.runtime.resources import get_resource_profile
from finesub.speech.recognition import lang_redecode
from finesub.speech.recognition import transcribe as asr_align
from finesub.speech.recognition import checkpoint as checkpoint_store
from finesub.speech.verification import qwen_referee


class FakeReader:
    def __init__(self, audio_path):
        pass

    def read(self, start, end):
        return np.zeros(int(max(0.0, end - start) * 16000), dtype=np.float32)


class FakeReferee:
    """Replies per probed interval, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.batches = 0

    def transcribe_batch(self, clips, **kwargs):
        self.batches += 1
        assert len(clips) == len(self.replies)
        return list(self.replies)


def make_align_fn(text, calls=None, *, hand_back_first=0):
    """align_fn stub: forced-language redecode returning one segment per call.

    ``hand_back_first`` intervals are returned unconsumed on the first call,
    exercising the consume-it-all loop.
    """

    state = {"first": True}

    def align_fn(pending, audio, sr, *, model, gap_sec, language, **kwargs):
        if calls is not None:
            calls.append({"language": language, "pending": len(pending)})
        unconsumed = []
        consumed = pending
        if state["first"] and hand_back_first:
            state["first"] = False
            consumed = pending[:-hand_back_first]
            unconsumed = pending[-hand_back_first:]
        segments = [
            {
                "start": float(consumed[0]["start"]),
                "end": float(consumed[-1]["end"]),
                "text": text,
                "lang": language,
            }
        ]
        return segments, unconsumed

    return align_fn


GROUP = [
    {"start": 217.9, "end": 230.0},
    {"start": 231.0, "end": 246.8},
]
OLD_SEGMENTS = [
    {"start": 217.9, "end": 246.8, "text": "I'm not going to die.", "lang": "en"}
]


def redecoder(replies, monkeypatch, **kwargs):
    monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
    return lang_redecode.LangRedecoder(
        FakeReferee(replies), "unused.wav", **kwargs
    )


def run_maybe(
    rd,
    *,
    align_fn,
    history,
    history_before,
    recent="ja",
    segments=OLD_SEGMENTS,
    group=GROUP,
):
    return rd.maybe_redecode(
        align_fn=align_fn,
        model=object(),
        group=list(group),
        segments=list(segments),
        audio=None,
        sr=16000,
        gap_sec=0.3,
        auto_language_history=history,
        history_before=history_before,
        recent_language=recent,
        audio_loader=None,
        tail_real_limit_sec=0.7,
    )


class TestTrigger:
    def test_cold_start_none_majority_does_not_trigger(self, monkeypatch) -> None:
        # The truthiness guard IS the criterion: None majority means "nothing
        # to contradict", and a naive != would fire here.
        rd = redecoder([], monkeypatch)
        out = run_maybe(
            rd,
            align_fn=make_align_fn("x"),
            history=["en"],
            history_before=[],
            recent=None,
        )
        assert out == OLD_SEGMENTS
        assert rd.events == []

    def test_matching_vote_does_not_trigger(self, monkeypatch) -> None:
        rd = redecoder([], monkeypatch)
        out = run_maybe(
            rd,
            align_fn=make_align_fn("x"),
            history=["ja", "ja"],
            history_before=["ja"],
        )
        assert out == OLD_SEGMENTS
        assert rd.events == []

    def test_voteless_group_does_not_trigger(self, monkeypatch) -> None:
        rd = redecoder([], monkeypatch)
        history = ["ja"]
        out = run_maybe(
            rd,
            align_fn=make_align_fn("x"),
            history=history,
            history_before=["ja"],
        )
        assert out == OLD_SEGMENTS
        assert rd.events == []


class TestAdjudication:
    def test_empty_evidence_hard_gate_keeps_original(self, monkeypatch) -> None:
        # Pure laughter: referee hears nothing; no redecode is adjudicable.
        calls: list = []
        rd = redecoder([("", None), ("", None)], monkeypatch)
        history = ["ja", "en"]
        out = run_maybe(
            rd, align_fn=make_align_fn("x", calls), history=history,
            history_before=["ja"],
        )
        assert out == OLD_SEGMENTS
        assert calls == []  # never redecoded
        assert history == ["ja", "en"]  # original vote stays
        assert rd.events[0]["rejected"] == "no-evidence"

    def test_similarity_branch_adopts_and_rolls_back_history(
        self, monkeypatch
    ) -> None:
        evidence = "財政はパンタローネに一人する"
        calls: list = []
        rd = redecoder(
            [(evidence, "Japanese"), ("", None)], monkeypatch
        )
        history = ["ja", "en"]
        out = run_maybe(
            rd,
            align_fn=make_align_fn(evidence, calls),
            history=history,
            history_before=["ja"],
        )
        assert [s["text"] for s in out] == [evidence]
        assert calls == [{"language": "ja", "pending": 2}]
        # Ledger: hallucinated vote rolled back, one majority vote cast.
        assert history == ["ja", "ja"]
        event = rd.events[0]
        assert event["adopted"] is True
        assert event["sim_new"] > event["sim_old"]
        # ...and the audit's record of this group follows the ledger. It is
        # written BEFORE the decision (the early returns need it), so an
        # adopted redecode has to correct it, or the audit would warn about a
        # language that is no longer in the product.
        assert [language for *_, language in rd.observations()] == ["ja"]

    def test_vote_branch_adopts_when_similarity_is_inconclusive(
        self, monkeypatch
    ) -> None:
        # Short evidence, low similarity either way — the duration-weighted
        # language vote decides (asr-align.md, measurements: 2nd material).
        rd = redecoder(
            [("うん", "Japanese"), ("そう", "Japanese")], monkeypatch
        )
        history = ["ja", "en"]
        out = run_maybe(
            rd,
            align_fn=make_align_fn("全然違うテキストがここにある"),
            history=history,
            history_before=["ja"],
        )
        assert [s["text"] for s in out] == ["全然違うテキストがここにある"]
        assert rd.events[0]["adopted"] is True
        assert rd.events[0]["vote_share"] == 1.0

    def test_disagreeing_evidence_keeps_original(self, monkeypatch) -> None:
        # Referee votes the hallucinated language: a real language switch.
        history = ["ja", "en"]
        rd = redecoder(
            [("Real English speech here", "English"), ("", None)], monkeypatch
        )
        out = run_maybe(
            rd,
            align_fn=make_align_fn("音訳ごみ"),
            history=history,
            history_before=["ja"],
        )
        assert out == OLD_SEGMENTS
        assert history == ["ja", "en"]
        assert rd.events[0]["rejected"] == "evidence-disagrees"

    def test_redecode_consumes_handed_back_intervals(self, monkeypatch) -> None:
        evidence = "財政はパンタローネに一人する"
        calls: list = []
        rd = redecoder([(evidence, "Japanese"), ("", None)], monkeypatch)
        out = run_maybe(
            rd,
            align_fn=make_align_fn(evidence, calls, hand_back_first=1),
            history=["ja", "en"],
            history_before=["ja"],
        )
        assert [call["pending"] for call in calls] == [2, 1]
        assert len(out) == 2

    def test_empty_redecode_is_never_adopted(self, monkeypatch) -> None:
        # The vote branch never inspects the redecode's output; without the
        # second hard gate a confident language vote would adopt an empty
        # replacement — a deletion, which is outside the feature's boundary.
        rd = redecoder(
            [("うん", "Japanese"), ("そう", "Japanese")], monkeypatch
        )
        history = ["ja", "en"]

        def empty_align(pending, *args, **kwargs):
            return [], []

        out = run_maybe(
            rd, align_fn=empty_align, history=history, history_before=["ja"]
        )
        assert out == OLD_SEGMENTS
        assert history == ["ja", "en"]
        assert rd.events[0]["rejected"] == "redecode-empty"

    def test_stalled_redecode_keeps_original(self, monkeypatch) -> None:
        rd = redecoder([("何か", "Japanese"), ("", None)], monkeypatch)

        def stalled(pending, *args, **kwargs):
            return [], list(pending)

        history = ["ja", "en"]
        out = run_maybe(
            rd, align_fn=stalled, history=history, history_before=["ja"]
        )
        assert out == OLD_SEGMENTS
        assert history == ["ja", "en"]
        assert rd.events[0]["rejected"] == "redecode-stalled"

    def test_history_trim_after_adoption(self, monkeypatch) -> None:
        evidence = "財政はパンタローネに一人する"
        keep = asr_align.AUTO_LANGUAGE_HISTORY_GROUPS
        history_before = ["ja"] * keep
        history = history_before + ["en"]
        rd = redecoder([(evidence, "Japanese"), ("", None)], monkeypatch)
        run_maybe(
            rd,
            align_fn=make_align_fn(evidence),
            history=history,
            history_before=history_before,
        )
        assert history == ["ja"] * keep


class TestRefereeDevice:
    """Where the Qwen referee runs, given the tier's VRAM and the resident pool.

    The tier is the *card's* size, so these read as "on a mainstream card, does
    the referee fit beside large-v3-turbo" -- not as a budget anyone chose.
    """

    @pytest.fixture(autouse=True)
    def _torch_can_use_the_card(self, monkeypatch):
        """These pin the placement *arithmetic*, not the host's hardware.

        Referee placement asks `cuda_usable()` since the ASR stage got its own
        oracle (a card CT2 can decode on may still be one torch has no kernels
        for). Without this the whole class would pass on a machine with a card
        and fail on one without -- which is every CI runner. The cases where
        that answer is False are pinned separately, in
        `TestTorchAndCt2Disagree`.
        """

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )

    def profile(self, tier: str):
        from finesub.speech.runtime.resources import get_resource_profile

        return get_resource_profile(tier)

    def test_entry_tier_stays_on_cpu(self) -> None:
        assert (
            lang_redecode.referee_device(
                "cuda", self.profile("entry"), "large-v3-turbo"
            )
            == "cpu"
        )

    def test_turbo_co_resides_from_the_standard_tier_up(self) -> None:
        for tier in ("standard", "high"):
            assert (
                lang_redecode.referee_device(
                    "cuda", self.profile(tier), "large-v3-turbo"
                )
                == "cuda"
            ), tier

    def test_cpu_run_stays_on_cpu(self) -> None:
        """An explicit `--device cpu`: the request, not the resolved device,
        is what keeps the referee off the card."""

        assert (
            lang_redecode.referee_device(
                "cpu", self.profile("high"), "large-v3-turbo", requested_device="cpu"
            )
            == "cpu"
        )

    def test_large_v3_co_resides_from_the_standard_tier_up(self) -> None:
        """It used to need `high`, on a residency figure that was the B=8 one.

        The B=1 re-measure (3.89 GiB, 2026-09-02) leaves 2.61 GiB on
        `standard` -- 0.11 over the referee's reserve. Thin on purpose: the
        eager referee's measured peak is 2.3 GiB and the compiled path is kept
        out by its own gate. `entry` still cannot hold both.
        """

        assert (
            lang_redecode.referee_device("cuda", self.profile("entry"), "large-v3")
            == "cpu"
        )
        for tier in ("standard", "high"):
            assert (
                lang_redecode.referee_device("cuda", self.profile(tier), "large-v3")
                == "cuda"
            ), tier

    def test_the_japanese_finetune_is_measured_like_large_v3(self) -> None:
        """Named by repository id, and the lookup lower-cases it.

        A capitalised repo id falling through to the unknown-model branch
        would be a silent downgrade to a CPU referee, not an error.
        """

        for name in (
            "TransWithAI/whisper-ja-1.5B-ct2",
            "transwithai/whisper-ja-1.5b-ct2",
        ):
            assert (
                lang_redecode.referee_device("cuda", self.profile("entry"), name)
                == "cpu"
            ), name
            assert (
                lang_redecode.referee_device("cuda", self.profile("standard"), name)
                == "cuda"
            ), name

    def test_a_batched_decode_grows_the_pool_and_evicts_the_referee(self) -> None:
        """The residency table is B=1; `--asr-decode-batch` moves off it.

        This is the guard the B=1 correction needs. While large-v3 carried the
        B=8 figure as if it were B=1, a batched run could not put the referee
        on the GPU by accident. With the honest number it fits on paper, and
        `standard` is written for an 8 GB card where it would not -- CT2
        answers a CUDA OOM with a process-level abort, so there is no
        recovering from getting this wrong.
        """

        standard = self.profile("standard")
        assert lang_redecode.referee_device("cuda", standard, "large-v3", 1) == "cuda"
        for batch in (4, 8):
            assert (
                lang_redecode.referee_device("cuda", standard, "large-v3", batch)
                == "cpu"
            ), batch
        # turbo is small enough that the same batch keeps both resident.
        assert (
            lang_redecode.referee_device("cuda", standard, "large-v3-turbo", 8)
            == "cuda"
        )

    def test_the_batch_scaling_reproduces_the_measured_b8_residency(self) -> None:
        """3.89 GiB at B=1 plus the per-item slope must land on the B=8 number
        the tier table recorded (6.15 GB = 5.73 GiB), or the slope is wrong and
        every budget derived from it is too."""

        assert lang_redecode.whisper_resident_gib("large-v3", 1) == 3.89
        assert lang_redecode.whisper_resident_gib("large-v3", 8) == pytest.approx(
            5.85, abs=0.15
        )
        assert lang_redecode.whisper_resident_gib("large-v3-turbo", 8) == pytest.approx(
            2.92, abs=0.15
        )
        assert lang_redecode.whisper_resident_gib("nobody-measured-this", 8) is None

    def test_the_headroom_tier_buys_the_referee_its_compiled_path(self) -> None:
        """That is the whole reason `standard_large_vram` exists.

        Same two separator workers as `standard`, `high`'s VRAM figure. On
        `standard` a large-v3-class pool leaves 2.7 GiB, which co-resides but
        is under `COMPILE_MIN_VRAM_GIB`, so the referee runs eager; here it
        leaves 6.2 and compiles -- and still compiles once an opted-in decode
        batch has grown the pool, where `standard` has to give up the GPU
        entirely.
        """

        from finesub.speech.verification.qwen_referee import COMPILE_MIN_VRAM_GIB

        ja = "TransWithAI/whisper-ja-1.5B-ct2"
        standard = self.profile("standard")
        headroom = self.profile("standard_large_vram")

        assert headroom.vocal_separator_instances == standard.vocal_separator_instances
        assert headroom.usable_gpu_gb == self.profile("high").usable_gpu_gb

        def budget(profile, batch):
            return lang_redecode.referee_vram_budget(
                profile, ja, beside_pool=True, decode_batch=batch
            )

        assert lang_redecode.referee_device("cuda", standard, ja, 1) == "cuda"
        assert budget(standard, 1) < COMPILE_MIN_VRAM_GIB
        assert lang_redecode.referee_device("cuda", standard, ja, 8) == "cpu"

        for batch in (1, 8):
            assert lang_redecode.referee_device("cuda", headroom, ja, batch) == "cuda"
            assert budget(headroom, batch) >= COMPILE_MIN_VRAM_GIB, batch

    def test_unknown_model_stays_on_cpu(self) -> None:
        assert (
            lang_redecode.referee_device("cuda", self.profile("high"), "custom")
            == "cpu"
        )


class TestLanguageCodes:
    def test_names_map_to_whisper_codes(self) -> None:
        assert lang_redecode.qwen_language_code("Japanese") == "ja"
        assert lang_redecode.qwen_language_code("English") == "en"
        assert lang_redecode.qwen_language_code("ja") == "ja"
        assert lang_redecode.qwen_language_code("") is None
        assert lang_redecode.qwen_language_code("Klingon") is None


class StubRedecoder:
    """align_segments-level stub: replaces every group's segments."""

    def __init__(self, replacement_text):
        self.replacement_text = replacement_text
        self.calls: list[dict] = []
        self.observed: list[tuple[int, str]] = []

    def observe(self, group, language):
        self.observed.append((len(group), language))

    def observations(self):
        return [(0.0, 1.0, language) for _, language in self.observed]

    def restore_observations(self, rows):
        self.observed = [(0, str(row[2])) for row in rows or ()]

    def maybe_redecode(self, *, group, segments, **kwargs):
        self.calls.append({"group": list(group), "segments": list(segments)})
        return [
            {
                "start": float(group[0]["start"]),
                "end": float(group[-1]["end"]),
                "text": self.replacement_text,
                "lang": "ja",
            }
        ]


class TestAlignSegmentsWiring:
    def _run(self, monkeypatch, *, language, redecoder, **extra):
        intervals = [
            {"start": 0.0, "end": 1.0},
            {"start": 10.0, "end": 11.0},
        ]
        monkeypatch.setattr(
            asr_align,
            "build_alignment_groups",
            lambda remaining, **_: [[remaining[0]]],
        )
        monkeypatch.setattr(
            asr_align,
            "_align_intervals_group",
            lambda group, audio, sr, **kwargs: (
                [
                    {
                        "start": float(group[0]["start"]),
                        "end": float(group[-1]["end"]),
                        "text": "original",
                        "lang": "en",
                    }
                ],
                [],
            ),
        )
        monkeypatch.setattr(
            asr_align, "_build_recall_temp_groups", lambda *a, **k: []
        )
        return asr_align.align_segments(
            intervals,
            None,
            16000,
            model=object(),
            gap_sec=0.3,
            language=language,
            lang_redecode=redecoder,
            **extra,
        )

    def test_redecoder_replaces_group_output(self, monkeypatch) -> None:
        stub = StubRedecoder("replaced")
        segments = self._run(monkeypatch, language=None, redecoder=stub)
        assert [s["text"] for s in segments] == ["replaced", "replaced"]
        assert len(stub.calls) == 2

    def test_forced_language_is_observed_but_never_redecoded(
        self, monkeypatch
    ) -> None:
        """Under --language there is no vote to contradict, so no trigger --
        but the run can still be uniformly mislabelled *by the user*, and the
        audit needs those groups."""

        stub = StubRedecoder("replaced")
        segments = self._run(monkeypatch, language="ja", redecoder=stub)
        assert [s["text"] for s in segments] == ["original", "original"]
        assert stub.calls == []
        assert [language for _, language in stub.observed] == ["ja", "ja"]

    def test_no_redecoder_is_inert(self, monkeypatch) -> None:
        segments = self._run(monkeypatch, language=None, redecoder=None)
        assert [s["text"] for s in segments] == ["original", "original"]


class TestAuditLedgerSurvivesResume:
    """The audit samples the groups it was told about, so its ledger is run
    state. If a resume starts it empty, where the interruption fell decides
    what the run reports -- and under MIN_ANSWERED it reports nothing."""

    def test_observations_round_trip_through_the_checkpoint_payload(self) -> None:
        redecoder = lang_redecode.LangRedecoder(referee=None, audio_path="audio.wav")
        redecoder.observe(
            [{"start": 0.0, "end": 4.0}, {"start": 5.0, "end": 6.0}], "ja"
        )
        redecoder.observe([{"start": 10.0, "end": 14.0}], "en")
        carried = redecoder.observations()

        resumed = lang_redecode.LangRedecoder(referee=None, audio_path="audio.wav")
        resumed.restore_observations(carried)

        assert resumed.observations() == carried
        assert [language for *_, language in carried] == ["ja", "en"]

    def test_the_checkpoint_carries_the_ledger_and_the_resume_reads_it(
        self, monkeypatch, tmp_path
    ) -> None:
        """The payload is the contract: a writer that drops the key, or a
        loader that ignores it, both leave a resumed run auditing half a
        timeline while every other test stays green."""

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"riff")
        checkpoint = tmp_path / "clip.asr-partial.json"
        key = checkpoint_store.build_key(
            model_name="m", language="ja", gap_sec=0.3,
            audio_path=audio, lang_redecode=True,
        )
        # align_segments clears the checkpoint on a clean finish, so spy on the
        # writes rather than reading the file afterwards.
        written: list[dict] = []
        real_write = checkpoint_store.write
        monkeypatch.setattr(
            checkpoint_store,
            "write",
            lambda path, payload: (written.append(dict(payload)),
                                   real_write(path, payload))[1],
        )
        TestAlignSegmentsWiring()._run(
            monkeypatch, language="ja", redecoder=StubRedecoder("replaced"),
            checkpoint_path=checkpoint, checkpoint_key=key,
        )
        assert written, "the run wrote no checkpoint at all"
        assert written[-1].get("lang_observations"), (
            "the ASR checkpoint must carry the audit ledger"
        )

        resumed = StubRedecoder("replaced")
        resumed.restore_observations(written[-1]["lang_observations"])
        assert [language for *_, language in resumed.observations()] == ["ja", "ja"]

    def test_a_resume_restores_the_ledger_before_decoding_anything(
        self, monkeypatch, tmp_path
    ) -> None:
        """The loader half of the same contract. A writer that stores the key
        and a loader that ignores it is still a resumed run auditing only what
        it decoded itself."""

        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"riff")
        checkpoint = tmp_path / "clip.asr-partial.json"
        key = checkpoint_store.build_key(
            model_name="m", language="ja", gap_sec=0.3,
            audio_path=audio, lang_redecode=True,
        )
        intervals = [{"start": 0.0, "end": 1.0}, {"start": 10.0, "end": 11.0}]
        fingerprint = dict(key)
        fingerprint["intervals"] = checkpoint_store.intervals_digest(intervals)
        checkpoint_store.write(checkpoint, {
            "version": checkpoint_store.SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "processed_intervals": 1,
            "group_idx": 1,
            "segments": [{"start": 0.0, "end": 1.0, "text": "earlier", "lang": "ja"}],
            "prev_tail_segments": [],
            "auto_language_history": ["ja"],
            "lang_observations": [[0.0, 1.0, "ja"]],
        })

        stub = StubRedecoder("replaced")
        TestAlignSegmentsWiring()._run(
            monkeypatch, language="ja", redecoder=stub,
            checkpoint_path=checkpoint, checkpoint_key=key,
        )

        assert len(stub.observations()) == 2, (
            "the resumed run must audit the group from before the break too"
        )

    def test_a_garbled_payload_is_dropped_rather_than_crashing(self) -> None:
        redecoder = lang_redecode.LangRedecoder(referee=None, audio_path="audio.wav")
        redecoder.restore_observations([("x", 1.0, "ja"), (0.0, 1.0), None])
        assert redecoder.observations() == []


class TestAuditRecordsWhatShipped:
    def test_an_adopted_redecode_updates_the_group_record(self) -> None:
        """`observe` runs before the decision, so an adopted redecode has to
        correct it: the ledger votes the forced language and the text that
        ships is in it, and auditing the raw detection would warn about a
        language no longer present in the product."""

        redecoder = lang_redecode.LangRedecoder(referee=None, audio_path="audio.wav")
        redecoder.observe([{"start": 0.0, "end": 4.0}], "en")
        assert [language for *_, language in redecoder.observations()] == ["en"]

        redecoder.amend_last_observation("ja")

        spans = redecoder.observations()
        assert [language for *_, language in spans] == ["ja"]
        assert spans[0][:2] == (0.0, 4.0), "the span must not move"

    def test_amending_with_nothing_leaves_the_record_alone(self) -> None:
        redecoder = lang_redecode.LangRedecoder(referee=None, audio_path="audio.wav")
        redecoder.observe([{"start": 0.0, "end": 4.0}], "en")
        for empty in ("", None, "None", "   "):
            redecoder.amend_last_observation(empty)
        assert [language for *_, language in redecoder.observations()] == ["en"]


class TestCheckpointKey:
    def test_lang_redecode_changes_the_fingerprint(self, tmp_path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"riff")

        def key(flag):
            return checkpoint_store.build_key(
                model_name="large-v3-turbo",
                language=None,
                gap_sec=0.3,
                audio_path=audio,
                lang_redecode=flag,
            )

        assert key(False) != key(True)
        assert key(False)["lang_redecode"] is False
        assert key(True)["lang_redecode"] is True

def _pretend_torch_has_no_kernels(monkeypatch) -> None:
    """Make torch disown the real card without touching CUDA itself.

    `cuda_usable()` disqualifies a card only when its capability is **below**
    the minimum of `torch.cuda.get_arch_list()`, so raising that floor above
    the installed card reproduces "this PyTorch build has no kernels for it"
    exactly -- while CTranslate2, which keeps its own architecture list and its
    own CUDA runtime, still sees the device.

    That asymmetry is the whole reason the ASR stage stopped asking torch, and
    it is the one case the reference machine cannot produce by hardware. This
    produces it in process, so the branch is covered by a real
    `ct2_cuda_unusable_reason()` rather than by a stub of it.
    """

    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_130"])


class TestTorchAndCt2Disagree:
    """CT2 can use the card, torch cannot -- the case that used to be lost."""

    def test_the_simulation_actually_splits_the_two_backends(self, monkeypatch) -> None:
        """Guard the guard: if this stops holding, the tests below prove nothing."""

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device; the asymmetry cannot be staged here")
        _pretend_torch_has_no_kernels(monkeypatch)
        assert device_module.cuda_usable() is False
        assert device_module.ct2_cuda_unusable_reason() is None

    def test_asr_keeps_the_gpu_that_torch_disowned(self, monkeypatch) -> None:
        if not torch.cuda.is_available():
            pytest.skip("no CUDA device; the asymmetry cannot be staged here")
        _pretend_torch_has_no_kernels(monkeypatch)
        # Before this change the stage asked torch and went to the CPU here.
        assert device_module.resolve_asr_device("cuda") == "cuda"

    def test_the_referee_does_not_follow_it_onto_that_card(self, monkeypatch) -> None:
        """The bug 4b would have introduced on its own.

        The referee is a transformers model. Handing it a card torch has no
        kernels for -- just because CTranslate2 is happily decoding on it --
        would be a worse failure than the one 4b fixes.
        """

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device; the asymmetry cannot be staged here")
        _pretend_torch_has_no_kernels(monkeypatch)
        profile = get_resource_profile("standard")
        assert (
            lang_redecode.referee_device("cuda", profile, "large-v3-turbo") == "cpu"
        )


class TestRefereePlacementIsFourQuestions:
    def test_an_explicit_cpu_request_keeps_the_referee_off_an_idle_card(self) -> None:
        """`--device cpu` means "leave the card alone", not "for Whisper only"."""

        profile = get_resource_profile("standard")
        assert (
            lang_redecode.referee_device(
                "cpu", profile, "large-v3-turbo", requested_device="cpu"
            )
            == "cpu"
        )

    def test_the_cpu_tier_keeps_it_off_too(self) -> None:
        assert (
            lang_redecode.referee_device(
                "cpu", get_resource_profile("cpu"), "large-v3-turbo",
                requested_device="cuda",
            )
            == "cpu"
        )

    def test_an_unchosen_device_is_the_default_not_the_resolved_asr_device(
        self, monkeypatch
    ) -> None:
        """The desktop's "automatic" arrives as `None`.

        Reading None as "use whatever the ASR resolved to" pinned the referee
        to the CPU whenever CTranslate2 had fallen back there -- beside an idle
        card torch could use. None is the absence of a choice, i.e. the code
        default, and the default is a request for the card (review
        2026-09-02).
        """

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        profile = get_resource_profile("standard")
        for unchosen in ({"requested_device": None}, {}):
            assert (
                lang_redecode.referee_device(
                    "cpu", profile, "large-v3-turbo", **unchosen
                )
                == "cuda"
            ), unchosen

    def test_a_capability_fallback_hands_it_the_whole_budget(self, monkeypatch) -> None:
        """Whisper is not on the card, so nothing has to fit beside it.

        This is the payoff the plan called step 3: the ASR fell back for a
        reason that says nothing about torch, and an idle card is an idle card.
        Note the model here is one whose residency is unknown -- irrelevant now,
        because there is no resident pool to subtract.
        """

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        profile = get_resource_profile("standard")
        assert (
            lang_redecode.referee_device(
                "cpu", profile, "nobody-measured-this", requested_device="cuda"
            )
            == "cuda"
        )
        # ...and with a pool actually resident, the old arithmetic still rules.
        assert (
            lang_redecode.referee_device(
                "cuda", profile, "nobody-measured-this", requested_device="cuda"
            )
            == "cpu"
        )
