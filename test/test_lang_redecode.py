"""Inline language-collapse redecode: trigger, adjudication, ledger rollback."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.runtime import device as device_module
from finesub.speech.runtime.resources import get_resource_profile
from finesub.speech.recognition import lang_redecode
from finesub.speech.recognition import vad_asr_stage
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

        Question 5 (live free VRAM) is pinned to *unknown* here for the same
        reason and with more at stake: left alone it reads this machine's card
        right now, so the whole class would start passing or failing on what
        else the developer happens to have open. Unknown is the answer that
        keeps the tier's verdict, which is exactly what these cases are about.
        `TestLiveVramVeto` pins question 5 itself.
        """

        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: None
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


class TestRefereePlacementIsFiveQuestions:
    @pytest.fixture(autouse=True)
    def _live_vram_is_unknown(self, monkeypatch):
        """Question 5 pinned to unknown, so these keep testing questions 1-4.

        Two of these cases reach it (the ones that patch `cuda_usable`), and
        left alone they would read the developer's own card -- passing or
        failing on what else happens to be open. Unknown is the answer that
        defers to the tier, which is what they are about. Question 5 has its
        own class.
        """

        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: None
        )

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


class _WarningRecorder(NullReporter):
    """Warnings as the caller shaped them: `TerminalReporter` renders the
    message and the impact but drops the code, and the code is half the
    contract."""

    def __init__(self) -> None:
        self.warnings: list[dict[str, str]] = []

    def warning(self, code, message, *, impact="", action="") -> None:
        self.warnings.append(
            {"code": code, "message": message, "impact": impact, "action": action}
        )


class TestLiveVramVeto:
    """Question 5: the tier is a budget, the driver has the measurement.

    The regression this pins: `standard` promises 6.5 GiB, so question 4
    answers "4.43 GiB spare" even on a card whose driver has 2.4 GiB left with
    something else open. The referee's `Module.to(cuda)` then lands beside a
    decoding CTranslate2 pool, and the failure is not a catchable OOM -- the
    decode crawls with nothing to show for it, or the process takes an access
    violation mid-group and CT2 reports neither, because it aborts.

    The wording of the veto is *not* pinned here: it belongs to the caller,
    and the two callers mean different things by it. See
    `TestPlacementCallersWordTheirOwnVeto`.
    """

    @pytest.fixture(autouse=True)
    def _torch_can_use_the_card(self, monkeypatch):
        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )

    def _free(self, monkeypatch, value):
        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: value
        )

    def _placed(self, **kwargs):
        """Placement plus whatever the veto was told, as (device, calls)."""

        calls: list[tuple[float, float]] = []
        placed = lang_redecode.referee_device(
            kwargs.pop("asr_device", "cuda"),
            get_resource_profile(kwargs.pop("tier", "standard")),
            kwargs.pop("model_name", "large-v3-turbo"),
            live_vram_veto=lambda free, needed: calls.append((free, needed)),
            **kwargs,
        )
        return placed, calls

    def test_a_card_with_room_keeps_the_tier_verdict(self, monkeypatch) -> None:
        self._free(monkeypatch, 8.0)
        placed, calls = self._placed(pool_resident=True)

        assert placed == "cuda"
        assert calls == []

    def test_an_unknown_figure_keeps_the_tier_verdict(self, monkeypatch) -> None:
        """The direction every other unknown in this module takes: defer,
        never invent a veto."""

        self._free(monkeypatch, None)
        placed, calls = self._placed(pool_resident=True)

        assert placed == "cuda"
        assert calls == []

    def test_a_full_card_sends_it_to_the_cpu_and_says_by_how_much(
        self, monkeypatch
    ) -> None:
        self._free(monkeypatch, 2.4)
        placed, calls = self._placed(pool_resident=True)

        assert placed == "cpu"
        # Both figures reach the caller: a user who has to decide what to close
        # needs the gap, not just the verdict. The need is the referee alone
        # here, because the pool is already paid for.
        assert calls == [(2.4, lang_redecode.QWEN_REFEREE_GIB)]

    def test_without_a_veto_question_five_is_not_asked_at_all(
        self, monkeypatch
    ) -> None:
        """A caller that has nothing to say about the veto does not get it.

        The tail referee is that caller, deliberately: its pool is already
        closed, so there is nothing to collide with, and the figure would move
        a decode path that is not bit-exact.
        """

        def unreachable():  # pragma: no cover - the point is that it is not called
            raise AssertionError("question 5 was asked without a veto")

        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", unreachable
        )

        assert (
            lang_redecode.referee_device(
                "cuda", get_resource_profile("standard"), "large-v3-turbo"
            )
            == "cuda"
        )

    def test_the_pool_is_bought_only_when_it_is_not_yet_resident(
        self, monkeypatch
    ) -> None:
        """The one arithmetic that is not a rounding error if it is wrong.

        Same card, same free figure, two callers: the language-vote call runs
        before the pool exists and still has to buy Whisper out of what it
        reads, while the warm runs after and must not pay for it twice. Get
        this backwards and the warm goes to the CPU on exactly the tiers it
        exists for.
        """

        resident = lang_redecode.whisper_resident_gib("large-v3-turbo", 1)
        assert resident is not None
        # Enough for the referee alone, not for the referee plus the pool.
        free = lang_redecode.QWEN_REFEREE_GIB + resident / 2
        self._free(monkeypatch, free)

        assert self._placed(pool_resident=True)[0] == "cuda"
        placed, calls = self._placed(pool_resident=False)
        assert placed == "cpu"
        assert calls == [(free, lang_redecode.QWEN_REFEREE_GIB + resident)]

    def test_an_idle_pool_does_not_exempt_the_card_from_the_check(
        self, monkeypatch
    ) -> None:
        """Whisper on the CPU means nothing of *ours* is on the card -- it does
        not mean the card is empty."""

        self._free(monkeypatch, 1.0)
        placed, calls = self._placed(asr_device="cpu", requested_device="cuda")

        assert placed == "cpu"
        assert calls == [(1.0, lang_redecode.QWEN_REFEREE_GIB)]


class TestPlacementCallersWordTheirOwnVeto:
    """The same veto costs different things, so the two callers say different
    things -- and one of them is not a warning at all.

    The bug this pins: a single message owned by the oracle said "the check
    will run on the CPU; it will be noticeably slower; rerun and it goes back
    to the card". True at the redecode call site, false at the warm one, where
    a veto only skips the preload and the tail referee goes on the card
    anyway. On a full card with the defaults, both sites ask, so the false one
    was printed on every such run -- next to the true one.
    """

    @pytest.fixture(autouse=True)
    def _a_card_that_is_full(self, monkeypatch):
        monkeypatch.setattr(
            "finesub.speech.runtime.device.cuda_usable", lambda: True
        )
        monkeypatch.setattr(
            "finesub.speech.runtime.device.free_vram_gib", lambda: 0.5
        )

    def test_the_redecode_site_does_not_ask_at_all(self) -> None:
        """The card is full and it still goes on the card -- on purpose.

        This referee's answer decides whether a group's decode is *replaced*,
        and placement switches dtype (`bfloat16` on CUDA, `float32` on CPU).
        Nothing in the repository shows those two agree: the measurement that
        reads like it does compares CPU float32 against CPU bf16/fp16, a
        different pair. Letting a live figure choose between them would make
        the same audio produce different subtitles depending on what else was
        open at the time. The tier decides here, stably.
        """

        recorder = _WarningRecorder()
        with reporting_to(recorder):
            placed = vad_asr_stage.redecode_referee_device(
                "cuda", get_resource_profile("standard"), "large-v3-turbo"
            )

        assert placed == "cuda"
        assert recorder.warnings == []

    def test_the_warm_site_does_not_warn_because_nothing_moves_to_the_cpu(
        self,
    ) -> None:
        """It only skips the preload; the tail referee is placed later, by
        `tail_verify_device`, which does not ask question 5."""

        recorder = _WarningRecorder()
        with reporting_to(recorder):
            warm = vad_asr_stage.referee_warm_device(
                qwen_verify="auto",
                device="cuda",
                resource_profile=get_resource_profile("standard"),
                model_name="large-v3-turbo",
            )

        assert warm is None
        assert recorder.warnings == []
        # And the tail still goes on the card -- on this very card, which the
        # fixture leaves with 0.5 GiB free. That is what makes the warm site's
        # old wording false, and it is also question 5 not being asked here.
        assert vad_asr_stage.tail_verify_device("cuda") == "cuda"


class TestTailRefereePlacement:
    """The tail referee follows the resolved ASR device -- back to that.

    A 2026-09-04 reroute sent it through `referee_device` so intent, tier and
    torch capability would be answered here too; the argument was that a
    Whisper which fell back for CTranslate2 reasons says nothing about torch,
    so the referee was being kept off a usable card. That argument was purely
    about **speed**, resting on "placement changes how long it takes, not what
    it says".

    The premise is false -- CUDA runs bf16 and CPU runs float32, and the
    measurement that looks like it licenses the swap compares CPU float32
    against CPU bf16/fp16. And this referee is not inert: `stabilization`
    reads its text to decide whether a noise-leg drop stands down, i.e.
    whether a line survives into the subtitle. With the only justification
    gone, the reroute went with it.
    """

    def test_it_follows_the_resolved_asr_device(self) -> None:
        assert vad_asr_stage.tail_verify_device("cuda") == "cuda"
        assert vad_asr_stage.tail_verify_device("cuda:1") == "cuda:1"

    def test_a_cpu_asr_keeps_the_referee_on_the_cpu(self) -> None:
        """⚠ Including the case the reroute existed to fix: a card CT2 cannot
        decode on but torch can. Keeping the referee on the CPU there costs
        speed -- and buying that speed means moving a numeric path that feeds
        a subtitle-affecting decision, which is not a trade anyone has
        measured. Re-open it with the comparison in `asr-align.md` 待标定,
        not with the CPU-only measurement.
        """

        assert vad_asr_stage.tail_verify_device("cpu") == "cpu"
        assert vad_asr_stage.tail_verify_device("") == "cpu"


class _RefereeThatWillNotLoad:
    """The shape of a lazy referee whose weights fail on first use."""

    requested_device = "cpu"

    def transcribe_batch(self, clips, **kwargs):
        raise RuntimeError("weights would not load")


class TestRefereeFailureFollowsTheMode:
    """What a referee that will not load costs, per `--lang-redecode`.

    The module already answers this twice for evidence that never arrives:
    `no-evidence` and `redecode-stalled` both keep the original decode, the
    latter saying outright that "the safe exit is keeping the original decode,
    not killing the run". A referee that fails to load is a third way for the
    evidence not to arrive -- and it used to be the only one that ended the
    run, on the default setting.

    ⚠ Keeping the original decode is *not* a lesser version of redecoding. The
    trigger is a suspicion; the referee is what adjudicates it. Forcing the
    majority language without evidence would corrupt a correct decode on
    genuinely bilingual material -- which is why the `no-evidence` gate exists.
    """

    def _redecoder(self, monkeypatch, *, contained):
        monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
        return lang_redecode.LangRedecoder(
            _RefereeThatWillNotLoad(), "unused.wav", contained=contained
        )

    def _trigger(self, rd):
        return run_maybe(
            rd,
            align_fn=make_align_fn("should not be reached"),
            history=["en"],
            history_before=[],
        )

    def test_auto_keeps_the_original_decode_and_says_so_once(
        self, monkeypatch
    ) -> None:
        """One warning, then silence: every later group would fail the same
        way and print the same line."""

        rd = self._redecoder(monkeypatch, contained=True)
        recorder = _WarningRecorder()

        with reporting_to(recorder):
            first = self._trigger(rd)
            second = self._trigger(rd)

        assert first == OLD_SEGMENTS
        assert second == OLD_SEGMENTS
        assert [warned["code"] for warned in recorder.warnings] == [
            "lang-redecode-failed"
        ]
        # Same impact as the construction-time branch in `vad_asr_stage`: the
        # user-visible consequence is identical, only the timing differs.
        assert recorder.warnings[0]["impact"] == "语言票翻转窗口不会被重解"

        # ⚠ Both triggers must be in the ledger, not just the first. The
        # latch that stops re-probing sits *after* the event is recorded on
        # purpose: `triggers` is documented as counting every trigger, and it
        # is the denominator `stats()` tells callers to calibrate against.
        # Silencing the warning must not also silence the count -- and it
        # would go wrong only on runs that already hit a fault, which is the
        # worst place to lose a number. An earlier version returned before
        # the ledger and this assertion is what catches that.
        assert rd.stats()["triggers"] == 2
        assert rd.stats()["adjudicated"] == 0
        assert [event["rejected"] for event in rd.events] == [
            "referee-unavailable",
            "referee-unavailable",
        ]

    def test_on_still_ends_the_run(self, monkeypatch) -> None:
        """`on` is a caller requiring the redecode; returning quietly without
        it would be the wrong answer, exactly as for `--qwen-verify on`."""

        rd = self._redecoder(monkeypatch, contained=False)

        with pytest.raises(RuntimeError):
            self._trigger(rd)


class TestLedgerKeepsThePopulationsApart:
    """A trigger is a suspicion; four different things stop it becoming a
    redecode, and only two of them are the rule speaking.

    The concrete harm this prevents is a wrong denominator: `adopted /
    triggers` counts the runs where the referee never loaded as runs where the
    rule declined, and that ratio is exactly what this feature's uncalibrated
    thresholds will eventually be measured against (`asr-align.md` 待标定).
    """

    def test_the_denominator_drops_what_the_rule_never_judged(
        self, monkeypatch
    ) -> None:
        rd = redecoder([], monkeypatch)
        # Written straight into the ledger: `stats` is a pure function of it,
        # and driving five different failure paths for real would test the
        # paths (which their own tests already do) rather than the arithmetic.
        rd.events.extend(
            [
                {"adopted": True},
                {"adopted": False, "rejected": "evidence-disagrees"},
                {"adopted": False, "rejected": "redecode-stalled"},
                {"adopted": False, "rejected": "no-evidence"},
                {"adopted": False, "rejected": "referee-unavailable"},
            ]
        )

        stats = rd.stats()

        assert stats["triggers"] == 5
        # Five fired, but the referee answered for only three of them.
        assert stats["adjudicated"] == 3
        assert stats["adopted"] == 1
        assert stats["rejected"] == {
            "evidence-disagrees": 1,
            "redecode-stalled": 1,
            "no-evidence": 1,
            "referee-unavailable": 1,
        }

    def test_every_rejection_reason_is_classified(self) -> None:
        """The guard that keeps the split honest.

        `UNADJUDICATED_REASONS` is a set someone has to remember to update.
        A new reason added to the module without being classified would land
        in the denominator by default -- silently, and in the direction that
        flatters nothing. So the reasons are read back out of the source and
        every one of them must be on a list.
        """

        import inspect
        import re

        adjudicated = {"evidence-disagrees", "redecode-stalled", "redecode-empty"}
        source = inspect.getsource(lang_redecode)
        written = set(
            re.findall(r'event\["rejected"\] = "([a-z-]+)"', source)
        )

        assert written, "the scan found nothing -- the assignment shape moved"
        assert written == adjudicated | set(lang_redecode.UNADJUDICATED_REASONS), (
            "a rejection reason is not classified as adjudicated or not; "
            "decide which and update UNADJUDICATED_REASONS or this test"
        )
