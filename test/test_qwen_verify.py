"""Second-model verification: suspect collection, evidence, stabilize use."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finesub.reporting import NullReporter, reporting_to
from finesub.speech.postprocessing import stabilization as asr_stabilize
from finesub.speech.recognition import vad_asr_stage
from finesub.speech.runtime import hf_weights
from finesub.speech.verification import qwen_referee


JA_RUN_FILLER = (
    "日本語のセグメントがたくさんあって全体としては日本語配信の書き起こしですこの調子で会話が続きます"
) * 3


def seg(text, start, end, *, confidence=0.9, energy=5.0, **extra):
    return {
        "start": start,
        "end": end,
        "text": text,
        "words": [
            {"word": text, "start": start, "end": end, "confidence": confidence}
        ],
        "confidence": confidence,
        "vad_weighted_energy_db": energy,
        **extra,
    }


class TestSuspectCollection:
    def test_normal_rate_closing_phrase_is_suspect(self) -> None:
        segments = [seg(JA_RUN_FILLER, 0.0, 5.0), seg("おわり", 10.0, 10.9)]
        assert qwen_referee.collect_suspect_indices(segments) == [1]

    def test_latin_run_is_suspect_only_in_cjk_run(self) -> None:
        latin = seg("The great plan of the master", 6.0, 8.0)
        assert qwen_referee.collect_suspect_indices(
            [seg(JA_RUN_FILLER, 0.0, 5.0), latin]
        ) == [1]
        # Latin-dominant run: gate off.
        assert (
            qwen_referee.collect_suspect_indices(
                [seg("All English content here", 0.0, 5.0), latin]
            )
            == []
        )

    def test_prospective_drop_tag_is_suspect(self) -> None:
        filler = seg("あ", 10.0, 10.3, confidence=0.1, energy=3.0)
        segments = [seg(JA_RUN_FILLER, 0.0, 5.0), filler]
        assert qwen_referee.collect_suspect_indices(segments) == [1]

    def test_english_boilerplate_is_suspect_in_an_english_run(self) -> None:
        # The 61 residue lines that survived the P1 fallback run were never
        # probed: the run is Latin-dominant (so the lang-switch leg is off),
        # the energy legs exempted them for high word confidence, and no
        # English phrase was listed. Only the phrase leg can reach them.
        english_run = seg("A long stretch of ordinary English narration", 0.0, 5.0)
        boilerplate = seg("Thank you.", 100.0, 111.6, confidence=0.95, energy=-63.0)
        half = seg("Thank", 120.0, 124.0, confidence=0.95, energy=-63.0)
        assert qwen_referee.collect_suspect_indices(
            [english_run, boilerplate, half]
        ) == [1, 2]

    def test_ordinary_english_sentence_is_not_suspect(self) -> None:
        # The whole-segment length bound is what keeps the short phrases from
        # swallowing real sentences that merely contain them.
        english_run = seg("A long stretch of ordinary English narration", 0.0, 5.0)
        real = seg("Thank you for coming to the workshop today.", 10.0, 13.0)
        assert qwen_referee.collect_suspect_indices([english_run, real]) == []

    def test_plain_segment_is_not_suspect(self) -> None:
        segments = [seg(JA_RUN_FILLER, 0.0, 5.0), seg("普通の話です", 6.0, 8.0)]
        assert qwen_referee.collect_suspect_indices(segments) == []


class TestGapCollection:
    def test_uncovered_span_meets_minimum(self) -> None:
        intervals = [{"start": 0.0, "end": 20.0}]
        segments = [seg("a", 0.0, 5.0), seg("b", 16.0, 20.0)]
        assert qwen_referee.collect_gaps(intervals, segments) == [(5.0, 16.0)]

    def test_short_gaps_are_skipped(self) -> None:
        intervals = [{"start": 0.0, "end": 10.0}]
        segments = [seg("a", 0.0, 4.0), seg("b", 6.0, 10.0)]
        assert qwen_referee.collect_gaps(intervals, segments) == []


class FakeReferee:
    _model_name = "fake"

    def __init__(self, replies, device="cpu"):
        self.replies = list(replies)
        self.calls = 0
        self.requested_device = device

    def transcribe_batch(self, clips, *, on_batch=None):
        self.calls += 1
        assert len(clips) == len(self.replies)
        # One "batch" per call: enough to exercise the callback contract
        # without pretending to know the real planner's grouping.
        if on_batch is not None and clips:
            on_batch(len(clips), len(clips))
        return self.replies


class FakeReader:
    def __init__(self, audio_path):
        pass

    def read(self, start, end):
        return np.zeros(int(max(0.0, end - start) * 16000), dtype=np.float32)


class TestApplyVerification:
    def test_evidence_attach_and_gap_recovery(self, monkeypatch) -> None:
        monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
        segments = [
            seg(JA_RUN_FILLER, 0.0, 5.0),
            seg("おわり", 10.0, 10.9),
        ]
        intervals = [{"start": 0.0, "end": 5.0}, {"start": 10.0, "end": 20.0}]
        referee = FakeReferee([("あ。", "Japanese"), ("認識された台詞", "Japanese")])
        out, stats = qwen_referee.apply_verification(
            segments,
            vad_intervals=intervals,
            audio_path="unused.wav",
            referee=referee,
        )
        assert out[1]["qwen_verify"] == {"text": "あ。", "language": "Japanese"}
        assert "qwen_verify" not in out[0]
        assert stats["suspects"] == 1 and stats["gaps_probed"] == 1
        # Nobody chose this device -- `referee_device` derived it -- so a run
        # that quietly verified on the CPU has to be tellable afterwards.
        assert stats["device"] == "cpu"
        assert stats[qwen_referee.GAP_RECOVERY_KEY] == [
            {
                "start": 10.9,
                "end": 20.0,
                "text": "認識された台詞",
                "language": "Japanese",
            }
        ]
        assert referee.calls == 1  # one batched call

    def test_all_three_suspect_families_keep_segment_spans(
        self, monkeypatch
    ) -> None:
        spans: list[tuple[float, float]] = []

        class RecordingReader(FakeReader):
            def read(self, start, end):
                spans.append((start, end))
                return super().read(start, end)

        monkeypatch.setattr(qwen_referee, "_SpanReader", RecordingReader)
        segments = [
            seg(JA_RUN_FILLER, 0.0, 5.0),
            seg("I'm not going to die.", 10.0, 11.0),
            seg("おわり", 20.0, 20.9),
            seg("あ", 30.0, 30.3, confidence=0.1, energy=3.0),
        ]
        referee = FakeReferee(
            [("証拠", "Japanese"), ("証拠", "Japanese"), ("証拠", "Japanese")]
        )
        qwen_referee.apply_verification(
            segments,
            vad_intervals=[],
            audio_path="unused.wav",
            referee=referee,
        )
        pad = qwen_referee.SEGMENT_PAD_SEC
        assert spans == [
            (10.0 - pad, 11.0 + pad),
            (20.0 - pad, 20.9 + pad),
            (30.0 - pad, 30.3 + pad),
        ]

    def test_degenerate_clip_reads_as_no_speech_without_model_call(
        self, monkeypatch
    ) -> None:
        # A suspect span clipped away by the audio bounds must not reach the
        # model; its evidence is "" (and with no usable clips the referee is
        # invoked with an empty batch, which never loads the model).
        class EmptyReader(FakeReader):
            def read(self, start, end):
                return np.zeros(0, dtype=np.float32)

        monkeypatch.setattr(qwen_referee, "_SpanReader", EmptyReader)
        segments = [seg(JA_RUN_FILLER, 0.0, 5.0), seg("おわり", 10.0, 10.9)]
        referee = FakeReferee([])
        out, stats = qwen_referee.apply_verification(
            segments,
            vad_intervals=[{"start": 0.0, "end": 5.0}],
            audio_path="unused.wav",
            referee=referee,
        )
        assert out[1]["qwen_verify"] == {"text": "", "language": None}
        assert stats["suspects"] == 1

    def test_empty_gap_text_is_not_recorded(self, monkeypatch) -> None:
        monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
        segments = [seg(JA_RUN_FILLER, 0.0, 5.0)]
        intervals = [{"start": 0.0, "end": 10.0}]
        referee = FakeReferee([("", None)])
        _, stats = qwen_referee.apply_verification(
            segments,
            vad_intervals=intervals,
            audio_path="unused.wav",
            referee=referee,
        )
        assert stats[qwen_referee.GAP_RECOVERY_KEY] == []


class TestStabilizeConsumption:
    def payload(self, *segments):
        return {"segments": list(segments), "metadata": {}}

    def test_normal_rate_phrase_with_absent_evidence_is_dropped(self) -> None:
        ghost = seg(
            "おわり", 10.0, 10.9,
            confidence=0.24,
            qwen_verify={"text": "あ。", "language": "Japanese"},
        )
        result, report = asr_stabilize.stabilize_payload(
            self.payload(ghost), profile=0
        )
        assert result["segments"] == []
        assert report.tag_counts[asr_stabilize.TAG_PHRASE_GHOST] == 1

    def test_normal_rate_phrase_with_confirming_evidence_is_kept(self) -> None:
        real = seg(
            "ありがとうございました", 10.0, 11.05,
            confidence=0.999, energy=0.7,
            qwen_verify={"text": "ありがとうございました。", "language": "Japanese"},
        )
        result, _ = asr_stabilize.stabilize_payload(self.payload(real), profile=0)
        assert [s["text"] for s in result["segments"]] == [real["text"]]

    def test_normal_rate_phrase_without_evidence_stays_kept(self) -> None:
        unknown = seg("おわり", 10.0, 10.9, confidence=0.24)
        result, _ = asr_stabilize.stabilize_payload(
            self.payload(unknown), profile=0
        )
        assert [s["text"] for s in result["segments"]] == ["おわり"]

    def test_rate_ghost_drops_even_with_bleed_evidence(self) -> None:
        # Neighbor speech bleeding into the evidence clip must not rescue a
        # physically impossible squeeze.
        ghost = seg(
            "それではまた。", 10.0, 10.28,
            confidence=0.2, energy=0.7,
            qwen_verify={"text": "隣の言葉", "language": "Japanese"},
        )
        result, _ = asr_stabilize.stabilize_payload(self.payload(ghost), profile=0)
        assert result["segments"] == []

    def test_verify_speech_vetoes_noise_leg_drop(self) -> None:
        # kaguya あ! family: filler-shaped at positive energy, Qwen heard it.
        shout = seg(
            "あ!", 10.0, 10.4,
            confidence=0.27, energy=9.4,
            qwen_verify={"text": "啊！", "language": "Chinese"},
        )
        result, report = asr_stabilize.stabilize_payload(
            self.payload(shout), profile=0
        )
        assert [s["text"] for s in result["segments"]] == ["あ!"]
        assert report.suspicious_segments_dropped == 0

    def test_empty_verify_does_not_veto(self) -> None:
        filler = seg(
            "あ", 10.0, 10.4,
            confidence=0.1, energy=-25.0,
            qwen_verify={"text": "", "language": None},
        )
        result, _ = asr_stabilize.stabilize_payload(self.payload(filler), profile=0)
        assert result["segments"] == []


class TestRefereePrefetchGate:
    """The prefetch serves the manifest's model and no other."""

    def test_a_custom_model_is_not_prefetched(self, monkeypatch) -> None:
        from finesub_bootstrap import model_ensure

        calls = []
        monkeypatch.setattr(
            model_ensure,
            "ensure_hf_model",
            lambda model_id, **_kwargs: calls.append(model_id),
        )

        assert qwen_referee._ensure_referee_weights("acme/other-model") == (None, False)
        assert calls == []

    def test_the_default_model_gets_the_manifest_revision(self, monkeypatch) -> None:
        """`from_pretrained` must load the verified snapshot, not today's `main`."""

        from finesub_bootstrap import model_ensure

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", lambda *_a, **_k: None)
        monkeypatch.setattr(
            model_ensure, "pinned_snapshot_loadable", lambda _id: False
        )

        assert qwen_referee._ensure_referee_weights(
            qwen_referee.DEFAULT_QWEN_MODEL
        ) == ("abc123", False)

    def test_weights_on_disk_let_the_load_stay_offline(self, monkeypatch) -> None:
        """The gate for `local_files_only`: the pinned snapshot is where the
        loader will look, so nothing has to be asked of huggingface.co."""

        from finesub_bootstrap import model_ensure

        monkeypatch.setattr(model_ensure, "pinned_revision", lambda _id: "abc123")
        monkeypatch.setattr(model_ensure, "ensure_hf_model", lambda *_a, **_k: None)
        monkeypatch.setattr(
            model_ensure, "pinned_snapshot_loadable", lambda _id: True
        )

        assert qwen_referee._ensure_referee_weights(
            qwen_referee.DEFAULT_QWEN_MODEL
        ) == ("abc123", True)


class TestRefereeLoadArguments:
    """What the two `from_pretrained` calls are actually handed.

    Stubs stand in for torch and transformers: the point is the arguments, and
    the light suite loads no models.
    """

    def _load_with(self, monkeypatch, *, revision, local_only) -> list[dict]:
        import types

        calls: list[dict] = []

        class _Loader:
            @staticmethod
            def from_pretrained(name, **kwargs):
                calls.append({"name": name, **kwargs})
                return types.SimpleNamespace(eval=lambda: None)

        monkeypatch.setitem(
            sys.modules,
            "transformers",
            types.SimpleNamespace(
                AutoProcessor=_Loader, AutoModelForMultimodalLM=_Loader
            ),
        )
        monkeypatch.setitem(
            sys.modules, "torch", types.SimpleNamespace(bfloat16=1, float32=2)
        )
        monkeypatch.setattr(
            qwen_referee,
            "_ensure_referee_weights",
            lambda _name: hf_weights.HfLoad(revision, local_only),
        )
        qwen_referee.QwenReferee(device="cpu")._load_model()
        return calls

    def test_weights_on_disk_load_without_the_network(self, monkeypatch) -> None:
        """The regression this exists for: with `local_files_only` unset,
        `AutoProcessor` lists the repo's chat templates over the network and an
        offline machine dies there with every byte already cached."""

        calls = self._load_with(monkeypatch, revision="abc123", local_only=True)
        assert len(calls) == 2
        assert all(call["local_files_only"] is True for call in calls)
        assert all(call["revision"] == "abc123" for call in calls)

    def test_weights_not_vouched_for_may_still_be_fetched(self, monkeypatch) -> None:
        """The other half: `local_files_only` must not turn a first run, or a
        model the manifest never described, into a failure."""

        calls = self._load_with(monkeypatch, revision=None, local_only=False)
        assert all(call["local_files_only"] is False for call in calls)


class _ProgressRecorder(NullReporter):
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def progress(self, stage, *, completed, total=None, unit="", detail="") -> None:
        self.events.append(
            {
                "stage": stage,
                "completed": completed,
                "total": total,
                "unit": unit,
                "detail": detail,
            }
        )


class _Recorder(NullReporter):
    """Every warning as the caller shaped it -- `TerminalReporter` renders the
    message and the impact but drops the code, and the code is half the
    contract here."""

    def __init__(self) -> None:
        self.warnings: list[dict[str, str]] = []

    def warning(self, code, message, *, impact="", action="") -> None:
        self.warnings.append(
            {"code": code, "message": message, "impact": impact, "action": action}
        )


def _unreachable_hub():
    raise TimeoutError("hub unreachable")


class TestContainedVerification:
    """Under `auto` a failed referee costs a layer of evidence; under `on` it
    costs the run.

    The regression: the referee produces evidence and never decisions, yet any
    exception out of it used to take the whole stage down -- discarding an
    alignment pass that had already finished, over a second model the user
    never asked for (`auto` is the default).
    """

    def test_success_passes_the_result_through(self) -> None:
        outcome = vad_asr_stage.contained_verification(
            lambda: (["segment"], {"checked": 1}), qwen_verify="auto"
        )
        assert outcome == (["segment"], {"checked": 1})

    def test_auto_contains_the_failure_and_names_it(self) -> None:
        recorder = _Recorder()
        with reporting_to(recorder):
            outcome = vad_asr_stage.contained_verification(
                _unreachable_hub, qwen_verify="auto"
            )

        assert outcome is None
        assert len(recorder.warnings) == 1
        warned = recorder.warnings[0]
        assert warned["code"] == "qwen-verify-failed"
        # The type and the message both, so a log says which network fault it
        # was rather than only that one happened.
        assert "TimeoutError" in warned["message"]
        assert "hub unreachable" in warned["message"]
        # Same impact string as the `transformers 5.x` branch beside it: both
        # are "the evidence did not happen".
        assert warned["impact"] == "少一层校验证据"

    def test_on_lets_the_failure_end_the_run(self) -> None:
        """`on` is a caller asking for the evidence; returning quietly without
        it would be the wrong answer."""

        with pytest.raises(TimeoutError):
            vad_asr_stage.contained_verification(
                _unreachable_hub, qwen_verify="on"
            )


class TestVerificationProgress:
    """The tail pass has to say it is running.

    It used to be the longest unbroken silence in the pipeline -- 168 s of a
    197 s stage on a contended card, with zero events -- and a silence that
    long is indistinguishable from a hung run. That is not a cosmetic
    complaint: an access violation inside a co-resident referee comes out of
    exactly the same silence, because CTranslate2 aborts rather than raising.
    """

    def _verify(self, recorder, monkeypatch, replies):
        monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
        segments = [seg("all right", 0.0, 1.0), seg("thanks for watching", 1.0, 2.0)]
        with reporting_to(recorder):
            return qwen_referee.apply_verification(
                segments,
                vad_intervals=[{"start": 0.0, "end": 2.0}],
                audio_path="x.wav",
                referee=FakeReferee(replies),
            )

    def test_it_announces_the_work_before_it_starts_and_counts_it_off(
        self, monkeypatch
    ) -> None:
        """The zero comes before the load, not after the first batch: the load
        and the first `generate` are the slow part, so what a watcher needs
        first is what is running and how much of it there is."""

        recorder = _ProgressRecorder()
        self._verify(recorder, monkeypatch, [("ok", "en")])

        assert recorder.events, "the pass reported nothing at all"
        first, last = recorder.events[0], recorder.events[-1]
        assert first["completed"] == 0
        assert last["completed"] == last["total"] == first["total"]
        # Reported under the ASR stage, because it is that stage's tail -- a
        # second stage name after `aligned` hit 100% would read as a new stage
        # starting.
        assert {event["stage"] for event in recorder.events} == {"aligned"}
        assert {event["unit"] for event in recorder.events} == {"clips"}
        assert all(event["detail"] for event in recorder.events)

    def test_nothing_to_check_reports_nothing(self, monkeypatch) -> None:
        """`0/0` would be the one progress line that never moves."""

        recorder = _ProgressRecorder()
        monkeypatch.setattr(qwen_referee, "_SpanReader", FakeReader)
        with reporting_to(recorder):
            qwen_referee.apply_verification(
                [seg("nothing suspicious here", 0.0, 1.0)],
                vad_intervals=[{"start": 0.0, "end": 1.0}],
                audio_path="x.wav",
                referee=FakeReferee([]),
            )

        assert recorder.events == []
