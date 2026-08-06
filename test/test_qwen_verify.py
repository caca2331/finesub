"""Second-model verification: suspect collection, evidence, stabilize use."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_playground.speech.postprocessing import stabilization as asr_stabilize
from asr_playground.speech.verification import qwen_referee


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

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def transcribe_batch(self, clips):
        self.calls += 1
        assert len(clips) == len(self.replies)
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
        assert stats[qwen_referee.GAP_RECOVERY_KEY] == [
            {
                "start": 10.9,
                "end": 20.0,
                "text": "認識された台詞",
                "language": "Japanese",
            }
        ]
        assert referee.calls == 1  # one batched call

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
