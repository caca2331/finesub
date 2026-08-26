"""Inline language-collapse redecode: trigger, adjudication, ledger rollback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

    def transcribe_batch(self, clips):
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
    def profile(self, budget):
        from finesub.speech.runtime.resources import get_resource_profile

        return get_resource_profile(budget)

    def test_4gb_stays_on_cpu(self) -> None:
        assert (
            lang_redecode.referee_device(
                "cuda", self.profile(4), "large-v3-turbo"
            )
            == "cpu"
        )

    def test_turbo_8gb_and_up_co_reside_on_cuda(self) -> None:
        for budget in (8, 12, 16):
            assert (
                lang_redecode.referee_device(
                    "cuda", self.profile(budget), "large-v3-turbo"
                )
                == "cuda"
            )

    def test_cpu_run_stays_on_cpu(self) -> None:
        assert (
            lang_redecode.referee_device(
                "cpu", self.profile(16), "large-v3-turbo"
            )
            == "cpu"
        )

    def test_large_v3_needs_12gb_for_co_residency(self) -> None:
        assert (
            lang_redecode.referee_device("cuda", self.profile(8), "large-v3")
            == "cpu"
        )
        assert (
            lang_redecode.referee_device("cuda", self.profile(12), "large-v3")
            == "cuda"
        )

    def test_unknown_model_stays_on_cpu(self) -> None:
        assert (
            lang_redecode.referee_device("cuda", self.profile(16), "custom")
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
    def _run(self, monkeypatch, *, language, redecoder):
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
        )

    def test_redecoder_replaces_group_output(self, monkeypatch) -> None:
        stub = StubRedecoder("replaced")
        segments = self._run(monkeypatch, language=None, redecoder=stub)
        assert [s["text"] for s in segments] == ["replaced", "replaced"]
        assert len(stub.calls) == 2

    def test_forced_language_bypasses_redecoder(self, monkeypatch) -> None:
        stub = StubRedecoder("replaced")
        segments = self._run(monkeypatch, language="ja", redecoder=stub)
        assert [s["text"] for s in segments] == ["original", "original"]
        assert stub.calls == []

    def test_no_redecoder_is_inert(self, monkeypatch) -> None:
        segments = self._run(monkeypatch, language=None, redecoder=None)
        assert [s["text"] for s in segments] == ["original", "original"]


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
