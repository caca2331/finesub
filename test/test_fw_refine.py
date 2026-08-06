from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import pytest

# fw_refine imports scipy and the backend imports faster_whisper, both [asr]-only.
# CI installs [harness,dev] and cannot install [asr] at all -- the patched
# CTranslate2 wheel is Windows-only -- so skip the module rather than error out
# during collection.
pytest.importorskip("scipy", reason="[asr] extra not installed")
pytest.importorskip("faster_whisper", reason="[asr] extra not installed")

from asr_playground.speech.recognition import fw_refine  # noqa: E402
from asr_playground.speech.recognition import fw_refine_backend
from asr_playground.speech.recognition import transcribe as asr_transcribe
from asr_playground.speech.recognition.fw_refine_backend import RefinedWhisperModel


class _Tokenizer:
    eot = 100
    timestamp_begin = 200

    _pieces = {
        1: " Hello",
        2: ",",
        3: " world",
        5: "本",
        6: " again",
        7: "�",  # never resolves: an unrepresentable byte, not a prefix
    }

    def decode_with_timestamps(self, tokens: list[int]) -> str:
        if tokens == [4]:
            return "\ufffd"
        if tokens == [4, 5]:
            return "日本"
        return "".join(
            f"<|{(token - self.timestamp_begin) * 0.02:.2f}|>"
            if token >= self.timestamp_begin
            else self._pieces.get(token, "")
            for token in tokens
        )

    def decode(self, tokens: list[int]) -> str:
        return self.decode_with_timestamps(tokens)


def test_split_timestamp_spans_retains_trace_rows() -> None:
    spans = fw_refine.split_timestamp_spans(
        [200, 1, 201, 201, 2, 202],
        timestamp_begin=200,
    )

    assert spans == [
        fw_refine.TimestampSpan(0, 2, (200, 1, 201)),
        fw_refine.TimestampSpan(3, 5, (201, 2, 202)),
    ]


def test_early_eot_repair_uses_retained_terminal_trace_row() -> None:
    repaired = fw_refine.repair_early_eot_span(
        [200, 1, 2],
        timestamp_begin=200,
        eot=100,
        attention_steps=4,
        logprob_steps=4,
    )

    assert repaired == fw_refine.TimestampSpan(0, 3, (200, 1, 2, 100))
    assert (
        fw_refine.repair_early_eot_span(
            [200, 1, 2],
            timestamp_begin=200,
            eot=100,
            attention_steps=3,
            logprob_steps=4,
        )
        is None
    )


def test_early_eot_alignment_excludes_temporary_boundary_from_output() -> None:
    result = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 3, (200, 1, 3, 100)),
        path=[(0, 0), (1, 1), (2, 2), (3, 4)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -0.3, -0.5],
    )

    assert [word["word"] for word in result.words] == ["Hello", "world"]
    assert result.confidence == round(math.exp(-0.2), 3)


def test_nonincreasing_end_repair_only_changes_temporary_alignment_tokens() -> None:
    original = fw_refine.TimestampSpan(4, 7, (203, 1, 2, 203))
    logits = np.full(220, -10.0, dtype=np.float32)
    logits[207] = -0.1

    repaired = fw_refine.repair_nonincreasing_end_span(
        original,
        timestamp_begin=200,
        endpoint_logprobs=logits,
    )

    assert original.tokens == (203, 1, 2, 203)
    assert repaired == fw_refine.TimestampSpan(4, 7, (203, 1, 2, 207))


def test_splitters_match_wt_word_contract() -> None:
    tokenizer = _Tokenizer()
    tokens = [200, 1, 2, 3, 201]

    words, token_texts, token_ids = fw_refine.split_tokens_on_spaces(tokens, tokenizer)

    assert words == ["<|0.00|>", "Hello,", "world", "<|0.02|>"]
    assert token_texts == [["<|0.00|>"], [" Hello", ","], [" world"], ["<|0.02|>"]]
    assert token_ids == [[200], [1, 2], [3], [201]]


def test_unicode_splitter_waits_for_complete_character() -> None:
    words, token_texts, token_ids = fw_refine.split_tokens_on_unicode(
        [200, 4, 5, 201],
        _Tokenizer(),
    )

    assert words == ["<|0.00|>", "日本", "<|0.02|>"]
    assert token_texts == [["<|0.00|>"], ["", "日本"], ["<|0.02|>"]]
    assert token_ids == [[200], [4, 5], [201]]


def test_unicode_splitter_emits_never_resolving_replacement_char() -> None:
    """A span may end on an unrepresentable byte (hallucination hitting the
    decode limit). Waiting for a completion that never arrives used to drop the
    token, desynchronising the groups from the decoder trace."""

    tokens = [200, 1, 7]
    words, _token_texts, token_ids = fw_refine.split_tokens_on_unicode(
        tokens,
        _Tokenizer(),
    )

    assert words == ["<|0.00|>", " Hello", "�"]
    assert token_ids == [[200], [1], [7]]
    assert sum(len(group) for group in token_ids) == len(tokens)


def test_alignment_consumes_span_ending_on_unrepresentable_token() -> None:
    """The shape a repetition hallucination takes when it hits the decode
    limit: an unfinished span whose final token is an unrepresentable byte."""

    result = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 2, (200, 1, 7), True),
        path=[(0, 0), (1, 1), (2, 2)],
        tokenizer=_Tokenizer(),
        language="ja",
        chosen_logprobs=[-9.0, -0.1, -0.2],
        collect_refine_signals=True,
    )

    assert [word["word"] for word in result.words] == [" Hello", "�"]


def test_alignment_words_exclude_timestamps_and_punctuation_from_confidence() -> None:
    span = fw_refine.TimestampSpan(0, 4, (200, 1, 2, 3, 201))
    result = fw_refine.align_span_words(
        span=span,
        path=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -2.0, -0.3, -9.0],
    )

    assert result.words == (
        {"word": "Hello,", "start": 0.02, "end": 0.04, "confidence": 0.905},
        {"word": "world", "start": 0.06, "end": 0.08, "confidence": 0.741},
    )
    assert result.confidence == round(math.exp(-0.2), 3)


def test_unfinished_alignment_keeps_final_text_token_and_confidence() -> None:
    result = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 2, (200, 1, 3), unfinished=True),
        path=[(0, 0), (1, 1), (2, 4)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -0.3],
    )

    assert [word["word"] for word in result.words] == ["Hello", "world"]
    assert result.confidence == round(math.exp(-0.2), 3)


def test_disfluency_detection_omits_leading_gap_and_preserves_lexical_ends(
    monkeypatch,
) -> None:
    def fake_find_peaks(values, *, width, prominence):
        assert width == 3
        assert prominence == 0.02
        if values.size and values[0] in {1.0, 2.0}:
            return np.asarray([1, 2]), {"left_ips": np.asarray([0.5, 2.5])}
        return np.asarray([], dtype=np.int64), {"left_ips": np.asarray([])}

    monkeypatch.setattr(fw_refine, "find_peaks", fake_find_peaks)
    weights = np.zeros((4, 12), dtype=np.float32)
    weights[1, 1:6] = 1.0
    weights[2, 6:10] = 2.0

    options = {
        "span": fw_refine.TimestampSpan(0, 3, (200, 1, 3, 201)),
        "path": [(0, 0), (1, 1), (2, 6), (3, 10), (3, 11)],
        "tokenizer": _Tokenizer(),
        "language": "en",
        "chosen_logprobs": [-9.0, -0.1, -0.3, -9.0],
    }
    baseline = fw_refine.align_span_words(**options)
    result = fw_refine.align_span_words(
        **options,
        alignment_weights=weights.ravel(),
        alignment_frame_start=0,
        detect_disfluencies=True,
    )

    assert [word["word"] for word in result.words] == ["Hello", "[*]", "world"]
    assert result.words[0]["start"] == 0.06
    assert result.words[1]["start"] == 0.12
    assert result.words[1]["end"] == 0.16
    assert result.words[2]["start"] == 0.16
    baseline_ends = [word["end"] for word in baseline.words]
    detected_ends = [word["end"] for word in result.words if word["word"] != "[*]"]
    assert detected_ends == baseline_ends
    candidates = [event for event in result.events if event["type"] == "disfluency_candidate"]
    assert len(candidates) == 2
    assert candidates[0]["is_leading_word"] is True
    assert candidates[1]["is_trailing_word"] is True
    assert not any(event["type"] == "boundary_uncertainty" for event in result.events)


def test_refine_signal_collection_reports_stack_boundary_and_zero_tail() -> None:
    weights = np.zeros((5, 6), dtype=np.float32)
    weights[1, 1:4] = [0.2, 0.6, 0.2]
    weights[3, 2:5] = [0.2, 0.3, 0.5]

    result = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 4, (200, 1, 3, 6, 201)),
        path=[(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (4, 5)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -0.2, -0.3, -9.0],
        alignment_weights=weights.ravel(),
        collect_refine_signals=True,
        collect_attention_signals=True,
    )

    event_types = [event["type"] for event in result.events]
    assert event_types.count("alignment_stack") == 1
    assert event_types.count("boundary_uncertainty") == 2
    assert event_types.count("zero_duration_chunk_tail") == 1
    stack = next(event for event in result.events if event["type"] == "alignment_stack")
    assert stack["token_count"] == 3
    assert stack["active_frames"] == 0


def test_path_signal_collection_does_not_require_attention_weights() -> None:
    result = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 4, (200, 1, 3, 6, 201)),
        path=[(0, 0), (1, 1), (2, 1), (3, 1), (4, 1), (4, 5)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -0.2, -0.3, -9.0],
        collect_refine_signals=True,
    )

    assert [event["type"] for event in result.events] == [
        "alignment_stack",
        "zero_duration_chunk_tail",
    ]


def test_path_signal_collection_reports_long_token_and_decoder_motif() -> None:
    long_token = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(0, 2, (200, 1, 201)),
        path=[(0, 0), (1, 1), (2, 301), (2, 302)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-9.0, -0.1, -9.0],
        collect_refine_signals=True,
    )
    assert [event["type"] for event in long_token.events] == ["long_token_span"]

    repeated = fw_refine.align_span_words(
        span=fw_refine.TimestampSpan(
            0,
            10,
            (200, 1, 3, 1, 3, 1, 3, 1, 3, 1, 201),
        ),
        path=[(index, index) for index in range(11)] + [(10, 12)],
        tokenizer=_Tokenizer(),
        language="en",
        chosen_logprobs=[-0.1] * 11,
        collect_refine_signals=True,
    )
    motif = next(event for event in repeated.events if event["type"] == "decoder_repetition")
    assert motif["motif_token_count"] == 2
    assert motif["repeat_count"] == 4


def test_no_space_languages_match_wt_codes_and_names() -> None:
    assert not fw_refine.should_use_space("ja")
    assert not fw_refine.should_use_space("Chinese")
    assert fw_refine.should_use_space("en")


def test_group_transcribe_uses_the_backend_call() -> None:
    class Model:
        def __init__(self) -> None:
            self.options = None

        def transcribe_wt(self, audio, **options):
            self.options = options
            return {"segments": [], "language": "en"}

    model = Model()
    result = asr_transcribe._transcribe_with_teacher_force_fallback(
        model,
        np.zeros(160, dtype=np.float32),
        asr_transcribe._build_transcribe_kwargs(language="en"),
        group_start=0.0,
    )

    assert result == {"segments": [], "language": "en"}
    assert model.options is not None


def test_refine_backend_falls_back_to_teacher_force_instead_of_raising() -> None:
    """A one-pass trace desync must cost one group at most, never the run.
    The backend owns the retry: teacher-force alignment skips the pairing
    call here, so the retry has to force the backend's teacher-force path."""

    class Model:
        def __init__(self, always_fail: bool) -> None:
            self.always_fail = always_fail
            self.attempts: list[bool] = []

        def transcribe_wt(self, audio, **options):
            forced = bool(options.get("force_teacher_force"))
            self.attempts.append(forced)
            if self.always_fail or not forced:
                raise ValueError("word token groups do not consume the decoded text tokens")
            return {"segments": [], "language": "ja"}

    recovered = Model(always_fail=False)
    result = asr_transcribe._transcribe_with_teacher_force_fallback(
        recovered,
        np.zeros(160, dtype=np.float32),
        asr_transcribe._build_transcribe_kwargs(language="ja"),
        group_start=0.0,
    )
    assert result == {"segments": [], "language": "ja"}
    assert recovered.attempts == [False, True]

    dropped = Model(always_fail=True)
    assert asr_transcribe._transcribe_with_teacher_force_fallback(
        dropped,
        np.zeros(160, dtype=np.float32),
        asr_transcribe._build_transcribe_kwargs(language="ja"),
        group_start=0.0,
    ) is None
    assert dropped.attempts == [False, True]


def test_one_pass_trace_is_not_collected_for_text_only_decode() -> None:
    options = SimpleNamespace(
        temperatures=[0.0],
        beam_size=1,
        best_of=1,
        word_timestamps=False,
        without_timestamps=False,
    )

    assert not RefinedWhisperModel._can_refine_one_pass(options)
    options.word_timestamps = True
    assert RefinedWhisperModel._can_refine_one_pass(options)
    options.beam_size = 5
    assert RefinedWhisperModel._can_refine_one_pass(options)
    options.temperatures = [0.0, 0.2]
    assert not RefinedWhisperModel._can_refine_one_pass(options)



def _replaying_model(playback):
    """A RefinedWhisperModel with only the replay state populated."""

    model = RefinedWhisperModel.__new__(RefinedWhisperModel)
    model._playback = playback
    model._force_teacher_force = False
    model._pending_refine_trace = None
    model._real_audio_frames = 0
    model.feature_extractor = SimpleNamespace(nb_max_frames=3000)
    model.input_stride = 2
    return model


def _fake_result(tokens, logprobs, score):
    return SimpleNamespace(
        sequences_ids=[list(tokens)],
        token_logprobs=list(logprobs),
        refine_alignments=[],
        scores=[score],
    )


def test_batched_replay_serves_the_first_decode_from_the_batch() -> None:
    result = _fake_result((100, 200), (-0.1, -0.2), -0.5)
    model = _replaying_model(
        fw_refine_backend._Playback(
            encoder_output="batched-encoder-output",
            real_audio_frames=1400,
            result=result,
            length_penalty=1.0,
        )
    )

    encoder_output = model.encode(np.zeros((80, 3000), dtype=np.float32))
    served, average_logprob, no_speech, _ratio = model.generate_with_fallback(
        encoder_output, [1], _Tokenizer(), SimpleNamespace(length_penalty=1.0)
    )

    assert encoder_output == "batched-encoder-output"
    assert model._real_audio_frames == 1400
    assert served is result
    assert no_speech == 0.0
    assert math.isclose(average_logprob, -0.5 * (2**1.0) / 3)
    assert model._pending_refine_trace.tokens == (100, 200)


def test_batched_replay_is_spent_after_one_decode(monkeypatch) -> None:
    """faster-whisper seeks again when a decode stops before the end of the
    window. The batch holds nothing for that pass, so it has to reach the real
    encoder rather than silently reuse the first window's output."""

    from faster_whisper.transcribe import WhisperModel

    encoded: list[int] = []
    monkeypatch.setattr(
        WhisperModel,
        "encode",
        lambda self, features: encoded.append(np.asarray(features).shape[-1]),
    )
    model = _replaying_model(
        fw_refine_backend._Playback(
            encoder_output="batched-encoder-output",
            real_audio_frames=1400,
            result=_fake_result((1,), (-0.1,), -0.1),
            length_penalty=1.0,
        )
    )

    assert model.encode(np.zeros((80, 3000), dtype=np.float32)) == "batched-encoder-output"
    assert encoded == []
    model.encode(np.zeros((80, 3000), dtype=np.float32))
    assert encoded == [3000]


def test_batch_window_rejects_audio_beyond_one_encoder_window() -> None:
    extractor = lambda audio: np.zeros((80, 1502), dtype=np.float32)
    extractor.nb_max_frames = 1500
    extractor.time_per_frame = 0.02
    model = SimpleNamespace(feature_extractor=extractor)

    with pytest.raises(ValueError, match="up to 30s"):
        fw_refine_backend._encoder_window(model, np.zeros(16000, dtype=np.float32))


def test_transcribe_batch_requires_an_explicit_language() -> None:
    """Auto-detection would give items different prompts; CTranslate2 needs one
    prompt shape for the whole batch."""

    assert fw_refine_backend.transcribe_batch(object(), []) == []
    with pytest.raises(ValueError, match="explicit language"):
        fw_refine_batch_without_language()


def fw_refine_batch_without_language():
    return fw_refine_backend.transcribe_batch(
        object(), [np.zeros(16000, dtype=np.float32)]
    )


def test_missing_gemm_backend_names_the_device_and_the_remedy() -> None:
    """CTranslate2 reports this from deep inside encode as a bare 'No SGEMM
    backend on CPU'; the build flag that causes it is not discoverable from
    get_supported_compute_types, so the message has to carry the pointer."""

    original = RuntimeError("No SGEMM backend on CPU")
    raised = fw_refine_backend._missing_gemm_backend(original, "cpu")

    assert raised is not original
    assert "cpu" in str(raised)
    assert "ct2-patches" in str(raised)
    assert "--asr-backend wt" in str(raised)


def test_unrelated_runtime_errors_pass_through_untouched() -> None:
    original = RuntimeError("something else entirely")

    assert fw_refine_backend._missing_gemm_backend(original, "cuda") is original
