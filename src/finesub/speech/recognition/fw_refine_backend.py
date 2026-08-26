"""Patched faster-whisper backend for the WT-compatible one-pass path."""

from __future__ import annotations

import contextlib
import inspect
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from faster_whisper.transcribe import WhisperModel, get_compression_ratio

from ..runtime.cuda_libs import ensure_cublas_available
from .fw_refine import (
    AlignedSpan,
    TimestampSpan,
    align_span_words,
)


@dataclass(frozen=True)
class _PendingAlignment:
    span: TimestampSpan
    path: tuple[tuple[int, int], ...]
    frame_start: int = 0
    weights: tuple[float, ...] = ()


@dataclass(frozen=True)
class _PendingTrace:
    tokens: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    refine_alignments: tuple[_PendingAlignment, ...]


def _missing_gemm_backend(exc: RuntimeError, device: str) -> RuntimeError:
    """Explain a CTranslate2 build that has no matrix backend for this device.

    ``get_supported_compute_types`` does not reveal this -- a CUDA-only build
    still reports float32 for CPU -- so the first encode is where it surfaces,
    as a bare "No SGEMM backend on CPU" from deep inside the library.
    """

    if "GEMM" not in str(exc):
        return exc
    return RuntimeError(
        f"the patched CTranslate2 build has no matrix backend for device "
        f"'{device}' ({exc}). It must be compiled with a backend for that "
        f"device -- see tools/wt_refine_port/ct2-patches/README.md for the "
        f"required build flags. Until then use --asr-backend wt on machines "
        f"where fw-refine cannot run."
    )


def _trace_from_result(result: Any) -> _PendingTrace:
    """Lift one CTranslate2 generation result into the refine trace.

    Shared by the single-window path and the batched driver so the two cannot
    drift in how they read the compact trace.
    """

    tokens = tuple(int(token) for token in result.sequences_ids[0])
    selected_logprobs = tuple(float(value) for value in result.token_logprobs)
    if len(selected_logprobs) < len(tokens):
        raise RuntimeError("CTranslate2 returned an incomplete WT refine trace")
    return _PendingTrace(
        tokens=tokens,
        token_logprobs=selected_logprobs,
        refine_alignments=tuple(
            _PendingAlignment(
                span=TimestampSpan(
                    int(item.token_start),
                    int(item.token_end),
                    tuple(int(token) for token in item.alignment_tokens),
                    bool(item.unfinished),
                ),
                path=tuple(
                    (int(token), int(frame)) for token, frame in item.alignments
                ),
                frame_start=int(getattr(item, "alignment_frame_start", 0)),
                weights=tuple(
                    float(value) for value in getattr(item, "alignment_weights", ())
                ),
            )
            for item in result.refine_alignments
        ),
    )


@dataclass
class _Playback:
    """A decode already produced by the batched driver, replayed into the
    normal transcribe path so segment and word assembly stay single-sourced.

    It covers the window's *first* decode only. faster-whisper's seek loop can
    ask for a second one when the decode stops before the end of the audio; the
    batch has nothing for that, so those fall through to the ordinary path and
    run sequentially. Most windows never get there.
    """

    encoder_output: Any
    real_audio_frames: int
    result: Any
    length_penalty: float
    encoder_taken: bool = False
    result_taken: bool = False


class RefinedWhisperModel(WhisperModel):
    """WhisperModel that reuses decoder attention for word timing.

    The exact one-pass contract (one temperature and one returned hypothesis)
    is intercepted for greedy and beam search. Other decoding modes retain
    faster-whisper's normal generation and teacher-force word alignment.
    """

    def __init__(self, *args: Any, refine_sec: float = 1.0, **kwargs: Any) -> None:
        # CTranslate2 loads cuBLAS by name from C++, so its directory has to be
        # on the process search path before any GPU work -- and nothing here
        # ships it. Until now it was found only because torch had happened to
        # be imported first; `cuda_libs` asks for it deliberately instead.
        # `auto` is included: it resolves to CUDA whenever CUDA is usable.
        if str(kwargs.get("device", "auto")).strip().lower() != "cpu":
            ensure_cublas_available()
        super().__init__(*args, **kwargs)
        generate_doc = inspect.getdoc(self.model.generate) or ""
        if "return_refine_paths" not in generate_doc:
            raise RuntimeError(
                "CTranslate2 was not built with the WT refine trace extension"
            )
        self.refine_frames = max(0, round(float(refine_sec) / self.time_precision))
        self._pending_refine_trace: _PendingTrace | None = None
        self._segment_confidences: deque[float] = deque()
        self._segment_alignment_events: deque[
            tuple[int, tuple[int, ...], tuple[dict[str, object], ...]]
        ] = deque()
        self._real_audio_frames = 0
        self._detect_disfluencies = False
        self._collect_refine_signals = False
        self._collect_attention_signals = False
        # Escape hatch for the caller's retry ladder: the one-pass path pairs
        # word groups with the decoder trace, so a desync there is recoverable
        # by re-running through faster-whisper's own teacher-force alignment.
        self._force_teacher_force = False
        self._playback: _Playback | None = None

    def encode(self, features: np.ndarray):
        """Remember the non-padding encoder boundary for compact path generation."""

        if self._playback is not None and not self._playback.encoder_taken:
            self._playback.encoder_taken = True
            self._real_audio_frames = self._playback.real_audio_frames
            return self._playback.encoder_output
        values = np.asarray(features)
        frame_count = int(values.shape[-1])
        if frame_count > 1:
            tail = values[..., -1]
            index = frame_count - 2
            while index >= 0 and np.array_equal(values[..., index], tail):
                index -= 1
            frame_count = index + 2
        self._real_audio_frames = min(
            self.feature_extractor.nb_max_frames // self.input_stride,
            max(1, math.ceil(frame_count / self.input_stride)),
        )
        try:
            return super().encode(features)
        except RuntimeError as exc:
            raise _missing_gemm_backend(exc, self.model.device) from exc

    def transcribe(self, *args: Any, **kwargs: Any):
        self._pending_refine_trace = None
        self._segment_confidences.clear()
        self._segment_alignment_events.clear()
        return super().transcribe(*args, **kwargs)

    @staticmethod
    def _can_refine_one_pass(options: Any) -> bool:
        return (
            list(options.temperatures) == [0.0]
            and int(options.beam_size) >= 1
            and options.word_timestamps
            and not options.without_timestamps
        )

    def generate_with_fallback(
        self,
        encoder_output,
        prompt: list[int],
        tokenizer,
        options,
    ):
        if self._playback is not None and not self._playback.result_taken:
            # The batched driver already ran this decode; assembling it here
            # keeps segment/word construction identical to the single path.
            self._playback.result_taken = True
            return self._finish_generation(
                self._playback.result, tokenizer, self._playback.length_penalty
            )

        if self._force_teacher_force or not self._can_refine_one_pass(options):
            self._pending_refine_trace = None
            return super().generate_with_fallback(
                encoder_output,
                prompt,
                tokenizer,
                options,
            )

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        max_length = (
            len(prompt) + options.max_new_tokens
            if options.max_new_tokens is not None
            else self.max_length
        )
        if max_length > self.max_length:
            raise ValueError(
                f"prompt and max_new_tokens require {max_length} tokens, "
                f"but the model limit is {self.max_length}"
            )

        result = self.model.generate(
            encoder_output,
            [prompt],
            beam_size=int(options.beam_size),
            num_hypotheses=1,
            patience=options.patience,
            length_penalty=options.length_penalty,
            repetition_penalty=options.repetition_penalty,
            no_repeat_ngram_size=options.no_repeat_ngram_size,
            max_length=max_length,
            return_scores=True,
            return_no_speech_prob=True,
            return_refine_paths=True,
            return_refine_weights=(
                self._detect_disfluencies or self._collect_attention_signals
            ),
            refine_frames=self.refine_frames,
            real_audio_frames=self._real_audio_frames,
            suppress_blank=options.suppress_blank,
            suppress_tokens=options.suppress_tokens,
            max_initial_timestamp_index=max_initial_timestamp_index,
        )[0]
        return self._finish_generation(result, tokenizer, options.length_penalty)

    def _finish_generation(self, result: Any, tokenizer, length_penalty: float):
        """Turn one generation result into faster-whisper's return tuple and
        install its refine trace for the word-timestamp pass."""

        trace = _trace_from_result(result)
        seq_len = len(trace.tokens)
        cumulative_logprob = result.scores[0] * (seq_len**length_penalty)
        average_logprob = cumulative_logprob / (seq_len + 1)
        compression_ratio = get_compression_ratio(
            tokenizer.decode(trace.tokens).strip()
        )
        self._pending_refine_trace = trace
        return result, average_logprob, 0.0, compression_ratio

    def _split_segments_by_timestamps(
        self,
        tokenizer,
        tokens: list[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ):
        """Retain WT's decoding-limit tail instead of silently re-decoding it."""

        original_seek = seek
        segments, next_seek, single_timestamp_ending = super()._split_segments_by_timestamps(
            tokenizer,
            tokens,
            time_offset,
            segment_size,
            segment_duration,
            seek,
        )
        pending = self._pending_refine_trace
        if pending is None or not pending.refine_alignments:
            return segments, next_seek, single_timestamp_ending
        span = pending.refine_alignments[-1].span
        if not span.unfinished:
            return segments, next_seek, single_timestamp_ending
        raw_tokens = list(tokens[span.token_start : span.token_end + 1])
        if segments and list(segments[-1]["tokens"]) == raw_tokens:
            return segments, original_seek + segment_size, True
        start_position = raw_tokens[0] - tokenizer.timestamp_begin
        segments.append(
            {
                "seek": original_seek,
                "start": time_offset + start_position * self.time_precision,
                "end": time_offset + segment_duration,
                "tokens": raw_tokens,
            }
        )
        return segments, original_seek + segment_size, True

    def _align_pending_segments(
        self,
        segments: list[list[dict[str, Any]]],
        tokenizer,
        num_frames: int,
    ) -> list[AlignedSpan] | None:
        pending = self._pending_refine_trace
        if pending is None or len(segments) != 1:
            return None
        decoded_segments = segments[0]
        spans = [item.span for item in pending.refine_alignments]
        if len(spans) != len(decoded_segments):
            return None
        if any(
            tuple(int(token) for token in segment["tokens"])
            != pending.tokens[span.token_start : min(span.token_end + 1, len(pending.tokens))]
            for segment, span in zip(decoded_segments, spans)
        ):
            return None
        aligned: list[AlignedSpan] = []
        for item in pending.refine_alignments:
            span = item.span
            aligned.append(
                align_span_words(
                    span=span,
                    path=item.path,
                    tokenizer=tokenizer,
                    language=tokenizer.language_code,
                    chosen_logprobs=pending.token_logprobs[
                        span.token_start : span.token_end + 1
                    ],
                    alignment_weights=item.weights,
                    alignment_frame_start=item.frame_start,
                    detect_disfluencies=self._detect_disfluencies,
                    collect_refine_signals=self._collect_refine_signals,
                    collect_attention_signals=self._collect_attention_signals,
                )
            )
        return aligned

    def add_word_timestamps(
        self,
        segments,
        tokenizer,
        encoder_output,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ):
        aligned = self._align_pending_segments(segments, tokenizer, num_frames)
        self._pending_refine_trace = None
        if aligned is None:
            return super().add_word_timestamps(
                segments,
                tokenizer,
                encoder_output,
                num_frames,
                prepend_punctuations,
                append_punctuations,
                last_speech_timestamp,
            )

        time_offset = segments[0][0]["seek"] / self.frames_per_second
        for segment, result in zip(segments[0], aligned):
            words = [
                {
                    "word": str(word["word"]),
                    "start": round(time_offset + float(word["start"]), 2),
                    "end": round(time_offset + float(word["end"]), 2),
                    "probability": float(word["confidence"]),
                }
                for word in result.words
            ]
            segment["words"] = words
            translated_events: list[dict[str, object]] = []
            for event in result.events:
                translated = dict(event)
                for field in (
                    "start",
                    "end",
                    "original_start",
                    "refined_start",
                    "peak_time",
                ):
                    value = translated.get(field)
                    if isinstance(value, (int, float)):
                        translated[field] = round(time_offset + float(value), 3)
                translated_events.append(translated)
            self._segment_alignment_events.append(
                (
                    int(segment["seek"]),
                    tuple(int(token) for token in segment["tokens"]),
                    tuple(translated_events),
                )
            )
            if words:
                segment["start"] = words[0]["start"]
                segment["end"] = words[-1]["end"]
                last_speech_timestamp = float(segment["end"])
                self._segment_confidences.append(result.confidence)
        return last_speech_timestamp

    def take_segment_confidence(self, words: Iterable[Any]) -> float:
        if self._segment_confidences:
            return self._segment_confidences.popleft()
        probabilities = [
            max(float(getattr(word, "probability", 0.0)), np.finfo(float).tiny)
            for word in words
        ]
        if not probabilities:
            return 0.0
        return round(math.exp(sum(math.log(value) for value in probabilities) / len(probabilities)), 3)

    def take_segment_alignment_events(
        self,
        segment: Any,
    ) -> list[dict[str, object]]:
        if not self._segment_alignment_events:
            return []
        expected = (int(segment.seek), tuple(int(token) for token in segment.tokens))
        seek, tokens, events = self._segment_alignment_events.popleft()
        if (seek, tokens) != expected:
            self._segment_alignment_events.clear()
            return []
        return [dict(event) for event in events]

    def transcribe_wt(self, audio: np.ndarray, **options: Any) -> dict[str, object]:
        """Accept the small WT option surface used by the production caller."""

        beam_size = int(options.get("beam_size") or 1)
        best_of = int(options.get("best_of") or beam_size)
        previous_detect = self._detect_disfluencies
        previous_collect = self._collect_refine_signals
        previous_attention = self._collect_attention_signals
        previous_teacher_force = self._force_teacher_force
        self._detect_disfluencies = bool(options.get("detect_disfluencies", False))
        self._collect_refine_signals = bool(options.get("collect_refine_signals", False))
        self._collect_attention_signals = bool(
            options.get("collect_attention_signals", False)
        )
        self._force_teacher_force = bool(options.get("force_teacher_force", False))
        if self._detect_disfluencies or self._collect_attention_signals:
            generate_doc = inspect.getdoc(self.model.generate) or ""
            if "return_refine_weights" not in generate_doc:
                raise RuntimeError(
                    "CTranslate2 was not built with WT disfluency weight support"
                )
        try:
            return transcribe_to_wt_result(
                self,
                audio,
                language=options.get("language"),
                beam_size=beam_size,
                best_of=best_of,
                temperature=float(options.get("temperature", 0.0)),
                condition_on_previous_text=True,
                vad_filter=False,
            )
        finally:
            self._detect_disfluencies = previous_detect
            self._collect_refine_signals = previous_collect
            self._collect_attention_signals = previous_attention
            self._force_teacher_force = previous_teacher_force


MAX_BATCH_WINDOW_SEC = 30.0


def _encoder_window(model: RefinedWhisperModel, audio: np.ndarray) -> np.ndarray:
    """The single encoder window faster-whisper's seek loop would build.

    Padding happens in the feature domain, exactly as ``generate_segments``
    does it. Zero-padding the audio to 30s instead is not equivalent: the
    log-mel of digital silence is a large negative constant, not zero, so the
    encoder would see different content and decode differently.
    """

    from faster_whisper.audio import pad_or_trim

    features = model.feature_extractor(np.asarray(audio, dtype=np.float32))
    content_frames = features.shape[-1] - 1
    segment_size = min(model.feature_extractor.nb_max_frames, content_frames)
    if content_frames > model.feature_extractor.nb_max_frames:
        raise ValueError(
            f"batched fw-refine takes windows up to {MAX_BATCH_WINDOW_SEC:.0f}s, "
            f"got {content_frames * model.feature_extractor.time_per_frame:.2f}s"
        )
    return pad_or_trim(features[:, :segment_size])


def _concat_encoder_outputs(outputs: list[Any]) -> Any:
    """Join per-window encoder outputs into one batch without leaving the GPU.

    Encoder batching buys no throughput (measured flat from B=1 to B=16), so
    windows are encoded one at a time and only the decoder is batched. That
    makes this concat the seam between the two -- keep it zero-copy.
    """

    import ctranslate2
    import torch

    if len(outputs) == 1:
        return outputs[0]
    tensors = [torch.as_tensor(item, device=_storage_device(item)) for item in outputs]
    return ctranslate2.StorageView.from_array(torch.cat(tensors, dim=0))


def _storage_device(storage: Any) -> str:
    return "cuda" if str(getattr(storage, "device", "cpu")).startswith("cuda") else "cpu"


def transcribe_batch(
    model: RefinedWhisperModel,
    audios: list[np.ndarray],
    **options: Any,
) -> list[dict[str, object]]:
    """Decode several <=30s windows in one batched generate call.

    Split-encode: each window is encoded at batch 1 and only the decoder runs
    batched. Assembly then replays each item through the ordinary transcribe
    path, so segments, words and events are built by exactly the code the
    single-window backend uses.

    The caller owns batch composition. This function only enforces what the
    contract cannot express in types -- window length, a single explicit
    language, and no conditioning prompt (a non-empty prompt has to fall back
    to the sequential path, since CTranslate2 requires one prompt shape for the
    whole batch).
    """

    if not audios:
        return []
    language = options.get("language")
    if not language:
        raise ValueError("batched fw-refine needs an explicit language")
    beam_size = int(options.get("beam_size") or 1)

    windows = [_encoder_window(model, audio) for audio in audios]
    encoder_outputs = []
    real_frames = []
    for features in windows:
        encoder_outputs.append(model.encode(np.stack([features])))
        real_frames.append(model._real_audio_frames)

    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import get_suppressed_tokens

    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language=language,
    )
    prompt = model.get_prompt(tokenizer, [], without_timestamps=False)
    # -1 is a faster-whisper convention, not a CTranslate2 one: it has to be
    # expanded into the non-speech and special token ids here, or the batched
    # decode silently runs with no suppression at all.
    suppress_tokens = get_suppressed_tokens(tokenizer, [-1])
    results = model.model.generate(
        _concat_encoder_outputs(encoder_outputs),
        [prompt] * len(audios),
        beam_size=beam_size,
        num_hypotheses=1,
        max_length=model.max_length,
        return_scores=True,
        return_no_speech_prob=True,
        return_refine_paths=True,
        return_refine_weights=(
            model._detect_disfluencies or model._collect_attention_signals
        ),
        refine_frames=model.refine_frames,
        real_audio_frames=real_frames,
        suppress_blank=True,
        suppress_tokens=suppress_tokens,
        max_initial_timestamp_index=int(round(1.0 / model.time_precision)),
    )
    if len(results) != len(audios):
        raise RuntimeError(
            f"CTranslate2 returned {len(results)} results for {len(audios)} windows"
        )

    outputs: list[dict[str, object]] = []
    for audio, encoder_output, frames, result in zip(
        audios, encoder_outputs, real_frames, results
    ):
        model._playback = _Playback(
            encoder_output=encoder_output,
            real_audio_frames=frames,
            result=result,
            length_penalty=1.0,
        )
        try:
            outputs.append(
                transcribe_to_wt_result(
                    model,
                    audio,
                    language=language,
                    beam_size=beam_size,
                    best_of=beam_size,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                )
            )
        finally:
            model._playback = None
    return outputs


def transcribe_to_wt_result(
    model: RefinedWhisperModel,
    audio: np.ndarray,
    **options: Any,
) -> dict[str, object]:
    """Materialize faster-whisper output in the WT result schema."""

    segments, info = model.transcribe(audio, word_timestamps=True, **options)
    output: list[dict[str, object]] = []
    for segment in segments:
        words = list(segment.words or ())
        events = model.take_segment_alignment_events(segment)
        item: dict[str, object] = {
            "text": segment.text,
            "start": float(segment.start),
            "end": float(segment.end),
            "tokens": list(segment.tokens),
            "words": [
                {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                    "confidence": float(word.probability),
                }
                for word in words
            ],
            "confidence": model.take_segment_confidence(words),
            "no_speech_prob": float(segment.no_speech_prob),
            "avg_logprob": float(segment.avg_logprob),
        }
        if events:
            item["alignment_events"] = events
        output.append(item)
    return {"segments": output, "language": info.language}


_MODEL_LOAD_LOCK = threading.Lock()


class FwRefineModelPool:
    """Independent patched faster-whisper instances for ASR shard workers."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        size: int,
        refine_sec: float = 1.0,
        revision: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._size = max(1, int(size))
        self._refine_sec = float(refine_sec)
        self._revision = revision
        self._idle: list[RefinedWhisperModel] = []
        self._loaded = 0
        self._condition = threading.Condition()

    @contextlib.contextmanager
    def lease(self):
        model = self._acquire()
        try:
            yield model
        finally:
            self._release(model)

    def _acquire(self) -> RefinedWhisperModel:
        with self._condition:
            while True:
                if self._idle:
                    return self._idle.pop()
                if self._loaded < self._size:
                    self._loaded += 1
                    break
                self._condition.wait()
        try:
            with _MODEL_LOAD_LOCK:
                # The stage may have just ensured and verified the weights at a
                # pinned revision; loading without it would let the hub
                # re-resolve `main` past the snapshot that was checked.
                return RefinedWhisperModel(
                    self._model_name,
                    device=self._device,
                    compute_type=(
                        "float16"
                        if self._device.strip().lower().startswith("cuda")
                        else "float32"
                    ),
                    revision=self._revision,
                    refine_sec=self._refine_sec,
                )
        except BaseException:
            with self._condition:
                self._loaded -= 1
                self._condition.notify()
            raise

    def _release(self, model: RefinedWhisperModel) -> None:
        with self._condition:
            self._idle.append(model)
            self._condition.notify()

    def warm(self) -> None:
        models = []
        try:
            for _ in range(self._size):
                models.append(self._acquire())
        finally:
            for model in models:
                self._release(model)

    def close(self) -> None:
        with self._condition:
            models, self._idle, self._loaded = self._idle, [], 0
        models.clear()
