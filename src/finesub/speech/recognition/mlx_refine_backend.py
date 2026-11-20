"""Apple-Silicon MLX backend with one-pass WT-compatible word alignment.

The decode recorder is intentionally tied to mlx-whisper 0.4.3.  It replaces
two small strategy objects in that release's DecodingTask instead of forking
the complete seek loop: the selected-token log probabilities and the selected
alignment-head QK row are retained at each greedy step, then the existing
FineSub refine core consumes that trace.  The upstream word-timestamp pass is
kept as the recovery path when a trace cannot be paired defensively.

Adapted against the MIT-licensed sources in ml-explore/mlx-examples:
whisper/mlx_whisper/{decoding.py,transcribe.py,timing.py} at release 0.4.3.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import math
import threading
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime import hf_weights, phase_timing
from . import asr_backend
from .fw_refine import (
    AlignedSpan,
    TimestampSpan,
    align_span_words,
    prepare_alignment_weights,
    repair_nonincreasing_end_span,
    split_timestamp_spans,
    trace_alignment_path,
)

MLX_WHISPER_VERSION = "0.4.3"
MLX_VERSION = "0.32.2"
TRACE_CONTRACT_VERSION = 3


@dataclass(frozen=True)
class MlxDecodeTrace:
    tokens: tuple[int, ...]
    token_logprobs: tuple[float, ...]
    attention: np.ndarray
    tail_logprobs: np.ndarray
    endpoint_logprobs: dict[int, np.ndarray] = field(default_factory=dict)


def _require_compatible_runtime() -> None:
    expected = {"mlx-whisper": MLX_WHISPER_VERSION, "mlx": MLX_VERSION}
    problems = []
    for package, wanted in expected.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "not installed"
        if actual != wanted:
            problems.append(f"{package}=={wanted} required (found {actual})")
    if problems:
        raise RuntimeError(
            "incompatible MLX ASR runtime: "
            + "; ".join(problems)
            + '. Reinstall the pinned macOS ASR dependencies with `pip install -e ".[asr]"`.'
        )


def _require_internal_contract() -> None:
    """Fail before model loading when 0.4.3 internals were repackaged incompatibly."""

    decoding = importlib.import_module("mlx_whisper.decoding")
    transcribe = importlib.import_module("mlx_whisper.transcribe")
    required = (
        (decoding, "DecodingTask"),
        (decoding, "GreedyDecoder"),
        (transcribe, "ModelHolder"),
        (transcribe, "add_word_timestamps"),
    )
    missing = [name for module, name in required if not hasattr(module, name)]
    if not missing:
        for method in ("_get_audio_features", "_detect_language", "_get_initial_tokens"):
            if not callable(getattr(decoding.DecodingTask, method, None)):
                missing.append(f"DecodingTask.{method}")
    decoder_parameters = (
        inspect.signature(decoding.GreedyDecoder.update).parameters if not missing else {}
    )
    if missing or not {"tokens", "logits", "sum_logprobs"}.issubset(decoder_parameters):
        details = ", ".join(missing) if missing else "GreedyDecoder.update signature"
        raise RuntimeError(
            "mlx-whisper 0.4.3 internal decode contract does not match FineSub "
            f"trace adapter ({details}); reinstall the pinned lock instead of continuing"
        )
    from .mlx_seek import adapt_seek_source

    adapt_seek_source(inspect.getsource(transcribe.transcribe))


class _TraceInference:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.keep_device = getattr(model, "keep_attention_on_device", False)
        self.kv_cache = None
        self.rows: list[np.ndarray] = []

    def logits(self, tokens: Any, audio_features: Any):
        import mlx.core as mx

        logits, self.kv_cache, cross_qk = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        heads = self.model.alignment_heads.tolist()
        selected = mx.stack(
            [cross_qk[int(layer)][0, int(head), -1] for layer, head in heads]
        )
        self.rows.append(selected.astype(mx.float32) if self.keep_device else np.asarray(selected, dtype=np.float32))
        return logits.astype(mx.float32)

    def rearrange_kv_cache(self, source_indices: Iterable[int]) -> None:
        from mlx.utils import tree_map

        values = list(source_indices)
        if values != list(range(len(values))):
            self.kv_cache = tree_map(lambda item: item[values], self.kv_cache)

    def reset(self) -> None:
        self.kv_cache = None
        self.rows.clear()


def _recording_decoder(base: Any, recorder: _TraceDecodingTask):
    class RecordingGreedy(base):
        def update(self, tokens, logits, sum_logprobs):
            import mlx.core as mx

            active = bool(tokens[0, -1].item() != self.eot)
            updated, completed, sums = super().update(tokens, logits, sum_logprobs)
            if not active:
                return updated, completed, sums
            selected = updated[:, -1]
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            chosen = mx.take_along_axis(logprobs, selected[:, None], axis=1).squeeze(1)
            mx.eval(selected, chosen, logprobs)
            recorder.selected_tokens.extend(int(item) for item in selected.tolist())
            recorder.selected_logprobs.extend(float(item) for item in chosen.tolist())
            recorder.tail_logprobs = np.asarray(logprobs[0], dtype=np.float32)
            if int(selected[0].item()) >= recorder._task.tokenizer.timestamp_begin:
                recorder.endpoint_logprobs[len(recorder.selected_tokens) - 1] = (
                    recorder.tail_logprobs.copy()
                )
            return updated, completed, sums

    return RecordingGreedy


class _TraceDecodingTask:
    """Composition wrapper around the pinned upstream DecodingTask."""

    def __init__(self, model: Any, options: Any) -> None:
        decoding = importlib.import_module("mlx_whisper.decoding")
        self._task = decoding.DecodingTask(model, options)
        self.selected_tokens: list[int] = []
        self.selected_logprobs: list[float] = []
        self.tail_logprobs = np.empty(0, dtype=np.float32)
        self.endpoint_logprobs: dict[int, np.ndarray] = {}
        self.inference = _TraceInference(model)
        self._task.inference = self.inference
        decoder_type = _recording_decoder(decoding.GreedyDecoder, self)
        self._task.decoder = decoder_type(options.temperature, self._task.tokenizer.eot)

    def run(self, mel: Any):
        return self._task.run(mel)

    def trace(self) -> MlxDecodeTrace | None:
        if not self.inference.rows or not self.selected_tokens:
            return None
        if self.inference.keep_device:
            import mlx.core as mx
            attention = mx.stack(self.inference.rows[:len(self.selected_tokens)], axis=0)
        else:
            attention = np.stack(self.inference.rows[: len(self.selected_tokens)], axis=0)
        return MlxDecodeTrace(
            tokens=tuple(self.selected_tokens),
            token_logprobs=tuple(self.selected_logprobs),
            attention=attention,
            tail_logprobs=self.tail_logprobs.copy(),
            endpoint_logprobs=dict(self.endpoint_logprobs),
        )


def _geometric_confidence(words: Iterable[dict[str, object]]) -> float:
    values = [
        max(
            float(word.get("probability", word.get("confidence", 0.0))),
            np.finfo(float).tiny,
        )
        for word in words
    ]
    return (
        round(math.exp(sum(math.log(value) for value in values) / len(values)), 3)
        if values
        else 0.0
    )


def _trace_spans(trace: MlxDecodeTrace, tokenizer: Any) -> list[TimestampSpan]:
    tokens = tuple(token for token in trace.tokens if token != tokenizer.eot)
    spans = split_timestamp_spans(tokens, timestamp_begin=tokenizer.timestamp_begin)
    consumed = spans[-1].token_end + 1 if spans else 0
    tail = tokens[consumed:]
    if len(tail) >= 2 and tail[0] >= tokenizer.timestamp_begin:
        if len(trace.tokens) > len(tokens):
            spans.append(
                TimestampSpan(
                    consumed,
                    len(tokens),
                    (*tail, int(tokenizer.eot)),
                )
            )
        else:
            spans.append(
                TimestampSpan(consumed, len(tokens) - 1, tuple(tail), unfinished=True)
            )
    return spans


class MlxRefineModel:
    capabilities = frozenset()
    accepted_decode_options = frozenset({
        "language", "temperature", "detect_disfluencies", "collect_refine_signals",
        "collect_attention_signals", "force_teacher_force",
    })
    backend_name = "mlx-refine"
    trace_contract_version = TRACE_CONTRACT_VERSION

    def __init__(
        self,
        model_name: str,
        *,
        refine_sec: float = 1.0,
        load: hf_weights.HfLoad = hf_weights.UNMANAGED,
        condition_on_previous_text: bool = True,
    ) -> None:
        _require_compatible_runtime()
        _require_internal_contract()
        import mlx.core as mx
        from huggingface_hub import snapshot_download
        from mlx_whisper.load_models import load_model

        self.model_name = str(model_name)
        self.condition_on_previous_text = condition_on_previous_text
        if self.model_name == asr_backend.MLX_DEFAULT_MODEL and load.revision is None:
            load = hf_weights.HfLoad(asr_backend.MLX_MODEL_REVISION, False)
        self.model_revision = load.revision
        self.runtime_versions = {
            "mlx_whisper": version("mlx-whisper"),
            "mlx": version("mlx"),
        }
        self.refine_frames = max(0, round(float(refine_sec) / 0.02))
        self._latest_trace: MlxDecodeTrace | None = None
        self._force_teacher_force = False
        self._detect_disfluencies = False
        self._collect_refine_signals = False
        self._collect_attention_signals = False
        self.one_pass_count = 0
        self.teacher_force_count = 0
        self.peak_memory_bytes = 0
        self.fallback_reasons: Counter[str] = Counter()
        self.search_stats = Counter({
            f"{mode}_{metric}": 0
            for mode in ("greedy", "beam")
            for metric in ("windows", "one_pass", "teacher_force", "decode_sec",
                           "teacher_force_sec", "peak_memory_bytes")
        })
        self._search_mode = "greedy"
        self._last_fallback_reason = "trace-unavailable"

        def build(plan: hf_weights.HfLoad):
            source = self.model_name
            if not Path(source).exists() and plan.revision is not None:
                source = snapshot_download(
                    repo_id=source,
                    revision=plan.revision,
                    local_files_only=plan.local_files_only,
                )
            return source, load_model(source, dtype=mx.float16)

        self.model_path, self.model = hf_weights.offline_first(
            build, load, what="mlx asr"
        )
        from .mlx_session import CachedEncoder, RequestModel
        self.encoder = CachedEncoder(self.model.encoder)
        self.request_model = RequestModel(self)
        self.alignment_mode = "cpu"
        self._request_lock = threading.RLock()

    def _decode(self, _model: Any, mel: Any, options: Any):
        mode = "beam" if (options.beam_size or 1) > 1 else "greedy"
        started = time.perf_counter()
        try:
            result = self._decode_inner(_model, mel, options)
            sink = getattr(self, "trace_sink", None)
            if sink is not None:
                sink(result, self._latest_trace, options)
            return result
        finally:
            self.search_stats[f"{mode}_decode_sec"] += time.perf_counter() - started

    def _decode_inner(self, _model: Any, mel: Any, options: Any):
        import mlx.core as mx

        self._search_mode = "beam" if (options.beam_size or 1) > 1 else "greedy"
        self.search_stats[f"{self._search_mode}_windows"] += 1
        single = mel.ndim == 2
        batch = mel[None] if single else mel
        if options.temperature != 0 or not single:
            self._latest_trace = None
            decoding = importlib.import_module("mlx_whisper.decoding")
            return (
                decoding.DecodingTask(_model, options).run(batch)[0]
                if single
                else decoding.DecodingTask(_model, options).run(batch)
            )
        task = _TraceDecodingTask(_model, options)
        results = task.run(batch)
        mx.eval(results[0].audio_features)
        self._latest_trace = task.trace()
        return results[0]

    def _align_trace(
        self,
        segments: list[dict[str, Any]],
        tokenizer: Any,
        num_frames: int,
    ) -> list[AlignedSpan] | None:
        trace = self._latest_trace
        if trace is None or not segments:
            self._last_fallback_reason = "trace-unavailable"
            return None
        spans = _trace_spans(trace, tokenizer)
        if len(spans) != len(segments):
            self._last_fallback_reason = "span-count-mismatch"
            return None
        decoded_tokens = tuple(
            token for token in trace.tokens if token != tokenizer.eot
        )
        if any(
            tuple(int(token) for token in segment["tokens"])
            != decoded_tokens[
                span.token_start : min(span.token_end + 1, len(decoded_tokens))
            ]
            for segment, span in zip(segments, spans)
        ):
            self._last_fallback_reason = "trace-token-desync"
            return None
        real_audio_frames = max(
            1, min(trace.attention.shape[-1], math.ceil(num_frames / 2))
        )
        aligned: list[AlignedSpan] = []
        for raw_span in spans:
            span = raw_span
            if (
                not span.unfinished
                and span.tokens[-1] >= tokenizer.timestamp_begin
                and span.tokens[-1] <= span.tokens[0]
                and (
                    span.token_end in trace.endpoint_logprobs
                    or trace.tail_logprobs.size
                )
            ):
                span = repair_nonincreasing_end_span(
                    span,
                    timestamp_begin=tokenizer.timestamp_begin,
                    endpoint_logprobs=trace.endpoint_logprobs.get(
                        span.token_end, trace.tail_logprobs
                    ),
                )
            start_frame = max(
                0, int(span.tokens[0]) - tokenizer.timestamp_begin - self.refine_frames
            )
            if span.tokens[-1] >= tokenizer.timestamp_begin:
                end_anchor = int(span.tokens[-1]) - tokenizer.timestamp_begin
            else:
                end_anchor = real_audio_frames
            frame_end = min(
                trace.attention.shape[-1],
                max(start_frame + len(span.tokens), end_anchor + self.refine_frames),
            )
            if frame_end <= start_frame:
                self._last_fallback_reason = "empty-frame-range"
                return None
            path, weights = self._alignment(trace.attention, span=span,
                frame_start=start_frame, frame_end=frame_end, real_audio_frames=real_audio_frames)
            aligned.append(
                align_span_words(
                    span=span,
                    path=path,
                    tokenizer=tokenizer,
                    language=getattr(tokenizer, "language", None),
                    chosen_logprobs=trace.token_logprobs[
                        span.token_start : span.token_end + 1
                    ],
                    alignment_weights=weights,
                    alignment_frame_start=start_frame,
                    detect_disfluencies=self._detect_disfluencies,
                    collect_refine_signals=self._collect_refine_signals,
                    collect_attention_signals=self._collect_attention_signals,
                )
            )
        return aligned

    def _alignment(self, attention, **kwargs):
        attention = np.asarray(attention)
        return (trace_alignment_path(attention, **kwargs), prepare_alignment_weights(attention, **kwargs))

    def _add_word_timestamps(
        self, original, *, segments, tokenizer, num_frames, **kwargs
    ):
        if self._force_teacher_force:
            self._last_fallback_reason = "forced"
            self._record_search_alignment("teacher_force")
            self.teacher_force_count += 1
            self.fallback_reasons["forced"] += 1
            with phase_timing.phase("asr.refine_teacher_force"):
                return self._teacher_force_timing(original,
                    segments=segments,
                    tokenizer=tokenizer,
                    num_frames=num_frames,
                    **kwargs,
                )
        try:
            with phase_timing.phase("asr.refine"):
                aligned = self._align_trace(segments, tokenizer, num_frames)
        except Exception as exc:  # noqa: BLE001 - any trace defect uses the safe oracle
            self._last_fallback_reason = f"{type(exc).__name__}"
            aligned = None
        finally:
            self._latest_trace = None
        if aligned is None:
            self._record_search_alignment("teacher_force")
            self.teacher_force_count += 1
            self.fallback_reasons[self._last_fallback_reason] += 1
            with phase_timing.phase("asr.refine_teacher_force"):
                return self._teacher_force_timing(original,
                    segments=segments,
                    tokenizer=tokenizer,
                    num_frames=num_frames,
                    **kwargs,
                )

        self.one_pass_count += 1
        self._record_search_alignment("one_pass")
        time_offset = float(segments[0]["seek"]) * 160 / 16000
        for segment, result in zip(segments, aligned):
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
            segment["confidence"] = result.confidence
            if result.events:
                translated = []
                for event in result.events:
                    item = dict(event)
                    for field in (
                        "start",
                        "end",
                        "original_start",
                        "refined_start",
                        "peak_time",
                    ):
                        value = item.get(field)
                        if isinstance(value, (int, float)):
                            item[field] = round(time_offset + float(value), 3)
                    translated.append(item)
                segment["alignment_events"] = translated
            if words:
                segment["start"] = words[0]["start"]
                segment["end"] = words[-1]["end"]

    def _record_search_alignment(self, outcome: str) -> None:
        if hasattr(self, "search_stats"):
            self.search_stats[f"{self._search_mode}_{outcome}"] += 1
            if outcome == "teacher_force":
                self.search_stats[f"{self._search_mode}_fallback:{self._last_fallback_reason}"] += 1

    def _teacher_force_timing(self, original, **kwargs):
        started = time.perf_counter()
        try:
            return original(**kwargs)
        finally:
            if hasattr(self, "search_stats"):
                self.search_stats[f"{self._search_mode}_teacher_force_sec"] += time.perf_counter() - started

    def transcribe_wt(self, audio: np.ndarray, **options: Any) -> dict[str, object]:
        asr_backend.reject_unknown(options, self.accepted_decode_options, backend=self.backend_name)
        with self._request_lock:
            return self._transcribe_wt(audio, **options)

    def _transcribe_wt(self, audio: np.ndarray, **options: Any) -> dict[str, object]:
        import mlx.core as mx

        beam_size = 1
        transcribe_mod = importlib.import_module("mlx_whisper.transcribe")
        previous = (
            self._force_teacher_force,
            self._detect_disfluencies,
            self._collect_refine_signals,
            self._collect_attention_signals,
        )
        self._force_teacher_force = bool(options.get("force_teacher_force", False))
        self._detect_disfluencies = bool(options.get("detect_disfluencies", False))
        self._collect_refine_signals = bool(
            options.get("collect_refine_signals", False)
        )
        self._collect_attention_signals = bool(
            options.get("collect_attention_signals", False)
        )
        original_timing = transcribe_mod.add_word_timestamps

        def timing_hook(**kwargs):
            return self._add_word_timestamps(original_timing, **kwargs)

        try:
            mx.reset_peak_memory()
            from .mlx_session import bind_transcribe
            run, self.transcribe_source_sha256 = bind_transcribe(self.request_model, timing_hook)
            self.runtime_versions["transcribe_source_sha256"] = self.transcribe_source_sha256
            with phase_timing.phase("asr.transcribe_window"):
                result = run(
                    audio,
                    path_or_hf_repo=str(self.model_path),
                    verbose=None,
                    temperature=float(options.get("temperature", 0.0)),
                    language=options.get("language"),
                    condition_on_previous_text=self.condition_on_previous_text,
                    word_timestamps=True,
                    fp16=True,
                    beam_size=None,
                )
            self.peak_memory_bytes = max(
                self.peak_memory_bytes, int(mx.get_peak_memory())
            )
            memory_key = f"{'beam' if beam_size > 1 else 'greedy'}_peak_memory_bytes"
            self.search_stats[memory_key] = max(self.search_stats[memory_key], int(mx.get_peak_memory()))
            self.search_stats["encoder_cache_hits"] = self.encoder.hits
            self.search_stats["encoder_cache_misses"] = self.encoder.misses
            self.search_stats["encoder_cache_evictions"] = self.encoder.evictions
            self.search_stats["encoder_cache_bytes"] = self.encoder.bytes
        finally:
            (
                self._force_teacher_force,
                self._detect_disfluencies,
                self._collect_refine_signals,
                self._collect_attention_signals,
            ) = previous

        output = []
        for segment in result.get("segments", []):
            words = [
                {
                    "word": word["word"],
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                    "confidence": float(word.get("probability", 0.0)),
                }
                for word in segment.get("words", [])
            ]
            item = {
                "text": str(segment.get("text", "")),
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "tokens": [int(token) for token in segment.get("tokens", [])],
                "words": words,
                "confidence": float(
                    segment.get(
                        "confidence", _geometric_confidence(segment.get("words", []))
                    )
                ),
                "no_speech_prob": float(segment.get("no_speech_prob", 0.0)),
                "avg_logprob": float(segment.get("avg_logprob", 0.0)),
            }
            events = segment.get("alignment_events")
            if isinstance(events, list) and events:
                item["alignment_events"] = [dict(event) for event in events]
            output.append(item)
        return {
            "segments": output,
            "language": result.get("language") or options.get("language"),
        }

    def close(self) -> None:
        self._latest_trace = None
        self.encoder.clear()
        self.encoder.encoder = None  # release the bound raw encoder/model too
        self.model = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001,S110 - cache cleanup is best effort
            pass

    def provenance(self) -> dict[str, object]:
        total = self.one_pass_count + self.teacher_force_count
        return {
            "runtime_versions": dict(self.runtime_versions),
            "model_revision": self.model_revision,
            "trace_contract_version": self.trace_contract_version,
            "one_pass_windows": self.one_pass_count,
            "teacher_force_windows": self.teacher_force_count,
            "one_pass_hit_rate": self.one_pass_count / total if total else None,
            "fallback_reasons": dict(self.fallback_reasons),
            "search_stats": dict(self.search_stats),
            "peak_memory_bytes": self.peak_memory_bytes,
        }


_MODEL_LOAD_LOCK = threading.Lock()


class MlxRefineModelPool:
    def __init__(
        self,
        model_name: str,
        *,
        size: int,
        refine_sec: float = 1.0,
        load: hf_weights.HfLoad = hf_weights.UNMANAGED,
    ) -> None:
        self._model_name = str(model_name)
        self._size = max(1, int(size))
        if self._size != 1:
            raise ValueError("MLX in-process pool size must be 1; parallel inference is not supported")
        self._refine_sec = float(refine_sec)
        self._load = load
        self._idle: list[MlxRefineModel] = []
        self._loaded = 0
        self._condition = threading.Condition()

    @contextlib.contextmanager
    def lease(self):
        model = self._acquire()
        try:
            yield model
        finally:
            self._release(model)

    def _acquire(self) -> MlxRefineModel:
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
                model = MlxRefineModel(
                    self._model_name,
                    refine_sec=self._refine_sec,
                    load=self._load,
                )
                return model
        except BaseException:
            with self._condition:
                self._loaded -= 1
                self._condition.notify()
            raise

    def _release(self, model: MlxRefineModel) -> None:
        with self._condition:
            self._idle.append(model)
            self._condition.notify()

    def warm(self) -> None:
        with self.lease():
            pass

    def close(self) -> None:
        with self._condition:
            models = list(self._idle)
            self._idle.clear()
            self._loaded -= len(models)
            self._condition.notify_all()
        for model in models:
            model.close()
