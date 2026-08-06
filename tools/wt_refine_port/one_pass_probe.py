"""Compare 1-pass greedy attention with CT2 teacher-force alignment.

The probe decodes one 30-second window once, slices the selected alignment-head
trace by decoded timestamp spans, applies WT's attention postprocessing in
Python, and compares the resulting DTW path with patched ``Whisper.align``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import median_filter

from tools.wt_refine_port.full_window_probe import parse_decoded_span
from tools.wt_refine_port.oracle import alignment_path
from tools.wt_refine_port.teacher_force_probe import (
    AUDIO_TIME_PER_TOKEN,
    N_FRAMES,
    SAMPLE_RATE,
    _add_dll_directory,
    _load_audio,
    _path_word_timings,
    _verify_patched_ct2,
    compare_word_timings,
)


@dataclass(frozen=True)
class TraceSpan:
    index: int
    token_start: int
    token_end: int
    raw_tokens: tuple[int, ...]


def split_timestamp_spans(
    tokens: Sequence[int],
    *,
    timestamp_begin: int,
) -> list[TraceSpan]:
    """Split FW's normal timestamp-delimited spans and retain trace indices."""

    values = tuple(int(token) for token in tokens)
    consecutive = [
        index
        for index in range(1, len(values))
        if values[index - 1] >= timestamp_begin and values[index] >= timestamp_begin
    ]
    single_timestamp_ending = (
        len(values) >= 2
        and values[-2] < timestamp_begin <= values[-1]
    )
    boundaries = list(consecutive)
    if single_timestamp_ending:
        boundaries.append(len(values))
    if not boundaries and values and values[0] >= timestamp_begin and values[-1] >= timestamp_begin:
        boundaries.append(len(values))

    spans: list[TraceSpan] = []
    start = 0
    for end in boundaries:
        raw = values[start:end]
        if len(raw) >= 3 and raw[0] >= timestamp_begin and raw[-1] >= timestamp_begin:
            spans.append(
                TraceSpan(
                    index=len(spans),
                    token_start=start,
                    token_end=end - 1,
                    raw_tokens=raw,
                )
            )
        start = end
    return spans


def trace_alignment_path(
    attention: np.ndarray,
    *,
    token_start: int,
    token_end: int,
    frame_start: int,
    frame_end: int,
    real_audio_frames: int,
    median_filter_width: int = 9,
) -> list[tuple[int, int]]:
    """Apply WT postprocessing to ``steps x heads x frames`` trace attention."""

    if attention.ndim != 3:
        raise ValueError("attention must have shape steps x heads x frames")
    rows = attention[token_start : token_end + 1]
    if not len(rows):
        return []
    weights = rows.transpose(1, 0, 2)[..., frame_start:frame_end].astype(
        np.float32,
        copy=True,
    )
    weights = median_filter(
        weights,
        size=(1, 1, median_filter_width),
        mode="reflect",
    )
    weights -= weights.max(axis=-1, keepdims=True)
    np.exp(weights, out=weights)
    weights /= weights.sum(axis=-1, keepdims=True)
    weights = weights.mean(axis=0)
    norms = np.linalg.norm(weights, axis=0, keepdims=True)
    weights /= np.maximum(norms, np.finfo(np.float32).tiny)

    local_real_end = real_audio_frames - frame_start
    if 0 < local_real_end < weights.shape[1]:
        weights[:-1, local_real_end:] = 0
    path = alignment_path(weights, encourage_early=True, allow_empty_subwords=True)
    return [(token, frame + frame_start) for token, frame in path]


def _run(
    audio: np.ndarray,
    *,
    model_path: Path,
    language: str,
    refine_frames: int,
    max_segments: int,
) -> dict[str, Any]:
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer

    model = WhisperModel(str(model_path), device="cuda", compute_type="float16")
    _verify_patched_ct2(model)
    generate_doc = inspect.getdoc(model.model.generate) or ""
    if "return_refine_trace" not in generate_doc:
        raise RuntimeError("CTranslate2 runtime does not expose 1-pass Whisper attention")
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language=language,
    )
    from whisper_timestamped.transcribe import should_use_space

    use_space = should_use_space(language)
    padded = np.pad(audio, (0, max(0, 30 * SAMPLE_RATE - len(audio))))[: 30 * SAMPLE_RATE]
    features = model.feature_extractor(padded)[:, :N_FRAMES]
    encoder_output = model.encode(features)

    started = time.perf_counter()
    generated = model.model.generate(
        encoder_output,
        [list(tokenizer.sot_sequence)],
        beam_size=1,
        num_hypotheses=1,
        max_length=model.max_length,
        return_scores=True,
        return_refine_trace=True,
        sampling_temperature=1.0,
    )[0]
    decode_sec = time.perf_counter() - started
    tokens = [int(token) for token in generated.sequences_ids[0]]
    trace_spans = split_timestamp_spans(tokens, timestamp_begin=tokenizer.timestamp_begin)
    trace = np.asarray(generated.attention[0], dtype=np.float32).reshape(
        -1,
        generated.attention_heads,
        generated.attention_frames,
    )
    real_frames = min(
        N_FRAMES // 2,
        round(len(audio) / (SAMPLE_RATE * AUDIO_TIME_PER_TOKEN)),
    )

    reports: list[dict[str, Any]] = []
    for trace_span in trace_spans[:max_segments]:
        decoded = parse_decoded_span(
            index=trace_span.index,
            text=tokenizer.decode(list(trace_span.raw_tokens)),
            raw_tokens=trace_span.raw_tokens,
            timestamp_begin=tokenizer.timestamp_begin,
            eot=tokenizer.eot,
            refine_frames=refine_frames,
        )
        if decoded is None:
            continue
        trace_path = trace_alignment_path(
            trace,
            token_start=trace_span.token_start,
            token_end=trace_span.token_end,
            frame_start=decoded.frame_start,
            frame_end=decoded.frame_end,
            real_audio_frames=real_frames,
        )
        teacher = model.model.align(
            encoder_output,
            list(tokenizer.sot_sequence),
            [list(decoded.text_tokens)],
            decoded.frame_end * 2,
            median_filter_width=9,
            frame_ranges=[(decoded.frame_start, decoded.frame_end)],
            real_audio_frames=real_frames * 2,
            prefix_tokens=decoded.start_token,
            use_boundary_queries=True,
            use_wt_attention_postprocessing=True,
            encourage_early=True,
            allow_empty_subwords=True,
        )[0]
        teacher_path = [(int(token), int(frame)) for token, frame in teacher.alignments]
        trace_words = _path_word_timings(
            path=trace_path,
            tokens=decoded.text_tokens,
            tokenizer=tokenizer,
            use_space=use_space,
            timestamp_begin=decoded.start_token,
            end_timestamp=decoded.end_token,
            time_offset=0,
        )
        teacher_words = _path_word_timings(
            path=teacher_path,
            tokens=decoded.text_tokens,
            tokenizer=tokenizer,
            use_space=use_space,
            timestamp_begin=decoded.start_token,
            end_timestamp=decoded.end_token,
            time_offset=0,
        )
        shared = min(len(trace_path), len(teacher_path))
        frame_delta = [
            abs(trace_path[index][1] - teacher_path[index][1])
            for index in range(shared)
            if trace_path[index][0] == teacher_path[index][0]
        ]
        reports.append(
            {
                "span": asdict(decoded),
                "trace_indices": [trace_span.token_start, trace_span.token_end],
                "same_path": trace_path == teacher_path,
                "trace_path": trace_path,
                "teacher_path": teacher_path,
                "max_shared_frame_delta": max(frame_delta, default=None),
                "word_comparison": compare_word_timings(trace_words, teacher_words),
            }
        )

    return {
        "contract": "wt-refine-one-pass-v1",
        "tokens": tokens,
        "attention": {
            "steps": int(trace.shape[0]),
            "heads": int(trace.shape[1]),
            "frames": int(trace.shape[2]),
            "bytes": int(trace.nbytes),
        },
        "logit_steps": len(generated.token_logprobs),
        "tail_logprob_rows": len(generated.tail_logprobs),
        "decode_sec": decode_sec,
        "reports": reports,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--fw-model", type=Path, required=True)
    parser.add_argument("--language", default="ja")
    parser.add_argument("--refine-sec", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ct2-python", type=Path)
    parser.add_argument("--ct2-bin", type=Path)
    parser.add_argument("--cuda-bin", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.ct2_python:
        sys.path.insert(0, str(args.ct2_python.resolve()))
    _add_dll_directory(args.ct2_bin)
    _add_dll_directory(args.cuda_bin)
    audio = _load_audio(args.audio)
    if len(audio) > 30 * SAMPLE_RATE:
        raise SystemExit("one_pass_probe requires audio no longer than 30 seconds")
    payload = _run(
        audio,
        model_path=args.fw_model,
        language=args.language,
        refine_frames=round(args.refine_sec / AUDIO_TIME_PER_TOKEN),
        max_segments=args.max_segments,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decode_sec": payload["decode_sec"],
                "attention": payload["attention"],
                "reports": [
                    {
                        "same_path": report["same_path"],
                        "max_shared_frame_delta": report["max_shared_frame_delta"],
                        "word_comparison": report["word_comparison"],
                    }
                    for report in payload["reports"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
