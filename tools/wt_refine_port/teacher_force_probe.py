"""Compare patched CT2 alignment with WT on the same decoded tokens and audio window.

This is a development probe, not a production backend. It intentionally runs the
two model implementations sequentially so their word boundaries can be compared
without conflating decoder text divergence or requiring both models in VRAM.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
N_FRAMES = 3_000
AUDIO_TIME_PER_TOKEN = 0.02
AUDIO_SAMPLES_PER_TOKEN = 320


@dataclass(frozen=True)
class CapturedSegment:
    index: int
    start: float
    end: float
    text: str
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class AlignmentWindow:
    segment_index: int
    start: float
    end: float
    tokens: tuple[int, ...]


@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float
    end: float


def plan_window(
    segment: CapturedSegment,
    *,
    audio_duration: float,
    refine_sec: float,
) -> AlignmentWindow:
    """Freeze one segment's expanded window independently of either aligner."""

    start = max(0.0, segment.start - refine_sec)
    end = min(audio_duration, segment.end + refine_sec)
    if end <= start:
        end = min(audio_duration, start + AUDIO_TIME_PER_TOKEN)
    if end <= start:
        raise ValueError(f"segment {segment.index} has no alignable audio")
    return AlignmentWindow(segment.index, start, end, segment.tokens)


def compare_word_timings(
    candidate: Sequence[WordTiming],
    reference: Sequence[WordTiming],
) -> dict[str, Any]:
    candidate_text = [word.text.strip() for word in candidate]
    reference_text = [word.text.strip() for word in reference]
    same_text = candidate_text == reference_text
    result: dict[str, Any] = {
        "candidate_words": len(candidate),
        "reference_words": len(reference),
        "same_text": same_text,
    }
    if not same_text:
        result["candidate_text"] = candidate_text
        result["reference_text"] = reference_text
        return result

    start_delta = [abs(a.start - b.start) for a, b in zip(candidate, reference)]
    end_delta = [abs(a.end - b.end) for a, b in zip(candidate, reference)]
    for name, values in (("start", start_delta), ("end", end_delta)):
        result[f"{name}_median_sec"] = statistics.median(values) if values else 0.0
        result[f"{name}_max_sec"] = max(values, default=0.0)
        result[f"{name}_within_20ms"] = all(value <= 0.020001 for value in values)
    return result


def _load_audio(path: Path) -> np.ndarray:
    import whisper

    return whisper.load_audio(str(path)).astype(np.float32)


def _add_dll_directory(path: Path | None) -> None:
    if path and os.name == "nt" and path.is_dir():
        os.add_dll_directory(str(path))


def _verify_patched_ct2(model: Any) -> None:
    doc = inspect.getdoc(model.model.align) or ""
    required = (
        "use_timestamp_prefix",
        "use_boundary_queries",
        "use_wt_attention_postprocessing",
        "real_audio_frames",
        "prefix_tokens",
        "encourage_early",
        "allow_empty_subwords",
    )
    missing = [name for name in required if name not in doc]
    if missing:
        raise RuntimeError(
            "CTranslate2 runtime is not the WT-refine build; missing align arguments: "
            + ", ".join(missing)
        )


def _decode_and_ct2_align(
    audio: np.ndarray,
    *,
    model_path: Path,
    language: str,
    refine_sec: float,
    max_segments: int,
) -> tuple[list[CapturedSegment], list[AlignmentWindow], list[dict[str, Any]], float]:
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer

    started = time.perf_counter()
    model = WhisperModel(str(model_path), device="cuda", compute_type="float16")
    _verify_patched_ct2(model)
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language=language,
    )
    decoded = list(
        model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
        )[0]
    )
    captured: list[CapturedSegment] = []
    for index, segment in enumerate(decoded):
        tokens = tuple(token for token in (segment.tokens or ()) if token < tokenizer.eot)
        if tokens:
            captured.append(
                CapturedSegment(
                    index=index,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=str(segment.text),
                    tokens=tokens,
                )
            )
        if len(captured) >= max_segments:
            break

    audio_duration = len(audio) / SAMPLE_RATE
    windows = [
        plan_window(segment, audio_duration=audio_duration, refine_sec=refine_sec)
        for segment in captured
    ]
    raw_alignments: list[dict[str, Any]] = []
    for window in windows:
        begin = round(window.start * SAMPLE_RATE)
        finish = round(window.end * SAMPLE_RATE)
        clip = audio[begin:finish]
        padded = np.pad(clip, (0, max(0, 30 * SAMPLE_RATE - len(clip))))[: 30 * SAMPLE_RATE]
        real_encoder_frames = min(N_FRAMES // 2, round(len(clip) / AUDIO_SAMPLES_PER_TOKEN))
        alignment_encoder_frames = min(
            N_FRAMES // 2,
            real_encoder_frames + round(refine_sec / AUDIO_TIME_PER_TOKEN),
        )
        num_frames = alignment_encoder_frames * 2
        real_audio_frames = real_encoder_frames * 2
        features = model.feature_extractor(padded)[:, :N_FRAMES]
        encoder_output = model.encode(features)
        result = model.model.align(
            encoder_output,
            list(tokenizer.sot_sequence),
            [list(window.tokens)],
            num_frames,
            median_filter_width=9,
            real_audio_frames=real_audio_frames,
            use_timestamp_prefix=True,
            use_boundary_queries=True,
            use_wt_attention_postprocessing=True,
            encourage_early=True,
            allow_empty_subwords=True,
        )[0]
        raw_alignments.append(
            {
                "segment_index": window.segment_index,
                "path": [[int(token), int(frame)] for token, frame in result.alignments],
                "text_token_probs": [float(value) for value in result.text_token_probs],
            }
        )

    elapsed = time.perf_counter() - started
    del tokenizer, decoded, model
    gc.collect()
    return captured, windows, raw_alignments, elapsed


def _path_word_timings(
    *,
    path: Sequence[Sequence[int]],
    tokens: Sequence[int],
    tokenizer: Any,
    use_space: bool,
    timestamp_begin: int,
    end_timestamp: int,
    time_offset: float,
) -> list[WordTiming]:
    from whisper_timestamped.transcribe import (
        _punctuation,
        split_tokens_on_spaces,
        split_tokens_on_unicode,
    )

    if not path:
        return []

    full_tokens = [timestamp_begin, *tokens, end_timestamp]
    splitter = split_tokens_on_spaces if use_space else split_tokens_on_unicode
    words, word_tokens, _ = splitter(
        full_tokens,
        tokenizer,
        remove_punctuation_from_words=False,
    )
    text_indices = np.asarray([pair[0] for pair in path], dtype=np.int64)
    time_indices = np.asarray([pair[1] for pair in path], dtype=np.int64)
    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps]
    jump_times = np.pad(jump_times, (0, 1), constant_values=time_indices[-1])
    boundaries = np.pad(np.cumsum([len(group) for group in word_tokens]), (1, 0))
    punctuation_counts = np.asarray(
        [0 if len(group) == 1 or group[-1] not in _punctuation else 1 for group in word_tokens],
        dtype=np.int64,
    )
    begin = jump_times[boundaries[:-1]] * AUDIO_TIME_PER_TOKEN
    end = jump_times[boundaries[1:] - punctuation_counts] * AUDIO_TIME_PER_TOKEN
    return [
        WordTiming(str(word), float(start + time_offset), float(stop + time_offset))
        for word, start, stop in zip(words[1:-1], begin[1:-1], end[1:-1])
        if not str(word).startswith("<|")
    ]


def _wt_align_and_compare(
    audio: np.ndarray,
    *,
    model_name: str,
    language: str,
    refine_sec: float,
    windows: Sequence[AlignmentWindow],
    ct2_alignments: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    import torch
    import whisper
    wt_transcribe = importlib.import_module("whisper_timestamped.transcribe")
    from whisper_timestamped.transcribe import (
        AUDIO_TIME_PER_TOKEN as WT_AUDIO_TIME_PER_TOKEN,
        N_FRAMES as WT_N_FRAMES,
        get_alignment_heads,
        get_tokenizer,
        perform_word_alignment,
        should_use_space,
    )

    try:
        from whisper.model import disable_sdpa
    except ImportError:
        from contextlib import nullcontext as disable_sdpa

    assert WT_AUDIO_TIME_PER_TOKEN == AUDIO_TIME_PER_TOKEN
    assert WT_N_FRAMES == N_FRAMES
    started = time.perf_counter()
    model = whisper.load_model(model_name, device="cuda")
    tokenizer = get_tokenizer(model, task="transcribe", language=language)
    use_space = should_use_space(language)
    alignment_heads = get_alignment_heads(model)
    perform_word_alignment.__globals__["num_alignment_for_plot"] = 0

    attention_weights: list[Any] = [None] * len(model.decoder.blocks)
    hooks = []
    for index, block in enumerate(model.decoder.blocks):
        def capture(_layer: Any, _inputs: Any, outputs: Any, *, slot: int = index) -> None:
            attention_weights[slot] = outputs[1]

        hooks.append(block.cross_attn.register_forward_hook(capture))

    reports: list[dict[str, Any]] = []
    try:
        for window, ct2_alignment in zip(windows, ct2_alignments):
            begin = round(window.start * SAMPLE_RATE)
            finish = round(window.end * SAMPLE_RATE)
            clip = audio[begin:finish]
            mfcc = whisper.log_mel_spectrogram(clip, n_mels=model.dims.n_mels)
            mfcc = whisper.pad_or_trim(mfcc, N_FRAMES).to(model.device).unsqueeze(0)
            sot_sequence = tokenizer.sot_sequence
            prompt = [*sot_sequence, tokenizer.timestamp_begin, *window.tokens]
            with torch.no_grad(), disable_sdpa():
                model(
                    mfcc,
                    torch.tensor(prompt, dtype=torch.long, device=model.device).unsqueeze(0),
                )

            end_timestamp = tokenizer.timestamp_begin + min(
                N_FRAMES // 2,
                round(len(clip) / AUDIO_SAMPLES_PER_TOKEN),
            )
            aligned_tokens = [tokenizer.timestamp_begin, *window.tokens, end_timestamp]
            row_start = len(sot_sequence) - 1
            selected_attention = [weights[:, :, row_start:, :] for weights in attention_weights]
            captured_path: list[list[int]] = []
            original_dtw = wt_transcribe.dtw.dtw

            def capture_dtw(*dtw_args: Any, **dtw_kwargs: Any) -> Any:
                alignment = original_dtw(*dtw_args, **dtw_kwargs)
                captured_path.extend(
                    [int(token), int(frame)]
                    for token, frame in zip(alignment.index1s, alignment.index2s)
                )
                return alignment

            wt_transcribe.dtw.dtw = capture_dtw
            try:
                wt_words = perform_word_alignment(
                    aligned_tokens,
                    selected_attention,
                    tokenizer,
                    use_space=use_space,
                    alignment_heads=alignment_heads,
                    remove_punctuation_from_words=False,
                    refine_whisper_precision_nframes=round(refine_sec / AUDIO_TIME_PER_TOKEN),
                    detect_disfluencies=False,
                    medfilt_width=9,
                    mfcc=mfcc,
                    plot=False,
                    subwords_can_be_empty=True,
                )
            finally:
                wt_transcribe.dtw.dtw = original_dtw
            reference = [
                WordTiming(
                    str(word.get("text") or word.get("word") or ""),
                    float(word["start"]) + window.start,
                    float(word["end"]) + window.start,
                )
                for word in wt_words
            ]
            candidate = _path_word_timings(
                path=ct2_alignment["path"],
                tokens=window.tokens,
                tokenizer=tokenizer,
                use_space=use_space,
                timestamp_begin=tokenizer.timestamp_begin,
                end_timestamp=end_timestamp,
                time_offset=window.start,
            )
            reports.append(
                {
                    "segment_index": window.segment_index,
                    "ct2_path": ct2_alignment["path"],
                    "wt_path": captured_path,
                    "same_path": ct2_alignment["path"] == captured_path,
                    "ct2_words": [asdict(word) for word in candidate],
                    "wt_words": [asdict(word) for word in reference],
                    "comparison": compare_word_timings(candidate, reference),
                }
            )
    finally:
        for hook in hooks:
            hook.remove()
        del tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

    return reports, time.perf_counter() - started


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--fw-model", type=Path, required=True)
    parser.add_argument("--ow-model", default="large-v3-turbo")
    parser.add_argument("--language", default="ja")
    parser.add_argument("--refine-sec", type=float, default=1.0)
    parser.add_argument("--max-segments", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ct2-python", type=Path)
    parser.add_argument("--ct2-bin", type=Path)
    parser.add_argument("--cuda-bin", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_segments < 1:
        raise SystemExit("--max-segments must be positive")
    if args.ct2_python:
        sys.path.insert(0, str(args.ct2_python.resolve()))
    _add_dll_directory(args.ct2_bin)
    _add_dll_directory(args.cuda_bin)

    audio = _load_audio(args.audio)
    captured, windows, ct2_alignments, ct2_elapsed = _decode_and_ct2_align(
        audio,
        model_path=args.fw_model,
        language=args.language,
        refine_sec=args.refine_sec,
        max_segments=args.max_segments,
    )
    reports, wt_elapsed = _wt_align_and_compare(
        audio,
        model_name=args.ow_model,
        language=args.language,
        refine_sec=args.refine_sec,
        windows=windows,
        ct2_alignments=ct2_alignments,
    )
    payload = {
        "contract": "wt-refine-teacher-force-v1",
        "audio": str(args.audio.resolve()),
        "language": args.language,
        "refine_sec": args.refine_sec,
        "captured_segments": [asdict(segment) for segment in captured],
        "windows": [asdict(window) for window in windows],
        "ct2_alignments": ct2_alignments,
        "reports": reports,
        "timing_sec": {"fw_decode_and_ct2_align": ct2_elapsed, "wt_align": wt_elapsed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reports": reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
