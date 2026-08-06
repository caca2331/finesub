"""Compare CT2 and WT alignment on full-window decoded timestamp spans.

Unlike teacher_force_probe, this probe keeps the decoder's real start/end
timestamp tokens, encodes the audio window once per backend, and aligns several
segments against that shared encoder output. It is the bridge from isolated
alignment parity to the complete WT refine state machine.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.wt_refine_port.teacher_force_probe import (
    AUDIO_TIME_PER_TOKEN,
    N_FRAMES,
    SAMPLE_RATE,
    WordTiming,
    _add_dll_directory,
    _load_audio,
    _path_word_timings,
    _verify_patched_ct2,
    compare_word_timings,
)


@dataclass(frozen=True)
class DecodedSpan:
    index: int
    text: str
    raw_tokens: tuple[int, ...]
    text_tokens: tuple[int, ...]
    start_token: int
    end_token: int
    frame_start: int
    frame_end: int


def parse_decoded_span(
    *,
    index: int,
    text: str,
    raw_tokens: Sequence[int],
    timestamp_begin: int,
    eot: int,
    refine_frames: int,
) -> DecodedSpan | None:
    """Parse a normal FW timestamp-delimited segment without repairing it."""

    raw = tuple(int(token) for token in raw_tokens)
    if len(raw) < 3 or raw[0] < timestamp_begin or raw[-1] < timestamp_begin:
        return None
    content = tuple(token for token in raw[1:-1] if token < eot)
    if not content:
        return None
    start_frame = raw[0] - timestamp_begin
    end_frame = raw[-1] - timestamp_begin
    end_frame = min(N_FRAMES // 2, max(end_frame, start_frame + len(raw)))
    frame_start = max(0, start_frame - refine_frames)
    frame_end = min(N_FRAMES // 2, end_frame + refine_frames)
    if frame_end <= frame_start:
        return None
    return DecodedSpan(
        index=index,
        text=text,
        raw_tokens=raw,
        text_tokens=content,
        start_token=raw[0],
        end_token=raw[-1],
        frame_start=frame_start,
        frame_end=frame_end,
    )


def _decode_and_ct2_align(
    audio: np.ndarray,
    *,
    model_path: Path,
    language: str,
    refine_frames: int,
    max_segments: int,
) -> tuple[list[DecodedSpan], list[dict[str, Any]], list[dict[str, Any]], float]:
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
    spans: list[DecodedSpan] = []
    rejected: list[dict[str, Any]] = []
    for index, segment in enumerate(decoded):
        span = parse_decoded_span(
            index=index,
            text=str(segment.text),
            raw_tokens=segment.tokens or (),
            timestamp_begin=tokenizer.timestamp_begin,
            eot=tokenizer.eot,
            refine_frames=refine_frames,
        )
        if span is None:
            rejected.append(
                {
                    "index": index,
                    "text": str(segment.text),
                    "raw_tokens": [int(token) for token in (segment.tokens or ())],
                    "reason": "not_timestamp_delimited",
                }
            )
            continue
        spans.append(span)
        if len(spans) >= max_segments:
            break

    padded = np.pad(audio, (0, max(0, 30 * SAMPLE_RATE - len(audio))))[: 30 * SAMPLE_RATE]
    features = model.feature_extractor(padded)[:, :N_FRAMES]
    encoder_output = model.encode(features)
    real_encoder_frames = min(N_FRAMES // 2, round(len(audio) / (SAMPLE_RATE * AUDIO_TIME_PER_TOKEN)))
    alignments: list[dict[str, Any]] = []
    for span in spans:
        result = model.model.align(
            encoder_output,
            list(tokenizer.sot_sequence),
            [list(span.text_tokens)],
            span.frame_end * 2,
            median_filter_width=9,
            frame_ranges=[(span.frame_start, span.frame_end)],
            real_audio_frames=real_encoder_frames * 2,
            prefix_tokens=span.start_token,
            use_boundary_queries=True,
            use_wt_attention_postprocessing=True,
            encourage_early=True,
            allow_empty_subwords=True,
        )[0]
        alignments.append(
            {
                "segment_index": span.index,
                "path": [[int(token), int(frame)] for token, frame in result.alignments],
                "text_token_probs": [float(value) for value in result.text_token_probs],
            }
        )

    elapsed = time.perf_counter() - started
    del encoder_output, tokenizer, decoded, model
    gc.collect()
    return spans, alignments, rejected, elapsed


def _wt_align(
    audio: np.ndarray,
    *,
    model_name: str,
    language: str,
    refine_frames: int,
    spans: Sequence[DecodedSpan],
    ct2_alignments: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    import torch
    import whisper
    from whisper_timestamped.transcribe import (
        get_alignment_heads,
        get_tokenizer,
        perform_word_alignment,
        should_use_space,
    )

    wt_transcribe = importlib.import_module("whisper_timestamped.transcribe")
    try:
        from whisper.model import disable_sdpa
    except ImportError:
        from contextlib import nullcontext as disable_sdpa

    started = time.perf_counter()
    model = whisper.load_model(model_name, device="cuda")
    tokenizer = get_tokenizer(model, task="transcribe", language=language)
    use_space = should_use_space(language)
    alignment_heads = get_alignment_heads(model)
    wt_transcribe.num_alignment_for_plot = 0
    mfcc = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
    mfcc = whisper.pad_or_trim(mfcc, N_FRAMES).to(model.device).unsqueeze(0)

    attention_weights: list[Any] = [None] * len(model.decoder.blocks)
    hooks = []
    for index, block in enumerate(model.decoder.blocks):
        def capture(_layer: Any, _inputs: Any, outputs: Any, *, slot: int = index) -> None:
            attention_weights[slot] = outputs[1]

        hooks.append(block.cross_attn.register_forward_hook(capture))

    reports: list[dict[str, Any]] = []
    try:
        for span, ct2_alignment in zip(spans, ct2_alignments):
            prompt = [*tokenizer.sot_sequence, span.start_token, *span.text_tokens]
            with torch.no_grad(), disable_sdpa():
                model(
                    mfcc,
                    torch.tensor(prompt, dtype=torch.long, device=model.device).unsqueeze(0),
                )
            row_start = len(tokenizer.sot_sequence) - 1
            selected_attention = [weights[:, :, row_start:, :] for weights in attention_weights]
            captured_path: list[list[int]] = []
            original_dtw = wt_transcribe.dtw.dtw

            def capture_dtw(*dtw_args: Any, **dtw_kwargs: Any) -> Any:
                alignment = original_dtw(*dtw_args, **dtw_kwargs)
                captured_path.extend(
                    [int(token), int(frame) + span.frame_start]
                    for token, frame in zip(alignment.index1s, alignment.index2s)
                )
                return alignment

            wt_transcribe.dtw.dtw = capture_dtw
            try:
                words = perform_word_alignment(
                    list(span.raw_tokens),
                    selected_attention,
                    tokenizer,
                    use_space=use_space,
                    alignment_heads=alignment_heads,
                    remove_punctuation_from_words=False,
                    refine_whisper_precision_nframes=refine_frames,
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
                    float(word["start"]),
                    float(word["end"]),
                )
                for word in words
            ]
            candidate = _path_word_timings(
                path=ct2_alignment["path"],
                tokens=span.text_tokens,
                tokenizer=tokenizer,
                use_space=use_space,
                timestamp_begin=span.start_token,
                end_timestamp=span.end_token,
                time_offset=0,
            )
            reports.append(
                {
                    "segment_index": span.index,
                    "same_path": ct2_alignment["path"] == captured_path,
                    "ct2_path": ct2_alignment["path"],
                    "wt_path": captured_path,
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
    parser.add_argument("--max-segments", type=int, default=4)
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
    if len(audio) > 30 * SAMPLE_RATE:
        raise SystemExit("full_window_probe requires audio no longer than 30 seconds")
    refine_frames = round(args.refine_sec / AUDIO_TIME_PER_TOKEN)
    spans, ct2_alignments, rejected, ct2_elapsed = _decode_and_ct2_align(
        audio,
        model_path=args.fw_model,
        language=args.language,
        refine_frames=refine_frames,
        max_segments=args.max_segments,
    )
    reports, wt_elapsed = _wt_align(
        audio,
        model_name=args.ow_model,
        language=args.language,
        refine_frames=refine_frames,
        spans=spans,
        ct2_alignments=ct2_alignments,
    )
    payload = {
        "contract": "wt-refine-full-window-v1",
        "audio": str(args.audio.resolve()),
        "language": args.language,
        "refine_sec": args.refine_sec,
        "spans": [asdict(span) for span in spans],
        "rejected": rejected,
        "ct2_alignments": ct2_alignments,
        "reports": reports,
        "timing_sec": {"fw_decode_and_ct2_align": ct2_elapsed, "wt_align": wt_elapsed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [report["comparison"] for report in reports]
    print(json.dumps({"output": str(args.output), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
