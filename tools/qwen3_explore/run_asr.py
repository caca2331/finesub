"""Run Qwen3-ASR over the segment windows of an existing pipeline `*-aligned.json`.

Reusing the baseline's own segmentation keeps the comparison honest: both systems see
byte-identical audio windows, so any text difference is the acoustic model's doing.

    python -m tools.qwen3_explore.run_asr \
        --audio out/reference/BV1kYLR6AEXv/BV1kYLR6AEXv-vocal.flac \
        --aligned out/reference/BV1kYLR6AEXv/BV1kYLR6AEXv-aligned.json \
        --out out/qwen-explore/BV1kYLR6AEXv-qwen-seg.json --language ja
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .common import cut, load_aligned, load_audio_16k

DEFAULT_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"


def build_inputs(processor, chunks, language, context):
    """`apply_transcription_request` only exposes `language`; context needs the raw path.

    The chat template concatenates every system text block, so a context block placed
    before the language name reproduces the upstream `context` system prompt.
    """
    if not context:
        return processor.apply_transcription_request(audio=chunks, language=language)

    from transformers.models.qwen3_asr.processing_qwen3_asr import resolve_language

    lang_name = resolve_language(language)
    system_text = context if lang_name is None else f"{context}\n{lang_name}"
    conversations = [
        [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [{"type": "audio", "audio": chunk}]},
        ]
        for chunk in chunks
    ]
    return processor.apply_chat_template(
        conversations, tokenize=True, add_generation_prompt=True, return_dict=True
    )


def make_windows(segments: list[dict], mode: str, duration: float) -> list[dict]:
    """Turn baseline segments into decode windows.

    `segment` keeps the 1:1 mapping used for per-segment diffing; `full` and
    `group:<sec>` trade that away for the cross-sentence context the baseline's own
    grouped Whisper decode enjoys.
    """
    if mode == "segment":
        return [{"start": s["start"], "end": s["end"], "text": s.get("text", ""), "lang": s.get("lang")} for s in segments]
    if mode == "full":
        return [{"start": 0.0, "end": duration, "text": " ".join(s.get("text", "") for s in segments), "lang": None}]
    if mode.startswith("group:"):
        span = float(mode.split(":", 1)[1])
        out: list[dict] = []
        for seg in segments:
            if out and seg["end"] - out[-1]["start"] <= span:
                out[-1]["end"] = seg["end"]
                out[-1]["text"] += seg.get("text", "")
            else:
                out.append({"start": seg["start"], "end": seg["end"], "text": seg.get("text", ""), "lang": seg.get("lang")})
        return out
    raise ValueError(f"unknown --window mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--aligned", required=True, help="baseline *-aligned.json (segment windows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", default=None, help="ja / en / ... ; omit for auto-detect")
    ap.add_argument("--context", default=None, help="hotword / vocabulary system prompt")
    ap.add_argument("--context-file", default=None)
    ap.add_argument("--pad", type=float, default=0.0, help="seconds of context around each window")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument(
        "--window",
        default="segment",
        help="segment | full | group:<seconds> — how baseline segments are batched into decode windows",
    )
    args = ap.parse_args()

    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8").strip()

    wav = load_audio_16k(args.audio)
    segments = load_aligned(args.aligned)["segments"]
    if args.limit:
        segments = segments[: args.limit]
    windows = make_windows(segments, args.window, len(wav) / 16000.0)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    results: list[dict] = []
    audio_sec = 0.0
    t0 = time.perf_counter()
    for i in range(0, len(windows), args.batch_size):
        batch = windows[i : i + args.batch_size]
        chunks = [cut(wav, s["start"], s["end"], args.pad) for s in batch]
        audio_sec += sum(len(c) for c in chunks) / 16000.0

        inputs = build_inputs(processor, chunks, args.language, context)
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        parsed = processor.batch_decode(gen, skip_special_tokens=True)
        parsed = processor.parse_output(parsed)

        for seg, item in zip(batch, parsed):
            results.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": item["transcription"],
                    "lang": item["language"],
                    "baseline_text": seg.get("text", ""),
                    "baseline_lang": seg.get("lang"),
                }
            )
        print(f"  {len(results)}/{len(windows)} windows", flush=True)

    wall = time.perf_counter() - t0
    payload = {
        "metadata": {
            "model": args.model,
            "audio": str(args.audio),
            "aligned": str(args.aligned),
            "language": args.language,
            "context": context,
            "pad_sec": args.pad,
            "batch_size": args.batch_size,
            "segments": len(results),
            "audio_seconds": round(audio_sec, 2),
            "wall_seconds": round(wall, 2),
            "rtf": round(wall / audio_sec, 4) if audio_sec else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        },
        "segments": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload["metadata"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
