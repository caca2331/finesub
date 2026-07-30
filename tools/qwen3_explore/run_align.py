"""Run Qwen3-ForcedAligner over the same windows the baseline aligned, and diff timings.

The transcript can come from the baseline itself (`--transcript baseline`), which isolates
the aligner: identical audio, identical text, only the timestamp mechanism differs
(Whisper cross-attention DTW vs. Qwen's non-autoregressive timestamp head).

    python -m tools.qwen3_explore.run_align --audio ...-vocal.flac --aligned ...-aligned.json \
        --out out/qwen-explore/x-align.json --language ja
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

from .common import cut, load_aligned, load_audio_16k, normalize_ja

DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


def char_time_map(words: list[dict], key_start: str, key_end: str, key_text: str) -> list[tuple[str, float, float]]:
    """Explode word timings to one entry per kept character, so two tokenizers can be diffed."""
    out: list[tuple[str, float, float]] = []
    for w in words:
        text = normalize_ja(w[key_text])
        if not text:
            continue
        span = (w[key_end] - w[key_start]) / len(text)
        for i, ch in enumerate(text):
            out.append((ch, w[key_start] + i * span, w[key_start] + (i + 1) * span))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--language", default="ja")
    ap.add_argument(
        "--transcript",
        default="baseline",
        help="'baseline' (text from --aligned) or a run_asr.py arm JSON to take text from",
    )
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    wav = load_audio_16k(args.audio)
    base_segments = load_aligned(args.aligned)["segments"]
    if args.limit:
        base_segments = base_segments[: args.limit]

    if args.transcript == "baseline":
        texts = [s.get("text", "") for s in base_segments]
    else:
        arm = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        arm_segments = arm["segments"]
        if len(arm_segments) == len(base_segments):
            texts = [s["text"] for s in arm_segments]
        else:
            # A long-window arm (e.g. `--window full`): align its own windows instead.
            base_segments = [
                {"start": s["start"], "end": s["end"], "text": s["text"], "words": []}
                for s in arm_segments
            ]
            texts = [s["text"] for s in arm_segments]

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    todo = [(i, s, t) for i, (s, t) in enumerate(zip(base_segments, texts)) if normalize_ja(t)]
    results: list[dict] = []
    audio_sec = 0.0
    t0 = time.perf_counter()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i : i + args.batch_size]
        chunks = [cut(wav, s["start"], s["end"], args.pad) for _, s, _ in batch]
        audio_sec += sum(len(c) for c in chunks) / 16000.0

        inputs, word_lists = processor.prepare_forced_aligner_inputs(
            audio=chunks, transcript=[t for _, _, t in batch], language=args.language
        )
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            outputs = model(**inputs)
        decoded = processor.decode_forced_alignment(
            logits=outputs.logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )

        for (idx, seg, text), words in zip(batch, decoded):
            offset = seg["start"] - args.pad
            results.append(
                {
                    "index": idx,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": text,
                    "words": [
                        {
                            "word": w["text"],
                            "start": round(offset + w["start_time"], 3),
                            "end": round(offset + w["end_time"], 3),
                        }
                        for w in words
                    ],
                    "baseline_text": seg.get("text", ""),
                    "baseline_words": seg.get("words", []),
                }
            )
        print(f"  {len(results)}/{len(todo)} windows", flush=True)

    wall = time.perf_counter() - t0
    payload = {
        "metadata": {
            "model": args.model,
            "audio": str(args.audio),
            "transcript": args.transcript,
            "language": args.language,
            "pad_sec": args.pad,
            "windows": len(results),
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
