"""Boundary spot-check: does the audio inside a claimed word span actually contain those words?

Both systems carry the *same* text, so cutting each system's span and re-transcribing it is a
pure test of the boundaries. Step 1 (this file, `qwen-asr` env) picks the spans, dumps one WAV
per span per system, and re-transcribes with Qwen3-ASR. Step 2 (`verify_whisper.py`, `asr` env)
re-transcribes the identical WAVs with Whisper, so the verdict does not rest on a verifier that
shares an encoder family with the aligner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

from .common import TARGET_SR, cer, load_audio_16k, normalize_ja
from .run_align import char_time_map


def pick_spans(align_json: dict, min_sec: float, min_chars: int, stride: int) -> list[dict]:
    """Consecutive word runs long enough to be transcribable, taken every `stride`-th segment."""
    spans: list[dict] = []
    for seg in align_json["segments"][::stride]:
        qmap = char_time_map(seg["words"], "start", "end", "word")
        bmap = char_time_map(seg["baseline_words"], "start", "end", "word")
        if not qmap or not bmap:
            continue
        text_q = "".join(c for c, _, _ in qmap)
        if text_q != "".join(c for c, _, _ in bmap):
            continue
        n = len(qmap)
        # middle slice of the segment: boundaries there are not propped up by the VAD cut
        lo, hi = n // 4, max(n // 4 + 1, (3 * n) // 4)
        while hi < n and (qmap[hi - 1][2] - qmap[lo][1] < min_sec or hi - lo < min_chars):
            hi += 1
        if hi - lo < min_chars or qmap[hi - 1][2] - qmap[lo][1] < min_sec:
            continue
        spans.append(
            {
                "index": seg["index"],
                "text": text_q[lo:hi],
                "qwen": [qmap[lo][1], qmap[hi - 1][2]],
                "baseline": [bmap[lo][1], bmap[hi - 1][2]],
            }
        )
    return spans


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--align", required=True, help="run_align.py output (carries both timings)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--min-sec", type=float, default=1.0)
    ap.add_argument("--min-chars", type=int, default=5)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "wav").mkdir(parents=True, exist_ok=True)

    align_json = json.loads(Path(args.align).read_text(encoding="utf-8"))
    wav = load_audio_16k(args.audio)
    spans = pick_spans(align_json, args.min_sec, args.min_chars, args.stride)[: args.limit]

    clips: list[dict] = []
    for i, span in enumerate(spans):
        for system in ("qwen", "baseline"):
            a, b = span[system]
            path = outdir / "wav" / f"{i:03d}-{system}.wav"
            sf.write(path, wav[int(a * TARGET_SR) : int(b * TARGET_SR)], TARGET_SR)
            clips.append(
                {
                    "id": i,
                    "system": system,
                    "wav": str(path),
                    "text": span["text"],
                    "start": a,
                    "end": b,
                    "dur": round(b - a, 3),
                }
            )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-1.7B-hf")
    model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B-hf", dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    for i in range(0, len(clips), 8):
        batch = clips[i : i + 8]
        audio = [np.asarray(sf.read(c["wav"])[0], dtype=np.float32) for c in batch]
        inputs = processor.apply_transcription_request(audio=audio, language=args.language)
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1] :]
        for clip, parsed in zip(batch, processor.parse_output(processor.batch_decode(gen, skip_special_tokens=True))):
            clip["qwen_asr"] = parsed["transcription"]
            clip["qwen_asr_cer"] = round(cer(clip["text"], parsed["transcription"]), 3)

    (outdir / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    summarize(clips, "qwen_asr_cer")


def summarize(clips: list[dict], key: str) -> None:
    import statistics

    for system in ("qwen", "baseline"):
        vals = [c[key] for c in clips if c["system"] == system and key in c]
        if not vals:
            continue
        print(
            f"  {system:9s} n={len(vals)} mean CER={statistics.mean(vals):.3f} "
            f"median={statistics.median(vals):.3f} "
            f"exact={sum(1 for v in vals if v == 0)}/{len(vals)}"
        )


if __name__ == "__main__":
    main()
