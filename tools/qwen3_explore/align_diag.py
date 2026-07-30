"""Attribute the aligner's zero-duration words: bad text, or the aligner + its repair pass?

Runs the same audio through four text conditions and reports the raw (pre-repair) and final
zero-duration rates for each. `agree` is text both ASR systems independently produced, so it is
as close to known-correct as this corpus gets; `shuffled` and `foreign` are deliberately wrong
text, giving the "hallucinated transcript" regime a measured reference point instead of a guess.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoProcessor
from transformers.models.qwen3_asr.processing_qwen3_asr import _fix_timestamps

from .common import cut, load_aligned, load_audio_16k, normalize_ja

MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"


def run_condition(model, proc, wav, items, language, dump=None):
    """items: list of (segment, text). Returns per-word raw/fixed timestamp stats."""
    raw_zero = fixed_zero = made_zero = total = 0
    nonmono = 0
    traces = []
    dumped: list[dict] = []
    for i in range(0, len(items), 4):
        batch = items[i : i + 4]
        chunks = [cut(wav, s["start"], s["end"]) for s, _ in batch]
        inputs, word_lists = proc.prepare_forced_aligner_inputs(
            audio=chunks, transcript=[t for _, t in batch], language=language
        )
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            logits = model(**inputs).logits
        pred = logits.argmax(-1)
        for b, wl in enumerate(word_lists):
            mask = inputs["input_ids"][b] == model.config.timestamp_token_id
            raw = (pred[b][mask].float() * proc.timestamp_segment_time).cpu().numpy()
            fixed = _fix_timestamps(raw)
            nonmono += sum(1 for x, y in zip(raw, raw[1:]) if y < x)
            for k in range(len(wl)):
                total += 1
                r0, r1, f0, f1 = raw[2 * k], raw[2 * k + 1], fixed[2 * k], fixed[2 * k + 1]
                raw_zero += r0 == r1
                fixed_zero += f0 == f1
                made_zero += (r0 != r1) and (f0 == f1)
            if len(traces) < 2:
                traces.append((batch[b][1], wl, [float(x) for x in raw], fixed))
            if dump is not None:
                seg = batch[b][0]
                dumped.append(
                    {
                        "index": len(dumped),
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": batch[b][1],
                        "words": [
                            {
                                "word": w,
                                "start": round(seg["start"] + fixed[2 * k] / 1000.0, 3),
                                "end": round(seg["start"] + fixed[2 * k + 1] / 1000.0, 3),
                            }
                            for k, w in enumerate(wl)
                        ],
                        "baseline_words": [],
                    }
                )
    if dump is not None:
        Path(dump).parent.mkdir(parents=True, exist_ok=True)
        Path(dump).write_text(
            json.dumps({"metadata": {"condition": Path(dump).stem}, "segments": dumped}, ensure_ascii=False),
            encoding="utf-8",
        )
    return {
        "words": total,
        "raw_zero_pct": round(100 * raw_zero / max(1, total), 1),
        "final_zero_pct": round(100 * fixed_zero / max(1, total), 1),
        "created_by_repair_pct": round(100 * made_zero / max(1, total), 1),
        "nonmono_per_100_words": round(100 * nonmono / max(1, total), 1),
    }, traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--aligned", required=True)
    ap.add_argument("--arm", required=True, help="run_asr.py segment-mode output, for the agree/disagree split")
    ap.add_argument("--language", default="ja")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace", action="store_true", help="print a raw-vs-repaired timestamp trace")
    ap.add_argument("--dump-dir", default=None, help="write per-condition alignments for collapse_scan.py")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wav = load_audio_16k(args.audio)
    base = load_aligned(args.aligned)["segments"]
    arm = json.loads(Path(args.arm).read_text(encoding="utf-8"))["segments"]

    agree, disagree = [], []
    for seg, other in zip(base, arm):
        text = seg.get("text", "")
        if not normalize_ja(text):
            continue
        (agree if normalize_ja(text) == normalize_ja(other["text"]) else disagree).append((seg, text))

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0"
    ).eval()

    def shuffled(pairs):
        out = []
        for seg, text in pairs:
            words = proc.split_words_for_alignment(text, args.language)
            rng.shuffle(words)
            out.append((seg, "".join(words)))
        return out

    def foreign(pairs):
        rotated = pairs[len(pairs) // 2 :] + pairs[: len(pairs) // 2]
        return [(seg, other_text) for (seg, _), (_, other_text) in zip(pairs, rotated)]

    conditions = {
        "agree (两套 ASR 独立一致，文本基本可信)": agree,
        "disagree (两套 ASR 不一致，文本可疑)": disagree,
        "shuffled (词序打乱，词还是对的)": shuffled(agree),
        "foreign (换成别段的文本，完全不对)": foreign(agree),
    }

    print(f"{'condition':46s} {'words':>6s} {'raw0%':>6s} {'final0%':>8s} {'+repair%':>9s} {'nonmono/100w':>13s}")
    for name, items in conditions.items():
        if not items:
            continue
        dump = f"{args.dump_dir}/{name.split(' ')[0]}.json" if args.dump_dir else None
        stats, traces = run_condition(model, proc, wav, items, args.language, dump)
        print(
            f"{name:46s} {stats['words']:6d} {stats['raw_zero_pct']:6.1f} {stats['final_zero_pct']:8.1f} "
            f"{stats['created_by_repair_pct']:9.1f} {stats['nonmono_per_100_words']:13.1f}"
        )
        if args.trace and name.startswith("agree"):
            text, wl, raw, fixed = traces[0]
            print(f"\n  trace: {text}")
            print(f"  {'word':10s} {'raw start/end (ms)':>22s} {'repaired':>18s}")
            for k, w in enumerate(wl):
                print(
                    f"  {w:10s} {raw[2 * k]:10.0f}{raw[2 * k + 1]:11.0f}   "
                    f"{fixed[2 * k]:8.0f}{fixed[2 * k + 1]:9.0f}"
                    + ("   <- collapsed" if fixed[2 * k] == fixed[2 * k + 1] else "")
                )
            print()


if __name__ == "__main__":
    main()
