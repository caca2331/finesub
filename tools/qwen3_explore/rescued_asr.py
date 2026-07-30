"""Qwen ASR + aligner wrapped in the same rescue engineering the baseline gets.

`asr_align.py` does not hand raw Whisper output to the pipeline: it groups VAD intervals into
~30 s windows, checks that the decode actually covered the window's speech, and on failure peels
the first interval into its own window and re-decodes, converging to per-interval. That ladder is
what makes the baseline's recall look good, and none of it is Whisper-specific — so this arm
gives Qwen the same treatment before any quality comparison.

Coverage is measured differently here, and more directly: Qwen's ASR returns no timestamps, so
the forced aligner supplies them, and the aligner's own failure signatures (off-grid
interpolation blocks, absurd per-character durations) become extra rescue triggers the baseline
has no equivalent for.

Runs in the `qwen-asr` env. Output is an aligned-JSON-shaped file; run `apply_split.py` in the
production `asr` env afterwards to put it through the real `segment_split`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoModelForTokenClassification, AutoProcessor

from .common import TARGET_SR, load_audio_16k, normalize_ja

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B-hf"

# Mirrors asr_align.py's grouping and coverage rule.
GROUP_TARGET_SEC = 30.0
MIN_GROUP_LENGTH = 15.0
COVERAGE_MIN_RATIO = 0.6
COVERAGE_TOLERANCE_SEC = 2.0
SMALL_BATCH_EXEMPT_SEC = 3.3
COVERAGE_MIN_INTERVAL_SEC = 1.0   # intervals shorter than this are interjection territory

# Collapse triggers (FINDINGS.md §4.3). Only metrics that measure actual harm are triggers:
# an off-grid timestamp merely says `_fix_timestamps` interpolated there, which happens for any
# benign two-word swap as well — using it as a trigger fired on 8/8 windows and drove RTF to 3.4.
# It is kept as an annotation instead.
MAX_SEC_PER_CHAR = 3.0        # バカバ at 3.6 s/char is a real collapse; a drawn-out mora at 2.2 is not
MAX_CHAR_DENSITY = 25.0       # measured: 46 in the real collapse, <=20 everywhere else
GRID_MS = 80.0

# Japanese speech tops out well under this; the margin covers dense speech plus the language
# prefix, while still cutting a runaway loop short.
TOKENS_PER_SEC = 18


def group_intervals(intervals, target=GROUP_TARGET_SEC, min_len=MIN_GROUP_LENGTH):
    groups, current = [], []
    for iv in intervals:
        current.append(iv)
        span = current[-1][1] - current[0][0]
        if span >= target:
            groups.append(current)
            current = []
    if current:
        if groups and current[-1][1] - current[0][0] < min_len:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def coverage_ok(words, intervals):
    """Coverage over *substantial* intervals only.

    The check exists to catch silent truncation of real speech. Qwen legitimately returns
    nothing for sub-second interjection intervals (ヤホッ, ん?, あー — 20+% of intervals on
    reaction-heavy clips), and counting those in the denominator made coverage fail
    structurally: 12 of 18 windows on BV1cJjE6cEt8, driving the ladder to per-interval decode
    every time. Short intervals still get transcribed when the model has something to say;
    they just no longer make a window look truncated.
    """
    scored = [(a, b) for a, b in intervals if b - a >= COVERAGE_MIN_INTERVAL_SEC]
    speech = sum(b - a for a, b in scored)
    if speech < SMALL_BATCH_EXEMPT_SEC:
        return True, speech, speech
    covered = 0.0
    for a, b in scored:
        covered += sum(overlap(a, b, w["start"], w["end"]) for w in words)
    return covered >= COVERAGE_MIN_RATIO * speech - COVERAGE_TOLERANCE_SEC, covered, speech


def collapse_ok(words):
    """Reject the alignment failure modes of FINDINGS.md §4.3 before they reach the pipeline."""
    if not words:
        return False, "empty"
    for w in words:
        chars = max(1, len(normalize_ja(w["word"])))
        if (w["end"] - w["start"]) / chars > MAX_SEC_PER_CHAR:
            return False, f"word {w['word']!r} spans {w['end'] - w['start']:.1f}s"
    for i, w in enumerate(words):
        j = i
        chars = 0
        while j < len(words) and words[j]["start"] < w["start"] + 1.0:
            chars += len(normalize_ja(words[j]["word"]))
            j += 1
        if chars > MAX_CHAR_DENSITY:
            return False, f"{chars} chars in one second at {w['start']:.1f}s"
    return True, ""


class Engine:
    def __init__(self, language):
        self.language = language
        self.asr_proc = AutoProcessor.from_pretrained(ASR_MODEL)
        self.asr = AutoModelForMultimodalLM.from_pretrained(
            ASR_MODEL, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        self.aln_proc = AutoProcessor.from_pretrained(ALIGNER_MODEL)
        self.aln = AutoModelForTokenClassification.from_pretrained(
            ALIGNER_MODEL, dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()

    def transcribe(self, chunk, duration=None):
        inputs = self.asr_proc.apply_transcription_request(audio=[chunk], language=self.language)
        inputs = inputs.to(self.asr.device, self.asr.dtype)
        # Budget the generation to what the audio can plausibly contain. A flat 1024 let a
        # repetition loop run to the cap: BV1nxje63ERi averaged 40 s per ASR call with only 30
        # calls, which is generation runaway, not rescue thrashing.
        cap = 1024 if duration is None else max(48, min(1024, int(duration * TOKENS_PER_SEC)))
        with torch.inference_mode():
            out = self.asr.generate(**inputs, max_new_tokens=cap, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1] :]
        return self.asr_proc.parse_output(self.asr_proc.batch_decode(gen, skip_special_tokens=True))[0]

    def align(self, chunk, text, offset):
        inputs, word_lists = self.aln_proc.prepare_forced_aligner_inputs(
            audio=[chunk], transcript=[text], language=self.language
        )
        inputs = inputs.to(self.aln.device, self.aln.dtype)
        with torch.inference_mode():
            logits = self.aln(**inputs).logits
        decoded = self.aln_proc.decode_forced_alignment(
            logits=logits,
            input_ids=inputs["input_ids"],
            word_lists=word_lists,
            timestamp_token_id=self.aln.config.timestamp_token_id,
        )[0]
        return [
            {
                "word": w["text"],
                "start": round(offset + w["start_time"], 3),
                "end": round(offset + w["end_time"], 3),
                "space_before": False,
            }
            for w in decoded
        ]


def decode_span(engine, wav, intervals, depth, stats):
    """Decode one contiguous span once and judge it."""
    start, end = intervals[0][0], intervals[-1][1]
    chunk = wav[int(start * TARGET_SR) : int(end * TARGET_SR)]
    stats["asr_calls"] += 1
    parsed = engine.transcribe(chunk, end - start)
    text = parsed["transcription"]

    words = []
    if normalize_ja(text):
        stats["align_calls"] += 1
        words = engine.align(chunk, text, start)

    ok_cov, covered, speech = coverage_ok(words, intervals)
    ok_col, why = collapse_ok(words) if words else (not normalize_ja(text), "no text")
    offgrid = sum(1 for w in words if abs(round(w["start"] * 1000) % GRID_MS) > 1e-6)
    seg = {
        "start": round(start, 3),
        "end": round(end, 3),
        "words": words,
        "text": text,
        "lang": parsed.get("language"),
        "rescue_depth": depth,
        "interpolated_words": offgrid,
    }
    reason = "coverage %.0f%%" % (100 * covered / max(speech, 1e-6)) if not ok_cov else why
    return seg, (ok_cov and ok_col), reason


def decode_group(engine, wav, intervals, stats):
    """Bounded rescue ladder: whole window -> halves -> per interval.

    The original peel-one-and-recurse mirrored `asr_align.py`, but on this model it degenerated:
    coverage keeps failing because Qwen returns nothing for interjection-only intervals, so it
    peeled all the way down while re-decoding the whole tail at every step (180 ASR calls for 18
    windows on BV1cJjE6cEt8, RTF up to 3.0). The failure mode it is recovering from is not
    gradual, so a fixed three-level ladder reaches the same place with a bounded 1 + 2 + n calls.
    """
    seg, ok, reason = decode_span(engine, wav, intervals, 0, stats)
    if ok or len(intervals) == 1:
        if not ok:
            stats["unrescued"] += 1
        return [seg]

    stats["rescues"] += 1
    stats["rescue_reasons"].append(f"[{intervals[0][0]:.1f}-{intervals[-1][1]:.1f}] {reason}")

    mid = len(intervals) // 2
    if mid:
        halves = [decode_span(engine, wav, part, 1, stats) for part in (intervals[:mid], intervals[mid:])]
        if all(o for _s, o, _r in halves):
            return [s for s, _o, _r in halves]

    out = []
    for iv in intervals:
        s, o, _r = decode_span(engine, wav, [iv], 2, stats)
        if not o:
            stats["unrescued"] += 1
        out.append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--vad", required=True, help="dump_vad.py output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--group-sec", type=float, default=GROUP_TARGET_SEC)
    args = ap.parse_args()

    wav = load_audio_16k(args.audio)
    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    intervals = [tuple(x) for x in vad["speech"]]
    groups = group_intervals(intervals, args.group_sec)

    engine = Engine(args.language)
    stats = {"asr_calls": 0, "align_calls": 0, "rescues": 0, "unrescued": 0, "rescue_reasons": []}

    segments = []
    t0 = time.perf_counter()
    for i, group in enumerate(groups):
        segments.extend(decode_group(engine, wav, group, stats))
        print(f"  window {i + 1}/{len(groups)}", flush=True)
    wall = time.perf_counter() - t0

    payload = {
        "metadata": {
            "asr_align": {
                "model": ASR_MODEL,
                "aligner": ALIGNER_MODEL,
                "language": args.language,
                "group_target_sec": args.group_sec,
                "asr_coverage_min_ratio": COVERAGE_MIN_RATIO,
                "asr_coverage_tolerance_sec": COVERAGE_TOLERANCE_SEC,
                "collapse_max_sec_per_char": MAX_SEC_PER_CHAR,
                "collapse_max_char_density": MAX_CHAR_DENSITY,
            },
            "run": {
                "windows": len(groups),
                "segments": len(segments),
                "wall_seconds": round(wall, 2),
                "rtf": round(wall / (len(wav) / TARGET_SR), 4),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                **{k: v for k, v in stats.items() if k != "rescue_reasons"},
            },
            "rescue_reasons": stats["rescue_reasons"],
        },
        "segments": segments,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload["metadata"]["run"], ensure_ascii=False, indent=1))
    for reason in stats["rescue_reasons"]:
        print("  rescue:", reason)


if __name__ == "__main__":
    main()
