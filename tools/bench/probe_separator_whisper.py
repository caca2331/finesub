"""Whispered ASMR transcribes to boilerplate. Is that the ASR, or the separator?

A vocal separator is trained on sung/spoken voice, which is voiced. A whisper
is unvoiced -- broadband turbulence with no pitch. If the separator treats it
as noise, the ASR never gets a chance, and the failure is upstream of anything
17.10's level rule can see.

Test: take the same spans from (a) the ORIGINAL downloaded audio and (b) the
separated `-vocal.ogg`, and ask Qwen3-ASR both. If the original yields words
where the vocal track yields nothing, the separator is the one eating it.
"""
import glob
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from finesub.speech.verification import qwen_referee as qr

OUT = "tmp/whisper-corpus/out"
RAW = "data/whisper-corpus"
N_SPANS = 6
SPAN_SEC = 8.0


def spans_for(stem):
    """Evenly spaced spans that the VAD marked as speech, so both sides get
    the same audio and the comparison is not about interval selection."""

    with open(os.path.join(OUT, stem, f"{stem}-vad.json"), encoding="utf-8") as h:
        payload = json.load(h)
    segments = [s for s in payload["segments"]
                if float(s["end"]) - float(s["start"]) >= 1.0]
    if not segments:
        return []
    picks = [segments[min(len(segments) - 1, int((i + 0.5) * len(segments) / N_SPANS))]
             for i in range(N_SPANS)]
    out = []
    for segment in picks:
        start = float(segment["start"])
        out.append((start, min(start + SPAN_SEC, float(segment["end"]) + SPAN_SEC)))
    return out


def wav16k(path, out_path):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path,
                    "-ac", "1", "-ar", "16000", out_path], check=True)
    return out_path


referee = qr.QwenReferee(device="cuda")
for run_dir in sorted(glob.glob(f"{OUT}/*/")):
    stem = os.path.basename(os.path.normpath(run_dir))
    vocal = os.path.join(run_dir, f"{stem}-vocal.ogg")
    source = os.path.join(RAW, f"{stem}.wav")
    if not (os.path.exists(vocal) and os.path.exists(source)):
        continue
    picks = spans_for(stem)
    if not picks:
        continue
    original = wav16k(source, f"tmp/whisper-corpus/{stem}-orig16k.wav")
    print(f"\n=== {stem} ===")
    readers = {"original": qr._SpanReader(original), "separated": qr._SpanReader(vocal)}
    clips, keys = [], []
    for label, reader in readers.items():
        for start, end in picks:
            clip = reader.read(start, end)
            if len(clip) >= int(0.05 * qr.TARGET_SR):
                clips.append(clip)
                keys.append((label, start))
    replies = referee.transcribe_batch(clips)
    heard = {}
    for (label, start), reply in zip(keys, replies):
        heard.setdefault(start, {})[label] = str((reply or ("", None))[0] or "").strip()
    for start in sorted(heard):
        row = heard[start]
        print(f"  {start:>7.1f}s  original {row.get('original', '')[:34]!r:<38} "
              f"separated {row.get('separated', '')[:34]!r}")
referee.close()

# Result on the 5-file whisper corpus (bench-baselines 17.13, fifth finding):
# two DIFFERENT failures. On the loud whisper files the separated track still
# reads fine and it is faster-whisper that returns boilerplate; on the quiet
# ones the separator itself removes the whisper and nothing downstream can
# recover it. A whisper is unvoiced -- turbulence with no fundamental -- and
# the separator is trained on voiced material.
