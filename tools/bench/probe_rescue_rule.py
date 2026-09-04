"""Score the windowed separator-rescue rule, and its third condition.

The rule (bench-baselines 17.15 / 17.17), per 30 s window:

    attenuation(original vs separated) > 12 dB
    AND the separated track's coverage in that window < 5%
    AND the referee hears words in the ORIGINAL audio of that window
    -> decode that window from the original

The first two conditions alone misfire on instrumental music, which is a
stretch the separator SHOULD empty: measured 10/19 and 17/19 triggering windows
on solo piano and lofi, at higher attenuation than any whisper file. The third
condition is what separates "the separator destroyed a whisper" from "the
separator correctly emptied an instrumental" -- those two are identical in
level, and differ only in whether the original contains words.

    python -m tools.bench.probe_rescue_rule windows RUNDIR SOURCE [RUNDIR SOURCE ...]
    python -m tools.bench.probe_rescue_rule probe   RUNDIR SOURCE [RUNDIR SOURCE ...]

`windows` scores the first two conditions and prints a per-window trigger map;
`probe` runs the referee over the original audio of the triggering windows.
Corpora: `tools/bench/music_corpus.jsonl`, `tools/bench/whisper_corpus.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from finesub.speech.verification import qwen_referee as qr

STEP = 30.0
ATTEN_DB = 12.0
COVERAGE = 0.05
PROBE_SEC = 8.0
MAX_PROBES = 6


def _dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("nan")
    return 10.0 * np.log10(float(np.mean(np.square(samples))) + 1e-30)


def _sixteen_k(source: str, stem: str) -> str:
    """Both sides are read at 16 kHz mono so the ASR track's own resample is
    controlled for and the only difference measured is the separation."""

    out = os.path.join("tmp", f"rescue-{stem}-16k.wav")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source,
                        "-ac", "1", "-ar", "16000", out], check=True)
    return out


def _windows(run_dir: str, source: str):
    """(attenuation dB, coverage, window start) per 30 s window."""

    stem = os.path.basename(os.path.normpath(run_dir))
    vocal = os.path.join(run_dir, f"{stem}-vocal.ogg")
    vad_path = os.path.join(run_dir, f"{stem}-vad.json")
    if not (os.path.exists(vocal) and os.path.exists(vad_path)):
        return stem, []
    with open(vad_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    spans = [(float(s["start"]), float(s["end"])) for s in payload["segments"]]
    duration = float(payload["audio_duration"])
    a = qr._SpanReader(_sixteen_k(source, stem))
    b = qr._SpanReader(vocal)
    rows = []
    for lo in np.arange(0.0, duration - STEP, STEP):
        hi = lo + STEP
        x, y = a.read(lo, hi), b.read(lo, hi)
        if not (x.size and y.size):
            continue
        speech = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in spans)
        rows.append((_dbfs(x) - _dbfs(y), speech / STEP, float(lo)))
    return stem, rows


def _fires(row) -> bool:
    return row[0] > ATTEN_DB and row[1] < COVERAGE


def cmd_windows(pairs) -> int:
    print(f"trigger = attenuation > {ATTEN_DB:.0f} dB AND window coverage < "
          f"{COVERAGE * 100:.0f}%   ({STEP:.0f} s windows)")
    print(f"{'run':<16} {'wins':>5} {'atten p50':>10} {'atten max':>10} "
          f"{'triggering':>12}  map")
    for run_dir, source in pairs:
        stem, rows = _windows(run_dir, source)
        if not rows:
            print(f"{stem:<16} (no vad.json / vocal.ogg)")
            continue
        atten = np.array([r[0] for r in rows])
        marks = "".join("*" if _fires(r) else "." for r in rows)
        print(f"{stem:<16} {len(rows):>5} {np.percentile(atten, 50):>9.1f} "
              f"{atten.max():>9.1f} {sum(1 for r in rows if _fires(r)):>6} "
              f"/ {len(rows):<4} {marks}")
    return 0


def cmd_probe(pairs) -> int:
    referee = qr.QwenReferee(device="cuda")
    try:
        for run_dir, source in pairs:
            stem, rows = _windows(run_dir, source)
            fired = [r[2] for r in rows if _fires(r)]
            if not fired:
                print(f"{stem:<16} no triggering window")
                continue
            picks = [fired[min(len(fired) - 1,
                               int((i + 0.5) * len(fired) / MAX_PROBES))]
                     for i in range(min(MAX_PROBES, len(fired)))]
            reader = qr._SpanReader(_sixteen_k(source, stem))
            clips = [reader.read(lo, lo + PROBE_SEC) for lo in picks]
            replies = referee.transcribe_batch([c for c in clips if c.size])
            print(f"\n{stem:<16} {len(fired)} triggering windows, "
                  f"probing {len(picks)}")
            for lo, reply in zip(picks, replies):
                heard = str((reply or ("", None))[0] or "").strip()
                print(f"    {lo:>6.0f}s  referee on the ORIGINAL: {heard[:52]!r}")
    finally:
        referee.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("windows", "probe"))
    parser.add_argument("pairs", nargs="+",
                        help="RUNDIR SOURCE pairs (a stage output dir and the "
                             "unseparated media it came from)")
    args = parser.parse_args(argv)
    if len(args.pairs) % 2:
        parser.error("pairs must come as RUNDIR SOURCE")
    pairs = list(zip(args.pairs[0::2], args.pairs[1::2]))
    return cmd_windows(pairs) if args.mode == "windows" else cmd_probe(pairs)


if __name__ == "__main__":
    raise SystemExit(main())
