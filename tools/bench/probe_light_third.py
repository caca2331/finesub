"""Are there cheap alternatives to the referee as the rescue rule's third
condition? Measured answer: no.

The third condition (bench-baselines 17.17) runs a transcription model on the
ORIGINAL audio of every window the level conditions fire on. That is heavy for
what it is being asked -- "is there a voice here, rather than an instrument" --
so this scores four signals that cost an FFT or a small model instead.

The trap this probe exists to make visible: on a two-way problem (tonal music
vs whisper) spectral flatness separates perfectly. Add BROADBAND non-speech --
rain, a fireplace -- and it collapses, because rain reads flatter than whisper
does. Any candidate has to be scored on the three-way problem:

    MUSIC    tonal non-speech
    RESIDUE  broadband non-speech
    WHISPER  broadband speech      <- the only one that must be rescued

    python -m tools.bench.probe_light_third GROUP RUNDIR SOURCE [GROUP RUNDIR SOURCE ...]

GROUP is a free label used to bucket the results (MUSIC / RESIDUE / WHISPER in
17.18). Corpora: `music_corpus.jsonl`, `whisper_corpus.jsonl`, `p1_corpus.jsonl`.
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

STEP, PROBE = 30.0, 8.0
ATTEN_DB, COVERAGE = 12.0, 0.05
SR = 16000
PER_FILE = 5


def _dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return 10.0 * np.log10(float(np.mean(np.square(x))) + 1e-30)


def flatness(x: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    """Geometric over arithmetic mean of the power spectrum. Near 0 is tonal."""

    if x.size < n_fft:
        return float("nan")
    window = np.hanning(n_fft).astype(np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2 + 1e-12
    return float(np.mean(np.exp(np.log(spec).mean(-1)) / spec.mean(-1)))


def mod4_8(x: np.ndarray, n_fft: int = 512, hop: int = 160) -> float:
    """Share of envelope-modulation energy at the syllable rate (4-8 Hz)."""

    if x.size < n_fft:
        return float("nan")
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), n_fft)[::hop]
    env = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-12)
    env = env - env.mean()
    if env.size < 32:
        return float("nan")
    spec = np.abs(np.fft.rfft(env * np.hanning(env.size))) ** 2
    freq = np.fft.rfftfreq(env.size, d=hop / SR)
    return float(spec[(freq >= 4) & (freq <= 8)].sum()
                 / (spec[(freq >= 0.5) & (freq <= 16)].sum() + 1e-30))


def spectral_flux(x: np.ndarray, n_fft: int = 512, hop: int = 160) -> float:
    """How much the normalised spectrum moves frame to frame."""

    if x.size < n_fft * 2:
        return float("nan")
    window = np.hanning(n_fft).astype(np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(x.astype(np.float64), n_fft)[::hop]
    spec = np.abs(np.fft.rfft(frames * window, axis=-1))
    spec = spec / (spec.sum(-1, keepdims=True) + 1e-12)
    return float(np.mean(np.abs(np.diff(spec, axis=0)).sum(-1)))


def _sixteen_k(source: str, stem: str) -> str:
    out = os.path.join("tmp", f"light-{stem}-16k.wav")
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source,
                        "-ac", "1", "-ar", "16000", out], check=True)
    return out


def _triggering(run_dir: str, original: str):
    stem = os.path.basename(os.path.normpath(run_dir))
    vocal = os.path.join(run_dir, f"{stem}-vocal.ogg")
    vad_path = os.path.join(run_dir, f"{stem}-vad.json")
    if not (os.path.exists(vocal) and os.path.exists(vad_path)):
        return []
    with open(vad_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    spans = [(float(s["start"]), float(s["end"])) for s in payload["segments"]]
    duration = float(payload["audio_duration"])
    a, b = qr._SpanReader(original), qr._SpanReader(vocal)
    fired = []
    for lo in np.arange(0.0, duration - STEP, STEP):
        hi = lo + STEP
        x, y = a.read(lo, hi), b.read(lo, hi)
        if not (x.size and y.size):
            continue
        speech = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in spans)
        if _dbfs(x) - _dbfs(y) > ATTEN_DB and speech / STEP < COVERAGE:
            fired.append(float(lo))
    return fired


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("triples", nargs="+", help="GROUP RUNDIR SOURCE triples")
    parser.add_argument("--no-referee", action="store_true",
                        help="skip the model, score the cheap features only")
    args = parser.parse_args(argv)
    if len(args.triples) % 3:
        parser.error("arguments must come as GROUP RUNDIR SOURCE")
    cases = list(zip(args.triples[0::3], args.triples[1::3], args.triples[2::3]))

    referee = None if args.no_referee else qr.QwenReferee(device="cuda")
    rows = []
    try:
        for group, run_dir, source in cases:
            stem = os.path.basename(os.path.normpath(run_dir))
            original = _sixteen_k(source, stem)
            fired = _triggering(run_dir, original)
            if not fired:
                print(f"{group:<8} {stem:<14} no triggering window")
                continue
            picks = [fired[min(len(fired) - 1, int((i + 0.5) * len(fired) / PER_FILE))]
                     for i in range(min(PER_FILE, len(fired)))]
            reader = qr._SpanReader(original)
            clips = [reader.read(lo, lo + PROBE) for lo in picks]
            replies = ([("", None)] * len(clips) if referee is None
                       else referee.transcribe_batch(clips))
            for lo, clip, reply in zip(picks, clips, replies):
                rows.append({
                    "group": group, "stem": stem, "at": lo,
                    "flatness": flatness(clip), "mod4_8": mod4_8(clip),
                    "flux": spectral_flux(clip),
                    "referee": str((reply or ("", None))[0] or "").strip(),
                })
    finally:
        if referee is not None:
            referee.close()

    print(f"\n{'group':<8} {'run':<14} {'at':>6} {'flat':>7} {'mod4-8':>7} "
          f"{'flux':>7}  referee on the ORIGINAL")
    for r in rows:
        print(f"{r['group']:<8} {r['stem']:<14} {r['at']:>6.0f} "
              f"{r['flatness']:>7.4f} {r['mod4_8']:>7.3f} {r['flux']:>7.4f}  "
              f"{r['referee'][:40]!r}")

    groups = sorted({r["group"] for r in rows})
    print("\nrange per group -- a candidate is only usable if one group is "
          "disjoint from every other:")
    for key in ("flatness", "mod4_8", "flux"):
        cells = []
        for g in groups:
            v = np.array([r[key] for r in rows if r["group"] == g])
            v = v[np.isfinite(v)]
            cells.append(f"{g[:3]} [{v.min():.3f}, {v.max():.3f}]" if v.size else f"{g[:3]} -")
        print(f"  {key:<9} " + "  ".join(cells))
    if referee is not None or any(r["referee"] for r in rows):
        print("\n  referee   " + "  ".join(
            f"{g[:3]} {sum(1 for r in rows if r['group'] == g and r['referee'])}"
            f"/{sum(1 for r in rows if r['group'] == g)}" for g in groups))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
