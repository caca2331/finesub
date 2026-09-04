"""How much does the separator attenuate, on the same spans?

The Qwen comparison showed the CONTENT disappears. This measures the level
directly, so "the separator eats whisper" stops being an inference from a
downstream reader. Both sides are read at 16 kHz mono, so the ASR track's
resample is controlled for and the only difference is the separation itself.
"""
import glob
import os
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from finesub.speech.verification import qwen_referee as qr

OUT = "tmp/whisper-corpus/out"
RAW = "data/whisper-corpus"
STEP = 30.0


def dbfs(samples):
    if samples.size == 0:
        return float("nan")
    return 10.0 * np.log10(float(np.mean(np.square(samples))) + 1e-30)


print(f"{'file':<16} {'spans':>6} {'original p50':>13} {'separated p50':>14} "
      f"{'median attenuation':>19}")
for run_dir in sorted(glob.glob(f"{OUT}/*/")):
    stem = os.path.basename(os.path.normpath(run_dir))
    vocal = os.path.join(run_dir, f"{stem}-vocal.ogg")
    original = f"tmp/whisper-corpus/{stem}-orig16k.wav"
    if not (os.path.exists(vocal) and os.path.exists(original)):
        source = os.path.join(RAW, f"{stem}.wav")
        if not os.path.exists(source):
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source,
                        "-ac", "1", "-ar", "16000", original], check=True)
    a, b = qr._SpanReader(original), qr._SpanReader(vocal)
    rows = []
    for start in np.arange(0.0, 600.0 - STEP, STEP):
        x, y = a.read(start, start + STEP), b.read(start, start + STEP)
        if x.size and y.size:
            rows.append((dbfs(x), dbfs(y)))
    if not rows:
        continue
    orig = np.array([r[0] for r in rows])
    sep = np.array([r[1] for r in rows])
    print(f"{stem:<16} {len(rows):>6} {np.percentile(orig, 50):>12.1f} "
          f"{np.percentile(sep, 50):>13.1f} "
          f"{np.percentile(orig - sep, 50):>18.1f} dB")

# Measured on the whisper corpus vs four production runs (bench-baselines
# 17.13): ordinary speech loses 0.0-0.3 dB, whisper loses 5-41 dB, and the
# files that fail downstream are exactly the ones past 20 dB. So "whisper is
# quiet" is mostly something this pipeline does, not a property of whisper.
