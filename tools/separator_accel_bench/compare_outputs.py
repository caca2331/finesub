"""Compare two separator outputs the way this project accepts a change.

A single waveform score is not enough and never was. E11 recorded a variant with
the *best* SI-SDR whose downstream boundaries were nine seconds out, and the
in-memory demix runner (E15) shipped two bugs that a global score waved through:
one arm silently running FP32, and a block whose normalisation gain differed by
1.27x. Both were invisible in cosine similarity and obvious here.

Four views, in the order they are worth reading:

``vad``     segment count and per-boundary deltas against the reference. This is
            the criterion; a boundary inside one 20ms frame is the same decision.
``diff``    the segments with no counterpart, with how loud each one is -- a
            0.3s fragment at -50dBFS crossing the threshold is a different event
            from a 0.9s one at -13dBFS.
``snr``     per-window SNR **stratified by loudness**, because digital silence
            otherwise owns the bottom of any "worst" list.
``gain``    the best-fit scale between the two in each window. A residual that
            is really a level difference reads as catastrophic SNR until this
            separates them, which is exactly how the normalisation bug looked.
``lag``     the best integer sample offset on the worst window. Together with
            ``gain`` this closes the diagnosis: a bad residual is a time shift
            (chunk mis-assembly, a resampler off by a few samples), a level
            difference, or genuinely different audio -- and they are told apart
            here, not by staring at the number.

    PYTHONPATH=src python -m tools.separator_accel_bench.compare_outputs \
        reference.flac candidate.flac
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

#: One VAD frame. Boundaries closer than this agree on every frame.
_FRAME_SEC = 0.02


def _segments(path: Path):
    from finesub.speech.preprocessing.energy import run_vad_file

    result, _params, duration, _track = run_vad_file(str(path))
    return [(float(s["start"]), float(s["end"])) for s in result], duration


def _level_db(path: Path, start: float, end: float) -> tuple[float, float]:
    info = sf.info(str(path))
    data, _rate = sf.read(
        str(path),
        start=int(start * info.samplerate),
        frames=max(1, int((end - start) * info.samplerate)),
        dtype="float32",
        always_2d=True,
    )
    peak = float(np.abs(data).max()) or 1e-12
    rms = float(np.sqrt((data.astype(np.float64) ** 2).mean())) or 1e-12
    return 20 * np.log10(peak), 20 * np.log10(rms)


def _report_vad(reference: Path, candidate: Path) -> None:
    ref, duration = _segments(reference)
    got, _ = _segments(candidate)
    print(f"\n[vad] {reference.name}: {len(ref)} segments over {duration:.2f}s"
          f"   {candidate.name}: {len(got)} segments")
    if not ref and not got:
        # A silent or instrumental track separates to nothing, which is a
        # legitimate result and not a reason to stop before the waveform views.
        print("       both sides found no speech")
    elif len(ref) == len(got):
        starts = [abs(a[0] - b[0]) for a, b in zip(ref, got)]
        ends = [abs(a[1] - b[1]) for a, b in zip(ref, got)]
        exact = sum(1 for s, e in zip(starts, ends) if s == 0 and e == 0)
        deltas = starts + ends
        beyond = sum(1 for d in deltas if d > _FRAME_SEC + 1e-4)
        print(f"       {exact}/{len(ref)} segments identical, "
              f"{beyond}/{len(deltas)} boundaries off by more than one frame "
              f"(max {max(deltas):.4f}s)")
    speech_ref = sum(b - a for a, b in ref)
    speech_got = sum(b - a for a, b in got)
    print(f"       total speech {speech_got:.3f}s vs {speech_ref:.3f}s "
          f"({speech_got - speech_ref:+.3f}s)")

    unmatched = False
    for label, mine, theirs, source in (
        (reference.name, ref, got, reference),
        (candidate.name, got, ref, candidate),
    ):
        for segment in mine:
            if any(segment[0] < o[1] and o[0] < segment[1] for o in theirs):
                continue
            unmatched = True
            peak, rms = _level_db(source, *segment)
            print(f"       only in {label}: {segment[0]:.3f}-{segment[1]:.3f}s "
                  f"({segment[1] - segment[0]:.3f}s) peak {peak:.1f}dBFS "
                  f"rms {rms:.1f}dBFS")
    if not unmatched:
        print("       every segment has a counterpart")


def _best_lag(left, right, start: int, frames: int, search: int = 40):
    """Integer offset that best aligns one window, and the SNR it buys.

    Edges are trimmed by the search width so the rolled copy never wraps real
    audio in from the far end -- which is also why a window has to be several
    times the search width to say anything. Returns ``None`` when it is not.
    """

    search = min(search, frames // 4)
    if search < 1:
        return None
    a = left[start : start + frames, 0]
    b = right[start : start + frames, 0]
    best = (0, -np.inf)
    for lag in range(-search, search + 1):
        trim = slice(search, -search)
        x, y = a[trim], np.roll(b, lag)[trim]
        noise = float(((x - y) ** 2).mean())
        snr = 10 * np.log10(float((x**2).mean()) / noise) if noise > 0 else np.inf
        if snr > best[1]:
            best = (lag, snr)
    return best


def _report_waveform(reference: Path, candidate: Path, window_sec: float) -> None:
    left, rate = sf.read(str(reference), dtype="float64", always_2d=True)
    right, rate_right = sf.read(str(candidate), dtype="float64", always_2d=True)
    if rate != rate_right:
        print(f"\n[snr] rate mismatch: {rate} vs {rate_right}")
        return
    print(f"\n[snr] {len(left)} vs {len(right)} frames at {rate} Hz")
    frames = min(len(left), len(right))
    left, right = left[:frames], right[:frames]

    step = max(1, int(window_sec * rate))
    at, snr, gain, level = [], [], [], []
    for start in range(0, frames - step + 1, step):
        a, b = left[start : start + step], right[start : start + step]
        power = float((a**2).mean())
        if power <= 0:
            continue
        noise = float(((a - b) ** 2).mean())
        scale = float((a * b).sum() / max((b * b).sum(), 1e-30))
        matched = float(((a - scale * b) ** 2).mean())
        at.append(start / rate)
        snr.append(10 * np.log10(power / noise) if noise > 0 else np.inf)
        gain.append(scale)
        level.append(10 * np.log10(power))

    if not at:
        print("       no windows with signal")
        return
    snr_arr, level_arr, gain_arr = np.array(snr), np.array(level), np.array(gain)
    finite = np.isfinite(snr_arr)
    if not finite.any():
        print("       identical everywhere")
        return
    # `>=`, not `>`: with one window, or with every window at the same level,
    # a strict split against the median selects nothing and every statistic
    # below raises on an empty array. Including ties is also the more honest
    # reading of "the louder half" when the levels are flat.
    loud = finite & (level_arr >= np.percentile(level_arr[finite], 50))
    if not loud.any():
        loud = finite
    print(f"       median {np.median(snr_arr[finite]):.2f} dB over "
          f"{finite.sum()} windows; the louder half: "
          f"min {snr_arr[loud].min():.2f} dB, p05 "
          f"{np.percentile(snr_arr[loud], 5):.2f} dB")
    print(f"[gain] best-fit scale over the louder half: "
          f"median {np.median(gain_arr[loud]):.5f}, "
          f"range {gain_arr[loud].min():.5f}-{gain_arr[loud].max():.5f}")
    if abs(np.log(max(gain_arr[loud].max(), 1e-9))) > 0.01 or abs(
        np.log(max(gain_arr[loud].min(), 1e-9))
    ) > 0.01:
        print("       ^ a scale away from 1.0 means a level difference, not a "
              "different separation; re-read the SNR with that in mind")
    print("       worst windows in the louder half:")
    order = np.argsort(np.where(loud, snr_arr, np.inf))[:6]
    for index in order:
        print(f"         {at[index]:9.2f}s  SNR {snr_arr[index]:7.2f} dB   "
              f"level {level_arr[index]:6.1f} dB   gain {gain_arr[index]:.5f}")

    worst = int(order[0])
    aligned = _best_lag(left, right, int(at[worst] * rate), step)
    if aligned is None:
        print("[lag]  window too short to search for an offset")
        return
    lag, lag_snr = aligned
    print(f"[lag]  worst window ({at[worst]:.2f}s): best integer offset {lag:+d} "
          f"samples, SNR there {lag_snr:.2f} dB")
    if lag != 0:
        print("       ^ a non-zero offset is a timeline bug, not a quality "
              "difference: look at the chunk assembly and the resampler")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--skip-vad", action="store_true",
                        help="waveform views only; the VAD pass decodes twice more")
    args = parser.parse_args()

    if not args.skip_vad:
        _report_vad(args.reference, args.candidate)
    _report_waveform(args.reference, args.candidate, args.window_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
