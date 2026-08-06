"""Dump the VAD's two dB tracks as SRT, one cue per 0.1 s, for eyeballing.

The detector's whole decision is `energy_db <= noise_floor_db + margin`, and both
sides are invisible in any artifact the pipeline writes. Rendering them as subtitles
puts them on the same timeline as the audio and the word-level SRTs, so a player or
subtitle editor becomes the inspection tool.

Two files, deliberately not merged: load them as separate tracks and the vertical
gap between them *is* the SNR the detector is thresholding.

The floor defaults to the pre-branch estimator, since that is the one the existing
annotations were produced against. `--floor production` renders whatever is in
`energy.py` right now instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def _bucket(values: np.ndarray, starts: np.ndarray, duration: float,
            bucket: float, agg: str) -> list:
    """(start, end, value) per bucket, aggregating the frames whose start falls in it."""
    fn = {"median": np.median, "mean": np.mean, "max": np.max, "min": np.min}[agg]
    n = max(1, int(np.ceil(duration / bucket)))
    idx = np.clip((starts / bucket).astype(int), 0, n - 1)
    out = []
    order = np.argsort(idx, kind="stable")
    idx_s, val_s = idx[order], values[order]
    bounds = np.searchsorted(idx_s, np.arange(n + 1))
    for k in range(n):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        if hi <= lo:
            continue
        out.append((k * bucket, min((k + 1) * bucket, duration), float(fn(val_s[lo:hi]))))
    return out


def write_srt(path: Path, cues, fmt: Callable[[float], str]) -> None:
    from asr_playground.speech.preprocessing.energy import format_srt_time

    lines = []
    for i, (s, e, v) in enumerate(cues, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(s)} --> {format_srt_time(e)}")
        lines.append(fmt(v))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="the separated vocals the VAD sees")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", default=None, help="output basename (default: audio stem)")
    ap.add_argument("--bucket", type=float, default=0.1)
    ap.add_argument("--agg", default="median", choices=("median", "mean", "max", "min"))
    ap.add_argument("--floor", default="legacy", choices=("legacy", "production"),
                    help="legacy = the pre-branch estimator (default)")
    ap.add_argument("--intervals", action="store_true",
                    help="also render the speech intervals the ASR is handed")
    ap.add_argument("--pad-right-ms", type=float, default=None,
                    help="override NEGATIVE_PAD_RIGHT_MS for the interval render")
    ap.add_argument("--group-left-lead-sec", type=float, default=0.0,
                    help="render the intervals as the ASR group assembly widens "
                         "them: up to this much real audio prepended, bounded by "
                         "the previous interval's end")
    args = ap.parse_args()

    from energy_sweep import compute_tracks

    audio = Path(args.audio)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = args.stem or audio.stem

    tr = compute_tracks(audio)
    energy = tr.energy_db.numpy().astype(np.float64)
    starts = tr.frame_starts.numpy().astype(np.float64)
    if args.floor == "legacy":
        from floor_lab import legacy
        floor = legacy()(energy, starts, tr.duration)
    else:
        floor = tr.noise_floor.numpy().astype(np.float64)

    # Cue text is read at a glance while the audio plays, so it is kept as short as
    # it can be: integer dB, no unit, and everything at or under the DB_EPS clamp
    # collapsed to -99. Nothing is lost -- below -99 there is no signal to resolve.
    def fmt(v: float) -> str:
        return f"{max(-99.0, v):.0f}"

    for name, values in (("energy-db", energy), ("noise-floor-db", floor)):
        cues = _bucket(values, starts, tr.duration, args.bucket, args.agg)
        out = outdir / f"{stem}-vad-{name}.srt"
        write_srt(out, cues, fmt)
        print(f"{out}  {len(cues)} cues  "
              f"[{fmt(min(c[2] for c in cues))} .. {fmt(max(c[2] for c in cues))}] dB")

    if args.intervals:
        # The *speech* intervals, not the non-speech ones production writes: these
        # are what the ASR is handed, so they are what lines up with a word-level
        # SRT. Each cue is labelled with its index and length, because a track of
        # identical empty captions is unreadable in an editor.
        from asr_playground.speech.preprocessing import energy as E

        raw = E._score_to_non_speech_intervals(
            tr.energy_db, torch.from_numpy(floor.astype(np.float32)), tr.frame_dbfs,
            tr.frame_starts, tr.frame_ends, tr.duration,
            enter_margin_db=6.0, weighted=bool(E.WEIGHTED_INTERVAL))
        saved = E.NEGATIVE_PAD_RIGHT_MS
        try:
            if args.pad_right_ms is not None:
                E.NEGATIVE_PAD_RIGHT_MS = args.pad_right_ms
            padded = E._apply_negative_padding(raw, tr.duration)
            pr = E.NEGATIVE_PAD_RIGHT_MS
        finally:
            E.NEGATIVE_PAD_RIGHT_MS = saved
        speech = [(float(a), float(b))
                  for a, b in E.invert_intervals(padded, tr.duration) if b > a]
        lead = max(0.0, float(args.group_left_lead_sec))
        if lead > 0:
            # Same rule as transcribe.build_combined_audio: reach back into the
            # silence, never past the previous interval, so this is what the
            # decoder is handed rather than what the VAD marked.
            widened = []
            for i, (s0, e0) in enumerate(speech):
                floor_s = 0.0 if i == 0 else speech[i - 1][1]
                widened.append((max(s0 - lead, floor_s), e0))
            speech = widened
        cues = [(s, e, (i, e - s)) for i, (s, e) in enumerate(speech, start=1)]
        suffix = "-asr-input" if lead > 0 else ""
        out = outdir / f"{stem}-vad-speech-intervals{suffix}.srt"
        write_srt(out, cues, lambda t: f"#{t[0]} {t[1]:.2f}s")
        total = sum(e - s for s, e, _ in cues)
        print(f"{out}  {len(cues)} speech intervals  {total:.0f}s "
              f"= {total/tr.duration:.1%}  (padR={pr:.0f}ms)")

    snr = energy - floor
    print(f"\nfor reference, energy - floor over the whole file: "
          f"p10={np.quantile(snr, .1):.1f} med={np.median(snr):.1f} "
          f"p90={np.quantile(snr, .9):.1f} dB   (the detector enters non-speech "
          f"under +6 dB, with the absolute gate also satisfied)")


if __name__ == "__main__":
    main()
