"""How quiet does real speech get, and what can a level floor safely remove?

`bench-baselines.md` 17.6b established that the absolute level of the admitted
frames separates separator residue from real sparse speech where raw coverage
does not. Turning that into a usable filter needs the other half: the
conservative lower bound of real speech, measured rather than assumed.

Three passes, all over artifacts that already exist (no decoding, no GPU):

  intervals  per-interval peak/p90 dBFS, pooled per group. This is the level a
             floor can be set on -- NOT the frame level: inside real production
             speech intervals a quarter of the frames are below -43 dBFS
             (pauses, breath, trailing silence), so a per-frame floor is
             hopeless before it starts.
  coverage   coverage recomputed counting only intervals whose peak clears each
             floor, marked against the 1% line the removed failover predicate
             warned at. The pass condition is "residue drops under the
             threshold, real speech does not move".
  texts      every aligned text sitting inside an interval below a floor -- the
             direct evidence for where real speech stops. Mind that a long
             segment can overlap a short interval it did not come from.
  tiers      a two-condition rule (peak AND power mean), scored as a DROP tier
             (intervals never decoded -- cost counted in transcript, not
             seconds) and as a VERIFY tier (intervals forced into the referee's
             suspect set -- cost counted in clips). See 17.10.

    python -m tools.bench.probe_soft_speech_floor intervals RUNDIR [RUNDIR ...]
    python -m tools.bench.probe_soft_speech_floor coverage  RUNDIR [RUNDIR ...]
    python -m tools.bench.probe_soft_speech_floor texts --floor -45 RUNDIR ...
    python -m tools.bench.probe_soft_speech_floor tiers --peak -45 --pmean -55 RUNDIR ...

⚠ Every number this produces is bounded by the material: this machine has no
whisper or ASMR recording. The softest verified speech is soft-spoken guided
meditation, so a margin measured here is a margin against SOFT-SPOKEN, not
against whispering.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterator, NamedTuple

import numpy as np

FLOORS = (-60.0, -55.0, -50.0, -45.0, -40.0)
QUANTILES = (0.5, 1, 5, 10, 25, 50, 90)


class Interval(NamedTuple):
    peak_dbfs: float
    p90_dbfs: float
    pmean_dbfs: float
    duration: float
    text: str


def _read(run_dir: str) -> tuple[str, float, list[Interval]] | None:
    """(stem, audio duration, intervals) for a stage output directory."""

    stem = os.path.basename(os.path.normpath(run_dir))
    vad_path = os.path.join(run_dir, f"{stem}-vad.json")
    if not os.path.exists(vad_path):
        return None
    with open(vad_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    track = payload.get("energy_track") or {}
    npz_path = os.path.join(run_dir, str(track.get("arrays") or ""))
    if not os.path.exists(npz_path):
        return None
    hop = float(track["hop_sec"])
    with np.load(npz_path) as arrays:
        dbfs = np.asarray(arrays["frame_dbfs"], dtype=np.float64)

    aligned_path = os.path.join(run_dir, f"{stem}-aligned.json")
    segments: list[dict] = []
    if os.path.exists(aligned_path):
        with open(aligned_path, encoding="utf-8") as handle:
            segments = json.load(handle)["segments"]

    intervals: list[Interval] = []
    for interval in payload["segments"]:
        start, end = float(interval["start"]), float(interval["end"])
        window = dbfs[int(start / hop): int(end / hop) + 1]
        window = window[np.isfinite(window)]
        if not window.size:
            continue
        text = "".join(
            str(s.get("text") or "") for s in segments
            if float(s["end"]) > start and float(s["start"]) < end
        ).strip()
        pmean = 10.0 * np.log10(np.mean(np.power(10.0, window / 10.0)) + 1e-30)
        intervals.append(Interval(
            float(window.max()), float(np.percentile(window, 90)), float(pmean),
            end - start, text
        ))
    return stem, float(payload["audio_duration"]), intervals


def _runs(run_dirs: list[str]) -> Iterator[tuple[str, float, list[Interval]]]:
    for run_dir in run_dirs:
        got = _read(run_dir)
        if got is None:
            print(f"  (skipped, no vad.json/npz: {run_dir})")
            continue
        yield got


def cmd_intervals(run_dirs: list[str]) -> int:
    pooled = [i for _, _, items in _runs(run_dirs) for i in items]
    if not pooled:
        return 1
    print(f"pooled intervals: {len(pooled)}")
    for label, values in (
        ("peak", np.array([i.peak_dbfs for i in pooled])),
        ("p90 ", np.array([i.p90_dbfs for i in pooled])),
    ):
        cells = "  ".join(f"{np.percentile(values, q):>6.1f}" for q in QUANTILES)
        head = "  ".join(f"{'p' + format(q, 'g'):>6}" for q in QUANTILES)
        print(f"  {label} dBFS  {head}")
        print(f"            {cells}")
    total = sum(i.duration for i in pooled)
    print("\n  share of SPEECH SECONDS kept by an interval-peak floor:")
    for floor in FLOORS:
        kept = sum(i.duration for i in pooled if i.peak_dbfs > floor)
        print(f"    {floor:>6.0f} dBFS  {100 * kept / total:>6.2f}%")
    return 0


def _coverage(run_dirs: list[str], min_coverage: float) -> int:
    print(f"'!' marks coverage under {min_coverage * 100:.0f}% "
          f"-- where the removed failover predicate used to warn")
    print(f"{'run':<18} {'raw':>8} " + " ".join(f"{f:>8.0f}" for f in FLOORS))
    for stem, duration, intervals in _runs(run_dirs):
        if duration <= 0:
            continue

        def cell(kept: float) -> str:
            share = kept / duration
            return f"{'!' if share < min_coverage else ' '}{100 * share:>6.2f}"

        raw = sum(i.duration for i in intervals)
        cells = " ".join(
            cell(sum(i.duration for i in intervals if i.peak_dbfs > floor))
            for floor in FLOORS
        )
        print(f"{stem:<18} {cell(raw):>8} {cells}")
    return 0


def cmd_texts(run_dirs: list[str], floor: float) -> int:
    rows = [
        (i.peak_dbfs, i.duration, stem, i.text)
        for stem, _, intervals in _runs(run_dirs)
        for i in intervals
        if i.peak_dbfs < floor and i.text
    ]
    total = sum(
        1 for _, _, intervals in _runs(run_dirs)
        for i in intervals if i.peak_dbfs < floor
    )
    print(f"\nintervals below {floor:.0f} dBFS: {total}; carrying any text: {len(rows)}")
    for peak, duration, stem, text in sorted(rows):
        print(f"  peak {peak:>6.1f}  {duration:>5.2f}s  {stem:<14} {text[:44]!r}")
    return 0


#: The 1% line the removed failover predicate warned at
#: (`speech/preprocessing/vad_failover.py`, deleted 2026-08-30 --
#: `bench-baselines.md` 17.12). Kept here as a reference mark for reading
#: coverage tables, not as live behaviour.
FALLBACK_MIN_COVERAGE = 0.01


def _min_coverage() -> float:
    return FALLBACK_MIN_COVERAGE


def cmd_tiers(run_dirs: list[str], peak: float, pmean: float) -> int:
    """Score `peak < PEAK and pmean < PMEAN` over the given runs."""

    pooled = [(stem, i) for stem, _, items in _runs(run_dirs) for i in items]
    if not pooled:
        return 1
    picked = [(stem, i) for stem, i in pooled
              if i.peak_dbfs < peak and i.pmean_dbfs < pmean]
    total = sum(i.duration for _, i in pooled)
    seconds = sum(i.duration for _, i in picked)
    with_text = [(stem, i) for stem, i in picked if i.text]
    print(f"rule: peak < {peak:.0f} and pmean < {pmean:.0f} dBFS")
    print(f"  intervals   {len(picked)} / {len(pooled)} "
          f"({100 * len(picked) / len(pooled):.1f}%)")
    print(f"  seconds     {seconds:.0f} / {total:.0f} "
          f"({100 * seconds / total:.2f}%)  <- decode time saved as a DROP tier, "
          f"referee load as a VERIFY tier")
    if not picked:
        print("  nothing matched -- on a real-speech arm that is the pass condition")
        return 0
    print(f"  no text     {len(picked) - len(with_text)} "
          f"({100 * (len(picked) - len(with_text)) / len(picked):.0f}%)")
    print(f"  with text   {len(with_text)}  <- read these before trusting the rule")
    for stem, item in sorted(with_text, key=lambda x: -x[1].peak_dbfs):
        print(f"    peak {item.peak_dbfs:>6.1f}  pmean {item.pmean_dbfs:>6.1f}  "
              f"{item.duration:>5.2f}s  {stem:<14} {item.text[:38]!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("intervals", "coverage", "texts", "tiers"))
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--floor", type=float, default=-45.0,
                        help="texts mode: report intervals peaking below this")
    parser.add_argument("--min-coverage", type=float, default=_min_coverage(),
                        help="coverage mode: the warn threshold to mark against")
    parser.add_argument("--peak", type=float, default=-45.0,
                        help="tiers mode: the peak condition")
    parser.add_argument("--pmean", type=float, default=-55.0,
                        help="tiers mode: the power-mean condition")
    args = parser.parse_args(argv)

    if args.mode == "intervals":
        return cmd_intervals(args.run_dirs)
    if args.mode == "coverage":
        return _coverage(args.run_dirs, args.min_coverage)
    if args.mode == "tiers":
        return cmd_tiers(args.run_dirs, args.peak, args.pmean)
    return cmd_texts(args.run_dirs, args.floor)


if __name__ == "__main__":
    raise SystemExit(main())
