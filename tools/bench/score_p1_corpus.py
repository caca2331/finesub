"""The VAD-failover predicate on genuinely sparse recordings.

Stage two of P1 needs a false-trigger rate, and the production corpus cannot
supply one: 9.45h of talk streams whose longest silence is 14.7s
(`bench-baselines.md` 17.2). This scores the predicate on material picked for
the property that corpus lacks -- long recordings that are almost entirely
non-speech -- split into two arms that must come out differently:

* **arm A** has no speech at all. Whatever coverage the energy VAD reports here
  is the *residue floor*: what separator output measures when there is nothing
  to find. The threshold has to sit below it or the predicate cannot tell a
  broken detector from a quiet room.
* **arm B** has real speech, very sparse. The predicate must stay quiet:
  firing here is the false trigger that would send an auto-fallback off on a
  recording whose VAD was right.

⚠ Scope: this is ambient/meditation material, not livestream capture. Its noise
floor is not the owner's production distribution, so these numbers bound the
predicate's *discriminative power*, not its rate on real work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: The predicate this scored lived in `speech/preprocessing/vad_failover.py`
#: and was REMOVED on 2026-08-30 (`bench-baselines.md` 17.12: it could only
#: fire when the whole file was under 1% speech -- a failure the user already
#: sees as a nearly empty subtitle file -- and the pipeline exposes no VAD knob
#: to act on it with). Its conditions are inlined here so 17.5 stays
#: reproducible; they are a historical record, not live behaviour.
MIN_AUDIO_SEC = 120.0
MIN_COVERAGE = 0.01
VERY_LONG_MULTIPLE = 4.0
FEW_INTERVAL_COUNT = 2
FEW_INTERVAL_MAX_COVERAGE = 0.10


def inspect(intervals, audio_duration):
    """(coverage, interval count, fires) -- the removed predicate's verdict."""

    speech = 0.0
    count = 0
    for interval in intervals:
        start, end = float(interval["start"]), float(interval["end"])
        count += 1
        if end > start:
            speech += end - start
    coverage = speech / audio_duration if audio_duration > 0 else 0.0
    if audio_duration < MIN_AUDIO_SEC:
        return coverage, count, False
    if count == 0 or coverage < MIN_COVERAGE:
        return coverage, count, True
    fires = (count <= FEW_INTERVAL_COUNT
             and audio_duration >= VERY_LONG_MULTIPLE * MIN_AUDIO_SEC
             and coverage < FEW_INTERVAL_MAX_COVERAGE)
    return coverage, count, fires


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="directory of pipeline run dirs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("p1_corpus.jsonl"),
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    arms = {}
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            arms[row["id"]] = row

    rows = []
    for vad_path in sorted(args.out.glob("*/*-vad.json")):
        run = vad_path.parent.name
        meta = arms.get(run, {})
        payload = json.loads(vad_path.read_text(encoding="utf-8"))
        duration = float(payload["audio_duration"])
        # `raw_segments`, because that is what the stage handed the predicate.
        coverage, interval_count, fires = inspect(payload["raw_segments"], duration)
        rows.append(
            {
                "run": run,
                "arm": meta.get("arm", "?"),
                "texture": meta.get("texture", ""),
                "min": duration / 60,
                "coverage": coverage,
                "intervals": interval_count,
                "suspect": fires,
            }
        )

    if not rows:
        print(f"FAIL: no <stem>-vad.json under {args.out}")
        return 2

    print(f"threshold: coverage < {MIN_COVERAGE * 100:.0f}%, or "
          f"<= {FEW_INTERVAL_COUNT} intervals with low coverage "
          f"(predicate removed 2026-08-30; replayed here)\n")
    print("arm run              texture                min   coverage  intervals  fires")
    for row in sorted(rows, key=lambda r: (r["arm"], r["coverage"])):
        print(
            f" {row['arm']}  {row['run']:<16} {row['texture']:<22}"
            f"{row['min']:>5.1f}  {row['coverage'] * 100:>7.2f}%  {row['intervals']:>8}"
            f"   {'YES' if row['suspect'] else 'no'}"
        )

    for arm, label in (("A", "no speech at all"), ("B", "sparse real speech")):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        fired = sum(1 for r in arm_rows if r["suspect"])
        floor = min(r["coverage"] for r in arm_rows)
        print(
            f"\narm {arm} ({label}): fires {fired}/{len(arm_rows)}, "
            f"lowest coverage {floor * 100:.2f}% "
            f"({floor / MIN_COVERAGE:.1f}x the threshold)"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
