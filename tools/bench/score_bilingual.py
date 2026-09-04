"""Score a bilingual run against the span labels it was built from.

The ground truth here did not come from a detector -- each span's language is
the language of the asset it was cut from -- so this is the first thing in the
project that can say a language vote is *wrong* rather than merely *different
from the majority*.

Scoring is by time overlap, not by index: the pipeline runs its own VAD and
re-segments, so its segments never line up 1:1 with the spans that were
concatenated. Each output segment takes the label of the truth span it overlaps
most.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


#: Language labels the scorer will accept. A truth file that says something
#: else is a generation bug, not data: a ja+ja control was once built without
#: `--b-lang ja`, so half of it claimed "en" and any score computed from it was
#: meaningless. Refusing beats printing a number.
KNOWN_LANGUAGES = frozenset({"ja", "en", "de", "fr", "zh", "ko", "es", "it", "ru"})

#: An output segment must sit mostly inside one truth span to inherit its
#: label. A segment straddling a switch belongs to neither and would otherwise
#: be scored against whichever side it happened to touch first.
MIN_SPAN_SHARE = 0.6


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory holding aligned.json + truth")
    args = parser.parse_args()

    truth = json.loads((args.run / "bilingual-truth.json").read_text(encoding="utf-8"))
    payload = json.loads((args.run / "aligned.json").read_text(encoding="utf-8"))
    segments = payload["segments"]
    align = payload["metadata"]["asr_align"]

    labels = {str(span.get("language")) for span in truth}
    unknown = labels - KNOWN_LANGUAGES
    if unknown:
        print(f"REFUSED: truth carries unknown language label(s) {sorted(unknown)}")
        return 2
    missing = [span for span in truth if "source" not in span]
    if missing:
        print(f"REFUSED: {len(missing)} truth span(s) do not name their source")
        return 2

    matched = 0
    correct = 0
    skipped_straddling = 0
    confusion: collections.Counter = collections.Counter()
    by_truth: collections.Counter = collections.Counter()
    for segment in segments:
        start, end = float(segment.get("start", 0)), float(segment.get("end", 0))
        duration = max(1e-6, end - start)
        best, best_span = 0.0, None
        for span in truth:
            value = _overlap(start, end, span["start"], span["end"])
            if value > best:
                best, best_span = value, span
        if best_span is None or best <= 0:
            continue
        if best / duration < MIN_SPAN_SHARE:
            # Straddles a language switch: it belongs to neither side.
            skipped_straddling += 1
            continue
        matched += 1
        expected = best_span["language"]
        got = str(segment.get("lang"))
        by_truth[expected] += 1
        confusion[(expected, got)] += 1
        correct += expected == got

    print(f"=== {args.run.name} ===")
    print(f"output segments {len(segments)}   matched to a truth span {matched}")
    if skipped_straddling:
        print(
            f"  skipped {skipped_straddling} segment(s) straddling a switch "
            f"(<{MIN_SPAN_SHARE:.0%} inside one span)"
        )
    if matched:
        print(f"language correct {correct}/{matched} = {100 * correct / matched:.1f}%")
    print("\nconfusion (truth -> detected):")
    for (expected, got), n in sorted(confusion.items(), key=lambda kv: -kv[1]):
        flag = "" if expected == got else "   <-- WRONG"
        print(f"  {expected} -> {got:<6} {n:4d}   ({100 * n / by_truth[expected]:5.1f}% of {expected}){flag}")

    redecode = align.get("lang_redecode") or {}
    print(f"\nlang_redecode: triggers={redecode.get('triggers')} adopted={redecode.get('adopted')}")
    for event in redecode.get("events") or []:
        print(f"  {json.dumps(event, ensure_ascii=False)}")
    qwen = align.get("qwen_verify") or {}
    print(f"qwen_verify: {json.dumps(qwen, ensure_ascii=False)}")
    coverage = align.get("audio_coverage") or {}
    print(f"audio_coverage: {json.dumps(coverage, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
