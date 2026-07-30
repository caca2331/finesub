"""Score 断句 quality against the project's own bands (docs/segment_split.md).

Ideal 5-20 weighted chars / 1.2-4.5 s; acceptable 3-36 chars / 0.6-8.0 s. The doc also says
cuts should land on VAD-certified silence and that sentence punctuation is the strongest
non-VAD cut signal — both are counted here, because an ASR that emits punctuation hands the
splitter cut points the other one simply does not have.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from subtitle_metrics import weighted_char_count  # noqa: E402
from utils.text import punct_class  # noqa: E402

CHAR_IDEAL = (5.0, 20.0)
CHAR_OK = (3.0, 36.0)
DUR_IDEAL = (1.2, 4.5)
DUR_OK = (0.6, 8.0)


def band(value, lo, hi):
    return lo <= value <= hi


HAS_KANJI = __import__("re").compile(r"[一-鿿]")


def analyse(path: str, intervals, label: str, lexical_only: bool = False) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = [s for s in data["segments"] if str(s.get("text") or "").strip()]
    if lexical_only:
        # Whisper emits a lot of tiny interjection cues that sit inside the ideal char band
        # for free. Those are dropped downstream anyway, so scoring with them in flatters it.
        segs = [s for s in segs if HAS_KANJI.search(s["text"])]
    starts = [a for a, _ in intervals]

    chars = [weighted_char_count(s["text"]) for s in segs]
    durs = [s["end"] - s["start"] for s in segs]

    on_vad = 0
    for s in segs[1:]:
        if any(abs(s["start"] - a) <= 0.25 for a in starts):
            on_vad += 1

    ends_punct = sum(1 for s in segs if punct_class(s["text"].strip()[-1:]) in {"sentence", "clause"})

    return {
        "label": label,
        "cues": len(segs),
        "char_ideal": 100 * sum(band(c, *CHAR_IDEAL) for c in chars) / len(segs),
        "char_ok": 100 * sum(band(c, *CHAR_OK) for c in chars) / len(segs),
        "char_over": 100 * sum(c > CHAR_OK[1] for c in chars) / len(segs),
        "char_under": 100 * sum(c < CHAR_OK[0] for c in chars) / len(segs),
        "dur_ideal": 100 * sum(band(d, *DUR_IDEAL) for d in durs) / len(segs),
        "dur_ok": 100 * sum(band(d, *DUR_OK) for d in durs) / len(segs),
        "dur_over": 100 * sum(d > DUR_OK[1] for d in durs) / len(segs),
        "dur_under": 100 * sum(d < DUR_OK[0] for d in durs) / len(segs),
        "median_chars": sorted(chars)[len(chars) // 2],
        "median_dur": round(sorted(durs)[len(durs) // 2], 2),
        "starts_on_vad": 100 * on_vad / max(1, len(segs) - 1),
        "ends_on_punct": 100 * ends_punct / len(segs),
        "total_chars": round(sum(chars)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad", required=True)
    ap.add_argument("--arm", action="append", required=True, help="LABEL=path/to/aligned.json")
    ap.add_argument("--lexical-only", action="store_true", help="score only cues containing kanji")
    args = ap.parse_args()

    vad = json.loads(Path(args.vad).read_text(encoding="utf-8"))
    intervals = [(float(a), float(b)) for a, b in vad["speech"]]

    rows = [
        analyse(spec.split("=", 1)[1], intervals, spec.split("=", 1)[0], args.lexical_only)
        for spec in args.arm
    ]

    head = (
        f"{'arm':16s} {'cues':>5s} {'chars med':>9s} {'ideal%':>7s} {'ok%':>6s} {'over%':>6s} "
        f"{'dur med':>8s} {'ideal%':>7s} {'ok%':>6s} {'over%':>6s} {'cut@VAD%':>9s} {'end.punct%':>11s}"
    )
    print(head)
    for r in rows:
        print(
            f"{r['label']:16s} {r['cues']:5d} {r['median_chars']:9.1f} {r['char_ideal']:7.0f} "
            f"{r['char_ok']:6.0f} {r['char_over']:6.0f} {r['median_dur']:8.2f} {r['dur_ideal']:7.0f} "
            f"{r['dur_ok']:6.0f} {r['dur_over']:6.0f} {r['starts_on_vad']:9.0f} {r['ends_on_punct']:11.0f}"
        )


if __name__ == "__main__":
    main()
