"""End-to-end check of a VAD change: what actually reached the subtitles.

The interval-level tables in FINDINGS say the VAD hands the ASR more audio in more
pieces. Neither is the deliverable. This asks the three questions that are:

  text        did anything appear or disappear, and is what appeared real speech
  segmenter   did 10% more intervals fragment the output (the splitter runs after,
              so more seams could mean more short segments)
  junk        did the extra audio buy hallucination, per asr-stabilize profile 0 --
              the pipeline's own verdict, not a metric invented here
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))


def load(path: Path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    segs = d["segments"] if isinstance(d, dict) else d
    meta = d.get("metadata", {}) if isinstance(d, dict) else {}
    return segs, meta


def text_of(segs) -> str:
    return "".join(str(s.get("text", "")).strip() for s in segs)


def describe(name: str, segs, meta):
    words = [w for s in segs for w in (s.get("words") or [])]
    durs = np.array([float(s["end"]) - float(s["start"]) for s in segs] or [0.0])
    chars = np.array([len(str(s.get("text", ""))) for s in segs])
    tags = Counter(t for s in segs for t in (s.get("tags") or []))
    print(f"--- {name}")
    print(f"    segments={len(segs)} words={len(words)} chars={int(chars.sum())}")
    print(f"    seg dur   med={np.median(durs):.2f} p10={np.quantile(durs, .1):.2f} "
          f"p90={np.quantile(durs, .9):.2f} max={durs.max():.2f} "
          f"under1s={int((durs < 1.0).sum())} under0.5s={int((durs < 0.5).sum())}")
    print(f"    chars/seg med={np.median(chars):.0f} p10={np.quantile(chars, .1):.0f} "
          f"p90={np.quantile(chars, .9):.0f} under5={int((chars < 5).sum())}")
    if tags:
        print(f"    tags: {dict(tags)}")
    sp = (meta.get("asr_align") or {}).get("segment_split") or {}
    if sp:
        print(f"    segment_split: {json.dumps(sp, ensure_ascii=False)[:160]}")
    tm = meta.get("timing") or (meta.get("asr_align") or {}).get("timing") or {}
    if tm:
        keep = {k: round(v, 1) for k, v in tm.items()
                if isinstance(v, (int, float)) and v > 1}
        print(f"    timing: {keep}")
    return {"segs": len(segs), "words": len(words), "chars": int(chars.sum())}


def diff_report(a_text: str, b_text: str, a_name: str, b_name: str, top: int = 12):
    sm = difflib.SequenceMatcher(None, a_text, b_text, autojunk=False)
    print(f"    whole-transcript similarity: {sm.ratio():.4f}")
    only_a, only_b = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace") and i2 > i1:
            only_a.append(a_text[i1:i2])
        if tag in ("insert", "replace") and j2 > j1:
            only_b.append(b_text[j1:j2])
    print(f"    only in {a_name}: {sum(len(x) for x in only_a)} chars "
          f"in {len(only_a)} runs")
    print(f"    only in {b_name}: {sum(len(x) for x in only_b)} chars "
          f"in {len(only_b)} runs")
    for label, runs in ((a_name, only_a), (b_name, only_b)):
        longest = sorted(runs, key=len, reverse=True)[:top]
        print(f"    longest only-in-{label}:")
        for r in longest:
            print(f"      {r[:60]}")


def word_diff(a_segs, b_segs, a_name: str, b_name: str, tol: float = 1.5):
    """Symmetric word-level diff: matched by text within `tol` seconds.

    The interval-level `lost` metric in FINDINGS asks whether a reference word sits
    inside non-speech. That is a proxy: `inserted_gap_parts` keeps up to 0.7 s of
    real audio after each interval, so the decoder often sees a word the interval
    boundaries appear to hide. This measures what actually came out.
    """
    def words(segs):
        out = []
        for s in segs:
            for w in (s.get("words") or []):
                t = str(w.get("word", "")).strip()
                if t:
                    out.append((float(w["start"]), t))
        return sorted(out)

    aw, bw = words(a_segs), words(b_segs)
    def unmatched(x, y):
        by_text = {}
        for t, w in y:
            by_text.setdefault(w, []).append(t)
        miss = []
        for t, w in x:
            cand = by_text.get(w)
            if not cand or min(abs(c - t) for c in cand) > tol:
                miss.append((t, w))
        return miss

    only_a = unmatched(aw, bw)
    only_b = unmatched(bw, aw)
    print(f"    words: {a_name}={len(aw)} {b_name}={len(bw)}")
    print(f"    in {a_name} only: {len(only_a)} ({len(only_a)/max(len(aw),1):.1%})")
    print(f"    in {b_name} only: {len(only_b)} ({len(only_b)/max(len(bw),1):.1%})")
    for label, miss in ((a_name, only_a), (b_name, only_b)):
        joined = "".join(w for _t, w in miss)
        print(f"    {label}-only text ({len(joined)} chars): {joined[:100]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--before-name", default="before")
    ap.add_argument("--after-name", default="after")
    ap.add_argument("--stabilize", action="store_true")
    ap.add_argument("--probe", nargs="*", default=[],
                    help="TIME:TEXT pairs to look for in both arms")
    args = ap.parse_args()

    pairs = [(args.before_name, Path(args.before)), (args.after_name, Path(args.after))]

    print("=== aligned (pre-stabilization) ===")
    for name, p in pairs:
        segs, meta = load(p)
        describe(name, segs, meta)
        print()

    a_segs, _ = load(pairs[0][1])
    b_segs, _ = load(pairs[1][1])
    print("=== transcript diff (aligned) ===")
    diff_report(text_of(a_segs), text_of(b_segs), args.before_name, args.after_name)
    print()
    print("=== word-level diff (what actually came out) ===")
    word_diff(a_segs, b_segs, args.before_name, args.after_name)
    print()

    if args.probe:
        print("=== targeted probe: words appendix E watched disappear ===")
        for spec in args.probe:
            t_str, needle = spec.split(":", 1)
            t = float(t_str)
            for name, segs in ((args.before_name, a_segs), (args.after_name, b_segs)):
                near = [s for s in segs
                        if float(s["start"]) - 3 <= t <= float(s["end"]) + 3]
                txt = "".join(str(s.get("text", "")) for s in near)
                print(f"    {t:>8.2f} {needle:<12} {name:<8} "
                      f"{'HIT ' if needle in txt else 'miss'} | {txt[:56]}")
        print()

    if args.stabilize:
        from asr_playground.speech.postprocessing.stabilization import stabilize_json_file

        print("=== after asr-stabilize profile 0 ===")
        stable = {}
        for name, p in pairs:
            outp, report = stabilize_json_file(
                p, output_path=p.with_name(p.stem.replace("-aligned", "") + "-stable.json"),
                profile=0)
            segs, meta = load(outp)
            words = [w for s in segs for w in (s.get("words") or [])]
            print(f"--- {name}")
            print(f"    segments {report.input_segments} -> {report.output_segments} "
                  f"(dropped {report.suspicious_segments_dropped} suspicious, "
                  f"{report.phrase_occurrences_removed} phrase occurrences)")
            print(f"    tags: {dict(report.tag_counts) or '-'}")
            print(f"    surviving: segments={len(segs)} words={len(words)} "
                  f"chars={sum(len(str(s.get('text', ''))) for s in segs)}")
            stable[name] = segs
            print()
        print("=== transcript diff (stabilized) ===")
        diff_report(text_of(stable[args.before_name]), text_of(stable[args.after_name]),
                    args.before_name, args.after_name)


if __name__ == "__main__":
    main()
