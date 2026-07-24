"""Offline DP segment-split explorer (docs/segment_split.md).

Thin wrapper over the production splitter ``src/segment_split.py``: reads an
existing aligned/stable JSON plus a cached VAD interval list (computed once
from the vocal audio, no ASR rerun), runs the DP over every segment with
CLI-tunable scoring constants, and reports proposed splits, before/after
stats, and an optional SRT for audition.

Usage:
  python -m tools.split_explorer out/yui-exp/yui-cov2-stable.json \
      --audio out/yui-exp/yui-vocal.flac [--srt out.srt] [--seg N] [params]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from subtitle_metrics import weighted_char_count  # noqa: E402
import segment_split as sp  # noqa: E402


def load_intervals(audio: Path, cache: Path) -> list[tuple[float, float]]:
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        raw_intervals = data.get("intervals")
        if raw_intervals is None:
            raw_intervals = [
                [segment["start"], segment["end"]]
                for segment in data.get("segments", [])
            ]
        return [tuple(x) for x in raw_intervals]
    print(f"Info: computing VAD intervals from {audio} (one-time, cached)", file=sys.stderr)
    import vad_asr

    _raw, segments, *_rest = vad_asr.load_and_detect_segments(audio)
    intervals = [(float(s["start"]), float(s["end"])) for s in segments]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "audio": str(audio),
                "intervals": intervals,
                # Keep the cache directly consumable by asr_align.py.  This
                # makes an explorer dataset reproducible without rerunning
                # VAD or maintaining a second interval artifact.
                "segments": [
                    {"start": start, "end": end} for start, end in intervals
                ],
            }
        ),
        encoding="utf-8",
    )
    print(f"Info: cached {len(intervals)} intervals -> {cache}", file=sys.stderr)
    return intervals


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stable_json", type=Path)
    ap.add_argument("--audio", type=Path, help="vocal audio (for one-time VAD cache)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="VAD interval cache path (default: <audio>.vadcache.json)")
    ap.add_argument("--srt", type=Path, default=None, help="write split SRT here")
    ap.add_argument("--seg", type=int, default=None,
                    help="verbose: dump all boundary scores for segment #N")
    ap.add_argument("--max-report", type=int, default=30)
    for f in fields(sp.SplitParams):
        ap.add_argument(f"--{f.name.replace('_', '-')}", type=float,
                        default=getattr(sp.DEFAULT_SPLIT_PARAMS, f.name))
    args = ap.parse_args()

    params = sp.SplitParams(
        **{f.name: getattr(args, f.name) for f in fields(sp.SplitParams)}
    )

    cache = args.cache
    if cache is None:
        if args.audio is None:
            ap.error("--audio or --cache required")
        cache = args.audio.with_suffix(args.audio.suffix + ".vadcache.json")
    interval_spans = load_intervals(args.audio, cache)
    zones = sp.build_zones(interval_spans)

    data = json.loads(args.stable_json.read_text(encoding="utf-8"))
    segments = [s for s in data["segments"] if s.get("text") and s.get("words")]

    before_stats, after_stats = [], []
    out_entries = []  # (start, end, text)
    reported = 0
    n_split = 0

    for seg_idx, seg in enumerate(segments):
        adj = sp.adjust_words(seg["words"], interval_spans, zones)
        boundaries = sp.score_boundaries(adj, interval_spans, params)
        result = sp.dp_split(adj, boundaries, params)

        d0 = float(seg["end"]) - float(seg["start"])
        c0 = weighted_char_count(str(seg["text"]))
        before_stats.append((d0, c0))

        if args.seg is not None and seg_idx == args.seg:
            print(f"=== segment #{seg_idx} {seg['start']:.2f}-{seg['end']:.2f} "
                  f"d={d0:.2f}s c={c0:g} ===")
            for k, bd in enumerate(boundaries):
                cut_here = any(a == k + 1 for a, _ in result.pieces)
                mark = " <cut>" if cut_here else ""
                nogap = " nogap" if bd.no_gap else ""
                word_l, word_r = adj[k], adj[k + 1]
                print(f"  [{k}] {word_l.text!r}|{word_r.text!r} "
                      f"case={word_l.case}/{word_r.case} g={bd.g:.2f} T={bd.t:.1f} "
                      f"B={'INF' if bd.banned else f'{bd.b:.2f}'}{nogap}{mark}")

        seg_start, seg_end = float(seg["start"]), float(seg["end"])
        if len(result.pieces) <= 1:
            # Production split_segments passes a no-split source segment
            # through bit-identically.  Virtual gap-word adjustment must not
            # leak into explorer stats or SRT output in this branch.
            pieces_out = [(seg_start, seg_end, str(seg["text"]))]
            after_stats.append((d0, c0))
        else:
            pieces_out = []
            for a, b in result.pieces:
                ps = min(max(adj[a].start, seg_start), seg_end)
                pe = min(max(adj[b - 1].end, ps), seg_end)
                txt = sp.piece_text(adj, a, b)
                pieces_out.append((ps, pe, txt))
                after_stats.append((pe - ps, weighted_char_count(txt)))
        # mirror to_srt: zero/negative-duration entries never reach the SRT
        out_entries.extend(x for x in pieces_out if x[1] > x[0])

        if len(result.pieces) > 1:
            n_split += 1
            if reported < args.max_report:
                reported += 1
                gain = result.no_split - result.total
                print(f"--- #{seg_idx} {seg['start']:8.2f}-{seg['end']:8.2f} "
                      f"d={d0:5.2f}s c={c0:4g} -> {len(result.pieces)} pieces "
                      f"(score {result.no_split:.2f} -> {result.total:.2f}, gain {gain:.2f})")
                for i, (ps, pe, txt) in enumerate(pieces_out):
                    print(f"    {ps:8.2f}-{pe:8.2f} d={pe-ps:5.2f} "
                          f"c={weighted_char_count(txt):4g}  {txt[:44]}")
                    if i < len(pieces_out) - 1:
                        cut_k = result.pieces[i][1] - 1
                        bd = boundaries[cut_k]
                        print(f"      cut: g={bd.g:.2f} T={bd.t:.1f} B={bd.b:.2f}")

    def summarize(label, stats):
        durs = [d for d, _ in stats]
        chars = [c for _, c in stats]
        print(f"{label}: n={len(stats)}  dur med={statistics.median(durs):.2f} "
              f">4.5s={sum(1 for x in durs if x > 4.5)} >8s={sum(1 for x in durs if x > 8)} "
              f"<0.6s={sum(1 for x in durs if x < 0.6)}  "
              f"chars med={statistics.median(chars):.0f} "
              f">20c={sum(1 for x in chars if x > 20)} >36c={sum(1 for x in chars if x > 36)} "
              f"<3c={sum(1 for x in chars if x < 3)}")

    print()
    print(f"segments split: {n_split}/{len(segments)}")
    summarize("before", before_stats)
    summarize("after ", after_stats)

    if args.srt:
        lines = []
        for i, (ps, pe, txt) in enumerate(out_entries, start=1):
            lines.append(f"{i}\n{fmt_ts(ps)} --> {fmt_ts(pe)}\n{txt}\n")
        args.srt.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {args.srt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
