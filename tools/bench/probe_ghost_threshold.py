"""Are the ghost-drop thresholds right, or just fitted to the clip they were
tuned on?

`drop_ghost_duplicate_segments` drops a segment on three conditions, and the
three constants in them (`GHOST_SEGMENT_MAX_SPAN_SEC`, `GHOST_SEGMENT_MIN_CHARS`,
`GHOST_SEGMENT_CONTEXT_SEC`) were hand-fitted on the 2026-07 clips (owner,
2026-08-31). "It looks good on that data" is therefore circular. This probe
answers the question on held-out artifacts, with an oracle the rule never had.

**The oracle.** The rule's own promise is "removing this cannot lose real
speech". The Qwen referee answers exactly that, by listening to the span rather
than pattern-matching neighbouring text. So every candidate drop can be checked:
if the referee hears speech there, the drop is wrong.

⚠ This needs `end`, which the drop telemetry only started recording on
2026-09-01 (`bench-baselines.md` 20.6). On older artifacts a dropped ghost's
span is unrecoverable and the widened window bleeds neighbouring speech into
the clip, which is what made 20.3's criterion (1) unanswerable.

**The criterion is pre-registered** (`bench-baselines.md` 20.7) and this file
does not get to move it: loosening any threshold is REFUSED if a single one of
the newly dropped segments has the referee hearing speech. Tightening is judged
on the drops it gives up.

    python -m tools.bench.probe_ghost_threshold verify --root out/ts-baseline-...
    python -m tools.bench.probe_ghost_threshold sweep  --root out/ts-baseline-...
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterator, List, Sequence, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from finesub.speech.recognition import segments as segment_ops
from finesub.text import normalized_compact


def _clips(root: pathlib.Path) -> Iterator[Tuple[pathlib.Path, Dict[str, Any]]]:
    for path in sorted(root.rglob("*-aligned.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _vocal_for(path: pathlib.Path) -> pathlib.Path | None:
    for suffix in (".ogg", ".flac", ".wav"):
        candidate = path.parent / f"{path.parent.name}-vocal{suffix}"
        if candidate.exists():
            return candidate
    return None


def _referee_hears(cases: Sequence[Dict[str, Any]]) -> List[str]:
    """What the referee hears in each span, reading the same clip production
    reads (`SEGMENT_PAD_SEC` either side of the real span)."""

    from finesub.speech.verification import qwen_referee

    referee = qwen_referee.QwenReferee(device="cuda")
    heard: List[str] = []
    reader = None
    current = None
    try:
        for case in cases:
            if case["audio"] != current:
                reader = qwen_referee._SpanReader(str(case["audio"]))
                current = case["audio"]
            clip = reader.read(
                case["start"] - qwen_referee.SEGMENT_PAD_SEC,
                case["end"] + qwen_referee.SEGMENT_PAD_SEC,
            )
            answer = referee.transcribe_batch([clip])[0]
            heard.append(str(answer[0] if isinstance(answer, tuple) else answer))
    finally:
        referee.close()
    return heard


def cmd_verify(root: pathlib.Path) -> int:
    """Criterion (1) at last: is every drop the CURRENT rule makes correct?"""

    cases = []
    for path, data in _clips(root):
        audio = _vocal_for(path)
        for record in data["metadata"]["asr_align"].get(
            "ghost_duplicate_segments_dropped"
        ) or []:
            if not isinstance(record, dict) or "end" not in record:
                print(f"  skip {path.parent.name}: telemetry has no `end` "
                      f"(pre-2026-09-01 artifact)")
                continue
            if audio is None:
                print(f"  skip {path.parent.name}: no vocal track")
                continue
            cases.append({"clip": path.parent.name, "audio": audio,
                          "start": float(record["start"]),
                          "end": float(record["end"]),
                          "text": str(record.get("text") or "")})

    print(f"现行规则丢掉的段: {len(cases)}\n")
    if not cases:
        return 0
    heard = _referee_hears(cases)
    wrong = 0
    print(f"{'clip':16s} {'t':>9} {'dur':>6}  被丢文本 -> 裁判听到")
    for case, text in zip(cases, heard):
        bad = bool(normalized_compact(text))
        wrong += bad
        mark = "  ⚠ 裁判听到语音" if bad else ""
        print(f"{case['clip']:16s} {case['start']:9.2f} "
              f"{case['end'] - case['start']:6.2f}  "
              f"{case['text'][:20]!r} -> {text[:24]!r}{mark}")
    print(f"\n错杀 {wrong} / {len(cases)}")
    return 0


def _candidates(
    data: Dict[str, Any], *, max_span: float, min_chars: int, context: float
) -> List[int]:
    """Indices the rule drops under the given thresholds.

    ⚠ Drives the REAL `drop_ghost_duplicate_segments` with its module constants
    swapped, rather than reimplementing the decision. A parameterised copy is
    how a sweep ends up measuring a rule that is not the one shipping -- and
    the copy stays right only until someone edits one of the two. The
    constants are read at call time, so swapping them is enough.

    The decoder-event condition is not in the grid and is never relaxed:
    without it the duplicate test matches real rapid repeats (twice-shouted
    calls, sung refrains), which is the whole reason it exists.
    """

    names = {
        "GHOST_SEGMENT_MAX_SPAN_SEC": max_span,
        "GHOST_SEGMENT_MIN_CHARS": min_chars,
        "GHOST_SEGMENT_CONTEXT_SEC": context,
    }
    saved = {name: getattr(segment_ops, name) for name in names}
    try:
        for name, value in names.items():
            setattr(segment_ops, name, value)
        _, dropped = segment_ops.drop_ghost_duplicate_segments(data["segments"])
    finally:
        for name, value in saved.items():
            setattr(segment_ops, name, value)
    return [int(record["index"]) for record in dropped]


def cmd_sweep(root: pathlib.Path) -> int:
    """Every candidate a LOOSER threshold reaches, judged by the referee once.

    ⚠ Only loosening. The artifacts are post-drop: what the shipped thresholds
    already removed is not in the file, so a tighter grid point would count
    from an already-filtered set and report a meaningless zero. Tightening is
    the `verify` mode's question -- "were the drops it already made right" --
    and the two together cover the space.
    """

    from finesub.speech.recognition.segments import (
        GHOST_SEGMENT_CONTEXT_SEC, GHOST_SEGMENT_MAX_SPAN_SEC,
        GHOST_SEGMENT_MIN_CHARS,
    )

    # Loosening only, current value first (see the docstring).
    grid = {
        "max_span": (GHOST_SEGMENT_MAX_SPAN_SEC, 0.2, 0.4),
        "min_chars": (GHOST_SEGMENT_MIN_CHARS, 1),
        "context": (GHOST_SEGMENT_CONTEXT_SEC, 6.0, 12.0),
    }
    #: The union of everything any grid point can select -- judged once, so the
    #: referee is not re-run per combination.
    widest = dict(max_span=max(grid["max_span"]), min_chars=min(grid["min_chars"]),
                  context=max(grid["context"]))
    print(f"现行阈值 max_span={GHOST_SEGMENT_MAX_SPAN_SEC} "
          f"min_chars={GHOST_SEGMENT_MIN_CHARS} "
          f"context={GHOST_SEGMENT_CONTEXT_SEC}；只往松的方向扫\n")

    universe: List[Dict[str, Any]] = []
    per_clip: List[Tuple[Dict[str, Any], pathlib.Path, List[int]]] = []
    for path, data in _clips(root):
        audio = _vocal_for(path)
        if audio is None:
            continue
        picked = _candidates(data, **widest)
        per_clip.append((data, path, picked))
        for index in picked:
            segment = data["segments"][index]
            universe.append({
                "clip": path.parent.name, "audio": audio, "index": index,
                "start": float(segment["start"]), "end": float(segment["end"]),
                "text": str(segment.get("text") or ""),
            })

    print(f"最宽档能选中的候选共 {len(universe)} 条，交给裁判判一次\n")
    if not universe:
        return 0
    heard = _referee_hears(universe)
    speech = {
        (case["clip"], case["index"]): bool(normalized_compact(text))
        for case, text in zip(universe, heard)
    }
    for case, text in zip(universe, heard):
        flag = "语音" if normalized_compact(text) else "静默"
        print(f"  {case['clip']:16s} t={case['start']:8.2f} "
              f"dur={case['end'] - case['start']:.2f} "
              f"{case['text'][:18]!r} -> {text[:20]!r}  [{flag}]")

    print(f"\n{'max_span':>9} {'min_chars':>10} {'context':>8} "
          f"{'丢弃':>6} {'其中裁判听到语音':>18}")
    base = (GHOST_SEGMENT_MAX_SPAN_SEC, GHOST_SEGMENT_MIN_CHARS,
            GHOST_SEGMENT_CONTEXT_SEC)
    for max_span in grid["max_span"]:
        for min_chars in grid["min_chars"]:
            for context in grid["context"]:
                total = wrong = 0
                for data, path, _ in per_clip:
                    for index in _candidates(
                        data, max_span=max_span, min_chars=min_chars,
                        context=context,
                    ):
                        total += 1
                        wrong += speech.get((path.parent.name, index), False)
                here = " <- 现行" if (max_span, min_chars, context) == base else ""
                verdict = "" if not wrong else "  ⚠ 否决"
                print(f"{max_span:9.2f} {min_chars:10d} {context:8.1f} "
                      f"{total:6d} {wrong:18d}{verdict}{here}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe_ghost_threshold")
    parser.add_argument("mode", choices=("verify", "sweep"))
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    return cmd_verify(root) if args.mode == "verify" else cmd_sweep(root)


if __name__ == "__main__":
    raise SystemExit(main())
