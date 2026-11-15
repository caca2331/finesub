"""D3 (docs/plans/argon-audio-judge-plan.md): where the split arm's cuts fell, and whether the judge did worse near them.

    python tools/diarization_explore/split_cuts_readout.py

Recomputes each judge clip's cuts with the trial driver's own rule (`split_audio_for_agy` of `b3355c57`: equal-length
targets moved to the quietest 0.2 s within 30 s), places them against the judge prompt's rows (clip seconds), and
counts the judge's row outcomes on rows within NEAR seconds of a cut, split runs against bare runs on the same rows.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as M  # noqa: E402

def _trial_ffmpeg():
    """The trial driver's ffmpeg module (finesub is already this checkout's, so load the file by path)."""
    import importlib.util
    path = M.S.r9.MAIN / "tmp" / "spk-merge" / "accept" / "code-b3355c57" / "src" / "finesub" / "media" / "ffmpeg.py"
    spec = importlib.util.spec_from_file_location("trial_ffmpeg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


F = _trial_ffmpeg()

ACCEPT = M.S.r9.MAIN / "tmp" / "spk-merge" / "accept"
ROW = re.compile(r"^▶ q(\d+) ([\d.]+)\|([\d.]+)\|")
NEAR = 3.0
#: (key, judge unit) -> (a run holding that clip, split runs, bare runs)
CLIPS = {
    ("hina-22-c0102", "0001"): ("audio-rsplit1", ("38F split r1", "38F split r2", "38F split r3", "38F ovl r1"),
                                ("38F bare r2", "38F bare r3", "38F bare r4")),
    ("hina-22-c0102", "0002"): ("audio-rsplit1", ("38F split r1", "38F split r2", "38F split r3", "38F ovl r1"),
                                ("38F bare r2", "38F bare r3", "38F bare r4")),
    ("hina-46-c0001", "0001"): ("audio-rsplit1", ("38F split r1", "38F split r2", "38F split r3", "38F ovl r1"),
                                ("38F bare r2", "38F bare r3", "38F bare r4")),
    ("hina-46-c0001", "0002"): ("audio-rsplit1", ("38F split r1", "38F split r2", "38F split r3", "38F ovl r1"),
                                ("38F bare r2", "38F bare r3", "38F bare r4")),
    ("hina-17-c0002", "0001"): ("audio-rlsplit1", ("38F lsplit1",), ("38F lbare1",)),
    ("hina-17-c0002", "0002"): ("audio-rlsplit1", ("38F lsplit1",), ("38F lbare1",)),
    ("manosaba-b-c0001", "0001"): ("audio-rlsplit1", ("38F lsplit1",), ("38F lbare1",)),
}


def rows_of(run, key, unit):
    """id -> (start, end) in clip seconds, for the rows the judge was asked about."""
    ex = next((ACCEPT / run / key / "llm" / "exchanges").glob(f"*-speaker_judge-{unit}-attempt*.md"))
    text = ex.read_text(encoding="utf-8").split("## 模型响应", 1)[0]
    return {m.group(1): (float(m.group(2)), float(m.group(2)) + float(m.group(3)))
            for m in map(ROW.match, text.splitlines()) if m}


def main():
    import numpy as np
    wins, results = M.load(None)
    ffmpeg = F.resolve_ffmpeg()
    near_tally, far_tally = Counter(), Counter()
    for (key, unit), (run, splits, bares) in CLIPS.items():
        clip = ACCEPT / run / key / "llm" / "clips" / f"speaker-judge-{unit}.aac"
        pcm = F._decode_pcm(ffmpeg, clip)
        cuts = F._quiet_cuts(pcm, F.AGY_AUDIO_PIECE_SECONDS, F.AGY_AUDIO_CUT_SEARCH_SECONDS)
        rows = rows_of(run, key, unit)
        loud = 10 * np.log10(np.mean(np.square(pcm)) + 1e-9)
        print(f"=== {key} judge {unit}: {len(pcm) / 16000:.0f} s, {len(rows)} rows, cuts {[round(c, 1) for c in cuts]}")
        near_ids = set()
        for c in cuts:
            inside = [i for i, (s, e) in rows.items() if s < c < e]
            edge = min(min(abs(c - s), abs(c - e)) for s, e in rows.values())
            seg = pcm[int((c - 0.1) * 16000): int((c + 0.1) * 16000)]
            db = 10 * np.log10(np.mean(np.square(seg)) + 1e-9) - loud
            near = [i for i, (s, e) in rows.items() if s - NEAR <= c <= e + NEAR]
            near_ids |= set(near)
            text = {i: wins[key].segs[wins[key].ids.index(i)]["text"] for i in inside} if key in wins else {}
            print(f"  cut {c:7.1f}s  {db:+5.1f} dB vs clip  nearest row edge {edge:4.2f}s  "
                  f"inside rows {inside or '-'} {text or ''}  rows within {NEAR:.0f}s {near}")
        if key not in wins:
            continue
        index = {k: i for i, k in enumerate(wins[key].ids)}
        for arm, names in (("split", splits), ("bare", bares)):
            for name in names:
                got = results.get((name, key), (None,))[0]
                if not got:
                    continue
                for rid in rows:
                    kind = got.get(index[rid], {}).get("kind")
                    if kind:
                        (near_tally if rid in near_ids else far_tally)[arm, kind] += 1
                for rid in near_ids:
                    kind = got.get(index[rid], {}).get("kind")
                    if kind in ("wrong", "fix"):
                        print(f"    {name}: q{rid} {kind}")
    print(f"\nrow outcomes summed over runs (rows within {NEAR:.0f} s of a cut vs the rest):")
    for arm in ("split", "bare"):
        print(f"  {arm:5}: near {dict((k, v) for (a, k), v in near_tally.items() if a == arm)}"
              f" | rest {dict((k, v) for (a, k), v in far_tally.items() if a == arm)}")


if __name__ == "__main__":
    main()
