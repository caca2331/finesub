"""agy hands the model only part of a long audio attachment wrapped as a single-frame mp4 (probes 2026-10-24/25: of a
7, 9.5, 10, 12 and 20-minute file the model hears the first 5, 5, 5, 10 and 15 minutes -- the last multiple of PART
below the length; a real 0.25 fps video track gets the whole 20 minutes). Every audio judge window longer than PART has
rows nobody heard: a natural control inside each run (runs in `HEARD_WHOLE` heard everything and are left out). Wrong moves and fixes of the audio runs, split by whether the
row's audio fell inside the heard part of the window's attachment, and by BIN-second place in the attachment.

    python tools/diarization_explore/judge_heard_split.py

A row's place in the attachment: the window's clip (`llm/clips/speaker-judge-<window>.aac`) is the samples, then the
window's recording; the recording is its last part, so a row sits at clip length - (recording end - row start), with the
recording ending at the window's last row end plus `prompt.EDGE_PAD` (reconstructed, not read from the prompt: when the
recording also covers context rows after the window's last row, a row is placed that much too late, so a few heard
rows near the cut count as unheard). Unheard rows are the tail of long windows: the silence runs, which hear nothing,
show how much of a heard / unheard gap is place in the window rather than hearing.
"""
from __future__ import annotations

import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import judge_moves_readout as M  # noqa: E402

from finesub.llm.stages.speaker_judge import prompt as P  # noqa: E402

#: the server's part length, seconds
PART = 300.0
#: width of the finer bins, seconds into the attachment
BIN = 300


def seconds(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout.strip()
    return float(out)


def heard(length):
    """Seconds of a single-frame attachment of this length the model hears."""
    return length if length <= PART else math.ceil(length / PART - 1) * PART


def main():
    wins, results = M.load(None)
    tally = defaultdict(Counter)
    for (name, key), (rows, _, payload) in results.items():
        folder, _, media, *_ = M.RUNS[name]
        if media != "audio" or name in M.HEARD_WHOLE:
            continue
        win = wins[key]
        index = {k: i for i, k in enumerate(win.ids)}
        place = {}
        for window in payload["windows"]:
            clip = M.SPK / folder / key / "llm" / "clips" / f"speaker-judge-{window['id']}.aac"
            if not clip.exists():
                continue
            length = seconds(clip)
            cap = heard(length)
            lo, hi = index[str(window["first"])], index[str(window["last"])]
            end = float(win.segs[hi]["end"]) + P.EDGE_PAD
            for i in range(lo, hi + 1):
                place[i] = (length - (end - float(win.segs[i]["start"])), cap)
        group = f"{M.RUNS[name][1]} {'silence' if 'silence' in name else 'audio'}"
        for i in win.scored:
            if i not in place:
                continue
            at, cap = place[i]
            part = "heard" if at < cap else "unheard"
            low = int(at // BIN) * BIN // 60
            for c in (tally[group, part], tally[group, f"{low:02d}-{low + BIN // 60:02d} min"]):
                count(c, win, rows, i)
    for (group, part), c in sorted(tally.items()):
        print(f"{group:13} {part:10} rows {c['rows']:5} | wrong {c['wrong']:3}/{c['a_right']:4} ({100 * c['wrong'] / max(1, c['a_right']):.1f}%)"
              f" cross-sex {c['cross']:3} ({100 * c['cross'] / max(1, c['a_right']):.2f}%) | fix {c['fix']:3}/{c['a_wrong']:4} ({100 * c['fix'] / max(1, c['a_wrong']):.1f}%)")


def count(c, win, rows, i):
    """Add row i to the tally c."""
    c["rows"] += 1
    a = win.right(win.acoustic, win.amap, i)
    c["a_right"] += a
    c["a_wrong"] += not a
    kind = rows.get(i, {}).get("kind")
    if kind in ("wrong", "fix"):
        c[kind] += 1
        c["cross"] += rows[i]["obvious"]


if __name__ == "__main__":
    main()
