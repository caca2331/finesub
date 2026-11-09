"""Round 9's window judges re-scored with the move gates F1 / F1b (docs/plans/argon-audio-judge-plan.md 「改法」).

    python tools/diarization_explore/judge_gate_r9.py

Round 9 (agy Gemini 3.7 Flash, raw rows, audio and text) is the only audio judge with row truth on the merged-track
windows (manosaba-b, hina-71), where the judge earns its keep; its frozen answers are under
data/diarization/speaker-llm-runs-2026-10/. Its prompt showed the same track cells as the products' acoustic stable
(checked row by row, 1850 rows), so production's `cell_tracks` stands for what it saw. Scored like
`judge_moves_readout.py`: round 9's scoring windows, many-to-one mapping per labelling, one-person main rows;
`right` is the row accuracy after the judge, mapping shifts included.

- judge: every answer as given;
- F1: a move onto a track outside the row's cell is not taken;
- F1b: the same on short rows only (`judge_moves_readout.short`);
- F1c: the same on isolated moves only (neither neighbouring row answered the same track);
- fused (P9 brainstorm, `p9_gates.py`'s textbase-incell): the text arm's answers, plus the audio arm's moves onto a
  track in the row's cell on rows the text arm kept.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import judge_moves_readout as M  # noqa: E402
import row_truth_score as S  # noqa: E402

from finesub.llm.stages.speaker_judge.prompt import cell_tracks, rows_of  # noqa: E402

R9 = S.r9.MAIN / "data" / "diarization" / "speaker-llm-runs-2026-10" / "tmp" / "diarization-retrack" / "r9" / "pass1"
MERGED = ("manosaba-b", "hina-71")
ARMS = ("audio-raw", "text-raw")
GATES = ("judge", "F1", "F1b", "F1c")
GATE_OF = {"judge": None, "F1": "cell", "F1b": "short", "F1c": "isolated"}


def cells_after(answer, acoustic, rows, ident, gate):
    cells = dict(acoustic)
    tracks = {int(q[1:]): int(str(a["track"])[1:]) for q, a in answer.items() if str(a.get("track", "")).startswith("s")}
    for q, a in answer.items():
        i = int(q[1:])
        track = str(a.get("track", "keep"))
        if track in ("keep", "abstain") or i >= len(rows):
            continue
        if track.startswith("s"):
            if M.gated(GATE_OF[gate], rows, i, int(track[1:]), tracks):
                continue
            cells[i] = [track]
        elif track.startswith("@"):
            cells[i] = [f"@{ident}/{track}"]
    return cells


def fused(text, audio, rows):
    """The text arm's answers, plus the audio arm's moves onto a track in the cell on rows the text arm kept."""
    out = dict(text)
    for q, a in audio.items():
        i, track = int(q[1:]), str(a.get("track", "keep"))
        if str(text.get(q, {}).get("track", "keep")) not in ("keep", "abstain") or i >= len(rows):
            continue
        if track.startswith("s") and int(track[1:]) in cell_tracks(rows[i], marks=True):
            out[q] = a
    return out


def main():
    totals = defaultdict(Counter)
    for window in S.windows("score"):
        segs = S.stable(window.ident)
        rows = rows_of(segs)
        acoustic = M.acoustic_cells(segs)
        amap = S.fit(window, acoustic)
        scored = [i for i in window.rows
                  if window.truth[i].cls == "one" and window.truth[i].prio == "main" and window.truth[i].person]
        group = "merged" if window.ident in MERGED else "other"
        answers = {}
        for arm in ARMS:
            path = next(R9.joinpath(arm).glob(f"{window.ident}-*.json"), None)
            if path is not None:
                answers[arm] = json.loads(path.read_text(encoding="utf-8"))["answer"]
        if len(answers) == len(ARMS):
            answers["fused"] = fused(answers["text-raw"], answers["audio-raw"], rows)
        for arm, answer in answers.items():
            for gate in GATES if arm == "audio-raw" else ("judge",):
                cells = cells_after(answer, acoustic, rows, window.ident, gate)
                jmap = S.fit(window, cells)
                c = totals[arm, gate, group]
                for i in scored:
                    a = bool(acoustic.get(i)) and amap.get(acoustic[i][0]) == window.truth[i].person
                    j = bool(cells.get(i)) and jmap.get(cells[i][0]) == window.truth[i].person
                    c.update(scored=1, aright=a, right=j)
                    if (cells.get(i) or [None])[0] != (acoustic.get(i) or [None])[0]:
                        c["wrong" if a and not j else "fix" if j and not a else "other"] += 1
    for arm in (*ARMS, "fused"):
        for gate in GATES if arm == "audio-raw" else ("judge",):
            parts = []
            for group in ("merged", "other"):
                c = totals[arm, gate, group]
                parts.append(f"{group}: wrong {c['wrong']:3} fix {c['fix']:3} net {c['fix'] - c['wrong']:+4}"
                             f" right {100 * c['right'] / c['scored']:.1f}% (acoustic {100 * c['aright'] / c['scored']:.1f}%,"
                             f" n {c['scored']})")
            print(f"{arm:9} {gate:5} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
