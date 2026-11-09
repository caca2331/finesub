"""The speaker judge's row moves against the row truth (docs/plans/argon-audio-judge-plan.md 「指标」).

    python tools/diarization_explore/judge_moves_readout.py [--gate cell|short|isolated] [--profile] [--samples] [--mai2] [--rows out.csv]

Reads every run in RUNS that has a judge sidecar (`input/<key>-speaker-judge.json`, beside the stable it judged) and
scores the rows the judge moved: a moved row's new lead label against its acoustic one, each labelling mapped to
people by the row-truth rule (`row_truth_score.fit`: every label to the person with the most of its rows in the
window; a new `@` label on one row therefore always maps to that row's own person). Only rows the ruler marks one
person, main content, are scored. A run whose sidecar holds a window that is not `success` (still running, or given
up) is skipped for that file, with a warning: its rows would read as not moved.

- wrong move: acoustic right, judged wrong; fix: acoustic wrong, judged right; the rest of the moves are listed apart;
- obvious: a wrong move onto a person of the other sex (GENDER: the ruler's role names are the refined subtitles'
  roles, read against the judges' cast names). The first try (the new person absent from the 5 rows on either side)
  never held: in these four-voice talk shows everyone speaks within five rows. Sex is a rough stand-in for "a voice
  far off": 温迪 (村瀬歩, a high male voice) to 派蒙 counts as cross-sex though the two are close;
- in a run: a wrong move whose nearest scored rows before and after are the row's own person;
- audio only: an audio judge's wrong move that every text judge of the same shape (TEXT_OF) on that file got right;
- overlap: wrong moves shared by two audio runs over the wrong moves of either;
- gate (`--gate cell`): every readout as if the harness took a move only onto a track the row's cell already shows
  (`cell_tracks`: its tracks and a low mark's alternative) or onto a new `@` label, and kept the acoustic track
  otherwise -- candidate fix F1, designed after seeing these data; `--gate short` gates only short rows (under
  SHORT_SECONDS or at most SHORT_CHARS characters) -- F1b; `--gate isolated` gates only isolated moves (neither
  neighbouring row moved onto the same track) -- F1c, which keeps the run-level reassignments of merged tracks.
  `judge_gate_r9.py` scores them on round 9's windows. For the v3 runs (D8, D8b) the gate also counts a low mark's
  alternative, which their prompt may not show. Only moved rows are counted: a row the judge left can still turn
  right or wrong when the window's mapping is refitted, so a gated count may differ by one from a whole-window one;
- profile (`--profile`): wrong moves and fixes against every scored row -- row length, the cell (a low mark, more
  than one track, the new track outside the cell), minutes into the judge window's recording, and whether the new
  person speaks within NEARS seconds of the row (truth rows overlapping the row widened by that much) against a
  wrong person drawn by how much each speaks in the window. For a fix the new person is the row's own, who mostly
  speaks in the rows around it, so that column is no control for the wrong moves;
- samples (`--samples`): who speaks, by the truth, in each track's voice samples (production's `profiles`, rebuilt
  per judge window), and for the wrong moves onto an acoustic track, whether that track's samples hold the row's own
  person -- the judge hearing the right voice under the wrong track;
- MAI2 (`--mai2`): a second ruler for the held-out cross-sex rows, by sex only (MAI2's labels split by sex, not by
  person; MAI2_SEX), against its own disagreement on acoustic-right rows no audio judge moved, by row length.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import row_truth_score as S  # noqa: E402

from finesub.llm.stages.speaker_judge.prompt import cell_tracks, profiles, rows_of  # noqa: E402
from finesub.llm.stages.speaker_judge.windows import JudgeWindow  # noqa: E402
from finesub.speech.diarization.judgments import segment_ids  # noqa: E402
from finesub.speech.diarization.labels import speaker_cell, speaker_of  # noqa: E402

#: the ruler's people per file -> sex (男主 堀江瞬, 派蒙 古賀葵, 钟离 前野智昭, 温迪 村瀬歩; 嘉宾 differs per episode:
#: hina-17 三森すずこ, hina-24 木村良平, hina-22 前川涼子, hina-46 木野日菜; 心海 is the game's voice)
_HOSTS = {"男主": "m", "派蒙": "f", "钟离": "m", "温迪": "m", "岸大河": "m", "心海": "f"}
GENDER = {
    "hina-17": {**_HOSTS, "嘉宾": "f"},
    "hina-24": {**_HOSTS, "嘉宾": "m"},
    "hina-31": dict(_HOSTS),
    "hina-22": {**_HOSTS, "嘉宾": "f"},
    "hina-46": {**_HOSTS, "嘉宾": "f"},
    # P3's merged windows: Mr. Owl is a he; manosaba's nine are all young women (no cross-sex move possible)
    "hina-71": {"猫头鹰先生": "m", "昔涟": "f", "女开拓者": "f", "大黑塔": "f"},
    "manosaba-b": {},
}
SPK = S.r9.MAIN / "tmp" / "spk-merge"
DEV = ("hina-17-c0002", "hina-24-c0002", "hina-31-c0005")
HELDOUT = ("hina-22-c0102", "hina-46-c0001")
#: P3's merged-track windows (docs/plans/argon-audio-judge-plan.md 「P3 并轨窗」)
MERGED = ("manosaba-b-c0001", "hina-71-c0001")
GROUPS = (("dev", DEV), ("held-out", HELDOUT), ("merged", MERGED))
#: run name -> (folder under tmp/spk-merge, model, media, judge prompt, shaped (pass 1 of two or three passes), files)
RUNS = {
    "38F D8": ("two-pass-d8", "3.8F", "audio", "v3", True, DEV),
    "38F D8b": ("two-pass-d8b", "3.8F", "audio", "v3", True, DEV),
    "38F rflash1": ("accept/audio-rflash1", "3.8F", "audio", "v7", True, DEV),
    "Argon rargon1": ("accept/audio-rargon1", "Argon", "audio", "v7", True, DEV + HELDOUT),
    "38F audio-r1": ("accept/audio-r1", "3.8F", "audio", "v6", True, HELDOUT),
    # P2 (docs/plans/argon-audio-judge-plan.md): today's prompt on hina-22, with its recording and with silence
    "38F v7": ("accept/audio-rf38v7", "3.8F", "audio", "v7", True, ("hina-22-c0102",)),
    "38F silence": ("accept/audio-rsilence1", "3.8F", "audio", "v7", True, ("hina-22-c0102", *MERGED)),
    # P3: the same on the merged windows, with the text judge of the same prompt beside them
    "38F v7 merged": ("accept/audio-rf38v7", "3.8F", "audio", "v7", True, MERGED),
    "38F text v7": ("accept/main-rf38v7", "3.8F", "text", "v7", True, MERGED),
    # the audio as a bare m4a, heard whole (owner 2026-10-29; driver of branch trial/agy-bare-audio, 88811913)
    "38F bare": ("accept/audio-rbare1", "3.8F", "audio", "v7", True, HELDOUT),
    "38F bare merged": ("accept/audio-rbare1", "3.8F", "audio", "v7", True, MERGED),
    # P7 (owner 2026-10-29): S = pieces under 5 minutes, each under a 1-second frame (trial/agy-split-audio b3355c57);
    # B = bare m4a (trial/agy-bare-audio 88811913); three runs each
    "38F split r1": ("accept/audio-rsplit1", "3.8F", "audio", "v7", True, HELDOUT),
    "38F split r2": ("accept/audio-rsplit2", "3.8F", "audio", "v7", True, HELDOUT),
    "38F split r3": ("accept/audio-rsplit3", "3.8F", "audio", "v7", True, HELDOUT),
    "38F bare r2": ("accept/audio-rbare2", "3.8F", "audio", "v7", True, HELDOUT),
    "38F bare r3": ("accept/audio-rbare3", "3.8F", "audio", "v7", True, HELDOUT),
    "38F bare r4": ("accept/audio-rbare4", "3.8F", "audio", "v7", True, HELDOUT),
    # P9: the split arm plus the audio note's line on overlapping voices (trial/judge-overlap-prompt)
    "38F ovl r1": ("accept/audio-rovl1", "3.8F", "audio", "v7-ovl", True, HELDOUT),
    # P11: the split arm with the audio note rewritten from the judge's own feedback (trial/judge-heard-prompt)
    "38F heard r1": ("accept/audio-rheard1", "3.8F", "audio", "v7-heard", True, HELDOUT),
    "38F iter1 r1": ("accept/audio-riter1", "3.8F", "audio", "v7-iter1", True, HELDOUT),
    "38F iter2 r1": ("accept/audio-riter2", "3.8F", "audio", "v7-iter2", True, HELDOUT),
    "38F iter3 r1": ("accept/audio-riter3", "3.8F", "audio", "v7-iter3", True, HELDOUT),
    "38F iter4 r1": ("accept/audio-riter4", "3.8F", "audio", "v7-iter4", True, HELDOUT),
    "38F iter5 r1": ("accept/audio-riter5", "3.8F", "audio", "v7-iter5", True, HELDOUT),
    "38F iter6 r1": ("accept/audio-riter6", "3.8F", "audio", "v7-iter6", True, HELDOUT),
    # P13: a text-only draft, then the recording in the same session patching it (trial/judge-text-then-audio);
    # "draft" reads the draft the same run kept (`patch.draft`), a text judge v7 under a silent placeholder
    "38F p13a r1": ("accept/audio-rp13a", "3.8F", "audio", "v7-p13a", True, HELDOUT),
    "38F p13a draft": ("accept/audio-rp13a", "3.8F", "text", "v7-p13a", True, HELDOUT),
    # P13's control: the text judge v7 alone on the held-out windows (no placeholder), also one of D1's text runs
    "38F text tv7a": ("accept/main-rtv7a", "3.8F", "text", "v7", True, HELDOUT),
    "38F p13a2 r1": ("accept/audio-rp13a2", "3.8F", "audio", "v7-p13a2", True, HELDOUT),
    "38F p13a2 draft": ("accept/audio-rp13a2", "3.8F", "text", "v7-p13a2", True, HELDOUT),
    "38F p13b r1": ("accept/audio-rp13b", "3.8F", "audio", "v7-p13b", True, HELDOUT),
    "38F p13b draft": ("accept/audio-rp13b", "3.8F", "text", "v7-p13b", True, HELDOUT),
    "38F p13b2 r1": ("accept/audio-rp13b2", "3.8F", "audio", "v7-p13b", True, HELDOUT),
    "38F p13b2 draft": ("accept/audio-rp13b2", "3.8F", "text", "v7-p13b", True, HELDOUT),
    "38F p13c r1": ("accept/audio-rp13c", "3.8F", "audio", "v7-p13c", True, HELDOUT),
    "38F p13c draft": ("accept/audio-rp13c", "3.8F", "text", "v7-p13c", True, HELDOUT),
    "38F p13c2 r1": ("accept/audio-rp13c2", "3.8F", "audio", "v7-p13c", True, HELDOUT),
    "38F p13c2 draft": ("accept/audio-rp13c2", "3.8F", "text", "v7-p13c", True, HELDOUT),
    "38F p13d r1": ("accept/audio-rp13d", "3.8F", "audio", "v7-p13d", True, HELDOUT),
    "38F p13d draft": ("accept/audio-rp13d", "3.8F", "text", "v7-p13d", True, HELDOUT),
    "38F p13e r1": ("accept/audio-rrp13e", "3.8F", "audio", "v7-p13e", True, HELDOUT),
    "38F p13e draft": ("accept/audio-rrp13e", "3.8F", "text", "v7-p13e", True, HELDOUT),
    "38F p13f r1": ("accept/audio-rp13f", "3.8F", "audio", "v7-p13f", True, HELDOUT),
    "38F p13f draft": ("accept/audio-rp13f", "3.8F", "text", "v7-p13f", True, HELDOUT),
    "38F p13g1 r1": ("accept/audio-rp13g1", "3.8F", "audio", "v7-p13gA", True, HELDOUT),
    "38F p13g1 draft": ("accept/audio-rp13g1", "3.8F", "text", "v7-p13gA", True, HELDOUT),
    "38F p13g2 r1": ("accept/audio-rp13g2", "3.8F", "audio", "v7-p13gA", True, HELDOUT),
    "38F p13g2 draft": ("accept/audio-rp13g2", "3.8F", "text", "v7-p13gA", True, HELDOUT),
    "38F p13gb1 r1": ("accept/audio-rp13gb1", "3.8F", "audio", "v7-p13gB", True, HELDOUT),
    "38F p13gb1 draft": ("accept/audio-rp13gb1", "3.8F", "text", "v7-p13gB", True, HELDOUT),
    "38F p13gb2 r1": ("accept/audio-rp13gb2", "3.8F", "audio", "v7-p13gB", True, HELDOUT),
    "38F p13gb2 draft": ("accept/audio-rp13gb2", "3.8F", "text", "v7-p13gB", True, HELDOUT),
    # P10 (owner 2026-10-30): P7's two arms on windows of 17.5 and 20.2 minutes, one run each
    "38F lsplit1": ("accept/audio-rlsplit1", "3.8F", "audio", "v7", True, ("hina-17-c0002", "manosaba-b-c0001")),
    "38F lbare1": ("accept/audio-rlbare1", "3.8F", "audio", "v7", True, ("hina-17-c0002", "manosaba-b-c0001")),
    # P4: P2's silence on Argon, beside its own held-out run with the recording ("Argon rargon1")
    "Argon silence": ("accept/audio-rargonsilence1", "Argon", "audio", "v7", True, HELDOUT),
    # the last of the week's Argon quota (2026-10-25): Argon on the merged windows, its text judge, a second audio run
    "Argon merged": ("accept/audio-rargon1", "Argon", "audio", "v7", True, MERGED),
    "Argon rargon2": ("accept/audio-rargon2", "Argon", "audio", "v7", True, HELDOUT),
    "Argon text": ("accept/main-rargon1", "Argon", "text", "v7", True, HELDOUT + MERGED),
    "38F jao-r1": ("accept/judge-audio-only-r1", "3.8F", "audio", "v6", False, HELDOUT),
    "38F jao-r2": ("accept/judge-audio-only-r2", "3.8F", "audio", "v6", False, HELDOUT),
    "38F text D-may": ("two-pass-may-agy-r1", "3.8F", "text", "v3", True, DEV),
    "38F text D5": ("two-pass-d5-fresh", "3.8F", "text", "v3", True, DEV),
    "38F text main-r1": ("accept/main-r1", "3.8F", "text", "v6", True, HELDOUT),
    "38F text main-r2": ("accept/main-r2", "3.8F", "text", "v6", True, HELDOUT),
    # Sol (owner 2026-10-31): the diar-marks acceptance's Sol text judge (prompt v2, whole episodes, 2026-10-09; the
    # same stable rows and tracks as these windows, its answers cut to them), and today's text judge v7 on Sol
    "Sol text v2": ("accept/sol-v2-frozen", "Sol", "text", "v2", True, HELDOUT),
    "Sol text v7 r1": ("accept/main-text-rsol7a", "Sol", "text", "v7", True, HELDOUT),
    "Sol text v7 r2": ("accept/main-text-rsol7b", "Sol", "text", "v7", True, HELDOUT),
    "38F text base-r1": ("accept/base-judge-r1", "3.8F", "text", "v6", False, HELDOUT),
    "38F text base-r2": ("accept/base-judge-r2", "3.8F", "text", "v6", False, HELDOUT),
    # D1 rerun (owner 2026-10-31): text v7 and audio v7 (pieces) on agy, text v7 on Sol, P13g A with patch note v8
    "38F text d1a": ("accept/main-rd1a", "3.8F", "text", "v7", True, HELDOUT + ("hina-71-c0001",)),
    "38F text d1b": ("accept/main-rd1b", "3.8F", "text", "v7", True, HELDOUT + ("hina-71-c0001",)),
    "Sol text d1a": ("accept/main-text-rd1sola", "Sol", "text", "v7", True, ("hina-71-c0001",)),
    "Sol text d1b": ("accept/main-text-rd1solb", "Sol", "text", "v7", True, ("hina-71-c0001",)),
    "38F split d1a": ("accept/audio-rd1a", "3.8F", "audio", "v7", True, ("hina-71-c0001",)),
    "38F split d1b": ("accept/audio-rd1b", "3.8F", "audio", "v7", True, ("hina-71-c0001",)),
    "38F p13gv8a r1": ("accept/audio-rp13gv8a", "3.8F", "audio", "v8-p13gA", True, HELDOUT + ("hina-71-c0001",)),
    "38F p13gv8a draft": ("accept/audio-rp13gv8a", "3.8F", "text", "v8-p13gA", True, HELDOUT + ("hina-71-c0001",)),
    "38F p13gv8b r1": ("accept/audio-rp13gv8b", "3.8F", "audio", "v8-p13gA", True, HELDOUT + ("hina-71-c0001",)),
    "38F p13gv8b draft": ("accept/audio-rp13gv8b", "3.8F", "text", "v8-p13gA", True, HELDOUT + ("hina-71-c0001",)),
}
#: audio runs whose attachment the model heard whole (bare m4a or pieces under 5 minutes); every other audio run had
#: agy's single-frame mp4, of which the model hears only the part below the last multiple of 5 minutes
HEARD_WHOLE = {name for name in RUNS if name.startswith(("38F bare", "38F split", "38F ovl", "38F heard", "38F iter", "38F lsplit", "38F lbare", "38F p13gv8"))}
NEARS = (2.0, 5.0, 10.0)
SHORT_SECONDS, SHORT_CHARS = 1.5, 4
MAI2 = S.r9.MAIN / "data" / "mai2-2026-10" / "hina-mix"
#: MAI2's anonymous speakers -> sex, read off their many-to-one people (hina-22: S0 钟离, S1 男主, S2 嘉宾 and 派蒙)
MAI2_SEX = {"hina-22": {"S0": "m", "S1": "m", "S2": "f"}, "hina-46": {"S1": "m", "S0": "f", "S2": "f"}}


def audio_runs():
    return [name for name, run in RUNS.items() if run[2] == "audio"]


def text_of(name, key):
    """The text runs of the same shape on this file."""
    shaped = RUNS[name][4]
    return [other for other, run in RUNS.items() if run[2] == "text" and run[4] == shaped and key in run[5]]


def acoustic_cells(segs):
    return {i: S.cell_labels(speaker_cell([speaker_of(s)])) for i, s in enumerate(segs)}


def settled(payload):
    units = [unit for window in payload["windows"] for unit in (window.get("parts") or [window])]
    return bool(units) and all(unit.get("status") == "success" for unit in units)


def short(row):
    return row.seconds < SHORT_SECONDS or len(re.sub(r"[\W_]", "", row.text)) <= SHORT_CHARS


def gated(gate, rows, i, track, tracks):
    """Whether `gate` refuses moving row i onto `track` (`tracks`: row -> the track the judge answered)."""
    if not gate or track in cell_tracks(rows[i], marks=True):
        return False
    if gate == "short":
        return short(rows[i])
    if gate == "isolated":
        return tracks.get(i - 1) != track and tracks.get(i + 1) != track
    return True


def judged(payload, segs, acoustic, *, gate=None):
    """({row: labels} after the judge, {row: why} for the rows it moved); with `gate` (`cell`, `short`, `isolated`),
    a move onto a track the row's cell does not show is not taken (on short rows only; when isolated only)."""
    index = {key: i for i, key in enumerate(segment_ids(segs))}
    rows = rows_of(segs) if gate else None
    tracks = {index[str(result["row"])]: int(result["track"]) for window in payload["windows"]
              for unit in (window.get("parts") or [window]) for result in unit.get("results") or [] if "track" in result}
    out, why = dict(acoustic), {}
    for window in payload["windows"]:
        for unit in window.get("parts") or [window]:
            for result in unit.get("results") or []:
                i = index[str(result["row"])]
                if "track" in result:
                    if gated(gate, rows, i, int(result["track"]), tracks):
                        continue
                    out[i] = [f"s{result['track']}"]
                elif "label" in result:
                    out[i] = [f"@{unit.get('id')}/{result['label']}"]
                elif "carried" in result:
                    out[i] = [f"@{result['carried']['window']}/{result['carried']['label']}"]
                else:
                    continue
                why[i] = result.get("why", "")
    return out, why


class Win:
    """One correction window: its stable's rows with the truth of each (keyed by the stable's row)."""

    def __init__(self, key, segs):
        ident = key.rsplit("-c", 1)[0]
        product = S.stable(ident)
        at = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(product)}
        truth_all = S.truth_rows(ident)
        truth = {}
        for i, s in enumerate(segs):
            k = at.get((round(s["start"], 3), round(s["end"], 3)))
            if k in truth_all:
                truth[i] = truth_all[k]
        self.key, self.segs, self.ids, self.rows = key, segs, segment_ids(segs), rows_of(segs)
        self.window = S.Window(key, ident, sorted(truth), "", False, truth)
        self.truth = truth
        self.scored = [i for i in sorted(truth) if truth[i].cls == "one" and truth[i].prio == "main" and truth[i].person]
        self.acoustic = acoustic_cells(segs)
        self.amap = S.fit(self.window, self.acoustic)
        self.a_right = sum(self.right(self.acoustic, self.amap, i) for i in self.scored)
        #: how many main rows each person speaks in the file, for the weighted chance
        self.weight = Counter(truth[i].person for i in self.scored)

    def seconds(self, i):
        return float(self.segs[i]["end"]) - float(self.segs[i]["start"])

    def right(self, cells, mapping, i):
        labels = cells.get(i) or []
        return bool(labels) and mapping.get(labels[0]) == self.truth[i].person

    def obvious(self, i, new_person):
        sex = GENDER.get(self.window.ident, {})  # a file with no table yet counts no cross-sex move
        return {sex.get(self.truth[i].person), sex.get(new_person)} == {"m", "f"}

    def near_people(self, i, near):
        s = self.segs[i]
        lo, hi = float(s["start"]) - near, float(s["end"]) + near
        out = set()
        for j, row in self.truth.items():
            if j != i and float(self.segs[j]["start"]) < hi and float(self.segs[j]["end"]) > lo:
                out |= row.present
        return out

    def chance_near(self, i, near, sex=None):
        """The chance that a wrong person, drawn by how many rows each speaks (of `sex` when given), is near."""
        own = self.truth[i].person
        sexes = GENDER.get(self.window.ident, {})
        pool = {p: w for p, w in self.weight.items() if p != own and (sex is None or sexes.get(p) == sex)}
        total = sum(pool.values())
        return sum(w for p, w in pool.items() if p in self.near_people(i, near)) / total if total else 0.0

    def in_run(self, i):
        person = self.truth[i].person
        before = [j for j in self.scored if j < i][-1:]
        after = [j for j in self.scored if j > i][:1]
        return bool(before and after) and self.truth[before[0]].person == person == self.truth[after[0]].person


def window_starts(win, payload):
    """{row: start seconds of the judge window holding it}."""
    index = {key: i for i, key in enumerate(win.ids)}
    out = {}
    for window in payload["windows"]:
        lo, hi = index[str(window["first"])], index[str(window["last"])]
        for i in range(lo, hi + 1):
            out[i] = float(win.segs[lo]["start"])
    return out


def score(win, payload, *, gate=None):
    cells, why = judged(payload, win.segs, win.acoustic, gate=gate)
    starts = window_starts(win, payload)
    jmap = S.fit(win.window, cells)
    rows = {}
    for i in win.scored:
        if i not in why or (cells[i] or [None])[0] == (win.acoustic.get(i) or [None])[0]:
            continue
        a, j = win.right(win.acoustic, win.amap, i), win.right(cells, jmap, i)
        kind = "wrong" if a and not j else "fix" if j and not a else "both right" if a else "both wrong"
        new = jmap.get(cells[i][0])
        shown = cell_tracks(win.rows[i], marks=True)
        rows[i] = dict(kind=kind, obvious=kind == "wrong" and win.obvious(i, new),
                       in_run=kind == "wrong" and win.in_run(i), new=new,
                       minute=(float(win.segs[i]["start"]) - starts.get(i, 0.0)) / 60,
                       outside=cells[i][0].startswith("s") and int(cells[i][0][1:]) not in shown,
                       old=win.acoustic[i][0], new_label=cells[i][0], why=why[i])
    correct = {i: win.right(cells, jmap, i) for i in win.scored}
    return rows, correct, payload


def load(gate):
    wins, results = {}, {}
    for name, (folder, model, media, prompt, shaped, keys) in RUNS.items():
        for key in keys:
            sidecar = SPK / folder / key / "input" / f"{key}-speaker-judge.json"
            if not sidecar.exists():
                continue
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            if name.endswith(" draft"):
                for window in payload["windows"]:
                    for unit in window.get("parts") or [window]:
                        if "patch" in unit:
                            unit["results"] = unit["patch"]["draft"]
            if not settled(payload):
                print(f"warning: {name} {key}: a judge window is not `success`, skipped")
                continue
            if key not in wins:
                segs = json.loads((sidecar.parent / f"{key}-stable.json").read_text(encoding="utf-8"))["segments"]
                wins[key] = Win(key, segs)
                if key.rsplit("-c", 1)[0] not in GENDER:
                    print(f"warning: {key}: no entry in GENDER, its cross-sex moves read as 0")
            results[name, key] = score(wins[key], payload, gate=gate if media == "audio" else None)
    return wins, results


def audio_only(results, name, key, i):
    texts = [results[other, key][1] for other in text_of(name, key) if (other, key) in results]
    return bool(texts) and all(t[i] for t in texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=("cell", "short", "isolated"),
                    help="take only moves onto a track the row's cell shows (F1); that on short rows only (F1b),"
                         " or on isolated moves only (F1c)")
    ap.add_argument("--rows", help="write every wrong move to this CSV")
    ap.add_argument("--profile", action="store_true", help="wrong moves and fixes against every scored row")
    ap.add_argument("--samples", action="store_true", help="who speaks in each track's voice samples")
    ap.add_argument("--mai2", action="store_true", help="MAI2's sex on the held-out cross-sex rows")
    args = ap.parse_args()
    wins, results = load(args.gate)
    if args.gate:
        print(f"GATE {args.gate}: moves onto a track outside the row's cell are not taken"
              f"{dict(short=' on short rows', isolated=' when isolated').get(args.gate, '')} (audio runs)")

    print("per file and run: wrong / acoustic-right rows | cross-sex | in a run | fix / acoustic-wrong rows | net"
          " | other moves | audio only")
    totals = {}
    for group, keys in GROUPS:
        for key in keys:
            if key not in wins:
                continue
            win = wins[key]
            a_wrong = len(win.scored) - win.a_right
            print(f"=== {key}: {len(win.segs)} rows, {len(win.scored)} scored (one person, main), acoustic right {win.a_right}")
            for name, run in RUNS.items():
                if (name, key) not in results:
                    continue
                rows = results[name, key][0]
                c = Counter(r["kind"] for r in rows.values())
                ob = sum(r["obvious"] for r in rows.values())
                in_run = sum(r["in_run"] for r in rows.values())
                only = sum(r["kind"] == "wrong" and audio_only(results, name, key, i) for i, r in rows.items()) \
                    if run[2] == "audio" else 0
                t = totals.setdefault((group, name), Counter())
                t.update(wrong=c["wrong"], obvious=ob, run=in_run, fix=c["fix"], aright=win.a_right, awrong=a_wrong,
                         other=c["both right"] + c["both wrong"], only=only, files=1)
                print(f"  {name:17} {run[3]} {'shaped  ' if run[4] else 'unshaped'} wrong {c['wrong']:3}/{win.a_right}"
                      f"  cross-sex {ob:3}  in a run {in_run:3}  fix {c['fix']:3}/{a_wrong}  net {c['fix'] - c['wrong']:+4}"
                      f"  other {c['both right'] + c['both wrong']:3}  audio only {only:3}")
    for group, keys in GROUPS:
        print(f"=== totals, {group} (only runs that read every file of the group)")
        for (g, name), t in totals.items():
            if g != group or t["files"] != sum(key in wins for key in keys):
                continue
            print(f"  {name:17} wrong {t['wrong']:3}/{t['aright']} ({100 * t['wrong'] / t['aright']:.1f}%)"
                  f"  cross-sex {t['obvious']:3}  in a run {t['run']:3}"
                  f"  fix {t['fix']:3}/{t['awrong']} ({100 * t['fix'] / max(t['awrong'], 1):.1f}%)"
                  f"  net {t['fix'] - t['wrong']:+4}  other {t['other']:3}  audio only {t['only']:3}")

    print("=== overlap of wrong moves between audio runs (shared / either)")
    names = audio_runs()
    for x, y in [(x, y) for n, x in enumerate(names) for y in names[n + 1:]]:
        shared = either = 0
        for key in wins:
            if (x, key) in results and (y, key) in results:
                wa = {i for i, r in results[x, key][0].items() if r["kind"] == "wrong"}
                wb = {i for i, r in results[y, key][0].items() if r["kind"] == "wrong"}
                shared, either = shared + len(wa & wb), either + len(wa | wb)
        if either:
            print(f"  {x:17} x {y:17} {shared}/{either}")

    if args.profile:
        profile(wins, results)
    if args.samples:
        samples(wins, results)
    if args.mai2:
        mai2(wins, results)
    if args.rows:
        with open(args.rows, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window", "row", "run", "cross_sex", "in_run", "outside_cell", "seconds", "minute", "audio_only",
                        "truth", "truth_src", "old_label", "old_person", "new_label", "new_person", "text", "why"])
            for (name, key), (rows, _, _) in results.items():
                win = wins[key]
                for i, r in sorted(rows.items()):
                    if r["kind"] != "wrong":
                        continue
                    w.writerow([key, win.ids[i], name, int(r["obvious"]), int(r["in_run"]), int(r["outside"]),
                                round(win.seconds(i), 2), round(r["minute"], 1),
                                int(RUNS[name][2] == "audio" and audio_only(results, name, key, i)),
                                win.truth[i].person, win.truth[i].src, r["old"], win.amap.get(r["old"]),
                                r["new_label"], r["new"], win.truth[i].text, r["why"]])


def sample_people(win, payload):
    """{row: {track: [truth person or kind of each sample row]}} for the judge window holding each row."""
    index = {key: i for i, key in enumerate(win.ids)}
    out = {}
    for window in payload["windows"]:
        lo, hi = index[str(window["first"])], index[str(window["last"])] + 1
        got = profiles(win.rows, JudgeWindow(window["id"], window.get("chunk", window["id"]), lo, hi)).samples
        who = {track: [(win.truth[j].person or win.truth[j].cls) if j in win.truth else "?" for j in picks]
               for track, picks in got.items()}
        for i in range(lo, hi):
            out[i] = who
    return out


def samples(wins, results):
    print("=== voice samples: track (acoustic owner) -> truth of its sample rows, per judge window")
    seen = set()
    hits = Counter()
    for (name, key), (rows, _, payload) in results.items():
        if RUNS[name][2] != "audio":
            continue
        win = wins[key]
        who = sample_people(win, payload)
        if key not in seen:
            seen.add(key)
            for n, table in enumerate({id(x): x for x in who.values()}.values()):
                for track, people in sorted(table.items()):
                    print(f"  {key} window {n + 1} s{track} ({win.amap.get(f's{track}')}): {people}")
        for i, r in rows.items():
            if r["kind"] != "wrong" or not r["new_label"].startswith("s"):
                continue
            picks = who.get(i, {}).get(int(r["new_label"][1:]), [])
            hits[RUNS[name][1], "cross-sex" if r["obvious"] else "other", "no samples" if not picks else
                 "own person in samples" if win.truth[i].person in picks else
                 "new person in samples" if r["new"] in picks else "neither"] += 1
    print("=== wrong moves onto an acoustic track: what that track's samples hold (model, cross-sex or not, verdict)")
    for k, v in sorted(hits.items()):
        print(f"  {k}: {v}")


def mai2(wins, results):
    print("=== MAI2's sex against the truth's, acoustic-right rows: cross-sex wrong moves vs rows no audio judge moved")
    for key in HELDOUT:
        if key not in wins:
            continue
        win, ident = wins[key], key.rsplit("-c", 1)[0]
        payload = json.loads((MAI2 / f"{ident}-mix.json").read_text(encoding="utf-8"))
        offset = payload.get("offset") or 0.0
        moved, cross = set(), set()
        for (name, k), (rows, _, _) in results.items():
            if k == key and RUNS[name][2] == "audio":
                moved |= set(rows)
                cross |= {i for i, r in rows.items() if r["obvious"]}

        def sex_of(i):
            s, e = float(win.segs[i]["start"]), float(win.segs[i]["end"])
            over = Counter()
            for g in payload["segments"]:
                o = min(e, g["end"] + offset) - max(s, g["start"] + offset)
                if o > 0:
                    over[g["speaker"]] += o
            return MAI2_SEX[ident].get(over.most_common(1)[0][0]) if over else None

        tally = defaultdict(Counter)
        for i in win.scored:
            if not win.right(win.acoustic, win.amap, i) or (i in moved and i not in cross):
                continue
            band = "<1 s" if win.seconds(i) < 1 else "1-2 s" if win.seconds(i) < 2 else ">=2 s"
            group = "cross-sex wrong moves" if i in cross else "unmoved"
            got = sex_of(i)
            tally[group, band]["n"] += 1
            tally[group, band]["none" if got is None else "agree" if got == GENDER[ident].get(win.truth[i].person)
                               else "disagree"] += 1
        for (group, band), c in sorted(tally.items()):
            print(f"  {key} {group:22} {band:6} n {c['n']:3}  MAI2 disagrees with the truth {c['disagree']:3}"
                  f"  ({100 * c['disagree'] / c['n']:.0f}%)  no label {c['none']}")


def profile(wins, results):
    """Wrong moves and fixes of the audio runs (pooled per model) against the scored rows of the same files."""
    def bands(values, edges):
        names = [f"<{e:g}" for e in edges] + [f">={edges[-1]:g}"]
        out = Counter(next((f"<{e:g}" for e in edges if v < e), names[-1]) for v in values)
        return "  ".join(f"{name} {100 * out[name] / len(values):3.0f}%" for name in names)

    def cell_share(items):
        low = sum(win.rows[i].low for win, i, _ in items)
        multi = sum(len(win.rows[i].tracks) > 1 for win, i, _ in items)
        return f"low mark {100 * low / len(items):3.0f}%  2+ tracks {100 * multi / len(items):3.0f}%"

    print("=== profile (share of rows per band; pooled runs count a row once per run)")
    for group, keys in GROUPS:
        base = [(wins[key], i, None) for key in keys if key in wins for i in wins[key].scored]
        if not base:
            continue
        print(f"  [{group}]")
        minutes = []
        for w, i, _ in base:
            payload = next(p for (name, key), (_, _, p) in results.items() if key == w.key and RUNS[name][2] == "audio")
            minutes.append((float(w.segs[i]["start"]) - window_starts(w, payload).get(i, 0.0)) / 60)
        print(f"  {'all scored rows':30} n {len(base):4}  seconds {bands([w.seconds(i) for w, i, _ in base], (1, 2, 4))}"
              f"  | {cell_share(base)}  | minute {bands(minutes, (5, 10, 15))}")
        for model in ("3.8F", "Argon"):
            for label, want in (("wrong", lambda r: r["kind"] == "wrong"),
                                ("cross-sex", lambda r: r["obvious"]),
                                ("fix", lambda r: r["kind"] == "fix")):
                items = [(wins[key], i, r) for (name, key), (rows, _, _) in results.items()
                         if RUNS[name][1] == model and RUNS[name][2] == "audio" and key in keys
                         for i, r in rows.items() if want(r)]
                if not items:
                    continue
                outside = sum(r["outside"] for _, _, r in items)
                near = []
                for d in NEARS:
                    hit = sum(r["new"] in w.near_people(i, d) for w, i, r in items)
                    chance = sum(w.chance_near(i, d, GENDER[w.window.ident].get(r["new"]) if label == "cross-sex" else None)
                                 for w, i, r in items)
                    near.append(f"{d:g}s {100 * hit / len(items):3.0f}% ({100 * chance / len(items):3.0f}%)")
                print(f"  {model + ' ' + label:30} n {len(items):4}  seconds {bands([w.seconds(i) for w, i, _ in items], (1, 2, 4))}"
                      f"  | {cell_share(items)}  new track outside the cell {100 * outside / len(items):3.0f}%"
                      f"  | minute {bands([r['minute'] for _, _, r in items], (5, 10, 15))}"
                      f"  | new person near (weighted chance) {'  '.join(near)}")


if __name__ == "__main__":
    main()
