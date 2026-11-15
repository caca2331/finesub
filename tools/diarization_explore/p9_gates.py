"""P9 brainstorm (docs/plans/argon-audio-judge-plan.md): harness gates and audio + text fusions, scored offline.

    python tools/diarization_explore/p9_gates.py

"Outside the cell": a move onto a known track the row's cell does not show (`cell_tracks`). "Cross-band": the old
and new tracks' pitch levels lie at least BAND octaves apart; a track's level is the median of its row medians over the
file's long single-track rows -- the acoustic layer's own tracks, no truth -- so production could compute it. A move
onto a new `@` label is never gated. Variants of one audio run (each needing a text run of its model on the file is
averaged over the pairs, CONSENSUS):
- none; short (F1b): outside the cell on a short row (`judge_moves_readout.short`) is refused;
- consensus: outside the cell is refused unless the text judge answered the same track for the row;
- short+pitch: F1b, and on the other outside-cell moves, cross-band ones whose row pitch sides with the old track;
- short+pitchall: F1b, and any cross-band move (in the cell too, any length) whose row pitch sides with the old track;
- short+bandtext: F1b, and any cross-band move the text judge did not make the same way;
- band-out: outside the cell and cross-band is refused, whatever the row's pitch (the signature the audio judge's
  cross-sex wrong moves share, `p9_signature.py`); short+band-out: that, or F1b;
- textbase: the text judge's answers, plus the audio judge's on rows the text judge left, minus those F1b refuses and
  cross-band ones;
- textbase-incell: the text judge's answers, plus the audio judge's moves onto a track in the cell on rows the text
  judge left (no other audio move, no new label of the audio judge).
Scored as in judge_moves_readout (wrong: acoustic right, judged wrong; cross-sex wrong; fix: acoustic wrong, judged
right; net = fix - wrong), summed over the group's files per run, then the mean over a family's runs.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_f0_check as F  # noqa: E402
import judge_moves_readout as M  # noqa: E402

from finesub.llm.stages.speaker_judge.prompt import cell_tracks  # noqa: E402

BAND = 0.4
LEVEL_MAX = 40
CACHE = M.S.r9.MAIN / "tmp" / "p9-gates"
#: text runs a model's audio run is paired with, per group
CONSENSUS = {
    ("3.8F", "held-out"): ("38F text main-r1", "38F text main-r2"),
    ("3.8F", "merged"): ("38F text v7",),
    ("Argon", "held-out"): ("Argon text",),
    ("Argon", "merged"): ("Argon text",),
}
FAMILIES = {
    "38F v7": ("38F split r1", "38F split r2", "38F split r3", "38F bare r2", "38F bare r3", "38F bare r4",
               "38F v7 merged", "38F bare merged"),
    "38F P9 overlap": ("38F ovl r1",),
    # P12: one family per rewritten audio note
    **{f"38F P12 iter{k}": (f"38F iter{k} r1",) for k in range(1, 7)},
    "Argon": ("Argon rargon1", "Argon rargon2", "Argon merged"),
}
GROUPS = (("held-out", M.HELDOUT), ("merged", M.MERGED))
GATES = ("none", "short", "band-out", "short+band-out", "consensus", "short+pitch", "short+pitchall", "short+bandtext",
         "textbase", "textbase-incell")
PAIRED = ("consensus", "short+bandtext", "textbase", "textbase-incell")

CTX = {}
_plain = M.gated


def answers(payload, win):
    """{row: track} of a judge's moves onto known tracks."""
    index = {key: i for i, key in enumerate(win.ids)}
    return {index[str(r["row"])]: int(r["track"]) for w in payload["windows"] for u in (w.get("parts") or [w])
            for r in u.get("results") or [] if "track" in r}


class Levels:
    """Pitch level of each acoustic track of one file, and each row's pitch (cached on disk)."""

    def __init__(self, win):
        self.win, self.pitch = win, None
        self.path = CACHE / f"{win.key}.json"
        self.cache = {tuple(k): tuple(v) for k, v in json.loads(self.path.read_text())} if self.path.exists() else {}
        rows = defaultdict(list)
        for i, labels in win.acoustic.items():
            if len(labels) == 1 and labels[0].startswith("s") and win.seconds(i) >= F.REF_SECONDS:
                rows[int(labels[0][1:])].append(i)
        self.level = {}
        for track, members in rows.items():
            values = [self.row(i) for i in sorted(members, key=win.seconds, reverse=True)[:LEVEL_MAX]]
            values = [f for f, n in values if f is not None and n >= F.MIN_VOICED]
            self.level[track] = float(np.median(values)) if values else None

    def row(self, i):
        seg = self.win.segs[i]
        span = (float(seg["start"]), float(seg["end"]))
        if span not in self.cache:
            self.pitch = self.pitch or F.Pitch(self.win.key)
            self.cache[span] = self.pitch.row(seg)
        return self.cache[span]

    def save(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([[list(k), list(v)] for k, v in self.cache.items()]))

    def old_new(self, i, new):
        old = self.win.acoustic.get(i) or []
        if not old or not old[0].startswith("s"):
            return None, None
        return self.level.get(int(old[0][1:])), self.level.get(new)

    def cross(self, i, new):
        a, b = self.old_new(i, new)
        return a is not None and b is not None and abs(a - b) >= BAND

    def sides_old(self, i, new):
        a, b = self.old_new(i, new)
        return F.side(*self.row(i), a, b) == "ruler"


def gated(gate, rows, i, track, tracks):
    if gate in (None, "none"):
        return False
    outside = track not in cell_tracks(rows[i], marks=True)
    if gate in ("short", "cell", "isolated"):
        return outside and _plain(gate, rows, i, track, tracks)
    if gate == "consensus":
        return outside and CTX["text"].get(i) != track
    if gate == "band-out":
        return outside and CTX["levels"].cross(i, track)
    if outside and M.short(rows[i]):
        return True
    lv = CTX["levels"]
    if gate == "short+pitch":
        return outside and lv.cross(i, track) and lv.sides_old(i, track)
    if gate == "short+pitchall":
        return lv.cross(i, track) and lv.sides_old(i, track)
    if gate == "short+band-out":
        return outside and lv.cross(i, track)
    if gate == "short+bandtext":
        return lv.cross(i, track) and CTX["text"].get(i) != track
    raise ValueError(gate)


M.gated = gated


def textbase(audio, text, win, lv, incell=False):
    """The text judge's answers, then the audio judge's on rows the text judge left: minus F1b and cross-band moves,
    or (`incell`) only its moves onto a track in the row's cell."""
    index = {key: i for i, key in enumerate(win.ids)}
    left = {str(r["row"]) for w in text["windows"] for u in (w.get("parts") or [w]) for r in u.get("results") or []}
    extra = []
    for w in audio["windows"]:
        for u in w.get("parts") or [w]:
            for r in u.get("results") or []:
                if str(r["row"]) in left:
                    continue
                i = index[str(r["row"])]
                if incell:
                    if "track" in r and int(r["track"]) in cell_tracks(win.rows[i], marks=True):
                        extra.append(r)
                    continue
                if "track" in r:
                    t = int(r["track"])
                    if (t not in cell_tracks(win.rows[i], marks=True) and M.short(win.rows[i])) or lv.cross(i, t):
                        continue
                    extra.append(r)
                elif "label" in r:
                    extra.append({**r, "label": f"a{u.get('id')}-{r['label']}"})
                elif "carried" in r:
                    extra.append({"row": r["row"], "why": r.get("why", ""),
                                  "label": f"c{r['carried']['window']}-{r['carried']['label']}"})
    added = {"id": "audio", "first": audio["windows"][0]["first"], "last": audio["windows"][0]["first"],
             "status": "success", "results": extra}
    return {"windows": list(text["windows"]) + [added]}


def tally(win, payload, gate=None):
    rows = M.score(win, payload, gate=gate)[0]
    c = Counter(r["kind"] for r in rows.values())
    return Counter(wrong=c["wrong"], cross=sum(r["obvious"] for r in rows.values()), fix=c["fix"])


def main():
    wins, results = M.load(None)
    levels = {}
    table = defaultdict(Counter)  # (group, variant, run) -> counts
    for group, keys in GROUPS:
        for key in keys:
            if key not in wins:
                continue
            win = wins[key]
            lv = levels.setdefault(key, Levels(win))
            CTX["levels"] = lv
            for name, (folder, model, media, *_rest) in M.RUNS.items():
                if (name, key) not in results:
                    continue
                if media == "text":
                    table[group, "text alone", name].update(tally(win, results[name, key][2]))
                    table[group, "text alone", name]["files"] += 1
                    continue
                if not any(name in f for f in FAMILIES.values()):
                    continue
                payload = results[name, key][2]
                for gate in GATES:
                    if gate in PAIRED:
                        texts = [t for t in CONSENSUS.get((model, group), ()) if (t, key) in results]
                        if not texts:
                            continue
                        got = Counter()
                        for t in texts:
                            CTX["text"] = answers(results[t, key][2], win)
                            got.update(tally(win, textbase(payload, results[t, key][2], win, lv, gate == "textbase-incell"))
                                       if gate.startswith("textbase")
                                       else tally(win, payload, gate))
                        got = Counter({k: v / len(texts) for k, v in got.items()})
                    else:
                        got = tally(win, payload, gate)
                    table[group, gate, name].update(got)
                    table[group, gate, name]["files"] += 1
            lv.save()
    for key, lv in levels.items():
        print(f"{key}: track pitch " + ", ".join(f"s{t} {round(2 ** v)}Hz" for t, v in sorted(lv.level.items()) if v))

    def line(label, runs):
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("wrong", "cross", "fix")}
        return (f"      {label:16} wrong {mean['wrong']:5.1f}  cross-sex {mean['cross']:5.1f}  fix {mean['fix']:5.1f}"
                f"  net {mean['fix'] - mean['wrong']:+6.1f}")

    for group, keys in GROUPS:
        print(f"\n=== {group} ({', '.join(keys)}): mean per run over the runs that read every file")
        for family, names in FAMILIES.items():
            full = [n for n in names if table[group, "none", n]["files"] == len(keys)]
            if not full:
                continue
            print(f"  {family} (n={len(full)}: {', '.join(full)})")
            for gate in GATES:
                runs = [table[group, gate, n] for n in full if table[group, gate, n]["files"] == len(keys)]
                print(line(gate, runs) if runs else f"      {gate:16} -")
        print("  text judges alone")
        for (g, variant, name), t in sorted(table.items()):
            if g == group and variant == "text alone" and t["files"] == len(keys):
                print(line(name, [t]))


if __name__ == "__main__":
    main()
