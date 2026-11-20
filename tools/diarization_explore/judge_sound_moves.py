"""Moves the audio judge justifies by sound, with what a text judge also makes taken out, against random moves
(owner 2026-11-14).

    python tools/diarization_explore/judge_sound_moves.py

Same runs, windows and scoring as `judge_moves_overlap.py` (the five plain hina windows, no gate, production's runs
as their " raw" answers; the silence placeholders read separately as the no-sound control). A move is:

- by sound: its reason (`why`) names timbre, voice, a male / female voice, the recording, hearing (`SOUND`);
- out of text reach: no text run of the same window moved that row to the same person (a text judge reaches the
  semantic moves; what it never makes is the candidate for hearing).

Baselines, per moved row:

- random target: move the row to another person of the window (not the acoustic one), drawn by how many scored rows
  each speaks; right only when the acoustic layer was wrong;
- matched random: pick the row at random among the window's scored rows with the same production-visible features
  (low mark, two tracks in the cell, turn edge, short; `judge_moves_overlap.Rows.features`), then a random target;
  its net is the expected fixes minus the acoustic-right rows it would move.

The ruler is 3.8 Flash with audio; `SOUND` is a word list; the held-out windows have more text runs than the dev
ones, so their moves are stripped of semantics more strictly; runs are pooled and the same row recurs across runs.
"""
from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_overlap as O  # noqa: E402
import judge_moves_readout as R  # noqa: E402

SOUND = re.compile(r"音色|声音|声线|声線|嗓|男声|女声|男聲|女聲|低沉|高音|低音|尖细|沙哑|浑厚|音调|语调|声调|听到|听起来"
                   r"|录音中|录音里|音频里|音频中|发声|音质|胸腔")
PLAIN = R.DEV + R.HELDOUT
FEATURES = ("low", "tracks2", "turn_edge", "short")
#: recording vs the silence placeholder, same prompt (v7) and windows
PAIRS = (("38F v7", "38F silence", ("hina-22-c0102",)), ("Argon rargon1", "Argon silence", R.HELDOUT),
         ("Argon rargon2", "Argon silence", R.HELDOUT))


def pool(name):
    if R.RUNS[name][2] == "text":
        return "text"
    return "silence" if "silence" in name else O.pool_of(name)


def chance(win, i):
    """P(right) of moving row i to a person other than its acoustic one, drawn by rows spoken."""
    own = win.amap.get(win.acoustic[i][0])
    others = {p: w for p, w in win.weight.items() if p != own}
    total = sum(others.values())
    return others.get(win.truth[i].person, 0) / total if total else 0.0


class Tally:
    def __init__(self):
        self.c, self.chance, self.m_awrong, self.m_chance = Counter(), 0.0, 0.0, 0.0

    def add(self, win, i, r, stratum):
        self.c.update(n=1, right=r["kind"] in ("fix", "both right"), awrong=r["kind"] in ("fix", "both wrong"),
                      fix=r["kind"] == "fix", wrong=r["kind"] == "wrong")
        self.chance += chance(win, i)
        n, awrong, ch = stratum
        self.m_awrong += awrong / n
        self.m_chance += ch / n

    def line(self, label):
        c = self.c
        n = c["n"]
        if not n:
            return f"  {label:44} no moves"
        fixed = c["fix"] / c["awrong"] if c["awrong"] else 0.0
        return (f"  {label:44} {n:5}  right {c['right'] / n:6.1%}  random target {self.chance / n:6.1%}"
                f"  matched random {self.m_chance / n:6.1%} | acoustic-wrong rows {c['awrong'] / n:6.1%}"
                f" (matched {self.m_awrong / n:6.1%}), fixed there {fixed:6.1%} | fix {c['fix']} wrong {c['wrong']}"
                f" net {c['fix'] - c['wrong']:+} (matched random {self.m_chance - (n - self.m_awrong):+.0f})")


def main():
    wins, results = R.load(None)
    feats = {k: O.Rows(wins[k], 0.2, O.mai2_segments(k.rsplit("-c", 1)[0])) for k in PLAIN if k in wins}

    def key_of(k, i):
        return tuple(feats[k].features(i)[x] for x in FEATURES)

    strata = {}
    for k in feats:
        win, s = wins[k], defaultdict(lambda: [0, 0, 0.0])
        for i in win.scored:
            c = s[key_of(k, i)]
            c[0] += 1
            c[1] += not win.right(win.acoustic, win.amap, i)
            c[2] += chance(win, i)
        strata[k] = s
    reach = defaultdict(set)   # (window, row) -> people a text run moved the row to
    for (name, k), (rows, _, _) in results.items():
        if pool(name) == "text" and k in feats:
            for i, r in rows.items():
                reach[k, i].add(r["new"])

    print("random row and target, per window: acoustic-wrong share of scored rows / right share of random moves")
    for k in feats:
        win = wins[k]
        print(f"  {k}: {sum(not win.right(win.acoustic, win.amap, i) for i in win.scored) / len(win.scored):.1%}"
              f" / {sum(chance(win, i) for i in win.scored) / len(win.scored):.1%}")

    groups = defaultdict(Tally)
    for (name, k), (rows, _, _) in results.items():
        p = pool(name)
        if p is None or k not in feats:
            continue
        model = R.RUNS[name][1]
        for i, r in rows.items():
            sound, semantic = bool(SOUND.search(r["why"] or "")), r["new"] in reach[k, i]
            labels = [f"{p}: all moves"]
            if p != "text":
                labels += [f"{p}: by sound"] * sound + [f"{p}: out of text reach"] * (not semantic)
                if sound and not semantic:
                    labels += [f"{p}: by sound, out of text reach", f"{p} {model}: by sound, out of text reach"]
            for label in labels:
                groups[label].add(wins[k], i, r, strata[k][key_of(k, i)])
    print("\npooled over runs (moves, right share, baselines | how the rows were picked | net)")
    for label in ("text: all moves", "audio: all moves", "audio: by sound", "audio: out of text reach",
                  "audio: by sound, out of text reach", *(f"audio {m}: by sound, out of text reach"
                                                          for m in ("3.8F", "Hydrogen", "Argon")),
                  "silence: all moves", "silence: by sound, out of text reach"):
        print(groups[label].line(label))

    print("\nrecording vs silence placeholder, same windows and prompt")
    for runs in PAIRS:
        audio, silent, keys = runs
        for name in (audio, silent):
            for label, keep in (("all moves", lambda s, t: True), ("by sound", lambda s, t: s),
                                ("by sound, out of text reach", lambda s, t: s and not t)):
                t = Tally()
                for k in keys:
                    for i, r in results.get((name, k), ({},))[0].items():
                        if keep(bool(SOUND.search(r["why"] or "")), r["new"] in reach[k, i]):
                            t.add(wins[k], i, r, strata[k][key_of(k, i)])
                print(t.line(f"{name} ({'+'.join(k[:7] for k in keys)}) {label}"))
        print()


if __name__ == "__main__":
    main()
