"""P9 and P11 (docs/plans/argon-audio-judge-plan.md 「P9 提醒叠音的 prompt」「P11 把 agy 自己的反馈写进 prompt」): what a
new audio note changes beyond the main table.

    python tools/diarization_explore/judge_moves_readout.py --rows <dir>/rows.csv
    python tools/diarization_explore/judge_f0_check.py --rows <dir>/f0.csv
    python tools/diarization_explore/p9_readout.py <dir> [run ...]

Per run and window: the answers the judge gave (keep / abstain / a track, read from its last exchange per unit),
splits, fills and inserts; cross-sex wrong moves and those both witnesses (pitch from f0.csv, MAI2) side with the ruler
on; cross-sex fixes (the acoustic layer's person and the truth differ in sex); moves whose reason talks about overlap;
and P11's rule reading: moves whose reason opens with what was heard (`听到：男声…`), and those that went to the other
sex of what the judge itself says it heard.
Then the rows the baseline moved onto the other sex in at least two of its runs and a new-prompt run left alone, with
their witnesses; and P8's eight rows, owner-heard, run by run.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as M  # noqa: E402

BASE = ("38F split r1", "38F split r2", "38F split r3")
NEW = ("38F ovl r1", "38F heard r1", "38F iter1 r1", "38F iter2 r1", "38F iter3 r1", "38F iter4 r1", "38F iter5 r1", "38F iter6 r1")
KEYS = ("hina-22-c0102", "hina-46-c0001")
P8 = ("171", "189", "192", "214", "221", "291", "295", "337")
OVERLAP = re.compile(r"叠|重叠|同时|盖|笑声|附和|插话|背景")
TRACK = re.compile(r'"track"\s*:\s*"([^"]+)"')
HEARD = re.compile(r"^\s*听到[：:]\s*(男女声都有|男声|女声|听不清)")
SAID = {"男声": "m", "女声": "f"}


def answers(run, key):
    """Counter of the judge's answers over its units: keep, abstain, move (a track or label), split, fill, insert."""
    out = Counter()
    folder = M.SPK / M.RUNS[run][0] / key / "llm" / "exchanges"
    last = {}
    for path in sorted(folder.glob("*-speaker_judge-*-attempt*.md")):
        unit = path.name.split("-speaker_judge-")[1].split("-attempt")[0]
        last[unit] = path
    for path in last.values():
        reply = path.read_text(encoding="utf-8").split("## 模型响应", 1)[-1]
        for line in reply.splitlines():
            if not line.lstrip().startswith("{"):
                continue
            got = TRACK.search(line)
            if got:
                out[got.group(1) if got.group(1) in ("keep", "abstain") else "move"] += 1
            for shape in ("split", "fill", "insert"):
                out[shape] += f'"{shape}"' in line
    return out


def witnesses(directory):
    """(window, row 0-based, person moved to) -> (pitch verdict, MAI2 verdict), each 'ruler', 'judge' or other."""
    f0 = {(r["window"], int(r["row"]), r["judge"]): r["verdict"]
          for r in csv.DictReader(open(directory / "f0.csv", encoding="utf-8"))}
    segs, mai = {}, {}

    def mai_says(key, i, truth):
        ident = key.rsplit("-c", 1)[0]
        if key not in segs:
            segs[key] = json.loads((M.SPK / M.RUNS[BASE[0]][0] / key / "input" / f"{key}-stable.json")
                                   .read_text(encoding="utf-8"))["segments"]
            mai[key] = json.loads((M.MAI2 / f"{ident}-mix.json").read_text(encoding="utf-8"))
        offset = mai[key].get("offset") or 0.0
        s, e = float(segs[key][i]["start"]), float(segs[key][i]["end"])
        over = Counter()
        for g in mai[key]["segments"]:
            o = min(e, g["end"] + offset) - max(s, g["start"] + offset)
            if o > 0:
                over[g["speaker"]] += o
        sex = M.MAI2_SEX[ident].get(over.most_common(1)[0][0]) if over else None
        return "none" if sex is None else "ruler" if sex == M.GENDER[ident].get(truth) else "judge"
    return lambda key, i, judge, truth: (f0.get((key, i, judge), "missing"), mai_says(key, i, truth))


def main():
    directory = Path(sys.argv[1])
    runs = tuple(sys.argv[2:]) or BASE + NEW
    wins, results = M.load(None)
    witness = witnesses(directory)
    print("run          window  keep abstain move split fill insert | x-sex wrong confirmed | x-sex fixes | overlap-why"
          " | heard-prefix | against own ear")
    for run in runs:
        for key in KEYS:
            if (run, key) not in results:
                continue
            win, ident = wins[key], key.rsplit("-c", 1)[0]
            sex = M.GENDER[ident]
            got = results[run, key][0]
            a = answers(run, key)
            wrong = [(i, r) for i, r in got.items() if r["kind"] == "wrong" and r["obvious"]]
            confirmed = sum(witness(key, i, r["new"], win.truth[i].person) == ("ruler", "ruler") for i, r in wrong)
            fixes = sum(1 for i, r in got.items() if r["kind"] == "fix"
                        and sex.get(win.amap.get((win.acoustic.get(i) or [None])[0])) not in (None, sex.get(win.truth[i].person)))
            why = sum(bool(OVERLAP.search(r["why"] or "")) for r in got.values())
            heard = [HEARD.match(r["why"] or "") for r in got.values()]
            against = sum(1 for h, r in zip(heard, got.values())
                          if h and h.group(1) in SAID and sex.get(r["new"]) not in (None, SAID[h.group(1)]))
            print(f"{run:12} {key[:7]}  {a['keep']:4} {a['abstain']:7} {a['move']:4} {a['split']:5} {a['fill']:4} {a['insert']:6} |"
                  f" {len(wrong):11} {confirmed:9} | {fixes:11} | {why:3} of {len(got)}"
                  f" | {sum(map(bool, heard)):3} of {len(got)} | {against}")
    print("\nrows the baseline moved onto the other sex in >= 2 runs, left alone by a new-prompt run:")
    for key in KEYS:
        win = wins[key]
        moved = defaultdict(list)
        for run in BASE:
            for i, r in results[run, key][0].items():
                if r["kind"] == "wrong" and r["obvious"]:
                    moved[i].append(r["new"])
        for run in (r for r in runs if r in NEW and (r, key) in results):
            got = results[run, key][0]
            for i, people in sorted(moved.items()):
                if len(people) >= 2 and not (i in got and got[i]["kind"] == "wrong" and got[i]["obvious"]):
                    truth = win.truth[i].person
                    pitch, mai = witness(key, i, people[0], truth)
                    print(f"  {run} {key} {win.ids[i]:>4} {win.seconds(i):4.1f}s truth {truth} -> {people[0]} in {len(people)}/3"
                          f"  pitch {pitch}  MAI2 {mai}  now: {got[i]['kind'] + ' ' + got[i]['new'] if i in got else 'not moved'}")
    print("\nP8's eight rows (owner heard: ruler right):")
    win = wins["hina-46-c0001"]
    index = {k: i for i, k in enumerate(win.ids)}
    for rid in P8:
        i = index[rid]
        cells = []
        for run in runs:
            r = results.get((run, "hina-46-c0001"), ({},))[0].get(i)
            cells.append(f"{run.replace('38F ', '')}: {r['new'] if r else '—'}")
        print(f"  {rid} truth {win.truth[i].person}  " + "  ".join(cells))
        for run in runs:
            r = results.get((run, "hina-46-c0001"), ({},))[0].get(i)
            if r and run in NEW:
                print(f"      {run} why: {r['why']}")


if __name__ == "__main__":
    main()
