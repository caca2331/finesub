"""P8 readout: per row x per session, what agy answered (`agy_judge_interrogate.py score`).

R1 and F: sex right (truth sex; 「听不清」 counts wrong and is shown as ?), nearest voice sample right (a track the
window's mapping gives to the true person), and for F the dictation's share of the ASR text's character 3-grams
(「原文对」 at >= 0.5; rows of 3 characters or fewer are only a hint). R2: what it says now about its own moves and
about two of its fixes (the decoys).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

import agy_judge_interrogate as P
import judge_moves_readout as M

TEXT_OK = 0.5
LINE = re.compile(r"^\W*q?(\d+)\s*[｜|]\s*(.+)$", re.M)


def grams(text):
    text = re.sub(r"[\s、。，,.!?！？「」…ー〜（）()＿]", "", text)
    return {text[i:i + 3] for i in range(len(text) - 2)}


def parse(answer):
    return {m.group(1): [x.strip() for x in re.split(r"[｜|]", m.group(2))] for m in LINE.finditer(answer)}


def sex_of(word):
    return "m" if "男" in word else "f" if "女" in word else "?"


def main():
    m = P.materials()
    answers = json.loads((P.WORK / "answers.json").read_text(encoding="utf-8"))
    wins, _ = M.load(None)
    win = wins[P.KEY]
    tracks_of = defaultdict(set)
    for label, person in win.amap.items():
        tracks_of[person].add(str(label))
    _, rows = P.prompt_parts()
    asr = {r[0]: r[3] for r in rows if r[0]}
    truth = {t["id"]: t for t in m["targets"]} | {c["id"]: c for c in m["controls"]}

    def judge(rid, sex_word, sample):
        t = truth[rid]
        s = sex_of(sex_word)
        sample_ok = any(re.fullmatch(rf"s?{re.escape(k.lstrip('s'))}", sample.strip()) for k in tracks_of[t["truth"]])
        return ("✓" if s == t["sex"] else "?" if s == "?" else "✗") + ("✓" if sample_ok else "✗")

    cols = []
    for a in answers:
        if a["turn"] not in ("r1", "f"):
            continue
        got = parse(a["answer"])
        cell = {}
        for rid, fields in got.items():
            if rid not in truth:
                continue
            if a["turn"] == "r1" and len(fields) >= 2:
                cell[rid] = judge(rid, fields[0], fields[1])
            elif a["turn"] == "f" and len(fields) >= 3:
                q = grams(asr[rid])
                share = len(grams(fields[0]) & q) / len(q) if q else 0.0
                text = ("T" if share >= TEXT_OK else "t") + ("*" if len(re.sub(r"\W", "", asr[rid])) <= 3 else "")
                cell[rid] = text + judge(rid, fields[1], fields[2])
        cols.append((f"{a['turn']}:{a['label']}", cell))
    print("cell: [F: T/t dictation >=0.5 or not, * = 3 chars or fewer] sex ✓/✗/? (? = 听不清) sample ✓/✗; - = no answer")
    print("row  kind       truth  sec  " + "  ".join(name for name, _ in cols))
    order = []
    for t in m["targets"]:
        kind = "suspect" if t["id"] in P.SUSPECT else "confirmed" if t["id"] in ("171", "221") else "target"
        order.append((t["id"], kind))
        order += [(c["id"], f"ctrl {t['id']}") for c in m["controls"] if c["for"] == t["id"]]
    for rid, kind in order:
        t = truth[rid]
        print(f"{rid:>4} {kind:10} {t['truth']:4} {t['seconds']:4.1f}  " + "  ".join(f"{cell.get(rid, '-'):>{len(name)}}" for name, cell in cols))
    for a in answers:
        if a["turn"] != "r2":
            continue
        s = m["sessions"][a["label"]]
        decoys = {d["id"] for d in s["decoys"]}
        print(f"\nR2 {a['label']}:")
        for rid, fields in parse(a["answer"]).items():
            role = "decoy" if rid in decoys else "move"
            now = fields[2] if len(fields) > 2 else ""
            print(f"  q{rid} {role:5} who={fields[1] if len(fields) > 1 else ''}  now={now}")


if __name__ == "__main__":
    main()
