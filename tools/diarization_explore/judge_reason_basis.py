"""What the audio judge's moves rest on, read off its reasons, and whether the moves resting on sound can be told
apart and gated (owner 2026-11-14).

    python tools/diarization_explore/judge_reason_basis.py items     # items.jsonl, questions.json, sample.tsv
    python ~/.agents/my-agent-dispatch/scripts/dispatch.py decide --items <OUT>/items.jsonl --questions <OUT>/questions.json --out <OUT>/decide
    python tools/diarization_explore/judge_reason_basis.py agree     # the Decisions answers against sample.tsv's hand labels
    python tools/diarization_explore/judge_reason_basis.py read      # moves by basis and feature; gates replayed

Every distinct reason (`why`) of the audio pool's moves (`judge_moves_overlap.pool_of`, all seven windows) goes to the
Gemini Decisions API once, with three questions: does it rest on sound (timbre, a male / female voice, what the
recording sounds like, a voice sample); does it rest on content (what is said, who answers whom, the plot, a name,
the turn); which of the two carries it. A move's basis:

- sound: rests on sound and not on content, or sound carries it;
- both: rests on both, content carries it;
- content: rests on content only (or on neither: a bare track name).

`sample.tsv` holds a fixed-seed sample labelled by hand (column `label`: sound / both / content) to check the API's
reading before it is used. `read` gives fixes and wrong moves per basis and model on the plain windows, crossed with the
production-visible features of `judge_moves_overlap.Rows`, and replays gates that drop moves by basis (alone and on
top of F1b) by the whole-window reading of `judge_gate_long.py` on the dev, held-out and merged windows.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_overlap as O  # noqa: E402
import judge_moves_readout as R  # noqa: E402
import judge_sound_moves as SM  # noqa: E402

OUT = R.S.r9.MAIN / "data" / "argon-audio-judge-2026-10" / "tmp" / "spk-merge" / "accept" / "reason-basis"
SEED, SAMPLE = 20261114, 80
CONTEXT = "下面是一个说话人裁判写的理由：它把一行字幕从声学模型给的说话人改挂到另一个人名下，这句理由解释为什么。\n理由："
QUESTIONS = {
    "sound": {"type": "noul", "instructions": "这条理由有没有拿声音本身作依据？比如音色、男声或女声、音高、声线、"
              "录音里听到的是谁、声音像某条轨的声音样本。写成「录音中是某某」「录音里为某某」「声音属于某轨」「为某某的声音」"
              "也算：那是说它听出了是谁。只提到「附和」「笑」「喊」这类动作、没说声音的，不算。",
              "criteria": {"true_description": "理由里有对声音本身的判断", "false_description": "理由里没有对声音本身的判断"}},
    "content": {"type": "noul", "instructions": "这条理由有没有拿台词内容或对话关系作依据？比如这句话说了什么、"
                "接着谁的话、回应谁、剧情、自称或报名字、话轮的先后。",
                "criteria": {"true_description": "理由里有内容或对话关系上的依据", "false_description": "理由里没有内容或对话关系上的依据"}},
    "carry": {"type": "choice", "instructions": "这条理由主要靠哪一种依据把这一行改给新的人？",
              "criteria": {"sound": "主要靠声音本身（音色、男女声、录音里听到的）", "content": "主要靠台词内容、对话关系或剧情",
                           "other": "两样都没有，或看不出"}},
}


def items_of(results):
    """{reason id: reason} over the audio pool's moves, every window."""
    out = {}
    for (name, _), (rows, _, _) in results.items():
        if O.pool_of(name) == "audio":
            for r in rows.values():
                why = (r["why"] or "").strip()
                if why:
                    out[hashlib.sha1(why.encode("utf-8")).hexdigest()[:16]] = why
    return out


def basis_of(answer):
    if answer is None:
        return "none"
    sound, content = answer["sound"]["noul"] >= 0.5, answer["content"]["noul"] >= 0.5
    carry = answer["carry"]["choice"]
    if sound and (not content or carry == "sound"):
        return "sound"
    return "both" if sound else "content"


def answers():
    out = {}
    path = OUT / "decide" / "answers.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") == "answered":
            out[row["id"]] = row["answers"]
    return out


def cmd_items(results):
    reasons = items_of(results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "questions.json").write_text(json.dumps(QUESTIONS, ensure_ascii=False, indent=1), encoding="utf-8")
    sample = OUT / "sample.tsv"
    if not sample.exists():   # hand labels live here; never overwrite them
        rng = random.Random(SEED)
        loud = sorted(rid for rid, why in reasons.items() if SM.SOUND.search(why))
        quiet = sorted(rid for rid, why in reasons.items() if not SM.SOUND.search(why))
        picked = rng.sample(loud, SAMPLE // 2) + rng.sample(quiet, SAMPLE // 2)
        rng.shuffle(picked)
        with sample.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["id", "label", "why"])
            for rid in picked:
                w.writerow([rid, "", reasons[rid]])
    # the labelled sample first, so `decide --limit SAMPLE` answers just it
    first = [r["id"] for r in csv.DictReader(sample.open(encoding="utf-8"), delimiter="\t")]
    with (OUT / "items.jsonl").open("w", encoding="utf-8") as f:
        for rid in first + sorted(set(reasons) - set(first)):
            f.write(json.dumps({"id": rid, "input": CONTEXT + reasons[rid]}, ensure_ascii=False) + "\n")
    print(f"{len(reasons)} distinct reasons -> {OUT}")


def cmd_agree():
    got = answers()
    rows = list(csv.DictReader((OUT / "sample.tsv").open(encoding="utf-8"), delimiter="\t"))
    pairs = Counter()
    for r in rows:
        if r["label"]:
            pairs[r["label"], basis_of(got.get(r["id"]))] += 1
    n = sum(pairs.values())
    print(f"hand label x Decisions basis ({n} labelled): agree {sum(v for (a, b), v in pairs.items() if a == b) / n:.1%}")
    for (a, b), v in sorted(pairs.items()):
        print(f"  {a:8} -> {b:8} {v}")
    for r in rows:
        b = basis_of(got.get(r["id"]))
        if r["label"] and r["label"] != b:
            print(f"  differs: hand {r['label']} / api {b}: {r['why']}")


def cmd_read(wins, results):
    got = answers()

    def basis(why):
        return basis_of(got.get(hashlib.sha1((why or "").strip().encode("utf-8")).hexdigest()[:16]))

    feats = {k: O.Rows(wins[k], 0.2, O.mai2_segments(k.rsplit("-c", 1)[0])) for k in SM.PLAIN if k in wins}
    reach = defaultdict(set)
    for (name, k), (rows, _, _) in results.items():
        if SM.pool(name) == "text" and k in feats:
            for i, r in rows.items():
                reach[k, i].add(r["new"])
    t = defaultdict(Counter)
    for (name, k), (rows, _, _) in results.items():
        if O.pool_of(name) != "audio" or k not in feats:
            continue
        model = R.RUNS[name][1]
        for i, r in rows.items():
            if r["kind"] not in ("wrong", "fix"):
                continue
            b, f = basis(r["why"]), feats[k].features(i, r["new"])
            cell = "in cell" if not r["outside"] else "outside cell"
            for label in (f"all / {b}", f"{model} / {b}", f"{b} / {cell}", f"{b} / {'short' if f['short'] else 'long'}",
                          f"{b} / {cell} / {'short' if f['short'] else 'long'}",
                          f"{b} / {'target audible' if f['target_audible'] else 'target silent'}",
                          f"{b} / {'text also makes it' if r['new'] in reach[k, i] else 'out of text reach'}"):
                t[label][r["kind"]] += 1
    print("plain windows, audio pool, moves by the basis of their reason (fix / wrong / share right)")
    for label in sorted(t):
        c = t[label]
        print(f"  {label:44} fix {c['fix']:4}  wrong {c['wrong']:4}  right {c['fix'] / max(1, c['fix'] + c['wrong']):6.1%}")

    def drop(payload, keep):
        out = copy.deepcopy(payload)
        for window in out["windows"]:
            for unit in window.get("parts") or [window]:
                unit["results"] = [x for x in unit.get("results") or [] if keep(x)]
        return out

    gates = {"none": (None, lambda x: True), "F1b": ("short", lambda x: True),
             "drop sound": (None, lambda x: "track" not in x or basis(x.get("why")) != "sound"),
             "F1b + drop sound": ("short", lambda x: "track" not in x or basis(x.get("why")) != "sound"),
             "F1b + drop sound outside cell": ("short", None)}
    table = defaultdict(Counter)
    for (name, k), (_, _, payload) in results.items():
        if O.pool_of(name) != "audio":
            continue
        win = wins[k]
        group = next(g for g, keys in R.GROUPS if k in keys)
        for gname, (gate, keep) in gates.items():
            if keep is None:
                shown = {str(key): R.cell_tracks(win.rows[i], marks=True) for i, key in enumerate(win.ids)}
                keep = lambda x, shown=shown: ("track" not in x or basis(x.get("why")) != "sound"  # noqa: E731
                                               or int(x["track"]) in shown[str(x["row"])])
            _, correct, _ = R.score(win, drop(payload, keep), gate=gate)
            table[group, gname].update(runs=1, whole=sum(correct.values()) - win.a_right)
    print("\nwhole window, per run-window (dev / held-out / merged)")
    for gname in gates:
        cells = [f"{table[g, gname]['whole'] / table[g, gname]['runs']:+6.1f}" if table[g, gname]["runs"] else "   -  "
                 for g, _ in R.GROUPS]
        print(f"  {gname:32} " + "  ".join(cells) + "   (" + "/".join(str(table[g, gname]["runs"]) for g, _ in R.GROUPS) + " run-windows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("items", "agree", "read"))
    args = ap.parse_args()
    if args.cmd == "agree":
        return cmd_agree()
    wins, results = R.load(None)
    (cmd_items if args.cmd == "items" else lambda r: cmd_read(wins, r))(results)


if __name__ == "__main__":
    main()
