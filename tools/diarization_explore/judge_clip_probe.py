"""Probe P1 (docs/plans/argon-audio-judge-plan.md): can agy Gemini 3.8 Flash tell the two voices apart on one row cut out
of the window, when inside the whole window it moved that row to the wrong person?

    python tools/diarization_explore/judge_clip_probe.py build [--pair cross|same]
    python tools/diarization_explore/judge_clip_probe.py run [--pair ..] --shard K --shards 4   # one process per agy lane
    python tools/diarization_explore/judge_clip_probe.py score [--pair ..]

`--pair cross` (P1, default) pairs people of opposite sex; `--pair same` (P1b) people of the same sex, and takes only
moves onto a track outside the row's cell (what F4 would recheck); its control partner is the most-speaking other
person of the same sex.

Items, all from the windows and audio runs of `judge_moves_readout.RUNS`, each a row and two people of opposite sex:

  wrong    a cross-sex wrong move of some audio run: the ruler's person (acoustic layer agrees) vs the judge's person
  fix      a cross-sex fix of some audio run: the ruler's person (the judge agrees) vs the acoustic layer's person
  control  a short acoustic-right row no audio run moved vs the most-speaking person of the other sex
           (`judge_f0_check`'s control rows)

Every item also carries the pitch witness's side (`judge_f0_check.side`). One call per item, audio only: two voice
samples of each person (their longest 2-5 s rows in the window that the ruler and the acoustic layer agree on, at least
AWAY seconds from the item), then the row with PAD seconds either side. Which person is 甲 is drawn per item (fixed
seed). The row's text is not given: the probe asks about the voice alone.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from string import Template

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_f0_check as P  # noqa: E402
import judge_moves_readout as M  # noqa: E402

ROOT = M.S.r9.MAIN / "tmp" / "judge-probe"
#: --pair -> output folder: the two people of an item are of opposite sex (P1) or of the same sex (P1b)
FOLDERS = {"cross": "p1", "same": "p1b"}
TARGET = "local-agy-gemini-3_8-flash"
SEED = 20261024
SAMPLE = (2.0, 5.0)
AWAY, PAD, GAP = 15.0, 0.25, 1.0
NAMES = ("甲", "乙")
CONTROL_PER_WINDOW = 10


def samples(win, person, avoid):
    rows = [i for i in win.scored if win.truth[i].person == person and win.right(win.acoustic, win.amap, i)
            and SAMPLE[0] <= win.seconds(i) <= SAMPLE[1] and abs(float(win.segs[i]["start"]) - avoid) > AWAY]
    return sorted(sorted(rows, key=win.seconds, reverse=True)[:2])


def pair_ok(pair, a, b):
    """Two sexes (`m` / `f` / None) make an item of this --pair."""
    if a not in ("m", "f") or b not in ("m", "f"):
        return False
    return (a != b) if pair == "cross" else (a == b)


def build(pair, out_dir):
    import soundfile as sf
    wins, results = M.load(None)
    items = {}
    for (name, key), (rows, _, _) in results.items():
        if M.RUNS[name][2] != "audio":
            continue
        win = wins[key]
        sexes = M.GENDER.get(win.window.ident, {})
        for i, r in rows.items():
            person = win.truth[i].person
            # P1b takes only moves onto a track outside the cell: those are what F4 would recheck
            if pair == "same" and not r["outside"]:
                continue
            if r["kind"] == "wrong" and pair_ok(pair, sexes.get(person), sexes.get(r["new"])):
                items.setdefault((key, i, "wrong"), dict(other=r["new"], runs=set()))["runs"].add(name)
            elif r["kind"] == "fix":
                old = win.amap.get(win.acoustic[i][0])
                if pair_ok(pair, sexes.get(person), sexes.get(old)):
                    items.setdefault((key, i, "fix"), dict(other=old, runs=set()))["runs"].add(name)
    moved_any = {(k, i) for (n, k), (rows, _, _) in results.items() if M.RUNS[n][2] == "audio" for i in rows}
    for key in sorted({k for k, _, _ in items}):
        win = wins[key]
        sexes = M.GENDER.get(win.window.ident, {})

        def partner(person):
            """The most-speaking other person this --pair allows."""
            pool = [p for p in win.weight if p != person and pair_ok(pair, sexes.get(person), sexes.get(p))]
            return max(pool, key=win.weight.get, default=None)
        picked = [i for i in win.scored if win.seconds(i) < P.CONTROL_SECONDS and (key, i) not in moved_any
                  and win.right(win.acoustic, win.amap, i) and partner(win.truth[i].person)]
        for i in picked[::max(1, len(picked) // CONTROL_PER_WINDOW)][:CONTROL_PER_WINDOW]:
            items[key, i, "control"] = dict(other=partner(win.truth[i].person), runs=set())
    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    by_key = defaultdict(list)
    for (key, i, kind), item in sorted(items.items()):
        by_key[key].append((i, kind, item))
    for key, todo in by_key.items():
        win, pitch = wins[key], P.Pitch(key)
        levels = {}
        wave = pitch.wave
        for i, kind, item in todo:
            person, other = win.truth[i].person, item["other"]
            seg = win.segs[i]
            picks = {p: samples(win, p, float(seg["start"])) for p in (person, other)}
            if any(len(v) < 2 for v in picks.values()):
                print(f"skip {key} {i} {kind}: fewer than two samples")
                continue
            for p in (person, other):
                if p not in levels:
                    values = [pitch.row(win.segs[j]) for j in picks[p]]
                    levels[p] = float(np.median([f for f, _ in values if f is not None])) if values else None
            order = [person, other] if rng.random() < 0.5 else [other, person]
            gap = np.zeros(int(GAP * P.SR), dtype=np.float32)
            pieces, layout, cursor = [], [], 0.0
            for name, p in zip(NAMES, order):
                for j in picks[p]:
                    piece = wave[int(float(win.segs[j]["start"]) * P.SR):int(float(win.segs[j]["end"]) * P.SR)]
                    layout.append(f"| {name}的声音样本 | {cursor:.1f}–{cursor + len(piece) / P.SR:.1f} 秒 |")
                    pieces += [piece, gap]
                    cursor += len(piece) / P.SR + GAP
            a, b = max(0.0, float(seg["start"]) - PAD), float(seg["end"]) + PAD
            piece = wave[int(a * P.SR):int(b * P.SR)]
            layout.append(f"| 待判的一句 | {cursor:.1f}–{cursor + len(piece) / P.SR:.1f} 秒 |")
            pieces.append(piece)
            ident = f"{key}-{i}-{kind}"
            sf.write(str(out_dir / f"{ident}.wav"), np.concatenate(pieces), P.SR, subtype="PCM_16")
            f, n = pitch.row(seg)
            out.append(dict(id=ident, key=key, row=i, kind=kind, runs=sorted(item["runs"]), seconds=round(win.seconds(i), 2),
                            text=seg.get("text", ""), src=win.truth[i].src, truth=person, other=other,
                            truth_name=NAMES[order.index(person)], layout="\n".join(layout),
                            audio_seconds=round(cursor + len(piece) / P.SR, 1),
                            pitch=P.side(f, n, levels.get(person), levels.get(other))))
    (out_dir / "items.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    c = Counter(x["kind"] for x in out)
    print(f"{len(out)} items: {dict(c)}; audio {sum(x['audio_seconds'] for x in out):.0f} s; pitch "
          + str({k: dict(Counter(x['pitch'] for x in out if x['kind'] == k)) for k in c}))


def parse(text):
    m = re.search(r"\{[^{}]*\}", text or "", re.S)
    if not m:
        return None
    try:
        reply = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return reply if str(reply.get("answer", "")).strip() in (*NAMES, "分不出") else None


def run(out_dir, shard, shards):
    import llm_retrack_av as av
    complete = av.client_for(TARGET)
    system = (HERE / "judge_clip_probe_system.md").read_text(encoding="utf-8")
    user = Template((HERE / "judge_clip_probe_user.md").read_text(encoding="utf-8"))
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    for item in items[shard::shards]:
        path = out_dir / f"{item['id']}.json"
        if path.exists():
            continue
        answer, text = None, ""
        for attempt in range(4):
            try:
                text = complete(system, user.substitute(layout=item["layout"]), out_dir / f"{item['id']}.wav")
            except SystemExit:
                raise
            except Exception as error:  # noqa: BLE001 - a transient provider error
                print(f"{item['id']}: {type(error).__name__} {str(error)[:80]}; retry", flush=True)
                time.sleep(10 * (attempt + 1))
                continue
            # ⚠ agy's answer channel is malformed now and then: retried, never scored.
            answer = parse(text)
            if answer is not None:
                break
            print(f"{item['id']}: malformed reply; retry", flush=True)
        path.write_text(json.dumps({"answer": answer, "raw": text}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{item['id']}: {answer['answer'] if answer else 'GAVE UP'}", flush=True)


def score(out_dir):
    items = json.loads((out_dir / "items.json").read_text(encoding="utf-8"))
    tally, picked_first = defaultdict(Counter), Counter()
    for item in items:
        path = out_dir / f"{item['id']}.json"
        got = json.loads(path.read_text(encoding="utf-8"))["answer"] if path.exists() else None
        if got is None:
            continue
        said = str(got["answer"]).strip()
        verdict = "unsure" if said == "分不出" else "right" if said == item["truth_name"] else "wrong"
        picked_first[said] += 1
        for group in (item["kind"], f"{item['kind']} pitch={item['pitch']}"):
            tally[group][verdict] += 1
        if item["kind"] != "control" and verdict != "right":
            print(f"  {item['kind']:5} {item['key'][:7]} {item['row']:4} {item['seconds']:4}s pitch {item['pitch']:7}"
                  f" | {item['truth']} vs {item['other']} -> {verdict} | {item['text'][:20]} | {got.get('why', '')[:50]}")
    for group in sorted(tally):
        c = tally[group]
        n = sum(c.values())
        print(f"{group:24} n {n:3}  right {c['right']:3} ({100 * c['right'] / n:3.0f}%)  wrong {c['wrong']:3}  unsure {c['unsure']:3}")
    print("answers:", dict(picked_first))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("build", "run", "score"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--pair", choices=sorted(FOLDERS), default="cross")
    args = ap.parse_args()
    out_dir = ROOT / FOLDERS[args.pair]
    if args.step == "build":
        build(args.pair, out_dir)
    elif args.step == "score":
        score(out_dir)
    else:
        run(out_dir, args.shard, args.shards)


if __name__ == "__main__":
    main()
