"""A listening page of the audio judge's moves, a few per category of `judge_moves_overlap.py` (owner 2026-11-14).

    python tools/diarization_explore/judge_moves_overlap_page.py [--thr 0.2]   # <MAIN>/out/judge-moves-overlap/

Draws, with a fixed seed, QUOTA distinct moves (window, row, the person moved to) per group -- wrong moves and fixes
of the audio pool, each in its category A-D -- at least APART seconds from another place of the same window. A move
several runs made is one place; its note says how many runs of the window made it and its lines show one run's reason.
The page is `explore/od8_overlap_review/build_page.py`'s: a clip per place (row +- PAD s from the vocal track the
judges heard), lanes aligned under a play head:

- ASR rows coloured by the person the row truth gives (the moved row alone in the next lane, orange);
- the acoustic layer's timeline, each track as the person it maps to (overlapping segments stack: that is the overlap);
- MAI2's diarized transcript (speakers split by sex only);
- the subtitle group's lines (the reference the ruler read).

A person keeps one colour across lanes and places (`colors`). The owner may mark per place who is right.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_overlap as O  # noqa: E402
import judge_moves_readout as R  # noqa: E402
import row_truth as rt  # noqa: E402

MAIN = R.S.r9.MAIN
OUT = MAIN / "out" / "judge-moves-overlap"
BUILD = HERE.parents[1] / "explore" / "od8_overlap_review"
SEED = 20261114
PAD, MAX_CLIP, APART = 3.0, 14.0, 12.0
QUOTA = {("wrong", "A"): 6, ("wrong", "B"): 6, ("wrong", "C"): 3, ("wrong", "D"): 6,
         ("fix", "A"): 5, ("fix", "B"): 2, ("fix", "C"): 2, ("fix", "D"): 5}
GROUPS = {f"{kind}-{c}": f"{'挪错' if kind == 'wrong' else '挪对'} {c}" for kind, c in QUOTA}
CAT_TEXT = {"A": "有叠音，挪给了叠音里的另一个声音", "B": "有叠音，挪给了叠音之外的人",
            "C": "没叠音，被挪去的人在这一行里出声", "D": "没叠音，被挪去的人在这一行里几乎没出声"}
PALETTE = ("#7b5ea7", "#2a9d8f", "#c08a1e", "#b5485d", "#3b7dd8", "#6a994e", "#8d6e63", "#5c6bc0")


def vocal(key):
    for folder in sorted({run[0] for run in R.RUNS.values()}):
        path = R.SPK / folder / key / "input" / f"{key}-vocal.ogg"
        if path.exists():
            return path
    raise FileNotFoundError(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.2)
    args = ap.parse_args()
    wins, results = R.load(None)
    feats = {key: O.Rows(wins[key], args.thr, O.mai2_segments(key.rsplit("-c", 1)[0])) for key in O.WINDOWS if key in wins}

    # distinct moves per group, with every audio run that made them
    moves = defaultdict(lambda: defaultdict(list))   # group -> (key, i, new) -> [(run, result)]
    runs_of = defaultdict(set)
    for (name, key), (rows, _, _) in results.items():
        if O.pool_of(name) != "audio" or key not in feats:
            continue
        runs_of[key].add(name)
        for i, r in rows.items():
            if r["kind"] in ("wrong", "fix"):
                c = O.category(feats[key].features(i, r["new"]))[0]
                moves[r["kind"], c][key, i, r["new"]].append((name, r))

    rng = random.Random(SEED)
    chosen = []
    for group, quota in QUOTA.items():
        pool = sorted(moves[group])
        rng.shuffle(pool)
        taken = []
        for key, i, new in pool:
            start = float(wins[key].segs[i]["start"])
            if any(k == key and abs(float(wins[k].segs[j]["start"]) - start) < APART for k, j, _, _ in chosen + taken):
                continue
            taken.append((key, i, new, group))
            if len(taken) == quota:
                break
        chosen += taken

    people = sorted({p for f in feats.values() for p in f.win.weight}, key=lambda p: -sum(f.win.weight[p] for f in feats.values()))
    colors = {p: PALETTE[k % len(PALETTE)] for k, p in enumerate(people)}
    refs = {key: rt.reference_lines(key.rsplit("-c", 1)[0]) for key in {k for k, *_ in chosen}}
    places = []
    for n, (key, i, new, (kind, c)) in enumerate(chosen, 1):
        f, win = feats[key], wins[key]
        s, e = float(win.segs[i]["start"]), float(win.segs[i]["end"])
        lo = max(0.0, s - PAD)
        hi = min(lo + MAX_CLIP, e + PAD)
        made = moves[kind, c][key, i, new]
        name, r = made[0]
        share, sim_people = f.simultaneous(i)
        cov = {p: round(v, 2) for p, v in f.people_cov(i).most_common()}
        in_clip = lambda a, b: a < hi and b > lo  # noqa: E731
        asr = [{"start": float(g["start"]), "end": float(g["end"]), "text": str(g.get("text", "")).strip(),
                "speaker": (win.truth[j].person or win.truth[j].cls) if j in win.truth else "（尺子外）"}
               for j, g in enumerate(win.segs) if in_clip(float(g["start"]), float(g["end"]))]
        diar = [{"start": t["start"], "end": t["end"], "text": f"s{t['track']}",
                 "speaker": win.amap.get(f"s{t['track']}", f"s{t['track']}（未配上）")}
                for t in f.timeline if in_clip(t["start"], t["end"])]
        lanes = [{"name": "ASR 行（颜色 = 行真值的人）", "rows": asr},
                 {"name": "被挪的那一行", "rows": [{"start": s, "end": e, "text": win.truth[i].text, "cls": "L"}]},
                 {"name": "声学层时间线（轨 → 对到的人；叠起来的就是同时出声）", "rows": diar}]
        if f.mai2 is not None:
            segs, offset = f.mai2
            lanes.append({"name": "MAI2（只按性别分人）", "rows": [
                {"start": g["start"] + offset, "end": g["end"] + offset, "text": g.get("text", ""), "speaker": f"MAI2 {g['speaker']}"}
                for g in segs if in_clip(g["start"] + offset, g["end"] + offset)]})
        lanes.append({"name": "字幕组（尺子读的参照）", "rows": [
            {"start": a, "end": b, "text": text, "speaker": who.removesuffix(" 日").removesuffix(" 中")}
            for a, b, who, text in refs[key] if in_clip(a, b)]})
        places.append({
            "n": n, "id": f"{key} 第 {win.ids[i]} 行", "at": f"{s:.1f}s", "group": f"{kind}-{c}",
            "note": f"{len(made)} / {len(runs_of[key])} 遍带音频运行这样挪",
            "lines": [
                ["l", "被挪的行", f"「{win.truth[i].text}」 {e - s:.2f} 秒"],
                ["", "行真值", f"{win.truth[i].person}；声学层 {r['old']}（{win.amap.get(r['old'])}）"
                               f" → 裁判 {r['new_label']}（{new}）：{'挪错' if kind == 'wrong' else '挪对'}"],
                ["", f"{c} 类", f"{CAT_TEXT[c]}；两人同时出声占这一行 {share:.0%}"
                                + (f"（{'、'.join(sorted(sim_people))}）" if sim_people else "")
                                + f"；各人覆盖 {json.dumps(cov, ensure_ascii=False)}"],
                ["b", f"理由（{name}）", r["why"] or "（没写）"],
                ["", "做了这一挪的运行", "、".join(sorted(m for m, _ in made))],
            ],
            "lanes": lanes, "audio": str(vocal(key)), "lo": lo, "hi": hi})

    counts = defaultdict(int)
    for p in places:
        counts[p["group"]] += 1
    spec = {"title": "带音频说话人裁判：挪错与挪对各类的样本（2026-11-14）",
            "summary": f"共 {len(places)} 处：" + "、".join(f"{GROUPS[g]} {n}" for g, n in counts.items())
                       + f"；叠音阈值 {args.thr:.0%}",
            "help": "每处是一次带音频裁判的挪动（同一次挪动多遍运行都做过的只算一处）。"
                    "四类：A 有叠音、挪给了叠音里的另一个声音；B 有叠音、挪给了叠音之外的人；"
                    "C 没叠音、被挪去的人在这一行里出声；D 没叠音、被挪去的人在这一行里几乎没出声。"
                    "<b style=\"color:var(--L)\">橙色</b>是被挪的那一行；同一个人在各条轨道上同一个颜色；"
                    "点时间轴从那里播放，空格暂停。对错按行真值（3.8 Flash 带音频标的）；"
                    "想标的话，选「行真值对」或「裁判对」，可加标签。",
            "key": "judge-moves-overlap-2026-11-14",
            "choices": [["truth", "行真值对"], ["judge", "裁判对"], ["neither", "都不对"], ["unsure", "听不出"]],
            "tags": ["真同时说", "笑声 / 附和", "声学层串轨", "片段坏"],
            "groups": GROUPS, "colors": colors, "places": places}
    OUT.mkdir(parents=True, exist_ok=True)
    spec_path = OUT / "spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    sys.path.insert(0, str(BUILD))
    import build_page  # noqa: E402
    build_page.main(str(spec_path), str(OUT / "judge-moves-overlap.html"))


if __name__ == "__main__":
    main()
