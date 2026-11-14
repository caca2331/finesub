"""P13g (docs/plans/argon-audio-judge-plan.md 「P13g 设计单」): back in arm A's sessions, tell each what its patch turn
did -- rows it broke, rows the row-split truth splits that it left whole, candidate rows still wrong whose ruler person
is on another candidate track -- and ask why it did not split or move them, how it listened, how much the draft's
reasoning steered it, and how the candidate and split wording should read.

    python tools/diarization_explore/p13g_ask.py show
    python tools/diarization_explore/p13g_ask.py ask <label,...>

Rows are told by whole-recording seconds, with the attachment and its own seconds (p13_ask.pieces). A split row's
people are told as the tracks the acoustic map gives them (preferring a track in the row's cell); a person on no track
is told as such. Same-sex or other-sex is said from GENDER; no pitch or MAI2 witness is read for this round, so rows
say so. Answers and agy's per-request ledger go to <MAIN>/tmp/p13g-ask/answers.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from string import Template

import p13_ask as P
from judge_ask import HERE, MAIN, ledger

#: label -> (round, window key, patch exchange file, agy session); arm A's patched sessions (r2's hina-46 unit 2 was
#: rebuilt in a fresh session, its unit 1 has little to ask)
SESSIONS = {
    "r1-h22-u1": ("p13g1", "hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "4c0c60e6-0228-4286-90ca-2b3d90608573"),
    "r1-h22-u2": ("p13g1", "hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "1901569b-cfb4-4c63-a05e-bd6598a8f4f6"),
    "r1-h46-u1": ("p13g1", "hina-46-c0001", "002-speaker_judge-0001-attempt0-patch.md", "d519ec3a-3ad6-4969-9398-97346cbba13a"),
    "r1-h46-u2": ("p13g1", "hina-46-c0001", "004-speaker_judge-0002-attempt0-patch.md", "2f0e9e70-8012-4528-b2bd-452d771ca0e9"),
    "r2-h22-u1": ("p13g2", "hina-22-c0102", "002-speaker_judge-0001-attempt0-patch.md", "185f9529-4168-433c-a780-4ef716f9afd2"),
    "r2-h22-u2": ("p13g2", "hina-22-c0102", "004-speaker_judge-0002-attempt0-patch.md", "3dbb85b6-fa52-4315-861b-d3c18cfc5ef1"),
}
TEMPLATE = "ask_patch_v5.md"
POSITION = re.compile(r"^q(\d+)｜([\d.]+)–([\d.]+)(?:｜.*)?$")
MAX_ROWS = 12  # per table; the rest are counted, not listed


def _load():
    import judge_moves_readout as M
    import row_split as R
    import row_truth_score as S
    from finesub.llm.stages.speaker_judge.prompt import cell_tracks
    items, _ = R.load("1")
    split = {(it["ident"], it["i"]): it["truth"] for it in items if it["truth"] and it["truth"].get("kind") == "split"}
    wins, results = M.load(None)
    return M, S, cell_tracks, split, wins, results


M, S, cell_tracks, SPLIT, WINS, RESULTS = _load()


def units(payload):
    return [u for w in payload["windows"] for u in (w.get("parts") or [w])]


def k_of(key):
    ident = key.rsplit("-c", 1)[0]
    at = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(S.stable(ident))}
    return ident, {i: at.get((round(s["start"], 3), round(s["end"], 3))) for i, s in enumerate(WINS[key].segs)}


def track_of(win, person, cell):
    """The track the acoustic map gives a person, one in the row's cell first; None when no track is theirs."""
    tracks = [t for t, p in win.amap.items() if p == person]
    inside = [t for t in tracks if int(str(t).lstrip("s")) in cell]
    pick = (inside or tracks or [None])[0]
    return None if pick is None else f"s{str(pick).lstrip('s')}"


def sex(win, a, b):
    g = M.GENDER.get(win.window.ident, {})
    if not a or not b or not g.get(a) or not g.get(b):
        return ""
    return "（男女不同）" if g[a] != g[b] else "（同性别）"


def candidates(row):
    tracks = list(dict.fromkeys(cell_tracks(row, marks=True)))  # an alternative may repeat a parallel track
    return tracks if len(tracks) > 1 else []


def classify(key, rnd, unit):
    """Rows of one unit: broke (draft right, patch wrong), unsplit (truth split, no split placed), left (a candidate row
    the draft left, still wrong, whose ruler person is on another candidate track), right (a truth split row split
    within a cut's ±1 character)."""
    win = WINS[key]
    patched, draft = f"38F {rnd} r1", f"38F {rnd} draft"
    d, p = RESULTS[draft, key], RESULTS[patched, key]
    ident, kmap = k_of(key)
    lo, hi = int(unit["first"]), int(unit["last"])
    touched = {str(r["row"]) for r in unit["patch"]["draft"]}
    split_done = {str(s["row"]): s for s in unit.get("splits") or []}
    out = {"broke": [], "unsplit": [], "left": [], "right": []}
    for i, s in enumerate(win.segs):
        rid = win.ids[i]
        if not lo <= int(rid) <= hi:
            continue
        truth = SPLIT.get((ident, kmap[i]))
        if truth and rid not in split_done:
            out["unsplit"].append((i, truth))
        elif truth:
            cut = sum(len(x["word"]) for x in s["words"][:split_done[rid]["at"]])
            if any(a - 1 <= cut <= b + 1 for a, b in truth["cuts"]):
                out["right"].append((i, split_done[rid]))
        if i not in win.scored:
            continue
        if d[1][i] and not p[1][i]:
            out["broke"].append(i)
        cands = candidates(win.rows[i])
        if rid not in touched and cands and not p[1][i]:
            want = track_of(win, win.truth[i].person, cands)
            if want and int(want[1:]) in cands and int(want[1:]) != cands[0]:
                out["left"].append((i, want))
    return out


def summary():
    c = Counter()
    for rnd in ("p13g1", "p13g2"):
        patched, draft = f"38F {rnd} r1", f"38F {rnd} draft"
        for key in M.HELDOUT:
            win = WINS[key]
            for r in RESULTS[draft, key][0].values():
                c["d " + r["kind"]] += 1
            for r in RESULTS[patched, key][0].values():
                c["p " + r["kind"]] += 1
            for u in units(RESULTS[patched, key][2]):
                pt = u.get("patch") or {}
                c["c_proposed"] += (pt.get("proposed") or {}).get("candidate", 0)
                if pt.get("status") != "success":
                    continue
                touched = {str(r["row"]) for r in pt["draft"]}
                for i in win.scored:
                    rid = win.ids[i]
                    if not int(u["first"]) <= int(rid) <= int(u["last"]):
                        continue
                    dr, pr = RESULTS[draft, key][1][i], RESULTS[patched, key][1][i]
                    if dr != pr:
                        kind = "t" if rid in touched else "c" if candidates(win.rows[i]) else "o"
                        c[f"{kind}_{'broke' if dr else 'fixed'}"] += 1
                for kind in ("left",):
                    c["c_left"] += len(classify(key, rnd, u)[kind])
            ident, kmap = k_of(key)
            c["truth_splits"] += sum((ident, kmap[i]) in SPLIT for i in range(len(win.segs)))
            index = {r: i for i, r in enumerate(win.ids)}
            for u in units(RESULTS[patched, key][2]):
                for sp in u.get("splits") or []:
                    i = index[str(sp["row"])]
                    t = SPLIT.get((ident, kmap[i]))
                    if t:
                        cut = sum(len(x["word"]) for x in win.segs[i]["words"][:sp["at"]])
                        c["splits_right"] += any(a - 1 <= cut <= b + 1 for a, b in t["cuts"])
    net = lambda pre: f"{c[pre + ' fix'] - c[pre + ' wrong']:+d}"  # noqa: E731
    return dict(d_net=net("d"), p_net=net("p"), t_fixed=c["t_fixed"], t_broke=c["t_broke"],
                c_proposed=c["c_proposed"], c_fixed=c["c_fixed"], c_broke=c["c_broke"], c_left=c["c_left"],
                truth_splits=c["truth_splits"] // 2, splits_right=c["splits_right"])


def told(label):
    rnd, key, exchange, sid = SESSIONS[label]
    win = WINS[key]
    unit = next(u for u in units(RESULTS[f"38F {rnd} r1", key][2])
                if exchange.split("-")[2] == u["id"])
    text = (M.SPK / "accept" / f"audio-r{rnd}" / key / "llm" / "exchanges" / exchange).read_text(encoding="utf-8")
    at = {m.group(1): (float(m.group(2)), float(m.group(3)))
          for m in map(POSITION.match, text.split("## 模型响应", 1)[0].splitlines()) if m}
    cut = P.pieces(sid)
    reply = text.split("## 模型响应", 1)[1]
    said, split_said = {}, {}
    for block in re.findall(r"<judge>(.*?)</judge>", reply, re.S)[-1:]:
        for line in block.splitlines():
            try:
                entry = json.loads(line)
                (split_said if "split" in entry else said)[str(entry["id"]).lstrip("q")] = str(entry.get("why", ""))
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
    rows = classify(key, rnd, unit)
    d, p = RESULTS[f"38F {rnd} draft", key], RESULTS[f"38F {rnd} r1", key]

    def head(i):
        rid = win.ids[i]
        s, e = at[rid]
        return f"q{rid}｜{P.where(s, e, cut)}｜{win.segs[i]['text']}"

    broke = []
    for i in rows["broke"]:
        rid = win.ids[i]
        acoustic = win.acoustic[i][0]
        dl = (d[0].get(i) or {}).get("new_label", acoustic)
        pl = (p[0].get(i) or {}).get("new_label", acoustic)
        new = (p[0].get(i) or {}).get("new")
        basis = P.SUSPECT.get((key, rid)) or f"标注认为初稿对{sex(win, win.truth[i].person, new) if new else ''}"
        broke.append(f"{head(i)}｜{win.seconds(i):.2f} 秒｜{dl}｜{pl}｜{said.get(rid, '')}｜{basis}")
    unsplit = []
    for i, truth in rows["unsplit"]:
        words = "".join(x["word"] for x in win.segs[i]["words"])
        a, b = truth["cuts"][0]
        cell = list(dict.fromkeys(cell_tracks(win.rows[i], marks=True)))
        first, second = truth["people"][0], truth["people"][1]
        who = lambda people: "、".join(track_of(win, x, cell) or "不在任何轨上的人" for x in people)  # noqa: E731
        more = f"（共换了 {truth['nparts'] - 1} 次人，这里只写第一处）" if truth["nparts"] > 2 else ""
        part = lambda people: f"{who(people)} 的人" + ("齐声" if len(people) > 1 else "")  # noqa: E731
        alone = len(first) == len(second) == 1
        unsplit.append(f"{head(i)}｜{'、'.join(f's{t}' for t in cell)}｜「{words[:a]}」之后换人：前面是 {part(first)}，"
                       f"后面是 {part(second)}{sex(win, first[0], second[0]) if alone else ''}{more}")
    left = []
    for i, want in rows["left"]:
        cands = candidates(win.rows[i])
        left.append(f"{head(i)}｜{win.seconds(i):.2f} 秒｜{'、'.join(f's{t}' for t in cands)}（原轨 s{cands[0]}）｜{want}"
                    f"{sex(win, win.truth[i].person, win.amap.get(cands[0]) or win.amap.get(str(cands[0])))}")

    right = [f"{head(i)}｜「{split['prefix']}」之后｜{split_said.get(win.ids[i], '')}" for i, split in rows["right"]]

    def table(lines):
        if not lines:
            return "（没有）"
        rest = len(lines) - MAX_ROWS
        return "\n".join(lines[:MAX_ROWS]) + (f"\n（另有 {rest} 行，不一一列出）" if rest > 0 else "")

    return dict(summary(), broke_count=len(broke), broke=table(broke), unsplit_count=len(unsplit),
                unsplit=table(unsplit), left_count=len(left), left=table(left),
                right_count=len(right), right=table(right)), sid


def render(label):
    fields, sid = told(label)
    return Template((HERE / "judge_ask_templates" / TEMPLATE).read_text(encoding="utf-8")).substitute(fields), sid


def ask(labels):
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    work = MAIN / "tmp" / "p13g-ask"
    work.mkdir(parents=True, exist_ok=True)
    for label in labels:
        text, sid = render(label)
        started = time.time()
        driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=1200))
        result = driver.run([{"role": "user", "content": text}], task="p13g-ask", session_scope="assignment",
                            conversation_key=f"p13g-ask-{label}-{int(started)}", conversation_handle=sid,
                            continuation=True)
        row = {"label": label, "conversation": result.conversation_handle, "same_session": result.conversation_handle == sid,
               "answer": result.content.strip(), "seconds": round(time.time() - started, 1), "ledger": ledger(sid),
               "prompt": text}
        out = work / "answers.json"
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k not in ("answer", "prompt")}, ensure_ascii=False), flush=True)


def main():
    if sys.argv[1] == "show":
        for label in SESSIONS:
            text, sid = render(label)
            print(f"===== {label} ({sid})\n{text}\n")
    else:
        ask(sys.argv[2].split(","))


if __name__ == "__main__":
    main()
