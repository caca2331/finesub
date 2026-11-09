"""P8 (docs/plans/argon-audio-judge-plan.md 「P8 回到出错的会话里追问」): ask agy 3.8 Flash what it heard on the rows
its speaker judge moved onto the other sex, back in the judge's own session (R) and in fresh sessions (F).

    python tools/diarization_explore/agy_judge_interrogate.py build
    python tools/diarization_explore/agy_judge_interrogate.py feas                  # resume one session, two controls
    python tools/diarization_explore/agy_judge_interrogate.py r1|r2 <label,...>     # labels from SESSIONS
    python tools/diarization_explore/agy_judge_interrogate.py --code <snapshot> f <bare|split> <n>
    python tools/diarization_explore/agy_judge_interrogate.py score

hina-46-c0001's second judge window (9:14). Targets: the rows the four sessions moved onto the other sex; controls: two
per target, the target's true sex and length band, acoustic-right rows no audio judge moved, nearest in time. R asks
sex and nearest voice sample only (its context holds the row texts); F gets the rows with the targets and controls
blanked and dictates them too. F runs the trial driver of its form: pass `--code` = the P7 snapshot
(`tmp/spk-merge/accept/code-88811913` bare, `code-b3355c57` split), which then supplies `finesub`. Answers and agy's
per-request ledger go to <MAIN>/tmp/p8-interrogate/answers.json.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
#: as `accept_3pass.py` ran the judges: agy's capsules then land under the project the judge sessions were opened in,
#: and a resumed session can read its new task file
os.environ.setdefault("FINESUB_ROOT", str(HERE.parents[1]))
CODE = Path(sys.argv[sys.argv.index("--code") + 1]) if "--code" in sys.argv else HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE / "src"))

MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
SPK = MAIN / "tmp" / "spk-merge"
sys.path.insert(0, str(MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16"))
from agy_media_resume_probe_ledger import ledger  # noqa: E402

WORK = MAIN / "tmp" / "p8-interrogate"
TEMPLATES = HERE / "p8_templates"
KEY, UNIT = "hina-46-c0001", "0002"
#: label -> (readout run, agy session)
SESSIONS = {
    "bare-r2": ("38F bare r2", "33057531-9ecc-449f-9364-072d0a84579b"),
    "bare-r4": ("38F bare r4", "eff4b4e8-cbf5-4d25-972f-a539dd7d1472"),
    "split-r1": ("38F split r1", "85ae6cb7-b80b-4543-a611-8c6941a63fe7"),
    "split-r2": ("38F split r2", "b10a162b-93c0-4413-a979-c44f2f356f1b"),
}
TARGETS = ("171", "189", "192", "214", "221", "291", "295", "337")
#: in the ruler's unheard tail, pitch with the judge: read apart
SUSPECT = ("337",)
#: a control shorter than this has too little voice to answer on
MIN_SECONDS = 0.3
CLIP = SPK / "accept" / "audio-rbare2" / KEY / "llm" / "clips" / f"speaker-judge-{UNIT}.aac"
PROMPT = SPK / "accept" / "audio-rbare2" / KEY / "llm" / "exchanges" / f"002-speaker_judge-{UNIT}-attempt0.md"
ROW = re.compile(r"^(?:▶ q(\d+) )?([\d.]+)\|([\d.]+)\|[\d.]+\|[^|]*\|(.*)$")


def band(seconds):
    return "<1" if seconds < 1 else "1-2" if seconds < 2 else ">=2"


def prompt_parts():
    """The judge prompt's layout table and its rows (id or '', start, length, text)."""
    user = PROMPT.read_text(encoding="utf-8").split("## 请求（user）", 1)[1].split("## 模型响应", 1)[0]
    lines = user.splitlines()
    at = lines.index("音频布局：")
    layout = []
    for line in lines[at + 1:]:
        if not line.strip():
            break
        layout.append(line)
    rows = [(m.group(1) or "", float(m.group(2)), float(m.group(3)), m.group(4).strip())
            for m in map(ROW.match, lines) if m]
    return "\n".join(layout), rows


def build():
    import judge_moves_readout as M  # loads finesub from this checkout: only build and score need it
    wins, results = M.load(None)
    win = wins[KEY]
    index = {k: i for i, k in enumerate(win.ids)}
    sex = M.GENDER["hina-46"]
    _, rows = prompt_parts()
    asked = {r[0]: r for r in rows if r[0]}
    moved = set()
    for (name, key), (got, _, _) in results.items():
        if key == KEY and M.RUNS[name][2] == "audio":
            moved |= set(got)
    targets = []
    for tid in TARGETS:
        i = index[tid]
        targets.append({"id": tid, "truth": win.truth[i].person, "sex": sex[win.truth[i].person],
                        "seconds": round(win.seconds(i), 2), "text": win.truth[i].text})
    used, controls = set(TARGETS), []
    for t in targets:
        i = index[t["id"]]
        pool = [j for j in win.scored if win.ids[j] in asked and win.ids[j] not in used and j not in moved
                and win.right(win.acoustic, win.amap, j) and sex.get(win.truth[j].person) == t["sex"]
                and band(win.seconds(j)) == band(t["seconds"]) and win.seconds(j) >= MIN_SECONDS]
        pool.sort(key=lambda j: abs(float(win.segs[j]["start"]) - float(win.segs[i]["start"])))
        for j in pool[:2]:
            used.add(win.ids[j])
            controls.append({"id": win.ids[j], "for": t["id"], "truth": win.truth[j].person, "sex": t["sex"],
                             "seconds": round(win.seconds(j), 2), "text": win.truth[j].text})
    sessions = {}
    for label, (run, sid) in SESSIONS.items():
        got = results[run, KEY][0]
        moves, decoys = [], []
        for i, r in sorted(got.items()):
            if win.ids[i] not in asked:
                continue
            entry = {"id": win.ids[i], "old": r["old"], "new": r["new_label"], "why": r["why"]}
            if r["kind"] == "wrong" and win.ids[i] in TARGETS and r["obvious"]:
                moves.append(entry)
            elif r["kind"] == "fix":
                decoys.append(entry)
        sessions[label] = {"run": run, "session": sid, "moves": moves, "decoys": decoys[:2]}
    WORK.mkdir(parents=True, exist_ok=True)
    out = {"targets": targets, "controls": controls, "sessions": sessions,
           "tracks": {r[0]: None for r in rows if r[0] in used}}
    (WORK / "materials.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for t in targets:
        cs = [c for c in controls if c["for"] == t["id"]]
        print(t["id"], t["truth"], t["sex"], t["seconds"], t["text"], "| controls", [(c["id"], c["truth"], c["seconds"]) for c in cs])
    for label, s in sessions.items():
        print(label, "moves", [m["id"] for m in s["moves"]], "decoys", [d["id"] for d in s["decoys"]])


def materials():
    return json.loads((WORK / "materials.json").read_text(encoding="utf-8"))


def ask_lines(ids, seed):
    _, rows = prompt_parts()
    at = {r[0]: r for r in rows if r[0]}
    ids = list(ids)
    random.Random(seed).shuffle(ids)
    return [f"q{i}｜{at[i][1]:.1f}–{at[i][1] + at[i][2]:.1f} 秒" for i in ids], ids


def template(name, **values):
    return Template((TEMPLATES / name).read_text(encoding="utf-8")).substitute(values)


def driver():
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    return AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=900))


def record(row):
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "answers.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in row.items() if k != "prompt"}, ensure_ascii=False), flush=True)


def resume(label, turn, text):
    sid = SESSIONS[label][1]
    started = time.time()
    result = driver().run([{"role": "user", "content": text}], task=f"p8-{turn}", session_scope="assignment",
                          conversation_key=f"p8-{label}-{turn}-{int(started)}", conversation_handle=sid,
                          continuation=True)
    record({"arm": "R", "turn": turn, "label": label, "conversation": result.conversation_handle,
            "same_session": result.conversation_handle == sid, "answer": result.content.strip(),
            "seconds": round(time.time() - started, 1), "ledger": ledger(sid), "prompt": text})


def r1(labels, ids=None):
    m = materials()
    for label in labels:
        lines, _ = ask_lines(ids or [t["id"] for t in m["targets"]] + [c["id"] for c in m["controls"]], f"r1-{label}")
        resume(label, "r1" if ids is None else "feas", template("r1_user_v1.md", asks="\n".join(lines), count=len(lines)))


def r2(labels):
    m = materials()
    _, rows = prompt_parts()
    at = {r[0]: r for r in rows if r[0]}
    for label in labels:
        s = m["sessions"][label]
        items = s["moves"] + s["decoys"]
        random.Random(f"r2-{label}").shuffle(items)
        lines = [f"q{x['id']}｜{at[x['id']][1]:.1f}–{at[x['id']][1] + at[x['id']][2]:.1f} 秒｜{x['old']}｜{x['new']}｜{x['why']}"
                 for x in items]
        resume(label, "r2", template("r2_user_v1.md", moves="\n".join(lines), count=len(lines)))


def fresh(form, n):
    m = materials()
    layout, rows = prompt_parts()
    blank = {t["id"] for t in m["targets"]} | {c["id"] for c in m["controls"]}
    table = "\n".join(f"{('q' + i) if i else '—'}｜{s:.1f}｜{d:.1f}｜{'＿＿' if i in blank else text}"
                      for i, s, d, text in rows)
    for k in range(n):
        label = f"F-{form}-{k + 1}"
        lines, _ = ask_lines(sorted(blank, key=int), label)
        text = template("f_user_v1.md", layout=layout, rows=table, asks="\n".join(lines), count=len(lines))
        user = {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "file", "file": {"file_id": CLIP.name, "filename": CLIP.name,
                                                                      "format": "audio/aac", "local_path": str(CLIP)}}]}
        started = time.time()
        result = driver().run([{"role": "system", "content": template("f_system_v1.md")}, user], task="p8-f",
                              session_scope="assignment", conversation_key=f"p8-{label}-{int(started)}",
                              conversation_handle="")
        record({"arm": "F", "turn": "f", "label": label, "form": form, "code": str(CODE),
                "conversation": result.conversation_handle, "answer": result.content.strip(),
                "seconds": round(time.time() - started, 1), "ledger": ledger(result.conversation_handle),
                "prompt": text})


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if a != "--code" and sys.argv[i - 1] != "--code"]
    cmd = args[0]
    if cmd == "build":
        build()
    elif cmd == "feas":
        m = materials()
        r1(["bare-r2"], ids=[m["controls"][0]["id"], m["controls"][1]["id"]])
    elif cmd == "r1":
        r1(args[1].split(","))
    elif cmd == "r2":
        r2(args[1].split(","))
    elif cmd == "f":
        fresh(args[1], int(args[2]))
    elif cmd == "score":
        import agy_judge_interrogate_score
        agy_judge_interrogate_score.main()


if __name__ == "__main__":
    main()
