"""P12 (docs/plans/argon-audio-judge-plan.md 「P12 迭代改 prompt」): back in the sessions of a prompt round's run, tell
each judge what changed against the old prompt (rows it now leaves alone, fixes it lost) and which rows it moved
wrongly, ask what it heard and how the audio note should be rewritten. Owner 2026-10-30: say plainly where it erred.

    python tools/diarization_explore/judge_ask.py <round> show            # what each session will be told
    python tools/diarization_explore/judge_ask.py <round> ask <label,...>

Rows told as wrong, by what backs the ruler (other-sex moves no witness backs the ruler on are left out: the ruler may
well be the one that is wrong there); each says in how many of the old prompt's three runs that row was also wrong:
- hina-46 rows the owner heard (P8);
- other-sex moves: pitch (f0.csv of the round's readout) and MAI2, both, one, or split;
- same-sex moves: the ruler alone (pitch and MAI2 only tell the sexes apart), listed apart and asked about together.
Against the old prompt (P7's three split runs): rows it moved wrongly in at least two of them that this run leaves
alone, and rows it fixed in at least two that this run does not. Answers and agy's per-request ledger go to
<MAIN>/tmp/<round>-ask/answers.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
os.environ.setdefault("FINESUB_ROOT", str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
sys.path.insert(0, str(MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16"))
from agy_media_resume_probe_ledger import ledger  # noqa: E402

#: round -> (readout run name, run dir under tmp/spk-merge/accept, readout dir holding f0.csv, template,
#:           {label: (window key, judge unit, exchange file of the answer kept, agy session)})
ROUNDS = {
    "p12": ("38F heard r1", "audio-rheard1", MAIN / "tmp" / "p11-heard" / "readout", "ask_user_v2.md", {
        "h22-u1": ("hina-22-c0102", "0001", "002-speaker_judge-0001-attempt1.md", "f3b4edb0-dda6-4bb1-b93d-d8f3b4a38eca"),
        "h22-u2": ("hina-22-c0102", "0002", "003-speaker_judge-0002-attempt0.md", "e871920e-b64b-4062-bba3-f201b15e9bbe"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "ae48d45c-0850-4c87-8268-919bade550d4"),
    }),
    "p12r2": ("38F iter1 r1", "audio-riter1", MAIN / "tmp" / "p12-iter1" / "readout", "ask_user_v3.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "d26fdda7-59a5-4667-8096-44e577a7e72a"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "021a40fe-ca19-4c3f-a5ab-a0926a2ed0ae"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "3c063751-d2c5-4af1-83bc-8bd4eab5ba7b"),
    }),
    "p12r3": ("38F iter2 r1", "audio-riter2", MAIN / "tmp" / "p12-iter2" / "readout", "ask_user_v4.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "aa4d82b9-69c1-4133-bc12-7512a2d73e54"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "ccad26d8-c99d-4996-b8e2-ccc07c6a11d7"),
        "h46-u1": ("hina-46-c0001", "0001", "001-speaker_judge-0001-attempt0.md", "7683ced5-d82c-4222-81cc-e259e6123e5d"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "156b8879-7005-4c17-8984-e6f03101ded3"),
    }),
    "p12r4": ("38F iter3 r1", "audio-riter3", MAIN / "tmp" / "p12-iter3" / "readout", "ask_user_v5.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "d126638f-ca2d-4915-8231-e81ad654809d"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "6130d2be-3129-4e27-8a7d-1d4dbd944d03"),
        "h46-u1": ("hina-46-c0001", "0001", "001-speaker_judge-0001-attempt0.md", "46261816-3881-4d62-a93d-7ff00d116632"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "1ea3f14c-b399-4eaf-94e9-c2308e561bbb"),
    }),
    "p12r5": ("38F iter4 r1", "audio-riter4", MAIN / "tmp" / "p12-iter4" / "readout", "ask_user_v6.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "afcc0221-751b-4633-8ce3-23ecb4ac9e45"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "8727ae5d-96ba-4086-86fd-068fa3d49de3"),
        "h46-u1": ("hina-46-c0001", "0001", "001-speaker_judge-0001-attempt0.md", "d4c71fbd-b898-4b0d-b9d0-3eff49b8270a"),
        "h46-u2": ("hina-46-c0001", "0002", "003-speaker_judge-0002-attempt1.md", "2d0b0998-9f91-4c22-ac97-ec8a97be693a"),
    }),
    "p12r6": ("38F iter5 r1", "audio-riter5", MAIN / "tmp" / "p12-iter5" / "readout", "ask_user_v7.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "398299ba-2b38-4dba-b55d-54409a4e3028"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "7267b52e-458d-4286-87a0-fa5bf6f8a4d3"),
        "h46-u1": ("hina-46-c0001", "0001", "001-speaker_judge-0001-attempt0.md", "cdf35172-fe94-4252-b11b-5b3feeefa28f"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "29139b9d-05cd-4751-a627-59b49cd6d084"),
    }),
    "p12r7": ("38F iter6 r1", "audio-riter6", MAIN / "tmp" / "p12-iter6" / "readout", "ask_user_v8.md", {
        "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "29b3fb27-049e-4adf-8a70-9f5f2dc9fdde"),
        "h22-u2": ("hina-22-c0102", "0002", "002-speaker_judge-0002-attempt0.md", "0e939e93-b93e-41e4-93a6-df879bb91ed3"),
        "h46-u1": ("hina-46-c0001", "0001", "001-speaker_judge-0001-attempt0.md", "4e8faf0b-06f8-4264-9e32-7902216f6e6e"),
        "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "1f2c588b-1da0-484e-9926-1aa26d893307"),
    }),
}
BLIND_SECONDS = 1.5  # rows this short go into the blind table (round 5 on)
OWNER_HEARD = {"hina-46-c0001"}  # P8's rows there are owner-heard (ruler right)
ROW = re.compile(r"^▶ q(\d+) ([\d.]+)\|([\d.]+)\|")
BASIS = {
    ("ruler", "ruler"): "基频与另一套分人系统都站原来那个人",
    "one": "一个旁证（基频或另一套分人系统）站原来那个人，另一个判不了",
    "split": "旁证不一（基频与另一套分人系统一个站原来那个人、一个站你），可能是标注错了，请照实说",
    "same": "改前改后同性别，只有逐行标注（基频与另一套分人系统分不出），可能是标注错了，请照实说",
}


def basis(witness, key, i, r, truth):
    """Why we believe the ruler on this wrong move, or None to leave the row out."""
    import p9_readout as P
    if key in OWNER_HEARD and r["id"] in P.P8:
        return "人工逐行听过，确实是原来那个人说的"
    if not r["obvious"]:
        return BASIS["same"]
    got = witness(key, i, r["new"], truth)
    if got == ("ruler", "ruler"):
        return BASIS[got]
    if "ruler" not in got:
        return None
    return BASIS["split"] if "judge" in got else BASIS["one"]


def told(round_, label):
    import judge_moves_readout as M
    import p9_readout as P
    name, run, readout, template, sessions = ROUNDS[round_]
    key, unit, exchange, sid = sessions[label]
    wins, results = M.load(None)
    win = wins[key]
    got = results[name, key][0]
    base = [results[b, key][0] for b in P.BASE]
    witness = P.witnesses(readout)
    index = {k: i for i, k in enumerate(win.ids)}
    text = (M.SPK / "accept" / run / key / "llm" / "exchanges" / exchange).read_text(encoding="utf-8")
    at = {m.group(1): (float(m.group(2)), float(m.group(3)))
          for m in map(ROW.match, text.split("## 模型响应", 1)[0].splitlines()) if m}
    # a template with $blind gets the short rows the old version often moved as one unlabeled table, scored offline
    blind_short = "$blind" in (HERE / "judge_ask_templates" / template).read_text(encoding="utf-8")
    wrong, same, better, lost, blind = [], [], [], [], []
    for rid, (s, d) in at.items():
        i = index[rid]
        line = f"q{rid}｜{s:.1f}–{s + d:.1f}｜{win.segs[i]['text']}"
        r = got.get(i)
        old_wrong = [b[i]["new_label"] for b in base if i in b and b[i]["kind"] == "wrong"]
        old_fix = [b[i]["new_label"] for b in base if i in b and b[i]["kind"] == "fix"]
        if blind_short and win.seconds(i) <= BLIND_SECONDS and (len(old_wrong) >= 2 or len(old_fix) >= 2):
            blind.append((f"{line}｜原来的轨号 {win.acoustic[i][0]}", len(old_wrong) >= 2))
            old_wrong = old_fix = []
        if r and r["kind"] == "wrong":
            why = basis(witness, key, i, dict(r, id=rid), win.truth[i].person)
            if why:
                (same if why == BASIS["same"] else wrong).append(
                    f"{line}｜{r['old']} → {r['new_label']}｜{r['why']}｜{why}｜旧版三遍里 {len(old_wrong)} 遍也改错")
        if len(old_wrong) >= 2 and not (r and r["kind"] == "wrong"):
            better.append(f"{line}｜旧版三遍里 {len(old_wrong)} 遍把 {win.acoustic[i][0]} 改成 {old_wrong[0]}（错）"
                          f"｜这次{'改成 ' + r['new_label'] + '（对）' if r else '没改（对）'}")
        if len(old_fix) >= 2 and not (r and r["kind"] == "fix"):
            lost.append(f"{line}｜原来的 {win.acoustic[i][0]} 是错的，旧版三遍里 {len(old_fix)} 遍改成 {old_fix[0]}（对）"
                        f"｜这次{'改成 ' + r['new_label'] + '（也错）' if r else '没改'}")
    return wrong, same, better, lost, blind, sid


def render(round_, label):
    wrong, same, better, lost, blind, sid = told(round_, label)
    template = ROUNDS[round_][3]
    text = Template((HERE / "judge_ask_templates" / template).read_text(encoding="utf-8")).substitute(
        count=len(wrong), wrong="\n".join(wrong) or "（没有）", same_count=len(same), same="\n".join(same) or "（没有）",
        better="\n".join(better) or "（没有）", lost="\n".join(lost) or "（没有）",
        blind="\n".join(line for line, _ in blind) or "（没有）")
    return text, sid, {line.split("｜", 1)[0]: original for line, original in blind}


def ask(round_, labels):
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    work = MAIN / "tmp" / f"{round_}-ask"
    work.mkdir(parents=True, exist_ok=True)
    for label in labels:
        text, sid, blind_key = render(round_, label)
        started = time.time()
        driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=1200))
        result = driver.run([{"role": "user", "content": text}], task=f"{round_}-ask", session_scope="assignment",
                            conversation_key=f"{round_}-ask-{label}-{int(started)}", conversation_handle=sid,
                            continuation=True)
        row = {"label": label, "conversation": result.conversation_handle, "same_session": result.conversation_handle == sid,
               "answer": result.content.strip(), "seconds": round(time.time() - started, 1), "ledger": ledger(sid),
               "prompt": text, "blind_key": blind_key}
        out = work / "answers.json"
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k not in ("answer", "prompt")}, ensure_ascii=False), flush=True)


def main():
    round_, verb = sys.argv[1], sys.argv[2]
    if verb == "show":
        for label in ROUNDS[round_][4]:
            text, sid, blind_key = render(round_, label)
            print(f"===== {label} ({sid})\n{text}\n")
    else:
        ask(round_, sys.argv[3].split(","))


if __name__ == "__main__":
    main()
