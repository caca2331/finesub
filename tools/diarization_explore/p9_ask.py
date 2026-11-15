"""P9 follow-up (docs/plans/argon-audio-judge-plan.md 「P9 之后：直接问出错的会话」): back in the P9 run's own agy
sessions, tell the judge which rows it moved wrongly and ask what it heard, how it decided, and how it should be
prompted. Owner 2026-10-30: say plainly where it erred, no decoys.

    python tools/diarization_explore/p9_ask.py show            # the rows each session will be told about
    python tools/diarization_explore/p9_ask.py ask <label,...>

Rows told: the run's wrong moves onto the other sex that the owner heard (hina-46) or that pitch and MAI2 both side
the ruler on; in hina-22's second window, where no such row is left, its cross-sex wrong moves with the witnesses as
they stand. Answers and agy's per-request ledger go to <MAIN>/tmp/p9-ask/answers.json.
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
RUN = MAIN / "tmp" / "spk-merge" / "accept" / "audio-rovl1"
sys.path.insert(0, str(MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16"))
from agy_media_resume_probe_ledger import ledger  # noqa: E402

WORK = MAIN / "tmp" / "p9-ask"
#: label -> (window key, judge unit, exchange file of the answer kept, agy session, rows told, basis)
SESSIONS = {
    "h46-u2": ("hina-46-c0001", "0002", "002-speaker_judge-0002-attempt0.md", "12bb07d4-d234-4660-8528-d713623df027",
               ("171", "189", "221"), "人工逐行听过，确实是原来那个人说的"),
    "h22-u1": ("hina-22-c0102", "0001", "001-speaker_judge-0001-attempt0.md", "f80a3bb4-2156-4e9c-9636-60673fce2c26",
               ("31", "44", "208", "210", "211", "212", "214", "215", "216", "217", "242"),
               "基频与另一套分人系统都站原来那个人（第 208、217、242 行基频判不了），其中第 208–217 行是男主持在念听众来信"),
    "h22-u2": ("hina-22-c0102", "0002", "003-speaker_judge-0002-attempt1.md", "384233c0-0d5b-421d-aa45-ccee4e95044c",
               ("322", "326", "378", "391", "422", "438", "453"),
               "按我们的逐行标注是原来那个人；但这几行的基频与另一套分人系统不全同意，可能是标注错了，所以也请照实说"),
}
ROW = re.compile(r"^▶ q(\d+) ([\d.]+)\|([\d.]+)\|")


def told(label):
    import judge_moves_readout as M
    key, unit, exchange, sid, ids, basis = SESSIONS[label]
    wins, results = M.load(None)
    win = wins[key]
    got = results["38F ovl r1", key][0]
    index = {k: i for i, k in enumerate(win.ids)}
    text = (RUN / key / "llm" / "exchanges" / exchange).read_text(encoding="utf-8").split("## 模型响应", 1)[0]
    at = {m.group(1): (float(m.group(2)), float(m.group(3))) for m in map(ROW.match, text.splitlines()) if m}
    lines = []
    for rid in ids:
        r = got[index[rid]]
        assert r["kind"] == "wrong", (label, rid)
        s, d = at[rid]
        lines.append(f"q{rid}｜{s:.1f}–{s + d:.1f}｜{win.segs[index[rid]]['text']}｜{r['old']} → {r['new_label']}｜{r['why']}")
    return lines, basis, sid


def ask(labels):
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    WORK.mkdir(parents=True, exist_ok=True)
    for label in labels:
        lines, basis, sid = told(label)
        text = Template((HERE / "judge_ask_templates" / "ask_user_v1.md").read_text(encoding="utf-8")).substitute(
            count=len(lines), basis=basis, rows="\n".join(lines))
        started = time.time()
        driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=1200))
        result = driver.run([{"role": "user", "content": text}], task="p9-ask", session_scope="assignment",
                            conversation_key=f"p9-ask-{label}-{int(started)}", conversation_handle=sid,
                            continuation=True)
        row = {"label": label, "conversation": result.conversation_handle, "same_session": result.conversation_handle == sid,
               "answer": result.content.strip(), "seconds": round(time.time() - started, 1), "ledger": ledger(sid),
               "prompt": text}
        out = WORK / "answers.json"
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k not in ("answer", "prompt")}, ensure_ascii=False), flush=True)


def main():
    if sys.argv[1] == "show":
        for label in SESSIONS:
            lines, basis, sid = told(label)
            print(f"== {label} ({sid}): {basis}")
            print("\n".join(lines))
    else:
        ask(sys.argv[2].split(","))


if __name__ == "__main__":
    main()
