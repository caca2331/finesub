"""P10 (docs/plans/argon-audio-judge-plan.md 「P10 更长的片段」): on judge clips of 17.5 and 20.2 minutes, does agy 3.8
Flash hear the whole clip and find the rows, with the clip split into pieces under 5 minutes or sent as bare audio?

    python tools/diarization_explore/p10_long_probe.py --code <snapshot> <split|bare> <align|clock> <key,...>
    python tools/diarization_explore/p10_long_probe.py score

The clip is a production judge clip (voice samples, then the window's recording), attached as `audio/aac` the way
production attaches it; the trial driver of `--code` (P7's snapshots: `tmp/spk-merge/accept/code-b3355c57` split,
`code-88811913` bare) wraps it as its form does. Two tasks, one call each:
- align: the judge prompt's rows (clip seconds) with one row per minute hidden (the first from 20 s into the minute
  lasting >= 1.5 s with >= 8 characters, plus one in the last minute); it writes the hidden rows out. A row counts as
  heard when its ASR text's character 3-grams are at least half in the answer (as the 11-minute acceptance).
- clock: quote the ~10 s from each point (every 60 s from the first row, plus the end). Per point, `here` is the
  3-gram share found among rows starting within [-20, +30] s of it and `best` the best share at any point of the
  clip (`where`): heard but misplaced shows as a low `here` with a high `best` at another `where`.
Answers and agy's per-request ledger go to <MAIN>/tmp/p10-long/probes.json.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("FINESUB_ROOT", str(HERE.parents[1]))
CODE = Path(sys.argv[sys.argv.index("--code") + 1]) if "--code" in sys.argv else HERE.parents[1]
sys.path.insert(0, str(CODE / "src"))

MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
SPK = MAIN / "tmp" / "spk-merge" / "accept"
sys.path.insert(0, str(MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16"))
from agy_media_resume_probe_ledger import ledger  # noqa: E402

WORK = MAIN / "tmp" / "p10-long"
#: key -> the run whose judge clip and prompt are reused (one judge window each)
RUNS = {"hina-17-c0002": "audio-rflash1", "manosaba-b-c0001": "audio-rbare1"}
ROW = re.compile(r"^(?:▶ q\d+ )?([\d.]+)\|([\d.]+)\|[\d.]*\|[^|]*\|(.*)$")
SYSTEM = "你会收到一段日语录音的音频附件。这是一次听辨核对，不是字幕任务。只按听到的声音如实回答：听不清就写「听不清」，不要猜。"


def clip(key):
    return SPK / RUNS[key] / key / "llm" / "clips" / "speaker-judge-0001.aac"


def rows(key):
    """(start, length, text) of the judge prompt's rows, clip seconds."""
    text = (SPK / RUNS[key] / key / "llm" / "exchanges" / "001-speaker_judge-0001-attempt0.md").read_text(encoding="utf-8")
    user = text.split("## 模型响应", 1)[0]
    return [(float(m.group(1)), float(m.group(2)), m.group(3).strip()) for m in map(ROW.match, user.splitlines()) if m]


def length(key):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip(key))],
                         capture_output=True, text=True, check=True)
    return float(out.stdout)


def hidden(key):
    rs, end = rows(key), length(key)
    first = rs[0][0]
    out = []
    for at in [first + 20 + 60 * k for k in range(int((end - first) // 60))] + [end - 60]:
        k = next((i for i, (s, d, t) in enumerate(rs) if s >= at and d >= 1.5 and len(t) >= 8 and i not in out), None)
        if k is not None:
            out.append(k)
    return out


def points(key):
    first, end = rows(key)[0][0], length(key)
    return [round(first + 5 + 60 * k) for k in range(int((end - first - 20) // 60) + 1)] + [int(end - 15)]


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def ask(key, task):
    rs, end, first = rows(key), length(key), rows(key)[0][0]
    head = (f"附件是一段 {end / 60:.1f} 分钟的音频：开头约 {first:.0f} 秒是几段声音样本，之后是一段日语对话的录音。"
            "时间都按这个音频文件内的位置（分:秒）算。")
    if task == "clock":
        ps = "、".join(f"{p // 60}:{p % 60:02d}" for p in points(key))
        return head + f"每个时间点，逐字写出从那一刻起约 10 秒里说的话，每个时间点一行，格式「m:ss｜原文」：{ps}。"
    hide = set(hidden(key))
    table = "\n".join(f"{i + 1}｜{clock(s)}–{clock(s + d)}｜{'＿＿' if i in hide else t}" for i, (s, d, t) in enumerate(rs))
    return (head + f"下表是录音的 ASR 逐行文本（行号｜起–止｜文本），其中 {len(hide)} 行的文本隐去了，写作「＿＿」。"
            "听录音，逐字写出这些行说的话，每行一行，格式「行号｜原文」，只写隐去的行。\n\n" + table)


def grams(text):
    text = re.sub(r"[\s、。，,.!?！？「」…ー〜（）()]", "", text)
    return {text[i:i + 3] for i in range(len(text) - 2)}


def score_align(key, answer):
    rs = rows(key)
    got = {int(m.group(1)) - 1: m.group(2) for m in re.finditer(r"^\W*(\d+)\s*[｜|]\s*(.+)$", answer, re.M)}
    return [(round(rs[k][0]), round(len(grams(got.get(k, "")) & grams(rs[k][2])) / max(1, len(grams(rs[k][2]))), 2))
            for k in hidden(key)]


def score_clock(key, answer):
    rs, end = rows(key), length(key)
    near = lambda t: grams("".join(x for s, d, x in rs if t - 20 <= s <= t + 30))  # noqa: E731
    quotes = {}
    for line in answer.splitlines():
        m = re.match(r"\s*[*-]*\s*(\d+):(\d\d)\s*[｜|:：]\s*(.+)", line)
        if m:
            quotes[int(m.group(1)) * 60 + int(m.group(2))] = m.group(3)
    out = []
    for p in points(key):
        q = grams(quotes.get(p, ""))
        if not q:
            out.append((p, 0.0, 0.0, None))
            continue
        here = len(q & near(p)) / len(q)
        where, best = max(((u, len(q & near(u)) / len(q)) for u in range(0, int(end), 5)), key=lambda x: x[1])
        out.append((p, round(here, 2), round(best, 2), where))
    return out


def run(form, task, keys):
    from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver
    driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=1200))
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "probes.json"
    for key in keys:
        text = ask(key, task)
        user = {"role": "user", "content": [{"type": "text", "text": text},
                                            {"type": "file", "file": {"file_id": clip(key).name, "filename": clip(key).name,
                                                                      "format": "audio/aac", "local_path": str(clip(key))}}]}
        started = time.time()
        result = driver.run([{"role": "system", "content": SYSTEM}, user], task="p10-probe", session_scope="assignment",
                            conversation_key=f"p10-{form}-{task}-{key}-{int(started)}", conversation_handle="")
        row = {"key": key, "form": form, "task": task, "code": str(CODE), "conversation": result.conversation_handle,
               "answer": result.content.strip(), "seconds": round(time.time() - started, 1),
               "ledger": ledger(result.conversation_handle)}
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in row.items() if k != "answer"}, ensure_ascii=False), flush=True)


def score():
    for row in json.loads((WORK / "probes.json").read_text(encoding="utf-8")):
        key, task = row["key"], row["task"]
        if task == "align":
            got = score_align(key, row["answer"])
            heard = sum(s >= 0.5 for _, s in got)
            print(f"{key} {row['form']:5} align: heard {heard}/{len(got)} | " + " ".join(f"{clock(t)}:{s}" for t, s in got))
        else:
            got = score_clock(key, row["answer"])
            ok = sum(h >= 0.5 for _, h, _, _ in got)
            heard = sum(b >= 0.5 for _, _, b, _ in got)
            print(f"{key} {row['form']:5} clock: right place {ok}/{len(got)}, heard anywhere {heard}/{len(got)} | " +
                  " ".join(f"{p}s:{h}/{b}@{w}" for p, h, b, w in got))


def main():
    args = [a for i, a in enumerate(sys.argv[1:], 1) if a != "--code" and sys.argv[i - 1] != "--code"]
    if args[0] == "score":
        score()
    else:
        run(args[0], args[1], args[2].split(","))


if __name__ == "__main__":
    main()
