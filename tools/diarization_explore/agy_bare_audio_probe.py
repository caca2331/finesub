"""Bare audio on agy (owner 2026-10-29): where does the clock drift come from, and does reading audio drop the cache?

agy 1.1.18+ takes bare audio; on 3.8 Flash an 11-minute m4a is heard whole but the times the model gives drift, while
mp4 wrappings did not drift where checked. Two experiments, agy Gemini 3.8 Flash, one call per arm:

  drift -- one 5-minute stretch of the hina-22 vocal (SEG seconds into `agy_media_cut_probe`'s 20 minutes), encoded
           once to AAC; the very same audio stream (`-c:a copy`) as
             drA  bare m4a
             drS  mp4 under one black frame spanning the whole clip (as `containerize_audio_for_agy` does)
             drV  mp4 under a black video track of one 1-second frame
           The model quotes the ~10 s from each of POINTS; `drift_score` finds, per quote, the second whose ASR rows
           match it best (1-second steps): drift = that second - the asked second.
  cache -- one session, task padded past agy's ~16k cache-write threshold; the model reads the task in its first step,
           then in a separate step opens one more thing, then answers. Per request from agy's ledger: uncached /
           cached. The request after that second read shows whether it dropped the cache:
             caT  re-read the first lines of the task file (control: no media)
             caS  the 60 s audio wrapped single-frame (the driver today)
             caA  the same audio stream as a bare m4a
           Production opens task and media in one first step (`agy_first_step_media`); it is switched off here.

⚠ drA / caA need a driver that sends `audio/*` attachments as they are (a temporary patch, not committed: the
committed driver wraps them in a single-frame mp4 again).

    python tools/diarization_explore/agy_bare_audio_probe.py <arm,...>   # appends to <MAIN>/tmp/agy-media-cut/bare-probes.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agy_media_cut_probe as A  # noqa: E402

from finesub.llm.agent import local_agent  # noqa: E402
from finesub.media.ffmpeg import AUDIO_CODEC_ARGS  # noqa: E402

OUT = A.WORK / "bare-probes.json"
SEG, LENGTH = 300, 300
POINTS = tuple(range(15, LENGTH, 30))
CACHE_AUDIO = 60
PAD = (A.MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16" / "fixed30k.txt").read_text(
    encoding="utf-8")[:26000]


def run_ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def audio(name, start, length):
    """`length` seconds from `start` of the 20 minutes, encoded once as the driver encodes audio."""
    target = A.WORK / f"{name}.m4a"
    if not target.exists():
        run_ffmpeg("-ss", start, "-t", length, "-i", A.cut("full20", 0, 1200).with_suffix(".wav"), *AUDIO_CODEC_ARGS, target)
    return target


def single_frame(name, source, length):
    """`source`'s audio stream copied under one black frame lasting the whole clip."""
    target = A.WORK / f"{name}.mp4"
    if not target.exists():
        run_ffmpeg("-i", source, "-f", "lavfi", "-i", f"color=c=black:s=16x16:r={1 / length:.12f}", "-map", "1:v:0",
                   "-map", "0:a:0", "-frames:v", "1", "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                   "-c:a", "copy", target)
    return target


def one_second(name, source):
    """`source`'s audio stream copied under a black video track of one 1-second frame."""
    target = A.WORK / f"{name}.mp4"
    if not target.exists():
        run_ffmpeg("-f", "lavfi", "-i", "color=c=black:s=16x16:r=1:d=1", "-i", source, "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy", target)
    return target


def drift_files(arm):
    m4a = audio("drift5", SEG, LENGTH)
    return {"drA": m4a, "drS": single_frame("drift5-s", m4a, LENGTH), "drV": one_second("drift5-v", m4a)}[arm]


def cache_files(arm):
    if arm == "caT":
        return []
    m4a = audio("cache60", 0, CACHE_AUDIO)
    return [m4a if arm == "caA" else single_frame("cache60-s", m4a, CACHE_AUDIO)]


def drift_ask():
    points = "、".join(f"{t // 60}:{t % 60:02d}" for t in POINTS)
    return (f"附件是一段 5 分钟的日语录音。下面每个时间点，逐字写出从那一刻起约 10 秒里说的话，每个时间点一行，"
            f"格式「m:ss｜原文」：{points}。听不到就写「拿不到」，不要猜。")


CACHE_ASK = {
    "caT": "读完任务之后，在单独的一步里（不要和别的调用放在同一步）用 view_file 再打开一次任务文件的第 1 到 5 行，然后回答：任务文件第一行写的是什么？",
    "caS": "读完任务之后，在单独的一步里（不要和别的调用放在同一步）用 view_file 打开附件，听完回答：录音最开头的一句话是什么？",
}
CACHE_ASK["caA"] = CACHE_ASK["caS"]
NOTE = "\n\n下面是背景资料，与问题无关，不用理会：\n\n"


def drift_score(answer):
    """Per point: (asked second, best-matching second, drift, overlap there)."""
    def grams(text):
        text = re.sub(r"[\s、。，,.!?！？「」…ー〜（）()*]", "", text)
        return {text[i:i + 3] for i in range(len(text) - 2)}

    stable = json.loads((A.SRC.parent / "hina-22-c0102-stable.json").read_text(encoding="utf-8"))
    segs = stable["segments"] if isinstance(stable, dict) else stable
    rows = [(float(s["start"]) - A.OFFSET - SEG, s.get("text", "")) for s in segs]
    quotes = {int(m.group(1)) * 60 + int(m.group(2)): m.group(3)
              for m in re.finditer(r"^\W*(\d+):(\d\d)\s*[｜|:：]\s*(.+)$", answer, re.M)}
    out = []
    for t in POINTS:
        q = grams(quotes.get(t, ""))
        if not q:
            out.append((t, None, None, 0.0))
            continue
        u, best = max(((u, len(q & grams("".join(x for at, x in rows if u - 3 <= at <= u + 12))) / len(q))
                       for u in range(-20, LENGTH + 20)), key=lambda x: x[1])
        out.append((t, u, u - t, round(best, 2)))
    return out


def main():
    driver = local_agent.AgyLocalAgentDriver(local_agent.AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=900))
    first_step = local_agent.agy_first_step_media
    for arm in sys.argv[1].split(","):
        cache = arm.startswith("ca")
        files = cache_files(arm) if cache else [drift_files(arm)]
        text = CACHE_ASK[arm] + NOTE + PAD if cache else drift_ask()
        local_agent.agy_first_step_media = (lambda *a, **k: []) if cache else first_step
        user = {"role": "user", "content": [{"type": "text", "text": text}, *(A.attachment(f) for f in files)]}
        started = time.time()
        result = driver.run([{"role": "system", "content": "如实回答。"}, user], task="probe-bare-audio",
                            session_scope="assignment", conversation_key=f"bare-{arm}-{int(started)}", conversation_handle="")
        row = {"arm": arm, "conversation": result.conversation_handle, "answer": result.content.strip(),
               "seconds": round(time.time() - started, 1), "ledger": A.ledger(result.conversation_handle)}
        if not cache:
            row["drift"] = drift_score(row["answer"])
        print(json.dumps({k: v for k, v in row.items() if k != "answer"}, ensure_ascii=False), flush=True)
        old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
        OUT.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
