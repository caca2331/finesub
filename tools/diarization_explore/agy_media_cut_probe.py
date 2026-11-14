"""How much of an audio attachment does the model behind agy hear, and what gets it all? (2026-10-25)

agy's `view_file` sends the whole mp4 (the conversation store keeps every byte), but the model sees one part,
`part_0: 00:00 to ...`, whose end the server picks. The driver wraps audio as an mp4 with one black frame
(`containerize_audio_for_agy`); each arm is one call to agy Gemini 3.8 Flash on cuts of the hina-22 vocal:

  one7 / one95 / one20   one single-frame attachment of 7 / 9.5 / 20 minutes
  two10                  two single-frame 10-minute attachments
  four5                  four single-frame 5-minute attachments
  one20f / one20m / one20q   one 20-minute attachment with a real black video track at 0.25 fps / 1 per 60 s / 1 per 300 s
  one20e                 one 20-minute attachment with two black frames, at the start and at 1199 s
  acc11s / acc11e / acc11q / acc11f / acc11v   the acceptance check (owner 2026-10-25): an 11-minute attachment,
                         single-frame / frames at both ends / a frame every 300 s / every 4 s / a video track of one
                         1-second frame, quoted at ACCEPT_POINTS (`score`)
  tiny1 / tinyf1 / tinyf4 / tinyn1 / tinyv1   30 seconds of audio under one frame / a black frame per second / per 4 s /
                         a different frame per second / one 1-second frame: what the frames themselves cost
  acc11pad / acc11end    one attachment: the 11 minutes padded with silence to PADDED seconds, single-frame / one
                         2-second frame at the end only
  acc11p                 the same 11 minutes as attachments of at most PIECE seconds, each under a 1-second video track
  tinya / acc11a / acc11pa   30 seconds / the 11 minutes / the 11 minutes in PIECE-second attachments, as bare audio
                         (AAC in m4a, no video track; 2026-10-29, agy 1.4.7).
                         ⚠ ran with a temporary driver patch that copies `audio/*` attachments as they are and keeps their
                         MIME; the committed driver wraps them in a single-frame mp4 again
  alignA / alignP / alignN   the 11 minutes' ASR rows with 11 texts hidden, to be written out from the audio: bare (patched
                         driver) / split under 1-second video tracks / no audio at all (`align_score`)
⚠ owner 2026-10-25: no more agy runs on attachments over 11 minutes (quota); the free API arms of
`api_media_cut_probe.py` behave the same on generation and come first.

The model quotes fixed points (checked by hand against the window's ASR) and copies the part labels it got; the
ledger (`[request, uncached, cached, output]` from agy's own store) gives the input-token jump the media brings.

    python tools/diarization_explore/agy_media_cut_probe.py <arm,...>     # appends to <MAIN>/tmp/agy-media-cut/probes.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import judge_moves_readout as M  # noqa: E402

from finesub.llm.agent.local_agent import AgyDriverConfig, AgyLocalAgentDriver  # noqa: E402
from finesub.media.ffmpeg import containerize_audio_for_agy  # noqa: E402

MAIN = M.S.r9.MAIN
sys.path.insert(0, str(MAIN / "data" / "session-ab-2026-10" / "probes-live" / "cache-reuse-2026-10-16"))
from agy_media_resume_probe_ledger import ledger  # noqa: E402

WORK = MAIN / "tmp" / "agy-media-cut"
SRC = M.SPK / "accept" / "audio-rf38v7" / "hina-22-c0102" / "input" / "hina-22-c0102-vocal.ogg"
OFFSET = 60.0
SYSTEM = "你会收到音频附件。如实回答，听不到就说听不到。"
TAIL = ("某一段你拿不到音频就写「拿不到」，不要猜。最后：照原样列出你收到的每个媒体部分的标签与时间范围"
        "（例如工具结果里写的 part 编号和起止时间），并说明你实际能听到的时间范围。")
TWENTY = "请分别逐字写出 5:00–5:10、9:50–10:00、10:30–10:40、15:00–15:10、19:40–19:50 这五段里说的话。"
#: acceptance (owner 2026-10-25): on an 11-minute attachment the model must hear the start, the end and every point
#: between -- clip seconds asked for, one quote each
ACCEPT_SECONDS = 660
ACCEPT_POINTS = (5, *range(60, 601, 60), 650)


def accept_ask(note, split=False):
    points = "、".join(f"{t // 60}:{t % 60:02d}" for t in ACCEPT_POINTS)
    what = ("附件是一段 11 分钟的日语录音，按顺序切成了 3 个附件：附件 1 是 0:00–5:00，附件 2 是 5:00–10:00，附件 3 是 10:00–11:00"
            f"{note}，每个都要用 view_file 打开。下面的时间点按整段录音算" if split else f"附件是一段 11 分钟的日语录音{note}。下面")
    return f"{what}每个时间点，逐字写出从那一刻起约 10 秒里说的话，每个时间点一行，格式「m:ss｜原文」：{points}。"


ASK = {
    "one7": "附件是一段 7 分钟的日语录音。请分别逐字写出 2:00–2:10、4:50–5:00、5:30–5:40、6:40–6:50 这四段里说的话。",
    "one95": "附件是一段 9 分半的日语录音。请分别逐字写出 4:50–5:00、5:30–5:40、9:10–9:20 这三段里说的话。",
    "one20": "附件是一段 20 分钟的日语录音。" + TWENTY,
    "one20f": "附件是一段 20 分钟的日语录音（画面是黑屏）。" + TWENTY,
    "one20m": "附件是一段 20 分钟的日语录音（画面是黑屏）。" + TWENTY,
    "one20q": "附件是一段 20 分钟的日语录音（画面是黑屏）。" + TWENTY,
    "one20e": "附件是一段 20 分钟的日语录音（画面是黑屏）。" + TWENTY,
    **{arm: accept_ask(note) for arm, note in (("acc11s", ""), ("acc11e", "（画面是黑屏）"), ("acc11q", "（画面是黑屏）"),
                                              ("acc11f", "（画面是黑屏）"), ("acc11v", "（画面是黑屏）"))},
    "acc11p": accept_ask("（画面是黑屏）", split=True),
    **{arm: accept_ask("，后面接着约 4 分钟静音（画面是黑屏）") for arm in ("acc11pad", "acc11end")},
    **{arm: "附件是一段 30 秒的日语录音。逐字写出 0:20–0:30 里说的话。" for arm in ("tiny1", "tinyf1", "tinyf4", "tinyn1", "tinyv1")},
    "two10": ("附件有两个，各 10 分钟，第二个紧接第一个。两个都要用 view_file 打开。请分别逐字写出：附件 1 的 5:00–5:10、9:50–10:00；"
              "附件 2 的 0:30–0:40、5:00–5:10、9:40–9:50。"),
    "four5": ("附件有四个，各 5 分钟，依次相接。四个都要用 view_file 打开。请分别逐字写出：附件 1 的 4:45–4:55；附件 2 的 0:30–0:40；"
              "附件 3 的 4:45–4:55；附件 4 的 4:40–4:50。"),
}


def cut(name, start, length):
    """A single-frame attachment of the vocal from OFFSET + start, as the driver wraps audio."""
    target = WORK / f"{name}.mp4"
    if not target.exists():
        WORK.mkdir(parents=True, exist_ok=True)
        wav = WORK / f"{name}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(OFFSET + start), "-t", str(length), "-i", str(SRC),
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        containerize_audio_for_agy(wav, target)
    return target


def framed(name, length, rate):
    """The first `length` seconds with a real black video track at `rate` fps instead of one frame."""
    target = WORK / f"{name}.mp4"
    if not target.exists():
        wav = cut("full20", 0, 1200).with_suffix(".wav")
        # ⚠ not -shortest: with a sparse video track it cuts the audio to a few seconds
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r={rate}:d={length}",
                        "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "32k",
                        "-t", str(length), str(target)], check=True)
    return target


def ends(name="full20e", length=1200):
    """The first `length` seconds with two black frames, at 0 s and at the last whole second."""
    target = WORK / f"{name}.mp4"
    if not target.exists():
        wav = cut("full20", 0, 1200).with_suffix(".wav")
        # ⚠ not color=r=1/<length>: that makes one frame; pick the first and last frame of a 1 fps track instead
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=black:s=64x64:r=1:d={length}", "-i", str(wav),
                        "-vf", f"select='eq(n\\,0)+eq(n\\,{length - 1})'", "-fps_mode", "vfr", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "32k", "-t", str(length), str(target)], check=True)
    return target


def score(answer, segs):
    """Per acceptance point: the share of the quote's character 3-grams found in the ASR rows starting within
    [-20, +30] s of it (`here`), and the best such share at any point of the clip (`best`, at `where`): heard but
    misplaced shows as a low `here` with a high `best`."""
    import re

    def grams(text):
        text = re.sub(r"[\s、。，,.!?！？「」…ー〜（）()]", "", text)
        return {text[i:i + 3] for i in range(len(text) - 2)}

    rows = [(float(r["start"]) - OFFSET, r.get("text", "")) for r in segs]
    near = lambda t: grams("".join(x for at, x in rows if t - 20 <= at <= t + 30))  # noqa: E731
    quotes = {}
    for line in answer.splitlines():
        m = re.match(r"\s*[*-]*\s*(\d+):(\d\d)\s*[｜|:：]\s*(.+)", line)
        if m:
            quotes[int(m.group(1)) * 60 + int(m.group(2))] = m.group(3)
    out = []
    for t in ACCEPT_POINTS:
        q = grams(quotes.get(t, ""))
        if not q:
            out.append((t, 0.0, 0.0, None))
            continue
        here = len(q & near(t)) / len(q)
        where, best = max(((u, len(q & near(u)) / len(q)) for u in range(0, ACCEPT_SECONDS, 10)), key=lambda x: x[1])
        out.append((t, here, best, where))
    return out


def short(name, length, start=0):
    """`length` seconds of audio from `start` under a black video track of one 1-second frame: the video ends at 1 s."""
    target = WORK / f"{name}.mp4"
    if not target.exists():
        wav = cut("full20", 0, 1200).with_suffix(".wav")
        # no -t / -shortest on the output: it lasts as long as its longest stream, the audio
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=1:d=1",
                        "-ss", str(start), "-t", str(length), "-i", str(wav), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "32k", str(target)], check=True)
    return target


#: the frame-cost check (owner 2026-10-25: frame count did change the cost in earlier tests; confirm on a few seconds
#: before trusting the long probes): TINY seconds of audio, the video track varied, one short question
TINY = 30
TINY_TRACKS = {
    "tiny1": None,                                        # one frame over the whole clip (the driver today)
    "tinyf1": "color=c=black:s=64x64:r=1",                # a black frame per second
    "tinyf4": "color=c=black:s=64x64:r=0.25",             # a black frame per 4 s
    "tinyn1": "testsrc2=s=64x64:r=1",                     # a different frame per second
    "tinyv1": "short",                                    # one frame lasting 1 s, then audio alone
}


def tiny(arm):
    track = TINY_TRACKS[arm]
    if track is None:
        return cut(arm, 0, TINY)
    if track == "short":
        return short(arm, TINY)
    target = WORK / f"{arm}.mp4"
    if not target.exists():
        wav = cut("full20", 0, 1200).with_suffix(".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{track}:d={TINY}", "-t", str(TINY), "-i", str(wav),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "32k", "-t", str(TINY), str(target)],
                       check=True)
    return target


#: one attachment longer than PIECE: the server drops its last PIECE-part, so pad the audio with silence until that
#: part is only padding (ACCEPT_SECONDS of speech, then silence up to PADDED seconds)
PADDED = 901


def padded_wav():
    wav = WORK / "acc11-padded.wav"
    if not wav.exists():
        src = cut("full20", 0, 1200).with_suffix(".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", str(ACCEPT_SECONDS), "-i", str(src), "-af", f"apad=whole_dur={PADDED}",
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


def padded(name, frame_at=None):
    """The padded clip as one attachment: the driver's single-frame wrapping (`frame_at` None), or one black frame
    lasting 2 s that starts at `frame_at` seconds (the video track begins there)."""
    target = WORK / f"{name}.mp4"
    if not target.exists():
        if frame_at is None:
            containerize_audio_for_agy(padded_wav(), target)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-itsoffset", str(frame_at), "-f", "lavfi", "-i",
                            "color=c=black:s=64x64:r=1:d=2", "-i", str(padded_wav()), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "32k", str(target)], check=True)
    return target


#: one attachment per this many seconds at most (the server's part length)
PIECE = 300


def pieces(name, length):
    """The first `length` seconds as attachments of at most PIECE seconds, each under a 1-second video track."""
    return [short(f"{name}-{k}", min(PIECE, length - at), at) for k, at in enumerate(range(0, length, PIECE))]


FILES = {
    "one7": lambda: [cut("full7", 0, 420)],
    "one95": lambda: [cut("full95", 0, 570)],
    "one20": lambda: [cut("full20", 0, 1200)],
    "one20f": lambda: [framed("full20f", 1200, "0.25")],
    "one20m": lambda: [framed("full20m", 1200, "1/60")],
    "one20q": lambda: [framed("full20q", 1200, "1/300")],
    "one20e": lambda: [ends()],
    "acc11s": lambda: [cut("acc11s", 0, ACCEPT_SECONDS)],
    "acc11e": lambda: [ends("acc11e", ACCEPT_SECONDS)],
    "acc11q": lambda: [framed("acc11q", ACCEPT_SECONDS, "1/300")],
    "acc11f": lambda: [framed("acc11f", ACCEPT_SECONDS, "0.25")],
    "acc11v": lambda: [short("acc11v", ACCEPT_SECONDS)],
    "acc11p": lambda: pieces("acc11p", ACCEPT_SECONDS),
    "acc11pad": lambda: [padded("acc11pad")],
    "acc11end": lambda: [padded("acc11end", PADDED - 2)],
    **{arm: (lambda arm=arm: [tiny(arm)]) for arm in TINY_TRACKS},
    "two10": lambda: [cut("half1", 0, 600), cut("half2", 600, 600)],
    "four5": lambda: [cut(f"q{k}", 300 * k, 300) for k in range(4)],
}


def bare(name, length, start=0):
    """`length` seconds from `start` as bare audio (AAC in m4a, no video track): agy 1.1.18+ takes audio files as they are."""
    target = WORK / f"{name}.m4a"
    if not target.exists():
        wav = cut("full20", 0, 1200).with_suffix(".wav")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(length), "-i", str(wav), "-c:a", "aac",
                        "-b:a", "32k", str(target)], check=True)
    return target


FILES |= {"tinya": lambda: [bare("tinya", TINY)], "acc11a": lambda: [bare("acc11a", ACCEPT_SECONDS)],
          "acc11pa": lambda: [bare(f"acc11pa-{k}", min(PIECE, ACCEPT_SECONDS - at), at)
                              for k, at in enumerate(range(0, ACCEPT_SECONDS, PIECE))]}
ASK |= {"tinya": ASK["tiny1"], "acc11a": accept_ask(""), "acc11pa": accept_ask("", split=True)}


def align_rows():
    """The window's ASR rows inside the 11 minutes (clip time), and the rows to hide: per minute, the first row from
    0:20 into it lasting at least 1.5 s with at least 8 characters, plus one from 10:30."""
    stable = json.loads((SRC.parent / "hina-22-c0102-stable.json").read_text(encoding="utf-8"))
    segs = stable["segments"] if isinstance(stable, dict) else stable
    rows = [(float(s["start"]) - OFFSET, float(s["end"]) - OFFSET, s.get("text", "").strip()) for s in segs]
    rows = [r for r in rows if 0 <= r[0] < ACCEPT_SECONDS and r[2]]
    hidden = []
    for at in (*range(20, 620, 60), 630):
        k = next(i for i, (s, e, t) in enumerate(rows) if s >= at and e - s >= 1.5 and len(t) >= 8)
        hidden.append(k)
    return rows, hidden


def align_ask(split):
    """Hand the model the ASR rows with a dozen texts hidden; it has to find each hidden row in the audio by its
    neighbours and time (the owner's 2026-10-29 question: does audio line up with ASR text even when its clock drifts)."""
    rows, hidden = align_rows()
    clock = lambda x: f"{int(x // 60)}:{x % 60:04.1f}"  # noqa: E731
    table = "\n".join(f"{i + 1}｜{clock(s)}–{clock(e)}｜{'＿＿' if i in hidden else t}" for i, (s, e, t) in enumerate(rows))
    what = ("附件是一段 11 分钟的日语录音，按顺序切成了 3 个附件：附件 1 是 0:00–5:00，附件 2 是 5:00–10:00，附件 3 是 10:00–11:00，"
            "每个都要用 view_file 打开。下表的时间按整段录音算。" if split else "附件是一段 11 分钟的日语录音。")
    return (f"{what}下表是这段录音的 ASR 逐行文本（行号｜起–止｜文本），其中 {len(hidden)} 行的文本隐去了，写作「＿＿」。"
            "听录音，逐字写出这些行说的话，每行一行，格式「行号｜原文」，只写隐去的行。\n\n" + table)


def align_score(answer):
    """Per hidden row: the share of its ASR text's character 3-grams the model's line for it contains."""
    import re

    def grams(text):
        text = re.sub(r"[\s、。，,.!?！？「」…ー〜（）()]", "", text)
        return {text[i:i + 3] for i in range(len(text) - 2)}

    rows, hidden = align_rows()
    got = {int(m.group(1)) - 1: m.group(2) for m in re.finditer(r"^\W*(\d+)\s*[｜|]\s*(.+)$", answer, re.M)}
    return [(round(rows[k][0]), round(len(grams(got.get(k, "")) & grams(rows[k][2])) / max(1, len(grams(rows[k][2]))), 2))
            for k in hidden]


FILES |= {"alignA": FILES["acc11a"], "alignP": FILES["acc11p"], "alignN": lambda: []}
ASK |= {"alignA": align_ask(False), "alignP": align_ask(True),
        # control: no audio, so whatever it gets right comes from the neighbouring rows
        "alignN": align_ask(False).replace("附件是一段 11 分钟的日语录音。", "这次录音没有附上，凭上下文尽量猜。")
                                  .replace("听录音，逐字写出", "逐字猜出")}


def attachment(path):
    mime = "audio/mp4" if path.suffix == ".m4a" else "video/mp4"
    return {"type": "file", "file": {"file_id": path.name, "filename": path.name, "format": mime,
                                     "local_path": str(path), "agy_prepared": True}}


def main():
    driver = AgyLocalAgentDriver(AgyDriverConfig(model="gemini-3.8-flash", timeout_seconds=900))
    out = WORK / "probes.json"
    for arm in sys.argv[1].split(","):
        user = {"role": "user", "content": [{"type": "text", "text": ASK[arm] + TAIL}, *(attachment(f) for f in FILES[arm]())]}
        started = time.time()
        result = driver.run([{"role": "system", "content": SYSTEM}, user], task="probe-media-cap", session_scope="assignment",
                            conversation_key=f"cap-{arm}-{int(started)}", conversation_handle="")
        row = {"arm": arm, "conversation": result.conversation_handle, "answer": result.content.strip(),
               "seconds": round(time.time() - started, 1), "ledger": ledger(result.conversation_handle)}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
        out.write_text(json.dumps(old + [row], ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
