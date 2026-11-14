"""Row truth (findings 「真值素材的限制」, 行真值; the closed plan: `git show 423458df:docs/plans/asr-row-truth-plan.md`): a speaker truth for every ASR row of one frozen ASR result.

One agy call (Gemini 3.8 Flash; effort high by default) per window, with the window's audio (production's vocal track), the ASR
rows, the acoustic layer's labels as production writes them in `spk` (owner 2026-10-12: the ruler gets everything,
the acoustic labels included), the reference lines of the same span (refined subtitles for hina / onetonton, the
on-screen text and nameplates for 魔裁) and the people the reference names. The prompt does not narrow what it may say:
off-cast people get names it makes up, and it may say which acoustic track is whom. `--no-labels` builds the same call
without the acoustic labels, the anchoring control of the plan's phase 1.

Windows follow production's correction windows (owner 2026-10-12: no fixed row count): one call per correction window
whose core rows reach into one of round 9's eleven scoring windows, judging its core rows with CONTEXT rows shown either
side. Rows, labels and the correction plan come from the acceptance run (`tmp/diarization-retrack/accept/<ident>/`).
The two validation slices around the owner's hand labels sit on the stable the owner's review was cut from
(`window_judge_r3.load`); their correction windows are planned with production's planner and the acceptance runs'
limits and profile, which reproduce the acceptance plans window for window.

    python tools/diarization_explore/row_truth.py check                     # ASR rows the same as the scored ones?
    python tools/diarization_explore/row_truth.py build [--only hina-71-c0001] [--no-labels]   # render, count, no calls
    python tools/diarization_explore/row_truth.py run --only hina-71-c0001 --rep 1 [--no-labels] [--effort high]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_accept as acc  # noqa: E402

r9 = acc.r9
r3 = acc.r3
lr = r9.lr
lj = r3.lj

from finesub.speech.diarization.labels import speaker_cell, speaker_of  # noqa: E402

OUT = lr.OUT / "row-truth"
CONTEXT = 10        # rows shown either side of the judged ones
PAD = 1.0           # seconds of audio before the first and after the last shown row
#: the owner's review rows outside the eleven windows (plan「校验」): (ident, first row, end row) on r3's stable
SLICES = (("hina-06", 544, 649), ("onetonton-13", 415, 597))
#: what the acceptance runs' correction plans were planned with (their plan.json): reproduces them window for window
PLAN_LIMITS = dict(prompt_input_limit=194000, output_limit=65536)
PLAN_PROFILE = dict(media="text", retrieval="local", difficulty="quality")
CLS = ("one", "multi", "bg", "filler", "halluc", "unsure")
#: agy names the effort in the model id (`gemini-3.8-flash-high`); 3.7 Flash, which built the first phase-1 rulers of
#: hina-71 and hina-24, left agy on 2026-10-12; each ruler records its model
MODEL = "gemini-3.8-flash"
AGY_TIMEOUT = 3600
#: agy has no other way to attach media in print mode; without the length and the "complete" the model has taken the
#: media for a truncated text view and gone looking for commands (denied), ending the turn without an answer
#: the turn's first reply right after the media degenerated in over half the runs (`,part_1:15:00`, `}`): the model
#: continues the media. A one-line text file read in the same step, after the audio, puts text last again (read as a
#: separate step it came too late: the reply between the two reads degenerated as before).
AUDIO_NOTE = ("这一段的录音在 {path}，长 {seconds:.0f} 秒，另有一句提示在 {after}。"
              "在同一次回复里调用两次 view_file，先读录音、后读提示（录音会完整返回，不会截断），两者都读完再开始判断；"
              "之后不要再读文件，也不要运行任何命令。")
AFTER_AUDIO = "录音已经完整读完。现在按上面的要求判断，直接输出 <people>、<rows>、<missing>、<tracks> 四个块；不要转写录音。\n"
SRC = ("ref", "infer")
PRIO = ("main", "side")
ASR = ("ok", "miss")
#: styles that carry no speech (notes, screen text, options); other non-person styles (`全员`, `pv旁白`, `路人`…) are
#: speech and stay in the reference under their own name, though `refined_multi` drops them as "not one person"
NOT_SPEECH = re.compile(r"注释|水印|watermark|STAFF|staff|声明|标题|提示|选项|大字|答案|名片|赞助|温馨提醒|非台词|右侧字幕|底部|顶部|"
                        r"战斗知识|系统解锁|闪现文字|邀约|^Default$|^default$")

MATERIALS = {
    "hina": dict(
        title="一档游戏相关的日语节目（声优以所配角色的身份出演的特别节目，标题略）",
        scene="素材是一档游戏相关的日语节目：声优以所配角色的身份出演、主持人串场，中间可能插播游戏的宣传片（PV）或剧情片段。",
        bg="例如插播的宣传片（PV）、剧情片段、中场短片、小动画里的配音。已确认属于播放片子的声音就算 `bg`，哪怕出演的声优就是这个角色，"
           "哪怕片中的角色对着观众说话、参考的样式名直接写这个角色；认得出时 `人` 写角色名，`src` 按下面 `src` 一项的规则。"
           "辨认播放片子的线索有报幕放片或宣布休息、参考里的片段标记、前后话题与接话关系、声音场景或音质的变化，要结合起来判断："
           "没有接话、出现混响都不能单独证明是片子（这份录音做过人声分离，配乐多半已经去掉）。"
           "出演者在当前节目中用角色身份主持、对谈或念台词不算 `bg`。",
        reference="粉丝字幕组校对、翻译过的字幕，说话人取自字幕的样式名：这一档多数写的是角色名，也有不指某一个人的样式，"
                  "如 `全员`、`众人`（几个人一起说）、`pv旁白`、`小动画旁白`（播放的视频里的声音）、`路人`。"
                  "说话人后的 `日` / `中` 只标这一行是日文原文还是中文译文，不是名字的一部分；有日文原文的句子只列日文那行。"),
    "onetonton": dict(
        title="一档日语多人对谈节目（标题略）",
        scene="素材是一档日语多人对谈节目。",
        bg="例如节目里播放的视频、广告的声音。",
        reference="粉丝字幕组翻译成中文、校对过的字幕，说话人取自字幕的样式名（这一档写的是真人的名字）。只有中文译文，没有日文原文。"),
    "manosaba": dict(
        title="一部日本视觉小说游戏的剧情录像（无解说，标题略）",
        scene="素材是一部日本视觉小说游戏的剧情录像，没有解说：每句台词是一个游戏角色的配音。",
        bg="这份素材里，游戏角色的台词都不是 `bg`（当面说的、打电话说的都一样，按 `one` 定是哪个角色）；`bg` 只指剧情里另外播放出来的声音，例如电视、广播、录音。",
        reference="游戏画面上台词框里的文字（OCR 识别，可能有错字）与台词框上方的名牌，每条台词带说话的角色；"
                  "说话人写作 `旁白` 的是没有名牌的叙述文字，不配音，录音里没有。"),
}
CAVEATS = {
    "refined": "它是另一个人另做的，不是照这份 ASR 改的：切行、合句、拆句不同是正常的；时间只是大致对得上，结束时间常被延后，"
               "时间重叠不等于是同一句，文字也对得上才算对应；附和、语气词、口头禅常被删掉，参考里没有不代表录音里没有；"
               "参考里可能有与语音无关的行（注释、屏幕文字）；参考自己也会标错说话人或漏句，与录音明显冲突时以录音为准，在 `note` 里写明。",
    "ocr": "OCR 可能有错字、偶尔漏掉一条台词，名牌偶尔认错或缺失；画面上别的文字（手机界面、聊天记录、提示）也会混进来，那些不是台词。"
           "带括号的心声可能配了音，也可能没有，以录音为准。时间是台词框显示的时间，比配音长：时间重叠不等于是同一句，"
           "文字也对得上才算对应。与录音明显冲突时以录音为准，在 `note` 里写明。",
}
LABELS = dict(
    head="声学标签|",
    note="- **声学标签**：一个自动分人模型给每行挂的轨号（`s0`、`s1`…，只在这一份录音里有效，编号不是人名）。"
         "`s5+s4`：这一行时间里两条轨都有声音，前面那条占得多；空着：没挂上；`s3?`：模型自己对这一行没把握，"
         "括号里是它的备选轨，如 `s3?(s5)`；`s4*`：原本没挂上、按前后补上的。\n",
    rule="4. **声学标签是证据之一，不是答案。** 它多数时候是对的，但已知会错：几个声音相近的人并进同一条轨"
         "（一条轨里可能是几个人，一个人也可能分在两条轨），短句挂错轨，没检出的行挂不上。"
         "它与参考或录音冲突时，以参考和录音为准。\n",
    tracks="声学标签里出现过的轨（只写轨号本身，不带 `?`、`*`、括号），你看得出它是谁、或混了哪几个人的，每行 `s5 = 名字`，混了人的写 `s5 = 名字A + 名字B + …（混了人）`；看不出的不写，没有就留空。",
    #: the example's label column (`$l0`…) and its tracks block
    example=dict(l0="s0|", l1="s1|", l2="s2|", l3="s1?(s2)|", l4="s0+s1|", lnone="|",
                 tracks="s0 = 佐藤一郎 + 男声A（混了人）\ns1 = 山田花子 + 田中健（混了人）\ns2 = 田中健 + pv旁白（混了人）"),
)
NO_LABELS = dict(head="", note="", rule="", tracks="这一轮没有声学标签，留空。",
                 example=dict(l0="", l1="", l2="", l3="", l4="", lnone="", tracks=""))


def kind(ident):
    return "manosaba" if ident.startswith("manosaba-") else "onetonton" if ident.startswith("onetonton-") else "hina"


# --- material -----------------------------------------------------------------------------------------

def cores(ident, source):
    """[(chunk id, first core row, end row)] of the file's correction windows."""
    if source == "accept":
        return [(w.chunk, w.low, w.high) for w in acc.judge_windows(ident)]
    import tempfile

    from finesub.llm.chunking import load_segments_from_stable_json, plan_correction_windows
    from finesub.llm.routing.config import ModelLimits
    from finesub.llm.routing.profiles import resolve_profile
    from finesub.speech.diarization.judgments import segment_ids
    stable = r3.load(ident).stable
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stable.json"
        path.write_text(json.dumps({"segments": stable}, ensure_ascii=False), encoding="utf-8")
        planned = plan_correction_windows(load_segments_from_stable_json(path), limits=ModelLimits(**PLAN_LIMITS),
                                          profile=resolve_profile(**PLAN_PROFILE))
    position = {str(key): index for index, key in enumerate(segment_ids(stable))}
    out = []
    for w in planned:
        core = [str(s) for s in w.source_ids[len(w.overlap_segments):]]
        out.append((w.chunk_id, position[core[0]], position[core[-1]] + 1))
    return out


def windows():
    """[(key, ident, first judged row, end row, source)]: every correction window whose core reaches into a scoring
    window (round 9's eleven) or a validation slice."""
    wanted = [(w["ident"], w["a"], w["b"], "accept") for w in r9.plan()] + [(i, a, b, "review") for i, a, b in SLICES]
    out = []
    for ident, a, b, source in wanted:
        for chunk, low, high in cores(ident, source):
            key = f"{ident}-c{chunk}"
            if low < b and high > a and key not in {o[0] for o in out}:
                out.append((key, ident, low, high, source))
    return out


def ready():
    """Before a readout: the frozen inputs (`frozen_inputs`), then round 9's windows load, hina first (loading
    onetonton rebinds hina's reference reader)."""
    import io
    from contextlib import redirect_stdout

    from tools.ts_diar_explore.common import frozen_inputs

    frozen_inputs(HERE / "row_truth_frozen.sha256", r9.MAIN / "data" / "diarization" / "row-truth")
    with redirect_stdout(io.StringIO()):
        for w in sorted(r9.plan(), key=lambda w: not w["ident"].startswith("hina")):
            r9.load(w["ident"])


def rows_and_cells(ident, source):
    """(stable segments, spk cell per row, vocal path)."""
    if source == "accept":
        stable = json.loads(acc.stable_path(ident).read_text(encoding="utf-8"))["segments"]
        cells = [speaker_cell([speaker_of(s)]) for s in stable]
        return stable, cells, acc.folder(ident) / f"{ident}-vocal.ogg"
    m = r3.load(ident)
    return m.stable, [speaker_cell([s]) for s in m.speakers], m.vocal


def reference_lines(ident):
    """[(start, end, speaker shown, text)] of the whole file."""
    if kind(ident) == "manosaba":
        import manosaba_truth as mt
        return [(a, b, who or "旁白", text) for a, b, text, who in mt.cues(ident, mt.names())]
    if kind(ident) == "onetonton":
        import refined_ass as ra
        path = next(p for p in ra.episodes() if int(re.search(r"#(\d+)", p).group(1)) == int(ident[-2:]))
        return [(a, b, who, text) for a, b, who, text in ra.events(path) if who != ra.NOTE and text]
    import refined_multi as rm
    # the path `refined_multi.sources()` gives, read here: loading onetonton rebinds that function, and the id list
    # lives in the main checkout's tmp/, not a worktree's
    title = json.loads(acc.IDS.read_text(encoding="utf-8"))[ident]["title"]
    ass = r9.MAIN / "data" / "refined-multi" / "希娜小姐应援会" / f"{title}.ass"
    out = []
    for line in ass.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        parts = line.split(",", 9)
        style = parts[3].strip()
        text = re.sub(r"\{[^}]*\}", "", parts[9]).replace(r"\N", " ").strip()
        if NOT_SPEECH.search(style) or not text or text.startswith("※"):
            continue
        who = rm.speaker(style) or re.sub(r"\s+正文$", "", re.sub(r"-(中文|日文)\d*$", "", style)).strip()
        lang = "日" if re.search(r"日文\d*$", style) else "中" if re.search(r"中文\d*$", style) else ""
        out.append((rm.seconds(parts[1]), rm.seconds(parts[2]), f"{who} {lang}".strip(), text))
    japanese = {(a, b, who[:-2]) for a, b, who, _ in out if who.endswith(" 日")}
    return [r for r in out if not (r[2].endswith(" 中") and (r[0], r[1], r[2][:-2]) in japanese)]


def cast(ident, refs):
    """The people the reference names: for hina, the styles `refined_multi` reads as one person."""
    names = {re.sub(r" [日中]$", "", who) for *_, who, _ in refs} - {"旁白"}
    if kind(ident) == "hina":
        import refined_multi as rm
        names = {n for n in names if rm.speaker(n) == n}
    if kind(ident) == "manosaba":
        import manosaba_window_judge as mw
        aliases = mw.display_names()
    else:
        import window_judge_r5 as r5
        aliases = r5.ALIASES
    return [f"{n}（{aliases[n]}）" if n in aliases else n for n in sorted(names)]


def cut_audio(vocal, lo, hi, target):
    if not target.exists():  # written aside and renamed, as the agy clip
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(f"{target.stem}.{os.getpid()}.part.wav")  # per process: parallel runs may cut the same window
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{lo:.3f}", "-to", f"{hi:.3f}", "-i", str(vocal),
                        "-ac", "1", "-ar", str(lj.SR), str(part)], check=True)
        part.replace(target)
    return target


def call(key, ident, a, b, source, labels=True, folder=None):
    """(system, user, judged ids, wav, audio seconds)."""
    stable, cells, vocal = rows_and_cells(ident, source)
    shown = range(max(0, a - CONTEXT), min(len(stable), b + CONTEXT))
    lo = max(0.0, float(stable[shown[0]]["start"]) - PAD)
    hi = float(stable[shown[-1]]["end"]) + PAD
    at = lambda t: f"{float(t) - lo:.1f}"  # noqa: E731
    parts = LABELS if labels else NO_LABELS
    rows = [("▶ " if a <= i < b else "") + f"q{i}|{at(stable[i]['start'])}–{at(stable[i]['end'])}|"
            + (f"{cells[i]}|" if labels else "") + str(stable[i].get("text", "")).strip() for i in shown]
    refs_all = reference_lines(ident)
    refs = [f"{at(s)}–{at(e)}|{who}|{' '.join(text.split())}" for s, e, who, text in refs_all if e > lo and s < hi]
    spec = MATERIALS[kind(ident)]
    system = lr.template("row_truth_system.md").substitute(
        scene=spec["scene"], reference_note=spec["reference"],
        reference_caveats=CAVEATS["ocr" if kind(ident) == "manosaba" else "refined"],
        label_head=parts["head"], labels_note=parts["note"], labels_rule=parts["rule"], bg_examples=spec["bg"],
        tracks_block=parts["tracks"], example=lr.template("row_truth_example.md").substitute(parts["example"]).strip())
    ids = [f"q{i}" for i in range(a, b)]
    user = lr.template("row_truth_user.md").substitute(
        title=spec["title"], cast="、".join(cast(ident, refs_all)), refs="\n".join(refs) or "（这一段没有）",
        label_head=parts["head"], rows="\n".join(rows), count=len(ids))
    wav = cut_audio(vocal, lo, hi, (folder or OUT / "audio") / f"{key}.wav")
    return system, user, ids, wav, hi - lo


# --- answers --------------------------------------------------------------------------------------------

def parse(text, ids, texts):
    """{id: row} when every judged id is answered with a well-formed `id|cls|people|prio|src|asr|note` line, else None.
    A row is {"cls", "who" (people in order), "parts" ([{"from", "who"}] for multi, else None), "prio" (`main` / `side`
    for one and multi, else ""), "src", "asr" (`ok` / `miss`, "" for halluc), "note"}, plus "split_fallback": True on a
    multi row whose split points did not line up and were folded into one part; a stray `|` in the note is kept in it
    rather than failing the window;
    `texts` ({id: ASR text}) checks that a multi row's parts start where its text says, in order."""
    if not text or text.lstrip().lower().startswith("thought"):
        return None
    lines = block(text, "rows")
    rows = {}
    for line in lines:
        cells = [c.strip() for c in line.split("|")]
        # aligned on the src field: an extra empty field before it (`q5|bg||||infer|…`, seen in answers rewritten
        # after a cut-off) is dropped; extra text there still fails; the note keeps any stray `|`
        at = next((j for j in range(4, len(cells)) if cells[j] in SRC), None)
        if at is None or len(cells) < at + 2 or any(cells[2:at - 1][1:]) or cells[0] not in texts or cells[0] in rows:
            return None
        key, cls, people, prio, src, asr, note = (cells[0], cells[1], cells[2], cells[at - 1], cells[at], cells[at + 1],
                                                 "|".join(cells[at + 2:]))
        if cls not in CLS or src not in SRC or (prio not in PRIO if cls in ("one", "multi") else prio):
            return None
        if (asr not in ASR) if cls != "halluc" else asr:
            return None
        parts, fallback = None, False
        if cls == "multi":
            pieces = [piece.split(">", 1) if ">" in piece else [piece, ""] for piece in people.split(";")]
            parts = [{"from": start.strip(), "who": [n.strip() for n in names.split("+") if n.strip()]} for names, start in pieces]
            if not all(part["who"] for part in parts):
                return None
            who = list(dict.fromkeys(n for part in parts for n in part["who"]))
            cursor = 0
            for part in parts:
                found = texts[key].find(part["from"], cursor) if part["from"] else -1
                if found < 0:  # split points that do not line up with the ASR text: kept as the prompt's own
                    # fallback (one part naming everyone, split unclear) rather than losing the window over one row
                    parts, fallback = [{"from": texts[key][:6], "who": who}], True
                    break
                cursor = found + 1
        else:
            who = [people] if people else []
            if cls == "one" and not who:
                return None
        rows[key] = {"cls": cls, "who": who, "parts": parts, "prio": prio, "src": src, "asr": asr, "note": note}
        if fallback:
            rows[key]["split_fallback"] = True
    return rows if set(rows) == set(ids) else None


def block(text, name):
    match = re.search(rf"<{name}>(.*?)</{name}>", text or "", re.S)
    return [line.strip() for line in match.group(1).splitlines() if line.strip()] if match else []


def agy(content, workdir, effort):
    """One agy print-mode turn: the task text goes in on stdin (stream-json takes text blocks only, no media, and
    `@` mentions are not expanded in print mode), the audio sits in `workdir` for the model to `view_file` once.
    -> (response, usage, conversation id, status, error)."""
    message = json.dumps({"event": "user", "message": {"content": content}}, ensure_ascii=False)
    done = subprocess.run(["agy", "--input-format", "stream-json", "--output-format", "stream-json", "--model", f"{MODEL}-{effort}",
                           "--print="], input=message + "\n", capture_output=True, text=True,
                          encoding="utf-8", cwd=str(workdir), env={**os.environ, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"},
                          timeout=AGY_TIMEOUT)
    results = [json.loads(line)["result"] for line in done.stdout.splitlines()
               if line.startswith("{") and json.loads(line).get("event") == "result"]
    if not results:
        return "", {}, "", "NO_RESULT", (done.stderr or "")[-400:]
    r = results[-1]
    return r.get("response") or "", r.get("usage") or {}, r.get("conversation_id", ""), r.get("status", ""), r.get("error", "")


def last_reply(conversation):
    """The model's answer in agy's own transcript of the conversation: its plain replies after the last tool result,
    joined. After an empty or malformed reply agy retries inside the turn; after a reply cut off at the output limit
    it asks the model to continue where it stopped, so the answer is split over two replies; either way the `result`
    event can report an error while the transcript holds the whole answer."""
    path = Path.home() / ".gemini" / "antigravity-cli" / "brain" / conversation / ".system_generated" / "logs" / "transcript_full.jsonl"
    if not conversation or not path.exists():
        return ""
    steps = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tools = [k for k, s in enumerate(steps) if s.get("type") == "GENERIC" or s.get("tool_calls")]
    after = steps[tools[-1] + 1:] if tools else steps[1:]
    replies = [s.get("content") or "" for s in after if s.get("type") == "PLANNER_RESPONSE" and not s.get("tool_calls")]
    replies = [r for r in replies if r.strip()]
    if not replies:
        return ""
    # a retry after an empty / malformed reply starts the answer over: keep from the last reply that opens a block
    opens = [k for k, r in enumerate(replies) if r.lstrip().startswith("<")]
    return "".join(replies[opens[-1] if opens else 0:])


#: bumped when the prompt or the answer format changes, so rulers from different prompts never share a folder
#: (v1: the phase-1 rulers in the untagged folders; v2: `prio`, group reactions default to multi, played clips are bg;
#: v3: `asr`, ok / bad for an obvious gap or a meaning-changing error; v4: `asr` is ok / miss, omissions only (owner))
PROMPT_TAG = "v4"


def run_dir(labels, rep, effort):
    return OUT / f"{PROMPT_TAG}-{'labels' if labels else 'nolabels'}-{effort}-r{rep}"


def run(only, labels, rep, effort):
    folder = run_dir(labels, rep, effort)
    folder.mkdir(parents=True, exist_ok=True)
    for key, ident, a, b, source in windows():
        if only and key not in only:
            continue
        path = folder / f"{key}.json"
        if path.exists():
            continue
        workdir = folder / "work" / key
        workdir.mkdir(parents=True, exist_ok=True)
        system, user, ids, wav, seconds = call(key, ident, a, b, source, labels)
        content = system + "\n\n=====\n\n" + audio_note(wav, seconds, workdir) + "\n\n" + user
        stable = rows_and_cells(ident, source)[0]
        texts = {f"q{i}": str(stable[i].get("text", "")).strip() for i in range(a, b)}
        ok, replies = ask(content, workdir, ids, texts, effort, path, key)
        print(f"{key} {'labels' if labels else 'nolabels'} {effort} r{rep}: {'ok' if ok else 'GAVE UP'} "
              f"after {replies} replies", flush=True)


def audio_note(wav, seconds, workdir):
    """The clip agy reads (`workdir/audio.mp4`) and the note telling the model to read it, then the after-audio line."""
    clip = workdir / "audio.mp4"
    if not clip.exists():  # written aside and renamed: an interrupted ffmpeg must not leave a truncated clip behind
        from finesub.media.ffmpeg import containerize_audio_for_agy
        part = clip.with_name(f"audio.{os.getpid()}.part.mp4")
        containerize_audio_for_agy(wav, part)
        part.replace(clip)
    after = workdir / "after-audio.txt"
    after.write_text(AFTER_AUDIO, encoding="utf-8")
    return AUDIO_NOTE.format(path=clip.resolve(), seconds=seconds, after=after.resolve())


def ask(content, workdir, ids, texts, effort, path, key, extra=None):
    """Up to four agy turns until one parses (an empty, `thought`-led or malformed reply is redone; the transcript is
    read when the result event lost the answer); writes `path` (with `extra` merged in), or `<path stem>.failed.json`
    after the last try.
    -> (parsed?, replies)."""
    answer, text, attempts = None, "", []
    for attempt in range(4):
        text, usage, conversation, status, error = agy(content, workdir, effort)
        attempts.append({"usage": usage, "conversation": conversation, "status": status, "error": error})
        answer = parse(text, ids, texts) if status == "SUCCESS" else None
        if answer is None:
            kept = last_reply(conversation)
            answer = parse(kept, ids, texts)
            if answer is not None:
                text = kept
                attempts[-1]["from_transcript"] = True
        if answer is not None:
            break
        if status != "SUCCESS":
            print(f"{key}: agy {status} {error[:120]}; retry", flush=True)
            time.sleep(30 * (attempt + 1))
        else:
            print(f"{key}: malformed reply; retry", flush=True)
    payload = {"rows": answer, "people": block(text, "people"), "missing": block(text, "missing"),
               "tracks": block(text, "tracks"), "raw": text, "model": f"{MODEL}-{effort}", "attempts": attempts, **(extra or {})}
    target = path if answer else path.with_suffix(".failed.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    if answer:
        path.with_suffix(".failed.json").unlink(missing_ok=True)
    return answer is not None, len(attempts)


# --- checks and dry renders -------------------------------------------------------------------------------

def check():
    """The rows a ruler is built on must be the rows that get scored: acceptance stable vs round 9's, slice by slice."""
    for key, ident, a, b, source in windows():
        if source != "accept":
            continue
        ours = json.loads(acc.stable_path(ident).read_text(encoding="utf-8"))["segments"]
        theirs = r9.load(ident).stable
        bad = [i for i in range(max(0, a - CONTEXT), min(len(ours), b + CONTEXT))
               if i >= len(theirs) or (ours[i].get("text"), round(float(ours[i]["start"]), 3)) != (theirs[i].get("text"), round(float(theirs[i]["start"]), 3))]
        print(f"{key}: {'same rows' if not bad else f'{len(bad)} rows differ, first q{bad[0]}'}")


def build(only, labels):
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    folder = OUT / "build"
    total_text = total_audio = 0
    for key, ident, a, b, source in windows():
        if only and key not in only:
            continue
        system, user, ids, wav, seconds = call(key, ident, a, b, source, labels, folder)
        text = count(system) + count(user)
        total_text += text
        total_audio += seconds * 32
        print(f"{key}: {len(ids)} judged rows, text input {text}, audio {seconds:.0f} s (~{seconds * 32:.0f} tokens)")
        (folder / f"{key}-{'labels' if labels else 'nolabels'}.md").write_text(system + "\n\n=====\n\n" + user, encoding="utf-8")
    print(f"text input {total_text}, audio ~{total_audio:.0f} tokens (thinking, output, retries not counted)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("check", "build", "run"))
    parser.add_argument("--only", nargs="*", help="window keys, e.g. hina-71-c0001")
    parser.add_argument("--no-labels", action="store_true", help="leave the acoustic labels out (anchoring control)")
    parser.add_argument("--rep", type=int, default=1)
    # high (owner): on 3.8 Flash the lower levels drop thinking on long, complex inputs; high thinks hard and now and
    # then runs past the output limit, which agy continues in place (`last_reply` joins the pieces)
    parser.add_argument("--effort", default="high", choices=("low", "medium", "high"), help="agy reasoning effort")
    args = parser.parse_args()
    ready()
    if args.command == "check":
        check()
    elif args.command == "build":
        build(args.only, not args.no_labels)
    else:
        run(args.only, not args.no_labels, args.rep, args.effort)


if __name__ == "__main__":
    main()
