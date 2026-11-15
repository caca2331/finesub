"""Two passes with the shape folded into the judge (docs/reports/2026-10-17-three-pass.md 「两 pass 原型」): production's
speaker judge (text) answers, per row, its track and also whether the row joins the line before it and whether it is
dropped; then `two_pass.py run` corrects the grouped rows with merging switched off.

    python tools/diarization_explore/judge_shape.py judge <key>...   # judge + shape on a copy of the D6 stable
    python tools/diarization_explore/two_pass.py run <key>...        # same TRIAL_LABEL: group, correct, map back

Writes into two_pass's folder (tmp/spk-merge/two-pass-<TRIAL_LABEL>/<key>/): input/ (the stable copy the judge's sidecar
sits next to, and the judge's run), judged/<key>-stable.json (the judge's verdicts applied, as `merge_trial.py judged`),
shape.json (the joins and drops, and the judge calls' usage). Model: TRIAL_MODEL, as merge_trial.

TRIAL_PASS1 picks the pass 1 (docs/reports/2026-10-17-three-pass.md 「D4、D5、D8 的试跑」): unset, the text judge plus the
shape section; `trackless`, the shape alone on rows without tracks (D4); `audio`, judge-audio plus `split` / `insert`
(D8), whose edits shape.json keeps under `edits` for `two_pass.shaped`. TRIAL_SESSION=1 runs the judge on agy's
continued-conversation path and logs its conversation (input/sessions.jsonl) for pass 2 to continue (D5). DRY=1 only
renders the prompts.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import unicodedata
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import merge_trial as mt  # noqa: E402
import two_pass as tp  # noqa: E402

from finesub.llm.prompt_constants import THRESHOLDS  # noqa: E402

FORMAT = ("`开始秒|轨号|文字`", "`开始秒|时长|gap|轨号|文字`")
FIELDS = ("每行三个字段：`id`、`track`（你的答案）、`why`（一句理由）。",
          "每行三个字段：`id`、`track`（你的答案）、`why`（一句理由）；后面「定字幕的形状」一节还有三个可选字段。")
#: the shape section's wording that depends on whether pass 1 also judges the tracks (shape_rules.md)
TRACKED = {"intro": "除了轨号，每个 ▶ 行还可以带三个字段，写进同一行 JSON；用不上就不写，也不要写 `false`。",
           "same_person": "按你判完的说话人，不同的人绝不 join；这一行或它要接的那一行答了 `abstain`，不 join。轨号相同本身不算同一个人"
                          "（一条轨里可能混着几个声音相近的人），要从对话结构、称呼与口吻看出是同一个人接着说。",
           "drop_track": "drop 的行照常答 `track`。", "example_track": '"track":"keep",'}
TRACKLESS = {"intro": "每个 ▶ 行可以带下面三个字段之一，写进这一行的 JSON；三个都用不上的行不写。",
             "same_person": "这里没有说话人编号，要从对话结构、称呼与口吻看出是同一个人接着说。",
             "drop_track": "", "example_track": ""}


def shape_rules(words):
    return Template((HERE.parent / "shape_rules.md").read_text(encoding="utf-8")).substitute(THRESHOLDS, **words)


#: PASS1=audio: appended to the shape section; the geometry only listening can fix (docs/llm_harness_behavior.md
#: 「说话人裁判」, three passes' pass 1)
AUDIO_SHAPE_RULES = """
有音频时，还可以做三件只有听才能做的事（同样写进这一行的 JSON，用不上就不写）：
- `"split":"<切点之前的原文>"`：这一行里换了人，从换人处切成两行。`track` 答前半段的说话人，`"track2"` 答后半段的（轨号或 `@标签`，不能写 `keep`、`abstain`）；值逐字照抄这一行从开头到切点的原文。只在听得出是两个人说的、听得出切点时用；一个人说的话不切，句中停顿也不切。一行里换了几次人时，只切最清楚的一处；原文里找不到对应切点的（ASR 写漏或写乱了），不切。
  同一行既写 `join`（或 `may_join`）又写 `split`：前半段接到上一行，后半段单独成一条；下一行写 `join` 或 `may_join`，接的是后半段。
- `"fill":"<补全之后这一行的原文>"`：这一行只写出了一句话的一部分，录音里同一个人紧接着这一行的开头之前或结尾之后还说了、ASR 漏掉的，补上。值是补全后的整行：原有的字一字不改、顺序不变，只在开头或结尾加上听到的部分；行中间漏的字不补，原有的字听错了也不在这里改（下一步纠错会改）。只补听得清的、这一行那句话没说完的部分：同一个人中间停了约 1 秒以上、隔着别人的话、或另起了一句的，不算，用下面的 `insert`。一行只写 `split` 与 `fill` 中的一个，两件都要做时写 `split`。
- `"insert":{"start":<开始秒>,"end":<结束秒>,"text":"<听到的原文>","track":"<轨号或 @标签>"}`：这一行之后有 ASR 整段漏掉、听得清的话（另一个人说的，或同一个人另起的一句；别人说的半句也算），补一行。这一行写下的只是附和、插话这类次要的声音，同时另一个人在说的话被盖掉、没写出来的，也算漏掉，照样补，这一行照常留着。`start`、`end` 用字幕行同样的时间轴，是你听到的这句话的起止，落在这一行开始到下一行结束之间（可以和原有的行重叠）。只补听得清、有内容的话；听不清的、只有语气声或附和声的不补。一行之后只补一句，漏了几句时补最清楚的那句；没有下一行时，落在这一行结束后 5 秒以内。补出的行单独成一条：它不接别人，它后面的那一行也不隔着它去接。
"""
#: PASS1=audio: production's sentences that the two fields above change (system prompt, audio note)
AUDIO_EDITS = (("判这一行**大部分是谁说的**。", "判这一行**大部分是谁说的**；听得出是两个人、听得出切点时，按「定字幕的形状」里的 `split` 切开。"),
               ("不用为它们找行。", "不用为它们找行；有内容、听得清的整句漏写，按「定字幕的形状」里的 `insert` 补一行。"),
               ("后面「定字幕的形状」一节还有三个可选字段。", "后面「定字幕的形状」一节还有几个可选字段。"))
#: which pass 1: "" (production's judge plus the shape section), "trackless" (the shape alone, no tracks shown or
#: answered: D4), "audio" (judge-audio plus split / insert: D8)
PASS1 = os.environ.get("TRIAL_PASS1", "")
_TRACK = re.compile(r"s(\d+)")


def trackless_prompt(rows, window, *, title, summary, cast, **_):
    """D4's prompt: the judge's window and candidates, the rows without tracks, the shape section alone."""
    from finesub.llm.stages.speaker_judge import prompt as jp

    system = Template((HERE.parent / "trackless_judge_system.md").read_text(encoding="utf-8")).substitute(
        summary=jp._summary(summary), rules=shape_rules(TRACKLESS))
    lines, ids = [], []
    for index in window.shown(rows):
        row = rows[index]
        judged = window.low <= index < window.high
        if judged:
            ids.append(f"q{row.id}")
        gap = f"{row.start - rows[index - 1].end:.1f}" if index > 0 else ""
        lines.append((f"▶ q{row.id} " if judged else "") + f"{row.start:.1f}|{row.seconds:.1f}|{gap}|{row.text}")
    user = Template((HERE.parent / "trackless_judge_user.md").read_text(encoding="utf-8")).substitute(
        title=title, cast="、".join(c.display for c in cast) or jp.PHRASES["no_cast"], rows="\n".join(lines),
        count=len(ids))
    return system, user, ids


def patched_judge(args):
    """Production's correction run with the judge's prompt and parser extended, stopping once the judge is done."""
    from finesub.llm import correction_translation, prompt_compose
    from finesub.llm.stages.speaker_judge import answers as ja
    from finesub.llm.stages.speaker_judge import prompt as jp
    from finesub.llm.stages.speaker_judge import run as jr
    from finesub.llm.stages.speaker_judge import task as jt

    out, lock = Path(os.environ["SHAPE_OUT"]), threading.Lock()
    found = {"acts": {}, "splits": {}, "fills": {}, "inserts": [], "refused": []}
    layouts = {}  # judged id -> (where the window's recording starts in the audio file, the source time it starts at)
    if os.environ.get("TRIAL_SESSION_LOG"):
        import session_patch
        session_patch.install(Path(os.environ["TRIAL_SESSION_LOG"]))
    original_template, original_prompt, original_parse = prompt_compose.load_prompt_template, jr.judge_prompt, jr.parse_judge

    def template(name, **kwargs):
        text = original_template(name, **kwargs).replace(*FORMAT).replace(*FIELDS)
        if name == jp.SYSTEM_TEMPLATE:
            text += shape_rules(TRACKED) + (AUDIO_SHAPE_RULES if PASS1 == "audio" else "")
        if PASS1 == "audio":
            for old, new in AUDIO_EDITS:
                text = text.replace(old, new)
        return text

    def judge_prompt(rows, window, **kwargs):
        system, user, ids = original_prompt(rows, window, **kwargs)
        audio = kwargs.get("audio")
        when = (lambda row: audio.offset + row.start - audio.origin) if audio is not None else (lambda row: row.start)
        for index in window.shown(rows):
            row = rows[index]
            shown = f"{jp.cell(row, marks=kwargs['marks'])}|{row.text}"
            gap = f"{row.start - rows[index - 1].end:.1f}" if index > 0 else ""
            old = f"{when(row):.1f}|{shown}"
            assert old in user, f"row {row.id} not found in the judge prompt"
            user = user.replace(old, f"{when(row):.1f}|{row.seconds:.1f}|{gap}|{shown}", 1)
        if audio is not None:
            layouts.update({key: (audio.offset, audio.origin) for key in ids})
        return system, user, ids

    def track_token(value, known):
        value = str(value or "").strip()
        match = _TRACK.fullmatch(value)
        if match is not None and int(match.group(1)) in known:
            return value
        return value if value.startswith("@") and value[1:].strip() else None

    def parse_judge(text, ids, tracks):
        if PASS1 == "trackless":
            verdicts = {key: ja.Verdict("keep") for key in ids}  # no tracks asked: the acoustic ones stand
            body = ja._block(text, "judge")
        else:
            verdicts = original_parse(text, ids, tracks)  # refuses what production refuses
            body = re.search(r"<judge>(.*?)</judge>", text, re.S).group(1)
        rows = {}
        for line in body.splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ja.AnswerError(f"row is not JSON: {line!r}") from error
                if str(item.get("id")) not in ids:
                    raise ja.AnswerError(f"unexpected id {item.get('id')!r}")
                rows[str(item["id"])] = item
        with lock:
            for key, item in rows.items():
                row = int(key.lstrip("q"))
                act = ("drop" if item.get("drop") is True else "join" if item.get("join") is True
                       else "may" if item.get("may_join") is True else None)
                if act:
                    found["acts"][row] = act
                if PASS1 != "audio":
                    continue
                known = set(tracks)
                if isinstance(item.get("split"), str) and item["split"].strip():
                    second = track_token(item.get("track2"), known)
                    if second is None:
                        found["refused"].append({"row": row, "edit": "split", "why": f"track2 {item.get('track2')!r}"})
                    else:
                        found["splits"][row] = {"prefix": item["split"], "track2": second}
                if isinstance(item.get("fill"), str) and item["fill"].strip():
                    if row in found["splits"]:
                        found["refused"].append({"row": row, "edit": "fill", "why": "the row is split"})
                    else:
                        found["fills"][row] = item["fill"].strip()
                insert = item.get("insert")
                if isinstance(insert, dict):
                    who = track_token(insert.get("track"), known)
                    try:
                        start = float(insert.get("start"))
                    except (TypeError, ValueError):
                        start = None
                    if who is None or start is None or not str(insert.get("text") or "").strip():
                        found["refused"].append({"row": row, "edit": "insert", "why": f"fields {insert!r}"})
                    elif key not in layouts:  # the route fell back to text: no audio timeline to read `start` on
                        found["refused"].append({"row": row, "edit": "insert", "why": "no audio layout"})
                    else:
                        offset, origin = layouts[key]
                        try:
                            asked_end = float(insert.get("end")) - offset + origin
                        except (TypeError, ValueError):
                            asked_end = None
                        found["inserts"].append({"row": row, "start": start - offset + origin, "end": asked_end,
                                                 "text": str(insert["text"]).strip(), "track": who})
        return verdicts

    def write():
        out.write_text(json.dumps(found, ensure_ascii=False), encoding="utf-8")

    atexit.register(write)
    prompt_compose.load_prompt_template, jp.load_prompt_template = template, template
    shaped_prompt = trackless_prompt if PASS1 == "trackless" else judge_prompt
    jr.judge_prompt, jt.judge_prompt, jr.parse_judge = shaped_prompt, shaped_prompt, parse_judge  # task: the dry run's preview
    original_safely = correction_translation._judge_safely

    def judge_then_stop(*a, **k):
        original_safely(*a, **k)
        write()
        os._exit(0)

    correction_translation._judge_safely = judge_then_stop
    sys.argv = ["correction_translation", *args]
    sys.exit(correction_translation.main())


def judged_payload(stable):
    """The stable with the speaker judge's sidecar next to it applied: each moved row gets the judged track alone."""
    from finesub.speech.diarization.judgments import read_judgments
    from finesub.subtitles.stable_rows import read_rows

    payload = json.loads(Path(stable).read_text(encoding="utf-8"))
    ids = {str(row.get("id") or n): n - 1 for n, row in enumerate(read_rows(stable), start=1)}
    for row_id, track in read_judgments(stable).moved.items():
        segment = payload["segments"][ids[row_id]]
        old = segment.get("speaker") or {}
        segment["speaker"] = {"labels": old.get("labels", []), "block": old.get("block", 0), "track": track,
                              "tracks": [track], "coverage": {str(track): 1.0}}
    return payload


def _plain(text):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text))


def cut_word(segment, prefix):
    """(word index the split goes before, None) or (None, why refused): the word boundary nearest to where
    `prefix`, compared without spaces and punctuation, ends in the row's words."""
    words = segment.get("words") or []
    whole, target = _plain("".join(w["word"] for w in words)), _plain(prefix)
    if not target or not whole.startswith(target):
        return None, "prefix does not match the row"
    if len(target) >= len(whole):
        return None, "prefix is the whole row"
    bounds, seen = [], 0
    for k, word in enumerate(words[:-1], start=1):
        seen += len(_plain(word["word"]))
        bounds.append((abs(seen - len(target)), k))
    return (min(bounds)[1], None) if bounds else (None, "one word")


def edits(segs, found):
    """shape.json's `edits`: the audio judge's splits snapped to word boundaries and its inserts on the source
    timeline, tracks as numbers (a `@label` gets a number of its own from 1000 up), and what could not be applied.
    A split whose second half names the first half's track is refused (split is for a change of speaker). A fill must
    keep the row's text whole, adding only at its head or tail (compared without spaces and punctuation); the row's
    time grows by the added text's length (speech near 7 characters a second) into the silence beside it, never over a
    neighbour. Subtitles do not overlap yet: an insert keeps the times it was asked for (for when they may) and is
    placed between its row's end and the next row's start, its end from the judge or else from its length; with
    less than 0.5 s of room it is deferred (kept here, not shown)."""
    labels, refused = {}, list(found["refused"])

    def number(token):
        match = _TRACK.fullmatch(token)
        return int(match.group(1)) if match else labels.setdefault(token, 1000 + len(labels))

    splits = {}
    for row, split in sorted(found["splits"].items(), key=lambda kv: int(kv[0])):
        if found["acts"].get(row) == "drop":
            refused.append({"row": row, "edit": "split", "why": "the row is dropped"})
            continue
        first = (segs[int(row) - 1].get("speaker") or {}).get("track")
        match = _TRACK.fullmatch(split["track2"])
        if match is not None and int(match.group(1)) == first:
            refused.append({"row": row, "edit": "split", "why": "second half is the same track"})
            continue
        at, why = cut_word(segs[int(row) - 1], split["prefix"])
        if at is None:
            refused.append({"row": row, "edit": "split", "why": why, "prefix": split["prefix"]})
            continue
        splits[str(row)] = {"at": at, "track2": number(split["track2"]), "prefix": split["prefix"]}
    fills = {}
    for row, text in sorted(found.get("fills", {}).items(), key=lambda kv: int(kv[0])):
        seg, n = segs[int(row) - 1], int(row)
        whole, plain_text = _plain(text), _plain(seg["text"])
        at = whole.find(plain_text) if plain_text else -1
        if at < 0 or len(whole) == len(plain_text):
            refused.append({"row": row, "edit": "fill", "why": "changes the row's own text" if at < 0 else "adds nothing",
                            "fill": text})
            continue
        if found["acts"].get(row) == "drop":
            refused.append({"row": row, "edit": "fill", "why": "the row is dropped"})
            continue
        before, after = fills.get(str(n - 1), segs[n - 2])["end"] if n > 1 else seg["start"] - 5.0, segs[n]["start"] if n < len(segs) else seg["end"] + 5.0
        head, tail = at, len(whole) - at - len(plain_text)
        fills[str(row)] = {"text": text, "head": head, "tail": tail,
                           "start": round(max(before, seg["start"] - head / 7), 3) if head else seg["start"],
                           "end": round(min(after, seg["end"] + tail / 7), 3) if tail else seg["end"]}
    inserts = []
    for insert in found["inserts"]:
        row = int(insert["row"])
        low = segs[row - 1]["start"]
        high = segs[row]["end"] if row < len(segs) else segs[row - 1]["end"] + 5.0
        asked_start = min(max(insert["start"], low), high)
        seconds = min(max(len(_plain(insert["text"])) / 7, 0.8), 5.0)
        asked_end = insert["end"] if insert.get("end") and insert["end"] > asked_start else asked_start + seconds
        # the room left once the fills have grown their rows
        following = (fills.get(str(row + 1), segs[row])["start"]) if row < len(segs) else float("inf")
        start = max(asked_start, fills.get(str(row), segs[row - 1])["end"])
        end = min(start + max(asked_end - asked_start, 0.5), following)
        inserts.append({"row": row, "start": round(start, 3), "end": round(end, 3),
                        "deferred": end - start < 0.5, "asked_end": round(asked_end, 3),
                        "text": insert["text"], "track": number(insert["track"]), "asked_start": round(asked_start, 3)})
    return {"splits": splits, "fills": fills, "inserts": inserts, "refused": refused, "labels": labels}


def judge(keys):
    from finesub.speech.diarization.judgments import read_judgments

    env = {**os.environ, "PYTHONPATH": str(mt.CHECKOUT / "src")}
    for key in keys:
        folder = tp.ROOT / key
        if (folder / "shape.json").exists():
            print(f"{key}: judged before")
            continue
        stable = folder / "input" / f"{key}-stable.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mt.stable_for("acoustic", key), stable)
        ident = key.rsplit("-c", 1)[0]
        context = mt.MAIN / "tmp/diarization-retrack/accept" / ident / "llm" / f"{ident}-research-context.json"
        judge_mode = ["--speaker-judge", "audio", "--audio", str(mt.MAIN / "tmp/diarization-retrack/accept" / ident
                                                                 / f"{ident}-vocal.ogg")] if PASS1 == "audio" else ["--speaker-judge", "text"]
        args = [str(stable), "--llm-model", mt.MODEL, "--context-file", str(context), "--retrieval", "none",
                "--media", "text", "--fast", "off", *judge_mode, "-o", str(folder / "input" / f"{key}.srt"),
                "--task-artifact-dir", str(folder / "input" / "llm"),
                *(["--prompt-dir", str(folder / "input" / "prompts")] if os.environ.get("DRY") else ["--execute"])]
        session = {"TRIAL_SESSION_LOG": str(folder / "input" / "sessions.jsonl")} if os.environ.get("TRIAL_SESSION") else {}
        with open(folder / "input" / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, str(HERE), "patched-judge", *args], stdout=log,
                                   stderr=subprocess.STDOUT, cwd=str(mt.CHECKOUT),
                                   env={**env, **session, "SHAPE_OUT": str(folder / "input" / "acts.json")})
        if os.environ.get("DRY"):
            print(f"{key}: dry run exit {code}, prompts in {folder / 'input' / 'prompts'}")
            continue
        verdicts = read_judgments(stable)
        if code != 0 or verdicts is None or not (folder / "input" / "acts.json").exists():
            print(f"{key}: judge failed (exit {code})")
            continue
        payload = judged_payload(stable)
        judged = folder / "judged" / f"{key}-stable.json"
        judged.parent.mkdir(parents=True, exist_ok=True)
        judged.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        usage = [json.loads(line)["payload"]["usage"]
                 for line in (folder / "input" / "llm" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("kind") == "speaker_judge_response"]
        found = json.loads((folder / "input" / "acts.json").read_text(encoding="utf-8"))
        found["acts"] = {str(k): v for k, v in found["acts"].items()}
        shape = {"acts": found["acts"], "join_seconds": THRESHOLDS["hard_seconds"], "may_join": True,
                 "pass1": PASS1 or "tracked", "calls": [{"usage": u} for u in usage]}
        if PASS1 == "audio":
            shape["edits"] = edits(payload["segments"], found)
        (folder / "shape.json").write_text(json.dumps(shape, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: {len(verdicts.moved)} rows moved, acts {sorted(set(found['acts'].values()))} x{len(found['acts'])}"
              + (f", splits {len(shape['edits']['splits'])}, inserts {len(shape['edits']['inserts'])}, "
                 f"fills {len(shape['edits']['fills'])}, refused {len(shape['edits']['refused'])}" if PASS1 == "audio" else ""), flush=True)


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    {"judge": judge, "patched-judge": patched_judge}[mode](rest)
