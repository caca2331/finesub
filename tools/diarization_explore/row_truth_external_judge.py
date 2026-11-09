"""An LLM speaker judge on the integrated models' own rows (owner 2026-10-13: try a judge on MOSS Pro, 35t and MAI2,
on one to three representative windows, agy text-only and with audio once each), scored by their frozen dedicated
rulers (`row_truth_external.py`).

The judge is round 9's (`window_judge_r9.py`: its system prompt, the open options, the user template), on agy Gemini
3.8 Flash high -- round 9 ran 3.7 Flash, gone from agy since 2026-10-12, at the research role's high. It reads the rows
the ruler judged and their context (`rows_shown`), each with the model's label (`s0`, `s1`… as the ruler saw them),
the cast (the people the ruler found in the window, as round 9 gave the true speakers), and per label a profile: its
share of the window's speech, up to REF_LINES example rows (8 characters or more, context rows first) and, with audio,
up to two voice samples (rows of 2–5 s). The audio arm gets the samples and the window's span of production's vocal
track, the rows' times moved into the file. Calls go through the agy CLI (`row_truth.agy`) with the model named, as the
rulers did, not through the production agy driver.

A judged row takes the judge's pick when it is a label (`s3`) or a new one (`@名字`); `keep` / `abstain` leave the
model's label. The ruler saw the model's labels, so it leans towards leaving them (anchoring; on the main line about
0.4 points): a judge's moves are, if anything, under-credited.

    python tools/diarization_explore/row_truth_external_judge.py run [--windows hina-24 ...] [--arms mosspro ...] [--modality text audio]
    python tools/diarization_explore/row_truth_external_judge.py score
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_retrack as lr  # noqa: E402
import llm_retrack_judge as lj  # noqa: E402
import row_truth as rt  # noqa: E402
import row_truth_cross as rc  # noqa: E402
import row_truth_external as ext  # noqa: E402
import row_truth_score as rs  # noqa: E402
import window_judge_r3 as r3  # noqa: E402
import window_judge_r9 as r9  # noqa: E402

OUT = rt.OUT / "external-judge"  # runs; the answers scored are the frozen copies in FROZEN
FROZEN = rs.DATA / "external-judge-high"  # text/, audio/ (2026-10-13)
WINDOWS = ("hina-24", "hina-47", "manosaba-c")  # dev normal (MOSS behind Sol), held-out normal, held-out 魔裁
ARMS = ("mosspro", "g35t-w300", "mai2-diar")
EFFORT = "high"
AFTER = "录音已经完整读完。现在按上面的要求判断，只输出 <judge> 与 <tracks> 两个块。\n"
TEXT_NOTE = "要看的材料都在下面：不要读文件，也不要运行任何命令，直接按要求输出 <judge> 与 <tracks> 两个块。"


def ruler(ident, arm):
    return ext.ruler_rows(rc.FROZEN / f"{ident}-{arm}.json")


def profiles(shown):
    """(profile text, {label: voice-sample rows}): per label its share of the rows' time, example rows (context first,
    then the longest) and voice samples (rows of lj.SAMPLE seconds, context first)."""
    seconds = lambda r: r["end"] - r["start"]  # noqa: E731
    by = {}
    for r in shown.values():
        if r["label"]:
            by.setdefault(r["label"], []).append(r)
    share = {k: sum(seconds(r) for r in v) for k, v in by.items()}
    total = sum(share.values()) or 1.0
    order = lambda r: (r["judged"], -len(r["text"]))  # noqa: E731
    lines, voice = [], {}
    for label in sorted(by, key=lambda k: -share[k]):
        rows = by[label]
        picks = sorted((r for r in rows if len(re.sub(r"[\W_]", "", r["text"])) >= 8), key=order)[:r3.REF_LINES]
        voice[label] = sorted((r for r in rows if lj.SAMPLE[0] <= seconds(r) <= lj.SAMPLE[1]), key=order)[:2]
        lines.append(f"- {label}：占语音 {share[label] / total * 100:.0f}%。例句："
                     + (" / ".join(f"「{r['text'].strip()}」" for r in sorted(picks, key=lambda r: r["start"])) or "（没有可靠的例句）"))
    return "\n".join(lines), voice


def call(ident, arm, modality, workdir):
    """(content, judged ids)."""
    stored, truth, shown = ruler(ident, arm)
    kind = rt.kind(ident)
    spec = r3.MATERIALS[kind]
    profile, voice = profiles(shown)
    rows = [shown[i] for i in sorted(shown, key=lambda i: (shown[i]["start"], i))]
    when = lambda t: f"{t:.1f}"  # noqa: E731
    note = TEXT_NOTE
    if modality == "audio":
        import soundfile as sf
        from tools.ts_diar_explore.common import vocal_track
        lo, hi = stored["span"]
        wav_all = workdir / "span.wav"
        rt.cut_audio(vocal_track(ident), lo, hi, wav_all)  # every row shown lies in the span
        wave, sr = sf.read(str(wav_all), dtype="float32")
        at_lo = lambda t: int(max(0.0, t - lo) * sr)  # noqa: E731
        gap = np.zeros(int(lj.GAP * sr), dtype=np.float32)
        pieces, layout, cursor = [], [], 0.0
        for label in sorted(voice):
            for r in voice[label]:
                piece = wave[at_lo(r["start"]):at_lo(r["end"])]
                layout.append(f"| {label} 的声音样本 | {cursor:.1f}–{cursor + len(piece) / sr:.1f} 秒 |")
                pieces += [piece, gap]
                cursor += len(piece) / sr + lj.GAP
        at = cursor
        pieces.append(wave)
        layout.append(f"| 这一段字幕的录音 | {at:.1f}–{at + hi - lo:.1f} 秒 |")
        silent = [label for label in sorted(voice) if not voice[label]]
        profile += ("\n\n音频布局：\n| 段落 | 音频文件内时间 |\n| --- | --- |\n" + "\n".join(layout)
                    + (f"\n\n没有声音样本的轨：{'、'.join(silent)}" if silent else ""))
        wav = workdir / "judge.wav"
        sf.write(str(wav), np.concatenate(pieces), sr, subtype="PCM_16")
        when = lambda t: f"{at + t - lo:.1f}"  # noqa: E731
        note = rt.audio_note(wav, at + hi - lo, workdir)
        (workdir / "after-audio.txt").write_text(AFTER, encoding="utf-8")
    text = "\n".join((f"▶ {r['id']} " if r["judged"] else "") + f"{when(r['start'])}|{r['label'] or '??'}|{r['text'].strip()}"
                     for r in rows)
    ids = [r["id"] for r in rows if r["judged"]]
    cast = sorted({p for row in truth.values() for p in row.present})
    user = lr.template("window_judge_r3_user.md").substitute(
        title=spec["title"], cast="、".join(cast) or "（无）", profiles=profile, rows=text, count=len(ids))
    system = r9.system_prompt(modality, types.SimpleNamespace(kind=kind))
    return system + "\n\n=====\n\n" + note + "\n\n" + user, ids


def run(windows, arms, modalities):
    for modality in modalities:
        folder = OUT / f"{modality}-{EFFORT}"
        folder.mkdir(parents=True, exist_ok=True)
        for ident in windows:
            for arm in arms:
                path = folder / f"{ident}-{arm}.json"
                if path.exists() or not (rc.FROZEN / f"{ident}-{arm}.json").exists():
                    continue
                workdir = folder / "work" / f"{ident}-{arm}"
                workdir.mkdir(parents=True, exist_ok=True)
                content, ids = call(ident, arm, modality, workdir)
                answer, text, attempts = None, "", []
                for attempt in range(4):
                    text, usage, conversation, status, error = rt.agy(content, workdir, EFFORT)
                    attempts.append({"usage": usage, "conversation": conversation, "status": status, "error": error})
                    answer = lr.parse(text, "judge", "id", ids)
                    if answer is None and (kept := rt.last_reply(conversation)):
                        if (answer := lr.parse(kept, "judge", "id", ids)) is not None:
                            text = kept
                    if answer is not None:
                        break
                    print(f"{ident}-{arm} {modality}: {status} {error[:80]}; retry", flush=True)
                payload = {"answer": answer, "raw": text, "model": f"{rt.MODEL}-{EFFORT}", "attempts": attempts}
                (path if answer else path.with_suffix(".failed.json")).write_text(
                    json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{ident}-{arm} {modality}: {'ok' if answer else 'GAVE UP'} after {len(attempts)}", flush=True)


def scored(ident, arm, picks=None):
    """The model's rows on one window, its labels or the judge's `picks` {id: track}, by its ruler (as
    `row_truth_cross.external`); -> (sums, minutes of the scoring window, moved rows)."""
    stored, truth, shown = ruler(ident, arm)
    rows = sorted(truth)
    cells, moved = {}, 0
    for i in rows:
        label = shown[i]["label"]
        pick = str(((picks or {}).get(f"q{i}") or {}).get("track", "")).strip()
        if re.fullmatch(r"s\d+", pick) or pick.startswith("@"):
            moved += pick != label
            cells[i] = [pick]
        else:
            cells[i] = [label] if label else []
    times = {i: (shown[i]["start"], shown[i]["end"]) for i in rows}
    every = [shown[i] for i in sorted(shown)]
    overlapping = frozenset(int(every[k]["id"][1:]) for k in ext.overlaps(every))
    sums, _ = rc.score_rows(truth, {i: stored["rows"][f"q{i}"] for i in rows}, rows, cells, times, overlapping)
    lo, hi = stored["scored_time"]
    return sums, (hi - lo) / 60, moved


def score():
    print("per window: edit cost per minute (the cross readout's: cost + merge + ASR merge), right % of main-row time; "
          "moved = judged rows whose label the judge changed")
    folders = sorted(p for p in FROZEN.iterdir() if p.is_dir())
    total = Counter()
    for ident in WINDOWS:
        for arm in ARMS:
            if not (rc.FROZEN / f"{ident}-{arm}.json").exists():
                continue
            base, minutes, _ = scored(ident, arm)
            cost = lambda s: (s["cost"] + s["merge"] + s["asr_merge"]) / minutes  # noqa: E731
            right = lambda s: 100 * s["right"] / max(s["counted"], 1e-9)  # noqa: E731
            line = f"  {ident:11s} {ext.MODELS[arm]:9s} alone {cost(base):5.1f} ({right(base):5.1f}%)"
            total[(arm, "alone")] += cost(base) * minutes
            total[(arm, "minutes")] += minutes
            for folder in folders:
                path = folder / f"{ident}-{arm}.json"
                if path.exists():
                    s, _, moved = scored(ident, arm, json.loads(path.read_text(encoding="utf-8"))["answer"])
                    line += f" | +{folder.name} {cost(s):5.1f} ({right(s):5.1f}%, moved {moved})"
                    total[(arm, folder.name)] += cost(s) * minutes
            print(line)
    print("all windows tried:")
    for arm in ARMS:
        m = total[(arm, "minutes")]
        if m:
            print(f"  {ext.MODELS[arm]:9s} alone {total[(arm, 'alone')] / m:.1f}" + "".join(
                f" | +{f.name} {total[(arm, f.name)] / m:.1f}" for f in folders if total[(arm, f.name)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "score", "render"])
    ap.add_argument("--windows", nargs="*", default=list(WINDOWS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--modality", nargs="*", default=["text", "audio"])
    args = ap.parse_args()
    rt.ready()
    if args.cmd == "run":
        run(args.windows, args.arms, args.modality)
    elif args.cmd == "render":  # dry: write one call's content per (window, arm, modality) under OUT/render
        for modality in args.modality:
            for ident in args.windows:
                for arm in args.arms:
                    work = OUT / "render" / f"{modality}-{ident}-{arm}"
                    work.mkdir(parents=True, exist_ok=True)
                    content, ids = call(ident, arm, modality, work)
                    (work / "content.md").write_text(content, encoding="utf-8")
                    print(f"{modality} {ident} {arm}: {len(ids)} judged rows, {len(content)} chars")
    else:
        score()


if __name__ == "__main__":
    main()
