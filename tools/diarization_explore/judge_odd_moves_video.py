"""A review video of the rows the audio judges keep moving onto the other sex (owner 2026-10-29): who really speaks?

The ruler on these rows is the row truth (3.8 Flash with audio), which the owner has not heard here. Each clip is the
row with PAD seconds either side (at most MAX_CLIP), cut from the window's vocal track, in a fixed shuffled order. On
screen: the cast line (the window's people numbered by rows in the ruler), the ▶ row and up to four rows around it with
their start in the clip, what the ruler gives and the acoustic layer's track; for the ▶ row, how every audio and text
judge run in `judge_moves_readout.RUNS` left it; the subtitle group's lines (reference) under them; ticks `|n` along the
bottom where each clip starts. Answers go into `OWNER` keyed by (window, row id), then `score`.

    python tools/diarization_explore/judge_odd_moves_video.py render [cut]   # <MAIN>/out/judge-odd-moves/<cut>.mp4
    python tools/diarization_explore/judge_odd_moves_video.py score [cut]
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import judge_moves_readout as M  # noqa: E402
import p2_review_video as pv  # noqa: E402
import row_truth as rt  # noqa: E402

OUT = M.S.r9.MAIN / "out" / "judge-odd-moves"
SEED = 20261029
#: cut -> (window, row ids): P8's targets (`agy_judge_interrogate.py`), none of them in the owner's earlier labels or
#: the row-truth review videos
CUTS = {"hina-46-p8": ("hina-46-c0001", ("171", "189", "192", "214", "221", "291", "295", "337"))}
PAD, GAP, MAX_CLIP = 3.0, 0.8, 15.0
LAYERS = {"head": 18, "cast": 52, "asr": 92, "ref": 560, "tick": 720}
COLOURS = {"head": "&H00FFFF&", "cast": "&HFFFFFF&", "asr": "&HC0C0C0&", "ref": "&HE0B080&", "tick": "&H808080&"}
LIVE = "{\\c&H40FF40&}"

#: (window, row id) -> what the owner heard in the ▶ row: people from the cast line; None = cannot tell
OWNER: dict[tuple[str, str], set | None] = {}


def runs_on(wins, results, key, i):
    """One line per judge group: how its runs left row i ('没动' or the track and the person it maps to)."""
    groups = {}
    for name, (folder, model, media, *_rest) in M.RUNS.items():
        if (name, key) not in results:
            continue
        silent = "silence" in name
        group = f"{'38F' if model == '3.8F' else model} {'静音' if silent else '带音频' if media == 'audio' else '纯文本'}"
        got = results[name, key][0].get(i)
        groups.setdefault(group, Counter())[f"改成 {got['new_label']}（{got['new']}）" if got else "没动"] += 1
    return [f"{g} {sum(c.values())} 遍：" + "，".join(f"{n} 遍{what}" for what, n in c.most_common())
            for g, c in groups.items()]


def clips(cut):
    key, ids = CUTS[cut]
    wins, results = M.load(None)
    win = wins[key]
    index = {k: i for i, k in enumerate(win.ids)}
    people = [p for p, _ in win.weight.most_common()]
    vocal = M.SPK / M.RUNS["38F split r1"][0] / key / "input" / f"{key}-vocal.ogg"
    out = []
    for rid in ids:
        i = index[rid]
        s, e = float(win.segs[i]["start"]), float(win.segs[i]["end"])
        lo = max(0.0, s - PAD)
        hi = min(lo + MAX_CLIP, e + PAD)
        near = [j for j, g in enumerate(win.segs) if float(g["start"]) < hi and float(g["end"]) > lo]
        near = sorted(sorted(near, key=lambda j: (j != i, abs(j - i)))[:5])

        def track(j):
            label = (win.acoustic.get(j) or [None])[0]
            return f"{label}（{win.amap.get(label, '?')}）" if label else "??"
        out.append({"key": key, "row": rid, "start": s, "clip": [lo, hi], "vocal": str(vocal), "people": people,
                    "runs": runs_on(wins, results, key, i),
                    "rows": [{"row": win.ids[j], "start": float(win.segs[j]["start"]), "end": float(win.segs[j]["end"]),
                              "text": str(win.segs[j].get("text", "")).strip(),
                              "truth": win.truth[j].person or win.truth[j].cls if j in win.truth else "（不在尺子里）",
                              "track": track(j)} for j in near]})
    rng = np.random.default_rng(SEED)
    return [out[k] for k in rng.permutation(len(out))]


def render(cut):
    import soundfile as sf
    chosen = clips(cut)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{cut}-clips.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    refs = rt.reference_lines(chosen[0]["key"].rsplit("-c", 1)[0])
    wave, rate = sf.read(chosen[0]["vocal"], dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != pv.SR:
        import librosa
        wave = librosa.resample(wave, orig_sr=rate, target_sr=pv.SR)
    events, audio, cursor, starts = [], [], 0.0, []
    for n, c in enumerate(chosen, 1):
        lo, hi = c["clip"]
        audio += [wave[int(lo * pv.SR):int(hi * pv.SR)], np.zeros(int(GAP * pv.SR), dtype=np.float32)]
        t0, t1 = cursor, cursor + hi - lo
        starts.append(t0)
        cursor = t1 + GAP
        at = lambda x: t0 + (x - lo)  # noqa: E731

        def live(style, start, end, text, y):
            events.append((0, style, t0, t1, text, y))
            if start < hi and end > lo:
                body = text.split("}", 1)[-1] if text.startswith("{") else text
                events.append((1, style, at(max(lo, start)), at(min(hi, end)), LIVE + body, y))
        events.append((0, "head", t0, t1, pv.esc(f"C{n:02d} · {c['key']} · {c['start']:.1f}s · 标 ▶ 那一行是谁在说（人物编号或名字）"), LAYERS["head"]))
        events.append((0, "cast", t0, t1, pv.esc("人物：" + "   ".join(f"{k} {p}" for k, p in enumerate(c["people"], 1))), LAYERS["cast"]))
        y = LAYERS["asr"]
        for r in c["rows"]:
            mine = r["row"] == c["row"]
            colour = "{\\c&H00FFFF&}" if mine else ""
            live("asr", r["start"], r["end"], colour + pv.esc(f"{'▶ ' if mine else ''}{max(0.0, r['start'] - lo):4.1f}s  {r['text']}"), y)
            events.append((0, "asr", t0, t1, colour + pv.esc(f"    尺子 {r['truth']}   声学层 {r['track']}"), y + 28))
            y += 60
            if mine:
                for line in c["runs"]:
                    events.append((0, "asr", t0, t1, colour + pv.esc(f"    {line}"), y))
                    y += 26
                y += 8
        shown = sorted((x for x in refs if x[0] < hi and x[1] > lo), key=lambda x: x[0])[:4]
        for k, (ra, rb, who, text) in enumerate(shown):
            live("ref", ra, rb, pv.esc(f"{max(0.0, ra - lo):4.1f}s  字幕组 {who}：{text}"), LAYERS["ref"] + 27 * k)
    for n, t in enumerate(starts, 1):
        x = int(20 + (pv.WIDTH - 40) * t / cursor)
        events.append((0, "tick", 0.0, cursor, f"{{\\an2\\pos({x},{LAYERS['tick']})}}|{n}", None))
    sf.write(str(OUT / f"{cut}.wav"), np.concatenate(audio), pv.SR, subtype="PCM_16")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {pv.WIDTH}
PlayResY: {pv.HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(
        f"Style: {k},Microsoft YaHei,{24 if k == 'head' else 14 if k == 'tick' else 19 if k == 'ref' else 20},{COLOURS[k]},&H000000FF,&H00000000,&H00000000,{-1 if k == 'head' else 0},0,0,0,100,100,0,0,1,1.5,0,7,20,20,0,1"
        for k in LAYERS) + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    lines = [f"Dialogue: {layer},{pv.ass_time(s0)},{pv.ass_time(s1)},{style},,0,0,0,,"
             f"{text if y is None else f'{{\\pos(20,{y})}}{text}'}" for layer, style, s0, s1, text, y in events]
    (OUT / f"{cut}.ass").write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={pv.WIDTH}x{pv.HEIGHT}:r=10:d={cursor:.2f}", "-i", f"{cut}.wav",
           "-vf", f"subtitles={cut}.ass:fontsdir=C\\\\:/Windows/Fonts", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "96k", "-shortest", f"{cut}.mp4"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True)
    print(f"{len(chosen)} clips, {cursor:.0f} s -> {OUT / (cut + '.mp4')}")


def score(cut):
    agree = Counter()
    for n, c in enumerate(json.loads((OUT / f"{cut}-clips.json").read_text(encoding="utf-8")), 1):
        heard = OWNER.get((c["key"], c["row"]), "missing")
        if heard == "missing":
            continue
        truth = next(r["truth"] for r in c["rows"] if r["row"] == c["row"])
        ok = heard is not None and truth in heard
        agree["ruler agrees" if ok else "ruler disagrees" if heard is not None else "owner cannot tell"] += 1
        print(f"C{n:02d} {c['key']} q{c['row']}: ruler {truth}, owner {heard} -> {'ok' if ok else 'NO'}")
    print(dict(agree))


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else next(iter(CUTS)))
