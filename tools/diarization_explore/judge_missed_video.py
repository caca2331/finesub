"""A look-only video of the rows the production judge left wrong (owner 2026-10-09): how many are the ruler's fault?

The aligned acceptance readout (GPT-6.1 Sol, `window_judge_accept.py`) counts 144 main rows (fresh windows, not
back-channel) that the refined-subtitle ruler calls wrong after the judge and that the judge did not move (kept or
abstained). The owner watches some of them, without labelling, to see roughly how many are the ruler's mistake rather
than the label's. Clips come from the three talk windows with the most such rows (the labelling-video recipe: a few
long windows, not many scattered ones), a seeded draw per window, at least APART seconds apart.

Each clip is the row with PAD seconds either side from the vocal track production used. On screen for the whole clip:
the row and up to four rows around it with their start in clip seconds; under each, the acoustic label and the judge's
answer (＝ kept, 弃权 abstained, or the track it moved the row to) with the person the ruler's matching gives that label,
(`未配上` when the window's one-to-one matching gave it nobody -- the scoring counts that wrong) and the name the
judge gave the track; then the refined reference lines (the ruler) in the clip. Every line turns green
while it is spoken; ticks along the bottom mark where each clip starts, for seeking. No right / wrong mark is shown.

The labelling-video recipe, for any new video the owner labels (owner 2026-10-06/07, each point a snag met while labelling).
Start from this script or `row_truth_review_video.py`, not `judge_review_video.py`:
- Pick 2-3 representative long windows (later prompt tests rerun only these); draw stratified at random with a fixed seed,
  shuffle the order, and settle what the strata are first.
- Each clip is the labelled row +/- 3 s, at most 15 s, 0.8 s of silence between clips. For the whole clip show the labelled
  row and a few rows around it with their start in clip seconds; the row being spoken lights up.
- Under the title, one line of person numbers (ranked by each person's truth-row count in the window, the same number throughout a window), so
  the owner answers "C03 2"; show every arm's label, with ＝ for unchanged; the truth subtitle may show, right / wrong never.
- At the very bottom, a `|<number>` tick where each clip starts.
- Record labels by (file, row), not clip number; record the owner's "？" separately.

    python tools/diarization_explore/judge_missed_video.py     # <MAIN>/out/judge-missed/judge-missed.mp4
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
import p2_review_video as pv  # noqa: E402
import window_judge_accept as acc  # noqa: E402
import window_judge_r9 as r9  # noqa: E402
import window_partition_score as ps  # noqa: E402

OUT = r9.MAIN / "out" / "judge-missed"
SEED = 20261010
#: the talk windows with the most rows the judge left wrong (33, 18, 18 of the 144)
FILES = {"hina-24": 7, "hina-17": 7, "hina-57": 7}
APART, PAD, GAP, MAX_CLIP = 10.0, 3.0, 0.8, 15.0
LAYERS = {"head": 18, "cast": 52, "asr": 92, "ref": 462, "why": 590, "tick": 720}
COLOURS = {"head": "&H00FFFF&", "cast": "&HFFFFFF&", "asr": "&HC0C0C0&", "ref": "&HE0B080&", "why": "&H90EE90&",
           "tick": "&H808080&"}
LIVE = "{\\c&H40FF40&}"


def candidates():
    """Every main row of the chosen windows that the ruler calls wrong after the judge and the judge did not move."""
    r3 = acc.r3
    r3.OUT = r9.ROOT / "build"
    base = r3.outcome({})
    chosen, units = acc.picks(acc.aligned_stable)
    out = []
    for ident, a, b in r9.windows():
        if ident not in FILES:
            continue
        m = r9.load(ident)
        rows = [i for i in range(a, b) if (ident, a, i) in base and base[(ident, a, i)]["labelled"]
                and not r9.backchannel(m, i)]
        labels = {"acoustic": ps.final_labels(r3, m, ident, a, b, {}), "judge": ps.final_labels(r3, m, ident, a, b, chosen)}
        found = {name: ps.match(rows, labels[name], m.truth) for name in labels}
        weight = Counter()
        for i in range(a, b):
            for person in m.truth.get(i, ()):
                weight[person] += 1 / len(m.truth[i])
        people = [person for person, _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))]
        segments = json.loads(acc.aligned_stable(ident).read_text(encoding="utf-8"))["segments"]
        from finesub.speech.diarization.judgments import segment_ids
        ids = segment_ids(segments)
        verdict, named = {}, {}
        for unit in units[ident]:
            named.update(unit.get("names") or {})
            for result in unit.get("results") or ():
                verdict[ids.index(str(result["row"]))] = result

        def ends(name, j):
            """(label, the person the window's matching gives it -- or, unmatched, `未配上` and the track's owner
            over the file); rows outside the window keep production's track."""
            lead = m.leader(j)
            label = labels[name].get(j, f"s{lead}" if lead is not None else None)
            person = found[name].get(label) if j in rows else found[name].get(label) or m.owner.get(lead)
            return label, person or f"未配上，整份轨主 {m.owner.get(lead, '?')}"

        for i in rows:
            # wrong as the readout scores it: the label's person under the window's one-to-one matching (an unmatched
            # label is wrong) -- not the display's fallbacks below
            if found["judge"].get(labels["judge"][i]) in m.truth[i] or labels["judge"][i] != labels["acoustic"][i]:
                continue
            s, e = float(m.stable[i]["start"]), float(m.stable[i]["end"])
            lo = max(0.0, s - PAD)
            hi = min(lo + MAX_CLIP, e + PAD)
            near = [j for j in range(len(m.stable)) if float(m.stable[j]["start"]) < hi and float(m.stable[j]["end"]) > lo]
            near = sorted(sorted(near, key=lambda j: (j != i, abs(j - i)))[:5])
            out.append({
                "file": ident, "window": a, "row": i, "start": s, "end": e, "clip": [lo, hi], "vocal": str(m.vocal),
                "truth": sorted(m.truth[i]), "people": people, "named": named,
                "abstain": bool(verdict.get(i, {}).get("abstain")), "why": str(verdict.get(i, {}).get("why", "")),
                "rows": [{"row": j, "start": float(m.stable[j]["start"]), "end": float(m.stable[j]["end"]),
                          "text": str(m.stable[j].get("text", "")).strip(), "acoustic": ends("acoustic", j),
                          "judge": ends("judge", j), "result": verdict.get(j)} for j in near],
            })
    return out


def pick(cands):
    rng = np.random.default_rng(SEED)
    chosen = []
    for ident, quota in FILES.items():
        pool = [c for c in cands if c["file"] == ident]
        taken = []
        for k in rng.permutation(len(pool)):
            c = pool[k]
            if all(abs(c["start"] - x["start"]) >= APART for x in taken):
                taken.append(c)
            if len(taken) == quota:
                break
        chosen += taken
    return [chosen[k] for k in rng.permutation(len(chosen))]


def judge_text(r, named):
    label, person = r["judge"]
    result = r["result"] or {}
    if result.get("abstain"):
        head = "弃权"
    elif label == r["acoustic"][0]:
        head = "＝"
    else:
        head = f"{label}（{person or '?'}）"
    name = named.get(str(label)) if label else None
    return head + (f"  裁判叫它「{name}」" if name else "")


def render():
    import soundfile as sf
    import p2_features as pf
    clips = pick(candidates())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    refs = {}
    for f in sorted({c["file"] for c in clips}):
        pf.reference(f)
        refs[f] = pv.reference_lines(f)
    events, audio, cursor, waves, starts = [], [], 0.0, {}, []
    for n, c in enumerate(clips, 1):
        lo, hi = c["clip"]
        if c["vocal"] not in waves:
            wave, rate = sf.read(c["vocal"], dtype="float32", always_2d=True)
            wave = wave.mean(axis=1)
            if rate != pv.SR:
                import librosa
                wave = librosa.resample(wave, orig_sr=rate, target_sr=pv.SR)
            waves[c["vocal"]] = wave
        audio += [waves[c["vocal"]][int(lo * pv.SR):int(hi * pv.SR)], np.zeros(int(GAP * pv.SR), dtype=np.float32)]
        t0, t1 = cursor, cursor + hi - lo
        starts.append(t0)
        cursor = t1 + GAP
        at = lambda x: t0 + (x - lo)  # noqa: E731

        def live(style, start, end, text, y):
            events.append((0, style, t0, t1, text, y))
            if start < hi and end > lo:
                body = text.split("}", 1)[-1] if text.startswith("{") else text
                events.append((1, style, at(max(lo, start)), at(min(hi, end)), LIVE + body, y))

        events.append((0, "cast", t0, t1, pv.esc("人物：" + "   ".join(f"{k} {p}" for k, p in enumerate(c["people"], 1))),
                       LAYERS["cast"]))
        events.append((0, "head", t0, t1, pv.esc(f"C{n:02d} · {c['file']} · {c['start']:.1f}s · 裁判没改到的 ▶ 行（只看，不标）"),
                       LAYERS["head"]))
        for k, r in enumerate(c["rows"]):
            mine = r["row"] == c["row"]
            colour = "{\\c&H00FFFF&}" if mine else ""
            label, person = r["acoustic"]
            y = LAYERS["asr"] + 74 * k
            live("asr", r["start"], r["end"], colour + pv.esc(f"{'▶ ' if mine else ''}{max(0.0, r['start'] - lo):4.1f}s  {r['text']}"), y)
            events.append((0, "asr", t0, t1, colour + pv.esc(f"    声学 {label}（{person or '?'}）  →  裁判 {judge_text(r, c['named'])}"),
                           y + 30))
        shown = sorted((x for x in refs[c["file"]] if x[0] < hi and x[1] > lo), key=lambda x: x[0])[:4]
        for k, (ra, rb, who, text) in enumerate(shown):
            live("ref", ra, rb, pv.esc(f"{max(0.0, ra - lo):4.1f}s  真值 {who}：{text}"), LAYERS["ref"] + 30 * k)
        if c["why"]:
            events.append((0, "why", t0, t1, pv.esc(f"裁判弃权的理由：{c['why'][:60]}"), LAYERS["why"]))
    # ticks along the bottom where each clip starts, for dragging the player's progress bar (owner 2026-10-07)
    for n, t in enumerate(starts, 1):
        x = int(20 + (pv.WIDTH - 40) * t / cursor)
        events.append((0, "tick", 0.0, cursor, f"{{\\an2\\pos({x},{LAYERS['tick']})}}|{n}", None))
    sf.write(str(OUT / "judge-missed.wav"), np.concatenate(audio), pv.SR, subtype="PCM_16")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {pv.WIDTH}
PlayResY: {pv.HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(
        f"Style: {k},Microsoft YaHei,{24 if k == 'head' else 14 if k == 'tick' else 22},{COLOURS[k]},&H000000FF,&H00000000,&H00000000,{-1 if k == 'head' else 0},0,0,0,100,100,0,0,1,1.5,0,7,20,20,0,1"
        for k in LAYERS) + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    lines = [f"Dialogue: {layer},{pv.ass_time(s0)},{pv.ass_time(s1)},{style},,0,0,0,,"
             f"{text if y is None else f'{{\\pos(20,{y})}}{text}'}" for layer, style, s0, s1, text, y in events]
    (OUT / "judge-missed.ass").write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={pv.WIDTH}x{pv.HEIGHT}:r=10:d={cursor:.2f}", "-i",
           "judge-missed.wav", "-vf", "subtitles=judge-missed.ass:fontsdir=C\\\\:/Windows/Fonts", "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", "judge-missed.mp4"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True)
    print(f"{len(clips)} clips {dict(Counter(c['file'] for c in clips))}, {cursor:.0f} s -> {OUT / 'judge-missed.mp4'}")


if __name__ == "__main__":
    render()
