"""A review video of the speaker judges' errors under the row truth (owner 2026-10-12): are they real errors?

Rows of round 9's scoring windows that the row truth scores (main `one` / `multi` with a person) and that a judge ends
wrong on, under the readout's rule (each label to the person with the most of its rows in the window, many-to-one):
Sol (production's acceptance run) and round 9's agy 3.7 Flash judges on raw ASR (text, audio). Drawn with a fixed seed,
QUOTA per judge, at most PER_WINDOW clips per window and APART seconds between clips of one file; a row wrong for several
judges counts for the first stratum that draws it.

Each clip is the row with PAD seconds either side, cut from production's vocal track. On screen: the cast line (the
window's people numbered by rows in the truth), the ▶ row and up to four rows around it, each with the person the
truth gives and what production and every judge end on (label and the person it maps to; ＝ when a judge left
production's label), then the reference lines and what each judge wrote on the ▶ row; every line turns green while it
is spoken; ticks along the bottom mark
where each clip starts, for seeking. Labels go into `OWNER`
(keyed by file and row), then `score`.

    python tools/diarization_explore/row_truth_review_video.py render [cut]   # <MAIN>/out/row-truth-review/<cut>.mp4
    python tools/diarization_explore/row_truth_review_video.py score [cut]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import window_judge_r9 as r9  # noqa: E402

# round 9's loader first: listing the validation slices loads onetonton, which rebinds hina's reference reader
LOADED = {w["ident"]: r9.load(w["ident"]) for w in sorted(r9.plan(), key=lambda w: not w["ident"].startswith("hina"))}
import p2_review_video as pv  # noqa: E402
import row_truth as rt  # noqa: E402
import row_truth_score as rs  # noqa: E402

OUT = r9.MAIN / "out" / "row-truth-review"
SEED = 20261012
JUDGES = (("Sol", "sol"), ("agy 文字", "text-raw"), ("agy 音频", "audio-raw"))
#: the cuts: name -> clips per judge; a later cut skips the rows of the earlier ones (owner 2026-10-12: the first was
#: mostly Sol's errors, the second is agy's, the audio judge first)
CUTS = {"row-truth-review": {"sol": 8, "text-raw": 6, "audio-raw": 6},
        "row-truth-review-agy": {"audio-raw": 14, "text-raw": 6}}
PER_WINDOW, APART = 4, 10.0
PAD, GAP, MAX_CLIP = 3.0, 0.8, 15.0
LAYERS = {"head": 18, "cast": 52, "asr": 92, "ref": 532, "why": 616, "tick": 720}
COLOURS = {"head": "&H00FFFF&", "cast": "&HFFFFFF&", "asr": "&HC0C0C0&", "ref": "&HE0B080&", "why": "&H90EE90&",
           "tick": "&H808080&"}
LIVE = "{\\c&H40FF40&}"

#: (file, row) -> what the owner heard in the ▶ row: names (a person number from the cast line is entered as the name it
#: stands for); None = cannot tell
OWNER: dict[tuple[str, int], set | None] = {}


def judge_cells(ident, a, judge, acoustic):
    """{row: [labels]} of one judge: production's cells with the judge's moves applied."""
    if judge == "sol":
        return rs.candidate_cells(ident, "sol")
    out = dict(acoustic)
    for (kid, ka, i), pick in r9.picks_of(judge, 1).items():
        if kid == ident and ka == a and (re.fullmatch(r"s\d+", pick) or pick.startswith("@")):
            out[i] = [pick]
    return out


def reasons(ident, a, judge):
    """{row: (why, the judge's own name for the track it chose)} of one judge in one scoring window. Sol writes a reason
    only for rows it changes or abstains on; round 9's agy judges wrote one for every row."""
    out = {}
    if judge == "sol":
        from finesub.speech.diarization.judgments import segment_ids, sidecar_path
        path = rs.PRODUCTS / "sol" / ident / f"{ident}-stable.json"
        index = {key: i for i, key in enumerate(segment_ids(rs.stable(ident, "sol")))}
        for window in json.loads(sidecar_path(path).read_text(encoding="utf-8"))["windows"]:
            for unit in window.get("parts") or [window]:
                names = unit.get("names") or {}
                for result in unit.get("results") or []:
                    chosen = f"s{result['track']}" if "track" in result else f"@{result['label']}" if "label" in result else ""
                    out[index[str(result["row"])]] = (result.get("why", ""), names.get(chosen, ""))
        return out
    path = r9.pass_dir(1, judge) / f"{ident}-{a}.json"
    for key, answer in (json.loads(path.read_text(encoding="utf-8")).get("answer") or {}).items():
        out[int(key[1:])] = (str(answer.get("why", "")), "")
    return out


def candidates():
    """Every scored row some judge ends wrong on, with what the video needs."""
    out = []
    spans = [(k, kid, lo, hi) for k, kid, lo, hi, src in rt.windows() if src == "accept"]
    for w in rs.windows("score"):
        a = w.rows[0]
        acoustic = rs.candidate_cells(w.ident, "acoustic")
        cells = {"production": acoustic, **{judge: judge_cells(w.ident, a, judge, acoustic) for _, judge in JUDGES}}
        mapping = {name: rs.fit(w, c) for name, c in cells.items()}
        why = {judge: reasons(w.ident, a, judge) for _, judge in JUDGES}
        weight = Counter(p for i in w.rows for p in w.truth[i].present)
        people = [p for p, _ in sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))]
        segments = rs.stable(w.ident)
        ruler = rs.rulers()

        def ends(name, j):
            labels = cells[name].get(j) or []
            label = labels[0] if labels else None
            return label, mapping[name].get(label) if label else None

        def note(j):
            key = next(k for k, kid, lo, hi in spans if kid == w.ident and lo <= j < hi)
            return ruler[key][f"q{j}"].get("note", "")
        for i in w.rows:
            row = w.truth[i]
            if row.prio != "main" or not row.person:
                continue
            wrong = [judge for _, judge in JUDGES if ends(judge, i)[1] != row.person]
            if not wrong:
                continue
            s, e = float(segments[i]["start"]), float(segments[i]["end"])
            lo = max(0.0, s - PAD)
            hi = min(lo + MAX_CLIP, e + PAD)
            near = [j for j in range(len(segments)) if float(segments[j]["start"]) < hi and float(segments[j]["end"]) > lo]
            near = sorted(sorted(near, key=lambda j: (j != i, abs(j - i)))[:5])
            out.append({
                "file": w.ident, "row": i, "wrong": wrong, "start": s, "end": e, "clip": [lo, hi],
                "vocal": str(rt.acc.folder(w.ident) / f"{w.ident}-vocal.ogg"), "people": people,
                "truth": rs.truth_cell(row), "note": note(i),
                "why": {judge: why[judge].get(i, ("", "")) for _, judge in JUDGES},
                "rows": [{"row": j, "start": float(segments[j]["start"]), "end": float(segments[j]["end"]),
                          "text": str(segments[j].get("text", "")).strip(),
                          "truth": rs.truth_cell(w.truth[j]) if j in w.truth else "（不在尺子里）",
                          "ends": {name: ends(name, j) for name in cells}} for j in near]})
    return out


def pick(cands, quota, taken):
    rng = np.random.default_rng(SEED)
    pools = {judge: [cands[k] for k in rng.permutation([k for k, c in enumerate(cands) if judge in c["wrong"]])]
             for judge in quota}
    chosen, need, taken = [], dict(quota), set(taken)
    while any(need.values()):
        progressed = False
        for judge in quota:
            while need[judge] and pools[judge]:
                c = pools[judge].pop(0)
                same = [x for x in chosen if x["file"] == c["file"]]
                if (c["file"], c["row"]) in taken or len(same) >= PER_WINDOW \
                        or any(abs(x["start"] - c["start"]) < APART for x in same):
                    continue
                chosen.append({**c, "stratum": judge})
                taken.add((c["file"], c["row"]))
                need[judge] -= 1
                progressed = True
                break
        if not progressed:
            raise SystemExit(f"cannot fill the quota: still need {need}")
    return [chosen[k] for k in rng.permutation(len(chosen))]  # strata mixed on screen


def label_text(label, person, same=False):
    if label is None:
        return "??"
    return f"{'＝' if same else ''}{label}（{person or '?'}）"


def render(name):
    import soundfile as sf
    earlier = list(CUTS)[:list(CUTS).index(name)]
    taken = {(c["file"], c["row"]) for cut in earlier
             for c in json.loads((OUT / f"{cut}-clips.json").read_text(encoding="utf-8"))}
    saved = OUT / f"{name}-clips.json"
    if saved.exists():   # the same clips, redrawn with what the tool shows now
        keep = {(c["file"], c["row"]): c["stratum"] for c in json.loads(saved.read_text(encoding="utf-8"))}
        order = list(keep)
        found = {(c["file"], c["row"]): c for c in candidates() if (c["file"], c["row"]) in keep}
        clips = [{**found[k], "stratum": keep[k]} for k in order]
    else:
        clips = pick(candidates(), CUTS[name], taken)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}-clips.json").write_text(json.dumps(clips, ensure_ascii=False, indent=1), encoding="utf-8")
    refs = {f: rt.reference_lines(f) for f in sorted({c["file"] for c in clips})}
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
        events.append((0, "cast", t0, t1, pv.esc("人物：" + "   ".join(f"{k} {p}" for k, p in enumerate(c["people"], 1))), LAYERS["cast"]))
        events.append((0, "head", t0, t1, pv.esc(f"C{n:02d} · {c['file']} · {c['start']:.1f}s · 标 ▶ 那一行是谁在说（人物编号或名字）"), LAYERS["head"]))
        for k, r in enumerate(c["rows"]):
            mine = r["row"] == c["row"]
            colour = "{\\c&H00FFFF&}" if mine else ""
            prod = r["ends"]["production"]
            judges = "  ".join(f"{label} {label_text(*r['ends'][judge], same=r['ends'][judge][0] == prod[0])}" for label, judge in JUDGES)
            y = LAYERS["asr"] + 88 * k
            live("asr", r["start"], r["end"], colour + pv.esc(f"{'▶ ' if mine else ''}{max(0.0, r['start'] - lo):4.1f}s  {r['text']}"), y)
            events.append((0, "asr", t0, t1, colour + pv.esc(f"    尺子 {r['truth']}"), y + 28))
            events.append((0, "asr", t0, t1, colour + pv.esc(f"    生产 {label_text(*prod)}  →  {judges}"), y + 56))
        shown = sorted((x for x in refs[c["file"]] if x[0] < hi and x[1] > lo), key=lambda x: x[0])[:3]
        for k, (ra, rb, who, text) in enumerate(shown):
            live("ref", ra, rb, pv.esc(f"{max(0.0, ra - lo):4.1f}s  参考 {who}：{text}"), LAYERS["ref"] + 27 * k)
        # what each judge wrote on the ▶ row (owner 2026-10-12): Sol only on rows it changed or abstained on
        for k, (label, judge) in enumerate(JUDGES):
            text, named = c["why"].get(judge) or ("", "")
            body = (text or ("（没动这一行，没写理由）" if judge == "sol" else "（没写理由）")) + (f"（它叫这条轨 {named}）" if named else "")
            events.append((0, "why", t0, t1, pv.esc(f"{label} 的理由：{body[:62]}"), LAYERS["why"] + 28 * k))
    # ticks along the bottom where each clip starts, for dragging the player's progress bar (owner 2026-10-07)
    for n, t in enumerate(starts, 1):
        x = int(20 + (pv.WIDTH - 40) * t / cursor)
        events.append((0, "tick", 0.0, cursor, f"{{\\an2\\pos({x},{LAYERS['tick']})}}|{n}", None))
    sf.write(str(OUT / f"{name}.wav"), np.concatenate(audio), pv.SR, subtype="PCM_16")
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {pv.WIDTH}
PlayResY: {pv.HEIGHT}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(
        f"Style: {k},Microsoft YaHei,{24 if k == 'head' else 14 if k == 'tick' else 19 if k in ('ref', 'why') else 21},{COLOURS[k]},&H000000FF,&H00000000,&H00000000,{-1 if k == 'head' else 0},0,0,0,100,100,0,0,1,1.5,0,7,20,20,0,1"
        for k in LAYERS) + "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    lines = [f"Dialogue: {layer},{pv.ass_time(s0)},{pv.ass_time(s1)},{style},,0,0,0,,"
             f"{text if y is None else f'{{\\pos(20,{y})}}{text}'}" for layer, style, s0, s1, text, y in events]
    (OUT / f"{name}.ass").write_text(header + "\n".join(lines) + "\n", encoding="utf-8-sig")
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"color=c=0x101010:s={pv.WIDTH}x{pv.HEIGHT}:r=10:d={cursor:.2f}", "-i", f"{name}.wav",
           "-vf", f"subtitles={name}.ass:fontsdir=C\\\\:/Windows/Fonts", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
           "-b:a", "96k", "-shortest", f"{name}.mp4"]
    subprocess.run(cmd, cwd=OUT, check=True, capture_output=True)
    print(f"{len(clips)} clips {dict(Counter(c['stratum'] for c in clips))}, {cursor:.0f} s -> {OUT / (name + '.mp4')}")


def score(name):
    clips = json.loads((OUT / f"{name}-clips.json").read_text(encoding="utf-8"))
    agree = Counter()
    for n, c in enumerate(clips, 1):
        heard = OWNER.get((c["file"], c["row"]), "missing")
        if heard == "missing":
            continue
        truth = re.sub(r"^(one|multi) |（.*|·次$", "", c["truth"]).split("+")
        ok = heard is not None and bool(set(truth) & heard)
        agree["ruler agrees" if ok else "ruler disagrees" if heard is not None else "owner cannot tell"] += 1
        print(f"C{n:02d} {c['file']} q{c['row']} {c['stratum']}: truth {c['truth']}, owner {heard} -> {'ok' if ok else 'NO'}")
    print(dict(agree))


if __name__ == "__main__":
    {"render": render, "score": score}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else "row-truth-review")
