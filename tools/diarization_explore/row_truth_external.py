"""Dedicated rulers for the integrated models' own rows (the closed plan's 一体模型支线, `git show 423458df:docs/plans/asr-row-truth-plan.md`).

MOSS Pro, 35t (Gemini 3.5 Transcribe, 300 s and 540 s windows) and MAI2 transcribe and diarize in one pass, so a ruler
built on production's ASR rows cannot score them: each output gets its own ruler, built on its own rows by the same
call as the main line's (owner 2026-10-12: one call per output; the reading is what that ruler changed). The model's
speaker labels stand where the acoustic labels stood (anonymised to `s0`, `s1`… in order of first appearance).

Rows, per model (owner 2026-10-12):
  MOSS Pro   its own segments; rows may overlap in time
  MAI2       its own segments (word and segment timing; no overlap, like production)
  35t        word timing only: production's sentence DP (`tools/ts_diar_explore/seg_arm.py`, arm `punct`, the
             acceptance run's VAD) over each speaker's words on their own, so rows of different speakers may overlap
             in time like MOSS Pro's. 35t interleaves the words of two people talking at once; one merged stream
             (`--merged`, the plan's first definition) glues them into mixed rows and fragments -- on the hina windows
             about half the rows -- which is our cutting, not 35t's diarization. A merged row's speaker is the one with
             most word time in it. Words longer than MAX_WORD (one ran 512 s) are dropped first.

One call per (window, output) over the output's span (the scoring window's rows and context, ±0.5 s; see
`window_external.span`): rows whose midpoint lies in the scoring window's time are judged (▶), the rest are context.
Text only by default (owner 2026-10-12: the integrated models hardly hallucinate); `--audio` builds the main line's
audio call instead, for the plan's audio-vs-text comparison on three outputs.

    python tools/diarization_explore/row_truth_external.py rows [--only hina-24-mosspro]   # cut rows, count, no calls
    python tools/diarization_explore/row_truth_external.py build [--only …] [--audio]      # render, count tokens, no calls
    python tools/diarization_explore/row_truth_external.py run --only … [--audio] [--rep 1] [--effort high]
    python tools/diarization_explore/row_truth_external.py compare [--only …]          # audio vs text, text vs its rerun

The frozen rulers (2026-10-13, 43 outputs on the eleven windows) are in `data/diarization/row-truth/external-v4-text-high/`,
hashed in `row_truth_frozen.sha256`; `row_truth_cross.py` reads them there. To measure another model (owner 2026-10-13:
the rulers are meant to be reused): add it to MODELS and cut its rows in `rows` (one label per row; rows of different
speakers may overlap), check them with `rows` and the token count with `build`, `run` one ruler per window, copy the
results into the frozen folder, append their hashes and add the route to `row_truth_cross.ROUTES`. Text only, as the
frozen ones, unless the model hallucinates (then `--audio`, and say so: the two kinds of ruler are not the same
measure). A new window needs the main line's ruler first (`row_truth.py`, audio and the acoustic labels, prompt v4,
high) on a production run of it, and what the readouts then read from tmp/ and out/ for it copied under the frozen
folder's inputs/ with its hash (`row_truth.ready` checks those before every readout; trace the readout's opens to
find them). Routes compare only on the windows they all have.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import row_truth as rt  # noqa: E402
import window_external as we  # noqa: E402

acc, r9, r3, lr = rt.acc, rt.r9, rt.r3, rt.lr

from tools.ts_diar_explore import arms as A  # noqa: E402

OUT = rt.OUT / "external"
#: arm name -> how the readouts name it
MODELS = {"mosspro": "MOSS Pro", "g35t-w300": "35t 300 s", "g35t": "35t 540 s", "mai2-diar": "MAI2"}
MAX_WORD = we.MAX_WORD
#: the pause that opens a new DP input segment when 35t is cut per speaker: tuned on the segmentation gold set
#: (must-cut recall 85.7% -> 88.4% vs 1.0 s, production 90.6%); the fansub line boundaries prefer longer pauses, so
#: 0.85 is a compromise. SBD as a DP signal gained noise-level amounts and a standalone SBD cutter did worse (2026-10-13)
PAUSE = 0.85
#: a 35t row shorter than this with at most FRAGMENT_CHARS characters (punctuation removed) is a fragment
FRAGMENT_SECONDS, FRAGMENT_CHARS = 1.0, 4

LABELS = dict(
    head="模型标签|",
    note="- **模型标签**：转写模型自己给每行标的说话人编号（`s0`、`s1`…，只在这一段里有效，编号不是人名）；空着：模型没给这一行说话人。\n",
    #: shown when the window's rows overlap (MOSS Pro, 35t cut per speaker; MAI2 now and then)
    overlap="  这个模型允许两行时间重叠：同一段时间里两个人各说各的，会写成两行。\n",
    rule_audio="4. **模型标签是证据之一，不是答案。** 它多数时候是对的，但已知会错：一个人被拆成几个编号，几个声音相近的人并进同一个编号，"
               "短句标错。它与参考或录音冲突时，以参考和录音为准。\n",
    rule_text="3. **模型标签是证据之一，不是答案。** 它多数时候是对的，但已知会错：一个人被拆成几个编号，几个声音相近的人并进同一个编号，"
              "短句标错。它与参考或内容冲突时，以参考和内容为准。\n",
    tracks="模型标签里出现过的编号，你看得出它是谁（也可以是 `pv旁白` 这类参考里的泛称）、或混了哪几个人的，每行 `s5 = 名字`，混了人的写 `s5 = 名字A + 名字B + …（混了人）`；"
           "看不出的不写，没有就留空。",
    example=dict(l0="s0|", l1="s1|", l2="s2|", l3="s1|", l4="s0|", lnone="|",
                 tracks="s0 = 佐藤一郎 + 男声A（混了人）\ns1 = 山田花子 + 田中健（混了人）\ns2 = 田中健 + pv旁白（混了人）"),
)


# --- rows ---------------------------------------------------------------------------------------------

def outputs():
    """[(key, ident, arm, span, scoring time)] for every window and model with an output on disk."""
    out = []
    for w in r9.plan():
        ident, a, b = w["ident"], w["a"], w["b"]
        lo, hi = we.span(r3.load(ident), a, b)
        stable = json.loads(acc.stable_path(ident).read_text(encoding="utf-8"))["segments"]
        scored = (float(stable[a]["start"]), float(stable[b - 1]["end"]))
        for arm in MODELS:
            if A.path(ident, arm, span=(lo, hi)).exists():
                out.append((f"{ident}-{arm}", ident, arm, (lo, hi), scored))
    return out


def dp_rows(ident, words, merged=False, pause=None):
    """35t: production's DP over each speaker's words (or, `merged`, over one stream) -> [{start, end, text, speaker,
    mixed}]."""
    from tools.ts_diar_explore import seg_arm
    vad = json.loads((acc.folder(ident) / f"{ident}-vad.json").read_text(encoding="utf-8"))["segments"]
    if not merged:  # no speaker on the words: seams fall back to pauses, as for a model without speakers
        out = []
        for speaker in sorted({w.get("speaker") for w in words}, key=str):
            mine = [{**w, "speaker": None} for w in words if w.get("speaker") == speaker]
            out += [{**r, "speaker": speaker} for r in dp_rows(ident, mine, merged=True, pause=PAUSE)]
        return sorted(out, key=lambda r: (r["start"], r["end"]))
    dp_words, _ = seg_arm.model_words({"words": words})
    rows = []
    for piece in seg_arm.split(dp_words, vad, lam=0.0, pause=pause or seg_arm.SEAM_PAUSE):
        time_by = Counter()
        for w in piece.get("words") or []:
            time_by[w.get("speaker")] += float(w["end"]) - float(w["start"])
        speakers = [s for s in time_by if s]
        lead = max(speakers, key=lambda s: (time_by[s], s)) if speakers else None
        rows.append({"start": float(piece["start"]), "end": float(piece["end"]), "text": piece.get("text", ""),
                     "speaker": lead, "mixed": len(speakers) > 1})
    return rows


def rows(ident, arm, span, merged=False):
    """(rows sorted by start, words dropped as broken): {start, end, text, speaker, mixed}."""
    out = A.load(ident, arm, span=span)
    if arm.startswith("g35t"):
        words = [w for w in out["words"] if float(w["end"]) - float(w["start"]) <= MAX_WORD]
        return sorted(dp_rows(ident, words, merged), key=lambda r: (r["start"], r["end"])), len(out["words"]) - len(words)
    segs = [{"start": float(s["start"]), "end": float(s["end"]), "text": str(s.get("text", "")).strip(),
             "speaker": s.get("speaker"), "mixed": False} for s in out["segments"] if str(s.get("text", "")).strip()]
    return sorted(segs, key=lambda s: (s["start"], s["end"])), 0


def anonymise(rows):
    """Model labels -> `s0`, `s1`… in order of first appearance in the window."""
    names = {}
    for r in rows:
        if r["speaker"] and r["speaker"] not in names:
            names[r["speaker"]] = f"s{len(names)}"
    return [{**r, "label": names.get(r["speaker"], "")} for r in rows], names


def judged(row, scored):
    return scored[0] <= (row["start"] + row["end"]) / 2 <= scored[1]


def fragment(row):
    chars = "".join(ch for ch in row["text"] if ch.isalnum())
    return row["end"] - row["start"] < FRAGMENT_SECONDS and len(chars) <= FRAGMENT_CHARS


def overlaps(rows):
    """Indices of rows that overlap another row in time."""
    hit = set()
    for i, r in enumerate(rows):
        for j in range(i + 1, len(rows)):
            if rows[j]["start"] >= r["end"]:
                break
            hit |= {i, j}
    return hit


def list_rows(only, merged):
    for key, ident, arm, span, scored in outputs():
        if only and key not in only:
            continue
        rs, dropped = rows(ident, arm, span, merged)
        rs, names = anonymise(rs)
        inside = [r for r in rs if judged(r, scored)]
        extra = ""
        if arm.startswith("g35t"):
            extra = (f", mixed {sum(r['mixed'] for r in inside)}, fragments {sum(fragment(r) for r in inside)}"
                     f", dropped words {dropped}")
        print(f"{key}: {len(inside)} judged / {len(rs)} rows, {len(names)} labels, unlabelled {sum(not r['label'] for r in inside)}, "
              f"overlapping {len(overlaps(rs))}{extra}")


# --- prompt and calls ---------------------------------------------------------------------------------

#: text mode: the main line's wording where it leans on the recording, rewritten for text alone
TEXT_REPLACE = {
    "、声音场景或音质的变化，要结合起来判断：没有接话、出现混响都不能单独证明是片子（这份录音做过人声分离，配乐多半已经去掉）。":
        "，要结合起来判断：没有接话不能单独证明是片子。",
    "与录音明显冲突时以录音为准，在 `note` 里写明。": "与内容、上下文明显冲突时按内容判断，在 `note` 里写明。",
    "带括号的心声可能配了音，也可能没有，以录音为准。": "带括号的心声可能配了音，也可能没有：ASR 写出了它，就是配了音。",
}
#: the review prompt's fix for blank answers: a print-mode turn that tries a tool is denied and ends with no answer
TEXT_NOTE = "要看的材料都在下面：不要读文件，也不要运行任何命令，直接按要求输出四个块。"


def text_only(s):
    for old, new in TEXT_REPLACE.items():
        s = s.replace(old, new)
    return s


def call(key, ident, arm, span, scored, audio, folder):
    """(system, user, judged ids, {id: text}, wav or None, seconds, record). `record` is what the ruler was built on --
    the rows (time, text, label, judged) and the label mapping -- stored with it: the rows are cut afresh from the
    output, the acceptance run's VAD and production's DP code, any of which may change or be cleaned out of tmp/."""
    rs, names = anonymise(rows(ident, arm, span)[0])
    lo, hi = span
    at = lambda t: f"{float(t) - lo:.1f}"  # noqa: E731
    shown = [("▶ " if judged(r, scored) else "") + f"q{i}|{at(r['start'])}–{at(r['end'])}|{r['label']}|{r['text']}"
             for i, r in enumerate(rs)]
    ids = [f"q{i}" for i, r in enumerate(rs) if judged(r, scored)]
    texts = {f"q{i}": r["text"] for i, r in enumerate(rs) if judged(r, scored)}
    refs_all = rt.reference_lines(ident)
    refs = [f"{at(s)}–{at(e)}|{who}|{' '.join(text.split())}" for s, e, who, text in refs_all if e > lo and s < hi]
    spec = rt.MATERIALS[rt.kind(ident)]
    caveats = rt.CAVEATS["ocr" if rt.kind(ident) == "manosaba" else "refined"]
    note = LABELS["note"] + (LABELS["overlap"] if overlaps(rs) else "")
    example = dict(LABELS["example"])
    if not audio:  # no 男声A without the recording: the text example leaves q113 unsure
        example["tracks"] = "s0 = 佐藤一郎\ns1 = 山田花子 + 田中健（混了人）\ns2 = 田中健 + pv旁白（混了人）"
    system = lr.template("row_truth_system.md" if audio else "row_truth_text_system.md").substitute(
        scene=spec["scene"], reference_note=spec["reference"],
        reference_caveats=caveats if audio else text_only(caveats),
        label_head=LABELS["head"], labels_note=note, labels_rule=LABELS["rule_audio" if audio else "rule_text"],
        bg_examples=spec["bg"] if audio else text_only(spec["bg"]), tracks_block=LABELS["tracks"],
        example=lr.template("row_truth_example.md" if audio else "row_truth_text_example.md").substitute(example).strip())
    user = lr.template("row_truth_user.md").substitute(
        title=spec["title"], cast="、".join(rt.cast(ident, refs_all)), refs="\n".join(refs) or "（这一段没有）",
        label_head=LABELS["head"], rows="\n".join(shown), count=len(ids))
    wav = rt.cut_audio(acc.folder(ident) / f"{ident}-vocal.ogg", lo, hi, (folder or OUT / "audio") / f"{key}.wav") if audio else None
    record = {"rows_shown": [{"id": f"q{i}", "start": r["start"], "end": r["end"], "text": r["text"], "label": r["label"],
                              "speaker": r["speaker"], "judged": judged(r, scored)} for i, r in enumerate(rs)],
              "labels": names, "span": [lo, hi], "scored_time": list(scored), "arm": arm, "audio": audio}
    return system, user, ids, texts, wav, hi - lo, record


def run_dir(audio, rep, effort):
    return OUT / f"{rt.PROMPT_TAG}-{'audio' if audio else 'text'}-{effort}-r{rep}"


def run(only, audio, rep, effort):
    folder = run_dir(audio, rep, effort)
    folder.mkdir(parents=True, exist_ok=True)
    for key, ident, arm, span, scored in outputs():
        if only and key not in only:
            continue
        path = folder / f"{key}.json"
        if path.exists():
            continue
        workdir = folder / "work" / key
        workdir.mkdir(parents=True, exist_ok=True)
        system, user, ids, texts, wav, seconds, record = call(key, ident, arm, span, scored, audio, None)
        note = rt.audio_note(wav, seconds, workdir) if audio else TEXT_NOTE
        content = system + "\n\n=====\n\n" + note + "\n\n" + user
        ok, replies = rt.ask(content, workdir, ids, texts, effort, path, key, extra=record)
        print(f"{key} {'audio' if audio else 'text'} {effort} r{rep}: {'ok' if ok else 'GAVE UP'} after {replies} replies",
              flush=True)


def build(only, audio):
    from finesub.llm.token_budget import default_token_counter
    count = default_token_counter().count_text
    folder = OUT / "build"
    folder.mkdir(parents=True, exist_ok=True)
    total_text = total_audio = 0
    for key, ident, arm, span, scored in outputs():
        if only and key not in only:
            continue
        system, user, ids, _, _, seconds, _ = call(key, ident, arm, span, scored, audio, folder)
        note = rt.AUDIO_NOTE.format(path="<audio.mp4>", seconds=seconds, after="<after-audio.txt>") if audio else TEXT_NOTE
        content = system + "\n\n=====\n\n" + note + "\n\n" + user
        tokens = count(content)
        total_text += tokens
        total_audio += seconds * 32 if audio else 0
        print(f"{key}: {len(ids)} judged rows, text input {tokens}" + (f", audio {seconds:.0f} s" if audio else ""))
        (folder / f"{key}-{'audio' if audio else 'text'}.md").write_text(content, encoding="utf-8")
    print(f"text input {total_text}" + (f", audio ~{total_audio:.0f} tokens" if audio else "")
          + " (thinking, output, retries not counted)")


# --- the audio-vs-text comparison ------------------------------------------------------------------------

def ruler_rows(path):
    """(stored ruler, {row index: Row}, {row index: shown row}) for one ruler file."""
    import row_truth_score as rs
    ruler = json.loads(path.read_text(encoding="utf-8"))
    shown = {int(r["id"][1:]): r for r in ruler["rows_shown"]}
    truth = {}
    for key, row in ruler["rows"].items():
        i = int(key[1:])
        text = shown[i]["text"]
        speaking = row["cls"] in ("one", "multi")
        present = frozenset(n for n in row["who"] if n != rs.EVERYONE) if speaking else frozenset()
        truth[i] = rs.Row(i, row["cls"], row["prio"], row["src"], present, rs.person_of(row, text) if speaking else None, text)
    return ruler, truth, shown


def credit(row, raw, text, person, seconds):
    """(counted seconds, right seconds) of one main row: `one` whole; `multi` split by its parts' characters, only when
    it has two parts or more; otherwise (0, 0). A part naming only `全员` (people not recognised) keeps its share and is
    never right, as `row_truth_score.person_of` leaves such a row without a main person when that part is the largest;
    a row whose parts all name only `全员` is not counted."""
    if row.cls == "one":
        return seconds, seconds * (person == row.person)
    parts = raw.get("parts") or []
    if row.cls != "multi" or len(parts) < 2:
        return 0.0, 0.0
    starts, cursor = [], 0
    for part in parts:
        cursor = text.find(part["from"], cursor)
        starts.append(cursor)
        cursor += 1
    sizes = [end - start for start, end in zip(starts, [*starts[1:], len(text)])]
    if not any(n != "全员" for part in parts for n in part["who"]):
        return 0.0, 0.0
    total = sum(sizes) or 1
    return seconds, seconds * sum(size for size, part in zip(sizes, parts) if person in part["who"]) / total


def ruler_score(path):
    """{row index: (counted seconds, right seconds)} over main rows, the judged seconds, and the ruler's tokens."""
    import types
    import row_truth_score as rs
    ruler, truth, shown = ruler_rows(path)
    cells = {i: [shown[i]["label"]] if shown[i]["label"] else [] for i in truth}
    mapping = rs.fit(types.SimpleNamespace(rows=list(truth), truth=truth), cells)
    out = {}
    for i, row in truth.items():
        if row.prio != "main" or row.cls not in ("one", "multi"):
            continue
        seconds = shown[i]["end"] - shown[i]["start"]
        label = shown[i]["label"]
        out[i] = credit(row, ruler["rows"][f"q{i}"], shown[i]["text"], mapping.get(label) if label else None, seconds)
    judged_seconds = sum(shown[i]["end"] - shown[i]["start"] for i in truth)
    tokens = sum((a.get("usage") or {}).get("total_tokens", 0) for a in ruler["attempts"])
    return out, judged_seconds, tokens


def accuracy(scores, keep):
    counted = sum(scores[i][0] for i in keep)
    return 100 * sum(scores[i][1] for i in keep) / counted if counted else float("nan")


def compare(only):
    """Per output: each ruler's coverage, accuracy and tokens; text vs audio and text vs its rerun on common rows."""
    runs = {"text": run_dir(False, 1, "high"), "rerun": run_dir(False, 2, "high"), "audio": run_dir(True, 1, "high")}
    for key, *_ in outputs():
        if only and key not in only:
            continue
        got = {name: ruler_score(folder / f"{key}.json") for name, folder in runs.items() if (folder / f"{key}.json").exists()}
        if not got:
            continue
        line = []
        for name, (scores, judged_seconds, tokens) in got.items():
            counted = {i for i, (c, _) in scores.items() if c}
            cover = 100 * sum(scores[i][0] for i in counted) / judged_seconds
            line.append(f"{name}: right {accuracy(scores, counted):.1f}% on {len(counted)} rows, coverage {cover:.0f}%, {tokens / 1000:.0f}k tokens")
        print(f"{key}: " + "; ".join(line))
        for a, b in (("text", "audio"), ("text", "rerun")):
            if a in got and b in got:
                common = {i for i in got[a][0] if got[a][0][i][0] and got[b][0].get(i, (0, 0))[0]}
                print(f"  {a} - {b} on {len(common)} common rows: {accuracy(got[a][0], common) - accuracy(got[b][0], common):+.1f}")
        if "text" in got and "audio" in got:
            print(f"  audio / text tokens: {got['audio'][2] / max(1, got['text'][2]):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("rows", "build", "run", "compare"))
    parser.add_argument("--only", nargs="*", help="output keys, e.g. hina-24-mosspro")
    parser.add_argument("--audio", action="store_true", help="the main line's audio call instead of text only")
    parser.add_argument("--rep", type=int, default=1)
    parser.add_argument("--effort", default="high", choices=("low", "medium", "high"), help="agy reasoning effort")
    parser.add_argument("--merged", action="store_true", help="rows: cut 35t over one merged word stream instead")
    args = parser.parse_args()
    rt.ready()
    if args.command == "rows":
        list_rows(args.only, args.merged)
    elif args.command == "build":
        build(args.only, args.audio)
    elif args.command == "compare":
        compare(args.only)
    else:
        run(args.only, args.audio, args.rep, args.effort)


if __name__ == "__main__":
    main()
