"""Two-pass prototype (docs/reports/2026-10-17-three-pass.md 「两 pass 原型」): a shape pass decides merges and whole-row
drops on the judged rows; production's correction then runs on the grouped rows under the owner's merge principles
(relaxed_rules.py), adding the longer joins and never splitting.

    python tools/diarization_explore/two_pass.py prompt <key>...   # render the shape prompt (no call)
    python tools/diarization_explore/two_pass.py shape <key>...    # call the shape pass
    python tools/diarization_explore/two_pass.py run <key>...      # group, correct, map the sources back
    python tools/diarization_explore/two_pass.py report <key>...   # joins, drops, pass-2 merges, tokens

Input: `merge_trial.py judged sol`'s stables (Sol's verdicts applied). Output: tmp/spk-merge/two-pass[-<TRIAL_LABEL>]/<key>/;
the `<key>-annotated.csv` there names the original rows, so `merge_audit.py compare` reads it next to the one-pass runs.
Same model and options as `merge_trial.py run`.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import merge_trial as mt  # noqa: E402

sys.path.insert(0, str(mt.CHECKOUT / "src"))
from finesub.llm.prompt_constants import THRESHOLDS  # noqa: E402

LABEL = os.environ.get("TRIAL_LABEL", "")
ROOT = mt.OUT / (f"two-pass-{LABEL}" if LABEL else "two-pass")
#: put in front of the merge rules (relaxed_rules.py, the owner's principles): pass 1 joined only short, clearly unfinished
#: pieces, so the longer joins of one sentence are pass 2's, and it never splits (owner 2026-10-16)
PREGROUPED = ("输入的行已经由上一步初步分好：上一步只并了同一个人明显没说完、并完很短的碎片，多数行就是一条字幕，逐行纠错、翻译。"
              "同一个人连着说的同一句话被切成了两行、合起来仍在下面规定的长度里的，照下面的合并规则在这里合并；不拆开任何一行。"
              "下面「合并行检查」里缩窄、拆回的要求只针对你这一步新并的行：输入里本来就是一行的，照原样当作一个源。\n")
#: a may_join run: the input lines' `turn` column reads MAY_CELL where pass 1 called the seam the same person
#: carrying on, and the turn fragments say merges go only there
MAY_CELL = "可并"
PREGROUPED_MAY = ("输入的行已经由上一步初步分好：上一步只并了同一个人明显没说完、并完很短的碎片，多数行就是一条字幕，逐行纠错、翻译。"
                  "上一步还认过人：`turn` 列标着「换人」的行与上一行是不同的人；标着「可并」的行，是同一个人接着上一行往下说、只是没在那一步并的；"
                  "这些行要不要并到上一行，由你按下面的合并规则、译文长度与用户要的翻译风格决定。没标的行不与上一行合并；不拆开任何一行。"
                  "下面「合并行检查」里缩窄、拆回的要求只针对你这一步新并的行：输入里本来就是一行的，照原样当作一个源。\n")
MAY_TURNS = {
    "fragment_turn_cell_v1.md": "换人、可并或空|",
    "fragment_turn_column_v1.md": "4. `turn` 列：上一步判断本行与上一行不是同一个人时标着「换人」；判断本行是同一个人接着上一行往下说、可以并到上一行时标着「可并」；为空的行（不是同一句话接着说，或判不准）也不与上一行合并。",
    "fragment_turn_rule_v1.md": "只有 `turn` 列标着「可并」的行可以并到上一行；合并行里除第一行外，每一行都要标着「可并」（三源也一样）。",
    "fragment_turn_doubt_v1.md": "标着「可并」的行已经判过是同一个人接着说，不用再判是不是同一个人；并不并看停顿、长度与是不是同一句话。",
    "fragment_turn_check_v1.md": "合并行里除第一行外有任何一行没标「可并」，void 后拆开。",
    "fragment_turn_reminder_v1.md": "只有 `turn` 列标「可并」的行可以与上一行合并。",
}


#: D5: TRIAL_SESSION=continue runs pass 2 in pass 1's agy conversation (its first message opens with CONTINUE_BRIDGE),
#: fresh in a new one on the same path; unset, on production's default transport
SESSION = os.environ.get("TRIAL_SESSION", "")
CONTINUE_BRIDGE = ("（接着上一轮：上一轮你判了这一段字幕的说话人与形状，判的结果已经体现在下面输入的分行和 `turn` 列里。"
                   "这一轮是下一步：按下面的要求纠错、翻译。上一轮的输出格式不再适用。）")


def fork(args):
    """`fork <label> <key>...`: copy another run's pass 1 (input/, judged/, shape.json) into this TRIAL_LABEL's
    folder, so a second pass 2 reads exactly the same pass 1."""
    source, keys = mt.OUT / f"two-pass-{args[0]}", args[1:]
    for key in keys:
        target = ROOT / key
        if (target / "shape.json").exists():
            print(f"{key}: forked before")
            continue
        target.mkdir(parents=True, exist_ok=True)
        for name in ("input", "judged"):
            shutil.copytree(source / key / name, target / name)
        shutil.copyfile(source / key / "shape.json", target / "shape.json")
        print(f"{key}: pass 1 from {source.name}")


def segments(key):
    """The rows the shape was decided on: `judge_shape.py`'s judged copy when it wrote one, else Sol's (merge_trial)."""
    own = ROOT / key / "judged" / f"{key}-stable.json"
    return json.loads((own if own.exists() else mt.stable_for("judged-sol", key)).read_text(encoding="utf-8"))


def track(segment):
    return (segment.get("speaker") or {}).get("track")


def shape_prompt(key):
    payload = segments(key)
    segs = payload["segments"]
    ident = key.rsplit("-c", 1)[0]
    context = json.loads((mt.MAIN / "tmp/diarization-retrack/accept" / ident / "llm" / f"{ident}-research-context.json")
                         .read_text(encoding="utf-8"))["context_pack"]["general_context"]
    rows = []
    for n, s in enumerate(segs):
        gap = f"{segs[n + 1]['start'] - s['end']:.1f}" if n + 1 < len(segs) else ""
        spk = "??" if track(s) is None else f"s{track(s)}"
        rows.append(f"q{n + 1}|{s['start']:.1f}|{s['end'] - s['start']:.1f}|{gap}|{spk}|{s['text'].strip()}")
    system = Template((HERE.parent / "two_pass_shape_system.md").read_text(encoding="utf-8")).substitute(THRESHOLDS)
    user = Template((HERE.parent / "two_pass_shape_user.md").read_text(encoding="utf-8")).substitute(
        title=ident, summary=context.get("global_summary", ""), rows="\n".join(rows), count=len(rows))
    return system, user, len(segs)


def parse(text, count):
    """{row (1-based): "join" | "drop"}; refuses a reply that is not one."""
    match = re.search(r"<shape>(.*?)</shape>", text or "", re.S)
    if match is None:
        raise ValueError("no closed <shape> block")
    out = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        row, act = int(str(item["id"]).lstrip("q")), item["act"]
        if not 1 <= row <= count or row in out or act not in ("join", "drop"):
            raise ValueError(f"bad line {line!r}")
        out[row] = act
    return out


def shape(keys):
    from finesub.llm.client import RoleClient, extract_token_distribution
    from finesub.llm.routing.config import LLMRole
    from finesub.llm.routing.model_routes import install_runtime_preferred, parse_llm_model_args

    install_runtime_preferred(parse_llm_model_args([mt.MODEL]))
    client = RoleClient()
    for key in keys:
        target = ROOT / key / "shape.json"
        if target.exists():
            print(f"{key}: shaped before")
            continue
        system, user, count = shape_prompt(key)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        calls = []
        for _ in range(2):
            result = client.complete(LLMRole.AUDIO_MULTIMODAL, messages, task_group="correction-text")
            calls.append({"model": result.model, "text": result.content,
                          "usage": extract_token_distribution(result.raw_response)})
            try:
                acts = parse(result.content, count)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                calls[-1]["error"] = str(error)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"acts": acts, "calls": calls}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{key}: {Counter(acts.values())}", flush=True)
            break
        else:
            print(f"{key}: no answer: {calls[-1].get('error')}")


def may_join_run(folder):
    """Whether pass 1 of this run answered `may_join` (pass 2 then merges only where it did)."""
    return json.loads((Path(folder) / "shape.json").read_text(encoding="utf-8")).get("may_join", False)


def load_shape(folder):
    """(acts by 1-based row, the seconds a join may span) from a run's shape.json. The judge-led pass 1 records its
    prompt's ceiling (`join_seconds`: longer same-speaker lines are pass 2's call), since the prompt alone does not hold
    it; the earlier shape pass recorded none and keeps the absolute seconds."""
    shaped = json.loads((Path(folder) / "shape.json").read_text(encoding="utf-8"))
    return ({int(k): v for k, v in shaped["acts"].items()},
            shaped.get("join_seconds", THRESHOLDS["absolute_seconds"]))


def shaped(folder, payload):
    """(rows, acts by 1-based row, each row's original 1-based row or None) for grouping: the judged rows with the
    audio pass 1's edits applied when it made some (shape.json `edits`): a split row becomes its two halves (the first
    keeps the row's act and track, the second gets `track2` and stands alone), a filled row takes the fuller text and
    times (its words stay the ASR's), an insert becomes a line of its own right after the row it was asked after, acted
    "insert" (a deferred one, with no room without overlapping, is left out). Without edits: the judged rows as they are."""
    from finesub.subtitles.stable_rows import halves

    acts, _ = load_shape(folder)
    edits = json.loads((Path(folder) / "shape.json").read_text(encoding="utf-8")).get("edits")
    segs = payload["segments"]
    if not edits:
        return segs, acts, list(range(1, len(segs) + 1))
    speaker = lambda n, block: {"labels": [], "block": block, "track": n, "tracks": [n], "coverage": {str(n): 1.0}}
    rows, origin, out_acts = [], [], {}
    for n, seg in enumerate(segs, start=1):
        split = edits["splits"].get(str(n))
        block = (seg.get("speaker") or {}).get("block", 0)
        fill = (edits.get("fills") or {}).get(str(n))
        if fill is not None:
            seg = {**seg, "text": fill["text"], "start": fill["start"], "end": fill["end"]}
        pieces = [seg] if split is None else list(halves(seg, split["at"]))
        if split is not None:
            pieces[0]["speaker"], pieces[1]["speaker"] = seg.get("speaker"), speaker(split["track2"], block)
        for k, piece in enumerate(pieces):
            rows.append(piece)
            origin.append(n)
            if k == 0 and n in acts:
                out_acts[len(rows)] = acts[n]
        for insert in (i for i in edits["inserts"] if i["row"] == n and not i.get("deferred")):
            rows.append({"start": insert["start"], "end": insert["end"], "text": insert["text"], "words": [],
                         "lang": "ja", "confidence": 1.0, "speaker": speaker(insert["track"], block)})
            origin.append(None)
            out_acts[len(rows)] = "insert"
    return rows, out_acts, origin


def group(segs, acts, seconds_cap=THRESHOLDS["absolute_seconds"]):
    """([[0-based rows]] per kept line, [(row, why)] joins refused): a join goes to the previous kept line, unless the
    two lines' speakers differ, the joined line runs past `seconds_cap` (the char thresholds count the Chinese
    translation, which this pass does not have), or it would break production's source cap."""
    groups, refused = [], []
    for i, seg in enumerate(segs):
        act = acts.get(i + 1)
        if act == "drop":
            continue
        if act == "join" and groups and acts.get(groups[-1][0] + 1) == "insert":
            refused.append((i + 1, "insert"))  # an inserted line stands alone, and nothing joins across it
        elif act == "join" and groups:
            prev = groups[-1]
            tracks = {track(segs[k]) for k in [*prev, i]} - {None}
            seconds = seg["end"] - segs[prev[0]]["start"]
            # production's source cap: two rows, or three when the middle one is a lone filler (by length here)
            rows = len(prev) + 1
            sandwich = rows == 3 and len(segs[prev[1]]["text"].strip()) <= THRESHOLDS["filler_max_chars"]
            why = ("speaker" if len(tracks) > 1 else "seconds" if seconds > seconds_cap
                   else "rows" if rows > 3 or (rows == 3 and not sandwich) else None)
            if why is None:
                prev.append(i)
                continue
            refused.append((i + 1, why))
        groups.append([i])
    return groups, refused


def mergeable(segs, acts, groups, refused):
    """Indexes of the grouped lines pass 2 may merge into the line before: pass 1 called the seam the same person
    carrying on (`may_join`, or a join the script refused for length or the row cap), the two lines' tracks agree,
    and together they stay within the absolute seconds."""
    long_join = {row for row, why in refused if why in ("seconds", "rows")}
    out = set()
    for g in range(1, len(groups)):
        row = groups[g][0] + 1
        tracks = {track(segs[k]) for k in [*groups[g - 1], *groups[g]]} - {None}
        # two grouped lines past the absolute seconds would be a merge no prompt has held off
        span = segs[groups[g][-1]]["end"] - segs[groups[g - 1][0]]["start"]
        inserted = "insert" in (acts.get(groups[g - 1][0] + 1), acts.get(row))
        if (acts.get(row) == "may" or row in long_join) and len(tracks) <= 1 and span <= THRESHOLDS["absolute_seconds"] \
                and not inserted:
            out.add(g)
    return out


def grouped_stable(payload, groups, may=None):
    """The grouped lines as a stable JSON. With `may` (a may_join run), each line's speaker is a stand-in that changes
    at every seam pass 2 must not merge across, so the correction's `turn` column marks exactly those (the trial's
    prompt shows it as 「可并」 on the other seams)."""
    out = []
    stand_in = 0
    for g, rows in enumerate(groups):
        first, last = payload["segments"][rows[0]], payload["segments"][rows[-1]]
        seg = {k: v for k, v in first.items() if k not in ("speaker", "words", "alignment_events")}
        # a group never spans two tracks: its tracked row's speaker lets pass 2 see the turns between lines
        tracked = next((payload["segments"][k] for k in rows if track(payload["segments"][k]) is not None), None)
        if may is not None:
            stand_in += g not in may
            seg["speaker"] = {"labels": [], "block": 0, "track": stand_in, "tracks": [stand_in],
                              "coverage": {str(stand_in): 1.0}}
        elif tracked is not None:
            seg["speaker"] = tracked["speaker"]
        seg.update(end=last["end"], text="".join(payload["segments"][k]["text"].strip() for k in rows),
                   confidence=min(payload["segments"][k].get("confidence", 1.0) for k in rows))
        out.append(seg)
    return {**payload, "segments": out}


def run(keys):
    env = {**os.environ, "PYTHONPATH": str(mt.CHECKOUT / "src")}
    for key in keys:
        folder = ROOT / key
        if (folder / f"{key}-annotated.csv").exists():
            print(f"{key}: done before")
            continue
        rows, acts, origin = shaped(folder, segments(key))
        payload = {**segments(key), "segments": rows}
        groups, refused = group(rows, acts, load_shape(folder)[1])
        may = mergeable(rows, acts, groups, refused) if may_join_run(folder) else None
        stable = folder / "grouped" / f"{key}-stable.json"
        stable.parent.mkdir(parents=True, exist_ok=True)
        stable.write_text(json.dumps(grouped_stable(payload, groups, may), ensure_ascii=False), encoding="utf-8")
        turn_starts = folder / "grouped" / "turn-starts.json"
        if may is not None:
            from finesub.llm.chunking import Seam, load_segments_from_stable_json
            loaded = load_segments_from_stable_json(stable)
            # the stand-in speakers only work through production's loader: check it reads them as meant
            assert {g for g, s in enumerate(loaded) if s.seam is Seam.NONE and g > 0} == may, f"{key}: stand-ins misread"
            segs = rows
            # a line's track is its tracked row's, as grouped_stable takes it
            line_track = [next((track(segs[k]) for k in g if track(segs[k]) is not None), None) for g in groups]
            turn_starts.write_text(json.dumps([round(segs[g[0]]["start"], 3) for n, g in enumerate(groups)
                                               if n > 0 and None not in (line_track[n], line_track[n - 1])
                                               and line_track[n] != line_track[n - 1]]), encoding="utf-8")
        ident = key.rsplit("-c", 1)[0]
        context = mt.MAIN / "tmp/diarization-retrack/accept" / ident / "llm" / f"{ident}-research-context.json"
        pass2 = folder / "pass2"
        media = (["--media", "audio", "--audio", str(mt.MAIN / "tmp/diarization-retrack/accept" / ident / f"{ident}-vocal.ogg")]
                 if os.environ.get("TRIAL_MEDIA") == "audio" else ["--media", "text"])
        args = [str(stable), "--llm-model", mt.MODEL, "--context-file", str(context), "--retrieval", "none",
                *media, "--fast", "off", "-o", str(pass2 / f"{key}.srt"), "--task-artifact-dir",
                str(pass2 / "llm"), *(["--prompt-dir", str(pass2 / "prompts")] if os.environ.get("DRY") else ["--execute"])]
        pass2.mkdir(parents=True, exist_ok=True)
        session = {}
        if SESSION:  # D5: pass 2 on agy's continued-conversation path, continuing pass 1's conversation or a new one
            import session_patch
            seed = session_patch.last_handle(folder / "input" / "sessions.jsonl") if SESSION == "continue" else ""
            assert seed or SESSION == "fresh", f"{key}: pass 1 left no conversation to continue"
            session = {"TRIAL_SESSION_LOG": str(pass2 / "sessions.jsonl"), "TRIAL_SESSION_SEED": seed}
        with open(pass2 / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, str(HERE), "correct", *args], stdout=log, stderr=subprocess.STDOUT,
                                   cwd=str(mt.CHECKOUT), env={**env, **session,
                                                              "TRIAL_MAY_JOIN": str(turn_starts) if may is not None else ""})
        annotated = pass2 / f"{key}-annotated.csv"
        if code == 0 and annotated.exists():
            remap(annotated, [[origin[k] - 1 for k in g if origin[k] is not None] for g in groups],
                  folder / f"{key}-annotated.csv")
        print(f"{key}: exit {code}", flush=True)


def remap(annotated, groups, target):
    """The pass-2 annotated CSV with each source position (a grouped line) spelled as its original rows (0-based in
    `groups`; the two halves of a split row are the same row once, an inserted line none: a line made only of
    inserted ones reads `ins`)."""
    lines = []
    for line in annotated.read_text(encoding="utf-8").splitlines():
        cells = line.split("|")
        if not line.startswith("#") and len(cells) > 1 and re.fullmatch(r"[\d,]+", cells[1]):
            rows = [k + 1 for g in cells[1].split(",") for k in groups[int(g) - 1]]
            cells[1] = ",".join(str(r) for n, r in enumerate(rows) if n == 0 or r != rows[n - 1]) or "ins"
        lines.append("|".join(cells))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def correct(args):
    """Production's correction with the relaxed merge rules and PREGROUPED in front of them (in this process only)."""
    import relaxed_rules
    from finesub.llm import correction_translation

    if os.environ.get("TRIAL_SESSION_LOG"):
        import session_patch
        session_patch.install(Path(os.environ["TRIAL_SESSION_LOG"]), seed=os.environ.get("TRIAL_SESSION_SEED", ""),
                              bridge=CONTINUE_BRIDGE if os.environ.get("TRIAL_SESSION_SEED") else "")
    if os.environ.get("TRIAL_MAY_JOIN"):
        relaxed_rules.install(prefix=PREGROUPED_MAY, turns=MAY_TURNS, turn_cell=MAY_CELL,
                              rules=HERE.parent / "may_merge_rules.md",
                              turn_starts=json.loads(Path(os.environ["TRIAL_MAY_JOIN"]).read_text(encoding="utf-8")))
    else:
        relaxed_rules.install(prefix=PREGROUPED)
    sys.argv = ["correction_translation", *args]
    sys.exit(correction_translation.main())

def report(keys):
    import row_split as R

    truth = {(it["ident"], it["i"]): it["truth"] for it in R.load("1")[0]}
    for key in keys:
        ident, folder = key.rsplit("-c", 1)[0], ROOT / key
        payload = segments(key)
        segs = payload["segments"]
        acts, cap = load_shape(folder)
        groups, refused = group(segs, acts, cap)
        acoustic = json.loads((mt.MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json")
                              .read_text(encoding="utf-8"))["segments"]
        index = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(acoustic)}
        kinds = Counter(((truth.get((ident, index.get((round(segs[r - 1]["start"], 3), round(segs[r - 1]["end"], 3)))))
                          or {}).get("kind", "none")) for r, a in acts.items() if a == "drop")
        pass2 = [line.split("|") for line in (folder / "pass2" / f"{key}-annotated.csv").read_text(encoding="utf-8").splitlines()
                 if line and not line.startswith("#")] if (folder / "pass2" / f"{key}-annotated.csv").exists() else []
        shape_tokens = sum(c["usage"].get("total_tokens", 0) for c in json.loads((folder / "shape.json").read_text(encoding="utf-8"))["calls"])
        artifacts = folder / "pass2" / "llm" / "task-artifacts.jsonl"
        pass2_tokens = sum(json.loads(line)["payload"].get("usage", {}).get("total_tokens", 0)
                           for line in artifacts.read_text(encoding="utf-8").splitlines()
                           if json.loads(line).get("kind") == "correction_window_response") if artifacts.exists() else 0
        print(f"{key}: rows {len(segs)} -> lines {len(groups)}; joins {sum(a == 'join' for a in acts.values())} "
              f"(refused {Counter(w for _, w in refused)}); drops {dict(kinds)}; "
              f"pass 2 merged {sum(1 for c in pass2 if c[0] == 'sub' and ',' in c[1])}, "
              f"other rows {Counter(c[0] for c in pass2 if c[0] != 'sub')}; tokens shape {shape_tokens} + text {pass2_tokens}")


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "prompt":
        for key in rest:
            system, user, _ = shape_prompt(key)
            (ROOT / key).mkdir(parents=True, exist_ok=True)
            (ROOT / key / "shape-prompt.md").write_text(f"{system}\n\n=====\n\n{user}", encoding="utf-8")
            print(ROOT / key / "shape-prompt.md")
    else:
        {"shape": shape, "run": run, "correct": correct, "report": report, "fork": fork}[mode](rest)
