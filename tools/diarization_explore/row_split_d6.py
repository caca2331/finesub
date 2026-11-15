"""D6 of docs/diarization-findings.md 6.5: does correction treat the split halves right?

Ruler 1's 13 correction windows, each cut out of its file as a stable of its own (core rows only), in two versions:
`orig` as it is, `split` with every row the settled configuration (`row_split.phase1_cut(..., "final")`) cuts replaced
by its two halves. Correction runs on each with gemini flash-lite (`--test-profile`), the acceptance run's research
context reused so no arm prepares its own; `split` runs twice as the noise baseline.

    python tools/diarization_explore/row_split_d6.py build          # stables + cut manifest -> tmp/row-split/d6/
    python tools/diarization_explore/row_split_d6.py commands [--execute] [--target T]   # the command per window and arm
    python tools/diarization_explore/row_split_d6.py run [--parallel 3] [--target T]     # run the missing ones
    python tools/diarization_explore/row_split_d6.py read [--target T]                   # the three counts

--target is a route target for `--llm-model` (e.g. gemini-free-3_8-flash); without it, flash-lite via --test-profile.
`read` flags a run whose output drifts by a row (line n carries source n+1's text, every structural check passing)
and compares untouched rows only on the windows no arm drifted in.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import shlex
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import row_split as R  # noqa: E402
import row_split_lib as L  # noqa: E402

D6 = R.CACHE / "d6"
#: arm -> folder; split runs twice (split, split2) on the same input
ARMS = {"orig": "orig", "split": "split", "split2": "split"}


def _runs(target):
    """Run folder: flash-lite (`--test-profile`) in runs/, any other route target (`--llm-model`) in runs-<target>/."""
    return D6 / (f"runs-{target}" if target else "runs")


def _text(words):
    return "".join((" " if k and w.get("space_before") else "") + str(w.get("word", "")) for k, w in enumerate(words))


def halves(seg, cut):
    words = seg["words"]
    a, b = copy.deepcopy(seg), copy.deepcopy(seg)
    a["words"], b["words"] = copy.deepcopy(words[:cut]), copy.deepcopy(words[cut:])
    a["text"], b["text"] = _text(a["words"]), _text(b["words"]).lstrip()
    a["end"] = float(words[cut - 1]["end"])
    b["start"] = float(words[cut]["start"])
    return a, b


def build():
    items, _ = R.load("1")
    by_window = {}
    for it in items:
        by_window.setdefault(it["key"], []).append(it)
    manifest = []
    for key, rows in by_window.items():
        rows.sort(key=lambda it: it["i"])
        stable_path = L.MAIN / "data" / "diarization" / "row-truth" / "products" / "acoustic" / rows[0]["ident"] / f"{rows[0]['ident']}-stable.json"
        payload = json.loads(stable_path.read_text(encoding="utf-8"))
        orig, split = [], []
        for it in rows:
            seg = it["seg"]
            assert _text(seg["words"]).strip() == str(seg["text"]).strip(), (key, it["i"])
            orig.append(seg)
            cut = R.phase1_cut("1", it, "final")
            if not cut:
                split.append(seg)
                continue
            tier, detail = R.grade("1", it, cut, [])
            kind = {"single": "one", "split": "two"}.get(it["truth"]["kind"], "other")
            a, b = halves(seg, cut)
            manifest.append(dict(key=key, ident=it["ident"], row=it["i"], orig_line=len(orig), split_line=len(split) + 1,
                                 cut=cut, kind=kind, tier=tier, detail=detail, label=R.label(R.key_of("1", it, cut)),
                                 left=a["text"], right=b["text"]))
            split += [a, b]
        for folder, segs in (("orig", orig), ("split", split)):
            out = D6 / folder / key / f"{key}-stable.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({**payload, "segments": segs}, ensure_ascii=False), encoding="utf-8")
        print(f"{key}: {len(orig)} rows, {len(split) - len(orig)} split")
    (D6 / "manifest.jsonl").write_text("".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest), encoding="utf-8")
    print(Counter((m["kind"], m["tier"]) for m in manifest))


def _jobs(execute, target=None, windows=None):
    """(arm, key, run folder, argv) for every window with at least one cut: the others are the same input in both arms."""
    split_keys = {json.loads(x)["key"] for x in (D6 / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x}
    for arm, folder in ARMS.items():
        for key in sorted(split_keys & set(windows) if windows else split_keys):
            ident = key.rsplit("-c", 1)[0]
            stable = D6 / folder / key / f"{key}-stable.json"
            run = _runs(target) / arm / key
            context = R.ACCEPT / ident / "llm" / f"{ident}-research-context.json"
            cmd = [sys.executable, "-m", "finesub.llm.correction_translation", str(stable),
                   *(["--llm-model", target] if target else ["--test-profile"]),
                   "--context-file", str(context), "--retrieval", "none", "--media", "text", "--fast", "off",
                   "-o", str(run / f"{key}.srt"), "--task-artifact-dir", str(run / "llm"), "--prompt-dir", str(run / "prompts")]
            yield arm, key, run, cmd + (["--execute"] if execute else [])


def commands(execute, target):
    for *_, cmd in _jobs(execute, target):
        print(shlex.join(cmd))


def run(parallel, target, windows=None):
    """Execute every job whose annotated CSV is missing, `parallel` at a time; logs next to each run."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        arm, key, folder, cmd = job
        if (folder / f"{key}-annotated.csv").exists():
            return f"{arm} {key}: done before"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "run.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(L.WT))
        return f"{arm} {key}: exit {code}"

    with ThreadPoolExecutor(parallel) as pool:
        for line in pool.map(one, list(_jobs(True, target, windows))):
            print(line, flush=True)


def _positions(cell):
    """Source line numbers of one output line: `7`, `8,9`, `8-10`."""
    out = []
    for part in str(cell).split(","):
        lo, _, hi = part.strip().partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def _lines(path):
    """source line -> (output line index, corrected text); discarded sources are absent."""
    out = {}
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.reader((x for x in f if not x.startswith("#")), delimiter="|")]
    for k, r in enumerate(rows):
        if not r or r[0] != "sub":
            continue
        for n in _positions(r[1]):
            out[n] = (k, r[4])
    return out


#: a run drifted when more than this share of its lines read closer to the next source than to their own
DRIFT = 0.1


def _drift(arm, key, lines):
    import difflib
    src = json.loads((D6 / ARMS[arm] / key / f"{key}-stable.json").read_text(encoding="utf-8"))["segments"]
    nxt = sum(1 for n, (_, t) in lines.items() if n < len(src)
              and difflib.SequenceMatcher(None, t, src[n]["text"]).ratio() > difflib.SequenceMatcher(None, t, src[n - 1]["text"]).ratio())
    return nxt / max(1, len(lines))


def read(target):
    manifest = [json.loads(x) for x in (D6 / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x]
    cuts = {}
    for m in manifest:
        cuts.setdefault(m["key"], []).append(m)
    runs = _runs(target)
    got = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for key in cuts:
            path = runs / arm / key / f"{key}-annotated.csv"
            if path.exists():
                got[arm][key] = _lines(path)
    keys = [k for k in cuts if all(k in got[a] for a in ARMS)]
    print(f"{target or 'flash-lite'}: {len(keys)} / {len(cuts)} windows in every arm")
    drifted = {(arm, key) for arm in ARMS for key in keys if _drift(arm, key, got[arm][key]) > DRIFT}
    print("drifted runs:", sorted(drifted) or "none")
    merges = Counter()
    for arm in ARMS:
        for key in keys:
            lines = got[arm][key]
            groups = Counter(k for k, _ in lines.values())
            merges[arm] += sum(n - 1 for n in groups.values() if n > 1)
    print("merged source lines per arm (all rows):", dict(merges))
    # 1, 2: the halves of each cut, per split run
    for arm in ("split", "split2"):
        tally = Counter()
        for key in keys:
            lines = got[arm][key]
            for m in cuts[key]:
                a, b = lines.get(m["split_line"]), lines.get(m["split_line"] + 1)
                if a is None or b is None:
                    state = "discarded"
                else:
                    state = "merged" if a[0] == b[0] else "apart"
                group = f"one {m['label'] or m['tier']}" if m["kind"] == "one" else f"two {m['tier']}"
                tally[(group, state)] += 1
                if (m["kind"] == "one" and state == "apart") or (m["kind"] == "two" and state == "merged"):
                    print(f"  {arm} {key} row {m['row']} {group} {state}: {m['left']} | {m['right']}")
        print(f"{arm}: " + "; ".join(f"{g} {s}={n}" for (g, s), n in sorted(tally.items())))
    # 3: everything else -- rows no cut touched, compared by the corrected text of the line holding them
    def text_of(arm, key):
        """orig line -> corrected text of the output line holding it, for every row no cut touched."""
        cut_rows = sorted(m["orig_line"] for m in cuts[key])
        n = len(json.loads((D6 / "orig" / key / f"{key}-stable.json").read_text(encoding="utf-8"))["segments"])
        lines, out = got[arm][key], {}
        for row in range(1, n + 1):
            if row in cut_rows:
                continue
            src = row + sum(1 for c in cut_rows if c < row) if ARMS[arm] == "split" else row
            if src in lines:
                out[row] = lines[src][1]
        return out

    clean = [k for k in keys if not any((arm, k) in drifted for arm in ARMS)]
    for x, y in (("orig", "split"), ("orig", "split2"), ("split", "split2")):
        same = diff = 0
        for key in clean:
            a, b = text_of(x, key), text_of(y, key)
            for row in a.keys() & b.keys():
                if a[row] == b[row]:
                    same += 1
                else:
                    diff += 1
        print(f"untouched rows ({len(clean)} windows), corrected text differs {x} vs {y}: {diff} / {same + diff} ({diff / max(1, same + diff):.1%})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    c = sub.add_parser("commands")
    c.add_argument("--execute", action="store_true")
    c.add_argument("--target")
    r = sub.add_parser("run")
    r.add_argument("--parallel", type=int, default=3)
    r.add_argument("--target")
    r.add_argument("--windows", nargs="+", help="only these windows (e.g. a fallback model for the ones a quota stopped)")
    rd = sub.add_parser("read")
    rd.add_argument("--target")
    a = ap.parse_args()
    if a.cmd == "build":
        return build()
    if a.cmd == "commands":
        return commands(a.execute, a.target)
    if a.cmd == "run":
        return run(a.parallel, a.target, a.windows)
    return read(a.target)


if __name__ == "__main__":
    main()
