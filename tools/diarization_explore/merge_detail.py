"""Seam-level detail for the speaker-aware merge trials (docs/reports/2026-10-17-three-pass.md): first answer vs final, cross / same merges
split by what the acoustic marks (A) and the judged marks (J) said about the seam: the arm's own judge when it ran one
(judge_shape's judged/ stable, or the speaker-judge sidecar in input/), else Sol's.
`merge_detail.py marks [arm]` (how well A and J mark the clean seams) | `merge_detail.py <arm or run folder>...`."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agy_submissions as agy_subs  # noqa: E402

MAIN = Path("C:/Users/Carl/Documents/Carl/projects/asr-playground")
sys.path.insert(0, str(MAIN / "tools/diarization_explore"))
import row_split as R  # noqa: E402

TMP = MAIN / "tmp"
KEYS = ["hina-17-c0002", "hina-24-c0002", "hina-31-c0005"]
truth = {(it["ident"], it["i"]): it["truth"] for it in R.load("1")[0]}


def ends(t):
    if t is None:
        return None
    if t["kind"] == "single":
        p = t["people"]
        return (p[0], p[-1]) if len(p) == 1 else None
    if t["kind"] == "split":
        return (t["people"][0], t["people"][-1])
    return None


def marks(payload):
    segs = payload["segments"]
    sp = [s.get("speaker") or {} for s in segs]
    return segs, [False] + [a.get("track") is not None and b.get("track") is not None and a.get("block", 0) == b.get("block", 0)
                            and a["track"] != b["track"] for a, b in zip(sp, sp[1:])]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir(arm):
    return Path(arm) if "/" in arm else TMP / "spk-merge" / arm


def judged(arm, key):
    """The judged rows the arm's correction saw: its own judge's when it ran one, else Sol's."""
    run = run_dir(arm) / key if arm else None
    if run and (run / "judged" / f"{key}-stable.json").exists():
        return load(run / "judged" / f"{key}-stable.json")
    if run and (run / "input" / f"{key}-speaker-judge.json").exists():
        import judge_shape
        return judge_shape.judged_payload(run / "input" / f"{key}-stable.json")
    return load(TMP / "spk-merge/stables-judged-sol" / key / f"{key}-stable.json")


def seams(key, arm=None):
    """[(n, cross?, acoustic mark, judged mark)] for seams n-1|n (1-based n) with clean truth."""
    ident = key.rsplit("-c", 1)[0]
    segs, am = marks(load(TMP / "row-split/d6/orig" / key / f"{key}-stable.json"))
    _, jm = marks(judged(arm, key))
    acoustic = json.loads((MAIN / "data/diarization/row-truth/products/acoustic" / ident / f"{ident}-stable.json").read_text(encoding="utf-8"))["segments"]
    index = {(round(s["start"], 3), round(s["end"], 3)): k for k, s in enumerate(acoustic)}
    src = [ends(truth.get((ident, index.get((round(s["start"], 3), round(s["end"], 3)))))) for s in segs]
    return [(n, src[n - 1][1] != src[n][0], am[n], jm[n]) for n in range(1, len(segs)) if src[n - 1] and src[n]]


def merged_pairs(text, groups=None):
    """{n: True} for every seam n-1|n (0-based positions n-1, n) an answer's sub rows join."""
    out = set()
    for line in text.splitlines():
        cells = line.split("|")
        if len(cells) > 2 and cells[0].strip() == "sub" and re.fullmatch(r"[\d,]+", cells[1].strip()) and "<void>" not in line:
            pos = [int(p) for p in cells[1].split(",")]
            if groups:
                pos = [k + 1 for g in pos for k in groups[g - 1]]
            out |= {b - 1 for a, b in zip(pos, pos[1:]) if b == a + 1}
    return out


def translated(text):
    m = re.search(r"<translated>(.*?)(</translated>|$)", text, re.S)
    return m.group(1) if m else text


def answers(arm, key):
    """(first answer's merged seams, final merged seams), 0-based seam index n = later row's 0-based position."""
    run = run_dir(arm) / key
    groups = None
    if arm.startswith("two-pass"):
        sys.path.insert(0, str(MAIN / ".worktrees/spk-merge/tools/diarization_explore"))
        import os
        os.environ["TRIAL_LABEL"] = arm.removeprefix("two-pass-")
        import importlib
        import two_pass
        importlib.reload(two_pass)
        rows, acts, origin = two_pass.shaped(run, two_pass.segments(key))
        shaped_groups, _ = two_pass.group(rows, acts, two_pass.load_shape(run)[1])
        # in original rows: a split row's halves are the row, an inserted line none
        groups = [[origin[k] - 1 for k in g if origin[k] is not None] for g in shaped_groups]
    final_text = next(run.glob("*-annotated.csv")).read_text(encoding="utf-8")
    final = merged_pairs(final_text)  # already in original rows for two-pass
    llm = run / ("pass2" if groups else "") / "llm"
    if "agy" in arm:
        subs = agy_subs.submissions(agy_subs.session(run / ("pass2" if groups else "")))
        first = merged_pairs(translated(subs[0][0]), groups) if subs else final
        rounds = len(subs)
    else:
        exchanges = sorted((llm / "exchanges").glob("*.md"))
        first_text = exchanges[0].read_text(encoding="utf-8").split("## 模型响应", 1)[1]
        first = merged_pairs(translated(first_text), groups)
        rounds = len(exchanges)
    return first, final, rounds


def tally(merged, key, arm):
    c = Counter()
    for n, cross, am, jm in seams(key, arm):
        if n in merged:
            c["cross" if cross else "same", "A" if am else "a", "J" if jm else "j"] += 1
    return c


LABELS = [("cross", "A", "j", "跨人·声学标、裁判没标"), ("cross", "a", "j", "跨人·都没标"),
          ("cross", "A", "J", "跨人·都标了"), ("cross", "a", "J", "跨人·只有裁判标"),
          ("same", "A", "j", "同人·声学标、裁判没标"), ("same", "a", "j", "同人·都没标"),
          ("same", "A", "J", "同人·都标了"), ("same", "a", "J", "同人·只有裁判标")]

if __name__ == "__main__":
    if sys.argv[1] == "marks":
        arm = sys.argv[2] if len(sys.argv) > 2 else None
        tot = Counter()
        for key in KEYS:
            if not (arm is None or (run_dir(arm) / key / "input").exists() or (run_dir(arm) / key / "judged").exists()):
                continue
            c = Counter()
            for _, cr, am, jm in seams(key, arm):
                kind = "cross" if cr else "same"
                c[kind] += 1
                c[kind, "A"] += am
                c[kind, "J"] += jm
            tot.update(c)
            print(f"{key}: cross {c['cross']} (A marks {c['cross', 'A']}, J marks {c['cross', 'J']})"
                  f"  same {c['same']} (A marks {c['same', 'A']}, J marks {c['same', 'J']})")
        print(f"total ({arm or 'Sol'}): cross {tot['cross']} (A {tot['cross', 'A']}, J {tot['cross', 'J']})"
              f"  same {tot['same']} (A {tot['same', 'A']}, J {tot['same', 'J']})")
        sys.exit()
    for arm in sys.argv[1:]:
        tot = {"first": Counter(), "final": Counter()}
        rounds = []
        for key in KEYS:
            first, final, r = answers(arm, key)
            rounds.append(r)
            tot["first"].update(tally(first, key, arm))
            tot["final"].update(tally(final, key, arm))
        print(f"== {arm}  rounds per window {rounds}")
        for t, a, j, name in LABELS:
            f, g = tot["first"][t, a, j], tot["final"][t, a, j]
            if f or g:
                print(f"   {name:22} first {f:3}  final {g:3}")
