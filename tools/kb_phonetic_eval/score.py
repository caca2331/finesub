"""Score the phonetic-hints A/B arms against the human-refined 四月一日 subtitles (maintainers only).

    python tools/kb_phonetic_eval/score.py --kb <knowledge.sqlite> --index <clips.csv> \
        --arm A1=<dir> --arm B1=<dir> ... [--clips id,id] [--json <file>]

An arm dir is what ``run_ab.py arm`` wrote: one sub-directory per clip id with <id>-stable.json and
<id>-translated.srt. ``--index`` is the run root's clips.csv (``clip id | refined subtitles | stable json |
note``, paths relative to the main checkout); the refined file may be SRT or ASS (``{...}`` blocks, staff,
watermark and 注释 lines dropped). ``--kb`` must be the store the arms ran with: which spellings it knows
decides the classes below, so two scorings are comparable only on the same store.

A term's Chinese forms are its Chinese name and its aliases (each at least 2 characters): refined lines
often carry only a short name (玲王 for 御影玲王). An occurrence is a refined line containing one of a
term's forms (one count per term and line; reported under the Chinese name). It is "present" when the raw ASR within +-1 s already has the term's surface or an
alias, else "misheard". Each occurrence is also flagged:
  note      the term's name or surface is in the clip's note, which every arm receives as extra info
  residual  a stored misheard spelling of the term sits in the raw within +-1 s (the store still knows
            this very mishearing; ``strip_leak.py`` removes only what it can trace to the test clips)

  M1  misheard occurrences where the arm wrote one of the term's forms within +-1 s
  M2  present occurrences, same test
  M3  insertions: a term's form in the arm's output where the refined lines within +-1 s exist but
      none carries any of its forms. Where the refined file has no line at all (a stretch its editor skipped) there
      is nothing to judge against, so those are counted apart as "unjudged".
      Common-word names (火花, 妖精) count too, so read the per-term list, and read M1 with M3:
      writing names everywhere raises both.
  M4  from the exchanges of every correction call of the scored clips (request rounds included): input tokens uncached /
      cached (cache reads), output tokens, calls, and the hint block's size in characters

Arms are compared on the clips they all have; missing clips are printed, never silently dropped.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from finesub.llm.knowledge.node.matching import ExactIndex, scan_normalize  # noqa: E402
from finesub.llm.knowledge.node.store import KnowledgeStore  # noqa: E402
from run_ab import MAIN, read_index  # noqa: E402

TOLERANCE = 1.0


SKIP_STYLES = ("staff", "watermark", "水印", "注释")


def stamp(text: str) -> float:
    h, m, s = text.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def subs(path: Path) -> list[tuple[float, float, str]]:
    if path.suffix.lower() != ".ass":
        return srt(path)
    out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("Dialogue:"):
            continue
        cols = line.split(",", 9)
        if any(word in cols[3].lower() for word in SKIP_STYLES):
            continue
        text = re.sub(r"\{[^}]*\}", "", cols[9]).replace("\\N", " ").replace("\\n", " ").strip()
        if text:
            out.append((stamp(cols[1]), stamp(cols[2]), text))
    return out


def srt(path: Path) -> list[tuple[float, float, str]]:
    out = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            stamps = re.findall(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[1])
            s, e = (int(h) * 3600 + int(m) * 60 + int(x) + int(ms) / 1000 for h, m, x, ms in stamps)
            out.append((s, e, " ".join(lines[2:])))
    return out


def carries(text: str, forms: set[str]) -> bool:
    return any(form in text for form in forms)


def near(lines, start: float, end: float):
    return [text for s, e, text in lines if e > start - TOLERANCE and s < end + TOLERANCE]


def load_terms(kb: str):
    """Chinese name -> (spellings the raw may carry, stored misheard spellings, surfaces, Chinese forms)."""

    conn = sqlite3.connect(f"file:{kb}?mode=ro", uri=True)
    zh_of = {}
    surface_of = collections.defaultdict(set)
    for local_id, payload in conn.execute(
        "SELECT v.local_id, v.payload FROM node_versions v JOIN nodes n ON n.local_id = v.local_id"
        " WHERE n.kind = 'term' AND v.valid_to_rev IS NULL"
    ):
        data = json.loads(payload)
        name = data.get("zh") or ""
        if len(name) >= 2:
            zh_of[local_id] = name
            surface_of[name].add(str(data.get("surface") or ""))
    keys = collections.defaultdict(set)
    misheard = collections.defaultdict(set)
    forms = {name: {name} for name in zh_of.values()}
    with KnowledgeStore(kb) as store:
        for text, owners in ExactIndex.build(store, include_tentative=True)._keys.items():  # noqa: SLF001
            for key in owners:
                if key.node_id in zh_of:
                    (misheard if key.kind == "misheard" else keys)[zh_of[key.node_id]].add(text)
                    if key.kind == "alias" and len(key.raw) >= 2:
                        forms[zh_of[key.node_id]].add(key.raw)
    return sorted(forms), keys, misheard, surface_of, forms


def tokens(arm_dir: Path, clips: list[str]) -> dict[str, int]:
    total = collections.Counter()
    hint_chars = []
    paths = [p for clip in clips for p in glob.glob(str(arm_dir / clip / "**" / "exchanges" / "*correction*.md"), recursive=True)]
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        found = re.search(r"input_tokens: (\d+) / (\d+) / \d+", text)
        if not found:
            continue
        total["uncached"] += int(found.group(1))
        total["cached"] += int(found.group(2))
        total["calls"] += 1
        found = re.search(r"output_tokens_breakdown: \d+ / \d+ / (\d+)", text)
        if found:
            total["output"] += int(found.group(1))
        # anchored: the slimmed entries' label names the tag inline
        block = re.search(r"^<kb_phonetic_hints>\n(.*?)\n</kb_phonetic_hints>$", text, re.S | re.M)
        if block:
            hint_chars.append(len(block.group(1)))
    total["hint_blocks"] = len(hint_chars)
    total["hint_chars_max"] = max(hint_chars, default=0)
    return dict(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kb", required=True)
    ap.add_argument("--index", required=True, help="the run root's clips.csv")
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--clips", default="")
    ap.add_argument("--json")
    args = ap.parse_args()

    names, keys, misheard, surface_of, forms = load_terms(args.kb)
    index = read_index(Path(args.index))
    notes = {clip: scan_normalize(row["note"]) for clip, row in index.items()}
    wanted = [c for c in args.clips.split(",") if c] or sorted(index)
    arms = dict(spec.split("=", 1) for spec in args.arm)
    have = {name: {c for c in wanted if (Path(d) / c / f"{c}-translated.srt").exists()} for name, d in arms.items()}
    common = [c for c in wanted if all(c in got for got in have.values())]
    for name, got in have.items():
        if len(got) < len(wanted):
            print(f"{name}: missing {sorted(set(wanted) - got)}")
    if len(common) < len(wanted):
        print(f"scoring the {len(common)} clips every arm has")

    report = {"kb": str(Path(args.kb).resolve()), "clips": common, "arms": {}}
    for name, arm_dir in arms.items():
        arm = Path(arm_dir)
        counts = collections.Counter()
        details = []
        inserted = collections.Counter()
        for clip in common:
            output = srt(arm / clip / f"{clip}-translated.srt")
            refined = subs(MAIN / index[clip]["refined"])
            segments = json.loads((arm / clip / f"{clip}-stable.json").read_text(encoding="utf-8"))["segments"]
            for start, end, text in refined:
                raw = scan_normalize("".join(
                    g["text"] for g in segments if g["end"] > start - TOLERANCE and g["start"] < end + TOLERANCE
                ))
                for name_zh in names:
                    if not carries(text, forms[name_zh]):
                        continue
                    kind = "present" if any(k in raw for k in keys[name_zh]) else "misheard"
                    hit = any(carries(line, forms[name_zh]) for line in near(output, start, end))
                    note = name_zh in notes[clip] or any(
                        s and scan_normalize(s) in notes[clip] for s in surface_of[name_zh]
                    )
                    residual = kind == "misheard" and any(k in raw for k in misheard[name_zh])
                    counts[kind] += 1
                    counts[f"{kind}_hit"] += hit
                    if kind == "misheard":
                        for flag, on in (("note", note), ("residual", residual)):
                            if on:
                                counts[f"misheard_{flag}"] += 1
                                counts[f"misheard_{flag}_hit"] += hit
                    details.append({"clip": clip, "t": round(start, 1), "term": name_zh, "class": kind,
                                    "hit": hit, "note": note, "residual": residual})
            for start, end, text in output:
                around = near(refined, start, end)
                for name_zh in names:
                    if carries(text, forms[name_zh]) and not any(carries(line, forms[name_zh]) for line in around):
                        if not around:
                            counts["unjudged"] += 1
                            details.append({"clip": clip, "t": round(start, 1), "term": name_zh, "class": "unjudged"})
                            continue
                        inserted[name_zh] += 1
                        details.append({"clip": clip, "t": round(start, 1), "term": name_zh, "class": "insertion"})
        counts["insertions"] = sum(inserted.values())
        m1 = counts["misheard_hit"] / max(counts["misheard"], 1)
        m2 = counts["present_hit"] / max(counts["present"], 1)
        report["arms"][name] = {"counts": dict(counts), "M1": round(m1, 3), "M2": round(m2, 3),
                                "M3": counts["insertions"], "M3_terms": dict(inserted.most_common()),
                                "M4": tokens(arm, common), "details": details}
        print(f"{name:6s} M1 {counts['misheard_hit']}/{counts['misheard']} ({m1:.1%})"
              f" [note {counts['misheard_note_hit']}/{counts['misheard_note']},"
              f" residual {counts['misheard_residual_hit']}/{counts['misheard_residual']}]"
              f"  M2 {counts['present_hit']}/{counts['present']} ({m2:.1%})  M3 {counts['insertions']}"
              f" (+{counts['unjudged']} unjudged)"
              f" {dict(inserted.most_common(5))}  M4 {report['arms'][name]['M4']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
