"""Which misheard occurrences a hinted arm's hint block covered (maintainers only).

    python tools/kb_phonetic_eval/hint_cover.py <root> <score.json> [--base A1] [--hinted B1]

<root> is ``run_ab.py``'s root and <score.json> what ``score.py --json`` wrote for both arms. Reads the
hinted arm's correction prompt per clip (clips must be one window each; more fails) for its hint rows
and window rows. An occurrence counts as covered when a hint for its term names a row starting within
3 s of it -- looser than the scorer's +-1 s, because a refined line and the raw row that carried the
name rarely start together.

Prints one line per misheard occurrence (hint=Y when covered, then whether each arm wrote the name),
then ``covered N; <base> already right on M of them``: M of the N covered occurrences needed no hint,
so N - M bounds what the hints could have added. Last, every hint with the raw row it was attached to.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROW = re.compile(r"^(\d+)\|([\d.]+)\|([\d.]+)\|[^|\n]*\|(.*)$", re.M)
BLOCK = re.compile(r"<kb_phonetic_hints>\nrows?\|entry\|term\n(.*?)</kb_phonetic_hints>", re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("score", type=Path)
    ap.add_argument("--base", default="A1")
    ap.add_argument("--hinted", default="B1")
    args = ap.parse_args()
    score = json.loads(args.score.read_text(encoding="utf-8"))["arms"]

    rows: dict[tuple[str, int], tuple[float, str]] = {}
    hints: dict[str, list[tuple[int, str]]] = {}
    for clip_dir in sorted(p for p in (args.root / args.hinted).iterdir() if p.is_dir()):
        clip = clip_dir.name
        if (clip_dir / "prompts" / "correction-0002.txt").exists():
            raise SystemExit(f"{clip}: more than one window; this tool reads only the first")
        text = (clip_dir / "prompts" / "correction-0001.txt").read_text(encoding="utf-8")
        if "<kb_phonetic_hints>" not in text:
            raise SystemExit(f"{clip}: no hint block in {args.hinted}; was it run with --hints?")
        # The block sits before the window's rows; the prompt's examples come earlier still.
        for m in ROW.finditer(text.split("</kb_phonetic_hints>")[-1]):
            rows[(clip, int(m.group(1)))] = (float(m.group(2)), m.group(4))
        block = BLOCK.search(text)
        for line in block.group(1).splitlines() if block else []:
            parts = line.split("|")
            for row in parts[0].split(","):
                hints.setdefault(clip, []).append((int(row), parts[3]))

    base = {(d["clip"], d["t"], d["term"]): d["hit"] for d in score[args.base]["details"] if d["class"] == "misheard"}
    covered = covered_base_hit = 0
    for d in score[args.hinted]["details"]:
        if d["class"] != "misheard":
            continue
        clip, t, term = d["clip"], d["t"], d["term"]
        hinted = any(zh == term and abs(rows[(clip, r)][0] - t) < 3 for r, zh in hints.get(clip, []))
        base_hit = base[(clip, t, term)]
        covered += hinted
        covered_base_hit += hinted and base_hit
        raw = [text for (c, _), (start, text) in rows.items() if c == clip and abs(start - t) < 1.5]
        print(f"{clip} {t:7.1f} {term:8s} hint={'Y' if hinted else '-'} "
              f"{args.base}={'Y' if base_hit else '-'} {args.hinted}={'Y' if d['hit'] else '-'} raw={raw[:2]}")
    print(f"covered {covered}; {args.base} already right on {covered_base_hit} of them")
    for clip, found in hints.items():
        for row, zh in found:
            print(f"hint {clip} {row} {zh} <- {rows[(clip, row)][1][:40]}")


if __name__ == "__main__":
    main()
