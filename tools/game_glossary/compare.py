"""Compare two builds of one game's glossary: which names moved tier, and which tags appeared or went (maintainers).

    python -I tools/game_glossary/compare.py <earlier glossary.json> <new glossary.json> [--show N]

Run it after every rebuild, before the reader's pass and the release: the same inputs give another file on another
machine or with other versions of the readers (``provenance.readers``), and a rule change moves names. Terms are
matched by ``(ja, zh)``; a tag is counted by its key (``risk:common-word``, ``noise``), so changed numbers show too.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

KEYED = ("risk", "evidence", "everyday", "important", "override", "skip")  # tags counted with their second part


def tier(t: dict) -> str:
    return "skip" if t.get("skip") else "rare" if t.get("rare") else "exact" if t.get("exact") else "common"


def mark(t: dict) -> str:
    return tier(t) + "+core" * bool(t.get("core")) + "+fold" * bool(t.get("fold"))


def key(tag: str) -> str:
    head, _, rest = tag.partition(":")
    return f"{head}:{rest.split(':', 1)[0]}" if head in KEYED and rest else head


def compare(old: dict, new: dict, show: int) -> list[str]:
    a = {(t["ja"], t["zh"]): t for t in old["terms"]}
    b = {(t["ja"], t["zh"]): t for t in new["terms"]}
    lines = [f"readers: {old.get('provenance', {}).get('readers') or '-'} -> {new.get('provenance', {}).get('readers') or '-'}",
             f"terms: {len(a)} -> {len(b)}; only earlier {len(a.keys() - b.keys())}, only new {len(b.keys() - a.keys())}"]
    lines += [f"  only earlier: {ja}={zh}" for ja, zh in sorted(a.keys() - b.keys())[:show]]
    lines += [f"  only new: {ja}={zh}" for ja, zh in sorted(b.keys() - a.keys())[:show]]
    added, gone, moves = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    for k in a.keys() & b.keys():
        ta, tb = set(a[k]["tags"]), set(b[k]["tags"])
        added.update(key(x) for x in tb - ta)
        gone.update(key(x) for x in ta - tb)
        if tier(a[k]) != tier(b[k]) or a[k].get("core") != b[k].get("core") or a[k].get("fold") != b[k].get("fold"):
            moves[f"{mark(a[k])}->{mark(b[k])}"].append((k, sorted(ta ^ tb)))
    lines.append(f"tags added: {dict(added.most_common())}")
    lines.append(f"tags gone: {dict(gone.most_common())}")
    for move, items in sorted(moves.items(), key=lambda x: -len(x[1])):
        lines.append(f"{move}: {len(items)}")
        lines += [f"  {ja}={zh}: {' '.join(diff)}" for (ja, zh), diff in items[:show]]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old", type=Path)
    ap.add_argument("new", type=Path)
    ap.add_argument("--show", type=int, default=10, help="examples per group")
    args = ap.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    print("\n".join(compare(load(args.old), load(args.new), args.show)))


if __name__ == "__main__":
    main()
