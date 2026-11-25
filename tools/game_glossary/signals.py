"""Step 1 of the README: how often the game's own text mentions each name (``build.py --signals``; maintainers only).

    python -I tools/game_glossary/signals.py <terms.jsonl> --ja <TextMap JA.json> [...] --en <TextMap EN.json> [...] \
        --out <signals.json>

The text tables are the game's strings (dialogue, quest text, item descriptions), read locally and never published.
For each term, ``mentions`` counts the strings' occurrences of its Japanese name where it stands as a word of its own
script (a katakana name not inside a longer katakana run, a kanji name not inside a longer kanji run) and, when the
term has an English name, the occurrences of that name as a whole word; the smaller of the two is kept (one language
alone overcounts a short name: リン inside リンゴ, "Rin" is rarer). A string that is just the name (its own table entry,
a speaker label) is not counted. Output: ``{"<term id>|<ja>": {"mentions": n}}``: Genshin's several Japanese
forms of one word (淑女 / シニョーラ) share an id, so the key names the form.

``build.py`` turns it into a ``mentions:<n>`` tag and reads an important NPC from it (with the dialogue line counts
``extract.py`` writes for the games whose tables give speakers).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_KATA = "\u30a1-\u30fa\u30fc"
_KANJI = "\u3400-\u9fff々"


def _corpus(files: list[Path]) -> str:
    strings: dict[str, str] = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, str):
                strings[key] = value
    return "\n" + "\n".join(s.replace("\n", " ") for s in strings.values()) + "\n"


def _class(ch: str) -> str:
    if re.match(f"[{_KATA}]", ch):
        return _KATA
    if re.match(f"[{_KANJI}]", ch):
        return _KANJI
    return ""


def _count(text: str, name: str, before: str, after: str) -> int:
    """Occurrences of ``name`` not preceded by a character of class ``before`` nor followed by one of ``after``
    (a literal search first: a pattern that opens with a lookbehind scans the whole text slowly)."""

    if not name or text.count(name) == 0:
        return 0
    behind = re.compile(f"[{before}]") if before else None
    ahead = re.compile(f"[{after}]") if after else None
    n, at, size = 0, text.find(name), len(name)
    while at != -1:
        prev, nxt = text[at - 1:at], text[at + size:at + size + 1]
        if not (behind and prev and behind.match(prev)) and not (ahead and nxt and ahead.match(nxt)):
            n += 1
        at = text.find(name, at + 1)
    return max(0, n - _whole(text, name))


def _whole(text: str, name: str) -> int:
    """The strings that are just ``name``: overlapping finds, since two such strings in a row share the newline
    between them (``str.count`` would count every second one)."""

    whole, n = f"\n{name}\n", 0
    at = text.find(whole)
    while at != -1:
        n += 1
        at = text.find(whole, at + 1)
    return n


def count_ja(text: str, name: str) -> int:
    return _count(text, name, _class(name[0]), _class(name[-1])) if name else 0


def count_en(text: str, name: str) -> int:
    if not name or not re.search(r"[A-Za-z]", name):
        return 0
    return _count(text, name, "A-Za-z", "A-Za-z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("terms", type=Path)
    ap.add_argument("--ja", type=Path, action="append", required=True)
    ap.add_argument("--en", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ja_text, en_text = _corpus(args.ja), _corpus(args.en) if args.en else ""
    out = {}
    for line in args.terms.read_text(encoding="utf-8").splitlines():
        term = json.loads(line)
        ja = count_ja(ja_text, term["ja"])
        en = count_en(en_text, term.get("en") or "") if en_text and term.get("en") else None
        out[f"{term['id']}|{term['ja']}"] = {"mentions": ja if en is None else min(ja, en)}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"{len(out)} terms -> {args.out}")


if __name__ == "__main__":
    main()
