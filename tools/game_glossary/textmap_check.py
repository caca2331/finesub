"""Check a built glossary's names against the game's own text tables (maintainers only; local).

    python -I tools/game_glossary/textmap_check.py <game> <glossary.json> <text dir> <out.tsv> [--all] [--readables DIR]

The values a listed name carries out (``ja``, ``zh``, ``en``) are copied by the correction model as they are and
checked by nothing after this, so each has to be a string the game writes. ``<text dir>`` holds the game's text
tables, keyed so that one key is one sentence in each language:

  genshin   Dimbreath/animegamedata2 ``TextMap{JP,CHS,EN}.json`` and ``TextMap_Medium{JP,CHS,EN}.json``;
            ``--readables`` adds its ``Readable/{JP,CHS,EN}`` books (one key per file)
  starrail  Dimbreath/turnbasedgamedata ``TextMap{JP,CHS,EN}.json`` and ``TextMapMain{JP,CHS,EN}.json``
  zzz       dimbreath/ZenlessData ``TextMap{_JA,,_EN}TemplateTb.json`` and their ``Overwrite`` tables

The tables are used only on the maintainer's machine; nothing from them ships, and they are never committed.
Listed names are checked (``--all``: every name). Each gets a category; ``ok`` and ``variant-ok`` are left out:

  ok             one key's ja, zh and en are the name's as written (an empty en is not compared;
                 each of an en's " / " alternatives must be the en of one such key)
  punct-differs  as ok once quotes, spacing and case are ignored, but not as written (see below): the game's
                 strings are given (an unclosed 「, quotes the game does not write, LightKeepers)
  en-differs     ja and zh share a key, an en alternative is the en of none of them (the game's en is given)
  zh-differs     ja is a whole table string, no key with it carries zh (the game's zh is given)
  ja-differs     zh is a whole table string, no key with it carries ja
  both-found     ja and zh are each whole table strings, never under one key
  ja-missing     ja is no whole table string: names the source took from dialogue or a book
  variant-*      a variant (``variant-of:``) is a spelling ASR may write, not the game's: its ja is only looked
                 for, whole or inside a string; its zh, which goes out, must be written (closed, brackets aside)
                 inside some zh string; ``variant-ja-missing`` is a misspelling whose zh the game writes,
                 ``variant-zh-missing`` a zh the game does not write (whatever its ja)

A name not ``ok`` is then looked for inside longer strings: ``cooc`` counts the keys whose ja text holds its ja
and whose zh text holds its zh, ``en_cooc`` how many of those hold each en alternative as whole words. The
category gets a prefix: ``cooc-ok:`` (the game writes the three together, as written), ``cooc-punct-differs:``
(only once quotes, spacing or an unclosed bracket are let pass), ``cooc-en-differs:`` (an alternative never with them),
``no-cooc:``. Only ``cooc-ok:ja-missing`` passes: a name the game writes only in dialogue or a book. Every other
row needs reading before release, ``cooc-ok:`` on a name with its own key included (the name's own key is what
the game calls it; a line elsewhere may spell it another way). Variants get no prefix; a misspelling variant
(``variant-ja-missing``) is as intended. Finding ignores markup (``{…}`` ruby, ``<…>``), ・·•「」『』《》【】,
quotes, 。 and spaces in ja and zh, and quotes, a closing full stop, a leading "The" and case in en. The written
check then wants ja and zh inside the game's text and en one of the game's names (``en_forms``), with only markup
removed, spacing collapsed and the middle dots ・·• taken as one; 「」『』《》【】 may differ (the game quotes a name
where it stands in a line: 断片領域「タウ」), but ours must close what it opens. At a name's own key en's case must
match too, unless only the case differs and a line writes ours (a label "Element", "element" in a line); a name
found only in lines is compared in any case (a line may start with it).
A confirmed slip goes into ``GAMES[...].fixes``; Genshin's two-character slips also into
``genshin-textmap-fixes.json`` (``fixes`` wins).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import unicodedata
from pathlib import Path

FILES = {
    "genshin": {"ja": ("TextMapJP.json", "TextMap_MediumJP.json"), "zh": ("TextMapCHS.json", "TextMap_MediumCHS.json"),
                "en": ("TextMapEN.json", "TextMap_MediumEN.json")},
    "starrail": {"ja": ("TextMapJP.json", "TextMapMainJP.json"), "zh": ("TextMapCHS.json", "TextMapMainCHS.json"),
                 "en": ("TextMapEN.json", "TextMapMainEN.json")},
    "zzz": {"ja": ("TextMap_JATemplateTb.json", "TextMap_JAOverwriteTemplateTb.json"),
            "zh": ("TextMapTemplateTb.json", "TextMapOverwriteTemplateTb.json"),
            "en": ("TextMap_ENTemplateTb.json", "TextMap_ENOverwriteTemplateTb.json")},
}
READABLE_DIRS = (("ja", "JP", "_JP"), ("zh", "CHS", ""), ("en", "EN", "_EN"))
PASSING = "cooc-ok:ja-missing"
_MARKUP = re.compile(r"\{[^}]*\}|<[^>]*>|^#")
_STRIP = re.compile(r"[\s・·•「」『』《》【】\"“”。]")
_BRACKETS = re.compile(r"[「」『』《》【】]")
_QUOTES = re.compile(r"[\"“”]")
_DOTS = str.maketrans("·•･", "・・・")
_WRAP = re.compile(r'^(?:"(.*)"|“(.*)”|「(.*)」|『(.*)』)$')
_PAIRS = dict(zip("「『《【“（(", "」』》】”）)"))


def lit(text: str) -> str:
    """The text as written: markup removed, spacing collapsed, the three middle dots one."""
    return re.sub(r"\s+", " ", _MARKUP.sub("", unicodedata.normalize("NFKC", text or ""))).strip().translate(_DOTS)


def _unbracket(text: str) -> str:
    return _BRACKETS.sub("", text)


def norm(text: str) -> str:
    """ja / zh for finding: brackets, quotes, ・, 。 and spaces dropped too."""
    return _STRIP.sub("", lit(text))


def en_forms(text: str) -> set[str]:
    """A game en as a name may be written: as it stands, or without a pair of quotes around it, a closing full stop
    or ! / ?, or a leading "The" (an article, not the name)."""
    forms = {lit(text)}
    for f in list(forms):
        if m := _WRAP.match(f):
            forms.add(next(g for g in m.groups() if g is not None))
    forms |= {f.rstrip(".!?") for f in forms}
    forms |= {f[4:] for f in forms if f.startswith("The ")}
    return {f for f in forms if f}


def norm_en(text: str) -> str:
    """en for finding: quotes, a closing full stop and a leading "the" dropped, case folded."""
    text = _QUOTES.sub("", lit(text)).rstrip(".!?").casefold()
    return text[4:] if text.startswith("the ") else text


def _words(alt: str) -> re.Pattern:
    return re.compile(r"(?<![0-9a-z])" + re.escape(alt) + r"(?![0-9a-z])")


def closed(text: str) -> bool:
    """Every bracket opened is closed, in order, and ASCII quotes come in pairs."""
    stack = []
    for ch in text:
        if ch in _PAIRS:
            stack.append(_PAIRS[ch])
        elif ch in _PAIRS.values() and (not stack or stack.pop() != ch):
            return False
    return not stack and text.count('"') % 2 == 0


class Tables:
    """The game's strings per language (``{key: text}``), normalised, with the keys of each whole ja / zh string."""

    def __init__(self, raw: dict[str, dict[str, str]]):
        self.raw = raw
        self.lit = {lang: {k: lit(v) for k, v in tab.items()} for lang, tab in raw.items()}
        self.ub = {lang: {k: _unbracket(v) for k, v in self.lit[lang].items()} for lang in ("ja", "zh")}
        self.n = {lang: ({k: norm_en(v) for k, v in tab.items()} if lang == "en" else
                         {k: _STRIP.sub("", v) for k, v in tab.items()}) for lang, tab in self.lit.items()}
        self.whole = {lang: collections.defaultdict(list) for lang in ("ja", "zh")}
        for lang, index in self.whole.items():
            for key, value in self.n[lang].items():
                if value:
                    index[value].append(key)
        self.ja_keys = [k for k, v in self.n["ja"].items() if v]

    @classmethod
    def load(cls, game: str, folder: Path, readables: Path | None = None) -> "Tables":
        raw: dict[str, dict[str, str]] = {}
        for lang, names in FILES[game].items():
            raw[lang] = {}
            for name in names:
                if (folder / name).is_file():
                    raw[lang].update((k, v) for k, v in json.loads((folder / name).read_text(encoding="utf-8")).items()
                                     if isinstance(v, str))
        for lang, sub, suffix in READABLE_DIRS if readables else ():
            for path in (readables / sub).glob("*.txt"):
                stem = path.stem.removesuffix(suffix) if suffix else path.stem
                raw[lang]["readable:" + stem] = path.read_text(encoding="utf-8", errors="replace")
        return cls(raw)

    def _game(self, lang: str, keys) -> str:
        return " | ".join(sorted({self.lit[lang].get(k, "") for k in keys})[:4])

    def _written_whole(self, keys, ja: str, zh: str, alts: list[str], fold: bool = False) -> bool:
        """At ``keys`` the game's ja and zh hold ours as written, brackets aside (ours must close its own), and each
        en alternative is the en of one of those same keys (``en_forms``; with ``fold``, in any case): a ja / zh
        written at one key and an en at another is no string the game writes."""
        if not (closed(ja) and closed(zh) and all(closed(a) for a in alts)):
            return False
        ja, zh = _unbracket(ja), _unbracket(zh)
        keys = [k for k in keys if ja in self.ub["ja"][k] and zh in self.ub["zh"].get(k, "")]
        same = (lambda a, f: a.casefold() == f.casefold()) if fold else (lambda a, f: a == f)
        return bool(keys) and all(any(same(a, f) for k in keys for f in en_forms(self.raw["en"].get(k, "")))
                                  for a in alts)

    def _written_inside(self, keys, ja: str, zh: str, alts: list[str], case: bool = False) -> bool:
        """At ``keys`` the game writes ours inside its text: ja and zh as written, brackets aside, each en
        alternative as whole words (in any case, or with ``case`` in ours)."""
        if not (closed(ja) and closed(zh) and all(closed(a) for a in alts)):
            return False
        ja, zh = _unbracket(ja), _unbracket(zh)
        keys = [k for k in keys if ja in self.ub["ja"][k] and zh in self.ub["zh"].get(k, "")]
        return bool(keys) and all(any(re.search(r"(?<![0-9A-Za-z])" + re.escape(a) + r"(?![0-9A-Za-z])",
                                                self.lit["en"].get(k, ""), 0 if case else re.IGNORECASE)
                                      for k in keys) for a in alts)

    def _hits(self, ja: str, zh: str) -> list[str]:
        return [k for k in self.ja_keys if ja in self.n["ja"][k] and zh in self.n["zh"].get(k, "")]

    def classify(self, ja_raw: str, zh_raw: str, en_raw: str, variant: bool = False) -> tuple[str, str]:
        ja, zh = norm(ja_raw), norm(zh_raw)
        ja_lit, zh_lit = lit(ja_raw), lit(zh_raw)
        alts = [lit(a) for a in en_raw.split(" / ") if a.strip()]
        jk, zk = self.whole["ja"].get(ja, []), self.whole["zh"].get(zh, [])
        if variant:  # the ja may be a misspelling ASR writes; the zh goes out as it is, so as written
            ja_in = len(jk) or sum(1 for v in self.n["ja"].values() if ja in v)
            zh_ub = _unbracket(zh_lit)
            zh_in = sum(1 for v in self.ub["zh"].values() if zh_ub in v) if closed(zh_lit) and zh_ub else 0
            cat = "variant-ok" if ja_in and zh_in else "variant-ja-missing" if zh_in else "variant-zh-missing"
            return cat, f"ja whole={len(jk)} found={ja_in} zh whole={len(zk)} found={zh_in}"
        if both := set(jk) & set(zk):
            game_en = {norm_en(self.raw["en"].get(k, "")) for k in both}
            if any(norm_en(a) not in game_en for a in alts):
                cat, evidence = "en-differs", "game en: " + self._game("en", both)
            elif self._written_whole(both, ja_lit, zh_lit, alts) or (    # only the case differs from the name's
                    self._written_whole(both, ja_lit, zh_lit, alts, fold=True)   # key and a line writes ours:
                    and self._written_inside(self._hits(ja, zh), ja_lit, zh_lit, alts, case=True)):  # "element"
                return "ok", ""
            else:
                cat = "punct-differs"
                evidence = f"game: {self._game('ja', both)} ;; {self._game('zh', both)} ;; {self._game('en', both)}"
        elif jk and zk:
            cat, evidence = "both-found", f"game zh at ja: {self._game('zh', jk)} ;; game ja at zh: {self._game('ja', zk)}"
        elif jk:
            cat, evidence = "zh-differs", "game zh: " + self._game("zh", jk)
        elif zk:
            cat, evidence = "ja-differs", "game ja: " + self._game("ja", zk)
        else:
            cat, evidence = "ja-missing", ""
        hits = self._hits(ja, zh)
        per_alt = [sum(1 for k in hits if _words(norm_en(a)).search(self.n["en"].get(k, ""))) for a in alts]
        if not hits:
            prefix = "no-cooc:"
        elif not all(per_alt):
            prefix = "cooc-en-differs:"
        elif self._written_inside(hits, ja_lit, zh_lit, alts):
            prefix = "cooc-ok:"
        else:
            prefix = "cooc-punct-differs:"
        en_cooc = "/".join(map(str, per_alt)) or "-"
        return prefix + cat, f"{evidence} ;; cooc={len(hits)} en_cooc={en_cooc}".lstrip(" ;")


def check(tables: Tables, terms: list[dict], everything: bool = False) -> list[list[str]]:
    rows = []
    for t in terms:
        if not everything and (t["rare"] or t["skip"]):
            continue
        variant = next((x.split(":", 1)[1] for x in t["tags"] if x.startswith("variant-of:")), "")
        cat, evidence = tables.classify(t["ja"], t["zh"], t["en"], bool(variant))
        if cat not in ("ok", "variant-ok"):
            rows.append([cat, t["ja"], t["zh"], t["en"], t["label"], variant, evidence])
    return sorted(rows, key=lambda r: (r[0] == PASSING, r))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("game", choices=sorted(FILES))
    p.add_argument("glossary", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--all", action="store_true", help="check rare and skip names too")
    p.add_argument("--readables", type=Path, help="Genshin: animegamedata2's Readable directory")
    a = p.parse_args()
    terms = json.loads(a.glossary.read_text(encoding="utf-8"))["terms"]
    rows = check(Tables.load(a.game, a.folder, a.readables), terms, a.all)
    a.out.write_text("cat\tja\tzh\ten\tlabel\tvariant_of\tevidence\n" + "".join("\t".join(r) + "\n" for r in rows),
                     encoding="utf-8")
    counts = collections.Counter(r[0] for r in rows)
    to_read = sum(n for c, n in counts.items() if c not in (PASSING, "variant-ja-missing"))
    print(f"{to_read} names to read, {counts[PASSING]} only in dialogue or books, {counts['variant-ja-missing']}"
          " misspelling variants ->", a.out,
          dict(counts.most_common()))


if __name__ == "__main__":
    main()
