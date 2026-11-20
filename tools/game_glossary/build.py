"""Build a game glossary file (format 1, read by ``finesub.llm.glossary_store``) for publishing as a Release asset.

    PYTHONPATH=<checkout>/src python tools/game_glossary/build.py genshin \
        --terms <terms.jsonl> --dates <dates.json> --famous <famous.json> --background <background.tsv> \
        --source-commit <upstream commit> --out <glossary.json> [--built YYYY-MM-DD]

Run from the checkout root (the background list's paths are relative to it); needs the ``kb-phonetic`` extra.

Inputs, all made by the maintainer before this step:

- ``--terms``: one JSON object per line, one per official name: ``ja``, ``reading`` (kana, may be empty), ``zh``,
  ``zh_alts``, ``en``, ``tags``, ``note_zh``; extracted from the upstream named in ``GAMES`` (``source``) at
  ``--source-commit``. Its license allows redistribution until its owner asks to stop, which is why the list is a
  withdrawable download and never ships inside a wheel.
- ``--dates``: ``{en name: {"date": "YYYY-MM-DD"}}``, when the upstream added a name (or tagged it playable, the
  later of the two); a name missing here counts as old. The upstream adds a character when it is announced, which can
  be well before release (Natlan's cast in 2024-07, an NPC years before): ``Game.released`` corrects the playable
  characters released inside the window whose upstream date falls before it.
- ``--famous``: ``{zh name: right answers}``, a closed-book probe of the resident-list pool: a model with no tools
  is asked three times for the official Chinese of each Japanese name. ``FAMOUS_PASSES`` right answers make it
  famous: it stays off the resident list, and a window does not list it when ASR wrote it right.
- ``--background``: ``<label>\\t<raw.srt path>`` per recording that is NOT about the game; its rows give each
  (name, ASR spelling) pair's per-row rate of matching by sound, which the window matcher uses to drop pairs that
  fire on ordinary talk. Only the rates are published, never the rows.

What a glossary must satisfy (the resident list, simplified Chinese, what counts as a name, the reader's pass and
how to judge it) is in docs/knowledge.md, section 一份名单要满足什么; the steps from inputs to a published Release are
in this directory's README.md. "Last two years" is counted back from ``--built``, which is why a build goes stale
(``glossary_store.STALE_DAYS``).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from finesub.llm import glossary_match as gm
from finesub.llm import glossary_store as gs
from finesub.llm import kb_phonetic_hints as K
from finesub.text import t2s_converter

FAMOUS_PASSES = 3
NOTE_CHARS = 40
RECENT_YEARS = 2
# Characters OpenCC's 繁→简 rewrites that simplified Chinese also uses, so an official name keeps them: 篠冢, 乾子,
# 灵濛山 (it would write 筿冢, 干子, 灵蒙山).
T2S_KEEP = frozenset("篠乾濛簾")
# Japanese forms the upstream sometimes writes in a Chinese name (隠玉之泉), which OpenCC leaves alone. Other kanji in
# Chinese names are mostly official (薙草之稻光, 天云峠, 柊千里) and stay.
SHINJITAI = str.maketrans("隠髄歩舎焔穏", "隐髓步舍焰稳")
_QUOTED_JAPANESE = re.compile(r"(「[^」]*[\u3041-\u3096\u30a1-\u30fa][^」]*」)")
LABELS = (  # first match wins
    ("character-main", "可玩角色"), ("fatui", "愚人众"), ("character-sub", "角色"), ("organization", "组织"),
    ("location", "地点"), ("element", "元素 / 机制"), ("enemy", "敌人"), ("weapon", "武器"), ("artifact", "圣遗物"),
    ("quest", "任务"), ("event", "活动"), ("food", "料理"), ("item", "物品"), ("title", "称号"),
)


@dataclass(frozen=True)
class Game:
    name: str
    aliases: tuple[str, ...]
    intro: str
    source: str                        # the upstream the terms come from; its license governs redistribution
    core_tags: frozenset[str]          # a term with one of these is a core name ...
    fatui_excluded: frozenset[str]     # ... and so is a Fatui term without any of these (its NPCs and foot soldiers
                                       # took half the list and were almost never said)
    not_names: frozenset[str]          # tags of address forms (相棒): never core unless also a playable character
    released: dict[str, str]           # en name -> release date, for playable characters the upstream named early
    fixes: dict[str, dict[str, str]]   # upstream ja -> fields to replace (``ja``, ``zh``, ``label``): its slips
    drop: frozenset[str]               # upstream ja of terms that pair two things, carry no usable name, or are
                                       # address forms that are everyday words (相棒)
    ordinary: frozenset[str]           # core names that are also everyday words with another sense (無効, "invalid",
                                       # is the 免疫 mechanic): off the resident list, which every window carries;
                                       # still window candidates where a row writes them


GAMES = {
    "genshin": Game(
        name="原神",
        aliases=("Genshin Impact", "Genshin", "げんしん", "げんしんインパクト"),
        source="https://github.com/xicri/genshin-langdata",
        intro="原神官方日中英专名表（只读）。录音讲原神时点名它：纠错窗口会自动附上常驻名单、本片反复出现的名字与本窗的原样 / 音近候选",
        core_tags=frozenset({"character-main", "organization", "element"}),
        fatui_excluded=frozenset({"character-sub", "enemy", "enemy-boss", "enemy-legend"}),
        not_names=frozenset({"how-to-call"}),
        released={"Chasca": "2024-11-20", "Ororon": "2024-11-20", "Citlali": "2025-01-01", "Mavuika": "2025-01-01",
                  "Iansan": "2025-03-26"},
        ordinary=frozenset({"無効", "開花", "運輸局"}),
        # From a reader's pass (2026-11-13), kept where the slip is plain: a typo, a dropped character, a swapped
        # kana, a series written two ways.
        fixes={
            "スペランザ": {"zh": "斯佩兰扎"}, "曚雲神社": {"zh": "曚云神社"}, "リューバ": {"zh": "利乌芭"},
            "最勝のアメシスト・欠片": {"zh": "最胜紫晶断片"}, "最勝のアメシスト・砕屑": {"zh": "最胜紫晶碎屑"},
            "護法仙衆夜叉録": {"zh": "护法仙众夜叉录"}, "コアホイール": {"zh": "核心轮毂"},
            "賢医の仮面": {"zh": "贤医的假面"}, "ライトキーバー": {"ja": "ライトキーパー"},
            "パティサラ": {"label": ""}, "征討領域": {"label": ""}, "アルケウム": {"label": "物品"},
            "イノシシプリンセス": {"label": "物品"},
        },
        drop=frozenset({"無妄の翁", "ビュンビュンスパイ1号", "相棒", "お嬢様", "お嬢", "可愛い子ちゃん"}),
    ),
}


def _is_kana(text: str) -> bool:
    return all("぀" <= ch <= "ヿ" or ch in "・ー" for ch in text)


def _kata(text: str) -> str:
    return "".join(chr(ord(ch) + 0x60) if "ぁ" <= ch <= "ゖ" else ch for ch in text)


def _bare(name: str) -> str:
    """A name without its qualifier or quotes: 炎 (元素) -> 炎, 「月の少女」 -> 月の少女."""

    for opener in (" (", "(", "（"):
        if opener in name:
            name = name.split(opener, 1)[0]
    name = name.strip()
    if name[:1] in "「『" and name[-1:] in "」』":  # only a whole-name quote: 「死兆星号」級二番艦 keeps its own
        name = name[1:-1]
    return name.strip()


def _kana(ja: str, reading: str) -> str:
    """What the sound matcher compares: the reading (or a kana name's spelling) in katakana; "" when a kanji name's
    reading is not known in full."""

    if reading or _is_kana(ja):
        return re.sub(r"[・･\s]", "", _kata(reading or _bare(ja)))
    kana = K.reading(ja)
    return _kata(kana) if kana and K.reading_complete(ja, kana) else ""


def _short_reading(ja: str, reading: str) -> bool:
    """Too short for sound matching to find: fewer than ``K.MIN_MORA`` morae, or a loose reading under 4 kana that is
    also under ``K.MIN_ROMAJI`` romaji letters. Such names can only be helped by the resident list."""

    if reading or _is_kana(ja):
        kana = "".join(ch for ch in _kata(reading or ja) if "ァ" <= ch <= "ヺ" or ch == "ー")
    else:
        kana = K.reading(ja)
        kana = kana if K.reading_complete(ja, kana) else ""
    if not kana or K.mora_count(kana) < K.MIN_MORA:
        return True
    loose = K.loose(kana)
    return len(loose) < 4 and len(K.romaji(loose)) < K.MIN_ROMAJI


def _simplified(text: str) -> str:
    """``text`` in simplified characters: the upstream writes a few names and notes in traditional ones. Japanese a
    note quotes (「鍾離さん」, told by its kana) stays as written."""

    cc = t2s_converter()
    parts = _QUOTED_JAPANESE.split(text)  # odd items are the quotes
    return "".join(part if i % 2 else "".join(ch if ch in T2S_KEEP else cc.convert(ch)
                                              for ch in part.translate(SHINJITAI))
                   for i, part in enumerate(parts))


def _not_a_name(ja: str, zh: str) -> bool:
    """An upstream row that names nothing a transcript would write: a bracketed description (``(ガチャで) 引く``), a
    dish template (``美味しそうな〇〇``), a demonym's fragment (``稲妻の`` = 稻妻人)."""

    return ja[:1] in "(（" or "〇〇" in ja or (ja.endswith("の") and zh.endswith("人"))


def _note(text: str) -> str:
    """The whole sentences of a note that fit in ``NOTE_CHARS`` (none when the first does not): a note cut mid-sentence
    reads as a different claim."""

    text = _simplified(text).replace("|", "/").replace("\n", " ").strip()
    kept = ""
    for sentence in re.findall(r"[^。！？]+[。！？]?", text):
        if len(kept + sentence) > NOTE_CHARS:
            break
        kept += sentence
    return kept.strip()


def _label(tags: set[str]) -> str:
    return next((label for key, label in LABELS if any(t == key or t.startswith(key) for t in tags)), "")


def _groups(raw: list[dict]) -> dict[str, str]:
    """Chinese name -> its entity key: names some term lists together (女士 / 席诺拉) are one entity."""

    group: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for term in raw:
        names = [_simplified(n) for n in term.get("zh_alts") or [term["zh"]] if len(n) >= 2]
        if not names:
            continue
        key = next((group[n] for n in names if n in group), names[0])
        for n in names:
            if n in group and group[n] != key:
                for m in members.pop(group[n]):
                    group[m] = key
                    members.setdefault(key, set()).add(m)
            group[n] = key
            members.setdefault(key, set()).add(n)
    return group


def _srt_rows(path: Path) -> list[str]:
    rows = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8-sig").strip()):
        lines = block.splitlines()
        arrow = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if arrow is not None and (text := " ".join(x.strip() for x in lines[arrow + 1:] if x.strip())):
            rows.append(text)
    return rows


def build(game: Game, terms_file: Path, dates_file: Path, famous_file: Path, background: Path, built: str,
          source_commit: str) -> dict:
    raw = [json.loads(line) for line in terms_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    dates = json.loads(dates_file.read_text(encoding="utf-8"))
    famous = {_simplified(zh): n for zh, n in json.loads(famous_file.read_text(encoding="utf-8")).items()}
    group = _groups(raw)
    built_day = date.fromisoformat(built)
    cut = built_day.replace(year=built_day.year - RECENT_YEARS).isoformat()

    terms, seen, unprobed = [], set(), []
    for r in raw:
        fix = game.fixes.get(r["ja"].strip(), {})
        ja, zh = fix.get("ja", r["ja"].strip()), fix.get("zh") or _simplified(r["zh"].strip())
        if not ja or not zh or (ja, zh) in seen or r["ja"].strip() in game.drop or _not_a_name(ja, zh):
            continue
        seen.add((ja, zh))
        tags, reading = set(r.get("tags") or ()), str(r.get("reading") or "")
        address = bool(tags & game.not_names) and "character-main" not in tags
        core_name = not address and (
            bool(tags & game.core_tags) or ("fatui" in tags and not tags & game.fatui_excluded))
        en = r.get("en", "")
        recent = max((dates.get(en) or {}).get("date") or "", game.released.get(en, "")) >= cut
        core = core_name and ja not in game.ordinary and (recent or _short_reading(ja, reading))
        if core and zh not in famous:
            unprobed.append(f"{ja}={zh}")
        terms.append({
            "ja": ja, "zh": zh, "en": str(r.get("en") or "").strip(),
            "kana": _kana(ja, reading), "reading": bool(reading) or not _is_kana(ja),
            "label": fix.get("label", _label(tags)),
            "note": _note(str(r.get("note_zh") or "")),
            "group": group.get(zh, ja), "core": core, "famous": core and famous.get(zh, 0) >= FAMOUS_PASSES,
        })
    if unprobed:  # the resident-list pool the probe must cover: an empty --famous file lists all of it
        raise SystemExit(f"error: {len(unprobed)} core names missing from the famous probe; probe them and rerun:\n"
                         + "\n".join(unprobed))

    rows = []
    for line in background.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows += _srt_rows(Path(line.split("\t", 1)[1].strip()))
    matcher = gm.SoundMatcher(gm.Term(t["ja"], t["kana"], t["reading"]) for t in terms if t["kana"])
    per_text = collections.Counter(rows)  # match() keys by distinct text; a repeated row counts each time
    fired: collections.Counter = collections.Counter()
    for text, hits in matcher.match(rows).items():
        for pair in {f"{h.ja}|{h.surface}" for h in hits}:
            fired[pair] += per_text[text]
    noise = {pair: float(f"{n / len(rows):.3g}") for pair, n in sorted(fired.items())}

    core = [t for t in terms if t["core"]]
    print(f"{len(terms)} terms; resident list {sum(not t['famous'] for t in core)} of {len(core)} core "
          f"(cut {cut}); {len(noise)} noise pairs from {len(rows)} background rows", file=sys.stderr)
    return {
        "format": gs.FORMAT,
        "game": {"name": game.name, "aliases": list(game.aliases), "intro": game.intro},
        "built": built,
        "provenance": {"source": game.source, "source_commit": source_commit, "famous_passes": FAMOUS_PASSES, "recent_years": RECENT_YEARS, "background_rows": len(rows)},
        "terms": terms,
        "noise": noise,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=sorted(GAMES))
    ap.add_argument("--terms", type=Path, required=True)
    ap.add_argument("--dates", type=Path, required=True)
    ap.add_argument("--famous", type=Path, required=True)
    ap.add_argument("--background", type=Path, required=True)
    ap.add_argument("--source-commit", required=True, help="the upstream commit the terms were extracted from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--built", default=date.today().isoformat())
    args = ap.parse_args()
    data = build(GAMES[args.game], args.terms, args.dates, args.famous, args.background, args.built,
                 args.source_commit)
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    glossary = gs.parse(body)  # what users will install must load
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    print(f"{args.out}: {len(body)} bytes, sha256 {glossary.sha256}", file=sys.stderr)


if __name__ == "__main__":
    main()
