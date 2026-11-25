"""Step 1 of the README: extract an upstream into the term table ``build.py --terms`` reads (maintainers only).

    python -I tools/game_glossary/extract.py genshin <genshin-langdata clone> <out.jsonl>
    python -I tools/game_glossary/extract.py starrail <StarRailRes clone> <out.jsonl> --textmap <dir>
    python -I tools/game_glossary/extract.py zzz <ZenlessData TextMap dir> <out.jsonl>

Each output line is one term: ``{"id", "game", "ja", "reading", "zh", "zh_alts", "zh_tw", "en", "tags", "note_zh",
"file", "variants"}``, plus ``lines`` / ``story_lines`` for a speaker of the dialogue tables. ``reading`` is the kana reading when the source gives one; ``variants`` lists other Japanese
spellings of the same name (``[{"ja", "zh"}]``: the upstream's variants, a character's given name) that
``build.py`` turns into names of the same entity. ``id`` is stable across upstream versions and keys the dates
(``dates.py``).

Upstreams (see ``build.py``'s ``GAMES`` for which one a game uses):

- ``genshin``: xicri/genshin-langdata, ``dataset/dictionary/*.ts``. The files are TypeScript object literals, read
  as data and never executed: comments, the import/export wrapper and trailing commas are stripped.
- ``starrail``: Mar-7th/StarRailRes ``index_new/{jp,cn,en}`` (characters, paths, elements, light cones, relic sets,
  materials), plus the factions, Aeons, worlds and enemy factions that StarRailRes does not index, read from the
  game's text tables (``--textmap``: a directory holding ``TextMap{JP,CHS,EN}.json`` and the ``ExcelOutput`` files
  ``AvatarCamp``, ``MonsterCamp``, ``RogueAeonDisplay``, ``WorldDataConfig``, ``MazePlane``, and for the less
  common names ``MonsterTemplateConfig`` (enemies, by rank), ``MainMission`` (missions, by type), ``NPCData`` and
  ``TalkSentenceConfig`` (NPCs, each speaker with its line count) from Dimbreath/turnbasedgamedata; a missing file
  leaves its names out).
- ``zzz``: Dimbreath/ZenlessData ``TextMap`` (``TextMap_JATemplateTb.json``, ``TextMapTemplateTb.json`` (简体),
  ``TextMap_ENTemplateTb.json``), whose keys name what a string is (``Partner_Name_1011``, ``PartnerCamp_Name_01``,
  ``Item_Weapon_S_1021_Name``): agents, factions, attributes, W-engines, drive discs, Bangboo, places, bosses,
  enemies, commissions, materials, and every speaker of the dialogue keys (``..._Name_3``) with its line count and
  how many of the lines are in the main story (``Main_`` keys).

Only names are taken: no descriptions, dialogue or story text. A table string or speaker label must read as a name
(``name_like``): a description (考え込む作業員), a template (``{0}``), a UI string or a sentence is left out.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


def _strip_comments(text: str) -> str:
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            i = text.index("*/", i) + 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_ts_literal(text: str) -> list[dict]:
    body = _strip_comments(text)
    start = body.index("export default") + len("export default")
    body = body[start:]
    body = re.sub(r"\]\s*(as const)?\s*(satisfies [^;]*)?;?\s*$", "]", body.strip())
    parts = re.split(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""", body)
    for i in range(1, len(parts), 2):  # single-quoted literals become JSON strings
        if parts[i].startswith("'"):
            parts[i] = json.dumps(parts[i][1:-1].replace("\\'", "'"), ensure_ascii=False)
    for i in range(0, len(parts), 2):  # the even parts are outside string literals
        parts[i] = re.sub(r'([{,\s])([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', parts[i])
        parts[i] = re.sub(r",(\s*[\]}])", r"\1", parts[i])
    return json.loads("".join(parts))


def alternatives(name: str) -> list[str]:
    """``女士 / 席诺拉`` -> both; a qualifier is dropped (``杰克(枫丹)`` -> ``杰克``, ``ジャック(フォンテーヌ)`` ->
    ``ジャック``); the first stays first. Used for both languages."""

    out = []
    for part in name.split(" / "):
        part = re.sub(r"\s*[(（][^()（）]*[)）]$", "", part.strip())
        if len(part) > 2 and part[0] + part[-1] in ("「」", "『』") and part[0] not in part[1:-1]:
            part = part[1:-1]
        if part and part not in out:
            out.append(part.strip())
    return out or [name]


def _parts(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[・·]", text) if p.strip()]


def _given_name(ja: str, zh: str) -> dict | None:
    """``ドリー・サングマハベイ`` / ``多莉·桑歌玛哈巴依`` -> ``{"ja": "ドリー", "zh": "多莉"}``: the name a transcript
    writes for a character the upstream lists by full name."""

    jp, cp = _parts(ja), _parts(zh)
    if len(jp) >= 2 and len(jp) == len(cp) and len(jp[0]) >= 2:
        return {"ja": jp[0], "zh": cp[0], "given": True}
    return None


def _variant(form: str, ja: str, zh: str) -> dict:
    """A variant spelling with its Chinese: the matching part of a ``・``-joined name (``シュミット`` of
    ``ミカ・シュミット`` is ``施密特``), else the whole name's."""

    jp, cp = _parts(ja), _parts(zh)
    if len(jp) >= 2 and len(jp) == len(cp) and form in jp:
        return {"ja": form, "zh": cp[jp.index(form)]}
    return {"ja": form, "zh": zh}


def genshin(repo: Path) -> list[dict]:
    terms = []
    for path in sorted((repo / "dataset" / "dictionary").glob("*.ts")):
        if path.stem == "tmp":
            continue
        for word in parse_ts_literal(path.read_text(encoding="utf-8")):
            if not word.get("ja") or not word.get("zhCN"):
                continue
            # One record per Japanese form (執行官 / ファトゥス); the kana reading belongs to the first.
            forms = alternatives(word["ja"])
            zh_alts = alternatives(word["zhCN"])
            tags = word.get("tags", []) or []
            for n, form in enumerate(forms):
                variants = []
                if n == 0:
                    variants = [_variant(v, form, zh_alts[0]) for v in (word.get("variants") or {}).get("ja", [])]
                    given = _given_name(form, zh_alts[0]) if {"character-main", "character-sub"} & set(tags) else None
                    if given and given["ja"] not in {v["ja"] for v in variants}:
                        variants.append(given)
                terms.append({
                    "id": f"genshin:{path.stem}:{word.get('en', '')}",
                    "game": "genshin",
                    "ja": form,
                    "reading": word.get("pronunciationJa", "") if n == 0 else "",
                    "zh": zh_alts[0],
                    "zh_alts": zh_alts,
                    "zh_tw": word.get("zhTW", ""),
                    "en": word.get("en", ""),
                    "tags": tags,
                    "note_zh": word.get("notesZh", ""),
                    "file": path.stem,
                    "variants": variants,
                })
    return terms


# --- Honkai: Star Rail ---------------------------------------------------------------------------------------------

# StarRailRes index file -> tags. Items: the materials and currencies a stream names (the upstream's books,
# missions, phone themes and event props are not names a transcript writes).
STARRAIL_FILES = {"characters": ["character-main"], "paths": ["path"], "elements": ["element"],
                  "light_cones": ["light-cone"], "relic_sets": ["relic"]}
STARRAIL_ITEMS = {("Material", "Material"), ("Material", "CommonMonsterDrop"), ("Material", "WeeklyMonsterDrop"),
                  ("Material", "ComposeMaterial"), ("Material", "TracePath"), ("Virtual", "Virtual"),
                  ("Usable", "Food")}
# ExcelOutput file -> (name field, tags, or a row -> tags function: None leaves the row out) for the names
# StarRailRes does not index. Enemies by rank (a BigBoss / LittleBoss is a boss), missions by type, NPCs by kind
# (an "Avatar" NPC is a playable character's field model: its name is already a character).
STARRAIL_MONSTER_RANK = {"BigBoss": ["enemy-boss"], "LittleBoss": ["enemy-boss"], "Elite": ["enemy-elite"],
                         "MinionLv2": ["enemy"], "Minion": ["enemy"]}
STARRAIL_MISSION_TYPE = {"Main": ["quest-main"], "Companion": ["quest-companion"], "Branch": ["quest-side"],
                         "Daily": ["quest-daily"], "Gap": ["quest-side"]}
STARRAIL_EXCEL = {
    "AvatarCamp": ("Name", ["faction"]),
    "MonsterCamp": ("Name", ["enemy-faction"]),
    "RogueAeonDisplay": ("RogueAeonName", ["aeon"]),
    "WorldDataConfig": ("WorldName", ["world"]),
    "MazePlane": ("PlaneName", ["location"]),
    "MonsterTemplateConfig": ("MonsterName", lambda row: STARRAIL_MONSTER_RANK.get(row.get("Rank"), ["enemy"])),
    "MainMission": ("Name", lambda row: STARRAIL_MISSION_TYPE.get(row.get("Type"))),
    "NPCData": ("DefaultNPCName", lambda row: None if row.get("SubType") in ("Avatar", "PlayerAvatar", "Monster")
                else ["npc"]),
}
# Not a name: what a dialogue line or a speaker label is when it describes someone (憂鬱な外計局職員, 謎の声) or is
# a template / UI string. A name keeps at most these hiragana runs (さん, じい, おじさん, ちゃん).
_NAME_KANA_OK = re.compile(r"(?:さん|くん|ちゃん|じい|じいさん|ばあ|ばあさん|おじさん|おばさん|おじいさん|おばあさん|の|ノ)")
_NOT_A_NAME = re.compile(r"[{}<>%#\[\]0-9０-９？?！!。、…\n]|^[\s・ー]*$|クリア|テスト|test|Test|選択肢|プレイヤー")


def name_like(ja: str, person: bool = False) -> bool:
    """A speaker label or table string that reads as a proper name: short, no template or sentence marks, and no
    hiragana beyond an address suffix or, for what is not a person, a particle の (``ホンおじさん`` yes;
    ``考え込む作業員``, ``ドリームレスコーヒーの店員`` no)."""

    if not ja or len(ja) > 24 or _NOT_A_NAME.search(ja) or ja.isascii() or (person and "の" in ja):
        return False
    rest = _NAME_KANA_OK.sub("", ja)
    return not re.search(r"[\u3041-\u3096]", rest)
_RUBY = re.compile(r"\{RUBY_B#([^}]*)\}(.*?)\{RUBY_E#\}")
_MARKUP = re.compile(r"<[^>]+>|\{[^}]*\}")
_KANJI = re.compile(r"[\u3400-\u9fff々]")


def _clean(text: str) -> str:
    """ja / zh as the game writes them: markup and ruby readings removed, no space around a middle dot."""
    return re.sub(r"\s*([・·•])\s*", r"\1", _clean_en(text).replace("･", "・"))


def _clean_en(text: str) -> str:
    """en as the game writes it: markup removed, spacing collapsed (Aventurine • Waveflair keeps its spaces)."""
    return re.sub(r"\s+", " ", _MARKUP.sub("", _RUBY.sub(r"\2", text or ""))).strip()


def _ruby_reading(text: str) -> str:
    """The kana reading a ruby-annotated string gives in full (``{RUBY_B#たいぼくし}太卜司{RUBY_E#}``), else ""."""

    if "RUBY_B#" not in (text or ""):
        return ""
    read = _MARKUP.sub("", _RUBY.sub(r"\1", text)).strip()
    return "" if _KANJI.search(read) else read


def _concepts(names: dict[str, list[str]], ja_map: dict, zh_map: dict, en_map: dict, game: str) -> list[dict]:
    """World terms no index lists (the Aeon of Trailblaze, the Hollows), found by their official Chinese: the
    Japanese string most often paired with exactly that Chinese across the text table, with a ruby reading when
    one copy carries it."""

    by_zh: dict[str, list[str]] = {}
    wanted = {zh for zhs in names.values() for zh in zhs}
    for key, zh in zh_map.items():
        zh = zh.strip() if isinstance(zh, str) else ""
        if zh in wanted and ja_map.get(key):
            by_zh.setdefault(zh, []).append(key)
    terms = []
    for tag, zhs in names.items():
        for zh in zhs:
            keys = by_zh.get(zh, [])
            counts: dict[str, int] = {}
            for k in keys:
                counts[_clean(ja_map[k])] = counts.get(_clean(ja_map[k]), 0) + 1
            if not counts:
                continue
            ja = max(counts, key=lambda j: (counts[j], -len(j)))
            reading = next((r for k in keys if _clean(ja_map[k]) == ja and (r := _ruby_reading(ja_map[k]))), "")
            en = next((_clean_en(en_map.get(k, "")) for k in keys if _clean(ja_map[k]) == ja and en_map.get(k)), "")
            terms.append({"id": f"{game}:concept:{zh}", "game": game, "ja": ja, "reading": reading, "zh": zh,
                          "zh_alts": [zh], "zh_tw": "", "en": en, "tags": [tag], "note_zh": "", "file": "concept",
                          "variants": []})
    return terms


# Official Chinese of world terms and groups the indexes above miss; a name not found in the text table is skipped.
STARRAIL_CONCEPTS = {
    "concept": ["开拓者", "星神", "命途", "令使", "无名客", "寰宇蝗灾", "铁墓", "绝灭大君", "模拟宇宙", "差分宇宙",
                "忘却之庭", "虚构叙事", "末日幻影", "黄金裔", "泰坦", "火种", "再创世", "裂界", "开拓"],
    "aeon": ["阿基维利"],
    "faction": ["仙舟联盟", "云骑军", "十王司", "太卜司", "地衡司", "丹鼎司", "工造司", "步离人", "丰饶民", "天才俱乐部",
                "列车组", "银鬃铁卫", "筑城者"],
}
ZZZ_CONCEPTS = {
    "concept": ["代理人", "绳匠", "以骸", "邦布", "以太", "空洞", "零号空洞", "虚狩", "绳网", "侵蚀", "丁尼"],
    "faction": ["空洞调查协会"],
}


def _row_id(row: dict) -> str:
    for key in ("ID", "DisplayID", "PlaneID", "AeonID", "MonsterTemplateID", "MainMissionID"):
        if key in row:
            return str(row[key])
    return ""


def starrail(repo: Path, textmap: Path | None) -> list[dict]:
    terms = []
    idx = {lang: repo / "index_new" / lang for lang in ("jp", "cn", "en")}

    def load(lang: str, name: str) -> dict:
        path = idx[lang] / f"{name}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    for name, tags in [*STARRAIL_FILES.items(), ("items", None)]:
        jp, cn, en = load("jp", name), load("cn", name), load("en", name)
        for key, value in jp.items():
            other = cn.get(key)
            if not (isinstance(value, dict) and isinstance(other, dict)):
                continue
            if tags is None and (value.get("type"), value.get("sub_type")) not in STARRAIL_ITEMS:
                continue
            ja, zh = _clean(value.get("name", "")), _clean(other.get("name", ""))
            if not ja or not zh or "{" in str(value.get("name", "")):
                continue
            term_tags = list(tags or ["item", f"item-{str(value.get('sub_type', '')).lower()}"])
            if name == "characters" and value.get("path"):
                term_tags.append(f"path-{value['path'].lower()}")
            terms.append({
                "id": f"starrail:{name}:{key}", "game": "starrail", "ja": ja, "reading": "", "zh": zh,
                "zh_alts": [zh], "zh_tw": "", "en": _clean_en((en.get(key) or {}).get("name", "")), "tags": term_tags,
                "note_zh": "", "file": name, "variants": [],
            })
    if textmap is not None:
        text = {lang: json.loads((textmap / f"TextMap{lang}.json").read_text(encoding="utf-8"))
                for lang in ("JP", "CHS", "EN")}
        for lang in ("JP", "CHS", "EN"):  # the story's strings: speakers and lines
            main = textmap / f"TextMapMain{lang}.json"
            if main.is_file():
                for key, value in json.loads(main.read_text(encoding="utf-8")).items():
                    text[lang].setdefault(key, value)
        for name, (field, tags_of) in STARRAIL_EXCEL.items():
            path = textmap / f"{name}.json"
            if not path.is_file():
                print(f"starrail: no {path.name}, its names are left out", file=sys.stderr)
                continue
            rows = json.loads(path.read_text(encoding="utf-8"))
            for row in rows if isinstance(rows, list) else rows.values():
                tags = tags_of(row) if callable(tags_of) else tags_of
                h = str((row.get(field) or {}).get("Hash", ""))
                ja, zh = _clean(text["JP"].get(h, "")), _clean(text["CHS"].get(h, ""))
                if not tags or not ja or not zh or (callable(tags_of) and not name_like(ja, tags == ["npc"])):
                    continue
                if tags[0].startswith("enemy"):  # 「（バグ）」「（スペック）」: a stage's copy of the same enemy
                    ja, zh = _QUALIFIER.sub("", ja), _QUALIFIER.sub("", zh)
                terms.append({
                    "id": f"starrail:{name}:{_row_id(row)}", "game": "starrail", "ja": ja,
                    "reading": _ruby_reading(text["JP"].get(h, "")), "zh": zh,
                    "zh_alts": [zh], "zh_tw": "", "en": _clean_en(text["EN"].get(h, "")), "tags": list(tags),
                    "note_zh": "", "file": name, "variants": [],
                })
        terms += _concepts(STARRAIL_CONCEPTS, text["JP"], text["CHS"], text["EN"], "starrail")
        talk = textmap / "TalkSentenceConfig.json"
        if talk.is_file():
            hashes = (str((row.get("TextmapTalkSentenceName") or {}).get("Hash", ""))
                      for row in json.loads(talk.read_text(encoding="utf-8")))
            terms += _speakers("starrail", collections.Counter(hashes), {}, text["JP"], text["CHS"], text["EN"])
    return terms


def _speakers(game: str, lines: collections.Counter, story: collections.Counter, ja_map: dict, zh_map: dict,
              en_map: dict) -> list[dict]:
    """Every name-like speaker of the dialogue tables as an ``npc`` term with how many lines it speaks (``lines``)
    and how many of them are in the main story (``story_lines``, where the table says): the signal ``build.py``
    reads for an important NPC. One term per Japanese name, lines summed over the keys that print it."""

    by_pair: dict[tuple[str, str], dict] = {}
    for key, n in lines.items():
        ja, zh = _clean(ja_map.get(key, "")), _clean(zh_map.get(key, ""))
        if not key or not ja or not zh or not name_like(ja, person=True):
            continue
        t = by_pair.setdefault((ja, zh), {
            "id": f"{game}:speaker:{ja}", "game": game, "ja": ja, "reading": _ruby_reading(ja_map.get(key, "")),
            "zh": zh, "zh_alts": [zh], "zh_tw": "", "en": _clean_en(en_map.get(key, "")), "tags": ["npc"],
            "note_zh": "", "file": "speaker", "variants": [], "lines": 0, "story_lines": 0})
        t["lines"] += n
        t["story_lines"] += story.get(key, 0)
    # A label printed one way in Japanese and another in Chinese for a scene (リン shown as 黑屏, a black screen)
    # is not that speaker's name: each Japanese name keeps the Chinese it is printed with most.
    best: dict[str, dict] = {}
    for t in by_pair.values():
        if t["ja"] not in best or t["lines"] > best[t["ja"]]["lines"]:
            best[t["ja"]] = t
    return list(best.values())


# --- Zenless Zone Zero ---------------------------------------------------------------------------------------------

# TextMap key pattern -> tags, first match wins.
ZZZ_KEYS = (
    (r"Partner_Name_\d+", ["character-main"]),
    (r"Avatar_(?:Female|Male)_Size\d+_[A-Za-z]+", ["character-main", "short-name"]),
    (r"PartnerCamp_Name_\d+", ["faction"]),
    (r"Camp[A-Z][A-Za-z.\-0-9]*", ["faction"]),
    (r"ElementType_[A-Za-z]+", ["element"]),
    (r"ProfessionName_[A-Za-z]+", ["specialty"]),
    (r"Item_Weapon_[SAB](?:_Common)?_\d+_Name", ["w-engine"]),
    (r"EquipmentSuit_\d+_name", ["drive-disc"]),
    (r"Bangboo_Name_\d+", ["bangboo"]),
    (r"CityLocation_\d+", ["location"]),
    (r"TeleportAreaName_\d+", ["location"]),
    (r"Zenkov_SubArea_\d+", ["location", "minor"]),
    (r"BossBattle_QuestName_\d+", ["enemy-boss"]),
    (r"ActivityBallGame_Monster_Name_\d+", ["enemy"]),
    (r"QuestName_\d+", ["quest"]),
    (r"Item_Avatar(?:Skill|Break|Rank)Material_\d+_name", ["item"]),
    (r"Item_SmallBoss\d+_name", ["item"]),
)
# Dialogue keys whose ``_Name`` strings are the speaker of a line (Main_GalGame_Chapter1_2_Name_3); Main_ ones are
# the main story.
ZZZ_SPEAKER = re.compile(r"(?=.*(?:Chat|GalGame|Galgame|Ongoing|Bubble|Talk|Story))[A-Za-z0-9_#&\-]*_Name_?\d+$")
# A boss hunt names its boss after a prefix (要警戒狩り：初生のデッドエンドブッチャー).
_PREFIXED = re.compile(r"^[^：:]*[：:]\s*|^BOSS")
_QUALIFIER = re.compile(r"\s*[（(][^）)]*[）)]$")


def zzz(textmap: Path) -> list[dict]:
    def load(name: str) -> dict:
        data = json.loads((textmap / f"{name}TemplateTb.json").read_text(encoding="utf-8"))
        over = textmap / f"{name}OverwriteTemplateTb.json"
        if over.is_file():  # strings a later patch rewrote
            data.update(json.loads(over.read_text(encoding="utf-8")))
        return data

    ja_map, zh_map, en_map = load("TextMap_JA"), load("TextMap"), load("TextMap_EN")
    patterns = [(re.compile(rx + "$"), tags) for rx, tags in ZZZ_KEYS]
    terms = []
    for key, raw in ja_map.items():
        tags = next((t for rx, t in patterns if rx.match(key)), None)
        if tags is None or "Node" in key:
            continue
        ja, zh, en = _clean(raw), _clean(zh_map.get(key, "")), _clean_en(en_map.get(key, ""))
        if "enemy-boss" in tags:
            ja, zh, en = (_unquote(_PREFIXED.sub("", x)) for x in (ja, zh, en))
        if "{" in raw:  # a template (「激戦試練：末路」第{0}階)
            continue
        if not ja or not zh or ja.isascii() or ja.startswith("Test-"):  # an English logo (PartnerBg_Camp...) is not a Japanese name
            continue
        if tags[0] in ("location", "enemy-boss", "enemy", "quest", "item") and not name_like(ja):
            continue
        variants = []
        if "character-main" in tags and "short-name" not in tags:
            given = _given_name(ja, zh)
            variants = [given] if given else []
        terms.append({
            "id": f"zzz:{key}", "game": "zzz", "ja": ja, "reading": "", "zh": zh, "zh_alts": [zh], "zh_tw": "",
            "en": en, "tags": list(tags), "note_zh": "", "file": key.split("_")[0],
            "variants": variants,
        })
    lines: collections.Counter = collections.Counter()
    story: collections.Counter = collections.Counter()
    by_text: dict[str, str] = {}  # one key per printed speaker, so the zh and en of the same speaker line up
    for key in ja_map:
        if ZZZ_SPEAKER.match(key):
            first = by_text.setdefault(f"{ja_map[key]}\t{zh_map.get(key, '')}", key)
            lines[first] += 1
            story[first] += key.startswith("Main_")
    return (_attach_short_names(terms) + _concepts(ZZZ_CONCEPTS, ja_map, zh_map, en_map, "zzz")
            + _speakers("zzz", lines, story, ja_map, zh_map, en_map))


def _unquote(text: str) -> str:
    text = text.strip()
    return text[1:-1] if len(text) > 2 and text[0] in "「“\"" and text[-1] in "」”\"" else text


def _attach_short_names(terms: list[dict]) -> list[dict]:
    """The short names the game shows (``猫又``, ``アンビー``) become variants of the agent's full name
    (``猫宮又奈``, ``アンビー・デマラ``) when one contains the other in either language; the rest stay names of
    their own (``0号・アンビー``)."""

    full = [t for t in terms if "character-main" in t["tags"] and "short-name" not in t["tags"]]
    out = []
    for t in terms:
        if "short-name" not in t["tags"]:
            out.append(t)
            continue
        owner = next((f for f in full if f["ja"] == t["ja"]), None) or next(
            (f for f in full if t["ja"] in _parts(f["ja"]) or t["zh"] in _parts(f["zh"])), None)
        if owner is None:
            t["tags"] = [tag for tag in t["tags"] if tag != "short-name"]
            out.append(t)
        elif owner["ja"] != t["ja"]:  # the game's own short name replaces the guessed given name (キング of
            # キング・シーザー is 金, but the game calls him シーザー / 凯撒)
            owner["variants"] = [v for v in owner["variants"] if v.get("given") is None] + [{"ja": t["ja"], "zh": t["zh"]}]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=("genshin", "starrail", "zzz"))
    ap.add_argument("source", type=Path, help="the upstream checkout (genshin, starrail) or TextMap directory (zzz)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--textmap", type=Path, help="starrail: the text tables for factions, Aeons and worlds")
    args = ap.parse_args()
    if args.game == "genshin":
        terms = genshin(args.source)
    elif args.game == "starrail":
        terms = starrail(args.source, args.textmap)
    else:
        terms = zzz(args.source)
    first: dict[tuple[str, str], dict] = {}
    unique = []
    for term in terms:
        key = (term["ja"], term["zh"])
        if key not in first:
            first[key] = term
            unique.append(term)
        else:  # the same name from a later table: keep the first, with the speaker counts a later one has
            kept = first[key]
            for field in ("lines", "story_lines"):
                if term.get(field):
                    kept[field] = max(kept.get(field, 0), term[field])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for term in unique:
            handle.write(json.dumps(term, ensure_ascii=False) + "\n")
    print(f"{args.game}: {len(terms)} terms, {len(unique)} unique (ja, zh) pairs -> {args.out}")


if __name__ == "__main__":
    main()
