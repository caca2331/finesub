"""Build a game glossary file (format 2, read by ``finesub.llm.glossary_store``) for publishing as a Release asset.

    PYTHONPATH=<checkout>/src python tools/game_glossary/build.py <game> \
        --terms <terms.jsonl> --dates <dates.json> [--dates <more.json>] --famous <famous.json> \
        (--background <background.tsv> | --noise-from <earlier glossary.json> | --no-background) \
        [--signals <signals.json>] [--versions <versions.json>] [--readings-from <earlier glossary.json>] \
        --source-commit <upstream commit> --out <glossary.json> [--built YYYY-MM-DD]

Run from the checkout root (the background list's paths are relative to it); needs the ``kb-phonetic`` extra.

Inputs, all made by the maintainer before this step:

- ``--terms``: one JSON object per line, one per official name: ``id``, ``ja``, ``reading`` (kana, may be empty),
  ``zh``, ``zh_alts``, ``en``, ``tags``, ``note_zh``, ``variants``; extracted (``extract.py``) from the upstream named
  in ``GAMES`` (``source``) at ``--source-commit``. Only names go into the glossary, which is why it is a separate
  download that never ships inside a wheel and can be withdrawn if an upstream asks.
- ``--dates``: ``{term id or en name: {"date": "YYYY-MM-DD"}}`` (``dates.py``), when a name entered the upstream
  (or became playable: the latest of the given files' dates wins); a name missing everywhere counts as old.
  Upstreams add a character when it is announced or shipped hidden, which can be well before release (Natlan's cast
  in 2024-07, an NPC years before): ``Game.released`` gives the date of a playable character whose upstream date is
  wrong, and wins over the files: the ones released inside the window the upstream dates before it, and the ones it
  marked playable again long after release (Genshin's 2025-07-08 re-tag of three 2022-23 characters).
- ``--famous``: ``{zh name: right answers}``, a closed-book probe (``probe_pool``: the resident-list pool and the
  important names the short / everyday-word rule leaves without evidence): a model with no tools is asked three times
  for the official Chinese of each Japanese name. ``FAMOUS_PASSES`` right answers make it famous: it stays off the
  resident list, a window does not list it when ASR wrote it right, and it is evidence for the risk rule.
- ``--readings-from`` (optional): an earlier build whose ``kana`` a name that is not all kana keeps (``_carried``;
  ``reading:carried`` where this machine reads it otherwise); readers differ between machines and versions.
- ``--signals`` (optional, more than once): ``{"<term id>|<ja>": {"mentions": n}}`` (``signals.py``; Genshin's
  several Japanese forms of one word share an id), how often the game's
  text mentions a name; with the dialogue lines ``extract.py`` writes, the evidence for an important NPC.
- ``--versions`` (optional): ``{version: [commit, date]}`` of the game's data repository, for the ``version:`` tag.
- ``--background``: ``<label>\\t<raw.srt path>`` per recording that is NOT about the game; its rows give each
  (name, ASR spelling) pair's per-row rate of matching by sound, which the window matcher uses to drop pairs that
  fire on ordinary talk. Only the rates are published, never the rows. ``--noise-from`` instead carries the rates
  of an earlier build of the same game over (the pairs whose name is still in the list), for a rebuild away from the
  recordings. ``--no-background`` publishes no rates (the matcher's thresholds alone decide, so windows get more
  candidates): for a first build away from the recordings, to be rebuilt with ``--background`` before long.

Every extracted name stays in the file with its ``tags``, the evidence for where it lands (owner 2026-11-13):
``kind:`` (what it names, from the upstream's tags through ``Game.kinds``), ``up:`` (the upstream's tags),
``important:<reason>`` (playable, faction, concept, major-place, boss, alias, story-npc), ``added:`` / ``version:`` /
``recent``, ``short-reading``, ``lines:`` / ``story-lines:`` / ``mentions:`` (the story signals), ``famous:n/3``,
``everyday:listed`` / ``everyday:dictionary``, ``noise:`` (the name's background rates summed over the pairs under ``NOISE_LINE``), ``variant-of:``,
``skip:<reason>``; then the rule tags ``override:<tier>:<reason>`` (``Game.tiers``) and ``risk:`` / ``evidence:``
(``risk_tags``: the short / everyday-word rule). ``derive`` turns the tags into the marks the runtime reads, by rules
written there: ``skip``, ``rare`` (kept in the file, never listed or matched), ``exact`` (matched only as written),
``core`` (the resident list's pool), ``famous`` and ``core_skip``; ``apply_rules`` then sets ``fold`` (an exact katakana
name also found in hiragana) and a variant's ``famous``. A later build moves a name by changing a rule or
adding an override; ``retag.py`` reapplies the rule tags to a built file (``apply_rules``).

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
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from finesub.llm import glossary_match as gm
from finesub.llm import glossary_store as gs
from finesub.llm import kb_phonetic_hints as K
from finesub.text import t2s_converter

sys.path.insert(0, str(Path(__file__).parent))
from extract import name_like  # noqa: E402  (the sibling script: what reads as a proper name)

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
GENSHIN_LABELS = (  # tag -> label, first match wins
    ("character-main", "可玩角色"), ("fatui", "愚人众"), ("character-sub", "角色"), ("organization", "组织"),
    ("location", "地点"), ("element", "元素 / 机制"), ("enemy", "敌人"), ("weapon", "武器"), ("artifact", "圣遗物"),
    ("quest", "任务"), ("event", "活动"), ("food", "料理"), ("item", "物品"), ("title", "称号"),
)
STARRAIL_LABELS = (
    ("character-main", "可玩角色"), ("aeon", "星神"), ("path", "命途"), ("element", "属性"), ("faction", "阵营"),
    ("enemy-faction", "阵营"), ("concept", "概念"), ("world", "地点"), ("location", "地点"), ("light-cone", "光锥"),
    ("relic", "遗器"), ("item", "物品"), ("enemy", "敌人"), ("quest", "任务"), ("npc", "角色"),
)
ZZZ_LABELS = (
    ("character-main", "代理人"), ("faction", "阵营"), ("element", "属性"), ("specialty", "特性"),
    ("concept", "概念"), ("bangboo", "邦布"), ("w-engine", "音擎"), ("drive-disc", "驱动盘"), ("location", "地点"),
    ("enemy", "敌人"), ("quest", "委托"), ("item", "物品"), ("npc", "角色"),
)
# Upstream tag -> the kind of thing a term names, first match wins (``kind:`` tags). Every game maps into the same
# kinds, so one set of rules (``_important``, ``_derive``) places every term.
GENSHIN_KINDS = (
    ("character-main", "playable"), ("how-to-call", "address"), ("enemy-boss", "boss"), ("enemy-legend", "boss"),
    ("enemy", "enemy"), ("character-sub", "npc"), ("organization", "faction"), ("fatui", "faction"),
    ("element", "concept"), ("location", "place"), ("domain", "place-minor"), ("facility", "place-minor"),
    ("weapon", "weapon"), ("artifact", "artifact"), ("quest", "quest"), ("event", "event"), ("food", "food"),
    ("title", "title"), ("living-being", "creature"), ("dialogue", "line"),
)
STARRAIL_KINDS = (
    ("character-main", "playable"), ("aeon", "concept"), ("path", "concept"), ("element", "concept"),
    ("enemy-faction", "faction"), ("faction", "faction"), ("concept", "concept"), ("world", "place"),
    ("location", "place"), ("enemy-boss", "boss"), ("enemy", "enemy"), ("npc", "npc"), ("quest", "quest"),
    ("light-cone", "weapon"), ("relic", "artifact"), ("item-food", "food"), ("item", "item"),
)
ZZZ_KINDS = (
    ("character-main", "playable"), ("faction", "faction"), ("element", "concept"), ("specialty", "concept"),
    ("concept", "concept"), ("minor", "place-minor"), ("location", "place"), ("enemy-boss", "boss"),
    ("enemy", "enemy"), ("bangboo", "npc"), ("npc", "npc"), ("quest", "quest"), ("w-engine", "weapon"),
    ("drive-disc", "artifact"), ("item", "item"),
)
# Kind -> the ``important:`` reason it gives by itself. An NPC is important only by the story signals
# (``_story_npc``). A term with any ``important:`` tag is common (a window candidate); the rest are rare.
IMPORTANT_KINDS = {"playable": "playable", "faction": "faction", "concept": "concept", "place": "major-place",
                   "boss": "boss", "address": "alias"}
# The important reasons whose recent or short names make up the resident list's pool (the owner's 2026-11-12 / 13
# rules: playable characters, factions and world concepts).
CORE_REASONS = frozenset({"playable", "faction", "concept"})
EVERYDAY = "日常词"
# The short / everyday-word rule (``risk_tags``, decided 2026-11-20 with the owner's leave, agreed by a peer agent
# the same day with changes; docs/knowledge.md「短名字与普通词同形的名字」). A name with a risk tag needs an evidence
# tag to stay common, and then it is matched only as written (``exact``), never by sound; without evidence it is
# rare. The rule never skips: a noisy name loses only its sound matching, the one source of its background hits.
NOISE_RISK = 2e-4   # summed background rate of a name's false sound hits that makes it a risk: about 5 hits on the
                    # 25119 Genshin background rows (1e-3 would be about 25)
# Kinds whose upstream classification is evidence by itself, even for an everyday word (炎, 銀狼); a boss or a major
# place that is an everyday word (ピーク, 荒海, 幕府) needs other evidence (``risk_tags``).
EVIDENT_REASONS = frozenset({"playable", "faction", "concept"})
EVIDENT_UNLESS_WORD = frozenset({"major-place", "boss"})
RULE_PREFIXES = ("risk:", "evidence:", "override:")  # what ``apply_rules`` rewrites
# Slips of genshin-langdata the game's own text table settles (2026-11-13, by English name: the Japanese or the
# Chinese differs by at most two characters from the pair the game writes for the same English name; dialogue
# lines and table quirks left out; 2026-11-20 took back ten whose text-table side reads as a UI string or an
# abbreviation: 荒 / 芒 for 荒性 / 芒性, 开启供奉, 重投骰子, 伤害值, 好感, 精炼阶, 合成, 《旧都梦夜奇谭》, and
# 非有非无有之笼, which the 2026-11-21 check found the game writing everywhere: it is back in ``fixes``).
# Upstream ja -> fields, like ``Game.fixes``; the hand-checked ones win.
GENSHIN_TEXTMAP_FIXES = json.loads(Path(__file__).with_name("genshin-textmap-fixes.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Game:
    name: str
    aliases: tuple[str, ...]
    intro: str
    source: str                        # the upstream the terms come from (see extract.py)
    labels: tuple[tuple[str, str], ...]  # tag -> label, first match wins
    released: dict[str, str]           # en name -> the date a playable character counts from, over the upstream's
    fixes: dict[str, dict[str, str]]   # upstream ja -> fields to replace (``ja``, ``zh``, ``en``, ``label``): its slips
    drop: dict[str, str]               # upstream ja -> why it is marked ``skip``: it pairs two things, carries no
                                       # usable name, or is an address form that is an everyday word (相棒)
    ordinary: frozenset[str]           # core names that are also everyday words with another sense (無効, "invalid",
                                       # is the 免疫 mechanic): marked ``core_skip``, off the resident list, which
                                       # every window carries; still window candidates where a row writes them
    kinds: tuple[tuple[str, str], ...] = ()  # upstream tag -> kind (``kind:`` tag), first match wins
    tiers: dict[str, tuple[str, str]] = field(default_factory=dict)  # ja, or "ja|kind" for one of two rows with the
                                       # same ja -> ("common" | "rare" | "sound", reason):
                                       # a per-term override of the derived tier, or ``sound`` to let a risky name be
                                       # matched by sound again (an ``override:`` tag)
    risk_rule: bool = True             # the short / everyday-word rule (``risk_tags``); off writes no risk: tags
    simplify: bool = False             # the upstream writes some Chinese in traditional characters: convert it.
                                       # Off where the Chinese is the game's own table, which is simplified already
                                       # and keeps the odd official character (星铁 迴星港, 工造迴轮)
    story_lines: int = 30              # an NPC with this many dialogue lines (where the tables give speakers) ...
    story_mentions: int = 60           # ... or this many mentions in the game's text is an important story NPC
    variant_keep: frozenset[str] = frozenset()  # variants kept although the dictionary knows them as a word
    extra_variants: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)  # upstream ja -> (ja, zh)
                                       # short names the upstream lacks


GAMES = {
    "genshin": Game(
        name="原神",
        aliases=("Genshin Impact", "Genshin", "げんしん", "げんしんインパクト"),
        source="https://github.com/xicri/genshin-langdata",
        intro="原神官方日中英专名表（只读）。录音讲原神时点名它：纠错窗口会自动附上常驻名单、本片反复出现的名字与本窗的原样 / 音近候选",
        labels=GENSHIN_LABELS,
        kinds=GENSHIN_KINDS,
        # No speakers in the upstream or the local text table, so mentions alone: 40 keeps the lore names the story
        # keeps returning to (狐斎宮 56, 花散里 57, 帰終 73) and leaves the one-quest NPCs (about 280 NPCs pass).
        story_mentions=40,
        simplify=True,
        # Address forms the risk rule would leave without evidence (an everyday word, no line counts in Genshin)
        # that the story uses as the fixed name of one character (2026-11-20).
        # Nations and Harbingers whose names are everyday words, which the risk rule would leave without evidence
        # (a major place or a boss needs more than its kind when it is an everyday word).
        tiers={"旅人": ("common", "主角的固定称呼，译作旅行者"), "ダイン": ("common", "戴因斯雷布的通称"),
               "稲妻": ("common", "七国之一"), "ナタ": ("common", "七国之一"),
               "淑女": ("common", "愚人众执行官的称号，主线首领"), "シニョーラ": ("common", "愚人众执行官，主线首领"),
               "博士": ("common", "愚人众执行官的称号"),
               # Full names no line or book of the 7.2 text writes (textmap_check.py, 2026-11-21): what the correction
               # model is given has to be the game's own string, so they stay in the file, unlisted. A variant takes
               # its name's override, so the given names the game does write are listed again by their own.
               "ディオナ・カッツェレイン": ("rare", "游戏文本里没有这个全名"),
               "バーバラ・ペイジ": ("rare", "游戏文本里没有这个全名"), "バーバラ・ペッチ": ("rare", "游戏文本里没有这个全名"),
               "ミカ・シュミット": ("rare", "游戏文本里没有这个全名"),
               "ディオナ": ("common", "游戏里写的名字（全名不在文本里）"), "バーバラ": ("common", "游戏里写的名字（全名不在文本里）"),
               "ミカ": ("common", "游戏里写的名字（全名不在文本里）"), "シュミット": ("common", "游戏里写的姓（全名不在文本里）")},
        # genshin-langdata marked Dori, Diona and Mika playable again on 2025-07-08, which made them recent and put
        # Dori on the resident list: they count from the day the upstream added them (all three shipped in 2020-23).
        released={"Chasca": "2024-11-20", "Ororon": "2024-11-20", "Citlali": "2025-01-01", "Mavuika": "2025-01-01",
                  "Iansan": "2025-03-26",
                  "Dori Sangemah Bay": "2022-09-04", "Diona Kätzlein": "2023-04-09", "Mika Schmidt": "2023-03-25"},
        ordinary=frozenset({"無効", "開花", "運輸局"}),
        # From a reader's pass (2026-11-13), kept where the slip is plain: a typo, a dropped character, a swapped
        # kana, a series written two ways. Checked against the game's own text table (2026-11-13, AnimeGameData
        # TextMap, read locally only; textmap_check.py): the ones marked TM are what the game writes. Rechecked
        # 2026-11-20 on animegamedata2@a311cd77 (7.2): each fixed pair is one entry of the table, punctuation width
        # aside, but 染血骑士 (quoted inside longer strings) and デルポイ・ピュートーン (its line is gone in 7.2); the
        # game writes 核心轮縠, as the upstream does (a fix to 毂 was dropped).
        fixes={
            **GENSHIN_TEXTMAP_FIXES,
            # The five the 2026-11-13 reader's pass left to the upstream, settled by the text table: the quest
            # 「血染めの騎士」と対峙する is 与「染血骑士」对峙; the line じゃあデルポイ・ピュートーンは？ is
            # 德尔斐·皮托呢？ (呢 is the question particle); the weapon is 烏髄の孤灯 / 乌髓孑灯; the mountain is
            # カピトリヌス山 (in the table file); the line is 炎食いの刑！ / 食炎之罚！.
            "血染めの騎士": {"zh": "染血骑士"}, "デルポイ・ピュートーン": {"zh": "德尔斐·皮托", "en": "Delphi Pytho"},
            "鳥髄の狐灯": {"ja": "烏髄の孤灯", "zh": "乌髓孑灯"}, "炎食いの刑!": {"ja": "炎食いの刑！", "zh": "食炎之罚！"},
            "スペランザ": {"zh": "斯佩兰扎"}, "曚雲神社": {"zh": "曚云神社"}, "リューバ": {"zh": "利乌芭"},
            "最勝のアメシスト・欠片": {"zh": "最胜紫晶断片"}, "最勝のアメシスト・砕屑": {"zh": "最胜紫晶碎屑"},
            "護法仙衆夜叉録": {"zh": "护法仙众夜叉录"},
            "賢医の仮面": {"zh": "贤医的假面"}, "ライトキーバー": {"ja": "ライトキーパー", "en": "Lightkeepers"},
            "パティサラ": {"label": ""}, "征討領域": {"label": ""}, "アルケウム": {"label": "物品"},
            "イノシシプリンセス": {"label": "物品"},
            # The text-table leftovers (textmap_check.py, more than two characters off or both sides differing),
            # read one by one on 2026-11-20 and kept where the game's own pair for the same English name is plainly
            # the current spelling (a typo, a dropped word, a renamed NPC whose upstream entry kept the old names).
            # Left alone: dialogue lines, the table's quirks (an English zh, a 测试 string, the # gender
            # placeholder), and names the game writes two ways (アムリタ学院 / 生論派, 雷コレクター).
            "茲白": {"ja": "兹白"}, "リュドヒカ・シュナイツェフナ": {"ja": "リュドヒカ・スネージヴナ"},
            "サイモン・ペイジ": {"ja": "サイモン・ペッチ"}, "ケーナウ": {"ja": "クーナウ", "zh": "昆纳乌"},
            "オロラク": {"ja": "コヌラフ", "zh": "科奴拉芙"}, "ジェリック・ジェストニン": {"ja": "ジェリック・シェストニン"},
            "時間の執政": {"ja": "時の執政", "zh": "时之执政"}, "フレイムグレネード・爛熱": {"ja": "フレイムグレネード・爛熟"},
            "野伏・火付番": {"zh": "野伏·火付番"}, "霊覚の勇士・瞑思師": {"ja": "霊覚の勇士・冥思師"},
            "クク竜戦士・鍛治の槌": {"ja": "クク竜戦士・鍛冶の槌"}, "プレイズ・ミートシチュー": {"ja": "ブレイズ・ミートシチュー"},
            "最高の知恵・生活": {"zh": "至高的智慧（生活）"}, "怪盗レッド・ミラーの伝説": {"zh": "大盗雷德·米勒传奇"},
            "誰かのボロボロなノート・1": {"zh": "谁人的残破笔记（一）"}, "誰かのボロボロなノート・2": {"zh": "谁人的残破笔记（二）"},
            "腐食したボロボロなノート・3": {"ja": "腐蝕したボロボロのノート・3", "zh": "被腐蚀的残破笔记（三）"},
            "キラキラエッグ・聖龍降臨": {"ja": "キラキラエッグ・聖龍君臨"}, "荊夫港": {"ja": "ドーンマンポート"},
            "ホルトゥス・ユーゲレティス": {"ja": "ホルトゥス・ユーレゲティス"}, "カキシュの塔": {"ja": "ヴクブ・カキシュの塔"},
            "集いの場・魔女の楽園": {"ja": "集いの場・魔女の花園"},
            "或る人への薔薇〜荊のような刺々しさ〜": {"ja": "或る人への薔薇～荊のような刺々しさ～"},
            "或る人への薔薇〜長い一日の過ごし方〜": {"ja": "或る人への薔薇～長い一日の過ごし方～"},
            "翼の折れた猟鷹": {"zh": "折翼的猎鹰"}, "ハラハラ潜入捜査": {"zh": "隐行谛勘，秽金之罪"},
            "月の東、太陽の西": {"zh": "月之东，日之西"}, "月のそばの小さな画家・２": {"ja": "月のそばの小さな画家・2"},
            "遺跡を探検? 知行合一プロジェクト!": {"zh": "参究故墟·稳健知行"},
            "西の彼方の宝物・満ちゆく月": {"ja": "虹の彼方の宝物・満ちゆく月"},
            "シーッ！ 潜入ロボット！": {"zh": "静音！谍械潜行！"}, "のんびりナタののびのびサマー!": {"zh": "悠悠纳塔，悠悠夏天！"},
            "彩る紙鳶、春梳きて": {"ja": "春立つ風を梳かす彩鳶"}, "騎士が血に染めた時": {"ja": "騎士が血に染まりし時"},
            # Listed names textmap_check.py found the game writing otherwise (2026-11-21, animegamedata2 7.2 with
            # its Readable books; each confirmed by agy Gemini 4 Hydrogen on the cited key, and the key reread):
            # the English the aligned lines use, the ja the game writes (永結, 機巧, 霜, スネージ-), the zh at the
            # same key. 非有非無の檻 is 非有非无有之笼 under all three keys; 何爺 is 老何 under all five.
            "シニョーラ": {"en": "Signora"}, "アームスヴァルトニル湖": {"en": "Amsvartnir"},
            "エスコフィエ": {"en": "Escoffier"}, "ジャック": {"en": "Jack"},  # 杰克; 雅克 (Jacques) shares the ja
            "スチームバード新聞社": {"en": "The Steambird"}, "ティティ島": {"en": "Tete Isle"},
            "テケメカンの谷": {"en": "Tequemecan Valley"}, "パイン峠": {"en": "Guiztli Ridge"},
            "パイン町": {"en": "Villa Guiztli"}, "ヤヌスの門": {"en": "Initium Iani"},
            "千柱中央エリア": {"en": "Pillar Hall Central Zone"}, "小さなロボット": {"en": "Little Robot"},
            "星岩の墜落地": {"en": "Fallingstar Fields"}, "星電導": {"en": "Stellar-Conduct"},
            "月感電": {"en": "Lunar-Charged"}, "祝福の森": {"en": "Forest of Blessings"},
            "秘源機兵・統御デバイス": {"en": "Secret Source Automaton: Overseer Device"}, "結晶": {"en": "Crystallize"},
            "ワルカシュナ": {"en": "Valuka Shuna"}, "偉大なる聖龍": {"en": "Almighty Dragonlord"},
            "断片領域タウ": {"en": "Segment-Domain Tau"}, "月の三女神": {"en": "Three Moon Sisters"},
            "第八席": {"en": "Eighth Lord"}, "第十一席": {"en": "Eleventh Lord"},
            "氷結の凍土": {"ja": "永結の凍土"}, "雪の精の村": {"ja": "霜の精の村"},
            "秘源機兵・精巧デバイス": {"ja": "秘源機兵・機巧デバイス", "en": "Secret Source Automaton: Configuration Device"},
            "シュナイツェビッチ": {"ja": "スネージヴィッチ"}, "シュナイツェフナ": {"ja": "スネージヴナ"},
            "ラグヴィンド家": {"zh": "莱艮芬德家"}, "栄誉騎士のお兄ちゃん": {"zh": "荣誉骑士大哥哥"},
            "栄誉騎士のお姉ちゃん": {"zh": "荣誉骑士大姐姐"}, "何爺": {"zh": "老何"}, "非有非無の檻": {"zh": "非有非无有之笼"},
            "カピバラ キング!": {"ja": "カピバラキング", "zh": "豚兽大王", "en": "Capybara King"},
            # The same check made stricter on the peer's review (2026-11-21: every value as written, each " / "
            # alternative, the name's own key over a line that spells it otherwise): the English at the name's own
            # key (Lunar-Crystallize as Lunar-Charged above; 游侠 = Rebel is another レンジャー), quotes the game
            # writes only around the name in a line, an unclosed 「 from the upstream, and the ship rather than the
            # quest label at its ja's own key (「死兆星」二号舰领航 / The Alcor 2 — Lead Ship).
            "邪知深き狂者": {"en": "Scamface"}, "月結晶": {"en": "Lunar-Crystallize"}, "月開花": {"en": "Lunar-Bloom"},
            "サロンメンバー": {"en": "Salon Members"}, "レンジャー": {"zh": "巡林员", "en": "Forest Ranger"},
            "璃月七星": {"en": "Liyue Qixing"}, "ヴァインショット": {"en": "Sprawling Shots"},
            "虹雨の祭庭": {"en": "Temple of Cascading Colors"}, "南十字船隊": {"en": "The Crux"},
            "淑女": {"en": "La Signora"}, "月の少女": {"en": "Moon Maiden"}, "無風の地": {"en": "Windless Land"},
            "故郷を偲ぶ孤独の狼": {"zh": "望乡的孤狼"}, "燼寂海": {"en": "Mare Jivari"},
            "「死兆星号」級二番艦": {"en": "Alcor-class No. 2"},
        },
        drop={"無妄の翁": "一条写着两样东西", "ビュンビュンスパイ1号": "一条写着两样东西", "相棒": "日常称呼",
              "お嬢様": "日常称呼", "お嬢": "日常称呼", "可愛い子ちゃん": "日常称呼"},
    ),
    "starrail": Game(
        name="崩坏：星穹铁道",
        aliases=("Honkai: Star Rail", "Honkai Star Rail", "Star Rail", "星穹铁道", "崩坏星穹铁道", "星铁",
                 "崩壊：スターレイル", "崩壊スターレイル", "スターレイル", "スタレ"),
        source="https://github.com/Mar-7th/StarRailRes (+ Dimbreath/turnbasedgamedata text tables)",
        intro="崩坏：星穹铁道官方日中英专名表（只读）。录音讲星穹铁道时点名它：纠错窗口会自动附上常驻名单、本片反复出现的名字与本窗的原样 / 音近候选",
        labels=STARRAIL_LABELS,
        kinds=STARRAIL_KINDS,
        # TalkSentenceConfig counts the overworld chatter too: a Xianzhou stall keeper says 30-50 lines and is never
        # mentioned, so a speaker needs 100 lines (about 180 do) or 60 mentions.
        story_lines=100,
        # Playable characters, Aeons, Paths, combat types, factions (the playable ones and the enemy ones) and world
        # terms (星神, 令使, 开拓者): the owner's 2026-11-13 list. Light cones, relics, materials and places are
        # window candidates only.
        released={},
        ordinary=frozenset({"開拓", "運命", "銀河", "異世界", "物理", "炎", "氷", "雷", "風", "量子", "虚数",
                            "記憶", "知恵", "調和", "存護", "豊穣", "愉悦", "壊滅", "虚無", "巡狩", "タイタン"}),
        # From the 2026-11-20 reader's pass (DeepSeek), checked against the text table: the game's English for 命途
        # is Path; ？？？ names a place not yet revealed; speaker labels that are a job, a generic guard or an enemy
        # unit, though the text mentions the word often (the faction シルバーメイン keeps its own row). Left as they
        # are, the text table or the upstream agreeing: 仙舟「玉殿」, パルメニデ and イレーネ = 伊来内 = Alain are what
        # the game writes; 星神★アッハ is a playable character (4.7), 模擬宇宙 an enemy faction, 銀河 and 異世界
        # character factions (matched only as written, off the resident list), 開拓者 a world term.
        # The 2026-11-21 text-table check: the English the enemy's own key writes ((Complete) is the スペック form).
        fixes={"運命": {"en": "Path"}, "スウォーム・真蟄虫": {"en": "Swarm: True Sting"}},
        drop={"その他": "不是名字", "？？？駅": "占位名，不是名字", "？？？": "占位名，不是名字"},
        tiers={"シルバーメイン|npc": ("rare", "说话人标签：泛指卫兵"), "ドリームメーカー": ("rare", "职业称呼"),
               "配信者": ("rare", "职业称呼"), "狼兵": ("rare", "敌人兵种，不是人名"),
               "エリュシオン": ("common", "主线大地点，文本里提到近 300 次"),
               "トパーズ|boss": ("rare", "托帕幻象：首领形态，托帕本人另有一行"),
               # Rare by their upstream kind, though the story keeps them (2026-11-21, agy Hydrogen on the reader's
               # comments): ダン is filed as an elite enemy; ディエス・ドミニ's listed row is its boss form.
               "ダン|enemy": ("common", "银鬃铁卫的剧情人物，台词 183 行，上游只收作精英敌人"),
               "ディエス・ドミニ|npc": ("common", "匹诺康尼主线首领的本名，列出的那行是首领形态")},
    ),
    "zzz": Game(
        name="绝区零",
        aliases=("Zenless Zone Zero", "ZZZ", "ゼンレスゾーンゼロ", "ゼンゼロ", "绝区零"),
        source="https://git.mero.moe/dimbreath/ZenlessData (TextMap)",
        intro="绝区零官方日中英专名表（只读）。录音讲绝区零时点名它：纠错窗口会自动附上常驻名单、本片反复出现的名字与本窗的原样 / 音近候选",
        labels=ZZZ_LABELS,
        kinds=ZZZ_KINDS,
        # A speaker label that is a job or a placeholder, not a person's name, though it talks often.
        tiers={"治安官": ("rare", "职务称呼，不是人名"), "助手二号": ("rare", "临时代称"),
               "知能構造体": ("rare", "说话人标签，不是人名"), "傍白": ("rare", "说话人标签：旁白"),
               "暗転": ("rare", "说话人标签：黑屏"), "コック長": ("rare", "职务称呼"),
               "治安局": ("common", "新艾利都治安局的通称，文本里提到 1737 次")},
        # Agents, factions, attributes, specialties and world terms (代理人, 空洞, 以骸): the owner's 2026-11-13 list.
        # Agents the text table names a version or more before they were playable (banner dates, checked 2026-11-13).
        released={"Lighter": "2024-11-27", "Hoshimi Miyabi": "2024-12-18", "Asaba Harumasa": "2024-12-18"},
        ordinary=frozenset({"電気", "炎", "氷", "物理", "風", "強攻", "撃破", "異常", "支援", "防護",
                            "エージェント", "侵蝕"}),
        # From the 2026-11-20 reader's pass (DeepSeek), checked against the text table: the zh the game writes
        # for these ja almost everywhere (危局强袭战 12 to 1, 幕间余韵 26 to 1); a demo entrance, a placeholder
        # boss and a menu option (找零) are no names; HIAセンター, a speaker label, is the club.
        fixes={"危局強襲戦": {"zh": "危局强袭战"}, "幕間の余韻": {"zh": "幕间余韵"}, "HIAセンター": {"label": "地点"},
               # The 2026-11-21 text-table check: the boss's own name key (a quest title spells her 特莉薇娅 once;
               # the game writes 特里薇亚 / Truvia 237 times).
               "「エーテル特異体」トリヴィア": {"ja": "エーテル特異体・トリヴィア", "zh": "以太异体·特里薇亚",
                                          "en": "Etheric Other - Truvia"}},
        drop={"なし": "不是名字", "レミエール デモ入口": "演示入口，不是名字", "章ボス-新規人形ボス": "占位名，不是名字",
              "オツリ": "界面选项，不是人名"},
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


def _carried(ja: str, earlier: dict[str, str]) -> str:
    """The kana an earlier build gave a name that is not all kana, or "": readers differ between machines and
    versions (2026-11-20: this machine's pyopenjtalk-plus reads 六分街 ロクブガイ, 若陀竜王 ワカオー, 雲璃 クモ
    where the earlier build read ロップンガイ, ワカダリューオー, クモリ), so a rebuild keeps the readings the
    published file and its reader's pass had, and the local reader reads only the names new since."""

    return "" if _is_kana(ja) else earlier.get(ja, "")  # a name it could not read either is read here


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


def _simplified(text: str, on: bool) -> str:
    """``text`` in simplified characters when ``on`` (``Game.simplify``): the upstream writes a few names and notes
    in traditional ones. Japanese a note quotes (「鍾離さん」, told by its kana) stays as written."""

    if not on:
        return text
    cc = t2s_converter()
    parts = _QUOTED_JAPANESE.split(text)  # odd items are the quotes
    return "".join(part if i % 2 else "".join(ch if ch in T2S_KEEP else cc.convert(ch)
                                              for ch in part.translate(SHINJITAI))
                   for i, part in enumerate(parts))


def _not_a_name(ja: str, zh: str) -> bool:
    """An upstream row that names nothing a transcript would write: a bracketed description (``(ガチャで) 引く``), a
    dish template (``美味しそうな〇〇``), a demonym's fragment (``稲妻の`` = 稻妻人)."""

    return ja[:1] in "(（" or "〇〇" in ja or (ja.endswith("の") and zh.endswith("人"))


def _note(text: str, simplify: bool) -> str:
    """The whole sentences of a note that fit in ``NOTE_CHARS`` (none when the first does not): a note cut mid-sentence
    reads as a different claim."""

    text = _simplified(text, simplify).replace("|", "/").replace("\n", " ").strip()
    kept = ""
    for sentence in re.findall(r"[^。！？]+[。！？]?", text):
        if len(kept + sentence) > NOTE_CHARS:
            break
        kept += sentence
    return kept.strip()


def _label(game: Game, tags: set[str]) -> str:
    return next((label for key, label in game.labels if any(t == key or t.startswith(key) for t in tags)), "")


_JOINED_OOV = 0x0FFFFFFF  # the word id sudachipy 0.6.10 gives a katakana run it joined into one, with is_oov() False


def _in_dictionary(m) -> bool:
    """Whether a Sudachi morpheme is one dictionary word: not OOV, nor a katakana run Sudachi joined into one morpheme,
    which some sudachipy versions report as not OOV: unknown katakana (サンドローネ, チャスカ) or known words written
    together (ホイールコア)."""

    return not m.is_oov() and (m.word_id() & 0x0FFFFFFF) != _JOINED_OOV


def _everyday(word: str) -> bool:
    """Whether the dictionary knows ``word`` as one ordinary word (王冠, 末路, スターライト), not a proper noun: such a
    variant written verbatim is mostly not the game's name."""

    from finesub.llm.glossary_match import _sudachi_tok
    from sudachipy import tokenizer

    morphemes = _sudachi_tok().tokenize(word, tokenizer.Tokenizer.SplitMode.C)
    if len(morphemes) != 1 or not _in_dictionary(morphemes[0]):
        return False
    pos = morphemes[0].part_of_speech()
    return pos[0] in ("名詞", "動詞", "形容詞", "副詞") and pos[1] != "固有名詞"


def _hiragana_word(ja: str) -> bool:
    """Whether the hiragana spelling of a katakana name is one word the dictionary knows, not a proper noun
    (サンポ: さんぽ 散歩, ボッチ: ぼっち, コリン: こりん): such a name is not folded (``fold``), its hiragana is the word."""

    from finesub.llm.glossary_match import _sudachi_tok
    from sudachipy import tokenizer

    hira = "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in ja)
    morphemes = _sudachi_tok().tokenize(hira, tokenizer.Tokenizer.SplitMode.C)
    return len(morphemes) == 1 and _in_dictionary(morphemes[0]) and morphemes[0].part_of_speech()[1] != "固有名詞"


def _ordinary_words(word: str) -> bool:
    """Whether ``word`` is built only of ordinary dictionary words (メンテナンスエンジニア, ベテラン鉱車運転手): a role a
    speaker label names, unless the game's text calls someone by it."""

    from finesub.llm.glossary_match import _sudachi_tok
    from sudachipy import tokenizer

    morphemes = _sudachi_tok().tokenize(word, tokenizer.Tokenizer.SplitMode.C)
    return bool(morphemes) and all(_in_dictionary(m) and m.part_of_speech()[1] != "固有名詞"
                                   and m.part_of_speech()[0] in ("名詞", "接頭辞", "接尾辞", "補助記号") for m in morphemes)


def _variant_skip(game: Game, ja: str, person: bool) -> str:
    """The mark of a short form: one character is too short to look for; a word the dictionary knows (悠久 for
    悠久の磐岩) is everyday speech, except a person's short name (ジン, ニコ, カリン), which people say as a name."""

    if len(ja) < 2:
        return "太短"
    if not person and ja not in game.variant_keep and _everyday(ja):
        return EVERYDAY
    return ""


def _groups(raw: list[dict], game: Game) -> dict[str, str]:
    """Chinese name -> its entity key: names some term lists together (女士 / 席诺拉) are one entity. A fixed zh
    stands in for the upstream's (the upstream calls two books 谁人的残破笔记; fixed, they are （一） and （二））."""

    group: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for term in raw:
        upstream, fixed = term["zh"].strip(), game.fixes.get(term["ja"].strip(), {}).get("zh")
        names = [fixed if fixed and n.strip() == upstream else _simplified(n, game.simplify)
                 for n in term.get("zh_alts") or [upstream]]
        if fixed and fixed not in names:
            names.insert(0, fixed)
        names = [n for n in names if len(n) >= 2]
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


def _dates(files: list[Path]) -> dict[str, str]:
    """Term id or en name -> the latest date any of ``files`` gives it."""

    out: dict[str, str] = {}
    for f in files:
        for key, value in json.loads(f.read_text(encoding="utf-8")).items():
            when = (value or {}).get("date") or "" if isinstance(value, dict) else str(value or "")
            out[key] = max(out.get(key, ""), when)
    return out


def _noise(terms: list[dict], background: Path | None, noise_from: Path | None) -> tuple[dict, int]:
    """(pair -> background rate, rows read): measured on ``background``, or carried over from an earlier build."""

    if background is None:
        if noise_from is None:
            return {}, 0
        old = json.loads(noise_from.read_text(encoding="utf-8"))
        names = {t["ja"] for t in terms if t["kana"] and not t["skip"] and not t["rare"]}
        noise = {pair: rate for pair, rate in sorted(old.get("noise", {}).items()) if pair.split("|", 1)[0] in names}
        return noise, int(old.get("provenance", {}).get("background_rows", 0))
    rows = []
    for line in background.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows += _srt_rows(Path(line.split("\t", 1)[1].strip()))
    matcher = gm.SoundMatcher(gm.Term(t["ja"], t["kana"], t["reading"]) for t in terms
                              if t["kana"] and not t["skip"] and not t["rare"])
    per_text = collections.Counter(rows)  # match() keys by distinct text; a repeated row counts each time
    fired: collections.Counter = collections.Counter()
    for text, hits in matcher.match(rows).items():
        for pair in {f"{h.ja}|{h.surface}" for h in hits}:
            fired[pair] += per_text[text]
    return {pair: float(f"{n / len(rows):.3g}") for pair, n in sorted(fired.items())}, len(rows)


def _kind(game: Game, tags: set[str]) -> str:
    kind = next((k for key, k in game.kinds if any(t.startswith(key) for t in tags)), "")
    return kind or ("item" if tags else "other")


ROLE_MENTIONS = 5


def _story_npc(game: Game, ja: str, lines: int, mentions: int) -> bool:
    """An NPC the story makes a name people say: a proper name (not a description, not one character), with enough
    dialogue lines or mentions in the game's text (``Game.story_lines``, ``Game.story_mentions``). A name the
    dictionary knows as a word (スコート, シッポ, 将軍) needs the lines: counting a word's mentions counts the word. A
    speaker label made of ordinary words that the rest of the text hardly says (オークショニア, 1 mention) is a role
    with lines, not a name (``ROLE_MENTIONS``)."""

    if len(ja) < 2 or not name_like(ja, person=True):
        return False
    if mentions < ROLE_MENTIONS and _ordinary_words(ja):
        return False
    if _everyday(ja):
        return lines >= game.story_lines
    return lines >= game.story_lines or mentions >= game.story_mentions


def _versions(files: list[Path]) -> list[tuple[str, str]]:
    """(start date, version) of each game version, oldest first, from ``{version: [commit, date]}`` files."""

    out = set()
    for f in files:
        for version, (_, when) in json.loads(f.read_text(encoding="utf-8")).items():
            out.add((when, version))
    return sorted(out)


def _version_tag(versions: list[tuple[str, str]], when: str) -> str:
    """``version:4.7`` for the version a date falls in; ``version:≤2.6`` when it is the first version the data
    covers (the name may be older)."""

    if not versions or not when or when < versions[0][0]:
        return ""
    v = [ver for start, ver in versions if start <= when][-1]
    return f"version:≤{v}" if v == versions[0][1] else f"version:{v}"


def derive(tags: list[str] | tuple[str, ...]) -> dict:
    """The marks a term's tags give it; the rules, in order:

    - ``skip`` is the reason of a ``skip:`` tag (not a name, an everyday address form, a variant too short);
    - ``rare`` unless an ``important:`` tag makes it common; a name with a ``risk:`` tag (an everyday word, a short
      reading, a high background rate) also needs an ``evidence:`` tag; an ``override:common:`` /
      ``override:rare:`` tag wins;
    - ``exact``: common with a ``risk:`` tag: matched only as written, never by sound, unless ``override:sound:``;
    - ``core`` (the resident list's pool): not skipped, not rare, not a variant (``apply_rules`` lets a short one
      stand in for a name off the pool), an ``important:`` reason in
      ``CORE_REASONS``, and ``recent`` (added within the window) or ``short-reading``;
    - ``famous``: core and ``famous:3/3`` (``FAMOUS_PASSES`` right answers);
    - ``core_skip``: core and ``everyday:listed`` (a hand-listed everyday word, ``Game.ordinary``).
    """

    tagset = set(tags)
    skip = next((t.split(":", 1)[1] for t in tags if t.startswith("skip:")), "")
    overrides = {t.split(":")[1] for t in tags if t.startswith("override:")}
    risky = any(t.startswith("risk:") for t in tags)
    proven = not risky or any(t.startswith("evidence:") for t in tags)
    common = "common" in overrides or ("rare" not in overrides and proven
                                       and any(t.startswith("important:") for t in tags))
    core = (not skip and common and not any(t.startswith("variant-of:") for t in tags)
            and any(f"important:{r}" in tagset for r in CORE_REASONS)
            and ("recent" in tagset or "short-reading" in tagset))
    famous = core and f"famous:{FAMOUS_PASSES}/{FAMOUS_PASSES}" in tagset
    core_skip = EVERYDAY if core and "everyday:listed" in tagset else ""
    exact = common and not skip and risky and "sound" not in overrides
    return {"core": core, "famous": famous, "skip": skip, "core_skip": core_skip, "rare": not common, "exact": exact}


def _number(tags, key: str) -> float:
    return next((float(t.split(":", 1)[1]) for t in tags if t.startswith(key + ":")), 0)


def risk_tags(game: Game, tags: list[str], full_name: str = "") -> list[str]:
    """The ``risk:`` and ``evidence:`` tags the short / everyday-word rule reads from a term's fact tags.

    Risks: ``risk:common-word`` (``everyday:dictionary`` / ``everyday:listed``: the word is said for its own sense),
    ``risk:short-reading`` (too short for sound matching to tell apart) and ``risk:noise`` (``noise:`` at least
    ``NOISE_RISK``; ``noise:`` sums only the pairs under ``glossary_match.NOISE_LINE``, as the matcher drops the
    others by itself). Evidence that the game's name is meant:
    ``evidence:lines`` (``Game.story_lines`` dialogue lines), ``evidence:mentions`` (``Game.story_mentions``
    mentions; not for an everyday word, whose mentions count the word), ``evidence:full-name:<ja>`` (a longer name of
    the same entity holds it: 心海 in 珊瑚宮心海), ``evidence:famous`` (``famous:3/3``) and ``evidence:kind:<reason>``
    (the upstream lists it as a playable character, a faction or a world concept, ``EVIDENT_REASONS``: the game's
    name by the upstream's own classification, 炎 and 銀狼 included; what stays open is only whether a row means it,
    which ``exact`` answers; a major place or a boss counts only when it is not an everyday word,
    ``EVIDENT_UNLESS_WORD``). So the names that can turn rare are story NPCs, address forms, and bosses or places
    that are everyday words, with no other evidence (お嬢様, ピーク, 荒海). Variants carry no line or mention counts
    of their own, nor their name's probe score (it asks the name as written), only its ``important:``: a risky
    variant has ``evidence:kind:`` when its name is playable, a faction or a concept, ``evidence:full-name`` when its
    name holds it, or none (かざり, an address form of 花散里 which does not hold it)."""

    if not game.risk_rule:
        return []
    tagset = set(tags)
    common_word = "everyday:dictionary" in tagset or "everyday:listed" in tagset
    noise = _number(tags, "noise")
    risks = (["risk:common-word"] * common_word + ["risk:short-reading"] * ("short-reading" in tagset)
             + ["risk:noise"] * (noise >= NOISE_RISK))
    if not risks:
        return []
    evidence = (["evidence:lines"] * (_number(tags, "lines") >= game.story_lines)
                + ["evidence:mentions"] * (not common_word and _number(tags, "mentions") >= game.story_mentions)
                + [f"evidence:full-name:{full_name}"] * bool(full_name)
                + ["evidence:famous"] * (f"famous:{FAMOUS_PASSES}/{FAMOUS_PASSES}" in tagset)
                + [f"evidence:kind:{r}" for r in sorted(EVIDENT_REASONS | (set() if common_word else EVIDENT_UNLESS_WORD))
                   if f"important:{r}" in tagset])
    return risks + evidence


def override_tags(game: Game, ja: str, facts: list[str]) -> list[str]:
    """The ``override:`` tag ``Game.tiers`` gives a term: by ``ja|kind`` (one of two rows sharing a ja), then by
    ``ja``; a variant (``variant-of:``) takes its name's the same way."""

    of = next((x[len("variant-of:"):] for x in facts if x.startswith("variant-of:")), "")
    kind = next((x[len("kind:"):] for x in facts if x.startswith("kind:")), "")
    tier = next((game.tiers[k] for k in (f"{ja}|{kind}", ja, f"{of}|{kind}", of) if k and k in game.tiers), None)
    return [f"override:{tier[0]}:{tier[1]}"] if tier else []


def apply_rules(game: Game, terms: list[dict]) -> None:
    """Rewrite every term's rule tags (``risk:``, ``evidence:``, ``override:``) from its fact tags, ``Game.tiers``
    and the rule constants, then its marks (``derive``). The build ends with it, and ``retag.py`` reruns it on a
    built file when a rule or an override changes. A variant takes its name's overrides and famous mark, and a
    short one stands for its name on the resident list's pool (``stands_in``)."""

    names = collections.defaultdict(list)  # entity -> its names
    for t in terms:
        names[t["group"]].append(t["ja"])
    # a variant's name, by its ja and entity: two rows may share a ja (トパーズ the character and the boss form)
    named = {(t["ja"], t["group"]): t for t in terms if not any(x.startswith("variant-of:") for x in t["tags"])}
    for t in terms:
        facts = [x for x in t["tags"] if not x.startswith(RULE_PREFIXES)]
        of = next((x[len("variant-of:"):] for x in facts if x.startswith("variant-of:")), "")
        full = next((n for n in [of, *names[t["group"]]] if len(n) > len(t["ja"]) and t["ja"] in n), "")
        t["tags"] = facts + override_tags(game, t["ja"], facts) + risk_tags(game, facts, full)
        t.update(derive(t["tags"]))
        # an exact katakana name that is no everyday word is also found written in hiragana (ろーえん)
        t["fold"] = (t["exact"] and "risk:common-word" not in t["tags"] and len(t["ja"]) >= gm.FOLD_MIN
                     and all("ァ" <= ch <= "ヺ" or ch == "ー" for ch in t["ja"]))
        if t["fold"] and _hiragana_word(t["ja"]):  # but not when its hiragana is a word (さんぽ)
            t["fold"] = False
            t["tags"].append("risk:hiragana-word")
        t["tags"] = t.pop("tags")  # last, after the marks
    standing = set()
    for t in terms:  # famous when the name is
        of = next((x[len("variant-of:"):] for x in t["tags"] if x.startswith("variant-of:")), "")
        if (name := named.get((of, t["group"]))) is None:
            continue
        if id(name) not in standing and stands_in(t, name):
            standing.add(id(name))
            t["core"] = True
            t["famous"] = f"famous:{FAMOUS_PASSES}/{FAMOUS_PASSES}" in name["tags"]
            t["core_skip"] = EVERYDAY if "everyday:listed" in t["tags"] else ""
        else:
            t["famous"] = name["famous"]


def stands_in(variant: dict, name: dict) -> bool:
    """A short variant of a core kind (ドリー, ジン, アンビー) is in the resident list's pool when its name is not: the
    list is for names every window may need that sound matching cannot catch, and a short reading is that, while the
    name off the pool (an old full name, long) is not how anyone says it. Only a piece of the name (シーザー of
    キング・シーザー; not a misspelling or a hiragana form the game never writes). Famous by its name's probe (the
    probe asks the name, not an address form of it); one variant per name, the first."""

    return (not variant["rare"] and not variant["skip"] and not name["core"] and variant["ja"] in name["ja"]
            and "short-reading" in variant["tags"]
            and any(f"important:{r}" in variant["tags"] for r in CORE_REASONS))


def probe_pool(terms: list[dict]) -> list[dict]:
    """The names the famous probe must cover (after ``apply_rules``): the resident list's pool (core, not
    ``core_skip``; for a variant standing in, its name), whose famous names stay off the list, and the important names the short / everyday-word rule
    leaves rare for want of evidence, which ``evidence:famous`` keeps (石門, 行政区). Not a name on behalf of a
    demoted variant: the probe asks the name as written, which says nothing of an address form of it (かざり, a word
    of its own, would take 花散里's score)."""

    named = {(t["ja"], t["group"]): t for t in terms if not any(x.startswith("variant-of:") for x in t["tags"])}
    out = []
    for t in terms:
        tags = t["tags"]
        of = next((x[len("variant-of:"):] for x in tags if x.startswith("variant-of:")), "")
        if of and t["core"] and not t["core_skip"] and (name := named.get((of, t["group"]))) is not None:
            out.append(name)  # a variant standing in for its name: the name is asked
            continue
        demoted = (t["rare"] and not t["skip"] and not any(x.startswith("variant-of:") for x in tags)
                   and any(x.startswith("important:") for x in tags) and any(x.startswith("risk:") for x in tags)
                   and not any(x.startswith(("evidence:", "override:rare:")) for x in tags))
        if (t["core"] and not t["core_skip"]) or demoted:
            out.append(t)
    return out


def _signals(files: list[Path]) -> dict[str, dict]:
    """``"<term id>|<ja>"`` -> its signals, from ``signals.py`` files. A file keyed by the term id alone (before
    2026-11-20) matches nothing and would silently drop every important NPC that rests on mentions: refused."""

    out: dict[str, dict] = {}
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data and not all("|" in key for key in data):
            raise SystemExit(f"error: {f} is keyed by term id alone; rerun signals.py (keys are now 'id|ja')")
        out.update(data)
    return out


def _reader_versions() -> dict[str, str]:
    """The libraries whose dictionaries decide readings (``kana``, ``short-reading``) and everyday words: another
    version gives another file from the same inputs, so the file records them."""

    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in ("pyopenjtalk-plus", "sudachipy", "sudachidict-core"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = ""
    return out


def unused_overrides(game: Game, raw: list[dict], terms: list[dict]) -> dict[str, list[str]]:
    """The keys of ``Game``'s hand-written tables that met no name in this build, by table. An upstream that
    respells a name (its ja, or the en ``released`` reads) leaves the override written for the old spelling
    silently unused: dead where the upstream fixed its own slip, lost where it did not."""

    upstream_ja = {r["ja"].strip() for r in raw}
    upstream_en = {str(r.get("en") or "").strip() for r in raw}
    is_variant = [any(x.startswith("variant-of:") for x in t["tags"]) for t in terms]
    names = {t["ja"] for t, v in zip(terms, is_variant) if not v}
    variants = {t["ja"] for t, v in zip(terms, is_variant) if v}
    tiers_keys = {t["ja"] for t in terms} | {
        f"{t['ja']}|{x[len('kind:'):]}" for t in terms for x in t["tags"] if x.startswith("kind:")}
    found = {
        "fixes": [k for k in game.fixes if k not in upstream_ja],
        "drop": [k for k in game.drop if k not in upstream_ja],
        "extra_variants": [k for k in game.extra_variants if k not in upstream_ja],
        "released": [k for k in game.released if k not in upstream_en],
        "ordinary": sorted(k for k in game.ordinary if k not in names),
        "tiers": [k for k in game.tiers if k not in tiers_keys],
        "variant_keep": sorted(k for k in game.variant_keep if k not in variants),
    }
    return {table: keys for table, keys in found.items() if keys}


def build(game: Game, terms_file: Path, dates_files: list[Path], famous_file: Path, background: Path | None,
          built: str, source_commit: str, noise_from: Path | None = None, signals_files: list[Path] = (),
          versions_files: list[Path] = (), readings_from: Path | None = None) -> dict:
    raw = [json.loads(line) for line in terms_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    earlier = {t["ja"]: t["kana"] for t in json.loads(readings_from.read_text(encoding="utf-8"))["terms"]
               if t.get("kana")} if readings_from else {}
    dates = _dates(dates_files)
    famous = {_simplified(zh, game.simplify): n for zh, n in json.loads(famous_file.read_text(encoding="utf-8")).items()}
    signals = _signals(signals_files)
    versions = _versions(list(versions_files))
    group = _groups(raw, game)
    built_day = date.fromisoformat(built)
    cut = built_day.replace(year=built_day.year - RECENT_YEARS).isoformat()

    terms, seen = [], set()
    for r in raw:
        upstream_ja = r["ja"].strip()
        fix = game.fixes.get(upstream_ja, {})
        ja, zh = fix.get("ja", upstream_ja), fix.get("zh") or _simplified(r["zh"].strip(), game.simplify)
        if not ja or not zh or (ja, zh) in seen:
            continue
        seen.add((ja, zh))
        up, reading = set(r.get("tags") or ()), str(r.get("reading") or "")
        carried = "" if reading else _carried(ja, earlier)   # a fixed ja reads afresh: the earlier was the slip's
        differs = bool(carried) and carried != _kana(ja, "")  # tagged where it is not what this machine reads
        reading = reading or carried
        kind = _kind(game, up)
        en = fix.get("en", str(r.get("en") or "").strip())
        upstream_en = str(r.get("en") or "").strip()
        when = game.released.get(upstream_en) or max(dates.get(r.get("id") or "", ""), dates.get(upstream_en, ""))
        lines, story = int(r.get("lines") or 0), int(r.get("story_lines") or 0)
        mentions = (signals.get(f"{r.get('id') or ''}|{r['ja']}") or {}).get("mentions")
        tags = [f"kind:{kind}", *(f"up:{t}" for t in sorted(up))]
        reason = IMPORTANT_KINDS.get(kind)
        if kind == "place-minor" or (kind == "place" and "minor" in up):
            reason = None
        if reason:
            tags.append(f"important:{reason}")
        elif kind == "npc" and _story_npc(game, ja, lines, mentions or 0):
            tags.append("important:story-npc")
        if when:
            tags += [f"added:{when}"] + ([v] if (v := _version_tag(versions, when)) else [])
            if when >= cut:
                tags.append("recent")
        if _short_reading(ja, reading):
            tags.append("short-reading")
        if differs:
            tags.append("reading:carried")
        tags += [f"lines:{lines}"] * bool(lines) + [f"story-lines:{story}"] * bool(story)
        if mentions is not None:
            tags.append(f"mentions:{mentions}")
        if ja in game.ordinary:
            tags.append("everyday:listed")
        if _everyday(ja):
            tags.append("everyday:dictionary")
        if zh in famous:
            tags.append(f"famous:{famous[zh]}/{FAMOUS_PASSES}")
        skip = (game.drop.get(upstream_ja, "") or ("不是名字" if _not_a_name(ja, zh) else "")
                or ("说话人标签残缺" if kind == "npc" and re.search(r"[&＆]$|[&＆]{2}", ja) else ""))
        if skip:
            tags.append(f"skip:{skip}")
        label = fix.get("label", _label(game, up))
        entity = group.get(zh, ja)
        terms.append({
            "ja": ja, "zh": zh, "en": en,
            "kana": _kana(ja, reading), "reading": bool(reading) or not _is_kana(ja),
            "label": label, "note": _note(str(r.get("note_zh") or ""), game.simplify), "group": entity, "tags": tags,
        })
        # Other spellings of the same name (the upstream's variants, a character's given name): names of the same
        # entity with its kind, importance and probe result, never on the resident list.
        variants = [*(r.get("variants") or ()),
                    *({"ja": a, "zh": b} for a, b in game.extra_variants.get(upstream_ja, ()))]
        person = kind in ("playable", "npc")
        for v in variants:
            vja, vzh = str(v.get("ja") or "").strip(), _simplified(str(v.get("zh") or zh).strip(), game.simplify)
            if not vja or (vja, vzh) in seen:
                continue
            seen.add((vja, vzh))
            # not famous: the probe asked the name as written (a variant takes its famous mark in apply_rules)
            vtags = [t for t in tags if t.startswith(("kind:", "important:", "skip:"))]
            vtags.append(f"variant-of:{ja}")
            vcarried = _carried(vja, earlier)
            if _short_reading(vja, vcarried):
                vtags.append("short-reading")
            if vcarried and vcarried != _kana(vja, ""):
                vtags.append("reading:carried")
            if _everyday(vja):
                vtags.append("everyday:dictionary")
            if not skip and (vskip := _variant_skip(game, vja, person)):
                vtags.append(f"skip:{vskip}")
            terms.append({
                "ja": vja, "zh": vzh, "en": "", "kana": _kana(vja, vcarried), "reading": not _is_kana(vja),
                "label": label, "note": "", "group": entity, "tags": vtags,
            })
    for t in terms:  # the tiers before the risk rule, overrides included: the names the background is measured on
        t.update(derive(t["tags"] + override_tags(game, t["ja"], t["tags"])))
    noise, rows = _noise(terms, background, noise_from)
    per_name: collections.Counter = collections.Counter()
    for pair, rate in noise.items():
        if rate < gm.NOISE_LINE:  # the matcher drops a pair at the line by itself
            per_name[pair.split("|", 1)[0]] += rate
    for t in terms:
        if per_name[t["ja"]]:
            t["tags"].append(f"noise:{per_name[t['ja']]:.3g}")
    apply_rules(game, terms)
    unprobed = [f"{t['ja']}={t['zh']}" for t in probe_pool(terms) if t["zh"] not in famous]
    if unprobed:  # the names the probe must cover: an empty --famous file lists all of them
        raise SystemExit(f"error: {len(unprobed)} names missing from the famous probe; probe them and rerun:\n"
                         + "\n".join(unprobed))
    # Every measured pair stays, an exact, skipped or demoted name's too: the runtime reads only the pairs of names it
    # matches by sound, and the rest are the evidence behind the ``noise:`` tags.
    resident = [t for t in terms if t["core"] and not t["famous"] and not t["skip"] and not t["core_skip"]]
    pool = [t for t in terms if t["core"] and not t["skip"] and not t["core_skip"]]
    common = sum(not t["skip"] and not t["rare"] for t in terms)
    print(f"{len(terms)} terms ({common} common, {sum(not t['skip'] and t['rare'] for t in terms)} rare, "
          f"{sum(bool(t['skip']) for t in terms)} skip, "
          f"{sum(bool(t['core_skip']) for t in terms)} core_skip, {sum(t['exact'] for t in terms)} exact); resident list {len(resident)} of {len(pool)} "
          f"in the pool (cut {cut}); {len(noise)} noise pairs from {rows} background rows"
          + (" (carried over)" if background is None and noise_from else ""), file=sys.stderr)
    for table, keys in unused_overrides(game, raw, terms).items():
        print(f"unused override: Game.{table} met no name this build (respelt upstream? rekey or delete): "
              + ", ".join(keys), file=sys.stderr)
    provenance = {"source": game.source, "source_commit": source_commit, "famous_passes": FAMOUS_PASSES,
                  "recent_years": RECENT_YEARS, "background_rows": rows, "readers": _reader_versions()}
    if background is None:
        provenance["noise"] = f"carried over from {noise_from.name}" if noise_from else "none: no background recordings"
    return {
        "format": gs.FORMAT,
        "game": {"name": game.name, "aliases": list(game.aliases), "intro": game.intro},
        "built": built,
        "provenance": provenance,
        "terms": terms,
        "noise": noise,
    }


def encode(data: dict) -> bytes:
    """The file's bytes: compact JSON with one term and one noise pair per line, so that the copy kept in git
    (``builds/<game>/glossary.json``) diffs and compresses by term."""
    def one(value) -> str:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    parts = []
    for key, value in data.items():
        if key == "terms":
            body = "[\n" + ",\n".join(one(t) for t in value) + "\n]"
        elif key == "noise":
            body = "{\n" + ",\n".join(f"{one(k)}:{one(v)}" for k, v in value.items()) + "\n}"
        else:
            body = one(value)
        parts.append(f"{one(key)}:{body}")
    return ("{" + ",\n".join(parts) + "}\n").encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("game", choices=sorted(GAMES))
    ap.add_argument("--terms", type=Path, required=True)
    ap.add_argument("--dates", type=Path, action="append", required=True)
    ap.add_argument("--famous", type=Path, required=True)
    noise = ap.add_mutually_exclusive_group(required=True)
    noise.add_argument("--background", type=Path)
    noise.add_argument("--noise-from", type=Path, help="an earlier build of the game whose background rates to keep")
    noise.add_argument("--no-background", action="store_true",
                       help="publish no background rates (a first build away from the recordings; not for release)")
    ap.add_argument("--signals", type=Path, action="append", default=[], help="signals.py output: mentions")
    ap.add_argument("--versions", type=Path, action="append", default=[],
                    help="{version: [commit, date]} of the game's data repository, for version: tags")
    ap.add_argument("--readings-from", type=Path,
                    help="an earlier build of the game whose kana its names keep (reading:carried where this"
                         " machine's reader reads them otherwise); new names, and names it has no kana for,"
                         " are read here")
    ap.add_argument("--source-commit", required=True, help="the upstream commit the terms were extracted from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--built", default=date.today().isoformat())
    args = ap.parse_args()
    data = build(GAMES[args.game], args.terms, args.dates, args.famous, args.background, args.built,
                 args.source_commit, args.noise_from, args.signals, args.versions, args.readings_from)
    body = encode(data)
    glossary = gs.parse(body)  # what users will install must load
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(body)
    print(f"{args.out}: {len(body)} bytes, sha256 {glossary.sha256}", file=sys.stderr)


if __name__ == "__main__":
    main()
