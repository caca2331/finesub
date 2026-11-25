"""The maintainer's glossary build: the marks a term's tags give it (tools/game_glossary/build.py ``derive``)."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("sudachipy")
pytest.importorskip("pyopenjtalk")

TOOLS = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def build():
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location("glossary_build", TOOLS / "build.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["glossary_build"] = module  # dataclasses look their module up while it runs
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop("glossary_build", None)
        sys.modules.pop("extract", None)
        sys.path.remove(str(TOOLS))


def test_an_important_recent_name_is_core_and_the_probe_decides_famous(build):
    marks = build.derive(["kind:playable", "important:playable", "added:2026-01-01", "recent", "famous:1/3"])
    assert marks == {"core": True, "famous": False, "skip": "", "core_skip": "", "rare": False,
                     "exact": False}
    assert build.derive(["kind:playable", "important:playable", "recent", "famous:3/3"])["famous"] is True


def test_without_an_important_tag_a_name_is_rare_and_never_core(build):
    marks = build.derive(["kind:npc", "mentions:3", "recent"])
    assert marks["rare"] is True and marks["core"] is False
    assert build.derive(["kind:npc", "important:story-npc", "recent"]) == {
        "core": False, "famous": False, "skip": "", "core_skip": "", "rare": False,
        "exact": False}  # common, but not a core reason


def test_overrides_skip_variants_and_everyday_words(build):
    assert build.derive(["kind:food", "override:common:主线反复出现"])["rare"] is False
    assert build.derive(["kind:boss", "important:boss", "override:rare:只在一个活动里"])["rare"] is True
    assert build.derive(["kind:faction", "important:faction", "recent", "skip:不是名字"])["core"] is False
    assert build.derive(["kind:playable", "important:playable", "recent", "variant-of:ドリー・サングマハベイ"])["core"] is False
    assert build.derive(["kind:concept", "important:concept", "short-reading", "everyday:listed"])["core_skip"] == "日常词"


def test_kinds_and_story_npcs(build):
    genshin = build.GAMES["genshin"]
    assert build._kind(genshin, {"character-sub", "fatui"}) == "npc"
    assert build._kind(genshin, {"fatui"}) == "faction"
    assert build._kind(genshin, {"quest-world", "liyue"}) == "quest"
    assert build._kind(genshin, set()) == "other"
    assert build._story_npc(genshin, "カチーナ", 0, 500) and not build._story_npc(genshin, "カチーナ", 0, 5)
    assert not build._story_npc(genshin, "考え込む作業員", 99, 999)  # a description, not a name
    starrail = build.GAMES["starrail"]
    assert not build._story_npc(starrail, "メンテナンスエンジニア", 109, 2)  # a role with lines, hardly said
    assert build._story_npc(starrail, "スコート", 492, 803)


def test_a_risky_name_needs_evidence_and_is_then_matched_only_as_written(build):
    starrail = build.GAMES["starrail"]
    # 将軍: an everyday word with one line; its mentions count the word, not the name
    general = ["kind:npc", "important:story-npc", "everyday:dictionary", "lines:1", "mentions:1044"]
    risk = build.risk_tags(starrail, general)
    assert risk == ["risk:common-word"] and build.derive(general + risk)["rare"] is True
    # トパーズ: an everyday word, but hundreds of dialogue lines
    topaz = ["kind:npc", "important:story-npc", "everyday:dictionary", "lines:603", "mentions:681"]
    marks = build.derive(topaz + build.risk_tags(starrail, topaz))
    assert not marks["rare"] and marks["exact"]
    # a short reading with enough mentions, a given name with its full name, a playable character, an element
    assert build.risk_tags(starrail, ["short-reading", "mentions:80"]) == ["risk:short-reading", "evidence:mentions"]
    assert "evidence:full-name:珊瑚宮心海" in build.risk_tags(starrail, ["short-reading"], "珊瑚宮心海")
    flame = ["kind:concept", "important:concept", "everyday:dictionary"]
    assert build.risk_tags(starrail, flame) == ["risk:common-word", "evidence:kind:concept"]
    # no risk: nothing written, nothing changes
    assert build.risk_tags(starrail, ["kind:npc", "important:story-npc", "lines:300"]) == []


def test_a_boss_or_major_place_that_is_a_word_needs_more_than_its_kind(build):
    genshin = build.GAMES["genshin"]
    peak = ["kind:boss", "important:boss", "everyday:dictionary"]  # ピーク = 尖峰
    assert build.risk_tags(genshin, peak) == ["risk:common-word"] and build.derive(peak + ["risk:common-word"])["rare"]
    sea = ["kind:place", "important:major-place", "everyday:dictionary", "lines:3"]  # 荒海
    assert build.derive(sea + build.risk_tags(genshin, sea))["rare"] is True
    famous = [*sea, "famous:3/3"]
    assert "evidence:famous" in build.risk_tags(genshin, famous)
    # no everyday word: a short boss name keeps its kind as evidence; a playable one always does
    short = ["kind:boss", "important:boss", "short-reading"]
    assert build.risk_tags(genshin, short) == ["risk:short-reading", "evidence:kind:boss"]
    playable = ["kind:playable", "important:playable", "everyday:dictionary"]
    assert build.risk_tags(genshin, playable) == ["risk:common-word", "evidence:kind:playable"]


def test_noise_overrides_and_the_switch(build):
    genshin = build.GAMES["genshin"]
    # however noisy, a name is only matched as written: noise counts sound matches, which exact has none of
    noisy = ["kind:concept", "important:concept", "noise:0.0012"]
    tags = noisy + build.risk_tags(genshin, noisy)
    assert tags[-2:] == ["risk:noise", "evidence:kind:concept"]
    assert build.derive(tags)["exact"] is True and not build.derive(tags)["skip"]
    mild = ["kind:concept", "important:concept", "noise:0.0003"]
    assert build.derive(mild + build.risk_tags(genshin, mild))["exact"] is True
    captain = ["kind:npc", "important:story-npc", "everyday:dictionary", "mentions:209", "risk:common-word"]
    assert build.derive(captain)["rare"] is True
    kept = build.derive(captain + ["override:common:主线固定称呼"])
    assert not kept["rare"] and kept["exact"]  # kept as a candidate, still only as written
    assert build.derive(captain + ["override:common:x", "override:sound:x"])["exact"] is False
    assert build.risk_tags(dataclasses.replace(genshin, risk_rule=False), noisy) == []  # the switch


def test_apply_rules_rewrites_rule_tags_and_retag_counts_moves(build):
    game = dataclasses.replace(build.GAMES["zzz"], tiers={"治安官": ("rare", "职务称呼")})
    terms = [
        {"ja": "治安官", "zh": "治安官", "group": "治安官",
         "tags": ["kind:npc", "important:story-npc", "lines:40", "risk:stale", "override:common:old"]},
        {"ja": "リン", "zh": "铃", "group": "铃", "tags": ["kind:playable", "important:playable", "short-reading"]},
        {"ja": "師匠", "zh": "师父", "group": "师父",
         "tags": ["kind:npc", "important:story-npc", "everyday:dictionary", "lines:7"]},
        {"ja": "ローエン", "zh": "罗恩", "group": "罗恩", "tags": ["kind:playable", "important:playable", "noise:0.0003"]},
        {"ja": "トパーズ", "zh": "托帕", "group": "托帕",
         "tags": ["kind:playable", "important:playable", "everyday:dictionary"]},
    ]
    build.apply_rules(game, terms)
    assert terms[0]["tags"][-1] == "override:rare:职务称呼" and "risk:stale" not in terms[0]["tags"]
    assert terms[0]["rare"] and terms[1]["exact"] and not terms[1]["rare"] and terms[2]["rare"]
    assert list(terms[1])[-1] == "tags"
    # an exact katakana name: folded unless it is a word (トパーズ) or under FOLD_MIN kana (リン)
    assert terms[3]["exact"] and terms[3]["fold"] and terms[4]["exact"] and not terms[4]["fold"]
    assert not terms[1]["fold"] and not terms[0]["fold"]
    # nor when its hiragana is a word: さんぽ (散歩), ぼっち; あきら is a given name, ろーえん no word
    pytest.importorskip("sudachipy")
    assert build._hiragana_word("サンポ") and build._hiragana_word("ボッチ")
    assert not build._hiragana_word("アキラ") and not build._hiragana_word("ローエン")
    sampo = [{"ja": "サンポ", "zh": "桑博", "group": "桑博", "tags": ["kind:playable", "important:playable", "noise:0.0003"]}]
    build.apply_rules(game, sampo)
    assert sampo[0]["exact"] and not sampo[0]["fold"] and "risk:hiragana-word" in sampo[0]["tags"]

    spec = importlib.util.spec_from_file_location("glossary_retag", TOOLS / "retag.py")
    retag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retag)
    data = {"terms": [dict(t) for t in terms], "noise": {"リン|りん": 0.0001, "師匠|しよう": 0.0001}}
    data["terms"][2]["tags"] = [*data["terms"][2]["tags"], "x"]
    out, moved = retag.retag(retag.build.GAMES["zzz"], data)  # the shipped tiers: 治安官 is rare there too
    assert moved == {} and out["noise"] == data["noise"]  # the noise table is kept whole, for audit
    flipped = {"terms": [dict(sampo[0], fold=True)], "noise": {}, "provenance": {"readers": {"sudachipy": "0"}}}
    out, moved = retag.retag(retag.build.GAMES["starrail"], flipped)  # another dictionary had folded it
    assert moved == {"fold-": 1} and out["provenance"]["fold_readers"] == retag.build._reader_versions()


def test_two_rows_sharing_a_ja_keep_their_own_overrides_and_famous_marks(build):
    game = dataclasses.replace(build.GAMES["starrail"], tiers={"トパーズ|boss": ("rare", "首领形态")})
    terms = [
        {"ja": "トパーズ", "zh": "托帕", "group": "托帕", "tags": ["kind:playable", "important:playable", "recent",
                                                           "famous:3/3"]},
        {"ja": "トパーズ", "zh": "托帕幻象", "group": "托帕幻象", "tags": ["kind:boss", "important:boss"]},
        {"ja": "トパ", "zh": "托帕", "group": "托帕", "tags": ["kind:playable", "important:playable",
                                                       "variant-of:トパーズ"]},
        {"ja": "トパ", "zh": "托帕幻象", "group": "托帕幻象", "tags": ["kind:boss", "important:boss",
                                                           "variant-of:トパーズ"]},
    ]
    build.apply_rules(game, terms)
    assert terms[0]["famous"] and not terms[0]["rare"] and terms[1]["rare"]
    assert terms[2]["famous"] and not terms[3]["famous"]  # each variant follows its own entity's row
    assert "override:rare:首领形态" in terms[3]["tags"] and not any(x.startswith("override:") for x in terms[2]["tags"])


def test_an_override_counts_before_the_background_is_measured(build):
    game = dataclasses.replace(build.GAMES["genshin"], tiers={"ゴロー饅頭": ("common", "主线反复出现")})
    tags = ["kind:food"]
    assert build.derive(tags)["rare"]
    assert not build.derive(tags + build.override_tags(game, "ゴロー饅頭", tags))["rare"]  # what build() measures


def test_a_signals_file_keyed_by_id_alone_is_refused(build, tmp_path):
    new, old = tmp_path / "new.json", tmp_path / "old.json"
    new.write_text('{"genshin:npc:Signora|淑女": {"mentions": 90}}', encoding="utf-8")
    old.write_text('{"genshin:npc:Signora": {"mentions": 90}}', encoding="utf-8")
    assert build._signals([new]) == {"genshin:npc:Signora|淑女": {"mentions": 90}}
    with pytest.raises(SystemExit, match="rerun signals.py"):
        build._signals([new, old])


def test_a_rebuild_keeps_the_earlier_readings_and_reads_only_new_names(build):
    earlier = {"六分街": "ロップンガイ", "アッハ": "アッハ"}
    assert build._carried("六分街", earlier) == "ロップンガイ"  # kept, whatever this machine reads
    assert build._carried("雲璃", earlier) == "" and build._carried("アッハ", earlier) == ""  # new, or kana


def test_the_probe_covers_the_pool_and_the_names_the_risk_rule_demotes(build):
    game = build.GAMES["genshin"]
    terms = [
        {"ja": "チャスカ", "zh": "恰斯卡", "group": "恰斯卡", "tags": ["kind:playable", "important:playable", "recent"]},
        {"ja": "ピーク", "zh": "尖峰", "group": "尖峰", "tags": ["kind:boss", "important:boss", "everyday:dictionary"]},
        {"ja": "かざり", "zh": "花散里", "group": "花散里",
         "tags": ["kind:npc", "important:story-npc", "short-reading", "variant-of:花散里"]},
        {"ja": "花散里", "zh": "花散里", "group": "花散里", "tags": ["kind:npc", "important:story-npc", "mentions:57"]},
        {"ja": "モブ", "zh": "路人", "group": "路人", "tags": ["kind:npc", "short-reading"]},
    ]
    build.apply_rules(game, terms)
    assert [t["ja"] for t in build.probe_pool(terms)] == ["チャスカ", "ピーク"]  # not 花散里 for かざり


def test_a_variant_takes_no_evidence_from_its_names_probe(build, tmp_path):
    terms, dates, famous = tmp_path / "t.jsonl", tmp_path / "d.json", tmp_path / "f.json"
    terms.write_text('{"id": "genshin:characters-inazuma:Hanachirusato", "ja": "花散里", "zh": "花散里", "en": '
                     '"Hanachirusato", "tags": ["character-sub"], "variants": [{"ja": "かざり"}]}\n', encoding="utf-8")
    dates.write_text("{}", encoding="utf-8")
    famous.write_text('{"花散里": 3}', encoding="utf-8")
    signals = tmp_path / "s.json"
    signals.write_text('{"genshin:characters-inazuma:Hanachirusato|花散里": {"mentions": 57}}', encoding="utf-8")
    data = build.build(build.GAMES["genshin"], terms, [dates], famous, None, "2026-11-13", "x",
                       signals_files=[signals])
    name, kazari = data["terms"]
    assert not name["rare"] and "famous:3/3" in name["tags"]
    assert kazari["ja"] == "かざり" and kazari["rare"] and not any(t.startswith("famous:") for t in kazari["tags"])


def test_a_short_variant_stands_in_for_a_name_off_the_resident_pool(build):
    def row(ja, zh, *tags):
        return {"ja": ja, "zh": zh, "group": "多莉", "skip": "", "tags": ["kind:playable", "important:playable", *tags]}

    full = row("ドリー・サングマハベイ", "多莉·桑歌玛哈巴依", "added:2022-09-04", "famous:1/3")
    short = row("ドリー", "多莉", "variant-of:ドリー・サングマハベイ", "short-reading")
    build.apply_rules(build.GAMES["genshin"], [full, short])
    assert not full["core"] and short["core"] and not short["famous"]   # an old long name; how she is called
    assert build.probe_pool([full, short]) == [full]                    # the probe asks the name
    full["tags"].append("famous:3/3")
    build.apply_rules(build.GAMES["genshin"], [full, short])
    assert short["core"] and short["famous"]                            # known by name: off the list
    recent = row("ドリー・サングマハベイ", "多莉·桑歌玛哈巴依", "recent")
    build.apply_rules(build.GAMES["genshin"], [recent, short])
    assert recent["core"] and not short["core"]                         # the name is on it already


def test_retag_stops_on_names_a_rule_change_leaves_to_probe(build):
    spec = importlib.util.spec_from_file_location("glossary_retag", TOOLS / "retag.py")
    retag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retag)
    peak = {"ja": "ピーク", "zh": "尖峰", "group": "尖峰", "skip": "", "rare": False, "core": False,
            "tags": ["kind:boss", "important:boss", "everyday:dictionary", "override:common:x"]}
    with pytest.raises(SystemExit, match="ピーク=尖峰"):  # the shipped tiers have no override for it
        retag.retag(retag.build.GAMES["genshin"], {"terms": [peak], "noise": {}})


def test_a_joined_unknown_katakana_run_is_no_dictionary_word(build):
    # sudachipy 0.6.10 joins an unknown katakana run into one morpheme with is_oov() False (word id 0x0FFFFFFF)
    assert not build._everyday("サンドローネ") and not build._everyday("チャスカ")
    assert build._everyday("ピーク") and build._everyday("散歩")


def test_an_override_the_upstream_no_longer_spells_is_listed(build):
    game = dataclasses.replace(
        build.GAMES["zzz"], fixes={"アンビー": {"zh": "安比"}, "旧い綴り": {"zh": "x"}},
        drop={"消えた名": "理由"}, released={"Anby": "2024-07-04", "Gone": "2024-07-04"},
        ordinary=frozenset({"本部", "無い"}), tiers={"本部|place": ("rare", "r"), "ニコ": ("common", "c"), "無": ("rare", "r")},
        variant_keep=frozenset({"アン", "ない"}), extra_variants={"いない": (("い", "い"),)})
    raw = [{"ja": "アンビー", "en": "Anby"}, {"ja": "本部", "en": "HQ"}]
    terms = [{"ja": "アンビー", "tags": ["kind:playable"]}, {"ja": "アン", "tags": ["kind:playable", "variant-of:アンビー"]},
             {"ja": "本部", "tags": ["kind:place"]}, {"ja": "ニコ", "tags": ["kind:playable"]},
             {"ja": "ニコ", "tags": ["kind:playable", "variant-of:ニコラ"]}]  # a name and another's variant
    game = dataclasses.replace(game, variant_keep=game.variant_keep | {"ニコ"})
    assert build.unused_overrides(game, raw, terms) == {
        "fixes": ["旧い綴り"], "drop": ["消えた名"], "extra_variants": ["いない"], "released": ["Gone"],
        "ordinary": ["無い"], "tiers": ["無"], "variant_keep": ["ない"]}
    assert not build.unused_overrides(dataclasses.replace(
        game, fixes={}, drop={}, extra_variants={}, released={}, ordinary=frozenset(), tiers={},
        variant_keep=frozenset()), raw, terms)


def test_strings_that_are_just_the_name_are_not_mentions():
    spec = importlib.util.spec_from_file_location("glossary_signals", TOOLS / "signals.py")
    signals = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(signals)
    text = "\n" + "\n".join(["Rin"] * 120 + ["Rin came", "a Ring"]) + "\n"
    assert signals.count_en(text, "Rin") == 1
    assert signals.count_ja("\nリン\nリン\nリンが来た\nリンゴ\n", "リン") == 1


def test_the_file_puts_one_term_and_one_noise_pair_per_line(build):
    data = {"format": 2, "game": {"name": "原神", "aliases": ["Genshin"]}, "built": "2026-11-13",
            "terms": [{"ja": "若", "zh": "若", "tags": ["kind:npc"]}, {"ja": "ナヒーダ", "zh": "纳西妲", "tags": []}],
            "noise": {"ナヒーダ|ナイーダ": 4e-05, "若|若い": 1e-04}}
    body = build.encode(data)
    assert json.loads(body) == data
    lines = body.decode("utf-8").splitlines()
    assert '{"ja":"若","zh":"若","tags":["kind:npc"]},' in lines and '"若|若い":0.0001' in lines


def test_compare_lists_moved_names_and_tag_changes():
    spec = importlib.util.spec_from_file_location("glossary_compare", TOOLS / "compare.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    old = {"terms": [{"ja": "チャスカ", "zh": "恰斯卡", "rare": False, "tags": ["kind:playable"]},
                     {"ja": "ピーク", "zh": "尖峰", "rare": True, "tags": ["risk:common-word"]}]}
    new = {"provenance": {"readers": {"sudachipy": "0.6.10"}},
           "terms": [{"ja": "チャスカ", "zh": "恰斯卡", "rare": True, "tags": ["kind:playable", "risk:common-word"]},
                     {"ja": "ピーク", "zh": "尖峰", "rare": True, "tags": ["risk:common-word"]}]}
    out = "\n".join(compare.compare(old, new, 5))
    assert "common->rare: 1" in out and "チャスカ=恰斯卡: risk:common-word" in out and "'risk:common-word': 1" in out


def test_textmap_check_reads_each_listed_name_against_one_key():
    spec = importlib.util.spec_from_file_location("glossary_textmap_check", TOOLS / "textmap_check.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    tables = check.Tables({
        "ja": {"1": "非有非無の檻", "2": "何爺", "3": "「何爺」はどこだ？", "4": "結晶", "5": "霜の{RUBY#[S]ジャック}精の村",
               "6": "故郷を偲ぶ孤独の狼", "7": "邪知深き狂者を倒せ", "8": "月結晶", "9": "月結晶の反応だ",
               "10": "レンジャーだ", "11": "ムラタ人"},
        "zh": {"1": "非有非无有之笼", "2": "老何", "3": "「老何」在哪？", "4": "结晶", "5": "霜精的村落",
               "6": "「望乡的孤狼」", "7": "击败诈狂", "8": "月结晶", "9": "是月结晶反应", "10": "是巡林员",
               "11": "穆纳塔人"},
        "en": {"1": "Cage of Suchness", "2": "Uncle He", "3": "Where is Uncle He?", "4": "<color=#FFF>Crystallize</color>",
               "5": "Jack Frost Village", "6": "Lonely Wolf", "7": "Defeat Scamface", "8": "Lunar-Crystallize",
               "9": "A Lunar Crystallize reaction", "10": "Forest Rangers!", "11": "Muratans"}})

    def term(ja, zh, en, rare=False, tags=()):
        return {"ja": ja, "zh": zh, "en": en, "label": "", "rare": rare, "skip": "", "tags": list(tags)}

    rows = {r[1]: r[0] for r in check.check(tables, [
        term("非有非無の檻", "非有非无之笼", "Cage of Suchness"),   # the game writes 有 twice
        term("何爺", "老何", "Uncle He"),                          # one key: ok, left out
        term("結晶", "结晶", "Crystalize"),                        # markup ignored, the typo is not
        term("雪の精の村", "霜精的村落", "Jack Frost Village"),    # ruby ignored: the game's ja is 霜の精の村
        term("故郷を偲ぶ孤独の狼", "「望乡的孤狼", "Lonely Wolf"),  # found once brackets are dropped, not as written
        term("邪知深き狂者", "诈狂", "Scamfac"),                   # only inside Scamface: not a whole word
        term("月結晶", "月结晶", "Lunar Crystallize"),             # its own key says Lunar-Crystallize
        term("レンジャー", "巡林员", "Forest Rangers / Rebel"),    # every alternative must be with it
        term("ムラタ人", "穆纳塔人", '"Muratans"'),                 # quotes the game does not write
        term("何じい", "老何", "", tags=["variant-of:何爺"]),       # a misspelling variant, zh whole
        term("何爷", "何爷", "", rare=True)])}                    # rare: not checked
    assert rows == {"非有非無の檻": "no-cooc:zh-differs", "結晶": "cooc-en-differs:en-differs",
                    "雪の精の村": "no-cooc:ja-differs", "故郷を偲ぶ孤独の狼": "cooc-punct-differs:punct-differs",
                    "邪知深き狂者": "cooc-en-differs:ja-missing", "月結晶": "cooc-ok:en-differs",
                    "レンジャー": "cooc-en-differs:ja-missing", "ムラタ人": "cooc-punct-differs:punct-differs",
                    "何じい": "variant-ja-missing"}
    assert tables.classify("何爺", "老何", "Uncle He", variant=False)[0] == "ok"
    assert tables.classify("何爺", "老何", "")[0] == "ok"
    assert tables.classify("故郷を偲ぶ孤独の狼", "「望乡的孤狼」", "Lonely Wolf")[0] == "ok"
    assert tables.classify("レンジャー", "巡林员", "Forest Rangers")[0] == "cooc-ok:ja-missing"
    assert check.PASSING == "cooc-ok:ja-missing"
    notation = check.Tables({"ja": {"1": "アル·マジュジ峡谷", "2": "「アトラトの祝福」"}, "zh": {"1": "巨人峡谷", "2": "「阿特拉托的佑助」"},
                             "en": {"1": "Wadi Al-Majuj.", "2": '"Atlatl\'s Blessing"'}})
    assert notation.classify("アル・マジュジ峡谷", "巨人峡谷", "Wadi Al-Majuj")[0] == "ok"   # the dots are one; a full stop
    assert notation.classify("アトラトの祝福", "阿特拉托的佑助", "Atlatl's Blessing")[0] == "ok"  # the game's quotes
    quoted = check.Tables({"ja": {"1": "月の少女", "2": "月の少女に祈る"}, "zh": {"1": "月之少女", "2": "向月之少女祈祷"},
                           "en": {"1": "Moon Maiden", "2": 'Pray to the "Moon Maiden"'}})
    assert quoted.classify("月の少女", "月之少女", '"Moon Maiden"')[0] == "cooc-ok:punct-differs"  # its key: no quotes
    assert quoted.classify("月の少女", "月之少女", "moon maiden")[0] == "cooc-ok:punct-differs"    # no line writes it
    assert quoted.classify("月のしょうじょ", "月之少女", "", variant=True)[0] == "variant-ja-missing"
    assert quoted.classify("月のしょうじょ", "月の少女", "", variant=True)[0] == "variant-zh-missing"  # zh not the game's
    assert quoted.classify("月のしょうじょ", "「月 之少女", "", variant=True)[0] == "variant-zh-missing"  # unclosed, spaced
    # ja / zh written at one key, the en at another: no string the game writes
    split = check.Tables({"ja": {"1": "花火", "2": "花 火"}, "zh": {"1": "花火", "2": "花 火"},
                          "en": {"1": "Sparkle", "2": "Fireworks"}})
    assert split.classify("花火", "花火", "Sparkle")[0] == "ok"
    assert split.classify("花火", "花火", "Fireworks")[0] != "ok"
