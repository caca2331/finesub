"""Game glossaries: the store beside the knowledge base, the per-run view and the window block."""

from __future__ import annotations

import json
from datetime import date

import pytest

from finesub.llm import game_glossary as gg
from finesub.llm import glossary_match as gm
from finesub.llm import glossary_store as gs
from finesub.llm.prompts import _load_prompt_template


def _term(ja, zh, *, kana="", reading=False, label="角色", core=False, famous=False, group="", en="", **marks):
    return {"ja": ja, "zh": zh, "en": en, "kana": kana, "reading": reading, "label": label, "note": "",
            "group": group, "core": core, "famous": famous, **marks}


def _file(terms=None, *, name="原神", built="2026-11-12", noise=None, format=2) -> bytes:
    return json.dumps({
        "format": format,
        "game": {"name": name, "aliases": ["Genshin Impact", "げんしん"], "intro": "原神的官方日中专名对照"},
        "built": built,
        "terms": terms or [
            _term("テイワット", "提瓦特", kana="テイワット", label="地区", core=True),
            _term("ヌヴィレット", "那维莱特", kana="ヌヴィレット", core=True, famous=True),
            _term("鍾離", "钟离", kana="ショウリ", reading=True, core=True, famous=True),
            _term("フリーナ", "芙宁娜", kana="フリーナ", core=True, famous=True),
            _term("オデット", "奥黛特", kana="オデット"),
            _term("パイモン", "派蒙", kana="パイモン"),
        ],
        "noise": noise or {},
    }, ensure_ascii=False).encode("utf-8")


# --- the store ---------------------------------------------------------------------


def test_install_replaces_the_game_and_keeps_recent_files(tmp_path):
    first = gs.install(tmp_path, _file())
    assert gs.installed(tmp_path)["原神"].sha256 == first.sha256
    hashes = [first.sha256]
    for day in range(13, 17):
        hashes.append(gs.install(tmp_path, _file(built=f"2026-11-{day}")).sha256)
    assert gs.installed(tmp_path)["原神"].sha256 == hashes[-1]
    kept = {p.stem for p in (tmp_path / "glossary").glob("*.json")} - {"installed"}
    assert kept == set(hashes[-gs.KEEP_FILES:])
    assert gs.load(tmp_path, hashes[0]) is None


def test_an_installed_file_this_finesub_cannot_read_is_warned_and_left_out(tmp_path, reported):
    glossary = gs.install(tmp_path, _file())
    (tmp_path / "glossary" / f"{glossary.sha256}.json").write_bytes(_file(format=1))  # the 2026-11-12 Genshin file
    assert gs.load_selection(tmp_path, [{"game": "原神", "sha256": glossary.sha256}]) == []
    assert reported.codes() == ["game-glossary-unreadable"]
    gs.install(tmp_path, _file(built="2026-11-13"))  # fetched again: a task pinned to the old file still goes without
    assert gs.load_selection(tmp_path, [{"game": "原神", "sha256": glossary.sha256}]) == []
    assert reported.codes() == ["game-glossary-unreadable"] * 2


def test_a_task_whose_only_glossary_cannot_be_read_gets_no_block(tmp_path, reported):
    from finesub.llm.correction_translation import _task_glossaries
    from finesub.llm.prompt_artifacts import _preview_glossary

    glossary = gs.install(tmp_path, _file())
    (tmp_path / "glossary" / f"{glossary.sha256}.json").write_bytes(_file(format=1))
    assert _task_glossaries(tmp_path, "原神", None, "ja") is None
    assert _preview_glossary(tmp_path, "原神", [], "ja", None) is None


@pytest.mark.parametrize("raw, message", [
    (b"{", "not a JSON file"),
    (json.dumps({"format": 9}).encode(), "unsupported glossary format"),
    (json.dumps({"format": 1, "game": {"name": "x", "intro": "y"}, "built": "2026-11-12", "terms": []}).encode(),
     "unsupported glossary format"),
    (json.dumps({"format": 2, "game": {"name": "x", "intro": "y"}, "built": "2026-11-12", "terms": []}).encode(),
     "no terms"),
    (json.dumps({"format": 2, "game": {"name": "x", "intro": "y"}, "built": "2026-11-12",
                 "terms": [_term("なし", "无", skip="不是名字"), _term("饅頭", "馒头", rare=True)]}).encode(), "no terms"),
])
def test_parse_refuses_unusable_files(raw, message):
    with pytest.raises(gs.GlossaryError, match=message):
        gs.parse(raw)


def test_selection_by_notes_and_by_request(tmp_path):
    installed = gs.install(tmp_path, _file())
    pick = [{"game": "原神", "sha256": installed.sha256}]
    assert gs.selection(tmp_path, "今天玩 genshin impact 的新版本", ()) == pick
    assert gs.selection(tmp_path, "", ["glossary/げんしん"]) == pick
    assert gs.selection(tmp_path, "原神", ["glossary/原神"]) == pick
    assert gs.selection(tmp_path, "", ["原神"]) == []  # a bare name is the user's own entry
    assert gs.split_requests(["glossary/原神", "原神"]) == (["glossary/原神"], ["原神"])
    assert gs.index_lines(tmp_path) == "- glossary/原神 | Genshin Impact、げんしん | 原神的官方日中专名对照"


def test_a_pruned_pin_falls_back_to_the_installed_one(tmp_path):
    current = gs.install(tmp_path, _file())
    loaded = gs.load_selection(tmp_path, [{"game": "原神", "sha256": "0" * 64}])
    assert [g.sha256 for g in loaded] == [current.sha256]


def test_a_task_reads_its_preparation_pick_or_else_its_notes(tmp_path):
    from finesub.llm.correction_translation import _task_glossaries

    installed = gs.install(tmp_path, _file())
    by_notes = _task_glossaries(tmp_path, "原神の新版本", None, "ja")
    assert [g.sha256 for g in by_notes.glossaries] == [installed.sha256] and by_notes.source_language == "ja"
    context = tmp_path / "research-context.json"
    context.write_text(json.dumps({"keep_entries": []}), encoding="utf-8")
    assert _task_glossaries(tmp_path, "原神", context, "ja") is None  # the preparation task picked none
    context.write_text(json.dumps({"glossary_selection": [{"game": "原神", "sha256": installed.sha256}]}),
                       encoding="utf-8")
    assert _task_glossaries(tmp_path, "", context, "zh").glossaries[0].sha256 == installed.sha256
    assert _task_glossaries(None, "原神", None, "ja") is None  # the task does not read the store


def test_fetch_downloads_what_the_manifest_pins_and_installs_it(tmp_path, monkeypatch):
    import hashlib

    from finesub_bootstrap import model_fetch

    data = _file()
    sha = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"glossaries": [{"game": "原神", "aliases": ["Genshin Impact"],
                                                    "url": "https://example.invalid/g.json", "size": len(data),
                                                    "sha256": sha}]}), encoding="utf-8")
    fetched = []

    def fake_fetch(entry, directory, **_):
        (file,) = entry.files
        fetched.append((file.url, file.sha256))
        (directory / file.name).write_bytes(data)

    monkeypatch.setattr(model_fetch, "fetch_fixed_files", fake_fetch)
    root = tmp_path / "kb"
    glossary = gs.fetch(root, "genshin  impact", manifest=manifest)
    assert fetched == [("https://example.invalid/g.json", sha)]
    assert gs.installed(root)["原神"].sha256 == glossary.sha256 == sha
    assert sorted(p.name for p in (root / "glossary").iterdir()) == [f"{sha}.json", "installed.json"]
    with pytest.raises(gs.GlossaryError, match="published: 原神"):
        gs.fetch(root, "崩坏：星穹铁道", manifest=manifest)


def test_the_shipped_manifest_parses():
    assert isinstance(gs.published(), list)


def test_no_knowledge_path_sees_the_glossary(tmp_path):
    from finesub.llm.knowledge.base import load_index_texts
    from finesub.llm.knowledge.node.presets import CATEGORIES

    gs.install(tmp_path, _file())
    assert "glossary" not in CATEGORIES
    assert "glossary" not in load_index_texts(tmp_path)


# --- the window block ----------------------------------------------------------------


def _view(texts, source="ja", sound=False, **kw):
    return gg.GlossaryView([gs.parse(_file(**kw))], texts, source, sound=sound)


def _part(block, tag):
    return block.split(f"<{tag}>\n")[1].split(f"</{tag}>")[0]


def _block(view, rows, skip="", limit=3000):
    return view.block(rows, _load_prompt_template, len, limit, skip)


def test_lists_and_verbatim_candidates():
    texts = ["フリーナとオデット", "オデットがいた", "鍾離先生", "パイモンだ"]
    view = _view(texts)
    block = _block(view, texts[:3])
    core = _part(block, "game_glossary_core")
    assert core.strip() == "地区：テイワット=提瓦特"  # famous core names are left out
    frequent = _part(block, "game_glossary_frequent")
    assert frequent.strip() == "角色：オデット=奥黛特"  # verbatim on two rows
    cands = _part(block, "game_glossary_candidates")
    assert "フリーナ" not in cands and "鍾離" not in cands  # famous and written right: no help needed
    assert "オデット" not in cands  # already on a list
    assert cands.strip().splitlines()[0] == "rows|日文|中文|依据|类别|注"


def test_two_games_that_key_a_group_alike_keep_their_names_apart():
    genshin = gs.parse(_file([_term("ジャック", "杰克", group="杰克"), _term("テイワット", "提瓦特", label="地区")]))
    zzz = gs.parse(_file([_term("ネイヴ", "奈维", group="杰克")], name="绝区零"))
    view = gg.GlossaryView([genshin, zzz], ["ジャックだ", "ジャックが来た"], "ja", sound=False)
    assert [t.zh for t in view.frequent] == ["杰克"]


def test_two_games_keep_their_own_english_for_one_japanese_name():
    genshin = gs.parse(_file([_term("氷", "冰", en="Cryo", label="元素"), _term("テイワット", "提瓦特", label="地区")]))
    starrail = gs.parse(_file([_term("氷", "冰", en="Ice", label="属性")], name="崩坏：星穹铁道"))
    for order in ([genshin, starrail], [starrail, genshin]):
        view = gg.GlossaryView(order, ["We use Ice.", "Cryo again"], "en", sound=False)
        assert sorted((c.term.en, c.rows) for c in view.candidates(["We use Ice.", "Cryo again"])) == [
            ("Cryo", (2,)), ("Ice", (1,))]


def test_english_names_match_whole_words():
    terms = [_term("雷電将軍", "雷电将军", en="Ei"), _term("契約", "契约", en="contract")]
    texts = ["Either way", "Eight hours", "a contract", "the contract"]
    view = _view(texts, source="en", terms=terms)
    assert view.frequent == []  # Ei is not inside Either or Eight; a lower-case word is not a name
    assert _view(["Ei said", "so Ei"], source="en", terms=terms).frequent[0].zh == "雷电将军"


def test_user_entries_win():
    view = _view(["テイワットの話"])
    assert "テイワット=提瓦特" not in _block(view, ["テイワットの話"], skip="テイワット：提瓦特，我的条目")


def test_chinese_source_lists_chinese_names():
    view = _view(["提瓦特"], source="zh")
    assert "地区：提瓦特" in _block(view, ["提瓦特"])


def test_sound_candidates_and_the_cap():
    pytest.importorskip("pyopenjtalk")
    pytest.importorskip("sudachipy")
    pytest.importorskip("rapidfuzz")
    view = _view(["おでっとだよね", "ヘイワットの話"], sound=True, noise={"テイワット|ヘイワット": 3e-4})
    block = _block(view, ["おでっとだよね", "ヘイワットの話"])
    assert "1|オデット|奥黛特|音近" in block
    assert "テイワット" not in _part(block, "game_glossary_candidates")  # on the resident list already
    cut = _block(view, ["おでっとだよね"], limit=len(_block(view, [])) + 5)
    assert "（另有 1 个候选因篇幅略去）" in cut


def test_only_a_stale_build_warns(reported):
    gg.GlossaryPick((gs.parse(_file(built="2020-01-01")),), "zh").run(["x"], len)
    gg.GlossaryPick((gs.parse(_file(built=date.today().isoformat())),), "zh").run(["x"], len)
    assert reported.codes() == ["game-glossary-stale"]


def test_without_the_readers_a_japanese_source_gets_verbatim_only(reported, monkeypatch):
    monkeypatch.setattr(gg.gm, "available", lambda: False)
    run = gg.GlossaryPick((gs.parse(_file()),), "ja").run(["おでっとだよね"], len)
    assert run.view.sound == {} and reported.codes() == ["game-glossary-readers"]
    gg.GlossaryPick((gs.parse(_file()),), "zh").run(["x"], len)  # no sound matching to miss: no warning
    assert reported.codes() == ["game-glossary-readers"]


def test_names_on_one_row_come_in_a_fixed_order():
    names = ["パンタローネ", "ドットーレ", "アルレッキーノ", "コロンビーナ", "サンドローネ"]
    view = gg.GlossaryView([gs.parse(_file([_term(n, n + "zh") for n in names]))], ["".join(names)], "ja",
                           sound=False)
    assert [c.term.ja for c in view.candidates(["".join(names)])] == sorted(names)  # not the set's hash order


def test_a_parallel_run_gives_every_window_the_same_block():
    texts = ["ナヒーダです", "ナヒーダかな", "ニィロウ"]
    pick = gg.GlossaryPick((gs.parse(_file([_term("ナヒーダ", "纳西妲"), _term("ニィロウ", "妮露")])),), "zh")
    run = pick.run(texts, len, fixed=True)
    assert run.block(texts[:1]) == run.block(texts[2:]) != pick.run(texts, len).block(texts[2:])


def test_parallel_block_keeps_recurring_candidates_without_rows():
    texts = ["ナヒーダです", "ナヒーダかな", "ニィロウ"]
    terms = [_term("ナヒーダ", "纳西妲"), _term("ニィロウ", "妮露")]
    view = gg.GlossaryView([gs.parse(_file(terms))], texts, "ja", sound=False)
    block = view.fixed_block(texts, _load_prompt_template, len, 3000)
    cands = _part(block, "game_glossary_candidates")
    assert "ニィロウ" not in cands  # once only
    assert cands.splitlines()[0] == "日文|中文|依据|类别|注"



def test_the_dry_run_shows_the_index_and_the_block_the_notes_select(tmp_path):
    from finesub.llm import prompt_artifacts
    from finesub.llm.length_tier import TaskLengthScale

    from .support.fakes import FakeTokenCounter

    gs.install(tmp_path / "kb", _file())
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "オデットだ。"}]}),
                      encoding="utf-8")
    artifacts = prompt_artifacts.build_prompt_artifacts(
        stable_json=stable, knowledge_root=tmp_path / "kb", extra_info="原神の配信", counter=FakeTokenCounter(),
        length_scale=TaskLengthScale(1.0, "default"), source_language="ja")
    research = json.dumps(artifacts["research_messages"], ensure_ascii=False)
    window = json.dumps(artifacts["correction_messages"], ensure_ascii=False)
    assert "glossary/原神" in research
    assert "テイワット=提瓦特" in window and "オデット|奥黛特|原样" in window


def test_a_parallel_dry_run_previews_the_fixed_block(tmp_path):
    from finesub.llm.prompt_artifacts import _preview_glossary

    gs.install(tmp_path, _file())
    texts = [type("Row", (), {"text": t})() for t in ("オデットだ", "テイワット")]
    assert _preview_glossary(tmp_path, "原神", texts, "ja", type("C", (), {"count_text": len})(),
                             fixed=True).fixed_texts == ("オデットだ", "テイワット")


def test_marked_names_stay_in_the_file_but_off_the_lists():
    terms = [
        _term("ドリー・サングマハベイ", "多莉·桑歌玛哈巴伊", core=True, group="多莉"),
        _term("ドリー", "多莉", group="多莉"),  # a short form: its own row of the same entity
        _term("開花", "绽放", label="元素反应", core=True, core_skip="日常词"),
        _term("先遣隊", "愚人众先遣队", label="组织", skip="日常词"),
        _term("ゴロー饅頭", "五郎馒头", label="料理", rare=True, tags=["kind:food", "added:2022-01-01"]),
    ]
    glossary = gs.parse(_file(terms))
    assert [t.ja for t in glossary.terms if t.resident] == ["ドリー・サングマハベイ"]
    assert glossary.terms[3].skip == "日常词" and glossary.terms[2].core_skip == "日常词"
    texts = ["ドリーが来た", "ドリーだ", "開花反応", "先遣隊がいる"]
    view = gg.GlossaryView([glossary], texts, "ja", sound=False)
    block = _block(view, texts)
    assert _part(block, "game_glossary_core").strip() == "角色：ドリー・サングマハベイ=多莉·桑歌玛哈巴伊"
    assert "ドリー=多莉" in _part(block, "game_glossary_frequent")  # the short name matches on its own
    assert [c.term.ja for c in view.candidates(texts)] == ["開花"]  # off the resident list, still a candidate
    assert "先遣隊" not in block  # skipped: never a candidate
    assert glossary.terms[4].tags == ("kind:food", "added:2022-01-01") and not glossary.terms[4].matched
    rare_texts = ["ゴロー饅頭だ", "ゴロー饅頭を", "ゴロー饅頭が"]
    rare_view = gg.GlossaryView([glossary], rare_texts, "ja", sound=False)
    assert "ゴロー饅頭" not in _block(rare_view, rare_texts)  # rare: kept in the file, never listed or matched
    assert rare_view.candidates(rare_texts) == []


def test_a_format_1_file_has_no_marks():
    glossary = gs.parse(_file())
    assert gs.FORMAT == 2 and not any(t.skip or t.core_skip or t.rare or t.tags for t in glossary.terms)
    assert [t.ja for t in glossary.terms if t.resident] == ["テイワット"]


def test_a_name_written_the_same_in_both_languages_is_no_candidate():
    terms = [_term("八重神子", "八重神子"), _term("ニィロウ", "妮露")]
    view = gg.GlossaryView([gs.parse(_file(terms))], ["八重神子とニィロウ"], "ja", sound=False)
    assert [c.term.ja for c in view.candidates(["八重神子とニィロウ"])] == ["ニィロウ"]


def test_a_sound_match_brings_every_name_of_its_japanese():
    pytest.importorskip("pyopenjtalk")
    pytest.importorskip("sudachipy")
    pytest.importorskip("rapidfuzz")
    terms = [_term("オデット", "奥黛特", kana="オデット"), _term("オデット", "小奥黛特", kana="オデット")]
    view = gg.GlossaryView([gs.parse(_file(terms))], ["おでっとだよね"], "ja", sound=True)
    assert sorted(c.term.zh for c in view.candidates(["おでっとだよね"])) == ["奥黛特", "小奥黛特"]


def test_an_exact_name_is_found_as_written_but_never_by_sound():
    pytest.importorskip("pyopenjtalk")
    pytest.importorskip("sudachipy")
    pytest.importorskip("rapidfuzz")
    terms = [_term("オデット", "奥黛特", kana="オデット", exact=True, tags=["risk:short-reading", "evidence:lines"])]
    glossary = gs.parse(_file(terms))
    assert glossary.terms[0].exact and glossary.terms[0].matched
    view = gg.GlossaryView([glossary], ["おでっとだよね", "オデットだ"], "ja", sound=True)
    assert view.sound == {} or not any(view.sound.values())
    assert [(c.term.ja, c.rows, c.basis) for c in view.candidates(["おでっとだよね", "オデットだ"])] == [
        ("オデット", (2,), "原样")]
    assert not gs.parse(_file()).terms[0].exact  # no mark: matched by sound too


def test_a_name_half_in_each_width_is_found_whole():
    terms = [_term("ヴェルーリヤ・ミラージュ", "璃彩海市", label="地区"), _term("ヴェルー", "韦吕", fold=True)]
    rows = ["ヴェルーﾘﾔ・ミラージュに行く"]
    view = gg.GlossaryView([gs.parse(_file(terms))], rows, "ja", sound=False)
    assert [c.term.zh for c in view.candidates(rows)] == ["璃彩海市"]
    assert gm.widen_kana("炎食いの刑！ﾛｰｴﾝ") == "炎食いの刑！ローエン"
    assert gm.widen_kana("トハﾟースﾞ") == "トパーズ"  # half-width marks after full-width kana compose with them
    assert gm.widen_kana("「ロﾋ゙ン」だ") == "「ロビン」だ"  # a combining mark after a half-width kana too
    topaz = gg.GlossaryView([gs.parse(_file([_term("トパーズ", "托帕", exact=True)]))], ["トハﾟースﾞだ"], "ja",
                            sound=False)
    assert [c.term.zh for c in topaz.candidates(["トハﾟースﾞだ"])] == ["托帕"]
    # both verbatim paths read the same widening: a name with a full-width ！ masks its folded name inside
    terms = [_term("トーマ！", "托马！", label="活动"), _term("トーマ", "托马", exact=True, fold=True)]
    view = gg.GlossaryView([gs.parse(_file(terms))], ["トーマ！"], "ja", sound=False)
    assert [c.term.zh for c in view.candidates(["トーマ！"])] == ["托马！"]


def _fold(text, forms):
    return gm.kana_verbatim(text, forms, gm.by_length(forms))


def test_a_folded_exact_name_is_also_found_in_hiragana_and_half_width_kana():
    pytest.importorskip("pyopenjtalk")
    pytest.importorskip("sudachipy")
    pytest.importorskip("rapidfuzz")
    assert _fold("ろーえんが来た", ["ローエン"]) == {"ローエン"}
    assert _fold("ﾛｰｴﾝだ", ["ローエン"]) == {"ローエン"}
    assert _fold("ろーえンス", ["ローエン"]) == set()  # touches katakana: part of another word
    assert _fold("のいず", ["ノイ"]) == set()  # under FOLD_MIN kana
    assert _fold("「とーま」だよ", ["トーマ"]) == {"トーマ"}
    assert _fold("ろーえんさん、ごろー", ["ローエン", "ゴロー"]) == {"ローエン", "ゴロー"}
    # a hiragana spelling inside a longer word is not the name (the release files' fold names, found by a reviewer)
    for row, form in [("これいいね", "コレイ"), ("これいじょう", "コレイ"), ("ありすぎる", "アリス"),
                      ("あきらめないで", "アキラ"), ("ぼっちだった", "ボッチ"), ("どりーむ", "ドリー"),
                      ("くれーぷ", "クレー"), ("あいらしい", "アイラ"), ("れいんこーと", "レイン"),
                      ("ごろーん", "ゴロー"), ("そのありすぎ", "アリス"), ("まろーえん", "ローエン")]:
        assert _fold(row, [form]) == set(), row
    # さんぽにいこう passes the edges (a particle follows): サンポ is never folded, its hiragana is a word (build.py)
    rows = ["ろーえんが来た", "とぱーずの指輪"]
    terms = [_term("ローエン", "罗恩", kana="ローエン", exact=True, fold=True, tags=["risk:noise", "evidence:kind:playable"]),
             _term("トパーズ", "托帕", kana="トパーズ", exact=True, tags=["risk:common-word", "evidence:lines"])]
    glossary = gs.parse(_file(terms))
    assert glossary.terms[0].fold and not glossary.terms[1].fold
    view = gg.GlossaryView([glossary], rows, "ja", sound=True)
    assert [(c.term.ja, c.rows) for c in view.candidates(rows)] == [("ローエン", (1,))]  # a word stays as written


def test_a_longer_name_masks_a_folded_name_inside_it():
    names = gm.by_length(["ローエンハイム", "ローエン"])
    for row in ["ローエンハイムが来た", "ﾛｰｴﾝﾊｲﾑだ", "ろーえんはいむだ"]:  # は after ろーえん would pass the edges
        assert gm.kana_verbatim(row, ["ローエン"], names) == set(), row
    assert gm.kana_verbatim("ろーえんとローエンハイム", ["ローエン"], names) == {"ローエン"}  # its own word elsewhere
    terms = [_term("ローエン", "罗恩", exact=True, fold=True), _term("ローエンハイム", "罗恩海姆", label="地点")]
    rows = ["ローエンハイムが来た", "ろーえんはいむだ"]
    view = gg.GlossaryView([gs.parse(_file(terms))], rows, "ja", sound=False)
    assert [(c.term.ja, c.rows) for c in view.candidates(rows)] == [("ローエンハイム", (1,))]
    # a longer name with hiragana in it masks too
    assert gm.kana_verbatim("とーまのお茶だ", ["トーマ"], gm.by_length(["トーマのお茶", "トーマ"])) == set()


def test_a_sound_hit_on_one_row_does_not_bring_an_exact_row_of_the_same_ja():
    pytest.importorskip("pyopenjtalk")
    pytest.importorskip("sudachipy")
    pytest.importorskip("rapidfuzz")
    terms = [_term("オデット", "奥黛特", kana="オデット"), _term("オデット", "奥黛特幻影", kana="オデット", exact=True)]
    view = gg.GlossaryView([gs.parse(_file(terms))], ["おでっとだよね"], "ja", sound=True)
    assert [c.term.zh for c in view.candidates(["おでっとだよね"])] == ["奥黛特"]


def test_parse_refuses_a_mark_that_is_not_a_boolean():
    with pytest.raises(gs.GlossaryError, match="must be booleans"):
        gs.parse(_file([_term("ローエン", "罗恩", exact="yes")], format=2))
    for tags in (7, "kind:npc", ["kind:npc", 3]):
        with pytest.raises(gs.GlossaryError, match="tags must be a list of strings"):
            gs.parse(_file([_term("ローエン", "罗恩", tags=tags)]))


def test_a_request_turn_leaves_out_the_names_its_carried_entries_show(monkeypatch):
    from types import SimpleNamespace

    from finesub.llm.stages.correction import request_turn

    skips = []
    glossary = SimpleNamespace(block=lambda rows, skip: skips.append(skip) or "")
    run = SimpleNamespace(glossary=glossary, knowledge_enabled=True, knowledge_index_texts={},
                          max_search_queries_per_window=0, style_catalog=None, extra_style="", context_pack=None,
                          style_block="", task_update_feedback=False, profile=None)
    monkeypatch.setattr(request_turn, "build_window_task_messages", lambda **kwargs: kwargs)
    window = SimpleNamespace(segments=[SimpleNamespace(text="テイワット")])
    request_turn.request_turn_messages(run, window, audio_label="", previous_advice="",
                                       carried_entries_text="テイワット：我的译名")
    assert skips == ["テイワット：我的译名"]
