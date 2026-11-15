"""Slimmed knowledge entries and phonetic hints in serial correction windows (``FINESUB_KB_SLIM``, off by default)."""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import pytest

from finesub.llm import kb_phonetic_hints as kh
from finesub.llm.knowledge.base import load_entry_texts
from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.prompts import _load_prompt_template, build_correction_csv_messages
from finesub.llm.stages.correction import serial
from finesub.llm.stages.correction.commit import WindowEntries, _entry_details_signature


def window(*texts):
    return SimpleNamespace(segments=[SimpleNamespace(text=text) for text in texts])


def readers():
    for name in ("pyopenjtalk", "pykakasi", "rapidfuzz"):
        pytest.importorskip(name)


def _seed(root) -> int:
    with KnowledgeStore(root / "knowledge.sqlite") as store:
        with store.begin("import") as txn:
            txn.create_node(
                "S", "subject", {"surface": "原神", "intro": "游戏", "category": "common", "section_order": ["角色"]}
            )
            txn.create_node("T1", "term", {"surface": "スカラマシュ", "zh": "散兵", "desc": "执行官"})
            txn.create_node("T2", "term", {"surface": "サンドローネ", "zh": "桑多涅", "desc": "木偶"})
            txn.create_membership("M1", "S", "T1", "角色", 0)
            txn.create_membership("M2", "S", "T2", "角色", 1)
            txn.create_item("I1", "T1", "misheard", "残表", fuzzy_enabled=True)
            txn.create_node(
                "S2", "subject", {"surface": "崩坏", "intro": "游戏", "category": "common", "section_order": ["角色"]}
            )
            txn.create_node("T3", "term", {"surface": "キアナ", "zh": "琪亚娜", "desc": "主角"})
            txn.create_membership("M3", "S2", "T3", "角色", 0)
        return store.current_rev()


@pytest.fixture
def kb(tmp_path):
    rev = _seed(tmp_path)
    kh.library_terms.cache_clear()
    yield SimpleNamespace(root=str(tmp_path), rev=rev, bodies=load_entry_texts(tmp_path, ["原神", "崩坏"], rev)[0])
    kh.library_terms.cache_clear()


# --- matching -------------------------------------------------------------------


def test_misheard_and_soundalike_rows_are_hinted_verbatim_ones_are_not(kb):
    readers()
    slim = kh.slim_window_entries(
        window("残表が出た", "サンドロネらしい戦い方", "スカラマシュ強い", "今日はいい天気"),
        kb.bodies, root=kb.root, rev=kb.rev,
    )
    rows = slim.hints.splitlines()
    assert rows[0].startswith("1|原神 / 角色|スカラマシュ|散兵")  # stored misheard spelling
    assert rows[1].startswith("2|原神 / 角色|サンドローネ|桑多涅")  # sounds alike
    assert len(rows) == 2  # row 3 already says the name; row 4 sounds like nothing
    assert slim.hinted == ("スカラマシュ", "サンドローネ")


def test_a_term_heard_on_several_rows_is_one_line_and_max_hits_counts_terms(kb, monkeypatch):
    readers()
    monkeypatch.setattr(kh, "MAX_HITS", 1)
    slim = kh.slim_window_entries(window("残表が出た", "今日はいい天気", "また残表", "サンドロネ"), kb.bodies, root=kb.root, rev=kb.rev)
    assert slim.hints.splitlines() == ["1,3|原神 / 角色|スカラマシュ|散兵||执行官"]


def test_a_short_text_is_not_found_inside_a_long_key():
    pytest.importorskip("rapidfuzz")
    assert not kh.fuzzy_hit("sandorone", "ane", 90)
    assert kh.fuzzy_hit("sandorone", "korehasandoonerashii", 85)


def test_a_reading_with_dropped_characters_is_incomplete():
    assert not kh.reading_complete("绝区零", "レー")
    assert kh.reading_complete("サンドローネ", "サンドローネ")


def test_hint_rows_are_the_local_ids_of_the_window_rows(kb):
    """The block's row numbers are the ``local_id`` column of ``<asr_result>``, on a real window."""

    readers()
    from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

    texts = ["今日はいい天気", "サンドロネらしい戦い方", "まあね", "また残表"]
    current = dataclasses.replace(plan_correction_windows(
        [SubtitleSegment(str(i + 10), float(i), i + 0.8, text) for i, text in enumerate(texts)]
    )[0], length_scale=1.0)
    slim = kh.slim_window_entries(current, kb.bodies, root=kb.root, rev=kb.rev)
    user = build_correction_csv_messages(window=current, entry_details="# 原神\n", phonetic_hints=slim.hints)[1]["content"]
    asr = user.split("<asr_result>\n", 2)[-1].split("</asr_result>", 1)[0].splitlines()[1:]
    text_of = {row.split("|", 1)[0]: row.rsplit("|", 1)[-1] for row in asr}
    hinted = {line.split("|")[2]: line.split("|", 1)[0].split(",") for line in slim.hints.splitlines()}
    assert hinted == {"サンドローネ": ["2"], "スカラマシュ": ["4"]}
    assert "サンドロネ" in text_of["2"] and "残表" in text_of["4"]


# --- slimming -------------------------------------------------------------------


BODY = """# 原神
游戏

## 档案

[本名] 原神

## 角色

スカラマシュ|散兵|国崩|执行官
サンドローネ|桑多涅||木偶
フリンズ|菲林斯||挪德卡莱角色
很长的一段说明文字

## 元数据

最近更新日期: 2026-09-03
"""


def test_slimming_keeps_the_head_and_the_names_said_and_leaves_hinted_ones_to_the_block():
    slim = kh._slim_body(BODY, {"スカラマシュ|散兵|国崩|执行官"}, {"サンドローネ|桑多涅||木偶"})
    assert slim.startswith("# 原神\n游戏\n\n## 档案\n\n[本名] 原神\n\n## 角色\n")
    assert "スカラマシュ|散兵|国崩|执行官" in slim and "サンドローネ" not in slim
    assert "フリンズ|菲林斯" in slim and "挪德卡莱角色" not in slim
    assert "很长" not in slim and "元数据" not in slim


def test_every_entry_of_the_window_is_slimmed_from_the_real_rendering_and_an_unknown_one_stays_whole(kb):
    """Scope is the window's injected set; also guards ``_slim_body`` against the renderer's shape."""

    readers()
    bodies = {**kb.bodies, "不存在": "# 不存在\n随便|什么||说明\n"}
    slim = kh.slim_window_entries(window("スカラマシュ強い", "キアナ"), bodies, root=kb.root, rev=kb.rev)
    assert slim.bodies["不存在"] == bodies["不存在"]
    genshin = slim.bodies["原神"]
    assert genshin.startswith("# 原神") and "执行官" in genshin  # said verbatim: the whole line
    assert "桑多涅" in genshin and "木偶" not in genshin  # not said: the name only
    assert "主角" in slim.bodies["崩坏"]  # the second entry's term was said too


def test_without_the_readers_the_entries_stay_whole_with_one_warning(kb, monkeypatch):
    warnings = []

    class Shown:
        def warning(self, code, message, **kwargs):
            warnings.append(message)

    monkeypatch.setattr(kh, "current_reporter", Shown)
    monkeypatch.setattr(kh, "reporter_delivers", lambda reporter: True)
    monkeypatch.setattr(kh, "_warned_missing", False)
    monkeypatch.setitem(sys.modules, "pyopenjtalk", None)  # import raises ImportError
    kh._missing_reader.cache_clear()
    try:
        for _ in range(2):
            assert kh.slim_window_entries(window("残表"), kb.bodies, root=kb.root, rev=kb.rev) is None
    finally:
        kh._missing_reader.cache_clear()
    assert len(warnings) == 1 and "pyopenjtalk" in warnings[0]


# --- the serial window's entries ---------------------------------------------------


class Counter:
    def count_text(self, text):
        return len(text or "")


def _run(tmp_path, *, request_channel=False):
    return SimpleNamespace(
        token_counter=Counter(), request_channel=request_channel, knowledge_root=tmp_path, knowledge_version="rev:3"
    )


@pytest.fixture
def fake_slim(monkeypatch):
    calls = []

    def slim(current, bodies, *, root, rev):
        calls.append(rev)
        return kh.SlimmedEntries({key: f"# {key}\n" for key in bodies}, "2|原神 / 角色|サンドローネ|桑多涅||木偶", ("サンドローネ",))

    monkeypatch.setattr(kh, "slim_window_entries", slim)
    return calls


UNION = {"原神": BODY}


def entries_for(run, current, carried):
    view = serial._carried_view(run, current, carried)
    return serial._window_entries(run, view, dict(view.bodies))


def test_off_the_window_gets_whole_entries_and_no_record(tmp_path, monkeypatch, fake_slim):
    monkeypatch.delenv(kh.ENV, raising=False)
    entries = entries_for(_run(tmp_path), window("x"), UNION)
    assert BODY.strip() in entries.details and entries.hints == "" and entries.slim is None
    assert not entries.slimmed and entries.signature == _entry_details_signature(entries.details)
    assert fake_slim == []


def test_under_the_gate_the_entries_go_in_whole_without_hints(tmp_path, monkeypatch, fake_slim):
    monkeypatch.setenv(kh.ENV, "on")
    whole = entries_for(_run(tmp_path), window("x"), UNION)
    monkeypatch.setattr(kh, "GATE_TOKENS", len(whole.details))
    entries = entries_for(_run(tmp_path), window("x"), UNION)
    assert entries.details == whole.details and entries.hints == "" and not entries.slimmed
    assert entries.slim == {"whole_tokens": len(whole.details), "gate": False}
    assert fake_slim == []


def test_over_the_gate_the_entries_are_slimmed_with_hints_and_recorded(tmp_path, monkeypatch, fake_slim):
    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    entries = entries_for(_run(tmp_path), window("x"), UNION)
    assert "# 原神" in entries.details and "很长" not in entries.details and entries.slimmed
    assert entries.hints.startswith("2|原神")
    assert entries.slim["gate"] is True and entries.slim["hinted"] == ["サンドローネ"]
    assert entries.slim["entry_tokens"] < entries.slim["whole_tokens"]
    assert fake_slim == [3]  # the run's pinned revision
    # the hints are part of the window's identity
    assert entries.signature != WindowEntries(details=entries.details).signature


def test_the_real_carried_view_reads_the_run_s_knowledge_root_and_revision(kb, monkeypatch):
    """No stand-in for the slimming: the run's root and pinned revision reach the store."""

    readers()
    from pathlib import Path

    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    run = SimpleNamespace(
        token_counter=Counter(), request_channel=False, knowledge_root=Path(kb.root), knowledge_version=f"rev:{kb.rev}"
    )
    view = serial._carried_view(run, window("サンドロネらしい戦い方"), kb.bodies)
    assert view.slimmed and view.slim["hinted"] == ["サンドローネ"]
    assert view.hints.startswith("1|原神 / 角色|サンドローネ|桑多涅") and "木偶" not in view.bodies["原神"]


def test_missing_readers_leave_the_entries_whole_and_say_so(tmp_path, monkeypatch):
    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    monkeypatch.setattr(kh, "slim_window_entries", lambda *args, **kwargs: None)
    entries = entries_for(_run(tmp_path), window("x"), UNION)
    assert BODY.strip() in entries.details and entries.hints == "" and not entries.slimmed
    assert entries.slim["skipped"] == "reading libraries missing"


def test_a_batched_window_requests_a_slimmed_carried_entry_back_whole(tmp_path, monkeypatch, fake_slim):
    """The request turn and the delivery show the same slimmed block and hints; the
    entry the request turn asks for comes back whole in the fill."""


    from .test_llm_window_request_turn import DELIVERY, ENTRY_REQUEST, _run as run_batched

    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    out, calls, _search, _kinds, records = run_batched(
        tmp_path, monkeypatch, [ENTRY_REQUEST, DELIVERY], retrieval="none", initial_transfer_keys=["主播A"]
    )
    assert out.exists() and len(calls) == 2
    opening, delivery = calls[0][1], calls[1][1]
    assert delivery[:2] == opening[:2]  # the replayed opening is the one the request turn saw
    user = opening[1]["content"]
    assert "主播资料" not in user and "<kb_phonetic_hints>" in user
    assert "本窗的条目经过精简" in user and "回填会给出整个条目的全文" in user
    fill = delivery[3]["content"]
    assert "主播资料" in fill and "精简过的" in fill
    window = next(r for r in records if r["kind"] == "correction_window_response")["payload"]
    assert window["kb_slim"]["whole_on_request"] == ["主播A"]
    assert fake_slim  # slimmed once, before the request turn


def test_of_two_slimmed_carried_entries_only_the_requested_one_comes_back_whole(tmp_path, monkeypatch, fake_slim):
    from . import test_llm_window_request_turn as rt

    def with_second_entry(path):
        root = base(path)
        (root / "streamer" / "index.md").write_text(
            "- 主播A | エーちゃん | 测试主播\n- 主播B | ビーちゃん | 测试主播\n", encoding="utf-8"
        )
        (root / "streamer" / "主播B.md").write_text(
            "# 主播B\n主播B资料。\n\n## 档案\n本名: 主播B\n\n## 元数据\n最近更新日期: 2026-08-01\n",
            encoding="utf-8",
        )
        return root

    base = rt._knowledge_root
    monkeypatch.setattr(rt, "_knowledge_root", with_second_entry)
    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    _out, calls, _search, _kinds, records = rt._run(
        tmp_path, monkeypatch, [rt.ENTRY_REQUEST, rt.DELIVERY], retrieval="none", initial_transfer_keys=["主播A", "主播B"]
    )
    fill = calls[1][1][3]["content"]
    details = fill.split("<entry_details>", 1)[1].split("</entry_details>", 1)[0]
    assert "主播资料" in details  # requested: whole
    assert "# 主播B" in details and "主播B资料" not in details  # carried, not requested: still slimmed
    assert "你请求的词条是全文" in fill and "精简过的" in fill
    window = next(r for r in records if r["kind"] == "correction_window_response")["payload"]
    assert window["kb_slim"]["whole_on_request"] == ["主播A"]


# --- the prompt -------------------------------------------------------------------


def test_the_fragment_carries_the_rows_verbatim():
    block = _load_prompt_template("fragment_kb_phonetic_hints_v1.md", hint_rows="2,5|原神 / 角色|サンドローネ|桑多涅||木偶")
    assert "<kb_phonetic_hints>\nrows|entry|写法|定名|别名|描述\n2,5|原神 / 角色|サンドローネ|桑多涅||木偶\n</kb_phonetic_hints>" in block


def test_hints_sit_before_the_window_rows_and_no_hints_change_nothing():
    from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

    current = dataclasses.replace(plan_correction_windows([SubtitleSegment("1", 0.0, 1.0, "サンドオーネ")])[0], length_scale=1.0)
    plain = build_correction_csv_messages(window=current, entry_details="# 原神\n")
    assert build_correction_csv_messages(window=current, entry_details="# 原神\n", phonetic_hints="") == plain
    user = build_correction_csv_messages(window=current, entry_details="# 原神\n", phonetic_hints="1|原神 / 角色|サンドローネ|桑多涅||木偶")[1]["content"]
    assert "<kb_phonetic_hints>" not in plain[1]["content"]
    assert user.index("<kb_phonetic_hints>") < user.index("本窗口待处理 ASR：")


# --- end to end -------------------------------------------------------------------


def test_serial_windows_carry_slimmed_entries_and_hints_and_parallel_ones_never_do(tmp_path, monkeypatch, fake_slim):
    import json

    from .conftest import setattr_correction
    from .test_llm_parallel_windows import ScriptedClient, _run as run_windows

    class Recording(ScriptedClient):
        def complete(self, role, messages, **kwargs):
            built = messages("capableC") if callable(messages) else messages
            self.prompts.append(built[-1]["content"])
            return super().complete(role, messages, **kwargs)

    from finesub.llm.stages.task_graph import WindowFlow

    # No request channel (what an agentic window has): a batched window with a
    # base always opens one, and then its entries stay whole.
    setattr_correction(monkeypatch, "window_flow", lambda *a, **k: WindowFlow(window_requests=False, request_channel=False, agent_fetches=False))
    setattr_correction(monkeypatch, "load_index_texts", lambda root: {"streamer": "idx", "common": "idx"})
    setattr_correction(monkeypatch, "load_entry_texts", lambda root, keys: ({key: BODY for key in keys}, []))
    monkeypatch.setenv(kh.ENV, "on")
    monkeypatch.setattr(kh, "GATE_TOKENS", 10)
    for continuity in ("serial", "parallel"):
        client = Recording()
        client.prompts = []
        art = tmp_path / continuity
        run_windows(
            tmp_path, continuity=continuity, client=client, artifact_dir=art, out_name=f"{continuity}.srt",
            knowledge_enabled=True, initial_transfer_keys=["原神"], monkeypatch=monkeypatch,
            **({"parallel_window_limit": 2} if continuity == "parallel" else {}),
        )
        responses = [
            json.loads(line)["payload"]
            for line in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()
            if '"correction_window_response"' in line
        ]
        if continuity == "serial":
            assert client.prompts and all("<kb_phonetic_hints>" in p and "很长" not in p for p in client.prompts)
            assert all(r["kb_slim"]["gate"] for r in responses)
            assert len(fake_slim) == len(client.prompts)
        else:
            assert client.prompts and not any("<kb_phonetic_hints>" in p for p in client.prompts)
            assert all("很长" in p for p in client.prompts) and not any("kb_slim" in r for r in responses)
            assert len(fake_slim) == len(responses)  # unchanged: the serial run's calls only
