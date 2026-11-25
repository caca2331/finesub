"""What a correct-only task (a Chinese source) reads of the knowledge base (docs/knowledge.md「只纠错的任务读到什么」).

The run says how it reads where it says which revision (`knowledge.base.KnowledgeRun`), and every reader asks
there: the matchable entries come in the correct-only projection, a term's 中文定名 is a match key, the entry
signals are not booked. A translating task reads what it always did, byte for byte; so does a style, whoever
reads it.
"""

from __future__ import annotations

import concurrent.futures as cf
import json

import pytest

from finesub.llm.agent.agent_mcp_server import KB_TOOL_NAMES, HarnessToolServer
from finesub.llm.knowledge.base import (
    KnowledgeRun,
    active_knowledge_run,
    bind_knowledge_run,
    entry_signals_enabled,
    knowledge_signals_enabled,
    load_entry_texts,
    match_terms,
    pinned_generation_rev,
    reads_correct_only,
    repin_generation_rev,
)
from finesub.llm.knowledge.node.matching import ExactIndex
from finesub.llm.knowledge.node.render import render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.run_context import bind_llm_worker, llm_worker_context

TERM = "スカラマシュ|散兵|国崩|愚人众执行官。"
#: what a correct-only task reads of it: nobody checked the name (`name_trust`), so it carries the mark
MARKED_TERM = "スカラマシュ|散兵|国崩|（未核）愚人众执行官。"
NOTE = "读评论时译作「散兵大人」。"
HEAD_NOTE = "条目聚焦 Ver.6.0。"


@pytest.fixture(autouse=True)
def _no_run_left_bound():
    yield
    bind_knowledge_run(KnowledgeRun())


def _seed(root) -> KnowledgeRepo:  # type: ignore[no-untyped-def]
    """One game entry (head section with a free sentence, a section of a term and a free sentence, a section of
    free sentences alone) and one style."""

    repo = KnowledgeRepo.open(root)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "原神", "intro": "游戏", "category": "common",
                                         "entry_type": "游戏", "native_names": ["原神"],
                                         "section_order": ["档案", "角色", "口吻"]})
        txn.create_node("H", "note", {"text": HEAD_NOTE})
        txn.create_membership("MH", "S", "H", "档案", 0)
        txn.create_node("T", "term", {"surface": "スカラマシュ", "zh": "散兵", "alias_text": "", "reading": "",
                                      "desc": "愚人众执行官。"})
        txn.create_membership("MT", "S", "T", "角色", 0)
        txn.create_item("I1", "T", "aliases", "国崩")
        txn.create_node("N", "note", {"text": NOTE})
        txn.create_membership("MN", "S", "N", "角色", 1)
        txn.create_item("I2", "N", "aliases", "执行官大人")
        txn.create_node("N2", "note", {"text": "整体口吻偏轻松。"})
        txn.create_membership("MN2", "S", "N2", "口吻", 0)
        # same writing in both columns: one key, not two
        txn.create_node("T2", "term", {"surface": "稻妻", "zh": "稻妻", "alias_text": "", "reading": "", "desc": "地区。"})
        txn.create_membership("MT2", "S", "T2", "角色", 2)
        txn.create_node("ST", "subject", {"surface": "甲组", "intro": "简介", "category": "style",
                                          "section_order": ["档案", "约定"]})
        txn.create_node("SN", "note", {"text": "中文纠错", "label": "任务"})
        txn.create_membership("MS", "ST", "SN", "档案", 0)
        txn.create_node("SR", "note", {"text": "谐音梗加括注"})
        txn.create_membership("MR", "ST", "SR", "约定", 0)
    return repo


# ---- the projection --------------------------------------------------------------------------------------


def test_the_projection_keeps_the_head_and_the_term_lines(tmp_path) -> None:
    repo = _seed(tmp_path)
    whole = render_subject(repo.store, "S", mode="prompt")
    projected = render_subject(repo.store, "S", mode="prompt", correct_only=True)
    assert all(part in whole for part in (TERM, NOTE, HEAD_NOTE, "## 口吻"))
    assert projected.splitlines()[:2] == ["# 原神", "游戏"]
    assert HEAD_NOTE in projected, "the head section is kept whole"
    assert MARKED_TERM in projected and "稻妻|稻妻||（未核）地区。" in projected
    assert NOTE not in projected and "口吻" not in projected, "free sentences go, and a section left empty with them"
    # the lines that stay are the translating task's own (but for the trust mark), in its order
    kept = [line.replace("|（未核）", "|") for line in projected.splitlines() if line]
    assert [line for line in whole.splitlines() if line in kept] == kept


def test_a_style_is_never_projected(tmp_path) -> None:
    repo = _seed(tmp_path)
    assert (render_subject(repo.store, "ST", mode="prompt", correct_only=True)
            == render_subject(repo.store, "ST", mode="prompt"))


@pytest.mark.parametrize("kwargs", [{"mode": "human"}, {"mode": "prompt", "preview": "full"}])
def test_the_projection_is_a_read_only_prompt_face(tmp_path, kwargs) -> None:
    repo = _seed(tmp_path)
    with pytest.raises(ValueError):
        render_subject(repo.store, "S", correct_only=True, **kwargs)


def test_every_in_process_read_follows_the_run(tmp_path) -> None:
    """`load_entry_texts` is the one way a matchable entry's body reaches a prompt in this process: the caller
    passes nothing, the run's context decides."""

    repo = _seed(tmp_path)
    whole = repo.entry_injection_text("S")
    assert load_entry_texts(tmp_path, ["原神"])[0] == {"原神": whole}
    with pinned_generation_rev(tmp_path):
        assert load_entry_texts(tmp_path, ["原神"])[0] == {"原神": whole}, "a translating task: byte for byte"
    with pinned_generation_rev(tmp_path, correct_only=True):
        found, _missing = load_entry_texts(tmp_path, ["原神"])
        assert found == {"原神": render_subject(repo.store, "S", mode="prompt", correct_only=True)}
        assert NOTE not in found["原神"]
    assert not reads_correct_only(), "the reading ends with the pin"


def test_the_styles_of_a_correct_only_task_are_todays(tmp_path) -> None:
    from finesub.llm.knowledge.style import load_style_entries

    _seed(tmp_path)
    plain = load_style_entries(tmp_path, ("甲组",))
    with pinned_generation_rev(tmp_path, correct_only=True):
        assert load_style_entries(tmp_path, ("甲组",)) == plain
    assert "谐音梗加括注" in plain[0][1]


# ---- the run's reading travels with its pin ---------------------------------------------------------------


def test_pool_workers_inherit_the_reading_with_the_pin(tmp_path) -> None:
    _seed(tmp_path)
    with pinned_generation_rev(tmp_path, correct_only=True) as rev:
        context = llm_worker_context()
        try:
            with cf.ThreadPoolExecutor(max_workers=2, initializer=bind_llm_worker, initargs=(context,)) as pool:
                seen = list(pool.map(lambda _i: (reads_correct_only(), NOTE in load_entry_texts(tmp_path, ["原神"])[0]["原神"],
                                                 active_knowledge_run().pins), range(2)))
        finally:
            context.release_lanes()
        assert [(a, b) for a, b, _pins in seen] == [(True, False)] * 2
        assert all(list(pins.values()) == [rev] for _a, _b, pins in seen)


def test_a_repin_keeps_the_reading(tmp_path) -> None:
    repo = _seed(tmp_path)
    with pinned_generation_rev(tmp_path, correct_only=True) as rev:
        with repo.store.begin("harness") as txn:
            txn.create_item("I9", "S", "aliases", "GI")
        assert repin_generation_rev(tmp_path) == rev + 1
        assert reads_correct_only()


def test_an_agent_task_is_told_the_reading_in_its_manifest(tmp_path) -> None:
    """The MCP server is another process: the reading is written down where the revision is, for a task that
    names its root (a style grant, the preparation task) and for one that reads the pin alike."""

    from finesub.llm.client import _agent_task_inputs
    from finesub.llm.routing.config import CapabilityTier

    _seed(tmp_path)

    def metadata(**knowledge) -> dict:
        inputs = _agent_task_inputs(
            [{"role": "system", "content": "p"}, {"role": "user", "content": "u"}],
            validator_spec=None, variant="", capability_tier=CapabilityTier.CAPABLE,
            max_repair_attempts=0, retrieval="none",
            call_kwargs={"task": "knowledge", "kb_tools": "read", **knowledge},
        )
        return json.loads(json.dumps(inputs["metadata"]))

    explicit = {"knowledge_root": str(tmp_path), "knowledge_identity": "rev:1"}
    assert "kb_correct_only" not in metadata(**explicit)
    with pinned_generation_rev(tmp_path):
        assert "kb_correct_only" not in metadata() and "kb_signals" not in metadata()
    with pinned_generation_rev(tmp_path, correct_only=True):
        for built in (metadata(), metadata(**explicit)):
            assert built["kb_correct_only"] is True
            assert built["kb_signals"] == "off", "nor does the server book an exposure for it"


# ---- matching ----------------------------------------------------------------------------------------------


def test_the_canonical_name_is_a_key_only_when_asked(tmp_path) -> None:
    repo = _seed(tmp_path)
    plain = ExactIndex.build(repo.store)
    with_zh = ExactIndex.build(repo.store, zh_names=True)
    assert not [m for m in plain.scan("散兵来了") if m.key.node_id == "T"]
    assert [(m.key.kind, m.key.raw) for m in with_zh.scan("散兵来了") if m.key.node_id == "T"] == [("zh", "散兵")]
    # everything else is the same index: the added keys are the canonical names that differ from the first column
    assert plain._keys == {text: [key for key in keys if key.kind != "zh"]
                           for text, keys in with_zh._keys.items() if any(key.kind != "zh" for key in keys)}
    assert [key.raw for keys in with_zh._keys.values() for key in keys if key.kind == "zh"] == ["散兵"]


def test_a_title_in_chinese_finds_the_term_for_a_correct_only_task(tmp_path) -> None:
    _seed(tmp_path)
    title = "【原神实况】散兵登场"
    with pinned_generation_rev(tmp_path):
        assert match_terms(tmp_path, title, exclude_subjects=["原神"]) == []
        assert match_terms(tmp_path, title) == [], "a translating task matches the first column and the aliases"
    with pinned_generation_rev(tmp_path, correct_only=True):
        (hit,) = match_terms(tmp_path, title)
        assert (hit.subject, hit.section, hit.line, hit.via) == ("原神", "角色", MARKED_TERM, ())


def test_a_title_naming_a_free_sentence_by_its_alias_brings_no_sentence_to_a_correct_only_task(tmp_path) -> None:
    _seed(tmp_path)
    title = "执行官大人驾到"
    with pinned_generation_rev(tmp_path):
        (hit,) = match_terms(tmp_path, title)
        assert hit.line == NOTE
    with pinned_generation_rev(tmp_path, correct_only=True):
        assert match_terms(tmp_path, title) == []


def test_a_pin_opened_inside_a_correct_only_run_keeps_the_reading(tmp_path) -> None:
    _seed(tmp_path)
    with pinned_generation_rev(tmp_path, correct_only=True):
        with pinned_generation_rev(tmp_path / "other"):
            assert reads_correct_only()
        assert reads_correct_only()
    assert not reads_correct_only()


@pytest.mark.parametrize("langs, knowledge, reads", [
    ([("zh", "大家好。"), ("zh", "开始吧。")], "read", True),
    ([("zh", "大家好。"), ("zh", "开始吧。")], "none", False),  # nothing pinned, nothing read
    ([("ja", "こんにちは。"), ("ja", "はじめます。")], "read", False),
])
def test_the_task_itself_opens_the_reading_its_mode_asks_for(tmp_path, monkeypatch, langs, knowledge, reads) -> None:
    """The wiring: a Chinese task's whole run (research, windows, proofreading) sits inside the correct-only
    reading, a translating one's does not."""
    from finesub.llm import correction_translation as orchestration
    from finesub.llm.routing.profiles import resolve_profile

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"metadata": {}, "segments": [
        {"id": str(i), "start": i * 2.0, "end": i * 2.0 + 1.0, "text": text, "lang": lang}
        for i, (lang, text) in enumerate(langs, start=1)]}, ensure_ascii=False), encoding="utf-8")
    seen: list[bool] = []

    def run_task(request, artifact_dir, language):  # type: ignore[no-untyped-def]
        seen.append(reads_correct_only())
        return orchestration.CorrectionOutcome()

    monkeypatch.setattr(orchestration, "_run_task", run_task)
    orchestration.run_correction_task(orchestration.CorrectionRequest(
        stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan", knowledge=knowledge,
        knowledge_root=tmp_path / "knowledge", profile=resolve_profile("text", "local", "quality"),
        test_profile=True))
    assert seen == [reads]


def test_the_phonetic_hints_are_off_for_a_correct_only_task(tmp_path, monkeypatch) -> None:
    from finesub.llm import kb_phonetic_hints

    _seed(tmp_path)
    monkeypatch.setenv(kb_phonetic_hints.ENV, "on")
    assert kb_phonetic_hints.enabled()
    with pinned_generation_rev(tmp_path, correct_only=True):
        assert not kb_phonetic_hints.enabled()


# ---- signals -----------------------------------------------------------------------------------------------


def test_a_correct_only_task_books_no_entry_signal_and_keeps_the_style_ones(tmp_path) -> None:
    _seed(tmp_path)
    assert entry_signals_enabled(tmp_path) and knowledge_signals_enabled(tmp_path)
    with pinned_generation_rev(tmp_path, correct_only=True):
        assert not entry_signals_enabled(tmp_path)
        assert knowledge_signals_enabled(tmp_path), "style usage is booked and read as before"


def test_the_shadow_scan_and_the_landing_book_nothing_for_it(tmp_path) -> None:
    from types import SimpleNamespace

    from finesub.llm.stages.correction import run as correction_run

    repo = _seed(tmp_path)
    heard = SimpleNamespace(chunk_id="0001", segments=[SimpleNamespace(text="原神")], source_ids=["1"])
    misheard = SimpleNamespace(chunk_id="0001", segments=[SimpleNamespace(text="元神")], source_ids=["1"])
    rendered = [SimpleNamespace(source_ids=["1"], corrected_text="原神")]

    def events() -> int:
        return repo.store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def book() -> None:
        version = f"rev:{repo.rev}"
        correction_run._shadow_scan_windows(tmp_path, "task", version, [heard])
        correction_run._log_landed_windows(tmp_path, "task", version, [misheard], rendered)

    with pinned_generation_rev(tmp_path, correct_only=True):
        book()
    assert events() == 0
    with pinned_generation_rev(tmp_path):
        book()
    assert events() > 0, "control: a translating task books them through the same calls"


# ---- the agent's knowledge tools ----------------------------------------------------------------------------


class _ReadLedger:
    def record_kb_read(self, **_kwargs) -> None:
        pass


def _server(monkeypatch, root, *, correct_only: bool, entitlement: str = "read") -> HarnessToolServer:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FINESUB_MCP_KNOWLEDGE_ROOT", str(root))
    server = HarnessToolServer(_ReadLedger(), assignment_id="a", worker_id="w", session_id="s", instance_id="i",
                               tools=list(KB_TOOL_NAMES))
    server._task = {"task_id": "t", "lease_generation": 1}
    server._manifest = {"metadata": {"knowledge_identity": "rev:1", "kb_tools": entitlement,
                                     **({"kb_correct_only": True} if correct_only else {})}}
    return server


_RPC_IDS = iter(range(1, 10_000))


def _call(server: HarnessToolServer, name: str, **arguments):  # type: ignore[no-untyped-def]
    reply = server._call(name, arguments, rpc_id=next(_RPC_IDS))
    return reply["isError"], reply["content"][0]["text"]


def _payload(server: HarnessToolServer, name: str, **arguments) -> dict:
    is_error, text = _call(server, name, **arguments)
    assert is_error is False, text
    return json.loads(text)


def test_kb_read_answers_in_the_tasks_projection(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    plain = _server(monkeypatch, tmp_path, correct_only=False)
    only = _server(monkeypatch, tmp_path, correct_only=True)
    assert NOTE in _payload(plain, "kb_read", entry="原神")["text"]
    whole = _payload(only, "kb_read", entry="原神")["text"]
    assert MARKED_TERM in whole and HEAD_NOTE in whole and NOTE not in whole
    narrowed = _payload(only, "kb_read", entry="原神", sections=["角色"])["text"]
    assert "散兵" in narrowed and NOTE not in narrowed
    # a style is the task's own, read whole (fresh sessions: the handles count from the same start)
    styles = [_payload(_server(monkeypatch, tmp_path, correct_only=correct_only), "kb_read", entry="style/甲组")["text"]
              for correct_only in (False, True)]
    assert styles[0] == styles[1] and "谐音梗加括注" in styles[0]


def test_the_section_listing_of_an_oversized_entry_is_sized_on_the_projection(tmp_path, monkeypatch) -> None:
    repo = _seed(tmp_path)
    sizes = {}
    for correct_only in (False, True):
        server = _server(monkeypatch, tmp_path, correct_only=correct_only)
        server._kb_context("kb_read")
        sizes[correct_only] = server._kb_sections_hint(repo, "S", 1)
    assert sizes[True] != sizes[False]
    projected = render_subject(repo.store, "S", mode="prompt", correct_only=True, sections=["角色"],
                               handles=_server(monkeypatch, tmp_path, correct_only=True)._kb_handles)
    assert "角色 (" in sizes[True] and str(len(NOTE)) not in sizes[True] and projected


def test_kb_search_finds_canonical_names_and_no_free_sentence(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    plain = _server(monkeypatch, tmp_path, correct_only=False)
    only = _server(monkeypatch, tmp_path, correct_only=True)
    assert [row["kind"] for row in _payload(plain, "kb_search", query="散兵")["matches"]] == []
    assert [(row["kind"], row["value"]) for row in _payload(only, "kb_search", query="散兵")["matches"]] == [("zh", "散兵")]
    # an alias hung on a free sentence: listed for a translating task, absent for this one
    assert [row["value"] for row in _payload(plain, "kb_search", query="执行官大人")["matches"]] == ["执行官大人"]
    assert _payload(only, "kb_search", query="执行官大人")["matches"] == []


def test_the_search_index_is_cached_per_reading(tmp_path, monkeypatch) -> None:
    """A pseudo session serves task after task: a correct-only one must not search the previous task's index."""

    _seed(tmp_path)
    server = _server(monkeypatch, tmp_path, correct_only=False)
    assert _payload(server, "kb_search", query="散兵")["matches"] == []
    server._manifest["metadata"]["kb_correct_only"] = True
    assert len(_payload(server, "kb_search", query="散兵")["matches"]) == 1
    del server._manifest["metadata"]["kb_correct_only"]
    assert _payload(server, "kb_search", query="散兵")["matches"] == []


def test_kb_read_node_serves_no_free_sentence(tmp_path, monkeypatch) -> None:
    _seed(tmp_path)
    bindings = [{"handle": "@k1", "kind": "node", "id": "T", "expected_valid_from_rev": 1},
                {"handle": "@k2", "kind": "node", "id": "N", "expected_valid_from_rev": 1}]
    plain = _server(monkeypatch, tmp_path, correct_only=False)
    plain._manifest["metadata"]["kb_handle_bindings"] = bindings
    only = _server(monkeypatch, tmp_path, correct_only=True)
    only._manifest["metadata"]["kb_handle_bindings"] = bindings
    assert NOTE in _payload(plain, "kb_read_node", handle="@k2")["text"]
    assert "散兵" in _payload(only, "kb_read_node", handle="@k1")["text"]
    is_error, text = _call(only, "kb_read_node", handle="@k2")
    assert is_error and "unknown handle" in text


def test_a_writer_reads_the_full_preview_whatever_the_manifest_says(tmp_path, monkeypatch) -> None:
    """No correct-only task has a writer today; if one ever carries the mark, it still reads what it writes into."""

    _seed(tmp_path)
    writer = _server(monkeypatch, tmp_path, correct_only=True, entitlement="propose")
    assert NOTE in _payload(writer, "kb_read", entry="原神")["text"]
