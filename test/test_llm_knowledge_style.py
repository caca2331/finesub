"""The `style` category: a knowledge entry addressed by name, never matched.

One style (a subtitle group's whole taste) is ONE entry; each convention is a
line — i.e. a node — inside it. What these tests pin is the half of that design
that is invisible in the data: `style` deliberately stays out of the matchable
set, so every text-matching surface must keep missing it while the human and
`--style` surfaces must keep finding it.
"""

from __future__ import annotations

import inspect

import pytest

from finesub.llm.knowledge.base import KNOWLEDGE_CATEGORIES, load_entry_texts
from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.edit import EditError, edit_subject, harvest_rendered_edits
from finesub.llm.knowledge.node.model import CATEGORIES, MATCHABLE_CATEGORIES, STANDALONE_CATEGORIES
from finesub.llm.knowledge.entries import EntrySelection, pin_style_entries
from finesub.llm.knowledge.node.envelope import Binding
from finesub.llm.knowledge.node.presets import preset_for_category
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.proposals import translate_model_proposals
from finesub.llm.knowledge.style import (
    DEFAULT_STYLE_NAME,
    STYLE_MODES,
    StyleSelectionError,
    render_style_block,
    resolve_style_mode,
    resolve_style_names,
    resolve_style_selection,
    style_injects,
)
from finesub.llm.knowledge.node.repo import (
    RENDERED_DIRNAME,
    AmbiguousName,
    KnowledgeRepo,
)


def _new(root, category: str, key: str, intro: str = "一套风格", extra: list[str] | None = None) -> int:
    argv = ["--root", str(root), "new", category, key, "--intro", intro]
    if category == "common":
        argv += ["--type", "游戏"]
    return maintain_main(argv + (extra or []))


def _repo(root) -> KnowledgeRepo:
    KnowledgeRepo.forget(root)
    return KnowledgeRepo.open(root)


# ---- the namespace split ------------------------------------------------------


def test_style_is_not_in_the_matchable_set() -> None:
    """The guard for the decision itself.

    Adding `style` to the matchable set would be one plausible-looking edit
    with three consequences nobody would connect to it: entries reachable by
    text match, an index built for them, and `create_entry` refusing two
    groups a same-named style. `KNOWLEDGE_CATEGORIES` is asserted to BE the
    matchable tuple, not merely to equal it — a second literal is how the
    facade and the store would drift apart."""

    assert "style" in CATEGORIES and "style" in STANDALONE_CATEGORIES
    assert "style" not in MATCHABLE_CATEGORIES
    assert KNOWLEDGE_CATEGORIES is MATCHABLE_CATEGORIES


def test_style_entries_are_invisible_to_every_matching_surface(tmp_path) -> None:
    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)

    assert repo.resolve("某字幕组") is None  # free-text matching
    found, missing = load_entry_texts(tmp_path, ["某字幕组"])  # model-facing bulk read
    assert found == {} and missing == ["某字幕组"]
    with pytest.raises(ValueError):
        repo.index_text("style")  # an index is a matching surface

    resolved = repo.resolve_qualified("某字幕组")  # human/CLI face still finds it
    assert resolved is not None and resolved.category == "style"


def test_a_style_shares_no_namespace_with_the_matchable_categories(tmp_path) -> None:
    """Two namespaces, two rules: matchable keys are unique across
    streamer+common, a standalone category answers only for itself."""

    _new(tmp_path, "common", "引用标注", intro="一个游戏")
    assert _new(tmp_path, "style", "引用标注") == 0  # same name, other namespace: fine
    repo = _repo(tmp_path)
    assert {s.payload["category"] for s in repo.subjects()} == {"common", "style"}

    _new(tmp_path, "style", "某字幕组")
    _new(tmp_path, "style", "某字幕组")  # duplicate inside style: refused
    repo = _repo(tmp_path)
    assert [s.payload["surface"] for s in repo.subjects("style")].count("某字幕组") == 1


def test_a_model_cannot_reach_an_unwired_category(tmp_path) -> None:
    """The category exists before any prompt is wired to it, so the
    model-facing door stays shut: `translate_model_proposals` defaults to the
    matchable set, and the human CLI opts into the rest. Without this, a model
    guessing `category: "style"` would grow entries in a feature that has no
    caps, no scope rules and no reader yet."""

    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)

    def translate(proposal, **kwargs):
        _ops, report, _drafts, _bindings = translate_model_proposals(
            [proposal], repo=repo, knowledge_read_rev=repo.rev, bindings=[], **kwargs
        )
        return report

    created = translate({"op": "create_entry", "category": "style", "entry": "另一组",
                         "intro": "x", "reason": "r"})
    assert [r.reason for r in created.skipped] == ["bad category 'style'"]

    # ...and it cannot address the one the human made either. The refusal is
    # explicit rather than a fallback to the matchable set: see
    # `test_a_category_the_caller_may_not_touch_is_refused_not_retargeted`.
    appended = translate({"op": "append_lines", "category": "style", "entry": "某字幕组",
                          "section": "约定", "content": "偷偷加一条", "reason": "r"})
    assert [r.reason for r in appended.skipped] == ["category 'style' not allowed here"]

    # the human CLI passes the full set — that is how the entry above was made
    opened = translate({"op": "append_lines", "category": "style", "entry": "某字幕组",
                        "section": "约定", "content": "[引用标注] 用「」", "reason": "r"},
                       allow_categories=CATEGORIES)
    assert opened.skipped == []


# ---- the collision the design invites ------------------------------------------
#
# A style is usually named after the group or streamer it serves, so a name
# living in BOTH namespaces is the expected case, not a corner. Every path that
# turns a name into an entry has to face it; the review of 2026-09-02 found
# four that quietly picked the matchable side.


def _both_namespaces(tmp_path) -> None:
    _new(tmp_path, "common", "某字幕组", intro="一个游戏")
    _new(tmp_path, "style", "某字幕组")


def test_a_name_two_namespaces_answer_to_is_refused_not_guessed(tmp_path) -> None:
    _both_namespaces(tmp_path)
    repo = _repo(tmp_path)

    with pytest.raises(AmbiguousName) as excinfo:
        repo.resolve_qualified("某字幕组")
    assert "common/某字幕组" in str(excinfo.value) and "style/某字幕组" in str(excinfo.value)

    assert repo.resolve_qualified("style/某字幕组").category == "style"
    assert repo.resolve_qualified("common/某字幕组").category == "common"
    # a prefix that is not a category stays part of the name
    assert repo.resolve_qualified("不是类别/某字幕组") is None


def test_retire_never_reaches_across_the_namespaces(tmp_path, capsys) -> None:
    """`retire 某字幕组` used to retire the same-named common entry while both
    styles stayed live: the proposal carried no category, so it resolved in the
    matchable set only."""

    _both_namespaces(tmp_path)

    assert maintain_main(["--root", str(tmp_path), "retire", "某字幕组"]) == 1
    assert "请写成 <类别>/<名字>" in capsys.readouterr().err
    repo = _repo(tmp_path)
    assert len(repo.subjects()) == 2  # nothing retired

    assert maintain_main(["--root", str(tmp_path), "retire", "style/某字幕组",
                          "--reason", "test"]) == 0
    repo = _repo(tmp_path)
    assert [s.payload["category"] for s in repo.subjects()] == ["common"]


def test_a_category_the_caller_may_not_touch_is_refused_not_retargeted(tmp_path) -> None:
    """The first cut dropped a forbidden category hint and let the lookup fall
    through. With a same-named proper-noun entry that turned "write into style"
    into "edit the streamer entry" — silently."""

    _both_namespaces(tmp_path)
    repo = _repo(tmp_path)
    before = len(_lines(repo, repo.resolve("某字幕组").subject_id))

    _ops, report, _drafts, _bindings = translate_model_proposals(
        [{"op": "append_lines", "category": "style", "entry": "某字幕组",
          "section": "约定", "content": "偷偷加一条", "reason": "r"}],
        repo=repo, knowledge_read_rev=repo.rev, bindings=[],
    )
    assert [r.reason for r in report.skipped] == ["category 'style' not allowed here"]
    assert len(_lines(repo, repo.resolve("某字幕组").subject_id)) == before


def test_rename_follows_the_same_namespace_rule(tmp_path) -> None:
    _new(tmp_path, "common", "某游戏", intro="一个游戏")
    _new(tmp_path, "style", "甲组")
    _new(tmp_path, "style", "乙组")
    repo = _repo(tmp_path)

    def rename(entry, new_key):
        _ops, report, _drafts, _bindings = translate_model_proposals(
            [{"op": "rename_entry", "category": "style", "entry": entry,
              "new_key": new_key, "reason": "r"}],
            repo=repo, knowledge_read_rev=repo.rev, bindings=[],
            allow_categories=CATEGORIES,
        )
        return [r.reason for r in report.skipped]

    # the other namespace is not a collision — that is the whole point
    assert rename("甲组", "某游戏") == []
    # another style with that name is
    assert rename("甲组", "乙组") == ["new_key already exists ('乙组')"]


def test_merged_into_stays_inside_the_namespace(tmp_path, capsys) -> None:
    """`retire style/甲 --merged-into 目标` used to land on `common/目标`: the
    scoped lookup missed and the matchable fallback answered. A standalone
    category IS a partition, so there is nothing to fall back to."""

    _new(tmp_path, "common", "目标", intro="一个游戏")
    _new(tmp_path, "style", "甲组")
    repo = _repo(tmp_path)

    # translator level: no fallback, so the retire never lands on common/目标
    _ops, report, _drafts, _bindings = translate_model_proposals(
        [{"op": "retire_entry", "category": "style", "entry": "甲组",
          "merged_into": "目标", "reason": "内容已并入"}],
        repo=repo, knowledge_read_rev=repo.rev, bindings=[], allow_categories=CATEGORIES,
    )
    assert [r.reason for r in report.skipped] == ["merged_into '目标' does not exist"]

    # CLI level: the qualified name resolves, and the namespaces must match
    assert maintain_main(["--root", str(tmp_path), "retire", "style/甲组",
                          "--merged-into", "common/目标", "--reason", "x"]) == 1
    assert "不能并入另一个命名空间" in capsys.readouterr().err

    _new(tmp_path, "style", "乙组")
    assert maintain_main(["--root", str(tmp_path), "retire", "style/甲组",
                          "--merged-into", "style/乙组", "--reason", "x"]) == 0
    repo = _repo(tmp_path)
    assert sorted(s.payload["surface"] for s in repo.subjects()) == ["乙组", "目标"]


def test_a_category_without_an_alias_field_refuses_aliases(tmp_path, capsys) -> None:
    """An alias is a matching affordance. The style preset has no alias field,
    so an alias stored there would be invisible in `rendered/` and in the entry
    text while still answering in `resolve()` — two styles could share a hidden
    alias and `style/<别名>` would silently pick one (review 2026-09-02)."""

    assert _new(tmp_path, "style", "某字幕组", extra=["--alias", "某组"]) == 0
    assert "has no alias field" in capsys.readouterr().out
    repo = _repo(tmp_path)
    assert repo.subjects("style") == []  # refused outright, not stored half-way

    # the matchable categories keep theirs
    _new(tmp_path, "common", "某游戏", intro="一个游戏", extra=["--alias", "某作"])
    repo = _repo(tmp_path)
    assert repo.resolve("某作") is not None


# ---- the entry itself ---------------------------------------------------------


def test_preset_scaffolds_the_five_sections_and_shares_nothing(tmp_path) -> None:
    preset = preset_for_category("style")
    assert preset.section_names() == ("档案", "约定", "正例", "反例", "待归类")
    assert preset.strict_sections and preset.default_section is None
    # style has no external truth: nothing is ever sent out for verification,
    # and nothing is shared by default (taste is not a fact).
    for section in preset.section_names():
        assert preset.verify_for(section) == "none"
        assert preset.share_for(section) == "local"

    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    assert subject.payload["section_order"] == list(preset.section_names())
    # sparse means absent: only the identity line is materialized
    assert [n.payload.get("label") for n in _lines(repo, subject.local_id)] == ["本名"]


def _lines(repo: KnowledgeRepo, subject_id: str) -> list:
    return [
        repo.store.node(m.child_id)
        for m in repo.store.children(subject_id)
        if repo.store.node(m.child_id) is not None
    ]


def test_a_rule_and_its_examples_pair_by_label(tmp_path) -> None:
    """A node cannot hold structure (a line owns no children), so a rule and
    its examples are three lines in three sections joined by the label. That
    buys "at most one 正例 per rule" for free: labels are unique within a
    section, which the apply engine already enforces."""

    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    edit_subject(repo, subject, _entry_text([
        ("约定", ["[引用标注] 引用他人的话用「」，不用引号"]),
        # a real example carries the multi-speaker pipe — a note-only section
        # takes it verbatim
        ("正例", ["[引用标注] 甲|乙 → 甲说「不去」"]),
        ("反例", ['[引用标注] 甲|乙 → 甲说"不去"']),
    ]))

    by_section = {
        m.section: repo.store.node(m.child_id).payload
        for m in repo.store.children(subject.local_id)
        if repo.store.node(m.child_id) is not None and m.section != "档案"
    }
    assert {s for s in by_section} == {"约定", "正例", "反例"}
    assert {p["label"] for p in by_section.values()} == {"引用标注"}
    assert by_section["正例"]["text"] == "甲|乙 → 甲说「不去」"

    # a second 正例 for the same rule collides on the label: the engine keeps
    # the entry consistent instead of silently storing two
    with pytest.raises(EditError):
        edit_subject(repo, subject, _entry_text([
            ("约定", ["[引用标注] 引用他人的话用「」，不用引号"]),
            ("正例", ["[引用标注] a → b", "[引用标注] c → d"]),
        ]))


def test_conventions_round_trip_through_rendered(tmp_path) -> None:
    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    edit_subject(repo, subject, _entry_text([
        ("约定", ["[引用标注] 引用他人的话用「」，不用引号", "[自称] 第三人称自称转成「我」"]),
    ]))

    path = tmp_path / RENDERED_DIRNAME / "style" / "某字幕组.md"
    assert path.is_file()
    assert not (tmp_path / RENDERED_DIRNAME / "style" / "index.md").exists()

    # the rendered projection is editable: change one line, harvest it back
    path.write_text(
        path.read_text(encoding="utf-8").replace("第三人称自称转成「我」", "第三人称自称一律转成「我」"),
        encoding="utf-8",
    )
    results = harvest_rendered_edits(repo)
    assert [(r["file"], r["status"]) for r in results] == [("style/某字幕组.md", "applied")]
    bodies = [n.payload.get("text", "") for n in _lines(repo, subject.local_id)]
    assert "第三人称自称一律转成「我」" in bodies


def _entry_text(sections: list[tuple[str, list[str]]]) -> str:
    out = ["# 某字幕组", "", "一套风格", "", "## 档案", "", "- [本名] 某字幕组", ""]
    for name, lines in sections:
        out += [f"## {name}", ""] + [f"- {line}" for line in lines] + [""]
    return "\n".join(out)


# ---- selection and injection (plan §2.5) ---------------------------------------


def test_style_selection_precedence(tmp_path) -> None:
    """CLI argument → project config → nothing, and `""` is a decision."""

    config = tmp_path / "config.toml"
    config.write_text('[llm]\nstyle = "甲组, 乙组"\n', encoding="utf-8")

    assert resolve_style_names("丙组", config_path=config) == ("丙组",)
    assert resolve_style_names(None, config_path=config) == ("甲组", "乙组")
    # an explicit empty value is "no style this run", not "ask the config"
    assert resolve_style_names("", config_path=config) == ()
    # nothing named anywhere: empty. The implicit default is NOT injected here
    # — provenance has to survive to `resolve_style_selection`, or a user who
    # types `--style default_style` becomes indistinguishable from one who
    # typed nothing (review 2026-09-02).
    assert resolve_style_names(None, config_path=tmp_path / "missing.toml") == ()
    # duplicates collapse, order is kept
    assert resolve_style_names("甲组,甲组,乙组") == ("甲组", "乙组")


def test_a_named_style_that_cannot_be_resolved_is_loud(tmp_path) -> None:
    """Silence here would look exactly like a style with no effect: the run
    would translate the whole file without the conventions it was told to use."""

    _new(tmp_path, "common", "某字幕组", intro="一个游戏")

    with pytest.raises(StyleSelectionError, match="没有这个条目"):
        render_style_block(tmp_path, ["不存在的组"])
    with pytest.raises(StyleSelectionError, match="不是风格"):
        render_style_block(tmp_path, ["某字幕组"])  # resolves, but to a common entry

    _new(tmp_path, "style", "某字幕组")
    with pytest.raises(StyleSelectionError, match="多个类别"):
        render_style_block(tmp_path, ["某字幕组"])  # now ambiguous
    assert "某字幕组" in render_style_block(tmp_path, ["style/某字幕组"])


def test_the_injected_block_carries_the_conventions(tmp_path) -> None:
    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)
    edit_subject(repo, repo.subjects("style")[0], _entry_text([
        ("约定", ["[自称转换] 说话者用自己名字自指时改用「我」"]),
        ("正例", ["[自称转换] 甲|乙 → 我说的"]),
    ]))

    block = render_style_block(tmp_path, ["某字幕组"])
    assert "[自称转换] 说话者用自己名字自指时改用「我」" in block
    assert "甲|乙 → 我说的" in block  # the pipe survives into the prompt
    assert '<style name="某字幕组">' in block
    # the PROMPT projection: no `@k` handles, no markdown bullets
    assert "@k" not in block and "- [" not in block

    assert render_style_block(tmp_path, []) == ""  # no style named: no block


def test_the_update_task_pins_its_style_entry(tmp_path) -> None:
    """The style entry is the task's target, so it survives the ranking cut and
    leads the list; a ranked copy of it would be a second injection."""

    ranked = [
        EntrySelection(category="common", key="游戏X", score=9.0, exists=True),
        EntrySelection(category="style", key="某字幕组", score=1.0, exists=True),
    ]
    pinned = pin_style_entries(ranked, ["某字幕组"])
    assert [(s.category, s.key) for s in pinned] == [
        ("style", "某字幕组"), ("common", "游戏X"),
    ]
    assert pin_style_entries(ranked, []) == [ranked[0]]


# ---- what a knowledge-update task may do to a style entry ----------------------


def test_a_handle_does_not_authorize_retiring_or_renaming_the_entry(tmp_path) -> None:
    """The task is handed ONE style entry to propose conventions into. The
    handle is the grant for its LINES; it must not also authorize retiring the
    entry those lines live in (review 2026-09-02 reproduced exactly that with
    `retire_entry @k1`)."""

    _new(tmp_path, "style", "某字幕组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    handles = HandleMap()
    render_subject(repo.store, subject.local_id, mode="prompt", handles=handles)
    bindings = [Binding(**b) for b in handles.bindings()]

    def translate(proposal):
        _ops, report, _drafts, _bindings = translate_model_proposals(
            [proposal], repo=repo, knowledge_read_rev=repo.rev, bindings=bindings
        )
        return report

    retired = translate({"op": "retire_entry", "entry": "@k1", "reason": "并入别处"})
    assert [r.reason for r in retired.skipped] == [
        "retire_entry is not allowed on a style entry here"
    ]
    renamed = translate({"op": "rename_entry", "entry": "@k1", "new_key": "别的名字",
                         "reason": "r"})
    assert [r.reason for r in renamed.skipped] == [
        "rename_entry is not allowed on a style entry here"
    ]

    # line-level work through the same handle is exactly what it IS for
    appended = translate({"op": "append_lines", "entry": "@k1", "section": "约定",
                          "content": "[引用标注] 用「」", "reason": "r"})
    assert appended.skipped == []


def test_the_update_task_offers_a_style_only_against_refined_subtitles() -> None:
    """`artifacts_only` sees the machine's own raw/final text. Letting it edit
    the style would be the model learning its own habits back — so the entry is
    writable only in the refined variant, and at most one of them."""

    from finesub.llm.knowledge import update as knowledge_update

    source = inspect.getsource(knowledge_update._run_knowledge_update)
    assert "if refined and style_names:" in source
    assert "resolve_style_keys(knowledge_root, style_names)[:1]" in source


def test_the_chunk_identity_includes_the_style_it_may_write(tmp_path) -> None:
    """Rerunning the same material against another style is a different
    question; without the style in the ledger key the second run is skipped as
    already-applied and the new style never receives a proposal."""

    from finesub.llm.knowledge.update import _chunk_input_hash

    class _Chunk:
        def packs_text(self):
            return "--- window 0001 ---"

    class _Feedback:
        def research_slice_text(self):
            return ""

    class _Materials:
        general_context = ""
        feedback = _Feedback()

    chunk, materials = _Chunk(), _Materials()
    base = _chunk_input_hash(chunk, materials, [])
    assert _chunk_input_hash(chunk, materials, ["甲组"]) != base
    assert _chunk_input_hash(chunk, materials, ["乙组"]) != _chunk_input_hash(
        chunk, materials, ["甲组"]
    )
    # ...and the same style still recognizes its own applied chunk
    assert _chunk_input_hash(chunk, materials, ["甲组"]) == _chunk_input_hash(
        chunk, materials, ["甲组"]
    )


def test_the_line_cap_counts_the_label(tmp_path) -> None:
    """A free-form label with a short body would otherwise pass a cap that only
    measured the body, and the label is part of what the injection pays for."""

    _new(tmp_path, "style", "某组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    spec = preset_for_category("style").section("约定")
    body = "字" * (spec.max_body_chars - 10)
    with pytest.raises(EditError, match="含 \\[标记\\]"):
        edit_subject(repo, subject, _entry_text([
            ("约定", [f"[{'长' * 20}] {body}"]),
        ]))


def test_a_labelled_example_must_point_at_a_real_convention(tmp_path) -> None:
    """The pairing is the only structure a style entry has, so it is checked,
    not merely asked for: an orphan example proves nothing and still costs
    injection budget. Both directions are the same check — adding an example
    for no convention, and removing the convention out from under one."""

    _new(tmp_path, "style", "某组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]

    with pytest.raises(EditError, match="没有对应的那一条"):
        edit_subject(repo, subject, _entry_text([("正例", ["[没有的约定] 原文 → 译文"])]))

    paired = _entry_text([
        ("约定", ["[引用标注] 引用他人的话用「」"]),
        ("正例", ["[引用标注] 彼女は → 她说「」"]),
    ])
    assert edit_subject(repo, subject, paired).changed

    # removing the convention leaves the example orphaned: the whole batch fails
    with pytest.raises(EditError, match="没有对应的那一条"):
        edit_subject(repo, subject, _entry_text([("正例", ["[引用标注] 彼女は → 她说「」"])]))

    # an UNLABELLED example is the deliberate escape for a second example
    assert edit_subject(repo, subject, _entry_text([
        ("约定", ["[引用标注] 引用他人的话用「」"]),
        ("正例", ["[引用标注] 彼女は → 她说「」", "另一例 a → b"]),
    ])).changed


def test_the_per_window_identity_includes_the_style_and_stays_compatible() -> None:
    """`covered_by_prior_chunks` compares the per-window hashes, so the style
    has to be in THOSE too — the chunk hash alone left the skip in place
    (review 2026-09-02). Windows applied before styles existed keep their hash:
    the key is omitted when nothing is writable."""

    from finesub.llm.knowledge.update import _chunk_material_hashes

    class _Window:
        def pack_text(self):
            return "--- window 0001 ---"

    class _Chunk:
        windows = [_Window()]

    class _Feedback:
        def research_slice_text(self):
            return ""

    class _Materials:
        mode = "refined_aligned"
        general_context = ""
        feedback = _Feedback()

    chunk, materials = _Chunk(), _Materials()
    without = _chunk_material_hashes(chunk, materials)
    assert _chunk_material_hashes(chunk, materials, []) == without  # old ledgers
    assert _chunk_material_hashes(chunk, materials, ["甲组"]) != without
    assert _chunk_material_hashes(chunk, materials, ["甲组"]) != _chunk_material_hashes(
        chunk, materials, ["乙组"]
    )


def test_style_mode_is_the_same_tri_state_the_knowledge_switch_has(tmp_path) -> None:
    """`none` / `read` / `update`, resolved argument → config → `read`.

    `read` is the default because injecting is what a style is FOR; writing
    back changes stored data, so it is asked for (owner 2026-09-02)."""

    assert STYLE_MODES == ("none", "read", "update")
    config = tmp_path / "config.toml"
    config.write_text('[llm]\nstyle_mode = "update"\n', encoding="utf-8")

    assert resolve_style_mode("none", config_path=config) == "none"
    assert resolve_style_mode(None, config_path=config) == "update"
    assert resolve_style_mode(None, config_path=tmp_path / "missing.toml") == "read"
    with pytest.raises(StyleSelectionError, match="只能是"):
        resolve_style_mode("write", config_path=config)

    # `none` is the only one that does not inject
    assert not style_injects("none")
    assert style_injects("read") and style_injects("update")


def test_one_resolver_answers_both_questions(tmp_path) -> None:
    """`--style` + `--style-mode` decide two things — what goes into the
    correction prompt and what the post-task update may write. Both front ends
    ask the same function, because a second one re-deriving `writable` from
    `mode` is how the two would drift."""

    config = tmp_path / "config.toml"
    config.write_text('[llm]\nstyle = "配置里的"\n', encoding="utf-8")

    read = resolve_style_selection("甲组", "read", config_path=config)
    assert read.names == ("甲组",) and read.writable == ()

    update = resolve_style_selection("甲组,乙组", "update", config_path=config)
    assert update.names == ("甲组", "乙组") and update.writable == update.names

    off = resolve_style_selection("甲组", "none", config_path=config)
    assert off.names == () and off.writable == ()  # `none` reads nothing at all

    # unset falls through to the config for the names and to `read` for the mode
    fallback = resolve_style_selection(None, None, config_path=config)
    assert fallback.names == ("配置里的",) and fallback.mode == "read"


def test_style_is_on_by_default_but_only_if_the_default_entry_exists(tmp_path) -> None:
    """Owner 2026-09-02: style is on by default and read-only by default, so a
    base of house conventions applies without being asked for.

    A fresh install has no `default_style`, and that must be silence rather
    than an error — while a name the USER typed and got wrong stays loud."""

    missing_config = tmp_path / "missing.toml"

    # nothing in the store yet: the implicit default is dropped
    selection = resolve_style_selection(
        None, None, knowledge_root=tmp_path, config_path=missing_config
    )
    assert selection.names == () and selection.mode == "read"

    _new(tmp_path, "style", DEFAULT_STYLE_NAME)
    KnowledgeRepo.forget(tmp_path)
    selection = resolve_style_selection(
        None, None, knowledge_root=tmp_path, config_path=missing_config
    )
    assert selection.names == (DEFAULT_STYLE_NAME,)
    assert selection.writable == ()  # read by default: on, but not learning

    # an explicitly named style that does not exist is still an error
    with pytest.raises(StyleSelectionError, match="没有这个条目"):
        render_style_block(tmp_path, resolve_style_names("打错了的名字"))


def test_the_default_probe_creates_nothing_and_survives_a_missing_root(tmp_path) -> None:
    """Probing for the implicit default must not bring a store into being.

    `KnowledgeRepo.open` mkdirs the root, writes an empty `knowledge.sqlite`
    and may kick off the markdown auto-import — a run that names no style used
    to touch none of that, and the probe runs BEFORE ASR (review 2026-09-02).
    """

    from finesub.llm.knowledge.style import default_style_candidate

    fresh = tmp_path / "not-yet"
    assert default_style_candidate(fresh) == ()
    assert not fresh.exists()  # nothing created by asking

    # no root at all (packaged run, no user data): absent, not a TypeError
    assert default_style_candidate(None) == ()
    # ...and the whole selection survives it rather than raising downstream
    assert resolve_style_selection(
        None, None, knowledge_root=None, config_path=tmp_path / "missing.toml"
    ).names == ()


def test_a_named_default_style_is_not_silently_dropped(tmp_path) -> None:
    """`--style default_style` on a store that has no such entry must fail like
    any other missing name.

    The first cut decided "is this the implicit default?" by comparing the
    resolved names against `DEFAULT_STYLE_NAME`, which cannot tell a user who
    typed it from one who typed nothing — so the run went ahead with no style
    at all, and the pre-ASR check waved it through (review 2026-09-02)."""

    fresh = tmp_path / "kb"
    named = resolve_style_selection(
        DEFAULT_STYLE_NAME, "read", knowledge_root=fresh,
        config_path=tmp_path / "missing.toml",
    )
    assert named.names == (DEFAULT_STYLE_NAME,)  # kept, so the load below raises
    with pytest.raises(StyleSelectionError, match="没有这个条目"):
        render_style_block(fresh, named.names)

    # the config counts as naming it too
    config = tmp_path / "config.toml"
    config.write_text(f'[llm]\nstyle = "{DEFAULT_STYLE_NAME}"\n', encoding="utf-8")
    from_config = resolve_style_selection(None, None, knowledge_root=fresh, config_path=config)
    assert from_config.names == (DEFAULT_STYLE_NAME,)


def test_efficiency_turns_the_default_off_like_the_knowledge_switch(tmp_path) -> None:
    """The cheapest shape reads nothing; a default that injects ~600 chars into
    every window would quietly contradict that. Naming one still works."""

    _new(tmp_path, "style", DEFAULT_STYLE_NAME)
    KnowledgeRepo.forget(tmp_path)
    missing_config = tmp_path / "missing.toml"

    default_on = resolve_style_selection(
        None, None, knowledge_root=tmp_path, config_path=missing_config
    )
    assert default_on.names == (DEFAULT_STYLE_NAME,)

    efficiency = resolve_style_selection(
        None, None, knowledge_root=tmp_path, difficulty="efficiency",
        config_path=missing_config,
    )
    assert efficiency.mode == "none" and efficiency.names == ()

    asked_for = resolve_style_selection(
        DEFAULT_STYLE_NAME, "read", knowledge_root=tmp_path, difficulty="efficiency",
        config_path=missing_config,
    )
    assert asked_for.names == (DEFAULT_STYLE_NAME,)  # explicit still wins


def test_saying_no_style_is_not_the_same_as_saying_nothing(tmp_path) -> None:
    """`--style ""` turns style OFF; it must not fall through to the default.

    The provenance fix keyed on "did the resolved names come out empty", which
    is true for both "nobody said anything" and "said none" — so an explicit
    empty value quietly enabled `default_style` (review 2026-09-02)."""

    _new(tmp_path, "style", DEFAULT_STYLE_NAME)
    KnowledgeRepo.forget(tmp_path)
    missing_config = tmp_path / "missing.toml"

    said_nothing = resolve_style_selection(
        None, "read", knowledge_root=tmp_path, config_path=missing_config
    )
    assert said_nothing.names == (DEFAULT_STYLE_NAME,)

    for said_none in ("", (), []):
        selection = resolve_style_selection(
            said_none, "read", knowledge_root=tmp_path, config_path=missing_config
        )
        assert selection.names == (), said_none

    # the config can say it too
    blank = tmp_path / "blank.toml"
    blank.write_text('[llm]\nstyle = ""\n', encoding="utf-8")
    assert resolve_style_selection(
        None, "read", knowledge_root=tmp_path, config_path=blank
    ).names == ()


def _near_limit_sections(lines: int = 20, *, extra: list[str] | None = None) -> list:
    """Every section at its own cap — the case the whole-entry budget is for.

    One section at 20x200 is only ~3.3k tokens; the backstop exists because
    three of them together are not."""

    labels = [f"[规则{i:02d}]" for i in range(lines)]
    rules = [f"{label} " + "长" * 190 for label in labels]
    return [
        ("约定", rules + (extra or [])),
        ("正例", [f"{label} 原文 → " + "译" * 180 for label in labels]),
        ("反例", [f"{label} 原文 → " + "错" * 180 for label in labels]),
    ]


def _fill_to_budget(repo, subject) -> None:
    edit_subject(repo, subject, _entry_text(_near_limit_sections()))


def test_over_budget_an_entry_may_only_shrink(tmp_path) -> None:
    """The per-section caps bound a line and a section; this bounds the whole
    entry's injected projection.

    ⚠ It is monotone, not a plain refusal: fixing an oversized entry is itself
    a write, so refusing every write would lock it at its worst size and leave
    the model no move (owner 2026-09-02)."""

    from finesub.llm.knowledge.node.render import entry_prompt_tokens, over_budget_marker

    _new(tmp_path, "style", "某组")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    budget = preset_for_category("style").max_entry_tokens
    assert budget == 6000

    _fill_to_budget(repo, subject)
    used = entry_prompt_tokens(repo.store, subject.local_id)
    assert used > budget, used

    # the entry says so where the update/repair task reads it
    marker = over_budget_marker(repo.store, subject.local_id, "style")
    assert "只能压缩" in marker and str(budget) in marker

    # adding is refused...
    with pytest.raises(EditError, match="超注入预算"):
        edit_subject(repo, subject, _entry_text(
            _near_limit_sections(extra=["[再来一条] 短的"])
        ))
    # ...and so is making a line longer
    with pytest.raises(EditError, match="只能改短"):
        longer = _near_limit_sections()
        longer[0][1][0] += "再多一个字"
        edit_subject(repo, subject, _entry_text(longer))

    # compressing is the way out, and it works
    report = edit_subject(repo, subject, _entry_text([
        ("约定", [f"[规则{i:02d}] 短的" for i in range(20)]),
    ]))
    assert report.changed
    assert entry_prompt_tokens(repo.store, subject.local_id) <= budget
    assert over_budget_marker(repo.store, subject.local_id, "style") == ""


def test_a_category_without_a_budget_is_unbounded(tmp_path) -> None:
    """Only `style` declares one; the proper-noun entries grow with their
    subject and must not inherit a cap nobody set for them."""

    from finesub.llm.knowledge.node.render import over_budget_marker

    _new(tmp_path, "common", "某游戏", intro="一个游戏")
    repo = _repo(tmp_path)
    subject = repo.subjects("common")[0]
    assert preset_for_category("common").max_entry_tokens is None
    assert over_budget_marker(repo.store, subject.local_id, "common") == ""
