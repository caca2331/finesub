"""Aliases a writer can see, and the ones it cannot.

The entry-header `[别名]` row as a model-writable line (cold read Q5, owner
2026-10-03/04): it carries an `@a` handle; `update` rewrites the row as a set
(an empty body clears it), `remove` clears it, `add_item(aliases)` adds one.
A term line's alias column, rewritten by the model or by a person in
`rendered/`, diffs the same way. The diff base is what was rendered, so
tentative aliases are never touched, and naming one is no change."""

from __future__ import annotations

import json

import pytest

from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.render import HandleMap, render_subject, subject_aliases
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.share.conflict_repair import restrict_to_conflicted_nodes


def _repo(tmp_path, *aliases: str, tentative: tuple[str, ...] = ()):  # type: ignore[no-untyped-def]
    assert maintain_main(
        ["--root", str(tmp_path), "new", "common", "游戏X", "--intro", "一个游戏", "--type", "游戏"]
    ) == 0
    KnowledgeRepo.forget(tmp_path)
    repo = KnowledgeRepo.open(tmp_path)
    subject = repo.subjects("common")[0]
    with repo.store.begin("user") as txn:
        for n, value in enumerate(aliases):
            txn.create_item(f"A{n}", subject.local_id, "aliases", value)
        for n, value in enumerate(tentative):
            txn.create_item(f"T{n}", subject.local_id, "aliases", value, maturity="tentative")
    return repo, subject.local_id


def _apply(repo, subject_id, *proposals):  # type: ignore[no-untyped-def]
    handles = HandleMap()
    text = repo.entry_prompt_text(subject_id, handles)
    block = "\n".join(json.dumps(p, ensure_ascii=False) for p in proposals)
    report = apply_model_proposals(
        f"<knowledge_proposals>\n{block}\n</knowledge_proposals>",
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    return text, report


def _aliases(repo, subject_id):  # type: ignore[no-untyped-def]
    return sorted(subject_aliases(repo.store, subject_id))


def test_the_row_carries_its_own_handle_without_shifting_the_line_handles(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX")
    handles = HandleMap()
    text = repo.entry_prompt_text(sid, handles)
    assert "[别名] GX  <!-- @a1 -->" in text
    assert "[本名] 游戏X  <!-- @k2 -->" in text and "@k3" not in text
    # only where the slot renders, and not where the round may not write aliases
    assert "@a" not in render_subject(repo.store, sid, mode="prompt", handles=HandleMap())
    assert "@a" not in repo.entry_prompt_text(sid, HandleMap(), alias_row_handle=False)
    assert "@a" not in repo.entry_text(sid)


def test_the_handle_survives_the_agent_manifest_round_trip() -> None:
    from types import SimpleNamespace

    handles = HandleMap()
    handle = handles.alias_row_handle(SimpleNamespace(local_id="S", valid_from_rev=3))  # type: ignore[arg-type]
    replayed = HandleMap()
    replayed.seed(handles.bindings())
    assert replayed.aliases == {handle: ("S", 3)}
    with pytest.raises(ValueError, match="unknown handle binding kind"):
        HandleMap().seed([{"handle": "@z1", "kind": "zone", "id": "x"}])


def test_update_rewrites_the_row_as_a_set(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX", "GY")
    _text, report = _apply(repo, sid, {"op": "update", "id": "@a1", "line": "[别名] GX、Game X", "reason": "r"})
    assert report.rev is not None and not report.skipped
    assert _aliases(repo, sid) == ["GX", "Game X"]


@pytest.mark.parametrize(
    "proposal",
    [
        {"op": "update", "id": "@a1", "line": "[别名]", "reason": "r"},
        {"op": "update", "id": "@a1", "line": "[别名] ", "reason": "r"},
        {"op": "remove", "id": "@a1", "reason": "r"},
    ],
)
def test_an_empty_body_or_remove_clears_the_row_but_not_tentative_aliases(tmp_path, proposal) -> None:
    repo, sid = _repo(tmp_path, "GX", "GY", tentative=("影子",))
    _text, report = _apply(repo, sid, proposal)
    assert report.rev is not None and not report.skipped
    assert _aliases(repo, sid) == []
    assert [i.value for i in repo.store.items_of(sid) if i.field == "aliases"] == ["影子"]


@pytest.mark.parametrize(
    ("proposal", "reason"),
    [
        ({"op": "update", "id": "@a1", "line": "GX、GZ", "reason": "r"}, "marker"),
        ({"op": "update", "id": "@a1", "line": "[别名] ＧＸ", "reason": "r"}, "no change"),
        ({"op": "add_item", "id": "@a1", "field": "misheard", "value": "ゲーム", "reason": "r"}, "only aliases"),
        ({"op": "add_item", "id": "@a1", "field": "aliases", "value": "GX", "reason": "r"}, "no change"),
    ],
)
def test_what_the_row_refuses(tmp_path, proposal, reason) -> None:
    repo, sid = _repo(tmp_path, "GX")
    _text, report = _apply(repo, sid, proposal)
    assert report.rev is None
    assert any(reason in r.reason for r in report.skipped), report.skipped
    assert _aliases(repo, sid) == ["GX"]


def test_a_copied_handle_comment_is_not_an_alias(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX", "Game X")
    _text, report = _apply(repo, sid, {"op": "update", "id": "@a1", "line": "[别名] GX、Game X、GZ  <!-- @a1 -->", "reason": "r"})
    assert report.rev is not None and _aliases(repo, sid) == ["GX", "GZ", "Game X"]


def test_other_ops_on_the_row_say_why_they_are_refused(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX")
    _text, report = _apply(repo, sid, {"op": "move_line", "id": "@a1", "section": "待归类", "reason": "r"})
    assert report.rev is None and "takes only update / remove / add_item" in report.skipped[0].reason


def test_no_handle_where_the_slot_is_not_rendered(tmp_path) -> None:
    """A real `[别名]` line in the section takes the slot's place; a
    sub-entry renders only its identity slot."""

    repo, sid = _repo(tmp_path, "GX")
    with repo.store.begin("user") as txn:
        txn.create_node("N", "note", {"text": "人写的别名行", "label": "别名"})
        txn.create_membership("MN", sid, "N", "档案", 9)
    assert "@a" not in repo.entry_prompt_text(sid, HandleMap())
    repo2, parent = _repo(tmp_path / "p", "GX")
    with repo2.store.begin("user") as txn:
        txn.create_node("C", "subject", {"surface": "游戏X·角色", "intro": "子", "category": "common", "entry_type": "游戏"})
        txn.create_link("L", "C", "part_of", parent)
    child = repo2.entry_prompt_text("C", HandleMap())
    assert "属于：游戏X" in child and "@a" not in child


@pytest.mark.parametrize(
    ("line", "expected"),
    [("[别名] GX，GZ", ["GX", "GZ"]), ("[别名] —", []), ("[别名] GX、—", ["GX"])],
)
def test_the_row_splits_like_the_term_alias_column(tmp_path, line, expected) -> None:
    """Full-width commas separate and a word-less token is no name -- the
    same split a term line's alias column and a person's edit get."""

    repo, sid = _repo(tmp_path, "GX", "GY")
    _apply(repo, sid, {"op": "update", "id": "@a1", "line": line, "reason": "r"})
    assert _aliases(repo, sid) == expected


def test_add_item_on_the_row_adds_one_name_whole(tmp_path) -> None:
    """A name with a separator in it stays one alias: `add_item` never splits."""

    repo, sid = _repo(tmp_path, "GX")
    _apply(repo, sid, {"op": "add_item", "id": "@a1", "field": "aliases", "value": "彼女、お借りします", "reason": "r"})
    _apply(repo, sid, {"op": "add_item", "id": "@a1", "field": "aliases", "value": "「GY，GZ」", "reason": "r"})
    assert _aliases(repo, sid) == ["GX", "GY，GZ", "彼女、お借りします"]
    _text, report = _apply(repo, sid, {"op": "add_item", "id": "@a1", "field": "aliases", "value": "—", "reason": "r"})
    assert report.rev is None and "empty alias" in report.skipped[0].reason


def test_add_item_on_the_row_defaults_to_aliases(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX")
    _text, report = _apply(repo, sid, {"op": "add_item", "id": "@a1", "value": "GZ", "reason": "r"})
    assert report.rev is not None and _aliases(repo, sid) == ["GX", "GZ"]


def test_an_alias_added_earlier_in_the_batch_survives_a_row_rewrite(tmp_path) -> None:
    """The base is the row the model read, not the batch's running state."""

    repo, sid = _repo(tmp_path, "GX", "GY")
    _text, report = _apply(
        repo, sid,
        {"op": "add_item", "id": "@k1", "field": "aliases", "value": "GZ", "reason": "r"},
        {"op": "update", "id": "@a1", "line": "[别名] GX", "reason": "r"},
    )
    assert report.rev is not None
    assert _aliases(repo, sid) == ["GX", "GZ"]


def test_the_share_conflict_gate_refuses_the_row(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX")
    handles = HandleMap()
    repo.entry_prompt_text(sid, handles)
    ops = [
        {"op": "update", "id": "@a1", "line": "[别名]", "reason": "r"},
        {"op": "remove", "id": "@a1", "reason": "r"},
        {"op": "set_intro", "entry": "@a1", "intro": "x", "reason": "r"},
        {"op": "rename_entry", "entry": "@a1", "new_key": "Y", "reason": "r"},
    ]
    text = "<knowledge_proposals>\n" + "\n".join(json.dumps(o, ensure_ascii=False) for o in ops) + "\n</knowledge_proposals>"
    kept, refused = restrict_to_conflicted_nodes(
        text, handles, allowed={sid}, entry_ops={"set_intro": {sid}, "rename_entry": {sid}}
    )
    assert len(refused) == len(ops) and "@a1" not in kept


def test_a_model_removing_a_human_alias_leaves_a_human_version_mark(tmp_path) -> None:
    repo, sid = _repo(tmp_path, "GX")
    _apply(repo, sid, {"op": "remove", "id": "@a1", "reason": "r"})
    assert "别名「GX」" in repo.entry_text(sid)


def test_a_name_with_a_separator_round_trips_in_corner_brackets(tmp_path) -> None:
    """『彼女、お借りします』 is one alias: rendered inside 「」, and a row
    copied back -- by the model or by a person -- keeps it whole."""

    repo, sid = _repo(tmp_path, "彼女、お借りします", "かのかり")
    text, report = _apply(repo, sid, {
        "op": "update", "id": "@a1", "line": "[别名] 「彼女、お借りします」、かのかり、カノカリ", "reason": "r",
    })
    assert "[别名] 「彼女、お借りします」、かのかり  <!-- @a1 -->" in text
    assert report.rev is not None
    assert _aliases(repo, sid) == ["かのかり", "カノカリ", "彼女、お借りします"]
    assert "「彼女、お借りします」" in repo.entry_text(sid)
    index = "\n".join(repo.index_text(c) for c in ("common",))
    assert "「彼女、お借りします」" in index


def test_a_person_editing_the_row_keeps_a_bracketed_name_whole(tmp_path) -> None:
    from finesub.llm.knowledge.node.edit import edit_subject

    repo, sid = _repo(tmp_path, "彼女、お借りします")
    text = render_subject(repo.store, sid, mode="human", preview="full")
    old = "- [别名] 「彼女、お借りします」"
    assert old in text
    edit_subject(repo, repo.store.node(sid), text.replace(old, old + "、かのかり"), refresh=False)
    assert _aliases(repo, sid) == ["かのかり", "彼女、お借りします"]


def test_an_unpaired_corner_bracket_does_not_swallow_the_list() -> None:
    from finesub.llm.knowledge.base import _split_names, join_names

    assert list(_split_names(join_names(["A「B", "D"]))) == ["A「B", "D"]
    assert list(_split_names("「彼女、お借りします、かのかり、GX")) == ["「彼女", "お借りします", "かのかり", "GX"]


def test_the_h1_handle_registers_a_misheard_form_of_the_entry_name(tmp_path) -> None:
    """Cold read round 2, P175 (owner 2026-10-05): a misheard form of the entry
    name itself goes on the entry through its H1 handle, not into `[别名]`."""

    repo, sid = _repo(tmp_path, "GX")
    text, report = _apply(repo, sid, {"op": "add_item", "id": "@k1", "field": "misheard", "value": "ゆうぎX", "reason": "r"})
    assert "# 游戏X  <!-- @k1 -->" in text and report.rev is not None
    items = [(i.field, i.value) for i in repo.store.all_items() if i.local_id == sid]
    assert ("misheard", "ゆうぎX") in items and _aliases(repo, sid) == ["GX"]


@pytest.mark.parametrize(
    "proposal",
    [
        {"op": "add_item", "id": "@a1", "field": "aliases", "value": "Ｓｈａｄｏｗ", "reason": "r"},
        {"op": "update", "id": "@a1", "line": "[别名] GX、Shadow", "reason": "r"},
    ],
)
def test_a_name_matching_a_tentative_alias_is_no_change(tmp_path, proposal) -> None:
    """Not an `add_item` the engine would reject as already present: the
    tentative alias stays tentative and nothing is written."""

    repo, sid = _repo(tmp_path, "GX", tentative=("Shadow",))
    _text, report = _apply(repo, sid, proposal)
    assert report.rev is None
    assert [r.reason for r in report.skipped] == ["no change"]
    assert [(i.value, i.maturity) for i in repo.store.items_of(sid) if i.field == "aliases"] == [
        ("GX", "normal"), ("Shadow", "tentative"),
    ]


# ---- a term line's alias column: the same base, model and person alike ----


def _term_repo(tmp_path):  # type: ignore[no-untyped-def]
    from finesub.llm.knowledge.node.repo import KnowledgeRepo

    repo = KnowledgeRepo.open(tmp_path)
    with repo.store.begin("import") as txn:
        txn.create_node("S", "subject", {"surface": "游戏", "intro": "", "category": "common", "section_order": ["角色"]})
        txn.create_node("T", "term", {"surface": "アリス", "zh": "爱丽丝", "reading": "", "desc": "主角"})
        txn.create_membership("M", "S", "T", "角色", 0)
        txn.create_item("I", "T", "aliases", "Alice")
        txn.create_item("J", "T", "aliases", "Shadow", maturity="tentative")
    return repo


def _term_aliases(repo):  # type: ignore[no-untyped-def]
    return sorted((i.value, i.maturity) for i in repo.store.items_of("T") if i.field == "aliases")


def _rewrite_term(repo, line):  # type: ignore[no-untyped-def]
    handles = HandleMap()
    repo.entry_prompt_text("S", handles)
    term = next(h for h, (ident, _) in handles.nodes.items() if ident == "T")
    proposal = {"op": "update", "id": term, "line": line, "reason": "r"}
    return apply_model_proposals(
        f"<knowledge_proposals>\n{json.dumps(proposal, ensure_ascii=False)}\n</knowledge_proposals>",
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )


def _edit_term(repo, line):  # type: ignore[no-untyped-def]
    from finesub.llm.knowledge.node.edit import edit_subject

    text = repo.entry_text("S")
    assert "アリス|爱丽丝|Alice|主角" in text  # the tentative alias is not rendered
    return edit_subject(repo, repo.store.node("S"), text.replace("アリス|爱丽丝|Alice|主角", line))


@pytest.mark.parametrize("write", [_rewrite_term, _edit_term], ids=["model", "person"])
@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("アリス|爱丽丝|Ally|主角", [("Ally", "normal"), ("Shadow", "tentative")]),
        ("アリス|爱丽丝||主角", [("Shadow", "tentative")]),
        ("アリス|爱丽丝|Ｓｈａｄｏｗ|主角", [("Shadow", "tentative")]),
    ],
    ids=["replace", "clear", "drop-and-name-the-tentative"],
)
def test_a_term_line_rewrite_leaves_tentative_aliases_alone(tmp_path, write, line, expected) -> None:
    repo = _term_repo(tmp_path)
    write(repo, line)
    assert _term_aliases(repo) == expected


def test_a_term_line_naming_only_the_tentative_alias_is_no_change(tmp_path) -> None:
    repo = _term_repo(tmp_path)
    report = _rewrite_term(repo, "アリス|爱丽丝|Alice、Shadow|主角")
    assert report.rev is None and [r.reason for r in report.skipped] == ["no change"]
    before = repo.rev
    edited = _edit_term(repo, "アリス|爱丽丝|Alice、Shadow|主角")
    assert edited.rejected == [] and repo.rev == before
    assert _term_aliases(repo) == [("Alice", "normal"), ("Shadow", "tentative")]
