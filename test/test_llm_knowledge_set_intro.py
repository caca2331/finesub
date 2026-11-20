"""`set_intro` (owner 2026-10-01): the model may replace an entry's intro, under
the same one-line rules `create_entry` now enforces too."""

from __future__ import annotations

import json

import pytest

from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.envelope import Binding
from finesub.llm.knowledge.node.presets import CATEGORIES
from finesub.llm.knowledge.node.proposals import (
    apply_model_proposals,
    intro_problem,
    translate_model_proposals,
)
from finesub.llm.knowledge.node.render import HandleMap, render_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _new(root, category: str, key: str, intro: str = "一个游戏") -> None:
    argv = ["--root", str(root), "new", category, key, "--intro", intro]
    if category == "common":
        argv += ["--type", "游戏"]
    assert maintain_main(argv) == 0


def _repo(root) -> KnowledgeRepo:
    KnowledgeRepo.forget(root)
    return KnowledgeRepo.open(root)


def _handles(repo: KnowledgeRepo, subject_id: str) -> tuple[HandleMap, list[Binding]]:
    handles = HandleMap()
    render_subject(repo.store, subject_id, mode="prompt", handles=handles)
    return handles, [Binding(**b) for b in handles.bindings()]


def _skipped(repo: KnowledgeRepo, proposal: dict, bindings=(), allow=None) -> list[str]:  # type: ignore[no-untyped-def]
    kwargs = {"allow_categories": allow} if allow is not None else {}
    _ops, report, _drafts, _bindings = translate_model_proposals(
        [proposal], repo=repo, knowledge_read_rev=repo.rev, bindings=list(bindings), **kwargs
    )
    return [r.reason for r in report.skipped]


def test_the_model_replaces_the_intro_and_the_head_is_marked(tmp_path, monkeypatch) -> None:
    _new(tmp_path, "common", "游戏X")  # `new` is a human revision: the head is endorsed
    repo = _repo(tmp_path)
    subject = repo.subjects("common")[0]
    handles, _ = _handles(repo, subject.local_id)
    monkeypatch.setattr(KnowledgeRepo, "today", lambda self: "2099-01-01")
    report = apply_model_proposals(
        json.dumps({"op": "set_intro", "entry": "@k1", "intro": "  一款开放世界游戏  ", "reason": "素材里说明了"},
                   ensure_ascii=False),
        repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles,
    )
    assert report.rev is not None and not report.skipped
    after = repo.store.node(subject.local_id)
    assert after.payload["intro"] == "一款开放世界游戏"
    assert after.payload["updated_date"] == "2099-01-01"
    text = repo.entry_text(subject.local_id)
    assert "\n一款开放世界游戏\n<!-- 人改版 " in text and "简介「一个游戏」" in text


@pytest.mark.parametrize(
    ("intro", "reason"),
    [
        ("第一行\n第二行", "intro must be one line"),
        ("一行\r", "intro must be one line"),
        ("   ", "intro required"),
        ("# 像个标题", "intro must not start with #"),
    ],
)
def test_bad_intros_are_refused_by_set_intro_and_create_entry(tmp_path, intro, reason) -> None:
    assert intro_problem(intro) == reason
    _new(tmp_path, "common", "游戏X")
    repo = _repo(tmp_path)
    assert _skipped(repo, {"op": "set_intro", "entry": "游戏X", "intro": intro, "reason": "r"}) == [reason]
    assert _skipped(
        repo,
        {"op": "create_entry", "category": "common", "entry": "游戏Y", "entry_type": "游戏",
         "intro": intro, "reason": "r"},
    ) == [reason]


def test_an_unchanged_intro_is_not_a_write(tmp_path) -> None:
    _new(tmp_path, "common", "游戏X")
    repo = _repo(tmp_path)
    assert _skipped(repo, {"op": "set_intro", "entry": "游戏X", "intro": " 一个游戏 ", "reason": "r"}) == [
        "intro unchanged"
    ]


def test_a_style_entry_intro_is_out_of_a_knowledge_update_grant(tmp_path) -> None:
    """Entry-level, like rename_entry: the handle grants the style's LINES."""

    _new(tmp_path, "style", "某字幕组", intro="一套风格")
    repo = _repo(tmp_path)
    subject = repo.subjects("style")[0]
    _, bindings = _handles(repo, subject.local_id)
    proposal = {"op": "set_intro", "entry": "@k1", "intro": "新简介", "reason": "r"}
    assert _skipped(repo, proposal, bindings) == ["set_intro is not allowed on a style entry here"]
    # a session granted every category (the repair sessions) may
    assert _skipped(repo, proposal, bindings, allow=CATEGORIES) == []


def test_a_renamed_entry_can_keep_its_old_name_as_an_alias(tmp_path) -> None:
    """The engine does not keep the old H1 (owner 2026-10-02: the model will
    judge); the ops contract says how, and the H1 handle takes `add_item`."""

    from finesub.llm.prompt_compose import render_knowledge_ops_contract

    assert "留作条目的别名" in render_knowledge_ops_contract(CATEGORIES)
    _new(tmp_path, "common", "游戏X")
    repo = _repo(tmp_path)
    subject = repo.subjects("common")[0]
    handles, _ = _handles(repo, subject.local_id)
    text = "\n".join(
        json.dumps(op, ensure_ascii=False)
        for op in (
            {"op": "rename_entry", "entry": "@k1", "new_key": "游戏X正式名", "reason": "r"},
            {"op": "add_item", "id": "@k1", "field": "aliases", "value": "游戏X", "reason": "旧名仍常见"},
        )
    )
    report = apply_model_proposals(
        text, repo=repo, task_id="t", knowledge_read_rev=repo.rev, handles=handles
    )
    assert report.rev is not None and not report.skipped
    resolved = repo.resolve("游戏X")
    assert resolved is not None and resolved.key == "游戏X正式名"
