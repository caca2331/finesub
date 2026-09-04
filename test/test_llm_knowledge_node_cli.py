"""Human edit surface: CLI (new/edit/retire/log), round-trip diff, revert/restore."""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from finesub.llm.knowledge.maintain import main
from finesub.llm.knowledge.node.edit import EditError, edit_subject
from finesub.llm.knowledge.node.history import RevertError, restore_node, revert_revision
from finesub.llm.knowledge.node.proposals import apply_model_proposals
from finesub.llm.knowledge.node.repo import KnowledgeRepo


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    root.mkdir()
    return root


def _new_game(root: Path) -> KnowledgeRepo:
    assert (
        main(
            ["--root", str(root), "new", "common", "测试游戏", "--intro", "一个游戏",
             "--type", "游戏", "--alias", "TG"]
        )
        == 0
    )
    return KnowledgeRepo.open(root)


def _entry_text(repo: KnowledgeRepo) -> tuple[str, str]:
    resolved = repo.resolve("测试游戏")
    assert resolved is not None
    return resolved.subject_id, repo.entry_text(resolved.subject_id)


def test_cli_new_show_log(tmp_path, capsys) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    assert text.startswith("# 测试游戏\n一个游戏\n")
    assert "- [本名] 测试游戏" in text  # 别名 is only materialized when non-empty
    # the alias lives in items and reaches the index from there
    assert "| TG |" in repo.index_text("common")
    assert main(["--root", str(root), "show", "TG"]) == 0
    assert "# 测试游戏" in capsys.readouterr().out
    assert main(["--root", str(root), "log"]) == 0
    out = capsys.readouterr().out
    assert "user" in out and "cli:new:测试游戏" in out


def test_edit_round_trip_updates_creates_removes_and_syncs_items(tmp_path) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)

    # add a free section with one term line (misheard in the description,
    # which is transport syntax: it becomes an item and leaves the desc)
    edited = text.replace(
        "\n## 元数据",
        "\n## 角色\n\nアリス|爱丽丝|Alice|主角（误听:「アリズ」）\nボブ|鲍勃||配角\n\n## 元数据",
    )
    report = edit_subject(repo, repo.store.node(subject_id), edited)
    assert report.changed and report.created == 2 and report.rejected == []
    terms = {n.payload["surface"]: n for n in repo.store.nodes_of_kind("term")}
    alice = terms["アリス"]
    values = {(i.field, i.value) for i in repo.store.items_of(alice.local_id)}
    assert ("aliases", "Alice") in values and ("misheard", "アリズ") in values
    assert "误听" not in repo.store.node(alice.local_id).payload["desc"]

    # rewrite the line: alias column change syncs items both ways; node id survives
    text2 = repo.entry_text(subject_id)
    assert "- アリス|爱丽丝|Alice|主角" in text2  # desc stored stripped; human line bulleted
    report2 = edit_subject(
        repo, repo.store.node(subject_id),
        text2.replace("アリス|爱丽丝|Alice|主角", "アリス|爱丽丝|Ally|主角"),
    )
    # alias-only rewrite is pure item churn now: no payload update op
    assert report2.changed and report2.updated == 0 and report2.created == 0
    alias_values = [i.value for i in repo.store.items_of(alice.local_id) if i.field == "aliases"]
    assert alias_values == ["Ally"]
    # misheard is an add-only cache: the rewritten line no longer mentions it, item stays
    assert [i.value for i in repo.store.items_of(alice.local_id) if i.field == "misheard"] == ["アリズ"]

    # swap the two lines: pure reorder keeps both node ids
    text3 = repo.entry_text(subject_id)
    bob = terms["ボブ"]
    swapped = text3.replace(
        "- アリス|爱丽丝|Ally|主角\n- ボブ|鲍勃||配角",
        "- ボブ|鲍勃||配角\n- アリス|爱丽丝|Ally|主角",
    )
    report3 = edit_subject(repo, repo.store.node(subject_id), swapped)
    assert report3.changed and report3.created == 0 and report3.removed == 0
    order = [m.child_id for m in repo.store.children(subject_id) if m.section == "角色"]
    assert order == [bob.local_id, alice.local_id]

    # delete one line -> node retired
    text4 = repo.entry_text(subject_id)
    report4 = edit_subject(
        repo, repo.store.node(subject_id), text4.replace("- ボブ|鲍勃||配角\n", "")
    )
    assert report4.changed and report4.removed == 1
    assert repo.store.node(bob.local_id) is None

    # round-trip without changes is a no-op (no new revision)
    before = repo.rev
    report5 = edit_subject(repo, repo.store.node(subject_id), repo.entry_text(subject_id))
    assert not report5.changed and repo.rev == before


def test_edit_syncs_subject_derived_retrieval_fields(tmp_path) -> None:
    """Changing 档案 本名/别名 rows updates native_names/reading/alias items."""

    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    edited = text.replace(
        "- [本名] 测试游戏",
        "- [本名] テストゲーム（てすと）/ 测试游戏\n- [别名] TG、Test",
    )
    report = edit_subject(repo, repo.store.node(subject_id), edited)
    assert report.changed
    payload = repo.store.node(subject_id).payload
    assert "テストゲーム" in payload["native_names"]
    assert payload["reading"] == "てすと"
    assert repo.resolve("テストゲーム") is not None
    assert repo.resolve("Test") is not None
    assert "Test" in repo.index_text("common")

    # removing an alias from the 别名 row retires its item
    text2 = repo.entry_text(subject_id)
    edit_subject(repo, repo.store.node(subject_id), text2.replace("[别名] TG、Test", "[别名] Test"))
    assert repo.resolve("TG") is None
    assert repo.resolve("Test") is not None


def test_unicode_equivalent_alias_rewrite_keeps_items(tmp_path) -> None:
    """NFKC-equivalent rewrites are the same alias: no add, no remove."""

    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)

    # subject level: the 档案 别名 row rewritten in fullwidth moves nothing
    report = edit_subject(
        repo, repo.store.node(subject_id), text.replace("- [别名] TG", "- [别名] ＴＧ")
    )
    assert not report.changed and report.rejected == []
    assert [i.value for i in repo.store.items_of(subject_id) if i.field == "aliases"] == ["TG"]
    assert repo.resolve("ＴＧ") is not None

    # term level: the alias column rewritten in fullwidth
    text2 = repo.entry_text(subject_id)
    edit_subject(
        repo,
        repo.store.node(subject_id),
        text2.replace("\n## 元数据", "\n## 角色\n\nアリス|爱丽丝|Alice|主角\n\n## 元数据"),
    )
    alice = next(n for n in repo.store.nodes_of_kind("term") if n.payload["surface"] == "アリス")
    text3 = repo.entry_text(subject_id)
    report3 = edit_subject(
        repo,
        repo.store.node(subject_id),
        text3.replace("アリス|爱丽丝|Alice|主角", "アリス|爱丽丝|Ａｌｉｃｅ|主角"),
    )
    # an NFKC-equivalent alias rewrite moves nothing: same items, no revision
    assert not report3.changed
    assert [i.value for i in repo.store.items_of(alice.local_id) if i.field == "aliases"] == ["Alice"]


def test_edit_moves_line_across_sections_keeping_identity(tmp_path) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    edited = text.replace(
        "\n## 元数据",
        "\n## 角色\n\nアリス|爱丽丝||主角\n\n## 术语\n\n\n## 元数据",
    )
    edit_subject(repo, repo.store.node(subject_id), edited)
    alice = next(n for n in repo.store.nodes_of_kind("term") if n.payload["surface"] == "アリス")
    # move the line by editing the rendered text line-wise (the full preview
    # puts a guidance comment under each heading, so substring surgery on
    # "## 角色\n\n- …" no longer describes the file)
    text2 = repo.entry_text(subject_id)
    entry_line = "- アリス|爱丽丝||主角"
    lines = [row for row in text2.splitlines() if row != entry_line]
    at = lines.index("## 术语") + 1
    lines[at:at] = ["", entry_line]
    moved = "\n".join(lines) + "\n"
    report = edit_subject(repo, repo.store.node(subject_id), moved)
    assert report.changed and report.created == 0 and report.removed == 0
    assert repo.store.node(alice.local_id) is not None
    parents = repo.store.parents(alice.local_id)
    assert [m.section for m in parents] == ["术语"]


def test_edit_wraps_parse_errors(tmp_path) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, _ = _entry_text(repo)
    with pytest.raises(EditError):
        edit_subject(repo, repo.store.node(subject_id), "not an entry at all\n")


def test_examples_tree_round_trips_as_noop(tmp_path) -> None:
    """The tracked sample KB is written in the v3 grammar: ingesting it and
    re-rendering it must be a fixed point, in both directions."""

    examples = Path(__file__).resolve().parents[1] / "examples" / "knowledge"
    root = _root(tmp_path)
    repo = KnowledgeRepo.open(root)
    for category, name, intro in (
        ("streamer", "四月一日べレト", "个人势日本女仆 VTuber"),
        ("common", "原神", "HoYoverse 开放世界 RPG"),
    ):
        args = ["--root", str(root), "new", category, name, "--intro", intro]
        if category == "common":
            args += ["--type", "游戏"]
        assert main(args) == 0
        repo = KnowledgeRepo.open(root)
        subject = repo.resolve(name)
        text = (examples / category / f"{name}.md").read_text(encoding="utf-8")
        edit_subject(repo, repo.store.node(subject.subject_id), text)
        repo = KnowledgeRepo.open(root)

        def body(raw: str) -> str:
            return raw.split("## 元数据")[0]  # the date is harness-owned

        # what the sample says is exactly what the store renders back
        assert body(repo.entry_text(subject.subject_id)) == body(text)
        before = repo.rev
        report = edit_subject(repo, repo.store.node(subject.subject_id), text)
        assert not report.changed, name
        assert repo.rev == before


def test_edit_rejects_illegal_sections_and_rename_collisions(tmp_path) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    assert main(["--root", str(root), "new", "common", "别的游戏", "--intro", "另一款", "--type", "游戏"]) == 0
    with pytest.raises(EditError, match="collides"):
        edit_subject(repo, repo.store.node(subject_id), text.replace("# 测试游戏", "# 别的游戏"))
    streamer_root_entry = main(
        ["--root", str(root), "new", "streamer", "星野灯", "--intro", "个人势"]
    )
    assert streamer_root_entry == 0
    resolved = repo.resolve("星野灯")
    streamer_text = repo.entry_text(resolved.subject_id)
    with pytest.raises(EditError, match="not allowed"):
        edit_subject(
            repo,
            repo.store.node(resolved.subject_id),
            streamer_text.replace("\n## 元数据", "\n## 自定义\n\n随便\n\n## 元数据"),
        )


def test_revert_and_restore(tmp_path) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)

    edited = text.replace("\n## 元数据", "\n## 角色\n\nアリス|爱丽丝||主角\n\n## 元数据")
    report = edit_subject(repo, repo.store.node(subject_id), edited)
    edit_rev = report.rev
    assert "アリス" in repo.entry_text(subject_id)

    rev = revert_revision(repo.store, edit_rev)
    assert rev == repo.store.current_rev()
    assert "アリス" not in repo.entry_text(subject_id)
    # reverting the revert brings the line back (compensation is symmetric)
    revert_revision(repo.store, rev)
    assert "アリス" in repo.entry_text(subject_id)

    # a later touch on the same entity blocks the earlier revert
    text_now = repo.entry_text(subject_id)
    edit2 = edit_subject(
        repo, repo.store.node(subject_id), text_now.replace("アリス|爱丽丝||主角", "アリス|爱丽丝||女主角")
    )
    with pytest.raises(RevertError):
        revert_revision(repo.store, edit_rev)
    assert edit2.rev == repo.store.current_rev()

    # retire the whole entry, then restore it under the same local_id;
    # the retire cascades over items, so the alias goes dark and comes back
    assert main(["--root", str(root), "retire", "测试游戏"]) == 0
    assert repo.resolve("测试游戏") is None
    assert repo.resolve("TG") is None
    restore_rev = restore_node(repo.store, subject_id)
    assert restore_rev == repo.store.current_rev()
    assert repo.resolve("测试游戏") is not None
    assert repo.resolve("TG") is not None
    assert "女主角" in repo.entry_text(subject_id)


def test_cli_edit_from_file_and_refresh(tmp_path, capsys) -> None:
    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    edited_file = tmp_path / "edited.md"
    edited_file.write_text(
        text.replace("一个游戏", "一个很好玩的游戏"), encoding="utf-8"
    )
    assert main(["--root", str(root), "edit", "测试游戏", "--file", str(edited_file)]) == 0
    assert "rev " in capsys.readouterr().out
    assert repo.store.node(subject_id).payload["intro"] == "一个很好玩的游戏"
    rendered = root / "rendered" / "common" / "测试游戏.md"
    assert "一个很好玩的游戏" in rendered.read_text(encoding="utf-8")
    assert main(["--root", str(root), "refresh"]) == 0


def test_model_injection_paths_get_prompt_projection(tmp_path) -> None:
    """Round 12: load_entry_texts feeds research/search/correction prompts —
    it must carry the PROMPT projection (bare lines), while entry_text keeps
    the bulleted human face for rendered/ and the editor."""

    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    edit_subject(
        repo, repo.store.node(subject_id),
        text.replace("\n## 元数据", "\n## 角色\n\nアリス|爱丽丝|Alice|主角\n\n## 元数据"),
    )
    assert "- アリス|爱丽丝|Alice|主角" in repo.entry_text(subject_id)  # human face
    found, _missing = repo.load_entry_texts(["测试游戏"])
    injected = found["测试游戏"]
    assert "\nアリス|爱丽丝|Alice|主角" in injected  # bare line, no bullet
    assert "\n- " not in injected
    assert injected == repo.entry_injection_text(subject_id)


def test_alias_row_absent_leaves_items_alone_but_emptied_row_clears(tmp_path) -> None:
    """Review 2026-08-29 P1-1: the row is an items projection, so its ABSENCE
    and its EMPTINESS parse identically — only presence separates them."""

    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, text = _entry_text(repo)
    assert [i.value for i in repo.store.items_of(subject_id) if i.field == "aliases"] == ["TG"]

    # deleting the rendered row is not a request to wipe the retrieval index
    without = "\n".join(l for l in text.splitlines() if not l.startswith("- [别名]")) + "\n"
    edit_subject(repo, repo.store.node(subject_id), without)
    assert [i.value for i in repo.store.items_of(subject_id) if i.field == "aliases"] == ["TG"]

    # emptying its body IS
    text2 = repo.entry_text(subject_id)
    edit_subject(repo, repo.store.node(subject_id), text2.replace("- [别名] TG", "- [别名]"))
    assert [i.value for i in repo.store.items_of(subject_id) if i.field == "aliases"] == []


def test_relabelling_onto_a_taken_label_is_refused(tmp_path) -> None:
    """Review 2026-08-29 P1-3: "one line per label per section" is an
    invariant, so an `update` that relabels onto a taken name must fail too —
    append_lines checking it locally is not enough."""

    root = _root(tmp_path)
    assert main(["--root", str(root), "new", "streamer", "某主播", "--intro", "x"]) == 0
    repo = KnowledgeRepo.open(root)
    subject_id = repo.resolve("某主播").subject_id
    seeded = repo.entry_text(subject_id).replace("- [人设]", "- [人设] 冷静\n- [外观] 银发")
    edit_subject(repo, repo.store.node(subject_id), seeded)

    clash = repo.entry_text(subject_id).replace("- [外观] 银发", "- [人设] 银发")
    with pytest.raises(EditError, match="重复"):
        edit_subject(repo, repo.store.node(subject_id), clash)
    labels = [n.payload.get("label") for n in repo.store.nodes_of_kind("note")]
    assert labels.count("人设") == 1


def test_a_line_can_lose_its_label(tmp_path) -> None:
    """Review 2026-08-29 P2-1: a diff over the NEW payload alone never clears
    `payload.label`, so the label came back on the next render."""

    root = _root(tmp_path)
    assert main(["--root", str(root), "new", "streamer", "某主播", "--intro", "x"]) == 0
    repo = KnowledgeRepo.open(root)
    subject_id = repo.resolve("某主播").subject_id
    seeded = repo.entry_text(subject_id).replace("- [人设]", "- [人设] 冷静")
    edit_subject(repo, repo.store.node(subject_id), seeded)

    unlabelled = repo.entry_text(subject_id).replace("- [人设] 冷静", "- 冷静")
    edit_subject(repo, repo.store.node(subject_id), unlabelled)
    text = repo.entry_text(subject_id)
    assert "- 冷静" in text and "[人设] 冷静" not in text


def test_an_alias_label_outside_its_own_section_is_an_ordinary_line(tmp_path) -> None:
    """Review 2026-08-29 P2-3: `[别名]` was treated as THE structural alias row
    wherever it appeared, so writing one as a custom label in another section
    made the 档案 row look present-but-empty and wiped the retrieval index."""

    root = _root(tmp_path)
    assert main(["--root", str(root), "new", "streamer", "某主播", "--intro", "x",
                 "--alias", "TG"]) == 0
    repo = KnowledgeRepo.open(root)
    subject_id = repo.resolve("某主播").subject_id
    text = repo.entry_text(subject_id)

    lines = [l for l in text.splitlines() if not l.startswith("- [别名]")]
    lines.insert(lines.index("## 待归类") + 1, "- [别名] 粉丝私下这么叫，出处待查")
    edit_subject(repo, repo.store.node(subject_id), "\n".join(lines) + "\n")

    assert [i.value for i in repo.store.items_of(subject_id) if i.field == "aliases"] == ["TG"]
    parked = [n for n in repo.store.nodes_of_kind("note") if n.payload.get("label") == "别名"]
    assert len(parked) == 1 and parked[0].payload["text"].startswith("粉丝私下")


# ---------------------------------------------------------------------------
# ingest: a material distilled into one entry (2026-09-01)


def _material(tmp_path: Path, text: str) -> str:
    path = tmp_path / "material.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_ingest_dry_run_renders_a_prompt_and_calls_nothing(tmp_path, monkeypatch, capsys) -> None:
    """Dry-run is the default for every LLM entry point here, so the bare
    command must not be able to reach a model even by accident."""

    import finesub.llm.client as llm_client_module

    root = _root(tmp_path)
    _new_game(root)
    monkeypatch.setattr(
        llm_client_module, "RoleClient",
        lambda *a, **k: pytest.fail("dry-run must not construct a client"),
    )
    assert main(["--root", str(root), "ingest", "--subject", "测试游戏",
                 "--material", _material(tmp_path, "一段素材")]) == 0
    assert "dry-run: ingest prompt rendered" in capsys.readouterr().out


def test_ingest_says_what_to_do_when_the_entry_does_not_exist(tmp_path, capsys) -> None:
    """Routing a material to the right subject is an unspecified judgement
    call, so the command refuses rather than guessing -- and names the way
    forward instead of just failing."""

    root = _root(tmp_path)
    _new_game(root)
    assert main(["--root", str(root), "ingest", "--subject", "没这个条目",
                 "--material", _material(tmp_path, "x")]) == 1
    assert "new" in capsys.readouterr().err


def test_ingest_reads_the_material_from_stdin(tmp_path, monkeypatch, capsys) -> None:
    import io

    root = _root(tmp_path)
    _new_game(root)
    monkeypatch.setattr("sys.stdin", io.StringIO("从管道进来的素材"))
    assert main(["--root", str(root), "ingest", "--subject", "测试游戏",
                 "--material", "-"]) == 0
    assert "8 chars of material" in capsys.readouterr().out


def test_ingest_refuses_an_empty_material(tmp_path, capsys) -> None:
    root = _root(tmp_path)
    _new_game(root)
    assert main(["--root", str(root), "ingest", "--subject", "测试游戏",
                 "--material", _material(tmp_path, "   ")]) == 1
    assert "empty" in capsys.readouterr().err


def test_repair_no_longer_carries_the_material_mode(tmp_path) -> None:
    """`ingest` took it over: a different input deserves a different command,
    and sharing one made both harder to describe."""

    root = _root(tmp_path)
    _new_game(root)
    with pytest.raises(SystemExit):  # argparse: unrecognised argument
        main(["--root", str(root), "repair", "--material", "x"])


def test_the_ingest_prompt_carries_the_standard_and_the_steer(tmp_path) -> None:
    """What the round is allowed to write down, and what the user's own line
    may and may not move."""

    from finesub.llm.knowledge.node.repair import render_repair_prompt

    root = _root(tmp_path)
    repo = _new_game(root)
    resolved = repo.resolve("测试游戏")
    prompt = render_repair_prompt(
        repo, resolved.subject_id, material="素材正文", user_prompt="只要术语"
    )
    # rule 0: what the knowledge base is for at all
    assert "让 ASR 听对" in prompt and "让字幕写对" in prompt
    # no reference text this round -> no invented mishearings
    assert "没有 ASR 文本作对照" in prompt
    # the user's line steers, it does not lower the bar
    assert "只要术语" in prompt and "不能越过上面的收录标准" in prompt
    # the material is untrusted input, and the prompt says so
    assert "不可信的外部文本" in prompt

    bare = render_repair_prompt(repo, resolved.subject_id, material="素材正文")
    assert "（用户没有额外交代。）" in bare


def test_revert_undoes_a_whole_multi_entity_revision(tmp_path) -> None:
    """One apply is one rev, and a rev routinely spans several entities: a new
    subject, lines under it, alias items, lines added to another entry. The
    compensation has to undo ALL of it — the existing coverage reverted a
    single-entry edit, which cannot tell "reverts the revision" from "reverts
    the first thing it finds"."""

    root = _root(tmp_path)
    repo = _new_game(root)
    before = repo.rev
    existing_id, existing_text = _entry_text(repo)

    proposals = "\n".join([
        json.dumps({"op": "create_entry", "category": "common", "entry": "新游戏",
                    "entry_type": "游戏", "intro": "另一个游戏",
                    "aliases": ["新作"], "reason": "库中没有"}, ensure_ascii=False),
        json.dumps({"op": "append_lines", "category": "common", "entry": "新游戏",
                    "section": "角色", "content": "ボブ|鲍勃||配角",
                    "reason": "材料"}, ensure_ascii=False),
        json.dumps({"op": "append_lines", "category": "common", "entry": "测试游戏",
                    "section": "角色", "content": "キャロル|卡罗尔||另一个配角",
                    "reason": "材料"}, ensure_ascii=False),
    ])
    report = apply_model_proposals(
        f"<knowledge_proposals>\n{proposals}\n</knowledge_proposals>",
        repo=repo, task_id="multi", knowledge_read_rev=repo.rev,
    )
    rev = report.to_dict()["rev"]
    assert rev == before + 1  # four ops, ONE revision
    assert repo.resolve("新游戏") is not None
    assert repo.resolve("新作") is not None  # the alias item
    assert "キャロル" in repo.entry_text(existing_id)

    revert_revision(repo.store, rev)

    assert repo.resolve("新游戏") is None  # the subject
    assert repo.resolve("新作") is None  # ...its alias item
    assert "ボブ" not in "".join(  # ...and the line under it
        repo.entry_text(s.local_id) for s in repo.subjects()
    )
    assert "キャロル" not in repo.entry_text(existing_id)  # the OTHER entry too
    assert repo.entry_text(existing_id) == existing_text


def test_rendered_follows_a_revert(tmp_path) -> None:
    """`rendered/` is what a human reads. A revert that fixed the store and
    left the markdown showing the reverted line would look exactly like the
    revert not working."""

    root = _root(tmp_path)
    repo = _new_game(root)
    subject_id, _ = _entry_text(repo)
    repo.refresh_rendered()
    path = root / "rendered" / "common" / "测试游戏.md"
    assert "アリス" not in path.read_text(encoding="utf-8")

    edited = repo.entry_text(subject_id).replace(
        "\n## 元数据", "\n## 角色\n\n- アリス|爱丽丝||主角\n\n## 元数据"
    )
    rev = edit_subject(repo, repo.store.node(subject_id), edited).rev
    assert "アリス" in path.read_text(encoding="utf-8")

    assert main(["--root", str(root), "revert", str(rev)]) == 0
    assert "アリス" not in repo.entry_text(subject_id)
    assert "アリス" not in path.read_text(encoding="utf-8")

