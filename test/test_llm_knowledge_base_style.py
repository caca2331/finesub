"""The base-style migration (`node/base_style.py`): the pending list, the
proposal screen, the session with a canned client, the CLI and the warning."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from finesub.llm.knowledge import maintain
from finesub.llm.knowledge.node import base_style
from finesub.llm.knowledge.node.base_style import (
    PENDING_META_KEY,
    admit_proposals,
    parse_skipped,
    pending_styles,
    render_base_style_prompt,
    run_base_style_session,
    seed_pending,
)
from finesub.llm.knowledge.node.render import HandleMap
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.node.store import KnowledgeStore
from finesub.llm.knowledge.style import load_style_entries
from finesub.reporting import reporting_to

_CONVENTIONS = [
    "[口语化] 主播杂谈用直播社区的口语习惯重写，不逐字直译。",
    "[括号注释] 提到小众典故时用简短括号注释补足背景。",
    "[语气词] 语气词一律删掉。",
]
_ORDER = "[语序] 可为中文习惯微调语序，避免大幅重排。"
_LOANS = "[圈内借词] 圈内已通用的借词保留它的中文写法（如 ACG / VTuber 圈的「萌」），不留假名，也不换成语域更平的中文同义词。"
_LOANS_GOOD = "[圈内借词] 萌えな顔 → 表情相当萌"


def _append(section: str, content: str, entry: str = "@k1") -> dict:
    return {"op": "append_lines", "entry": entry, "section": section, "content": content, "reason": "r"}


def _old_style(root: Path, name: str = "某组", conventions: list[str] | None = None) -> tuple[KnowledgeRepo, str]:
    """A style written before the factory default: created, then recorded as
    pending the way the first open of an older store would have."""

    repo = KnowledgeRepo.open(root)
    proposals = [
        {"op": "create_entry", "category": "style", "entry": name, "intro": "某组的口味", "reason": "t"},
        {"op": "append_lines", "category": "style", "entry": name, "section": "约定",
         "content": "\n".join(conventions if conventions is not None else _CONVENTIONS), "reason": "t"},
    ]
    maintain._apply_user_proposal(repo, proposals, "test")
    repo.store.conn.execute("DELETE FROM meta WHERE key=?", (PENDING_META_KEY,))
    seed_pending(repo.store)
    resolved = repo.resolve_qualified(f"style/{name}")
    assert resolved is not None
    return repo, resolved.subject_id


class _Canned:
    """Answers with the given blocks; records the prompt it was sent."""

    def __init__(self, proposals: list[dict], skipped: list[dict] | None = None) -> None:
        self.proposals, self.skipped, self.prompt = proposals, skipped or [], ""

    def complete(self, role, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.prompt = messages[0]["content"]
        body = "\n".join(json.dumps(p, ensure_ascii=False) for p in self.proposals)
        skipped = "\n".join(json.dumps(s, ensure_ascii=False) for s in self.skipped)
        return SimpleNamespace(
            content=f"<reasoning>r</reasoning>\n<knowledge_proposals>\n{body}\n</knowledge_proposals>\n"
            f"<skipped>\n{skipped}\n</skipped>",
            requested_output_tokens=65_536,
            raw_response={},
        )


def _conventions(repo: KnowledgeRepo, subject_id: str) -> list[str]:
    text = repo.entry_injection_text(subject_id)
    block = text.split("## 约定", 1)[1].split("##", 1)[0]
    return [line for line in block.splitlines() if line.startswith("[")]


# ---- the pending list ------------------------------------------------------------------


def test_first_open_records_live_styles_and_a_new_style_is_not_pending(tmp_path) -> None:
    repo = KnowledgeRepo.open(tmp_path)
    assert repo.store.meta(PENDING_META_KEY) == "[]"  # an empty store records an empty list
    assert maintain.main(["--root", str(tmp_path), "new", "style", "新组", "--intro", "新的"]) == 0
    assert pending_styles(repo.store) == []

    repo, subject_id = _old_style(tmp_path / "old")
    assert pending_styles(repo.store) == [subject_id]


def test_a_second_open_does_not_rewrite_the_list(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    seed_pending(repo.store)
    assert pending_styles(repo.store) == [subject_id]
    repo.store.conn.execute("UPDATE meta SET value='[]' WHERE key=?", (PENDING_META_KEY,))
    seed_pending(repo.store)
    assert pending_styles(repo.store) == []


def test_concurrent_first_opens_write_the_key_once(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    repo.store.conn.execute("DELETE FROM meta WHERE key=?", (PENDING_META_KEY,))
    path = repo.store.path
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def _open() -> None:
        try:
            with KnowledgeStore(path) as store:
                barrier.wait()
                seed_pending(store)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=_open) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    rows = repo.store.conn.execute("SELECT value FROM meta WHERE key=?", (PENDING_META_KEY,)).fetchall()
    assert [json.loads(r["value"]) for r in rows] == [[subject_id]]


# ---- the screen ------------------------------------------------------------------------


def test_admission_takes_only_append_lines_into_this_style_and_its_three_sections(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    _other_repo, other_id = _old_style(tmp_path, "别组")
    maintain.main(["--root", str(tmp_path), "new", "streamer", "某组", "--intro", "同名主播"])
    handles = HandleMap()
    render_base_style_prompt(repo, subject_id, handles)
    other_handle = handles.node_handle(repo.store.node(other_id))
    proposals = [
        _append("约定", _LOANS),                                   # handle: kept
        _append("正例", _LOANS_GOOD, entry="某组"),                 # key: kept, its convention is in the batch
        {"op": "update", "id": "@k2", "line": "[口语化] 改", "reason": "r"},
        _append("约定", _ORDER, entry=other_handle),
        _append("约定", _ORDER, entry="别组"),
        {**_append("约定", _ORDER, entry="某组"), "category": "streamer"},
        _append("档案", "[作者] 出厂"),
        _append("待归类", _ORDER),
        {"op": "__invalid__", "raw": "{"},
        _append("约定", "[自创] 编出来的一条。"),                    # not a factory label
        _append("反例", "[语序] 甲 → 乙"),                           # no such factory example
        _append("反例", "[圈内借词] 萌えな顔 → 表情很可爱\n[语感] 甲 → 乙"),  # one line kept, one refused
    ]
    admitted, refused = admit_proposals(proposals, repo=repo, subject_id=subject_id, handles=handles)
    assert [(p["section"], p["content"]) for p in admitted] == [
        ("约定", _LOANS), ("正例", _LOANS_GOOD), ("反例", "[圈内借词] 萌えな顔 → 表情很可爱"),
    ]
    assert all(p["category"] == "style" for p in admitted)
    assert sum("只收这套风格" in r for r in refused) == 3
    assert sum("三节" in r for r in refused) == 2
    assert sum("标记不是出厂默认" in r for r in refused) == 3
    assert len(refused) == 10


def test_an_example_whose_convention_is_not_added_is_refused_not_rolled_back(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    handles = HandleMap()
    render_base_style_prompt(repo, subject_id, handles)
    admitted, refused = admit_proposals([_append("正例", _LOANS_GOOD)], repo=repo, subject_id=subject_id, handles=handles)
    assert admitted == [] and "例子的约定这次没加" in refused[0]


def test_prompt_carries_neither_the_update_standard_nor_the_full_grammar(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    prompt = render_base_style_prompt(repo, subject_id, HandleMap())
    assert "[口语化]" in prompt and "<factory_default>" in prompt and "[圈内借词]" in prompt
    assert "「约定」17 行、「正例」20 行、「反例」20 行" in prompt
    for absent in ("create_entry", "有没有参照物", "对谁都成立", "$"):
        assert absent not in prompt, absent


def test_parse_skipped_keeps_unparseable_rows_raw() -> None:
    text = (
        "<reasoning><skipped>not this</skipped></reasoning>\n"
        '<skipped>\n{"label":"不加注","section":"约定","why":"相反","line":"[括号注释] x"}\nfree text\n</skipped>'
    )
    assert parse_skipped(text) == [
        {"label": "不加注", "section": "约定", "why": "相反", "line": "[括号注释] x"},
        {"raw": "free text"},
    ]


# ---- the session -----------------------------------------------------------------------


def test_apply_adds_the_lines_keeps_the_users_and_crosses_the_style_off(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    client = _Canned(
        [
            _append("约定", f"{_ORDER}\n{_LOANS}"),
            _append("约定", "[语气词] 有表演价值的语气词保留。"),  # the user's label: the engine skips it
            _append("正例", _LOANS_GOOD),
        ],
        [{"label": "不加注", "section": "约定", "why": "相反", "line": "[括号注释] …"}],
    )
    result = run_base_style_session(repo, subject_id, client=client, apply=True)
    assert result["apply_report"]["rolled_back"] is False and result["refused"] == []
    assert sorted(result["added"]) == [("正例", _LOANS_GOOD), ("约定", _LOANS), ("约定", _ORDER)]
    assert sorted(_conventions(repo, subject_id)) == sorted([*_CONVENTIONS, _ORDER, _LOANS])
    assert any("已存在" in r["reason"] for r in result["apply_report"]["skipped"])
    assert result["skipped"][0]["label"] == "不加注"
    assert result["kept_pending"] == "" and pending_styles(repo.store) == []


@pytest.mark.parametrize(
    "reply, why",
    [
        ("<reasoning>r</reasoning>\n<skipped>\n</skipped>", "回答缺"),  # no proposals block
        ("", "回答缺"),
        (None, "拒收"),  # a refused proposal
    ],
    ids=["no-proposals-block", "empty", "refused"],
)
def test_a_broken_or_refused_answer_keeps_the_style_pending(tmp_path, reply, why) -> None:
    repo, subject_id = _old_style(tmp_path)
    client = _Canned([{"op": "remove", "id": "@k3", "reason": "越权"}])
    if reply is not None:
        client.complete = lambda role, messages, **kw: SimpleNamespace(  # type: ignore[method-assign]
            content=reply, requested_output_tokens=65_536, raw_response={})
    result = run_base_style_session(repo, subject_id, client=client, apply=True)
    assert why in result["kept_pending"]
    assert pending_styles(repo.store) == [subject_id]
    if why == "回答缺":
        assert result["refused"] == []  # not screened line by line


def test_an_execute_without_apply_writes_nothing(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    rev = repo.rev
    client = _Canned([_append("约定", _ORDER)])
    result = run_base_style_session(repo, subject_id, client=client)
    assert result["proposals"] and repo.rev == rev and pending_styles(repo.store) == [subject_id]


def test_nothing_added_still_crosses_off_and_a_rollback_does_not(tmp_path) -> None:
    repo, subject_id = _old_style(tmp_path)
    run_base_style_session(repo, subject_id, client=_Canned([]), apply=True)
    assert pending_styles(repo.store) == []

    full = [f"[规则{i}] 第 {i} 条。" for i in range(20)]
    repo, subject_id = _old_style(tmp_path / "full", conventions=full)
    client = _Canned([_append("约定", _ORDER)])
    result = run_base_style_session(repo, subject_id, client=client, apply=True)
    assert result["apply_report"]["rolled_back"] is True and result["kept_pending"] == "整批回滚"
    assert pending_styles(repo.store) == [subject_id]


# ---- the CLI ---------------------------------------------------------------------------


def test_cli_dry_run_renders_and_calls_nothing(tmp_path, monkeypatch, capsys) -> None:
    _old_style(tmp_path)
    monkeypatch.setattr(maintain, "_role_client", lambda _m: pytest.fail("dry-run called a model"))
    assert maintain.main(["--root", str(tmp_path), "adopt-base-style"]) == 0
    out = capsys.readouterr().out
    assert "— 某组" in out and "dry-run" in out
    assert "adopt-base-style" in maintain._MUTATING


def test_cli_apply_prints_added_and_skipped_and_then_has_nothing_left(tmp_path, monkeypatch, capsys) -> None:
    _old_style(tmp_path)
    client = _Canned(
        [_append("约定", _ORDER)],
        [{"label": "不加注", "section": "约定", "why": "相反", "line": "[括号注释] 提到小众典故时…"}],
    )
    monkeypatch.setattr(maintain, "_role_client", lambda _m: client)
    assert maintain.main(["--root", str(tmp_path), "adopt-base-style", "--execute", "--apply"]) == 0
    out = capsys.readouterr().out
    assert f"加了（约定）：{_ORDER}" in out
    assert "没加 [不加注]（约定）：相反——[括号注释]" in out
    assert maintain.main(["--root", str(tmp_path), "adopt-base-style"]) == 0
    assert "没有待迁移的风格" in capsys.readouterr().out


def test_cli_subject_runs_a_style_off_the_list_and_refuses_a_non_style(tmp_path, monkeypatch, capsys) -> None:
    assert maintain.main(["--root", str(tmp_path), "new", "style", "新组", "--intro", "新的"]) == 0
    maintain.main(["--root", str(tmp_path), "new", "streamer", "某人", "--intro", "主播"])
    capsys.readouterr()
    assert maintain.main(["--root", str(tmp_path), "adopt-base-style", "--subject", "style/新组"]) == 0
    assert "— 新组" in capsys.readouterr().out
    assert maintain.main(["--root", str(tmp_path), "adopt-base-style", "--subject", "某人"]) == 1
    assert "不是风格" in capsys.readouterr().err


# ---- the warning -----------------------------------------------------------------------


def test_a_pending_style_warns_once_by_id_and_a_migrated_one_does_not(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(base_style, "_warned", set())
    repo, subject_id = _old_style(tmp_path)
    maintain.main(["--root", str(tmp_path), "edit", "style/某组", "--file", str(_renamed(repo, subject_id, tmp_path))])
    warnings: list[tuple[str, str]] = []

    class _Collect:
        def warning(self, code, message, **kwargs):  # type: ignore[no-untyped-def]
            warnings.append((code, kwargs.get("action", "")))

        def __getattr__(self, _name):  # type: ignore[no-untyped-def]
            return lambda *a, **k: None

    with reporting_to(_Collect()):
        load_style_entries(tmp_path, ["改名组"])
        load_style_entries(tmp_path, ["改名组"])
    assert [code for code, _ in warnings] == ["knowledge-style-not-migrated"]
    assert "adopt-base-style --subject style/改名组" in warnings[0][1]

    warnings.clear()
    monkeypatch.setattr(base_style, "_warned", set())
    run_base_style_session(repo, subject_id, client=_Canned([]), apply=True)
    with reporting_to(_Collect()):
        load_style_entries(tmp_path, ["改名组"])
    assert warnings == []


def _renamed(repo: KnowledgeRepo, subject_id: str, tmp_path: Path) -> Path:
    path = tmp_path / "renamed.md"
    text = repo.entry_text(subject_id).replace("# 某组", "# 改名组", 1).replace("[本名] 某组", "[本名] 改名组")
    path.write_text(text, encoding="utf-8")
    return path
