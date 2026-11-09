"""`--style auto` (docs/knowledge.md, 「style 类别」): task-level style selection.

Pinned: `auto` is an intent, not a name; the catalog is descriptions only and
lists every style entry of the store revision; the pick is forced -- the
preparation task's `<style_selection>` block must name a candidate (an invalid
or, with `default_style` in the store, empty pick is a parse error that ends
in a failed task, never a fallback); the orchestrator reads the task's record
first, pins the pick before any window starts and limits the post-task update
to the selected style; with nothing to choose it records the no-choice result
silently; and a fast window with a choice always picks itself (the window side
is in `test_llm_auto_style_window.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm import correction_translation as orchestration
from finesub.llm.correction_request import CorrectionRequest
from finesub.llm.client import LLMCallResult
from finesub.llm.knowledge.base import append_task_artifact
from finesub.llm.knowledge.maintain import main as maintain_main
from finesub.llm.knowledge.node.edit import edit_subject
from finesub.llm.knowledge.node.repo import KnowledgeRepo
from finesub.llm.knowledge.style import (
    StyleCatalog,
    StyleChoice,
    StyleContractError,
    StyleSelection,
    StyleSelectionError,
    load_style_catalog,
    parse_style_selection_block,
    render_style_block,
    resolve_style_keys,
    resolve_style_selection,
)
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages.prepare_task import run_prepare_task

TRANSCRIPT = "--- window 0001 ---\n1|こんにちは\n2|ゲームBを始めます\n"
PACK = {"general_context": {"global_summary": "游戏B 实况"}, "window_contexts": []}
#: A fast window that answers with the finished translation straight away --
#: no `<style_selection>`, because nothing asked it to translate in a style yet.
DIRECT_DELIVERY = (
    "<translated>\ntype|position|corrected_text|translation|conf|note\n"
    "sub|1|一。|一|8|\n</translated>\n<next_advice></next_advice>"
)


def _entry_text(key: str, intro: str, author: str, scope: str, rules: list[str]) -> str:
    lines = [f"# {key}", "", intro, "", "## 档案", "", f"- [本名] {key}", f"- [作者] {author}", f"- [适用范围] {scope}", ""]
    lines += ["## 约定", ""] + [f"- {r}" for r in rules] + [""]
    return "\n".join(lines)


def _store(tmp_path: Path) -> Path:
    root = tmp_path / "kb"
    maintain_main(["--root", str(root), "new", "style", "甲组", "--intro", "甲组的口味"])
    maintain_main(["--root", str(root), "new", "style", "default_style", "--intro", "通用约定"])
    KnowledgeRepo.forget(root)
    repo = KnowledgeRepo.open(root)
    for subject in repo.subjects("style"):
        key = subject.payload.get("surface")
        if key == "甲组":
            edit_subject(repo, subject, _entry_text("甲组", "甲组的口味", "甲组", "主播A 的全部投稿", ["[直角引号] 对话统一用「」"]))
        else:
            edit_subject(repo, subject, _entry_text("default_style", "通用约定", "维护者", "所有素材", ["[口语化] 用直播社区口语"]))
    KnowledgeRepo.forget(root)
    return root


def _delivery(selected: str | None, reason: str = "适用范围对得上") -> str:
    text = "<reasoning>看完</reasoning>\n<context_pack>\n" + json.dumps(PACK, ensure_ascii=False) + "\n</context_pack>"
    if selected is not None:
        text += "\n<style_selection>\n" + json.dumps({"selected": selected, "reason": reason}, ensure_ascii=False) + "\n</style_selection>"
    return text


class RecordingClient:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []

    def complete(self, role, messages, **kwargs):
        self.calls.append((role, [dict(m) for m in messages], kwargs))
        return LLMCallResult(content=self.responses.pop(0), role=role, model="fake", fallback_used=False, raw_response={})


# --- the intent ---------------------------------------------------------------


def test_auto_is_an_intent_not_a_name(tmp_path) -> None:
    selection = resolve_style_selection("auto", knowledge_root=tmp_path, config_path=tmp_path / "none.toml")
    assert selection.auto and selection.names == ()
    assert resolve_style_selection("AUTO", knowledge_root=tmp_path, config_path=tmp_path / "none.toml").auto
    with pytest.raises(StyleSelectionError, match="style/auto"):
        resolve_style_selection("auto,甲组", knowledge_root=tmp_path, config_path=tmp_path / "none.toml")
    # a real entry called auto is addressed by its qualified name, which is not the intent
    plain = resolve_style_selection("style/auto", knowledge_root=tmp_path, config_path=tmp_path / "none.toml")
    assert not plain.auto and plain.names == ("style/auto",)
    assert selection.resolved(("甲组",)).names == ("甲组",)
    auto_update = resolve_style_selection("auto", style_mode="update", knowledge_root=tmp_path, config_path=tmp_path / "none.toml")
    assert auto_update.resolved(("甲组",)).writable == ("甲组",)
    # The write is intended from the start, so the prerequisite check
    # (`--knowledge update` + `--refined-srt`) has to read `wants_write`:
    # `writable` is empty until the preparation task picks, which silently
    # skipped that warning for every `auto` run.
    assert auto_update.writable == () and auto_update.wants_write is True
    assert selection.wants_write is False  # `--style-mode read`


# --- the catalog and the choice ----------------------------------------------


def test_catalog_lists_descriptions_of_every_style_entry(tmp_path) -> None:
    assert not load_style_catalog(tmp_path / "missing")  # no store: no candidates, nothing created
    catalog = load_style_catalog(_store(tmp_path))
    assert [c.key for c in catalog.candidates] == ["default_style", "甲组"]
    jia = next(c for c in catalog.candidates if c.key == "甲组")
    assert (jia.intro, jia.author, jia.scope) == ("甲组的口味", "甲组", "主播A 的全部投稿")
    rendered = catalog.render()
    assert "- 甲组｜甲组的口味｜作者：甲组｜适用范围：主播A 的全部投稿" in rendered
    assert "[直角引号]" not in rendered  # descriptions only: the rules are not the catalog


def test_decide_never_falls_back(tmp_path) -> None:
    """auto-style-gaps plan 「已裁定」 8: a forced pick and a silent fallback
    cannot both hold. A name outside the catalog is the reply being wrong; with
    `default_style` in the store an empty pick is a missed one (the default is
    itself a candidate); without it, an explicit empty pick is the legal
    "none of these"."""

    catalog = load_style_catalog(_store(tmp_path))
    assert catalog.has_default and catalog.has_choice
    assert catalog.decide(selected="甲组", reason="主播A").outcome == "selected"
    assert catalog.decide(selected="style/甲组", reason="").key == "甲组"
    explicit_default = catalog.decide(selected="default_style", reason="都对不上")
    assert explicit_default.outcome == "selected" and explicit_default.key == "default_style"
    with pytest.raises(StyleContractError, match="乙组"):
        catalog.decide(selected="乙组", reason="随便")
    with pytest.raises(StyleContractError, match="default_style"):
        catalog.decide(selected="", reason="对不上")

    without_default = StyleCatalog(
        candidates=tuple(c for c in catalog.candidates if c.key == "甲组"), rev=catalog.rev
    )
    none = without_default.decide(selected="", reason="对不上")
    assert none.outcome == "none" and none.key == "" and none.names == ()
    with pytest.raises(StyleContractError, match="不在候选里"):
        without_default.decide(selected="default_style", reason="")

    # A missing or unreadable block is a missed pick, never the legal empty one.
    with pytest.raises(StyleContractError, match="缺少"):
        without_default.choose([])
    with pytest.raises(StyleContractError, match="JSON"):
        without_default.choose(["not json"])
    with pytest.raises(StyleContractError, match="selected"):
        without_default.choose(['{"reason": "r"}'])
    with pytest.raises(StyleContractError, match="只能有一个"):
        catalog.choose(['{"selected": "甲组"}', '{"selected": "甲组"}'])
    assert parse_style_selection_block('{"selected": "甲组", "reason": "r"}') == ("甲组", "r")
    assert StyleChoice.from_dict(explicit_default.to_dict()) == explicit_default


def test_only_a_catalog_with_something_besides_the_default_is_a_choice(tmp_path) -> None:
    catalog = load_style_catalog(_store(tmp_path))
    only_default = StyleCatalog(
        candidates=tuple(c for c in catalog.candidates if c.key == "default_style"), rev=catalog.rev
    )
    assert catalog.has_choice
    assert not only_default.has_choice and only_default.has_default
    assert not StyleCatalog().has_choice and not StyleCatalog().has_default


def test_the_pick_is_addressed_by_category_not_by_bare_name(tmp_path) -> None:
    """A style is usually named after the group or streamer it serves, and the
    two namespaces are deliberately allowed to collide (`node/repo.resolve_in`).
    The catalog only ever lists style entries, so the category is the harness's
    to supply -- handing the bare name to a reader would send it into the
    matchable namespace and fail a run over a pick that was never ambiguous."""

    root = _store(tmp_path)
    maintain_main(["--root", str(root), "new", "streamer", "甲组", "--intro", "一个同名的主播"])
    KnowledgeRepo.forget(root)
    choice = load_style_catalog(root).decide(selected="甲组", reason="主播A")

    assert choice.outcome == "selected" and choice.key == "甲组"
    assert choice.names == ("style/甲组",)
    with pytest.raises(StyleSelectionError):
        render_style_block(root, (choice.key,))
    assert "[直角引号]" in render_style_block(root, choice.names)
    assert resolve_style_keys(root, choice.names) == ["甲组"]


# --- the preparation task ------------------------------------------------------


def test_prepare_task_shows_the_catalog_and_records_the_pick(tmp_path) -> None:
    root = _store(tmp_path)
    catalog = load_style_catalog(root)
    client = RecordingClient([_delivery("甲组")])
    payload = run_prepare_task(
        transcript=TRANSCRIPT,
        knowledge_root=root,
        knowledge_enabled=False,
        client=client,
        profile=resolve_profile("text", "none", "quality"),
        task_artifact_dir=tmp_path / "art",
        task_id="t",
        max_parse_retries=0,
        style_catalog=catalog,
        extra_style="人名保留日文",
    )
    system, user = client.calls[0][1][0]["content"], client.calls[0][1][1]["content"]
    assert "翻译风格选择" in system and "- 甲组｜" in system and "人名保留日文" in system
    assert "<style_selection>" in system and "`<style_selection>` 块" in user
    # The forced-pick clause: with `default_style` in the store, "nothing
    # fits" is answered by choosing it, never by leaving the pick empty.
    assert "选 `default_style`" in system and "不要留空" in system
    assert payload["style_selection"]["outcome"] == "selected"
    assert payload["style_selection"]["key"] == "甲组"
    assert payload["style_selection"]["rev"] == catalog.rev


@pytest.mark.parametrize(
    "first",
    [_delivery(None), _delivery(""), _delivery("乙组")],
    ids=["no-block", "empty-with-default", "not-a-candidate"],
)
def test_prepare_task_retries_a_missed_pick_and_fails_when_it_runs_out(tmp_path, first) -> None:
    """The pick is judged inside the parse (`parse_prepare_reply`), so a missed
    one takes the parse retries like any other malformed delivery; when they
    run out the task fails -- there is no fallback to land on."""

    root = _store(tmp_path)
    catalog = load_style_catalog(root)
    kwargs = dict(
        transcript=TRANSCRIPT, knowledge_root=root, knowledge_enabled=False,
        profile=resolve_profile("text", "none", "quality"), task_id="t", style_catalog=catalog,
    )

    retried = RecordingClient([first, _delivery("default_style", "都对不上")])
    payload = run_prepare_task(
        client=retried, task_artifact_dir=tmp_path / "a1", max_parse_retries=1, **kwargs
    )
    assert len(retried.calls) == 2
    # An explicit `default_style` is a real pick: selected, and writable.
    assert payload["style_selection"]["outcome"] == "selected"
    assert payload["style_selection"]["key"] == "default_style"

    with pytest.raises(RuntimeError, match="could not be parsed"):
        run_prepare_task(
            client=RecordingClient([first]), task_artifact_dir=tmp_path / "a2",
            max_parse_retries=0, **kwargs,
        )


def test_without_a_default_an_explicit_empty_pick_means_no_style(tmp_path, reported) -> None:
    root = _store(tmp_path)
    maintain_main(["--root", str(root), "retire", "style/default_style"])
    KnowledgeRepo.forget(root)
    catalog = load_style_catalog(root)
    assert not catalog.has_default and catalog.has_choice

    client = RecordingClient([_delivery("", "都对不上")])
    payload = run_prepare_task(
        transcript=TRANSCRIPT, knowledge_root=root, knowledge_enabled=False, client=client,
        profile=resolve_profile("text", "none", "quality"), task_artifact_dir=tmp_path / "art",
        task_id="t", max_parse_retries=0, style_catalog=catalog,
    )
    system = client.calls[0][1][0]["content"]
    assert "`selected` 填空字符串" in system and "default_style" not in system
    assert payload["style_selection"]["outcome"] == "none"
    assert payload["style_selection"]["key"] == ""
    assert "style-auto-none" not in reported.codes()


def test_a_checkpoint_that_never_picked_is_called_again(tmp_path) -> None:
    """A resume that replays a preparation checkpoint which does not meet the
    forced-pick contract (written before it existed) must not reuse it: the
    parse rejects it, the checkpoint is recorded invalid, and the task calls
    again."""

    from finesub.llm.session_checkpoint import SESSION_CHECKPOINT_FILENAME, _content_hash

    root = _store(tmp_path)
    catalog = load_style_catalog(root)
    art = tmp_path / "art"
    kwargs = dict(
        transcript=TRANSCRIPT, knowledge_root=root, knowledge_enabled=False,
        profile=resolve_profile("text", "none", "quality"), task_artifact_dir=art,
        task_id="t", max_parse_retries=0, style_catalog=catalog,
    )
    run_prepare_task(client=RecordingClient([_delivery("甲组")]), **kwargs)
    store = art / SESSION_CHECKPOINT_FILENAME
    rows = [json.loads(line) for line in store.read_text(encoding="utf-8").splitlines()]
    assert rows
    for row in rows:  # the same input hash, an old reply that never picked
        row["content"] = _delivery(None)
        row["content_hash"] = _content_hash(row["content"])
    store.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    again = RecordingClient([_delivery("default_style")])
    payload = run_prepare_task(client=again, **kwargs)

    assert len(again.calls) == 1
    assert payload["style_selection"]["key"] == "default_style"
    kinds = [json.loads(l)["kind"] for l in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "session_checkpoint_invalid" in kinds


# --- the orchestrator -----------------------------------------------------------


def _run_auto(
    tmp_path, monkeypatch, *, saved_selection, style_mode="read", knowledge="none",
    fast=False, window_pick=None, root=None, style="auto", task_id=None,
    difficulty="quality", mode="execute",
):
    root = root if root is not None else _store(tmp_path)
    if isinstance(saved_selection, dict) and "rev" not in saved_selection:
        # What a real run records: the revision it picked against. Tests that
        # care about a *stale* revision set it explicitly.
        saved_selection = {**saved_selection, "rev": load_style_catalog(root).rev}
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")

    prepared: dict = {}

    def fake_prepare(**kwargs):
        prepared.update(kwargs)
        Path(kwargs["context_path"]).write_text(
            json.dumps({"context_pack": PACK, "keep_entries": [], "style_selection": saved_selection}, ensure_ascii=False),
            encoding="utf-8",
        )
        from finesub.llm.prompts import ContextPack

        return ContextPack()

    monkeypatch.setattr(orchestration, "run_research_stage", fake_prepare)
    if fast:
        from finesub.llm.chunking import SubtitleSegment, plan_correction_windows

        class Counter:
            source = "test-fake"

            def count_text(self, text):
                return max(1, len(text) // 2)

            def count_texts(self, texts):
                return sum(self.count_text(t) for t in texts)

            def count_audio_seconds(self, s):
                return 0

        window = plan_correction_windows([SubtitleSegment("1", 0.0, 1.0, "一。")], counter=Counter())[0]
        monkeypatch.setattr(
            orchestration, "decide_fast_mode",
            lambda **kw: orchestration.FastDecision(mode="on", enabled=True, reason="t", window=window),
        )
    executed: dict = {}

    def fake_execute(**kwargs):
        executed.update(kwargs)
        if window_pick is not None:
            # What `request_turn` does on a fast run: the pick is made inside
            # the window, after the orchestrator resolved the selection, and
            # recorded as an artifact of the task.
            append_task_artifact(
                kwargs["task_artifact_dir"], kind="style_selection", task_id="t",
                payload={"chunk_id": "0001", **window_pick.to_dict()},
            )
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    updates: list[dict] = []
    monkeypatch.setattr(orchestration, "run_post_correction_knowledge_update", lambda **kw: updates.append(kw))
    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable,
            output_path=tmp_path / "out" / "clip.srt",
            postprocess_profile=None,
            knowledge=knowledge,
            knowledge_root=root,
            profile=resolve_profile("text", "none", difficulty),
            style=style,
            mode=mode,
            **({"task_id": task_id} if task_id else {}),
            # A one-segment fixture passes the fast gates on its own; the normal
            # path is what pins the pick, so ask for it explicitly.
            fast="on" if fast else "off",
            style_mode=style_mode,
            refined_srt=(tmp_path / "refined.srt") if style_mode == "update" else None,
            test_profile=True,
        )
    )
    artifacts = [json.loads(l) for l in (tmp_path / "out" / "clip.llm-artifacts" / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    executed["_prepared"] = prepared
    return executed, updates, next(
        (a["payload"] for a in artifacts if a["kind"] == "style_selection"), None
    )


def test_the_pick_is_pinned_before_the_windows_and_bounds_the_update(tmp_path, monkeypatch) -> None:
    chosen = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "主播A"}
    executed, updates, recorded = _run_auto(tmp_path, monkeypatch, saved_selection=chosen, style_mode="update", knowledge="update")
    assert '<style name="甲组">' in executed["style_block"] and "[直角引号]" in executed["style_block"]
    assert "default_style" not in executed["style_block"]
    assert recorded["outcome"] == "selected" and recorded["key"] == "甲组"
    # Qualified on the way out (see the collision test below); the recorded
    # choice keeps the plain catalog key.
    assert updates and updates[0]["style_names"] == ("style/甲组",)


def test_a_recorded_none_injects_the_factory_default(tmp_path, monkeypatch) -> None:
    """A recorded "none" is "no user style", which since 2026-10-31 renders the factory default."""
    executed, updates, recorded = _run_auto(
        tmp_path, monkeypatch,
        saved_selection={"intent": "auto", "outcome": "none", "key": "", "reason": "对不上"},
    )
    assert executed["style_block"] == render_style_block(tmp_path, ())
    assert '<style name="出厂默认">' in executed["style_block"]
    assert recorded["outcome"] == "none"


def _fast_pick(tmp_path, monkeypatch, **kwargs):  # type: ignore[no-untyped-def]
    """A fused window that picks 甲组 in its request turn; returns the calls and the artifact dir."""

    from finesub.llm.client import LLMCallResult as _Result
    from finesub.llm.stages.correction import execute_correction_windows
    from test.conftest import setattr_correction

    root = _store(tmp_path)
    catalog = load_style_catalog(root)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")
    request = (
        "<reasoning>先查</reasoning>\n<window_notes></window_notes>\n<search_queries>\n游戏B\n</search_queries>\n"
        "<style_selection>\n" + json.dumps({"selected": "甲组", "reason": "主播A"}, ensure_ascii=False) + "\n</style_selection>"
    )
    delivery = (
        "<translated>\ntype|position|corrected_text|translation|conf|note\n"
        "sub|1|一。|一|8|\n</translated>\n<next_advice></next_advice>"
    )

    class Client:
        responses = [request, delivery]
        calls: list = []

        def __init__(self, *a, **k) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            type(self).calls.append([dict(m) for m in messages])
            return _Result(content=type(self).responses.pop(0), role=role, model="fake", fallback_used=False, raw_response={"candidates": [{"finishReason": "STOP"}]})

    class Search:
        def search_many(self, requests, *, max_queries=None):
            return []

        def extract_many(self, requests, *, max_urls=None):
            return []

    setattr_correction(monkeypatch, "RoleClient", Client)
    art = tmp_path / "art"
    execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        profile=resolve_profile("text", "local", "quality"),
        knowledge_root=root,
        knowledge_enabled=False,
        search_client=Search(),
        task_artifact_dir=art,
        max_retries_per_window=1,
        resume=False,
        style_catalog=catalog,
        **kwargs,
        length_scale=1.0,
    )
    return Client.calls, art


def test_fast_batched_window_picks_in_its_request_turn(tmp_path, monkeypatch) -> None:
    """auto-style plan §4: the fused window selects in the request turn and the
    picked rules ride in the fill turn, before anything is translated."""

    calls, art = _fast_pick(tmp_path, monkeypatch)
    opening, fill = calls[0], calls[1]
    assert "翻译风格选择" in opening[0]["content"] and "- 甲组｜" in opening[0]["content"]
    assert '<style name="甲组">' in fill[3]["content"] and "[直角引号]" in fill[3]["content"]
    recorded = [json.loads(l) for l in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]
    choice = next(r["payload"] for r in recorded if r["kind"] == "style_selection")
    assert choice["outcome"] == "selected" and choice["key"] == "甲组" and choice["chunk_id"] == "0001"


def test_pass_3_proofreads_under_the_style_a_fast_window_picked(tmp_path, monkeypatch) -> None:
    """The task's block stays empty while a fast window picks; pass 3 gets the
    pick, not the factory default an empty block would otherwise mean."""

    import finesub.llm.stages.proofread.run as proofread_run

    seen: dict = {}

    def _capture(*_a, inputs, **_k):  # type: ignore[no-untyped-def]
        seen["style"] = inputs.style
        raise RuntimeError("captured")  # pass 3 is optional: the run keeps pass 2's rows

    monkeypatch.setattr(proofread_run, "run_proofread", _capture)
    _fast_pick(tmp_path, monkeypatch, proofread=True)
    assert '<style name="甲组">' in seen["style"] and "出厂默认" not in seen["style"]


def test_only_a_structural_no_style_is_a_warning(tmp_path, monkeypatch, reported) -> None:
    """auto-style plan §2.2: "一次无匹配是正常结果，记录简短理由，不作为 warning".

    The task having looked at the material and not matched is that normal
    result. The base having no style entries at all, or no record surviving,
    is a fact about the setup -- the user asked for `--style auto` and got
    nothing for a reason that has nothing to do with the material."""

    chosen = {"intent": "auto", "outcome": "none", "key": "", "reason": "都对不上"}
    _run_auto(tmp_path, monkeypatch, saved_selection=chosen)
    assert "style-auto-none" not in reported.codes()

    # No record at all is structural, and so is a base with no candidates.
    _run_auto(tmp_path / "b", monkeypatch, saved_selection=None)
    assert reported.codes().count("style-auto-none") == 1
    assert "没有留下风格选择记录" in reported.joined()


def test_a_resume_injects_the_rules_the_earlier_windows_saw(tmp_path, monkeypatch) -> None:
    """auto-style plan §5: "续跑复用已提交的选择与实际风格输入；新增、改名、改写
    或删除库中 style 不导致重新抽签".

    The record carries the revision it was picked against, so a rewrite between
    the two runs leaves the remaining windows on the rules the finished ones
    already followed. Recovering only the *name* and rendering from the current
    base split one task across two sets of rules.
    """

    root = _store(tmp_path)
    picked_at = load_style_catalog(root).rev

    # Someone rewrites the style between the two runs.
    repo = KnowledgeRepo.open(root)
    subject = next(s for s in repo.subjects("style") if s.payload.get("surface") == "甲组")
    edit_subject(repo, subject, _entry_text("甲组", "甲组的口味", "甲组", "主播A 的全部投稿", ["[书名号] 作品名用《》"]))
    KnowledgeRepo.forget(root)
    assert "[书名号]" in render_style_block(root, ("style/甲组",))  # the base really moved

    selection = StyleSelection(mode="read", names=(), auto=True).resolved(
        ("style/甲组",), rev=picked_at
    )
    block = render_style_block(root, selection.names, rev=selection.rev)

    assert "[直角引号]" in block  # what the earlier windows were given
    assert "[书名号]" not in block


def _apply_style(
    tmp_path: Path, root: Path, saved: dict, *, style_mode: str = "read", fast: bool = False,
    artifact_dir: Path | None = None,
):
    """`_apply_auto_style` → ``(selection, window_picks)``."""

    context = tmp_path / "ctx.json"
    if saved is not None:  # None = this task left no record at all
        context.write_text(
            json.dumps({"context_pack": PACK, "keep_entries": [], "style_selection": saved}, ensure_ascii=False),
            encoding="utf-8",
        )
    return orchestration._apply_auto_style(
        resolve_style_selection(
            "auto", style_mode, knowledge_root=root, config_path=tmp_path / "none.toml"
        ),
        load_style_catalog(root),
        knowledge_root=root,
        context_path=context,
        artifact_dir=artifact_dir,
        task_id="t",
        fast=fast,
    )


def test_a_record_from_another_store_falls_back_to_the_current_base(tmp_path, monkeypatch) -> None:
    """A revision is an integer with meaning only inside one store, so a record
    written against `knowledge/` before it was rebuilt (or against a different
    `--knowledge-root`) can point at a revision that has no such entry while the
    current base has it. Probing both revisions but rendering at the recorded
    one turned that into a `StyleSelectionError` at render time -- the very
    thing the probe exists to prevent."""

    chosen = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "旧库", "rev": 0}
    executed, _updates, recorded = _run_auto(tmp_path, monkeypatch, saved_selection=chosen)

    assert recorded["outcome"] == "selected" and recorded["key"] == "甲组"
    assert '<style name="甲组">' in executed["style_block"]
    assert "[直角引号]" in executed["style_block"]  # rendered off the current base


def test_all_styles_retired_still_resumes_on_the_recorded_one(tmp_path, monkeypatch) -> None:
    """The empty-catalog shortcut ran before the record was read, so a task
    whose styles were all retired since lost the style on its remaining
    windows -- even though history could still render the pick it already
    used."""

    root = _store(tmp_path)
    picked_at = load_style_catalog(root).rev
    for name in ("甲组", "default_style"):
        maintain_main(["--root", str(root), "retire", f"style/{name}"])
    KnowledgeRepo.forget(root)
    assert not load_style_catalog(root)  # nothing left to choose from

    selection, window_picks = _apply_style(
        tmp_path, root,
        {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "旧的", "rev": picked_at},
    )

    assert not window_picks
    assert selection.names == ("style/甲组",) and selection.rev == picked_at
    assert "[直角引号]" in render_style_block(root, selection.names, rev=selection.rev)


def test_a_retired_style_is_read_from_history_but_never_written_to(tmp_path, monkeypatch) -> None:
    """Reading and writing are two questions. The translation may render a
    retired entry out of history (§5); the post-task update writes to the
    current head, where the entry is gone -- `resolve_style_keys` would raise
    and take the update down with it."""

    root = _store(tmp_path)
    picked_at = load_style_catalog(root).rev
    maintain_main(["--root", str(root), "retire", "style/甲组"])
    KnowledgeRepo.forget(root)

    selection, _window_picks = _apply_style(
        tmp_path, root,
        {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "旧的", "rev": picked_at},
        style_mode="update",
    )

    assert "[直角引号]" in render_style_block(root, selection.names, rev=selection.rev)
    assert orchestration._writable_style_names(
        selection, load_style_catalog(root),
        context_path=tmp_path / "missing.json", artifact_dir=None,
    ) == ()


def test_a_retired_style_degrades_instead_of_killing_the_resume(
    tmp_path, monkeypatch, reported
) -> None:
    """A record naming something the base cannot render at *any* revision (here
    one that never existed; a record predating `rev` whose entry is now gone is
    the real-world shape) made `render_style_block` raise and take the whole
    resume with it -- over a style. It degrades to "no user style" (the factory
    default) and says so.
    ⚠ An explicit `--style <名>` still raises: there the name is the user's own
    instruction, not a recovered record."""

    chosen = {"intent": "auto", "outcome": "selected", "key": "乙组", "reason": "曾经选过", "rev": None}
    executed, updates, recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=chosen, style_mode="update", knowledge="update"
    )

    assert '<style name="出厂默认">' in executed["style_block"]
    assert recorded["outcome"] == "none" and "已无法从知识库渲染出正文" in recorded["reason"]
    assert updates and updates[0]["style_names"] == ()
    assert "style-auto-none" in reported.codes()


def test_the_cli_reads_the_pick_from_the_context_file_it_was_given(tmp_path, monkeypatch) -> None:
    """`--context-file` redirects which preparation result the run consumes, so
    that is the file `--style auto` must read its pick back from. Reading the
    default path instead found nothing, warned "准备阶段没有留下风格选择记录" and
    quietly ran with no style at all."""

    import sys

    root = _store(tmp_path)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
        encoding="utf-8",
    )
    given = tmp_path / "elsewhere" / "ctx.json"
    given.parent.mkdir()
    given.write_text(
        json.dumps(
            {
                "context_pack": PACK,
                "keep_entries": [],
                "style_selection": {
                    "intent": "auto", "outcome": "selected",
                    "key": "甲组", "reason": "主播A", "rev": 2,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    executed: dict = {}

    def fake_execute(**kwargs):
        executed.update(kwargs)
        out = Path(kwargs["output_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")
        return out

    monkeypatch.setattr(orchestration, "execute_correction_windows", fake_execute)
    monkeypatch.setattr(
        sys, "argv",
        [
            "prog", str(stable), "--execute", "--test-profile",
            "--output", str(tmp_path / "out" / "clip.srt"),
            "--context-file", str(given),
            "--knowledge-root", str(root),
            "--style", "auto", "--retrieval", "none", "--media", "text",
            # Normal mode: this test is about which file the pick is read
            # from (fast reads the same record first; covered separately).
            "--fast", "off",
        ],
    )

    assert orchestration._main() == 0
    assert '<style name="甲组">' in executed["style_block"]


def test_the_fast_window_pick_bounds_the_update_too(tmp_path, monkeypatch) -> None:
    """auto-style plan §6 holds in both columns. Fast has no preparation task,
    so the pick is made inside the window -- after the orchestrator resolved
    the selection. Carrying it back is what stops `--style-mode update` from
    injecting a style all run and then learning nothing from it."""

    picked = StyleChoice(outcome="selected", key="甲组", reason="主播A", rev=2)
    executed, updates, recorded = _run_auto(
        tmp_path, monkeypatch,
        saved_selection=None, fast=True, window_pick=picked,
        style_mode="update", knowledge="update",
    )

    # The window picks, so the orchestrator hands it the catalog, injects
    # nothing itself and records nothing: the only record is the window's.
    assert executed["style_catalog"] is not None and executed["style_catalog"].has_choice
    assert executed["style_block"] == ""
    assert recorded["key"] == "甲组" and recorded["chunk_id"] == "0001"
    assert updates and updates[0]["style_names"] == ("style/甲组",)


def test_an_interrupted_knowledge_update_resumes_against_the_same_style(tmp_path, monkeypatch) -> None:
    """The pick belongs to the task, not to the run that made it.

    "选中 → 出译文 → 知识更新中断 → 续跑" takes the early return above the
    windows, so nothing re-derives the choice in memory; it has to come back off
    the record the first run committed, or the update silently writes nowhere.
    """

    picked = StyleChoice(outcome="selected", key="甲组", reason="主播A", rev=2)
    executed, updates, _recorded = _run_auto(
        tmp_path, monkeypatch,
        saved_selection=None, fast=True, window_pick=picked,
        style_mode="update", knowledge="update",
    )
    assert updates and updates[0]["style_names"] == ("style/甲组",)

    # Second run: the translated SRT is already there, so the orchestrator
    # early-returns straight into the knowledge update.
    (tmp_path / "out" / "clip-translated.srt").write_text("", encoding="utf-8")
    updates.clear()
    executed.clear()
    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=tmp_path / "clip-stable.json",
            output_path=tmp_path / "out" / "clip.srt",
            postprocess_profile=None,
            knowledge="update",
            knowledge_root=tmp_path / "kb",
            profile=resolve_profile("text", "none", "quality"),
            style="auto",
            fast="on",
            style_mode="update",
            refined_srt=tmp_path / "refined.srt",
            test_profile=True,
        )
    )

    assert executed == {}  # no windows ran at all
    assert updates and updates[0]["style_names"] == ("style/甲组",)


def test_a_fast_window_with_a_choice_always_picks_itself(tmp_path, monkeypatch) -> None:
    """auto-style-gaps plan A: `style_pick` opens a channel on every fast
    shape (A1, A1', A2), so the orchestrator never records a structural
    "none" for a fast run with something to choose -- it steps aside."""

    executed, _updates, recorded = _run_auto(tmp_path, monkeypatch, saved_selection=None, fast=True)
    assert executed["style_catalog"] is not None
    assert executed["style_block"] == ""
    assert recorded is None  # the (stubbed) window recorded nothing either


# --- record first, and the no-choice shortcut (auto-style-gaps plan, C) ------


@pytest.mark.parametrize("fast", [False, True], ids=["normal", "fast"])
def test_a_recorded_pick_is_used_in_both_modes(tmp_path, fast) -> None:
    """Step 1: the record is read first in both modes. The fast branch used to
    hand a recorded decision back to the window, so a resume could re-pick."""

    root = _store(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    append_task_artifact(
        art, kind="style_selection", task_id="t",
        payload={"chunk_id": "0001", "intent": "auto", "outcome": "selected", "key": "甲组",
                 "reason": "主播A", "rev": load_style_catalog(root).rev},
    )
    selection, window_picks = _apply_style(tmp_path, root, None, fast=fast, artifact_dir=art)
    assert not window_picks
    assert selection.names == ("style/甲组",)


@pytest.mark.parametrize("fast", [False, True], ids=["normal", "fast"])
def test_a_resume_keeps_a_pick_whose_style_was_retired_since(tmp_path, fast) -> None:
    root = _store(tmp_path)
    picked_at = load_style_catalog(root).rev
    art = tmp_path / "art"
    art.mkdir()
    append_task_artifact(
        art, kind="style_selection", task_id="t",
        payload={"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "主播A",
                 "rev": picked_at},
    )
    maintain_main(["--root", str(root), "retire", "style/甲组"])
    KnowledgeRepo.forget(root)

    selection, window_picks = _apply_style(tmp_path, root, None, fast=fast, artifact_dir=art)
    assert not window_picks
    assert "[直角引号]" in render_style_block(root, selection.names, rev=selection.rev)


def _only_default_store(tmp_path: Path) -> Path:
    root = _store(tmp_path)
    maintain_main(["--root", str(root), "retire", "style/甲组"])
    KnowledgeRepo.forget(root)
    return root


@pytest.mark.parametrize("style", ["auto", None], ids=["explicit-auto", "unset"])
@pytest.mark.parametrize("fast", [False, True], ids=["normal", "fast"])
def test_only_default_style_is_todays_default_silently(tmp_path, monkeypatch, reported, fast, style) -> None:
    """Step 2: with only `default_style` there is nothing to choose. Nobody is
    asked (no catalog for the preparation task or the window), the default is
    injected and writable exactly as without `--style auto`, nothing warns,
    and the result is recorded so a resume sees the same decision."""

    root = _only_default_store(tmp_path)
    executed, updates, recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=None, fast=fast, root=root,
        style_mode="update", knowledge="update", style=style,
    )

    assert executed["style_catalog"] is None
    if not fast:
        assert executed["_prepared"]["runner_kwargs"] is None  # no catalog shown
    assert '<style name="default_style">' in executed["style_block"]
    assert executed["style_block"] == render_style_block(root, ("default_style",))
    assert recorded["outcome"] == "selected" and recorded["key"] == "default_style"
    assert recorded["reason"] == "没有可选的 style"
    assert updates and updates[0]["style_names"] == ("style/default_style",)
    assert "style-auto-none" not in reported.codes()


@pytest.mark.parametrize("style", ["auto", None], ids=["explicit-auto", "unset"])
@pytest.mark.parametrize("fast", [False, True], ids=["normal", "fast"])
def test_a_store_without_styles_runs_without_one_silently(tmp_path, monkeypatch, reported, fast, style) -> None:
    root = tmp_path / "kb"
    maintain_main(["--root", str(root), "new", "streamer", "某主播", "--intro", "不是风格"])
    KnowledgeRepo.forget(root)
    assert not load_style_catalog(root)

    executed, _updates, recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=None, fast=fast, root=root, style=style
    )

    assert executed["style_catalog"] is None
    assert executed["style_block"] == render_style_block(root, ())  # the factory default
    assert recorded["outcome"] == "none" and recorded["reason"] == "没有可选的 style"
    assert "style-auto-none" not in reported.codes()


@pytest.mark.parametrize("fast", [False, True], ids=["normal", "fast"])
def test_a_style_added_before_the_resume_does_not_reopen_the_choice(
    tmp_path, monkeypatch, reported, fast
) -> None:
    """Started with only `default_style` (the shortcut recorded it); a new
    style appears before the resume. The resume uses the recorded default --
    no record would have meant a pick, or worse, "no record left" and a
    warning (auto-style-gaps plan, C: the shortcut must not run off the
    current catalog)."""

    root = _only_default_store(tmp_path)
    art = tmp_path / "art"
    art.mkdir()
    first, window_picks = _apply_style(tmp_path, root, None, fast=fast, artifact_dir=art)
    assert not window_picks and first.names == ("style/default_style",)

    maintain_main(["--root", str(root), "new", "style", "乙组", "--intro", "新来的"])
    KnowledgeRepo.forget(root)
    assert load_style_catalog(root).has_choice

    resumed, window_picks = _apply_style(tmp_path, root, None, fast=fast, artifact_dir=art)
    assert not window_picks and resumed.names == ("style/default_style",)
    assert "style-auto-none" not in reported.codes()


def test_a_rerun_preparation_outranks_the_pinned_copy(tmp_path, monkeypatch) -> None:
    """The orchestrator copies every pin into a `style_selection` artifact. When
    the preparation reruns (reuse key changed, or the context was deleted) and
    picks anew, the new context is the task's decision: the old copy must not
    win, or the run injects and writes the old style while the context and the
    report name the new one (peer review of phase A, 2026-10-01)."""

    root = _store(tmp_path)
    maintain_main(["--root", str(root), "new", "style", "乙组", "--intro", "乙组的口味"])
    KnowledgeRepo.forget(root)
    first = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "主播A"}
    _executed, _updates, recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=first, root=root,
        style_mode="update", knowledge="update",
    )
    assert recorded["key"] == "甲组"  # the pin's artifact copy

    out = tmp_path / "out"
    (out / "clip.srt").unlink()
    for context in out.rglob("*research-context.json"):
        context.unlink()  # the preparation reruns
    second = {"intent": "auto", "outcome": "selected", "key": "乙组", "reason": "主播B"}
    executed, updates, _recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=second, root=root,
        style_mode="update", knowledge="update",
    )
    assert '<style name="乙组">' in executed["style_block"]
    assert "甲组" not in executed["style_block"]
    assert updates and updates[-1]["style_names"] == ("style/乙组",)


# --- the agentic preparation task can read the candidates (batch 2, item 6) ---


def test_an_agentic_preparation_task_gets_the_narrow_style_grant(tmp_path, monkeypatch) -> None:
    """`retrieval=local` with no readable index makes the preparation task
    agentic; its prompt offers `kb_read` on `style/<候选名>`, so the call must
    carry the narrow grant -- not the full read, not nothing."""

    import finesub.llm.stages.prepare_task as prepare_mod

    monkeypatch.setattr(prepare_mod, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "t"))
    root = _store(tmp_path)
    catalog = load_style_catalog(root)
    client = RecordingClient([_delivery("甲组")])
    run_prepare_task(
        transcript=TRANSCRIPT, knowledge_root=root, knowledge_enabled=False, client=client,
        profile=resolve_profile("text", "local", "quality"), task_artifact_dir=tmp_path / "art",
        task_id="t", max_parse_retries=0, style_catalog=catalog,
    )
    system = client.calls[0][1][0]["content"]
    assert "可用 `kb_read`（条目名写 `style/<候选名>`）" in system
    assert client.calls[0][2]["agent_task_extras"] == {
        "kb_tools": "style",
        "knowledge_root": str(root),
        "knowledge_identity": f"rev:{catalog.rev}",
        "kb_signal_task": "t",
    }

    # Without a catalog the task reads nothing; with a readable base it reads it.
    plain = RecordingClient([_delivery(None)])
    run_prepare_task(
        transcript=TRANSCRIPT, knowledge_root=root, knowledge_enabled=False, client=plain,
        profile=resolve_profile("text", "local", "quality"), task_artifact_dir=tmp_path / "art2",
        task_id="t", max_parse_retries=0,
    )
    assert plain.calls[0][2]["agent_task_extras"] is None


# --- the default flip (auto-style-gaps plan C) ---------------------------------


def test_an_unset_style_is_auto(tmp_path, monkeypatch) -> None:
    """Nobody naming a style is `auto`: with a choice in the store the fast
    window picks (the orchestrator hands it the catalog)."""

    executed, _updates, _recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=None, fast=True, style=None
    )
    assert executed["style_catalog"] is not None and executed["style_catalog"].has_choice


@pytest.mark.parametrize("style", ["auto", "甲组"])
def test_efficiency_refuses_a_named_style_in_both_front_ends(tmp_path, style) -> None:
    """owner 2026-10-30: a style named on the command line under efficiency,
    `auto` included, is refused -- by the correction task, and before ASR by
    the pipeline's early check."""

    from finesub.llm.correction_request import CorrectionRequest
    from finesub.stages import _validate_llm_configuration

    root = _store(tmp_path)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")
    with pytest.raises(StyleSelectionError, match="efficiency"):
        orchestration.run_correction_task(
            CorrectionRequest(
                stable_json=stable, output_path=tmp_path / "out" / "clip.srt",
                knowledge_root=root, profile=resolve_profile("text", "none", "efficiency"),
                style=style, test_profile=True,
            )
        )
    with pytest.raises(StyleSelectionError, match="efficiency") as refused:
        _validate_llm_configuration(
            knowledge_root=root, style=style, style_mode=None, llm_difficulty="efficiency"
        )
    # A resume that drops --style continues the style the task recorded (`_record_named_style`, `auto` that
    # does not ask reads the record first), and the refusal says so.
    assert "续跑的任务去掉 --style 照旧用它记下的风格" in str(refused.value)
    # The implicit default passes the early check: there it is the factory default.
    _validate_llm_configuration(
        knowledge_root=root, style=None, style_mode=None, llm_difficulty="efficiency"
    )


def test_a_plan_shows_the_default_style_when_there_is_nothing_to_pick(tmp_path) -> None:
    """A dry run pins nothing, so under the flipped default it shows what is
    certain: with only `default_style` in the store, its block -- the prompt
    the old default showed -- and no catalog anywhere."""

    from finesub.llm.correction_request import CorrectionRequest

    root = _only_default_store(tmp_path)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")
    seen: dict = {}
    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan",
            knowledge_root=root, profile=resolve_profile("text", "none", "quality"),
            test_profile=True, on_plan=seen.update,
        )
    )
    window_system = seen["correction_messages"][0][0]["content"]
    assert '<style name="default_style">' in window_system
    assert "翻译风格选择" not in seen["research_messages"][0][0]["content"]


def _plan_window_system(tmp_path, root, difficulty="quality", **kwargs):  # type: ignore[no-untyped-def]
    from finesub.llm.correction_request import CorrectionRequest

    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")
    seen: dict = {}
    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan",
            knowledge_root=root, profile=resolve_profile("text", "none", difficulty),
            test_profile=True, on_plan=seen.update, **kwargs,
        )
    )
    return seen["correction_messages"][0][0]["content"]


def test_a_plan_with_a_pick_pending_shows_no_style(tmp_path) -> None:
    """The pick is the model's: a plan shows no block, not the factory default
    as if the selection had settled on "no user style"."""

    assert "<style name=" not in _plan_window_system(tmp_path, _store(tmp_path))


def test_without_a_knowledge_base_the_factory_default_is_injected(tmp_path) -> None:
    """A root with no store yet, and `--knowledge none` (which opens nothing)."""

    assert '<style name="出厂默认">' in _plan_window_system(tmp_path, tmp_path / "no-kb")
    root = tmp_path / "unread"
    assert '<style name="出厂默认">' in _plan_window_system(tmp_path, root, knowledge="none")
    assert not (root / "knowledge.sqlite").exists()


def test_an_efficiency_plan_renders_only_the_factory_default(tmp_path) -> None:
    """The store's `default_style` is the user's default, which efficiency does
    not use (owner 2026-10-30), even from the config: it gets the factory default."""

    from finesub.llm.correction_request import CorrectionRequest

    root = _store(tmp_path)
    stable = tmp_path / "clip-stable.json"
    stable.write_text(json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}), encoding="utf-8")
    seen: dict = {}
    orchestration.run_correction_task(
        CorrectionRequest(
            stable_json=stable, output_path=tmp_path / "out" / "clip.srt", mode="plan",
            knowledge_root=root, profile=resolve_profile("text", "none", "efficiency"),
            test_profile=True, on_plan=seen.update,
        )
    )
    window_system = seen["correction_messages"][0][0]["content"]
    assert '<style name="出厂默认">' in window_system and '<style name="default_style">' not in window_system
    assert "翻译风格选择" not in seen["research_messages"][0][0]["content"]


# --- efficiency: only the factory default, and the task's own choice -----------


def _rerun(tmp_path, monkeypatch, root, **kwargs):
    """Run the same task again: the output is removed so the windows run, the
    research context stays (the resume reuses it)."""

    out = tmp_path / "out" / "clip.srt"
    if out.exists():
        out.unlink()
    return _run_auto(tmp_path, monkeypatch, saved_selection=None, root=root, **kwargs)


def test_efficiency_injects_the_factory_default_and_another_difficulty_keeps_that(
    tmp_path, monkeypatch, reported
) -> None:
    """owner 2026-10-30: efficiency uses only the factory default style -- not
    the store's `default_style`, the user's default. It asks nobody, and "no
    user style" is the task's choice: recorded where a pick would be, so a
    resume at a difficulty that does pick continues it rather than picking anew."""

    root = _store(tmp_path)
    first, _u, recorded = _run_auto(
        tmp_path, monkeypatch, saved_selection=None, root=root, style=None, difficulty="efficiency"
    )
    assert first["style_catalog"] is None
    assert first["_prepared"]["runner_kwargs"] is None  # nobody was asked
    assert first["style_block"] == render_style_block(root, ())  # the factory default
    assert (recorded["outcome"], recorded["key"]) == ("none", "")
    assert recorded["reason"] == "efficiency 只用出厂默认风格"

    second, _u, _r = _rerun(tmp_path, monkeypatch, root, style=None, difficulty="quality")
    assert second["style_block"] == first["style_block"]
    assert "style-auto-none" not in reported.codes()


def test_a_pick_then_efficiency_keeps_the_pick(tmp_path, monkeypatch, reported) -> None:
    root = _store(tmp_path)
    chosen = {"intent": "auto", "outcome": "selected", "key": "甲组", "reason": "主播A"}
    first, _u, _r = _run_auto(tmp_path, monkeypatch, saved_selection=chosen, root=root, style=None)
    assert '<style name="甲组">' in first["style_block"]

    second, _u, _r = _rerun(tmp_path, monkeypatch, root, style=None, difficulty="efficiency")
    assert second["style_block"] == first["style_block"]
    assert "style-auto-none" not in reported.codes()


def test_a_research_only_run_records_the_choice_for_the_full_run(tmp_path, monkeypatch, reported) -> None:
    """`--research-only` decides the task's style too: run when the store had
    nothing to choose, it records the no-choice result, and the full run after
    a style was added continues it."""

    root = _only_default_store(tmp_path)
    _run_auto(tmp_path, monkeypatch, saved_selection=None, root=root, style=None, mode="research")
    maintain_main(["--root", str(root), "new", "style", "乙组", "--intro", "新来的"])
    KnowledgeRepo.forget(root)

    full, _u, _r = _rerun(tmp_path, monkeypatch, root, style=None)
    assert full["style_catalog"] is None
    assert full["style_block"] == render_style_block(root, ("default_style",))
    assert "style-auto-none" not in reported.codes()


def test_a_named_style_is_continued_by_a_resume_that_names_none(tmp_path, monkeypatch, reported) -> None:
    root = _store(tmp_path)
    first, _u, recorded = _run_auto(tmp_path, monkeypatch, saved_selection=None, root=root, style="甲组")
    assert recorded["key"] == "甲组" and recorded["intent"] == "named"

    second, _u, _r = _rerun(tmp_path, monkeypatch, root, style=None)
    assert second["style_block"] == first["style_block"]
    assert "style-auto-none" not in reported.codes()

    # A name given again still wins over any record.
    third, _u, _r = _rerun(tmp_path, monkeypatch, root, style="default_style")
    assert '<style name="default_style">' in third["style_block"]


def test_the_style_lead_in_pairs_examples_only_when_a_style_has_some(monkeypatch, tmp_path):
    """Cold read round 3 (P263): the pairing rule for examples is not shown for a style that has none."""

    import finesub.llm.knowledge.style as style

    bare = "# 甲组\n\n## 约定\n\n[直角引号] 引语用「」"
    monkeypatch.setattr(style, "load_style_entries", lambda *_args, **_kw: [("甲组", bare)])
    assert "正例/反例" not in style.render_style_block(tmp_path, ("style/甲组",))
    shown = bare + "\n\n## 正例\n\n[直角引号] 「やだ」 → 「不要」"
    monkeypatch.setattr(style, "load_style_entries", lambda *_args, **_kw: [("甲组", shown)])
    assert "正例/反例按同名 [标记]" in style.render_style_block(tmp_path, ("style/甲组",))


def test_an_agentic_window_that_may_pick_none_sees_the_factory_default_conditionally(tmp_path) -> None:
    """It picks and translates in one delivery: a pick of "none" must find the
    factory default in the prompt already, worded so it binds only then. With
    `default_style` among the candidates "none" is not an answer, and every
    other shape gets its style from the harness after the pick."""

    from finesub.llm.prepare_prompts import render_style_selection_section

    root = _store(tmp_path)
    with_default = load_style_catalog(root).render()
    maintain_main(["--root", str(root), "retire", "style/default_style"])
    KnowledgeRepo.forget(root)
    without_default = load_style_catalog(root).render()

    section = render_style_selection_section(without_default, shape="agentic", read="required")
    assert "`selected` 填空字符串时，这一窗按下面这套「出厂默认」风格翻译" in section
    assert '<style name="出厂默认">' in section and "[圈内借词]" in section and "仍然优先" not in section
    assert "（本次另写的特殊翻译风格要求仍然优先）" in render_style_selection_section(
        without_default, extra_style="人名不译", shape="agentic", read="required")
    assert "出厂默认" not in render_style_selection_section(with_default, shape="agentic", read="required").replace(
        "只用出厂默认的基础口味", "")
    assert "<style name=" not in render_style_selection_section(without_default, shape="request")
