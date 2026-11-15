"""`--style auto` on the fast window (docs/knowledge.md, 「style 类别」, 谁来选).

Pinned: the common switch `style_pick` opens a channel on every fast shape
without touching `window_requests` (the transfer chain's question); a batched
window picks in a request turn that must carry a valid pick -- a direct
delivery or a request without one is malformed, and running out of retries
fails the task (A, A2); a window with only the pick to make gets a request
turn of its own whose fill carries the style alone (A2); an agentic window
picks in its delivery, must have read the picked style in full through the
narrow `kb_tools: "style"` grant, is judged by the same validator over the
same ledger on both sides, and its first valid pick pins the style for any
later half of a split (A1).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from finesub.llm.agent.agent_validators import (
    judge_style_pick,
    runtime_validators,
    serialize_window,
    style_pick_params,
)
from finesub.llm.chunking import SubtitleSegment, plan_correction_windows
from finesub.llm.client import LLMCallResult
from finesub.llm.correction_translation import _writable_style_names
from finesub.llm.knowledge.style import StyleSelection, load_style_catalog
from finesub.llm.routing.config import KB_WINDOW_TOTAL_ENTRIES, LLMRole
from finesub.llm.routing.profiles import resolve_profile
from finesub.llm.stages import task_graph
from finesub.llm.stages.correction import execute_correction_windows
from finesub.llm.stages.correction.request_turn import REQUEST_TURN_FORMAT_RETRIES
from test.conftest import setattr_correction
from test.test_llm_auto_style import DIRECT_DELIVERY, _store

STOP = {"candidates": [{"finishReason": "STOP"}]}


def _pick(name: str, reason: str = "主播A") -> str:
    return (
        "<style_selection>\n"
        + json.dumps({"selected": name, "reason": reason}, ensure_ascii=False)
        + "\n</style_selection>"
    )


def _request(*, pick: str | None = "甲组", queries: str | None = None) -> str:
    text = "<reasoning>先选</reasoning>\n<window_notes></window_notes>\n"
    if queries is not None:
        text += f"<search_queries>\n{queries}\n</search_queries>\n"
    if pick is not None:
        text += _pick(pick)
    return text


FULL_READ = {"entry": "甲组", "category": "style", "sections": False}


class _Client:
    """Pops scripted replies; ``(text, kb_reads)`` pairs carry the ledger an
    agentic call's execution attempt would bring back (`fold_kb_reads`)."""

    script: list = []
    calls: list = []

    def __init__(self, *a, **k) -> None:
        pass

    def complete(self, role, messages, **kwargs):
        if callable(messages):
            messages = messages("capableC")
        type(self).calls.append(([dict(m) for m in messages], kwargs))
        item = type(self).script.pop(0)
        if isinstance(item, BaseException):
            raise item
        text, reads = item if isinstance(item, tuple) else (item, None)
        return LLMCallResult(
            content=text, role=role, model="fake", fallback_used=False, raw_response=STOP,
            execution_attempts=[{"kb_reads": list(reads)}] if reads is not None else [],
        )


class _Search:
    def search_many(self, requests, *, max_queries=None):
        return []

    def extract_many(self, requests, *, max_urls=None):
        return []


def _artifacts(art: Path) -> list[dict]:
    return [json.loads(l) for l in (art / "task-artifacts.jsonl").read_text(encoding="utf-8").splitlines()]


def _run(
    tmp_path, monkeypatch, script, *, retrieval="none", strategy="batched", catalog=None,
    root=None, stable=None, max_retries=1,
):
    root = root if root is not None else _store(tmp_path)
    catalog = catalog if catalog is not None else load_style_catalog(root)
    if stable is None:
        stable = tmp_path / "clip-stable.json"
        stable.write_text(
            json.dumps({"segments": [{"id": "1", "start": 0.0, "end": 1.0, "text": "一。"}]}),
            encoding="utf-8",
        )
    monkeypatch.setattr(task_graph, "resolve_execution_strategy", lambda profile, **kw: (strategy, "t"))
    _Client.script = list(script)
    _Client.calls = []
    setattr_correction(monkeypatch, "RoleClient", _Client)
    art = tmp_path / "art"
    execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        profile=resolve_profile("text", retrieval, "quality"),
        knowledge_root=root,
        knowledge_enabled=False,
        search_client=_Search(),
        task_artifact_dir=art,
        max_retries_per_window=max_retries,
        max_replacements_per_window=0,
        resume=False,
        style_catalog=catalog,
        length_scale=1.0,
    )
    return root, catalog, art


# --- the common switch -------------------------------------------------------


def test_style_pick_opens_a_channel_but_never_asks_for_entries(monkeypatch) -> None:
    profile = resolve_profile("text", "none", "quality")
    for strategy in ("batched", "agentic"):
        monkeypatch.setattr(task_graph, "resolve_execution_strategy", lambda p, s=strategy, **kw: (s, "t"))
        plain = task_graph.window_flow(profile, knowledge_readable=False)
        assert plain == task_graph.WindowFlow(False, False, False)  # unchanged without a pick
        flow = task_graph.window_flow(profile, knowledge_readable=False, style_pick=True)
        assert flow.window_requests is False and flow.style_pick is True
        assert flow.request_channel is (strategy == "batched")
        assert flow.agent_fetches is (strategy == "agentic")


def test_a_style_only_window_keeps_the_whole_transfer_chain(tmp_path, monkeypatch) -> None:
    """`CorrectionRun.window_requests` is the flow's own answer: a channel open
    only for the pick cannot ask for a dropped entry back, so the chain keeps
    the full window budget and is never pruned."""

    from finesub.llm.stages.correction import run as run_module

    captured = {}

    def capture(run, windows):
        captured["run"] = run

    monkeypatch.setattr(run_module, "run_serial_windows", capture)
    _run(tmp_path, monkeypatch, [])
    run = captured["run"]
    assert run.request_channel is True and run.window_requests is False
    assert run.transfer_cap == KB_WINDOW_TOTAL_ENTRIES
    assert run.style_catalog is not None


def test_the_fast_gate_weighs_the_catalog_of_a_style_only_window(tmp_path) -> None:
    """The gate measures the batched opening; with only the pick to make that
    opening still has a request turn and the catalog in it."""

    from finesub.llm.stages.plan import decide_fast_mode
    from test.test_llm_fast_plan import FakeTokenCounter, _stable_json

    kwargs = dict(
        stable_json=_stable_json(tmp_path),
        fast="auto",
        profile=resolve_profile("text", "none", "quality"),
        knowledge_root=tmp_path / "kb",
        knowledge_enabled=False,
        token_counter=FakeTokenCounter(),
    )
    plain = decide_fast_mode(**kwargs)
    with_catalog = decide_fast_mode(
        **kwargs, style_catalog="- 甲组｜口味｜作者：甲组｜适用范围：主播A 的全部投稿"
    )
    assert with_catalog.opening_input_tokens > plain.opening_input_tokens


# --- A2: a request turn for the pick alone --------------------------------------


def test_a_style_only_window_picks_in_a_request_turn_of_its_own(tmp_path, monkeypatch) -> None:
    _root, _catalog, art = _run(tmp_path, monkeypatch, [_request(), DIRECT_DELIVERY])

    (opening, opening_kwargs), (delivery, _kw) = _Client.calls
    system = opening[0]["content"]
    assert "选择翻译风格（同一任务内，先选后译）" in system and "- 甲组｜" in system
    assert "获取缺失输入（同一任务内，至多一次）" not in system  # no search, no entries
    assert "知识库索引" not in system
    assert "首条回复只做选择" in system
    assert opening_kwargs.get("agent_task_extras") is None  # the harness reads the style
    fill = delivery[3]["content"]
    assert '<style name="甲组">' in fill and "[直角引号]" in fill
    assert "<keep_entries>" not in fill and "`<next_advice>`" in fill
    artifacts = _artifacts(art)
    assert not [a for a in artifacts if a["kind"] == "correction_query_format_retry"]
    choice = next(a["payload"] for a in artifacts if a["kind"] == "style_selection")
    assert choice["outcome"] == "selected" and choice["key"] == "甲组" and choice["chunk_id"] == "0001"


def test_a_style_only_reply_that_requests_is_malformed(tmp_path, monkeypatch) -> None:
    _root, _catalog, art = _run(
        tmp_path, monkeypatch,
        [_request(queries="游戏B"), _request(), DIRECT_DELIVERY],
    )
    assert len(_Client.calls) == 3
    retry = next(a["payload"] for a in _artifacts(art) if a["kind"] == "correction_query_format_retry")
    assert any("<search_queries>" in error for error in retry["errors"])


@pytest.mark.parametrize("retrieval", ["none", "local"])
def test_a_direct_delivery_is_retried_and_then_fails_the_task(tmp_path, monkeypatch, retrieval) -> None:
    """The pick is part of the delivery contract: the old "direct delivery
    picks none" is gone, and so is "retries spent, parse it anyway"."""

    attempts = REQUEST_TURN_FORMAT_RETRIES + 1
    with pytest.raises(RuntimeError, match="no valid style pick"):
        _run(tmp_path, monkeypatch, [DIRECT_DELIVERY] * attempts, retrieval=retrieval)
    assert len(_Client.calls) == attempts
    art = tmp_path / "art"
    assert not [a for a in _artifacts(art) if a["kind"] == "style_selection"]


@pytest.mark.parametrize(
    "bad",
    [_request(pick=None, queries="游戏B"), _request(pick="乙组", queries="游戏B")],
    ids=["requests-without-a-pick", "not-a-candidate"],
)
def test_a_request_turn_must_pick_as_well_as_request(tmp_path, monkeypatch, bad) -> None:
    """AND, not OR: `<search_queries>` without a valid pick is malformed."""

    _run(tmp_path, monkeypatch, [bad, _request(queries="游戏B"), DIRECT_DELIVERY], retrieval="local")
    assert len(_Client.calls) == 3
    with pytest.raises(RuntimeError, match="no valid style pick"):
        _run(tmp_path / "again", monkeypatch, [bad] * (REQUEST_TURN_FORMAT_RETRIES + 1), retrieval="local")


def test_a_pick_without_requests_passes_a_search_window(tmp_path, monkeypatch) -> None:
    """With search granted, "nothing to look up" is a reply that only picks."""

    _run(tmp_path, monkeypatch, [_request(), DIRECT_DELIVERY], retrieval="local")
    assert len(_Client.calls) == 2
    assert "不需要由 harness 代为取得的输入时也不能直接交付" in _Client.calls[0][0][0]["content"]


def test_a_request_turn_call_error_fails_the_pick(tmp_path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="provider down"):
        _run(tmp_path, monkeypatch, [RuntimeError("provider down")])


# --- A1: the agentic window picks in its delivery ------------------------------


def _delivered(pick: str | None = "甲组") -> str:
    return DIRECT_DELIVERY + ("\n" + _pick(pick) if pick is not None else "")


def test_an_agentic_window_picks_after_reading_the_style(tmp_path, monkeypatch) -> None:
    root, catalog, art = _run(
        tmp_path, monkeypatch, [(_delivered(), [FULL_READ])], strategy="agentic"
    )

    ((messages, kwargs),) = _Client.calls
    system = messages[0]["content"]
    assert "翻译风格选择" in system and "- 甲组｜" in system
    assert "不要带 `sections`" in system and "交付终稿时紧接在 `</translated>` 之后" in system
    # Only the style to fetch: no acquire note naming tools this task lacks.
    assert "获取缺失输入（由你自行取得" not in system
    assert kwargs["agent_task_extras"] == {
        "kb_tools": "style",
        "knowledge_root": str(root),
        "knowledge_identity": f"rev:{catalog.rev}",
        "kb_signal_task": "",
        "kb_signal_window": "0001",
    }
    assert kwargs["validator_spec"]["params"]["style_pick"] == style_pick_params(catalog)
    choice = next(a["payload"] for a in _artifacts(art) if a["kind"] == "style_selection")
    assert choice["outcome"] == "selected" and choice["key"] == "甲组" and choice["chunk_id"] == "0001"
    assert _writable_style_names(
        StyleSelection(mode="update", names=(), auto=True), catalog,
        context_path=tmp_path / "missing.json", artifact_dir=art,
    ) == ("style/甲组",)


@pytest.mark.parametrize(
    ("first", "message"),
    [
        ((_delivered(), []), "没有用 kb_read 完整读过"),
        ((_delivered(), [{**FULL_READ, "sections": True}]), "只读了部分章节"),
        ((_delivered(), [{**FULL_READ, "entry": "default_style"}]), "没有用 kb_read 完整读过"),
        ((_delivered(None), [FULL_READ]), "缺少 <style_selection>"),
        ((_delivered("乙组"), [FULL_READ]), "不在候选里"),
    ],
    ids=["never-read", "sections-only", "read-another", "no-block", "not-a-candidate"],
)
def test_an_agentic_pick_that_breaks_the_contract_is_repaired(tmp_path, monkeypatch, first, message) -> None:
    _root, _catalog, art = _run(
        tmp_path, monkeypatch, [first, (_delivered(), [FULL_READ])], strategy="agentic"
    )
    assert len(_Client.calls) == 2
    repair_errors = _Client.calls[1][1]["validation_errors"]
    assert any(message in error for error in repair_errors)
    responses = [a["payload"] for a in _artifacts(art) if a["kind"] == "correction_window_response"]
    assert responses[0]["validation_ok"] is False and responses[1]["validation_ok"] is True


def test_an_agentic_window_that_never_picks_fails_the_task(tmp_path, monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="failed validation"):
        _run(tmp_path, monkeypatch, [(_delivered(), [])] * 2, strategy="agentic")
    assert not [a for a in _artifacts(tmp_path / "art") if a["kind"] == "style_selection"]


def test_the_first_valid_pick_pins_the_style_across_a_split(tmp_path, monkeypatch) -> None:
    """A1 step 5: a truncated window splits into -a/-b, each delivering on its
    own. The first valid pick pins the task's style; the later half sees no
    catalog and gets the picked rules up front, so the halves cannot differ."""

    from test.test_llm_correction_translation import FakeTokenCounter

    stable = tmp_path / "clip-stable.json"
    stable.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0.0, "end": 1.0, "text": "一。"},
                    {"id": "2", "start": 1.2, "end": 2.0, "text": "二"},
                    {"id": "3", "start": 2.2, "end": 3.0, "text": "三。"},
                    {"id": "4", "start": 3.2, "end": 4.0, "text": "四"},
                    {"id": "5", "start": 4.2, "end": 5.0, "text": "五。"},
                ]
            }
        ),
        encoding="utf-8",
    )
    header = "type|position|corrected_text|translation|conf|note"
    first_half = (
        f"<singles>\n{header}\n"
        "sub|1|一。|一|8|译1字；宜保持独立\nsub|2|二|二|8|译1字；宜保持独立\n"
        "sub|3|三。|三|8|译1字；宜保持独立\n</singles>\n"
        f"<translated>\n{header}\nsub|1|一。|一|high|\nsub|2|二|二|high|\n"
        "sub|3|三。|三|high|\n</translated>\n<next_advice></next_advice>\n" + _pick("甲组")
    )
    second_half = (
        f"<singles>\n{header}\n"
        "sub|1|四|四|8|译1字；宜保持独立\nsub|2|五。|五|8|译1字；宜保持独立\n</singles>\n"
        f"<translated>\n{header}\nsub|1|四|四|high|\nsub|2|五。|五|high|\n"
        "</translated>\n<next_advice></next_advice>"
    )
    truncated = f"<translated>\n{header}\nsub|1|一。|一|high|\n"
    script = [("MAX_TOKENS", truncated), ("STOP", first_half), ("STOP", second_half)]
    calls: list = []

    class Client:
        def __init__(self, *a, **k) -> None:
            pass

        def complete(self, role, messages, **kwargs):
            if callable(messages):
                messages = messages("capableC")
            calls.append(([dict(m) for m in messages], kwargs))
            finish, content = script.pop(0)
            usage = (
                {"candidatesTokenCount": 60_000, "thoughtsTokenCount": 5_500}
                if finish == "MAX_TOKENS"
                else {"candidatesTokenCount": 100, "thoughtsTokenCount": 50}
            )
            return LLMCallResult(
                content=content, role=LLMRole.AUDIO_MULTIMODAL, model="fake", fallback_used=False,
                raw_response={"candidates": [{"finishReason": finish}], "usageMetadata": usage},
                execution_attempts=[{"kb_reads": [FULL_READ]}],
            )

    root = _store(tmp_path)
    monkeypatch.setattr(task_graph, "resolve_execution_strategy", lambda profile, **kw: ("agentic", "t"))
    setattr_correction(monkeypatch, "RoleClient", Client)
    art = tmp_path / "art"
    execute_correction_windows(
        stable_json=stable,
        output_path=tmp_path / "out.srt",
        token_counter=FakeTokenCounter(),
        profile=resolve_profile("text", "none", "quality"),
        knowledge_root=root,
        knowledge_enabled=False,
        task_artifact_dir=art,
        max_retries_per_window=2,
        max_replacements_per_window=0,
        resume=False,
        style_catalog=load_style_catalog(root),
        length_scale=1.0,
    )

    assert len(calls) == 3
    whole, first, second = calls
    assert "- 甲组｜" in first[0][0]["content"]
    assert "翻译风格选择" not in second[0][0]["content"]
    assert '<style name="甲组">' in second[0][0]["content"]
    assert second[1]["agent_task_extras"] is None  # nothing left to read
    assert "style_pick" not in second[1]["validator_spec"]["params"]
    picks = [a["payload"] for a in _artifacts(art) if a["kind"] == "style_selection"]
    assert [(p["key"], p["chunk_id"]) for p in picks] == [("甲组", "0001-a")]


# --- one validator, one ledger --------------------------------------------------


def _manifest(window, params) -> dict:
    return {
        "metadata": {
            "validator": {"window": serialize_window(window), "style_pick": params},
            "variant": "",
            "capability_tier": "capable",
        }
    }


@pytest.mark.parametrize(
    "reads",
    [[], [FULL_READ], [{**FULL_READ, "sections": True}]],
    ids=["unread", "read", "sections"],
)
def test_the_runtime_and_the_harness_judge_a_pick_alike(tmp_path, reads) -> None:
    """The MCP server runs `correction-window` inside the runtime's submit,
    handed the manifest plus the ledger it observed; the harness runs
    `judge_style_pick` over the copy folded into the call. Same verdict."""

    catalog = load_style_catalog(_store(tmp_path))
    params = style_pick_params(catalog)
    window = dataclasses.replace(plan_correction_windows(
        [SubtitleSegment("1", 0.0, 1.0, "一。")], counter=_Counter()
    )[0], length_scale=1.0)
    validate = runtime_validators("correction-window")["correction-window"]
    view = {**_manifest(window, params), "observed": {"kb_reads": reads}}

    runtime_verdict = validate(_delivered(), view)
    _choice, harness_errors, read_missing = judge_style_pick(_delivered(), params, reads)

    assert (runtime_verdict.status == "accepted") is (not harness_errors)
    assert [e for e in runtime_verdict.errors if "style" in e] == harness_errors
    # A verdict that leans on the ledger is not cached against the candidate.
    assert bool(runtime_verdict.metadata.get("observed_dependent")) is read_missing


class _Counter:
    source = "test-fake"

    def count_text(self, text):
        return max(1, len(text) // 2)

    def count_texts(self, texts):
        return sum(self.count_text(t) for t in texts)

    def count_audio_seconds(self, s):
        return 0


def test_a_search_only_fill_does_not_ask_for_keep_entries() -> None:
    """Batch 2, item 6: with no entry channel there is no transfer chain to
    continue, so the fill's closing order stops asking for `<keep_entries>`;
    a window that can request entries still asks."""

    from finesub.llm.prompts import ContextPack
    from finesub.llm.request_prompts import build_window_task_messages

    window = dataclasses.replace(plan_correction_windows([SubtitleSegment("1", 0.0, 1.0, "一。")], counter=_Counter())[0], length_scale=1.0)

    def fill(indexes):
        return build_window_task_messages(
            request_channel=True, request_reply="<search_queries>\n游戏B\n</search_queries>",
            requested_search=True, knowledge_indexes=indexes, window=window,
            context_pack=ContextPack(), audio_file_label="", search_results="（无）",
            profile=resolve_profile("text", "local", "quality"), knowledge_enabled=bool(indexes),
        )[-1]["content"]

    assert "<keep_entries>" not in fill(None) and "`<next_advice>`" in fill(None)
    assert "<keep_entries>" in fill({"streamer": "- 某主播｜简介"})


@pytest.mark.parametrize(
    "retrieval, indexes, agentic, lead",
    [
        ("local", {"streamer": "- 某主播｜简介"}, False, "需要由 harness 代为取得的输入时的请求块与 `<style_selection>`"),
        ("none", None, False, "只交 `<window_notes>` 与 `<style_selection>`"),
        ("none", None, True, "须已用 `kb_read` 完整读过选定的风格"),
    ],
)
def test_the_reminders_name_the_pick(retrieval, indexes, agentic, lead) -> None:
    """Cold read P002: the user message's first reminder item and its closing
    reminder used to describe a direct delivery; with a pick on the window
    they lead with what the first reply owes (or, agentic, with the read)."""

    from finesub.llm.prompts import ContextPack
    from finesub.llm.request_prompts import build_window_task_messages

    window = dataclasses.replace(plan_correction_windows([SubtitleSegment("1", 0.0, 1.0, "一。")], counter=_Counter())[0], length_scale=1.0)

    def user(catalog):
        return build_window_task_messages(
            request_channel=not agentic, agentic=agentic, knowledge_indexes=indexes,
            style_catalog=catalog, window=window, context_pack=ContextPack(), audio_file_label="",
            profile=resolve_profile("text", retrieval, "quality"), knowledge_enabled=bool(indexes),
        )[1]["content"]

    picked = user("甲组｜简介｜作者｜主播A")
    assert picked.count(lead) == 2  # the first reminder item and the closing one
    assert lead not in user("")


def test_the_request_channel_lists_the_pick_among_the_first_reply_blocks() -> None:
    """Cold read P012 (round 2): with a style to pick, the system's own list of
    what the first reply carries names `<style_selection>` too."""

    from finesub.llm.request_prompts import render_window_request_channel

    picking = render_window_request_channel(knowledge_indexes=None, search=True, style_pick=True)
    plain = render_window_request_channel(knowledge_indexes=None, search=True, style_pick=False)
    first = lambda text: next(line for line in text.splitlines() if line.startswith("1. "))  # noqa: E731
    assert "`<style_selection>`" in first(picking) and "`<style_selection>`" not in first(plain)
    assert "reasoning" not in picking
