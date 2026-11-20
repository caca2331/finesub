from __future__ import annotations

import json

import pytest

from finesub.llm.content_filter import (
    BLACKLIST_ARTIFACT_KIND,
    ContentFilterExhaustedError,
    DROPPED_UNITS_NOTE,
    InjectionUnit,
    load_content_filter_blacklist,
    run_content_filter_ladder,
    split_rendered_search_block,
    strip_blacklisted_units,
)


_RENDERED = (
    "<search_results>\n"
    "--- query: 原神 5.0 新角色 ---\n"
    "provider: exa\n"
    "- wiki (https://example.test/a)\n"
    "  角色介绍。\n"
    "\n"
    "--- 深度提取 url: https://example.test/page ---\n"
    "整页内容第一行。\n"
    "整页内容第二行。\n"
    "\n"
    "--- query: 主播 近况 ---\n"
    "provider: tavily\n"
    "- news (https://example.test/b)\n"
    "  近况摘要。\n"
    "\n"
    "（注入预算说明：1 条 query 结果被截断。）\n"
    "</search_results>"
)


def test_split_rendered_search_block_units_and_scaffolding() -> None:
    block = split_rendered_search_block(_RENDERED)

    assert [unit.kind for unit in block.units] == [
        "query_results",
        "url_extract",
        "query_results",
    ]
    assert block.units[0].stable_id == "原神 5.0 新角色"
    assert block.units[1].stable_id == "https://example.test/page"
    assert "<search_results>" in block.preamble
    assert "注入预算说明" in block.tail
    assert "</search_results>" in block.tail

    # Full rebuild keeps every unit and the scaffolding.
    full = block.render(block.units)
    for fragment in ("原神 5.0 新角色", "整页内容第一行。", "主播 近况", "注入预算说明"):
        assert fragment in full
    assert DROPPED_UNITS_NOTE not in full

    # Dropping the extract removes exactly that section and marks the drop.
    reduced = block.render([u for u in block.units if u.kind != "url_extract"])
    assert "整页内容第一行。" not in reduced
    assert "原神 5.0 新角色" in reduced and "主播 近况" in reduced
    assert DROPPED_UNITS_NOTE in reduced


def test_split_rendered_search_block_empty() -> None:
    assert split_rendered_search_block("").units == ()


def test_ladder_passes_through_when_not_blocked() -> None:
    unit = InjectionUnit(kind="query_results", stable_id="q", text="--- query: q ---\nx")
    outcome = run_content_filter_ladder(
        units=[unit],
        call=lambda active: {"blocked": False, "n": len(active)},
        blocked=lambda result: result["blocked"],
        stage="test",
        sleep=lambda _s: None,
    )
    assert outcome.level == -1 and outcome.attempts == 1
    assert not outcome.dropped_units


def test_ladder_leave_one_out_identifies_toxic_url_unit() -> None:
    toxic = InjectionUnit(kind="url_extract", stable_id="u-bad", text="bad page")
    ok_url = InjectionUnit(kind="url_extract", stable_id="u-ok", text="ok page")
    query = InjectionUnit(kind="query_results", stable_id="q", text="q text")
    calls: list[list[str]] = []

    def call(active):
        ids = [unit.stable_id for unit in active]
        calls.append(ids)
        return {"blocked": "u-bad" in ids}

    outcome = run_content_filter_ladder(
        units=[toxic, ok_url, query],
        call=call,
        blocked=lambda result: result["blocked"],
        stage="test",
        sleep=lambda _s: None,
    )
    # Full set blocked -> drop toxic first (leave-one-out) -> pass.
    assert outcome.level == 1
    assert [unit.stable_id for unit in outcome.identified_units] == ["u-bad"]
    assert [unit.stable_id for unit in outcome.dropped_units] == ["u-bad"]
    assert calls == [["u-bad", "u-ok", "q"], ["u-ok", "q"]]


def test_ladder_falls_through_to_wholesale_drops_and_exhaustion() -> None:
    url_a = InjectionUnit(kind="url_extract", stable_id="a", text="a")
    query = InjectionUnit(kind="query_results", stable_id="q", text="q")

    # Blocked unless every unit is gone -> rung 3.
    outcome = run_content_filter_ladder(
        units=[url_a, query],
        call=lambda active: {"blocked": bool(active)},
        blocked=lambda result: result["blocked"],
        stage="test",
        sleep=lambda _s: None,
    )
    assert outcome.level == 3
    assert {unit.stable_id for unit in outcome.dropped_units} == {"a", "q"}
    assert not outcome.identified_units

    # Always blocked -> exhausted with a diagnosable error.
    with pytest.raises(ContentFilterExhaustedError):
        run_content_filter_ladder(
            units=[url_a, query],
            call=lambda active: {"blocked": True},
            blocked=lambda result: result["blocked"],
            stage="窗口 0001",
            sleep=lambda _s: None,
        )


def test_ladder_blocked_exception_mode_and_plain_retry() -> None:
    class Blocked(Exception):
        pass

    attempts: list[int] = []

    def call(active):
        attempts.append(len(active))
        if len(attempts) == 1:
            raise Blocked()
        return "ok"

    outcome = run_content_filter_ladder(
        units=[],
        call=call,
        blocked_exception=Blocked,
        stage="test",
        plain_retry=True,
        sleep=lambda _s: None,
    )
    assert outcome.level == 0 and outcome.attempts == 2

    # Without units and without plain retry, a block exhausts immediately.
    with pytest.raises(ContentFilterExhaustedError):
        run_content_filter_ladder(
            units=[],
            call=lambda active: (_ for _ in ()).throw(Blocked()),
            blocked_exception=Blocked,
            stage="test",
            sleep=lambda _s: None,
        )


def test_blacklist_roundtrip(tmp_path) -> None:
    unit = InjectionUnit(kind="url_extract", stable_id="u", text="bad page")
    other = InjectionUnit(kind="query_results", stable_id="q", text="fine")
    artifact = {
        "kind": BLACKLIST_ARTIFACT_KIND,
        "payload": {"content_hash": unit.content_hash, "stable_id": "u"},
    }
    (tmp_path / "task-artifacts.jsonl").write_text(
        json.dumps(artifact, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    blacklist = load_content_filter_blacklist(tmp_path)
    assert unit.content_hash in blacklist
    kept, stripped = strip_blacklisted_units([unit, other], blacklist)
    assert [u.stable_id for u in kept] == ["q"]
    assert [u.stable_id for u in stripped] == ["u"]
    assert load_content_filter_blacklist(None) == set()


def test_run_injection_ladder_records_blacklist(tmp_path) -> None:
    from finesub.llm.content_filter import (
        LADDER_ARTIFACT_KIND,
        run_injection_ladder,
        split_rendered_search_block,
    )

    block = split_rendered_search_block(
        "--- query: ok ---\nsafe\n\n"
        "--- 深度提取 url: https://bad.test ---\ntoxic\n"
    )
    toxic = next(u for u in block.units if u.kind == "url_extract")
    blacklist: set[str] = set()
    calls: list[str] = []

    def call(text: str) -> str:
        calls.append(text)
        if "toxic" in text:
            raise RuntimeError("blocked")
        return "ok"

    outcome = run_injection_ladder(
        block=block,
        call=call,
        stage="research_round1",
        blocked_exception=RuntimeError,
        blacklist=blacklist,
        task_artifact_dir=tmp_path,
        task_id="t1",
        sleep=lambda _s: None,
    )
    assert outcome.level == 1
    assert toxic.content_hash in blacklist
    assert outcome.result == "ok"
    records = [
        json.loads(line)
        for line in (tmp_path / "task-artifacts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    kinds = {record["kind"] for record in records}
    assert LADDER_ARTIFACT_KIND in kinds
    assert BLACKLIST_ARTIFACT_KIND in kinds
    # Resume loader sees the new blacklist entry.
    assert toxic.content_hash in load_content_filter_blacklist(tmp_path)
    # Blacklist pre-strips on the next ladder before any call.
    calls.clear()
    second = run_injection_ladder(
        block=block,
        call=call,
        stage="research_round2",
        blocked_exception=RuntimeError,
        blacklist=blacklist,
        sleep=lambda _s: None,
    )
    assert second.level == -1 and second.attempts == 1
    assert calls and "toxic" not in calls[0]


class _Blocked(RuntimeError):
    def __init__(self, model: str) -> None:
        super().__init__(f"blocked by {model}")
        self.model = model


def _chain(blocking: set[str], avoided: set[str], calls: list[tuple[str, str]]):
    """A two-model chain behind a stage's call: the first model not avoided answers."""

    from finesub.llm.client import NoEligibleTargetError

    def call(text: str) -> str:
        model = next((m for m in ("gemini/a", "gemini/b") if m not in avoided), None)
        if model is None:
            raise NoEligibleTargetError("No eligible target for role x (content_filter_blockedx2).")
        calls.append((model, text))
        if model in blocking:
            raise _Blocked(model)
        return f"answer from {model}"

    return call


def test_an_exhausted_ladder_starts_over_on_the_next_model(tmp_path) -> None:
    from finesub.llm.content_filter import MODEL_SWITCH_ARTIFACT_KIND, run_injection_ladder

    block = split_rendered_search_block("--- 深度提取 url: https://x.test ---\npage\n")
    avoided: set[str] = set()
    calls: list[tuple[str, str]] = []
    outcome = run_injection_ladder(
        block=block, call=_chain({"gemini/a"}, avoided, calls), stage="窗口 0003",
        blocked_exception=_Blocked, task_artifact_dir=tmp_path, task_id="t", avoided=avoided,
        sleep=lambda _s: None,
    )

    assert outcome.result == "answer from gemini/b"
    assert avoided == {"gemini/a"} and outcome.avoided_models == ["gemini/a"]
    assert outcome.recovered
    # The next model gets the whole prompt again, injection included.
    assert calls[-1] == ("gemini/b", block.render(block.units))
    assert [model for model, _ in calls].count("gemini/a") == 2  # leave-one-out, then drop-all (deduped)
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "task-artifacts.jsonl").read_text("utf-8").splitlines()]
    assert kinds == [MODEL_SWITCH_ARTIFACT_KIND]


def test_the_ladder_gives_up_once_every_model_blocked() -> None:
    from finesub.llm.content_filter import run_injection_ladder

    avoided: set[str] = set()
    with pytest.raises(ContentFilterExhaustedError) as raised:
        run_injection_ladder(
            block=split_rendered_search_block(""), call=_chain({"gemini/a", "gemini/b"}, avoided, []),
            stage="prepare_task", blocked_exception=_Blocked, plain_retry=True, avoided=avoided,
            sleep=lambda _s: None,
        )
    assert raised.value.models == ["gemini/a", "gemini/b"]
    assert "on gemini/a, gemini/b" in str(raised.value)


def test_without_a_session_set_the_ladder_does_not_switch_models() -> None:
    from finesub.llm.content_filter import run_injection_ladder

    calls: list[tuple[str, str]] = []
    with pytest.raises(ContentFilterExhaustedError) as raised:
        run_injection_ladder(
            block=split_rendered_search_block(""), call=_chain({"gemini/a"}, set(), calls),
            stage="s", blocked_exception=_Blocked, plain_retry=True, sleep=lambda _s: None,
        )
    assert {model for model, _ in calls} == {"gemini/a"}
    assert raised.value.models == ["gemini/a"]


def test_a_model_a_parallel_half_already_avoided_still_moves_this_ladder_on() -> None:
    """The other half of a split window added gemini/a while this ladder was on it."""

    from finesub.llm.content_filter import run_injection_ladder

    avoided: set[str] = set()
    calls: list[tuple[str, str]] = []
    chain = _chain({"gemini/a"}, set(), calls)  # still routes to gemini/a, as it did when the ladder began

    def call(text: str) -> str:
        if len(calls) < 2:  # this ladder's two attempts both reach gemini/a ...
            if calls:
                avoided.add("gemini/a")  # ... while the other half finishes its own ladder on it
            return chain(text)
        return _chain({"gemini/a"}, avoided, calls)(text)

    outcome = run_injection_ladder(
        block=split_rendered_search_block(""), call=call, stage="窗口 0003-b", blocked_exception=_Blocked,
        plain_retry=True, avoided=avoided, sleep=lambda _s: None,
    )
    assert outcome.result == "answer from gemini/b"
    assert [model for model, _ in calls] == ["gemini/a", "gemini/a", "gemini/b"]


def test_a_session_with_no_model_left_reports_every_model_it_moved_off() -> None:
    """The request turn moved off both models; the delivery then has none to call."""

    from finesub.llm.content_filter import run_injection_ladder

    avoided = {"gemini/a", "gemini/b"}
    with pytest.raises(ContentFilterExhaustedError) as raised:
        run_injection_ladder(
            block=split_rendered_search_block(""), call=_chain(set(), avoided, []), stage="窗口 0003",
            blocked_exception=_Blocked, plain_retry=True, avoided=avoided, sleep=lambda _s: None,
        )
    assert raised.value.models == ["gemini/a", "gemini/b"]
