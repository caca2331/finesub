from __future__ import annotations

import json

import pytest

from llm.content_filter import (
    BLACKLIST_ARTIFACT_KIND,
    ContentFilterExhaustedError,
    DROPPED_UNITS_NOTE,
    InjectionUnit,
    evidence_pack_block,
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


def test_split_rendered_search_block_empty_and_pack() -> None:
    assert split_rendered_search_block("").units == ()
    pack = evidence_pack_block("## 结论\nF1 confirmed：…")
    assert len(pack.units) == 1 and pack.units[0].kind == "evidence_pack"
    assert pack.render(()) == DROPPED_UNITS_NOTE  # everything dropped, note only


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
    from llm.content_filter import (
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
