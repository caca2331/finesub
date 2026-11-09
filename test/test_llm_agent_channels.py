"""`agent/channels.py`: per-driver channel limits and fitting a reply to them
(docs/llm_agent_tool_protocol.md §6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.agent.channels import (
    CHANNEL_ENV,
    REPLY_LIMIT_ENV,
    ChannelLimits,
    fit_reply,
    part_list,
    read_spill_part,
    render_spill,
    reply_size,
    split_parts,
    text_units,
)


@pytest.mark.parametrize(
    "text",
    [
        "".join(f"行{index} 「台詞」\n" for index in range(500)),
        "x" * 100_000,  # one line longer than a part
        "😀" * 30_000 + "\nend\n",  # four bytes a character, cut on characters
        "",
    ],
    ids=["lines", "long-line", "emoji", "empty"],
)
def test_parts_rejoin_to_the_text_and_stay_in_bounds(text) -> None:
    parts = split_parts(text, budget=4_000, max_lines=50)
    assert "".join(parts) == text
    for part in parts:
        assert len(part.encode("utf-8")) <= 4_000
        assert len(part.splitlines()) <= 50


def test_the_renderer_keeps_every_field_its_type_and_text_whole() -> None:
    payload = {
        "status": "task",
        "blocks": {"protocol": "line one\n## not a heading\n", "payload": "a,b\n"},
        "rows": [{"entry": "原神", "n": 1}, "plain"],
        "count": 3,
        "flag": None,
    }
    document = render_spill(payload, skip=["status"])
    assert "@@ status" not in document
    assert "@@ blocks.protocol · text · 26 chars\nline one\n## not a heading\n" in document
    assert '@@ rows · list · 2 items\n{"entry":"原神","n":1}\n"plain"\n' in document
    assert "@@ count · 3\n" in document and "@@ flag · null\n" in document
    assert document.index("blocks.protocol") < document.index("blocks.payload") < document.index("rows")


def _fit(tmp_path, payload, **limits):  # type: ignore[no-untyped-def]
    return fit_reply(
        payload=payload, reply_id="r-1", limits=ChannelLimits(**limits),
        spill_root=tmp_path, spill_id="mcp-x", task_id="t",
    )


def test_a_reply_that_fits_goes_out_whole_with_its_id(tmp_path) -> None:
    reply = _fit(tmp_path, {"status": "ok", "text": "short"}, reply_ok_bytes=1_000)
    assert json.loads(reply["content"][0]["text"]) == {"reply_id": "r-1", "status": "ok", "text": "short"}
    assert not (tmp_path / "mcp-x").exists()
    error = fit_reply(payload=None, error="bad ref", reply_id="r-2", limits=ChannelLimits(),
                      spill_root=tmp_path, spill_id="mcp-y")
    assert error == {"content": [{"type": "text", "text": "[reply r-2] bad ref"}], "isError": True}


def test_a_stub_with_many_parts_falls_back_to_a_range(tmp_path) -> None:
    """Many parts and long paths: the stub names the directory once and the
    parts as a range, and keeps the control fields."""

    deep = tmp_path / ("d" * 60) / ("e" * 60)
    payload = {
        "status": "task", "task_id": "call-1",
        "text": "".join(f"{index:05d}\n" for index in range(40_000)),
    }
    reply = fit_reply(
        payload=payload, reply_id="r-1",
        limits=ChannelLimits(reply_ok_bytes=3_600, reply_error_bytes=3_600, overflow="files",
                             part_bytes=4_000, part_lines=700),
        spill_root=deep, spill_id="mcp-x", task_id="call-1",
    )
    text = reply["content"][0]["text"]
    assert len(text.encode("utf-8")) <= 3_600
    stub = json.loads(text)
    assert stub["status"] == "task" and stub["task_id"] == "call-1"
    assert stub["spilled"]["parts"] > 50
    assert "part-01.md to part-" in stub["spilled"]["files"]
    names = sorted(path.name for path in (deep / "mcp-x").glob("part-*.md"))
    assert len(names) == stub["spilled"]["parts"]
    joined = "".join((deep / "mcp-x" / name).read_text(encoding="utf-8") for name in names)
    assert payload["text"] in joined


def test_read_spill_parts_fit_the_reply_they_travel_in(tmp_path) -> None:
    """A part goes back inside a JSON reply: escapes count against the limit,
    twice where the CLI measures the whole result (Codex)."""

    limits = ChannelLimits(reply_ok_bytes=20_000, reply_error_bytes=20_000, overflow="tool", measure="json")
    payload = {"text": "".join('"q"\\\n' for _ in range(20_000))}
    stub = json.loads(fit_reply(payload=payload, reply_id="r", limits=limits, spill_root=tmp_path,
                                spill_id="mcp-x", task_id="t")["content"][0]["text"])
    text = ""
    for part in range(1, stub["spilled"]["parts"] + 1):
        body = read_spill_part(tmp_path, "mcp-x", part, task_id="t")
        result = {"content": [{"type": "text", "text": json.dumps({"reply_id": "r-123456789012", **body},
                                                                  ensure_ascii=False)}], "isError": False}
        assert reply_size(result, "json") <= 20_000
        text += body["text"]
    assert payload["text"] in text


def test_the_spill_note_reads_parts_one_a_call_where_the_cli_has_batch(tmp_path) -> None:
    """Codex has `batch` instead of parallel calls, and `batch` does not run
    `read_spill`: the note says so rather than inviting one step."""

    payload = {"text": "x" * 5_000}
    notes = {}
    for batch_tool in (False, True):
        limits = ChannelLimits(reply_ok_bytes=2_000, reply_error_bytes=2_000, overflow="tool", batch_tool=batch_tool)
        stub = fit_reply(payload=payload, reply_id="r", limits=limits, spill_root=tmp_path / str(batch_tool),
                         spill_id="mcp-x", task_id="t")
        notes[batch_tool] = json.loads(stub["content"][0]["text"])["note"]
    assert "in one step if you can make parallel calls" in notes[False]
    assert "one part a call" in notes[True] and "cannot run inside batch" in notes[True]
    assert "parallel" not in notes[True]


def test_the_part_list_counts_utf16_units() -> None:
    names = [f"part-{index:02d}.md" for index in range(1, 40)]
    directory = Path("C:/assignments/" + "長" * 100)
    full = part_list(directory, names)
    assert full.count("\n") == len(names) - 1
    bounded = part_list(directory, names, max_units=text_units(full, "utf16") - 1, unit="utf16")
    assert bounded.endswith(f"part-01.md to part-39.md ({len(names)} files, numbered in order)")
    assert text_units("😀", "utf16") == 2 and text_units("😀", "chars") == 1


def test_limits_round_trip_the_environment_and_the_dev_switch_lowers_them(monkeypatch) -> None:
    limits = ChannelLimits(reply_ok_bytes=400_000, reply_error_bytes=10_000, result_size_meta=500_000)
    assert ChannelLimits.from_env(limits.to_env()) == limits
    assert ChannelLimits.from_env("") == ChannelLimits()
    monkeypatch.setenv(REPLY_LIMIT_ENV, "3000,500")
    lowered = ChannelLimits.from_env(limits.to_env())
    assert (lowered.reply_ok_bytes, lowered.reply_error_bytes) == (3000, 500)
    assert CHANNEL_ENV == "FINESUB_MCP_CHANNEL"


def test_the_last_stub_fallback_leaves_every_field_in_the_parts(tmp_path) -> None:
    """Control fields too large for the stub are not lost: the parts carry
    every field."""

    owed = [{"kind": "payload", "ref": f"contexts/{index:064d}/payload.md#sha256:{index:064d}"} for index in range(20)]
    payload = {"status": "repairable", "task_id": "t", "owed_blocks": owed, "message": "fix row 3"}
    reply = fit_reply(
        payload=payload, reply_id="r-1",
        limits=ChannelLimits(reply_ok_bytes=3_600, reply_error_bytes=3_600, overflow="files"),
        spill_root=tmp_path, spill_id="mcp-x", task_id="t",
    )
    stub = json.loads(reply["content"][0]["text"])
    assert "owed_blocks" not in stub and stub["status"] == "repairable"
    parts = "".join(path.read_text(encoding="utf-8") for path in sorted((tmp_path / "mcp-x").glob("part-*.md")))
    assert "@@ owed_blocks · list · 20 items" in parts and owed[-1]["ref"] in parts and "fix row 3" in parts


@pytest.mark.parametrize("measure, budget", [("text", 398_000), ("json", 698_000)])
def test_a_line_longer_than_a_part_is_cut_in_one_pass(measure, budget) -> None:
    import time

    from finesub.llm.agent.channels import part_cost

    line = '台"詞\\' * 250_000
    cost = part_cost(ChannelLimits(overflow="tool", measure=measure, reply_ok_bytes=budget + 2_000))
    started = time.monotonic()
    parts = split_parts(line, budget=budget, cost=cost)
    assert time.monotonic() - started < 2.0
    assert "".join(parts) == line and all(cost(part) <= budget for part in parts)


def test_read_spill_returns_a_part_byte_for_byte(tmp_path) -> None:
    payload = {"text": "a\r\nb\rc\n" * 2_000}
    stub = json.loads(fit_reply(payload=payload, reply_id="r", limits=ChannelLimits(
        reply_ok_bytes=5_000, reply_error_bytes=5_000, overflow="tool"), spill_root=tmp_path,
        spill_id="mcp-x", task_id="t")["content"][0]["text"])
    text = "".join(read_spill_part(tmp_path, "mcp-x", part, task_id="t")["text"]
                   for part in range(1, stub["spilled"]["parts"] + 1))
    assert payload["text"] in text


def test_a_low_dev_limit_keeps_parts_usable() -> None:
    assert ChannelLimits(overflow="tool", reply_ok_bytes=500).spill_part_budget == 250
