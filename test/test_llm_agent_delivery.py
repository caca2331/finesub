"""`agent/delivery.py`: did each harness reply reach the model
(docs/llm_agent_tool_protocol.md §6)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from finesub.llm.agent.channels import ChannelLimits, fit_reply
from finesub.llm.agent.delivery import check, sent_replies


def _reply(reply_id: str, payload=None, error: str = "") -> dict:
    return fit_reply(payload=payload, error=error, reply_id=reply_id, limits=ChannelLimits(),
                     spill_root=Path("unused"), spill_id="x")


def _frame_log(tmp_path: Path, calls: list[tuple[str, str, dict]], session: str = "s1") -> Path:
    """A frame log of ``calls`` (tool, callId, CallToolResult), as `serve` writes it."""

    log = tmp_path / "mcp-frames.jsonl"
    rows = []
    for number, (tool, call_id, result) in enumerate(calls, start=1):
        params = {"name": tool, "arguments": {}, **({"_meta": {"callId": call_id}} if call_id else {})}
        rows.append({"dir": "in", "session": session, "frame": json.dumps(
            {"jsonrpc": "2.0", "id": number, "method": "tools/call", "params": params})})
        rows.append({"dir": "out", "session": session, "frame": json.dumps(
            {"jsonrpc": "2.0", "id": number, "result": result}, ensure_ascii=False)})
    # Another session's frames in the same log are not this session's.
    rows.append({"dir": "in", "session": "other", "frame": json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "kb_read"}})})
    log.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return log


def _text(result: dict) -> str:
    return result["content"][0]["text"]


TASK = _reply("r-000000000001", {"status": "task", "text": "台詞「引用」\n" * 200})
ERROR = _reply("r-000000000002", error="unknown ref")
ACCEPTED = _reply("r-000000000003", {"status": "accepted"})


def _sent(tmp_path, *, origin: str = "call_ab12"):
    log = _frame_log(tmp_path, [("next_task", origin, TASK), ("read_context", origin, ERROR),
                                ("submit", origin, ACCEPTED)])
    return sent_replies(log, "s1")


def _stream_json(tmp_path: Path, texts: list[str]) -> Path:
    """Claude Code's stream-json: each reply comes back as a `tool_result`."""

    path = tmp_path / "raw.jsonl"
    path.write_text("".join(json.dumps({"type": "user", "message": {"content": [
        {"type": "tool_result", "tool_use_id": f"t{index}", "content": [{"type": "text", "text": text}]}]}},
        ensure_ascii=False) + "\n" for index, text in enumerate(texts)), encoding="utf-8")
    return path


def test_the_sent_side_reads_ids_sizes_and_origins_of_one_session(tmp_path) -> None:
    sent = _sent(tmp_path, origin="exec-6f1c")
    assert [(reply.reply_id, reply.tool, reply.origin) for reply in sent] == [
        ("r-000000000001", "next_task", "exec"),
        ("r-000000000002", "read_context", "exec"),
        ("r-000000000003", "submit", "exec"),
    ]
    assert sent[0].bytes == len(_text(TASK).encode("utf-8")) and sent[2].accepted
    assert sent_replies(_frame_log(tmp_path, [("kb_read", "", TASK)]), "s1")[0].origin == "unknown"


def test_a_reply_the_model_saw_whole_is_delivered(tmp_path) -> None:
    record = _stream_json(tmp_path, [_text(TASK), _text(ERROR), _text(ACCEPTED)])
    booked = check(_sent(tmp_path), [(record, "model")])
    assert booked["evidence"] == "model" and booked["not_delivered"] == 0
    assert [row["status"] for row in booked["replies"]] == ["delivered"] * 3


def test_a_cut_on_the_model_side_is_not_delivered(tmp_path) -> None:
    """Head and tail kept, the middle elided: still shorter than sent."""

    text = _text(TASK)
    cut = text[:300] + "\n…truncated…\n" + text[-300:]
    record = _stream_json(tmp_path, [cut, _text(ERROR), _text(ACCEPTED)])
    booked = check(_sent(tmp_path), [(record, "model")])
    assert [row["status"] for row in booked["replies"]] == ["not_delivered", "delivered", "delivered"]
    assert booked["not_delivered"] == 1


def test_a_reply_replaced_by_a_file_path_is_not_delivered(tmp_path) -> None:
    """agy past its inline limit: the step names a saved file, not the reply."""

    path = tmp_path / "transcript_full.jsonl"
    path.write_text(json.dumps({"type": "TOOL", "content": "The output was large and was saved to: file:///x/output.txt"})
                    + "\n", encoding="utf-8")
    booked = check(_sent(tmp_path), [(path, "model")])
    assert booked["not_delivered"] == 3


def test_framing_and_reserialising_do_not_read_as_a_cut(tmp_path) -> None:
    """agy puts a header before the reply; a Codex rollout carries the whole
    result, compact, inside `function_call_output`."""

    compact = json.dumps(json.loads(_text(TASK)), ensure_ascii=False, separators=(",", ":"))
    agy = tmp_path / "agy.jsonl"
    agy.write_text(json.dumps({"content": "Created At: 2026-10-23\nCompleted At: 2026-10-23\n" + _text(ERROR)})
                   + "\n", encoding="utf-8")
    rollout = tmp_path / "rollout.jsonl"
    output = json.dumps({"content": [{"type": "text", "text": compact}], "isError": False})
    rollout.write_text(json.dumps({"type": "response_item", "payload": {
        "type": "function_call_output", "output": output}}) + "\n" + json.dumps(
        {"payload": {"output": _text(ACCEPTED)}}) + "\n", encoding="utf-8")
    booked = check(_sent(tmp_path), [(agy, "model"), (rollout, "model")])
    assert [row["status"] for row in booked["replies"]] == ["delivered"] * 3


def test_only_the_clis_copy_and_no_record_are_never_judged(tmp_path) -> None:
    text = _text(TASK)
    cli = tmp_path / "events.jsonl"
    cli.write_text(json.dumps({"type": "item.completed", "item": {"type": "mcp_tool_call", "result": {
        "content": [{"type": "text", "text": text[:500]}]}}}) + "\n", encoding="utf-8")
    booked = check(_sent(tmp_path), [(tmp_path / "missing-rollout.jsonl", "model"), (cli, "cli")])
    assert booked["evidence"] == "cli" and booked["not_delivered"] == 0
    assert booked["replies"][0]["status"] == "unconfirmed"
    nothing = check(_sent(tmp_path), [])
    assert nothing["evidence"] == "unknown" and {row["status"] for row in nothing["replies"]} == {"unknown"}


def test_an_accept_held_by_exit_on_accept_is_held_not_lost(tmp_path) -> None:
    record = _stream_json(tmp_path, [_text(TASK), _text(ERROR)])
    booked = check(_sent(tmp_path), [(record, "model")], ended_on_accept=True)
    assert [row["status"] for row in booked["replies"]] == ["delivered", "delivered", "held"]
    assert check(_sent(tmp_path), [(record, "model")])["not_delivered"] == 1


def test_a_batch_is_judged_by_its_own_reply(tmp_path) -> None:
    batch = _reply("r-0000000000b1", {"results": [{"tool": "kb_read", "result": {"text": "x" * 500}}]})
    log = _frame_log(tmp_path, [("batch", "call_1", batch)])
    sent = sent_replies(log, "s1")
    assert [(reply.reply_id, reply.tool) for reply in sent] == [("r-0000000000b1", "batch")]
    assert check(sent, [(_stream_json(tmp_path, [_text(batch)]), "model")])["not_delivered"] == 0


def test_the_driver_books_delivery_and_warns_once(tmp_path, reported, monkeypatch) -> None:
    from finesub.llm.agent import local_agent
    from finesub.llm.agent.local_agent import ClaudeCodeLocalAgentDriver

    monkeypatch.setattr(local_agent, "_WARNED_DELIVERY", set())
    log = _frame_log(tmp_path, [("next_task", "exec-1", TASK)])
    text = _text(TASK)
    raw = _stream_json(tmp_path, [text[:100]])
    with raw.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "result", "is_error": True,
                                 "result": "API Error: Claude's response exceeded the 32000 output token maximum."}) + "\n")
    driver = ClaudeCodeLocalAgentDriver()
    capsule = SimpleNamespace(raw_events_path=raw, episode_id="ep-1")
    server = {"env": {"FINESUB_MCP_LOG": str(log), "FINESUB_MCP_SESSION": "s1"}}
    for _ in range(2):
        attempt: dict = {}
        driver._check_delivery(capsule, attempt, [], server)
    assert attempt["delivery"]["not_delivered"] == 1 and attempt["delivery"]["output_cuts"] == 1
    # An `exec-` origin means something only on a CLI with code mode.
    assert "exec_calls" not in attempt["delivery"]
    codes = [warning.code for warning in reported.warnings]
    assert sorted(codes) == ["agent-output-cut", "agent-reply-not-delivered"]


def test_codex_warns_once_about_a_call_from_inside_exec(tmp_path, reported, monkeypatch) -> None:
    from finesub.llm.agent import local_agent
    from finesub.llm.agent.local_agent import CodexLocalAgentDriver

    monkeypatch.setattr(local_agent, "_WARNED_DELIVERY", set())
    log = _frame_log(tmp_path, [("next_task", "exec-1", TASK)])
    driver = CodexLocalAgentDriver()
    capsule = SimpleNamespace(raw_events_path=_stream_json(tmp_path, [_text(TASK)]), episode_id="ep-1")
    attempt: dict = {}
    driver._check_delivery(capsule, attempt, [], {"env": {"FINESUB_MCP_LOG": str(log), "FINESUB_MCP_SESSION": "s1"}})
    assert attempt["delivery"]["exec_calls"] == 1 and attempt["delivery"]["evidence"] == "cli"
    assert [warning.code for warning in reported.warnings] == ["agent-codex-exec-call"]


def test_agy_counts_this_turns_cut_off_continuations(tmp_path) -> None:
    from finesub.llm.agent.agy_transcript import transcript_path, turn_cut_offs

    path = transcript_path("conv", tmp_path)
    path.parent.mkdir(parents=True)
    steps = [
        {"step_index": 3, "type": "ERROR_MESSAGE", "content": "Your previous response was cut off. Please continue"},
        {"step_index": 9, "type": "ERROR_MESSAGE", "content": "Your previous response was cut off. Please continue"},
        {"step_index": 10, "type": "ERROR_MESSAGE", "content": "empty response"},
    ]
    path.write_text("".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8")
    assert turn_cut_offs("conv", 5, data_dir=tmp_path) == 1


def _spilled_stub(reply_id: str, spill_id: str, parts: int, files: str = "") -> dict:
    spilled = {"spill_id": spill_id, "parts": parts, **({"files": files} if files else {})}
    body = {"reply_id": reply_id, "status": "task", "spilled": spilled, "note": "read the parts"}
    return {"content": [{"type": "text", "text": json.dumps(body)}], "isError": False}


def test_a_spilled_reply_counts_only_once_every_part_was_read(tmp_path) -> None:
    """The stub reaching the model is not enough: each `read_spill` part
    must be delivered too."""

    stub = _spilled_stub("r-0000000000a1", "mcp-s1", 2)
    part1 = _reply("r-0000000000a2", {"spill_id": "mcp-s1", "part": 1, "parts": 2, "text": "x" * 300})
    part2 = _reply("r-0000000000a3", {"spill_id": "mcp-s1", "part": 2, "parts": 2, "text": "y" * 300})
    log = tmp_path / "log.jsonl"

    def sent_for(last):
        rows = []
        for number, (tool, arguments, result) in enumerate([
            ("next_task", {}, stub),
            ("read_spill", {"spill_id": "mcp-s1", "part": 1}, part1),
            ("read_spill", {"spill_id": "mcp-s1", "part": 2}, last),
        ], start=1):
            rows.append({"dir": "in", "session": "s1", "frame": json.dumps(
                {"jsonrpc": "2.0", "id": number, "method": "tools/call", "params": {"name": tool, "arguments": arguments}})})
            rows.append({"dir": "out", "session": "s1", "frame": json.dumps({"jsonrpc": "2.0", "id": number, "result": result})})
        log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return sent_replies(log, "s1")

    sent = sent_for(part2)
    assert sent[0].spill == ("mcp-s1", 2, ()) and sent[1].reads_part == ("mcp-s1", 1)
    whole = check(sent, [(_stream_json(tmp_path, [_text(stub), _text(part1), _text(part2)]), "model")])
    assert whole["replies"][0]["status"] == "delivered" and whole["replies"][0]["parts_read"] == "2/2"
    half = check(sent, [(_stream_json(tmp_path, [_text(stub), _text(part1)]), "model")])
    assert half["replies"][0]["status"] == "parts_unread" and half["not_delivered"] == 2

    # A part answered with an error (a lapsed hold) reached the model as an
    # error, not as the part.
    lapsed = _reply("r-0000000000a4", error="this session's hold on the task lapsed")
    sent = sent_for(lapsed)
    assert sent[2].reads_part is None
    errored = check(sent, [(_stream_json(tmp_path, [_text(stub), _text(part1), _text(lapsed)]), "model")])
    assert errored["replies"][0]["status"] == "parts_unread" and errored["replies"][0]["parts_read"] == "1/2"


def test_spilled_files_count_when_a_file_tool_opened_them(tmp_path) -> None:
    """agy: a part is opened when a record string is exactly its path (the
    `view_file` argument), not when a prompt lists it."""

    files = [str(tmp_path / "spill" / f"part-0{index}.md") for index in (1, 2)]
    stub = _spilled_stub("r-0000000000c1", "mcp-s2", 2, "\n".join(files))
    log = _frame_log(tmp_path, [("kb_read", "", stub)])
    transcript = tmp_path / "transcript.jsonl"

    def write(opened):
        steps = [{"type": "USER_INPUT", "content": "open these:\n" + "\n".join(files)},
                 {"type": "TOOL", "content": _text(stub)}]
        steps += [{"type": "PLANNER_RESPONSE", "tool_calls": [{"name": "view_file", "args": {"AbsolutePath": path}}]}
                  for path in opened]
        transcript.write_text("".join(json.dumps(step) + "\n" for step in steps), encoding="utf-8")

    write(files)
    assert check(sent_replies(log, "s1"), [(transcript, "model")])["replies"][0]["status"] == "delivered"
    write(files[:1])
    assert check(sent_replies(log, "s1"), [(transcript, "model")])["replies"][0]["status"] == "parts_unread"
    named = check([], [(transcript, "model")], named_files=files)
    assert named["named_files"] == {"named": 2, "opened": 1} and named["files_unopened"] == 1


def test_a_string_carrying_several_replies_or_a_cut_mark_is_not_counted(tmp_path) -> None:
    """An `exec` that printed two replies cannot say how much of each came
    through; one Codex marked as truncated did not."""

    first = _reply("r-0000000000d1", {"text": "a" * 400})
    second = _reply("r-0000000000d2", {"text": "b" * 400})
    log = _frame_log(tmp_path, [("kb_read", "exec-1", first), ("kb_read", "exec-2", second)])
    both = _stream_json(tmp_path, [_text(first)[:200] + "\n" + _text(second) + "\nmore output " * 50])
    assert [row["status"] for row in check(sent_replies(log, "s1"), [(both, "model")])["replies"]] == ["unknown", "unknown"]
    cut = _stream_json(tmp_path, ["Warning: truncated output (original token count: 9)\n" + _text(first) + " " * 900])
    assert check(sent_replies(log, "s1")[:1], [(cut, "model")])["replies"][0]["status"] == "not_delivered"
    both_cut = _stream_json(tmp_path, ["Warning: truncated output (original token count: 9)\n"
                                       + _text(first) + "\n" + _text(second)])
    booked = check(sent_replies(log, "s1"), [(both_cut, "model")])
    assert [row["status"] for row in booked["replies"]] == ["not_delivered", "not_delivered"]


def test_a_capsule_call_books_its_output_cuts(tmp_path, reported, monkeypatch) -> None:
    from finesub.llm.agent import local_agent
    from finesub.llm.agent.local_agent import ClaudeCodeLocalAgentDriver

    monkeypatch.setattr(local_agent, "_WARNED_DELIVERY", set())
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps({"type": "result", "is_error": True,
                               "result": "API Error: Claude's response exceeded the 32000 output token maximum."}) + "\n",
                   encoding="utf-8")
    attempt: dict = {}
    ClaudeCodeLocalAgentDriver()._check_delivery(SimpleNamespace(raw_events_path=raw, episode_id="ep"), attempt, [], None)
    assert attempt["delivery"]["output_cuts"] == 1 and attempt["delivery"]["replies"] == []
    assert [warning.code for warning in reported.warnings] == ["agent-output-cut"]
    quiet: dict = {}
    raw.write_text("{}\n", encoding="utf-8")
    ClaudeCodeLocalAgentDriver()._check_delivery(SimpleNamespace(raw_events_path=raw, episode_id="ep"), quiet, [], None)
    assert "delivery" not in quiet
