"""Reading agy's answer off its transcript after an in-turn retry (`agy_transcript`).

The shapes are the ones real cut-off, broken-stream and empty-reply turns left
(docs/llm_local_agent_agy.md §8).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from finesub.llm.agent.agy_transcript import recover_answer, transcript_path

CONV = "11111111-2222-3333-4444-555555555555"
CUT = "Your previous response was cut off because it exceeded the output token limit"


def user(index: int, text: str = "<USER_REQUEST>task") -> dict:
    return {"step_index": index, "type": "USER_INPUT", "content": text}


def reply(index: int, text: str, **extra) -> dict:
    return {"step_index": index, "type": "PLANNER_RESPONSE", "content": text, **extra}


def error(index: int, text: str = CUT) -> dict:
    return {"step_index": index, "type": "ERROR_MESSAGE", "error": text, "content": f"Error: {text}"}


def tool(index: int) -> dict:
    return {"step_index": index, "type": "GENERIC", "content": "audio"}


def write(data: Path, steps: list, *, raw_lines: tuple[str, ...] = ()) -> None:
    path = transcript_path(CONV, data)
    path.parent.mkdir(parents=True)
    lines = [json.dumps(step, ensure_ascii=False) for step in steps] + list(raw_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recover(data: Path, first_step: int = 0, max_bytes: int = 1_000_000):
    return recover_answer(CONV, first_step, max_bytes=max_bytes, data_dir=data)


def test_a_continuation_in_mid_row_is_joined_on(tmp_path: Path) -> None:
    write(tmp_path, [
        user(0), reply(1, "", tool_calls=[{"name": "view_file"}]), tool(2),
        reply(3, "<rows>\nq465|one|嘉宾"), error(4), reply(5, "|main|ref|\n</rows>\n"),
    ])

    got = recover(tmp_path)

    assert got.text == "<rows>\nq465|one|嘉宾|main|ref|\n</rows>\n"
    assert (got.segments, got.dropped, got.error) == (2, False, CUT)


def test_a_continuation_starting_at_a_later_block_is_joined_on(tmp_path: Path) -> None:
    write(tmp_path, [user(0), reply(1, "<rows>\nq1|one\n</rows>\n"), error(2), reply(3, "<missing>\n</missing>\n")])

    assert recover(tmp_path).text == "<rows>\nq1|one\n</rows>\n<missing>\n</missing>\n"


def test_a_rewrite_after_leaked_reasoning_keeps_only_the_rewrite(tmp_path: Path) -> None:
    """A production correction window (2026-10-08): the first reply was
    reasoning, with a drafted `<tracks>` block deep inside it; the model then
    wrote the whole answer again from its first block."""

    draft = "Ja, the whole audio...\n" + "x" * 500 + "\n<tracks>\ns1 = a\n</tracks>\n" + "more"
    answer = "<reasoning>\nok\n</reasoning>\n<translated>\n1|a\n</translated>\n"
    write(tmp_path, [user(0), reply(1, draft), error(2), reply(3, "  \n" + answer)])

    got = recover(tmp_path)

    assert got.text == "  \n" + answer
    assert (got.segments, got.dropped) == (1, True)


def test_a_rewrite_after_a_cut_off_answer_keeps_the_last_one(tmp_path: Path) -> None:
    write(tmp_path, [user(0), reply(1, "<people>\n</people>\n<rows>\nq1|"), error(2),
                     reply(3, "<people>\n</people>\n<rows>\nq1|one\n</rows>\n")])

    assert recover(tmp_path).text == "<people>\n</people>\n<rows>\nq1|one\n</rows>\n"


def test_free_text_is_joined_whole_and_leaked_reasoning_left_out(tmp_path: Path) -> None:
    write(tmp_path, [user(0), reply(1, "thought\nI should look first."), error(2),
                     reply(3, "The answer is "), error(4), reply(5, "42.")])

    got = recover(tmp_path)

    assert got.text == "The answer is 42."
    assert (got.segments, got.dropped) == (2, True)


def test_an_empty_reply_retry_gives_the_reply_after_it(tmp_path: Path) -> None:
    """The one shape that ends `SUCCESS`: the response is that same reply."""

    write(tmp_path, [user(0), reply(1, ""), error(2, "model output error: empty"), reply(3, "<rows>\nq1\n</rows>\n")])

    got = recover(tmp_path)

    assert got.text == "<rows>\nq1\n</rows>\n"
    assert got.error == "model output error: empty"


def test_only_this_turn_counts_in_a_resumed_conversation(tmp_path: Path) -> None:
    """A repair round resumes the conversation: the earlier turn's recovered
    answer is not this turn's, and this turn had no in-turn error."""

    write(tmp_path, [user(0), reply(1, "<rows>\nold"), error(2), reply(3, " rest\n</rows>\n"),
                     user(4, "<USER_REQUEST>fix it"), reply(5, "<rows>\nnew\n</rows>\n")])

    assert recover(tmp_path, first_step=0) is not None
    assert recover(tmp_path, first_step=4) is None


@pytest.mark.parametrize(
    "steps",
    [
        [user(0), reply(1, "<rows>\nq1")],  # no in-turn error
        [user(0), reply(1, "<rows>\nq1"), error(2)],  # nothing after it
        [user(0), reply(1, "<rows>\nq1"), error(2), reply(3, "  ")],  # only blank
        [user(0), reply(1, "<rows>\nq1"), error(2), reply(3, "", tool_calls=[{"name": "x"}]), tool(4)],
        [user(0), error(1), reply(2, "<rows>\nq1"), reply(3, "", tool_calls=[{"name": "x"}]), tool(4)],
    ],
)
def test_no_answer_after_the_last_error_recovers_nothing(tmp_path: Path, steps: list) -> None:
    write(tmp_path, steps)

    assert recover(tmp_path) is None


def test_unreadable_lines_and_string_indexes(tmp_path: Path) -> None:
    write(tmp_path, [user(0), {**reply(1, "<rows>\nq"), "step_index": "1"}, error(2), reply(3, "1\n</rows>\n")],
          raw_lines=("{not json", "[1, 2]"))

    assert recover(tmp_path).text == "<rows>\nq1\n</rows>\n"


def test_missing_transcript_bad_id_or_oversized_answer_recovers_nothing(tmp_path: Path) -> None:
    assert recover(tmp_path) is None
    write(tmp_path, [user(0), reply(1, "<rows>\nq"), error(2), reply(3, "1\n</rows>\n")])
    assert recover(tmp_path, max_bytes=10) is None
    assert recover_answer("../escape", 0, max_bytes=1000, data_dir=tmp_path) is None
