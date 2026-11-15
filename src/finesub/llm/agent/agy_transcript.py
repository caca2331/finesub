"""The answer agy wrote after retrying inside its own turn, read off agy's transcript.

When a reply runs past the single-reply output limit (65,536 tokens, thinking
included), the stream breaks off, or the model sends an empty or malformed
reply, agy inserts an ``ERROR_MESSAGE`` step ("Your previous response was cut
off ... Please continue from where you left off") and lets the model go on --
all inside the same ``agy --print`` call. The model usually finishes, but the
``result`` event still reports the last of those errors (``status=ERROR``), or,
after an empty-reply retry, reports ``SUCCESS`` with only the last reply as
``response``. The whole answer is in agy's transcript of the conversation.

The transcript's location and step layout are agy's implementation detail
(measured on agy 1.2-1.3, 2026-10-08 to 13; docs/llm_local_agent_agy.md §8):
a missing file, an unreadable line or an unexpected shape means there is
nothing to recover, never an error.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from finesub import jsonl

#: A transcript this large is not a turn worth parsing for an answer.
MAX_TRANSCRIPT_BYTES = 64 * 1024 * 1024
#: How a reply that is agy's reasoning shipped as text opens (output_protocol.LEAKED_THOUGHT_PREFIX).
_LEAKED_THOUGHT = "thought\n"
_OPENING_TAG = re.compile(r"\s*<([A-Za-z][\w.-]*)(?=[\s>/])")


@dataclass(frozen=True)
class RecoveredAnswer:
    text: str
    #: How many replies the answer was joined from (1: one reply, e.g. after an empty-reply retry).
    segments: int
    #: True when earlier replies were dropped: a later one started the answer
    #: over, or they were leaked reasoning.
    dropped: bool
    #: agy's last in-turn error, as that step's `error` field gives it (what `result.error` repeats).
    error: str


def agy_data_dir() -> Path:
    """agy's own data directory; brain/<conversation id>/ holds the transcript.

    Resolved per call, not at import, so it follows the home directory a caller
    (or a test) has set."""

    return Path.home() / ".gemini" / "antigravity-cli"


def transcript_path(conversation_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or agy_data_dir()) / "brain" / conversation_id / ".system_generated" / "logs" / "transcript_full.jsonl"


def recover_answer(
    conversation_id: str,
    first_step: int,
    *,
    max_bytes: int,
    data_dir: Path | None = None,
) -> RecoveredAnswer | None:
    """This turn's answer when agy retried inside it and the model answered after the last retry.

    ``first_step`` is the smallest ``step_index`` the call's own stream carried:
    the steps before it belong to earlier turns of a resumed conversation.
    ``None`` when the turn had no in-turn error, nothing came after the last
    one, the transcript is missing or unreadable, or the answer exceeds
    ``max_bytes``.
    """

    if not re.fullmatch(r"[0-9A-Za-z-]{8,64}", conversation_id or ""):
        return None
    steps = _turn_steps(transcript_path(conversation_id, data_dir), first_step)
    errors = [k for k, step in enumerate(steps) if step.get("type") == "ERROR_MESSAGE"]
    if not errors or not any(_is_reply(step) for step in steps[errors[-1] + 1:]):
        return None
    tools = [k for k, step in enumerate(steps) if step.get("type") == "GENERIC" or step.get("tool_calls")]
    replies = [str(step["content"]) for step in steps[tools[-1] + 1 if tools else 0:] if _is_reply(step)]
    # Reasoning shipped as a reply is no part of the answer, wherever the join would put it.
    answer = [reply for reply in replies if not reply.lstrip().startswith(_LEAKED_THOUGHT)] or replies
    start = _answer_start(answer)
    text = "".join(answer[start:])
    if not text.strip() or len(text.encode("utf-8")) > max_bytes:
        return None
    dropped = start > 0 or len(answer) < len(replies)
    return RecoveredAnswer(text, len(answer) - start, dropped, str(steps[errors[-1]].get("error") or ""))


def turn_cut_offs(conversation_id: str, first_step: int, *, data_dir: Path | None = None) -> int:
    """How often agy told the model, in this turn, that its reply was cut off
    and to continue (``ERROR_MESSAGE`` steps saying so)."""

    steps = _turn_steps(transcript_path(conversation_id, data_dir), first_step)
    return sum(
        step.get("type") == "ERROR_MESSAGE" and "cut off" in json.dumps(step, ensure_ascii=False)
        for step in steps
    )


def _turn_steps(path: Path, first_step: int) -> list[Mapping[str, Any]]:
    try:
        if path.stat().st_size > MAX_TRANSCRIPT_BYTES:
            return []
        records = list(jsonl.iter_records(path))
    except OSError:
        return []
    steps = []
    for step in records:
        index = _step_index(step.get("step_index"))
        if index is not None and index >= first_step:
            steps.append({**step, "step_index": index})
    steps.sort(key=lambda step: step["step_index"])
    return steps


def _step_index(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _is_reply(step: Mapping[str, Any]) -> bool:
    content = step.get("content")
    return (
        step.get("type") == "PLANNER_RESPONSE"
        and not step.get("tool_calls")
        and isinstance(content, str)
        and bool(content.strip())
    )


def _answer_start(replies: list[str]) -> int:
    """Where the answer begins among the replies.

    The anchor is the tag the first reply that *opens* with a tag opens with: a
    reply opening with it again started the answer over (after a cut-off draft
    or leaked reasoning), so the answer runs from the last such reply to the
    end. A continuation picks up mid-text, or at a later block (`<missing>`,
    `</rows>`), and is joined on. Replies that never open with a tag (free
    text) are joined whole.
    """

    anchor = next((match.group(1) for reply in replies if (match := _OPENING_TAG.match(reply))), None)
    if anchor is None:
        return 0
    return max(k for k, reply in enumerate(replies) if (match := _OPENING_TAG.match(reply)) and match.group(1) == anchor)
