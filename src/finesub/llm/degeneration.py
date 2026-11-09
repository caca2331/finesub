"""Signs that a model's answer degenerated, read from the answer and the window it answered -- no further call.

An answer can pass validation and still be a bad draw: the same pass 2 input, answered three times, merged 3, 0
and 32 of the window's lines read 可并 (`docs/llm_followups.md`「模型回答的退化」). Today these signals are only
recorded in the window's record, and a window left unmerged is named in the stage summary as a note, so their
false alarm rate can be measured on real runs before any of them starts a retry.

- `unmerged` (pass 2): the window offers at least `UNMERGED_MIN_SEAMS` lines read 可并 (the 7 second filter is
  already in the seam: `grouping.group_lines`) and the answer merged at most `UNMERGED_MAX_SHARE` of them. The two
  degenerate draws were 0 / 64 and 3 / 64, the normal one about 31 / 64.

One sign does start a retry: a call asked to think that reported no thinking at all (`thinking_of`), on the
steps that write an answer -- a correction window, the speaker judge, pass 3's proofread. The caller asks once
more in a fresh session and keeps one of the two answers (`pick_after_retry`; owner 2026-10-17).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from .chunking import Seam, SubtitleWindow
from .output_protocol import TranslatedCsvSegment

UNMERGED_MIN_SEAMS = 30
UNMERGED_MAX_SHARE = 0.05


@dataclass(frozen=True)
class Pass2Signals:
    #: the window's own lines read 可并, and how many of them the answer merged with the line before
    may_seams: int
    may_merged: int

    @property
    def signals(self) -> list[str]:
        out = []
        if self.may_seams >= UNMERGED_MIN_SEAMS and self.may_merged <= UNMERGED_MAX_SHARE * self.may_seams:
            out.append("unmerged")
        return out

    def to_metadata(self) -> Dict[str, Any]:
        return {"may_seams": self.may_seams, "may_merged": self.may_merged, "signals": self.signals}


def pass2_signals(window: SubtitleWindow, rows: Sequence[TranslatedCsvSegment]) -> Pass2Signals:
    """The signals of a grouped window's validated answer: `rows` are its rows, by grouped line id."""

    row_of = {source: index for index, row in enumerate(rows) for source in row.source_ids}
    lines = window.segments
    start = len(window.overlap_segments)
    may = merged = 0
    for index in range(max(start, 1), len(lines)):
        line, before = lines[index], lines[index - 1]
        if line.seam != Seam.MAY:
            continue
        may += 1
        merged += line.id in row_of and row_of.get(before.id) == row_of[line.id]
    return Pass2Signals(may, merged)


#: Provider thinking values a call may answer at without thinking at all. A call sent no thinking value (the
#: model's own default, or a model that takes none) may too. The abstract `low` before the model's mapping is
#: not one of them: a model that maps it to `medium` was asked to think (owner 2026-10-17).
LOW_THINKING = frozenset({"off", "none", "minimal", "low"})


@dataclass(frozen=True)
class Thinking:
    """How much one call thought. ``tokens`` is ``None`` when that is not known: no usage came back, the route
    does not report thinking (`reports_thinking`). A resumed conversation's running total stands in for a turn
    with no count of its own: it can only overstate, so its 0 is this turn's 0 too. ``level`` is the provider value sent (after the model's mapping), ``budget`` the token
    budget sent instead of or beside it."""

    tokens: int | None
    level: str = ""
    budget: int = 0

    @property
    def expected(self) -> bool:
        """Whether the call was asked to think."""
        level = self.level.strip().lower()
        return level not in LOW_THINKING and bool(level or self.budget > 0)

    @property
    def skipped(self) -> bool:
        """Asked to think, and reported none."""
        return self.tokens == 0 and self.expected

    @property
    def fine(self) -> bool:
        """Not asked to think, or known to have thought."""
        return not self.expected or (self.tokens or 0) > 0

    def to_metadata(self) -> Dict[str, Any]:
        return {"tokens": self.tokens, "level": self.level, **({"budget": self.budget} if self.budget else {})}


def reports_thinking(backend: str, usage: Any, attempt: Mapping[str, Any] | None = None) -> bool:
    """Whether a call's usage says how much it thought, so that 0 means none.

    - Gemini's API: always (`thoughtsTokenCount`, left out when 0).
    - A local agent: what its driver stamped on the answering attempt (`thinking_reported`,
      `LocalAgentDriver.reports_thinking`): Codex does, agy does for its Gemini models and not for others, Claude
      Code, CodeBuddy and dsh do not. An attempt without the stamp (a conversational session) does not.
    - Any other API: when its usage carries a reasoning count (`completion_tokens_details.reasoning_tokens`;
      Anthropic's folds thinking into the output and leaves it out).
    """

    if backend == "gemini_rest":
        return True
    if backend in ("local_agent", "conversational_agent"):
        return bool((attempt or {}).get("thinking_reported"))
    details = usage.get("completion_tokens_details") if isinstance(usage, Mapping) else None
    return isinstance(details, Mapping) and details.get("reasoning_tokens") is not None


def thinking_of(result: Any) -> Thinking:
    """How much the call behind ``result`` (an `LLMCallResult`) thought, and what it was asked to."""

    from .client import extract_token_distribution

    level, budget = str(getattr(result, "thinking_level", "") or ""), int(getattr(result, "thinking_budget", 0) or 0)
    attempts = [attempt for attempt in getattr(result, "execution_attempts", None) or () if isinstance(attempt, Mapping)]
    attempt = attempts[-1] if attempts else {}
    sent = (attempt.get("isolation") or {}).get("reasoning_effort")
    if isinstance(sent, str):
        level = sent  # what the CLI was really handed (agy caps `xhigh`, a configured effort overrides)
    raw = getattr(result, "raw_response", None)
    raw = raw if isinstance(raw, Mapping) else {}
    usage = raw.get("usageMetadata") or raw.get("usage_metadata") or raw.get("usage")
    tokens = extract_token_distribution(raw)
    if not any(tokens[key] for key in ("prompt_tokens", "total_input_tokens", "output_tokens", "thinking_tokens")):
        return Thinking(None, level, budget)  # usage missing: not a call that skipped thinking
    if not reports_thinking(str(getattr(result, "backend", "") or ""), usage, attempt):
        return Thinking(None, level, budget)
    turn = (attempt.get("usage") or {}).get("turn_reasoning_output_tokens")
    if isinstance(turn, int) and not isinstance(turn, bool):
        return Thinking(turn, level, budget)  # this turn's own, beside a resumed conversation's running total
    return Thinking(int(tokens["thinking_tokens"]), level, budget)


def pick_after_retry(original_errors: int, retry: Thinking, retry_errors: int) -> str:
    """Which answer a window keeps after asking again because the first did not think: ``retry`` or
    ``original``. Errors count what validation refused in each (0: it passed).

    The retry is kept when it thought (or was not asked to) and passed; when it thought and failed, the original.
    When it did not think either, or it is not known, the one with fewer errors, the original on a tie: the task
    really may be that simple."""

    if retry.fine:
        return "retry" if retry_errors == 0 else "original"
    return "retry" if retry_errors < original_errors else "original"


def zero_thinking_record(original: Thinking, retry: Thinking | None, chosen: str, *, error: str = "") -> Dict[str, Any]:
    """What a window's record keeps about a retry for missing thinking."""

    return {"signal": "zero_thinking", "original": original.to_metadata(),
            **({"retry": retry.to_metadata()} if retry is not None else {}),
            **({"retry_error": error} if error else {}), "chosen": chosen}
