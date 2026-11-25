"""Signs that a model's answer degenerated, read from the answer and the window it answered -- no further call.

An answer can pass validation and still be a bad draw: the same pass 2 input, answered three times, merged 3, 0
and 32 of the window's lines read 可并 (`docs/llm_design_notes.md`「模型回答退化的判据与重试的决策记录」).

- `zero_thinking` (a correction window, the speaker judge, pass 3's proofread): a call asked to think reported
  no thinking at all (`thinking_of`).
- `unmerged` (pass 2): the window's lines read 可并 were left apart (`Pass2Signals.unmerged`).

A passing answer with either sign is held and the window asked once more in a fresh session; the two signs share
that one retry, and the window keeps one of the two answers (`pick_after_retry`; owner 2026-10-17, 2026-11-21).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .chunking import Seam, SubtitleWindow
from .output_protocol import TranslatedCsvSegment

#: `unmerged` needs this many seams read 可并 and at most this percent of them merged ...
UNMERGED_MIN_SEAMS = 20
UNMERGED_MAX_PERCENT = 10
#: ... and then, with this many tight seams, at most this percent of those merged; with fewer, none merged at all.
#: A tight seam is one a normal answer merges most: at most `TIGHT_MAX_GAP` s of pause and the two lines together
#: at most `TIGHT_MAX_SPAN` s. A model that merges little when fine (Gemini 4 Hydrogen: 4%-15% of all seams)
#: still merges 17%-19% of the tight ones; a degenerate draw merges almost none. Readings and the plateau the
#: numbers sit on: `docs/llm_design_notes.md`「模型回答退化的判据与重试的决策记录」.
UNMERGED_MIN_TIGHT = 8
UNMERGED_MAX_TIGHT_PERCENT = 10
TIGHT_MAX_GAP = 0.3
TIGHT_MAX_SPAN = 5.0
_EPS = 1e-6  # a gap of exactly 0.3 s is tight however the floats round


@dataclass(frozen=True)
class Pass2Signals:
    #: the window's own lines read 可并, and how many of them the answer merged with the line before
    may_seams: int
    may_merged: int
    #: the same for the tight ones among them
    tight_seams: int = 0
    tight_merged: int = 0

    @property
    def unmerged(self) -> bool:
        if self.may_seams < UNMERGED_MIN_SEAMS or 100 * self.may_merged > UNMERGED_MAX_PERCENT * self.may_seams:
            return False
        if self.tight_seams >= UNMERGED_MIN_TIGHT:
            return 100 * self.tight_merged <= UNMERGED_MAX_TIGHT_PERCENT * self.tight_seams
        # Mostly loose seams: leaving them apart may be right, so only an answer that merged none of them counts.
        return self.may_merged == 0

    @property
    def signals(self) -> list[str]:
        return ["unmerged"] if self.unmerged else []

    def to_metadata(self) -> Dict[str, Any]:
        return {"may_seams": self.may_seams, "may_merged": self.may_merged, "tight_seams": self.tight_seams,
                "tight_merged": self.tight_merged, "signals": self.signals}


def tight(line: Any, before: Any) -> bool:
    """Whether the seam between ``before`` and ``line`` is tight: a short pause (an overlap reads as none) and
    the two lines short together. Times are the grouped lines' own, the ones the window's CSV shows."""

    gap = max(0.0, line.start - before.end)
    return gap <= TIGHT_MAX_GAP + _EPS and line.end - before.start <= TIGHT_MAX_SPAN + _EPS


def pass2_signals(window: SubtitleWindow, rows: Sequence[TranslatedCsvSegment]) -> Pass2Signals:
    """The signals of a grouped window's validated answer: `rows` are its rows, by grouped line id."""

    row_of = {source: index for index, row in enumerate(rows) for source in row.source_ids}
    lines = window.segments
    start = len(window.overlap_segments)
    may = merged = tight_may = tight_merged = 0
    for index in range(max(start, 1), len(lines)):
        line, before = lines[index], lines[index - 1]
        if line.seam != Seam.MAY:
            continue
        joined = line.id in row_of and row_of.get(before.id) == row_of[line.id]
        may += 1
        merged += joined
        if tight(line, before):
            tight_may += 1
            tight_merged += joined
    return Pass2Signals(may, merged, tight_may, tight_merged)


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
      Code does, CodeBuddy and dsh do not. An attempt without the stamp (a conversational session) does not. A
      Claude model never, for the reason below, whichever CLI ran it.
    - Anthropic: never. Its thinking is adaptive -- at an effort level the model decides whether to think, and a
      reply that did not is its call, which the same model asked the same way would make again -- so its count
      (`provider_transports` reads it into the usage for the books) does not trigger the zero-thinking retry.
    - Any other API: when its usage carries a reasoning count (`completion_tokens_details.reasoning_tokens`).
    """

    if backend == "gemini_rest":
        return True
    if backend == "anthropic":
        return False
    if backend in ("local_agent", "conversational_agent"):
        attempt = attempt or {}
        # Claude Code asks for its models by alias (`opus`, `haiku`), so the name alone misses them.
        if attempt.get("driver") == "claude-code":
            return False
        model = str(attempt.get("reported_model") or attempt.get("model") or "").lower()
        return bool(attempt.get("thinking_reported")) and not model.startswith("claude")
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


@dataclass(frozen=True)
class Draw:
    """One answer's degeneration state: how much its call thought, its pass 2 signals (pass 2 only), and
    whether a repair round's patch put it together (a patch is a few rows, so not thinking is no sign there)."""

    thinking: Thinking
    pass2: Pass2Signals | None = None
    patched: bool = False

    def held_for(self) -> List[str]:
        """Why a passing answer is held for the one retry: none, or its signs."""

        return [*(["zero_thinking"] if self.thinking.skipped and not self.patched else []),
                *(["unmerged"] if self.pass2 is not None and self.pass2.unmerged else [])]

    def to_metadata(self) -> Dict[str, Any]:
        return {"thinking": self.thinking.to_metadata(),
                **({"degeneration": self.pass2.to_metadata()} if self.pass2 is not None else {}),
                **({"patched": True} if self.patched else {})}


def pick_after_retry(original: Draw, retry: Draw, retry_passed: bool) -> str:
    """Which answer a window keeps after asking again because the first was held (`Draw.held_for`): ``retry``
    or ``original``.

    A retry that failed validation (or was cut off) loses. Otherwise the one with fewer signs wins, the original
    on a tie. The retry's signs are its own, except that against an original held for not thinking, a retry not
    known to have thought counts as not thinking either: keeping the original when the retry cannot be judged.
    Elsewhere a missing count is a gap in the books, not a call that skipped thinking."""

    if not retry_passed:
        return "original"
    held = original.held_for()
    signs = retry.held_for()
    if "zero_thinking" in held and "zero_thinking" not in signs and not retry.thinking.fine and not retry.patched:
        signs.append("zero_thinking")
    return "retry" if len(signs) < len(held) else "original"


def redo_record(original: Draw, retry: Draw | None, chosen: str, *, error: str = "") -> Dict[str, Any]:
    """What a record keeps about the one retry of a held answer: why it was held, both draws, which was kept,
    and the error when the retry could not answer."""

    return {"signals": original.held_for(), "original": original.to_metadata(),
            **({"retry": retry.to_metadata()} if retry is not None else {}),
            **({"retry_error": error} if error else {}), "chosen": chosen}
