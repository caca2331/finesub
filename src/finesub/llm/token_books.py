"""The token books of a task report, read from one booking of every call.

Every LLM call a task's artifacts record is booked once: its phase, its session,
who answered and what it spent. So are the tokens spent without an answer -- a
fallback's earlier target, a conversation rebuilt on the way, a call that failed
outright -- and a pseudo-conversational run's sessions, which are metered per CLI
session rather than per call. The phase, session and provider tables are three
groupings of that one list, so they add up to the same task total.

The writers' half lives here too (`call_trace`, `append_call_error`): what a call
record must carry for the books to read it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .exchange_metadata import infer_session_name, is_session_response_record, normalize_session_usage
from .knowledge.base import append_task_artifact

TOKEN_KEYS = (
    "uncached_input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "total_input_tokens",
    "thinking_tokens",
    "output_tokens",
    "total_output_tokens",
)

#: Response records of the calls outside the session kinds the round summary
#: reads (`exchange_metadata.SESSION_RESPONSE_KINDS`); each spends tokens all the same.
_STAGE_RESPONSE_KINDS = frozenset(
    {"speaker_judge_response", "proofread_response", "knowledge_proposal_repair_response"}
)

#: Record-kind prefix -> phase, in the order a task runs them.
_PHASES = (
    ("prepare_task", "prepare"),
    ("speaker_judge", "speaker judge"),
    ("correction_", "correction"),
    ("proofread", "proofread"),
    ("knowledge_", "knowledge update"),
)
AGENT_SESSIONS_PHASE = "agent sessions"

# A call error's session is named after the response its call would have written.
_RESPONSE_OF_ERROR = {
    "correction_window": "correction_window_response",
    "correction_query": "correction_query_response",
    "knowledge_update": "knowledge_update_response",
    "prepare_task": "prepare_task_response",  # the delivery turn's (`research`)
}

_CALL, _UNANSWERED, _SESSION = "call", "unanswered", "session"


@dataclass(frozen=True)
class Booking:
    phase: str
    session: str
    target: tuple[str, str]
    tokens: Mapping[str, int]
    #: ``call`` (one answered call), ``unanswered`` (an attempt that spent and did not
    #: answer) or ``session`` (a pseudo-conversational session's own meter).
    role: str
    #: False when the CLI does not count thinking apart from its answer.
    thinking_reported: bool = True


def call_trace(result: Any) -> dict[str, Any]:
    """What a call record carries beside its usage for the books: who answered, and
    the attempts that spent on the way (`LLMCallResult`)."""

    return {
        "route_decision": dict(getattr(result, "route_decision", None) or {}),
        "execution_attempts": list(getattr(result, "execution_attempts", None) or []),
    }


def append_call_error(
    artifact_dir: str | Path | None, kind: str, task_id: str, call: str, error: BaseException
) -> None:
    """Record a call that raised, with the attempts it spent before it did."""

    if artifact_dir is None:
        return
    append_task_artifact(artifact_dir, kind=kind, task_id=task_id, payload={
        "call": call,
        "error_type": type(error).__name__,
        "error": str(error),
        "execution_attempts": list(getattr(error, "_harness_execution_attempts", None) or []),
        "route_decision": dict(getattr(error, "_harness_route_decision", None) or {}),
    })


def book(
    records: Iterable[Mapping[str, Any]], agent_sessions: Sequence[Mapping[str, Any]] | None = None
) -> list[Booking]:
    bookings: list[Booking] = []
    for record in records:
        kind = str(record.get("kind") or "")
        payload = record.get("payload") or {}
        if not isinstance(payload, Mapping):
            continue
        failed = kind.endswith("_call_error")
        if not failed and not (is_session_response_record(kind, payload) or kind in _STAGE_RESPONSE_KINDS):
            continue
        phase = _phase_of(kind)
        if not phase:
            continue
        session = _session_of(kind, payload, failed)
        attempts = [item for item in payload.get("execution_attempts") or () if isinstance(item, Mapping)]
        if not failed:
            usage = payload.get("usage")
            answering = attempts[-1] if attempts else {}
            bookings.append(Booking(
                phase, session, answering_target(payload), _counts(usage), _CALL,
                bool(answering.get("thinking_reported", True)),
            ))
        # The answering attempt is the last one (`client.complete`); a failed call has none.
        for attempt in attempts if failed else attempts[:-1]:
            tokens = _agent_attempt_tokens(attempt)
            if tokens is None:
                continue
            target = _target_of(payload.get("route_decision"), str(attempt.get("target_id") or ""),
                                str(attempt.get("reported_model") or ""))
            bookings.append(Booking(phase, session, target, tokens, _UNANSWERED,
                                    bool(attempt.get("thinking_reported", True))))
    for item in agent_sessions or ():
        if not isinstance(item, Mapping) or not isinstance(item.get("usage"), Mapping):
            continue
        # Keyed exactly as the per-call rows are (`answering_target` passes the
        # tier through as recorded): case-folding here split one agent into two
        # rows, one with the calls and one with the tokens.
        target = (str(item.get("provider_tier") or "unknown"), str(item.get("model") or "unknown"))
        label = str(item.get("label") or f"{target[0]}/{target[1]}")
        bookings.append(Booking(AGENT_SESSIONS_PHASE, f"agent session {label}", target,
                                _counts(item["usage"]), _SESSION))
    return bookings


def phase_lines(bookings: Sequence[Booking]) -> list[str]:
    if not bookings:
        return ["- No retained LLM usage."]
    order = [name for _prefix, name in _PHASES] + [AGENT_SESSIONS_PHASE]
    lines = []
    for phase in sorted({item.phase for item in bookings}, key=order.index):
        lines.append(_bullet(_totals_line(phase, [item for item in bookings if item.phase == phase])))
    lines.append(_bullet(_totals_line("task total", bookings)))
    return lines


def session_lines(bookings: Sequence[Booking]) -> list[str]:
    if not bookings:
        return ["- No retained per-session LLM usage."]
    rows: dict[str, Counter[str]] = {}
    for item in bookings:
        row = rows.setdefault(item.session, Counter())
        row.update(normalize_session_usage(item.tokens))
        row["unanswered"] += item.role == _UNANSWERED
    keys = ("total_input_tokens", "total_output_tokens", "thinking_tokens", "output_tokens", "unanswered")
    lines = [
        "| Session | Total input | Total output | Thinking | Visible output | Unanswered attempts |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    totals: Counter[str] = Counter()
    for name, row in rows.items():
        lines.append(f"| {name} | " + " | ".join(str(row[key]) for key in keys) + " |")
        totals.update(row)
    lines.append("| **task total** | " + " | ".join(str(totals[key]) for key in keys) + " |")
    lines.append("")
    lines.append(_bullet(
        "Each row is one LLM session: its answer plus any attempt that spent tokens without one "
        "(a fallback's earlier target, a rebuilt conversation, a call that failed). A "
        "pseudo-conversational agent session is metered as a whole, so it is a row of its own "
        "and its windows read zero. Input/output come from provider usage metadata, not from "
        "summing component estimates in the exchange header. Thinking reads 0 for a CLI that "
        "does not count it apart (`n/a` in the provider table): its Visible output holds it."
    ))
    return lines


def provider_lines(bookings: Sequence[Booking]) -> list[str]:
    """Token cost grouped the way it is charged.

    The session table answers "which round spent this"; this one answers "who
    is billing for it". One session can fall back across tiers, and one tier
    serves many sessions.

    A call that names nobody and spent nothing is no bill line: it only ever
    rendered as `unknown / unknown` with zeros, which reads as a provider the
    run could not identify rather than a call that failed before any provider
    reported usage. An *identified* provider with no usage is still booked:
    a real call on a real tier whose CLI never said what it cost.
    """

    rows: dict[tuple[str, str], Counter[str]] = {}
    unreported: dict[tuple[str, str], bool] = {}
    for item in bookings:
        if item.target == ("unknown", "unknown") and not any(item.tokens.get(key) for key in TOKEN_KEYS):
            continue
        row = rows.setdefault(item.target, Counter())
        row["calls"] += item.role == _CALL
        row["unanswered"] += item.role == _UNANSWERED
        row.update({key: int(item.tokens.get(key) or 0) for key in TOKEN_KEYS})
        if item.role != _SESSION:
            unreported[item.target] = unreported.get(item.target, True) and not item.thinking_reported
    if not rows:
        return ["- No retained per-provider LLM usage."]
    lines = [
        "| Provider tier | Model | Calls | Unanswered attempts | Input (total) | of which cached | "
        "of which cache write | Visible output | Thinking |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    totals: Counter[str] = Counter()
    for (tier, model), counts in sorted(rows.items()):
        thinking = "n/a" if unreported.get((tier, model)) else str(counts["thinking_tokens"])
        lines.append(
            f"| {tier} | {model} | {counts['calls']} | {counts['unanswered']} | "
            f"{counts['total_input_tokens']} | {counts['cached_input_tokens']} | "
            f"{counts['cache_write_input_tokens']} | {counts['output_tokens']} | {thinking} |"
        )
        totals.update(counts)
    lines.append(
        f"| **task total** | | {totals['calls']} | {totals['unanswered']} | "
        f"{totals['total_input_tokens']} | {totals['cached_input_tokens']} | "
        f"{totals['cache_write_input_tokens']} | {totals['output_tokens']} | {totals['thinking_tokens']} |"
    )
    lines.append("")
    lines.append(_bullet(
        "Input is the full prompt side with the cache reads and cache writes called out, because "
        "the three are priced differently (a write costs more than plain input on some providers) "
        "-- and because a provider that reports only a total then reads as a total rather than as "
        "zero. The token columns include the unanswered attempts. Thinking `n/a`: that CLI does not "
        "count thinking apart from its answer, so Visible output holds it. Local-agent tiers are "
        "metered by subscription, not per token; their rows are for cache and context accounting."
    ))
    return lines


def answering_target(payload: Mapping[str, Any]) -> tuple[str, str]:
    """Which (provider tier, model) actually answered this call.

    A bill is keyed on the pair, not on the model: the same model on the free
    and the paid Gemini tier costs differently, and a local-agent tier costs
    nothing per token at all. `model` alone cannot say which, and only the route
    decision records which candidate answered -- so the pair is joined there.
    """

    route = payload.get("route_decision")
    answered = ""
    for candidate in (route.get("candidates") or ()) if isinstance(route, Mapping) else ():
        if isinstance(candidate, Mapping) and candidate.get("outcome") == "success":
            answered = str(candidate.get("target_id") or "")
            break
    return _target_of(route, answered, str(payload.get("model") or ""))


def _target_of(route: Any, target_id: str, model: str) -> tuple[str, str]:
    """The (provider tier, model) of ``target_id`` in the route's chain.

    No target id (an older artifact): the model name is the only handle left,
    and it is unambiguous unless one chain lists the same model on two tiers.
    """

    key, wanted = ("target_id", target_id) if target_id else ("model", model)
    for entry in (route.get("effective_chain") or ()) if isinstance(route, Mapping) else ():
        if not isinstance(entry, Mapping) or str(entry.get(key) or "") != wanted:
            continue
        return (str(entry.get("provider_tier") or "unknown"), str(entry.get("model") or model or "unknown"))
    return ("unknown", model or "unknown")


def _agent_attempt_tokens(attempt: Mapping[str, Any]) -> dict[str, int] | None:
    """An agent attempt's own usage, in the books' units; None when it reported none."""

    from .client import agent_usage_payload, extract_token_distribution

    usage = attempt.get("usage")
    if attempt.get("backend") != "local_agent" or not isinstance(usage, Mapping):
        return None
    if usage.get("source") in ("unavailable", "session"):
        return None
    tokens = extract_token_distribution({"usage": agent_usage_payload(usage)})
    return tokens if any(tokens[key] for key in TOKEN_KEYS) else None


def _counts(usage: Any) -> dict[str, int]:
    usage = usage if isinstance(usage, Mapping) else {}
    counts = {key: int(usage.get(key) or 0) for key in TOKEN_KEYS}
    # An older or a coarse payload may give only the totals.
    counts["total_input_tokens"] = counts["total_input_tokens"] or int(usage.get("prompt_tokens") or 0)
    counts["total_output_tokens"] = counts["total_output_tokens"] or counts["output_tokens"] + counts["thinking_tokens"]
    return counts


def _phase_of(kind: str) -> str:
    return next((name for prefix, name in _PHASES if kind.startswith(prefix)), "")


def _session_of(kind: str, payload: Mapping[str, Any], failed: bool) -> str:
    if payload.get("call"):
        name = str(payload["call"])
    else:
        base = kind[: -len("_call_error")] if failed else kind
        name = infer_session_name(_RESPONSE_OF_ERROR.get(base, base), payload)
    return f"{name} (failed)" if failed else name


def _totals_line(label: str, bookings: Sequence[Booking]) -> str:
    totals: Counter[str] = Counter()
    for item in bookings:
        totals.update({key: int(item.tokens.get(key) or 0) for key in TOKEN_KEYS})
    roles = Counter(item.role for item in bookings)
    counted = [f"{roles[_CALL]} calls"]
    if roles[_UNANSWERED]:
        counted.append(f"{roles[_UNANSWERED]} unanswered attempts")
    if roles[_SESSION]:
        counted.append(f"{roles[_SESSION]} agent sessions")
    return f"{label} ({', '.join(counted)}): " + ", ".join(f"{key}={totals[key]}" for key in TOKEN_KEYS)


def _bullet(text: str) -> str:
    return f"- {text}"
