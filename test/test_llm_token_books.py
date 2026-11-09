from __future__ import annotations

import re

from finesub.llm.client import extract_token_distribution
from finesub.llm.task_report import render_task_report


def _route(*targets: tuple[str, str, str], answered: str = "") -> dict:
    return {
        "effective_chain": [
            {"target_id": target_id, "provider_tier": tier, "model": model} for target_id, tier, model in targets
        ],
        "candidates": [{"target_id": answered, "outcome": "success"}] if answered else [],
    }


AGY = ("local-agy-media-gemini-3_8-flash", "LOCAL_AGY", "gemini-3.8-flash")
CODEX = ("local-codex-completion-gpt-6-luna", "LOCAL_CODEX", "gpt-6-luna")
CLAUDE = ("local-claude-completion-haiku-4_5", "LOCAL_CLAUDE", "haiku")


def _agy_attempt(input_tokens: int, cached: int, output: int, thinking: int) -> dict:
    return {
        "backend": "local_agent", "target_id": AGY[0], "thinking_reported": True,
        "usage": {"input_tokens": input_tokens, "cached_input_tokens": cached, "output_tokens": output,
                  "reasoning_output_tokens": thinking, "source": "agy_step_events"},
    }


def _codex_attempt(input_tokens: int, cached: int, output: int) -> dict:
    return {
        "backend": "local_agent", "target_id": CODEX[0], "thinking_reported": True,
        "usage": {"input_tokens": input_tokens, "cached_input_tokens": cached, "output_tokens": output},
    }


def _response(kind: str, attempts: list, *, route: dict, **payload) -> dict:
    from finesub.llm.client import agent_usage_payload

    usage = extract_token_distribution({"usage": agent_usage_payload(attempts[-1]["usage"])})
    return {"kind": kind, "payload": {"usage": usage, "execution_attempts": attempts, "route_decision": route,
                                      **payload}}


def _records() -> list[dict]:
    both = _route(AGY, CODEX, answered=CODEX[0])
    return [
        _response("prepare_task_request", [_codex_attempt(1000, 400, 50)], route=_route(CODEX, answered=CODEX[0])),
        _response("speaker_judge_response", [_agy_attempt(500, 0, 300, 200)], route=_route(AGY, answered=AGY[0]),
                  call="speaker_judge-0001-attempt0"),
        # agy spent and gave up; Codex answered.
        _response("correction_window_response", [_agy_attempt(2000, 100, 900, 600), _codex_attempt(3000, 1000, 120)],
                  route=both, chunk_id="0001", attempt=0),
        {"kind": "correction_window_call_error", "payload": {
            "chunk_id": "0002", "attempt": 0, "error_type": "LocalAgentPolicyViolationError", "error": "x",
            "route_decision": _route(AGY), "execution_attempts": [_agy_attempt(700, 0, 80, 40)]}},
        _response("correction_window_response", [_codex_attempt(2500, 2000, 100)], route=_route(CODEX, answered=CODEX[0]),
                  chunk_id="0002", attempt=1),
        _response("proofread_response", [_agy_attempt(400, 0, 60, 30)], route=_route(AGY, answered=AGY[0]),
                  call="proofread-0001-attempt0"),
        _response("knowledge_proposal_repair_response", [_codex_attempt(800, 0, 40)],
                  route=_route(CODEX, answered=CODEX[0]), call="knowledge-proposal-repair-attempt0"),
    ]


def _section(text: str, title: str) -> str:
    return text.split(f"## {title}", 1)[1].split("\n## ", 1)[0]


def test_the_three_books_add_up_to_one_task_total() -> None:
    """Each book used to see a different part of the task: the phase lines only the
    correction windows and the prepare request, the session and provider tables
    every session kind but neither the speaker judge nor pass 3, and none of them
    the attempts that spent without answering."""

    text = render_task_report(_records(), task_id="t")
    # agy's input leaves its cache read out; the books add it back (2000 + 100).
    spent_input = 1000 + 500 + 2100 + 3000 + 700 + 2500 + 400 + 800

    phase_total = re.search(r"- task total \((.*?)\): .*total_input_tokens=(\d+)", _section(text, "LLM Token Usage"))
    assert phase_total and int(phase_total.group(2)) == spent_input
    assert phase_total.group(1) == "6 calls, 2 unanswered attempts"
    for phase in ("prepare", "speaker judge", "correction", "proofread", "knowledge update"):
        assert f"- {phase} (" in text
    session_total = re.search(r"\| \*\*task total\*\* \| (\d+) \|", _section(text, "Session Token Totals"))
    assert session_total and int(session_total.group(1)) == spent_input
    provider_total = re.search(r"\| \*\*task total\*\* \| \| 6 \| 2 \| (\d+) \|", _section(text, "Provider Token Totals"))
    assert provider_total and int(provider_total.group(1)) == spent_input


def test_an_attempt_that_did_not_answer_is_billed_to_its_own_target() -> None:
    sessions = _section(render_task_report(_records(), task_id="t"), "Session Token Totals")
    providers = _section(render_task_report(_records(), task_id="t"), "Provider Token Totals")

    # The answer and the attempt before it are one session; the failed call is a row of its own.
    assert "| correction-0001-attempt0 | 5100 | 1020 | 600 | 420 | 1 |" in sessions
    assert "| correction-0002-attempt0 (failed) | 700 | 80 | 40 | 40 | 1 |" in sessions
    assert "| speaker_judge-0001-attempt0 | 500 | 300 | 200 | 100 | 0 |" in sessions
    # agy answered the judge and pass 3 and spent twice without answering.
    assert "| LOCAL_AGY | gemini-3.8-flash | 2 | 2 | 3700 | 100 | 0 | 470 | 870 |" in providers
    assert "| LOCAL_CODEX | gpt-6-luna | 4 | 0 | 7300 | 3400 | 0 | 310 | 0 |" in providers


def test_a_cli_that_does_not_count_thinking_reads_n_a_not_zero() -> None:
    attempt = {"backend": "local_agent", "target_id": CLAUDE[0], "thinking_reported": False,
               "usage": {"input_tokens": 6, "cache_creation_input_tokens": 28122, "cache_read_input_tokens": 33484,
                         "output_tokens": 38, "source": "claude_code_result_event"}}
    record = _response("correction_window_response", [attempt], route=_route(CLAUDE, answered=CLAUDE[0]))

    providers = _section(render_task_report([record], task_id="t"), "Provider Token Totals")

    # Claude Code's cache write is its own column, not part of uncached input.
    assert "| LOCAL_CLAUDE | haiku | 1 | 0 | 61612 | 33484 | 28122 | 38 | n/a |" in providers


def test_a_failed_judge_call_is_its_own_failed_session_with_its_attempts() -> None:
    error = {"kind": "speaker_judge_call_error", "payload": {
        "call": "speaker_judge-0002-attempt0", "error_type": "LocalAgentTransientError", "error": "exit 1",
        "route_decision": _route(AGY), "execution_attempts": [_agy_attempt(900, 100, 50, 20)]}}

    text = render_task_report([error], task_id="t")

    assert "| speaker_judge-0002-attempt0 (failed) | 1000 | 50 | 20 | 30 | 1 |" in text
    assert "- speaker judge (0 calls, 1 unanswered attempts)" in text
    # The failure list names the call, the only handle such a record has.
    assert "call=speaker_judge-0002-attempt0" in text


def test_a_failed_prepare_delivery_names_its_attempt() -> None:
    error = {"kind": "prepare_task_call_error", "payload": {
        "attempt": 1, "error_type": "X", "error": "x", "route_decision": _route(CODEX),
        "execution_attempts": [_codex_attempt(500, 0, 10)]}}

    assert "| prepare-delivery-attempt1 (failed) | 500 |" in render_task_report([error], task_id="t")


def test_a_pseudo_session_total_keeps_its_vendors_units() -> None:
    """The summed session usage dropped `source`, so agy's input -- which leaves
    the cache read out -- was read as the whole prompt and uncached as 0."""

    from types import SimpleNamespace

    from finesub.llm.agent.agent_session_host import AgentSessionHost
    from finesub.llm.client import agent_usage_payload

    host = SimpleNamespace(session_usage=[
        {"usage": {"input_tokens": 4000, "cached_input_tokens": 6000, "output_tokens": 50,
                   "source": "agy_result_event"}},
        {"usage": {"input_tokens": 5000, "cached_input_tokens": 10000, "output_tokens": 70,
                   "source": "agy_step_events"}},
    ])
    totals = AgentSessionHost.usage_totals(host)
    dist = extract_token_distribution({"usage": agent_usage_payload(totals)})

    assert (dist["total_input_tokens"], dist["uncached_input_tokens"], dist["cached_input_tokens"]) == (
        25000, 9000, 16000)
