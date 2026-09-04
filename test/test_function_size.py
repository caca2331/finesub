"""No function in `src/` may grow past 300 lines, and the ones already past may not grow.

This repository guards file boundaries (`test_import_boundaries.py`), option
defaults (`test_option_defaults.py`), test-suite partitioning
(`test_packaging.py`) and documentation links (`test_doc_links.py`) -- but until
2026-09-03 nothing at all watched **how big a single function is**. So
`run_search_loop` reached 882 lines without any moment where someone had to
decide that was fine.

The rule is deliberately not "big functions are bad". Some of these are one
algorithm chain or one state machine, and cutting them horizontally makes
navigation worse rather than better (`docs/report/2026-09-03-structure-clarity-
review.md` §5.2 argues that case for `transcribe.py`). The rule is:

- a **new** function may not be written past the limit, and
- an **existing** one may not get bigger than it is today.

`_ALLOWED` is therefore a ratchet, not an exemption list: every entry carries
the size it had when it was measured, the assertion is `<=`, and an entry that
falls back under the limit has to be deleted. Shrinking one is a normal part of
touching it; growing one takes a deliberate edit to this file, which is exactly
the moment the decision should be visible.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"

#: Lines, counted `def` line through the last line of the body -- the number a
#: reader actually scrolls. 300 is a ceiling for *new* code, not a target: the
#: median function here is under 20 lines, and everything in `_ALLOWED` predates
#: the rule.
MAX_FUNCTION_LINES = 300

#: `path:qualname` -> its size on 2026-09-03. **May only shrink.** Twenty
#: entries, ordered biggest first; the top four are the ones the structure
#: review nominated for splitting, and the ratchet is what keeps them from
#: drifting further while that work is scheduled.
_ALLOWED: dict[str, int] = {
    "finesub/llm/search_loop.py:run_search_loop": 882,
    # 838 not 815 since 2026-09-03 (`32890987`): the candidate loop grew a
    # second ceiling check -- a single-pool provider can clear
    # `max_input_tokens` and still not leave room for the answer. Legitimate
    # work, landed before this ratchet existed; recorded rather than waived so
    # the next +23 has to be argued for too. It is #22 on the structure
    # review's split list, and this is the ratchet saying it moved the wrong
    # way.
    "finesub/llm/client.py:RoleClient.complete": 838,
    "finesub/llm/routing/model_routes.py:load_model_routes": 723,
    "finesub/llm/stages/correction/attempts.py:run_window_attempts": 718,
    "finesub/llm/knowledge/update.py:_run_knowledge_update": 655,
    "finesub/speech/recognition/vad_asr_stage.py:run_vad_asr": 636,
    "finesub/llm/stages/correction/run.py:execute_correction_windows": 603,
    "finesub/llm/research.py:run_research": 555,
    "finesub/scheduler.py:run_batch": 522,
    "finesub/llm/agent/local_agent.py:LocalAgentDriver._run_episode": 498,
    # 441 not 486 since 2026-09-03: `--no-separate` would have pushed this the
    # other way, so the vocal branch moved out to `_run_vocal_stage` (142) and
    # the caller shrank instead. The ratchet is why that happened rather than a
    # fifty-seventh line being added quietly.
    "finesub/stages.py:run_pipeline": 441,
    "finesub/llm/stages/fast_session.py:run_fast_session": 469,
    # 429 not 414 since 2026-09-03: `--separate/--no-separate`. This function is
    # a flat list of `add_argument` calls, so every new user-facing option costs
    # it a block and no split makes it smaller -- splitting by topic would just
    # move the same lines behind three more names. Recorded, not waived: the
    # number still has to move by hand, which is the point.
    "finesub/pipeline.py:build_parser": 429,
    "finesub/llm/stages/correction/query_round.py:run_window_query_round": 404,
    "finesub/llm/task_report.py:render_task_report": 403,
    "finesub/llm/output_protocol.py:validate_translated_csv_text": 385,
    # 385 not 384 since 2026-09-03: the failure site now hands
    # `_record_api_attempt` the endpoint's own message, so the run log can say
    # *why* a call came back 429 rather than only that it did. One line, and
    # unlike the two functions above it there is no self-contained block to
    # lift out -- this is one candidate/key/retry loop all the way down.
    "finesub/llm/llm_runtime.py:chat_complete": 385,
    "finesub/speech/preprocessing/separator/separation.py:run_vocal_separation": 377,
    "finesub/llm/stages/correction/parallel.py:run_parallel_windows": 346,
    "finesub/llm/correction_translation.py:_main_impl": 338,
}


def _functions() -> dict[str, int]:
    """Every function in `src/`, keyed `path:qualname`, valued in lines.

    Qualnames rather than line numbers: a line number changes on every edit
    above it, so a ratchet keyed by position would need rewriting constantly and
    would say nothing when it did. Nested functions get their own entry *and*
    count inside their parent -- deliberate, since the parent is what a reader
    scrolls through.
    """

    sizes: dict[str, int] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        relative = path.relative_to(SOURCE_ROOT).as_posix()
        stack: list[tuple[ast.AST, str]] = [(tree, "")]
        while stack:
            node, prefix = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{prefix}{child.name}"
                    sizes[f"{relative}:{name}"] = child.end_lineno - child.lineno + 1
                    stack.append((child, f"{name}."))
                elif isinstance(child, ast.ClassDef):
                    stack.append((child, f"{prefix}{child.name}."))
    return sizes


def test_no_new_function_is_written_past_the_limit() -> None:
    sizes = _functions()
    offenders = sorted(
        (size, key)
        for key, size in sizes.items()
        if size > MAX_FUNCTION_LINES and key not in _ALLOWED
    )
    assert offenders == [], (
        f"these exceed {MAX_FUNCTION_LINES} lines and are not in the ratchet -- "
        "split them, or add them with their size and say why in the commit: "
        + ", ".join(f"{key} ({size})" for size, key in reversed(offenders))
    )


def test_the_ratchet_only_turns_one_way() -> None:
    """A listed function may shrink, never grow."""

    sizes = _functions()
    grown = sorted(
        (sizes[key] - allowed, key)
        for key, allowed in _ALLOWED.items()
        if key in sizes and sizes[key] > allowed
    )
    assert grown == [], (
        "these grew past their recorded size; the ratchet only turns one way: "
        + ", ".join(
            f"{key} +{delta} (now {sizes[key]}, was {_ALLOWED[key]})"
            for delta, key in reversed(grown)
        )
    )


def test_the_ratchet_carries_no_dead_or_redundant_entries() -> None:
    """Entries have to go when the function does -- or when it drops under the limit.

    Both halves matter. A renamed or deleted function leaves an entry that
    silently exempts nothing, and a function that has been cut back under the
    limit no longer needs an exemption -- keeping it would quietly re-permit
    growing back to the old size, which is the one thing this file exists to
    prevent.
    """

    sizes = _functions()
    dead = sorted(key for key in _ALLOWED if key not in sizes)
    assert dead == [], (
        "these no longer exist (renamed or deleted?); remove them from _ALLOWED: "
        + ", ".join(dead)
    )
    redundant = sorted(
        key
        for key, allowed in _ALLOWED.items()
        if sizes[key] <= MAX_FUNCTION_LINES
    )
    assert redundant == [], (
        f"these are now at or under {MAX_FUNCTION_LINES} lines -- delete their "
        "entries so the ratchet cannot be turned back: " + ", ".join(redundant)
    )
