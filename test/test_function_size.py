"""No function in `src/` may grow past 300 lines, and the ones already past may not grow.

This repository guards file boundaries (`test_import_boundaries.py`), option
defaults (`test_option_defaults.py`), test-suite partitioning
(`test_packaging.py`) and documentation links (`test_doc_links.py`) -- but until
2026-09-03 nothing at all watched **how big a single function is**. So
`run_search_loop` reached 882 lines without any moment where someone had to
decide that was fine.

The rule is deliberately not "big functions are bad". Some of these are one
algorithm chain or one state machine, and cutting them horizontally makes
navigation worse rather than better. `transcribe.py` is the clearest case: its
core is one linear call chain (`align_segments` -> `_align_intervals_group` ->
`align_group` -> `_transcribe_group_candidate`), and slicing it by line number
spreads one chain over several files; the part worth moving out is the argparse
CLI at its tail, which belongs in `speech/recognition/cli/`. The rule is:

- a **new** function may not be written past the limit, and
- an **existing** one may not get bigger than it is today.

`_ALLOWED` is therefore a ratchet, not an exemption list: every entry carries
the function's current size, and the check holds it there both ways -- growing
past it fails, and so does shrinking under it without lowering the entry, since
headroom left in the table is a size that can quietly grow back. An entry that
falls back under the limit has to be deleted. Shrinking one is a normal part of
touching it (and lowering its number is part of that edit); growing one takes a
deliberate edit to this file, which is exactly the moment the decision should
be visible.

Scan surface: every `*.py` under `src/` (both packages), every `def` at any
nesting depth, keyed by qualname.
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

#: `path:qualname` -> its current size. **May only shrink**, and a shrink is
#: recorded here in the same change. Roughly biggest first; the top four are the
#: ones judged worth splitting when the ratchet went in, and the ratchet is what
#: keeps them from drifting further while that work is scheduled.
_ALLOWED: dict[str, int] = {
    # 838 not 815 since 2026-09-03 (`32890987`): the candidate loop grew a
    # second ceiling check -- a single-pool provider can clear
    # `max_input_tokens` and still not leave room for the answer. Legitimate
    # work, landed before this ratchet existed; recorded rather than waived so
    # the next +23 has to be argued for too. It is one of the functions judged
    # worth splitting (vertically: candidate assembly vs one candidate's
    # attempt), and this is the ratchet saying it moved the wrong way.
    # 844 not 838 since 2026-09-04: splitting "how much to request" from "how
    # much to set aside" (`output_reserve`) costs a signature line, two
    # `requested_output_tokens=` lines, the call to `_output_budget` and the
    # two-line note on why an agent records 0. The resolution itself -- and the
    # single-pool clamp -- went into a module function instead, so the +19 this
    # started as landed as +6.
    # 849 not 844 since 2026-09-04: the single-pool clamp had to move out of
    # `_output_budget` and be applied *after* the oversized repair context is
    # dropped -- clamping against the rejected estimate sent a 1-token request
    # for a prompt with the whole window free. Five lines: a two-line note and
    # the three-line call. The helper itself is a module function.
    # 854 not 847 since 2026-11-09: a candidate on a host this chain already
    # failed to connect to is skipped, and a never-connected failure keeps the
    # media ref; the bookkeeping is `_UnconnectedHosts`, these are its calls.
    "finesub/llm/client.py:RoleClient.complete": 854,
    # 703 not 723 since 2026-09-11: packaged dsh / CodeBuddy pair emission
    # moved out of the load loop (`_wire_unclaimed_catalog_target`).
    # 637 since 2026-11-09: the preferred lookups learn which targets are declared.
    "finesub/llm/routing/model_routes.py:load_model_routes": 637,
    "finesub/llm/stages/correction/attempts.py:run_window_attempts": 692,
    "finesub/llm/knowledge/update.py:_run_knowledge_update": 636,
    # 562 not 636 since 2026-10-31: the chain after the decode moved out into
    # `finish_decoded`, which the collapse ladder's window decodes call too.
    "finesub/speech/recognition/vad_asr_stage.py:run_vad_asr": 562,
    # 553 not 603 since 2026-11-07: shrinkage that had gone unrecorded while the check was only `<=`.
    "finesub/llm/stages/correction/run.py:execute_correction_windows": 553,
    # 557 not 555 since 2026-09-04: one line at each of the two
    # `_call_and_parse` call sites, handing down the run's `--output-scale`
    # so the non-correction context reserve scales with it.
    # 479 not 522 since 2026-10-30: the asr bin's queue (bounded put, sentinels, re-key) moved into
    # `_AsrLanes` when the bin got its second lane (cloud ASR), which left the lane choice here.
    "finesub/scheduler.py:run_batch": 479,
    # 499 not 498 since 2026-09-04: one call line. A CLI that may hand the
    # session to a different model mid-run made "who was asked" and "who
    # answered" two different facts, and the attempt has to carry both or the
    # task report files a paid answer under the free model's name. The
    # bookkeeping itself is a method, not inline.
    # 483 not 499 since 2026-09-16: keeping an unaccepted harness session's
    # capsule needed a branch at the removal site. The whole decision -- and the
    # removal it guards -- went out to `_settle_episode`, and the supervisor's
    # inline try/except around the completion predicate became the same
    # `_completion_holds` the new check uses, so two copies of "a failing
    # predicate says no" are one. Tightened to lock the -16 in.
    # 458 not 462 since 2026-11-07: shrinkage that had gone unrecorded while the check was only `<=`.
    "finesub/llm/agent/local_agent.py:LocalAgentDriver._run_episode": 457,
    # 441 not 486 since 2026-09-03: `--no-separate` would have pushed this the
    # other way, so the vocal branch moved out to `_run_vocal_stage` (142) and
    # the caller shrank instead. The ratchet is why that happened rather than a
    # fifty-seventh line being added quietly.
    # 444 not 441 since 2026-09-19: `--diarize-backend`. Three lines and no
    # more -- one parameter, one comment saying the default lives in the stage,
    # and a wrapped call. The stage body did not move here: the backend registry
    # went into `speech/diarization/stage.py`, which is where the backends are
    # declared, so this file only gained the thread that carries the choice.
    # 446 not 444 since 2026-09-20: `--diarize-max-speakers`. Two lines, same
    # shape as the backend thread above -- a parameter and a keyword in the call.
    # The knob itself lives in `diarizen_cluster.cluster`, where the dendrogram
    # is cut (a perfect headcount plus one made 23 of 24 files worse), so this
    # file again only carries the choice down.
    # 447 not 446 since 2026-10-09: the LLM speaker judge, a level of `--diarize`
    # (`judge-text` / `judge-audio`). The keyword in the request and the second
    # export after the LLM stage; the export body and why it runs twice stay in
    # `_run_speaker_export_stage`, the warning when it cannot run in
    # `_warn_if_the_judge_cannot_run`. (The role layer that took two lines here
    # went when the judge replaced it.)
    # 409 not 447 since 2026-10-14: the raw-srt body moved out to `_write_raw_srt` when `--diarize-split`
    # touched it; the parameter and its pass-down to the diarization stage cost two lines here.
    # 410 not 409 since 2026-10-16: `--llm-proofread`'s parameter (its pass-down shares a line).
    # 412 not 410 since 2026-10-17: `--diarize`'s default (`default_diarize` / `task_diarize`): the batch
    # item's default parameter and the one line resolving it.
    # 416 not 412 since 2026-10-25: `--squeeze-repair` / `--gap-fill`, the two parameters and the two-line hook
    # passed to the stable step; resolving the switches and devices went into `_recall_repair`.
    # 418 not 416 since 2026-10-31: `--row-fill`'s parameter and its pass-down to the hook (the call took a third line).
    # 432 not 418 since 2026-10-31: cloud ASR (`--asr`, `--asr-aligner`). The two parameters with their
    # one-line defaults, the three lines deciding the task's backend, and `cloud` handed to the vocal stage,
    # the correction request and stabilization (one profile for it and its recall repair); the aligned stage, the
    # `--diarize` level and the speaker labels of a cloud task went to `_run_joint_aligned`, `_joint_task_diarize`
    # and `_run_joint_labels`.
    # 434 not 432 since 2026-11-02: `--asr-repair`'s parameter and its pass-down to `_run_joint_aligned` (the call
    # took a third line); resolving it is the cloud stage's.
    # 438 not 434 since 2026-11-07: `--empty-cues`' parameter, its comment and the one resolution both writers read (+3),
    # and the VAD-ASR stage imported where it runs, so `pipeline --help` no longer imports torch (+1).
    "finesub/stages.py:run_pipeline": 438,
    # 429 not 414 since 2026-09-03: `--separate/--no-separate`. This function is
    # a flat list of `add_argument` calls, so every new user-facing option costs
    # it a block and no split makes it smaller -- splitting by topic would just
    # move the same lines behind three more names. Recorded, not waived: the
    # number still has to move by hand, which is the point.
    # 439 not 429 since 2026-09-05: `--diarize`. Same reasoning, and the caller
    # it feeds went the other way -- `run_pipeline` absorbed the diarization
    # stage without growing, by moving the stage body and the planned-stage list
    # into `_run_diarization_stage` / `_planned_stages`.
    # 449 not 439 since 2026-09-06: `--speaker-tracks`. `run_pipeline` again
    # took its half without growing (`_run_speaker_export_stage`); this file is
    # where the option's cost is visible, which is the arrangement working.
    # 456 not 449 since 2026-09-20: `--diarize-max-speakers`. Seven lines, six
    # of them the help text -- which carries the measurement that says not to use
    # it (a perfect headcount plus one made 23 of 24 test files worse), because a
    # `--help` a user reads is the only place that warning reaches them.
    # 457 not 456 later the same day: `--diarize-backend`'s help said sortformer
    # was the default, which stopped being true when the default became a probe.
    # One line, and it buys the sentence that tells a user what actually happens.
    # 451 not 457 since 2026-10-09: `--diarize` gained the LLM speaker judge's two
    # levels (+6 lines of help saying what they write and that the single-track
    # SRT stays), and the role layer's `--speaker-roles` went with it (-11), less
    # what had shrunk since.
    # 462 not 451 since 2026-10-14: `--diarize-split`. Eleven lines, eight of them the help text saying
    # what is cut, what reads the split lines and that a task keeps its first run's choice.
    # 471 not 462 since 2026-10-16: `--llm-proofread`. Nine lines, four of them the help text saying when
    # `auto` runs it.
    # 472 not 471 since 2026-10-25: the call adding `--squeeze-repair` / `--gap-fill` (`_add_recall_repair_options`).
    # 495 not 472 since 2026-10-31: `--asr` and `--asr-aligner`, the help text naming the three backends,
    # that the audio leaves the machine and that a task keeps the backend its products record.
    # 506 not 495 since 2026-11-02: `--asr-repair`. Eleven lines, five of them the help text saying what each
    # backend's repair spends.
    # 514 not 506 since 2026-11-07: `--empty-cues` (owner's decision on the audit's D-2).
    # 509 not 514 since 2026-11-07: `--llm-correct-translate` deleted (audit Q-17).
    "finesub/pipeline.py:build_parser": 509,
    # 395 not 403 since 2026-09-16: naming every failed call and keeping calls
    # that bill nobody out of the provider table would have made this +21. The
    # booking and the bullet went out to `_book_provider_call` and
    # `_failed_call_line` instead, taking the two older inline blocks they
    # replaced with them, so it landed at -8. Tightened rather than left at 403,
    # which would quietly re-permit those eight lines.
    "finesub/llm/task_report.py:render_task_report": 357,
    # 324 not 385 since 2026-11-07: shrinkage that had gone unrecorded while the check was only `<=`.
    "finesub/llm/output_protocol.py:validate_translated_csv_text": 324,
    # 375 not 377 since 2026-11-07: each block trims by the padding it read, so the merge no longer takes the pad.
    "finesub/speech/preprocessing/separator/separation.py:run_vocal_separation": 375,
    # 365 not 385 since 2026-09-10: the soft/hard API timeout would have pushed
    # this past 385, so the per-attempt transport dispatch moved out to
    # `_generate_once` (now `_generate_hedged`) and the loop shrank instead. (The 2026-09-03 note here
    # said there was no self-contained block to lift -- that one was.)
    # 363 since 2026-10-23: streaming gave each attempt a `StreamProgress`,
    # whose clock replaced the attempt's own start bookkeeping. 349 since the
    # same day: the backup request records each request as it ends, so the
    # loop's two inline attempt records went to `_record_api_attempt`, and the
    # backup's rate-limit booking is `_reserve_backup`.
    # 342 not 349 since 2026-11-07: shrinkage that had gone unrecorded while the check was only `<=`.
    # 351 since 2026-11-09: a paused Anthropic search turn settles the combo
    # cooldown as an answer (`_note_combo_answered`) before moving on.
    # 359 not 351 since 2026-11-09: a judged-unreachable host gets one attempt
    # (`_HostAttempts`, these are its two calls) and a connect failure neither
    # sets sticky exhaustion nor starts a combo cooldown.
    "finesub/llm/llm_runtime.py:chat_complete": 359,
    # 327 not 346 since 2026-11-07: shrinkage that had gone unrecorded while the check was only `<=`.
    # 316 since 2026-11-08: its replay check is the ledger's (`ResumeLedger.replay`), shared with serial.
    "finesub/llm/stages/correction/parallel.py:run_parallel_windows": 316,
    # 341 not 338 since 2026-09-04: the CLI path was passing no `limits=` to
    # `decide_fast_mode`, so it planned the fused window against
    # DEFAULT_LIMITS whatever group was bound. Three lines to fix a wrong
    # number, taken deliberately rather than by shrinking the fix.
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
    """A listed function may shrink, never grow -- and a shrink is locked in.

    Only failing on growth would leave every line a refactor took off as
    headroom that the next edit can quietly spend, which is the drift this
    file exists to stop. So a recorded size above the measured one fails too,
    asking for the entry to come down to the new size.
    """

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
    shrunk = sorted(
        (allowed - sizes[key], key)
        for key, allowed in _ALLOWED.items()
        if key in sizes and sizes[key] < allowed
    )
    assert shrunk == [], (
        "these shrank -- lower their entries in _ALLOWED to the new size so the "
        "freed lines cannot grow back: "
        + ", ".join(
            f"{key} -> {sizes[key]} (recorded {_ALLOWED[key]})"
            for _, key in reversed(shrunk)
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
