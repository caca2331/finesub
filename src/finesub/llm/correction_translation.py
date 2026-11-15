"""One LLM correction task: ``run_correction_task``, and the module CLI over it.

Every front end -- this CLI, ``run_pipeline``'s LLM stage, reference ingest --
builds a ``CorrectionRequest`` and hands it to ``run_correction_task``, which is
the only place the task is orchestrated. The window-execution loop lives in
``finesub.llm.stages.correction``; research acquisition in
``finesub.llm.research``; the unified knowledge update in
``finesub.llm.knowledge.update``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, ContextManager, Dict, Mapping, Sequence

from finesub import jsonl
from finesub.run_metadata import (
    metadata_path_for_output,
    stage_record,
    summarize_llm_rounds,
    update_run_metadata,
)

from finesub.media.clips import probe_audio_duration
from finesub.reporting import current_reporter, reporting_to, terminal_reporter
from finesub_bootstrap.artifacts import ARTIFACT_DIR_SUFFIX, RESOLVED_SWITCHES_FILENAME
from .agent.agent_session_host import agent_session_scope, set_run_evidence_destination
from .client import write_agent_session_usage
from .correction_request import (
    KNOWLEDGE_MODES,
    CorrectionRequest,
    CorrectionRequestError,
    fast_mode_for_research_only,
    resolve_switches,
    resolved_request,
)
from .knowledge.node.repo import CATEGORY_SEPARATOR
from .knowledge.style import (
    DEFAULT_STYLE_NAME,
    STYLE_CATEGORY,
    StyleCatalog,
    StyleChoice,
    StyleSelection,
    CORRECT_ONLY_TASK,
    StyleSelectionError,
    for_correct_only,
    load_style_catalog,
    load_style_entries,
    parse_style_names,
    record_style_injected,
    resolve_style_keys,
    render_style_block,
    resolve_style_selection,
)
from .knowledge.base import (
    DEFAULT_KNOWLEDGE_ROOT,
    TASK_ARTIFACT_FILENAME,
    append_task_artifact,
    pinned_generation_rev,
)
from .knowledge.update import (
    ensure_research_context_path,
    run_knowledge_update,
)
from .routing.profiles import (
    CONTINUITY,
    DIFFICULTY,
    MEDIA,
    RETRIEVAL,
    TranslationProfile,
    resolve_profile,
)
from .routing.capabilities import (
    CapabilityUnavailableError,
    correction_planning_limits,
    validate_profile_capabilities,
)
from .prompts import ContextPack
from .prompt_constants import Thresholds
from .length_tier import task_length_scale
from .prompt_artifacts import build_prompt_artifacts, write_prompt_artifacts
from .research import (
    backup_unrecoverable_json,
    load_research_context,
    planning_metadata,
    research_reuse_key,
    run_research_stage,
)
from finesub.subtitles.postprocess import (
    SUPPORTED_POSTPROCESS_PROFILES,
    postprocess_srt_file,
)
from .stages.correction import (
    execute_correction_windows,
    refresh_annotated_speakers,
    resolve_correction_plan,
)
from .stages.plan import FastDecision, decide_fast_mode, fast_context_pack
from .stages import three_pass
from .stages.task_graph import (
    TaskGraph,
    grant_for_run,
    plan_task_graph,
    resolve_execution_strategy,
    task_graph_metadata,
)
from .task_report import write_task_report
from .token_budget import default_token_counter, TokenCounter
from .client import RoleClient
from .routing.config import WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS


def knowledge_reads(knowledge: str) -> bool:
    return knowledge in ("read", "update")


def collects_feedback(request: CorrectionRequest) -> bool:
    """`<task_update_feedback>` is asked for under `update` only, unless the
    caller says otherwise (`CorrectionRequest.collect_feedback`)."""

    if request.collect_feedback is not None:
        return request.collect_feedback
    return request.knowledge == "update"


def _load_reusable_research_context(
    context_path: Path,
    *,
    expected_planning: Mapping[str, Any],
) -> ContextPack | None:
    """Load a committed research context when it is still the right one (L3).

    The comparison is :func:`research_reuse_key`, not the whole planning dict:
    model, preset, policy, difficulty, continuity and knowledge content all
    changed how this context was produced without making it wrong. Window
    geometry does not gate it either -- the notes carry source-id intervals and
    remap onto any later plan. What still gates it is the source, the retrieval
    mode and the knobs a user turns *at this artifact* -- notes and search
    rounds -- where silently reusing the old result would ignore the request.

    Corrupt or incompatible files are not a recoverable checkpoint; they are
    copied aside before the caller overwrites them. A pack whose notes carry no
    source ranges counts as incompatible: it can only be injected half-empty.
    """

    try:
        saved = json.loads(context_path.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            raise ValueError("research context root is not an object")
        saved_planning = saved.get("planning")
        if not isinstance(saved_planning, dict):
            raise ValueError("missing planning metadata")
        saved_key = research_reuse_key(saved_planning)
        expected_key = research_reuse_key(expected_planning)
        if saved_key != expected_key:
            differing = sorted(
                key
                for key in set(saved_key) | set(expected_key)
                if saved_key.get(key) != expected_key.get(key)
            )
            current_reporter().warning(
                "research-context-stale",
                f"{context_path} was planned under different parameters "
                f"({', '.join(differing)})",
                impact="重跑 research",
            )
            return None
        return load_research_context(context_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        backup = backup_unrecoverable_json(context_path)
        current_reporter().warning(
            "research-context-unrecoverable",
            f"{context_path} is not a recoverable research context ({exc})",
            impact=f"已备份到 {backup}，重跑 research",
        )
        return None


def run_post_correction_knowledge_update(
    *,
    task_id: str,
    task_summary: str,
    result_srt_path: str | Path,
    output_path: str | Path,
    stable_json: str | Path,
    artifact_dir: str | Path,
    refined_srt: str | Path | None = None,
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    test_profile: bool = False,
    counter: TokenCounter | None = None,
    difficulty: str = "quality",
    style_names: Sequence[str] = (),
    user_summary: bool = False,
    parse_retries: int | None = None,
) -> Dict[str, Any]:
    """Run the unified knowledge update right after a correction task.

    ``difficulty`` is the run's own: it selects the knowledge cell, so a preset
    binding a cheaper group at intermediate has to reach this call rather than
    being resolved as the top tier.

    ``result_srt_path`` is whatever SRT the run produced (the postprocessed
    final SRT, or the translated SRT when postprocess was deferred) — the
    final_csv overlay uses it directly. Sibling paths derive from
    ``output_path`` (the task's final SRT anchor); ``stable_json`` is passed
    explicitly because it may live under a different stem (reference ingest).
    ``user_summary``: ``task_summary`` is the user's own `--task-summary`, not the default line.
    """

    out = Path(output_path).expanduser().resolve()
    base = out.with_suffix("")
    artifact_path = Path(artifact_dir).expanduser().resolve()
    report = run_knowledge_update(
        final_srt=result_srt_path,
        stable_json=stable_json,
        annotated_csv=base.with_name(f"{base.name}-annotated.csv"),
        research_context=ensure_research_context_path(
            artifact_dir=artifact_path,
            stem=base.name,
            run_dir=out.parent,
        ),
        artifact_dir=artifact_path,
        refined_srt=refined_srt,
        task_id=task_id,
        task_summary=task_summary,
        knowledge_root=knowledge_root,
        test_profile=test_profile,
        token_counter=counter,
        difficulty=difficulty,
        style_names=style_names,
        user_summary=user_summary,
        parse_retries=parse_retries,
    )
    write_task_report(
        artifact_path,
        task_id=task_id,
        outputs={
            "final_srt": str(Path(result_srt_path).expanduser().resolve()),
            "knowledge_update_ledger": report["ledger_path"],
        },
    )
    return report


def _fast_execute_kwargs(decision: FastDecision) -> Dict[str, Any]:
    """execute_correction_windows kwargs for an enabled fast decision: the whole
    input as one window task (plan §3, fast column)."""

    return {"windows_override": [decision.window]}


def _record_dispatch(
    artifact_dir: str | Path,
    *,
    task_id: str,
    fast_decision: FastDecision,
    profile: TranslationProfile,
    knowledge: str,
    style_selected: bool,
    test_profile: bool,
) -> TaskGraph:
    """The two dispatch artifacts: ``fast_decision``, then the ``task_graph`` it implies.

    Every research or execute run writes it, so plan / production / replay can
    be compared on one record (plan §8, last bullet).
    """

    append_task_artifact(
        artifact_dir, kind="fast_decision", task_id=task_id, payload=fast_decision.to_metadata()
    )
    strategy, source = resolve_execution_strategy(profile, test_profile=test_profile)
    # Resolved per task, so recorded per task: the preparation task runs on the
    # research chain and is text-only, which can be a tool session where a
    # media-bearing window cannot (agy + the default `--media audio` is exactly
    # that split). Recording only the window's would file such a run as
    # `batched` while its preparation fetched for itself. Not asked at all
    # under fast: there is no preparation task, and the question is not free
    # (it walks a chain and may probe a driver).
    prepare_strategy, prepare_source = (
        resolve_execution_strategy(profile, test_profile=test_profile, task_group="research")
        if not fast_decision.enabled
        else (None, "")
    )
    graph = plan_task_graph(
        strategy=strategy,
        prepare_strategy=prepare_strategy,
        fast_enabled=fast_decision.enabled,
        grant=grant_for_run(profile, knowledge=knowledge, style_selected=style_selected),
    )
    append_task_artifact(
        artifact_dir,
        kind="task_graph",
        task_id=task_id,
        payload=task_graph_metadata(
            graph, strategy_source=source, prepare_strategy_source=prepare_source
        ),
    )
    return graph


def _auto_style_catalog(knowledge_root: str | Path, *, correct_only: bool = False) -> StyleCatalog:
    """`--style auto`: the candidates the preparation task picks from (a correct-only task's: the styles
    written for it, `style.CORRECT_ONLY_TASK`)."""

    return load_style_catalog(knowledge_root, correct_only=correct_only)


def _apply_auto_style(
    selection: StyleSelection,
    catalog: StyleCatalog,
    *,
    knowledge_root: str | Path,
    context_path: Path,
    artifact_dir: Path | None,
    task_id: str,
    fast: bool,
) -> tuple[StyleSelection, bool]:
    """Pin `--style auto`'s pick before any window starts; ``(selection, window_picks)``.

    The record comes first, in both modes (auto-style-gaps plan, C 「先读记录」
    step 1): whatever this task already committed -- the preparation task's
    pick in `research-context.json` (which outranks any artifact copy), else a
    fast window's `style_selection` artifact or an earlier run's no-choice
    result (`_recorded_style_choice`) -- is what a resume uses, whatever the
    base holds now. Handing a recorded fast pick back to the
    window would let a resume re-pick, and the two halves of a split window
    mix styles.

    Without a record:

    * a selection that does not ask (``ask=False``: efficiency, which uses
      only the factory default style, owner 2026-10-30): a silent, persisted
      choice of that default: no user style, which renders the factory default.
    * nothing to choose (no styles, or only `default_style`): a silent,
      persisted choice of the static default -- `default_style` when it
      exists, else none (step 2). Not a fallback: nobody was asked, because
      there was nothing to ask about.
    * fast, with candidates: the window picks (``window_picks``) and records
      it itself -- the batched request turn or the agentic delivery.
    * normal mode, with candidates: the preparation task had to pick and
      record it (or fail the task), so no record is a fact about the setup
      -- a context made without auto, for one -- and says so.
    """

    choice = _recorded_style_choice(context_path=context_path, artifact_dir=artifact_dir)
    # Whether "no style this run" is a fact about the *setup* (worth saying) or
    # a recorded decision (the task looked and chose none -- auto-style plan
    # §2.2: "一次无匹配是正常结果，记录简短理由，不作为 warning").
    structural = False
    if choice is None:
        if not selection.ask:
            choice = StyleChoice(
                outcome="none", reason="efficiency 只用出厂默认风格", rev=catalog.rev
            )
        elif not catalog.has_choice:
            choice = StyleChoice(
                outcome="selected" if catalog.has_default else "none",
                key=DEFAULT_STYLE_NAME if catalog.has_default else "",
                reason="没有可选的 style",
                rev=catalog.rev,
            )
        elif fast:
            return selection, True
        else:
            choice = StyleChoice(
                outcome="none", reason="准备阶段没有留下风格选择记录", rev=catalog.rev
            )
            structural = True
    render_rev = choice.rev
    if choice.key:
        # The record is honoured before the catalog is judged: a task whose
        # styles have all been retired since can still render its pick out of
        # history, and dropping it would take the style off the remaining
        # windows of a task that already used it.
        renderable, render_rev = _renderable_style_rev(knowledge_root, choice)
        if not renderable:
            # Neither the revision it was picked at nor the current base can
            # produce this style's rules any more. `render_style_block` would
            # raise and take the whole resume with it, over a style: drop it and
            # say so. An explicit `--style <名>` still raises: there the name is
            # the user's own instruction, not a recovered record.
            choice = StyleChoice(
                outcome="none",
                reason=f"记录里选中的 {choice.key!r} 已无法从知识库渲染出正文",
                rev=catalog.rev,
            )
            structural = True
    if choice.outcome == "none" and structural:
        current_reporter().warning(
            "style-auto-none",
            f"--style auto 本次没有选中风格：{choice.reason}",
            impact="本次只注入出厂默认风格",
        )
    if artifact_dir:
        append_task_artifact(
            artifact_dir, kind="style_selection", task_id=task_id, payload=choice.to_dict()
        )
    return selection.resolved(choice.names, rev=render_rev), False


def _settle_style(
    selection: StyleSelection,
    catalog: StyleCatalog,
    *,
    knowledge_root: str | Path,
    context_path: Path,
    artifact_dir: Path,
    task_id: str,
    fast: bool,
) -> tuple[StyleSelection, bool]:
    """The task's style, decided and recorded: `auto`'s pick or record (`_apply_auto_style`), else the named
    selection as it is (`_record_named_style`). ``(selection, window_picks)`` as `_apply_auto_style`."""

    if selection.auto:
        return _apply_auto_style(
            selection, catalog, knowledge_root=knowledge_root, context_path=context_path,
            artifact_dir=artifact_dir, task_id=task_id, fast=fast,
        )
    _record_named_style(selection, knowledge_root=knowledge_root, artifact_dir=artifact_dir, task_id=task_id)
    return selection, False


def _with_length_scale(request: CorrectionRequest, choice: Any, selection: StyleSelection | None, *,
                       knowledge_root: str | Path) -> Any:
    """The task's choice of passes with its length ratio, settled once its style is (`length_tier`): the
    recorded one when resuming, else resolved from the injected style's tier and the rest. ``selection`` None:
    a fast window picks the style itself, whose tier does not take part."""

    scale = task_length_scale(
        request.split_length_scale, choice.length_scale, knowledge_root=knowledge_root,
        names=None if selection is None else selection.names + selection.tier_names,
        rev=None if selection is None else selection.rev,
    )
    if choice.length_scale is not None:
        return choice
    return replace(choice, length_scale=scale.value, length_source=scale.source)


def _record_named_style(
    selection: StyleSelection,
    *,
    knowledge_root: str | Path,
    artifact_dir: Path,
    task_id: str,
) -> None:
    """Record an explicitly named style as this task's choice.

    The name is the user's instruction for this run and is never overridden by
    a record. Recording it is for a later run of the same task that names
    nothing: under the default `auto` that resume reads the task's record first
    and continues the named style instead of picking anew or, finding no
    record, running without one (owner 2026-10-02, same root cause as the
    efficiency resume). One name, or an explicit "none"; a stacked
    `--style 甲组,乙组` has no single-choice record shape and is not recorded.
    """

    if selection.mode == "none" or len(selection.names) > 1:
        return
    if selection.names:
        keys = resolve_style_keys(knowledge_root, selection.names, rev=selection.rev)
        choice = StyleChoice(
            outcome="selected", key=keys[0], reason="--style 点名", rev=selection.rev, intent="named"
        )
    else:
        choice = StyleChoice(outcome="none", reason="--style 显式不用风格", intent="named")
    append_task_artifact(
        artifact_dir, kind="style_selection", task_id=task_id, payload=choice.to_dict()
    )


def _previewed_style_names(selection: StyleSelection, catalog: StyleCatalog) -> tuple[str, ...] | None:
    """What a plan's prompt artifacts show as the injected style; None while a model still has to pick.

    A plan runs no preparation and pins nothing, so under `auto` it can only
    show what is already certain: with no choice to make, the no-choice result
    (`default_style` when the store has it, else no user style: the factory
    default). With a choice the pick is the model's, and the plan shows no
    style block -- not the factory default, which is only one possible pick. A
    selection that does not ask (efficiency) has only the factory default.
    """

    if selection.names or not selection.auto:
        return selection.names
    if not selection.ask:
        return ()
    if not catalog.has_choice:
        return (f"{STYLE_CATEGORY}{CATEGORY_SEPARATOR}{DEFAULT_STYLE_NAME}",) if catalog.has_default else ()
    return None


def writable_styles_for_task(
    selection: StyleSelection,
    *,
    knowledge_root: str | Path,
    output_path: str | Path,
    artifact_dir: Path,
) -> tuple[str, ...]:
    """The styles a task's post-task update may write, for a caller that ran
    the correction task itself (`workflows.reference_ingest`): `selection`'s
    own names, or -- under `auto` -- the pick the task recorded."""

    if not selection.auto:
        return selection.writable
    anchor = Path(output_path).expanduser().resolve()
    context_path = ensure_research_context_path(
        artifact_dir=artifact_dir, stem=anchor.stem, run_dir=anchor.parent
    )
    return _writable_style_names(
        selection,
        load_style_catalog(knowledge_root),
        context_path=context_path,
        artifact_dir=artifact_dir,
    )


def _renderable_style_rev(
    knowledge_root: str | Path, choice: StyleChoice
) -> tuple[bool, int | None]:
    """``(can it render, at which revision)`` for a recorded pick.

    The revision it was made against first (auto-style plan §5: a resume
    injects the rules the earlier windows actually saw, so rewriting -- or
    retiring -- the entry afterwards changes nothing for this task), then the
    current base. ⚠ The two answers have to come back together: probing both
    and then rendering at the recorded one turned "the recorded revision has
    no such entry but the current base does" into a `StyleSelectionError` at
    render time -- exactly what this check exists to prevent. That cell is
    reachable whenever the recorded revision belongs to a *different* store
    (`knowledge/` rebuilt, or a different `--knowledge-root`), because a
    revision is an integer with meaning only inside one store.
    """

    for rev in ((choice.rev, None) if choice.rev is not None else (None,)):
        try:
            if load_style_entries(knowledge_root, choice.names, rev=rev):
                return True, rev
        except StyleSelectionError:
            continue
    return False, None


def _recorded_style_choice(
    *, context_path: Path | None, artifact_dir: Path | None
) -> StyleChoice | None:
    """The pick this *task* committed, read back from whatever recorded it.

    Two kinds of record, both durable and both belonging to the task rather
    than to one run:

    * `research-context.json`'s ``style_selection`` -- the preparation task's
      pick. **It wins whenever it is there.** The orchestrator copies every pin
      into a `style_selection` artifact as well, and that copy must never
      outrank the context it was copied from: when the preparation reruns
      (its reuse key changed, or the user deleted the context) and picks
      anew, the new context is the task's decision, and reading the old copy
      first would inject and grant writes to a style the context and the
      report no longer name (peer review of phase A, 2026-10-01).
    * the latest `style_selection` artifact -- where the record lives when the
      context holds none: a fast window's pick (with ``chunk_id``; fast runs no
      preparation, so its context has no pick), and the no-choice result
      `_apply_auto_style` records for a run with nothing to choose.

    One rule for both modes, so the pin and the post-task write read the same
    record -- and the write is reached from an early return that never decides
    fast or not.

    Reading it back rather than handing it along in memory is what covers the
    runs that never reach the windows at all: an SRT that already exists takes
    an early return, which is exactly the shape of "the knowledge update was
    interrupted, run it again".
    """

    if context_path is not None:
        try:
            saved = json.loads(context_path.read_text(encoding="utf-8")).get("style_selection")
        except (OSError, ValueError):
            saved = None
        if isinstance(saved, Mapping):
            return StyleChoice.from_dict(saved)
    if artifact_dir:
        path = Path(artifact_dir) / TASK_ARTIFACT_FILENAME
        try:
            records = list(jsonl.iter_records(path))
        except OSError:
            records = []
        for record in reversed(records):
            if record.get("kind") == "style_selection" and isinstance(
                record.get("payload"), Mapping
            ):
                return StyleChoice.from_dict(record["payload"])
    return None


def _writable_style_names(
    selection: StyleSelection,
    catalog: StyleCatalog,
    *,
    context_path: Path,
    artifact_dir: Path | None,
) -> tuple[str, ...]:
    """Which style the post-task knowledge update may write into (§6).

    `selection.writable` as soon as this run pinned a name. Under `--style
    auto` it may never have: the fast window picks inside itself, and an early
    return skips the pin entirely. The pick is a property of the task, so it is
    recovered from the durable record instead.

    ⚠ **Reading and writing are judged against different revisions.** The
    translation may render a retired entry out of history (§5: the resume
    injects the rules the earlier windows saw); a proposal has nowhere to land
    -- the update writes to the current head, and `resolve_style_keys` would
    raise on a name that is no longer there. So the write target is always
    checked against the *current* catalog, including the names this run pinned
    itself.
    """

    if selection.mode != "update":
        return ()
    if not selection.auto:
        # An explicit `--style <名>` is the user's own instruction; a name that
        # no longer resolves is their error to hear about, not ours to drop.
        # (The catalog is not even loaded for those runs.)
        return selection.writable
    names = selection.writable or ()
    if not names:
        choice = _recorded_style_choice(context_path=context_path, artifact_dir=artifact_dir)
        if choice is None or not choice.key:
            return ()
        names = choice.names
    # Compared as the qualified names both sides are built from, not by
    # splitting one back apart: `StyleChoice.names` and this set come from the
    # same key, so there is nothing to parse.
    current = {f"{STYLE_CATEGORY}/{c.key}" for c in catalog.candidates}
    return tuple(name for name in names if name in current)


def _prepare_stage_runs(fast_decision: FastDecision) -> bool:
    """Normal mode always prepares (plan §3: capabilities only shape the task);
    fast mode never does -- the whole input is one translate task that asks
    for its own inputs."""

    return not fast_decision.enabled


@dataclass
class CorrectionOutcome:
    """What one task produced, for the front end to report."""

    #: The SRT the task ended at: final, or translated when the request stops
    #: before postprocess. None for a plan or a research-only request.
    result: Path | None = None
    #: The result was already on disk: nothing ran but what it still owed
    #: (the postprocess tail, the knowledge update).
    result_reused: bool = False
    #: The research context the windows read, and whether an earlier run (or
    #: the caller, through ``context_file``) had already prepared it.
    research_context: Path | None = None
    research_reused: bool = False
    #: The post-task knowledge update's report, when ``knowledge="update"``.
    knowledge_update: Dict[str, Any] | None = None
    artifact_dir: Path | None = None


def run_correction_task(request: CorrectionRequest) -> CorrectionOutcome:
    """Run one correction task -- research, the windows, the knowledge update --
    inside its run boundary.

    Every front end comes through here (the module CLI, ``run_pipeline``'s LLM
    stage, reference ingest), so the task is orchestrated once: there used to
    be a second copy in the CLI, and fixes kept landing in only one of them.

    The boundary: the run's agent slot account, one agent session scope that
    research, the windows and the knowledge update share (docs/llm_local_agent.md
    §12.1.3), and the knowledge generation pin (``_generation_pin``;
    ``knowledge="none"`` never opens the store). An executed task also books
    its timing into the run metadata and writes the task report -- after the
    session scope has closed, so the report sees what those sessions spent.

    Refusals are ``CorrectionRequestError`` and come before any work.
    """

    request = _resolve_request(request)
    artifact_dir = _task_artifact_dir(request)
    request, language = _with_task_mode(request, artifact_dir)
    # Before the finished-task shortcuts here and in `_run_task`: they reuse or judge a finished task without
    # reaching `task_choice`, and a judge the split cannot run for is refused there too (owner 2026-10-30).
    three_pass.refuse_judge(request.profile, request.speaker_judge,
                            three_pass.recorded_choice(artifact_dir)
                            if request.resume and artifact_dir is not None else None)
    execute = request.mode == "execute"
    out = Path(request.output_path).expanduser().resolve() if request.output_path else None
    task_id = _task_id(request)
    existed = bool(
        execute and out and (out.exists() or out.with_name(f"{out.stem}-translated.srt").exists())
    )
    status = (
        "reused" if existed and request.knowledge != "update" and not _judge_owed(request, artifact_dir)
        else "executed"
    )
    started = time.perf_counter()
    if status == "reused":
        # An earlier run left the output and nothing is owed but, at most, the
        # postprocess tail: no agent slot, session or knowledge pin is needed,
        # and taking a slot would queue a finished item of a batch resume
        # behind the agent tasks still running.
        finished = _finished_output(request, out, artifact_dir=artifact_dir, task_id=task_id)
        if finished is not None:
            _book_run(out, artifact_dir, task_id=task_id, status=status, started=started)
            return CorrectionOutcome(result=finished, result_reused=True, artifact_dir=artifact_dir)
        status = "executed"
    _record_switches(request, artifact_dir)
    sessions: Any = None
    outcome: CorrectionOutcome | None = None
    try:
        with _task_slots(request.test_profile), agent_session_scope() as sessions, _generation_pin(
            str(request.knowledge), request.knowledge_root
        ):
            if artifact_dir is not None:
                # Said once, while the scope is open: it files what the
                # sessions kept when it closes.
                set_run_evidence_destination(artifact_dir)
            outcome = _run_task(request, artifact_dir, language)
        return outcome
    except BaseException:
        status = "failed"
        raise
    finally:
        if sessions is not None and artifact_dir is not None:
            write_agent_session_usage(artifact_dir, sessions)
        if execute and out is not None and artifact_dir is not None:
            _book_run(out, artifact_dir, task_id=task_id, status=status, started=started)
        elif outcome is not None and outcome.research_context and artifact_dir is not None:
            write_task_report(
                artifact_dir,
                task_id=task_id,
                outputs={"research_context": str(outcome.research_context)},
            )


def _resolve_request(request: CorrectionRequest) -> CorrectionRequest:
    """``resolved_request``, plus the runtime's knowledge root for an unset one."""

    request = resolved_request(request)
    if request.knowledge_root is None:
        request = replace(request, knowledge_root=DEFAULT_KNOWLEDGE_ROOT)
    return request

def _task_id(request: CorrectionRequest) -> str:
    return request.task_id or Path(request.stable_json).stem


def _record_switches(request: CorrectionRequest, artifact_dir: Path | None) -> None:
    """Write ``RESOLVED_SWITCHES_FILENAME`` -- what this task's correction
    actually ran with and where each switch came from, for whoever reads the
    run later (owner 2026-09-30: downstream readers, not the retired desktop's
    record) -- and note the switches nobody typed.

    Here rather than in a front end because every front end comes through
    this function, and because only here is the vector the one the task
    really uses (reference ingest's ASR step runs ``run_pipeline`` with no LLM
    switches at all). Notes go to ``debug`` -- the log file always, the
    terminal under ``--verbose`` -- like the other routing notes.
    """

    profile = request.profile
    values = {
        "media": profile.correction_media,
        "retrieval": profile.retrieval,
        "difficulty": profile.difficulty,
        "continuity": profile.continuity,
        "knowledge": str(request.knowledge),
        "parallel_windows": request.parallel_windows,
    }
    sources = dict(request.switch_sources or {key: "caller" for key in values})
    for key, source in sources.items():
        if source.startswith("preset:"):
            current_reporter().debug(f"{key}={values[key]} 来自预设 {source[len('preset:'):]}（命令行未指定）")
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / RESOLVED_SWITCHES_FILENAME).write_text(
        json.dumps(
            {"switches": values, "sources": sources, "active_preset": _active_preset_id()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _active_preset_id() -> str:
    from .routing.model_routes import default_model_routes

    return default_model_routes().active_preset_id


def _task_artifact_dir(request: CorrectionRequest) -> Path | None:
    """Where the task keeps its artifacts: as asked, else beside the output
    (beside the input when research runs without one). A plan without an
    explicit directory keeps none."""

    if request.task_artifact_dir:
        return Path(request.task_artifact_dir).expanduser().resolve()
    if request.mode == "plan":
        return None
    if request.output_path:
        return Path(request.output_path).expanduser().resolve().with_suffix(ARTIFACT_DIR_SUFFIX)
    source = Path(request.stable_json).expanduser().resolve()
    return source.with_name(f"{source.stem}{ARTIFACT_DIR_SUFFIX}")


def _research_context_path(request: CorrectionRequest, artifact_dir: Path) -> Path:
    anchor = Path(request.output_path or request.stable_json).expanduser().resolve()
    return ensure_research_context_path(
        artifact_dir=artifact_dir, stem=anchor.stem, run_dir=anchor.parent
    )


def _finished_output(
    request: CorrectionRequest, out: Path, *, artifact_dir: Path, task_id: str
) -> Path | None:
    """The task's output when an earlier run already produced it (stages skip
    on existence), finishing the postprocess tail on the way; None when the
    windows still have to run -- or pass 3 does: an earlier pass 3 that did
    not finish (failed, or stopped half way through a rerun) left pass 2's
    subtitles in the output, and a resumed run goes through the windows again
    (replayed from the ledger) to ask the windows pass 3 has no answer for."""

    if request.resume and _proofread_owed(artifact_dir):
        return None
    translated = out.with_name(f"{out.stem}-translated.srt")
    if request.postprocess_profile is None:
        return translated if translated.exists() else None
    if out.exists():
        return out
    if not translated.exists():
        return None
    # The pad bound is the duration recognition recorded, not a probe: by
    # now the source may be gone (a URL input whose download was cleaned up).
    from finesub.speech.postprocessing.stabilization import recorded_audio_end

    report = postprocess_srt_file(
        translated,
        output_path=out,
        profile=request.postprocess_profile,
        audio_end=recorded_audio_end(request.stable_json),
        keys=_finished_keys(request.stable_json, out, translated),
    )
    raw = out.with_name(f"{out.stem}-raw.srt")
    append_task_artifact(
        artifact_dir,
        kind="final_srt",
        task_id=task_id,
        payload={
            "path": str(out),
            **({"raw_path": str(raw)} if raw.exists() else {}),
            "translated_path": str(translated),
            "postprocess": report.to_dict(),
        },
    )
    return out


def _finished_keys(stable_json: Path, out: Path, translated: Path) -> list[str | None] | None:
    """The group keys of `translated`'s cues when the final SRT is finished from disk: each `-annotated.csv`
    row's `position` (its source ids, one row per cue) names its first product line, whose group key the
    run that wrote the rows keyed it by (`run.product_keys`); a row with an empty translation is no cue
    (`run._write_products`). None when no product line has one; None with a warning -- the cues are then kept
    apart as if one person spoke them all -- when the rows do not match."""
    from .chunking import load_product_segments
    from .output_protocol import KIND_SUB, split_annotated_row
    from finesub.subtitles.model import parse_srt

    try:
        key_by_id = {line.id: line.group_key for line in load_product_segments(stable_json)[0]}
    except (OSError, ValueError, TypeError):
        return None  # the lines are not readable: the finish goes on as it always did
    if not any(key is not None for key in key_by_id.values()):
        return None
    annotated = out.with_name(f"{out.stem}-annotated.csv")
    try:
        rows = [row for row in (split_annotated_row(line) for line in annotated.read_text(encoding="utf-8").splitlines()
                                if line.strip() and not line.startswith("#"))
                if row is None or row.translation.strip()]
        cues = len(parse_srt(translated.read_text(encoding="utf-8")))
    except (OSError, ValueError) as error:
        rows, cues, problem = [], 0, str(error)
    else:
        problem = None if len(rows) == cues else f"{len(rows)} 行对 {cues} 条字幕"
    keys: list[str | None] = []
    for row in rows if problem is None else ():
        first = row.position.split(",")[0].strip() if row is not None else ""
        if row is None or (row.kind == KIND_SUB and first not in key_by_id):
            problem = f"有一行认不出（{first or '格式不对'}）"
            break
        keys.append(key_by_id[first] if row.kind == KIND_SUB else None)
    if problem is not None:
        current_reporter().warning(
            "final-srt-keys", f"{annotated.name} 与 {translated.name} 对不上：{problem}",
            impact="不同说话人同时说话的字幕会互相截断")
        return None
    return keys if any(key is not None for key in keys) else None


def _proofread_owed(artifact_dir: Path | None) -> bool:
    """A pass 3 that started and is not done, in a task whose recorded choice still runs it (a task chosen
    again without pass 3 keeps its old record, which owes nothing)."""

    from .stages.proofread.run import proofread_owed

    choice = three_pass.recorded_choice(artifact_dir)
    return artifact_dir is not None and choice is not None and choice.proofread and proofread_owed(artifact_dir)


def _run_task(request: CorrectionRequest, artifact_dir: Path | None, language: str | None) -> CorrectionOutcome:
    """``request`` already carries the task's mode (`_with_task_mode`), ``language`` the source language read."""
    profile = request.profile
    root = request.knowledge_root
    knowledge = str(request.knowledge)
    reads = knowledge_reads(knowledge)
    feedback = collects_feedback(request)
    task_id = _task_id(request)
    task_summary = request.task_summary or f"LLM subtitle correction task {task_id}"
    execute = request.mode == "execute"
    out = Path(request.output_path).expanduser().resolve() if request.output_path else None
    audio_label = str(request.audio_path) if request.audio_path else ""
    outcome = CorrectionOutcome(artifact_dir=artifact_dir)

    # Before the early return below (it can still owe a knowledge update,
    # which writes into the pick) and before planning (a plan must render the
    # prompt the run would really send; review 2026-09-02).
    style_selection = _task_style_selection(request)
    style_catalog = (_auto_style_catalog(root, correct_only=profile.correct_only) if style_selection.auto
                     else StyleCatalog())
    # What a model is shown and asked to pick from: only a catalog with a
    # choice in it. Only `default_style`, or no style at all, asks nobody and
    # costs no call -- `_apply_auto_style` records the no-choice result.
    offered_catalog = (
        style_catalog if style_catalog.has_choice and style_selection.ask else StyleCatalog()
    )
    if execute:
        _warn_if_style_update_is_inert(style_selection, knowledge, request.refined_srt)
    # Where `--style auto` reads its pick back from: the preparation task
    # persisted it into the context this run uses, which `context_file`
    # redirects. Resolved before the early return, which is the resumed-update
    # path that reads it.
    context_path = (
        (Path(request.context_file) if request.context_file else _research_context_path(request, artifact_dir))
        if artifact_dir is not None and request.mode != "plan"
        else None
    )

    def knowledge_update(result: Path) -> None:
        if knowledge != "update":
            return
        outcome.knowledge_update = run_post_correction_knowledge_update(
            task_id=task_id,
            task_summary=task_summary,
            result_srt_path=result,
            output_path=out,
            stable_json=request.stable_json,
            artifact_dir=artifact_dir,
            refined_srt=request.refined_srt,
            knowledge_root=root,
            test_profile=request.test_profile,
            difficulty=profile.difficulty,
            style_names=_writable_style_names(
                style_selection, style_catalog, context_path=context_path, artifact_dir=artifact_dir
            ),
            user_summary=bool(request.task_summary.strip()),
            parse_retries=request.max_retries_per_window,
        )

    if execute:
        finished = _finished_output(request, out, artifact_dir=artifact_dir, task_id=task_id)
        if finished is not None:
            outcome.result, outcome.result_reused = finished, True
            if _judge_owed(request, artifact_dir):
                _judge_finished_task(request, out, artifact_dir=artifact_dir, task_id=task_id,
                                     context_path=context_path, knowledge_root=root if reads else None)
            knowledge_update(finished)
            return outcome

    # One pass or three (`stages/three_pass`): the task's own choice when resuming, else chosen now and
    # recorded once the windows are about to run.
    from finesub.subtitles.stable_rows import read_rows

    rows = read_rows(request.stable_json)
    choice = three_pass.task_choice(request, artifact_dir, rows, source_language=language)
    if profile.correct_only and choice.proofread:
        _correct_only_warning("中文素材只纠错，校对 pass 的 prompt 是按翻译写的（日语原文对中文译文），效果没量过；"
                            "--llm-proofread on 照跑")
    shape = (three_pass.pass1_shape(request.speaker_judge, warn=_three_pass_warning)
             if choice.mode == three_pass.THREE_PASS else None)
    grouped = three_pass.pass1_groups(shape, rows)
    # A resumed three-pass task whose cell has no grouped version now (a lower difficulty) corrects its windows
    # on one pass, bar those the ledger replays (`pass2_plan`): the dry run shows that prompt.
    cell_grouped = grouped and three_pass.grouped_prompts(profile)

    # Shared across planning, research and the windows: the sha cache makes
    # repeated countTokens calls over identical window texts free.
    token_counter = default_token_counter()
    try:
        fast_decision = decide_fast_mode(
            stable_json=request.stable_json,
            fast=request.fast,
            profile=profile,
            audio_path=request.audio_path,
            extra_info=request.extra_info,
            knowledge_root=root,
            knowledge_enabled=reads,
            token_counter=token_counter,
            # Required: omitting it silently plans against DEFAULT_LIMITS
            # (test_llm_fast_plan guards every call site and says why). The
            # group planning envelope (plan v2 D13) budgets the fast window
            # against the correction cell's group minimum.
            limits=correction_planning_limits(profile),
            max_window_subtitle_tokens=request.max_window_subtitle_tokens,
            style_catalog=offered_catalog.render(),
            # D3: a fast task runs its pass 3 as one window too, so that window has to fit.
            proofread_limits=_proofread_limits(profile) if choice.proofread else None,
        )
    except ValueError as exc:
        raise CorrectionRequestError(str(exc)) from exc
    if execute:
        # A plan works without configured provider keys; execution checks the
        # shape the fast planner chose before the first model call.
        try:
            validate_profile_capabilities(profile, test_profile=request.test_profile)
        except CapabilityUnavailableError as exc:
            raise CorrectionRequestError(str(exc)) from exc
    if request.on_plan is not None:
        previewed_styles = _previewed_style_names(style_selection, style_catalog)
        previewed_scale = task_length_scale(request.split_length_scale, choice.length_scale, knowledge_root=root,
                                            names=previewed_styles, rev=style_selection.rev)
        artifacts = build_prompt_artifacts(
            length_scale=previewed_scale,
            style_names=previewed_styles,
            style_catalog=offered_catalog.render(),
            extra_style=request.extra_style,
            test_profile=request.test_profile,
            stable_json=request.stable_json,
            audio_path=request.audio_path,
            video_path=request.video_path,
            audio_label=audio_label,
            extra_info=request.extra_info,
            knowledge_root=root,
            knowledge_enabled=reads,
            task_update_feedback=feedback,
            counter=token_counter,
            profile=profile,
            max_window_subtitle_tokens=request.max_window_subtitle_tokens,
            speakers=request.speaker_judge != "off",
            grouped=cell_grouped,
        )
        artifacts["fast_decision"] = fast_decision.to_metadata()
        artifacts["correction_mode"] = _previewed_mode(
            replace(choice, length_scale=previewed_scale.value, length_source=previewed_scale.source), shape,
            grouped=grouped, cell_grouped=cell_grouped,
        )
        if request.speaker_judge != "off" or shape is not None:
            artifacts.update(_preview_judge(request, fast_decision, token_counter, knowledge_root=root if reads else None,
                                            shape=shape))
        request.on_plan(artifacts)
    if request.mode == "plan":
        return outcome
    assert artifact_dir is not None and context_path is not None
    _record_dispatch(
        artifact_dir, task_id=task_id, fast_decision=fast_decision, profile=profile, knowledge=knowledge,
        # `auto` over a store with no style at all selects nothing -- the
        # same grant the old default gave that store.
        style_selected=bool(style_selection.names) or (style_selection.auto and bool(style_catalog)),
        test_profile=request.test_profile,
    )

    fast_kwargs: Dict[str, Any] = {}
    if request.context_file:
        if fast_decision.enabled:
            fast_kwargs = _fast_execute_kwargs(fast_decision)
        try:
            context_pack = load_research_context(request.context_file)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            # An explicitly named file is an instruction, not a checkpoint:
            # say why it cannot be used instead of quietly researching again.
            raise CorrectionRequestError(
                f"Error: {request.context_file} is not a usable research context ({exc})."
            ) from exc
        _seed_transfer_from_context(context_path, fast_kwargs)
        outcome.research_context, outcome.research_reused = context_path, True
    elif not _prepare_stage_runs(fast_decision):
        # Fast: one whole-input task, no preparation. The user's notes still
        # reach the window as its general context -- nothing else digests them.
        fast_kwargs = _fast_execute_kwargs(fast_decision)
        context_pack = fast_context_pack(request.extra_info)
    else:
        context_pack = None
        if context_path.exists():
            context_pack = _load_reusable_research_context(
                context_path,
                expected_planning=planning_metadata(
                    profile,
                    stable_json=request.stable_json,
                    extra_info=request.extra_info,
                    knowledge_root=root,
                    knowledge_enabled=reads,
                    collect_task_feedback=feedback,
                    audio_duration=(
                        probe_audio_duration(request.audio_path) if request.audio_path else None
                    ),
                    max_window_subtitle_tokens=request.max_window_subtitle_tokens,
                    limits=correction_planning_limits(profile),
                    test_profile=request.test_profile,
                ),
            )
        outcome.research_reused = context_pack is not None
        if context_pack is None:
            context_pack = run_research_stage(
                stable_json=request.stable_json,
                context_path=context_path,
                audio_path=request.audio_path,
                extra_info=request.extra_info,
                knowledge_root=root,
                knowledge_enabled=reads,
                test_profile=request.test_profile,
                task_artifact_dir=artifact_dir,
                task_id=task_id,
                token_counter=token_counter,
                profile=profile,
                collect_task_feedback=feedback,
                resume=request.resume,
                max_window_subtitle_tokens=request.max_window_subtitle_tokens,
                runner_kwargs={
                    **({"style_catalog": offered_catalog, "extra_style": request.extra_style} if offered_catalog else {}),
                    # The judge's cast candidates: asked only when it will run (owner 2026-10-09).
                    **({"speakers": True} if request.speaker_judge != "off" else {}),
                } or None,
            )
        outcome.research_context = context_path
        # v17: the preparation task's <keep_entries> seeds the first window's
        # transfer chain (persisted in the context JSON, so reuse keeps it).
        _seed_transfer_from_context(context_path, fast_kwargs)
    # The task's style is decided -- and recorded -- before a research-only
    # run returns: a later full run (perhaps after styles were added, or at
    # another difficulty) has to find the decision this task already made.
    style_selection, window_picks = _settle_style(
        style_selection, style_catalog, knowledge_root=root, context_path=context_path,
        artifact_dir=artifact_dir, task_id=task_id, fast=fast_decision.enabled)
    if profile.correct_only and style_selection.auto:
        style_selection = _drop_untagged_record(root, style_selection)
    if request.mode == "research":
        return outcome
    # The task's style is fixed here whenever the orchestrator decides it (an
    # explicit `--style`, the static `default_style`, or an `auto` pick or
    # record); a fast window that picks books its own when it does.
    record_style_injected(
        root, style_selection.names, task_id=task_id, rev=style_selection.rev
    )
    # Before the plan file: a task with windows and no recorded choice reads as one from before three passes.
    choice = _with_length_scale(request, choice, None if window_picks else style_selection, knowledge_root=root)
    three_pass.record_choice(artifact_dir, choice)
    # Resolved once and shared: a second resolution would find the plan file
    # this one wrote and read it as reused.
    correction_plan = resolve_correction_plan(
        stable_json=request.stable_json,
        task_artifact_dir=artifact_dir,
        test_profile=request.test_profile,
        audio_path=request.audio_path,
        video_path=request.video_path,
        token_counter=token_counter,
        resume=request.resume,
        profile=profile,
        max_window_subtitle_tokens=request.max_window_subtitle_tokens,
        windows_override=fast_kwargs.pop("windows_override", None),
    )
    correction_plan, grouping = _judge_and_group(
        request, correction_plan, context_pack, choice=choice, shape=shape, artifact_dir=artifact_dir,
        task_id=task_id, knowledge_root=root if reads else None, title=_judge_title(request, out))
    result = execute_correction_windows(
        plan=correction_plan,
        style_catalog=offered_catalog if window_picks else None,
        stable_json=request.stable_json,
        output_path=out,
        context_pack=context_pack,
        audio_label=audio_label,
        audio_path=request.audio_path,
        video_path=request.video_path,
        test_profile=request.test_profile,
        max_retries_per_window=request.max_retries_per_window,
        max_replacements_per_window=request.max_replacements_per_window,
        parallel_window_limit=request.parallel_windows,
        postprocess_profile=request.postprocess_profile,
        extra_style=request.extra_style,
        # A fast window that still picks gets no style up front: the factory default is one of its
        # possible picks, and injecting it before the pick would stack it under the style picked.
        style_block="" if window_picks else render_style_block(
            root, style_selection.names, rev=style_selection.rev, correct_only=profile.correct_only),
        knowledge_enabled=reads,
        task_artifact_dir=artifact_dir,
        task_id=task_id,
        task_update_feedback=feedback,
        token_counter=token_counter,
        resume=request.resume,
        profile=profile,
        knowledge_root=root,
        max_window_subtitle_tokens=request.max_window_subtitle_tokens,
        grouping=grouping,
        proofread=choice.proofread,
        proofread_title=_material_title(request, out),
        **fast_kwargs,
    )
    outcome.result = Path(result)
    knowledge_update(outcome.result)
    return outcome


def _previewed_mode(choice: Any, shape: str | None, *, grouped: bool, cell_grouped: bool) -> Dict[str, Any]:
    """The dry run's `correction_mode`: the task's choice and, with three passes, what pass 2 is previewed on."""

    if choice.mode != three_pass.THREE_PASS:
        return choice.to_metadata()
    pass2_input = (
        "waits for pass 1: previewed on the ASR lines, none marked 可并" if cell_grouped
        else "the correction cell has no grouped version now: every window the ledger does not replay gets "
        "the single pass prompt" if grouped
        else "pass 1 cannot run on these rows (no usable speaker labels): every window gets the single pass prompt"
    )
    return {**choice.to_metadata(), "pass1": shape, "pass2_input": pass2_input}


def _judge_and_group(request: CorrectionRequest, plan: Any, context_pack: Any, *, choice: Any, shape: str | None,
                     artifact_dir: Path, task_id: str, knowledge_root: Any, title: str) -> tuple[Any, Any]:
    """Before the windows: the task's length ratio (its ``choice``'s) written into every window, the judge --
    three passes' pass 1, else the judge whose verdicts the windows' 换人 reads -- and, with three passes, pass 2's
    windows and the grouping that maps them back, its checks on pass 1's joins at the task's ratio. The task's
    choice (its ratio included) and what pass 1 left go into the `correction_mode` artifact.

    The plan's windows were planned before the task's style was known, so this is where they get the ratio
    (a new plan, a resumed one rebuilt from its saved bounds, a fast task's window alike); the windows derived
    from them later (product lines, pass 2's groups, halves) carry it on."""

    thresholds = Thresholds.at(choice.length_scale)
    plan = replace(plan, windows=[replace(window, length_scale=choice.length_scale) for window in plan.windows])

    # A pass 3 still owed means every window was delivered: judging again now could only regroup them.
    if (request.speaker_judge != "off" or shape is not None) and not (request.resume and _proofread_owed(artifact_dir)):
        _judge_safely(request, plan.windows, context_pack, artifact_dir=artifact_dir, task_id=task_id,
                      knowledge_root=knowledge_root, fallback_title=title, shape=shape)
    plan = _on_product_lines(plan, request.stable_json)
    grouping = None
    record: Dict[str, Any] = {}
    if choice.mode == three_pass.THREE_PASS:
        windows, grouping, record = three_pass.pass2_plan(
            plan.windows, request.stable_json, shape,
            warn=_three_pass_warning, thresholds=thresholds,
            replayable=None if three_pass.grouped_prompts(request.profile) else _replayable(request, plan, artifact_dir))
        plan = replace(plan, windows=windows)
    append_task_artifact(artifact_dir, kind="correction_mode", task_id=task_id,
                         payload={**choice.to_metadata(), **record})
    return plan, grouping


def _replayable(request: CorrectionRequest, plan: Any, artifact_dir: Path) -> Any:
    """Whether the windows will replay this window from the ledger: the ledger they read, loaded on the same
    condition (`execute_correction_windows`: resume, and a plan reused or fast), asked the same question."""

    from .stages.correction.commit import WINDOW_CACHE_FILENAME
    from .stages.correction.context import ResumeLedger

    ledger = ResumeLedger(enabled=request.resume, path=artifact_dir / WINDOW_CACHE_FILENAME,
                          task_fingerprint=plan.task_fingerprint)
    if ledger.enabled and (plan.reused or plan.fast):
        ledger.load()
    return ledger.leaf_is_replayable


def _judge_owed(request: CorrectionRequest, artifact_dir: Path | None) -> bool:
    """The speaker judge is on and its sidecar does not yet cover this stable -- with the shape section, in a
    task that ran three passes."""

    if request.speaker_judge == "off":
        return False
    from finesub.speech.diarization.judgments import covers

    return not covers(request.stable_json, shape=_finished_shape(request, artifact_dir))


def _finished_shape(request: CorrectionRequest, artifact_dir: Path | None) -> str | None:
    """The pass 1 a task that already finished ran with: none for one that ran one pass."""

    choice = three_pass.recorded_choice(artifact_dir)
    if choice is None or choice.mode != three_pass.THREE_PASS:
        return None
    return three_pass.pass1_shape(request.speaker_judge, warn=_judge_warning)


def _judge_safely(request: CorrectionRequest, windows: Any, context_pack: Any, *, artifact_dir: Path, task_id: str,
                  knowledge_root: Any, fallback_title: str, shape: str | None = None) -> None:
    """The judge is an optional step: whatever goes wrong in it is a warning, never the correction's failure."""

    from .stages.speaker_judge.task import judge_task

    try:
        judge_task(request, windows, context_pack, client=RoleClient(test_profile=request.test_profile),
                   artifact_dir=artifact_dir, task_id=task_id, knowledge_root=knowledge_root,
                   fallback_title=fallback_title, warn=_judge_warning, counter=default_token_counter(), shape=shape)
    except Exception as error:  # noqa: BLE001 - see the docstring
        _judge_warning(f"说话人裁判没跑完（{type(error).__name__}: {error}），没判的行保留原有的标签；纠错照常进行")


def _on_product_lines(plan: Any, stable_json: str | Path) -> Any:
    """The plan's windows on the lines the correction reads (`chunking.product_windows`): the turns the judge's
    verdicts imply and the lines a judge that heard the audio split, filled or added. They were planned before
    it ran; a sidecar from an earlier run counts the same, judge on or off."""

    from .chunking import load_product_segments, product_windows

    try:
        segments, layer = load_product_segments(stable_json, warn=lambda message: current_reporter().warning(
            "speaker-judge-ignored", message, impact="纠错的换人标记按原有的标签写"))
    except (OSError, ValueError) as error:
        _judge_warning(f"裁判之后没能重读行（{type(error).__name__}: {error}），纠错的换人标记按原有的标签写")
        return plan
    return replace(plan, windows=product_windows(plan.windows, segments, layer))


def _judge_title(request: CorrectionRequest, out: Path | None) -> str:
    return (out or Path(request.stable_json)).stem.removesuffix("-stable")


def _material_title(request: CorrectionRequest, out: Path | None) -> str:
    """What pass 3 is told the material is: the extra-info block (URL, scraped title, description), else the
    output's name -- the judge's title too (`speaker_judge.task.judge_inputs`)."""

    return (request.extra_info or "").strip() or _judge_title(request, out)


def _proofread_limits(profile: TranslationProfile) -> Any:
    from .stages.proofread.run import planning_limits

    return planning_limits(profile.difficulty)


def _judge_warning(message: str) -> None:
    current_reporter().warning("speaker-judge", message)


def _with_task_mode(request: CorrectionRequest, artifact_dir: Path | None) -> tuple[CorrectionRequest, str | None]:
    """``request`` carrying its task's mode -- translate, or only correct a Chinese source -- in its profile
    (`TranslationProfile.correct_only`), and the source language read. Decided first of all: the knowledge
    base, the style, the finished-task shortcuts and the plan all follow it.

    A correct-only task neither reads nor writes the knowledge base (owner 2026-11-06): its entries and its
    update are written for translating, and what a Chinese task should keep there is not designed yet; nor does it
    ask the model for feedback to update it with. Said once when the user asked for it (`--knowledge read|update`,
    `--refined-srt`); the default `read` turns off silently."""
    correct_only, language = three_pass.task_correct_only(request, artifact_dir, warn=_correct_only_warning)
    if correct_only != request.profile.correct_only:
        request = replace(request, profile=replace(request.profile, correct_only=correct_only))
    if correct_only:
        if request.knowledge != "none" and (request.switch_sources or {}).get("knowledge", "caller") != "default":
            _correct_only_warning(f"中文素材只纠错，不读写知识库：--knowledge {request.knowledge} 这次按 none")
        if request.refined_srt:
            _correct_only_warning("中文素材只纠错，不读写知识库：--refined-srt 这次不用")
        request = replace(request, knowledge="none", refined_srt=None, collect_feedback=False)
    return request, language


def _task_style_selection(request: CorrectionRequest) -> StyleSelection:
    """The task's style pick (`_correct_only_style` for a correct-only task)."""
    selection = resolve_style_selection(
        request.style,
        request.style_mode,
        knowledge_root=request.knowledge_root,
        difficulty=request.profile.difficulty if request.style_efficiency_cap else "",
    )
    return _correct_only_style(request, selection) if request.profile.correct_only else selection


def _correct_only_style(request: CorrectionRequest, selection: StyleSelection) -> StyleSelection:
    """A correct-only task's style (owner 2026-11-05): only the styles written for it (`[任务] 中文纠错`,
    `style.CORRECT_ONLY_TASK`) are injected, whole -- `auto` picks among those alone (`_auto_style_catalog`); an
    untagged style the command line names is read for its `[字幕长度]` alone (`tier_names`), one from the config
    not at all. Nothing is written back, and `--extra-style` (a translation request) is not used."""

    if request.extra_style.strip():
        _correct_only_warning("中文素材只纠错，--extra-style 这次不起作用")
    if selection.mode == "update":
        _correct_only_warning("中文素材只纠错，不写回风格：--style-mode update 这次按 read")
        selection = replace(selection, mode="read")
    if selection.auto or not selection.names:
        return selection
    tagged, untagged = _split_tagged(request.knowledge_root, selection)
    tier_names = tuple(name for name in untagged if name in parse_style_names(request.style))
    if untagged:
        _correct_only_warning(
            f"风格 {'、'.join(untagged)} 没写「[任务] {CORRECT_ONLY_TASK}」，中文素材只纠错时不注入"
            + ("；命令行点名的只读它的 [字幕长度]" if tier_names else "")
        )
    return replace(selection, names=tagged, tier_names=tier_names)


def _split_tagged(knowledge_root: Any, selection: StyleSelection) -> tuple[tuple[str, ...], list[str]]:
    """The selection's styles written for correct-only (`style.CORRECT_ONLY_TASK`), and the rest."""

    entries = load_style_entries(knowledge_root, selection.names, rev=selection.rev)
    tagged = tuple(name for name, (_key, text) in zip(selection.names, entries) if for_correct_only(text))
    return tagged, [name for name in selection.names if name not in tagged]


def _drop_untagged_record(knowledge_root: Any, selection: StyleSelection) -> StyleSelection:
    """`auto`'s pick as a correct-only task settled it: a record this task kept from when it ran as a translation
    (a fast window's pick, or one recorded before the task was judged again with `--no-resume`) can name a style
    not written for correct-only, which is not injected. A fast task whose record is dropped whole runs without a
    style: its windows do not pick again (rare and cheap; delete the task's artifact directory to have them pick)."""

    if not selection.names:
        return selection
    tagged, untagged = _split_tagged(knowledge_root, selection)
    if untagged:
        _correct_only_warning(
            f"这个任务记录的风格选择 {'、'.join(untagged)} 没写「[任务] {CORRECT_ONLY_TASK}」"
            "（多半是按翻译跑时选的），中文素材只纠错时不注入"
        )
    return replace(selection, names=tagged)


def _correct_only_warning(message: str) -> None:
    current_reporter().warning("correct-only", message)


def _three_pass_warning(message: str) -> None:
    current_reporter().warning("three-pass", message)


def _judge_finished_task(
    request: CorrectionRequest, out: Path, *, artifact_dir: Path, task_id: str, context_path: Path | None,
    knowledge_root: Any,
) -> None:
    """The judge for a task whose correction already finished: the preparation task's context is
    reused when it is there, the windows are the saved correction plan's (else a fresh plan, not
    written), and only the `-annotated.csv` `spk` column is rendered again -- the correction itself is
    never re-run. A task that ran three passes is judged with pass 1's shape section, so its sidecar
    stays a pass 1's; the lines are not grouped again and passes 2 and 3 do not run (to regroup, delete
    the products)."""

    context_pack = None
    if context_path is not None and context_path.exists():
        try:
            context_pack = load_research_context(context_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            context_pack = None
    try:
        windows = resolve_correction_plan(
            stable_json=request.stable_json, task_artifact_dir=artifact_dir, test_profile=request.test_profile,
            audio_path=request.audio_path, video_path=request.video_path, token_counter=default_token_counter(),
            resume=True, profile=request.profile, max_window_subtitle_tokens=request.max_window_subtitle_tokens,
            persist=False,
        ).windows
    except Exception as error:  # noqa: BLE001 - the judge is optional: its windows failing is a warning too
        _judge_warning(f"说话人裁判没跑（排窗失败：{type(error).__name__}: {error}），行保留原有的标签；成品不变")
        return
    _judge_safely(request, windows, context_pack, artifact_dir=artifact_dir, task_id=task_id,
                  knowledge_root=knowledge_root, fallback_title=_judge_title(request, out),
                  shape=_finished_shape(request, artifact_dir))
    if not refresh_annotated_speakers(stable_json=request.stable_json, output_path=out):
        _judge_warning(f"{out.stem}-annotated.csv 不在，裁判结果只进了 sidecar 与分轨导出")


def _preview_judge(request: CorrectionRequest, fast_decision: FastDecision, counter: Any, *,
                   knowledge_root: Any, shape: str | None = None) -> Dict[str, Any]:
    """The dry run's judge prompts. Its windows nest in the plan this dry run computes itself, never
    written to the plan file -- a later real run must not take a dry run's plan as its own."""

    from .chunking import load_segments_from_stable_json, plan_correction_windows
    from .stages.speaker_judge.task import preview_task

    if fast_decision.enabled:
        windows = [fast_decision.window]
    else:
        windows = plan_correction_windows(
            load_segments_from_stable_json(request.stable_json), counter=counter,
            context_tokens=WINDOW_PLANNING_CONTEXT_RESERVE_TOKENS,
            audio_duration=probe_audio_duration(request.audio_path) if request.audio_path else None,
            profile=request.profile, max_window_subtitle_tokens=request.max_window_subtitle_tokens,
            limits=correction_planning_limits(request.profile),
        )
    out = Path(request.output_path) if request.output_path else None
    return preview_task(request, windows, knowledge_root=knowledge_root, fallback_title=_judge_title(request, out),
                        counter=counter, shape=shape)


def _warn_if_style_update_is_inert(
    selection: StyleSelection, knowledge: str, refined_srt: str | Path | None
) -> None:
    """`update` rides on the post-task knowledge update, and that task may only
    touch a style with refined subtitles in front of it. Asking for the write
    without either is a run that quietly behaves like `read`."""

    if not selection.wants_write:
        return
    missing = []
    if knowledge != "update":
        missing.append("--knowledge update")
    if not refined_srt:
        missing.append("--refined-srt")
    if missing:
        current_reporter().warning(
            "style-update-inert",
            f"--style-mode update 需要 {' 与 '.join(missing)}，本次缺少它们",
            impact="这套风格照常注入提示词，但本次学到的东西不会写回去",
            action="补上缺的参数，或改成 --style-mode read 让意图和行为一致",
        )


def _book_run(out: Path, artifact_dir: Path, *, task_id: str, status: str, started: float) -> None:
    """The executed task's line in the run metadata, and its task report."""

    metadata_path = metadata_path_for_output(out)
    update_run_metadata(
        metadata_path,
        {
            "task_id": task_id,
            "timing": {
                "stages": {
                    "llm_harness": stage_record(
                        status=status,
                        elapsed_sec=None if status == "reused" else time.perf_counter() - started,
                    )
                }
            },
            "llm_rounds": summarize_llm_rounds(artifact_dir),
        },
    )
    if status != "failed":
        write_task_report(
            artifact_dir,
            task_id=task_id,
            outputs={
                "translated_srt": str(out.with_name(f"{out.stem}-translated.srt")),
                **({"final_srt": str(out)} if out.exists() else {}),
            },
            run_metadata_path=metadata_path,
        )


def _generation_pin(knowledge: str, knowledge_root: str | Path | None) -> ContextManager[Any]:
    """Pin every knowledge read of this run to one revision (plan §2.5).

    research, the windows and the query rounds all read through the pin; the
    post-run knowledge update passes its own explicit ``working_rev`` per
    chunk, which overrides it. ``knowledge="none"`` never touches the base, so
    it never opens (or creates) the store either.
    """

    if knowledge == "none":
        return nullcontext()
    root = knowledge_root if knowledge_root is not None else DEFAULT_KNOWLEDGE_ROOT
    # Absorb pending rendered/ user edits BEFORE pinning (plan §11.3 / O5):
    # the run then reads the freshly harvested revision. Guarded (worktree
    # gate + write lock) and per-file isolated inside; never blocks the run.
    try:
        from .knowledge.node.edit import harvest_rendered_edits_at_run_start

        harvest_rendered_edits_at_run_start(root)
    except Exception as exc:
        current_reporter().warning("knowledge-rendered-harvest-failed", str(exc))
    return pinned_generation_rev(root)


def _task_slots(test_profile: bool) -> ContextManager[Any]:
    """This run's agent slot account (task-parallelism plan W4).

    A task whose routing can reach an agent target reserves its mandatory
    lane for the whole run; a pure-API or test-profile run books nothing.
    """

    from .run_context import default_task_slots

    return default_task_slots(test_profile=test_profile)


def _seed_transfer_from_context(context_path: Path, fast_kwargs: dict) -> None:
    """Read the preparation task's persisted <keep_entries> into the transfer seed."""

    try:
        saved_payload = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    raw_keep_entries = saved_payload.get("keep_entries")
    transfer_seed = [
        key
        for key in (raw_keep_entries or [])
        if isinstance(key, str) and key
    ]
    if transfer_seed:
        fast_kwargs["initial_transfer_keys"] = transfer_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run LLM-based ASR correction and Chinese subtitle translation."
    )
    parser.add_argument(
        "input",
        help="Path to *-stable.json from VAD-ASR; correction input is rendered from it.",
    )
    parser.add_argument("-o", "--output", help="Path to corrected translated SRT.")
    parser.add_argument(
        "--media",
        choices=list(MEDIA),
        default=None,
        help=(
            "What the correction window sees: text / +audio / +video, a "
            "ladder (video implies audio). text is also how a text-only "
            "strong model handles media material."
        ),
    )
    parser.add_argument(
        "--retrieval",
        choices=list(RETRIEVAL),
        default=None,
        help=(
            "none = no retrieval at all; local = the harness injection machinery "
            "(preparation fill, per-window request turn, local search agent); "
            "native = the model's own search tool, no harness search."
        ),
    )
    parser.add_argument(
        "--difficulty",
        choices=list(DIFFICULTY),
        default=None,
        help=(
            "Which prompt/thinking cell to use. quality and intermediate are "
            "for capable models (the shipped preset binds both to the capable "
            "group): quality runs pass 3 by default and turns the speaker judge "
            "on in the pipeline, intermediate does neither; both run the "
            "three-pass split when the speaker judge is on, else one pass. "
            "efficiency = the cheapest shape that still corrects and translates: "
            "the lite group, low thinking, one pass, only the factory default "
            "style (an explicit --style is an error), and no speaker "
            "judge (--speaker-judge is an error). It pins no other switch. Unset "
            "media/retrieval/difficulty/continuity/knowledge take the active "
            "preset's [llm.presets.<id>.defaults], then the built-in defaults."
        ),
    )
    parser.add_argument(
        "--continuity",
        choices=list(CONTINUITY),
        default=None,
        help=(
            "serial (default) keeps the chained inter-window context (advice "
            "ledger, entry transfer chain); parallel gives it up and "
            "dispatches correction windows concurrently for wall-clock "
            "(docs/llm_harness_behavior.md)."
        ),
    )
    parser.add_argument(
        "--parallel-windows",
        type=int,
        default=None,
        help=(
            "Concurrency cap for --continuity parallel (default: the active "
            "preset's parallel_windows, else 1 -- a preference, not a "
            "calibrated constant; see docs/manual/agent.md)."
        ),
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help=(
            "Scale k on the expected-output estimate k x c x csv_tokens; larger "
            "values plan smaller windows."
        ),
    )
    parser.add_argument(
        "--fast",
        choices=["auto", "on", "off"],
        default=None,
        help=(
            "Fast mode: the whole input as one correction task (no preparation "
            "task; the window asks for its own inputs). auto enables it when the "
            "whole input fits 0.8 x output_limit - 10k output and the opening "
            "message leaves the fill reserve; on errors out when it does not fit; "
            "off forces the normal flow."
        ),
    )
    parser.add_argument(
        "--audio",
        help=(
            "Path to original audio, not *-vocal.ogg. Required with --execute "
            "when a media switch uses audio: each window uploads its own "
            "mono-16k AAC clip (segment span plus padding) instead of the whole file."
        ),
    )
    parser.add_argument(
        "--video",
        help=(
            "Path to the source video (mm-high only; required with --execute). "
            "The correction round gets a low-res video+audio mp4 clip per "
            "window instead of the .aac; the query round stays audio-only."
        ),
    )
    parser.add_argument(
        "--extra-info",
        default=None,
        help="Extra user-provided info for research: source URL, content notes, requirements.",
    )
    parser.add_argument(
        "--extra-info-file",
        help="Path to a file with extra user-provided info for research.",
    )
    parser.add_argument(
        "--context-file",
        help="Existing research-context.json; skips the preparation task.",
    )
    parser.add_argument(
        "--research-only",
        action="store_true",
        help="Run only the preparation task, write research-context.json, then exit."
        " Like every run, it calls a model only with --execute; without it, only the plan and prompts.",
    )
    parser.add_argument(
        "--style",
        default=None,
        help="翻译风格条目名（可逗号分隔；同名时写 style/<名字>）；auto 让模型从库里选一套。"
             "不给则读配置 [llm] style，再不给就是 auto。没有用你的风格时注入出厂默认风格；"
             "difficulty=efficiency 只用出厂默认风格，配置不生效、点名报错",
    )
    parser.add_argument(
        "--style-mode",
        dest="style_mode",
        default=None,
        choices=("none", "read", "update"),
        help="拿这套风格做什么：none 不用 / read 只注入 / update 还把本次学到的写回去"
             "（需要 --refined-srt + --knowledge update）。不给则读配置 [llm] style_mode，默认 read",
    )
    parser.add_argument(
        "--extra-style",
        default=None,
        help="Extra translation style prompt injected into the correction system prompt.",
    )
    parser.add_argument(
        "--postprocess-profile",
        type=int,
        choices=SUPPORTED_POSTPROCESS_PROFILES,
        default=None,
        help=(
            "Final SRT postprocess profile: -1 semantic no-op re-render; "
            "0 t2s, overlap, duration, punctuation; 1 duration only; "
            "2 punctuation only; 3 t2s only; 4 overlap repair only."
        ),
    )
    parser.add_argument(
        "--task-summary",
        default=None,
        help="Task summary included in knowledge update prompts.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Stable task id used in retained artifacts and knowledge commits.",
    )
    parser.add_argument(
        "--task-artifact-dir",
        help="Retain selected LLM outputs and task metadata as JSONL in this directory.",
    )
    parser.add_argument(
        "--knowledge",
        choices=list(KNOWLEDGE_MODES),
        default=None,
        help=(
            "Knowledge switch. Default 'read': read and inject the base "
            "(indices, entries, style), collect nothing, write nothing back. "
            "'none': the knowledge base is not "
            "read or injected. 'update': read, have the windows and the "
            "preparation task emit task_update_feedback, and run the unified "
            "knowledge update after the task -- which a dry run never reaches, "
            "so it is accepted there and simply writes nothing."
        ),
    )
    parser.add_argument(
        "--refined-srt",
        help=(
            "User-refined SRT for the knowledge update (refined_aligned mode); "
            "only with --knowledge update."
        ),
    )
    parser.add_argument(
        "--knowledge-root",
        default=None,
        help="Root directory of the local Markdown knowledge base (embedded git repo).",
    )
    parser.add_argument(
        "--prompt-dir",
        help="Write plan.json plus Chinese research/correction prompts to this directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured LLM APIs. Default only writes/prints the plan.",
    )
    parser.add_argument(
        "--test-profile",
        action="store_true",
        default=None,
        help="Use gemini-3.5-flash-lite for every role.",
    )
    parser.add_argument(
        "--proofread",
        choices=("auto", "on", "off"),
        default=None,
        help="Proofread the finished subtitles in a separate pass, one call per correction window "
             "(default: auto = at the quality difficulty, whichever way the task runs).",
    )
    parser.add_argument(
        "--speaker-judge",
        choices=("off", "text", "audio"),
        default=None,
        help="Let a model re-check, one correction window at a time, which speaker track each line "
             "belongs to, reading the rows (text) or also hearing them (audio); needs speaker labels "
             "on the stable (the pipeline's --diarize judge-text / judge-audio). Results go to "
             "<stem>-speaker-judge.json and the annotated CSV's spk column, and the task runs the "
             "three-pass split, so it needs a correction cell with a grouped prompt set: refused at "
             "efficiency, except on a resume the judge was recorded for. Off by default.",
    )
    parser.add_argument(
        "--split-length-scale",
        type=float,
        default=None,
        help="How long a merged subtitle may get: scales the merge thresholds the correction prompt states and "
             "its validation holds to (0.6-1.6; below 1 = shorter). Default: the style's length tier "
             "([字幕长度] 偏短 0.8 / 标准 1.0 / 偏长 1.25), then [segmentation] length_scale in config.toml, then "
             "1.0. A resumed task keeps the ratio it recorded.",
    )
    parser.add_argument(
        "--max-retries-per-window",
        type=int,
        default=None,
        help=(
            "Tier 1 of the per-window retry budget: repair retries within "
            "one session chain -- each carries the previous output and the "
            "validation errors, and an agent backend resumes the same "
            "conversation. No longer the total-call cap: total calls per "
            "window = (this+1) x (--max-replacements-per-window+1)."
        ),
    )
    parser.add_argument(
        "--max-replacements-per-window",
        type=int,
        default=None,
        help=(
            "Tier 2 of the per-window retry budget: how many times a window "
            "whose repair chain is spent is handed to a fresh session -- "
            "repair context dropped, an agent starts a new conversation, a "
            "stateless endpoint throws blind."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=None,
        help=(
            "Disable LLM session and correction-window checkpoint reads/writes "
            "(default: resume validated sessions and completed windows in the "
            "task artifact directory)."
        ),
    )
    from .routing.model_routes import add_routing_arguments

    add_routing_arguments(parser)
    return parser.parse_args()


def _load_extra_info(args: argparse.Namespace) -> str:
    parts = [args.extra_info.strip()] if (args.extra_info or "").strip() else []
    if args.extra_info_file:
        parts.append(
            Path(args.extra_info_file).expanduser().read_text(encoding="utf-8").strip()
        )
    return "\n".join(part for part in parts if part)


def main() -> int:
    """The module CLI: binds the terminal renderer, then runs.

    The binding lives here rather than in the ``__main__`` guard so that any
    other caller of ``main()`` gets it too. Without it every
    ``current_reporter().warning(...)`` under this entry point is handed to the
    silent thread-local default and dropped -- measured 2026-09-03 on an
    unparseable hand edit in ``rendered/``: the run exited 0 and said nothing.
    The renderer writes to stderr, so redirecting stdout to capture a dry run's
    prompt JSON still shows the warnings that describe producing it.
    """

    with reporting_to(terminal_reporter()):
        return _main()


def _main() -> int:
    """Parse, build one request, run it, say what happened.

    Every refusal is the task's own (``CorrectionRequestError``, exit 2); this
    function only presents. The plan -- the prompts the run would send -- is
    shown through the request's ``on_plan`` hook, before anything runs.
    """

    args = parse_args()
    # Installed unconditionally (empty → clears) BEFORE any routing consumer:
    # planning envelopes, preflight, research and the correction drivers all
    # construct their clients off the memoized route loader.
    from .routing.model_routes import install_runtime_routing

    install_runtime_routing(args)
    try:
        switches = resolve_switches(
            {
                "media": args.media,
                "retrieval": args.retrieval,
                "difficulty": args.difficulty,
                "continuity": args.continuity,
                "knowledge": args.knowledge,
                "parallel_windows": args.parallel_windows,
            }
        )
        profile = resolve_profile(
            switches.media,
            switches.retrieval,
            switches.difficulty,
            switches.continuity,
            **({"output_scale": args.output_scale} if args.output_scale is not None else {}),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    def show_plan(artifacts: Mapping[str, Any]) -> None:
        # The plan is this CLI's product in a dry run: the prompts go to
        # --prompt-dir, or to stdout as JSON (minus the message bodies).
        if args.prompt_dir:
            write_prompt_artifacts(dict(artifacts), args.prompt_dir)
            print(f"Wrote prompt artifacts: {Path(args.prompt_dir).expanduser().resolve()}")
            return
        print(
            json.dumps(
                {
                    key: value
                    for key, value in artifacts.items()
                    if key not in {"research_messages", "correction_messages", "speaker_judge_messages"}
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    def path(value: str | None) -> Path | None:
        return Path(value) if value else None

    # argparse passes None for every flag the user did not give, and None is
    # dropped here: an unset option is answered by `CorrectionRequest`'s own
    # default, never by a second copy on this side (README_DEV 开发原则).
    given = {
        "output_path": path(args.output),
        "audio_path": path(args.audio),
        "video_path": path(args.video),
        "fast": args.fast,
        "extra_info": _load_extra_info(args) or None,
        "extra_style": args.extra_style,
        "style": args.style,
        "style_mode": args.style_mode,
        "knowledge": switches.knowledge,
        "switch_sources": switches.sources,
        "knowledge_root": args.knowledge_root,
        "refined_srt": path(args.refined_srt),
        "task_id": args.task_id,
        "task_summary": args.task_summary,
        "task_artifact_dir": path(args.task_artifact_dir),
        "context_file": path(args.context_file),
        "parallel_windows": switches.parallel_windows,
        "max_retries_per_window": args.max_retries_per_window,
        "max_replacements_per_window": args.max_replacements_per_window,
        "postprocess_profile": args.postprocess_profile,
        "speaker_judge": args.speaker_judge,
        "split_length_scale": args.split_length_scale,
        "proofread": args.proofread,
        "resume": args.resume,
        "test_profile": args.test_profile,
    }
    # `--research-only` says how far to go, `--execute` alone whether a model is
    # called. Its dry-run is a plan, held to the conflicts research would meet.
    if args.research_only and not args.execute:
        try:
            if args.context_file:
                raise ValueError("--research-only conflicts with --context-file")
            given["fast"] = fast_mode_for_research_only(args.fast, research_only=True)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
    request = CorrectionRequest(
        stable_json=Path(args.input),
        mode=("research" if args.research_only else "execute") if args.execute else "plan",
        profile=profile,
        on_plan=show_plan,
        **{name: value for name, value in given.items() if value is not None},
    )
    try:
        outcome = run_correction_task(request)
    except CorrectionRequestError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for line in _outcome_lines(outcome, request):
        print(line)
    return 0


def _outcome_lines(outcome: CorrectionOutcome, request: CorrectionRequest) -> list[str]:
    lines: list[str] = []
    if outcome.research_context is not None and request.context_file is None:
        verb = "Reused" if outcome.research_reused else "Wrote"
        lines.append(f"{verb} research context: {outcome.research_context}")
    if outcome.result is not None:
        lines.append(f"{'Reused' if outcome.result_reused else 'Wrote'} {outcome.result}")
    if outcome.knowledge_update is not None:
        update = outcome.knowledge_update
        lines.append(
            f"Knowledge update ({update['mode']}): {len(update['chunks'])} chunk(s);"
            f" ledger: {update['ledger_path']}"
        )
    if outcome.artifact_dir is not None and request.mode != "plan":
        report = outcome.artifact_dir / "task-report.md"
        if report.exists():
            lines.append(f"Task report: {report}")
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
