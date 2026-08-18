"""Unified knowledge update: one entry point for both evidence modes.

Replaces the old ``task_auto`` / ``post_task`` split (docs/
knowledge.md). Evidence is structured — per-window CSV packs
plus aggregated ``task_update_feedback`` — instead of a raw artifact dump;
providing ``--refined-srt`` switches from the ``artifacts_only`` prompt to the
``refined_aligned`` one (which alone may write the common-mistake ledger).

Multi-chunk tasks (CSV text over the 100k budget) apply sequentially: each
chunk's proposals land in the knowledge base before the next chunk's entry
excerpt is rendered, and an apply ledger (``knowledge-update-chunks.jsonl``
in the task artifact dir) makes reruns skip already-applied chunks instead of
double-appending history.
"""

from __future__ import annotations

import argparse
import hashlib
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping

from finesub.paths import is_linked_worktree

from ..client import (
    GeminiPromptBlockedError,
    RoleClient,
    extract_finish_reason,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
from .mistakes import apply_mistake_proposals, parse_mistake_proposals
from ..routing.config import LLMRole, SESSION_OUTPUT_MAX_TOKENS, planning_limits_for
from ..content_filter import (
    ContentFilterExhaustedError,
    run_injection_ladder,
    split_rendered_search_block,
)
from ..exchange_log import ExchangeLogger, messages_to_text
from ..exchange_metadata import llm_exchange_metadata
from .base import (
    DEFAULT_KNOWLEDGE_ROOT,
    GIT_MISSING_MESSAGE,
    append_task_artifact,
    apply_knowledge_proposals,
    commit_knowledge,
    ensure_knowledge_git,
    git_is_available,
    knowledge_git_head,
    knowledge_git_head_message,
    knowledge_git_is_clean,
    knowledge_write_lock,
    load_index_text,
    parse_knowledge_proposals,
)
from .entries import render_kb_entry_excerpt, select_kb_entries
from .materials import (
    KNOWLEDGE_CSV_TOKEN_BUDGET,
    KnowledgeChunk,
    KnowledgeMaterials,
    MODE_REFINED_ALIGNED,
    build_knowledge_materials,
)
from ..prompts import PROMPT_VERSION, build_knowledge_update_messages
from ..token_budget import default_token_counter, TokenCounter


CHUNK_LEDGER_FILENAME = "knowledge-update-chunks.jsonl"
# Align with research._call_and_parse: 1 retry → 2 attempts total.
DEFAULT_KNOWLEDGE_UPDATE_PARSE_RETRIES = 1


def _validate_knowledge_update_jsonl(text: str, *, refined: bool) -> None:
    """Raise ValueError if knowledge / mistake proposal JSONL is syntactically invalid."""

    parse_knowledge_proposals(text)
    if refined:
        parse_mistake_proposals(text)


# ---------------------------------------------------------------------------
# Path derivation (mirrors pipeline.default_pipeline_paths)


RESEARCH_CONTEXT_SUFFIX = "-research-context.json"


def research_context_filename(stem: str) -> str:
    return f"{stem}{RESEARCH_CONTEXT_SUFFIX}"


def research_context_in_artifact_dir(artifact_dir: str | Path, stem: str) -> Path:
    """Canonical research-context path: under the task artifact directory."""

    return Path(artifact_dir).expanduser().resolve() / research_context_filename(stem)


def ensure_research_context_path(
    *,
    artifact_dir: str | Path,
    stem: str,
    run_dir: str | Path | None = None,
) -> Path:
    """Return the artifact-dir path; migrate a legacy run-root sibling if present.

    Legacy layout kept ``<run_dir>/<stem>-research-context.json`` next to the
    SRT. New layout writes under ``artifact_dir``. If only the legacy file
    exists, it is moved once so subsequent reads hit the canonical location.
    """

    preferred = research_context_in_artifact_dir(artifact_dir, stem)
    if preferred.exists():
        return preferred
    legacy_root = Path(run_dir).expanduser().resolve() if run_dir else preferred.parent.parent
    legacy = legacy_root / research_context_filename(stem)
    if legacy.exists() and legacy.resolve() != preferred.resolve():
        preferred.parent.mkdir(parents=True, exist_ok=True)
        legacy.replace(preferred)
        print(f"Migrated research context: {legacy} -> {preferred}")
        return preferred
    return preferred


def derive_task_paths(final_srt: str | Path) -> Dict[str, Path]:
    """Sibling artifact paths from the standard final SRT path."""

    srt_path = Path(final_srt).expanduser().resolve()
    base = srt_path.with_suffix("")
    artifact_dir = base.with_name(f"{base.name}.llm-artifacts")
    return {
        "final_srt": srt_path,
        "stable_json": base.with_name(f"{base.name}-stable.json"),
        "annotated_csv": base.with_name(f"{base.name}-annotated.csv"),
        "artifact_dir": artifact_dir,
        "research_context": research_context_in_artifact_dir(artifact_dir, base.name),
    }


# ---------------------------------------------------------------------------
# Apply ledger (chunk idempotency)


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _task_fingerprint(
    *,
    mode: str,
    task_summary: str,
    refined_text: str,
    csv_token_budget: int,
    difficulty: str = "quality",
    execution_identity_override: Mapping[str, Any] | None = None,
) -> str:
    """Audit identity for the run that produced ledger rows.

    Reuse does not compare this value; applied material hashes are the
    idempotency boundary.
    """

    from ..routing.execution_policy import execution_identity

    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "mode": mode,
            "task_summary": task_summary,
            "refined_sha": _sha256(refined_text) if refined_text else "",
            "csv_token_budget": csv_token_budget,
            # Kept for provenance only; it is not an invalidation key.
            "difficulty": difficulty,
            "execution_identity": dict(
                execution_identity_override or execution_identity()
            ),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)


def _chunk_input_hash(chunk: KnowledgeChunk, materials: KnowledgeMaterials) -> str:
    """Chunk identity for the ledger: stable material text only.

    Excludes the KB-entry excerpt on purpose — it changes as earlier chunks
    apply, and a rerun must still recognize an already-applied chunk.
    """

    payload = json.dumps(
        {
            "packs": chunk.packs_text(),
            "general_context": materials.general_context,
            "research_feedback": materials.feedback.research_slice_text(),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)


def _load_chunk_ledger(path: Path) -> Dict[str, Dict[str, Any]]:
    """Applied-chunk records keyed by structural material hash.

    The task fingerprint remains in each row as audit metadata, but is not a
    reuse gate: changing model, prompt, difficulty, or chunk sizing must never
    apply already-committed knowledge a second time.
    """

    if not path.exists():
        return {}
    records: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        status = str(record.get("status") or "applied")
        if status not in {"applied", "recovered_after_commit"}:
            continue
        input_hash = record.get("input_hash")
        if isinstance(input_hash, str) and input_hash:
            records[input_hash] = record
    return records


def _load_pending_chunk_intents(path: Path) -> Dict[str, Dict[str, Any]]:
    """Latest uncompleted write-ahead records, keyed by chunk input hash."""

    pending: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return pending
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, dict):
            continue
        input_hash = record.get("input_hash")
        if not isinstance(input_hash, str) or not input_hash:
            continue
        status = str(record.get("status") or "applied")
        if status == "intent":
            pending[input_hash] = record
        elif status in {"applied", "recovered_after_commit"}:
            pending.pop(input_hash, None)
    return pending


def _append_chunk_ledger(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _chunk_material_hashes(
    chunk: KnowledgeChunk, materials: KnowledgeMaterials
) -> List[str]:
    """Per-window identities survive a later change in chunk boundaries."""

    hashes: List[str] = []
    for window in chunk.windows:
        payload = json.dumps(
            {
                "mode": materials.mode,
                "window": window.pack_text(),
                "general_context": materials.general_context,
                "research_feedback": materials.feedback.research_slice_text(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        hashes.append(_sha256(payload))
    return hashes


def _applied_entry_pairs(report: Dict[str, Any]) -> List[List[str]]:
    return [
        [str(record.get("category", "")), str(record.get("entry", ""))]
        for record in report.get("applied", [])
        if record.get("entry")
    ]


# ---------------------------------------------------------------------------
# Runner


def _aggregated_feedback_text(materials: KnowledgeMaterials) -> str:
    uncertainties = materials.feedback.merged_uncertainties()
    corrections = materials.feedback.merged_asr_corrections()
    if not uncertainties and not corrections:
        return ""
    return json.dumps(
        {"uncertainties": uncertainties, "asr_corrections": corrections},
        ensure_ascii=False,
    )


def _chunk_window_range(chunk: KnowledgeChunk) -> str:
    ids = chunk.window_ids
    if len(ids) == 1:
        return ids[0]
    return f"{ids[0]}–{ids[-1]}"


def _split_chunk(chunk: KnowledgeChunk) -> List[KnowledgeChunk]:
    half = len(chunk.windows) // 2
    first, second = chunk.windows[:half], chunk.windows[half:]
    return [
        KnowledgeChunk(index=chunk.index, windows=first, csv_tokens=0),
        KnowledgeChunk(index=chunk.index, windows=second, csv_tokens=0),
    ]


def _worktree_writes_allowed() -> bool:
    configured = os.environ.get("FINESUB_KNOWLEDGE_WRITE", "")
    return configured.strip().lower() in {"1", "true", "yes", "on"}


def run_knowledge_update(
    *,
    final_srt: str | Path,
    stable_json: str | Path | None = None,
    annotated_csv: str | Path | None = None,
    research_context: str | Path | None = None,
    artifact_dir: str | Path | None = None,
    refined_srt: str | Path | None = None,
    task_id: str = "",
    task_summary: str = "",
    knowledge_root: str | Path = DEFAULT_KNOWLEDGE_ROOT,
    test_profile: bool = False,
    execute: bool = True,
    apply: bool = True,
    prompt_dir: str | Path | None = None,
    resume: bool = True,
    csv_token_budget: int = KNOWLEDGE_CSV_TOKEN_BUDGET,
    token_counter: TokenCounter | None = None,
    client: RoleClient | None = None,
    # The run's difficulty selects the knowledge cell. Standalone invocations
    # (the module CLI) keep the top tier; the correction pipeline passes its
    # own so a preset binding a cheaper group at intermediate takes effect.
    difficulty: str = "quality",
) -> Dict[str, Any]:
    """Run the unified knowledge update for one finished correction task.

    Paths default to siblings of ``final_srt`` (``<stem>-stable.json`` etc.);
    ``research-context.json`` lives under the task artifact directory (legacy
    run-root siblings are migrated on first touch).
    Without ``execute`` only the per-chunk prompts are written/printed; with
    ``execute`` but not ``apply`` proposals are generated and retained without
    touching the knowledge base.
    """

    if execute and apply and is_linked_worktree() and not _worktree_writes_allowed():
        # Worktrees share the main checkout's knowledge base on purpose, which
        # also means a throwaway experiment in one would commit into the real
        # thing. Ask first: set FINESUB_KNOWLEDGE_WRITE=1 for a run that is
        # meant to update it.
        message = (
            "Warning: 当前在 git worktree 中运行，已跳过知识库更新以免写入主仓的知识库；"
            "确需更新请设 FINESUB_KNOWLEDGE_WRITE=1 后重跑（ledger 未推进）。"
        )
        print(message, file=sys.stderr)
        return {
            "mode": "",
            "task_fingerprint": "",
            "chunks": [],
            "warnings": [message],
            "ledger_path": "",
            "skipped": "worktree_readonly",
        }

    if execute and apply and not git_is_available():
        # Nothing in this project installs git, so a machine without it is an
        # ordinary state -- and the knowledge base is an embedded git repo whose
        # auto-apply commits there. Refuse before spending any LLM quota on a
        # proposal that could not be applied, and leave the chunk ledger
        # untouched so a later run with git present redoes the whole update.
        message = (
            f"Warning: {GIT_MISSING_MESSAGE} 本次跳过知识库更新；"
            f"装好 git 后重跑即可（ledger 未推进）。"
        )
        print(message, file=sys.stderr)
        return {
            "mode": "",
            "task_fingerprint": "",
            "chunks": [],
            "warnings": [message],
            "ledger_path": "",
            "skipped": "git_unavailable",
        }

    paths = derive_task_paths(final_srt)
    stable_json = Path(stable_json).expanduser() if stable_json else paths["stable_json"]
    annotated_csv = (
        Path(annotated_csv).expanduser() if annotated_csv else paths["annotated_csv"]
    )
    artifact_path = (
        Path(artifact_dir).expanduser().resolve()
        if artifact_dir
        else paths["artifact_dir"]
    )
    research_context = (
        Path(research_context).expanduser()
        if research_context
        else ensure_research_context_path(
            artifact_dir=artifact_path,
            stem=paths["final_srt"].stem,
            run_dir=paths["final_srt"].parent,
        )
    )
    llm_client = client
    if execute and llm_client is None:
        # Post-task update uses GENERAL_CAPABLE (3.5 → 3.6 → 3.5-lite).
        # Create it before the resume fingerprint so identity comes from the
        # client that will actually execute, but keep prompt-only mode offline.
        llm_client = RoleClient(test_profile=False)
    counter = token_counter or default_token_counter(
        execution_settings=getattr(llm_client, "execution_settings", None)
    )
    # Chunks are sized against the *bound* knowledge group, not the packaged
    # ceiling (2026-08-12): a chunk built at 194k for a group whose smallest
    # member holds 32k passes here and is then skipped at dispatch as
    # ``input_limit`` -- the model the user bound never answers, and a
    # single-member group fails outright.
    knowledge_limits = planning_limits_for("knowledge", difficulty)

    materials = build_knowledge_materials(
        stable_json=stable_json,
        annotated_csv=annotated_csv,
        final_srt=paths["final_srt"],
        research_context=research_context,
        artifact_dirs=[artifact_path],
        refined_srt=refined_srt,
        count_tokens=counter.count_text,
        csv_token_budget=csv_token_budget,
    )
    for warning in materials.warnings:
        print(f"Warning: knowledge update: {warning}", file=sys.stderr)

    refined = materials.mode == MODE_REFINED_ALIGNED
    refined_text = (
        Path(refined_srt).expanduser().read_text(encoding="utf-8") if refined_srt else ""
    )
    task_fingerprint = _task_fingerprint(
        mode=materials.mode,
        task_summary=task_summary,
        refined_text=refined_text,
        csv_token_budget=csv_token_budget,
        difficulty=difficulty,
        execution_identity_override=getattr(
            llm_client, "execution_identity", None
        ),
    )
    ledger_path = artifact_path / CHUNK_LEDGER_FILENAME
    ledger = _load_chunk_ledger(ledger_path) if resume else {}
    pending_intents = (
        _load_pending_chunk_intents(ledger_path) if resume else {}
    )
    applied_material_hashes = {
        str(material_hash)
        for record in ledger.values()
        for material_hash in (record.get("material_hashes") or [])
        if isinstance(material_hash, str) and material_hash
    }
    knowledge_repo_prepared = False

    research_hints = (
        list(materials.feedback.research_feedback.hints)
        if materials.feedback.research_feedback
        else []
    )
    aggregated_feedback = _aggregated_feedback_text(materials)
    # (category, key) pairs already written by earlier chunks of this task —
    # annotated in later chunks' entry excerpts so the model does not rewrite
    # the same section every chunk.
    applied_entries: set[tuple[str, str]] = set()
    for record in ledger.values():
        for category, entry in record.get("applied_entries", []):
            applied_entries.add((category, entry))

    exchange_logger = ExchangeLogger.for_task_artifact_dir(
        artifact_path if execute else None
    )
    prompt_dir_path = Path(prompt_dir).expanduser().resolve() if prompt_dir else None
    results: List[Dict[str, Any]] = []
    pending: List[KnowledgeChunk] = list(materials.chunks)
    multi_chunk = len(pending) > 1
    position = 0
    # Taken once, on the first chunk that actually applies, and held until the
    # run finishes: the knowledge base is one embedded git repository that the
    # desktop app, the CLI and a checkout can all reach at the same time, and a
    # second process committing between our apply and our commit would fold our
    # uncommitted files into its own commit.
    #
    # Released in a `finally`, not merely after the loop. The old reasoning was
    # that a raising run ends the process anyway -- but per-item failure
    # isolation means it does not: `batch` and `reference_ingest` carry on with
    # the next task in the *same* process, and both msvcrt and flock locks are
    # per handle, so a leaked one blocks us exactly as a stranger's would. Every
    # later task then waited out the full 90s timeout and skipped applying its
    # knowledge, reporting that another FineSub process was writing -- this one.
    knowledge_lock = ExitStack()
    try:
        while position < len(pending):
            chunk = pending[position]
            chunk_no = position + 1
            input_hash = _chunk_input_hash(chunk, materials)
            material_hashes = _chunk_material_hashes(chunk, materials)
            cached = ledger.get(input_hash)
            covered_by_prior_chunks = bool(material_hashes) and all(
                material_hash in applied_material_hashes
                for material_hash in material_hashes
            )
            pending_intent = pending_intents.get(input_hash)
            if cached is None and pending_intent is not None:
                head_before = str(pending_intent.get("git_head_before") or "")
                proposal_hash = str(pending_intent.get("proposal_hash") or "")
                head_now = knowledge_git_head(knowledge_root)
                commit_message = knowledge_git_head_message(knowledge_root)
                if (
                    proposal_hash
                    and head_now != head_before
                    and f"proposal: {proposal_hash}" in commit_message
                ):
                    recovered = {
                        "status": "recovered_after_commit",
                        "task_fingerprint": task_fingerprint,
                        "chunk_index": chunk_no,
                        "window_ids": list(chunk.window_ids),
                        "input_hash": input_hash,
                        "material_hashes": material_hashes,
                        "git_head_after": head_now,
                        "knowledge_report": None,
                        "mistake_report": None,
                        "applied_entries": [],
                    }
                    _append_chunk_ledger(ledger_path, recovered)
                    ledger[input_hash] = recovered
                    applied_material_hashes.update(material_hashes)
                    cached = recovered
            if cached is not None or covered_by_prior_chunks:
                results.append(
                    {
                        "chunk": chunk_no,
                        "window_ids": list(chunk.window_ids),
                        "skipped": "already_applied",
                        "knowledge_report": (
                            cached.get("knowledge_report") if cached else None
                        ),
                        "mistake_report": (
                            cached.get("mistake_report") if cached else None
                        ),
                    }
                )
                print(
                    f"Knowledge update chunk {chunk_no} already applied; skipping "
                    f"(ledger: {ledger_path})."
                )
                position += 1
                continue

            # §1.7: chunk entry scope = this chunk's window hints + ALL research
            # hints; the excerpt is re-rendered per chunk so it sees prior applies.
            window_hints = [
                hint
                for window in chunk.windows
                for hint in (
                    materials.feedback.window_feedback.get(window.chunk_id).hints
                    if materials.feedback.window_feedback.get(window.chunk_id)
                    else ()
                )
            ]
            selections = select_kb_entries(
                window_hints,
                knowledge_root=knowledge_root,
                research_origins=research_hints,
                applied_entries=applied_entries,
            )
            kb_entries_block = render_kb_entry_excerpt(
                selections, knowledge_root, count_tokens=counter.count_text
            )
            # v17: the update model must see the live index before proposing
            # create_entry; reload per chunk so entries created/renamed by the
            # previous chunk's apply are visible.
            streamer_index_text = load_index_text(knowledge_root, "streamer")
            common_index_text = load_index_text(knowledge_root, "common")
            messages = build_knowledge_update_messages(
                refined=refined,
                task_summary=task_summary,
                window_packs=chunk.packs_text(),
                general_context=materials.general_context,
                research_feedback=materials.feedback.research_slice_text(),
                aggregated_feedback=aggregated_feedback,
                kb_entries=kb_entries_block.text,
                streamer_index=streamer_index_text,
                common_index=common_index_text,
                chunk_index=chunk_no,
                multi_chunk=multi_chunk,
                window_range=_chunk_window_range(chunk),
            )
            prompt_tokens = counter.count_texts(
                str(message.get("content", "")) for message in messages
            )
            if prompt_tokens > knowledge_limits.prompt_input_limit:
                if len(chunk.windows) > 1:
                    pending[position : position + 1] = _split_chunk(chunk)
                    multi_chunk = True
                    print(
                        f"Knowledge update chunk {chunk_no} (~{prompt_tokens} tokens) "
                        "exceeds the prompt input limit; splitting on window boundary.",
                        file=sys.stderr,
                    )
                    continue
                raise RuntimeError(
                    f"Knowledge update chunk {chunk_no} is a single window but its "
                    f"prompt (~{prompt_tokens} tokens) exceeds the input limit "
                    f"{knowledge_limits.prompt_input_limit}."
                )

            if prompt_dir_path is not None:
                prompt_dir_path.mkdir(parents=True, exist_ok=True)
                prompt_path = prompt_dir_path / f"knowledge-update-chunk{chunk_no:02d}.txt"
                prompt_path.write_text(messages_to_text(messages), encoding="utf-8")
            if not execute:
                if prompt_dir_path is None:
                    print(messages_to_text(messages))
                results.append(
                    {
                        "chunk": chunk_no,
                        "window_ids": list(chunk.window_ids),
                        "prompt_tokens": prompt_tokens,
                        "executed": False,
                    }
                )
                position += 1
                continue

            assert llm_client is not None

            input_components = {
                "prompt_tokens_estimate": prompt_tokens,
                "window_packs_tokens": counter.count_text(chunk.packs_text()),
                "kb_entries_tokens": kb_entries_block.tokens,
                "max_output_tokens": SESSION_OUTPUT_MAX_TOKENS,
            }
            last_parse_error: Exception | None = None
            result = None
            finish_reason = None
            for attempt in range(DEFAULT_KNOWLEDGE_UPDATE_PARSE_RETRIES + 1):
                session = f"knowledge-update-chunk{chunk_no:02d}-attempt{attempt}"
                sampling = validation_retry_sampling_kwargs(attempt)
                last_call_route_decision: Dict[str, Any] = {}
                last_call_api_attempts: List[Mapping[str, Any]] = []
                last_call_execution_attempts: List[Mapping[str, Any]] = []

                def _knowledge_call(_injection: str = "", _sampling=sampling):
                    call_result = llm_client.complete(
                        LLMRole.GENERAL_CAPABLE,
                        messages,
                        max_tokens=SESSION_OUTPUT_MAX_TOKENS,
                        task_group="knowledge",
                        difficulty=difficulty,
                        **_sampling,
                    )
                    last_call_route_decision.update(call_result.route_decision)
                    last_call_api_attempts.extend(call_result.api_attempts)
                    last_call_execution_attempts.extend(
                        call_result.execution_attempts
                    )
                    if is_prompt_blocked(call_result.content, call_result.raw_response):
                        raise GeminiPromptBlockedError(
                            f"Knowledge update chunk {chunk_no} prompt was blocked by "
                            "the content filter"
                        )
                    return call_result

                try:
                    # Knowledge-update inputs are mostly task materials (not droppable
                    # web-retrieval units) — plain retry once, then a clear error.
                    knowledge_outcome = run_injection_ladder(
                        block=split_rendered_search_block(""),
                        call=_knowledge_call,
                        stage=f"knowledge_update_chunk_{chunk_no}",
                        blocked_exception=GeminiPromptBlockedError,
                        task_artifact_dir=artifact_path,
                        task_id=task_id,
                        plain_retry=True,
                    )
                except ContentFilterExhaustedError as exc:
                    append_task_artifact(
                        artifact_path,
                        kind="knowledge_update_call_error",
                        task_id=task_id,
                        payload={
                            "session": session,
                            "chunk": chunk_no,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "api_attempts": list(last_call_api_attempts),
                            "execution_attempts": list(last_call_execution_attempts),
                            "route_decision": dict(last_call_route_decision),
                        },
                    )
                    raise RuntimeError(
                        f"Knowledge update chunk {chunk_no}: prompt still blocked by "
                        "the content filter after a plain retry; the task materials "
                        "themselves likely trigger the filter."
                    ) from exc
                except Exception as exc:
                    append_task_artifact(
                        artifact_path,
                        kind="knowledge_update_call_error",
                        task_id=task_id,
                        payload={
                            "session": session,
                            "chunk": chunk_no,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "api_attempts": list(
                                getattr(exc, "_harness_api_attempts", []) or []
                            ),
                            "execution_attempts": list(
                                getattr(exc, "_harness_execution_attempts", []) or []
                            ),
                            "route_decision": dict(
                                getattr(exc, "_harness_route_decision", {}) or {}
                            ),
                        },
                    )
                    raise
                result = knowledge_outcome.result
                finish_reason = extract_finish_reason(result.raw_response)
                parse_error = ""
                try:
                    _validate_knowledge_update_jsonl(result.content, refined=refined)
                except (ValueError, json.JSONDecodeError) as exc:
                    last_parse_error = exc
                    parse_error = str(exc)
                append_task_artifact(
                    artifact_path,
                    kind="knowledge_update_response",
                    task_id=task_id,
                    payload={
                        "session": session,
                        "mode": materials.mode,
                        "chunk": chunk_no,
                        "attempt": attempt,
                        "window_ids": list(chunk.window_ids),
                        "input_hash": input_hash,
                        "model": result.model,
                        "fallback_used": result.fallback_used,
                        "usage": extract_token_distribution(result.raw_response),
                        "api_attempts": list(result.api_attempts),
                        "execution_attempts": list(result.execution_attempts),
                        "route_decision": dict(result.route_decision),
                        "input_components": input_components,
                        "finish_reason": finish_reason,
                        "parse_error": parse_error,
                        "injected_entries": [s.to_dict() for s in selections],
                        "entry_render_report": kb_entries_block.report(),
                        "response_content": result.content,
                    },
                )
                if exchange_logger:
                    exchange_logger.log(
                        session,
                        messages=messages,
                        response_text=result.content,
                        metadata=llm_exchange_metadata(
                            result,
                            session=session,
                            input_components=input_components,
                            mode=materials.mode,
                            chunk=chunk_no,
                            attempt=attempt,
                            finish_reason=finish_reason,
                            **({"parse_error": parse_error} if parse_error else {}),
                        ),
                    )
                if not parse_error:
                    break
            else:
                raise RuntimeError(
                    f"Knowledge update chunk {chunk_no} output could not be parsed "
                    f"after {DEFAULT_KNOWLEDGE_UPDATE_PARSE_RETRIES + 1} attempts: "
                    f"{last_parse_error}"
                )
            chunk_result: Dict[str, Any] = {
                "chunk": chunk_no,
                "window_ids": list(chunk.window_ids),
                "prompt_tokens": prompt_tokens,
                "executed": True,
                "proposal_text": result.content,
            }
            if apply and not knowledge_repo_prepared:
                if not knowledge_lock.enter_context(
                    knowledge_write_lock(knowledge_root)
                ):
                    print(
                        "Warning: 另一个 FineSub 进程正在写知识库，本次跳过自动应用；"
                        "提案已保留，稍后重跑会重做（ledger 未推进）。",
                        file=sys.stderr,
                    )
                    apply = False
                elif ensure_knowledge_git(
                    knowledge_root,
                    snapshot_dirty=True,
                    task_id=task_id,
                ):
                    knowledge_repo_prepared = True
                else:
                    # ensure_knowledge_git already said why (dirty tree, wrong
                    # branch, git unusable). Stop applying instead of aborting the
                    # task: the subtitle is the product, the knowledge base is a
                    # by-product, and failing the former over the latter helps
                    # nobody. Proposals are kept and the ledger is left where it
                    # is, so a later run redoes these chunks once the repository
                    # is back in order.
                    print(
                        "Warning: 知识库仓库不可用，跳过本次自动应用；"
                        "提案已保留，修好后重跑会重做（ledger 未推进）。",
                        file=sys.stderr,
                    )
                    apply = False
            if apply:
                source = f"llm.knowledge_update:{materials.mode}:chunk{chunk_no}"
                apply_task_id = f"{task_id or 'manual'}#chunk{chunk_no}" if multi_chunk else (
                    task_id or "manual"
                )
                # edit_lines is only valid against entries whose body was rendered
                # in full (line-numbered, untruncated) into THIS chunk's prompt.
                fully_rendered = set(kb_entries_block.included)
                line_editable = {
                    (selection.category, selection.key)
                    for selection in selections
                    if selection.exists
                    and f"{selection.category}/{selection.key}" in fully_rendered
                }
                proposal_hash = _sha256(result.content)
                _append_chunk_ledger(
                    ledger_path,
                    {
                        "status": "intent",
                        "task_fingerprint": task_fingerprint,
                        "chunk_index": chunk_no,
                        "window_ids": list(chunk.window_ids),
                        "input_hash": input_hash,
                        "material_hashes": material_hashes,
                        "git_head_before": knowledge_git_head(knowledge_root),
                        "proposal_hash": proposal_hash,
                    },
                )
                knowledge_report = apply_knowledge_proposals(
                    result.content,
                    knowledge_root=knowledge_root,
                    task_id=apply_task_id,
                    source=source,
                    line_editable=line_editable,
                    commit=False,
                ).to_dict()
                mistake_report = None
                if refined:
                    # artifacts_only never applies mistakes (design F/G): the
                    # prompt does not define the block and any stray one is ignored.
                    # 精选 is curated manually, so set_featured is refused here too.
                    # The chunk's material text backs the anti-fabrication check on
                    # add_mistake.wrong (audited runs kept inventing plausible
                    # mistranslations that never occurred).
                    mistake_report = apply_mistake_proposals(
                        result.content,
                        knowledge_root=knowledge_root,
                        task_id=apply_task_id,
                        source=source,
                        allow_featured=False,
                        evidence_text=chunk.packs_text(),
                        commit=False,
                    ).to_dict()
                changed = bool(knowledge_report.get("applied")) or bool(
                    mistake_report and mistake_report.get("applied")
                )
                committed = False
                uncommitted_changes = False
                if changed:
                    committed = commit_knowledge(
                        knowledge_root,
                        f"[{apply_task_id}] unified knowledge update\n\n"
                        f"source: {source}\nproposal: {proposal_hash}",
                    )
                    if not committed and not knowledge_git_is_clean(knowledge_root):
                        # The worst of the three: files on disk changed and nothing
                        # recorded them. Still not worth failing the task over --
                        # the subtitle is done -- but it needs a human, so the
                        # ledger must not advance past a chunk whose result was
                        # never committed.
                        uncommitted_changes = True
                        print(
                            f"Warning: 知识库文件已改动但 git 提交失败，"
                            f"{knowledge_root} 现在是未提交状态；请手动检查后再重跑"
                            f"（ledger 未推进）。",
                            file=sys.stderr,
                        )
                knowledge_report["committed"] = committed
                if mistake_report is not None:
                    mistake_report["committed"] = committed
                chunk_result["knowledge_report"] = knowledge_report
                chunk_result["mistake_report"] = mistake_report
                for category, entry in _applied_entry_pairs(knowledge_report):
                    applied_entries.add((category, entry))
                ledger_record = {
                    "status": "applied",
                    "task_fingerprint": task_fingerprint,
                    "chunk_index": chunk_no,
                    "window_ids": list(chunk.window_ids),
                    "input_hash": input_hash,
                    "material_hashes": material_hashes,
                    "proposal_text": result.content,
                    "knowledge_report": knowledge_report,
                    "mistake_report": mistake_report,
                    "applied_entries": _applied_entry_pairs(knowledge_report),
                    "git_head_after": knowledge_git_head(knowledge_root),
                }
                if not uncommitted_changes:
                    _append_chunk_ledger(ledger_path, ledger_record)
                    ledger[input_hash] = ledger_record
                    applied_material_hashes.update(material_hashes)
                append_task_artifact(
                    artifact_path,
                    kind="knowledge_update_apply_report",
                    task_id=task_id,
                    payload={
                        "chunk": chunk_no,
                        "knowledge_report": knowledge_report,
                        "mistake_report": mistake_report,
                    },
                )
            results.append(chunk_result)
            position += 1

    finally:
        knowledge_lock.close()
    return {
        "mode": materials.mode,
        "task_fingerprint": task_fingerprint,
        "chunks": results,
        "warnings": list(materials.warnings),
        "ledger_path": str(ledger_path),
    }


# ---------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified knowledge update from a finished correction task. The "
            "positional argument is the standard final SRT (out/<stem>/<stem>.srt); "
            "stable JSON, annotated CSV, research context and the task artifact "
            "dir derive from it unless overridden."
        )
    )
    parser.add_argument("final_srt", help="Path to the task's final SRT output.")
    parser.add_argument(
        "--refined-srt",
        help="User-refined SRT; switches to the refined_aligned evidence mode.",
    )
    parser.add_argument("--stable-json", help="Override the derived *-stable.json path.")
    parser.add_argument("--annotated-csv", help="Override the derived *-annotated.csv path.")
    parser.add_argument(
        "--research-context", help="Override the derived *-research-context.json path."
    )
    parser.add_argument(
        "--artifact-dir", help="Override the derived *.llm-artifacts directory."
    )
    parser.add_argument("--task-summary", default="", help="Short task summary.")
    parser.add_argument(
        "--task-id", default="", help="Stable task id used in knowledge commit messages."
    )
    parser.add_argument(
        "--knowledge-root",
        default=(
            str(DEFAULT_KNOWLEDGE_ROOT)
            if DEFAULT_KNOWLEDGE_ROOT is not None
            else None
        ),
        help="Root directory of the local Markdown knowledge base (embedded git repo).",
    )
    parser.add_argument(
        "--prompt-dir",
        help="Write the per-chunk update prompts to this directory.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Call the configured LLM to generate proposals. Default only writes/prints prompts.",
    )
    parser.add_argument(
        "--no-apply",
        dest="apply",
        action="store_false",
        help="With --execute: generate and retain proposals without applying them.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help=(
            "Ignore the chunk apply ledger and re-run every chunk "
            f"(default: skip chunks recorded in <artifact-dir>/{CHUNK_LEDGER_FILENAME})."
        ),
    )
    parser.add_argument(
        "--test-profile",
        action="store_true",
        help="Use gemini-3.5-flash-lite for model calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_id = args.task_id or Path(args.final_srt).stem
    try:
        report = run_knowledge_update(
            final_srt=args.final_srt,
            stable_json=args.stable_json,
            annotated_csv=args.annotated_csv,
            research_context=args.research_context,
            artifact_dir=args.artifact_dir,
            refined_srt=args.refined_srt,
            task_id=task_id,
            task_summary=args.task_summary,
            knowledge_root=args.knowledge_root,
            test_profile=args.test_profile,
            execute=args.execute,
            apply=args.execute and args.apply,
            prompt_dir=args.prompt_dir,
            resume=args.resume,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.execute:
        print(
            json.dumps(
                {
                    "mode": report["mode"],
                    "chunks": [
                        {
                            key: value
                            for key, value in chunk.items()
                            if key != "proposal_text"
                        }
                        for chunk in report["chunks"]
                    ],
                    "warnings": report["warnings"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
