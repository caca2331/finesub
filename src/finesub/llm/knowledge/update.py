"""Unified knowledge update: one entry point for both evidence modes.

Replaces the old ``task_auto`` / ``post_task`` split (docs/
knowledge.md). Evidence is structured — per-window CSV packs
plus aggregated ``task_update_feedback`` — instead of a raw artifact dump;
providing ``--refined-srt`` switches from the ``artifacts_only`` prompt to the
``refined_aligned`` one (which alone sees the refined subtitles, and alone
may propose into the run's style entry).

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
from typing import Any, Dict, List, Mapping, Sequence

from finesub.paths import is_linked_worktree
from finesub.reporting import current_reporter

from ..client import (
    GeminiPromptBlockedError,
    RoleClient,
    extract_finish_reason,
    extract_token_distribution,
    is_prompt_blocked,
    validation_retry_sampling_kwargs,
)
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
    append_task_artifact,
    knowledge_write_lock,
    load_index_text,
    parse_knowledge_proposals_jsonl,
)
from .entries import (
    EntrySelection,
    pin_style_entries,
    render_kb_entry_excerpt,
    select_kb_entries,
)
from .style import resolve_style_keys
from .node.proposals import apply_model_proposals
from .node.repo import KnowledgeRepo
from .materials import (
    KNOWLEDGE_CSV_TOKEN_BUDGET,
    KnowledgeChunk,
    KnowledgeMaterials,
    MODE_REFINED_ALIGNED,
    build_knowledge_materials,
)
from ..prompts import (
    PROMPT_VERSION,
    build_knowledge_conflict_repair_messages,
    build_knowledge_update_messages,
)
from ..token_budget import default_token_counter, TokenCounter


CHUNK_LEDGER_FILENAME = "knowledge-update-chunks.jsonl"
# Align with research._call_and_parse: 1 retry → 2 attempts total.
DEFAULT_KNOWLEDGE_UPDATE_PARSE_RETRIES = 1


def _validate_knowledge_update_jsonl(text: str) -> None:
    """Raise ValueError if the knowledge proposal JSONL is syntactically invalid.

    One block since the style entry took over from the mistake ledger: both
    variants now emit `<knowledge_proposals>` and nothing else.
    """

    parse_knowledge_proposals_jsonl(text)


# ---------------------------------------------------------------------------
# Path derivation (mirrors stages.default_pipeline_paths)


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
        current_reporter().debug(
            "migrated research context",
            {"from": str(legacy), "to": str(preferred)},
        )
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


# --- B': feeding a concurrent-write conflict back to the model ---------------

#: How many entries a repair round may carry. The round exists to re-decide a
#: handful of dropped lines, not to re-run the task: a conflict that touched
#: more than this is better served by rerunning the whole chunk.
MAX_REPAIR_ENTRIES = 6


def conflicted_entries(report: Mapping[str, Any]) -> List[tuple[str, str]]:
    """(category, entry) pairs whose proposals a concurrent writer took.

    Reads the report's own skipped records rather than the engine's conflict
    rows: the records name the ENTRY, which is what a repair round has to
    re-render, while a conflict row names an internal entity id. Rejections
    (bad op, missing parent) are not conflicts and are not repairable this way.
    """

    if not report.get("conflicts") or report.get("rolled_back"):
        return []
    pairs: List[tuple[str, str]] = []
    for record in report.get("skipped", []):
        reason = str(record.get("reason") or "")
        if "conflict" not in reason and "已被占用" not in reason and "并发" not in reason:
            continue
        pair = (str(record.get("category") or ""), str(record.get("entry") or ""))
        if pair[1] and pair not in pairs:
            pairs.append(pair)
    return pairs[:MAX_REPAIR_ENTRIES]


def _dropped_rows_text(
    report: Mapping[str, Any],
    pairs: Sequence[tuple[str, str]],
    *,
    proposal_text: str = "",
) -> str:
    """The dropped lines, VERBATIM, each under the reason it was dropped.

    The proposal itself, not a summary of it: an op name and an entry name are
    not something a model can re-decide from -- it would have to invent what it
    had wanted to write (reviewer 2026-08-31 P1). The original JSON carries the
    `content` it proposed AND the `reason` it gave for it, which is also why
    this round does not need the chunk's material injected a second time.
    """

    from .node.proposals import parse_model_proposals

    wanted = set(pairs)
    originals: Dict[tuple[str, str], List[str]] = {}
    if proposal_text:
        for proposal in parse_model_proposals(proposal_text):
            pair = (
                str(proposal.get("category") or ""),
                str(proposal.get("entry") or ""),
            )
            if pair in wanted:
                originals.setdefault(pair, []).append(
                    json.dumps(proposal, ensure_ascii=False, sort_keys=True)
                )
    rows: List[str] = []
    for record in report.get("skipped", []):
        pair = (str(record.get("category") or ""), str(record.get("entry") or ""))
        if pair not in wanted:
            continue
        head = (
            f"- {pair[0]}/{pair[1]}：`{record.get('op', '')}`"
            f"{('（' + str(record.get('section')) + ' 小节）') if record.get('section') else ''}"
            f" 被丢弃 —— {record.get('reason', '')}"
        )
        for line in originals.pop(pair, []):
            head += f"\n  你原来提的：`{line}`"
        rows.append(head)
    return "\n".join(rows)


def repair_knowledge_conflicts(
    report: Mapping[str, Any],
    *,
    knowledge_root: str | Path,
    llm_client: RoleClient,
    counter: TokenCounter,
    task_summary: str,
    read_rev: int,
    task_id: str,
    apply_task_id: str,
    source: str,
    difficulty: str,
    sampling: Mapping[str, Any],
    artifact_path: Path | None,
    proposal_text: str = "",
    exchange_logger: ExchangeLogger | None = None,
    session: str = "knowledge-conflict-repair",
) -> Dict[str, Any] | None:
    """One repair round for the proposals a concurrent writer displaced (B').

    The model is shown what it read (``read_rev``), what the store holds now,
    and which of its lines were dropped and why -- then asked to re-decide only
    those against the current contents. The entries are rendered in the same
    prompt projection the main round used, at the new revision, and the handle
    map that comes with them is what the second envelope binds to: the model
    can only reference versions it has now been shown, which is exactly the CAS
    precondition it failed the first time.

    Best effort by construction. The main proposals are already committed; a
    repair that errors, returns nothing, or is refused leaves the run exactly
    where a run without B' would have been, so every failure here is a warning.
    """

    pairs = conflicted_entries(report)
    if not pairs:
        return None
    repo = KnowledgeRepo.open(knowledge_root)
    current_rev = repo.rev
    if current_rev - read_rev <= (1 if report.get("rev") is not None else 0):
        # No FOREIGN revision since the read -- at most the main apply's own
        # commit landed. (`current_rev <= read_rev` alone missed this: the
        # main apply bumps the rev whenever it commits, competitor or not, so
        # it never held for a committed apply: reviewer 2026-08-31 P3.) The
        # drops collided with the model's own lines, and a repair round would
        # only show it what it just wrote; a competitor's write is what B'
        # exists for.
        return None
    block, handles = render_kb_entry_excerpt(
        [
            EntrySelection(category=category, key=key, score=0.0, exists=True)
            for category, key in pairs
        ],
        knowledge_root,
        count_tokens=counter.count_text,
        rev=current_rev,
    )
    messages = build_knowledge_conflict_repair_messages(
        task_summary=task_summary,
        read_rev=read_rev,
        current_rev=current_rev,
        dropped_rows=_dropped_rows_text(report, pairs, proposal_text=proposal_text),
        kb_entries=block.text,
    )
    def _failed(exc: BaseException) -> Dict[str, Any]:
        current_reporter().warning(
            "knowledge-conflict-repair-failed",
            f"冲突回喂轮失败：{type(exc).__name__}: {exc}",
            impact="被丢弃的提案保持丢弃，其余照常落库",
            action="重跑该任务即可（材料与 ledger 未变）",
        )
        return {"attempted": True, "error": f"{type(exc).__name__}: {exc}"}

    try:
        result = llm_client.complete(
            LLMRole.GENERAL_CAPABLE,
            messages,
            max_tokens=SESSION_OUTPUT_MAX_TOKENS,
            task_group="knowledge",
            difficulty=difficulty,
            agent_task_extras={
                "knowledge_root": str(knowledge_root),
                "knowledge_identity": f"rev:{current_rev}",
                "kb_tools": "propose",
                "kb_handle_bindings": handles.bindings(),
                "kb_signal_task": task_id or "manual",
                "kb_signal_window": "conflict-repair",
            },
            **dict(sampling),
        )
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        return _failed(exc)
    if exchange_logger is not None:
        # Before validation on purpose: the round that fails to validate is
        # precisely the one someone reads back, and logging only successes
        # left it with no record at all (reviewer 2026-08-31 P3).
        exchange_logger.log(
            session,
            messages=messages,
            response_text=result.content,
            metadata=llm_exchange_metadata(
                result, session=session, read_rev=read_rev, current_rev=current_rev
            ),
        )
    try:
        _validate_knowledge_update_jsonl(result.content)
        repair_report = apply_model_proposals(
            result.content,
            repo=repo,
            task_id=f"{apply_task_id}#repair",
            knowledge_read_rev=current_rev,
            handles=handles,
            assignment_id=f"{source}:repair",
            proposal_text_hash=_sha256(result.content),
        ).to_dict()
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        return _failed(exc)
    if artifact_path is not None:
        append_task_artifact(
            artifact_path,
            kind="knowledge_conflict_repair",
            task_id=task_id,
            payload={
                "read_rev": read_rev,
                "current_rev": current_rev,
                "entries": [list(pair) for pair in pairs],
                "report": repair_report,
            },
        )
    repair_report["attempted"] = True
    repair_report["read_rev"] = read_rev
    repair_report["current_rev"] = current_rev
    applied = len(repair_report.get("applied", []))
    current_reporter().debug(
        "knowledge conflict repair",
        {"entries": len(pairs), "reapplied": applied, "rev": repair_report.get("rev")},
    )
    return repair_report

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


def _chunk_input_hash(
    chunk: KnowledgeChunk,
    materials: KnowledgeMaterials,
    style_keys: Sequence[str] = (),
) -> str:
    """Chunk identity for the ledger: stable material text, plus the style
    entry this run may write into.

    Excludes the KB-entry excerpt on purpose — it changes as earlier chunks
    apply, and a rerun must still recognize an already-applied chunk. The
    STYLE is different: rerunning the same material against another style is a
    different question ("what does this material say about style B?"), and
    without it here the second run is skipped as already-applied and B never
    receives a single proposal (review 2026-09-02). This is the knowledge
    task's identity only — correction windows still do not invalidate on a
    style change (`translation-style-plan.md` §2.5).
    """

    payload: dict[str, Any] = {
        "packs": chunk.packs_text(),
        "general_context": materials.general_context,
        "research_feedback": materials.feedback.research_slice_text(),
    }
    if style_keys:
        # omitted when empty: a chunk applied before styles existed keeps its
        # identity and is still recognized as applied
        payload["style"] = list(style_keys)
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True))


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
    chunk: KnowledgeChunk,
    materials: KnowledgeMaterials,
    style_keys: Sequence[str] = (),
) -> List[str]:
    """Per-window identities survive a later change in chunk boundaries.

    This is the hash `covered_by_prior_chunks` compares, so the writable style
    belongs here for the same reason it belongs in the chunk hash: the same
    window against another style is a different question, and without it the
    second style is reported "already_applied" and receives nothing (review
    2026-09-02, second half of the same defect).

    The key is OMITTED when no style is writable, so every window applied
    before styles existed keeps its hash and stays recognized.
    """

    hashes: List[str] = []
    for window in chunk.windows:
        payload: dict[str, Any] = {
            "mode": materials.mode,
            "window": window.pack_text(),
            "general_context": materials.general_context,
            "research_feedback": materials.feedback.research_slice_text(),
        }
        if style_keys:
            payload["style"] = list(style_keys)
        hashes.append(_sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True)))
    return hashes


def _revision_with_proposal(
    knowledge_root: str | Path, proposal_text_hash: str, *, after: int
) -> int | None:
    """The revision (> ``after``) that applied the proposal with this raw-text
    hash, or ``None``. Backs crash recovery between store commit and ledger."""

    if not proposal_text_hash:
        return None
    store = KnowledgeRepo.open(knowledge_root).store
    row = store.conn.execute(
        "SELECT rev FROM revisions WHERE rev > ? AND note = ? ORDER BY rev LIMIT 1",
        (after, f"proposal_text:{proposal_text_hash}"),
    ).fetchone()
    return int(row["rev"]) if row else None


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


def run_knowledge_update(**kwargs: Any) -> Dict[str, Any]:
    """Book this task's agent slot account, then run the update.

    A standalone update (reference_ingest's second step, the module CLI) is a
    task of its own and must reserve its mandatory lane like any other, or it
    starves behind a sibling task's optional fan-out and stays invisible to
    ``A`` in the allocator (plan W4). Nested inside a correction run the
    account is already open and this reuses it -- one task, one reservation.
    """

    from ..run_context import default_task_slots

    with default_task_slots(test_profile=bool(kwargs.get("test_profile"))):
        return _run_knowledge_update(**kwargs)


def _run_knowledge_update(
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
    #: The run's `--style` selection: these entries are pinned into every
    #: chunk's `<kb_entries>` so the refined variant can propose conventions
    #: into them. Empty means the run named no style, and the prompt's style
    #: section then has nothing to act on.
    style_names: Sequence[str] = (),
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
            "当前在 git worktree 中运行，已跳过知识库更新以免写入主仓的知识库；"
            "确需更新请设 FINESUB_KNOWLEDGE_WRITE=1 后重跑（ledger 未推进）。"
        )
        current_reporter().warning("knowledge-worktree-skipped", message)
        return {
            "mode": "",
            "task_fingerprint": "",
            "chunks": [],
            "warnings": [f"Warning: {message}"],
            "ledger_path": "",
            "skipped": "worktree_readonly",
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
        current_reporter().warning("knowledge-update-material", warning)

    refined = materials.mode == MODE_REFINED_ALIGNED
    refined_text = (
        Path(refined_srt).expanduser().read_text(encoding="utf-8") if refined_srt else ""
    )
    if refined and execute and apply:
        # Deterministic evidence write-back (plan §5.4): the refined SRT
        # keeping/overturning a misheard-attributed correction confirms/refutes
        # exactly that claim. Runs before the LLM chunks — it reads only the
        # aligned materials — and is fail-soft, idempotent telemetry; the
        # the style half of the feedback travels as ordinary entry
        # proposals into the pinned style entry.
        try:
            from .node.signals import refined_alignment_evidence

            confirmed, refuted = refined_alignment_evidence(
                KnowledgeRepo.open(knowledge_root).store,
                (
                    (window.chunk_id, window.raw_csv, window.final_csv, window.refined_csv)
                    for chunk in materials.chunks
                    for window in chunk.windows
                ),
                task_id=task_id or "knowledge-update",
            )
            if confirmed or refuted:
                current_reporter().debug(
                    f"refined alignment evidence: {confirmed} confirmed, {refuted} refuted"
                )
        except Exception as exc:  # telemetry must never sink the update
            current_reporter().debug(f"refined alignment evidence skipped: {exc}")
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
    # A style entry is WRITABLE only with refined subtitles in hand: the
    # `artifacts_only` variant sees the machine's own raw/final text, and
    # letting it edit the style would be the model teaching itself its own
    # habits (review 2026-09-02). It is also at most ONE: the prompt tells the
    # model to write into "the" style entry, and with several pinned it could
    # not know which convention belongs where -- the rest stay injected on the
    # correction side and simply are not offered here.
    writable_style_keys: list[str] = []
    if refined and style_names:
        writable_style_keys = resolve_style_keys(knowledge_root, style_names)[:1]
        if len(style_names) > 1:
            current_reporter().warning(
                "knowledge-style-multi",
                f"本次指定了 {len(style_names)} 套风格，知识更新只写第一套"
                f"（{writable_style_keys[0] if writable_style_keys else ''}）",
                impact="其余几套照常注入纠错提示词，但不会收录本次的新约定",
                action="想更新另一套就单独跑一次，或把 --style 的第一位换成它",
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
            input_hash = _chunk_input_hash(chunk, materials, writable_style_keys)
            material_hashes = _chunk_material_hashes(chunk, materials, writable_style_keys)
            cached = ledger.get(input_hash)
            covered_by_prior_chunks = bool(material_hashes) and all(
                material_hash in applied_material_hashes
                for material_hash in material_hashes
            )
            pending_intent = pending_intents.get(input_hash)
            if cached is None and pending_intent is not None:
                # Crash between the store commit and the ledger write: the
                # revision table remembers every applied proposal hash, so a
                # revision after the intent's rev carrying ours means the chunk
                # landed (plan §2.5).
                rev_before = int(pending_intent.get("rev_before") or 0)
                proposal_hash = str(pending_intent.get("proposal_hash") or "")
                landed_rev = _revision_with_proposal(knowledge_root, proposal_hash, after=rev_before)
                if proposal_hash and landed_rev is not None:
                    recovered = {
                        "status": "recovered_after_commit",
                        "task_fingerprint": task_fingerprint,
                        "chunk_index": chunk_no,
                        "window_ids": list(chunk.window_ids),
                        "input_hash": input_hash,
                        "material_hashes": material_hashes,
                        "rev_after": landed_rev,
                        "knowledge_report": None,
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
                    }
                )
                current_reporter().debug(
                    "knowledge update chunk already applied",
                    {"chunk": chunk_no, "ledger": str(ledger_path)},
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
            # working_rev (plan §2.5): this chunk reads the store as it is *now*,
            # i.e. after the previous chunk's apply; the handle map binds every
            # rendered node to the version the model is about to see. Every read
            # below passes it explicitly — the run-level generation pin may
            # still be active, and read-your-writes must override it.
            working_rev = KnowledgeRepo.open(knowledge_root).rev
            selections = select_kb_entries(
                window_hints,
                knowledge_root=knowledge_root,
                research_origins=research_hints,
                applied_entries=applied_entries,
                rev=working_rev,
            )
            selections = pin_style_entries(selections, writable_style_keys)
            kb_entries_block, kb_handles = render_kb_entry_excerpt(
                selections, knowledge_root, count_tokens=counter.count_text, rev=working_rev
            )
            # v17: the update model must see the live index before proposing
            # create_entry; reload per chunk so entries created/renamed by the
            # previous chunk's apply are visible.
            streamer_index_text = load_index_text(knowledge_root, "streamer", rev=working_rev)
            common_index_text = load_index_text(knowledge_root, "common", rev=working_rev)
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
                    current_reporter().debug(
                        "knowledge update chunk exceeds the prompt input limit; "
                        "splitting on window boundary",
                        {"chunk": chunk_no, "prompt_tokens": prompt_tokens},
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
                    print(messages_to_text(messages))  # product output
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
                        # Agent-backed chunks (plan §6.5, 4c): the kb_* tools
                        # read this chunk's working_rev (not the run pin), the
                        # kb_validate pre-check is admitted, and the prompt's
                        # handle table rides the manifest so the tools and the
                        # proposal block share one handle space.
                        agent_task_extras={
                            "knowledge_root": str(knowledge_root),
                            "knowledge_identity": f"rev:{working_rev}",
                            "kb_tools": "propose",
                            "kb_handle_bindings": kb_handles.bindings(),
                            "kb_signal_task": task_id or "manual",
                            "kb_signal_window": f"chunk-{chunk_no}",
                        },
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
                    _validate_knowledge_update_jsonl(result.content)
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
                    current_reporter().warning(
                        "knowledge-locked",
                        "另一个 FineSub 进程正在写知识库，本次跳过自动应用",
                        impact="提案已保留，ledger 未推进",
                        action="稍后重跑会重做",
                    )
                    apply = False
                else:
                    knowledge_repo_prepared = True
            if apply:
                source = f"llm.knowledge_update:{materials.mode}:chunk{chunk_no}"
                apply_task_id = f"{task_id or 'manual'}#chunk{chunk_no}" if multi_chunk else (
                    task_id or "manual"
                )
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
                        "rev_before": working_rev,
                        "proposal_hash": proposal_hash,
                    },
                )
                # One store transaction per chunk (plan §2.5): handles bind the
                # model's references to the versions it saw at working_rev, and
                # the apply engine CAS-checks each entity once. The revision
                # records the *raw proposal text* hash so a crash between this
                # commit and the ledger write is recoverable (see above).
                knowledge_report = apply_model_proposals(
                    result.content,
                    repo=KnowledgeRepo.open(knowledge_root),
                    task_id=apply_task_id,
                    knowledge_read_rev=working_rev,
                    handles=kb_handles,
                    assignment_id=source,
                    proposal_text_hash=proposal_hash,
                ).to_dict()
                if knowledge_report.get("rolled_back"):
                    current_reporter().warning(
                        "knowledge-apply-rolled-back",
                        f"知识库提案整体回滚：{knowledge_report.get('rollback_reason', '')}",
                        impact="本块未写入知识库，ledger 照常推进",
                        action="看 apply report 里的 conflicts / skipped",
                    )
                elif knowledge_report.get("conflicts"):
                    # Typed drops used to be a near-impossible event; under
                    # task-level parallelism they are the normal signal that a
                    # concurrent writer won (plan W2) — silent means silently
                    # losing proposals.
                    conflict_rows = knowledge_report["conflicts"]
                    current_reporter().warning(
                        "knowledge-apply-conflict",
                        f"知识库并发冲突：{len(conflict_rows)} 处按类型化解"
                        f"（chunk {chunk_no}；其余提案照常落库）",
                        impact="冲突的 op 被丢弃或回退，未整包回滚",
                        action="看 apply report 里的 conflicts",
                    )
                    # B' (plan §4.2): ask the model to re-decide exactly those
                    # lines against what the winner wrote. Everything else is
                    # already committed, so this is additive by construction --
                    # see `repair_knowledge_conflicts` for why every failure in
                    # it is a warning rather than an error.
                    repair_report = repair_knowledge_conflicts(
                        knowledge_report,
                        knowledge_root=knowledge_root,
                        llm_client=llm_client,
                        counter=counter,
                        task_summary=task_summary,
                        read_rev=working_rev,
                        task_id=task_id,
                        apply_task_id=apply_task_id,
                        source=source,
                        difficulty=difficulty,
                        sampling=sampling,
                        artifact_path=artifact_path,
                        proposal_text=result.content,
                        exchange_logger=exchange_logger,
                        session=f"knowledge-conflict-repair-chunk{chunk_no}",
                    )
                    if repair_report is not None:
                        chunk_result["knowledge_repair_report"] = repair_report
                        for category, entry in _applied_entry_pairs(repair_report):
                            applied_entries.add((category, entry))
                committed = knowledge_report.get("rev") is not None
                knowledge_report["committed"] = committed
                chunk_result["knowledge_report"] = knowledge_report
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
                    "applied_entries": _applied_entry_pairs(knowledge_report),
                    "rev_after": KnowledgeRepo.open(knowledge_root).rev,
                }
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
