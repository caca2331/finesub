"""Two-phase parallel dispatch (harness plan A.3).

Every pending window's query round runs first; one barrier fixes the session
entry set for the whole run; then every correction window dispatches
concurrently and the batch drains before the first error is raised. Commits
are ordered at the end regardless of completion order.
"""

from __future__ import annotations

import concurrent.futures as cf
import threading
from typing import Any, Dict, List, Tuple

from ...chunking import SubtitleWindow
from ...routing.config import (
    CapabilityTier,
    INJECTION_SECTION_MAX_TOKENS,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    injection_block_token_limit,
)
from ...exchange_log import ExchangeBlock
from ...injection_budget import render_knowledge_entries_block
from ...knowledge.base import append_task_artifact, load_entry_texts
from ...output_protocol import CsvValidationResult, validate_correction_window_output
from ...prompt_variants import resolve_variant
from .attempts import run_window_attempts
from .commit import _entry_details_signature, _load_parallel_entry_set, _window_input_hash
from .context import CorrectionRun
from .query_round import QueryRoundProduct, run_window_query_round


# Circuit breaker for parallel dispatch (plan A.4 (3)): serial execution is a
# natural circuit breaker, but a full parallel dispatch would burn the whole
# rpd=20 on a systemic failure within minutes. After this many failed windows
# in one batch, no new window starts; in-flight ones drain to the cache.
PARALLEL_BREAKER_FAILURES = 3


def _execute_parallel_unit(
    run: CorrectionRun,
    base_window: SubtitleWindow,
    abort: threading.Event,
    fixed_details: str,
    fixed_sig: str,
    fixed_keys: List[str],
    exchange_block: ExchangeBlock | None,
) -> List[Tuple[SubtitleWindow, CsvValidationResult]]:
    """One base window (plus any split halves) under parallel dispatch.

    The serial loop's per-window body minus the chained state: no advice
    ledger in or out, no transfer-chain evolution -- the entry set is the
    session-fixed one and identical for every window (plan A.3).
    """

    unit_results: List[Tuple[SubtitleWindow, CsvValidationResult]] = []
    pending_units: List[SubtitleWindow] = [base_window]
    while pending_units:
        if abort.is_set():
            raise RuntimeError(
                f"Window {pending_units[0].chunk_id} not attempted: the "
                "parallel circuit breaker tripped."
            )
        current = pending_units.pop(0)
        outcome = run_window_attempts(
            run,
            current,
            previous_advice="",
            window_entry_details=fixed_details,
            window_entry_sig=fixed_sig,
            injected_keys=fixed_keys,
            chained=False,
            add_tail=lambda half: pending_units.insert(0, half),
            exchange_block=exchange_block,
        )
        if outcome.restart_halves is not None:
            pending_units[0:0] = list(outcome.restart_halves)
            continue
        unit_results.append((outcome.window, outcome.validation))
    return unit_results


def run_parallel_windows(run: CorrectionRun, windows: List[SubtitleWindow]) -> None:
    """Two-phase parallel dispatch (plan A.3).

    Every pending window\'s query round runs first; one barrier fixes the
    session entry set; then every correction window dispatches
    concurrently. Failures drain (plan A.5 (1)): every worker runs to
    completion or breaker-skip, each success is already in the JSONL, and
    only then is the first error raised. Commits are ordered at the end
    regardless of completion order.
    """

    def _split_expand(win: SubtitleWindow) -> List[SubtitleWindow]:
        record = run.ledger.record(win.chunk_id) or {}
        split_into = record.get("split_into")
        if isinstance(split_into, list) and split_into:
            halves = run.geometry.split(win)
            if halves is not None and [
                half.chunk_id for half in halves
            ] == [str(part) for part in split_into]:
                return [
                    leaf
                    for half in halves
                    for leaf in _split_expand(half)
                ]
        return [win]

    def _replay(win: SubtitleWindow) -> CsvValidationResult | None:
        record = run.ledger.record(win.chunk_id)
        if not record or record.get("split_into"):
            return None
        if record.get("input_hash_core") != _window_input_hash(win):
            return None
        cached_content = str(record.get("content") or "")
        cached_tier = CapabilityTier(
            record.get("capability_tier", CapabilityTier.CAPABLE.value)
        )
        cached_validation = validate_correction_window_output(
            cached_content,
            win,
            variant=resolve_variant(record.get("variant") or None, cached_tier),
        )
        if cached_validation.ok:
            if run.task_artifact_dir:
                append_task_artifact(
                    run.task_artifact_dir,
                    kind="correction_window_cached",
                    task_id=run.task_id,
                    payload={
                        "chunk_id": win.chunk_id,
                        "source_ids": list(win.source_ids),
                        "row_count": len(cached_validation.segments),
                        "reused_from": str(record.get("continuity") or "cache"),
                    },
                )
            return cached_validation
        return None

    slots: List[Dict[str, Any]] = []
    for base in windows:
        for leaf in _split_expand(base):
            cached = _replay(leaf) if run.ledger.enabled else None
            slots.append(
                {
                    "window": leaf,
                    "results": (
                        [(leaf, cached)] if cached is not None else None
                    ),
                }
            )
    pending = [slot for slot in slots if slot["results"] is None]

    # Exchange numbering is claimed here, in window order, BEFORE any
    # dispatch: one block per executing base window, sub-numbered per
    # call, so two identical runs name their files identically no matter
    # which windows finish first (plan A.6). Split leaves share their
    # base window's block.
    window_blocks: Dict[str, ExchangeBlock] = {}
    if run.exchange_logger:
        for slot in pending:
            base_id = slot["window"].chunk_id.split("-", 1)[0]
            if base_id not in window_blocks:
                window_blocks[base_id] = run.exchange_logger.reserve_block()

    # --- Phase 1: every pending window\'s query round, concurrently ---
    seed_bodies = run.carried.entry_bodies()
    carried_text = ""
    if seed_bodies:
        carried_text = render_knowledge_entries_block(
            seed_bodies,
            count_tokens=run.token_counter.count_text,
            entry_limit=INJECTION_SECTION_MAX_TOKENS,
            block_limit=injection_block_token_limit(KB_TRANSFER_MAX_ENTRIES),
        ).text
    worker_count = max(1, int(run.parallel_window_limit))

    def _query_round_for(win: SubtitleWindow) -> None:
        base_chunk_id = win.chunk_id.split("-", 1)[0]
        if base_chunk_id in run.query_round_cache:
            return
        if run.search_client is None:
            run.query_round_cache[base_chunk_id] = QueryRoundProduct()
            return
        query_file_ref = run.media.query_ref(win)
        run.query_round_cache[base_chunk_id] = run_window_query_round(
            knowledge_enabled=run.knowledge_enabled,
            client=run.client,
            window=win,
            context_pack=run.context_pack,
            audio_label=run.media.query_label(win),
            previous_advice="",
            file_ref=query_file_ref,
            search_client=run.search_client,
            knowledge_root=run.knowledge_root,
            streamer_index=run.streamer_index_text,
            common_index=run.common_index_text,
            carried_entries_text=carried_text,
            carried_key_count=len(seed_bodies),
            max_queries=run.max_search_queries_per_window,
            task_artifact_dir=run.task_artifact_dir,
            task_id=run.task_id,
            token_rows=run.token_rows,
            exchange_logger=window_blocks.get(base_chunk_id),
            token_counter=run.token_counter,
            profile=run.profile,
            resume=run.resume,
            checkpoint_store=run.session_checkpoint_store,
            checkpoint_extra_identity={
                "task_fingerprint": run.task_fingerprint,
            },
        )

    if run.external_injection and pending:
        # One query round per *base* chunk id: -a/-b leaves of one split
        # parent share the round, and mapping over the leaves would race
        # the same cache key from two workers (check-then-act) -- two
        # identical LLM calls, one overwriting the other.
        query_targets: Dict[str, SubtitleWindow] = {}
        for slot in pending:
            base_id = slot["window"].chunk_id.split("-", 1)[0]
            query_targets.setdefault(base_id, slot["window"])
        with cf.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="llm-query"
        ) as query_pool:
            list(query_pool.map(_query_round_for, query_targets.values()))

    # --- Barrier: fix the session entry set once (plan A.3) ---
    fixed_details = run.entry_details
    fixed_keys: List[str] = []
    if not fixed_details and run.knowledge_enabled:
        # A resumed task reuses the first barrier's decision: recomputing
        # from the *pending* windows only shrank the set and split one
        # task's "fixed" set between cached and fresh windows (2026-08-10
        # review P1). Keys are pinned; bodies re-render from the current
        # KB (A.5 (5) exemption).
        persisted_keys = (
            _load_parallel_entry_set(run.ledger.path, run.task_fingerprint)
            if run.ledger.enabled and run.ledger.path is not None
            else None
        )
        if persisted_keys is not None:
            found, _missing = load_entry_texts(run.knowledge_root, persisted_keys)
            union: Dict[str, str] = {
                key: found[key] for key in persisted_keys if key in found
            }
        else:
            request_counts: Dict[str, int] = {}
            first_seen: Dict[str, int] = {}
            for index, slot in enumerate(pending):
                base_chunk_id = slot["window"].chunk_id.split("-", 1)[0]
                product = (
                    run.query_round_cache.get(base_chunk_id) or QueryRoundProduct()
                )
                for key in product.requested_entry_keys:
                    if key in seed_bodies:
                        continue
                    request_counts[key] = request_counts.get(key, 0) + 1
                    first_seen.setdefault(key, index)
            # Seed-first, then requests by how many windows asked (ties by
            # first requesting window) -- same shape as the keep-first rule.
            ordered_requests = sorted(
                request_counts,
                key=lambda key: (-request_counts[key], first_seen[key]),
            )
            union = dict(seed_bodies)
            if ordered_requests:
                fresh_found, _fresh_missing = load_entry_texts(
                    run.knowledge_root, ordered_requests
                )
                for key in ordered_requests:
                    if key in fresh_found and key not in union:
                        union[key] = fresh_found[key]
        union = dict(list(union.items())[:KB_WINDOW_TOTAL_ENTRIES])
        fixed_keys = list(union)
        if union:
            fixed_details = render_knowledge_entries_block(
                union,
                count_tokens=run.token_counter.count_text,
                entry_limit=INJECTION_SECTION_MAX_TOKENS,
                block_limit=injection_block_token_limit(
                    KB_WINDOW_TOTAL_ENTRIES
                ),
            ).text
        if pending and persisted_keys is None:
            run.ledger.append_only(
                {
                    "parallel_entry_set": fixed_keys,
                    "task_fingerprint": run.task_fingerprint,
                }
            )
    fixed_sig = _entry_details_signature(fixed_details)
    if run.task_artifact_dir and pending:
        append_task_artifact(
            run.task_artifact_dir,
            kind="parallel_entry_set",
            task_id=run.task_id,
            payload={
                "seed_keys": list(seed_bodies),
                "fixed_keys": list(fixed_keys),
            },
        )

    # --- Phase 2: correction windows, concurrently, fully drained ---
    for slot in pending:
        run.media.schedule_correction(slot["window"])
    abort = threading.Event()
    errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def _run_slot(slot: Dict[str, Any]) -> None:
        if abort.is_set():
            # Breaker tripped before this window started: skip without
            # counting as a new failure; the batch is already doomed.
            return
        try:
            base_id = slot["window"].chunk_id.split("-", 1)[0]
            slot["results"] = _execute_parallel_unit(
                run,
                slot["window"],
                abort,
                fixed_details,
                fixed_sig,
                fixed_keys,
                window_blocks.get(base_id),
            )
        except BaseException as exc:
            # The breaker trips inside the worker, not in the consumer:
            # by the time the main thread has seen the failed future, the
            # same worker may already have started the next doomed window.
            with errors_lock:
                errors.append(exc)
                if len(errors) >= PARALLEL_BREAKER_FAILURES:
                    abort.set()
            raise

    with cf.ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="llm-corr"
    ) as correction_pool:
        future_map = {
            correction_pool.submit(_run_slot, slot): slot for slot in pending
        }
        cf.wait(list(future_map))
    if errors:
        # drain-then-raise (plan A.5 (1)): every completed window is
        # already in the cache; the rerun replays them and re-attempts
        # only what failed.
        raise errors[0]

    # --- Ordered commit (plan A.5 (2)) ---
    for slot in slots:
        results = slot.get("results")
        if results is None:
            raise RuntimeError(
                f"Window {slot['window'].chunk_id} was skipped by the "
                "parallel circuit breaker."
            )
        for win, validation in results:
            run.commit_window(win, validation, "")
