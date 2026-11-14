"""Serial dispatch: one window at a time, each reading the last one's notes.

This is the mode that has continuity. Each window's `<next_advice>` enters the
advice ledger the next window is shown, and its `<keep_entries>` is the only
continuation point of the knowledge transfer chain -- which is why a window
answered from the resume cache still has to re-derive both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from finesub.reporting import current_reporter
from ...chunking import SubtitleWindow
from ...routing.config import (
    ADVICE_LEDGER_MAX_TOKENS,
    KB_TRANSFER_MAX_ENTRIES,
    KB_WINDOW_TOTAL_ENTRIES,
    injection_block_token_limit,
)
from ... import kb_phonetic_hints
from ...injection_budget import render_knowledge_entries_block
from ...knowledge.base import append_task_artifact, knowledge_root_path, load_entry_texts, version_rev
from ...output_protocol import CsvValidationResult
from ...token_truncate import truncate_text_only
from .attempts import _extract_next_advice, run_window_attempts
from .commit import WindowEntries
from .context import CorrectionRun, RequestTurnProduct
from .request_turn import run_window_request_turn


@dataclass(frozen=True)
class _CarriedView:
    """What a window shows of its carried entries (`kb_phonetic_hints`).

    ``bodies`` are slimmed under ``FINESUB_KB_SLIM=on`` when the whole block is
    over the gate, else the entries as they are; ``hints`` and ``slim`` (the
    window's record) go with them."""

    bodies: Dict[str, str]
    hints: str = ""
    slim: Dict[str, Any] | None = None

    @property
    def slimmed(self) -> bool:
        return bool(self.slim and "entry_tokens" in self.slim)


def _render_entries(run: CorrectionRun, bodies: Dict[str, str], entries: int):
    return render_knowledge_entries_block(
        bodies,
        count_tokens=run.token_counter.count_text,
        block_limit=injection_block_token_limit(entries),
    )


def _carried_view(
    run: CorrectionRun, current: SubtitleWindow, carried: Dict[str, str]
) -> _CarriedView:
    """The carried entries slimmed, with phonetic hints, when the switch is on
    and they render over the gate. Computed before the request turn so that it
    and the delivery show the same block."""

    if not (carried and kb_phonetic_hints.enabled()):
        return _CarriedView(dict(carried))
    whole = run.token_counter.count_text(
        _render_entries(run, carried, KB_WINDOW_TOTAL_ENTRIES).text
    )
    slim: Dict[str, Any] = {"whole_tokens": whole, "gate": whole > kb_phonetic_hints.GATE_TOKENS}
    if not slim["gate"]:
        return _CarriedView(dict(carried), slim=slim)
    rev = version_rev(run.knowledge_version)
    slimmed = (
        kb_phonetic_hints.slim_window_entries(
            current, carried, root=str(knowledge_root_path(run.knowledge_root)), rev=rev
        )
        if rev is not None
        else None
    )
    if slimmed is None:
        slim["skipped"] = "no pinned knowledge revision" if rev is None else "reading libraries missing"
        return _CarriedView(dict(carried), slim=slim)
    slim.update(
        entry_tokens=run.token_counter.count_text(
            _render_entries(run, slimmed.bodies, KB_WINDOW_TOTAL_ENTRIES).text
        ),
        hint_tokens=run.token_counter.count_text(slimmed.hints),
        hinted=list(slimmed.hinted),
    )
    return _CarriedView(slimmed.bodies, slimmed.hints, slim)


def _window_entries(
    run: CorrectionRun, view: _CarriedView, union: Dict[str, str]
) -> WindowEntries:
    """The window's entry block: ``union`` rendered, with the carried view's hints."""

    if not union:
        return WindowEntries()
    block = _render_entries(run, union, KB_WINDOW_TOTAL_ENTRIES)
    return WindowEntries(
        details=block.text,
        # what the budget let in: an entry it dropped was never shown
        keys=tuple(block.injected),
        # A slimmed entry the budget cut counts its kept lines in the slimmed
        # body, which drops hinted terms: exposure logging may then under-count.
        extent={key: dict(kept) for key, kept in block.kept.items()},
        hints=view.hints,
        slimmed=view.slimmed,
        slim=view.slim,
    )


def _commit_replay(
    run: CorrectionRun, current: SubtitleWindow, record: Dict[str, Any],
    cached_validation: CsvValidationResult,
) -> None:
    """Commit a window answered from the ledger (`ResumeLedger.replay` passed it): its advice and keep list
    re-derived from the cached answer, as a first run would have folded them in."""

    cached_content = str(record.get("content") or "")
    # Replaying a parallel-produced window in serial mode is
    # allowed but lossy: parallel windows never wrote advice,
    # so the ledger later live windows read starts short of
    # what a first serial run would have built. Say so rather
    # than silently changing what those windows see.
    cached_continuity = str(record.get("continuity") or "")
    if (
        run.profile.continuity == "serial"
        and cached_continuity == "parallel"
        and current.chunk_id not in run.warned_parallel_replays
    ):
        run.warned_parallel_replays.add(current.chunk_id)
        current_reporter().warning(
            "correction-advice-missing",
            f"window {current.chunk_id} was produced in parallel "
            "mode; its advice was never recorded",
            impact="串行的 advice ledger 从此处起缺这一条",
        )
    run.commit_window(
        current,
        cached_validation,
        _extract_next_advice(
            cached_content, count_tokens=run.token_counter.count_text
        ),
    )
    run.note_replay(current)
    if run.task_artifact_dir:
        append_task_artifact(
            run.task_artifact_dir,
            kind="correction_window_cached",
            task_id=run.task_id,
            payload={
                "chunk_id": current.chunk_id,
                "source_ids": list(current.source_ids),
                "row_count": len(cached_validation.segments),
                "reused_from": cached_continuity or "cache",
            },
        )
    # Replay continues the transfer chain from the cached
    # keep list so later windows hash consistently.
    run.carried.keys = [
        key
        for key in (record.get("keep_entries") or [])
        if isinstance(key, str) and key
    ][:run.carried.cap]


def run_serial_windows(run: CorrectionRun, windows: List[SubtitleWindow]) -> None:
    """Execute `windows` in order, growing the list when one splits."""

    i = 0
    while i < len(windows):
        current = windows[i]
        transfer_found = run.carried.entry_bodies()
        if run.ledger.holds(current.chunk_id):
            record = run.ledger.records[current.chunk_id]
            split_into = record.get("split_into")
            if isinstance(split_into, list) and split_into:
                # A parent that split has no result of its own; re-derive
                # the halves (deterministic) and let each hit its record.
                halves = run.geometry.split(current)
                if halves is not None and [
                    half.chunk_id for half in halves
                ] == [str(part) for part in split_into]:
                    windows[i : i + 1] = list(halves)
                    continue
            cached_validation = run.ledger.replay(current)
            if cached_validation is not None:
                _commit_replay(run, current, record, cached_validation)
                i += 1
                continue
        # Below the resume check on purpose. Priming the next window's clip
        # is only worth it when this window is actually going to call the
        # model; a replay that answers every window from the cache used to
        # run one ffmpeg extraction and one Gemini Files upload per window
        # anyway -- ~19 of each for a 20-window task that made no LLM call
        # at all, each logged as an api_call and each spending upload quota.
        run.media.prefetch_next_correction(windows, i)
        # One exchange block per executing window, claimed in window
        # order (cached replays claim none), so serial numbering matches
        # the parallel scheme and stays deterministic across reruns.
        window_block = (
            run.exchange_logger.reserve_block() if run.exchange_logger else None
        )
        # Cumulative ledger cap: keep the most recent windows' advice
        # (front-truncated) within the token budget.
        previous_advice = truncate_text_only(
            run.carried.rendered_advice(),
            ADVICE_LEDGER_MAX_TOKENS,
            run.token_counter.count_text,
            keep="tail",
            prefer_natural_boundary=True,
        )
        view = _carried_view(run, current, transfer_found)
        request_product = RequestTurnProduct()
        if run.request_channel:
            # No search client only means nothing gets searched: an
            # entries-only channel (`retrieval=none|native`) still runs its turn.
            base_chunk_id = current.chunk_id.split("-", 1)[0]
            if base_chunk_id not in run.request_turn_cache:
                carried_text = ""
                if view.bodies:
                    carried_text = _render_entries(
                        run, view.bodies, KB_TRANSFER_MAX_ENTRIES
                    ).text
                run.request_turn_cache[base_chunk_id] = run_window_request_turn(
                    run,
                    current,
                    previous_advice=previous_advice,
                    carried_entries_text=carried_text,
                    carried_key_count=len(transfer_found),
                    exchange_logger=window_block,
                    phonetic_hints=view.hints,
                    entries_slimmed=view.slimmed,
                )
            request_product = run.request_turn_cache[base_chunk_id]
        # v17: one unified render — transfers first, then this window's new
        # requests, capped to the window total. Order is priority: a full
        # block shares out longest-first, and when even the entries' heads do
        # not fit, the last ones drop (`render_knowledge_entries_block`).
        union: Dict[str, str] = dict(view.bodies)
        # A slimmed carried entry the request turn asked for comes back whole.
        for key in request_product.requested_entry_keys:
            if key in transfer_found and union.get(key) != transfer_found[key]:
                union[key] = transfer_found[key]
                if view.slim is not None:
                    view.slim.setdefault("whole_on_request", []).append(key)
        fresh_keys = [
            key
            for key in request_product.requested_entry_keys
            if key not in union
        ]
        if fresh_keys:
            fresh_found, _fresh_missing = load_entry_texts(
                run.knowledge_root, fresh_keys
            )
            for key, body in fresh_found.items():
                if key not in union:
                    union[key] = body
        union = dict(list(union.items())[:KB_WINDOW_TOTAL_ENTRIES])
        outcome = run_window_attempts(
            run,
            current,
            previous_advice=previous_advice,
            entries=_window_entries(run, view, union),
            chained=True,
            add_tail=lambda half: (
                windows.insert(i + 1, half),
                run.progress and run.progress.add_units(1),
            ),
            exchange_block=window_block,
        )
        if outcome.restart_halves is not None:
            # Final-attempt truncation split: both halves re-enter the
            # window list as fresh units, with their own retry budgets
            # and their own resume-replay checks.
            windows[i : i + 1] = list(outcome.restart_halves)
            if run.progress is not None:
                run.progress.add_units(len(outcome.restart_halves) - 1)
            continue
        run.commit_window(outcome.window, outcome.validation, outcome.next_advice)
        run.carried.keys = outcome.next_transfer
        i += 1
