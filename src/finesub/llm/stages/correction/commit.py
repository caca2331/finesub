"""What survives a rerun, and what invalidates it.

The window ledger (`correction-windows.jsonl`), the boundary plan next to it,
and the two hashes that decide whether a committed window may be replayed: the
task fingerprint (does this task still have the same inputs) and the per-window
input hash (is this the same window). :class:`ResumeLedger` in ``context`` is
the only thing that writes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Mapping

from ...chunking import render_window_preceding_as_csv, render_window_segments_as_csv
from ...chunking import SubtitleWindow


def _entry_details_signature(text: str) -> str:
    """Identity of the rendered entry block a window was given.

    Only the signature enters the per-window input hash: the entry *bodies*
    come from a base that auto-commits between windows, and hashing them would
    let any other task's knowledge update discard this task's progress.
    """

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# Mid-loop resume: each successfully committed window's raw response is appended
# here so a rerun replays completed windows instead of re-calling the LLM.
WINDOW_CACHE_FILENAME = "correction-windows.jsonl"
# The persisted window plan (model-routing v2): chunk-id stability across reruns.
WINDOW_PLAN_FILENAME = "correction-window-plan.json"


# Resume identity has three layers, deliberately of different strictness.
#
#   L1  one uncommitted LLM call        exact input hash (session_checkpoint.py)
#   L2  one committed correction window WINDOW_INVALIDATION_INPUTS, below
#   L3  one committed stage artifact    research.research_reuse_key
#
# L1 is never relaxed: docs/llm_local_agent.md §4 makes its hash the submit
# compare-and-swap token and §12 makes it the single commit gate for the coming
# task runtime. L2 and L3 answer a different question -- not "was the config
# identical" but "can this committed result still be explained by the same
# source and structure" (docs/llm_local_agent.md §11: changing execution
# identity may invalidate an unsubmitted worker call, never a committed
# research/window/stage).
#
# L2 is an *include* list on purpose. A field nobody has classified stays out of
# the tuple below and therefore out of the fingerprint... which is the wrong
# default, so `_task_fingerprint` asserts the tuple covers its own payload and
# `test_llm_correction_translation` pins the tuple's contents. Forgetting to
# classify a new field then fails a test rather than silently reusing a window
# it should have invalidated.
WINDOW_INVALIDATION_INPUTS = (
    # The output contract itself; a bump invalidates by design (CLAUDE.md).
    "prompt_version",
    "test_profile",
    # The source subtitles, parsed rather than byte-hashed.
    "source_fingerprint",
    # Source media path+size. Deliberately not mtime: re-downloading or touching
    # the same audio is not a content change, and mtime made it look like one.
    "media_identity",
    # Fast mode's seeded round-1 injections.
    "extra",
)
#
# Everything else is whitelisted, each for a stated reason:
#
#   extra_style / style   the run's translation style: the free-text one and
#                         the named entries. Changing either mid-run leaves the
#                         file half in one voice and half in another — a
#                         difference the user CAN see, and owner 2026-09-02
#                         judged it acceptable rather than pay for discarding
#                         every finished window. (`translation-style-plan.md`
#                         §2.5 records the decision and what it overrides.)
#
#   execution_identity     model, preset, model group, thinking, policy, agent
#                          driver/effort/timeout. docs/llm_local_agent.md §11.
#                          This one line is what makes "switch model group and
#                          continue" work.
#   difficulty / variant   already per-window: each record carries the variant
#                          it really used and replays through that validator.
#   correction_media,      geometry knobs. The persisted boundary plan keeps
#   retrieval, continuity, chunk ids stable; pending leaves are refit against
#   output_scale, limits   the current envelope before dispatch.
#   knowledge entries      the knowledge base auto-commits, so any other task's
#                          update would otherwise discard this task's progress.
#                          The version actually used is recorded per window
#                          instead (docs/llm_local_agent.md §8).
#   task_update_feedback   only asks for an extra output block.
#   context_pack           window notes are addressed by source-id intervals.


def _task_fingerprint(
    *,
    prompt_version: str,
    test_profile: bool,
    source_fingerprint: str,
    media_identity: Mapping[str, Any],
    extra: str,
) -> str:
    """Fingerprint the inputs that make a *committed* window unusable."""

    payload = {
        "prompt_version": prompt_version,
        "test_profile": bool(test_profile),
        "source_fingerprint": source_fingerprint,
        "media_identity": dict(media_identity or {}),
        "extra": extra or "",
    }
    unclassified = set(payload) - set(WINDOW_INVALIDATION_INPUTS)
    if unclassified:
        raise AssertionError(
            "fingerprint inputs missing from WINDOW_INVALIDATION_INPUTS: "
            + ", ".join(sorted(unclassified))
        )
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
    )


def _media_identity(path: str | Path | None) -> Dict[str, Any]:
    """Local source identity: path and size, never mtime.

    A re-download or a `touch` produces the same audio with a new mtime;
    treating that as a new source discarded every completed window.
    """

    if path is None:
        return {}
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "exists": False}
    return {"path": str(resolved), "exists": True, "size": stat.st_size}


def _window_input_hash(window: SubtitleWindow, entry_details_sig: str = "") -> str:
    """Hash a window's exact model input (ids, timings, text, clip origin,
    read-only preceding context, plus all injected entry keys+bodies)."""

    text = render_window_segments_as_csv(window)
    preceding = render_window_preceding_as_csv(window)
    return (
        "sha256:"
        + hashlib.sha256(
            f"{preceding}\x1f{text}\x1f{entry_details_sig}".encode("utf-8")
        ).hexdigest()[:16]
    )


def _load_window_cache(path: Path, task_fingerprint: str) -> Dict[str, Dict[str, Any]]:
    """Load cached window records matching the stable source (last wins)."""

    if not path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
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
        if record.get("task_fingerprint") != task_fingerprint:
            continue
        chunk_id = record.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            cache[chunk_id] = record
    return cache


def _load_parallel_entry_set(
    path: Path | None, task_fingerprint: str
) -> List[str] | None:
    """The barrier's persisted session entry set for this fingerprint, if any.

    Recorded once per parallel task at the first barrier; a partial resume
    reuses it instead of recomputing the set from the *pending* windows only
    (which shrank it and split one task's "fixed" set in two). Keys are
    pinned; bodies re-render from the current KB (A.5 (5) exemption).
    """

    if path is None or not path.exists():
        return None
    result: List[str] | None = None
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
        if record.get("task_fingerprint") != task_fingerprint:
            continue
        keys = record.get("parallel_entry_set")
        if isinstance(keys, list):
            result = [str(key) for key in keys if isinstance(key, str) and key]
    return result


# Serializes the window-cache append: the JSONL semantics are append-only and
# concurrency-safe, the bare write was not (docs/llm_harness_behavior.md).
_WINDOW_CACHE_LOCK = threading.Lock()


def _append_window_cache(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _WINDOW_CACHE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.part{path.suffix}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
