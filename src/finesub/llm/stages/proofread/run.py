"""Pass 3's calls: one per correction window, each window's answer kept, the changes written into the rows.

Each window is asked once, and again once when the reply does not read -- with the reason, so the model can
fix it (the trial sent the same message twice). Quota-type errors stop the pass. A window whose answer is in
is kept in `proofread.json` beside the task's artifacts, under the hash of what it was given (`_input_digest`:
its lines, the title, the preparation task's context, the style, its knowledge terms -- not the prompt's
wording): a rerun asks only the windows that failed or whose inputs changed, and the pass is `done` once every
window has answered. An answer kept from another prompt version is reused and reported (`older_prompt`).
Changes from the windows that answered are applied even when another failed; the failed windows keep pass 2's
lines, and the status says `failed` so a rerun asks them again.

A change rewrites a line's corrected text and translation; its source ids, kind, times, doubt and note
stay -- pass 3 never touches the split.
"""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ...client import extract_token_distribution
from ...degeneration import pick_after_retry, thinking_of, zero_thinking_record
from ...exchange_log import ExchangeLogger
from ...exchange_metadata import llm_exchange_metadata
from ...knowledge.base import append_task_artifact
from ...output_protocol import TranslatedCsvSegment
from ...prompt_compose import load_prompt_template
from ...routing.config import LLMRole
from ...routing.model_router import FailureKind, classify_failure
from ...run_context import bind_llm_worker, llm_worker_context
from ...token_books import append_call_error, call_trace
from .answers import AnswerError, Change, answer_of, parse_changes
from .prompt import Line, Window, proofread_prompt, window_context_of

#: Pass 3's own prompt version, recorded with each call.
PROMPT_VERSION = "proofread-v10"
SIDECAR_NAME = "proofread.json"
#: What every window was built from, in the task artifact dir: `tools/session_replay`'s `proofread` session
#: rebuilds a window's prompt from it (docs/session_replay.md).
INPUT_NAME = "proofread-input.json"
SIDECAR_VERSION = 1
RETRY_TEMPLATE = "proofread_retry_v1.md"
#: Pass 3 reads text only.
TASK_GROUP = "correction-text"
VALIDATOR_ID = "proofread-window"
DONE = "done"
FAILED = "failed"


class ProofreadStopped(RuntimeError):
    """The provider said "not now" (quota, rate limit): stop the pass. ``kept`` is the window's entry when it
    already held an answer -- one waiting on its retry for missing thinking."""

    def __init__(self, message: str, kept: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.kept = kept


@dataclass(frozen=True)
class ProofreadInputs:
    """What every window is told: the material's title, the preparation task's context, the task's style
    (`prompt.style_text`), and per correction window the knowledge base's names and terms it used."""

    title: str
    context_pack: Any
    style: str
    kb_terms: Mapping[str, str] = field(default_factory=dict)


@dataclass
class ProofreadResult:
    """The rows after pass 3, and what happened: `status` is `done` when every window answered; `changes` lists
    each line it changed (1-based row, window, source ids, category, before and after, why)."""

    segments: list[TranslatedCsvSegment]
    status: str
    changes: list[dict[str, Any]]
    windows: int = 0
    reused: int = 0
    failed: list[str] = field(default_factory=list)
    stopped: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    #: reused windows that another prompt version answered: version -> count
    older_prompt: dict[str, int] = field(default_factory=dict)


def sidecar_path(artifact_dir: str | Path) -> Path:
    return Path(artifact_dir) / SIDECAR_NAME


def proofread_status(artifact_dir: str | Path) -> str | None:
    """The last pass 3's status in this task (`done` / `failed`), None when it never ran or its record does
    not read."""
    try:
        payload = json.loads(sidecar_path(artifact_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    status = payload.get("status") if isinstance(payload, dict) and payload.get("version") == SIDECAR_VERSION else None
    return status if status in (DONE, FAILED) else None


def proofread_owed(artifact_dir: str | Path) -> bool:
    """Whether this task's pass 3 started and is not `done`: failed, or stopped half way (a rerun of a failed
    pass 3 interrupted before its end leaves the status empty), or a record that does not read."""
    return sidecar_path(artifact_dir).exists() and proofread_status(artifact_dir) != DONE


def mark_failed(artifact_dir: str | Path) -> None:
    """Record that the last pass 3 did not finish, keeping the windows' answers: a pass 3 that could not run at
    all must not leave an earlier run's `done` standing."""
    path = sidecar_path(artifact_dir)
    payload = {"version": SIDECAR_VERSION, "status": FAILED, "windows": _stored(path)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    temporary.replace(path)


def planning_limits(difficulty: str) -> Any:
    """Pass 3's group's input envelope (`routing.config.planning_limits_for`); None with no route declaration to
    ask."""
    try:
        from ...routing.config import planning_limits_for

        return planning_limits_for(TASK_GROUP, difficulty)
    except Exception:  # noqa: BLE001 - nothing to check against
        return None


def lines_of(segments: Sequence[TranslatedCsvSegment], seconds: Sequence[float],
             asr: Mapping[str, str]) -> list[Line]:
    """The finished rows as pass 3 reads them: `seconds` each stays on screen (in row order), `asr` the ASR
    text by source id (a row's is its sources' joined)."""
    from ...grouping import joined_text

    if len(seconds) != len(segments):
        raise ValueError(f"{len(segments)} rows but {len(seconds)} cues")
    return [Line(seconds=float(shown), asr=joined_text(asr.get(str(key), "") for key in segment.source_ids),
                 corrected=segment.corrected_text, translation=segment.translation, conf=segment.conf,
                 note=segment.note) for segment, shown in zip(segments, seconds)]


def run_proofread(
    segments: Sequence[TranslatedCsvSegment],
    lines: Sequence[Line],
    windows: Sequence[Window],
    *,
    inputs: ProofreadInputs,
    client: Any,
    profile: Any,
    artifact_dir: str | Path,
    task_id: str,
    parallel: int = 1,
    resume: bool = True,
    retries: int | None = None,
    warn: Callable[[str], None] = lambda message: None,
) -> ProofreadResult:
    """Ask every window the record does not already hold an answer for (`resume=False`: every window), then
    apply every window's changes. `retries` is the calls a window may make after its first (a reply that does
    not read is asked again with the reason): the correction windows' `max_retries_per_window`."""
    artifact_dir = Path(artifact_dir)
    path = sidecar_path(artifact_dir)
    stored = _stored(path) if resume else {}
    prompts = {window.chunk_id: proofread_prompt(window, lines, title=inputs.title, context_pack=inputs.context_pack,
                                                 style=inputs.style, kb_terms=inputs.kb_terms.get(window.chunk_id, ""))
               for window in windows}
    digests = {window.chunk_id: _input_digest(window, lines, inputs) for window in windows}
    _dump_inputs(artifact_dir, lines, windows, inputs, profile=profile, task_id=task_id)
    entries: dict[str, dict[str, Any]] = {}
    todo = []
    older: dict[str, int] = {}
    for window in windows:
        before = stored.get(window.chunk_id)
        if isinstance(before, dict) and before.get("status") == "success" \
                and before.get("input") == digests[window.chunk_id]:
            entries[window.chunk_id] = before
            version = str(before.get("prompt_version") or "（未记录）")
            if version != PROMPT_VERSION:
                older[version] = older.get(version, 0) + 1
        else:
            todo.append(window)
    session = _Session(client, profile, artifact_dir, task_id, retries=_retries(retries))
    lock = threading.Lock()
    stopped: list[str] = []

    def write(status: str = "") -> None:
        payload = {"version": SIDECAR_VERSION, "status": status, "windows": entries}
        temporary = path.with_name(f".{path.name}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)

    def one(window: Window) -> None:
        if session.stopped.is_set():
            return
        system, user = prompts[window.chunk_id]
        try:
            entry = _ask(session, window, system, user, digests[window.chunk_id])
        except ProofreadStopped as stop:
            with lock:
                stopped.append(str(stop))
                if stop.kept is not None:
                    entries[window.chunk_id] = stop.kept
                    write()
            session.stopped.set()
            return
        with lock:
            entries[window.chunk_id] = entry
            write()

    with lock:
        write()
    worker_context = llm_worker_context()
    try:
        with ThreadPoolExecutor(max(1, parallel), thread_name_prefix="llm-proofread",
                                initializer=bind_llm_worker, initargs=(worker_context,)) as pool:
            list(pool.map(one, todo))
    finally:
        worker_context.release_lanes()
    failed = [window.chunk_id for window in windows if entries.get(window.chunk_id, {}).get("status") != "success"]
    status = FAILED if failed else DONE
    with lock:
        write(status)
    rows, changes = apply_changes(segments, windows, entries)
    if failed:
        warn(f"校对（pass 3）有 {len(failed)} 个窗没有答好（{', '.join(failed)}）" + (f"；停在 {stopped[0]}" if stopped else ""))
    return ProofreadResult(rows, status, changes, windows=len(windows), reused=len(windows) - len(todo),
                           failed=failed, stopped=stopped[0] if stopped else "", usage=session.usage,
                           older_prompt=older)


def _input_digest(window: Window, lines: Sequence[Line], inputs: ProofreadInputs) -> str:
    """What a window's answer depends on, generously: the window (its rows and read-only neighbours), every
    line it shows whole, the title, the preparation task's whole general context, the user's requests it
    relayed and its background for this window, the style and the window's knowledge terms -- what
    `_dump_inputs` records. Not the rendered prompt: a change to the templates' wording alone keeps the answers.
    A task without relayed requests, or a window without its own background, digests as before they were read."""
    general = getattr(inputs.context_pack, "general_context", None) or {}
    requests = list(getattr(inputs.context_pack, "user_requests", ()))
    own = window_context_of(inputs.context_pack, window.source_ids)
    shown = (*window.before, *window.rows, *window.after)
    payload = {"window": asdict(window), "lines": [asdict(lines[index]) for index in shown], "title": inputs.title,
               "general_context": general, **({"user_requests": requests} if requests else {}),
               **({"window_context": own} if own else {}), "style": inputs.style,
               "kb_terms": inputs.kb_terms.get(window.chunk_id, "")}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _dump_inputs(artifact_dir: Path, lines: Sequence[Line], windows: Sequence[Window], inputs: ProofreadInputs, *,
                 profile: Any, task_id: str) -> None:
    # pass 3 reads the preparation task's general context (`prompt.general_blocks`; a chunked one is picked by
    # window, which needs the chunks' ranges) and each window's own background, which needs the source order
    pack = inputs.context_pack
    payload = {"version": 2, "profile_id": getattr(profile, "profile_id", ""), "task_id": task_id,
               "title": inputs.title, "context_pack": pack.to_dict() if pack is not None else {},
               "source_order": list(getattr(pack, "source_order", ())), "style": inputs.style,
               "kb_terms": dict(inputs.kb_terms), "lines": [asdict(line) for line in lines],
               "windows": [asdict(window) for window in windows]}
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / INPUT_NAME).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def apply_changes(segments: Sequence[TranslatedCsvSegment], windows: Sequence[Window],
                  entries: Mapping[str, Mapping[str, Any]]) -> tuple[list[TranslatedCsvSegment], list[dict[str, Any]]]:
    """The rows with each answered window's changes written in, and the changes that changed something."""
    rows = list(segments)
    changes = []
    for window in windows:
        entry = entries.get(window.chunk_id) or {}
        if entry.get("status") != "success":
            continue
        for change in entry.get("changes") or ():
            index = window.rows[int(change["position"]) - 1]
            row = rows[index]
            corrected, translation = str(change["corrected"]), str(change["translation"])
            if corrected == row.corrected_text and translation == row.translation:
                continue
            rows[index] = replace(row, corrected_text=corrected, translation=translation)
            changes.append({
                "row": index + 1, "window": window.chunk_id, "source_ids": list(row.source_ids),
                "category": int(change["category"]), "why": str(change.get("why") or ""),
                **({"corrected": [row.corrected_text, corrected]} if corrected != row.corrected_text else {}),
                **({"translation": [row.translation, translation]} if translation != row.translation else {}),
            })
    return rows, changes


def _retries(retries: int | None) -> int:
    from ....settings import MAX_RETRIES_PER_WINDOW

    return MAX_RETRIES_PER_WINDOW.resolve(retries)


def _ask(session: "_Session", window: Window, system: str, user: str, digest: str) -> dict[str, Any]:
    """One window's calls: its record entry (under `digest`, `_input_digest`), `success` with the changes or
    `failed`. The window's rounds after
    its first call -- asking again with the reason a reply did not read, the repair rounds an agent session ran
    inside one call, a transient error -- share one budget, `session.retries`. A reply that read but reported no
    thinking although it was asked to gets one more call in a fresh session past that budget, and the window
    keeps one of the two (`degeneration.pick_after_retry`, owner 2026-10-17)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    error = ""
    spent = attempt = 0
    held: dict[str, Any] | None = None

    def success(answer: dict[str, Any], thinking: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"status": "success", "input": digest, "model": answer["model"],
                "prompt_version": PROMPT_VERSION, "changes": [_change_fields(change) for change in answer["changes"]],
                **({"thought_stripped": answer["stripped"]} if answer["stripped"] else {}),
                **({"thinking": dict(thinking)} if thinking is not None else {})}

    while spent <= session.retries or held is not None:
        name = f"proofread-{window.chunk_id}-attempt{attempt}"
        attempt += 1
        try:
            result = session.call(name, messages if held is None else messages[:2], count=len(window.rows),
                                  repairs=max(0, session.retries - spent), fresh=held is not None)
        except ProofreadStopped as stop:
            if held is not None:  # the answer it had stands; the stop goes on up
                raise ProofreadStopped(str(stop), kept=success(held, zero_thinking_record(
                    held["thinking"], None, "original", error=f"stopped: {stop}"))) from stop
            raise
        except Exception as failure:  # noqa: BLE001 - a transient error: try again
            if held is not None:
                return success(held, zero_thinking_record(held["thinking"], None, "original",
                                                          error=f"{type(failure).__name__}: {failure}"))
            error = f"{type(failure).__name__}: {failure}"
            spent += 1
            continue
        spent += int(getattr(result, "agent_repair_rounds", 0) or 0)
        reply = result.content or ""
        text = answer_of(reply)  # reasoning shipped as the reply: the answer at its end
        try:
            changes = parse_changes(text, len(window.rows))
        except AnswerError as refused:
            if held is not None:
                return success(held, zero_thinking_record(held["thinking"], thinking_of(result), "original"))
            error = str(refused)
            messages = [*messages[:2], {"role": "assistant", "content": reply},
                        {"role": "user", "content": load_prompt_template(RETRY_TEMPLATE, strict=True, error=error)}]
            spent += 1
            continue
        answer = {"model": result.model, "changes": changes, "stripped": len(reply) - len(text),
                  "thinking": thinking_of(result)}
        if held is not None:
            chosen = pick_after_retry(0, answer["thinking"], 0)
            return success(answer if chosen == "retry" else held,
                           zero_thinking_record(held["thinking"], answer["thinking"], chosen))
        if answer["thinking"].skipped:
            held = answer
            continue
        return success(answer)
    return {"status": "failed", "input": digest, "error": error, "prompt_version": PROMPT_VERSION}


def _change_fields(change: Change) -> dict[str, Any]:
    return {"position": change.position, "category": change.category, "corrected": change.corrected,
            "translation": change.translation, "why": change.why}


def _stored(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict) or payload.get("version") != SIDECAR_VERSION:
        return {}
    windows = payload.get("windows")
    return windows if isinstance(windows, dict) else {}


class _Session:
    """The calls of one pass 3: routing, logging, usage, the stop signal."""

    def __init__(self, client: Any, profile: Any, artifact_dir: Path, task_id: str, *, retries: int) -> None:
        self.client, self.profile, self.artifact_dir, self.task_id = client, profile, artifact_dir, task_id
        self.retries = retries
        self.logger = ExchangeLogger.for_task_artifact_dir(artifact_dir)
        self.stopped = threading.Event()
        self.usage: dict[str, int] = {}
        self._lock = threading.Lock()

    def call(self, name: str, messages: list[dict[str, str]], *, count: int, repairs: int,
             fresh: bool = False) -> Any:
        """One call; ``fresh`` asks for a new session (a `resume` lane's conversation is retired there)."""
        try:
            result = self.client.complete(
                LLMRole.AUDIO_MULTIMODAL, messages, task_group=TASK_GROUP, difficulty=self.profile.difficulty,
                validator_spec={"id": VALIDATOR_ID, "params": {"count": count}}, max_repair_attempts=repairs,
                **({"fresh_session": True} if fresh else {}))
        except Exception as error:  # noqa: BLE001 - classified below; the window decides what to do
            append_call_error(self.artifact_dir, "proofread_call_error", self.task_id, name, error)
            if classify_failure(error) in (FailureKind.QUOTA, FailureKind.RATE_LIMIT):
                raise ProofreadStopped(f"{type(error).__name__}: {error}") from error
            raise
        tokens = extract_token_distribution(result.raw_response)
        with self._lock:
            for key, value in tokens.items():
                self.usage[key] = self.usage.get(key, 0) + int(value or 0)
        append_task_artifact(self.artifact_dir, kind="proofread_response", task_id=self.task_id,
                             payload={"call": name, "model": result.model, "usage": tokens, **call_trace(result)})
        if self.logger is not None:
            self.logger.log(name, messages=messages, response_text=result.content,
                            metadata=llm_exchange_metadata(result, session=name, prompt_version=PROMPT_VERSION))
        return result
