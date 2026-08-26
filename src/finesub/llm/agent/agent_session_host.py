"""Pseudo-conversational sessions: one CLI serving a run's tasks in turn.

docs/llm_local_agent.md §12.1.3 and the plan in
docs/llm_followups.md "Agent 会话档位收敛与 pseudo/conversational 接线":

* a **host** owns one long-lived CLI invocation and one unsealed assignment
  (`session_scope=assignment`, one worker). The harness adds a task per
  call, the agent takes it over the MCP server, submits, and comes back for
  the next one; `next_task` parks (long poll) while the harness is between
  windows. The CLI runs on a background *supervisor* thread for the whole
  session; `driver.run()` is called once, with a completion predicate that
  only turns true once the assignment is sealed and every task is terminal
  (or the host gave the session up);
* every `complete()` only waits for **its own** task: the durable task row is
  the authority (`accepted` is the only completion point). A CLI that dies
  mid-task is handled per task -- first time the lease is kept and a fresh
  CLI resumes it (`reset_conversation`), second time the task is retired and
  the call fails to the route chain, while tasks still queued wait for the
  restarted session;
* a spent repair budget withdraws the task and reports `repair_exhausted`;
  the caller's replacement (`fresh_session=True`) ends this CLI first, so the
  replacement really does start a new conversation;
* usage is **session-level** (owner decision 2026-08-22, fifth review): the
  three CLI dialects report one terminal usage event per invocation, so a
  per-task split would be invented. Each task's execution attempt carries
  ``usage_attribution="session"``; the session total is reported when the
  host closes;
* the **registry** keys hosts by ``(provider tier, model, lane, mode)`` for
  the length of a run (`agent_session_scope`); closing it seals every
  assignment, waits a bounded time for the CLIs to leave on their own, and
  reclaims the ones that do not.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
import hashlib
import functools
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, TypeVar
import uuid

from finesub_bootstrap.locks import holding_activity

from finesub.reporting import current_reporter, reporter_delivers

from .agent_mcp_server import TOOL_NAMES, WEB_TOOL_NAMES
from .agent_task_runtime import (
    AgentTaskRuntime,
    AgentTaskSpec,
    AgentTaskRuntimeError,
    StaleLeaseError,
    lease_ttl_for,
)
from .agent_paths import resolve_evidence_locator
from .agent_transports import AgentRuntimeCallError
from .agent_validators import VALIDATOR_BUILDERS, runtime_validators
from .local_agent import ACCEPTED_EXIT_GRACE_SECONDS
from ..prompts import agent_tool_worker_session_bootstrap

_F = TypeVar("_F", bound=Callable[..., Any])

WORKER_ID = "worker-1"
# How often a waiting call re-reads the task row. A file read under the
# runtime lock; cheap next to a model turn.
_POLL_SECONDS = 0.25
# A session bounds itself per task (the caller's timeout); the single CLI
# invocation underneath must outlive the whole run.
SESSION_CALL_TIMEOUT_SECONDS = 7 * 24 * 3600.0
# Fresh CLI sessions a task may get after the one serving it left without
# submitting (docs/llm_agent_tool_protocol.md §4: a premature stop is a
# transport fault, not a replacement).
PREMATURE_STOP_RETRIES_PER_TASK = 1
# How long `close()` waits for the CLI to leave by itself after the seal
# before reclaiming it through the completion predicate.
CLOSE_GRACE_SECONDS = ACCEPTED_EXIT_GRACE_SECONDS + 10.0
# How long a conversational queue keeps its tree after sealing. A person's
# agent is not a process the harness can wait on: what it can do is leave the
# assignment readable for one more poll, so the agent is told
# `assignment_complete` instead of finding its working directory gone.
CONVERSATIONAL_SEAL_GRACE_SECONDS = 5.0
# How long a conversational call waits with nobody holding its task before it
# gives up. Not a setting: its only job is to stop a run hanging for ever, and
# every value a person might reach for it is worse than the answer that already
# exists -- run the speech stages first and the correction once you are at the
# keyboard (docs/manual/agent.md). Generous, because the cost of waiting is
# nothing and the cost of giving up early is the run.
CONVERSATIONAL_JOIN_WAIT_SECONDS = 3600.0
# How many times one task may be claimed without ever being accepted. An agent
# that takes the task and walks away is not the same failure as nobody coming,
# and counting seconds cannot tell them apart: each claim resets the wait, so
# without a cap a flapping agent keeps the run alive for ever. Three is two
# second chances.
MAX_CLAIMS_PER_TASK = 3


def _lease_is_live(record: Mapping[str, Any], now: float) -> bool:
    """Is somebody holding this task right now?

    Reads do not reclaim, so an owner alone proves nothing: the row keeps
    naming a worker that stopped renewing until some write sweeps it. The
    deadline is what separates "on the job" from "gone".
    """

    return bool(record.get("lease_owner")) and float(
        record.get("lease_expires_at") or 0.0
    ) > now


def absolute_pythonpath(value: str) -> str:
    """`PYTHONPATH` with every entry made absolute.

    The MCP server process inherits it from the CLI, which runs in its own
    working directory (agy: the slot project); a checkout's relative `src`
    would resolve there and the server would fail to import the package --
    measured 2026-08-22, the CLI then retried an invented tool name 18 times.
    """

    return os.pathsep.join(
        str(Path(entry).resolve()) if entry.strip() else entry
        for entry in value.split(os.pathsep)
    )


def write_audit_bundle(
    runtime: Any,
    *,
    result: Any,
    root: Path,
    assignment_id: str,
    worker_id: str,
    record: Mapping[str, Any],
    accepted_text: str,
    error: str = "",
    audit_name: str = "audit",
) -> bool:
    """The self-contained audit bundle of one tool session (docs §0-4).

    Written into the episode's capsule under ``audit_name`` so it follows
    the capsule's retention (kept on failure, pruned on success) and outlives
    the assignment root, which is deleted once the artifact is in hand:
    the manifest, every block the agent could read (the bodies, not just
    their digests), the pull ledger, the last rejected candidate with its
    errors, the accepted artifact and every JSON-RPC frame. Written into a
    temporary directory and renamed into place, so a bundle either exists
    whole or not at all. Returns whether it exists: a bundle that cannot be
    written must not fail a call that succeeded, but the caller must then
    keep the runtime root -- it is the only evidence left. A call whose
    capsule was already pruned (the driver's own rule for a clean run) has
    no locator and nothing to write: that is not a failure, and
    ``capsule_retained`` is how a caller tells the two apart.
    """

    locator = dict(getattr(result, "execution_attempt", {}) or {}).get("evidence_locator")
    if not isinstance(locator, Mapping):
        return False
    staging: Path | None = None
    try:
        episode = resolve_evidence_locator(locator)
        final = episode / audit_name
        audit = episode / f".audit-{uuid.uuid4().hex[:8]}"
        staging = audit
        audit.mkdir(parents=True, exist_ok=False)
        final.parent.mkdir(parents=True, exist_ok=True)
        manifest_ref = str(record.get("manifest_ref") or "")
        manifest = json.loads(runtime.read_artifact(manifest_ref)) if manifest_ref else {}
        (audit / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        blocks = audit / "blocks"
        blocks.mkdir(exist_ok=True)
        for block in manifest.get("required_blocks") or []:
            ref = str(block.get("ref") or "")
            if ref:
                (blocks / f"{block['kind']}.md").write_text(
                    runtime.read_artifact(ref), encoding="utf-8"
                )
        ledger = runtime.pull_status(
            assignment_id=assignment_id,
            task_id=str(record.get("task_id") or "call"),
            worker_id=worker_id,
        )
        conversation = runtime.conversation_state(
            assignment_id=assignment_id, worker_id=worker_id
        )
        outcome = {
            # The durable state *as it is now* -- after any reset/retire --
            # so the bundle agrees with the runtime it outlives.
            "task": dict(record),
            "ledger": ledger,
            "conversation": {
                "epoch": conversation.get("conversation_epoch"),
                "resets": list(conversation.get("resets") or []),
            },
            "accepted": bool(accepted_text),
            "error": error,
        }
        (audit / "outcome.json").write_text(
            json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        if accepted_text:
            (audit / "artifact.txt").write_text(accepted_text, encoding="utf-8")
        frames = root / "control" / "mcp-frames.jsonl"
        if frames.is_file():
            shutil.copyfile(frames, audit / "mcp-frames.jsonl")
        if final.exists():
            _remove_tree(final)
        audit.rename(final)
        return True
    except (OSError, ValueError, KeyError, TypeError, AgentTaskRuntimeError) as exc:
        # The runtime's own refusals count too: a task that is accepted (or
        # was withdrawn) no longer has a lease, and reading its ledger must
        # not turn a written answer into a failed call.
        current_reporter().warning(
            "agent-audit-bundle",
            f"could not write the agent audit bundle for {assignment_id}: {exc}",
        )
        if staging is not None and staging.exists():
            try:
                _remove_tree(staging)
            except OSError:
                pass
        return False

def capsule_retained(result: Any) -> bool:
    """Is there still an episode this call could be audited into?

    A driver prunes the capsule of a clean run and drops the locator with
    it, so "no bundle" then means "nothing left to write", not "the write
    failed". Only the latter is a reason to keep an assignment root.
    """

    locator = dict(getattr(result, "execution_attempt", {}) or {}).get("evidence_locator")
    return isinstance(locator, Mapping)


def _remove_tree(path: Path) -> None:
    from finesub_bootstrap.fsops import remove_tree

    remove_tree(path)


def discard_assignment_root(root: Path) -> None:
    """Remove an assignment tree nothing will read again.

    A session that ended with every task accepted leaves the same evidence a
    single-task tool session does -- the capsule. What is here is the run's
    whole subtitle text plus every MCP frame, and nothing bounds how many of
    those accumulate.
    """

    try:
        _remove_tree(root)
    except OSError as exc:  # noqa: BLE001 -- leftovers are `agent-clean`'s job
        current_reporter().debug("agent-session-keep", f"could not remove {root}: {exc}")


EVIDENCE_DIRECTORY_NAME = "evidence"


def discard_assignment_body(root: Path) -> None:
    """Remove the run's text but leave the control plane behind.

    A conversational worker parks in `await-next-task` for up to
    `CONVERSATIONAL_WATCH_SECONDS`, and the seal grace is five seconds, so a
    session that ends cleanly is *certain* to have somebody still waiting on
    it. Deleting the whole tree left that agent staring at a working
    directory that no longer exists instead of being told the assignment is
    complete. What has to go is the bulk -- this run's subtitle text, every
    protocol copy and manifest; what stays is the sealed index, its state and
    the evidence already lifted out of it, which together are a few KB and
    answer exactly one question: "we are done".
    """

    keep = {root / "control", root / EVIDENCE_DIRECTORY_NAME}
    try:
        for child in root.iterdir():
            if child not in keep:
                _remove_tree(child) if child.is_dir() else child.unlink(missing_ok=True)
        protocols = root / "control" / "protocols"
        if protocols.is_dir():
            _remove_tree(protocols)
    except OSError as exc:  # noqa: BLE001 -- leftovers are `agent-clean`'s job
        current_reporter().debug(
            "agent-session-keep", f"could not clear {root}: {exc}"
        )


CONVERSATIONAL_EVIDENCE_DIRNAME = "agent-conversational"


def file_conversational_evidence(artifact_dir: Path | str, registry: Any) -> list[Path]:
    """Move each conversational session's kept exchange into the artifacts.

    An API call leaves its prompt and answer where the run's other records
    are; a conversational call has them only inside the assignment tree that a
    clean finish clears, so without this a successful run left no trace of
    what was actually asked and answered.

    Called by `agent_session_scope` on the way out rather than by the two
    front ends that happen to know an artifact directory: a stage entered
    directly (`execute_correction_windows`, `run_reference_knowledge_update`)
    opens and closes its own scope, and its registry used to be dropped with
    the evidence still staged inside it. The destination is injected --
    `set_run_evidence_destination` -- because a session still must not learn
    where a run's artifacts go.
    """

    roots = [Path(root) for root in getattr(registry, "evidence_roots", list)()]
    if not roots:
        return []
    destination = Path(artifact_dir).expanduser().resolve() / CONVERSATIONAL_EVIDENCE_DIRNAME
    moved: list[Path] = []
    for root in roots:
        try:
            # Inside the guard: the scope's teardown may be running under an
            # exception, and a tree that went away in the meantime (another
            # front end's `agent-clean`) must not raise over it.
            task_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        except OSError as exc:  # noqa: BLE001
            current_reporter().debug(
                "agent-session-evidence", f"could not read {root}: {exc}"
            )
            continue
        for task_dir in task_dirs:
            target = destination / root.parent.name / task_dir.name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    # An artifact directory is reused across runs; the same
                    # task id from a previous one is stale, not a conflict.
                    _remove_tree(target)
                shutil.move(str(task_dir), str(target))
                moved.append(target)
            except OSError as exc:  # noqa: BLE001
                current_reporter().warning(
                    "agent-session-evidence",
                    f"could not file the conversational exchange {task_dir.name}: {exc}",
                )
    return moved


def set_run_evidence_destination(artifact_dir: Path | str | None) -> None:
    """Tell this run's scope where to file what its agent sessions kept.

    Whoever knows the artifact directory says so once; the scope does the
    filing when it closes. Safe to call from a stage that is running inside a
    larger run -- it is the run's registry either way, and every caller in one
    run names the same directory.
    """

    registry = current_registry()
    if registry is not None and artifact_dir:
        registry.set_evidence_destination(Path(artifact_dir))


def mcp_block_files(driver: Any) -> bool:
    """Whether this driver's CLI reads the task's blocks as files itself."""

    return bool(getattr(getattr(driver, "config", None), "mcp_block_files", False))


def mcp_page_chars(driver: Any) -> int:
    """The CLI's inline tool-reply limit for this driver (0 = unlimited)."""

    config = getattr(driver, "config", None)
    try:
        return max(0, int(getattr(config, "mcp_page_chars", 0) or 0))
    except (TypeError, ValueError):
        return 0


@dataclass
class TaskOutcome:
    """What one `complete()` gets back from the session for its task."""

    task_id: str
    started_at: float
    returned_at: float = 0.0
    record: dict[str, Any] = field(default_factory=dict)
    accepted_text: str = ""
    premature_stops: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class AgentSessionHost:
    """One pseudo-conversational session: a CLI, an assignment, a supervisor."""

    def __init__(
        self,
        driver: Any,
        *,
        root: Path,
        execution_identity: Mapping[str, Any],
        task_timeout_seconds: float,
        label: str = "",
    ) -> None:
        self.driver = driver
        # Absolute: the MCP server is spawned by the CLI in *its* working
        # directory (agy: the slot project), so a relative root would point
        # nowhere there and the session would start without tools.
        self.root = Path(root).resolve()
        self.label = label or self.root.name
        self.assignment_id = f"session-{uuid.uuid4().hex[:12]}"
        self.task_timeout_seconds = max(1.0, float(task_timeout_seconds))
        validators: dict[str, Any] = {}
        for validator_id in sorted(VALIDATOR_BUILDERS):
            validators.update(runtime_validators(validator_id))
        self.runtime = AgentTaskRuntime.start_assignment(
            self.root,
            assignment_id=self.assignment_id,
            worker_goal="serve the harness calls of one run",
            tasks=[],
            session_scope="assignment",
            execution_identity=dict(execution_identity),
            validators=validators,
            sealed=False,
            lease_ttl_seconds=lease_ttl_for(self.task_timeout_seconds),
        )
        self._lock = threading.RLock()
        self._input_hashes: dict[str, str] = {}
        self._task_seq = 0
        self._served_in_session = 0
        self._generation = 0
        self._thread: threading.Thread | None = None
        self._session_id = ""
        self._session_attempt: dict[str, Any] = {}
        self._session_result: Any = None
        self._session_error: BaseException | None = None
        self._session_ended = threading.Event()
        self._closing = False
        self._dead = False
        self.session_usage: list[dict[str, Any]] = []
        self.sessions_started = 0
        self.tasks_accepted = 0
        # Anything that left evidence worth reading later: a premature stop,
        # a withdrawal, a timeout, a retirement. Zero is what makes a session
        # disposable at close (docs/llm_agent_tool_protocol.md §5).
        self.incidents = 0

    # -- supervisor -----------------------------------------------------

    def _mcp_server_spec(self, session_id: str) -> dict[str, Any]:
        env = {
            "FINESUB_MCP_ROOT": str(self.root),
            "FINESUB_MCP_ASSIGNMENT": self.assignment_id,
            "FINESUB_MCP_WORKER": WORKER_ID,
            "FINESUB_MCP_SESSION": session_id,
            "FINESUB_MCP_LOG": str(self.root / "control" / "mcp-frames.jsonl"),
            # Exposure vs admission (docs §2): every tool is authorized at
            # launch, each call is admitted against the task it serves.
            "FINESUB_MCP_TOOLS": ",".join([*TOOL_NAMES, *WEB_TOOL_NAMES]),
        }
        if os.environ.get("PYTHONPATH"):
            env["PYTHONPATH"] = absolute_pythonpath(os.environ["PYTHONPATH"])
        env["FINESUB_MCP_WAIT_SECONDS"] = os.environ.get(
            "FINESUB_MCP_WAIT_SECONDS", ""
        ) or f"{self.next_task_wait_seconds():g}"
        env["FINESUB_MCP_PAGE_CHARS"] = os.environ.get("FINESUB_MCP_PAGE_CHARS", "") or str(
            mcp_page_chars(self.driver)
        )
        block_files = mcp_block_files(self.driver)
        if block_files:
            env["FINESUB_MCP_BLOCK_FILES"] = "1"
        return {
            "command": sys.executable,
            "args": ["-m", "finesub.llm.agent.agent_mcp_server"],
            "env": env,
            "tools": [*TOOL_NAMES, *WEB_TOOL_NAMES],
            # Where the CLI's file tool may read this call: the assignment
            # root, whose protocol/payload files the manifest names.
            "view_roots": [str(self.root)] if block_files else [],
        }

    def next_task_wait_seconds(self) -> float:
        """min(driver's measured ceiling, conversation TTL - 60s), at least 5s.

        Owner decision 2026-08-22 (second round): the long poll must end
        before the provider forgets the conversation it is keeping warm, and
        never outlast the CLI's own tool-call timeout.
        """

        config = getattr(self.driver, "config", None)
        ceiling = float(getattr(config, "next_task_wait_seconds", 25.0) or 25.0)
        ttl = float(getattr(config, "conversation_ttl_seconds", 0.0) or 0.0)
        if ttl > 0:
            ceiling = min(ceiling, ttl - 60.0)
        return max(5.0, ceiling)

    def _status(self) -> dict[str, Any]:
        return self.runtime.status(assignment_id=self.assignment_id, worker_id=WORKER_ID)

    def _completion(self) -> bool:
        if self._dead:
            return True
        try:
            return self._status()["status"] == "assignment_complete"
        except Exception:  # noqa: BLE001 -- the predicate must never kill the call
            return False

    def _parked(self) -> bool:
        try:
            return self._status()["status"] == "waiting"
        except Exception:  # noqa: BLE001
            return False

    # observer hook (`driver.run(observer=self)`)
    def started(self, attempt: Mapping[str, Any]) -> None:
        self._session_attempt = dict(attempt)

    def _supervise(self, session_id: str, run_kwargs: Mapping[str, Any]) -> None:
        bootstrap = [
            {
                "role": "user",
                "content": agent_tool_worker_session_bootstrap(
                    assignment_id=self.assignment_id, worker_id=WORKER_ID
                ),
            }
        ]
        try:
            self._session_result = self.driver.run(
                bootstrap,
                session_scope="task",
                mcp_server=self._mcp_server_spec(session_id),
                completion=self._completion,
                parked=self._parked,
                observer=self,
                timeout_seconds=SESSION_CALL_TIMEOUT_SECONDS,
                **run_kwargs,
                native_search=False,
            )
        except BaseException as exc:  # noqa: BLE001 -- reported to the waiter
            self._session_error = exc
        finally:
            self.session_usage.append(
                {
                    "session_id": session_id,
                    "usage": dict(getattr(self._session_result, "usage", {}) or {}),
                    "error": (
                        "" if self._session_error is None
                        else f"{type(self._session_error).__name__}: {self._session_error}"
                    ),
                }
            )
            self._session_ended.set()

    def _ensure_session(self, run_kwargs: Mapping[str, Any]) -> None:
        """Start the CLI if none is serving (first call, or after it left)."""

        if self._thread is not None and self._thread.is_alive():
            return
        if self._closing:
            raise AgentRuntimeCallError("the agent session is closing; no new task can be added")
        self._generation += 1
        self._session_id = uuid.uuid4().hex
        self._session_attempt = {}
        self._session_result = None
        self._session_error = None
        self._session_ended.clear()
        self._served_in_session = 0
        self._dead = False
        self.sessions_started += 1
        self._thread = threading.Thread(
            target=self._supervise,
            args=(self._session_id, dict(run_kwargs)),
            name=f"agent-session-{self.label}-{self._generation}",
            daemon=True,
        )
        self._thread.start()

    def _session_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _end_session(self, *, reason: str, wait: float = CLOSE_GRACE_SECONDS) -> None:
        """Make the running CLI leave: reclaim through the completion predicate."""

        if not self._session_alive():
            return
        self._dead = True
        self._session_ended.wait(wait + ACCEPTED_EXIT_GRACE_SECONDS)
        if self._session_alive():
            current_reporter().warning(
                "agent-session-reclaim",
                f"agent session {self.label} did not end within {wait:g}s after {reason}",
            )

    # -- tasks ----------------------------------------------------------

    def _record(self, task_id: str) -> dict[str, Any]:
        return self.runtime.task_record(assignment_id=self.assignment_id, task_id=task_id)

    def _harness_request_id(self, operation: str, task_id: str) -> str:
        material = f"{self._session_id}:{task_id}:{operation}".encode("utf-8")
        return "harness-" + hashlib.sha256(material).hexdigest()[:32]

    def _reset_task(self, record: Mapping[str, Any], reason: str) -> None:
        if not record.get("lease_owner"):
            return
        try:
            self.runtime.reset_conversation(
                assignment_id=self.assignment_id,
                task_id=str(record["task_id"]),
                worker_id=str(record["lease_owner"]),
                lease_generation=int(record["lease_generation"]),
                request_id=self._harness_request_id("reset", str(record["task_id"])),
                reason=reason,
            )
        except StaleLeaseError:
            return

    def _retire_task(self, record: Mapping[str, Any], reason: str) -> None:
        if not record.get("lease_owner"):
            return
        try:
            self.runtime.retire_task(
                assignment_id=self.assignment_id,
                task_id=str(record["task_id"]),
                worker_id=str(record["lease_owner"]),
                lease_generation=int(record["lease_generation"]),
                request_id=self._harness_request_id("retire", str(record["task_id"])),
                reason=reason,
            )
        except StaleLeaseError:
            return

    def _withdraw(self, task_id: str, reason: str) -> None:
        try:
            self.runtime.withdraw_task(
                assignment_id=self.assignment_id, task_id=task_id, reason=reason
            )
        except AgentTaskRuntimeError:
            pass

    def _task_attempt(self, outcome: TaskOutcome, *, index: int) -> dict[str, Any]:
        attempt = dict(self._session_attempt)
        attempt.update(
            {
                "backend": "local_agent",
                "driver": str(getattr(self.driver, "driver_id", attempt.get("driver", ""))),
                "task_id": outcome.task_id,
                "session_id": self._session_id,
                "session_task_index": index,
                "started_at": outcome.started_at,
                "returned_at": outcome.returned_at,
                "duration_ms": int(max(0.0, outcome.returned_at - outcome.started_at) * 1000),
                # The CLI reports usage once per invocation; a per-task split
                # would be invented (fifth review). Booked on the session.
                "usage": {"source": "session"},
                "usage_attribution": "session",
            }
        )
        return attempt

    def _result(self, outcome: TaskOutcome, content: str, *, index: int) -> Any:
        attempt = self._task_attempt(outcome, index=index)
        return SimpleNamespace(
            content=content,
            reported_model=str(
                attempt.get("reported_model")
                or getattr(getattr(self.driver, "config", None), "model", "")
                or "configured-default"
            ),
            episode_id=str(attempt.get("capsule_id") or ""),
            execution_attempt=attempt,
            normalized_events=(),
            usage={},
            conversation_handle="",
            turn_identity="",
        )

    def run_task(
        self,
        *,
        session_type: str,
        input_hash: str,
        validator_id: str,
        metadata: Mapping[str, Any],
        protocol_text: str,
        payload_text: str,
        retrieval_mode: str,
        run_kwargs: Mapping[str, Any],
        max_repairs: int,
        fresh_session: bool = False,
    ) -> tuple[Any, list[Mapping[str, Any]], bool, int, bool]:
        """Hand the session one task and wait for its verdict.

        Returns what the tool-session call returns: the result, the earlier
        sessions' attempts (premature stops), whether this task inherited an
        earlier task's history in the same CLI session (then the checkpoint
        is not resumable -- gate D answer C), repair rounds, exhausted.
        """

        with self._lock:
            if self._closing:
                raise AgentRuntimeCallError("the agent session is closing; no new task can be added")
            if fresh_session:
                # Tier 2: the replacement must not land in the conversation
                # it is escaping (docs/llm_local_agent.md §12.1.1).
                self._end_session(reason="replacement round")
            self._task_seq += 1
            task_id = f"call-{self._task_seq:04d}"
            context_key = f"payload-{task_id}"
            spec = AgentTaskSpec(
                task_id=task_id,
                session_type=session_type,
                input_hash=input_hash,
                goal="answer the harness call",
                validator_id=validator_id,
                protocol_key=session_type,
                context_key=context_key,
                retrieval_mode=retrieval_mode,
                metadata=dict(metadata),
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "payload", "digest": "@context"},
                ),
            )
            self._ensure_session(run_kwargs)
            inherited_history = self._served_in_session > 0
            self._input_hashes[task_id] = input_hash
            self.runtime.add_task(
                spec,
                protocol_documents={session_type: protocol_text},
                context_documents={context_key: payload_text},
            )
            outcome = TaskOutcome(task_id=task_id, started_at=time.time())
        # Deliberately outside the lock: a task's wait is as long as a model
        # turn, and `close()` -- which the run's scope may call from another
        # thread while this one is waiting -- must not queue behind it.
        return self._await(outcome, run_kwargs=run_kwargs, max_repairs=max_repairs,
                           inherited_history=inherited_history)

    def _audit(
        self,
        outcome: TaskOutcome,
        *,
        record: Mapping[str, Any],
        accepted_text: str,
        index: int,
        error: str = "",
    ) -> None:
        """One bundle per task, in the capsule of the CLI that served it.

        A session's tasks share one capsule, so they are told apart by name.
        The bundle follows the capsule's retention, which is what makes this
        session's assignment tree disposable at close: on a clean run both
        go, on a failing one both stay for `agent-clean`.
        """

        result = self._result(outcome, accepted_text, index=index)
        written = write_audit_bundle(
            self.runtime,
            result=result,
            root=self.root,
            assignment_id=self.assignment_id,
            worker_id=WORKER_ID,
            record=record,
            accepted_text=accepted_text,
            error=error,
            audit_name=f"audit-{outcome.task_id}",
        )
        if not written and capsule_retained(result):
            # There was a capsule to write into and the bundle did not get
            # there: this tree is now the only account of the task, so the
            # session stops being disposable (`write_audit_bundle` has
            # already said why). Without a capsule there was nothing to lose.
            self.incidents += 1

    def _await(
        self,
        outcome: TaskOutcome,
        *,
        run_kwargs: Mapping[str, Any],
        max_repairs: int,
        inherited_history: bool,
    ) -> tuple[Any, list[Mapping[str, Any]], bool, int, bool]:
        task_id = outcome.task_id
        deadline = outcome.started_at + self.task_timeout_seconds
        restarts = 0
        while True:
            record = self._record(task_id)
            outcome.record = record
            status = record["status"]
            if status == "accepted":
                outcome.returned_at = time.time()
                artifact = json.loads(self.runtime.read_artifact(record["accepted_artifact_ref"]))
                if artifact.get("input_hash") != self._input_hashes.get(task_id):
                    self.incidents += 1
                    raise AgentRuntimeCallError(
                        f"accepted artifact of {task_id} does not match the call"
                    )
                with self._lock:
                    self._served_in_session += 1
                    self.tasks_accepted += 1
                    index = self._served_in_session
                text = str(artifact.get("artifact") or "")
                self._audit(outcome, record=record, accepted_text=text, index=index)
                return (
                    self._result(outcome, text, index=index),
                    list(outcome.premature_stops),
                    inherited_history,
                    int(record.get("repair_attempts", 0)),
                    False,
                )
            if status == "failed":
                # The runtime gave up on this task (retired past its cap):
                # waiting out the deadline would only spend a whole model
                # timeout on a session that is not going to follow the
                # protocol.
                outcome.returned_at = time.time()
                self.incidents += 1
                self._audit(
                    outcome, record=record, accepted_text="",
                    index=self._served_in_session + 1, error="task failed",
                )
                error = AgentRuntimeCallError(
                    f"the agent session could not finish {task_id}: "
                    + ("; ".join(record.get("validation_errors") or []) or "no reason recorded")
                )
                setattr(error, "_harness_execution_attempts", list(outcome.premature_stops))
                raise error
            if status == "repairing" and int(record.get("repair_rounds_remaining", 0)) <= 0:
                # The budget is spent: the task comes back to the harness and
                # the session moves on; the caller's replacement is tier 2.
                outcome.returned_at = time.time()
                self._withdraw(task_id, "repair budget spent")
                with self._lock:
                    self._served_in_session += 1
                    self.incidents += 1
                self._audit(
                    outcome, record=record, accepted_text="",
                    index=self._served_in_session, error="repair budget spent",
                )
                return (
                    self._result(outcome, str(record.get("last_candidate") or ""),
                                 index=self._served_in_session),
                    list(outcome.premature_stops),
                    inherited_history,
                    max_repairs,
                    True,
                )
            if not self._session_alive():
                # The CLI left (or never started) while this task is open.
                # Durable state first: the task may have been accepted in the
                # same instant -- the loop above reads that next time round.
                record = self._record(task_id)
                if record["status"] == "accepted":
                    continue
                failure = self._session_error
                attempt = dict(self._session_attempt)
                if failure is not None:
                    attempt = dict(
                        (getattr(failure, "_harness_execution_attempts", None) or [attempt])[-1]
                    )
                attempt["premature_stop"] = True
                attempt["task_id"] = task_id
                outcome.premature_stops.append(attempt)
                self.incidents += 1
                if restarts < PREMATURE_STOP_RETRIES_PER_TASK and not self._closing:
                    restarts += 1
                    self._reset_task(record, "premature stop")
                    current_reporter().warning(
                        "agent-premature-stop",
                        f"agent session {self.label} ended before finishing {task_id}; "
                        "starting a fresh CLI on the same task",
                    )
                    with self._lock:
                        self._ensure_session(run_kwargs)
                    continue
                self._retire_task(record, "session ended twice")
                self._withdraw(task_id, "session ended twice")
                self._audit(
                    outcome, record=self._record(task_id), accepted_text="",
                    index=self._served_in_session + 1,
                    error=f"session ended without a submit: {failure}" if failure
                    else "session ended without a submit",
                )
                error = AgentRuntimeCallError(
                    f"agent session ended without an accepted submit for {task_id} "
                    f"(task status {record['status']!r}; "
                    f"{'; '.join(record.get('validation_errors') or []) or 'no submit seen'}"
                    + (f"; driver: {type(failure).__name__}: {failure}" if failure else "")
                    + ")"
                )
                setattr(error, "_harness_execution_attempts", list(outcome.premature_stops))
                raise error
            if time.time() >= deadline:
                self.incidents += 1
                self._retire_task(record, "task timeout")
                self._withdraw(task_id, "task timeout")
                self._audit(
                    outcome, record=self._record(task_id), accepted_text="",
                    index=self._served_in_session + 1, error="task timeout",
                )
                with self._lock:
                    self._end_session(reason="task timeout", wait=5.0)
                error = AgentRuntimeCallError(
                    f"agent session did not finish {task_id} within "
                    f"{self.task_timeout_seconds:g}s (status {record['status']!r})"
                )
                setattr(error, "_harness_execution_attempts", [
                    *outcome.premature_stops,
                    {**self._task_attempt(outcome, index=self._served_in_session + 1), "timeout": True},
                ])
                raise error
            self._session_ended.wait(_POLL_SECONDS)

    # -- lifecycle ------------------------------------------------------

    def close(self, *, keep_evidence: bool = False) -> None:
        """Seal, let the CLI leave, reclaim it if it does not. Idempotent.

        ``keep_evidence`` keeps the assignment tree even when this session
        served every task cleanly: the run around it failed, and its tree is
        then the readable account of what the agent was given and answered.
        """

        with self._lock:
            if self._closing:
                return
            self._closing = True
        try:
            self.runtime.seal()
        except Exception as exc:  # noqa: BLE001 -- closing must not raise past the caller
            current_reporter().warning(
                "agent-session-close", f"could not seal agent session {self.label}: {exc}"
            )
        if self._session_alive():
            self._session_ended.wait(CLOSE_GRACE_SECONDS)
            if self._session_alive():
                self._end_session(reason="close")
        totals = self.usage_totals()
        current_reporter().debug(
            "agent-session-usage",
            {
                "session": self.label,
                "assignment": self.assignment_id,
                "cli_sessions": self.sessions_started,
                "tasks": self._task_seq,
                "usage": totals,
            },
        )
        if self._disposable() and not keep_evidence:
            # Every task was accepted and nothing went wrong: the assignment
            # has no further use, exactly as a single-task tool session's
            # does not (`client._run_agent_tool_call`). The evidence a
            # successful call leaves is the capsule; what is here is this
            # run's whole subtitle text plus every MCP frame, and nothing
            # bounds how much of it accumulates.
            discard_assignment_root(self.root)
            return
        try:
            (self.root / "control" / "session-usage.json").write_text(
                json.dumps(
                    {"sessions": self.session_usage, "totals": totals, "tasks": self._task_seq},
                    ensure_ascii=False, indent=2, sort_keys=True,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def usage_totals(self) -> dict[str, int]:
        """What this session spent, summed over its CLI invocations.

        Session-level by decision (docs/llm_local_agent.md §12.1.3): the three
        CLI dialects report one terminal usage event per invocation, so a
        per-task split would be invented. The run books the total against the
        session's (tier, model).
        """

        totals: dict[str, int] = {}
        for row in self.session_usage:
            for key, value in dict(row.get("usage") or {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    totals[key] = int(totals.get(key, 0) + value)
        return totals

    def _disposable(self) -> bool:
        """Did every task this session took end accepted, with no incident?"""

        return (
            self.incidents == 0
            and self._task_seq > 0
            and self.tasks_accepted == self._task_seq
        )


class ConversationalQueue:
    """A run's queue for a person's own agent (docs/llm_local_agent.md §12.1.4).

    No process is owned here: the harness adds a task per call and waits for
    whoever joined through `finesub agent-join` to claim it over
    `finesub agent-task`, reading the protocol and payload files from the
    assignment root. The lease is the only hold on an agent that walks away:
    an expired lease puts the task back in the queue, and the call keeps
    waiting until its own deadline.
    """

    def __init__(
        self,
        *,
        parent: Path,
        activity_root: Path,
        execution_identity: Mapping[str, Any],
        call_timeout_seconds: float,
        wait_seconds: float = CONVERSATIONAL_JOIN_WAIT_SECONDS,
        label: str = "",
        max_workers: int = 8,
    ) -> None:
        # One id names both the assignment and its directory. They used to be
        # two independent `conv-<hex>`, which read as a mismatch to everyone
        # who saw the announcement and the bootstrap side by side.
        self.assignment_id = f"conv-{uuid.uuid4().hex[:12]}"
        self.root = (Path(parent) / self.assignment_id).resolve()
        self.label = label or self.assignment_id
        self.wait_seconds = max(1.0, float(wait_seconds))
        self.call_timeout_seconds = max(1.0, float(call_timeout_seconds))
        # Every agent call publishes an activity lease, and this one has to
        # publish it itself: the other transports get it from the CLI
        # invocation they wrap, while this path owns no process at all. The
        # tree now lives under the episode parent, which is exactly what
        # `finesub agent-clean` and `relocate` remove -- without a lease a
        # person clearing old evidence mid-run would take the live assignment
        # with it and cut off their own agent.
        self._activity = ExitStack()
        self._activity.enter_context(
            holding_activity(
                Path(activity_root), lease_id=f"conversational-{self.assignment_id}"
            )
        )
        # Everything after the lease is taken has to give it back on the way
        # out: an OS-level lock leaked here is held for the life of the
        # process, and it is the lock that tells `agent-clean` and `relocate`
        # a run is live. A queue that never got built is not a live run.
        try:
            validators: dict[str, Any] = {}
            for validator_id in sorted(VALIDATOR_BUILDERS):
                validators.update(runtime_validators(validator_id))
            self.runtime = AgentTaskRuntime.start_assignment(
                self.root,
                assignment_id=self.assignment_id,
                worker_goal="serve the harness calls of one run from your own agent",
                tasks=[],
                session_scope="task",
                execution_identity=dict(execution_identity),
                validators=validators,
                max_workers=max(1, int(max_workers)),
                sealed=False,
                # Same derivation as every other assignment: our deadline is
                # the agent's, plus the margin. It used to come off the wait
                # instead -- a number about when a *person* gets to the
                # keyboard, which has nothing to say about how long their
                # agent may go quiet.
                lease_ttl_seconds=lease_ttl_for(self.call_timeout_seconds),
            )
        except BaseException:
            self._activity.close()
            raise
        self._lock = threading.RLock()
        self._task_seq = 0
        self._announced = False
        self._input_hashes: dict[str, str] = {}
        self._closing = False
        self.tasks_accepted = 0
        self.incidents = 0

    @property
    def _joined(self) -> bool:
        """Has anybody registered as a worker on this queue?"""

        try:
            state = json.loads(
                self.runtime.read_artifact(
                    json.loads(self.runtime.index_path.read_text(encoding="utf-8"))["state_ref"]
                )
            )
        except (OSError, ValueError, KeyError):
            return False
        return bool(state.get("worker_ids"))

    def _announce(self) -> None:
        # Said once -- but only if anything was listening. The null reporter
        # accepts the call and returns, so marking it said regardless turned
        # "the first task ran on an unbound thread" into "this run never
        # mentions it again". Asking first is the whole difference; a later
        # task on a bound thread then still gets to say it.
        if self._announced:
            return
        reporter = current_reporter()
        reporter.warning(
            "agent-join",
            "this run is waiting for your own agent; in it, run: finesub agent-join"
            f" -- or name this one directly: finesub agent-join \"{self.root}\"",
            impact=f"each call waits up to {self.wait_seconds:g}s for a claim before failing",
        )
        self._announced = reporter_delivers(reporter)

    def _record(self, task_id: str) -> dict[str, Any]:
        return self.runtime.task_record(assignment_id=self.assignment_id, task_id=task_id)

    def _withdraw(self, task_id: str, reason: str) -> None:
        try:
            self.runtime.withdraw_task(
                assignment_id=self.assignment_id, task_id=task_id, reason=reason
            )
        except AgentTaskRuntimeError:
            pass

    @staticmethod
    def _result(content: str, *, task_id: str, started: float, record: Mapping[str, Any]) -> Any:
        returned = time.time()
        attempt = {
            "backend": "conversational_agent",
            "driver": "conversational",
            "task_id": task_id,
            "worker_id": str(record.get("accepted_by") or record.get("lease_owner") or ""),
            "started_at": started,
            "returned_at": returned,
            "duration_ms": int(max(0.0, returned - started) * 1000),
            "usage": {"source": "unavailable"},
            "usage_attribution": "none",
            "capsule_id": "",
        }
        return SimpleNamespace(
            content=content,
            reported_model="conversational-agent",
            episode_id="",
            execution_attempt=attempt,
            normalized_events=(),
            usage={},
            conversation_handle="",
            turn_identity="",
        )

    def run_task(
        self,
        *,
        session_type: str,
        input_hash: str,
        validator_id: str,
        metadata: Mapping[str, Any],
        protocol_text: str,
        payload_text: str,
        retrieval_mode: str,
        max_repairs: int,
    ) -> tuple[Any, list[Mapping[str, Any]], bool, int, bool]:
        with self._lock:
            if self._closing:
                raise AgentRuntimeCallError("the conversational queue is closing")
            self._task_seq += 1
            task_id = f"call-{self._task_seq:04d}"
            context_key = f"payload-{task_id}"
            spec = AgentTaskSpec(
                task_id=task_id,
                session_type=session_type,
                input_hash=input_hash,
                goal="answer the harness call",
                validator_id=validator_id,
                protocol_key=session_type,
                context_key=context_key,
                retrieval_mode=retrieval_mode,
                metadata=dict(metadata),
                required_blocks=(
                    {"kind": "protocol", "digest": "@protocol"},
                    {"kind": "payload", "digest": "@context"},
                ),
                claimable_by=("conversational",),
            )
            self._input_hashes[task_id] = input_hash
            self.runtime.add_task(
                spec,
                protocol_documents={session_type: protocol_text},
                context_documents={context_key: payload_text},
            )
            self._announce()
        started = time.time()
        # Two clocks, and two currencies. `wait_seconds` budgets the time this
        # task spends with *nobody on it*: it stops draining while an agent
        # holds a live lease -- every control command renews -- because an
        # agent that is working must not be cut off by a wall clock that
        # started before it arrived.
        #
        # A claim refills it. Somebody turning up is the evidence this run is
        # attended, and a person whose agent session dropped while they were
        # away from the desk should find the task still here when they rejoin,
        # not withdrawn because an earlier stretch of waiting had already eaten
        # the budget. What that leaves uncovered -- an agent that takes the
        # task and walks away, over and over -- is not a duration at all, so it
        # is counted instead: `lease_generation` rises by one per claim, and
        # past `MAX_CLAIMS_PER_TASK` the call gives up.
        unclaimed_left = self.wait_seconds
        last_tick = started
        claims = int(self._record(task_id).get("lease_generation", 0))
        while True:
            record = self._record(task_id)
            status = record["status"]
            if status == "accepted":
                artifact = json.loads(self.runtime.read_artifact(record["accepted_artifact_ref"]))
                if artifact.get("input_hash") != input_hash:
                    self.incidents += 1
                    raise AgentRuntimeCallError(f"accepted artifact of {task_id} does not match the call")
                with self._lock:
                    self.tasks_accepted += 1
                return (
                    self._result(str(artifact.get("artifact") or ""), task_id=task_id, started=started, record=record),
                    [],
                    False,
                    int(record.get("repair_attempts", 0)),
                    False,
                )
            if status == "repairing" and int(record.get("repair_rounds_remaining", 0)) <= 0:
                self._withdraw(task_id, "repair budget spent")
                self.incidents += 1
                return (
                    self._result(str(record.get("last_candidate") or ""), task_id=task_id, started=started, record=record),
                    [],
                    False,
                    max_repairs,
                    True,
                )
            if status == "failed":
                self._withdraw(task_id, "failed")
                self.incidents += 1
                raise AgentRuntimeCallError(
                    f"the conversational agent could not finish {task_id}: "
                    + "; ".join(record.get("validation_errors") or [])
                )
            now = time.time()
            generation = int(record.get("lease_generation", 0))
            if generation > claims:
                claims = generation
                unclaimed_left = self.wait_seconds
            if claims > MAX_CLAIMS_PER_TASK:
                self._withdraw(task_id, "claimed and abandoned too often")
                self.incidents += 1
                raise AgentRuntimeCallError(
                    f"{task_id} was claimed {claims} times and never finished; "
                    "the agent keeps taking it and going quiet long enough to "
                    f"lose the lease (waited {now - started:g}s in total)"
                )
            if not _lease_is_live(record, now):
                unclaimed_left -= now - last_tick
            last_tick = now
            if unclaimed_left <= 0:
                self._withdraw(task_id, "nobody claimed it in time")
                self.incidents += 1
                raise AgentRuntimeCallError(
                    f"no conversational agent picked up {task_id} within "
                    f"{self.wait_seconds:g}s of being unclaimed "
                    f"(status {status!r}, waited {now - started:g}s in total); "
                    f"join one with: finesub agent-join \"{self.root}\""
                )
            time.sleep(_POLL_SECONDS)

    @staticmethod
    def usage_totals() -> dict[str, int]:
        """Nobody meters a person's own agent; there is no total to report."""

        return {}

    def close(self, *, keep_evidence: bool = False) -> None:
        with self._lock:
            if self._closing:
                return
            self._closing = True
        try:
            self._close_locked(keep_evidence=keep_evidence)
        finally:
            # Released last, and unconditionally: while it is held nothing may
            # clear this tree, and an OS lock this process never gives back is
            # held until the process exits -- long enough to block the next
            # `agent-clean` for no reason.
            self._activity.close()

    def _close_locked(self, *, keep_evidence: bool) -> None:
        try:
            self.runtime.seal()
        except Exception as exc:  # noqa: BLE001
            current_reporter().warning(
                "agent-session-close", f"could not seal conversational queue {self.label}: {exc}"
            )
        if (
            not keep_evidence
            and self.incidents == 0
            and self._task_seq > 0
            and self.tasks_accepted == self._task_seq
        ):
            self._lift_evidence()
            try:
                # The tombstone is durable state, and a task that went through
                # a repair round is still carrying the whole rejected window
                # in it. Keeping the control plane must not mean keeping the
                # text -- that is the one thing this clean-up is for.
                self.runtime.forget_drafts(assignment_id=self.assignment_id)
            except (OSError, ValueError, AgentTaskRuntimeError) as exc:
                current_reporter().warning(
                    "agent-session-close",
                    f"could not clear the rejected drafts of {self.label}: {exc}",
                )
            if self._joined:
                time.sleep(CONVERSATIONAL_SEAL_GRACE_SECONDS)
            # Nothing went wrong, so this run's text has no reader left --
            # but the sealed control plane stays, see `discard_assignment_body`.
            discard_assignment_body(self.root)

    @property
    def evidence_root(self) -> Path:
        """Where this queue leaves what a reader will want afterwards.

        Lifted out of the tree before the body goes, and moved into the run's
        artifact directory by whoever knows where that is -- the client never
        learns it, and staging in memory would hold a whole run's subtitle
        text for no reason.
        """

        return self.root / EVIDENCE_DIRECTORY_NAME

    def _lift_evidence(self) -> None:
        """Copy each accepted task's exchange out before the body is cleared.

        An API call leaves its prompt and its answer in the run's artifacts;
        this path used to leave nothing at all, because everything it had was
        inside the tree that success deletes. Same shape, same place, so a
        person diagnosing a run does not have to know which backend answered.
        """

        for task_id in sorted(self._input_hashes):
            try:
                record = self._record(task_id)
                if record["status"] != "accepted":
                    continue
                manifest = json.loads(self.runtime.read_artifact(record["manifest_ref"]))
                artifact = json.loads(
                    self.runtime.read_artifact(record["accepted_artifact_ref"])
                )
                target = self.evidence_root / task_id
                target.mkdir(parents=True, exist_ok=True)
                (target / "protocol.md").write_text(
                    self.runtime.read_artifact(str(manifest["protocol_ref"])),
                    encoding="utf-8",
                )
                (target / "context.md").write_text(
                    self.runtime.read_artifact(str(manifest["context_ref"])),
                    encoding="utf-8",
                )
                (target / "answer.txt").write_text(
                    str(artifact.get("artifact") or ""), encoding="utf-8"
                )
                (target / "summary.json").write_text(
                    json.dumps(
                        {
                            "assignment_id": self.assignment_id,
                            "task_id": task_id,
                            "label": self.label,
                            "accepted_by": record.get("accepted_by", ""),
                            "repair_attempts": record.get("repair_attempts", 0),
                            "retirements": record.get("retirements", 0),
                            "lease_generation": record.get("lease_generation", 0),
                            "input_hash": self._input_hashes.get(task_id, ""),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except (OSError, ValueError, KeyError, AgentTaskRuntimeError) as exc:
                # Evidence is not the answer: a call that succeeded must not
                # fail because its record could not be copied.
                current_reporter().warning(
                    "agent-session-evidence",
                    f"could not keep the exchange of {task_id}: {exc}",
                )


class AgentSessionRegistry:
    """The hosts of one run, keyed by ``(provider tier, model, lane, mode)``."""

    def __init__(self) -> None:
        self._hosts: dict[tuple[str, str, int, str], AgentSessionHost] = {}
        self._usage_rows: list[dict[str, Any]] = []
        self._evidence_roots: list[Path] = []
        self._evidence_destination: Path | None = None
        self._lock = threading.Lock()
        self._closed = False

    def host_for(
        self,
        key: tuple[str, str, int, str],
        factory: Callable[[], AgentSessionHost],
    ) -> AgentSessionHost:
        with self._lock:
            if self._closed:
                raise AgentRuntimeCallError("the agent session registry is closed")
            host = self._hosts.get(key)
            if host is None:
                host = factory()
                self._hosts[key] = host
            return host

    @property
    def hosts(self) -> list[AgentSessionHost]:
        with self._lock:
            return list(self._hosts.values())

    def set_evidence_destination(self, artifact_dir: Path) -> None:
        """Where `agent_session_scope` files the staged exchanges."""

        with self._lock:
            self._evidence_destination = Path(artifact_dir)

    def evidence_destination(self) -> Path | None:
        with self._lock:
            return self._evidence_destination

    def evidence_roots(self) -> list[Path]:
        """Staged exchanges waiting to be moved into the run's artifacts.

        Filled by `close()`, for the same reason as `usage_rows`: what a
        session leaves behind is only final once it has ended. The registry
        holds paths, not contents -- a run's worth of subtitle text has no
        business sitting in memory.
        """

        return list(self._evidence_roots)

    def usage_rows(self) -> list[dict[str, Any]]:
        """What each session spent, keyed the way a bill is keyed.

        Filled by `close()`: a session's total is only final once its CLI has
        left. The run folds these into the task report's per-provider table,
        because the per-call rows carry no tokens at all -- usage is booked
        on the session, not split across its tasks (docs §12.1.3).
        """

        return list(self._usage_rows)

    def close(self, *, keep_evidence: bool = False) -> None:
        """Close every host; the first error is reported, none is raised.

        ``keep_evidence`` is set when the run is leaving by an exception:
        the sessions' trees are kept whatever their own outcome was.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            hosts = list(self._hosts.items())
            self._hosts.clear()
        for key, host in hosts:
            try:
                host.close(keep_evidence=keep_evidence)
            except Exception as exc:  # noqa: BLE001 -- the run's own error must surface, not this
                current_reporter().warning(
                    "agent-session-close", f"closing agent session {host.label} failed: {exc}"
                )
            evidence = getattr(host, "evidence_root", None)
            if evidence is not None and Path(evidence).is_dir():
                self._evidence_roots.append(Path(evidence))
            usage = host.usage_totals()
            if usage:
                self._usage_rows.append(
                    {
                        "provider_tier": key[0],
                        "model": key[1],
                        "lane": key[2],
                        "mode": key[3],
                        "label": host.label,
                        "usage": usage,
                    }
                )


_SCOPES: list[AgentSessionRegistry] = []
_SCOPES_LOCK = threading.Lock()


def current_registry() -> AgentSessionRegistry | None:
    with _SCOPES_LOCK:
        return _SCOPES[-1] if _SCOPES else None


@contextmanager
def agent_session_scope() -> Iterator[AgentSessionRegistry]:
    """One run's agent sessions: every `RoleClient` inside shares the
    registry, and leaving the block -- normally or by exception -- seals and
    reclaims every session (third review: a parked CLI must not outlive the
    run that started it)."""

    existing = current_registry()
    if existing is not None:
        # Re-entrant on purpose: a stage inside a run opens one of these to
        # be safe when it is called on its own, and must then share the run's
        # sessions rather than start a second set the run cannot reclaim.
        yield existing
        return
    registry = AgentSessionRegistry()
    failed = False
    with _SCOPES_LOCK:
        _SCOPES.append(registry)
    try:
        yield registry
    except BaseException:
        failed = True
        raise
    finally:
        with _SCOPES_LOCK:
            if registry in _SCOPES:
                _SCOPES.remove(registry)
        registry.close(keep_evidence=failed)
        # After the close, which is what stages the exchanges, and here rather
        # than in the front ends: a stage entered on its own owns this scope
        # too, and its evidence used to die with the registry.
        destination = registry.evidence_destination()
        if destination is not None:
            file_conversational_evidence(destination, registry)


def within_agent_session_scope(func: _F) -> _F:
    """Run ``func`` inside the run's agent session scope (re-entrant)."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with agent_session_scope():
            return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def private_registry() -> AgentSessionRegistry:
    """A registry for a client outside any run scope, closed at exit at the
    latest."""

    registry = AgentSessionRegistry()
    atexit.register(registry.close)
    return registry
