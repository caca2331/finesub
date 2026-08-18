"""Durable task protocol shared by conversational and headless agents.

An assignment declares how many workers may register; every worker keeps its
own active task, waiter and provider conversation lineage.  Worker ids, lease
generations, request ids and dependency state are all on disk, so the
scheduler can widen without moving the correctness boundary.

``control/index.json`` is the only reader entry point.  Full state is immutable
and generation-addressed; accepted submissions additionally use a tiny WAL so
a crash between artifact placement and dependency unlock is recoverable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from finesub_bootstrap.fsops import is_directory_link, write_atomic
from finesub_bootstrap.locks import holding_lock


AGENT_TASK_SCHEMA_VERSION = 3
AGENT_TASK_PROTOCOL_VERSION = "agent-task-v3"
# Liveness is "when did this worker last do anything", not "did it remember to
# ping". Every fenced call renews the lease, so a worker that is working keeps
# its task without a timer -- which a conversational Agent could not honour
# anyway: between two `finesub agent-task` invocations nothing of its is
# running, and a model in the middle of a long turn cannot execute code. The
# price of that is a coarser crash window, so the TTL is minutes, not seconds.
DEFAULT_LEASE_TTL_SECONDS = 30 * 60
# Ceiling for one `await_next_task` call. The conversational default below is
# the real constraint (host turn/tool timeouts); the ceiling only has to leave
# room for a headless worker parking until its provider conversation expires.
MAX_WATCH_SECONDS = 60 * 60
CONVERSATIONAL_WATCH_SECONDS = 28 * 60
# A waiter holds no lease, so a worker that dies while parked blocks no task --
# but it does keep its registration slot and waiter row forever. Anything that
# has not re-entered the watch within this multiple of its own wait bound is
# treated as gone, and its slot is returned.
WAITER_ABANDON_FACTOR = 2.0
# How many times a `blocked` submission may be thrown back to the queue for a
# fresh attempt. Repair (same lease, errors fed back) is the cheap loop and is
# tried first; this is the expensive one, so it is short.
DEFAULT_BLOCKED_REQUEUES = 2
# Generation files are append-only snapshots and only the newest is ever read
# (`index.json` names it). Older ones are forensics, so keep a bounded tail.
RETAINED_STATE_GENERATIONS = 20
DEFAULT_RETRIEVAL_BUDGET: Mapping[str, int | float] = {
    "max_queries": 8,
    "max_fetches": 8,
    "max_results": 40,
    "max_response_bytes": 1_048_576,
    "max_response_tokens": 64_000,
    "max_wall_seconds": 300.0,
    "max_parallel": 2,
}
_ACTIVE_STATES = {"leased", "executing", "repairing", "submitted"}
_TERMINAL_STATES = {"accepted"}
_FAILED_STATES = {"failed"}
# Operations whose replay must return the first answer rather than run again.
# That is nearly all of them -- a re-sent `reset_conversation` would open a
# second epoch, a re-sent `checkpoint_conversation` would break turn lineage --
# so the exclusion list is short and each entry needs a reason.
#
# `heartbeat` only pushes an expiry forward, which is safe to repeat, and
# nothing calls it in a loop any more; recording it grew a table that every
# later snapshot then had to carry.
_UNRECORDED_OPERATIONS = frozenset({"heartbeat"})


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_component(value: str, *, label: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned or Path(cleaned).name != cleaned or cleaned in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value!r}")
    if any(character in cleaned for character in ("/", "\\", "\0")):
        raise ValueError(f"Invalid {label}: {value!r}")
    return cleaned


def _json_copy(value: Any) -> Any:
    return json.loads(_stable_json(value))


def _normalized_retrieval_budget(value: Mapping[str, Any]) -> dict[str, int | float]:
    unknown = set(value) - set(DEFAULT_RETRIEVAL_BUDGET)
    if unknown:
        raise ValueError(f"Unknown retrieval budget fields: {sorted(unknown)}")
    budget: dict[str, int | float] = dict(DEFAULT_RETRIEVAL_BUDGET)
    budget.update(value)
    for key in (
        "max_queries",
        "max_fetches",
        "max_results",
        "max_response_bytes",
        "max_response_tokens",
        "max_parallel",
    ):
        number = int(budget[key])
        if number < 1:
            raise ValueError(f"{key} must be positive")
        budget[key] = number
    wall = float(budget["max_wall_seconds"])
    if wall <= 0:
        raise ValueError("max_wall_seconds must be positive")
    budget["max_wall_seconds"] = wall
    return budget


class AgentTaskRuntimeError(RuntimeError):
    """Base class for task protocol errors."""


class AssignmentConflictError(AgentTaskRuntimeError):
    """The requested assignment does not match durable state."""


class StaleLeaseError(AgentTaskRuntimeError):
    """A fenced mutation came from an expired or superseded lease."""


class StaleControlGenerationError(AgentTaskRuntimeError):
    """A queue mutation raced a newer control generation."""


@dataclass(frozen=True)
class AgentTaskSpec:
    task_id: str
    session_type: str
    input_hash: str
    goal: str
    dependencies: tuple[str, ...] = ()
    executor: str = "agent"
    validator_id: str = "accept"
    protocol_key: str = ""
    context_key: str = ""
    retrieval_mode: str = "none"
    retrieval_budget: Mapping[str, int | float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized(self) -> "AgentTaskSpec":
        task_id = _safe_component(self.task_id, label="task id")
        if self.executor not in {"agent", "external"}:
            raise ValueError("task executor must be 'agent' or 'external'")
        if not self.session_type.strip():
            raise ValueError(f"Task {task_id!r} has no session_type")
        if self.retrieval_mode not in {"none", "native", "local"}:
            raise ValueError("retrieval_mode must be none/native/local")
        dependencies = tuple(
            _safe_component(item, label="dependency id") for item in self.dependencies
        )
        if task_id in dependencies:
            raise ValueError(f"Task {task_id!r} cannot depend on itself")
        if not self.input_hash:
            raise ValueError(f"Task {task_id!r} has no input hash")
        return AgentTaskSpec(
            task_id=task_id,
            session_type=(self.session_type or "").strip(),
            input_hash=self.input_hash,
            goal=self.goal,
            dependencies=dependencies,
            executor=self.executor,
            validator_id=(self.validator_id or "accept").strip(),
            protocol_key=(self.protocol_key or self.session_type).strip(),
            context_key=self.context_key.strip(),
            retrieval_mode=self.retrieval_mode,
            retrieval_budget=_normalized_retrieval_budget(self.retrieval_budget),
            metadata=_json_copy(dict(self.metadata)),
        )


@dataclass(frozen=True)
class ValidationResult:
    status: str
    artifact: Any = None
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def accepted(
        cls, artifact: Any, *, metadata: Mapping[str, Any] | None = None
    ) -> "ValidationResult":
        return cls("accepted", artifact, (), dict(metadata or {}))

    @classmethod
    def repairable(cls, *errors: str) -> "ValidationResult":
        return cls("repairable", None, tuple(str(item) for item in errors if item))

    @classmethod
    def blocked(cls, *errors: str) -> "ValidationResult":
        return cls("blocked", None, tuple(str(item) for item in errors if item))

    def normalized(self) -> "ValidationResult":
        if self.status not in {"accepted", "repairable", "blocked"}:
            raise ValueError(f"Unknown validation status: {self.status!r}")
        return ValidationResult(
            status=self.status,
            artifact=_json_copy(self.artifact),
            errors=tuple(str(item) for item in self.errors),
            metadata=_json_copy(dict(self.metadata)),
        )


Validator = Callable[[Any, Mapping[str, Any]], ValidationResult]


def _accept_validator(candidate: Any, _manifest: Mapping[str, Any]) -> ValidationResult:
    return ValidationResult.accepted(candidate)


class AgentTaskRuntime:
    """One assignment's durable queue, lease store and submission boundary."""

    def __init__(
        self,
        root: str | Path,
        *,
        validators: Mapping[str, Validator] | None = None,
        lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
        blocked_requeues: int = DEFAULT_BLOCKED_REQUEUES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if lease_ttl_seconds < 1:
            raise ValueError("lease_ttl_seconds must be positive")
        if blocked_requeues < 0:
            raise ValueError("blocked_requeues must be non-negative")
        requested_root = Path(root).expanduser()
        if requested_root.exists() and is_directory_link(requested_root):
            raise ValueError("Assignment root must not be a directory link")
        self.root = requested_root.resolve()
        self.control_root = self.root / "control"
        self.index_path = self.control_root / "index.json"
        self.state_root = self.control_root / "state"
        self.lock_path = self.control_root / "runtime.lock"
        self.wal_path = self.control_root / "submit.wal.json"
        self.lease_ttl_seconds = int(lease_ttl_seconds)
        self.blocked_requeues = int(blocked_requeues)
        self._clock = clock
        self._validators: dict[str, Validator] = {"accept": _accept_validator}
        self._validators.update(validators or {})
        self._thread_lock = threading.RLock()
        self._changed = threading.Condition(self._thread_lock)
        if self.index_path.exists():
            with self._locked():
                self._recover_wal_locked()

    @classmethod
    def start_assignment(
        cls,
        root: str | Path,
        *,
        assignment_id: str,
        worker_goal: str,
        tasks: Sequence[AgentTaskSpec],
        session_scope: str = "task",
        bootstrap_text: str = "",
        protocol_documents: Mapping[str, str] | None = None,
        context_documents: Mapping[str, str] | None = None,
        knowledge_ref: str = "",
        knowledge_snapshot_identity: str = "",
        execution_identity: Mapping[str, Any] | None = None,
        max_workers: int = 1,
        sealed: bool = True,
        validators: Mapping[str, Validator] | None = None,
        lease_ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
        blocked_requeues: int = DEFAULT_BLOCKED_REQUEUES,
        clock: Callable[[], float] = time.time,
    ) -> "AgentTaskRuntime":
        runtime = cls(
            root,
            validators=validators,
            lease_ttl_seconds=lease_ttl_seconds,
            blocked_requeues=blocked_requeues,
            clock=clock,
        )
        runtime._initialize(
            assignment_id=assignment_id,
            worker_goal=worker_goal,
            tasks=tasks,
            session_scope=session_scope,
            bootstrap_text=bootstrap_text,
            protocol_documents=protocol_documents or {},
            context_documents=context_documents or {},
            knowledge_ref=knowledge_ref,
            knowledge_snapshot_identity=knowledge_snapshot_identity,
            execution_identity=execution_identity or {},
            max_workers=max_workers,
            sealed=sealed,
        )
        return runtime

    def _locked(self, *, create: bool = False):
        """Hold the assignment lock; only ``start_assignment`` may create it.

        Taking the lock used to create ``control/`` unconditionally, so a
        mistyped ``--root`` -- which the Agent's own bootstrap warns about --
        left a ``control/`` and a lock file behind in whatever directory the
        Agent happened to be standing in, and only then reported that there was
        no assignment there. Reads now refuse before touching the filesystem.
        """

        class _CombinedLock:
            def __init__(self, owner: "AgentTaskRuntime") -> None:
                self.owner = owner
                self.file_lock = None

            def __enter__(self):
                self.owner._thread_lock.acquire()
                try:
                    if create:
                        self.owner.control_root.mkdir(parents=True, exist_ok=True)
                    elif not self.owner.index_path.exists():
                        raise AssignmentConflictError(
                            f"No FineSub assignment at {self.owner.root} "
                            "(expected control/index.json)"
                        )
                    self.file_lock = holding_lock(self.owner.lock_path, timeout=10)
                    self.file_lock.__enter__()
                    return self
                except BaseException:
                    self.owner._thread_lock.release()
                    raise

            def __exit__(self, exc_type, exc, traceback):
                assert self.file_lock is not None
                try:
                    return self.file_lock.__exit__(exc_type, exc, traceback)
                finally:
                    self.owner._thread_lock.release()

        return _CombinedLock(self)

    def _initialize(
        self,
        *,
        assignment_id: str,
        worker_goal: str,
        tasks: Sequence[AgentTaskSpec],
        session_scope: str,
        bootstrap_text: str,
        protocol_documents: Mapping[str, str],
        context_documents: Mapping[str, str],
        knowledge_ref: str,
        knowledge_snapshot_identity: str,
        execution_identity: Mapping[str, Any],
        max_workers: int,
        sealed: bool,
    ) -> None:
        assignment_id = _safe_component(assignment_id, label="assignment id")
        if session_scope not in {"task", "assignment"}:
            raise ValueError("session_scope must be 'task' or 'assignment'")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        normalized_tasks = [item.normalized() for item in tasks]
        task_ids = [item.task_id for item in normalized_tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Task ids must be unique")
        known = set(task_ids)
        for task in normalized_tasks:
            missing = set(task.dependencies) - known
            if missing:
                raise ValueError(
                    f"Task {task.task_id!r} has unknown dependencies: {sorted(missing)}"
                )
            if task.validator_id not in self._validators and task.executor == "agent":
                raise ValueError(f"Unknown validator {task.validator_id!r}")
        self._assert_acyclic(normalized_tasks)

        definition = {
            "assignment_id": assignment_id,
            "worker_goal": worker_goal,
            "session_scope": session_scope,
            "sealed": bool(sealed),
            "tasks": [asdict(item) for item in normalized_tasks],
            "bootstrap_text": bootstrap_text,
            "protocol_documents": dict(protocol_documents),
            "context_documents": dict(context_documents),
            "knowledge_ref": knowledge_ref,
            "knowledge_snapshot_identity": knowledge_snapshot_identity,
            "execution_identity": dict(execution_identity),
            "max_workers": int(max_workers),
        }
        definition_digest = _sha256_text(_stable_json(definition))

        with self._locked(create=True):
            if self.index_path.exists():
                state = self._load_state_locked()
                if state.get("definition_digest") != definition_digest:
                    raise AssignmentConflictError(
                        "Existing assignment definition does not match the request"
                    )
                return
            self.root.mkdir(parents=True, exist_ok=True)
            refs = self._materialize_documents(
                bootstrap_text=bootstrap_text,
                protocol_documents=protocol_documents,
                context_documents=context_documents,
                knowledge_ref=knowledge_ref,
                knowledge_snapshot_identity=knowledge_snapshot_identity,
            )
            task_rows: dict[str, dict[str, Any]] = {}
            for spec in normalized_tasks:
                manifest = {
                    "schema_version": AGENT_TASK_SCHEMA_VERSION,
                    "protocol_version": AGENT_TASK_PROTOCOL_VERSION,
                    **asdict(spec),
                    "protocol_ref": refs["protocols"].get(spec.protocol_key, ""),
                    "context_ref": refs["contexts"].get(spec.context_key, ""),
                    "knowledge_ref": refs["knowledge"],
                    "knowledge_snapshot_identity": knowledge_snapshot_identity,
                    "completion_condition": "submit returns accepted",
                }
                manifest_text = _stable_json(manifest) + "\n"
                manifest_path = self.root / "tasks" / spec.task_id / "manifest.json"
                write_atomic(manifest_path, manifest_text)
                manifest_ref = self._reference(manifest_path, manifest_text)
                task_rows[spec.task_id] = {
                    "spec": asdict(spec),
                    "manifest_ref": manifest_ref,
                    "status": "queued",
                    "lease_generation": 0,
                    "lease": None,
                    "validation_errors": [],
                    "blocked_requeues": 0,
                    "progress": None,
                    "accepted_artifact_ref": "",
                    "retrieval": {
                        "limits": dict(spec.retrieval_budget),
                        "queries_used": 0,
                        "fetches_used": 0,
                        "results_returned": 0,
                        "response_bytes": 0,
                        "response_tokens": 0,
                        "wall_seconds": 0.0,
                        "allowed_fetch_urls": [],
                        "calls": {},
                    },
                }
            state = {
                "schema_version": AGENT_TASK_SCHEMA_VERSION,
                "protocol_version": AGENT_TASK_PROTOCOL_VERSION,
                "control_generation": 0,
                "definition_digest": definition_digest,
                "assignment_id": assignment_id,
                "worker_goal": worker_goal,
                "session_scope": session_scope,
                "sealed": bool(sealed),
                "worker_ids": [],
                "max_workers": int(max_workers),
                "execution_identity": _json_copy(execution_identity),
                "refs": refs,
                "tasks": task_rows,
                "waiters": {},
                "conversations": {},
                "request_results": {},
            }
            self._commit_state_locked(state)

    @staticmethod
    def _assert_acyclic(tasks: Sequence[AgentTaskSpec]) -> None:
        dependencies = {item.task_id: set(item.dependencies) for item in tasks}
        remaining = dict(dependencies)
        resolved: set[str] = set()
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if deps <= resolved)
            if not ready:
                raise ValueError("Task dependency graph contains a cycle")
            for key in ready:
                resolved.add(key)
                del remaining[key]

    def _materialize_documents(
        self,
        *,
        bootstrap_text: str,
        protocol_documents: Mapping[str, str],
        context_documents: Mapping[str, str],
        knowledge_ref: str,
        knowledge_snapshot_identity: str,
    ) -> dict[str, Any]:
        bootstrap_path = self.control_root / "bootstrap.md"
        bootstrap_content = bootstrap_text.rstrip() + "\n"
        write_atomic(bootstrap_path, bootstrap_content)
        protocols: dict[str, str] = {}
        for key, content in sorted(protocol_documents.items()):
            safe_key = _safe_component(key, label="protocol key")
            normalized = content.rstrip() + "\n"
            digest = _sha256_text(normalized).removeprefix("sha256:")
            path = self.control_root / "protocols" / safe_key / f"{digest}.md"
            write_atomic(path, normalized)
            protocols[key] = self._reference(path, normalized)
        contexts: dict[str, str] = {}
        for key, content in sorted(context_documents.items()):
            safe_key = _safe_component(key, label="context key")
            normalized = content.rstrip() + "\n"
            digest = _sha256_text(normalized).removeprefix("sha256:")
            path = self.root / "contexts" / digest / f"{safe_key}.md"
            write_atomic(path, normalized)
            contexts[key] = self._reference(path, normalized)
        return {
            "bootstrap": self._reference(bootstrap_path, bootstrap_content),
            "protocols": protocols,
            "contexts": contexts,
            "knowledge": knowledge_ref,
            "knowledge_snapshot_identity": knowledge_snapshot_identity,
        }

    def _reference(self, path: Path, content: str) -> str:
        relative = path.relative_to(self.root).as_posix()
        return f"{relative}#{_sha256_text(content)}"

    def _load_state_locked(self) -> dict[str, Any]:
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
            state_ref = str(index["state_ref"])
            relative, expected_digest = state_ref.rsplit("#", 1)
            path = (self.root / relative).resolve()
            if self.root not in path.parents:
                raise ValueError("state ref escapes assignment root")
            text = path.read_text(encoding="utf-8")
            if _sha256_text(text) != expected_digest:
                raise ValueError("state digest mismatch")
            state = json.loads(text)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AgentTaskRuntimeError(f"Unreadable agent task state: {exc}") from exc
        if state.get("schema_version") != AGENT_TASK_SCHEMA_VERSION:
            raise AgentTaskRuntimeError("Unsupported agent task schema")
        return state

    def _commit_state_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        prepared = deepcopy(state)
        prepared["control_generation"] = int(state.get("control_generation", 0)) + 1
        return self._write_prepared_state_locked(prepared)

    def _write_prepared_state_locked(self, state: dict[str, Any]) -> dict[str, Any]:
        generation = int(state["control_generation"])
        state_text = _stable_json(state) + "\n"
        state_path = self.state_root / f"{generation:08d}.json"
        write_atomic(state_path, state_text)
        state_ref = self._reference(state_path, state_text)
        index = self._index_for_state(state, state_ref=state_ref)
        write_atomic(self.index_path, _stable_json(index) + "\n")
        self._prune_state_generations_locked(generation)
        self._changed.notify_all()
        return state

    def _prune_state_generations_locked(self, current_generation: int) -> None:
        """Drop snapshots nothing can reach any more.

        Only the generation named by ``index.json`` is ever read; the WAL
        carries its target state inline rather than pointing at a file. The
        older snapshots are pure forensics, so a bounded tail is kept and the
        rest go -- otherwise a long-lived assignment leaves one file per state
        change behind for good.
        """

        oldest_kept = current_generation - RETAINED_STATE_GENERATIONS
        if oldest_kept < 1:
            return
        try:
            entries = list(self.state_root.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.suffix != ".json" or not entry.stem.isdigit():
                continue
            if int(entry.stem) < oldest_kept:
                try:
                    entry.unlink()
                except OSError:
                    # Losing a forensic snapshot must never fail a mutation
                    # that has already been committed.
                    pass

    def _index_for_state(
        self, state: Mapping[str, Any], *, state_ref: str
    ) -> dict[str, Any]:
        active = self._active_task(state)
        ready = (
            self._first_ready_task(state, executor="agent")
            if self._has_worker_capacity(state)
            else None
        )
        if active:
            next_action = "execute_active"
        elif ready:
            next_action = "claim_ready"
        elif self._assignment_complete(state):
            next_action = "assignment_complete"
        elif self._failed_tasks(state):
            # A task nobody can retry any further ends the assignment. Left as
            # "await_dependencies" it would park every other worker forever on
            # a dependency that is never coming.
            next_action = "assignment_failed"
        else:
            next_action = "await_dependencies"
        waiter = next(iter(dict(state.get("waiters") or {}).values()), None)
        waiting = None
        if next_action == "await_dependencies" and waiter:
            waiting = {
                "wait_token": waiter["wait_token"],
                "dependencies": self._unresolved_dependencies(state),
                "max_wait_seconds": CONVERSATIONAL_WATCH_SECONDS,
            }
        return {
            "schema_version": AGENT_TASK_SCHEMA_VERSION,
            "protocol_version": AGENT_TASK_PROTOCOL_VERSION,
            "control_generation": state["control_generation"],
            "assignment_id": state["assignment_id"],
            "session_scope": state["session_scope"],
            "next_action": next_action,
            "active_task": self._public_task(active) if active else None,
            "active_tasks": [
                self._public_task(task) for task in self._active_tasks(state)
            ],
            "ready_task": self._public_task(ready) if ready else None,
            "waiting": waiting,
            "failed_tasks": self._failed_tasks(state),
            "refs": state["refs"],
            "state_ref": state_ref,
            "completion_condition": "assignment_complete",
            "max_workers": int(state.get("max_workers", 1)),
        }

    @staticmethod
    def _public_task(task: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if task is None:
            return None
        spec = task["spec"]
        return {
            "task_id": spec["task_id"],
            "session_type": spec["session_type"],
            "manifest_ref": task["manifest_ref"],
            "status": task["status"],
            "lease_generation": task["lease_generation"],
            "lease_owner": (
                "" if task["lease"] is None else task["lease"]["worker_id"]
            ),
            "validation_errors": task["validation_errors"],
        }

    @staticmethod
    def _active_tasks(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return [
            task
            for task in dict(state["tasks"]).values()
            if task["status"] in _ACTIVE_STATES
        ]

    @classmethod
    def _active_task(cls, state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        active = cls._active_tasks(state)
        return active[0] if active else None

    @classmethod
    def _active_task_for_worker(
        cls, state: Mapping[str, Any], worker_id: str
    ) -> Mapping[str, Any] | None:
        for task in cls._active_tasks(state):
            lease = task.get("lease")
            if isinstance(lease, Mapping) and lease.get("worker_id") == worker_id:
                return task
        return None

    @classmethod
    def _has_worker_capacity(cls, state: Mapping[str, Any]) -> bool:
        return len(cls._active_tasks(state)) < int(state.get("max_workers", 1))

    @staticmethod
    def _failed_tasks(state: Mapping[str, Any]) -> list[str]:
        return sorted(
            task_id
            for task_id, task in dict(state["tasks"]).items()
            if task["status"] in _FAILED_STATES
        )

    @staticmethod
    def _task_dependencies_accepted(
        state: Mapping[str, Any], task: Mapping[str, Any]
    ) -> bool:
        return all(
            state["tasks"][dependency]["status"] == "accepted"
            for dependency in task["spec"]["dependencies"]
        )

    def _first_ready_task(
        self, state: Mapping[str, Any], *, executor: str
    ) -> Mapping[str, Any] | None:
        for task in dict(state["tasks"]).values():
            if (
                task["status"] == "queued"
                and task["spec"]["executor"] == executor
                and self._task_dependencies_accepted(state, task)
            ):
                return task
        return None

    def _assignment_complete(self, state: Mapping[str, Any]) -> bool:
        return bool(
            state["sealed"]
            and all(
                task["status"] in _TERMINAL_STATES
                for task in dict(state["tasks"]).values()
            )
        )

    @staticmethod
    def _unresolved_dependencies(state: Mapping[str, Any]) -> list[str]:
        return sorted(
            task_id
            for task_id, task in dict(state["tasks"]).items()
            if task["status"] not in _TERMINAL_STATES
        )

    def _assert_assignment(self, state: Mapping[str, Any], assignment_id: str) -> None:
        if state["assignment_id"] != assignment_id:
            raise AssignmentConflictError("assignment_id does not match durable state")

    @staticmethod
    def _deduplicated(
        state: Mapping[str, Any], *, request_id: str, operation: str
    ) -> dict[str, Any] | None:
        row = dict(state.get("request_results") or {}).get(request_id)
        if row is None:
            return None
        if row.get("operation") != operation:
            raise AssignmentConflictError(
                f"request_id {request_id!r} was already used for {row.get('operation')!r}"
            )
        return deepcopy(row["response"])

    @staticmethod
    def _record_request(
        state: dict[str, Any], *, request_id: str, operation: str, response: Mapping[str, Any]
    ) -> None:
        """Remember a response only where replaying the id can actually happen.

        Every record here is carried by every later snapshot, so recording an
        operation whose ids are one-shot (or whose repetition is harmless)
        bought nothing and grew the state without bound.
        """

        if not request_id:
            raise ValueError("request_id is required")
        if operation in _UNRECORDED_OPERATIONS:
            return
        state.setdefault("request_results", {})[request_id] = {
            "operation": operation,
            "response": _json_copy(response),
        }

    def _require_control_generation(
        self, state: Mapping[str, Any], expected_control_generation: int
    ) -> None:
        if int(state["control_generation"]) != int(expected_control_generation):
            raise StaleControlGenerationError(
                f"Expected control generation {expected_control_generation}, "
                f"found {state['control_generation']}"
            )

    def _validate_lease(
        self,
        state: Mapping[str, Any],
        *,
        task_id: str,
        worker_id: str,
        lease_generation: int,
    ) -> Mapping[str, Any]:
        task = dict(state["tasks"]).get(task_id)
        if task is None:
            raise StaleLeaseError(f"Unknown task {task_id!r}")
        lease = task.get("lease")
        if (
            task["status"] not in _ACTIVE_STATES
            or not isinstance(lease, Mapping)
            or lease.get("worker_id") != worker_id
            or int(task["lease_generation"]) != int(lease_generation)
            or float(lease.get("expires_at", 0)) <= self._clock()
        ):
            raise StaleLeaseError("Task lease is expired or has been superseded")
        return task

    @staticmethod
    def _assert_reservation_owner(
        row: Mapping[str, Any], *, worker_id: str, lease_generation: int
    ) -> None:
        if row.get("worker_id") != worker_id or int(
            row.get("lease_generation", -1)
        ) != int(lease_generation):
            raise StaleLeaseError("retrieval reservation belongs to another lease")

    def _renew_valid_lease_locked(
        self,
        state: dict[str, Any],
        *,
        task_id: str,
        worker_id: str,
        lease_generation: int,
    ) -> None:
        """Renew if the lease is still good; stay silent if it is not."""

        try:
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
        except StaleLeaseError:
            return
        self._renew_lease_locked(task)

    def _renew_lease_locked(self, task: dict[str, Any]) -> None:
        """Push a validated lease out by one TTL.

        Liveness is inferred from work, not from a keepalive: every fenced call
        a worker makes is proof it is still there, and those calls are writing
        state anyway, so renewing costs nothing extra. That is also the only
        thing a conversational Agent can offer -- it has no process of its own
        between two control commands, and a model mid-turn cannot run a timer.
        """

        lease = task.get("lease")
        if isinstance(lease, dict):
            lease["expires_at"] = self._clock() + self.lease_ttl_seconds

    def _reclaim_locked(self, state: dict[str, Any]) -> bool:
        """Return whatever a vanished worker was holding. Idempotent.

        Runs on every read as well as every claim: a worker parked in
        ``await_next_task`` never calls ``next_task`` again, so leaving
        reclamation there alone meant a crashed peer's task was never handed
        on and the survivors waited out the assignment.
        """

        now = self._clock()
        changed = False
        for task in state["tasks"].values():
            lease = task.get("lease")
            if (
                task["status"] in _ACTIVE_STATES
                and isinstance(lease, Mapping)
                and float(lease.get("expires_at", 0)) <= now
            ):
                task["status"] = "queued"
                task["lease"] = None
                task["validation_errors"] = []
                changed = True
        # A parked worker holds no lease, so nothing above notices it dying.
        # What it does hold is a registration slot, and with the slot gone
        # missing a replacement under a fresh id is refused outright.
        waiters = dict(state.get("waiters") or {})
        for worker_id, waiter in waiters.items():
            last_seen = float(waiter.get("last_seen") or 0.0)
            # Floored at the conversational bound: a caller that asked for a
            # short watch is not thereby declaring itself short-lived, and
            # letting its own bound set the threshold made a brief watch reap
            # itself on the very next poll.
            bound = max(
                float(waiter.get("max_wait_seconds") or 0.0),
                float(CONVERSATIONAL_WATCH_SECONDS),
            )
            if last_seen and now - last_seen > bound * WAITER_ABANDON_FACTOR:
                state["waiters"].pop(worker_id, None)
                state["worker_ids"] = [
                    item for item in state.get("worker_ids") or [] if item != worker_id
                ]
                changed = True
        return changed

    def _refresh_waiter_locked(
        self, state: dict[str, Any], *, worker_id: str, max_wait_seconds: float
    ) -> bool:
        """Re-stamp a waiter, but only once its stamp stops proving anything.

        Stamping on every entry would make the watcher a state mutation --
        bumping the control generation other workers compare against, for no
        news. Waiting out one full bound is when the stamp goes stale, so that
        is when it is rewritten. Reaping a live waiter early is cheap anyway:
        it loses its slot, sees ``stale`` and re-registers on the next round
        trip.

        The bound is also raised when this watch is longer than the one on
        record. The row is created with the conversational default, but a
        headless worker parking against a longer provider TTL asks for more --
        and judged by the shorter stored bound, its own reclaim pass dropped
        its own waiter part way through the first watch. Only ever raised: a
        caller asking for a shorter watch is not declaring itself shorter-lived.
        """

        waiter = dict(state.get("waiters") or {}).get(worker_id)
        if waiter is None:
            return False
        last_seen = float(waiter.get("last_seen") or 0.0)
        recorded_bound = float(waiter.get("max_wait_seconds") or 0.0)
        fresh = last_seen and self._clock() - last_seen < float(max_wait_seconds)
        if fresh and recorded_bound >= float(max_wait_seconds):
            return False
        state["waiters"][worker_id]["last_seen"] = self._clock()
        state["waiters"][worker_id]["max_wait_seconds"] = max(
            recorded_bound, float(max_wait_seconds)
        )
        return True

    def _lease_task_locked(
        self, state: dict[str, Any], task: dict[str, Any], *, worker_id: str
    ) -> None:
        task["lease_generation"] = int(task["lease_generation"]) + 1
        task["lease"] = {
            "worker_id": worker_id,
            "expires_at": self._clock() + self.lease_ttl_seconds,
        }
        task["status"] = "leased"
        state.setdefault("waiters", {}).pop(worker_id, None)

    def _response_for_state(
        self, state: Mapping[str, Any], *, worker_id: str = ""
    ) -> dict[str, Any]:
        active = (
            self._active_task_for_worker(state, worker_id)
            if worker_id
            else self._active_task(state)
        )
        if active:
            return {
                "status": "task",
                "control_generation": state["control_generation"],
                "task": self._public_task(active),
            }
        if self._assignment_complete(state):
            return {
                "status": "assignment_complete",
                "control_generation": state["control_generation"],
            }
        ready = (
            self._first_ready_task(state, executor="agent")
            if self._has_worker_capacity(state)
            else None
        )
        if ready is not None:
            return {
                "status": "ready",
                "control_generation": state["control_generation"],
                "ready_task": self._public_task(ready),
            }
        failed = self._failed_tasks(state)
        if failed:
            return {
                "status": "assignment_failed",
                "control_generation": state["control_generation"],
                "failed_tasks": failed,
                "validation_errors": [
                    error
                    for task_id in failed
                    for error in state["tasks"][task_id]["validation_errors"]
                ],
            }
        waiter = (
            dict(state.get("waiters") or {}).get(worker_id)
            if worker_id
            else next(iter(dict(state.get("waiters") or {}).values()), None)
        )
        return {
            "status": "waiting",
            "control_generation": state["control_generation"],
            "wait_token": "" if waiter is None else waiter["wait_token"],
            "dependencies": self._unresolved_dependencies(state),
            "max_wait_seconds": CONVERSATIONAL_WATCH_SECONDS,
        }

    def status(self, *, assignment_id: str, worker_id: str = "") -> dict[str, Any]:
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            # Reads reclaim too: the worker that would have noticed may be the
            # one that died, and every survivor is on this path.
            if self._reclaim_locked(state):
                state = self._commit_state_locked(state)
            response = self._response_for_state(state, worker_id=worker_id)
            response["worker_goal"] = state["worker_goal"]
            response["session_scope"] = state["session_scope"]
            return response

    def rehydrate(
        self, *, assignment_id: str, worker_id: str = ""
    ) -> dict[str, Any]:
        response = self.status(assignment_id=assignment_id, worker_id=worker_id)
        response["index_ref"] = "control/index.json"
        return response

    def conversation_state(
        self, *, assignment_id: str, worker_id: str
    ) -> dict[str, Any]:
        """Return durable provider lineage for one assignment worker."""

        worker_id = _safe_component(worker_id, label="worker id")
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            row = dict(state.get("conversations") or {}).get(worker_id)
            if row is None:
                return {
                    "conversation_epoch": 1,
                    "conversation_handle": "",
                    "turn_generation": 0,
                    "parent_turn_identity": "",
                    "harness_ack_digest": "",
                    "updated_at": 0.0,
                    "age_seconds": 0.0,
                    "resets": [],
                }
            row = deepcopy(row)
            updated_at = float(row.get("updated_at") or 0.0)
            row["age_seconds"] = (
                max(0.0, self._clock() - updated_at) if updated_at else 0.0
            )
            return row

    def checkpoint_conversation(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        conversation_epoch: int,
        conversation_handle: str,
        turn_generation: int,
        parent_turn_identity: str,
    ) -> dict[str, Any]:
        """Fence and persist the provider handle immediately after one turn."""

        if conversation_epoch < 1 or turn_generation < 1:
            raise ValueError("conversation epoch and turn generation must be positive")
        if not conversation_handle or not parent_turn_identity:
            raise ValueError("conversation handle and turn identity are required")
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="checkpoint_conversation"
            )
            if duplicate is not None:
                return duplicate
            self._renew_lease_locked(
                self._validate_lease(
                    state,
                    task_id=task_id,
                    worker_id=worker_id,
                    lease_generation=lease_generation,
                )
            )
            current = dict(state.get("conversations") or {}).get(worker_id)
            if current is not None:
                if int(current["conversation_epoch"]) != conversation_epoch:
                    raise AssignmentConflictError("conversation epoch changed unexpectedly")
                old_handle = str(current.get("conversation_handle") or "")
                if old_handle and old_handle != conversation_handle:
                    raise AssignmentConflictError("provider conversation handle changed")
                if int(current.get("turn_generation", 0)) + 1 != turn_generation:
                    raise AssignmentConflictError("conversation turn lineage is not monotonic")
                previous_ack = str(current.get("harness_ack_digest") or "")
            else:
                if turn_generation != 1:
                    raise AssignmentConflictError("a new conversation must start at turn 1")
                previous_ack = ""
            state.setdefault("conversations", {})[worker_id] = {
                "conversation_epoch": conversation_epoch,
                "conversation_handle": conversation_handle,
                "turn_generation": turn_generation,
                "parent_turn_identity": parent_turn_identity,
                "harness_ack_digest": previous_ack,
                "updated_at": self._clock(),
                "resets": list((current or {}).get("resets") or []),
            }
            response = {
                "status": "conversation_checkpointed",
                "control_generation": int(state["control_generation"]) + 1,
                "conversation_epoch": conversation_epoch,
                "turn_generation": turn_generation,
            }
            self._record_request(
                state,
                request_id=request_id,
                operation="checkpoint_conversation",
                response=response,
            )
            self._commit_state_locked(state)
            return response

    def reset_conversation(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Retire a provider conversation and open the next lineage epoch.

        The handle is only a performance cache: losing it, having it pruned by
        the provider, or finding a resume land in a different conversation must
        not strand the assignment. The next turn replays the full logical
        context into a new conversation under the same control namespace.
        """

        reason = reason.strip()
        if not reason:
            raise ValueError("a conversation reset must record why it happened")
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="reset_conversation"
            )
            if duplicate is not None:
                return duplicate
            self._renew_lease_locked(
                self._validate_lease(
                    state,
                    task_id=task_id,
                    worker_id=worker_id,
                    lease_generation=lease_generation,
                )
            )
            current = dict(state.get("conversations") or {}).get(worker_id) or {}
            epoch = int(current.get("conversation_epoch", 0)) + 1
            resets = list(current.get("resets") or [])
            resets.append(
                {
                    "conversation_epoch": int(current.get("conversation_epoch", 0)),
                    "conversation_handle": str(
                        current.get("conversation_handle") or ""
                    ),
                    "turn_generation": int(current.get("turn_generation", 0)),
                    "reason": reason,
                }
            )
            state.setdefault("conversations", {})[worker_id] = {
                "conversation_epoch": epoch,
                "conversation_handle": "",
                "turn_generation": 0,
                "parent_turn_identity": "",
                "harness_ack_digest": "",
                "updated_at": 0.0,
                "resets": resets,
            }
            response = {
                "status": "conversation_reset",
                "control_generation": int(state["control_generation"]) + 1,
                "conversation_epoch": epoch,
            }
            self._record_request(
                state,
                request_id=request_id,
                operation="reset_conversation",
                response=response,
            )
            self._commit_state_locked(state)
            return response

    @staticmethod
    def _ack_conversation(
        state: dict[str, Any], *, worker_id: str, response: Mapping[str, Any]
    ) -> None:
        row = dict(state.get("conversations") or {}).get(worker_id)
        if row is None:
            return
        state["conversations"][worker_id]["harness_ack_digest"] = _sha256_text(
            _stable_json(response)
        )

    def next_task(
        self,
        *,
        assignment_id: str,
        worker_id: str,
        request_id: str,
        expected_control_generation: int,
    ) -> dict[str, Any]:
        worker_id = _safe_component(worker_id, label="worker id")
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="next_task"
            )
            if duplicate is not None:
                return duplicate
            self._require_control_generation(state, expected_control_generation)
            self._reclaim_locked(state)
            registered_workers = list(state.get("worker_ids") or [])
            if worker_id not in registered_workers and len(registered_workers) >= int(
                state.get("max_workers", 1)
            ):
                raise AssignmentConflictError(
                    "assignment has reached its registered worker limit"
                )
            if worker_id not in registered_workers:
                registered_workers.append(worker_id)
                state["worker_ids"] = registered_workers
            active = self._active_task_for_worker(state, worker_id)
            if active is not None:
                self._renew_lease_locked(state["tasks"][active["spec"]["task_id"]])
                response = self._response_for_state(state, worker_id=worker_id)
            else:
                ready = (
                    self._first_ready_task(state, executor="agent")
                    if self._has_worker_capacity(state)
                    else None
                )
                if ready is not None:
                    self._lease_task_locked(state, ready, worker_id=worker_id)
                    response = self._response_for_state(state, worker_id=worker_id)
                elif self._assignment_complete(state) or self._failed_tasks(state):
                    response = self._response_for_state(state, worker_id=worker_id)
                else:
                    response = self._new_wait_response(state, worker_id=worker_id)
            response["control_generation"] = int(state["control_generation"]) + 1
            self._record_request(
                state, request_id=request_id, operation="next_task", response=response
            )
            self._commit_state_locked(state)
            return response

    def _new_wait_response(
        self, state: dict[str, Any], *, worker_id: str
    ) -> dict[str, Any]:
        token = "wait-" + secrets.token_hex(16)
        state.setdefault("waiters", {})[worker_id] = {
            "worker_id": worker_id,
            "wait_token": token,
            "last_seen": self._clock(),
            "max_wait_seconds": float(CONVERSATIONAL_WATCH_SECONDS),
        }
        return {
            "status": "waiting",
            "control_generation": int(state["control_generation"]) + 1,
            "wait_token": token,
            "dependencies": self._unresolved_dependencies(state),
            "max_wait_seconds": CONVERSATIONAL_WATCH_SECONDS,
        }

    def checkpoint_progress(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        progress: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="checkpoint_progress"
            )
            if duplicate is not None:
                return duplicate
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            self._renew_lease_locked(task)
            task["progress"] = _json_copy(progress)
            task["status"] = "executing"
            response = {
                "status": "checkpointed",
                "control_generation": int(state["control_generation"]) + 1,
            }
            self._record_request(
                state,
                request_id=request_id,
                operation="checkpoint_progress",
                response=response,
            )
            self._commit_state_locked(state)
            return response

    def begin_retrieval_call(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        operation: str,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reserve one harness-owned retrieval call without holding the lock over I/O."""

        if operation not in {"search", "fetch"}:
            raise ValueError("retrieval operation must be search or fetch")
        request_digest = _sha256_text(_stable_json(request))
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            if task["spec"]["retrieval_mode"] != "local":
                raise AssignmentConflictError(
                    "harness-owned retrieval is only available for retrieval_mode=local"
                )
            self._renew_lease_locked(task)
            ledger = task["retrieval"]
            self._abandon_orphaned_retrieval_calls(
                ledger, lease_generation=lease_generation
            )
            existing = ledger["calls"].get(request_id)
            if existing is not None:
                if (
                    existing["operation"] != operation
                    or existing["request_digest"] != request_digest
                ):
                    raise AssignmentConflictError(
                        f"request_id {request_id!r} was reused for different retrieval input"
                    )
                return self._retrieval_call_response(existing)
            for other_task_id, other_task in state["tasks"].items():
                if other_task_id != task_id and request_id in other_task["retrieval"]["calls"]:
                    raise AssignmentConflictError(
                        f"request_id {request_id!r} was already used by another task"
                    )

            if operation == "fetch":
                url = str(request.get("url") or "").strip()
                if url not in set(ledger.get("allowed_fetch_urls") or []):
                    raise AssignmentConflictError(
                        "web.fetch URL must come from a completed search result in this task"
                    )

            limits = ledger["limits"]
            counter = "queries_used" if operation == "search" else "fetches_used"
            maximum = "max_queries" if operation == "search" else "max_fetches"
            pending = sum(
                1 for row in ledger["calls"].values() if row["status"] == "in_progress"
            )
            reason = ""
            if int(ledger[counter]) >= int(limits[maximum]):
                reason = f"{maximum} exhausted"
            elif float(ledger["wall_seconds"]) >= float(limits["max_wall_seconds"]):
                reason = "max_wall_seconds exhausted"
            elif pending >= int(limits["max_parallel"]):
                reason = "max_parallel exhausted"
            if reason:
                row = {
                    "status": "budget_exhausted",
                    "operation": operation,
                    "request_digest": request_digest,
                    "response": {"status": "budget_exhausted", "reason": reason},
                }
            else:
                ledger[counter] = int(ledger[counter]) + 1
                row = {
                    "status": "in_progress",
                    "operation": operation,
                    "request_digest": request_digest,
                    "worker_id": worker_id,
                    "lease_generation": int(lease_generation),
                }
            ledger["calls"][request_id] = row
            response = self._retrieval_call_response(row)
            response["control_generation"] = int(state["control_generation"]) + 1
            response["budget"] = self._public_retrieval_budget(ledger)
            self._commit_state_locked(state)
            return response

    def complete_retrieval_call(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        result: Mapping[str, Any],
        result_count: int,
        response_tokens: int,
        wall_seconds: float,
    ) -> dict[str, Any]:
        """Commit one bounded retrieval response as a digest-addressed artifact."""

        if result_count < 0 or response_tokens < 0 or wall_seconds < 0:
            raise ValueError("retrieval accounting values must be non-negative")
        result_copy = _json_copy(result)
        result_text = _stable_json(result_copy) + "\n"
        response_bytes = len(result_text.encode("utf-8"))
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = state["tasks"].get(task_id)
            if task is None:
                raise StaleLeaseError(f"Unknown task {task_id!r}")
            ledger = task["retrieval"]
            row = ledger["calls"].get(request_id)
            if row is None:
                raise AssignmentConflictError("retrieval request was not reserved")
            if row["status"] != "in_progress":
                return self._retrieval_call_response(row)
            # Settled against the reservation, not against a live lease: a
            # fetch slower than the TTL would otherwise come back as a lease
            # error instead of its own result, and leave the reservation
            # in_progress holding a max_parallel slot nobody can release. The
            # reservation already names its owner, which is the fence that
            # matters here.
            self._assert_reservation_owner(
                row, worker_id=worker_id, lease_generation=lease_generation
            )
            self._renew_valid_lease_locked(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )

            limits = ledger["limits"]
            violations: list[str] = []
            if int(ledger["results_returned"]) + result_count > int(limits["max_results"]):
                violations.append("max_results")
            if int(ledger["response_bytes"]) + response_bytes > int(
                limits["max_response_bytes"]
            ):
                violations.append("max_response_bytes")
            if int(ledger["response_tokens"]) + response_tokens > int(
                limits["max_response_tokens"]
            ):
                violations.append("max_response_tokens")
            if float(ledger["wall_seconds"]) + wall_seconds > float(
                limits["max_wall_seconds"]
            ):
                violations.append("max_wall_seconds")
            ledger["wall_seconds"] = round(
                float(ledger["wall_seconds"]) + wall_seconds, 6
            )
            if violations:
                row["status"] = "budget_exhausted"
                row["response"] = {
                    "status": "budget_exhausted",
                    "reason": "retrieval response exceeded " + ", ".join(violations),
                }
            else:
                digest = _sha256_text(result_text).removeprefix("sha256:")
                result_path = (
                    self.root
                    / "tasks"
                    / task_id
                    / "retrieval"
                    / f"{hashlib.sha256(request_id.encode('utf-8')).hexdigest()}-{digest}.json"
                )
                write_atomic(result_path, result_text)
                row["status"] = "completed"
                row["result_ref"] = self._reference(result_path, result_text)
                if row["operation"] == "search":
                    urls = {
                        str(item.get("url") or "").strip()
                        for item in result_copy.get("items", [])
                        if isinstance(item, Mapping)
                    }
                    ledger["allowed_fetch_urls"] = sorted(
                        set(ledger.get("allowed_fetch_urls") or [])
                        | {url for url in urls if url}
                    )
                ledger["results_returned"] = int(ledger["results_returned"]) + result_count
                ledger["response_bytes"] = int(ledger["response_bytes"]) + response_bytes
                ledger["response_tokens"] = int(ledger["response_tokens"]) + response_tokens
            response = self._retrieval_call_response(row)
            response["control_generation"] = int(state["control_generation"]) + 1
            response["budget"] = self._public_retrieval_budget(ledger)
            self._commit_state_locked(state)
            return response

    def fail_retrieval_call(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        error: str,
        wall_seconds: float,
    ) -> dict[str, Any]:
        """Close a reserved call after a transport failure without refunding it."""

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = state["tasks"].get(task_id)
            if task is None:
                raise StaleLeaseError(f"Unknown task {task_id!r}")
            ledger = task["retrieval"]
            row = ledger["calls"].get(request_id)
            if row is None:
                raise AssignmentConflictError("retrieval request was not reserved")
            if row["status"] != "in_progress":
                return self._retrieval_call_response(row)
            # Same reasoning as ``complete_retrieval_call``: closing a call
            # must report why it failed, not why the lease is old.
            self._assert_reservation_owner(
                row, worker_id=worker_id, lease_generation=lease_generation
            )
            self._renew_valid_lease_locked(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            ledger["wall_seconds"] = round(
                float(ledger["wall_seconds"]) + max(0.0, wall_seconds), 6
            )
            row["status"] = "failed"
            row["response"] = {"status": "failed", "error": str(error)}
            response = self._retrieval_call_response(row)
            response["control_generation"] = int(state["control_generation"]) + 1
            response["budget"] = self._public_retrieval_budget(ledger)
            self._commit_state_locked(state)
            return response

    def record_native_retrieval(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        search_events: Sequence[Mapping[str, Any]],
        wall_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Book the provider-side searches of one native turn against the task.

        Native retrieval is a soft limit by construction: the driver can only
        report searches that already happened. Going over budget is recorded as
        a violation the business validator may act on — it is never reported as
        if the harness had prevented the call.
        """

        if wall_seconds < 0:
            raise ValueError("retrieval accounting values must be non-negative")
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            if task["spec"]["retrieval_mode"] != "native":
                raise AssignmentConflictError(
                    "native retrieval accounting is only for retrieval_mode=native"
                )
            self._renew_lease_locked(task)
            ledger = task["retrieval"]
            existing = ledger["calls"].get(request_id)
            if existing is not None:
                if existing["operation"] != "native":
                    raise AssignmentConflictError(
                        f"request_id {request_id!r} was reused for different retrieval input"
                    )
                return self._retrieval_call_response(existing)
            for other_task_id, other_task in state["tasks"].items():
                if (
                    other_task_id != task_id
                    and request_id in other_task["retrieval"]["calls"]
                ):
                    raise AssignmentConflictError(
                        f"request_id {request_id!r} was already used by another task"
                    )

            rows = [
                {
                    "query": str(event.get("query") or ""),
                    "urls": [str(url) for url in (event.get("urls") or [])],
                }
                for event in search_events
            ]
            result_text = _stable_json({"searches": rows}) + "\n"
            digest = _sha256_text(result_text).removeprefix("sha256:")
            result_path = (
                self.root
                / "tasks"
                / task_id
                / "retrieval"
                / f"{hashlib.sha256(request_id.encode('utf-8')).hexdigest()}-{digest}.json"
            )
            write_atomic(result_path, result_text)

            limits = ledger["limits"]
            ledger["queries_used"] = int(ledger["queries_used"]) + len(rows)
            ledger["results_returned"] = int(ledger["results_returned"]) + sum(
                len(row["urls"]) for row in rows
            )
            ledger["response_bytes"] = int(ledger["response_bytes"]) + len(
                result_text.encode("utf-8")
            )
            ledger["wall_seconds"] = round(
                float(ledger["wall_seconds"]) + wall_seconds, 6
            )
            violations = [
                name
                for name, used in (
                    ("max_queries", ledger["queries_used"]),
                    ("max_results", ledger["results_returned"]),
                    ("max_response_bytes", ledger["response_bytes"]),
                    ("max_wall_seconds", ledger["wall_seconds"]),
                )
                if float(used) > float(limits[name])
            ]
            ledger["calls"][request_id] = {
                "status": "recorded",
                "operation": "native",
                "enforcement": "soft",
                "result_ref": self._reference(result_path, result_text),
                "response": {
                    "status": "recorded",
                    "enforcement": "soft",
                    "result_ref": self._reference(result_path, result_text),
                    "searches": len(rows),
                    "violations": violations,
                },
            }
            response = self._retrieval_call_response(ledger["calls"][request_id])
            response["control_generation"] = int(state["control_generation"]) + 1
            response["budget"] = self._public_retrieval_budget(ledger)
            self._commit_state_locked(state)
            return response

    @staticmethod
    def _abandon_orphaned_retrieval_calls(
        ledger: dict[str, Any], *, lease_generation: int
    ) -> None:
        """Close reservations that their own lease can no longer settle.

        ``complete``/``fail`` both require the reserving lease, so a reservation
        left behind by an expired lease can never be closed by its owner. Left
        alone it would hold a ``max_parallel`` slot for the rest of the task.
        """

        for row in ledger["calls"].values():
            if row["status"] != "in_progress":
                continue
            if int(row.get("lease_generation", -1)) == int(lease_generation):
                continue
            row["status"] = "abandoned"
            row["response"] = {
                "status": "abandoned",
                "reason": "the reserving lease expired before the call was settled",
            }

    def _retrieval_call_response(self, row: Mapping[str, Any]) -> dict[str, Any]:
        status = str(row["status"])
        if status == "completed":
            return {
                "status": status,
                "result_ref": row["result_ref"],
                "result": json.loads(self._read_ref(str(row["result_ref"]))),
            }
        if "response" in row:
            return deepcopy(row["response"])
        return {"status": status}

    @staticmethod
    def _public_retrieval_budget(ledger: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "limits": deepcopy(ledger["limits"]),
            "used": {
                key: ledger[key]
                for key in (
                    "queries_used",
                    "fetches_used",
                    "results_returned",
                    "response_bytes",
                    "response_tokens",
                    "wall_seconds",
                )
            },
            "in_progress": sum(
                1 for row in ledger["calls"].values() if row["status"] == "in_progress"
            ),
        }

    def heartbeat(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
    ) -> dict[str, Any]:
        """Renew a lease without doing anything else.

        Nothing depends on this: every fenced call already renews, which is the
        only liveness signal a conversational Agent can actually produce. It is
        kept as an explicit way for a long-running caller to say "still here"
        and is deliberately absent from the Agent-facing command surface.
        """

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            # Checked but never recorded: a repeat is harmless, while reusing
            # an id that already belongs to some other operation still has to
            # be caught -- that is a caller bug either way.
            self._deduplicated(state, request_id=request_id, operation="heartbeat")
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            self._renew_lease_locked(task)
            response = {
                "status": "lease_renewed",
                "control_generation": int(state["control_generation"]) + 1,
                "lease_generation": task["lease_generation"],
                "expires_at": task["lease"]["expires_at"],
            }
            self._commit_state_locked(state)
            return response

    def release_task(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        reason: str = "",
    ) -> dict[str, Any]:
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="release_task"
            )
            if duplicate is not None:
                return duplicate
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            task["status"] = "queued"
            task["lease"] = None
            task["validation_errors"] = [reason] if reason else []
            response = {
                "status": "released",
                "control_generation": int(state["control_generation"]) + 1,
            }
            self._record_request(
                state, request_id=request_id, operation="release_task", response=response
            )
            self._commit_state_locked(state)
            return response

    def submit(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        input_hash: str,
        candidate: Any,
    ) -> dict[str, Any]:
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="submit"
            )
            if duplicate is not None:
                return duplicate
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            if task["spec"]["input_hash"] != input_hash:
                raise StaleLeaseError("submit input_hash does not match the leased task")
            validator_id = task["spec"]["validator_id"]
            validator = self._validators.get(validator_id)
            if validator is None:
                raise AgentTaskRuntimeError(f"Validator {validator_id!r} is unavailable")
            manifest = self._read_ref(task["manifest_ref"])
            result = validator(_json_copy(candidate), json.loads(manifest)).normalized()
            if result.status != "accepted":
                task["validation_errors"] = list(result.errors)
                response = {
                    "status": result.status,
                    "control_generation": int(state["control_generation"]) + 1,
                    "validation_errors": list(result.errors),
                }
                if result.status == "repairable":
                    # Cheap loop: same lease, same conversation, errors fed
                    # back. Renewing keeps a long repair round from losing the
                    # task it is in the middle of fixing.
                    task["status"] = "repairing"
                    self._renew_lease_locked(task)
                else:
                    # Expensive loop: the validator says repair will not help,
                    # so the session is thrown away and the task re-queued for
                    # a fresh full attempt. Bounded, because an unchanged input
                    # would otherwise block forever at the same wall -- and a
                    # blocked task with no exit parks every dependent worker.
                    used = int(task.get("blocked_requeues", 0))
                    if used < self.blocked_requeues:
                        task["blocked_requeues"] = used + 1
                        task["status"] = "queued"
                        task["lease"] = None
                        response["requeues_remaining"] = (
                            self.blocked_requeues - task["blocked_requeues"]
                        )
                    else:
                        task["status"] = "failed"
                        task["lease"] = None
                        response["status"] = "failed"
                        response["requeues_remaining"] = 0
                self._ack_conversation(state, worker_id=worker_id, response=response)
                self._record_request(
                    state, request_id=request_id, operation="submit", response=response
                )
                self._commit_state_locked(state)
                return response

            target = deepcopy(state)
            target_task = target["tasks"][task_id]
            artifact_text = _stable_json(
                {
                    "schema_version": AGENT_TASK_SCHEMA_VERSION,
                    "task_id": task_id,
                    "input_hash": input_hash,
                    "artifact": result.artifact,
                    "metadata": result.metadata,
                }
            ) + "\n"
            digest = _sha256_text(artifact_text).removeprefix("sha256:")
            artifact_path = self.root / "tasks" / task_id / "submissions" / f"{digest}.json"
            artifact_ref = self._reference(artifact_path, artifact_text)
            target_task["status"] = "accepted"
            target_task["lease"] = None
            target_task["validation_errors"] = []
            target_task["accepted_artifact_ref"] = artifact_ref
            next_ready = self._first_ready_task(target, executor="agent")
            if next_ready is not None and self._has_worker_capacity(target):
                self._lease_task_locked(target, next_ready, worker_id=worker_id)
            target["control_generation"] = int(state["control_generation"]) + 1
            response = self._response_for_state(target, worker_id=worker_id)
            response["accepted_task_id"] = task_id
            response["accepted_artifact_ref"] = artifact_ref
            self._ack_conversation(target, worker_id=worker_id, response=response)
            self._record_request(
                target, request_id=request_id, operation="submit", response=response
            )
            wal = {
                "schema_version": AGENT_TASK_SCHEMA_VERSION,
                "operation": "accepted_submit",
                "base_generation": state["control_generation"],
                "request_id": request_id,
                "artifact_ref": artifact_ref,
                "artifact_text": artifact_text,
                "target_state": target,
            }
            write_atomic(self.wal_path, _stable_json(wal) + "\n")
            write_atomic(artifact_path, artifact_text)
            self._write_prepared_state_locked(target)
            self.wal_path.unlink(missing_ok=True)
            return response

    def accept_external_task(
        self,
        *,
        assignment_id: str,
        task_id: str,
        request_id: str,
        expected_control_generation: int,
        input_hash: str,
        artifact: Any,
    ) -> dict[str, Any]:
        """Commit a scheduler/API-owned dependency and wake any waiter."""

        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="accept_external_task"
            )
            if duplicate is not None:
                return duplicate
            self._require_control_generation(state, expected_control_generation)
            task = state["tasks"].get(task_id)
            if task is None or task["spec"]["executor"] != "external":
                raise AgentTaskRuntimeError(f"Task {task_id!r} is not external")
            if task["status"] != "queued":
                raise AssignmentConflictError(f"External task {task_id!r} is not queued")
            if not self._task_dependencies_accepted(state, task):
                raise AssignmentConflictError("External task dependencies are not accepted")
            if task["spec"]["input_hash"] != input_hash:
                raise AssignmentConflictError("External task input_hash does not match")
            artifact_text = _stable_json(
                {
                    "schema_version": AGENT_TASK_SCHEMA_VERSION,
                    "task_id": task_id,
                    "input_hash": input_hash,
                    "artifact": _json_copy(artifact),
                }
            ) + "\n"
            digest = _sha256_text(artifact_text).removeprefix("sha256:")
            artifact_path = self.root / "tasks" / task_id / "submissions" / f"{digest}.json"
            artifact_ref = self._reference(artifact_path, artifact_text)
            target = deepcopy(state)
            target_task = target["tasks"][task_id]
            target_task["status"] = "accepted"
            target_task["accepted_artifact_ref"] = artifact_ref
            target["control_generation"] = int(state["control_generation"]) + 1
            response = {
                "status": "accepted",
                "control_generation": target["control_generation"],
                "accepted_task_id": task_id,
                "accepted_artifact_ref": artifact_ref,
            }
            self._record_request(
                target,
                request_id=request_id,
                operation="accept_external_task",
                response=response,
            )
            wal = {
                "schema_version": AGENT_TASK_SCHEMA_VERSION,
                "operation": "accepted_external",
                "base_generation": state["control_generation"],
                "request_id": request_id,
                "artifact_ref": artifact_ref,
                "artifact_text": artifact_text,
                "target_state": target,
            }
            write_atomic(self.wal_path, _stable_json(wal) + "\n")
            write_atomic(artifact_path, artifact_text)
            self._write_prepared_state_locked(target)
            self.wal_path.unlink(missing_ok=True)
            return response

    def _recover_wal_locked(self) -> None:
        if not self.wal_path.exists() or not self.index_path.exists():
            return
        try:
            wal = json.loads(self.wal_path.read_text(encoding="utf-8"))
            target = wal["target_state"]
            artifact_ref = str(wal["artifact_ref"])
            artifact_text = str(wal["artifact_text"])
            current = self._load_state_locked()
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AgentTaskRuntimeError(f"Unreadable submission WAL: {exc}") from exc
        request_id = str(wal.get("request_id") or "")
        if request_id in current.get("request_results", {}):
            self.wal_path.unlink(missing_ok=True)
            return
        if int(current["control_generation"]) != int(wal["base_generation"]):
            raise AgentTaskRuntimeError("Submission WAL does not follow current state")
        relative, expected_digest = artifact_ref.rsplit("#", 1)
        if _sha256_text(artifact_text) != expected_digest:
            raise AgentTaskRuntimeError("Submission WAL artifact digest mismatch")
        artifact_path = (self.root / relative).resolve()
        if self.root not in artifact_path.parents:
            raise AgentTaskRuntimeError("Submission WAL artifact escapes assignment root")
        write_atomic(artifact_path, artifact_text)
        self._write_prepared_state_locked(target)
        self.wal_path.unlink(missing_ok=True)

    def _read_ref(self, reference: str) -> str:
        relative, expected_digest = reference.rsplit("#", 1)
        path = (self.root / relative).resolve()
        if self.root not in path.parents:
            raise AgentTaskRuntimeError("Artifact reference escapes assignment root")
        text = path.read_text(encoding="utf-8")
        if _sha256_text(text) != expected_digest:
            raise AgentTaskRuntimeError(f"Artifact digest mismatch: {relative}")
        return text

    def read_artifact(self, reference: str) -> str:
        """Read one digest-bound assignment artifact without directory scans."""

        with self._locked():
            return self._read_ref(reference)

    def _watch_next_action(self, state: Mapping[str, Any], worker_id: str) -> str:
        """What the parked worker should do now, or "" to keep waiting."""

        if self._active_task_for_worker(state, worker_id) is not None:
            return "execute_active"
        if self._assignment_complete(state):
            return "assignment_complete"
        if self._failed_tasks(state):
            # Without this the watcher would keep parking on a dependency that
            # can no longer be produced -- the livelock a blocked task used to
            # cause for every worker except the one that submitted it.
            return "assignment_failed"
        if self._has_worker_capacity(state) and self._first_ready_task(
            state, executor="agent"
        ) is not None:
            return "claim_ready"
        return ""

    def await_next_task(
        self,
        *,
        assignment_id: str,
        worker_id: str,
        wait_token: str,
        max_wait_seconds: float = CONVERSATIONAL_WATCH_SECONDS,
    ) -> dict[str, Any]:
        """Event-driven wait with a bounded cross-process state recheck.

        The conversational command uses 28 minutes, which is a host turn/tool
        limit rather than anything this runtime cares about; a headless worker
        may park longer (up to :data:`MAX_WATCH_SECONDS`) so it can wake before
        its provider conversation goes cold.  Smaller values are accepted for
        contract tests.
        """

        if max_wait_seconds <= 0 or max_wait_seconds > MAX_WATCH_SECONDS:
            raise ValueError(f"max_wait_seconds must be within (0, {MAX_WATCH_SECONDS}]")
        deadline = time.monotonic() + max_wait_seconds
        worker_id = _safe_component(worker_id, label="worker id")
        stamped = False
        while True:
            with self._locked():
                self._recover_wal_locked()
                state = self._load_state_locked()
                self._assert_assignment(state, assignment_id)
                changed = self._reclaim_locked(state)
                if not stamped:
                    # At most one stamp per watch, and only when the previous
                    # one has aged out: this is how a worker that died while
                    # parked is eventually told apart from one still waiting.
                    stamped = True
                    changed |= self._refresh_waiter_locked(
                        state,
                        worker_id=worker_id,
                        max_wait_seconds=max_wait_seconds,
                    )
                if changed:
                    state = self._commit_state_locked(state)
                waiter = dict(state.get("waiters") or {}).get(worker_id)
                if waiter is None or waiter.get("wait_token") != wait_token:
                    index = json.loads(self.index_path.read_text(encoding="utf-8"))
                    next_action = self._watch_next_action(state, worker_id)
                    if next_action:
                        return {
                            "status": "ready",
                            "control_generation": index["control_generation"],
                            "next_action": next_action,
                            "index": "control/index.json",
                        }
                    return {
                        "status": "stale",
                        "control_generation": index["control_generation"],
                        "index": "control/index.json",
                    }
                next_action = self._watch_next_action(state, worker_id)
                if next_action:
                    return {
                        "status": "ready",
                        "control_generation": state["control_generation"],
                        "next_action": next_action,
                        "index": "control/index.json",
                    }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {
                    "status": "still_waiting",
                    "control_generation": state["control_generation"],
                    "wait_token": wait_token,
                    "index": "control/index.json",
                }
            with self._changed:
                self._changed.wait(timeout=min(remaining, 2.0))
