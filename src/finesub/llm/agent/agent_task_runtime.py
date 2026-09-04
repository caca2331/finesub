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
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from finesub_bootstrap.fsops import is_directory_link, write_atomic
from finesub_bootstrap.locks import holding_lock


AGENT_TASK_SCHEMA_VERSION = 4
# v4 (docs/llm_agent_tool_protocol.md §3, §4): required-block pull ledger
# with `pull_status` / `record_pull` and a submit gate over it, the atomic
# `retire_task`, and dedup records that carry an input fingerprint.
AGENT_TASK_PROTOCOL_VERSION = "agent-task-v4"
# The dedup table is assignment-level durable state carried by every later
# snapshot. Bounded, and fail-closed past the bound: a caller that runs into it
# is replaying something pathological, not working.
MAX_REQUEST_RESULTS = 512
# Missing-required-block rejections get one repair per context (docs
# §0-9, hard-coded on purpose); the next submit still owing anything means the
# session is not following the protocol and goes through the expensive loop.
PROTOCOL_REPAIRS_PER_CONTEXT = 1
# The repair budget a task carries when its manifest metadata names none
# (docs §7 "submit 内容幂等"): how many *distinct* rejected answers one
# context may follow up on. Durable on the task row, so a restarted MCP
# process cannot hand the session a fresh budget.
DEFAULT_MAX_REPAIR_ATTEMPTS = 5
# How many times one task may be retired -- a second protocol miss, a spent
# submit cap -- before it is failed instead of re-queued. Each retirement
# hands the task back with a fresh budget, so without a cap a session that
# will not follow the protocol simply starts over, and the only thing that
# ends it is the caller's per-task deadline (a whole model timeout spent on
# a task nobody was going to finish). Two is one full second chance.
MAX_RETIREMENTS_PER_TASK = 2
# Submits are deduplicated by content: re-sending an answer already judged
# replays the verdict and costs no repair budget. That makes a session that
# never changes its answer unable to exhaust the budget, so the number of
# submit *calls* per context is capped separately, at the repair budget plus
# this many. Past the cap the session is retired, like a protocol violation.
EXTRA_SUBMITS_PER_CONTEXT = 3
# Liveness is "when did this worker last do anything", not "did it remember to
# ping". Every fenced call renews the lease, so a worker that is working keeps
# its task without a timer -- which a conversational Agent could not honour
# anyway: between two `finesub agent-task` invocations nothing of its is
# running, and a model in the middle of a long turn cannot execute code. The
# price of that is a coarser crash window, so the TTL is minutes, not seconds.
#
# The TTL is not a knob and is not a second opinion about how long work takes.
# It answers one question -- *how long do we hold a task for the agent that is
# working on it* -- and it is derived from the one number that says how long
# that agent is allowed to work (`llm.local_agent_timeout_seconds`), plus a
# margin. See `lease_ttl_for`. This constant is what that derivation yields at
# the shipped default, and the fallback for a runtime built without settings.
DEFAULT_LEASE_TTL_SECONDS = 30 * 60
# The gap between "the agent's deadline" and "ours". Nothing renews while a
# driver is running -- the keepalive thread was dropped on purpose, and a
# person's own agent could never have offered one -- so a call that finishes
# just under its deadline still has to tear down, be read, and reach `submit`,
# which needs a live lease to renew. Without this the lease would expire on a
# *punctual* agent and the output it already paid for would die at submit.
LEASE_MARGIN_SECONDS = 120


def lease_ttl_for(call_timeout_seconds: float) -> int:
    """How long we hold a task for the agent working on it.

    One source for all three assignment shapes (one-shot tool session,
    pseudo-conversational host, conversational queue): our deadline is the
    agent's deadline plus the margin. Strictly later, never equal -- equal
    means reclaiming the task from somebody who met their deadline.
    """

    return max(1, int(call_timeout_seconds) + LEASE_MARGIN_SECONDS)
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
# How long a worker registration that never went on to do anything holds its
# id (`register_worker`). A person can run `finesub agent-join`, read the
# prompt and never hand it to an agent: that reservation has no lease and no
# waiter, so nothing else would ever reclaim it, and the queue would fill up
# with names that are not coming.
REGISTRATION_GRACE_SECONDS = CONVERSATIONAL_WATCH_SECONDS * WAITER_ABANDON_FACTOR
# How many times a `blocked` submission may be thrown back to the queue for a
# fresh attempt. Repair (same lease, errors fed back) is the cheap loop and is
# tried first; this is the expensive one, so it is short.
DEFAULT_BLOCKED_REQUEUES = 2
# Generation files are append-only snapshots and only the newest is ever read
# (`index.json` names it). Older ones are forensics, so keep a bounded tail.
RETAINED_STATE_GENERATIONS = 20
# Relaxed 2026-08-28 (owner decision, docs/report 2026-08-28 §2.4/§2.5): the
# binding constraint in real runs was max_response_tokens — one search returns
# ~12k tokens, so 64k starved a "verify a dozen names" task after four calls
# while max_queries sat unused. Sized for ~12 full-fat searches.
DEFAULT_RETRIEVAL_BUDGET: Mapping[str, int | float] = {
    "max_queries": 12,
    "max_fetches": 12,
    "max_results": 60,
    "max_response_bytes": 1_048_576,
    "max_response_tokens": 192_000,
    "max_wall_seconds": 600.0,
    "max_parallel": 2,
}
_ACTIVE_STATES = {"leased", "executing", "repairing", "submitted"}
# `withdrawn` is the harness taking a task back (pseudo-conversational: a
# spent repair budget, a call it stopped waiting for): terminal, but neither
# an accepted artifact nor a failure that ends the assignment.
_TERMINAL_STATES = {"accepted", "withdrawn"}
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


def _fp(**fields: Any) -> str:
    """Input fingerprint of one recorded operation (docs §2, dedup records).

    Same request id + same fingerprint is a transport replay and gets the
    first answer; same id + different fingerprint is an id reused for new
    work and is refused. Only a collision check, never an identity.
    """

    return _sha256_text(_stable_json(_json_copy(fields)))


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


def is_artifact_reference(reference: str) -> bool:
    """Whether a block's ``ref`` names a stored artifact.

    A stored artifact is ``<relative path>#<digest>``. A required block may
    instead be *served by a tool* -- ``kb_index`` is fetched by its own tool
    and carries the tool name as its ref -- and its identity is then the
    declared digest, not a file. Anything walking ``required_blocks`` has to
    ask before reaching for a body: assuming every ref is readable is what
    silently cost every knowledge-bound call its audit bundle.
    """

    return "#" in str(reference or "")


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
    # Static blocks the worker must have seen in its current context before a
    # submit is accepted (docs/llm_agent_tool_protocol.md §3). Identity is
    # ``(kind, digest)``; ``ref`` and ``tool`` tell the agent where to get it.
    # Empty means the push transport already put everything in the prompt.
    required_blocks: tuple[Mapping[str, Any], ...] = ()
    # Which kinds of worker may claim this task (docs/llm_local_agent.md
    # §12.1.4): ``headless`` is a CLI the harness runs, ``conversational`` is
    # a person's own agent driving `finesub agent-task`. Routing decides per
    # task; the queue enforces it at claim time.
    claimable_by: tuple[str, ...] = ("headless",)

    def normalized(self) -> "AgentTaskSpec":
        task_id = _safe_component(self.task_id, label="task id")
        required_blocks: list[dict[str, Any]] = []
        for block in self.required_blocks:
            kind = str(block.get("kind") or "").strip()
            digest = str(block.get("digest") or "").strip()
            if not kind or not digest:
                raise ValueError(
                    f"Task {task_id!r} declares a required block without kind/digest"
                )
            required_blocks.append(
                {
                    "kind": kind,
                    "digest": digest,
                    "ref": str(block.get("ref") or ""),
                    "tool": str(block.get("tool") or "read_context"),
                }
            )
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
        claimable_by = tuple(dict.fromkeys(str(kind).strip() for kind in self.claimable_by))
        if not claimable_by or not set(claimable_by) <= WORKER_KINDS:
            raise ValueError(
                f"Task {task_id!r} claimable_by must be a non-empty subset of {sorted(WORKER_KINDS)}"
            )
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
            required_blocks=tuple(required_blocks),
            claimable_by=claimable_by,
        )


WORKER_KINDS = frozenset({"headless", "conversational"})


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
            task_rows: dict[str, dict[str, Any]] = {
                spec.task_id: self._task_row(spec, refs)
                for spec in normalized_tasks
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

    def _task_row(self, spec: AgentTaskSpec, refs: Mapping[str, Any]) -> dict[str, Any]:
        """Write one task's manifest and build its durable row."""

        protocol_ref = dict(refs["protocols"]).get(spec.protocol_key, "")
        context_ref = dict(refs["contexts"]).get(spec.context_key, "")
        # A required block may name the task's own protocol or context
        # document by placeholder digest (`@protocol` / `@context`): the
        # caller declares the block before the runtime has materialized the
        # document, so the ref and digest are filled in here from what was
        # just written.
        required_blocks: list[dict[str, Any]] = []
        for block in spec.required_blocks:
            resolved = dict(block)
            placeholder = {"@protocol": protocol_ref, "@context": context_ref}.get(
                str(block.get("digest"))
            )
            if placeholder is not None:
                if not placeholder:
                    raise ValueError(
                        f"Task {spec.task_id!r} requires {block['digest']} but "
                        "has no such document"
                    )
                resolved["ref"] = placeholder
                resolved["digest"] = placeholder.rsplit("#", 1)[-1]
            required_blocks.append(resolved)
        spec = replace(spec, required_blocks=tuple(required_blocks))
        manifest = {
            "schema_version": AGENT_TASK_SCHEMA_VERSION,
            "protocol_version": AGENT_TASK_PROTOCOL_VERSION,
            **asdict(spec),
            "protocol_ref": protocol_ref,
            "context_ref": context_ref,
            "knowledge_ref": refs.get("knowledge", ""),
            "knowledge_snapshot_identity": refs.get("knowledge_snapshot_identity", ""),
            "completion_condition": "submit returns accepted",
        }
        manifest_text = _stable_json(manifest) + "\n"
        manifest_path = self.root / "tasks" / spec.task_id / "manifest.json"
        write_atomic(manifest_path, manifest_text)
        manifest_ref = self._reference(manifest_path, manifest_text)
        return {
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

    def add_task(
        self,
        spec: AgentTaskSpec,
        *,
        protocol_documents: Mapping[str, str] | None = None,
        context_documents: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Append one task to an unsealed assignment (pseudo-conversational).

        The documents are materialized by content digest, so a protocol the
        session already carries is not written twice and the task's refs
        point at the same file. Parked workers are woken through the state
        change: their next `await_next_task` recheck sees the queued task.
        """

        spec = spec.normalized()
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            if state.get("sealed"):
                raise AssignmentConflictError("assignment is sealed; no task can be added")
            if spec.task_id in state["tasks"]:
                raise AssignmentConflictError(f"Task {spec.task_id!r} already exists")
            missing = set(spec.dependencies) - set(state["tasks"])
            if missing:
                raise ValueError(
                    f"Task {spec.task_id!r} has unknown dependencies: {sorted(missing)}"
                )
            if spec.validator_id not in self._validators and spec.executor == "agent":
                raise ValueError(f"Unknown validator {spec.validator_id!r}")
            refs = dict(state["refs"])
            refs["protocols"] = dict(refs.get("protocols") or {})
            refs["contexts"] = dict(refs.get("contexts") or {})
            for key, content in sorted((protocol_documents or {}).items()):
                safe_key = _safe_component(key, label="protocol key")
                normalized = content.rstrip() + "\n"
                digest = _sha256_text(normalized).removeprefix("sha256:")
                path = self.control_root / "protocols" / safe_key / f"{digest}.md"
                if not path.exists():
                    write_atomic(path, normalized)
                refs["protocols"][key] = self._reference(path, normalized)
            for key, content in sorted((context_documents or {}).items()):
                safe_key = _safe_component(key, label="context key")
                normalized = content.rstrip() + "\n"
                digest = _sha256_text(normalized).removeprefix("sha256:")
                path = self.root / "contexts" / digest / f"{safe_key}.md"
                if not path.exists():
                    write_atomic(path, normalized)
                refs["contexts"][key] = self._reference(path, normalized)
            state["refs"] = refs
            state["tasks"][spec.task_id] = self._task_row(spec, refs)
            state = self._commit_state_locked(state)
            return self._public_task(state["tasks"][spec.task_id]) or {}

    def register_worker(
        self,
        *,
        assignment_id: str,
        worker_id: str = "",
        kind: str = "conversational",
        prefix: str = "conv",
    ) -> dict[str, Any]:
        """Reserve a worker id and its kind, under the lock. Idempotent by id.

        Two people joining the same run before either has claimed anything
        used to read the worker list and both pick the first free id, so the
        second one resumed the first one's task as if it were its own. The
        allocation and the registration have to be the same write; this is
        it. Re-registering an id already held by the same kind is a rejoin
        and returns the same row, and refreshes its reservation.

        A reservation is not free: it counts against ``max_workers`` like a
        worker that is already working, and it expires
        (`REGISTRATION_GRACE_SECONDS`) if it never becomes one. Otherwise a
        `finesub agent-join` nobody followed through on would hold a slot for
        the length of the run and the roster would only ever grow.
        """

        if kind not in WORKER_KINDS:
            raise ValueError(f"worker kind must be one of {sorted(WORKER_KINDS)}")
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            self._reclaim_locked(state)
            self._expire_registrations_locked(state)
            registered = list(state.get("worker_ids") or [])
            workers = state.setdefault("workers", {})
            if worker_id:
                worker_id = _safe_component(worker_id, label="worker id")
                known = str((workers.get(worker_id) or {}).get("kind") or "")
                if known and known != kind:
                    raise AssignmentConflictError(
                        f"worker {worker_id!r} is registered as {known!r}, not {kind!r}"
                    )
            else:
                number = 1
                while f"{prefix}-{number}" in registered:
                    number += 1
                worker_id = _safe_component(f"{prefix}-{number}", label="worker id")
            capacity = max(1, int(state.get("max_workers", 1)))
            if worker_id not in registered and len(registered) >= capacity:
                raise AssignmentConflictError(
                    f"this assignment takes {capacity} worker(s) and "
                    f"{sorted(registered)} already hold them"
                )
            if worker_id not in registered:
                registered.append(worker_id)
                state["worker_ids"] = registered
            workers[worker_id] = {"kind": kind, "registered_at": self._clock()}
            self._commit_state_locked(state)
            return {"worker_id": worker_id, "kind": kind}

    def _expire_registrations_locked(self, state: dict[str, Any]) -> bool:
        """Drop reservations that never became work. Idempotent.

        A worker that holds a lease, is parked in `await_next_task` or has
        accepted something is doing its job and is left alone; one that has
        only ever registered is judged by how long ago it did that.
        """

        workers = dict(state.get("workers") or {})
        if not workers:
            return False
        busy = {
            str((task.get("lease") or {}).get("worker_id") or "")
            for task in dict(state["tasks"]).values()
        } | {
            str(task.get("accepted_by") or "") for task in dict(state["tasks"]).values()
        } | set(dict(state.get("waiters") or {}))
        now = self._clock()
        dropped = False
        for worker_id, row in workers.items():
            if worker_id in busy:
                continue
            registered_at = float((row or {}).get("registered_at") or 0.0)
            if registered_at and now - registered_at <= REGISTRATION_GRACE_SECONDS:
                continue
            if not registered_at:
                # Written before reservations were timed: leave it be rather
                # than reclaim something whose age is unknown.
                continue
            state["workers"].pop(worker_id, None)
            state["worker_ids"] = [
                item for item in state.get("worker_ids") or [] if item != worker_id
            ]
            dropped = True
        return dropped

    def seal(self) -> dict[str, Any]:
        """No more tasks will come: once every task is terminal the assignment
        is complete and parked workers are told to leave. Idempotent."""

        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            if not state.get("sealed"):
                state["sealed"] = True
                state = self._commit_state_locked(state)
            return self._response_for_state(state)

    def withdraw_task(self, *, assignment_id: str, task_id: str, reason: str) -> dict[str, Any]:
        """The harness takes a task back: terminal, neither accepted nor failed.

        For a tool session whose repair budget is spent (the replacement is
        the caller's next call, on a fresh conversation) or a call the harness
        gave up waiting for. Whatever lease the task held is revoked; a late
        `submit` on it lands as a stale lease. A withdrawn task does not fail
        the assignment, so the other workers keep going.
        """

        reason = reason.strip()
        if not reason:
            raise ValueError("a task withdrawal must record why it happened")
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = dict(state["tasks"]).get(task_id)
            if task is None:
                raise StaleLeaseError(f"Unknown task {task_id!r}")
            if task["status"] in _TERMINAL_STATES:
                return self._response_for_state(state)
            task["status"] = "withdrawn"
            task["lease"] = None
            task["withdrawn_reason"] = reason
            self._prune_task_requests_locked(state, task_id)
            state = self._commit_state_locked(state)
            return {"status": "withdrawn", "task_id": task_id, **self._response_for_state(state)}

    @staticmethod
    def _prune_task_requests_locked(state: dict[str, Any], task_id: str) -> None:
        """Drop the request records a finished task no longer needs.

        The dedup table is bounded per assignment (`MAX_REQUEST_RESULTS`); a
        session that serves a whole run keeps taking tasks, so rows nothing
        can replay any more -- their lease is gone with the task -- must not
        accumulate. The one row a finished task still owes an answer for, the
        accept, moves onto the task row instead (`_accepted_replay_locked`).
        """

        table = state.get("request_results") or {}
        for request_id in [
            key for key, row in table.items() if row.get("task_id") == task_id
        ]:
            del table[request_id]

    @staticmethod
    def _accepted_replay_locked(
        state: Mapping[str, Any],
        *,
        task_id: str,
        request_id: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        """The stored answer to a re-sent accepted `submit`, or None.

        The accept is the one submit whose verdict a task must be able to
        replay after it is terminal -- a reconnected worker re-sending the
        answer that was taken has to hear "accepted", not a stale lease. It
        is kept on the task row rather than in the assignment-wide request
        table so a long session cannot fill that table (docs §7).
        """

        stored = dict(state.get("tasks") or {}).get(task_id, {}).get("accepted_submit")
        if not stored or str(stored.get("request_id")) != request_id:
            return None
        if str(stored.get("fingerprint") or "") != fingerprint:
            raise AssignmentConflictError(
                f"request_id {request_id!r} was replayed with different input"
            )
        return deepcopy(stored.get("response") or {})

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
            # Every task's manifest, whatever its state: a process that opens
            # this root cold (the MCP server) needs the validator ids and
            # retrieval modes before any task is active.
            "task_manifest_refs": {
                task_id: str(task["manifest_ref"])
                for task_id, task in sorted(dict(state["tasks"]).items())
            },
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

    @staticmethod
    def _worker_kind(state: Mapping[str, Any], worker_id: str) -> str:
        return str(dict(state.get("workers") or {}).get(worker_id, {}).get("kind") or "headless")

    def _first_ready_task(
        self, state: Mapping[str, Any], *, executor: str, kind: str = "headless"
    ) -> Mapping[str, Any] | None:
        for task in dict(state["tasks"]).values():
            spec = task["spec"]
            if (
                task["status"] == "queued"
                and spec["executor"] == executor
                and kind in tuple(spec.get("claimable_by") or ("headless",))
                # A person's agent has no harness-entitled native search: a
                # task that needs the CLI's own search tool is not for it.
                and not (kind == "conversational" and spec.get("retrieval_mode") == "native")
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
        state: Mapping[str, Any],
        *,
        request_id: str,
        operation: str,
        fingerprint: str = "",
    ) -> dict[str, Any] | None:
        """The first answer to a replayed request id, or None to run it.

        A replay is the same id *and* the same input: the same id with a
        different fingerprint is a caller reusing an id for new work, and
        answering it with the old response would silently drop the new
        input (docs/llm_agent_tool_protocol.md §2, dedup fingerprint).
        Records without a stored response (pull records) return None so the
        idempotent operation simply runs again.
        """

        row = dict(state.get("request_results") or {}).get(request_id)
        if row is None:
            return None
        if row.get("operation") != operation:
            raise AssignmentConflictError(
                f"request_id {request_id!r} was already used for {row.get('operation')!r}"
            )
        stored = str(row.get("fingerprint") or "")
        if stored and fingerprint and stored != fingerprint:
            raise AssignmentConflictError(
                f"request_id {request_id!r} was replayed with different input"
            )
        if row.get("response") is None:
            return None
        return deepcopy(row["response"])

    @staticmethod
    def _record_request(
        state: dict[str, Any],
        *,
        request_id: str,
        operation: str,
        response: Mapping[str, Any] | None,
        fingerprint: str = "",
        task_id: str = "",
    ) -> None:
        """Remember a response only where replaying the id can actually happen.

        Every record here is carried by every later snapshot, so recording an
        operation whose ids are one-shot (or whose repetition is harmless)
        bought nothing and grew the state without bound. ``response=None``
        records the id and fingerprint alone, for operations whose answer is
        large and re-derivable (a context pull). ``task_id`` tags the row so
        a finished task's rows can be pruned (`_prune_task_requests_locked`).
        """

        if not request_id:
            raise ValueError("request_id is required")
        if operation in _UNRECORDED_OPERATIONS:
            return
        table = state.setdefault("request_results", {})
        if request_id not in table and len(table) >= MAX_REQUEST_RESULTS:
            raise AssignmentConflictError(
                f"assignment request table is full ({MAX_REQUEST_RESULTS} records)"
            )
        row: dict[str, Any] = {"operation": operation}
        if fingerprint:
            row["fingerprint"] = fingerprint
        if task_id:
            row["task_id"] = task_id
        row["response"] = None if response is None else _json_copy(response)
        table[request_id] = row

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
        # Task scope has no conversation record: a lease *is* the context, so
        # the ledger starts empty with it (docs/llm_agent_tool_protocol.md
        # §3, context id for the `api` tier).
        task["pulled"] = []
        task["protocol_repairs"] = 0
        self._reset_submit_ledger(task)
        state.setdefault("waiters", {}).pop(worker_id, None)

    @staticmethod
    def _reset_submit_ledger(task: dict[str, Any]) -> None:
        """A new lease is a new context: its submits start from nothing.

        `reset_conversation` keeps the lease and therefore keeps this ledger --
        a fresh CLI re-sending the answer its predecessor had judged gets the
        same verdict back instead of paying for it twice (docs §7).
        """

        task["submissions"] = {}
        task["submit_count"] = 0
        task["repair_attempts"] = 0

    @staticmethod
    def _max_repair_attempts(task: Mapping[str, Any]) -> int:
        metadata = dict(task["spec"].get("metadata") or {})
        try:
            return max(
                0, int(metadata.get("max_repair_attempts", DEFAULT_MAX_REPAIR_ATTEMPTS))
            )
        except (TypeError, ValueError):
            return DEFAULT_MAX_REPAIR_ATTEMPTS

    @classmethod
    def _submit_budget(cls, task: Mapping[str, Any]) -> dict[str, int]:
        """The durable repair/submit counters the server reports to the model."""

        max_repairs = cls._max_repair_attempts(task)
        repairs = int(task.get("repair_attempts", 0))
        return {
            "max_repair_attempts": max_repairs,
            "repair_attempts": repairs,
            "repair_rounds_remaining": max(0, max_repairs - repairs),
            "submit_count": int(task.get("submit_count", 0)),
            "max_submits": max_repairs + EXTRA_SUBMITS_PER_CONTEXT,
        }

    # ------------------------------------------------------------------
    # Required-block ledger (protocol v4, docs/llm_agent_tool_protocol.md §3)
    # ------------------------------------------------------------------

    @staticmethod
    def _block_key(block: Mapping[str, Any]) -> str:
        return f"{block.get('kind')}@{block.get('digest')}"

    @staticmethod
    def _ledger_locked(
        state: dict[str, Any], task: dict[str, Any], *, worker_id: str
    ) -> dict[str, Any]:
        """The mutable container holding this context's pull ledger.

        Assignment scope: the worker's conversation record -- replaced
        wholesale by `reset_conversation`, so the ledger clears with it and
        needs no clearing step of its own. Task scope: the task row, reset on
        every lease.
        """

        if state.get("session_scope") != "assignment":
            task.setdefault("pulled", [])
            task.setdefault("protocol_repairs", 0)
            return task
        conversations = state.setdefault("conversations", {})
        row = conversations.get(worker_id)
        if row is None:
            row = {
                "conversation_epoch": 1,
                "conversation_handle": "",
                "turn_generation": 0,
                "parent_turn_identity": "",
                "harness_ack_digest": "",
                "updated_at": 0.0,
                "resets": [],
            }
            conversations[worker_id] = row
        row.setdefault("pulled", [])
        row.setdefault("protocol_repairs", 0)
        return row

    @classmethod
    def _owed_blocks(
        cls, task: Mapping[str, Any], ledger: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        pulled = set(ledger.get("pulled") or [])
        return [
            dict(block)
            for block in task["spec"].get("required_blocks") or []
            if cls._block_key(block) not in pulled
        ]

    def task_record(self, *, assignment_id: str, task_id: str) -> dict[str, Any]:
        """One task's durable row, for the harness reading a finished session.

        Status, validation errors, the accepted artifact ref and the last
        rejected candidate -- what a caller needs to turn "the CLI exited"
        into an outcome without being the worker.
        """

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = dict(state["tasks"]).get(task_id)
            if task is None:
                raise StaleLeaseError(f"Unknown task {task_id!r}")
            return {
                "task_id": task_id,
                "manifest_ref": str(task["manifest_ref"]),
                "status": task["status"],
                "lease_generation": int(task["lease_generation"]),
                "lease_owner": "" if task["lease"] is None else task["lease"]["worker_id"],
                # Reads do not reclaim, so `lease_owner` alone cannot tell a
                # worker that is still on the job from one that went silent
                # long enough to have lost the task. A caller deciding how
                # long to keep waiting needs the deadline itself.
                "lease_expires_at": (
                    0.0 if task["lease"] is None else float(task["lease"]["expires_at"])
                ),
                "validation_errors": list(task.get("validation_errors") or []),
                "accepted_artifact_ref": str(task.get("accepted_artifact_ref") or ""),
                "last_candidate": task.get("last_candidate"),
                "retirements": int(task.get("retirements", 0)),
                "accepted_by": str(task.get("accepted_by") or ""),
                **self._submit_budget(task),
            }

    def pull_status(
        self, *, assignment_id: str, task_id: str, worker_id: str
    ) -> dict[str, Any]:
        """Which required blocks this context still owes. Read-only."""

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = dict(state["tasks"]).get(task_id)
            if task is None:
                raise StaleLeaseError(f"Unknown task {task_id!r}")
            ledger = self._ledger_locked(state, task, worker_id=worker_id)
            owed = self._owed_blocks(task, ledger)
            return {
                "status": "pull_status",
                "control_generation": int(state["control_generation"]),
                "required_blocks": _json_copy(task["spec"].get("required_blocks") or []),
                "pulled": list(ledger.get("pulled") or []),
                "owed_blocks": owed,
            }

    def record_pull(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        blocks: Sequence[Mapping[str, Any]],
        source: str = "pull",
    ) -> dict[str, Any]:
        """Book blocks as seen by this context -- pulled by the agent or pushed
        by the harness in its first message (both satisfy the rule; the rule
        is "it is in the ledger", not "a tool call produced it")."""

        if source not in {"pull", "push"}:
            raise ValueError("pull source must be 'pull' or 'push'")
        keys = sorted({self._block_key(block) for block in blocks})
        fingerprint = _fp(
            task=task_id,
            worker=worker_id,
            lease=int(lease_generation),
            blocks=keys,
            source=source,
        )
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            self._deduplicated(
                state,
                request_id=request_id,
                operation="record_pull",
                fingerprint=fingerprint,
            )
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            self._renew_lease_locked(task)
            ledger = self._ledger_locked(state, task, worker_id=worker_id)
            pulled = list(ledger.get("pulled") or [])
            for key in keys:
                if key not in pulled:
                    pulled.append(key)
            ledger["pulled"] = pulled
            ledger.setdefault("pull_sources", {}).update({key: source for key in keys})
            response = {
                "status": "pull_recorded",
                "control_generation": int(state["control_generation"]) + 1,
                "owed_blocks": self._owed_blocks(task, ledger),
            }
            # Fingerprint only: the answer is re-derivable, and the block
            # bodies never belong in durable state.
            self._record_request(
                state,
                request_id=request_id,
                operation="record_pull",
                response=None,
                fingerprint=fingerprint,
                task_id=task_id,
            )
            self._commit_state_locked(state)
            return response

    def _retire_task_locked(
        self, state: dict[str, Any], task: dict[str, Any], *, worker_id: str, reason: str
    ) -> dict[str, Any]:
        """Reset the worker's conversation, revoke the lease, re-queue. One write."""

        current = dict(state.get("conversations") or {}).get(worker_id) or {}
        epoch = int(current.get("conversation_epoch", 0)) + 1
        resets = list(current.get("resets") or [])
        resets.append(
            {
                "conversation_epoch": int(current.get("conversation_epoch", 0)),
                "conversation_handle": str(current.get("conversation_handle") or ""),
                "turn_generation": int(current.get("turn_generation", 0)),
                "reason": f"retired: {reason}",
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
        task["lease"] = None
        task["pulled"] = []
        task["protocol_repairs"] = 0
        self._reset_submit_ledger(task)
        task["retirements"] = int(task.get("retirements", 0)) + 1
        spent = task["retirements"] >= MAX_RETIREMENTS_PER_TASK
        task["status"] = "failed" if spent else "queued"
        if spent:
            task["validation_errors"] = [
                *(task.get("validation_errors") or []),
                f"retired {task['retirements']} times; the last was: {reason}",
            ]
        return {
            "status": "failed" if spent else "retired",
            "control_generation": int(state["control_generation"]) + 1,
            "conversation_epoch": epoch,
            "retirements": task["retirements"],
        }

    def retire_task(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        request_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Retire a session that will not finish its task, atomically.

        One write: the worker's conversation is reset (new epoch, ledger
        cleared), the lease revoked, the task re-queued. The next claim --
        by the same worker id running a fresh CLI, per docs
        §5.3 -- gets a new lease generation. The CAS is the lease itself: a
        `submit` that landed first made the task non-active, so this raises
        `StaleLeaseError` and changes nothing; a retire that lands first
        leaves the late submit with the same stale lease.
        """

        reason = reason.strip()
        if not reason:
            raise ValueError("a task retirement must record why it happened")
        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="retire_task", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason)
            )
            if duplicate is not None:
                return duplicate
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            response = self._retire_task_locked(
                state, task, worker_id=worker_id, reason=reason
            )
            self._record_request(
                state, request_id=request_id, operation="retire_task", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason), response=response,
                task_id=task_id,
            )
            self._commit_state_locked(state)
            return response

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
                "sealed": bool(state.get("sealed")),
            }
        if self._assignment_complete(state):
            return {
                "status": "assignment_complete",
                "control_generation": state["control_generation"],
                "sealed": True,
            }
        ready = (
            self._first_ready_task(
                state, executor="agent",
                kind=self._worker_kind(state, worker_id) if worker_id else "headless",
            )
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
                state, request_id=request_id, operation="checkpoint_conversation", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), epoch=int(conversation_epoch), handle=conversation_handle, turn=int(turn_generation), parent=parent_turn_identity)
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
                # The ledger lives on this record and survives a checkpoint:
                # only a reset/retirement (a new context) clears it.
                "pulled": list((current or {}).get("pulled") or []),
                "pull_sources": dict((current or {}).get("pull_sources") or {}),
                "protocol_repairs": int((current or {}).get("protocol_repairs") or 0),
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
                operation="checkpoint_conversation", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), epoch=int(conversation_epoch), handle=conversation_handle, turn=int(turn_generation), parent=parent_turn_identity),
                response=response,
                task_id=task_id,
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
                state, request_id=request_id, operation="reset_conversation", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason)
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
            if state.get("session_scope") != "assignment":
                # Task scope keeps its pull ledger on the task row (docs
                # §2.4): a reset is a new context there too, same lease.
                task["pulled"] = []
                task["pull_sources"] = {}
                task["protocol_repairs"] = 0
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
                operation="reset_conversation", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason),
                response=response,
                task_id=task_id,
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
        worker_kind: str = "headless",
    ) -> dict[str, Any]:
        worker_id = _safe_component(worker_id, label="worker id")
        if worker_kind not in WORKER_KINDS:
            raise ValueError(f"worker_kind must be one of {sorted(WORKER_KINDS)}")
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="next_task", fingerprint=_fp(worker=worker_id, control=int(expected_control_generation))
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
            workers = state.setdefault("workers", {})
            known_kind = str((workers.get(worker_id) or {}).get("kind") or "")
            if known_kind and known_kind != worker_kind:
                raise AssignmentConflictError(
                    f"worker {worker_id!r} registered as {known_kind!r}, not {worker_kind!r}"
                )
            workers[worker_id] = {"kind": worker_kind}
            active = self._active_task_for_worker(state, worker_id)
            if active is not None:
                self._renew_lease_locked(state["tasks"][active["spec"]["task_id"]])
                response = self._response_for_state(state, worker_id=worker_id)
            else:
                ready = (
                    self._first_ready_task(state, executor="agent", kind=worker_kind)
                    if self._has_worker_capacity(state)
                    else None
                )
                if ready is not None:
                    self._lease_task_locked(state, ready, worker_id=worker_id)
                    if worker_kind == "conversational":
                        # The files are the hand-over (docs §12.1.4): a
                        # person's agent reads the protocol and payload from
                        # the assignment root with its own tools, so the
                        # ledger books them as pushed at the claim.
                        ledger = self._ledger_locked(
                            state, state["tasks"][ready["spec"]["task_id"]], worker_id=worker_id
                        )
                        ledger["pulled"] = sorted(
                            set(ledger.get("pulled") or [])
                            | {self._block_key(block) for block in ready["spec"].get("required_blocks") or []}
                        )
                    response = self._response_for_state(state, worker_id=worker_id)
                elif self._assignment_complete(state) or self._failed_tasks(state):
                    response = self._response_for_state(state, worker_id=worker_id)
                else:
                    response = self._new_wait_response(state, worker_id=worker_id)
            response["control_generation"] = int(state["control_generation"]) + 1
            # A claim that found nothing is not worth a durable record: a
            # parked session asks again and again (pseudo-conversational long
            # polling), `next_task` is claim-or-resume anyway, and the table
            # is bounded.
            if response.get("status") != "waiting":
                self._record_request(
                    state, request_id=request_id, operation="next_task", fingerprint=_fp(worker=worker_id, control=int(expected_control_generation)), response=response,
                    task_id=str((response.get("task") or {}).get("task_id") or ""),
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
                state, request_id=request_id, operation="checkpoint_progress", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), progress=progress)
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
                operation="checkpoint_progress", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), progress=progress),
                response=response,
                task_id=task_id,
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
                # The refusal must not read as "retrieval is unavailable": a
                # model that misreads it discards results it already has and
                # marks verified facts unverified (docs/report 2026-08-28
                # §2.4 — exactly that happened).
                row = {
                    "status": "budget_exhausted",
                    "operation": operation,
                    "request_digest": request_digest,
                    "response": {
                        "status": "budget_exhausted",
                        "reason": reason + self._budget_refusal_note(ledger),
                    },
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
                    "reason": (
                        "this call was REFUSED because its result would exceed "
                        + ", ".join(violations)
                        + (
                            f" (response_tokens {int(ledger['response_tokens'])}"
                            f"+{int(response_tokens)}/{int(limits['max_response_tokens'])})"
                            if "max_response_tokens" in violations
                            else ""
                        )
                        + self._budget_refusal_note(ledger)
                    ),
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

    def retrieval_search_events(
        self, *, assignment_id: str, task_id: str
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """This task's proxied retrieval, in the driver's search-event shape.

        A tool session searches through the harness (`web_search`/`web_fetch`),
        so the URLs land in this ledger and never in the driver's normalized
        events -- which is where every provenance consumer looks
        (`knowledge.feedback.retrieval_urls_from_response`,
        `research._mark_unverified_sources`). The result was that the one path
        where the harness *itself* fetched the sources recorded no source at
        all, and knowledge written from an agent round carried no provenance.

        Read-only and lease-free: this runs after the session is gone.

        Returns ``(rows, unreadable)``: a result this cannot read is dropped
        from the evidence, so the caller is handed its ref to report rather
        than left to publish a silently shortened source list.
        """

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = state["tasks"].get(task_id)
            if task is None:
                return [], []
            rows: list[dict[str, Any]] = []
            unreadable: list[str] = []
            for row in task["retrieval"]["calls"].values():
                if row.get("status") != "completed" or not row.get("result_ref"):
                    continue
                try:
                    result = json.loads(self._read_ref(str(row["result_ref"])))
                except (AgentTaskRuntimeError, OSError, ValueError):
                    unreadable.append(str(row["result_ref"]))
                    continue
                request = row.get("request") or {}
                urls = [
                    str(item["url"])
                    for item in (result.get("items") or [])
                    if isinstance(item, Mapping) and str(item.get("url") or "").startswith("http")
                ]
                url = str(request.get("url") or result.get("url") or "")
                if url.startswith("http") and url not in urls:
                    urls.append(url)
                operation = str(row.get("operation") or "search")
                rows.append(
                    {
                        "event": "item.completed",
                        # A fetch is not a search; consumers read `urls`, so
                        # there is nothing to buy by calling it one.
                        "item_type": f"web_{operation}",
                        "tool": f"web_{operation}",
                        "query": str(request.get("query") or result.get("query") or url),
                        "urls": urls,
                    }
                )
            return rows, unreadable

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
    def _budget_refusal_note(ledger: Mapping[str, Any]) -> str:
        """Tail every budget refusal with what the session ALREADY has.

        "budget_exhausted" alone was misread as "retrieval is unavailable":
        the model then reported zero sources and downgraded facts it had in
        fact retrieved (docs/report 2026-08-28 §2.4). A refusal is not a
        failure and earlier results stay valid — say so, with counts."""

        completed = sum(
            1 for row in ledger["calls"].values() if row.get("status") == "completed"
        )
        return (
            f"; this is a REFUSAL of the new call, not a retrieval failure — "
            f"{completed} earlier call(s) completed and returned "
            f"{int(ledger['results_returned'])} result(s), which remain valid; "
            "do not treat retrieval as unavailable or mark already-retrieved "
            "facts as unverified"
        )

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
                state, request_id=request_id, operation="release_task", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason)
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
                state, request_id=request_id, operation="release_task", fingerprint=_fp(task=task_id, worker=worker_id, lease=int(lease_generation), reason=reason), response=response,
                task_id=task_id,
            )
            self._commit_state_locked(state)
            return response

    def forget_drafts(self, *, assignment_id: str) -> int:
        """Drop every rejected draft this assignment is still carrying.

        `last_candidate` is a whole window of subtitle text, kept so that a
        session ending without an accepted answer can report what it last
        tried. A caller that is clearing the run's text has to clear these
        too -- they live in durable state, which is exactly what a
        conversational queue keeps as its tombstone, so leaving them would
        preserve the text the clean-up exists to drop.

        Clearing the current row is not enough. State is written as one
        append-only snapshot per change and a bounded tail of the older ones
        is kept for forensics, so the snapshot from the rejected submit still
        holds the draft in full -- and `control/state/` is exactly what the
        tombstone preserves. A single-task session never reaches the 20-
        generation trim line, so that tail is *always* still there. Every
        superseded snapshot therefore goes: only the generation `index.json`
        names is ever read, which is why they were forensics in the first
        place, and dropping the text is the whole point here.

        Not done at accept time: while the tree stands, the audit bundle
        reads the rejected draft as the record of a repair round.
        """

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            dropped = 0
            for task in state["tasks"].values():
                if task.pop("last_candidate", None) is not None:
                    dropped += 1
            if dropped:
                state = self._commit_state_locked(state)
            self._discard_superseded_states_locked(
                int(state["control_generation"])
            )
            return dropped

    def _discard_superseded_states_locked(self, current_generation: int) -> None:
        """Remove every state snapshot but the one in use."""

        try:
            entries = list(self.state_root.iterdir())
        except OSError:
            return
        for entry in entries:
            if entry.suffix != ".json" or not entry.stem.isdigit():
                continue
            if int(entry.stem) >= current_generation:
                continue
            try:
                entry.unlink()
            except OSError:
                continue

    def lint(
        self,
        *,
        assignment_id: str,
        task_id: str,
        worker_id: str,
        lease_generation: int,
        candidate: Any,
    ) -> dict[str, Any]:
        """Run the task's validator over a candidate without submitting it.

        The one check in this protocol that does not depend on the other side
        policing itself: a truncated answer, a missing column, a source id
        never covered are all things the agent cannot reliably see in its own
        output but the validator sees immediately. Finding out at submit time
        costs a repair round and, in a conversation, often another truncated
        rewrite of the whole thing.

        Deliberately not a submit: no budget spent, no repair counted, no
        verdict cached, task status untouched, nothing recorded in the request
        table. The lease *is* renewed -- every control command does, that is
        the liveness signal this protocol has instead of a heartbeat, and an
        agent should not lose its task for checking its work. Owed blocks are
        reported too, without spending a protocol repair, so that a clean lint
        really does predict a clean submit.
        """

        with self._locked():
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            validator_id = task["spec"]["validator_id"]
            validator = self._validators.get(validator_id)
            if validator is None:
                raise AgentTaskRuntimeError(f"Validator {validator_id!r} is unavailable")
            ledger = self._ledger_locked(state, task, worker_id=worker_id)
            owed = self._owed_blocks(task, ledger)
            manifest = self._read_ref(task["manifest_ref"])
            result = validator(_json_copy(candidate), json.loads(manifest)).normalized()
            self._renew_lease_locked(task)
            self._commit_state_locked(state)
            return {
                "status": "lint",
                "validator_id": validator_id,
                "verdict": result.status,
                "validation_errors": list(result.errors),
                "owed_blocks": owed,
            }

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
        fingerprint = _sha256_text(
            _stable_json(
                {
                    "task": task_id,
                    "lease": int(lease_generation),
                    "input_hash": input_hash,
                    "candidate": _json_copy(candidate),
                }
            )
        )
        with self._locked():
            self._recover_wal_locked()
            state = self._load_state_locked()
            self._assert_assignment(state, assignment_id)
            duplicate = self._deduplicated(
                state,
                request_id=request_id,
                operation="submit",
                fingerprint=fingerprint,
            )
            if duplicate is not None:
                return duplicate
            replayed_accept = self._accepted_replay_locked(
                state, task_id=task_id, request_id=request_id, fingerprint=fingerprint
            )
            if replayed_accept is not None:
                return replayed_accept
            task = self._validate_lease(
                state,
                task_id=task_id,
                worker_id=worker_id,
                lease_generation=lease_generation,
            )
            if task["spec"]["input_hash"] != input_hash:
                raise StaleLeaseError("submit input_hash does not match the leased task")
            # Content idempotence (docs §7): the request id above only catches
            # a transport replay on one live connection. A new id carrying an
            # answer this context already had judged is the same answer again
            # -- a reconnected CLI re-sending, or a model that will not change
            # its mind -- and gets the first verdict back without spending
            # repair budget. Every call still counts against the submit cap,
            # which is what ends the session that never changes its answer.
            task["submit_count"] = int(task.get("submit_count", 0)) + 1
            if task["submit_count"] > self._submit_budget(task)["max_submits"]:
                if isinstance(candidate, str):
                    task["last_candidate"] = candidate
                errors = [
                    f"submit cap reached ({self._submit_budget(task)['max_submits']} "
                    "calls in this context) without an accepted answer"
                ]
                task["validation_errors"] = errors
                response = self._retire_task_locked(
                    state, task, worker_id=worker_id, reason="submit cap exceeded"
                )
                response["validation_errors"] = errors
                response["protocol_violation"] = "submit_cap"
                self._record_request(
                    state,
                    request_id=request_id,
                    operation="submit",
                    response=response,
                    fingerprint=fingerprint,
                    task_id=task_id,
                )
                self._commit_state_locked(state)
                return response
            replay = dict(task.get("submissions") or {}).get(fingerprint)
            if replay is not None:
                response = deepcopy(replay)
                response.update(self._submit_budget(task))
                response["replayed"] = True
                self._renew_lease_locked(task)
                self._record_request(
                    state,
                    request_id=request_id,
                    operation="submit",
                    response=response,
                    fingerprint=fingerprint,
                    task_id=task_id,
                )
                self._commit_state_locked(state)
                return response
            validator_id = task["spec"]["validator_id"]
            validator = self._validators.get(validator_id)
            if validator is None:
                raise AgentTaskRuntimeError(f"Validator {validator_id!r} is unavailable")
            # The gate in front of the validator (docs §2.4): every required
            # block must be in this context's ledger. The first miss names
            # every owed block and the tool that fetches it; a second miss
            # in the same context is a session that is not following the
            # protocol, and goes through the expensive loop like `blocked`.
            ledger = self._ledger_locked(state, task, worker_id=worker_id)
            owed = self._owed_blocks(task, ledger)
            if owed:
                errors = [
                    f"required block {block['kind']} ({block['digest']}) was never "
                    f"read in this context: call {block['tool']}"
                    + (f" ref={block['ref']}" if block.get("ref") else "")
                    for block in owed
                ]
                repairs = int(ledger.get("protocol_repairs") or 0)
                if repairs >= PROTOCOL_REPAIRS_PER_CONTEXT:
                    # Still owing after its one repair: the session is not
                    # following the protocol. Retire it (docs §0-9 / §5.3) --
                    # conversation reset, lease revoked, task re-queued,
                    # counted as a replacement -- in this same write.
                    task["validation_errors"] = list(errors)
                    if isinstance(candidate, str):
                        task["last_candidate"] = candidate
                    response = self._retire_task_locked(
                        state, task, worker_id=worker_id, reason="missing required blocks twice"
                    )
                    response["validation_errors"] = list(errors)
                    response["protocol_violation"] = "missing_required_blocks"
                    response["owed_blocks"] = owed
                    self._record_request(
                        state,
                        request_id=request_id,
                        operation="submit",
                        response=response,
                        fingerprint=fingerprint,
                        task_id=task_id,
                    )
                    self._commit_state_locked(state)
                    return response
                ledger["protocol_repairs"] = repairs + 1
                result = ValidationResult(
                    status="repairable",
                    errors=tuple(errors),
                    metadata={"protocol_violation": "missing_required_blocks"},
                ).normalized()
            else:
                manifest = self._read_ref(task["manifest_ref"])
                result = validator(_json_copy(candidate), json.loads(manifest)).normalized()
            if result.status != "accepted":
                task["validation_errors"] = list(result.errors)
                response = {
                    "status": result.status,
                    "control_generation": int(state["control_generation"]) + 1,
                    "validation_errors": list(result.errors),
                }
                if result.metadata.get("protocol_violation"):
                    response["protocol_violation"] = result.metadata["protocol_violation"]
                    response["owed_blocks"] = owed
                if result.status == "repairable":
                    # Cheap loop: same lease, same conversation, errors fed
                    # back. Renewing keeps a long repair round from losing the
                    # task it is in the middle of fixing.
                    task["status"] = "repairing"
                    self._renew_lease_locked(task)
                    if not result.metadata.get("protocol_violation"):
                        # A distinct rejected answer spends one repair; a
                        # missing-block rejection is the ledger's business
                        # and is neither budgeted nor cached -- its verdict
                        # depends on what has been read since, not on the
                        # candidate.
                        task["repair_attempts"] = int(task.get("repair_attempts", 0)) + 1
                        task.setdefault("submissions", {})[fingerprint] = _json_copy(
                            response
                        )
                    # The harness reads this back when a tool-protocol session
                    # ends without an accepted submit: the last rejected
                    # output is what a replacement round reports.
                    if isinstance(candidate, str):
                        task["last_candidate"] = candidate
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
                response.update(self._submit_budget(task))
                self._ack_conversation(state, worker_id=worker_id, response=response)
                self._record_request(
                    state,
                    request_id=request_id,
                    operation="submit",
                    response=response,
                    fingerprint=fingerprint,
                    task_id=task_id,
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
            # The lease goes with the accept; who did it stays for the record.
            target_task["accepted_by"] = worker_id
            self._prune_task_requests_locked(target, task_id)
            next_ready = self._first_ready_task(
                target, executor="agent", kind=self._worker_kind(target, worker_id)
            )
            if next_ready is not None and self._has_worker_capacity(target):
                self._lease_task_locked(target, next_ready, worker_id=worker_id)
            target["control_generation"] = int(state["control_generation"]) + 1
            response = self._response_for_state(target, worker_id=worker_id)
            response["accepted_task_id"] = task_id
            response["accepted_artifact_ref"] = artifact_ref
            self._ack_conversation(target, worker_id=worker_id, response=response)
            # Not in the request table: that table is bounded per assignment
            # and a session serving a whole run would fill it with one
            # permanent row per accepted task. The answer to a replayed
            # accept lives on the task row instead, which the task already
            # pays for (`_accepted_replay_locked`).
            target_task["accepted_submit"] = {
                "request_id": request_id,
                "fingerprint": fingerprint,
                "response": _json_copy(response),
            }
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
            fingerprint = _fp(task=task_id, control=int(expected_control_generation), input_hash=input_hash, artifact=artifact)
            duplicate = self._deduplicated(
                state, request_id=request_id, operation="accept_external_task", fingerprint=fingerprint
            )
            if duplicate is not None:
                return duplicate
            replayed_accept = self._accepted_replay_locked(
                state, task_id=task_id, request_id=request_id, fingerprint=fingerprint
            )
            if replayed_accept is not None:
                return replayed_accept
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
            # Not in the request table, for the same reason an agent's accept
            # is not: the table is bounded per assignment and a scheduler
            # feeding one external dependency per window would fill it. The
            # answer to a replayed accept lives on the task row.
            self._prune_task_requests_locked(target, task_id)
            target_task["accepted_submit"] = {
                "request_id": request_id,
                "fingerprint": fingerprint,
                "response": _json_copy(response),
            }
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
        if not is_artifact_reference(reference):
            # A bare unpack raised ValueError here, and callers that log
            # ``str(exc)`` reported "not enough values to unpack" -- true, and
            # useless. Say which ref, in the runtime's own error type.
            raise AgentTaskRuntimeError(
                f"Not a digest-bound artifact reference: {reference!r}"
            )
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
            state, executor="agent", kind=self._worker_kind(state, worker_id)
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
