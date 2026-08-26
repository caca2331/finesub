"""The harness's own MCP server: how an agent takes a task and hands it back.

docs/llm_agent_tool_protocol.md §2 (the production tool table), §1 (the
server *is* the worker: it opens the runtime root it was told about and
claims with the worker id it was given), §3 (required blocks are booked as
the agent reads them) and §2 "request_id" (operation ids derive from the
server session, server-process instance and JSON-RPC id, so a transport replay
on one live connection hits the runtime's dedup while a restarted MCP process
cannot collide after its JSON-RPC counter resets).

Zero dependencies by design -- newline-delimited JSON-RPC over stdio is all
three CLIs need (measured 2026-08-17/21). The CLI spawns this process from
the per-invocation server entry the driver wrote; everything it needs arrives
in the environment:

    FINESUB_MCP_ROOT          assignment root (the runtime's ``root``)
    FINESUB_MCP_ASSIGNMENT    assignment id
    FINESUB_MCP_WORKER        worker id this server acts as
    FINESUB_MCP_SESSION       server session id (one per CLI invocation)
    FINESUB_MCP_LOG           optional: append every frame here (audit)
    FINESUB_MCP_TOOLS         optional: comma-separated tool names to expose
                              (a pseudo-conversational session is authorized
                              for every tool at launch; admission is per task)
    FINESUB_MCP_WAIT_SECONDS  optional: how long `next_task` may park before
                              answering `still_waiting` (default 25)
    FINESUB_MCP_BLOCK_FILES   optional: "1" hands every required block over
                              as a file (`path` + `read: "file"` in the
                              manifest) for the CLI's own file tool to read
    FINESUB_MCP_PAGE_CHARS    optional: largest tool reply the CLI shows the
                              model inline (0 = unlimited). Blocks longer than
                              this are not pushed by `next_task` and
                              `read_context` serves them page by page. agy
                              replaces a larger reply with a file path the
                              worker may not read (measured 2026-08-22).

Run with ``python -m finesub.llm.agent.agent_mcp_server``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

from .agent_task_runtime import (
    AgentTaskRuntime,
    AgentTaskRuntimeError,
    StaleControlGenerationError,
)
from .agent_validators import VALIDATOR_BUILDERS, runtime_validators

SERVER_NAME = "finesub"
PROTOCOL_VERSION = "2025-06-18"
# How many times one `next_task` re-plans against a control generation that
# moved under it before answering `still_waiting`. Each attempt re-reads the
# state that won, so losing repeatedly means the harness is writing fast, not
# that anything is wrong.
CLAIM_ATTEMPTS = 4
# How long one `next_task` call may park waiting for the harness to add the
# next task before answering `still_waiting` (pseudo-conversational). The
# ceiling is the CLI's own MCP tool-call timeout, which differs per vendor
# and is measured per driver; this is the conservative starting point
# (docs/llm_followups.md, second-round decisions).
DEFAULT_NEXT_TASK_WAIT_SECONDS = 25.0
# Required blocks small enough to ride the `next_task` answer (docs §0-6):
# the output contract and the material itself. Indexes and the like stay pull.
PUSHED_KINDS = frozenset({"protocol", "payload"})

# Annotations are honest (docs §2): Codex's `auto` approval gates on
# destructive/open-world, and a tool that claims read-only while it claims a
# lease would be lying to the thing deciding whether to let it run.
TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "next_task",
        "description": (
            "Claim (or resume) this worker's task. Returns the task manifest: "
            "session type, required blocks with the tool that fetches each, "
            "and the repair budget. `still_waiting` means no task is ready "
            "yet: call it again. `assignment_complete` means there is no more "
            "work: stop."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {
            "title": "Take the task",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "read_context",
        "description": (
            "Read one resource by its ref from the task manifest (protocol, "
            "payload, indexes), one page at a time: the reply carries `text`, "
            "`offset`, `total_chars` and `next_offset`; call again with "
            "`offset = next_offset` until `next_offset` is null. A required "
            "block is booked as seen once its last page was read."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
            },
            "required": ["ref"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Read context",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
    {
        "name": "pull_status",
        "description": "Which required blocks this context still owes before submit can be accepted.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {
            "title": "Pull status",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit the answer. Returns accepted (done), repairable (fix and "
            "submit again) or blocked/failed (stop)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Submit the answer",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    },
)
TOOL_NAMES: tuple[str, ...] = tuple(tool["name"] for tool in TOOL_DEFINITIONS)

# Exposed only when the task's retrieval mode is `local`: the harness's own
# retrieval chain (docs/llm_harness_research.md), metered by the runtime's
# retrieval ledger. Open-world and not idempotent -- a search is a search.
WEB_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "web_search",
        "description": (
            "Search the web through the harness retrieval proxy. Budgeted per "
            "task; returns result rows with URLs you may then fetch."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "guided_query": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Web search",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch one page whose URL came back from an earlier web_search in "
            "this task. Budgeted per task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "guided_query": {"type": "string"},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        "annotations": {
            "title": "Web fetch",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True,
        },
    },
)
WEB_TOOL_NAMES: tuple[str, ...] = tuple(tool["name"] for tool in WEB_TOOL_DEFINITIONS)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _page_end(text: str, offset: int, limit_bytes: int) -> int:
    """The end index of the page starting at ``offset`` that fits ``limit_bytes``.

    Bisected rather than grown one character at a time: the limit is a driver
    fact, and a wide one would otherwise re-encode the whole page per step.
    """

    if _utf8_len(text[offset:]) <= limit_bytes:
        return len(text)
    # Characters are at most four bytes, so this many always fit.
    low = offset + max(1, limit_bytes // 4)
    high = min(len(text), offset + limit_bytes)
    while low < high:
        middle = (low + high + 1) // 2
        if _utf8_len(text[offset:middle]) <= limit_bytes:
            low = middle
        else:
            high = middle - 1
    end = low
    cut = text.rfind("\n", offset, end)
    if cut > offset:
        end = cut + 1
    return end


def request_id_for(
    *,
    assignment_id: str,
    worker_id: str,
    session_id: str,
    instance_id: str,
    rpc_id: Any,
) -> str:
    """``H(caller_session_id, server_instance, caller_sequence)``.

    The JSON-RPC id keeps its type in the hash (``1`` and ``"1"`` are two
    requests). Some CLIs restart their MCP subprocess inside one invocation
    and reset the JSON-RPC counter; the instance nonce prevents a new process's
    id from colliding with a different operation handled by the old process.
    A replay on one live connection still has the same id and deduplicates.
    """

    material = json.dumps(
        [
            assignment_id,
            worker_id,
            session_id,
            instance_id,
            type(rpc_id).__name__,
            rpc_id,
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return "mcp-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def assignment_manifests(runtime: AgentTaskRuntime) -> list[dict[str, Any]]:
    """Every task manifest of the assignment behind ``runtime``, cold."""

    try:
        index = json.loads(runtime.index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    refs = index.get("task_manifest_refs")
    manifests: list[dict[str, Any]] = []
    if isinstance(refs, Mapping):
        for ref in refs.values():
            try:
                manifests.append(json.loads(runtime.read_artifact(str(ref))))
            except (OSError, ValueError):
                continue
    return manifests


class _ToolError(Exception):
    """A tool call that failed in a way the model should hear about."""


class HarnessToolServer:
    """Dispatches the four harness tools for one worker of one assignment."""

    def __init__(
        self,
        runtime: AgentTaskRuntime,
        *,
        assignment_id: str,
        worker_id: str,
        session_id: str,
        instance_id: str,
        tools: Sequence[str] | None = None,
        next_task_wait_seconds: float = DEFAULT_NEXT_TASK_WAIT_SECONDS,
        page_chars: int = 0,
        block_files: bool = False,
    ) -> None:
        # Blocks as files (docs/llm_local_agent_agy.md §5): every required
        # block carries its absolute `path` and `read: "file"`; the CLI's own
        # file tool reads it, and the ledger books it as handed over.
        self.block_files = bool(block_files)
        self.runtime = runtime
        self.assignment_id = assignment_id
        self.worker_id = worker_id
        self.session_id = session_id
        self.instance_id = instance_id
        self.next_task_wait_seconds = max(1.0, float(next_task_wait_seconds))
        # 0: every block rides `next_task` and `read_context` returns whole
        # documents. Otherwise the CLI's inline limit: a reply past it would
        # reach the model as a file path instead of text.
        self.page_chars = max(0, int(page_chars))
        self._task: dict[str, Any] | None = None
        self._manifest: dict[str, Any] | None = None
        # No repair counter here on purpose: the budget lives on the durable
        # task row (docs §7), so a restarted MCP process -- Claude Code does
        # that inside one CLI invocation -- reads the same numbers back.
        # Exposure vs admission (docs §2): a pseudo-conversational session is
        # authorized for every tool when the CLI launches, because tasks with
        # other retrieval modes arrive later; whether a call is *allowed* is
        # decided per task when it is made.
        self._web_tools = (
            any(name in WEB_TOOL_NAMES for name in tools)
            if tools is not None
            else self._assignment_offers_web_tools()
        )
        # Transport replays: the same JSON-RPC id asked again gets the same
        # answer, and nothing -- the repair count included -- moves twice.
        self._replies: dict[str, dict[str, Any]] = {}  # request_id -> {fingerprint, reply}

    def _assignment_offers_web_tools(self) -> bool:
        """Whether any task in this assignment retrieves through the harness."""

        for manifest in assignment_manifests(self.runtime):
            if manifest.get("retrieval_mode") == "local":
                return True
        return False

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        tools = [dict(tool) for tool in TOOL_DEFINITIONS]
        if self._web_tools:
            tools.extend(dict(tool) for tool in WEB_TOOL_DEFINITIONS)
        return tools

    # -- JSON-RPC -------------------------------------------------------

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        method = str(message.get("method") or "")
        rpc_id = message.get("id")
        params = message.get("params") if isinstance(message.get("params"), Mapping) else {}
        if method == "initialize":
            return self._result(
                rpc_id,
                {
                    "protocolVersion": str(params.get("protocolVersion") or PROTOCOL_VERSION),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": SERVER_NAME, "version": "1"},
                },
            )
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return self._result(rpc_id, {})
        if method == "tools/list":
            return self._result(rpc_id, {"tools": self.tool_definitions})
        if method == "tools/call":
            name = str(params.get("name") or "")
            arguments = params.get("arguments") if isinstance(params.get("arguments"), Mapping) else {}
            return self._result(rpc_id, self._call(name, arguments, rpc_id=rpc_id))
        if rpc_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    @staticmethod
    def _result(rpc_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": rpc_id, "result": dict(result)}

    def _call(self, name: str, arguments: Mapping[str, Any], *, rpc_id: Any) -> dict[str, Any]:
        handlers = {
            "next_task": self._next_task,
            "read_context": self._read_context,
            "pull_status": self._pull_status,
            "submit": self._submit,
        }
        if self._web_tools:
            handlers["web_search"] = self._web_search
            handlers["web_fetch"] = self._web_fetch
        handler = handlers.get(name)
        if handler is None:
            return self._tool_error(f"Unknown tool {name!r}")
        request_id = request_id_for(
            assignment_id=self.assignment_id,
            worker_id=self.worker_id,
            session_id=self.session_id,
            instance_id=self.instance_id,
            rpc_id=rpc_id,
        )
        fingerprint = hashlib.sha256(
            json.dumps([name, arguments], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        cached = self._replies.get(request_id)
        if cached is not None:
            if cached["fingerprint"] != fingerprint:
                # The runtime's own rule (docs §2): an id reused for different
                # input is refused, never answered with the old response.
                return self._tool_error(
                    f"request id {rpc_id!r} was already used for a different {name} call"
                )
            return dict(cached["reply"])
        try:
            payload = handler(arguments, request_id=request_id)
        except _ToolError as exc:
            reply = self._tool_error(str(exc))
        except AgentTaskRuntimeError as exc:
            reply = self._tool_error(f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001 -- the model's arguments must not kill the server
            reply = self._tool_error(f"{type(exc).__name__}: {exc}")
        else:
            reply = {
                "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False)}],
                "isError": False,
            }
        self._replies[request_id] = {"fingerprint": fingerprint, "reply": reply}
        return dict(reply)

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- tools ----------------------------------------------------------

    def _require_task(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._task is None or self._manifest is None:
            raise _ToolError("call next_task first")
        return self._task, self._manifest

    def _record(self) -> dict[str, Any]:
        task, _manifest = self._require_task()
        return self.runtime.task_record(
            assignment_id=self.assignment_id, task_id=str(task["task_id"])
        )

    _DONE_REPLY = {
        "status": "assignment_complete",
        "message": "there is no more work in this assignment; reply `done` and stop",
    }

    def _next_task(self, _arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.next_task_wait_seconds
        claims = 0
        while True:
            status = self.runtime.rehydrate(
                assignment_id=self.assignment_id, worker_id=self.worker_id
            )
            if status["status"] == "task":
                record = self.runtime.task_record(
                    assignment_id=self.assignment_id,
                    task_id=str(status["task"]["task_id"]),
                )
                if record["status"] == "repairing" and record["repair_rounds_remaining"] <= 0:
                    # Spent: the harness is taking this task back. Handing it
                    # out again would only repeat the exhausted verdict, so
                    # the session waits here until it is gone.
                    if time.monotonic() >= deadline:
                        return {
                            "status": "still_waiting",
                            "message": "no task is ready yet; call next_task again",
                        }
                    time.sleep(0.2)
                    continue
                break
            if status["status"] == "assignment_complete":
                # Reconciliation on start (docs §7): a server process that
                # comes up after the work was accepted -- a reconnect, a CLI
                # retry -- must not offer the session a second submit. For a
                # session that just finished its last task this is the exit.
                return dict(self._DONE_REPLY)
            if status["status"] == "waiting":
                if not status.get("wait_token"):
                    # Not registered as a waiter yet: the claim below parks us.
                    pass
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return {
                            "status": "still_waiting",
                            "message": "no task is ready yet; call next_task again",
                        }
                    # Park on the runtime until the harness adds a task or
                    # seals the assignment (pseudo-conversational long poll).
                    self.runtime.await_next_task(
                        assignment_id=self.assignment_id,
                        worker_id=self.worker_id,
                        wait_token=str(status["wait_token"]),
                        max_wait_seconds=max(0.5, remaining),
                    )
                    continue
            if status["status"] not in {"ready", "waiting"}:
                return {"status": status["status"]}
            claims += 1
            if claims > CLAIM_ATTEMPTS:
                # Losing the race means the harness wrote -- it added the next
                # window's task -- so the honest answer is the one the session
                # already knows how to act on. An error here is not actionable
                # by the model, and a driver whose worker treats it as fatal
                # ends the whole session over a lost write race.
                return {
                    "status": "still_waiting",
                    "message": "no task is ready yet; call next_task again",
                }
            try:
                status = self.runtime.next_task(
                    assignment_id=self.assignment_id,
                    worker_id=self.worker_id,
                    request_id=f"{request_id}-{claims}" if claims > 1 else request_id,
                    expected_control_generation=status["control_generation"],
                )
            except StaleControlGenerationError:
                continue
            if status["status"] == "task":
                break
            if status["status"] == "assignment_complete":
                return dict(self._DONE_REPLY)
            if status["status"] != "waiting":
                return {"status": status["status"]}
        task = dict(status["task"])
        manifest = json.loads(self.runtime.read_artifact(str(task["manifest_ref"])))
        self._task, self._manifest = task, manifest
        required = list(manifest.get("required_blocks") or [])
        # First turn pushes the protocol-mandated small blocks (docs §0-6 /
        # §2.4 "首轮直接全推"): the agent gets them in this very answer and
        # the ledger books them as pushed, so a well-behaved session needs no
        # round trip per block. `read_context` stays available for re-reads
        # (after a compact) and for anything not pushed.
        pushed: dict[str, str] = {}
        pushed_blocks = []
        budget = self.page_chars
        for block in required:
            if block.get("kind") not in PUSHED_KINDS:
                continue
            try:
                text = self.runtime.read_artifact(str(block["ref"]))
            except (OSError, ValueError):
                continue
            block["chars"] = len(text)
            block["bytes"] = _utf8_len(text)
            if self.block_files:
                # The file is the hand-over: its path is in the manifest and
                # the CLI reads it with its own tool, resuming past its own
                # truncation. Booked as pushed for the same reason a pushed
                # body is -- the model had it at hand.
                block["read"] = "file"
                block["path"] = str(self.runtime.root / str(block["ref"]).split("#", 1)[0])
                pushed_blocks.append(block)
                continue
            # Only what fits the CLI's inline limit is pushed; the rest is
            # read page by page, and the reply as a whole stays under it.
            pushed_bytes = sum(_utf8_len(value) for value in pushed.values())
            if budget and (_utf8_len(text) > budget or pushed_bytes + _utf8_len(text) > budget):
                block["read"] = "paged"
                continue
            pushed[str(block["kind"])] = text
            pushed_blocks.append(block)
        if pushed_blocks:
            self.runtime.record_pull(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id + "-push",
                blocks=pushed_blocks,
                source="push",
            )
        return {
            "status": "task",
            "task_id": task["task_id"],
            "session_type": manifest.get("session_type"),
            "goal": manifest.get("goal"),
            "required_blocks": required,
            "blocks": pushed,
            "refs": {
                "protocol": manifest.get("protocol_ref") or "",
                "context": manifest.get("context_ref") or "",
            },
            "repair_rounds_remaining": self._record()["repair_rounds_remaining"],
        }

    def _read_context(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, manifest = self._require_task()
        ref = str(arguments.get("ref") or "").strip()
        if not ref:
            raise _ToolError("read_context needs a ref")
        readable = self._readable_refs(manifest)
        if ref not in readable:
            # Name the valid refs: a model that guesses (measured on agy) burns
            # a turn per guess, and the list is short.
            raise _ToolError(
                f"ref {ref!r} is not a resource of this task; the readable refs are: "
                + ", ".join(sorted(readable))
            )
        text = self.runtime.read_artifact(ref)
        try:
            offset = max(0, int(arguments.get("offset") or 0))
        except (TypeError, ValueError):
            raise _ToolError("offset must be an integer") from None
        if offset > len(text):
            raise _ToolError(f"offset {offset} is past the end ({len(text)} chars)")
        if self.page_chars:
            # The limit is UTF-8 bytes (agy's cut is on the reply's size, and
            # CJK text is three bytes a character). Pages end on a line break
            # where one exists, so a CSV row or a subtitle line is never split
            # across two replies.
            end = _page_end(text, offset, self.page_chars)
        else:
            end = len(text)
        page = text[offset:end]
        next_offset = end if end < len(text) else None
        matched = [
            block for block in manifest.get("required_blocks") or [] if block.get("ref") == ref
        ]
        owed: list[dict[str, Any]] | None = None
        if matched and next_offset is None:
            # Booked on the last page: "had the chance to see it" means all
            # of it (docs/llm_agent_tool_protocol.md §3).
            booked = self.runtime.record_pull(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
                blocks=matched,
            )
            owed = list(booked.get("owed_blocks") or [])
        payload: dict[str, Any] = {
            "ref": ref,
            "text": page,
            "offset": offset,
            "total_chars": len(text),
            "next_offset": next_offset,
        }
        if owed is not None:
            payload["owed_blocks"] = owed
        return payload

    @staticmethod
    def _readable_refs(manifest: Mapping[str, Any]) -> set[str]:
        """The refs this task may read: what its manifest names, nothing else."""

        refs = {
            str(block.get("ref") or "")
            for block in manifest.get("required_blocks") or []
        }
        for key in ("protocol_ref", "context_ref", "knowledge_ref"):
            refs.add(str(manifest.get(key) or ""))
        refs.discard("")
        return refs

    def _pull_status(self, _arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, _manifest = self._require_task()
        status = self.runtime.pull_status(
            assignment_id=self.assignment_id,
            task_id=str(task["task_id"]),
            worker_id=self.worker_id,
        )
        return {"owed_blocks": status["owed_blocks"], "pulled": status["pulled"]}

    def _retrieval(self):
        from .agent_retrieval import AgentRetrievalAccess

        return AgentRetrievalAccess(self.runtime)

    def _web_search(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, manifest = self._require_task()
        if manifest.get("retrieval_mode") != "local":
            raise _ToolError("this task does not retrieve through the harness")
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise _ToolError("web_search needs a query")
        return dict(
            self._retrieval().search(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
                query=query,
                guided_query=str(arguments.get("guided_query") or ""),
            )
        )

    def _web_fetch(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, manifest = self._require_task()
        if manifest.get("retrieval_mode") != "local":
            raise _ToolError("this task does not retrieve through the harness")
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise _ToolError("web_fetch needs a url")
        return dict(
            self._retrieval().fetch(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
                url=url,
                guided_query=str(arguments.get("guided_query") or ""),
            )
        )

    _EXHAUSTED = {
        "status": "repair_exhausted",
        "message": "the repair budget for this session is spent; stop",
    }

    def _stop_reply(self, record: Mapping[str, Any]) -> dict[str, Any] | None:
        """What a session that may no longer submit is told, from durable state.

        Derived on every call rather than remembered, so a restarted server
        process says the same thing the old one would have (docs §7).
        """

        if record["status"] == "accepted":
            return {"status": "accepted", "message": "accepted; reply `submitted` and stop"}
        if record["lease_owner"] != self.worker_id:
            return {
                "status": "retired",
                "validation_errors": list(record.get("validation_errors") or []),
                "message": "this session was retired for not following the protocol; stop",
            }
        if record["status"] == "repairing" and record["repair_rounds_remaining"] <= 0:
            return dict(self._EXHAUSTED)
        return None

    def _submit(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, manifest = self._require_task()
        stop = self._stop_reply(self._record())
        if stop is not None:
            return stop
        payload = arguments.get("payload")
        if not isinstance(payload, str) or not payload.strip():
            raise _ToolError("submit needs a non-empty string payload")
        response = self.runtime.submit(
            assignment_id=self.assignment_id,
            task_id=str(task["task_id"]),
            worker_id=self.worker_id,
            lease_generation=int(task["lease_generation"]),
            request_id=request_id,
            input_hash=str(manifest["input_hash"]),
            candidate=payload,
        )
        status = str(response.get("status") or "")
        if response.get("accepted_task_id"):
            self._task, self._manifest = None, None
            if status == "assignment_complete":
                return {"status": "accepted", "message": "accepted; reply `submitted` and stop"}
            # An unsealed assignment (pseudo-conversational) has more coming:
            # the session goes back for the next task instead of leaving.
            return {"status": "accepted", "message": "accepted; call next_task for the next one"}
        if status == "retired":
            # The runtime retired this session (second protocol miss, or the
            # submit cap); its lease is gone, so nothing else it sends can land.
            return {
                "status": "retired",
                "validation_errors": list(response.get("validation_errors") or []),
                "message": "this session was retired for not following the protocol; stop",
            }
        out: dict[str, Any] = {
            "status": status,
            "validation_errors": list(response.get("validation_errors") or []),
        }
        if response.get("owed_blocks"):
            out["owed_blocks"] = response["owed_blocks"]
        if status == "repairable" and not response.get("protocol_violation"):
            remaining = int(response.get("repair_rounds_remaining", 0))
            out["repair_rounds_remaining"] = remaining
            if response.get("replayed"):
                out["message"] = "this exact answer was already judged; change it"
            if remaining <= 0:
                out.update(self._EXHAUSTED)
        return out


def serve(stream_in, stream_out, server: HarnessToolServer, *, log_path: str = "") -> None:
    """Pump newline-delimited JSON-RPC until stdin closes."""

    log = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for raw in stream_in:
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
            if not line.strip():
                continue
            if log is not None:
                log.write(json.dumps({"dir": "in", "frame": line.rstrip("\n")}, ensure_ascii=False) + "\n")
                log.flush()
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                reply: dict[str, Any] | None = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                }
            else:
                reply = server.handle(message) if isinstance(message, Mapping) else None
            if reply is None:
                continue
            encoded = json.dumps(reply, ensure_ascii=False) + "\n"
            if log is not None:
                log.write(json.dumps({"dir": "out", "frame": encoded.rstrip("\n")}, ensure_ascii=False) + "\n")
                log.flush()
            stream_out.write(encoded)
            stream_out.flush()
    finally:
        if log is not None:
            log.close()


def main() -> int:
    root = os.environ.get("FINESUB_MCP_ROOT", "")
    assignment_id = os.environ.get("FINESUB_MCP_ASSIGNMENT", "")
    worker_id = os.environ.get("FINESUB_MCP_WORKER", "")
    session_id = os.environ.get("FINESUB_MCP_SESSION", "")
    if not (root and assignment_id and worker_id and session_id):
        sys.stderr.write("agent_mcp_server: FINESUB_MCP_ROOT/ASSIGNMENT/WORKER/SESSION are required\n")
        return 2
    # Every registered validator, resolved by id: this process judges a
    # submit with the same function the harness registered, and a task added
    # after the server started (pseudo-conversational) may name one the
    # first task did not.
    validators: dict[str, Any] = {}
    for validator_id in sorted(VALIDATOR_BUILDERS):
        validators.update(runtime_validators(validator_id))
    runtime = AgentTaskRuntime(root, validators=validators)
    tools_env = os.environ.get("FINESUB_MCP_TOOLS", "")
    try:
        wait_seconds = float(os.environ.get("FINESUB_MCP_WAIT_SECONDS", "") or DEFAULT_NEXT_TASK_WAIT_SECONDS)
    except ValueError:
        wait_seconds = DEFAULT_NEXT_TASK_WAIT_SECONDS
    try:
        page_chars = int(os.environ.get("FINESUB_MCP_PAGE_CHARS", "") or 0)
    except ValueError:
        page_chars = 0
    server = HarnessToolServer(
        runtime,
        assignment_id=assignment_id,
        worker_id=worker_id,
        session_id=session_id,
        instance_id=uuid.uuid4().hex,
        tools=[name.strip() for name in tools_env.split(",") if name.strip()] if tools_env else None,
        next_task_wait_seconds=wait_seconds,
        page_chars=page_chars,
        block_files=os.environ.get("FINESUB_MCP_BLOCK_FILES", "") == "1",
    )
    # The frames carry subtitle text; MCP stdio is UTF-8 regardless of the
    # console code page this process inherited (cp936/cp1252 on Windows would
    # raise on the first non-ASCII character). `\n` stays `\n`: the clients
    # split on it and a translated `\r\n` is harmless but pointless.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", newline="\n")
        except (AttributeError, ValueError):
            pass
    serve(sys.stdin.buffer, sys.stdout, server, log_path=os.environ.get("FINESUB_MCP_LOG", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
