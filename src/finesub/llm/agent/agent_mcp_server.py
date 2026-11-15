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
    FINESUB_MCP_LOG           optional: append every frame here, with this
                              session's id (audit; `delivery.sent_replies`)
    FINESUB_MCP_TOOLS         optional: comma-separated tool names to expose
                              (a pseudo-conversational session is authorized
                              for every tool at launch; admission is per task)
    FINESUB_MCP_WAIT_SECONDS  optional: how long `next_task` may park before
                              answering `still_waiting` (default 25)
    FINESUB_MCP_CHANNEL       optional: the driver's `channels.ChannelLimits`
                              as JSON -- how large a reply reaches the model
                              whole, and where a larger one goes (parts read
                              with the CLI's file tool or with `read_spill`).
                              Absent: every reply goes out whole.
    FINESUB_MCP_REPLY_LIMIT   dev: lower both reply limits ("ok,error" bytes)
    FINESUB_MCP_PREDELIVERED  optional: comma-separated block keys
                              (``kind@digest``) the harness already gave the
                              CLI in its opening message, having claimed the
                              task and booked them as pushed. `next_task`
                              lists a protocol or `kb_index` among them as
                              `read: "seen"` instead of sending it again.
    FINESUB_MCP_HOLD_ACCEPTED_SECONDS
                              optional: sit this long on the `accepted` reply
                              that completes the assignment before sending it.
                              Exit-on-accept sets it, so the driver ends the
                              CLI before the reply can buy one more model
                              request (docs/llm_agent_tool_protocol.md §4).

Run with ``python -m finesub.llm.agent.agent_mcp_server``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid
from typing import Any, Callable, Iterator, Mapping, Sequence

from .agent_task_runtime import (
    AgentTaskRuntime,
    AgentTaskRuntimeError,
    StaleControlGenerationError,
    StaleLeaseError,
)
from .agent_validators import VALIDATOR_BUILDERS, runtime_validators
from .channels import (
    CHANNEL_ENV,
    SPILL_DIRNAME,
    ChannelLimits,
    fit_reply,
    fragment,
    read_spill_part,
    reply_id_for,
)

from finesub.text import (
    decodable_escape_count,
    looks_escaped,
    unescape_codepoints,
)

#: The environment key behind ``accepted_reply_hold_seconds``.
ACCEPTED_REPLY_HOLD_ENV = "FINESUB_MCP_HOLD_ACCEPTED_SECONDS"
#: The environment key behind ``predelivered``.
PREDELIVERED_ENV = "FINESUB_MCP_PREDELIVERED"
# Who `_delivered` names for a block the opening message carried: no task id
# is empty, so the task the opening was for still counts it as delivered.
_OPENING = ""

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
        # `resend` is left out of the description on purpose: only a session
        # that already has its task needs it, and the reply that says so
        # tells it how (fragment_next_task_already_sent_v1.md).
        "inputSchema": {
            "type": "object",
            "properties": {"resend": {"type": "boolean"}},
            "additionalProperties": False,
        },
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
            "payload, indexes), whole. "
            "A required block is booked as seen."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"ref": {"type": "string"}},
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

# Exposed on every call of a driver whose larger replies spill into parts the
# model reads through the server (`channels.ChannelLimits.overflow == "tool"`),
# so the tool table does not change between windows; used only when a reply
# says so.
READ_SPILL_DEFINITION: dict[str, Any] = {
    "name": "read_spill",
    "description": (
        "Read one part of a reply that was too large to arrive in one tool "
        "result; the reply names its spill_id and how many parts it has."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "spill_id": {"type": "string"},
            "part": {"type": "integer", "minimum": 1},
        },
        "required": ["spill_id", "part"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Read a spilled reply",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}
#: What `batch` may run: the read-only retrieval tools. Its own annotation is
#: read-only, which is honest only while every member is -- Codex's `auto`
#: approval lets a tool through on those hints, so a member that books or
#: writes (`read_context` books delivery) would make the honest annotation one
#: that is never approved. `read_spill` stays out too: a part already fills a
#: reply, so two together would spill again.
BATCH_MEMBERS = ("kb_index", "kb_search", "kb_read", "kb_read_node", "kb_validate", "web_search", "web_fetch")
# Exposed only to a CLI whose model cannot call tools in parallel (Codex runs
# GPT-6 without parallel calls): independent lookups in one round.
BATCH_DEFINITION: dict[str, Any] = {
    "name": "batch",
    "description": (
        "Run several independent lookups in one call: "
        + ", ".join(BATCH_MEMBERS)
        + ". Use it whenever you would otherwise make two or more of these "
        "calls in a row. The calls run in order and none sees another's "
        "result, so a call that needs an earlier result goes in a later "
        "round. Returns one result or error per call, in order."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "calls": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {"type": "string", "enum": list(BATCH_MEMBERS)},
                        "arguments": {"type": "object"},
                    },
                    "required": ["tool"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["calls"],
        "additionalProperties": False,
    },
    "annotations": {
        "title": "Batch lookups",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        # `web_search` / `web_fetch` are members.
        "openWorldHint": True,
    },
}

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

# Read-only knowledge-store tools (docs/archive/knowledge-node-plan.md §4.3, step 4a).
# Exposed only when the spawning harness hands the server a knowledge root
# (FINESUB_MCP_KNOWLEDGE_ROOT); each call is then admitted against the current
# task's manifest, which must carry ``metadata.knowledge_identity`` ("rev:N") —
# the pinned revision every reply reads. Hard reply cap, no pagination (v8):
# an oversized reply is an error telling the model how to narrow the request.
KB_REPLY_MAX_CHARS = 24_000
_KB_READONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
KB_TOOL_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "kb_index",
        "description": (
            "Both knowledge indexes (one line per entry: key, native names, "
            "aliases, one-line intro) at this task's pinned revision."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"title": "Knowledge index", **_KB_READONLY_ANNOTATIONS},
    },
    {
        "name": "kb_search",
        "description": (
            "Exact-match the query against entry surfaces, aliases and known "
            "misheard forms (kana-folded). Returns candidate rows — you judge "
            "relevance and disambiguate same-name candidates from context."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "entry": {"type": "string", "description": "restrict to one entry (key or alias)"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Knowledge search", **_KB_READONLY_ANNOTATIONS},
    },
    {
        "name": "kb_read",
        "description": (
            "One entry rendered at the pinned revision; nodes carry @k handles "
            "for kb_read_node. If the reply is too large, narrow it with "
            "sections=[...] (the section names are in the entry itself)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry": {"type": "string"},
                "sections": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["entry"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Knowledge entry", **_KB_READONLY_ANNOTATIONS},
    },
    {
        "name": "kb_read_node",
        "description": "One node by an @k handle from an earlier kb_read/kb_search reply of this session.",
        "inputSchema": {
            "type": "object",
            "properties": {"handle": {"type": "string"}},
            "required": ["handle"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Knowledge node", **_KB_READONLY_ANNOTATIONS},
    },
    {
        "name": "kb_validate",
        "description": (
            "Pre-check a <knowledge_proposals> block (or raw JSONL) against "
            "the pinned revision without writing anything: returns the ops "
            "that would apply, the rows that would be skipped, and overlay "
            "problems. Knowledge-update tasks only."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"proposals": {"type": "string"}},
            "required": ["proposals"],
            "additionalProperties": False,
        },
        "annotations": {"title": "Validate knowledge proposals", **_KB_READONLY_ANNOTATIONS},
    },
)
KB_TOOL_NAMES: tuple[str, ...] = tuple(tool["name"] for tool in KB_TOOL_DEFINITIONS)
#: Every tool this server can serve, whichever subset a call exposes. A name
#: outside it is no tool at all -- the driver's event check reads such a call
#: as a malformed one, not as reaching for a tool the call was not given.
SERVER_TOOL_NAMES: frozenset[str] = frozenset(
    (*TOOL_NAMES, *WEB_TOOL_NAMES, *KB_TOOL_NAMES, READ_SPILL_DEFINITION["name"], BATCH_DEFINITION["name"])
)
KB_READ_PROPOSE_NOTE = (
    " With a propose grant, the [别名] row carries an @a handle for "
    "<knowledge_proposals>; pre-check proposals with kb_validate."
)
#: Which kb tools each ``kb_tools`` entitlement may use (knowledge-node plan
#: §4.3 matrix). ``style`` is the narrow grant of a fast `--style auto` window
#: on a run that does not read the base (auto-style-gaps plan A): only
#: ``kb_read``, and only of ``style/<名字>`` -- turning on a style picker must
#: not hand out ordinary knowledge-base reads. ``kb_validate`` prechecks
#: proposals, so only ``propose`` (the knowledge update) gets it: a reading
#: task listing it is a tool it can never use.
KB_ENTITLEMENT_TOOLS: dict[str, tuple[str, ...]] = {
    "read": tuple(name for name in KB_TOOL_NAMES if name != "kb_validate"),
    "propose": KB_TOOL_NAMES,
    "style": ("kb_read",),
}


def kb_reply_chars(payload: Mapping[str, Any]) -> int:
    """What `KB_REPLY_MAX_CHARS` is measured on: the reply as canonical JSON."""

    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _frame_strings(value: Any) -> Iterator[str]:
    """Every string in one call's arguments, in no particular order."""

    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _frame_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _frame_strings(item)


def _decoded_frame(value: Any) -> Any:
    """`unescape_codepoints` over every string, once the frame is judged."""

    if isinstance(value, str):
        return unescape_codepoints(value)
    if isinstance(value, Mapping):
        return {key: _decoded_frame(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_decoded_frame(item) for item in value]
    return value


def _unescaped_arguments(value: Any) -> Any:
    r"""One call's arguments, with `\uXXXX`-escaped text decoded.

    antigravity-cli 1.1.24 cannot put a non-ASCII character into a tool
    argument: it writes every one of them as a literal escape *inside* the
    JSON string, so `json.loads` hands back the six characters. The repair
    sits here, where the frame enters -- before the request fingerprint and
    before any handler reads the arguments -- because a repaired argument is
    the argument the model wrote, and everything downstream should see that
    one.

    ⚠ **Judged over the whole frame, then applied to every string in it.** The
    fault is frame-shaped: the CLI escapes everything it sends. Density is not
    -- one mostly-English note carrying two CJK characters falls under
    `looks_escaped`'s floor by itself. Judging per field would decode some
    cells of a reply and leave others escaped, and a half-repaired row is
    worse than either whole answer.

    **Never silent, and once per frame rather than once per process**: the
    artifact records what the harness *received*, which after this runs is the
    repaired text, so nothing downstream keeps the original. The stderr line
    is therefore the only record that an edit happened at all -- it lands in
    the capsule's `events/stderr.log`, which is where
    `local_agent._empty_answer_error` already sends readers. It carries the
    escape count so a reader can tell a whole corrupted window (dozens) from a
    single ambiguous edit (one), which is the case worth a second look.

    This is the recovery half; the guarantee is `output_protocol._refused_reply`,
    which refuses escaped text outright. With the repair in place that refusal
    should never fire -- and if it does, it means the predicate missed a new
    shape, which is exactly when the loud failure is the one worth having
    (plan: decision one).
    """

    joined = "".join(_frame_strings(value))
    if not looks_escaped(joined):
        return value
    sys.stderr.write(
        "agent_mcp_server: this CLI escaped the non-ASCII characters of its "
        f"tool arguments; decoding {decodable_escape_count(joined)} "
        "of them\n"
    )
    return _decoded_frame(value)


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
        channel: ChannelLimits | None = None,
        accepted_reply_hold_seconds: float = 0.0,
        predelivered: Sequence[str] = (),
    ) -> None:
        # Exit-on-accept: the reply that completes the assignment is held
        # this long by `serve` (after it is logged), so the driver can end the
        # CLI before the CLI spends a request answering it. 0 sends it at once.
        self.accepted_reply_hold_seconds = max(0.0, float(accepted_reply_hold_seconds))
        self._reply_hold = 0.0
        # What this CLI carries to its model (docs/llm_agent_tool_protocol.md
        # §6): every reply is fitted to it on the way out, in `_call`.
        self.channel = channel or ChannelLimits()
        self.runtime = runtime
        self.assignment_id = assignment_id
        self.worker_id = worker_id
        self.session_id = session_id
        self.instance_id = instance_id
        self.next_task_wait_seconds = max(1.0, float(next_task_wait_seconds))
        self._task: dict[str, Any] | None = None
        self._manifest: dict[str, Any] | None = None
        # Block key -> the task whose reply first gave it to this process's
        # CLI (pushed in a reply, or read with `read_context`).
        # Kept in memory on purpose: the durable ledger outlives the CLI -- the
        # host can start a fresh CLI on the same conversation -- while this
        # process lives and dies with the one CLI that saw the replies.
        self._delivered: dict[str, str] = {key: _OPENING for key in predelivered}
        # Tasks whose payload a `next_task` reply of this process carried: a
        # later task with the same payload digest leaves `_delivered` naming
        # the first one.
        self._payload_sent_for: set[str] = set()
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
        # Built on the first web call and kept for the process: its search
        # client carries the pacing and the Gemma4 rate windows, which a
        # client per call restarted from zero.
        self._retrieval_access: Any = None
        # Knowledge tools: exposure follows the tools the spawner authorized
        # (or, absent an explicit list, the env root); per-call admission —
        # entitlement, identity, root — happens in `_kb_context` against the
        # current task's manifest. Handles are per-task, the index per-rev.
        self._kb_env_root = os.environ.get("FINESUB_MCP_KNOWLEDGE_ROOT", "")
        # Per tool, not all-or-nothing: a task spawned with the narrow
        # `style` grant is offered `kb_read` alone.
        self._kb_tool_names = (
            tuple(name for name in KB_TOOL_NAMES if name in tools)
            if tools is not None
            else (KB_TOOL_NAMES if self._kb_env_root else ())
        )
        self._kb_root = ""  # resolved per task in _kb_context
        self._kb_entitlement = ""  # "read" / "propose" / "style", set by _kb_context
        self._kb_handles: Any = None
        self._kb_task_id = ""
        self._kb_search_cache: tuple[int, Any] | None = None
        # Transport replays: the same JSON-RPC id asked again gets the same
        # answer, and nothing -- the repair count included -- moves twice.
        self._replies: dict[str, dict[str, Any]] = {}  # request_id -> {fingerprint, reply}

    @property
    def spill_root(self) -> Path:
        """Where replies too large for the channel are written as parts: under
        the assignment root, which an `overflow="files"` CLI may read."""

        return Path(self.runtime.root) / "control" / SPILL_DIRNAME

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
        for tool in KB_TOOL_DEFINITIONS:
            if tool["name"] not in self._kb_tool_names:
                continue
            tool = dict(tool)
            if tool["name"] == "kb_read" and "kb_validate" in self._kb_tool_names:
                # Only a propose grant writes proposals: told of the @a row and
                # kb_validate, a reading session would read of a tool and an
                # output block it does not have (cold read round 2, P178).
                tool["description"] += KB_READ_PROPOSE_NOTE
            tools.append(tool)
        if self.channel.reads_spills_by_tool:
            tools.append(dict(READ_SPILL_DEFINITION))
        if self.channel.batch_tool:
            tools.append(dict(BATCH_DEFINITION))
        if self.channel.result_size_meta:
            # Claude Code saves a text result past 50,000 characters to a file
            # the worker may not read, unless the tool declares a larger size
            # (docs/llm_local_agent.md, Claude Code section).
            for tool in tools:
                tool["_meta"] = {"anthropic/maxResultSizeChars": int(self.channel.result_size_meta)}
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
            # Before the request fingerprint and before any handler reads them.
            arguments = _unescaped_arguments(arguments)
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

    def _handlers(self) -> dict[str, Callable[..., dict[str, Any]]]:
        handlers = {
            "next_task": self._next_task,
            "read_context": self._read_context,
            "pull_status": self._pull_status,
            "submit": self._submit,
        }
        if self._web_tools:
            handlers["web_search"] = self._web_search
            handlers["web_fetch"] = self._web_fetch
        kb_handlers = {
            "kb_index": self._kb_index,
            "kb_search": self._kb_search,
            "kb_read": self._kb_read,
            "kb_read_node": self._kb_read_node,
            "kb_validate": self._kb_validate,
        }
        handlers.update(
            {name: kb_handlers[name] for name in self._kb_tool_names}
        )
        if self.channel.reads_spills_by_tool:
            handlers["read_spill"] = self._read_spill
        if self.channel.batch_tool:
            handlers["batch"] = self._batch
        return handlers

    def _call(self, name: str, arguments: Mapping[str, Any], *, rpc_id: Any) -> dict[str, Any]:
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
        payload, error = self._dispatch(name, arguments, request_id=request_id)
        # Every reply, success or error, leaves through here
        # (docs/llm_agent_tool_protocol.md §6): it carries an id the driver
        # can find again in the CLI's record, and when it is larger than the
        # CLI delivers whole it goes out as parts the model reads.
        task_id = str((self._task or {}).get("task_id") or "")
        reply = fit_reply(
            payload=payload,
            error=error,
            reply_id=reply_id_for(request_id),
            limits=self.channel,
            spill_root=lambda: self.spill_root,
            spill_id=request_id,
            task_id=task_id,
        )
        self._replies[request_id] = {"fingerprint": fingerprint, "reply": reply}
        return dict(reply)

    def _dispatch(
        self, name: str, arguments: Mapping[str, Any], *, request_id: str
    ) -> tuple[dict[str, Any] | None, str]:
        """(payload, "") for a reply, (None, message) for an error."""

        handler = self._handlers().get(name)
        if handler is None:
            return None, f"Unknown tool {name!r}"
        try:
            return dict(handler(arguments, request_id=request_id)), ""
        except _ToolError as exc:
            return None, str(exc)
        except AgentTaskRuntimeError as exc:
            return None, f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 -- the model's arguments must not kill the server
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    # -- tools ----------------------------------------------------------

    def _require_task(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._task is None or self._manifest is None:
            # No `next_task` answered in this process: the harness claimed
            # the task before starting the CLI (its opening carried the
            # blocks), or this is a restarted MCP process -- Claude Code does
            # that inside one CLI. The lease this worker holds says which task.
            status = self.runtime.rehydrate(
                assignment_id=self.assignment_id, worker_id=self.worker_id
            )
            if status["status"] == "ready" and _OPENING in self._delivered.values():
                # The harness claimed it for this session's opening, and the
                # lease lapsed before the CLI got going (a driver that queued
                # for a free slot): the task is back in the queue.
                raise _ToolError(
                    "this session holds no task -- its hold on the task lapsed; "
                    "call next_task to take it again"
                )
            if status["status"] != "task":
                raise _ToolError("call next_task first")
            task = dict(status["task"])
            self._task = task
            self._manifest = json.loads(self.runtime.read_artifact(str(task["manifest_ref"])))
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

    def _next_task(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
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
        task_id = str(task["task_id"])
        # A protocol this CLI already got for an earlier task -- same digest
        # -- is not sent again while the ledger still holds it: in a
        # pseudo-conversational session it is one document for the whole
        # run, and re-pushing it put a copy per window into a context that
        # keeps them all (rollouts 2026-09-30). The payload is the task
        # itself and always comes with it, even when an earlier task had the
        # same one (a re-asked preparation). Both conditions matter:
        # the ledger clears on a reset inside this CLI, and this process's
        # memory is gone with a fresh CLI. A repeated claim of the same task
        # is told it was sent (below) and gets everything again with `resend`.
        pulled = set(
            self.runtime.pull_status(
                assignment_id=self.assignment_id, task_id=task_id, worker_id=self.worker_id
            ).get("pulled")
            or []
        )
        seen = {
            key for key, owner in self._delivered.items() if owner != task_id and key in pulled
        }
        # This task's payload already reached this CLI -- in the opening, or
        # in an earlier `next_task` reply for the same task. A session that
        # calls again mostly still has it (CodeBuddy re-sent 37 KB that way,
        # acceptance 2026-10-24), so the reply only says so; `resend` gets
        # everything again, for a session that lost it (a compact, a reply
        # that never arrived).
        payload_key = next(
            (f"{block.get('kind')}@{block.get('digest')}" for block in required if block.get("kind") == "payload"),
            "",
        )
        held = (
            payload_key in pulled
            and (self._delivered.get(payload_key) == _OPENING or task_id in self._payload_sent_for)
            and arguments.get("resend") is not True
        )
        if held:
            seen = {key for key in self._delivered if key in pulled}
            # Nothing is booked, so nothing renews the lease on the way: a
            # `next_task` still does (it claims or resumes). Its own id, as
            # the push has: a re-claim past the lease already used this one.
            self.runtime.heartbeat(
                assignment_id=self.assignment_id,
                task_id=task_id,
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id + "-hold",
            )
        pushed: dict[str, str] = {}
        reply = {
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
        pushed_blocks = []
        for block in required:
            key = f"{block.get('kind')}@{block.get('digest')}"
            if block.get("kind") == "kb_index":
                # Fetched by its own tool, never by read_context: the gate is
                # "the kb_index reply's digest equals the manifest's" -- or
                # the opening carried it, and then it is already booked.
                block["read"] = "seen" if key in seen else "tool"
                block["tool"] = "kb_index"
                continue
            if block.get("kind") not in PUSHED_KINDS:
                continue
            if key in seen and (held or block.get("kind") == "protocol"):
                block["read"] = "seen"
                continue
            try:
                text = self.runtime.read_artifact(str(block["ref"]))
            except (OSError, ValueError):
                continue
            # Pushed whole: a reply larger than the CLI delivers goes out as
            # parts (`_call`), which hand the model the same text.
            block["chars"] = len(text)
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
        for block in pushed_blocks:
            self._delivered.setdefault(f"{block.get('kind')}@{block.get('digest')}", task_id)
            if block.get("kind") == "payload":
                self._payload_sent_for.add(task_id)
        if held:
            reply["note"] = fragment("fragment_next_task_already_sent_v1.md")
        return reply

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
        if ref == "kb_index":
            raise _ToolError("the kb_index block is fetched by the kb_index tool, not read_context")
        text = self.runtime.read_artifact(ref)
        matched = [
            block for block in manifest.get("required_blocks") or [] if block.get("ref") == ref
        ]
        payload: dict[str, Any] = {"ref": ref, "text": text}
        if matched:
            # Booked as seen: the reply carries all of it, inline or as parts
            # (docs/llm_agent_tool_protocol.md §3).
            booked = self.runtime.record_pull(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
                blocks=matched,
            )
            payload["owed_blocks"] = list(booked.get("owed_blocks") or [])
            for block in matched:
                self._delivered.setdefault(
                    f"{block.get('kind')}@{block.get('digest')}", str(task["task_id"])
                )
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

    def _read_spill(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        task, _manifest = self._require_task()
        # Reads book nothing, so the lease is checked here -- expired,
        # superseded or no longer active reads no part -- and, as work,
        # renews it.
        try:
            self.runtime.heartbeat(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
            )
        except StaleLeaseError:
            raise _ToolError(
                "this session's hold on the task lapsed; call next_task to take it again"
            ) from None
        try:
            return read_spill_part(
                self.spill_root,
                str(arguments.get("spill_id") or ""),
                arguments.get("part"),
                task_id=str(task["task_id"]),
            )
        except ValueError as exc:
            raise _ToolError(str(exc)) from None

    def _batch(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        """Independent lookups in one round, each dispatched as if called alone.

        Each call gets its own request id (`<id>-b<i>`): the replay cache,
        `record_pull` and the retrieval ledger key their idempotency on it,
        and one id shared by the calls would make the second one a replay of
        the first. A replayed `batch` is answered from `_call`'s cache whole.
        The reply is fitted once, as a whole, on its way out.
        """

        calls = arguments.get("calls")
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)) or not calls:
            raise _ToolError("batch needs a non-empty list of calls")
        results: list[dict[str, Any]] = []
        for index, call in enumerate(calls, start=1):
            call = call if isinstance(call, Mapping) else {}
            tool = str(call.get("tool") or "")
            if tool not in BATCH_MEMBERS:
                results.append({
                    "tool": tool,
                    "error": f"{tool!r} cannot run in a batch; it takes one of "
                    + ", ".join(BATCH_MEMBERS),
                })
                continue
            sub_arguments = call.get("arguments") if isinstance(call.get("arguments"), Mapping) else {}
            payload, error = self._dispatch(tool, sub_arguments, request_id=f"{request_id}-b{index}")
            results.append({"tool": tool, "result": payload} if payload is not None else {"tool": tool, "error": error})
        return {"results": results}

    def _retrieval(self):
        if self._retrieval_access is None:
            from .agent_retrieval import AgentRetrievalAccess

            self._retrieval_access = AgentRetrievalAccess(self.runtime)
        return self._retrieval_access

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

    # -- knowledge tools (read-only; docs/archive/knowledge-node-plan.md §4.3) ----

    def _kb_context(self, tool: str = ""):  # type: ignore[no-untyped-def]
        """(repo, rev) for the current task, or a tool error.

        Admission is the manifest's decision, never the server's: the task
        must carry a ``kb_tools`` entitlement (§4.3 matrix — a pin alone
        grants nothing) that admits ``tool`` (`KB_ENTITLEMENT_TOOLS`), a
        ``knowledge_identity`` ("rev:N") naming the revision every reply
        reads, and a root (``kb_root``, falling back to the spawn env for
        same-run tasks). A pseudo-conversational session exposes every tool
        at launch, so the per-tool check has to live here, per call.
        """

        _task, manifest = self._require_task()
        metadata = manifest.get("metadata") or {}
        entitlement = str(metadata.get("kb_tools") or "")
        if entitlement not in KB_ENTITLEMENT_TOOLS:
            raise _ToolError(
                "this task carries no kb entitlement; knowledge tools are not admitted for it"
            )
        if tool and tool not in KB_ENTITLEMENT_TOOLS[entitlement]:
            raise _ToolError(
                f"this task's kb entitlement ({entitlement}) does not admit {tool}; "
                + (
                    "it may only read style entries with kb_read (entry: style/<名字>)"
                    if entitlement == "style"
                    else "it admits " + ", ".join(KB_ENTITLEMENT_TOOLS[entitlement])
                )
            )
        self._kb_entitlement = entitlement
        root = str(metadata.get("kb_root") or "") or self._kb_env_root
        if not root:
            raise _ToolError("knowledge tools have no root for this task")
        self._kb_root = root
        identity = str(metadata.get("knowledge_identity") or "")
        if not identity.startswith("rev:"):
            raise _ToolError(
                "this task carries no knowledge identity; knowledge tools are not admitted for it"
            )
        rev = int(identity.split(":")[1])
        from ..knowledge.node.render import HandleMap
        from ..knowledge.node.repo import KnowledgeRepo

        task_id = str(_task.get("task_id") or "")
        if self._kb_handles is None or self._kb_task_id != task_id:
            # Handles are per-task identity (plan §2.3): a pseudo-conversational
            # session serves many tasks, and each task's prompt has its own
            # handle table riding its manifest — a stale seed would silently
            # resolve @k1 to the previous chunk's node.
            self._kb_handles = HandleMap()
            self._kb_handles.seed((manifest.get("metadata") or {}).get("kb_handle_bindings") or [])
            self._kb_task_id = task_id
        return KnowledgeRepo.open(self._kb_root), rev

    @staticmethod
    def _kb_finish(
        payload: dict[str, Any], *, capped: bool = True, narrow: Callable[[], str] | None = None
    ) -> dict[str, Any]:
        """Cap + audit digest: the digest rides the reply, so the frame log
        (FINESUB_MCP_LOG) carries (tool, args, rev, result digest) verbatim.

        The cap is how much knowledge one reply hands the model, not a
        transport limit (the channel carries any size). The index is exempt:
        it is a required block, and refusing it would shut the submit gate
        for good. ``narrow``, asked only past the cap, says how to ask for less.
        """

        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if capped and kb_reply_chars(payload) > KB_REPLY_MAX_CHARS:
            raise _ToolError(
                f"reply would be {len(text)} chars (limit {KB_REPLY_MAX_CHARS}); "
                + ((narrow() if narrow else "") or "narrow the request (kb_search: a longer query or entry=...)")
            )
        payload["result_digest"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return payload

    def _kb_index(self, _arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        repo, rev = self._kb_context("kb_index")
        from ..knowledge.base import kb_index_block_text

        text = kb_index_block_text(self._kb_root, rev)
        # Not capped (the index is a required block); the digest is taken
        # before the reply is fitted, so a spilled index books the same.
        payload = self._kb_finish({"knowledge_read_rev": rev, "text": text}, capped=False)
        # The required-block gate (plan §4.3): booked only when what the tool
        # returned is byte-for-byte what the manifest declared — "called it
        # once" does not pass.
        task, manifest = self._require_task()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        matched = [
            block
            for block in manifest.get("required_blocks") or []
            if block.get("kind") == "kb_index" and block.get("digest") == digest
        ]
        if matched:
            booked = self.runtime.record_pull(
                assignment_id=self.assignment_id,
                task_id=str(task["task_id"]),
                worker_id=self.worker_id,
                lease_generation=int(task["lease_generation"]),
                request_id=request_id,
                blocks=matched,
            )
            payload["owed_blocks"] = list(booked.get("owed_blocks") or [])
        return payload

    def _kb_search(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        repo, rev = self._kb_context("kb_search")
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise _ToolError("query is required")
        from ..knowledge.node.matching import ExactIndex

        if self._kb_search_cache is None or self._kb_search_cache[0] != rev:
            self._kb_search_cache = (rev, ExactIndex.build(repo.store, rev))
        keys = self._kb_search_cache[1].search(query)
        entry_filter = str(arguments.get("entry") or "").strip()
        if entry_filter:
            resolved = repo.resolve(entry_filter, rev)
            if resolved is None:
                raise _ToolError(f"unknown entry {entry_filter!r}")
            keys = [key for key in keys if key.subject_id == resolved.subject_id]
        # No silent truncation (plan §4.3 / §9): every match is returned or the
        # reply cap errors with a narrowing hint — hidden candidates would
        # defeat model-led retrieval.
        rows = []
        for key in keys:
            subject = repo.store.node(key.subject_id, rev)
            node = repo.store.node(key.node_id, rev)
            if subject is None or node is None:
                continue
            rows.append(
                {
                    "entry": subject.payload.get("surface", ""),
                    "category": subject.payload.get("category", ""),
                    "kind": key.kind,
                    "value": key.raw or key.text,
                    "handle": self._kb_handles.node_handle(node),
                }
            )
        return self._kb_finish({"knowledge_read_rev": rev, "matches": rows})

    def _kb_read(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        repo, rev = self._kb_context("kb_read")
        entry = str(arguments.get("entry") or "").strip()
        # `style/<名字>` and the like: a standalone category is only reachable
        # by its qualified name (`--style auto` lets the agent read a candidate
        # in full); a bare name keeps the matchable-categories lookup.
        from ..knowledge.node.repo import CATEGORY_SEPARATOR, AmbiguousName
        from ..knowledge.style import STYLE_CATEGORY

        style_only = self._kb_entitlement == "style"
        if style_only and not entry.startswith(f"{STYLE_CATEGORY}{CATEGORY_SEPARATOR}"):
            raise _ToolError(
                "this task may only read style entries; write the entry as style/<名字>"
            )
        from ..knowledge.node.presets import CATEGORIES

        # Qualified only when the prefix is a category: entry names may
        # contain `/` (`Fate/stay night`) and stay in the matchable lookup.
        prefix, separator, _rest = entry.partition(CATEGORY_SEPARATOR)
        try:
            resolved = (
                repo.resolve_qualified(entry, rev)
                if separator and prefix in CATEGORIES
                else repo.resolve(entry, rev)
            )
        except AmbiguousName as exc:
            raise _ToolError(str(exc)) from exc
        if resolved is None:
            raise _ToolError(f"unknown entry {entry!r}")
        if style_only and resolved.category != STYLE_CATEGORY:
            raise _ToolError("this task may only read style entries")
        raw_sections = arguments.get("sections")
        sections = (
            [str(name) for name in raw_sections]
            if isinstance(raw_sections, (list, tuple)) and raw_sections
            else None
        )
        # `tier` retired with section tiers (kb-line-grammar plan §3): the
        # caller names the sections it wants instead.
        #
        # The preview tier follows the task's ENTITLEMENT, same rule as every
        # other model-facing surface (review 2026-08-29 P2): the empty sections,
        # the collection discipline and the core empty slots exist to tell a
        # WRITER what belongs where, so a `propose` task gets the full preview
        # and a read-only task gets the partial one. Rendering scaffolding a
        # correction agent cannot act on is pure budget, and it invites the
        # model to report gaps nobody asked it about.
        from ..knowledge.node.render import render_subject

        text = render_subject(
            repo.store, resolved.subject_id, rev=rev, mode="prompt", handles=self._kb_handles,
            sections=sections,
            preview="full" if self._kb_entitlement == "propose" else "partial",
        )
        _task, manifest = self._require_task()
        asks_to_compress = str((manifest.get("metadata") or {}).get("kb_size_notice") or "") != "off"
        if self._kb_entitlement == "propose" and sections is None and asks_to_compress:
            # A writer reading the whole entry gets the same compression
            # notice the update prefetch hangs on it (plan kb-entry-size) --
            # unless the task may only re-decide what its input names
            # (`kb_size_notice: off`, the conflict repair; owner 2026-10-03).
            from ..knowledge.entry_size import size_notice

            notice = size_notice(repo.store, resolved.subject_id, rev, with_contract=True)
            if notice:
                text = f"{notice}\n{text}"
        payload = self._kb_finish(
            {"knowledge_read_rev": rev, "entry": resolved.key, "text": text},
            narrow=lambda: self._kb_sections_hint(repo, resolved.subject_id, rev),
        )
        # The context's read ledger (auto-style-gaps plan A1 step 3): booked only
        # after the size cap passed, so a reply the model never received never
        # counts as "read". Not fail-soft, unlike the telemetry below -- the
        # `--style auto` read-check stands on it, and an unbooked read would
        # fail a window whose model did read; erroring here lets it read again.
        task, _manifest = self._require_task()
        self.runtime.record_kb_read(
            assignment_id=self.assignment_id,
            task_id=str(task["task_id"]),
            worker_id=self.worker_id,
            lease_generation=int(task["lease_generation"]),
            entry=resolved.key,
            category=resolved.category,
            sections=sections is not None,
        )
        # Agent-side ``exposed`` (plan §4.2 item 5: what the tool actually
        # returned). After the cap check so an errored reply books nothing;
        # fail-soft because telemetry never outranks the reply.
        try:
            from ..knowledge.node.signals import log_exposed_nodes, subject_pack_node_ids

            if not self._kb_signals_booked():
                return payload
            signal_task, signal_window = self._kb_signal_identity()
            log_exposed_nodes(
                repo.store,
                (
                    (resolved.subject_id, node_id)
                    for node_id in subject_pack_node_ids(
                        repo.store, resolved.subject_id, rev, sections=sections
                    )
                ),
                task_id=signal_task,
                window_id=signal_window,
                rev=rev,
            )
        except Exception:
            pass
        return payload

    def _kb_sections_hint(self, repo: Any, subject_id: Any, rev: int) -> str:
        """How to read an oversized entry in parts: its sections and sizes.

        The model cannot see the entry it was refused, so the error names
        what `sections=[...]` takes. Kept short of Claude Code's cut on an
        `isError` text (about 11,000 characters).
        """

        from ..knowledge.node.render import load_subject_tree, render_subject

        rows = []
        for name in [view.name for view in load_subject_tree(repo.store, subject_id, rev).sections]:
            rendered = render_subject(
                repo.store, subject_id, rev=rev, mode="prompt", handles=self._kb_handles,
                sections=[name],
                preview="full" if self._kb_entitlement == "propose" else "partial",
            )
            rows.append(f"{name} ({len(rendered)} chars)")
        listing = "; ".join(rows)
        if len(listing) > 8_000:
            listing = listing[:8_000] + " ..."
        return f"read it in parts with sections=[...]; its sections: {listing}"

    def _kb_signals_booked(self) -> bool:
        """Whether this task's ``exposed`` events may be booked. The harness
        decides (``kb_signals: "off"`` when its knowledge-write gate is shut,
        e.g. a worktree reading the main checkout's base); this process does
        not re-derive it from its own cwd, which need not be the harness's.
        Absent means on, so a task built without the key never loses
        telemetry silently."""

        _task, manifest = self._require_task()
        return str((manifest.get("metadata") or {}).get("kb_signals") or "") != "off"

    def _kb_signal_identity(self) -> tuple[str, str]:
        """(task, window) the exposure ledger books under (plan §5.1).

        The runtime task_id is a constant ("call") on per-call assignments, so
        the caller's ``kb_signal_task``/``kb_signal_window`` — the run task
        and window this agent call serves — take precedence; without them the
        assignment id keeps distinct calls distinct."""

        task, manifest = self._require_task()
        metadata = manifest.get("metadata") or {}
        signal_task = str(metadata.get("kb_signal_task") or "")
        if not signal_task:
            signal_task = f"{self.assignment_id}:{task.get('task_id') or ''}"
        return signal_task, str(metadata.get("kb_signal_window") or "")

    def _kb_read_node(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        repo, rev = self._kb_context("kb_read_node")
        handle = str(arguments.get("handle") or "").strip()
        bound = self._kb_handles.nodes.get(handle) if self._kb_handles is not None else None
        if bound is None:
            raise _ToolError(
                f"unknown handle {handle!r}; handles come from this session's kb_read/kb_search replies"
            )
        from ..knowledge.node.render import format_line, node_aliases

        node = repo.store.node(bound[0], rev)
        if node is None:
            raise _ToolError(f"{handle} is not live at rev {rev}")
        payload = self._kb_finish(
            {
                "knowledge_read_rev": rev,
                "handle": handle,
                "kind": node.kind,
                "text": format_line(node, aliases=node_aliases(repo.store, node, rev)),
            }
        )
        try:
            from ..knowledge.node.signals import log_exposed_nodes

            if not self._kb_signals_booked():
                return payload
            # The event schema groups by owning subject: walk the membership
            # edges up to the top (bounded — the tree is a few levels deep).
            subject_id, seen = node.local_id, {node.local_id}
            while True:
                parents = repo.store.parents(subject_id, rev)
                if not parents or parents[0].parent_id in seen:
                    break
                subject_id = parents[0].parent_id
                seen.add(subject_id)
            signal_task, signal_window = self._kb_signal_identity()
            log_exposed_nodes(
                repo.store,
                [(subject_id, node.local_id)],
                task_id=signal_task,
                window_id=signal_window,
                rev=rev,
            )
        except Exception:
            pass
        return payload

    def _kb_validate(self, arguments: Mapping[str, Any], *, request_id: str) -> dict[str, Any]:
        repo, rev = self._kb_context("kb_validate")
        text = str(arguments.get("proposals") or "")
        if not text.strip():
            raise _ToolError("proposals is required (the <knowledge_proposals> block or raw JSONL)")
        from ..knowledge.node.apply import preview
        from ..knowledge.node.envelope import Binding, Envelope
        from ..knowledge.node.proposals import parse_model_proposals, translate_model_proposals

        ops, report, _drafts, bindings = translate_model_proposals(
            parse_model_proposals(text),
            repo=repo,
            knowledge_read_rev=rev,
            bindings=[Binding(**binding) for binding in self._kb_handles.bindings()],
        )
        engine_ops = [{key: value for key, value in op.items() if key != "_meta"} for op in ops]
        problems: list[tuple[str, str, str]] = []
        rejected: list[tuple[int, str]] = []
        if engine_ops:
            envelope = Envelope(
                task_id="kb-validate",
                assignment_id="",
                context_epoch=0,
                knowledge_read_rev=rev,
                ops=engine_ops,
                handle_bindings=bindings,
                draft_bindings=sorted(
                    {op["handle"] for op in engine_ops if op.get("op") == "create" and op.get("handle")}
                ),
            )
            overlay, problems = preview(repo.store, envelope)
            # fold rejections (duplicate items, unknown refs, bad payloads) are
            # per-op failures the apply would also skip: without them the
            # pre-check reads as a false all-clear (review 2026-08-26).
            rejected = list(overlay.rejected)
        return self._kb_finish(
            {
                "knowledge_read_rev": rev,
                "ops_translated": len(engine_ops),
                "ops_appliable": len(engine_ops) - len(rejected),
                "skipped": [record.to_dict() for record in report.skipped],
                "rejected": [{"op": index, "reason": reason} for index, reason in rejected],
                "problems": [f"{entity} {ident}: {problem}" for entity, ident, problem in problems],
            }
        )

    def take_reply_hold(self) -> float:
        """How long `serve` holds the reply it is about to send; asked once."""

        hold, self._reply_hold = self._reply_hold, 0.0
        return hold

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
                self._reply_hold = self.accepted_reply_hold_seconds
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
        if response.get("repair_instruction"):
            # The validator's own words for this repair (a patch, for a mostly right window answer).
            out["repair_instruction"] = response["repair_instruction"]
        for key in ("patch_merged", "rows_to_fix"):
            # A rejected patch: the errors are about the whole it made, and these are its rows still to fix.
            if response.get(key):
                out[key] = response[key]
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
                log.write(json.dumps(
                    {"dir": "in", "session": server.session_id, "frame": line.rstrip("\n")},
                    ensure_ascii=False,
                ) + "\n")
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
                log.write(json.dumps(
                    {"dir": "out", "session": server.session_id, "frame": encoded.rstrip("\n")},
                    ensure_ascii=False,
                ) + "\n")
                log.flush()
            # Logged first: the audit has the accept even when the driver
            # ends this process during the hold, which is the point of it.
            hold = server.take_reply_hold()
            if hold:
                time.sleep(hold)
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
        reply_hold = float(os.environ.get(ACCEPTED_REPLY_HOLD_ENV, "") or 0.0)
    except ValueError:
        reply_hold = 0.0
    server = HarnessToolServer(
        runtime,
        assignment_id=assignment_id,
        worker_id=worker_id,
        session_id=session_id,
        instance_id=uuid.uuid4().hex,
        tools=[name.strip() for name in tools_env.split(",") if name.strip()] if tools_env else None,
        next_task_wait_seconds=wait_seconds,
        channel=ChannelLimits.from_env(os.environ.get(CHANNEL_ENV, "")),
        accepted_reply_hold_seconds=reply_hold,
        predelivered=[key for key in os.environ.get(PREDELIVERED_ENV, "").split(",") if key],
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
