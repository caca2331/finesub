"""Durable draft contract for the agent front end (plan §6.5).

``kb_propose`` never touches the knowledge store; it appends to a draft that
the AgentTaskRuntime persists alongside its WAL, scoped to
``(task_id, context_epoch)``. Ops are deduplicated by canonical fingerprint
so a reconnect that replays a ``create`` cannot produce two nodes; a
conversation reset clears the whole draft together with its handle map.
``submit`` turns the current-epoch draft into an ``Envelope``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .envelope import OPS, Binding, Envelope, EnvelopeError
from .model import digest


@dataclass
class DraftOp:
    draft_op_id: str
    fingerprint: str
    op: dict[str, Any]


@dataclass
class Draft:
    task_id: str
    context_epoch: int
    knowledge_read_rev: int
    assignment_id: str = ""
    ops: list[DraftOp] = field(default_factory=list)
    handle_bindings: list[Binding] = field(default_factory=list)
    draft_handles: list[str] = field(default_factory=list)
    required_block_digests: list[str] = field(default_factory=list)
    _next: int = 1

    # ---- mutation ---------------------------------------------------------------------

    def propose(self, op: Mapping[str, Any]) -> tuple[str, bool]:
        """Append one op; returns ``(draft_op_id, duplicate)``. A duplicate
        (same canonical fingerprint) returns the existing id without appending."""

        name = op.get("op")
        if name not in OPS:
            raise EnvelopeError(f"unknown op {name!r}")
        fingerprint = digest(dict(op))
        for existing in self.ops:
            if existing.fingerprint == fingerprint:
                return existing.draft_op_id, True
        handle = op.get("handle")
        if name == "create" and handle:
            if handle in self.draft_handles or any(b.handle == handle for b in self.handle_bindings):
                raise EnvelopeError(f"draft handle {handle} already bound")
            self.draft_handles.append(str(handle))
        draft_op_id = f"d{self._next}"
        self._next += 1
        self.ops.append(DraftOp(draft_op_id, fingerprint, dict(op)))
        return draft_op_id, False

    def drop(self, draft_op_id: str) -> bool:
        for index, existing in enumerate(self.ops):
            if existing.draft_op_id == draft_op_id:
                handle = existing.op.get("handle")
                if existing.op.get("op") == "create" and handle in self.draft_handles:
                    self.draft_handles.remove(handle)
                del self.ops[index]
                return True
        return False

    def reset(self) -> None:
        """Conversation reset / requeue: the whole draft goes, handle map included."""

        self.ops.clear()
        self.draft_handles.clear()
        self.handle_bindings.clear()
        self.required_block_digests.clear()

    def bind(self, bindings: list[Binding]) -> None:
        known = {b.handle for b in self.handle_bindings}
        for binding in bindings:
            if binding.handle not in known:
                self.handle_bindings.append(binding)
                known.add(binding.handle)

    # ---- views --------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "context_epoch": self.context_epoch,
            "knowledge_read_rev": self.knowledge_read_rev,
            "ops": [{"draft_op_id": d.draft_op_id, "op": d.op.get("op"), "summary": _summary(d.op)} for d in self.ops],
            "draft_handles": list(self.draft_handles),
            "bound_handles": len(self.handle_bindings),
        }

    def to_envelope(self, *, preset_version: int = 1, validator_version: int = 1, input_hash: str = "") -> Envelope:
        envelope = Envelope(
            task_id=self.task_id,
            assignment_id=self.assignment_id,
            context_epoch=self.context_epoch,
            knowledge_read_rev=self.knowledge_read_rev,
            ops=[dict(d.op) for d in self.ops],
            handle_bindings=list(self.handle_bindings),
            draft_bindings=list(self.draft_handles),
            required_block_digests=list(self.required_block_digests),
            preset_version=preset_version,
            validator_version=validator_version,
            input_hash=input_hash,
        )
        problems = envelope.validate_shape()
        if problems:
            raise EnvelopeError("; ".join(problems))
        return envelope

    # ---- persistence (runtime durable state) -------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "context_epoch": self.context_epoch,
            "knowledge_read_rev": self.knowledge_read_rev,
            "assignment_id": self.assignment_id,
            "ops": [d.__dict__ for d in self.ops],
            "handle_bindings": [b.__dict__ for b in self.handle_bindings],
            "draft_handles": list(self.draft_handles),
            "required_block_digests": list(self.required_block_digests),
            "next": self._next,
        }

    @classmethod
    def load(cls, data: Mapping[str, Any] | None, *, task_id: str, context_epoch: int, knowledge_read_rev: int) -> "Draft":
        """Rehydrate; a draft from another task or epoch is discarded whole."""

        if not data or data.get("task_id") != task_id or int(data.get("context_epoch", -1)) != context_epoch:
            return cls(task_id=task_id, context_epoch=context_epoch, knowledge_read_rev=knowledge_read_rev)
        draft = cls(
            task_id=task_id,
            context_epoch=context_epoch,
            knowledge_read_rev=int(data.get("knowledge_read_rev", knowledge_read_rev)),
            assignment_id=str(data.get("assignment_id", "")),
            ops=[DraftOp(d["draft_op_id"], d["fingerprint"], dict(d["op"])) for d in data.get("ops", [])],
            handle_bindings=[Binding(**b) for b in data.get("handle_bindings", [])],
            draft_handles=[str(h) for h in data.get("draft_handles", [])],
            required_block_digests=[str(d) for d in data.get("required_block_digests", [])],
        )
        draft._next = int(data.get("next", len(draft.ops) + 1))
        return draft


def _summary(op: Mapping[str, Any]) -> str:
    name = op.get("op")
    if name == "create":
        return f"{op.get('kind')} in {op.get('section')} of {op.get('parent')}"
    if name == "update":
        return ", ".join(sorted(dict(op.get("set", {})).keys()))
    if name in ("add_item", "remove_item"):
        return f"{op.get('field', '')} {op.get('value', op.get('item', ''))}".strip()
    if name == "retire":
        return f"{op.get('id')} -> {op.get('merged_into', '')}".rstrip(" ->")
    return str(op.get("id") or op.get("membership") or "")
