"""Accepted proposal envelope (plan §6.5) and the proposal op vocabulary.

The envelope is what a knowledge-update task hands back to the parent
harness: self-contained (handle bindings included), hashed as a whole, and
re-validated against the task manifest before apply. Both front ends
(agent ``kb_propose``/``submit`` and the REST ``<knowledge_proposals>``
block) produce this same structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .model import digest

SCHEMA_VERSION = 1

OPS: tuple[str, ...] = (
    "create",
    "update",
    "add_item",
    "remove_item",
    "add_membership",
    "move_membership",
    "remove_membership",
    "link",
    "retire",
)

_REQUIRED: dict[str, tuple[str, ...]] = {
    "create": ("kind", "payload"),  # parent/section required unless kind == subject (checked at fold)
    "update": ("id", "set"),
    "add_item": ("id", "field", "value"),
    "remove_item": ("item",),
    "add_membership": ("id", "parent", "section"),
    "move_membership": ("membership",),
    "remove_membership": ("membership",),
    "link": ("id", "rel", "target"),
    "retire": ("id",),
}


class EnvelopeError(ValueError):
    pass


@dataclass(frozen=True)
class Binding:
    handle: str
    kind: str  # node | item | membership
    id: str
    expected_valid_from_rev: int


@dataclass
class Envelope:
    task_id: str
    assignment_id: str
    context_epoch: int
    knowledge_read_rev: int
    ops: list[dict[str, Any]]
    handle_bindings: list[Binding] = field(default_factory=list)
    draft_bindings: list[str] = field(default_factory=list)  # draft handles (@new1 …) in creation order
    required_block_digests: list[str] = field(default_factory=list)
    preset_version: int = 1
    validator_version: int = 1
    input_hash: str = ""
    schema_version: int = SCHEMA_VERSION

    # ---- canonical form -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "assignment_id": self.assignment_id,
            "context_epoch": self.context_epoch,
            "knowledge_read_rev": self.knowledge_read_rev,
            "ops": self.ops,
            "handle_bindings": [b.__dict__ for b in self.handle_bindings],
            "draft_bindings": list(self.draft_bindings),
            "required_block_digests": list(self.required_block_digests),
            "preset_version": self.preset_version,
            "validator_version": self.validator_version,
            "input_hash": self.input_hash,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Envelope":
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise EnvelopeError(f"unsupported envelope schema {data.get('schema_version')!r}")
        return cls(
            task_id=str(data["task_id"]),
            assignment_id=str(data.get("assignment_id", "")),
            context_epoch=int(data.get("context_epoch", 0)),
            knowledge_read_rev=int(data["knowledge_read_rev"]),
            ops=[dict(op) for op in data.get("ops", [])],
            handle_bindings=[Binding(**b) for b in data.get("handle_bindings", [])],
            draft_bindings=[str(h) for h in data.get("draft_bindings", [])],
            required_block_digests=[str(d) for d in data.get("required_block_digests", [])],
            preset_version=int(data.get("preset_version", 1)),
            validator_version=int(data.get("validator_version", 1)),
            input_hash=str(data.get("input_hash", "")),
        )

    def proposal_hash(self) -> str:
        """Covers the whole canonical envelope, not just ``ops`` (plan §6.5)."""

        return digest(self.to_dict())

    # ---- validation ------------------------------------------------------------------

    def validate_shape(self) -> list[str]:
        """Structural problems; empty list = well-formed. Does not touch the store."""

        problems: list[str] = []
        handles = {b.handle for b in self.handle_bindings} | set(self.draft_bindings)
        for index, op in enumerate(self.ops):
            name = op.get("op")
            if name not in OPS:
                problems.append(f"op[{index}]: unknown op {name!r}")
                continue
            missing = [key for key in _REQUIRED[name] if key not in op]
            if missing:
                problems.append(f"op[{index}] {name}: missing {missing}")
            for key in ("id", "parent", "target", "item", "membership"):
                value = op.get(key)
                if isinstance(value, str) and value.startswith("@") and value not in handles:
                    problems.append(f"op[{index}] {name}: unbound handle {value}")
        return problems

    def check_manifest(self, *, task_id: str, knowledge_read_rev: int, context_epoch: int | None = None) -> None:
        """Parent-side guard against cross-task replay (plan §6.5)."""

        if self.task_id != task_id:
            raise EnvelopeError(f"envelope task {self.task_id!r} != manifest task {task_id!r}")
        if self.knowledge_read_rev != knowledge_read_rev:
            raise EnvelopeError(
                f"envelope knowledge_read_rev {self.knowledge_read_rev} != manifest {knowledge_read_rev}"
            )
        if context_epoch is not None and self.context_epoch != context_epoch:
            raise EnvelopeError(f"envelope epoch {self.context_epoch} != current epoch {context_epoch}")
        for binding in self.handle_bindings:
            if binding.expected_valid_from_rev > self.knowledge_read_rev:
                raise EnvelopeError(
                    f"binding {binding.handle} references rev {binding.expected_valid_from_rev}"
                    f" beyond knowledge_read_rev {self.knowledge_read_rev}"
                )
