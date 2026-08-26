"""Canonical locations for one-shot local-agent evidence.

Keep this module import-light: the thin ``finesub agent-clean`` command runs
it on the launcher's Python 3.10 interpreter even when the managed runtime is
missing or broken.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

from finesub.paths import (
    checkout_data_enabled,
    resolve_checkout_root,
    resolve_managed_app_paths,
)
from finesub_bootstrap.locks import (
    AGENT_ACTIVITY_ROOT_VARIABLE,
    AGENT_CAPSULE_ROOT_VARIABLE,
    AGENT_IDENTITY_ANCHOR_VARIABLE,
    AGENT_LOCATOR_KIND_MANAGED,
    AGENT_LOCATOR_KIND_VARIABLE,
)


MACHINE_TEMP_ROOT_NAME = "finesub-agent-runtime"
MANAGED_DIRECTORY_NAME = "agent-capsules"
LOCATOR_MANAGED = AGENT_LOCATOR_KIND_MANAGED
LOCATOR_MACHINE_TEMP = "machine_temp"
# A parent someone named outright. It is not under the machine-temp root and
# its identity hash therefore cannot be used to find it again, so the locator
# has to carry the parent itself.
LOCATOR_EXPLICIT = "explicit_parent"


@dataclass(frozen=True)
class AgentEpisodeLocation:
    parent: Path
    domain_identity_anchor: Path
    activity_root: Path
    locator_kind: str
    location_identity: str


@dataclass(frozen=True)
class AgentEvidenceLocator:
    locator_kind: str
    location_identity: str
    episode_id: str
    absolute_at_write: str

    def as_dict(self) -> dict[str, str]:
        return {
            "locator_kind": self.locator_kind,
            "location_identity": self.location_identity,
            "episode_id": self.episode_id,
            "absolute_at_write": self.absolute_at_write,
        }


def _canonical_domain_path(path: Path) -> str:
    """Stable path spelling for cross-process domain identity."""

    value = os.path.normpath(os.path.realpath(str(path.expanduser().resolve())))
    if os.name == "nt":
        value = value.replace("/", "\\")
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        value = value.casefold()
    return value


def location_identity(locator_kind: str, anchor: Path) -> str:
    payload = locator_kind + "\0" + _canonical_domain_path(anchor)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def machine_temp_root() -> Path:
    return Path(tempfile.gettempdir()).resolve() / MACHINE_TEMP_ROOT_NAME


def managed_agent_capsule_parent() -> Path | None:
    paths = resolve_managed_app_paths()
    return None if paths is None else paths.agent_capsules


def resolve_agent_episode_location(
    explicit_parent: str | Path | None = None,
) -> AgentEpisodeLocation:
    """Resolve parent and coordination domain as one indivisible decision."""

    if explicit_parent is not None:
        parent = Path(explicit_parent).expanduser().resolve()
        anchor = parent
        kind = LOCATOR_EXPLICIT
        return AgentEpisodeLocation(
            parent=parent,
            domain_identity_anchor=anchor,
            activity_root=parent.parent,
            locator_kind=kind,
            location_identity=location_identity(kind, anchor),
        )

    configured = {
        "parent": os.environ.get(AGENT_CAPSULE_ROOT_VARIABLE, "").strip(),
        "activity": os.environ.get(AGENT_ACTIVITY_ROOT_VARIABLE, "").strip(),
        "anchor": os.environ.get(AGENT_IDENTITY_ANCHOR_VARIABLE, "").strip(),
        "kind": os.environ.get(AGENT_LOCATOR_KIND_VARIABLE, "").strip(),
    }
    if any(configured.values()):
        if not all(configured.values()):
            raise RuntimeError("Incomplete managed agent location environment")
        parent = Path(configured["parent"]).expanduser().resolve()
        activity = Path(configured["activity"]).expanduser().resolve()
        anchor = Path(configured["anchor"]).expanduser().resolve()
        kind = configured["kind"]
        if kind not in {LOCATOR_MANAGED, LOCATOR_MACHINE_TEMP, LOCATOR_EXPLICIT}:
            raise RuntimeError(f"Unknown agent locator kind: {kind}")
        return AgentEpisodeLocation(
            parent=parent,
            domain_identity_anchor=anchor,
            activity_root=activity,
            locator_kind=kind,
            location_identity=location_identity(kind, anchor),
        )

    checkout = resolve_checkout_root() if checkout_data_enabled() else None
    if checkout is not None:
        anchor = checkout / ".state"
        kind = LOCATOR_MACHINE_TEMP
        identity = location_identity(kind, anchor)
        return AgentEpisodeLocation(
            parent=machine_temp_root() / identity,
            domain_identity_anchor=anchor,
            activity_root=checkout,
            locator_kind=kind,
            location_identity=identity,
        )

    paths = resolve_managed_app_paths()
    if paths is None:  # pragma: no cover - bootstrap is a required dependency
        raise RuntimeError("FineSub managed paths are unavailable")
    anchor = paths.user_data
    kind = LOCATOR_MANAGED
    return AgentEpisodeLocation(
        parent=paths.agent_capsules,
        domain_identity_anchor=anchor,
        activity_root=paths.user_data,
        locator_kind=kind,
        location_identity=location_identity(kind, anchor),
    )


SESSION_LEDGER_NAME = "agent-sessions.jsonl"


def session_ledger_path(location: AgentEpisodeLocation) -> Path:
    """Where the list of sessions FineSub created is kept.

    Deliberately in the activity root -- beside ``config.toml``, in user data
    rather than under a task or a capsule. Every other record of a session dies
    with the thing that holds it: capsules are deleted the moment a call
    succeeds, and exchange files go when the run's artifacts do. The vendor's
    own transcripts outlive all of it, so without a durable list there is no
    way to look at ``~/.claude/projects/...`` and tell which sessions were ours.
    """

    return location.activity_root / SESSION_LEDGER_NAME


def evidence_locator(
    location: AgentEpisodeLocation, episode_id: str
) -> AgentEvidenceLocator:
    return AgentEvidenceLocator(
        locator_kind=location.locator_kind,
        location_identity=location.location_identity,
        episode_id=episode_id,
        absolute_at_write=str(location.parent / episode_id),
    )


def resolve_evidence_locator(locator: Mapping[str, object]) -> Path:
    """Resolve retained evidence from stable locator fields after relocation."""

    kind = str(locator.get("locator_kind") or "")
    identity = str(locator.get("location_identity") or "")
    episode_id = str(locator.get("episode_id") or "")
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise ValueError("Invalid agent evidence location identity")
    if kind == LOCATOR_EXPLICIT:
        # Nothing derives this parent, so the recorded path is the only way
        # back to it. Verify it still hashes to the identity that was written
        # rather than trusting the string.
        recorded = Path(str(locator.get("absolute_at_write") or "")).expanduser()
        parent = recorded.parent
        if not str(recorded) or identity != location_identity(kind, parent):
            raise ValueError("Agent evidence locator does not match its recorded parent")
        location = AgentEpisodeLocation(
            parent=parent,
            domain_identity_anchor=parent,
            activity_root=parent.parent,
            locator_kind=kind,
            location_identity=identity,
        )
    elif kind == LOCATOR_MACHINE_TEMP:
        location = AgentEpisodeLocation(
            parent=machine_temp_root() / identity,
            domain_identity_anchor=Path(),
            activity_root=Path(),
            locator_kind=kind,
            location_identity=identity,
        )
    elif kind == LOCATOR_MANAGED:
        current = resolve_agent_episode_location()
        if current.locator_kind == kind and current.location_identity == identity:
            location = current
        else:
            paths = resolve_managed_app_paths()
            if paths is None:
                raise RuntimeError("FineSub managed paths are unavailable")
            current_identity = location_identity(kind, paths.user_data)
            if identity != current_identity:
                raise ValueError("Agent evidence belongs to a different managed domain")
            location = AgentEpisodeLocation(
                parent=paths.agent_capsules,
                domain_identity_anchor=paths.user_data,
                activity_root=paths.user_data,
                locator_kind=kind,
                location_identity=identity,
            )
    else:
        raise ValueError(f"Unknown agent locator kind: {kind!r}")
    return episode_path(location, episode_id)


def episode_path(location: AgentEpisodeLocation, episode_id: str) -> Path:
    """Resolve one direct child without trusting an external episode id."""

    if not episode_id or Path(episode_id).name != episode_id:
        raise ValueError(f"Invalid agent episode id: {episode_id!r}")
    parent = location.parent.resolve()
    result = (parent / episode_id).resolve()
    if result.parent != parent:
        raise ValueError(f"Agent episode escapes its parent: {episode_id!r}")
    return result


CONVERSATIONAL_DIRECTORY_NAME = "conversational"


def conversational_assignment_parent(
    location: AgentEpisodeLocation | None = None,
) -> Path:
    """Where a run's conversational assignment trees live.

    Inside the episode parent, so it is partitioned by coordination domain
    like everything else here and one plain `finesub agent-clean` reaches it.
    A tree left behind by a failed run holds that run's whole subtitle text
    and every control frame -- the same evidence class as a capsule, and it
    must not outlive one by sitting somewhere the cleanup never looks.
    """

    resolved = location if location is not None else resolve_agent_episode_location()
    return resolved.parent / CONVERSATIONAL_DIRECTORY_NAME


def vendor_error_text(exc: BaseException) -> str:
    """Whatever the CLI put in its own error events, from the kept capsule.

    The transports deliberately keep vendor prose out of their exception
    messages and point at stderr instead -- but a spent Codex plan says so in
    an `error` *event* and leaves stderr empty, so the pointer led nowhere.
    Nothing here interprets the text: it is quoted so a person can read it.
    """

    return vendor_error_from_attempts(
        getattr(exc, "_harness_execution_attempts", None) or []
    )


def vendor_error_from_attempts(attempts: Iterable[Any]) -> str:
    """``vendor_error_text`` for callers that hold the attempt rows directly."""

    import json

    for attempt in attempts:
        locator = attempt.get("evidence_locator") if hasattr(attempt, "get") else None
        if not isinstance(locator, Mapping):
            continue
        try:
            events = resolve_evidence_locator(locator) / "events" / "agent-events.jsonl"
            lines = events.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError, RuntimeError):
            continue
        said: list[str] = []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(row.get("error") or "").strip() if isinstance(row, dict) else ""
            # Vendors repeat themselves: Codex reports the same sentence as an
            # `error` item and again inside `turn.failed`, wrapped in a dict
            # repr, so exact-match dedup leaves it printed twice.
            if not text or any(text in seen or seen in text for seen in said):
                continue
            said.append(text)
        if said:
            return " / ".join(said)
    return ""
