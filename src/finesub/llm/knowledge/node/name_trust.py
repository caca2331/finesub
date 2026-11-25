"""How far a term line's Chinese name can be trusted (docs/knowledge.md「中文定名的核查」).

Every term line with a 中文定名 (``payload.zh``) is in one of three states,
worked out when the line is rendered and never stored:

* 人定 -- a person wrote or endorsed this very name: ``user`` evidence on the
  current value of ``payload.zh``, or an ack taken while the line already
  carried this name (`human_marks.ack_lines`; the ack endorses the whole line
  as it read then, so its name is compared with today's);
* 已核 -- the name itself was checked twice: ``external`` or ``refined_srt``
  evidence confirming the current value of ``payload.zh``, the earliest and
  the latest at least `RECHECK_GAP` apart (one check rests on pages a model
  picked, and a placeholder name may since have been replaced by an official
  one), and nothing refuting it (a refutation outweighs a confirmation; 人定
  outweighs both);
* 未核 -- anything else, including a name checked only once (or twice within
  the gap) and a line only corroborated as a whole (``payload:core``,
  `knowledge verify`'s claim, which in practice is often backed by a
  foreign-language page).

Only evidence on ``payload.zh`` counts, and only on its current value:
evidence is keyed by the value's hash, so renaming a line voids what was
said about the old name by itself, and editing its description does not
touch the state. A line without a 中文定名 has no state.

A refutation counts only against the version it was booked on: once the
line is written again after it (a knowledge update writing back the name a
check corrected away, or any later edit), it no longer counts, and the line
is an ordinary 未核 one the next check may take. A confirmation booked
before the name's latest refutation never counts, whether that refutation
still does or not: a name found wrong and written back starts from zero,
like the new value a correction writes.

The same evidence says when the name check (`knowledge/name_check.py`) is
owed a line again (`check_due`): a refutation and a 根本 not-found are
terminal, a 暂时 not-found backs off, a single confirmation waits out the gap.

Read by two projections: a correction task's (`render.model_line`, translating
or correct-only), which marks the 未核 lines (`trust_marks`) -- the second place evidence reaches a
model prompt; the criteria that let it are in docs/knowledge.md「信号与证据」 --
and the human one (``rendered/``, ``show``), whose term lines end in a short
check note (`check_note`) that the harvest strips (`CHECK_NOTE_RE`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .model import NodeVersion, digest
from .signals import EVIDENCE_KIND_REFINED
from .store import KnowledgeStore

HUMAN = "人定"
CHECKED = "已核"
UNCHECKED = "未核"
#: What a correct-only task sees at the start of a 未核 line's description column. The prompt
#: rules quote it (`mode_texts_v1.toml`, `kb_term_lines_note` and `kb_conflict_spoken`).
UNCHECKED_MARK = f"（{UNCHECKED}）"
#: The same for a 未核 line whose current name a check searched for and found nothing on: it tells the
#: preparation task not to search for it again (暂时 and 根本 read the same; they only set the next check).
UNCHECKED_NOT_FOUND_MARK = f"（{UNCHECKED}，查不到）"

ZH_FIELD = "payload.zh"
_HUMAN_KINDS = frozenset({"user"})
_CHECKED_KINDS = frozenset({"external", EVIDENCE_KIND_REFINED})
#: How far apart a name's first and last confirmations must be for it to count as 已核.
RECHECK_GAP = timedelta(days=7)
#: A not-found's kind, kept in the evidence row's ``span``: 暂时 (a published work that may name it later)
#: is checked again after a backoff, 根本 (only the video itself says it) never.
NOT_FOUND_TEMPORARY = "temporary"
NOT_FOUND_PERMANENT = "permanent"
NOT_FOUND_KINDS = (NOT_FOUND_TEMPORARY, NOT_FOUND_PERMANENT)
#: The wait after the n-th 暂时 not-found: 30 × 2^(n−1) days, at most 180.
NOT_FOUND_FIRST_WAIT = timedelta(days=30)
NOT_FOUND_MAX_WAIT = timedelta(days=180)
_CHUNK = 500


@dataclass
class ZhStatus:
    """What the evidence on one line's current name says."""

    name: str
    state: str = UNCHECKED
    human: bool = False
    #: a refutation of the current name booked no earlier than the current version (`zh_statuses`)
    refuted: bool = False
    #: timestamps of the confirmations that count toward 已核, one per task (the earliest of its rows: a refined
    #: subtitle books per window, and a resumed task must not space its own windows into two confirmations);
    #: undated rows cannot space a recheck
    confirmed: list[datetime] = field(default_factory=list)
    #: ``(when, kind)`` of each not-found on the current name; ``when`` None for an undated row
    not_found: list[tuple[datetime | None, str]] = field(default_factory=list)
    #: the last time a name check booked anything on this line, whatever name it was about
    last_checked: datetime | None = None


def zh_name(node: NodeVersion) -> str | None:
    """The line's 中文定名 when it has one: a term with a non-blank ``zh``."""

    if node.kind != "term":
        return None
    value = node.payload.get("zh")
    return value if isinstance(value, str) and value.strip() else None


def zh_statuses(store: KnowledgeStore, nodes: Iterable[NodeVersion], rev: int | None = None) -> dict[str, ZhStatus]:
    """``{node id: status}`` for the given lines at ``rev``; lines without a 中文定名 are left out.

    One evidence query and one ack query per chunk of lines, so an entry costs two queries, not two per line.
    """

    at = store.current_rev() if rev is None else int(rev)
    nodes = list(nodes)
    names = {node.local_id: name for node in nodes if (name := zh_name(node)) is not None}
    written = _revision_times(store, {node.valid_from_rev for node in nodes if node.local_id in names})
    version_time = {node.local_id: written.get(node.valid_from_rev) for node in nodes if node.local_id in names}
    statuses = {node_id: ZhStatus(name) for node_id, name in names.items()}
    ids = list(names)
    for start in range(0, len(ids), _CHUNK):
        chunk = ids[start:start + _CHUNK]
        marks = ",".join("?" * len(chunk))
        rows = []
        for row in store.conn.execute(
            "SELECT node_id, value_hash, evidence_kind, verdict, span, created_at, task_id FROM evidence"
            f" WHERE field_path=? AND node_id IN ({marks})",
            (ZH_FIELD, *chunk),
        ):
            status = statuses[row["node_id"]]
            when = _timestamp(row["created_at"])
            if row["evidence_kind"] == "external" and when is not None:
                status.last_checked = max(filter(None, (status.last_checked, when)))
            if row["value_hash"] == digest(status.name):
                rows.append((row, when))
        # The latest refutation of each current name: confirmations before it do not count.
        refuted_at: dict[str, datetime] = {}
        for row, when in rows:
            if row["verdict"] == "refuted" and when is not None:
                node_id = row["node_id"]
                refuted_at[node_id] = max(when, refuted_at.get(node_id, when))
        first_seen: dict[tuple[str, str], datetime] = {}
        for row, when in rows:
            node_id = row["node_id"]
            status = statuses[node_id]
            if row["verdict"] == "refuted":
                # an undated refutation predates every dated version; a dated one counts unless the line was
                # written again after it (a tie counts: nothing was written after it)
                written_at = version_time[node_id]
                if when is not None and (written_at is None or when >= written_at):
                    status.refuted = True
            elif row["verdict"] == "unverifiable":
                kind = row["span"] if row["span"] in NOT_FOUND_KINDS else NOT_FOUND_TEMPORARY
                status.not_found.append((when, kind))
            elif row["evidence_kind"] in _HUMAN_KINDS:
                status.human = True
            elif row["evidence_kind"] in _CHECKED_KINDS and when is not None:
                if node_id in refuted_at and when <= refuted_at[node_id]:
                    continue
                key = (node_id, row["task_id"])
                first_seen[key] = min(when, first_seen.get(key, when))
        for (node_id, _task), when in first_seen.items():
            statuses[node_id].confirmed.append(when)
        # A person's name stands whatever a check said of it; a refutation of the current name outweighs any
        # confirmation of it (the two disagree, so the model is told to look again).
        for status in (statuses[node_id] for node_id in chunk):
            if status.human:
                status.state = HUMAN
            elif not status.refuted and status.confirmed and max(status.confirmed) - min(status.confirmed) >= RECHECK_GAP:
                status.state = CHECKED
        # The line an ack endorsed is the version live at the store rev it was taken at.
        for row in store.conn.execute(
            "SELECT a.node_id, v.payload FROM line_acks a JOIN node_versions v ON v.local_id = a.node_id"
            " AND v.valid_from_rev <= a.rev AND (v.valid_to_rev IS NULL OR v.valid_to_rev > a.rev)"
            f" WHERE a.rev <= ? AND a.node_id IN ({marks})",
            (at, *chunk),
        ):
            if json.loads(row["payload"]).get("zh") == names[row["node_id"]]:
                statuses[row["node_id"]].state = HUMAN
    return statuses


def zh_states(store: KnowledgeStore, nodes: Iterable[NodeVersion], rev: int | None = None) -> dict[str, str]:
    """``{node id: state}`` for the given lines at ``rev``; lines without a 中文定名 are left out."""

    return {node_id: status.state for node_id, status in zh_statuses(store, nodes, rev).items()}


def _revision_times(store: KnowledgeStore, revs: set[int]) -> dict[int, datetime | None]:
    """``{rev: created_at}`` for the given revisions."""

    times: dict[int, datetime | None] = {}
    ordered = sorted(revs)
    for start in range(0, len(ordered), _CHUNK):
        chunk = ordered[start:start + _CHUNK]
        marks = ",".join("?" * len(chunk))
        for row in store.conn.execute(f"SELECT rev, created_at FROM revisions WHERE rev IN ({marks})", chunk):
            times[int(row["rev"])] = _timestamp(row["created_at"])
    return times


def _timestamp(value: str) -> datetime | None:
    """An evidence row's ``created_at``; None for rows booked before it was recorded (they cannot space a recheck)."""

    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def zh_state(store: KnowledgeStore, node: NodeVersion, rev: int | None = None) -> str | None:
    """One line's state; None when it has no 中文定名."""

    return zh_states(store, [node], rev).get(node.local_id)


def trust_marks(store: KnowledgeStore, nodes: Iterable[NodeVersion], rev: int | None = None,
                now: datetime | None = None) -> dict[str, str]:
    """``{node id: mark}`` for the lines a marked projection marks (state 未核): `UNCHECKED_NOT_FOUND_MARK` while a
    check that found nothing on the current name still holds it back (`not_found_holds`), `UNCHECKED_MARK` otherwise.
    Once a 暂时 not-found's wait is over the line reads plain 未核 again, so the next task searches it."""

    now = now or datetime.now(timezone.utc)
    return {
        node_id: UNCHECKED_NOT_FOUND_MARK if not_found_holds(status, now) else UNCHECKED_MARK
        for node_id, status in zh_statuses(store, nodes, rev).items()
        if status.state == UNCHECKED
    }


#: The end of a human-projection term line: an HTML comment holding exactly one of `check_note`'s notes. The harvest
#: strips it (`edit._bare_line`), wherever a person left it standing; a comment of a person's own, or a note they
#: edited, is not one of these and stays text.
CHECK_NOTE_RE = re.compile(
    r"\s*<!--\s*(?:已核|未核(?:，印证 \d+ 次)?|查不到(?:，不再核|，\d{4}-\d{2}-\d{2} 重核)?)\s*-->\s*$")


def strip_check_note(line: str) -> str:
    """``line`` (no bullet) without its check note, when it is shaped as a term line by the grammar (a body of three or
    more ``|`` columns, `importer.classify_body`: four is a term, three a term missing its alias column -- an edit that
    split a line can leave the note on either); any other line keeps a trailing comment as its own text."""

    from .importer import split_label

    return CHECK_NOTE_RE.sub("", line) if split_label(line)[1].count("|") >= 2 else line


def check_note(status: ZhStatus, now: datetime | None = None) -> str:
    """The note a person reads at the end of a term line: ``已核``; ``查不到，不再核`` (根本) or ``查不到，<date> 重核``
    (暂时); ``未核，印证 1 次``; ``未核``. Empty for a 人定 line: the person wrote or endorsed it."""

    if status.state == HUMAN:
        return ""
    if status.state == CHECKED:
        return CHECKED
    if any(kind == NOT_FOUND_PERMANENT for _, kind in status.not_found):
        return "查不到，不再核"
    temporary = [when for when, kind in status.not_found if kind == NOT_FOUND_TEMPORARY and when is not None]
    if temporary:
        return f"查不到，{(max(temporary) + not_found_wait(len(temporary))).date().isoformat()} 重核"
    if status.not_found:
        return "查不到"
    if status.confirmed:
        return f"{UNCHECKED}，印证 {len(status.confirmed)} 次"
    return UNCHECKED


def check_notes(store: KnowledgeStore, nodes: Iterable[NodeVersion], rev: int | None = None) -> dict[str, str]:
    """``{node id: check_note}`` for the given lines (lines without a 中文定名, and 人定 ones, are left out)."""

    return {node_id: note for node_id, status in zh_statuses(store, nodes, rev).items() if (note := check_note(status))}


def not_found_wait(times_not_found: int) -> timedelta:
    """How long after its n-th 暂时 not-found a name waits for the next check: 30, 60, 120, 180, 180… days."""

    if times_not_found < 1:
        return timedelta(0)
    return min(NOT_FOUND_FIRST_WAIT * 2 ** (times_not_found - 1), NOT_FOUND_MAX_WAIT)


def not_found_holds(status: ZhStatus, now: datetime | None = None) -> bool:
    """Whether a not-found on the current name still holds the line back: a 根本 one always, a 暂时 one until
    `not_found_wait` has passed since the latest (an undated one cannot hold it)."""

    now = now or datetime.now(timezone.utc)
    if any(kind == NOT_FOUND_PERMANENT for _, kind in status.not_found):
        return True
    temporary = [when for when, kind in status.not_found if kind == NOT_FOUND_TEMPORARY and when is not None]
    return bool(temporary) and now - max(temporary) < not_found_wait(len(temporary))


def check_due(status: ZhStatus, now: datetime | None = None) -> bool:
    """Whether the name check is owed this line now (docs/knowledge.md「谁被送去核」).

    Only a 未核 name is; a refutation of the current name (the check found another official name and the correction
    did not land: a person's line, or a correction the validator refused) and a 根本 not-found are terminal; a 暂时 not-found
    waits `not_found_wait` from the latest one; a name confirmed once (or twice within the gap) waits until the gap
    has passed since its first confirmation, so the next check can make it 已核."""

    now = now or datetime.now(timezone.utc)
    if status.state != UNCHECKED or status.refuted or not_found_holds(status, now):
        return False
    if status.confirmed and now - min(status.confirmed) < RECHECK_GAP:
        return False
    return True
