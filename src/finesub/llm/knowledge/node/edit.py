"""Human edit surface: round-trip one subject through markdown (plan §8).

``knowledge edit <subject>`` renders the human projection, the user changes
the text, and this module reparses it with the importer's parser (the exact
format parity locked), diffs the result against the stored tree, and applies
the diff as one ``kind=user`` revision through the shared apply engine — the
markdown stays a projection, never a second truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..base import _match_normalize, _parse_entry_for_index
from .apply import apply_envelope
from .envelope import Binding, Envelope
from .importer import (
    EMPTY_SLOT_RE,
    ImportFormatError,
    alias_delta,
    classify_line,
    extract_misheard,
    parse_entry_text,
    storable_term_payload,
)
from .model import METADATA_SECTION, NodeVersion
from .presets import preset_for_category
from .render import format_line, node_aliases, render_subject
from .repo import KnowledgeRepo
from .store import MembershipVersion


class EditError(RuntimeError):
    pass


@dataclass
class EditReport:
    rev: int | None = None  # None = no change
    created: int = 0
    updated: int = 0
    removed: int = 0
    #: Lines nobody could classify, parked verbatim in the staging section
    #: instead of failing the whole edit (``on_invalid="stage"``).
    staged: list[str] = field(default_factory=list)
    #: Which section they were parked in — the name is the preset's, so a
    #: caller reporting it must not hardcode 「待归类」.
    staged_section: str = ""
    rejected: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.rev is not None


_PREFIX = {"node": "@k", "item": "@i", "membership": "@m"}


class _Ops:
    """Engine op list plus the synthetic handle table backing it."""

    def __init__(self) -> None:
        self.ops: list[dict[str, Any]] = []
        self.bindings: dict[str, Binding] = {}
        self.draft_handles: list[str] = []
        self._by_key: dict[tuple[str, str], str] = {}
        self._news = 0

    def bind(self, kind: str, ident: str, from_rev: int) -> str:
        key = (kind, ident)
        handle = self._by_key.get(key)
        if handle is None:
            handle = f"{_PREFIX[kind]}{len(self._by_key) + 1}"
            self._by_key[key] = handle
            self.bindings[handle] = Binding(
                handle=handle, kind=kind, id=ident, expected_valid_from_rev=from_rev
            )
        return handle

    def new_handle(self) -> str:
        self._news += 1
        handle = f"@new{self._news}"
        self.draft_handles.append(handle)
        return handle


#: What to do with a line the grammar cannot classify.
#:
#: ``reject`` — the whole edit fails with the parser's message. Right where a
#: human is sitting in front of the editor: they see the error and fix the
#: line they just wrote.
#: ``stage`` — the line is parked VERBATIM in the preset's staging section and
#: the rest of the edit applies. Right for the run-start harvest of
#: ``rendered/``, where nobody is watching: refusing the file throws away
#: every other edit in it, and the parked line is a ``staging-line`` candidate
#: by construction, so `knowledge repair` picks it up (plan O3/O4 -- the
#: deterministic pass first, the LLM only on what it flags).
ON_INVALID = ("reject", "stage")


def _stage_unparseable(parsed, preset) -> list[str]:  # type: ignore[no-untyped-def]
    """Move lines the preset cannot take into its staging section.

    Mutates ``parsed`` and returns the parked lines in file order. Two sources:
    a section this preset does not allow at all (only strict presets have
    those), and a line whose body classifies as ``invalid`` -- an empty or
    unregistered ``[标记]``, a three-pipe term row missing its alias column, or
    prose in a section that takes neither notes nor that kind.

    The parked text is the user's bytes, unchanged. It is stored as a note
    later on rather than re-classified: the staging section allows terms too,
    so a three-pipe line would fail there for the same reason it failed at
    home.

    The staging section itself is SKIPPED. Its lines are already parked, and
    scanning them would park them again on every later harvest of the same
    file: a repeated warning about nothing new, a retire+create that changes
    the node's identity (dropping whatever verdict `repair` recorded against
    it), and a revision that says nothing. Unchanged text diffs as ``equal``
    and produces no ops at all; a line the user *writes* there and the grammar
    cannot classify becomes a note instead (``invalid_as_note``), which is
    what the section is for.
    """

    staging = preset.staging_section()
    if staging is None:
        return []
    parked: list[str] = []
    kept_sections = []
    for section in parsed.sections:
        spec = preset.section(section.name) if section.name != METADATA_SECTION else None
        if section.name == METADATA_SECTION or spec is staging:
            kept_sections.append(section)
            continue
        if spec is None:
            # A whole section the preset does not know: its lines go, and the
            # section goes with them (an empty one would only be re-rendered
            # away on the next refresh).
            parked.extend(
                raw for raw in (_strip_bullet(line[1]) for line in section.lines)
                if raw.strip() and not EMPTY_SLOT_RE.match(raw)
            )
            continue
        survivors = []
        for entry in section.lines:
            raw = _strip_bullet(entry[1])
            if (
                not raw.strip()
                or EMPTY_SLOT_RE.match(raw)
                or _is_alias_line(raw, section.name, preset)
            ):
                survivors.append(entry)
                continue
            kind, _payload = classify_line(raw, spec, preset=preset)
            if kind == "invalid":
                parked.append(raw)
                continue
            survivors.append(entry)
        section.lines = survivors
        kept_sections.append(section)
    parsed.sections = kept_sections
    return parked


def edit_subject(
    repo: KnowledgeRepo,
    subject: NodeVersion,
    new_text: str,
    *,
    note: str | None = None,
    refresh: bool = True,
    on_invalid: str = "reject",
) -> EditReport:
    """Diff ``new_text`` (human projection format) against the stored subject
    and apply the result as one user revision. Returns a no-op report when the
    text round-trips unchanged.

    The v3 grammar needs no version parameter: brackets disambiguate labels
    and the body's shape decides its kind, so an edit written against an
    older rendering still parses the same way."""

    try:
        parsed = parse_entry_text(Path(f"{subject.payload.get('surface', 'edit')}.md"), new_text)
    except ImportFormatError as exc:
        raise EditError(str(exc)) from exc
    category = subject.payload.get("category") or "common"
    preset = preset_for_category(category)
    if on_invalid not in ON_INVALID:
        raise ValueError(f"on_invalid must be one of {ON_INVALID}, got {on_invalid!r}")
    staged = _stage_unparseable(parsed, preset) if on_invalid == "stage" else []
    # Only names a section when this call is allowed to park at all: under
    # `reject` an unclassifiable line in 待归类 is still an error to report,
    # because the human who typed it is right there.
    staging = preset.staging_section() if on_invalid == "stage" else None
    store = repo.store
    rev = repo.rev
    ops = _Ops()
    report = EditReport()
    subject_handle = ops.bind("node", subject.local_id, subject.valid_from_rev)
    subject_set: dict[str, Any] = {}

    if parsed.h1 != subject.payload.get("surface", ""):
        clash = repo.resolve_in(parsed.h1, category, rev)
        if clash is not None and clash.subject_id != subject.local_id:
            raise EditError(f"rename collides with existing entry {clash.key!r}")
        subject_set["payload.surface"] = parsed.h1
    if parsed.intro != subject.payload.get("intro", ""):
        subject_set["payload.intro"] = parsed.intro

    new_sections = [s for s in parsed.sections if s.name != METADATA_SECTION]
    for section in new_sections:
        if preset.section(section.name) is None:
            raise EditError(f"section {section.name!r} not allowed for preset {preset.name}")
    new_order = [s.name for s in new_sections]
    if new_order != list(subject.payload.get("section_order", [])):
        subject_set["payload.section_order"] = new_order

    # Derived retrieval fields follow the 档案 rows (the importer derived them
    # the same way): native_names / reading from 本名, subject alias items from
    # 别名. Delta against the *old rendered text*, not wholesale recompute —
    # the payload may carry names merged from legacy index rows that the entry
    # text never showed, and those must survive an unrelated edit.
    # the FULL preview is what the user edited, so the delta must read the
    # same face — the alias row only exists there
    old_text = render_subject(store, subject.local_id, rev=rev, mode="human", preview="full")
    _, _, old_native, old_aliases = _parse_entry_for_index(old_text)
    _, _, new_native, new_aliases = _parse_entry_for_index(new_text)
    if old_native != new_native:
        merged = [n for n in subject.payload.get("native_names", []) if n not in set(old_native) - set(new_native)]
        merged.extend(n for n in new_native if n not in merged)
        if merged != list(subject.payload.get("native_names", [])):
            subject_set["payload.native_names"] = merged
    old_reading = _derived_reading(old_text)
    new_reading = _derived_reading(new_text)
    if old_reading != new_reading and new_reading != subject.payload.get("reading", ""):
        subject_set["payload.reading"] = new_reading
    # Three states, and the text alone cannot tell the first two apart — an
    # ABSENT row and an EMPTY row both parse to "no aliases" (plan §6.3):
    #   row absent      -> leave the items alone (deleting a rendered row is
    #                      not a request to wipe the retrieval index)
    #   row, empty body -> explicit clear
    #   row with values -> normal delta
    added_aliases, removed_aliases = (
        alias_delta(old_aliases, new_aliases) if _has_alias_row(parsed, preset) else ([], [])
    )
    if added_aliases or removed_aliases:
        current_items = {
            _match_normalize(item.value): item
            for item in store.items_of(subject.local_id, rev)
            if item.field == "aliases"
        }
        for alias in added_aliases:
            if _match_normalize(alias) not in current_items:
                ops.ops.append({"op": "add_item", "id": subject_handle, "field": "aliases", "value": alias})
        for alias in removed_aliases:
            item = current_items.get(_match_normalize(alias))
            if item is not None:
                ops.ops.append(
                    {"op": "remove_item", "item": ops.bind("item", item.item_id, item.valid_from_rev)}
                )

    old_by_section: dict[str, list[tuple[MembershipVersion, NodeVersion]]] = {}
    for membership in store.children(subject.local_id, rev):
        child = store.node(membership.child_id, rev)
        if child is not None:
            old_by_section.setdefault(membership.section, []).append((membership, child))

    # (section, position, membership): edges that survive; moves resolved at the end
    survivors: list[tuple[str, int, MembershipVersion]] = []
    # deletions and insertions are reconciled globally after the per-section
    # diff, so an unchanged line that merely moved (within or across sections)
    # keeps its node instead of being retired and recreated
    deletions: list[tuple[MembershipVersion, NodeVersion, str]] = []
    insertions: list[tuple[str, int, str, Any]] = []  # (section, position, raw, spec)

    for section in new_sections:
        spec = preset.section(section.name)
        old_entries = old_by_section.get(section.name, [])
        old_rendered = [
            format_line(child, aliases=node_aliases(store, child, rev)) for _, child in old_entries
        ]
        # Human-mode entries carry a markdown bullet (one entry per line in
        # rendered markdown views); the diff runs on the bare grammar.
        new_lines = [
            line
            for line in (_strip_bullet(raw) for _, raw, _ in section.lines if raw.strip())
            # Dropping empty slots here gives the three states of plan §6.3 for
            # free: a slot nobody filled disappears (no-op), a line the user
            # EMPTIED disappears from the new text so the diff retires its node
            # (explicit clear), and a line absent from the edit was never here.
            if not EMPTY_SLOT_RE.match(line) and not _is_alias_line(line, section.name, preset)
        ]
        matcher = SequenceMatcher(a=old_rendered, b=new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                survivors.extend(
                    (section.name, j1 + offset, old_entries[i1 + offset][0])
                    for offset in range(i2 - i1)
                )
                continue
            olds = old_entries[i1:i2]
            positions = list(range(j1, j2))
            paired = min(len(olds), len(positions))
            for p in range(paired):
                membership, child = olds[p]
                raw = new_lines[positions[p]]
                kind, payload = _classify(
                    raw, spec, preset, invalid_as_note=spec is staging
                )
                if kind == child.kind:
                    handle = ops.bind("node", child.local_id, child.valid_from_rev)
                    alias_column: list[str] | None = None
                    if kind == "term":
                        payload, alias_column = storable_term_payload(payload)
                    # A line rewritten without its `[标记]` has no `label` key, and a diff over
                    # the NEW payload alone would silently keep the stored one — the label
                    # would come back on the next render (review 2026-08-29 P2-1). Removing a
                    # label is a legitimate edit, so clear it explicitly.
                    changes = {
                        f"payload.{k}": v for k, v in payload.items() if child.payload.get(k) != v
                    }
                    if "label" not in payload and child.payload.get("label"):
                        changes["payload.label"] = ""
                    if changes:
                        ops.ops.append({"op": "update", "id": handle, "set": changes})
                        report.updated += 1
                    _sync_items(ops, store, rev, child, alias_column, raw)
                    survivors.append((section.name, positions[p], membership))
                else:
                    deletions.append((membership, child, old_rendered[i1 + p]))
                    insertions.append((section.name, positions[p], raw, spec))
            for offset, (membership, child) in enumerate(olds[paired:]):
                deletions.append((membership, child, old_rendered[i1 + paired + offset]))
            for j in positions[paired:]:
                insertions.append((section.name, j, new_lines[j], spec))

    kept_sections = {section.name for section in new_sections}
    for name, entries in old_by_section.items():
        if name in kept_sections:
            continue
        for membership, child in entries:
            deletions.append(
                (membership, child, format_line(child, aliases=node_aliases(store, child, rev)))
            )

    # moved lines: identical text deleted here and inserted there is the same node
    remaining_insertions: list[tuple[str, int, str, Any]] = []
    for section_name, position, raw, spec in insertions:
        match = next((d for d in deletions if d[2] == raw), None)
        if match is not None:
            deletions.remove(match)
            survivors.append((section_name, position, match[0]))
        else:
            remaining_insertions.append((section_name, position, raw, spec))
    for membership, child, _rendered in deletions:
        _retire_or_unlink(ops, store, rev, membership, child, report)
    for section_name, position, raw, spec in remaining_insertions:
        _create_line(
            ops, subject_handle, section_name, position, raw, spec, report, preset,
            invalid_as_note=spec is staging,
        )
    if staged:
        # Parked lines are appended after whatever the edit itself put in the
        # staging section, as notes carrying the user's bytes unchanged. They
        # deliberately skip `_create_line`: it re-classifies, and the whole
        # point is that classification is what failed.
        staging_name = staging.name  # type: ignore[union-attr]
        used = [pos for name, pos, _ in survivors if name == staging_name]
        used += [pos for name, pos, _, _ in remaining_insertions if name == staging_name]
        start = max(used) + 1 if used else 0
        for offset, raw in enumerate(staged):
            ops.ops.append(
                {
                    "op": "create",
                    "handle": ops.new_handle(),
                    "kind": "note",
                    "parent": subject_handle,
                    "section": staging_name,
                    "order_key": start + offset,
                    "payload": {"text": raw},
                }
            )
            report.created += 1
        report.staged = list(staged)
        report.staged_section = staging_name
    for section_name, position, membership in survivors:
        if membership.order_key != position or membership.section != section_name:
            handle = ops.bind("membership", membership.membership_id, membership.valid_from_rev)
            ops.ops.append(
                {"op": "move_membership", "membership": handle, "section": section_name, "order_key": position}
            )

    if not ops.ops and not subject_set:
        return report
    subject_set["payload.updated_date"] = repo.today()
    ops.ops.append({"op": "update", "id": subject_handle, "set": subject_set})

    envelope = Envelope(
        task_id="knowledge-edit",
        assignment_id="cli",
        context_epoch=0,
        knowledge_read_rev=rev,
        ops=ops.ops,
        handle_bindings=list(ops.bindings.values()),
        draft_bindings=ops.draft_handles,
    )
    result = apply_envelope(store, envelope, revision_kind="user", note=note or f"edit:{parsed.h1}")
    if result.rolled_back:
        raise EditError(f"edit rolled back: {result.rollback_reason}")
    report.rev = result.rev
    report.rejected = [reason for _, reason in result.rejected_ops]
    report.rejected.extend(f"{c.entity} {c.id}: {c.reason}" for c in result.conflicts)
    if refresh:
        repo.refresh_rendered()
    return report


def harvest_rendered_edits(repo: KnowledgeRepo) -> list[dict[str, Any]]:
    """Absorb pending user edits from ``rendered/`` (plan §11.3 / O4-O5).

    Each dirty file (disk hash != manifest hash) is parsed with the edit
    round-trip and applied as its own ``kind=user`` revision (the author is a
    human regardless of what triggered the harvest). Files that fail to parse
    are reported and left dirty — per-file isolation, and NO LLM repair is
    ever attempted here (dry-run discipline: the repair flow is explicit).
    Returns one result row per dirty file."""

    from finesub.reporting import current_reporter

    results: list[dict[str, Any]] = []
    absorbed: list[str] = []
    for rel, path, subject_id in repo.rendered_dirty_files():
        subject = repo.store.node(subject_id)
        if subject is None or subject.kind != "subject":
            current_reporter().warning(
                "knowledge-rendered-edit-orphan",
                f"rendered/{rel} 有未收割的编辑，但对应条目已不存在——文件保留，请人工处理",
            )
            results.append({"file": rel, "status": "orphan"})
            continue
        try:
            report = edit_subject(
                repo, subject, path.read_text(encoding="utf-8"),
                note=f"rendered-edit:{rel}", refresh=False,
                # Nobody is watching a run-start harvest, so refusing the file
                # over one line would silently throw away every other edit in
                # it. The line is parked instead, and `knowledge repair` is
                # what turns it back into something classified.
                on_invalid="stage",
            )
        except EditError as exc:
            current_reporter().warning(
                "knowledge-rendered-edit-failed",
                f"rendered/{rel} 的用户编辑无法解析，已跳过（文件保留原样）：{exc}",
                action="修正该文件后重跑，或用 `python -m finesub.llm.knowledge edit` 编辑",
            )
            results.append({"file": rel, "status": "failed", "error": str(exc)})
            continue
        absorbed.append(rel)
        if report.staged:
            current_reporter().warning(
                "knowledge-rendered-edit-staged",
                f"rendered/{rel}：{len(report.staged)} 行没看懂，"
                f"已原样泊进「{report.staged_section}」"
                f"（首行：{report.staged[0][:40]}）",
                action="跑 `python -m finesub.llm.knowledge repair` 让 AI 提出归位方案，"
                       "或自己把它改成合语法的行",
            )
        results.append({"file": rel, "status": "applied" if report.changed else "no-op",
                        "rev": report.rev, "staged": list(report.staged),
                        "rejected": report.rejected})
    if absorbed or results:
        repo.refresh_rendered(force=absorbed)
    return results


def harvest_rendered_edits_at_run_start(knowledge_root) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    """Run-start harvest hook (plan §11.3 / O5): applies pending rendered/
    edits before the run pins its generation rev, under the same guards as
    every other knowledge write — worktree gate and the cross-process write
    lock. Never blocks or fails the run: guard misses degrade to a warning."""

    import os

    from finesub.paths import is_linked_worktree
    from finesub.reporting import current_reporter

    from ..base import knowledge_root_path, knowledge_write_lock

    repo = KnowledgeRepo.open(knowledge_root_path(knowledge_root))
    dirty = repo.rendered_dirty_files()
    if not dirty:
        return []
    if is_linked_worktree() and os.environ.get("FINESUB_KNOWLEDGE_WRITE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        current_reporter().warning(
            "knowledge-rendered-harvest-skipped",
            f"检测到 {len(dirty)} 个 rendered/ 用户编辑，但当前在 git worktree 中，"
            "已跳过收割以免写入主仓知识库（FINESUB_KNOWLEDGE_WRITE=1 可放行）",
        )
        return []
    with knowledge_write_lock(repo.root) as acquired:
        if not acquired:
            current_reporter().warning(
                "knowledge-rendered-harvest-skipped",
                "知识库写锁被其他进程占用，本次运行跳过 rendered/ 编辑收割",
            )
            return []
        return harvest_rendered_edits(repo)


_BULLET_RE = re.compile(r"^- ")


def _is_alias_line(line: str, section: str, preset) -> bool:  # type: ignore[no-untyped-def]
    """The `[别名]` row has no node of its own — it renders from the subject's
    alias items and is diffed back into them by `alias_delta`. Keeping it out
    of the section diff is what stops it becoming a note (plan §4).

    Only in the section the role is registered in: labels are free-form
    elsewhere, so a legitimate `[别名]` note in another section must stay an
    ordinary line rather than silently steer the retrieval index (review
    2026-08-29 P2-3)."""

    role = preset.label_by_role("aliases")
    if role is None or _match_normalize(section) != _match_normalize(role[0]):
        return False
    return line.startswith(f"[{role[1].name}]")


def _has_alias_row(parsed, preset) -> bool:  # type: ignore[no-untyped-def]
    """Is the alias row PRESENT in the edited text (empty body included)?

    The row is an items projection, so its absence and its emptiness parse
    identically — only presence separates "leave the index alone" from
    "clear it" (plan §6.3)."""

    return any(
        _is_alias_line(_strip_bullet(raw), section.name, preset)
        for section in parsed.sections
        for _lineno, raw, _blank in section.lines
    )


def _strip_bullet(raw: str) -> str:
    """Remove the human-projection entry bullet ONLY — everything else stays
    verbatim (legacy empty facts render as ``口癖: `` with a trailing space,
    and the round-trip diff must still see them as unchanged).

    ⚠ Exactly one, and that is why a line whose CONTENT starts with `- ` loses
    that dash on the way back in: the renderer does not put a second bullet in
    front of it (see `render_subject`), so the one stripped here is the
    user's. Absorbing it is the accepted cost of one-entry-per-line (owner
    2026-09-01); the archive projection is unaffected, it carries no
    bullets."""

    return _BULLET_RE.sub("", raw, count=1)


def _derived_reading(text: str) -> str:
    """The importer's reading derivation: the parenthesis in the 本名 row."""

    line = next(
        (
            row.split("]", 1)[1] if row.startswith("[") else row
            for row in map(_strip_bullet, text.splitlines())
            if row.startswith("[本名]") or row.startswith("本名")
        ),
        "",
    )
    paren = re.search(r"[（(]([^（）()]*)[）)]", line.split("/")[0]) if line else None
    return paren.group(1).strip() if paren else ""


def _classify(raw: str, spec, preset=None, *, invalid_as_note: bool = False) -> tuple[str, dict[str, Any]]:  # type: ignore[no-untyped-def]
    """``invalid_as_note`` is the staging section's rule: a line nobody could
    classify still belongs there, verbatim, because that is what the section
    holds. Everywhere else an unclassifiable line is an error the caller
    decides about (``on_invalid``)."""

    kind, payload = classify_line(raw, spec, preset=preset)
    if kind == "invalid":
        if invalid_as_note:
            return "note", {"text": raw}
        raise EditError(f"{payload.get('error', 'invalid line')}：{raw!r}")
    return kind, payload


def _create_line(ops: _Ops, subject_handle: str, section: str, order: int, raw: str, spec, report: EditReport, preset=None, *, invalid_as_note: bool = False) -> None:  # type: ignore[no-untyped-def]
    kind, payload = _classify(raw, spec, preset, invalid_as_note=invalid_as_note)
    aliases: list[str] = []
    if kind == "term":
        payload, aliases = storable_term_payload(payload)
    handle = ops.new_handle()
    ops.ops.append(
        {
            "op": "create",
            "handle": handle,
            "kind": kind,
            "parent": subject_handle,
            "section": section,
            "order_key": order,
            "payload": payload,
        }
    )
    for variant in extract_misheard(raw):
        ops.ops.append({"op": "add_item", "id": handle, "field": "misheard", "value": variant})
    for alias in dict.fromkeys(aliases):
        ops.ops.append({"op": "add_item", "id": handle, "field": "aliases", "value": alias})
    report.created += 1


def _retire_or_unlink(
    ops: _Ops,
    store,  # type: ignore[no-untyped-def]
    rev: int,
    membership: MembershipVersion,
    child: NodeVersion,
    report: EditReport,
) -> None:
    """Deleting a line retires the node — unless it also lives under another
    parent, in which case only this edge goes."""

    if len(store.parents(child.local_id, rev)) > 1:
        handle = ops.bind("membership", membership.membership_id, membership.valid_from_rev)
        ops.ops.append({"op": "remove_membership", "membership": handle})
    else:
        handle = ops.bind("node", child.local_id, child.valid_from_rev)
        ops.ops.append({"op": "retire", "id": handle})
    report.removed += 1


def _sync_items(ops: _Ops, store, rev: int, child: NodeVersion, alias_column: list[str] | None, raw: str) -> None:  # type: ignore[no-untyped-def]
    """Items follow the rewritten line: the alias column diffs both ways
    against the CURRENT alias items (the column renders from items now, plan
    §11.2), misheard extraction is add-only (plan §2.1/§4.1)."""

    items = store.items_of(child.local_id, rev)
    if child.kind == "term" and alias_column is not None:
        current = {_match_normalize(item.value): item for item in items if item.field == "aliases"}
        added, removed = alias_delta([item.value for item in current.values()], alias_column)
        handle = ops.bind("node", child.local_id, child.valid_from_rev)
        for alias in added:
            if _match_normalize(alias) not in current:
                ops.ops.append({"op": "add_item", "id": handle, "field": "aliases", "value": alias})
        for alias in removed:
            item = current.get(_match_normalize(alias))
            if item is not None:
                item_handle = ops.bind("item", item.item_id, item.valid_from_rev)
                ops.ops.append({"op": "remove_item", "item": item_handle})
    known_misheard = {item.value for item in items if item.field == "misheard"}
    handle = ops.bind("node", child.local_id, child.valid_from_rev)
    for variant in extract_misheard(raw):
        if variant not in known_misheard:
            ops.ops.append({"op": "add_item", "id": handle, "field": "misheard", "value": variant})
