"""Phase B of the re-import (kb-line-grammar plan §8): the deterministic
conversion from the frozen v1 archive shape into the v3 line grammar.

Phase A imports the markdown byte-for-byte (legacy kinds, ``check_parity``
as the gate). This module then rewrites those rows into ``subject`` /
``term`` / ``note`` plus labels, and emits a ROW-BY-ROW table — after this
step the legacy projection no longer applies, so the table is the only
thing standing between the conversion and a silent semantic change (plan §7).

Everything here is deterministic. Judgement calls (naming a relation's
Chinese column, splitting a run-on line into terms, applying a section's
``exclude`` discipline) are Phase C's, and land in ``待归类`` so they are
visible instead of guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .importer import (
    STAGING_SECTION,
    extract_misheard,
    storable_term_payload,
    strip_misheard_prose,
)
from .matching import scan_normalize
from .model import migration_id
from .presets import preset_for_category
from .store import KnowledgeStore


def _fold(text: str) -> str:
    return scan_normalize(text or "").replace(" ", "")



#: Where each frozen v1 section's content lands. A section that keeps its
#: name still gets its lines converted; the remapped ones are the structural
#: decisions from plan §3 (`说话风格` folded into `特点`, `重要经历` retired).
SECTION_MAP: dict[str, str] = {
    "档案": "档案",
    "直播内容": "直播内容",
    "说话风格": "特点",
    "频道用语": "频道用语",
    "喜好 / 特点": "特点",
    "重要经历": STAGING_SECTION,
    "人际关系": "人际关系",
}

#: v1 fact fields that become labels as-is.
LABEL_FIELDS = ("本名", "别名", "人设", "外观", "出道", "生日", "身高", "语体", "收录范围", "当前版本")
#: v1 fact fields whose value needs a judgement call before it can be a term
#: (a Chinese column, an alias split): parked in 待归类 with the label kept.
DEFERRED_FIELDS = ("自称", "他称", "粉丝名", "口癖", "会限互动", "听众统称")
#: The grab-bag: `其他: a；b；c` is three statements in one row.
GRAB_BAG_FIELDS = ("其他",)


@dataclass
class Row:
    """One source line and what became of it."""

    node_id: str
    subject: str
    old_section: str
    old_line: str
    new_section: str
    new_line: str
    action: str  # convert | split | park | drop | merge
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class PhaseBPlan:
    rev: int
    rows: list[Row] = field(default_factory=list)
    ops: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rev": self.rev, "rows": [row.to_dict() for row in self.rows]}


def _split_grab_bag(value: str) -> list[tuple[str, str]]:
    """``所属 X；出道 Y`` -> ``[("所属", "X"), ("出道", "Y")]``; a chunk whose
    head is not a known label keeps the whole chunk as an unlabelled line."""

    out: list[tuple[str, str]] = []
    for chunk in re.split(r"[；;]", value):
        chunk = chunk.strip()
        if not chunk:
            continue
        head = next((name for name in LABEL_FIELDS + DEFERRED_FIELDS if chunk.startswith(name)), "")
        if head:
            out.append((head, chunk[len(head) :].strip()))
        else:
            out.append(("", chunk))
    return out


def _term_ops(node_id: str, payload: dict[str, Any], raw_line: str) -> list[dict[str, Any]]:
    """Alias column and 误听 prose become items; the payload keeps only what
    the four columns render."""

    storable, aliases = storable_term_payload(payload)
    dropped = {f"payload.{key}": "" for key in ("alias_text", "reading", "sep") if key in payload}
    ops: list[dict[str, Any]] = [
        {"op": "update", "id": node_id,
         "set": {**dropped, **{f"payload.{k}": v for k, v in storable.items()}}}
    ]
    surface_fold = _fold(str(storable.get("surface") or ""))
    seen_folds = {surface_fold}
    for value in dict.fromkeys(aliases):
        # dedupe by NORMALIZED form: the reading repeated in the alias column,
        # or a width/kana variant of it, is one alias — and a reading that
        # folds equal to its surface carries no information at all
        fold = _fold(value)
        if not fold or fold in seen_folds:
            continue
        seen_folds.add(fold)
        ops.append({"op": "add_item", "id": node_id, "field": "aliases", "value": value})
    for variant in extract_misheard(raw_line):
        ops.append({"op": "add_item", "id": node_id, "field": "misheard", "value": variant})
    return ops


def build_plan(store: KnowledgeStore, rev: int | None = None) -> PhaseBPlan:
    from .render import format_line

    at = store.current_rev() if rev is None else rev
    plan = PhaseBPlan(rev=at)
    seen_labels: dict[tuple[str, str], set[str]] = {}

    # legacy placeholder items (the `—` the old english column left behind)
    # are punctuation, not names
    for item in store.all_items(at):
        if item.field == "aliases" and not re.search(r"\w", item.value):
            plan.ops.append({"op": "remove_item", "id": item.item_id})

    for subject in store.subjects(at):
        surface = str(subject.payload.get("surface") or "")
        category = str(subject.payload.get("category") or "common")
        preset = preset_for_category(category)
        if _fold_duplicate(plan, store, at, subject, surface):
            continue
        # subject payload residue the v3 model no longer carries
        residue = {key: "" for key in ("reading",) if subject.payload.get(key)}
        if residue or subject.payload.get("section_order"):
            plan.ops.append({
                "op": "update", "id": subject.local_id,
                "set": {
                    **{f"payload.{key}": value for key, value in residue.items()},
                    "payload.section_order": list(preset.section_names()),
                },
            })
        for membership in store.children(subject.local_id, at):
            child = store.node(membership.child_id, at)
            if child is None:
                continue
            old_section = membership.section
            new_section = SECTION_MAP.get(old_section, old_section)
            if preset.section(new_section) is None:
                new_section = STAGING_SECTION if preset.strict_sections else new_section
            old_line = format_line(child)
            rows_before = len(plan.rows)
            _convert_child(
                plan, store, at, subject_id=subject.local_id, surface=surface,
                child=child, membership=membership, old_section=old_section,
                new_section=new_section, old_line=old_line, preset=preset,
                seen_labels=seen_labels,
            )
            if len(plan.rows) == rows_before:  # nothing to do: already v3
                plan.rows.append(Row(
                    node_id=child.local_id, subject=surface, old_section=old_section,
                    old_line=old_line, new_section=new_section, new_line=old_line,
                    action="convert", note="unchanged",
                ))
    return plan


def _fold_duplicate(
    plan: PhaseBPlan,
    store: KnowledgeStore,
    at: int,
    subject,  # NodeVersion  # type: ignore[no-untyped-def]
    surface: str,
) -> bool:
    """Two entry files with the same H1 became two subjects; fold the loser in.

    The importer ranks them (newest, then longest) and stamps the loser with
    ``duplicate_of`` -- it cannot merge them itself, because the archive has to
    round-trip file by file and a merged projection would not. Here is where it
    can happen: every line moves into the winner's staging section and the
    loser subject retires. Nothing is dropped and nothing is guessed at -- the
    lines land where a human or the LLM pass decides what they were
    (owner 2026-09-01).

    ⚠ The moved lines keep their v1 shape: folding returns early, so
    `_convert_child` never sees them and `本名: 甲` lands in the staging
    section as free text rather than as `[本名] 甲`. Everything else parked
    there IS converted first, so this is the one inconsistent producer.
    Deliberate: converting and reparenting the same membership in one plan
    means two ops on one entity per revision, and the staging section exists
    precisely so a human or the LLM pass reshapes what needs a judgement call
    -- which is what these lines need anyway. It round-trips (an unlabelled
    line parses as a note), it just is not pretty until someone tidies it.

    Returns whether this subject was folded (and so must not be converted as
    an entry of its own).
    """

    aux = store.migration_aux(subject.local_id)
    winner_id = str((aux.layout if aux else {}).get("duplicate_of") or "")
    if not winner_id:
        return False
    if store.node(winner_id, at) is None:
        # The winner is gone (retired by hand between import and phase B):
        # folding into nothing would delete content, so leave the duplicate
        # standing and say so.
        plan.rows.append(Row(
            node_id=subject.local_id, subject=surface, old_section="", old_line=surface,
            new_section="", new_line=surface, action="park",
            note="标记为重复，但胜出条目已不存在——保持原样，交人工",
        ))
        return False
    moved = 0
    for membership in store.children(subject.local_id, at):
        plan.ops.append({
            "op": "move_membership", "membership": membership.membership_id,
            "parent": winner_id, "section": STAGING_SECTION,
        })
        moved += 1
    plan.ops.append({"op": "retire", "id": subject.local_id})
    plan.rows.append(Row(
        node_id=subject.local_id, subject=surface, old_section="", old_line=surface,
        new_section=STAGING_SECTION, new_line=surface, action="merge",
        note=f"与同名条目重复：{moved} 行并入胜出条目的「{STAGING_SECTION}」，本条目退休",
    ))
    return True


def _convert_child(
    plan: PhaseBPlan,
    store: KnowledgeStore,
    at: int,
    *,
    subject_id: str,
    surface: str,
    child: Any,
    membership: Any,
    old_section: str,
    new_section: str,
    old_line: str,
    preset: Any,
    seen_labels: dict[tuple[str, str], set[str]],
) -> None:
    payload = dict(child.payload)

    def record(new_line: str, action: str, note: str = "", section: str | None = None) -> None:
        plan.rows.append(Row(
            node_id=child.local_id, subject=surface, old_section=old_section,
            old_line=old_line, new_section=section or new_section, new_line=new_line,
            action=action, note=note,
        ))

    def move(section: str) -> None:
        if section != old_section:
            plan.ops.append({
                "op": "move_membership", "membership": membership.membership_id,
                "section": section, "order_key": membership.order_key,
            })

    def unique_label(section: str, label: str) -> str:
        used = seen_labels.setdefault((subject_id, section), set())
        if label in used:
            return ""  # a section holds one line per label: drop the duplicate label
        used.add(label)
        return label

    if child.kind == "term":
        plan.ops.extend(_term_ops(child.local_id, payload, old_line))
        move(new_section)
        storable, aliases = storable_term_payload(dict(payload))
        preview = "|".join((
            storable.get("surface", ""), storable.get("zh", ""),
            "、".join(dict.fromkeys(aliases)), storable.get("desc", ""),
        ))
        record(preview, "convert", "reading/alias/误听 → items")
        return

    if child.kind == "relation":
        target = str(payload.get("target") or "")
        description = str(payload.get("description") or "")
        new_payload = {
            "surface": target, "zh": "", "desc": strip_misheard_prose(description),
        }
        plan.ops.append({"op": "update", "id": child.local_id, "set": {
            "payload.surface": target, "payload.zh": "", "payload.desc": new_payload["desc"],
            "payload.target": "", "payload.description": "", "payload.sep": "",
        }})
        plan.ops.append({"op": "retype", "id": child.local_id, "kind": "term"})
        for variant in extract_misheard(description):
            plan.ops.append({"op": "add_item", "id": child.local_id, "field": "misheard", "value": variant})
        move(new_section)
        record(f"{target}||{'' }|{new_payload['desc']}", "convert",
               "relation → term 体；中文定名待 Phase C")
        return

    if child.kind == "event":
        occurred = str(payload.get("occurred_at") or "")
        description = str(payload.get("description") or "")
        text = f"{occurred} {description}".strip()
        plan.ops.append({"op": "update", "id": child.local_id, "set": {
            "payload.text": text, "payload.occurred_at": "", "payload.description": "",
            "payload.sep": "",
        }})
        plan.ops.append({"op": "retype", "id": child.local_id, "kind": "note"})
        move(STAGING_SECTION)
        record(text, "park", "重要经历退休——归位或删除由人决定", section=STAGING_SECTION)
        return

    if child.kind == "fact":
        field_name = str(payload.get("field") or "").strip()
        value = str(payload.get("value") or "").strip()
        if not value:
            plan.ops.append({"op": "retire", "id": child.local_id})
            record("", "drop", "空值——稀疏即缺席")
            return
        if field_name in GRAB_BAG_FIELDS:
            parts = _split_grab_bag(value)
            first = True
            for ordinal, (label, chunk) in enumerate(parts):
                line = f"[{label}] {chunk}" if label else chunk
                if first:
                    plan.ops.append({"op": "update", "id": child.local_id, "set": {
                        "payload.text": chunk, "payload.label": unique_label(new_section, label),
                        "payload.field": "", "payload.value": "", "payload.sep": "",
                    }})
                    plan.ops.append({"op": "retype", "id": child.local_id, "kind": "note"})
                    move(new_section)
                    first = False
                else:
                    plan.ops.append({
                        "op": "create", "kind": "note", "parent": subject_id,
                        "origin": child.local_id,  # the row this chunk was split out of
                        "ordinal": ordinal,        # ...and where in it (chunks repeat)
                        "section": new_section, "order_key": membership.order_key,
                        "payload": {"text": chunk, **({"label": unique_label(new_section, label)} if label else {})},
                    })
                record(line, "split", f"其他 大杂烩拆成 {len(parts)} 行")
            return
        if field_name in DEFERRED_FIELDS:
            plan.ops.append({"op": "update", "id": child.local_id, "set": {
                "payload.text": value, "payload.label": unique_label(STAGING_SECTION, field_name),
                "payload.field": "", "payload.value": "", "payload.sep": "",
            }})
            plan.ops.append({"op": "retype", "id": child.local_id, "kind": "note"})
            move(STAGING_SECTION)
            record(f"[{field_name}] {value}", "park",
                   "需要中文定名/别名拆分，交 Phase C", section=STAGING_SECTION)
            return
        aliases_role = preset.label_by_role("aliases")
        if aliases_role is not None and field_name == aliases_role[1].name:
            # The alias row has no node in v3: it renders from the items the
            # import already derived from this very line (plan §4). Keeping
            # the node would restore the payload/items duplication.
            plan.ops.append({"op": "retire", "id": child.local_id})
            record(f"[{field_name}] {value}", "convert", "别名行改由 items 渲染，节点退役")
            return
        label = field_name if field_name in LABEL_FIELDS else ""
        if not label:
            # a prose bullet the v1 fact regex split on a stray colon: put the
            # line back together instead of keeping the bogus field/value pair
            text = f"{field_name}{payload.get('sep', ': ')}{value}"
            plan.ops.append({"op": "update", "id": child.local_id, "set": {
                "payload.text": text, "payload.field": "", "payload.value": "", "payload.sep": "",
            }})
            plan.ops.append({"op": "retype", "id": child.local_id, "kind": "note"})
            move(new_section)
            record(text, "convert", "v1 把散文按冒号误切成 fact——整行复原")
            return
        plan.ops.append({"op": "update", "id": child.local_id, "set": {
            "payload.text": value, "payload.label": unique_label(new_section, label),
            "payload.field": "", "payload.value": "", "payload.sep": "",
        }})
        plan.ops.append({"op": "retype", "id": child.local_id, "kind": "note"})
        move(new_section)
        record(f"[{label}] {value}", "convert")
        return

    if child.kind == "note":
        move(new_section)
        if new_section != old_section:
            record(str(payload.get("text") or ""), "convert", f"{old_section} → {new_section}")


def execute_plan(store: KnowledgeStore, plan: PhaseBPlan, *, task_id: str = "phase-b") -> int:
    """Apply the conversion as ONE ``import`` revision.

    This writes through the store rather than the apply engine on purpose: a
    kind change is forbidden on the write path (``update`` may not change a
    line's type) precisely so models cannot do it — a migration is the one
    place it is legitimate, and it lives here where it is auditable."""

    with store.begin("import", task_id=task_id, note="phase-b §8") as txn:
        rev = txn.rev
        # kind first: a payload written against the OLD kind would fail its
        # structural check on the way in
        for op in plan.ops:
            if op["op"] == "retype":
                store.conn.execute("UPDATE nodes SET kind=? WHERE local_id=?", (op["kind"], op["id"]))
        for op in plan.ops:
            name = op["op"]
            if name == "retype":
                continue
            if name == "update":
                node = store.node(op["id"])
                if node is None:
                    continue
                payload = dict(node.payload)
                for key, value in op["set"].items():
                    field_name = key.split(".", 1)[1]
                    if value == "" and field_name in payload:
                        payload.pop(field_name)
                    elif value != "":
                        payload[field_name] = value
                txn.update_node(op["id"], payload=payload, expected_from_rev=node.valid_from_rev)
            elif name == "create":
                # Deterministic, but scoped to the SOURCE ROW *and the
                # position inside it*: two entries splitting an identical chunk
                # (`其他: 出道 2022-12`), or one row repeating a chunk
                # (`其他: 重复；重复`), would otherwise derive the same UUID5 and
                # trip nodes.local_id's UNIQUE constraint mid-migration
                # (review 2026-08-29 P2-2).
                local_id = migration_id(
                    "phase-b", op["parent"], str(op.get("origin") or ""),
                    str(op.get("ordinal", "")), op["section"],
                    json.dumps(op["payload"], sort_keys=True, ensure_ascii=False),
                )
                txn.create_node(local_id, op["kind"], op["payload"])
                txn.create_membership(
                    migration_id("phase-b-m", local_id), op["parent"], local_id,
                    op["section"], int(op.get("order_key", 0)),
                )
            elif name == "move_membership":
                current = next(
                    (m for m in store.conn.execute(
                        "SELECT valid_from_rev FROM membership_versions WHERE membership_id=?"
                        " AND valid_to_rev IS NULL", (op["membership"],)
                    ).fetchall()), None,
                )
                if current is None:
                    continue
                # `parent` and `order_key` are both optional: folding a
                # duplicate entry moves memberships to ANOTHER subject and has
                # no opinion about their order there.
                txn.move_membership(
                    op["membership"],
                    parent_id=op.get("parent"),
                    section=op["section"],
                    order_key=int(op["order_key"]) if "order_key" in op else None,
                    expected_from_rev=current["valid_from_rev"],
                )
            elif name == "add_item":
                item_id = migration_id("phase-b-i", op["id"], op["field"], op["value"])
                existing = {i.value for i in store.items_of(op["id"]) if i.field == op["field"]}
                if op["value"] not in existing:
                    txn.create_item(item_id, op["id"], op["field"], op["value"])
            elif name == "remove_item":
                item = next(
                    (row for row in store.conn.execute(
                        "SELECT item_id, valid_from_rev FROM item_versions WHERE item_id=?"
                        " AND valid_to_rev IS NULL", (op["id"],)).fetchall()), None,
                )
                if item is not None:
                    txn.tombstone_item(op["id"], expected_from_rev=item["valid_from_rev"])
            elif name == "retire":
                node = store.node(op["id"])
                if node is not None:
                    txn.tombstone_node(op["id"], expected_from_rev=node.valid_from_rev)
    return rev


def write_report(plan: PhaseBPlan, directory: str | Path) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "phase-b.json"
    path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    table = out / "phase-b.md"
    lines = ["| 条目 | 旧节 | 旧行 | 新节 | 新行 | 动作 | 备注 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in plan.rows:
        cells = [row.subject, row.old_section, row.old_line, row.new_section, row.new_line, row.action, row.note]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_plan(plan: PhaseBPlan) -> list[str]:
    counts: dict[str, int] = {}
    for row in plan.rows:
        counts[row.action] = counts.get(row.action, 0) + 1
    summary = "、".join(f"{action} {count}" for action, count in sorted(counts.items()))
    return [f"phase B @rev {plan.rev}: {len(plan.rows)} 行（{summary}），{len(plan.ops)} 个操作"]
