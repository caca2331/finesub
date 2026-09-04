"""Human signals report over the knowledge store (plan §5.5).

``python -m finesub.llm.knowledge.report`` — read-only, no model calls. Lists
per node/field the matched/exposed/landed event counts and the
confirmed/refuted evidence with its latest date, marks never-matched items,
admits high-false-trigger candidates by the §5.2 rule (exposed with a
correction opportunity, repeatedly, never landed), and closes with per-matcher
conversion. The numbers describe consistency, not causality (§5.1), and they
never flow back into any prompt (§5.4).
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any

from .base import knowledge_root_path
from .node.model import digest as value_digest
from .node.repo import KnowledgeRepo
from .node.store import KnowledgeStore

_Lines = list[str]


def _node_label(store: KnowledgeStore, node_id: str, rev: int) -> str:
    node = store.node(node_id, rev)
    if node is None:
        return f"{node_id} (retired)"
    payload = node.payload
    for key in ("surface", "field", "target", "occurred_at"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    text = str(payload.get("text") or "").strip()
    return (text[:24] + "…") if len(text) > 24 else (text or node_id)


def _window_key(row: Any) -> tuple[str, str, int]:
    """Join identity for one window's signals: task, window AND revision —
    correction task ids are stable filenames, so a rerun of the same material
    against a newer knowledge revision must not join rev-1 matches with
    rev-2 exposures (review 2026-08-27 round 6)."""

    return (row["task_id"], row["window_id"] or "", row["rev"])


def build_report(
    store: KnowledgeStore,
    *,
    rev: int | None = None,
    subject: str | None = None,
    min_exposures: int = 3,
) -> _Lines:
    at = store.current_rev() if rev is None else rev

    # ---- corpus at the pinned rev ------------------------------------
    subject_nodes = {node.local_id: node for node in store.subjects(at)}
    subject_ids = set(subject_nodes)
    if subject is not None:
        subject_ids = {
            node_id
            for node_id, node in subject_nodes.items()
            if node.payload.get("surface", "") == subject
        }
        if not subject_ids:
            return [f"no subject named {subject!r} at rev {at}"]

    items = {item.item_id: item for item in store.all_items(at)}

    # ---- raw rows ----------------------------------------------------
    event_rows = store.conn.execute(
        "SELECT kind, opportunity, task_id, window_id, subject_id, node_id, item_id, matcher, rev"
        " FROM events"
    ).fetchall()
    evidence_rows = store.conn.execute(
        "SELECT node_id, field_path, value_hash, verdict, evidence_kind, created_at FROM evidence"
    ).fetchall()

    # ---- aggregations ------------------------------------------------
    _Window = tuple[str, str, int]  # (task, window, rev) — see _window_key
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    matched_windows: dict[str, set[_Window]] = defaultdict(set)  # item_id
    exposed_corr_windows: dict[str, set[_Window]] = defaultdict(set)  # node_id
    landed_windows: dict[str, set[_Window]] = defaultdict(set)  # node_id
    matcher_matched: dict[str, set[tuple]] = defaultdict(set)  # (node, *window)
    exposed_triples: set[tuple] = set()
    landed_triples: set[tuple] = set()
    subjects_seen: set[str] = set()

    # First pass: which windows carried a misheard match for which node. The
    # REST path stamps opportunity=correction at exposure time; the agent path
    # cannot (the tool server has no window text), so the report derives the
    # same split from the join — same (node, task, window), misheard fired.
    item_field = {
        row["item_id"]: row["field"]
        for row in store.conn.execute("SELECT item_id, field FROM items").fetchall()
    }
    misheard_windows: dict[str, set[tuple[str, str, int]]] = defaultdict(set)  # node_id
    for row in event_rows:
        if (
            row["kind"] == "matched"
            and row["item_id"]
            and item_field.get(row["item_id"]) == "misheard"
        ):
            misheard_windows[row["node_id"] or row["subject_id"]].add(_window_key(row))

    for row in event_rows:
        if row["subject_id"] not in subject_ids:
            continue
        subjects_seen.add(row["subject_id"])
        node_id = row["node_id"] or row["subject_id"]
        window = _window_key(row)
        triple = (node_id, *window)
        kind = row["kind"]
        if kind == "matched":
            counts[node_id]["matched"] += 1
            if row["item_id"]:
                matched_windows[row["item_id"]].add(window)
            if row["matcher"]:
                matcher_matched[row["matcher"]].add(triple)
        elif kind == "exposed":
            counts[node_id]["exposed"] += 1
            exposed_triples.add(triple)
            if row["opportunity"] == "correction" or window in misheard_windows.get(node_id, ()):
                counts[node_id]["exposed_correction"] += 1
                exposed_corr_windows[node_id].add(window)
        elif kind == "landed":
            counts[node_id]["landed"] += 1
            landed_windows[node_id].add(window)
            landed_triples.add(triple)

    # Claims are (node, field_path, value_hash) — §5.3: evidence for the value
    # the model saw, not for whatever the field holds today. Rows whose hash
    # matches the *current* value attach to it; the rest stay visible as
    # stale-value history instead of decorating the new value.
    evidence: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"confirmed": 0, "refuted": 0, "latest": ""}
    )
    node_subject: dict[str, str] = {}
    for node_id in subject_ids:
        stack = [node_id]
        while stack:
            current = stack.pop()
            node_subject.setdefault(current, node_id)
            stack.extend(m.child_id for m in store.children(current, at))
    for row in evidence_rows:
        owner = node_subject.get(row["node_id"])
        if subject is not None and owner is None:
            continue
        cell = evidence[(row["node_id"], row["field_path"], row["value_hash"])]
        cell[row["verdict"]] = cell.get(row["verdict"], 0) + 1
        cell["latest"] = max(cell["latest"], row["created_at"] or "")

    def _claim_state(cell: dict, stale: bool) -> str:
        """The discrete claim-state vocabulary (plan §11.4): derived from the
        evidence rows on the fly, never stored, no numeric score."""

        confirmed = cell.get("confirmed", 0)
        refuted = cell.get("refuted", 0)
        if stale:
            return "evidence-stale"
        if confirmed and refuted:
            return "contested"
        if refuted:
            return "suspect"
        if confirmed:
            return "corroborated"
        if cell.get("unverifiable", 0):
            return "unverifiable"
        return "unverified"

    def _current_value_hash(node_id: str, field_path: str) -> str | None:
        if field_path.startswith("items/"):
            item = items.get(field_path.split("/", 1)[1])
            return value_digest(item.value) if item is not None else None
        if field_path.startswith("payload:"):
            # share group claims (payload:core/body/intro): recompute the
            # group hash so external evidence booked by an approval is not
            # forever [stale value]
            from .node.model import payload_group_hash

            node = store.node(node_id, at)
            if node is None:
                return None
            return payload_group_hash(node.kind, node.payload, field_path.split(":", 1)[1])
        if field_path.startswith("payload."):
            node = store.node(node_id, at)
            if node is None:
                return None
            value = node.payload.get(field_path.split(".", 1)[1])
            return value_digest(value) if isinstance(value, str) and value else None
        return None

    # ---- render ------------------------------------------------------
    lines: _Lines = [f"knowledge signals report — rev {at}"]
    for subject_id in sorted(subject_ids, key=lambda sid: subject_nodes[sid].payload.get("surface", "")):
        surface = subject_nodes[subject_id].payload.get("surface", subject_id)
        pack = [nid for nid, owner in node_subject.items() if owner == subject_id]
        signal_nodes = [nid for nid in pack if counts.get(nid) or any(k[0] == nid for k in evidence)]
        never_matched = [
            item
            for item in items.values()
            if item.local_id in pack and item.exact_enabled and not matched_windows.get(item.item_id)
        ]
        if not signal_nodes and not never_matched:
            continue
        lines.append("")
        lines.append(f"## {surface}")
        for nid in sorted(signal_nodes, key=lambda n: _node_label(store, n, at)):
            c = counts.get(nid, {})
            lines.append(
                f"- {_node_label(store, nid, at)}: matched {c.get('matched', 0)}"
                f" · exposed {c.get('exposed', 0)} (correction {c.get('exposed_correction', 0)})"
                f" · landed {c.get('landed', 0)}"
            )
            for (node_id, field_path, value_hash), cell in sorted(evidence.items()):
                if node_id != nid:
                    continue
                label = field_path
                if field_path.startswith("items/"):
                    item = items.get(field_path.split("/", 1)[1])
                    if item is not None:
                        label = f"{field_path} ({item.value})"
                current = _current_value_hash(node_id, field_path)
                stale = " [stale value]" if current != value_hash else ""
                lines.append(
                    f"    evidence {label}{stale}: confirmed {cell['confirmed']}"
                    f" · refuted {cell['refuted']}"
                    + (f" · unverifiable {cell['unverifiable']}" if cell.get("unverifiable") else "")
                    + f" · state={_claim_state(cell, bool(stale))}"
                    + (f" · latest {cell['latest']}" if cell["latest"] else "")
                )
        if never_matched:
            values = "、".join(item.value for item in never_matched)
            lines.append(f"  never matched: {values}")

    # ---- high false-trigger admission (§5.2) --------------------------
    lines.append("")
    lines.append(f"## high false-trigger candidates (≥{min_exposures} correction exposures, 0 landed)")
    admitted = 0
    for item_id, windows in sorted(matched_windows.items()):
        item = items.get(item_id)
        if item is None or item.field != "misheard":
            continue
        owner = item.local_id
        if subject is not None and owner not in node_subject:
            continue
        opportunity = windows & exposed_corr_windows.get(owner, set())
        if len(opportunity) < min_exposures:
            continue
        if opportunity & landed_windows.get(owner, set()):
            continue
        admitted += 1
        lines.append(
            f"- {item.value} → {_node_label(store, owner, at)}:"
            f" {len(opportunity)} exposed-correction window(s), never landed"
            " — consider exact_enabled=false"
        )
    if not admitted:
        lines.append("- none")

    # ---- per-matcher conversion ---------------------------------------
    lines.append("")
    lines.append("## per-matcher conversion (distinct node×window)")
    if not matcher_matched:
        lines.append("- no matched events")
    for matcher, triples in sorted(matcher_matched.items()):
        exposed = len(triples & exposed_triples)
        landed = len(triples & landed_triples)
        lines.append(f"- {matcher}: matched {len(triples)} → exposed {exposed} → landed {landed}")

    # ---- pending human decisions (candidate ledger, plan A6) ----------
    from .node.candidates import pending_human_reconciled

    _FRESHNESS_NOTE = {
        "current": "",
        "content-changed": "（候选内容已变化——下轮 repair 会重新裁定，本行届时作废）",
        "gone": "（当前扫描已不再产出该候选，多半已被其他修改解决——可直接 resolve）",
    }
    pending = pending_human_reconciled(store)
    lines.append("")
    lines.append("## 待人工裁定（repair 会话标记 needs_human 的候选）")
    if not pending:
        lines.append("- none")
    for row in pending:
        summary = row.get("candidate") or f"key {row['candidate_key']}"
        lines.append(
            f"- {summary}"
            f"（task {row['task_id'] or '?'}，{row['created_at']}）"
            + _FRESHNESS_NOTE.get(row.get("freshness", "current"), "")
        )
        if row["reason"]:
            lines.append(f"  判断：{row['reason']}")
        if row.get("missing"):
            lines.append(f"  缺证据：{row['missing']}")
        lines.append(
            f"  处理后：python -m finesub.llm.knowledge candidates --resolve {row['candidate_key']}"
        )

    if not subjects_seen and not evidence:
        lines.append("")
        lines.append("(no events or evidence recorded yet)")
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.knowledge.report",
        description="Read-only signals report over the knowledge store (plan §5.5).",
    )
    parser.add_argument("--root", default=None, help="knowledge root (default: resolved runtime root)")
    parser.add_argument("--subject", default=None, help="restrict to one subject (surface name)")
    parser.add_argument("--rev", type=int, default=None, help="pin the corpus to this revision")
    parser.add_argument(
        "--min-exposures",
        type=int,
        default=3,
        help="correction-exposure windows required before a never-landed item is reported (default 3)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo = KnowledgeRepo.open(knowledge_root_path(args.root))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for line in build_report(
        repo.store, rev=args.rev, subject=args.subject, min_exposures=args.min_exposures
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
