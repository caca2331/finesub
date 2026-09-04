"""Client CLI: ``python -m finesub.llm.knowledge.share <command>``.

``mark`` flips eligible nodes to ``visibility=shareable`` (push is an explicit
per-subject act, §6.4; only subject/term kinds inherit by default — prose notes
need ``--kinds`` as the second confirmation). ``push`` builds
the bundle from shareable nodes only and remembers the queue item locally so
``status`` can backfill the server-assigned canonical ids once approved.
``pull`` fetches, verifies and merges a snapshot (``sync.apply_snapshot``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..base import knowledge_root_path
from ..node.repo import AmbiguousName, KnowledgeRepo
from . import client, conflicts as conflict_ledger
from .exchange import build_push_bundle, bundle_content_digest, strip_local_maps
from .sync import apply_snapshot

_Lines = list[str]

PUSH_LOG_FILENAME = "share-pushes.jsonl"
DEFAULT_MARK_KINDS = ("subject", "term")


def _resolve(repo: KnowledgeRepo, name: str):  # type: ignore[no-untyped-def]
    """Human-facing name->entry with the namespace rule (`node/repo.py`):
    ``style/某字幕组`` addresses one category, a bare name that two categories
    answer to is refused rather than guessed. Pushing or marking the entry the
    user did not mean is the failure this prevents."""

    try:
        return repo.resolve_qualified(name)
    except AmbiguousName as exc:
        raise SystemExit(f"error: {exc}") from exc


def _repo(args: argparse.Namespace) -> KnowledgeRepo:
    return KnowledgeRepo.open(knowledge_root_path(args.root))


def _remote_id(url: str) -> str:
    return url.rstrip("/")


def _token_key(remote: str) -> str:
    return f"share:{remote}:token"


def _cmd_register(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    token = client.register(args.remote)
    repo.store.conn.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        (_token_key(_remote_id(args.remote)), token),
    )
    return ["registered; token stored in the knowledge store"]


def _visibility_targets(repo: KnowledgeRepo, args: argparse.Namespace, wanted: str) -> tuple[str, list[str]]:
    """Nodes under the subject whose visibility should become ``wanted``,
    keeping the shareable set **ancestry-closed** (review 2026-08-27 round 6):

    - marking a child also marks its parent chain up to the subject — a
      shareable node under a local ancestor would either be dropped by the
      bundle builder or arrive orphaned on the server;
    - unmarking cascades downward — nodes left shareable under an unmarked
      ancestor (including notes that inherited it at creation) would leak on
      the next push.

    ``--kinds`` widens past the subject/term default (the §6.4 second
    confirmation for prose notes); ``--match`` narrows to nodes whose LABEL
    contains the text, so one public line can be marked without dragging in
    the real-name one next to it."""

    resolved = _resolve(repo, args.subject)
    if resolved is None:
        raise SystemExit(f"error: no entry matches {args.subject!r}")
    kinds = set(DEFAULT_MARK_KINDS) | {k.strip() for k in (args.kinds or "").split(",") if k.strip()}
    match = (args.match or "").strip()
    from ..node.signals import subject_pack_node_ids

    store = repo.store
    rev = store.current_rev()
    pack = subject_pack_node_ids(store, resolved.subject_id, rev)
    selected: set[str] = set()
    for node_id in pack:
        node = store.node(node_id)
        if node is None or node.kind not in kinds:
            continue
        if match:
            label = " ".join(
                str(node.payload.get(key) or "")
                for key in ("surface", "zh", "label", "text")
            )
            if match not in label:
                continue
        selected.add(node_id)
    if wanted == "shareable":
        # upward closure: every selected node brings its parent chain
        for node_id in list(selected):
            walker, seen = node_id, {node_id}
            while walker != resolved.subject_id:
                parents = store.parents(walker)
                if not parents or parents[0].parent_id in seen:
                    break
                walker = parents[0].parent_id
                seen.add(walker)
                selected.add(walker)
    else:
        # downward cascade: descendants of an unmarked node go local too
        frontier = list(selected)
        while frontier:
            for membership in store.children(frontier.pop()):
                if membership.child_id not in selected:
                    selected.add(membership.child_id)
                    frontier.append(membership.child_id)
    targets = []
    for node_id in pack:  # pack order keeps the revision deterministic
        node = store.node(node_id)
        if node is not None and node_id in selected and node.visibility != wanted:
            targets.append(node_id)
    return resolved.key, targets


def _cmd_mark(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    key, targets = _visibility_targets(repo, args, "shareable")
    if not targets:
        return ["nothing to mark (already shareable, or excluded — see --kinds/--match)"]
    with repo.store.begin("user", note=f"share mark {key}") as txn:
        for node_id in targets:
            txn.update_node(node_id, visibility="shareable")
    return [f"marked {len(targets)} node(s) shareable under {key} (ancestors included)"]


def _cmd_unmark(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    key, targets = _visibility_targets(repo, args, "local")
    if not targets:
        return ["nothing to unmark"]
    with repo.store.begin("user", note=f"share unmark {key}") as txn:
        for node_id in targets:
            txn.update_node(node_id, visibility="local")
    return [f"unmarked {len(targets)} node(s) under {key} (descendants cascaded)"]


def _append_push_record(root: Path, record: dict[str, Any]) -> None:
    with (root / PUSH_LOG_FILENAME).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _cmd_push(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    store = repo.store
    remote = _remote_id(args.remote)
    token = store.meta(_token_key(remote))
    if not token:
        raise SystemExit(f"error: not registered with {remote} — run `share register` first")
    subject_ids = []
    for name in args.subjects:
        resolved = _resolve(repo, name)
        if resolved is None:
            raise SystemExit(f"error: no entry matches {name!r}")
        subject_ids.append(resolved.subject_id)
    bundle = build_push_bundle(store, subject_ids)
    wire = strip_local_maps(bundle)
    wire_digest = bundle_content_digest(wire)
    # Retry-safe idempotency (review 2026-08-27): the key survives a lost
    # response because the intent — key + content digest — is on disk BEFORE
    # the request leaves. A retry of the same content reuses the recorded key,
    # so the server dedupes instead of queueing a fork.
    prior = None
    log_path = repo.root / PUSH_LOG_FILENAME
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("remote") == remote and record.get("bundle_digest") == wire_digest:
                prior = record
    if prior is not None:
        bundle["idempotency_key"] = wire["idempotency_key"] = prior["idempotency_key"]
    else:
        _append_push_record(
            repo.root,
            {
                "remote": remote,
                "status": "intent",
                "idempotency_key": bundle["idempotency_key"],
                "bundle_digest": wire_digest,
                "handle_map": bundle["handle_map"],
                "item_handle_map": bundle["item_handle_map"],
            },
        )
    reply = client.push_bundle(remote, wire, token=token)
    _append_push_record(
        repo.root,
        {
            "remote": remote,
            "status": reply.get("status", "pending"),
            "queue_id": reply["queue_id"],
            "idempotency_key": bundle["idempotency_key"],
            "bundle_digest": wire_digest,
            "handle_map": bundle["handle_map"],
            "item_handle_map": bundle["item_handle_map"],
        },
    )
    return [
        f"queued as item {reply['queue_id']} ({'duplicate retry' if reply.get('duplicate') else 'new'});"
        f" check with: share status --remote {remote} --queue-id {reply['queue_id']}"
    ]


def _push_record(root: Path, remote: str, queue_id: int) -> dict[str, Any] | None:
    path = root / PUSH_LOG_FILENAME
    if not path.is_file():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("remote") == remote and record.get("queue_id") == queue_id:
            found = record  # last record wins
    return found


def _cmd_status(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    remote = _remote_id(args.remote)
    token = repo.store.meta(_token_key(remote))
    if not token:
        raise SystemExit(f"error: not registered with {remote} — run `share register` first")
    reply = client.push_status(remote, args.queue_id, token=token)
    lines = [f"queue item {args.queue_id}: {reply['status']}"]
    if reply.get("verdict"):
        lines.append(f"verdict note: {reply['verdict']}")
    if reply["status"] != "approved":
        return lines
    record = _push_record(repo.root, remote, args.queue_id)
    if record is None:
        lines.append("no local push record found; canonical ids not backfilled")
        return lines
    assigned = reply.get("assigned") or {}
    store = repo.store
    nodes = items = 0
    from ..node.model import digest as payload_digest

    for handle, local_id in (record.get("handle_map") or {}).items():
        canonical = assigned.get(handle)
        if not canonical:
            continue
        store.conn.execute(
            "UPDATE node_versions SET canonical_id=? WHERE local_id=? AND valid_to_rev IS NULL",
            (canonical, local_id),
        )
        row = store.conn.execute(
            "SELECT payload FROM node_versions WHERE local_id=? AND valid_to_rev IS NULL",
            (local_id,),
        ).fetchone()
        if row is not None:
            store.conn.execute(
                "INSERT OR REPLACE INTO sync_state(remote, canonical_id, local_id,"
                " last_server_rev, last_pulled_payload_hash) VALUES (?, ?, ?, NULL, ?)",
                (remote, canonical, local_id, payload_digest(json.loads(row["payload"]))),
            )
        nodes += 1
    for handle, item_id in (record.get("item_handle_map") or {}).items():
        canonical = assigned.get(handle)
        if not canonical:
            continue
        store.conn.execute(
            "UPDATE item_versions SET canonical_item_id=? WHERE item_id=? AND valid_to_rev IS NULL",
            (canonical, item_id),
        )
        items += 1
    lines.append(f"backfilled canonical ids: {nodes} node(s), {items} item(s)")
    return lines


def _cmd_pull(args: argparse.Namespace) -> _Lines:
    repo = _repo(args)
    remote = _remote_id(args.remote)
    snapshot = client.fetch_snapshot(remote)
    report = apply_snapshot(repo.store, snapshot, remote=remote)
    repo.refresh_rendered()
    lines = [
        f"pulled server rev {report.server_rev} → local rev {report.rev}:"
        f" +{report.created_nodes} nodes, ~{report.updated_nodes} updated,"
        f" -{report.retired_nodes} retired; +{report.created_items} items,"
        f" +{report.created_memberships} memberships, +{report.created_links} links"
    ]
    for conflict in report.conflicts:
        lines.append(f"conflict (local kept): {conflict.describe()}")
    for pending in report.pending_prose:
        lines.append(f"prose pending manual merge: {pending.describe()}")
    # The conflicts outlive this terminal. Written AFTER the merge committed:
    # a ledger row for a pull that then failed would point at a state that
    # never existed.
    opened, still_open = conflict_ledger.record_conflicts(repo.root, report.unsettled())
    if opened or still_open:
        lines.append(
            f"recorded in {conflict_ledger.CONFLICT_LOG_FILENAME}:"
            f" {len(opened)} new, {still_open} already open —"
            f" list them with: share conflicts --remote {remote}"
        )
    return lines


def _record_manual_verdict(repo: KnowledgeRepo, args: argparse.Namespace) -> _Lines:
    """Close one conflict by hand.

    Without this there is no way to close a conflict at all except by running
    the LLM round -- and that round skips any conflict whose local entry has
    been retired, so those would be listed and skipped forever (reviewer
    2026-09-01 P2). A person disagreeing with a machine verdict, or settling
    something the machine cannot see, needed a door regardless.

    `dismiss` and `resolve` differ in stickiness, exactly as they do when the
    repair round writes them: see `conflicts` for why.
    """

    identity = args.dismiss or args.resolve
    status = conflict_ledger.DISMISSED if args.dismiss else conflict_ledger.RESOLVED
    reason = args.reason.strip()
    if not reason:
        raise SystemExit("error: --reason is required; a verdict with no stated reason is not one")
    try:
        row = conflict_ledger.record_verdict(
            repo.root, identity, status=status, reason=reason,
            task_id="manual", applied_rev=repo.rev,
        )
    except KeyError:
        raise SystemExit(
            f"error: no conflict {identity!r} in the ledger"
            " — the id is the first column of `share conflicts`"
        ) from None
    sticky = "（粘住，之后的 pull 不再重开）" if status == conflict_ledger.DISMISSED else (
        "（不粘：同一条冲突再出现会重新打开，那说明没改成）"
    )
    return [f"{identity} → {status}{sticky}", f"  {conflict_ledger.describe(row)}"]


def _cmd_conflicts(args: argparse.Namespace) -> _Lines:
    """List what the last pulls could not settle, and optionally re-decide it.

    Three stages, the same shape as the maintenance CLI's `repair` and
    `share review`:
    no flag lists; --repair renders the prompt (dry-run, no API call);
    --repair --execute runs the session; --apply additionally writes, through
    the ordinary validate→apply path.
    """

    repo = _repo(args)
    if args.dismiss or args.resolve:
        return _record_manual_verdict(repo, args)
    remote = _remote_id(args.remote) if args.remote else ""
    rows = conflict_ledger.open_conflicts(repo.root, remote=remote)
    if not rows:
        return ["no open conflicts" + (f" for {remote}" if remote else "")]
    from .conflict_repair import human_only

    lines = [f"{len(rows)} open conflict(s):"]
    for row in rows:
        line = f"  {conflict_ledger.describe(row)}"
        reason = human_only(repo, row)
        if reason:
            # Say it next to the id `--dismiss`/`--resolve` wants, or the
            # reader watches `--repair` skip it with no explanation
            # (reviewer 2026-09-01 P2).
            line += f"  ⚠ {reason}——`--repair` 处理不了；人工看过后用 --dismiss/--resolve 关掉"
        lines.append(line)
    if not args.repair:
        return lines + [
            "re-decide them with: share conflicts --repair,"
            " or close one by hand: share conflicts --dismiss <id> --reason ...",
        ]

    from .conflict_repair import (
        conflict_handle_map,
        group_by_subject,
        human_only,
        render_conflict_repair_prompt,
        run_conflict_repair_session,
    )

    grouped = group_by_subject(repo, rows)
    unroutable = len(rows) - sum(len(group) for group in grouped.values())
    if unroutable:
        lines.append(
            f"  ⚠ {unroutable} conflict(s) skipped (see the ⚠ above them):"
            " a session cannot act on those; close them by hand"
        )
    client_ = None
    if args.execute:
        from ...client import RoleClient

        client_ = RoleClient()
    for subject_id, group in grouped.items():
        subject = repo.store.node(subject_id)
        surface = subject.payload.get("surface", subject_id) if subject else subject_id
        lines.append(f"— {surface}: {len(group)} conflict(s)")
        if not args.execute:
            prompt = render_conflict_repair_prompt(
                repo, subject_id, group, conflict_handles=conflict_handle_map(group)
            )
            lines.append(
                f"  dry-run: repair prompt rendered ({len(prompt)} chars);"
                " --execute 跑判定会话"
            )
            continue
        result = run_conflict_repair_session(
            repo, subject_id, group, client=client_, apply=args.apply
        )
        for row in result["verdicts"]:
            lines.append(f"  {row['conflict']}: {row['verdict']} — {row['reason']}")
        if not args.apply:
            lines.append("  proposals (not applied; --apply to run them through validate→apply):")
            lines.extend(
                f"    {line}" for line in result["proposals_text"].splitlines() if line.strip()
            )
            continue
        report = result["apply_report"]
        lines.append(
            f"  applied rev {report.get('rev')}: {len(report.get('applied', []))} op(s),"
            f" {len(report.get('skipped', []))} skipped"
        )
        for row in result["booked"]:
            lines.append(
                f"  {row['conflict']} → {row['status']}"
                + (f" — {row['note']}" if row.get("note") else "")
            )
    return lines


def _cmd_review(args: argparse.Namespace) -> _Lines:
    """Maintainer review (plan §6.3). Dry-run by default and **read-only**:
    it peeks the queue without leasing (looking must not lock, round 7) and
    renders the threshold pre-pass + review prompt. --execute leases ONE item
    at a time, runs the LLM session, and releases the lease if anything
    fails; --post additionally posts the verdict — the two are separate acts
    on purpose, the verdict stays the maintainer's. --override-thresholds
    lets the maintainer approve past the §6.3 gate, with the reason recorded
    server-side."""

    from .review import render_review_prompt, run_review_session, threshold_report

    remote = _remote_id(args.remote)
    peeked = client.peek_queue(
        remote, maintainer_token=args.maintainer_token, queue_id=args.queue_id
    )
    items = peeked.get("items") or []
    if not items:
        return ["queue is empty (or the requested item is not pending)"]
    lines: _Lines = []
    llm_client = None
    if args.execute:
        from ...client import RoleClient

        llm_client = RoleClient()
    for peek_item in items:
        queue_id = peek_item["queue_id"]
        lines.append(
            f"— queue item {queue_id} (contributor {peek_item.get('contributor', '')[:8]}…"
            + (", leased elsewhere" if peek_item.get("leased") else "")
            + ") —"
        )
        for row in threshold_report(peek_item.get("bundle") or {}):
            state = "OK" if row["satisfied"] else ("需外部印证" if row["needs_external"] else "人工")
            lines.append(f"  claim {row['node']} {row['field_path']} [{row['slot']}]: {state}")
        if not args.execute:
            prompt = render_review_prompt(peek_item)
            lines.append(f"  dry-run: review prompt rendered ({len(prompt)} chars); --execute 跑审核会话")
            continue
        leased = client.lease_queue(
            remote, maintainer_token=args.maintainer_token,
            seconds=args.lease_seconds, limit=1, queue_id=queue_id,
        ).get("items") or []
        if not leased:
            lines.append("  skipped: leased by another session")
            continue
        item = leased[0]
        # The lease is released on EVERY exit except a verdict the server
        # actually consumed (review 2026-08-27 round 8: a network error or a
        # threshold 409 from post_verdict used to strand the lease for its
        # full duration). A successful verdict clears the lease server-side.
        consumed = False
        try:
            review = run_review_session(item, client=llm_client)
            lines.append(
                f"  session verdict: {review['verdict']}"
                + (f" merge={review['merge']}" if review["merge"] else "")
                + f" — {review['reason']}"
            )
            for row in review["external_evidence"]:
                lines.append(f"  evidence {row['node']} {row['field_path']}: {row['url']}")
            if args.post:
                reply = client.post_verdict(
                    remote,
                    maintainer_token=args.maintainer_token,
                    queue_id=queue_id,
                    lease_token=item["lease_token"],
                    expected_version=item["verdict_version"],
                    verdict=review["verdict"],
                    reason=review["reason"],
                    merge=review["merge"],
                    evidence=review["external_evidence"],
                    override=args.override_thresholds or "",
                )
                consumed = True
                lines.append(f"  posted: {reply['status']}")
            else:
                lines.append("  not posted (--post to send); lease released")
        finally:
            if not consumed:
                try:
                    client.release_item(
                        remote, maintainer_token=args.maintainer_token,
                        queue_id=queue_id, lease_token=item["lease_token"],
                    )
                except Exception:
                    pass  # a lost release just lets the lease time out
    return lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m finesub.llm.knowledge.share",
        description="Share client: mark/push/pull against a share server (plan §6).",
    )
    parser.add_argument("--root", default=None, help="knowledge root (default: resolved runtime root)")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="obtain an anonymous contributor token")
    register.add_argument("--remote", required=True)
    register.set_defaults(func=_cmd_register)

    for name, handler, help_text in (
        ("mark", _cmd_mark, "mark a subject's nodes shareable (explicit opt-in)"),
        ("unmark", _cmd_unmark, "set nodes back to local (undo a mark)"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("subject")
        command.add_argument(
            "--kinds",
            default="",
            help="extra kinds beyond subject,term (note needs this second confirmation)",
        )
        command.add_argument(
            "--match",
            default="",
            help="only nodes whose label contains this text (share one fact without its neighbors)",
        )
        command.set_defaults(func=handler)

    push = sub.add_parser("push", help="push marked subjects to the review queue")
    push.add_argument("subjects", nargs="+")
    push.add_argument("--remote", required=True)
    push.set_defaults(func=_cmd_push)

    status = sub.add_parser("status", help="check a queued push; backfill canonical ids on approval")
    status.add_argument("--remote", required=True)
    status.add_argument("--queue-id", type=int, required=True)
    status.set_defaults(func=_cmd_status)

    pull = sub.add_parser("pull", help="fetch, verify and merge the server snapshot")
    pull.add_argument("--remote", required=True)
    pull.set_defaults(func=_cmd_pull)

    conflicts = sub.add_parser(
        "conflicts",
        help="list pull conflicts still unresolved; --repair re-decides them (dry-run by default)",
    )
    conflicts.add_argument("--remote", default="", help="only this remote's conflicts")
    conflicts.add_argument(
        "--repair", action="store_true",
        help="hand the open conflicts to a model against the entry's current contents",
    )
    conflicts.add_argument(
        "--execute", action="store_true",
        help="run the repair session (spends quota; default only renders the prompt)",
    )
    conflicts.add_argument(
        "--apply", action="store_true",
        help="apply the session's proposals through the normal validate→apply path",
    )
    conflicts.add_argument(
        "--dismiss", metavar="ID",
        help="close one by hand: local is right, stop reporting it (sticky)",
    )
    conflicts.add_argument(
        "--resolve", metavar="ID",
        help="close one by hand: it has been dealt with (reopens if it comes back)",
    )
    conflicts.add_argument("--reason", default="", help="why (required with --dismiss/--resolve)")
    conflicts.set_defaults(func=_cmd_conflicts)

    review = sub.add_parser(
        "review",
        help="maintainer: lease queue items and run the LLM review (dry-run by default)",
    )
    review.add_argument("--remote", required=True)
    review.add_argument("--maintainer-token", required=True)
    review.add_argument("--queue-id", type=int, default=None, help="review one specific item")
    review.add_argument("--lease-seconds", type=int, default=900)
    review.add_argument(
        "--execute", action="store_true",
        help="run the review LLM session (spends quota; default only renders the prompt)",
    )
    review.add_argument(
        "--post", action="store_true",
        help="post the session's verdict (+external evidence) back to the server",
    )
    review.add_argument(
        "--override-thresholds",
        default="",
        metavar="REASON",
        help="approve past the §6.3 gate; the reason is recorded in the verdict note",
    )
    review.set_defaults(func=_cmd_review)

    return parser.parse_args(argv)


#: Commands that write local state: they serialize behind the same
#: cross-process lock the maintenance CLI and auto-apply use — a pull racing
#: a knowledge update (or another pull) must not interleave. ``push`` is here
#: for the intent log: two concurrent pushes of the same content must agree
#: on one idempotency key, so the lookup/append (and, coarsely, the send)
#: happen under the lock.
_MUTATING = {"register", "mark", "unmark", "status", "pull", "push", "conflicts"}


def main(argv: list[str] | None = None) -> int:
    from ..base import knowledge_write_lock

    args = parse_args(argv)
    try:
        if args.command in _MUTATING:
            with knowledge_write_lock(knowledge_root_path(args.root)) as acquired:
                if not acquired:
                    print("error: knowledge base is locked by another process", file=sys.stderr)
                    return 1
                lines = args.func(args)
        else:
            lines = args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0
